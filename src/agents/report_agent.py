"""Morning Report (/report) TradingAgents-Lite 3-Step Pipeline & /us Sentinel."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend, is_degraded_llm_output
from src.agents.prompts import build_agent_system_instruction, build_wallet_context_text
from src.agents.semantic_auditor import audit_generated_output, build_fact_baseline
from src.agents.schemas import (
    MorningReportDecision,
    parse_or_repair_structured_json,
    render_morning_report,
    trim_text_at_line_boundary,
)
from src.agents.tools import MarketToolkit, _format_us_deep_intel, run_forced_market_grounding
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.analytics import (
    TechnicalSnapshot,
    build_nav_row,
    compute_exposure_summary,
    compute_technical_snapshot,
    compute_window_return_pct,
    format_technical_line,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    PortfolioData,
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import (
    get_market_calendar_alerts,
    get_uk_weekday_label,
    get_us_market_schedule_uk,
    is_uk_bank_holiday,
)
from src.engine.deployment_gate import (
    evaluate_deployment_gate,
    format_deployment_gate_banner,
    reconcile_pm_pick_gate,
)
from src.engine.portfolio_rules import (
    DisagreementItem,
    HoldingEvaluation,
    apply_risk_override_guardrail,
    build_disagreement_matrix,
    build_report_size_notes,
    evaluate_portfolio,
    format_disagreement_matrix_text,
    format_evaluations_text,
)
from src.storage.db import AdvisorDatabase, get_trading_cycle_id

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")
CONTEXT_PACK_VERSION = "1.0"
_PENDING_PROXIMITY_PCT = 1.5
_VIX_DANGER_LEVEL = 28.0


def _is_us_symbol(symbol: str) -> bool:
    return "." not in symbol and "=" not in symbol and "^" not in symbol


def _extract_holding_context(portfolio: PortfolioData) -> tuple[dict[str, str], dict[str, str]]:
    """Extract per-symbol ISIN and earliest open timestamp from live portfolio positions."""
    isin_map: dict[str, str] = {}
    created_at_map: dict[str, str] = {}
    for pos in portfolio.all_positions():
        sym = pos.yahoo_symbol
        if pos.isin and pos.isin.strip():
            isin_map.setdefault(sym, pos.isin.strip().upper())
        if pos.created_at and pos.created_at.strip():
            ts = pos.created_at.strip()
            existing = created_at_map.get(sym)
            if existing is None or ts < existing:
                created_at_map[sym] = ts
    return isin_map, created_at_map


def _format_snapshot_age(elapsed_hours: float) -> str:
    if elapsed_hours >= 24.0:
        return f"约{elapsed_hours / 24.0:.1f}天前"
    if elapsed_hours >= 1.0:
        return f"约{elapsed_hours:.1f}小时前"
    return f"约{max(1, int(round(elapsed_hours * 60)))}分钟前"


def _compute_portfolio_diff_text(
    current: PortfolioData,
    previous: PortfolioData | None,
    *,
    cash_flow_summary_text: str = "",
) -> str:
    """Summarize portfolio value/position changes with snapshot age & cash-flow attribution."""
    if previous is None:
        base = "（首次建档，暂无前一日净值基准；以 30 日实盘成交作依据）"
        return f"{base} | {cash_flow_summary_text.splitlines()[0]}" if cash_flow_summary_text else base

    elapsed_hours = max(0.0, (current.fetched_at_ms - previous.fetched_at_ms) / 3_600_000.0)
    span_str = _format_snapshot_age(elapsed_hours)
    fallback_tag = "，不足4h基准已回退最近快照" if elapsed_hours < 3.99 else ""
    anchor_note = f"对比 {previous.fetched_at_uk} UK（{span_str}{fallback_tag}）"

    def _map_pos(data: PortfolioData) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for acc in (data.isa, data.invest):
            if not acc.ok:
                continue
            for p in acc.positions:
                out[f"{acc.account_label}:{p.ticker}"] = {
                    "qty": p.quantity,
                    "account": acc.account_label,
                    "ticker": p.ticker,
                }
        return out

    prev_map = _map_pos(previous)
    cur_map = _map_pos(current)
    changes: list[str] = []

    val_diff = current.total_combined_value - previous.total_combined_value
    if abs(val_diff) >= 0.01:
        changes.append(
            f"总净值: {previous.total_combined_value:,.2f} → {current.total_combined_value:,.2f} GBP ({val_diff:+,.2f})"
        )

    def _add(sums: dict[str, float], ccy: str, amount: float) -> None:
        key = str(ccy or "GBP").strip().upper() or "GBP"
        sums[key] = sums.get(key, 0.0) + float(amount or 0.0)

    ext_by_ccy: dict[str, float] = {}
    int_by_ccy: dict[str, float] = {}
    fee_by_ccy: dict[str, float] = {}
    div_by_ccy: dict[str, float] = {}
    for acc in (current.isa, current.invest):
        for tx in acc.recent_transactions:
            if tx.date_time_ms >= previous.fetched_at_ms:
                if tx.tx_type in {"DEPOSIT", "WITHDRAW"}:
                    _add(ext_by_ccy, tx.currency, tx.amount)
                elif tx.tx_type in {"INTEREST_ON_FREE_CASH", "LENDING_INTEREST"}:
                    _add(int_by_ccy, tx.currency, tx.amount)
                elif tx.tx_type == "FEE":
                    _add(fee_by_ccy, tx.currency, tx.amount)
        for dv in acc.recent_dividends:
            if dv.paid_on_ms >= previous.fetched_at_ms:
                _add(div_by_ccy, dv.currency, dv.amount)

    def _fmt_sums(sums: dict[str, float], *, plus_only: bool = False) -> str:
        parts = [
            (f"+{amt:,.2f} {ccy}" if plus_only else f"{amt:+,.2f} {ccy}")
            for ccy, amt in sums.items()
            if abs(amt) >= 0.01
        ]
        return "、".join(parts)

    ext_txt = _fmt_sums(ext_by_ccy)
    if ext_txt:
        ext_gbp = ext_by_ccy.get("GBP", 0.0)
        changes.append(
            f"区间外部出入金 {ext_txt}（按 GBP 口径剔除出入金后净值变动约 {val_diff - ext_gbp:+,.2f} GBP）"
        )
    div_txt = _fmt_sums(div_by_ccy, plus_only=True)
    if div_txt:
        changes.append(f"区间派息入账 {div_txt}")
    int_txt = _fmt_sums(int_by_ccy, plus_only=True)
    if int_txt:
        changes.append(f"区间现金利息 {int_txt}")
    fee_txt = _fmt_sums(fee_by_ccy)
    if fee_txt:
        changes.append(f"区间平台扣费 {fee_txt}")

    for k, cur in cur_map.items():
        if k not in prev_map:
            changes.append(f"新增 [{cur['account']}] {cur['ticker']} ({cur['qty']:.4f}股)")
        else:
            dq = cur["qty"] - prev_map[k]["qty"]
            if abs(dq) > 1e-4:
                changes.append(f"{'加仓' if dq > 0 else '减仓'} [{cur['account']}] {cur['ticker']} ({dq:+.4f}股)")
    for k, prev in prev_map.items():
        if k not in cur_map:
            changes.append(f"清仓 [{prev['account']}] {prev['ticker']}")

    if not changes:
        return f"{anchor_note} 无净值或仓位数量变动"
    return f"{anchor_note} | " + "; ".join(changes)


def _account_change_pct(
    portfolio: PortfolioData,
    prev_snap: PortfolioData | None,
) -> float | None:
    """Signed % net-value change vs the previous snapshot (None when no usable baseline)."""
    if prev_snap is None:
        return None
    prev_val = float(prev_snap.total_combined_value or 0.0)
    cur_val = float(portfolio.total_combined_value or 0.0)
    if prev_val <= 0 or cur_val <= 0:
        return None
    return ((cur_val - prev_val) / prev_val) * 100.0


def _lookup_benchmark_hist(
    histories: dict[str, pd.DataFrame] | None,
    symbol: str,
) -> pd.DataFrame | None:
    """Fetch the benchmark history tolerating case / suffix differences (never bool() a DataFrame)."""
    if not histories:
        return None
    for key in (symbol, symbol.upper(), symbol.split(".")[0]):
        hist = histories.get(key)
        if hist is not None and not getattr(hist, "empty", True):
            return hist
    return None


def _format_benchmark_line(
    benchmark_symbol: str,
    hist: pd.DataFrame | None,
    *,
    account_change_pct: float | None = None,
) -> str:
    """One-line 'market vs me' dashboard: benchmark 7d/30d returns plus account change.

    Kept to ~60 characters of numbers so the /report 1800-char hard cap is never squeezed.
    """
    bench = (benchmark_symbol or "VWRP.L").strip().upper()
    ret_7 = compute_window_return_pct(hist, 5)  # 5 trading bars ≈ 7 calendar days
    ret_30 = compute_window_return_pct(hist, 21)  # 21 trading bars ≈ 30 calendar days
    if ret_7 is None and ret_30 is None:
        return f"{bench} 行情未就绪，暂无基准对照"
    parts = [f"{bench} 近7天 {ret_7:+.1f}%" if ret_7 is not None else f"{bench} 近7天 N/A"]
    parts.append(f"近30天 {ret_30:+.1f}%" if ret_30 is not None else "近30天 N/A")
    line = " / ".join(parts)
    if account_change_pct is not None:
        # Never subtract a 1-day snapshot change from a 7-day benchmark return: the
        # mismatched windows fabricate a fake "relative" alpha the PM then acts on.
        line += f"；本账户较上次快照 {account_change_pct:+.1f}%"
    return line


def _compact_lesson_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at_uk": str(row.get("created_at_uk") or ""),
        "account": str(row.get("account") or ""),
        "symbol": str(row.get("symbol") or ""),
        "action": str(row.get("action") or ""),
        "raw_return_pct": row.get("raw_return_pct"),
        "alpha_pct": row.get("alpha_pct"),
        "reflection_lesson": str(row.get("reflection_lesson") or "").strip()[:240],
    }


async def _fetch_symbol_lessons(
    db: AdvisorDatabase | None,
    symbols: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if db is None:
        return {}, []

    async def _load_recent() -> list[dict[str, Any]]:
        try:
            rows = await db.get_recent_lessons(limit=5)
            return [_compact_lesson_record(r) for r in rows]
        except Exception as exc:
            logger.debug("Failed to load recent lessons: %s", exc)
            return []

    async def _load_sym(sym: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            rows = await db.get_symbol_lessons(sym, limit=2)
            return sym, [_compact_lesson_record(r) for r in rows]
        except Exception as exc:
            logger.debug("Failed to load lessons for %s: %s", sym, exc)
            return sym, []

    recent_task = _load_recent()
    sym_tasks = [_load_sym(sym) for sym in symbols]
    results = await asyncio.gather(recent_task, *sym_tasks)
    recent_lessons = results[0]
    by_symbol = {sym: items for sym, items in results[1:] if items}
    return by_symbol, recent_lessons


def build_analysis_context_pack(
    *,
    portfolio: PortfolioData,
    tracked_syms: list[str],
    tech_map: dict[str, TechnicalSnapshot],
    bundles: dict[str, TickerMarketBundle],
    macro: MacroSnapshot,
    evals: list[HoldingEvaluation],
    disagreements: list[DisagreementItem],
    disagreement_text: str,
    exposure_text: str,
    calendar_alerts: list[str],
    isin_map: dict[str, str] | None = None,
    lessons_by_sym: dict[str, list[dict[str, Any]]] | None = None,
    recent_lessons: list[dict[str, Any]] | None = None,
    us_intel_map: dict[str, dict[str, Any]] | None = None,
    cash_flow_summary: str = "",
    portfolio_diff: str = "",
    cycle_date: str | None = None,
) -> dict[str, Any]:
    """Assemble compact JSON-serializable AnalysisContextPack for SQLite persistence & cross-command reuse."""
    now_utc = datetime.now(timezone.utc)
    resolved_cycle = cycle_date or get_trading_cycle_id(now_utc)
    isin_lookup = isin_map or {}
    lessons_lookup = lessons_by_sym or {}
    intel_lookup = us_intel_map or {}

    pos_by_sym: dict[str, list[dict[str, Any]]] = {}
    for pos in portfolio.all_positions():
        native_ret = (
            round(((pos.cur_price / pos.avg_price) - 1.0) * 100.0, 2)
            if pos.avg_price > 0 and pos.cur_price > 0
            else None
        )
        pos_by_sym.setdefault(pos.yahoo_symbol, []).append(
            {
                "account": pos.account,
                "ticker": pos.ticker,
                "inst_currency": pos.inst_currency,
                "weight_pct": round(pos.weight_pct, 2),
                "return_pct": round(pos.return_pct, 2),
                "native_return_pct": native_ret,
                "cur_price": pos.cur_price,
                "avg_price": pos.avg_price,
                "created_at": pos.created_at,
            }
        )

    eval_by_sym: dict[str, list[dict[str, Any]]] = {}
    for ev in evals:
        eval_by_sym.setdefault(ev.yahoo_symbol, []).append(ev.model_dump())

    dis_by_sym: dict[str, list[dict[str, Any]]] = {}
    for dis in disagreements:
        dis_by_sym.setdefault(dis.yahoo_symbol, []).append(dis.model_dump())

    symbols_pack: dict[str, dict[str, Any]] = {}
    missing_tech: list[str] = []
    for sym in tracked_syms:
        b = bundles.get(sym)
        t = tech_map.get(sym)
        if t is None:
            missing_tech.append(sym)
        identity = ""
        if b is not None:
            identity = b.identity_summary or b.build_identity_anchor(isin_override=isin_lookup.get(sym))
        held_list = pos_by_sym.get(sym, [])
        fallback_curr = held_list[0]["inst_currency"] if held_list else ""
        bundle_summary = (
            {
                "short_name": b.short_name,
                "long_name": b.long_name,
                "quote_type": b.quote_type,
                "exchange": b.exchange,
                "currency": b.currency or fallback_curr,
                "sector": b.sector,
                "industry": b.industry,
                "current_price": b.current_price,
                "target_mean_price": b.target_mean_price,
                "forward_pe": b.forward_pe,
                "trailing_pe": b.trailing_pe,
                "dividend_yield_pct": b.dividend_yield_pct,
                "beta": b.beta,
                "roe_pct": b.roe_pct,
                "profit_margin_pct": b.profit_margin_pct,
                "revenue_growth_pct": b.revenue_growth_pct,
                "debt_to_equity": b.debt_to_equity,
                "recommendation_key": b.recommendation_key,
                "earnings_date": b.earnings_date,
                "days_to_earnings": b.days_to_earnings,
                "ex_dividend_date": b.ex_dividend_date,
                "days_to_ex_dividend": b.days_to_ex_dividend,
                "news": list(b.news[:3]),
            }
            if b is not None
            else {}
        )
        symbols_pack[sym] = {
            "symbol": sym,
            "identity_summary": identity or sym,
            "isin": isin_lookup.get(sym) or (b.isin if b else ""),
            "held_accounts": held_list,
            "technical": t.model_dump() if t is not None else None,
            "technical_line": format_technical_line(sym, t, b),
            "bundle_summary": bundle_summary,
            "evaluations": eval_by_sym.get(sym, []),
            "disagreements": dis_by_sym.get(sym, []),
            "lessons": lessons_lookup.get(sym, []),
            "deep_intel": intel_lookup.get(sym.upper()) or intel_lookup.get(sym) or {},
        }

    return {
        "pack_version": CONTEXT_PACK_VERSION,
        "cycle_date": resolved_cycle,
        "generated_at_uk": now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
        "is_fallback": bool(portfolio.is_fallback),
        "calendar_alerts": list(calendar_alerts),
        "macro": macro.model_dump() | {"summary": macro.to_text()},
        "exposure_summary": exposure_text,
        "cash_flow_summary": cash_flow_summary,
        "portfolio_diff": portfolio_diff,
        "disagreements": [d.model_dump() for d in disagreements],
        "disagreement_text": disagreement_text,
        "recent_lessons": list(recent_lessons or []),
        "symbols": symbols_pack,
        "data_quality": {
            "portfolio_is_fallback": bool(portfolio.is_fallback),
            "macro_is_stale": bool(macro.is_stale),
            "missing_tech_symbols": missing_tech,
        },
    }


async def _assemble_fact_sheet_bundle(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    cfg: AppConfig,
    *,
    us_only: bool = False,
    db: AdvisorDatabase | None = None,
) -> tuple[
    str,
    dict[str, TechnicalSnapshot],
    dict[str, TickerMarketBundle],
    MacroSnapshot,
    dict[str, pd.DataFrame],
    list[HoldingEvaluation],
    list[DisagreementItem],
    str,
    list[str],
    dict[str, Any],
]:
    """Gather deterministic data, CYQ/weekly/ex-div technicals, disagreements, lessons & ContextPack."""
    all_tracked = portfolio.unique_yahoo_symbols(include_pending=True)
    held_all = portfolio.unique_yahoo_symbols(include_pending=False)
    tracked_syms = [s for s in all_tracked if _is_us_symbol(s)] if us_only else all_tracked
    held_syms_set = set(held_all)

    fetch_syms = all_tracked if (us_only and len(all_tracked) > len(tracked_syms)) else tracked_syms
    isin_map, created_at_map = _extract_holding_context(portfolio)
    fetch_isin_map = {sym: isin_map[sym] for sym in fetch_syms if sym in isin_map} or None

    us_tracked = [s for s in tracked_syms if _is_us_symbol(s)]

    async def _empty_intel() -> dict[str, dict[str, Any]]:
        return {}

    intel_coro = (
        market_data.get_us_deep_intel_batch(us_tracked)
        if (us_tracked and hasattr(market_data, "get_us_deep_intel_batch"))
        else _empty_intel()
    )

    (histories, bundles), macro, (lessons_by_sym, recent_lessons), us_intel_map = await asyncio.gather(
        market_data.get_bundles(
            fetch_syms,
            include_news=True,
            include_earnings=True,
            isin_map=fetch_isin_map,
        ),
        market_data.get_macro_snapshot(),
        _fetch_symbol_lessons(db, tracked_syms),
        intel_coro,
    )

    tech_map: dict[str, TechnicalSnapshot] = {}
    for sym in tracked_syms:
        snap = compute_technical_snapshot(
            sym,
            histories.get(sym),
            position_created_at=created_at_map.get(sym),
            bundle=bundles.get(sym),
        )
        if snap is not None:
            tech_map[sym] = snap

    evals = evaluate_portfolio(portfolio, tech_map, bundles, cfg.strategy, histories=histories)
    if us_only:
        evals = [e for e in evals if _is_us_symbol(e.yahoo_symbol)]
    eval_text = format_evaluations_text(evals)
    exposure_text = compute_exposure_summary(portfolio, bundles, histories=histories)

    disagreements = build_disagreement_matrix(portfolio, tech_map, bundles, cfg.strategy, evals=evals)
    if us_only:
        disagreements = [d for d in disagreements if _is_us_symbol(d.yahoo_symbol)]
    disagreement_text = format_disagreement_matrix_text(disagreements)

    calendar_alerts = get_market_calendar_alerts()

    identity_lines: list[str] = []
    for sym in tracked_syms:
        b = bundles.get(sym)
        if b is None:
            continue
        is_etf = b.quote_type.upper() == "ETF"
        is_pending_only = sym not in held_syms_set
        near_ex_div = b.days_to_ex_dividend is not None and -3 <= int(b.days_to_ex_dividend) <= 7
        near_earnings = b.days_to_earnings is not None and 0 <= int(b.days_to_earnings) <= 7
        if not (is_etf or is_pending_only or near_ex_div or near_earnings):
            continue
        anchor = b.identity_summary or b.build_identity_anchor(isin_override=isin_map.get(sym))
        if anchor:
            identity_lines.append(f"• {anchor}")

    held_lines = [
        format_technical_line(sym, tech_map.get(sym), bundles.get(sym))
        for sym in tracked_syms
        if sym in held_syms_set
    ]
    pending_only_syms = [sym for sym in tracked_syms if sym not in held_syms_set]
    pending_tech_lines = [format_technical_line(sym, tech_map.get(sym), bundles.get(sym)) for sym in pending_only_syms]

    lesson_lines: list[str] = []
    for sym in tracked_syms:
        for item in lessons_by_sym.get(sym, []):
            alpha_str = f"Alpha {item['alpha_pct']:+.1f}%" if item.get("alpha_pct") is not None else "已结算"
            lesson_lines.append(
                f"• {sym} [{item['created_at_uk']} {item['action']} | {alpha_str}]: {item['reflection_lesson']}"
            )

    deep_intel_lines: list[str] = []
    for sym in tracked_syms:
        if not _is_us_symbol(sym):
            continue
        b = bundles.get(sym)
        if b is not None and b.quote_type.upper() == "ETF":
            continue
        intel_raw = us_intel_map.get(sym.upper()) or us_intel_map.get(sym) or {}
        intel_fmt = _format_us_deep_intel(intel_raw)
        if intel_fmt:
            deep_intel_lines.append(f"• {sym}: {intel_fmt}")

    news_lines = []
    for sym in tracked_syms:
        b = bundles.get(sym)
        if b and b.news and b.quote_type.upper() != "ETF":
            t = tech_map.get(sym)
            is_high_interest = (
                sym not in held_syms_set
                or (b.days_to_earnings is not None and 0 <= int(b.days_to_earnings) <= 7)
                or (t is not None and t.ret_1d_pct is not None and abs(float(t.ret_1d_pct)) >= 3.0)
            )
            max_news = 3 if is_high_interest else 1
            titles = " / ".join(f"[{n['date']}] {n['title']}" for n in b.news[:max_news])
            news_lines.append(f"• {sym}: {titles}")

    calendar_section = (
        "\n\n【英美欧市场日历提醒】\n" + "\n".join(f"• {a}" for a in calendar_alerts)
        if calendar_alerts
        else ""
    )
    identity_section = (
        "\n\n【标的身份锚定（ISIN / 交易所 / 板块）】\n" + "\n".join(identity_lines)
        if identity_lines
        else ""
    )
    lesson_section = (
        "\n\n【标的 T+5 历史复盘教训（SQLite 记忆）】\n" + "\n".join(lesson_lines)
        if lesson_lines
        else ""
    )
    deep_intel_section = (
        "\n\n【美股机构评级、EPS惊喜与内部人动向（Finnhub）】\n" + "\n".join(deep_intel_lines)
        if deep_intel_lines
        else ""
    )

    fact_sheet = (
        f"【实时宏观与汇率（确定性数据源）】\n{macro.to_text()}"
        + calendar_section
        + f"\n\n【跨账户重叠与行业集中度（Python 实算）】\n{exposure_text}"
        + identity_section
        + f"\n\n【持仓量化技术面、目标价与财报雷达】\n"
        + ("\n".join(held_lines) if held_lines else "（无持仓技术数据）")
        + (
            "\n\n【未成交活动挂单标的实时技术面与估值（供审计挂单）】\n" + "\n".join(pending_tech_lines)
            if pending_tech_lines
            else ""
        )
        + "\n\n【Python 规则引擎：加仓 / 减仓 / 止损复核信号】\n"
        + eval_text
        + "\n\n【本地多空证据冲突表（Disagreement Matrix）】\n"
        + disagreement_text
        + deep_intel_section
        + lesson_section
        + "\n\n【近期公司新闻摘要】\n"
        + ("\n".join(news_lines) if news_lines else "（近7天无重要突发新闻或暂无抓取）")
    )

    cycle_id = (
        db._cycle_id()
        if db is not None
        else get_trading_cycle_id(
            reset_hour=cfg.schedule.morning_report_hour,
            reset_minute=cfg.schedule.morning_report_minute,
        )
    )
    context_pack = build_analysis_context_pack(
        portfolio=portfolio,
        tracked_syms=tracked_syms,
        tech_map=tech_map,
        bundles=bundles,
        macro=macro,
        evals=evals,
        disagreements=disagreements,
        disagreement_text=disagreement_text,
        exposure_text=exposure_text,
        calendar_alerts=calendar_alerts,
        isin_map=isin_map,
        lessons_by_sym=lessons_by_sym,
        recent_lessons=recent_lessons,
        us_intel_map=us_intel_map,
        cycle_date=cycle_id,
    )
    return (
        fact_sheet,
        tech_map,
        bundles,
        macro,
        histories,
        evals,
        disagreements,
        disagreement_text,
        calendar_alerts,
        context_pack,
    )


async def build_portfolio_fact_sheet(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    cfg: AppConfig,
    us_only: bool = False,
    *,
    db: AdvisorDatabase | None = None,
    context_pack_out: dict[str, Any] | None = None,
) -> tuple[str, dict[str, TechnicalSnapshot], dict[str, TickerMarketBundle], MacroSnapshot]:
    """Gather deterministic data for held positions AND pending orders + exposure breakdown."""
    (
        fact_sheet,
        tech_map,
        bundles,
        macro,
        _histories,
        _evals,
        _disagreements,
        _disagreement_text,
        _calendar_alerts,
        context_pack,
    ) = await _assemble_fact_sheet_bundle(
        portfolio,
        market_data,
        cfg,
        us_only=us_only,
        db=db,
    )
    if context_pack_out is not None:
        context_pack_out.update(context_pack)
    return fact_sheet, tech_map, bundles, macro


STEP2_ANALYST_FALLBACK = (
    "（本轮 Bull 机会分析未返回有效结果：模型调用超时或不可用。PM 请仅依据本地 FactSheet、"
    "多空证据对照与量化指标，按保守口径裁决，不得编造新的价位依据；无证据支撑则维持 HOLD。）"
)
STEP2_AUDITOR_FALLBACK = (
    "（本轮 Bear 独立审计未返回有效结果：模型调用超时或不可用。PM 请按保守口径自行裁决："
    "今日不放行超出既有计划的新增加仓，既有挂单维持原状不撤。）"
)


async def _gather_step2_notes(
    analyst_coro: Awaitable[str],
    auditor_coro: Awaitable[str],
    *,
    timeout_s: float,
) -> tuple[str, str, list[str]]:
    """Run Bull/Bear concurrently under one wall-clock budget.

    Timed-out tasks are cancelled and their notes replaced with neutral placeholders, and the
    `generate()` model-unavailable sentinel is swapped out too: OpsNotifier wording must never
    reach the PM prompt where it would read like a Bear audit opinion.
    """
    tasks: dict[str, asyncio.Task[str]] = {
        "analyst": asyncio.ensure_future(analyst_coro),
        "auditor": asyncio.ensure_future(auditor_coro),
    }
    degraded: list[str] = []
    if timeout_s > 0:
        _, pending = await asyncio.wait(set(tasks.values()), timeout=timeout_s)
        if pending:
            for role, task in tasks.items():
                if task in pending:
                    degraded.append(role)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Report Step 2 wall-clock budget %.0fs exceeded; degraded roles: %s",
                timeout_s,
                ", ".join(degraded),
            )
    else:
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    fallbacks = {"analyst": STEP2_ANALYST_FALLBACK, "auditor": STEP2_AUDITOR_FALLBACK}
    notes: dict[str, str] = {}
    for role, task in tasks.items():
        text = ""
        if not task.cancelled():
            try:
                text = (task.result() or "").strip()
            except Exception as exc:  # defensive: generate() already converts its own failures
                logger.warning("Report Step 2 %s task raised: %s", role, exc)
        if not text or is_degraded_llm_output(text):
            if role not in degraded:
                degraded.append(role)
            text = fallbacks[role]
        notes[role] = text
    return notes["analyst"], notes["auditor"], degraded


# ---------- v5.4.2: Step 2 round-2 debate (DeepSeek) + dual neutral index ----------

_DEBATE_NEUTRAL_SYSTEM = (
    "你是只读的中立辩论记录员/数字核验员，只做信息索引与事实比对，"
    "不做投资裁决、不给买卖建议、不新增任何事实或价位。"
)


@dataclass
class Step2DebateResult:
    """R1/R2/R3 debate output handed to the PM (full transcript goes to context_pack only)."""

    analyst_notes: str
    auditor_notes: str
    digest: str = ""
    factcheck: str = ""
    degraded_roles: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


async def _gather_named_notes(
    coros: dict[str, Awaitable[str]],
    *,
    timeout_s: float,
) -> tuple[dict[str, str], list[str]]:
    """Run named coroutines concurrently under one wall-clock budget.

    Unlike `_gather_step2_notes`, a failed round returns an empty string (no neutral
    placeholder): optional debate rounds must never leak ops wording into the PM prompt.
    """
    tasks = {name: asyncio.ensure_future(coro) for name, coro in coros.items()}
    degraded: list[str] = []
    if timeout_s > 0:
        _, pending = await asyncio.wait(set(tasks.values()), timeout=timeout_s)
        if pending:
            degraded.extend(name for name, task in tasks.items() if task in pending)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Report debate budget %.0fs exceeded; degraded roles: %s",
                timeout_s,
                ", ".join(degraded),
            )
    else:
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    texts: dict[str, str] = {}
    for name, task in tasks.items():
        text = ""
        if not task.cancelled():
            try:
                text = (task.result() or "").strip()
            except Exception as exc:  # defensive: generate() already converts its own failures
                logger.warning("Report debate %s task raised: %s", name, exc)
        if not text or is_degraded_llm_output(text):
            if name not in degraded:
                degraded.append(name)
            text = ""
        texts[name] = text
    return texts, degraded


def _build_rebuttal_prompt(
    *,
    side: str,
    own_notes: str,
    opponent_notes: str,
    shared_context: str,
    max_chars: int,
) -> str:
    if side == "bull":
        role = "Bull（机会方）"
        opponent = "Bear（风控审计官）"
        focus = "逐条反驳 Bear 的风险论断，指出哪些担忧已被事实底座化解、哪些被夸大，并守住你首轮的多方立论。"
    else:
        role = "Bear（风控审计官）"
        opponent = "Bull（机会方）"
        focus = "逐条攻击 Bull 的支撑位、目标价与加仓理由，指出其最坏情况与证据缺口，并守住你首轮的空方立论。"
    return (
        f"【任务：{role} 第二轮交叉质询（≤{max_chars} 字）】\n"
        "你在首轮已提交过一份草稿；本轮只做交叉质询，不重复首轮已说过的内容。\n"
        f"{focus}\n\n"
        f"=== 你的首轮草稿 ===\n{own_notes}\n\n"
        f"=== 对手 {opponent} 的首轮草稿 ===\n{opponent_notes}\n\n"
        "=== 唯一事实底座（与首轮相同） ===\n"
        f"{shared_context}\n\n"
        "硬规则：\n"
        "1. 开头先写「本轮最关键的 3 个反驳点」；\n"
        "2. 逐条回应对手的具体论点，必须引用底座中的数字/事实，不得编造；\n"
        "3. 不得新增首轮未出现的价位或新闻；\n"
        "4. 不要给最终买卖结论（PM 负责裁决）。"
    )


def _build_rebuttal_system(side: str, cfg: AppConfig, max_chars: int) -> str:
    if side == "bull":
        role = (
            f"你是 Bull 阵营的第二轮交叉质询者（≤{max_chars} 字）。"
            "基于同一份事实底座，逐条反驳 Bear 首轮的论点；不要重写研报、不要给最终买卖结论。"
        )
    else:
        role = (
            f"你是 Bear 阵营的第二轮交叉质询者（≤{max_chars} 字）。"
            "基于同一份事实底座，逐条攻击 Bull 首轮的支撑位/目标价/加仓理由；不要重写研报、不要给最终买卖结论。"
        )
    return build_agent_system_instruction(role, cfg.strategy)


def _build_debate_digest_prompt(
    *,
    bull_notes: str,
    bear_notes: str,
    bull_rebuttal: str,
    bear_rebuttal: str,
    disagreement_text: str,
    max_chars: int,
) -> str:
    return (
        f"【任务：中立辩论摘要（≤{max_chars} 字；只做索引，不做裁决）】\n"
        "你只负责整理这场 Bull/Bear 辩论，不给买卖建议、不新增事实。\n\n"
        f"=== Bull 首轮 ===\n{bull_notes}\n\n"
        f"=== Bear 首轮 ===\n{bear_notes}\n\n"
        f"=== Bull 对 Bear 的质询 ===\n{bull_rebuttal}\n\n"
        f"=== Bear 对 Bull 的质询 ===\n{bear_rebuttal}\n\n"
        f"=== 本地多空证据冲突对照 ===\n{disagreement_text}\n\n"
        "按以下四段输出（每段 ≤3 条）：\n"
        "1. 存活的多方论点；\n"
        "2. 存活的空方论点；\n"
        "3. 已被驳倒的论点及原因；\n"
        "4. 仍无证据、需 PM 亲自裁决的冲突。\n"
        "最后附「关键原文」：挑分歧最大的 2 个标的，各引用双方最关键的一句原话（每句 ≤200 字）。"
    )


def _build_factcheck_prompt(
    *,
    fact_sheet: str,
    bull_rebuttal: str,
    bear_rebuttal: str,
    max_chars: int,
) -> str:
    return (
        f"【任务：只读数字核验（≤{max_chars} 字）】\n"
        "下面给出唯一事实底座与两份第二轮质询。只核对质询中出现的价格、持仓成本、止损价、挂单价、百分比"
        "能否在事实底座中找到依据。\n\n"
        f"=== 只读事实底座 ===\n{fact_sheet}\n\n"
        f"=== Bull 质询 ===\n{bull_rebuttal}\n\n"
        f"=== Bear 质询 ===\n{bear_rebuttal}\n\n"
        "输出格式：逐条 `[无据] 标的 / 数字 / 出自 Bull|Bear`；没有发现则输出「未发现无据数字」。\n"
        "硬规则：只标记、不判违规；底座中找不到不等于错误，由 PM 判断；不得新增事实。"
    )


def _build_debate_pm_block(digest: str, factcheck: str) -> str:
    if not digest and not factcheck:
        return ""
    block = "\n\n=== 第二轮交叉质询与中立索引（DeepSeek，仅供索引，不是裁决） ===\n"
    if digest:
        block += f"【中立摘要】\n{digest}\n"
    if factcheck:
        block += f"【证据核验】\n{factcheck}\n"
    return block


async def _notify_debate_degraded(ops_notifier: OpsNotifier | None, roles: list[str]) -> None:
    if ops_notifier is None or not roles:
        return
    await ops_notifier.notify(
        severity="WARN",
        source="report:debate",
        alert_key="report_debate_degraded",
        message=f"/report 第二轮辩论降级：{'/'.join(roles)} 未返回有效结论，PM 将按首轮材料裁决。",
    )


async def _run_step2_debate(
    *,
    cfg: AppConfig,
    llm: LLMBackend,
    toolkit: MarketToolkit,
    shared_context: str,
    fact_sheet: str,
    disagreement_text: str,
    analyst_sys: str,
    auditor_sys: str,
    analyst_prompt: str,
    auditor_prompt: str,
    report_rounds: int,
    ops_notifier: OpsNotifier | None,
) -> Step2DebateResult:
    """Three-phase Step 2: R1 Gemini debate → R2 DeepSeek cross-exam → R3 dual neutral."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    total_budget = float(cfg.models.report_step2_timeout_seconds)
    r1_budget = min(float(cfg.models.report_step2_round1_seconds), total_budget)

    analyst_notes, auditor_notes, degraded_roles = await _gather_step2_notes(
        llm.generate(
            model=cfg.models.report_analyst_model,
            prompt=analyst_prompt,
            system_instruction=analyst_sys,
            thinking_level=cfg.models.report_analyst_thinking_level,
            tools=toolkit.declarations(),
            tool_executor=toolkit.as_executor(),
            temperature=0.2,
            max_tool_rounds=report_rounds,
            usage_label="report_analyst",
        ),
        llm.generate(
            model=cfg.models.report_auditor_model,
            prompt=auditor_prompt,
            system_instruction=auditor_sys,
            thinking_level=cfg.models.report_auditor_thinking_level,
            tools=toolkit.declarations(),
            tool_executor=toolkit.as_executor(),
            temperature=0.15,
            max_tool_rounds=report_rounds,
            usage_label="report_auditor",
        ),
        timeout_s=r1_budget,
    )

    payload: dict[str, Any] = {
        "enabled": bool(cfg.models.report_debate_enabled),
        "degraded": list(degraded_roles),
        "r1": {"bull": analyst_notes, "bear": auditor_notes},
        "r2": {},
        "r3": {},
        "elapsed": {"r1_s": round(loop.time() - started, 1)},
    }
    result = Step2DebateResult(
        analyst_notes=analyst_notes,
        auditor_notes=auditor_notes,
        degraded_roles=list(degraded_roles),
        payload=payload,
    )

    if degraded_roles:
        payload["skipped"] = "r1_degraded"
        return result
    if not cfg.models.report_debate_enabled or not ds_helper_available(llm, cfg):
        payload["skipped"] = "debate_disabled" if not cfg.models.report_debate_enabled else "deepseek_unavailable"
        return result

    remaining = total_budget - (loop.time() - started)
    r2_budget = min(float(cfg.models.report_step2_round2_seconds), max(0.0, remaining))
    if r2_budget <= 0:
        payload["skipped"] = "no_budget_r2"
        return result

    debate_round_timeout = max(30.0, r2_budget - 15.0)
    max_chars = int(cfg.models.report_debate_max_chars)
    r2_started = loop.time()
    r2_texts, r2_degraded = await _gather_named_notes(
        {
            "bull_rebuttal": llm.generate(
                model=cfg.models.report_debate_model,
                prompt=_build_rebuttal_prompt(
                    side="bull",
                    own_notes=analyst_notes,
                    opponent_notes=auditor_notes,
                    shared_context=shared_context,
                    max_chars=max_chars,
                ),
                system_instruction=_build_rebuttal_system("bull", cfg, max_chars),
                thinking_level=cfg.models.report_debate_thinking_level,
                temperature=0.25,
                usage_label="report_bull_rebuttal",
                round_timeout_seconds=debate_round_timeout,
                max_output_tokens=cfg.models.report_debate_max_output_tokens,
            ),
            "bear_rebuttal": llm.generate(
                model=cfg.models.report_debate_model,
                prompt=_build_rebuttal_prompt(
                    side="bear",
                    own_notes=auditor_notes,
                    opponent_notes=analyst_notes,
                    shared_context=shared_context,
                    max_chars=max_chars,
                ),
                system_instruction=_build_rebuttal_system("bear", cfg, max_chars),
                thinking_level=cfg.models.report_debate_thinking_level,
                temperature=0.2,
                usage_label="report_bear_rebuttal",
                round_timeout_seconds=debate_round_timeout,
                max_output_tokens=cfg.models.report_debate_max_output_tokens,
            ),
        },
        timeout_s=r2_budget,
    )
    r2_texts = {name: trim_text_at_line_boundary(text, max_chars) if text else "" for name, text in r2_texts.items()}
    payload["r2"] = r2_texts
    payload["elapsed"]["r2_s"] = round(loop.time() - r2_started, 1)
    if r2_degraded:
        payload["degraded"].extend(r2_degraded)
        await _notify_debate_degraded(ops_notifier, r2_degraded)
        return result

    remaining = total_budget - (loop.time() - started)
    r3_budget = min(float(cfg.models.report_step2_round3_seconds), max(0.0, remaining))
    if r3_budget <= 0:
        payload["skipped"] = "no_budget_r3"
        return result

    neutral_timeout = max(20.0, r3_budget - 10.0)
    digest_chars = int(cfg.models.report_neutral_digest_max_chars)
    factcheck_chars = int(cfg.models.report_neutral_factcheck_max_chars)
    r3_started = loop.time()
    r3_texts, r3_degraded = await _gather_named_notes(
        {
            "digest": llm.generate(
                model=cfg.models.report_neutral_model,
                prompt=_build_debate_digest_prompt(
                    bull_notes=analyst_notes,
                    bear_notes=auditor_notes,
                    bull_rebuttal=r2_texts.get("bull_rebuttal", ""),
                    bear_rebuttal=r2_texts.get("bear_rebuttal", ""),
                    disagreement_text=disagreement_text,
                    max_chars=digest_chars,
                ),
                system_instruction=_DEBATE_NEUTRAL_SYSTEM,
                thinking_level=cfg.models.report_neutral_thinking_level,
                temperature=0.1,
                usage_label="report_debate_digest",
                round_timeout_seconds=neutral_timeout,
                max_output_tokens=2048,
            ),
            "factcheck": llm.generate(
                model=cfg.models.report_neutral_model,
                prompt=_build_factcheck_prompt(
                    fact_sheet=fact_sheet,
                    bull_rebuttal=r2_texts.get("bull_rebuttal", ""),
                    bear_rebuttal=r2_texts.get("bear_rebuttal", ""),
                    max_chars=factcheck_chars,
                ),
                system_instruction=_DEBATE_NEUTRAL_SYSTEM,
                thinking_level=cfg.models.report_neutral_thinking_level,
                temperature=0.0,
                usage_label="report_debate_factcheck",
                round_timeout_seconds=neutral_timeout,
                max_output_tokens=1024,
            ),
        },
        timeout_s=r3_budget,
    )
    result.digest = trim_text_at_line_boundary(r3_texts.get("digest", ""), digest_chars) if r3_texts.get("digest") else ""
    result.factcheck = (
        trim_text_at_line_boundary(r3_texts.get("factcheck", ""), factcheck_chars)
        if r3_texts.get("factcheck")
        else ""
    )
    payload["r3"] = {"digest": result.digest, "factcheck": result.factcheck}
    payload["elapsed"]["r3_s"] = round(loop.time() - r3_started, 1)
    if r3_degraded:
        payload["degraded"].extend(r3_degraded)
        await _notify_debate_degraded(ops_notifier, r3_degraded)
    return result


def _evaluate_deployment_gate_safely(
    cfg: AppConfig,
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame],
    macro: MacroSnapshot,
    tech_map: dict[str, TechnicalSnapshot],
    gate_histories: dict[str, pd.DataFrame],
    gate_candidates: list[dict[str, Any]],
    prev_gate_states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Deterministic gate evaluation that must never break the /report pipeline."""
    try:
        return evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=cfg.strategy,
            bundles=bundles,
            histories={**gate_histories, **histories},
            macro=macro,
            tech_map=tech_map,
            candidates=gate_candidates,
            universe_etf_symbols=cfg.universe.etf_symbols,
            prev_gate_states=prev_gate_states,
            gbp_usd=getattr(macro, "gbp_usd", None),
        )
    except Exception as exc:  # noqa: BLE001 - gate failure must never block the report
        logger.warning("Deployment gate evaluation failed, banner omitted: %s", exc)
        return None


def _build_report_recommendations(
    decision: MorningReportDecision,
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    cfg: AppConfig,
) -> list[dict[str, Any]]:
    """Turn /report ADD/TRIM/STOP decisions into the same T+N ledger as /pick.

    HOLD/CASH are intentionally excluded. Report actions are marked source='report' and
    fill_status='ASSUMED' because price_zone may be a trigger zone rather than a fill price.
    """
    price_by_sym = {
        p.yahoo_symbol.upper(): float(p.cur_price)
        for p in portfolio.all_positions()
        if p.yahoo_symbol and float(getattr(p, "cur_price", 0.0) or 0.0) > 0
    }
    account_ok = {
        "ISA": bool(portfolio.isa.ok and not portfolio.isa.is_fallback),
        "INVEST": bool(portfolio.invest.ok and not portfolio.invest.is_fallback),
    }
    out: list[dict[str, Any]] = []
    for act in decision.top_actions:
        action = str(act.action or "").strip().upper()
        if action not in {"ADD", "TRIM", "STOP"}:
            continue
        sym = str(act.symbol or "").strip().upper()
        if not sym or sym == "CASH":
            continue
        acct = str(act.account or "").strip().upper()
        if acct == "ISA" and not account_ok["ISA"]:
            continue
        if acct == "INVEST" and not account_ok["INVEST"]:
            continue
        bundle = bundles.get(sym) or bundles.get(sym.split(".")[0])
        price = None
        if bundle is not None and bundle.current_price is not None and float(bundle.current_price) > 0:
            price = float(bundle.current_price)
        elif sym in price_by_sym:
            price = price_by_sym[sym]
        if price is None or price <= 0:
            continue
        out.append(
            {
                "account": act.account,
                "symbol": sym,
                "action": action,
                "entry_price": round(price, 4),
                "thesis": f"{act.reason} | {act.price_zone}"[:300],
                "horizon_days": max(1, int(cfg.strategy.settlement_holding_days)),
                "benchmark_symbol": resolve_market_benchmark(sym)[0],
                "source": "report",
                "fill_status": "ASSUMED",
            }
        )
    return out


async def _enrich_gate_candidate_bundles(
    market_data: MarketDataService,
    pool_syms: list[str],
    pool_snaps: dict[str, Any],
    bundles: dict[str, TickerMarketBundle],
    *,
    tolerance_pct: float = 1.5,
    max_symbols: int = 6,
) -> dict[str, TickerMarketBundle]:
    """Attach lightweight earnings/ex-div bundles to un-held MA50 pullback candidates.

    Deployment-gate A-trigger scans un-held candidates only, so they normally have no
    bundle; without this the 5-day earnings and ex-dividend exclusion windows are blind.
    One batched call, news disabled (0 LLM, earnings calendar only).
    """
    cand_bundles: dict[str, TickerMarketBundle] = dict(bundles)
    hit_syms = [
        sym
        for sym in pool_syms
        if sym in pool_snaps
        and getattr(pool_snaps[sym], "dist_ma50_pct", None) is not None
        and abs(float(pool_snaps[sym].dist_ma50_pct)) <= float(tolerance_pct)
        and sym not in bundles
    ][: max(0, int(max_symbols))]
    if not hit_syms or not callable(getattr(market_data, "get_bundles", None)):
        return cand_bundles
    try:
        fetched = await market_data.get_bundles(
            hit_syms, include_news=False, include_earnings=True
        )
        fetched_map = fetched[1] if isinstance(fetched, tuple) else fetched
        if isinstance(fetched_map, dict):
            for sym, bundle in fetched_map.items():
                if bundle is not None:
                    cand_bundles[sym] = bundle
    except Exception as exc:
        logger.debug("Deployment-gate candidate bundle fetch skipped: %s", exc)
    return cand_bundles


async def generate_morning_report(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """TradingAgents-Lite 3-step pipeline:
    1) Deterministic FactSheet + ContextPack + Concurrent Multi-Region Google Search Grounding
    2) Concurrent Analyst (Bull/Opportunity, HIGH, <=6 tool rounds) & Risk Auditor (Bear, Argon HIGH, <=6 tool rounds)
    3) Portfolio Manager Pydantic Structured Synthesis + RiskOverride Guardrail -> <= 1800 chars
    """
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 双账户数据均读取失败：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    tracked_syms = portfolio.unique_yahoo_symbols(include_pending=True)
    held_syms = portfolio.unique_yahoo_symbols(include_pending=False)

    prev_snap, past_lessons, fact_bundle = await asyncio.gather(
        db.get_previous_snapshot(min_age_hours=4.0),
        db.get_past_decision_context(limit=5),
        _assemble_fact_sheet_bundle(portfolio, market_data, cfg, us_only=False, db=db),
    )
    (
        fact_sheet,
        tech_map,
        bundles,
        macro,
        histories,
        evals,
        _disagreements,
        disagreement_text,
        calendar_alerts,
        context_pack,
    ) = fact_bundle

    # ---------- v5.4.0 deployment gate data pre-fetch (0 LLM calls, offline-tolerant) ----------
    gate_mode = str(getattr(cfg.strategy, "deployment_gate_mode", "off") or "off").strip().lower()
    gate_histories: dict[str, pd.DataFrame] = {}
    gate_candidates: list[dict[str, Any]] = []
    prev_gate_states: list[dict[str, Any]] = []
    if gate_mode != "off":
        regime_bench = str(getattr(cfg.strategy, "regime_benchmark_symbol", "SPY") or "SPY").strip().upper()
        core_bench = str(getattr(cfg.strategy, "benchmark_symbol", "VWRP.L") or "VWRP.L").strip().upper()
        etf_upper = {s.strip().upper() for s in (cfg.universe.etf_symbols or []) if s} | {core_bench}
        active_pool_syms: list[str] = []
        if callable(getattr(db, "get_active_universe_pool", None)):
            try:
                active_rows = await db.get_active_universe_pool()
                for row in active_rows or []:
                    if isinstance(row, dict) and not row.get("is_etf"):
                        sym_r = str(row.get("symbol") or "").strip()
                        if sym_r and sym_r.upper() not in etf_upper:
                            active_pool_syms.append(sym_r)
                            if len(active_pool_syms) >= 30:
                                break
            except Exception as exc:
                logger.debug("Deployment-gate active universe lookup skipped: %s", exc)
        pool_syms = list(
            dict.fromkeys(
                [
                    s
                    for s in (
                        *cfg.universe.invest_tactical,
                        *active_pool_syms,
                        *cfg.universe.isa_long_term,
                        *cfg.watchlist,
                    )
                    if s and s.strip().upper() not in etf_upper
                ]
            )
        )
        gate_syms = list(dict.fromkeys([regime_bench, core_bench, "^VIX", *pool_syms]))
        if callable(getattr(market_data, "get_histories", None)):
            try:
                fetched = await market_data.get_histories(gate_syms, period="2y")
                if isinstance(fetched, dict):
                    gate_histories = {k: v for k, v in fetched.items() if v is not None and not v.empty}
            except Exception as exc:
                logger.debug("Deployment-gate history pre-fetch failed; continuing without A-trigger scan: %s", exc)
        pool_snaps: dict[str, Any] = {}
        for sym in pool_syms:
            hist = gate_histories.get(sym)
            if hist is None or hist.empty:
                continue
            try:
                snap = compute_technical_snapshot(sym, hist, bundle=bundles.get(sym))
            except Exception as exc:
                logger.debug("Deployment-gate tech snapshot failed for %s: %s", sym, exc)
                continue
            if snap is not None:
                pool_snaps[sym] = snap
        # The 5-day earnings / ex-dividend mines only work when the un-held gate
        # candidates carry a bundle; fetch one lightweight batch for the MA50 pullbacks.
        cand_bundles = await _enrich_gate_candidate_bundles(
            market_data,
            pool_syms,
            pool_snaps,
            bundles,
            tolerance_pct=float(getattr(cfg.strategy, "trigger_ma_tolerance_pct", 1.5) or 1.5),
        )
        for sym in pool_syms:
            snap = pool_snaps.get(sym)
            if snap is not None:
                gate_candidates.append(
                    {"symbol": sym, "tech": snap, "bundle": cand_bundles.get(sym), "account": "Invest"}
                )
        if hasattr(db, "get_recent_context_packs"):
            try:
                for pack in await db.get_recent_context_packs(limit=5):
                    gate_prev = pack.get("deployment_gate") if isinstance(pack, dict) else None
                    if isinstance(gate_prev, dict) and gate_prev.get("regime"):
                        prev_gate_states.append(gate_prev)
            except Exception as exc:
                logger.debug("Deployment-gate previous-state lookup skipped: %s", exc)

    # Feed the latest GBP/USD mid so subsequent Invest wallet splits can be derived locally.
    if getattr(macro, "gbp_usd", None):
        set_rate = getattr(t212, "set_gbp_usd_rate", None)
        if callable(set_rate):
            set_rate(macro.gbp_usd)

    priority_symbols: list[str] = []
    for sym in held_syms:
        b = bundles.get(sym)
        t = tech_map.get(sym)
        near_earnings = (
            b is not None
            and b.days_to_earnings is not None
            and 0 <= int(b.days_to_earnings) <= 7
        )
        ex_days = (
            b.days_to_ex_dividend
            if (b is not None and b.days_to_ex_dividend is not None)
            else (t.days_to_ex_dividend if t is not None else None)
        )
        near_ex_div = ex_days is not None and -3 <= int(ex_days) <= 3
        move_1d = (
            t.ret_1d_pct
            if (t is not None and t.ret_1d_pct is not None)
            else _daily_move_pct_from_hist(histories.get(sym))
        )
        big_move = move_1d is not None and abs(float(move_1d)) >= 3.0
        if near_earnings or near_ex_div or big_move:
            priority_symbols.append(sym)

    quote_types = {
        sym: bundles[sym].quote_type
        for sym in tracked_syms
        if sym in bundles and bundles[sym].quote_type
    }
    grounding_per_symbol: dict[str, str] = {}
    grounding_brief = (
        await run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.report_search_model,
            scope="report",
            symbols=tracked_syms,
            quote_types=quote_types or None,
            priority_symbols=priority_symbols or None,
            priority_search_model=cfg.models.report_priority_search_model,
            priority_thinking_level=cfg.models.report_priority_search_thinking_level,
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
            db=db,
            persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
            per_symbol_out=grounding_per_symbol,
        )
        if cfg.models.enable_forced_search_grounding
        else ""
    )
    if grounding_per_symbol and isinstance(context_pack.get("symbols"), dict):
        for g_sym, g_text in grounding_per_symbol.items():
            entry = context_pack["symbols"].get(g_sym)
            if isinstance(entry, dict) and g_text:
                entry["grounding_news"] = trim_text_at_line_boundary(g_text, 320)

    trade_ctx = build_trade_context(
        portfolio,
        history_days=cfg.strategy.order_history_days,
        max_recent_lines=10,
        max_summary_lines=10,
        bundles=bundles,
        tech_map=tech_map,
        histories=histories,
        cooldown_days=cfg.strategy.rebuy_cooldown_days,
    )
    benchmark_line = _format_benchmark_line(
        cfg.strategy.benchmark_symbol,
        _lookup_benchmark_hist(histories, cfg.strategy.benchmark_symbol),
        account_change_pct=_account_change_pct(portfolio, prev_snap),
    )
    context_pack["benchmark_dashboard"] = benchmark_line
    cash_flow_text = trade_ctx.get("cash_flow_summary_text") or "（近30天无外部出入金或派息流水）"
    diff_text = _compute_portfolio_diff_text(
        portfolio,
        prev_snap,
        cash_flow_summary_text=trade_ctx.get("cash_flow_summary_text", ""),
    )
    context_pack["cash_flow_summary"] = cash_flow_text
    context_pack["portfolio_diff"] = diff_text

    us_sched = get_us_market_schedule_uk()
    uk_bank_holiday = is_uk_bank_holiday()
    now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
    dst_note = (
        f"今日美股 {us_sched.open_time_uk} UK 开盘（错位周）"
        if us_sched.is_dst_mismatch
        else f"今日美股 {us_sched.open_time_uk} UK 开盘"
    )
    if us_sched.phase in {"HOLIDAY", "WEEKEND"}:
        dst_note = "今日美股休市"
    if uk_bank_holiday:
        dst_note += " | 🇬🇧 今日英国 Bank Holiday（LSE 休市，英股/LSE ETF 暂不成交，按下一开盘日预案）"
    extra_alerts = [a for a in calendar_alerts if "英国 Bank Holiday" not in a]
    if extra_alerts:
        dst_note += " | " + "；".join(extra_alerts)

    wallet_text = build_wallet_context_text(portfolio, cfg.strategy)
    shared_context = (
        f"时间：英国时间 {now_uk}（{get_uk_weekday_label()} | {dst_note}）\n\n"
        f"{wallet_text}\n\n"
        f"=== 一、双账户状态 ===\n"
        f"【Stocks ISA】\n{format_account_overview(portfolio.isa)}\n持仓：\n{format_positions_text(portfolio.isa)}\n\n"
        f"【Invest】\n{format_account_overview(portfolio.invest)}\n持仓：\n{format_positions_text(portfolio.invest)}\n\n"
        f"【净值变动与出入金归因】{diff_text}\n"
        f"【基准对照（大盘 vs 本账户）】{benchmark_line}\n"
        f"【近{cfg.strategy.order_history_days}天出入金与派息流水（区分外部资金与交易盈亏）】\n{cash_flow_text}\n\n"
        f"【当前活动挂单（含市场现价/目标价）】\n{trade_ctx['pending_orders_text']}\n\n"
        f"【近7天实盘成交】\n{trade_ctx['recent_trades_text']}\n\n"
        f"【近{cfg.strategy.order_history_days}天行为统计】\n{trade_ctx['trade_summary_30d_text']}\n\n"
        f"【交易频率与冷却（Python 计算）】{trade_ctx['trade_behavior_text']}\n\n"
        f"=== 二、量化事实底座 ===\n{fact_sheet}"
        + (f"\n\n=== 三、实时 Google Search 全市场（英/欧/美/ETF/宏观）深度新闻 ===\n{grounding_brief}" if grounding_brief else "")
    )

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.report_search_model,
        ops_notifier=ops_notifier,
        db=db,
        allow_search_tool=False,
        search_cache_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
    )

    # ---------- Step 2A: Opportunity & Portfolio Analyst ----------
    analyst_sys = build_agent_system_instruction(
        "你是组合机会分析师（Bull/Opportunity 视角）。请基于用户提供的唯一真实事实数据，用干货要点输出完整内部研报草稿供 PM 拍板（1200 字内）：\n"
        "1. 宏观利率/汇率主线与今日整体交易气候；\n"
        "2. 双账户净值、FX 拆解与近 7/30 天实盘复盘要点（严格区分外部出入金/派息与交易盈亏）；\n"
        "3. 哪些持仓具备真实技术支撑（回踩 MA50/MA200 且基本面未坏）值得考虑加仓或顺势持有（写明支撑位/目标价与依据）；\n"
        f"4. 资金部署：若 ISA 现金占比 >{cfg.strategy.isa_cash_deploy_trigger_pct:.0f}% 且核心宽基未达标，"
        "给出核心宽基分批归拢价位与金额；Invest 机动仓严格遵循右侧猎物门控，"
        "无 R:R≥2:1 强信号时明确维持持币待命，严禁因现金占比高而勉强凑单；\n"
        "5. 近 7 日财报事件应对预案。\n"
        "切勿平铺常规正常持仓，仅点名关键标的；以上 FactSheet 与 Step 1 全市场 Google Search 摘要已涵盖全部持仓及挂单的量价、CYQ、目标价、财报日、深度情报与最新新闻，"
        "严禁对已列出数据的持仓重复调用 `inspect_ticker`；仅当文中明确缺失某未入表新代码数据时才允许调用工具。",
        cfg.strategy,
    )
    analyst_prompt = f"【任务：组合机会分析师（Bull 视角，深度推理并保留关键数字/支撑位，1200 字内）】\n{shared_context}"

    # ---------- Step 2B: Ruthless Risk & Pending-Order Auditor ----------
    auditor_sys = build_agent_system_instruction(
        "你是独立风控与挂单审计官（Bear/Conservative 视角）。你的唯一任务是冷酷挑刺与风险一票否决，提供完整内部审计草稿供 PM 拍板（1200 字内）：\n"
        "1. 行业集中度与跨账户重叠审计：基于 Python 算出的行业权重、CVaR 与跨账户重叠，明确哪些行业/标的是否已超载、今天必须冻结禁止加码（列出红线）。\n"
        "2. 破位与弱势持仓审计：跌破 MA200 或触及止损线（尤其 Invest 短线仓）或临近 5 日内财报的标的，明确减仓/硬止损/严禁补仓指令；"
        "所有持仓的技术面、Finnhub 新闻与 Step 1 全市场 Google Search 摘要已完整注入上方上下文，常规持仓严禁重复调 `inspect_ticker`，直接依据既有证据严格审计即可。\n"
        "3. 【活动挂单零情面独立审计（一律按挂单限价核算）】：逐一核查【当前活动挂单】！"
        "无论挂单是用户手动下单还是前序 AI 建议，都以该挂单的限价（而非现价）为成本基线，结合均线位置、分析师目标价、"
        f"{cfg.strategy.uk_stamp_duty_pct:.1f}% 印花税/{cfg.strategy.fx_fee_pct * 2:.2f}% 往返换汇费与买卖价差估算，核算成交后 R:R；"
        f"挂单占用资金在未成交前照常计息，距现价 -3%~-12% 且基本面未坏的回踩埋伏单属有效等待，"
        "严禁以“释放冻结现金吃息”或“现价空间不足”为由撤单；"
        "仅当按限价成交后 R:R <2:1、限价明显高于合理回踩区或基本面/技术面已证伪时，必须敢于裁决【CANCEL 建议撤单】或【MODIFY 建议下调限价至具体数字】！\n"
        # [用户授权调整] 1200 字草稿集中用于异动/分歧/挂单；IMP-12 请保留。
        "4. 篇幅聚焦：常规无异常 HOLD 持仓合并略过，把 1200 字集中分配给异动标的、分歧标的与活动挂单。",
        cfg.strategy,
    )
    auditor_prompt = f"【任务：独立风控与挂单审计官（Bear 视角，深度审计并保留完整依据，1200 字内）】\n{shared_context}"

    report_rounds = max(1, int(cfg.models.report_max_tool_rounds))
    debate = await _run_step2_debate(
        cfg=cfg,
        llm=llm,
        toolkit=toolkit,
        shared_context=shared_context,
        fact_sheet=fact_sheet,
        disagreement_text=disagreement_text,
        analyst_sys=analyst_sys,
        auditor_sys=auditor_sys,
        analyst_prompt=analyst_prompt,
        auditor_prompt=auditor_prompt,
        report_rounds=report_rounds,
        ops_notifier=ops_notifier,
    )
    analyst_notes, auditor_notes = debate.analyst_notes, debate.auditor_notes
    degraded_roles = debate.degraded_roles
    if degraded_roles and ops_notifier is not None:
        await ops_notifier.notify(
            severity="WARN",
            source="report:step2",
            alert_key="report_step2_degraded",
            message=(
                f"/report Step 2 降级运行：{'/'.join(degraded_roles)} 未返回有效结论"
                f"（首轮预算 {cfg.models.report_step2_round1_seconds:.0f}s），已用中性占位继续 PM 拍板。"
            ),
        )

    # ---------- Step 3: Portfolio Manager Structured Synthesis ----------
    debate_pm_block = _build_debate_pm_block(debate.digest, debate.factcheck)
    pm_role = (
        "你是最终投资组合经理（Portfolio Manager），负责拍板权衡「机会分析师（Bull）」与「独立风控审计官（Bear）」的冲突并输出结构化今日总纲（严格遵守 JSON Schema）。\n"
        "严格去重与排版准则（违反将导致重要信息被截断）：\n"
        "• 对【本地多空证据冲突对照】中列出的标的，必须正面裁决采纳 Bull 还是 Bear 依据，严禁和稀泥。\n"
        "• `top_actions`【只放现有持仓的 ADD/TRIM/STOP/HOLD】（1~3条）；所有活动挂单的撤改留在 `pending_order_audits`，严禁把 CANCEL_ORDER/MODIFY_ORDER 重复写进 `top_actions`！\n"
        "• `size_hint` 必填数量口径：TRIM 写减仓比例（如 30%、1/3）或英镑金额（如 £2,300），ADD 写加仓金额或比例，STOP 写「全出」，HOLD/CASH 留空；严禁写股数（由 Python 按持仓确定性换算）。\n"
        "• `pending_order_audits` 逐一简洁裁决 KEEP/MODIFY/CANCEL，不要和上文重复抄写理由。\n"
        "• `flagged_positions` 只列出【未在 top_actions 或 pending_order_audits 中出现过】的其他异动/观察持仓（如暴跌、财报临近但暂不操作者），严禁再复述上方已给过动作的标的！\n"
        "• `daily_vetoes` 将写入数据库强制约束今日后续 `/pick` 和 `/us`，只写简洁可执行的红线（2~4条，每条<=35字）。\n"
        # [用户授权调整] /report 目标篇幅调至 1500~1650 字、安全天花板 1800 字；IMP-12 请保留。
        "• 全文必须精炼直白，控制在 1500~1650 字左右（绝不可超 1800 字），确保财报预警与操作复盘完整展示。"
    )
    if debate_pm_block:
        pm_role += (
            "\n• 【第二轮交叉质询与中立索引】只是索引，不是裁决；仍须正面裁决 Bull 与 Bear，"
            "索引与首轮事实冲突时以首轮与本地 FactSheet 为准；"
            "被标记 [无据] 的数字不得作为加减仓的主要依据。"
        )
    pm_sys = build_agent_system_instruction(pm_role, cfg.strategy)
    calendar_block = (
        "=== 市场日历与休市/半日市提醒 ===\n" + "\n".join(f"• {a}" for a in calendar_alerts) + "\n\n"
        if calendar_alerts
        else ""
    )
    pm_prompt = (
        f"=== 历史决策回顾 (Memory Log) ===\n{past_lessons}\n\n"
        + calendar_block
        + f"=== 账户速览与净值变动 ===\n"
        f"{wallet_text}\n"
        f"净值变动：{diff_text}\n"
        f"基准对照（大盘 vs 本账户）：{benchmark_line}\n"
        f"交易频率与冷却：{trade_ctx['trade_behavior_text']}\n"
        f"资金流水：{cash_flow_text}\n"
        f"活动挂单：\n{trade_ctx['pending_orders_text']}\n\n"
        f"=== 本地多空证据冲突对照 (Bull vs Bear Disagreement Matrix) ===\n{disagreement_text}\n\n"
        f"=== 机会分析师报告 (Bull) ===\n{analyst_notes}\n\n"
        f"=== 独立风控审计官挑刺与挂单审计 (Bear) ===\n{auditor_notes}"
        + debate_pm_block
    )

    # v5.4.0: inject the deterministic gate bottom line into the PM prompt (advise mode only).
    gate_preview: dict[str, Any] | None = None
    if gate_mode == "advise":
        gate_preview = _evaluate_deployment_gate_safely(
            cfg,
            portfolio,
            bundles,
            histories,
            macro,
            tech_map,
            gate_histories,
            gate_candidates,
            prev_gate_states,
        )
    if gate_preview is not None:
        isa_g = gate_preview.get("isa") or {}
        inv_g = gate_preview.get("invest") or {}
        pm_prompt += (
            "\n\n=== 今日组合门控（Python 确定性底线；PM 仅可单向收紧） ===\n"
            f"市场状态 {gate_preview.get('regime')}｜ISA {isa_g.get('status')}（{isa_g.get('reason')}）"
            f"｜Invest {inv_g.get('status')}（{inv_g.get('reason')}）\n"
            "• 仅在发现重大宏观/事件风险时把 pick_gate.allow 置 false 收紧；"
            "严禁把 WAIT/DEFEND 升为 RUN；ROTATE 必须给出合法 funding_from 置换源。"
        )

    raw_pm = await llm.generate(
        model=cfg.models.report_model,
        prompt=pm_prompt,
        system_instruction=pm_sys,
        thinking_level=cfg.models.report_thinking_level,
        response_schema=MorningReportDecision,
        temperature=0.15,
        usage_label="report_pm",
        round_timeout_seconds=cfg.models.report_pm_timeout_seconds,
    )

    decision, resolved_pm_raw = await parse_or_repair_structured_json(
        raw_pm,
        MorningReportDecision,
        llm=llm,
        repair_model=cfg.models.fallback_model,
        thinking_level="LOW",
        usage_label="report_pm_repair",
        ops_notifier=ops_notifier,
        alert_source="report_pm",
    )
    holiday_banner = "🇬🇧 今日英国 Bank Holiday（LSE 休市，英股/LSE ETF 暂不交易）\n\n" if uk_bank_holiday else ""
    # [用户授权调整] /report 安全天花板由 1500 放宽至 1800（Prompt 控制 1500~1650 字）；IMP-12 接线时请保留 1800 上限。
    max_body_chars = 1800 - len(holiday_banner)
    if decision is not None:
        override_res = apply_risk_override_guardrail(
            decision,
            portfolio=portfolio,
            macro=macro,
            strategy=cfg.strategy,
            bundle_map=bundles,
            evals=evals,
            trade_behavior=trade_ctx.get("trade_behavior"),
        )
        resolved_decision = override_res.decision if isinstance(override_res.decision, MorningReportDecision) else decision

        # v5.4.0: evaluate the deterministic gate AFTER the guardrail, reconcile the PM pick_gate
        # one-way, persist to the context pack and (in advise mode) render the banner.
        # Zero extra LLM calls.
        gate_dict: dict[str, Any] | None = None
        gate_banner = ""
        if gate_mode != "off":
            gate_dict = _evaluate_deployment_gate_safely(
                cfg,
                portfolio,
                bundles,
                histories,
                macro,
                tech_map,
                gate_histories,
                gate_candidates,
                prev_gate_states,
            )
            if gate_dict is not None:
                pm_proposal = getattr(resolved_decision, "pick_gate", None)
                funding_whitelist = (
                    set(gate_dict.get("invest", {}).get("unfinanced_positions") or [])
                    | set(gate_dict.get("invest", {}).get("pending_new_equity_buys") or [])
                    | set(gate_dict.get("invest", {}).get("pending_add_equity_buys") or [])
                )
                gate_dict["pm"] = reconcile_pm_pick_gate(
                    gate_dict,
                    pm_proposal.model_dump() if pm_proposal is not None else None,
                    whitelist_symbols=funding_whitelist,
                )
                gate_dict["banner"] = format_deployment_gate_banner(gate_dict)
        if gate_dict is not None:
            context_pack["deployment_gate"] = gate_dict
            logger.info(
                "Deployment gate [%s]: regime=%s global=%s isa=%s invest=%s",
                gate_mode,
                gate_dict.get("regime"),
                gate_dict.get("global_status"),
                gate_dict.get("isa", {}).get("status"),
                gate_dict.get("invest", {}).get("status"),
            )
            if gate_mode == "advise":
                gate_banner = str(gate_dict.get("banner") or "")

        context_pack["debate"] = debate.payload
        context_pack["report_decision"] = {
            "headline": resolved_decision.headline,
            "top_actions": [a.model_dump() for a in resolved_decision.top_actions],
            "pending_order_audits": [a.model_dump() for a in resolved_decision.pending_order_audits],
            "daily_vetoes": list(resolved_decision.daily_vetoes),
        }
        final_report = holiday_banner + render_morning_report(
            resolved_decision,
            max_chars=max_body_chars,
            risk_override=override_res,
            gate_banner=gate_banner,
            size_notes=build_report_size_notes(resolved_decision.top_actions, portfolio),
        )
        guardrails = list(resolved_decision.daily_vetoes)
        actions_summary = "; ".join(
            f"[{a.account}] {a.action} {a.symbol}({a.price_zone})" for a in resolved_decision.top_actions[:4]
        )
        short_conclusion = f"{resolved_decision.headline} | 重点动作: {actions_summary}"
    else:
        logger.warning("Morning PM structured parse failed; refusing to persist a degraded SOT.")
        return (
            "⚠️ /report PM 结构化解析失败，未写入当日基线（避免用半成品晨报锁死当天门控）。"
            "请稍后重试 /report。"
        )

    if not portfolio.is_fallback:
        await db.save_snapshot(portfolio)
        # IMP-17: one lightweight GBP NAV row per day feeds the TWR performance loop.
        try:
            bench_symbol = cfg.strategy.benchmark_symbol
            bench_hist = histories.get(bench_symbol) if histories else None
            bench_close = None
            if bench_hist is not None and not getattr(bench_hist, "empty", True):
                close_col = "Adj Close" if "Adj Close" in bench_hist.columns else "Close"
                if close_col in bench_hist.columns:
                    bench_close = float(bench_hist[close_col].dropna().iloc[-1])
            await db.upsert_nav_history(build_nav_row(portfolio, benchmark_close=bench_close))
        except Exception as exc:  # never break /report on a performance-ledger hiccup
            logger.warning("nav_history upsert skipped: %s", exc)
    else:
        logger.warning("Skipping save_snapshot because portfolio is fallback data (%s)", portfolio.fallback_note)

    if not final_report.startswith("⚠️"):
        await db.save_morning_report(
            final_report,
            short_conclusion,
            guardrails=guardrails,
            context_pack=context_pack,
        )
        try:
            report_recs = _build_report_recommendations(resolved_decision, portfolio, bundles, cfg)
            if report_recs:
                # 5-day same (account, symbol, action) dedupe: repeating "TRIM X" every
                # morning would flood the T+20 ledger with 10 highly-correlated records.
                try:
                    recent = await db.get_recent_recommendations(limit=200)
                    cutoff_ms = int(time.time() * 1000) - 5 * 86_400_000
                    recent_keys = {
                        (
                            str(r.get("account") or "").strip().upper(),
                            str(r.get("symbol") or "").strip().upper(),
                            str(r.get("action") or "").strip().upper(),
                        )
                        for r in recent
                        if int(r.get("created_at_ms") or 0) >= cutoff_ms
                    }
                    report_recs = [
                        rec
                        for rec in report_recs
                        if (
                            str(rec.get("account") or "").strip().upper(),
                            str(rec.get("symbol") or "").strip().upper(),
                            str(rec.get("action") or "").strip().upper(),
                        )
                        not in recent_keys
                    ]
                except Exception as exc:
                    logger.debug("report recommendation 5d dedupe lookup skipped: %s", exc)
            if report_recs:
                await db.log_recommendations(
                    report_recs,
                    default_horizon_days=cfg.strategy.settlement_holding_days,
                    source="report",
                    dedupe_same_day=True,
                )
        except Exception as exc:  # the report must never fail because the ledger hiccuped
            logger.warning("report recommendation ledger write skipped: %s", exc)
        # Read-only semantic linter (DeepSeek sidecar): never rewrites or blocks the report text.
        await audit_generated_output(
            llm,
            cfg=cfg,
            scope="report",
            fact_baseline=build_fact_baseline(
                scope="report",
                portfolio=portfolio,
                evals=evals,
                trade_ctx=trade_ctx,
                guardrails=guardrails,
                fact_sheet=fact_sheet,
            ),
            rendered_text=final_report,
            strategy=cfg.strategy,
            ops_notifier=ops_notifier,
        )
    return final_report


def _daily_move_pct_from_hist(hist: pd.DataFrame | None) -> float | None:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
    closes = closes[closes > 0]
    if len(closes) < 2:
        return None
    prev = float(closes.iloc[-2])
    last = float(closes.iloc[-1])
    if prev <= 0:
        return None
    return ((last / prev) - 1.0) * 100.0


def _evaluate_us_anomaly_gate(
    *,
    portfolio: PortfolioData,
    cfg: AppConfig,
    histories: dict[str, pd.DataFrame],
    bundles: dict[str, TickerMarketBundle],
    macro: MacroSnapshot,
    vix_hist: pd.DataFrame | None,
    morning_context_pack: dict[str, Any] | None,
    realtime_quotes: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Evaluate local 0-LLM Anomaly Gate for /us sentinel (moves, order proximity, VIX delta)."""
    move_threshold = float(cfg.strategy.us_anomaly_move_pct)
    vix_chg_threshold = float(cfg.strategy.us_anomaly_vix_change_pct)
    quotes = realtime_quotes or {}

    anomaly_syms: list[str] = []
    reasons: list[str] = []

    us_held_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=False) if _is_us_symbol(s)]
    for sym in us_held_syms:
        move_pct: float | None = None
        quote = quotes.get(sym)
        if quote:
            q_pct = quote.get("change_pct")
            if q_pct is not None:
                move_pct = float(q_pct)
            else:
                q_cur = quote.get("current_price")
                q_prev = quote.get("prev_close")
                if q_cur and q_prev and float(q_prev) > 0:
                    move_pct = ((float(q_cur) / float(q_prev)) - 1.0) * 100.0
        if move_pct is None:
            move_pct = _daily_move_pct_from_hist(histories.get(sym))

        if move_pct is not None and abs(move_pct) >= move_threshold:
            anomaly_syms.append(sym)
            reasons.append(f"{sym} 日内/最新日线涨跌 {move_pct:+.2f}%（达异动阈值 ±{move_threshold:.1f}%）")

    for order in portfolio.all_pending_orders():
        sym = order.yahoo_symbol
        if not sym or not _is_us_symbol(sym):
            continue
        quote = quotes.get(sym) or {}
        b = bundles.get(sym)
        hist_sym = histories.get(sym)
        hist_last = None
        if hist_sym is not None and not hist_sym.empty and "Close" in hist_sym.columns:
            c_ser = pd.to_numeric(hist_sym["Close"], errors="coerce").dropna()
            if not c_ser.empty:
                hist_last = float(c_ser.iloc[-1])
        cur_price = quote.get("current_price") or (b.current_price if b else None) or hist_last
        if cur_price is None or cur_price <= 0:
            continue
        for label, trig in (("限价", order.limit_price), ("止损", order.stop_price)):
            if trig is None or trig <= 0:
                continue
            dist_pct = ((cur_price - trig) / trig) * 100.0
            crossed = (
                (order.side == "BUY" and cur_price <= trig)
                or (order.side == "SELL" and label == "止损" and cur_price <= trig)
                or (order.side == "SELL" and label == "限价" and cur_price >= trig)
            )
            if crossed or abs(dist_pct) <= _PENDING_PROXIMITY_PCT:
                if sym not in anomaly_syms:
                    anomaly_syms.append(sym)
                reasons.append(
                    f"[{order.account}] {sym} {order.side} 挂单{label} {trig:.2f} 临近/触及现价 {cur_price:.2f}（距 {dist_pct:+.2f}%）"
                )

    vix_chg_pct = _daily_move_pct_from_hist(vix_hist)
    if vix_chg_pct is None and macro.vix is not None and morning_context_pack:
        base_vix = (morning_context_pack.get("macro") or {}).get("vix")
        if base_vix is not None and float(base_vix) > 0:
            vix_chg_pct = ((macro.vix / float(base_vix)) - 1.0) * 100.0

    if vix_chg_pct is not None and abs(vix_chg_pct) >= vix_chg_threshold:
        reasons.append(f"VIX 日变动 {vix_chg_pct:+.2f}%（达阈值 ±{vix_chg_threshold:.1f}%）")
    if macro.vix is not None and macro.vix >= _VIX_DANGER_LEVEL:
        reasons.append(f"VIX 处于高危水平 {macro.vix:.2f}（>= {_VIX_DANGER_LEVEL:.0f}）")

    return bool(reasons), anomaly_syms, reasons


async def generate_us_sentinel(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    is_scheduled_premarket: bool = False,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Delta US Sentinel (/us & premarket): local Anomaly Gate -> calm LOW <=450 / anomaly search+MEDIUM <=600."""
    sched = get_us_market_schedule_uk()
    portfolio = await t212.get_portfolio(use_cache=not is_scheduled_premarket)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 数据获取失败，无法执行美股哨兵：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    us_held_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=False) if _is_us_symbol(s)]
    us_tracked_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=True) if _is_us_symbol(s)]

    async def _fetch_vix_hist() -> dict[str, pd.DataFrame]:
        try:
            return await market_data.get_histories(["^VIX"], period="1mo")
        except Exception as exc:
            logger.debug("VIX history fetch failed in /us gate: %s", exc)
            return {}

    async def _fetch_us_quotes() -> dict[str, dict[str, Any]]:
        if not us_tracked_syms:
            return {}
        out: dict[str, dict[str, Any]] = {}
        if getattr(getattr(market_data, "finnhub", None), "enabled", False):
            pairs = await asyncio.gather(
                *(market_data.get_us_realtime_quote(sym) for sym in us_tracked_syms),
                return_exceptions=True,
            )
            for sym, res in zip(us_tracked_syms, pairs):
                if isinstance(res, dict) and res:
                    out[sym] = res

        if sched.phase == "PRE":
            need_prepost = list(us_tracked_syms)
        else:
            need_prepost = [
                sym
                for sym in us_tracked_syms
                if sym not in out
                or out[sym].get("change_pct") is None
                or abs(float(out[sym].get("change_pct") or 0.0)) < 1e-9
            ]

        if need_prepost and hasattr(market_data, "get_us_prepost_quotes"):
            try:
                prepost_map = await market_data.get_us_prepost_quotes(need_prepost)
                for sym, pp in (prepost_map or {}).items():
                    if isinstance(pp, dict) and pp.get("current_price"):
                        out[sym] = {**(out.get(sym) or {}), **pp}
            except Exception as exc:
                logger.debug("US prepost quotes fallback failed in /us sentinel: %s", exc)
        return out

    (
        fact_bundle,
        (morning_report_full, morning_conclusion, morning_stale),
        (guardrails, guardrails_stale),
        morning_pack,
        vix_hist_map,
        realtime_quotes,
    ) = await asyncio.gather(
        _assemble_fact_sheet_bundle(portfolio, market_data, cfg, us_only=True, db=db),
        db.get_morning_context_with_status(include_premarket=False),
        db.get_daily_guardrails_with_status(),
        db.get_context_pack(allow_fallback=False),
        _fetch_vix_hist(),
        _fetch_us_quotes(),
    )
    (
        fact_sheet,
        tech_map,
        bundles,
        macro,
        histories,
        _evals,
        _disagreements,
        _disagreement_text,
        calendar_alerts,
        _context_pack,
    ) = fact_bundle

    has_anomaly, anomaly_syms, anomaly_reasons = _evaluate_us_anomaly_gate(
        portfolio=portfolio,
        cfg=cfg,
        histories=histories,
        bundles=bundles,
        macro=macro,
        vix_hist=vix_hist_map.get("^VIX"),
        morning_context_pack=morning_pack,
        realtime_quotes=realtime_quotes,
    )

    us_grounding = ""
    if has_anomaly and cfg.models.enable_forced_search_grounding:
        search_targets = anomaly_syms or us_held_syms or us_tracked_syms
        if search_targets:
            us_grounding = await run_forced_market_grounding(
                llm=llm,
                search_model=cfg.models.search_model,
                scope="us",
                symbols=search_targets,
                ttl_seconds=cfg.app.market_data_ttl_seconds,
                ops_notifier=ops_notifier,
                db=db,
                persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
                bypass_cache=True,
            )

    trade_ctx = build_trade_context(
        portfolio,
        max_recent_lines=5,
        max_summary_lines=5,
        bundles=bundles,
        pending_us_only=True,
        tech_map=tech_map,
        histories=histories,
    )
    phase_desc = {
        "PRE": f"美股盘前（{sched.open_time_uk} UK 开盘，距开盘 {max(sched.minutes_to_open, 0)} 分钟）",
        "OPEN": f"美股盘中（已开盘 {max(sched.minutes_since_open, 0)} 分钟）",
        "CLOSED": f"美股已收盘（{sched.close_time_uk} UK 收盘）",
        "HOLIDAY": "美股法定休市日",
        "WEEKEND": "周末美股休市",
    }.get(sched.phase, sched.phase)

    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（今日尚未生成 /report 红线）"
    if guardrails_stale:
        veto_block = "（以下为上一交易日旧红线，仅作参考，严禁当作今日裁决）\n" + veto_block
    # [用户授权调整] 随 /report 放宽至 1800 同步放宽基线切片；IMP-12 请保留。
    morning_baseline = morning_report_full[:1800] if morning_report_full else (morning_conclusion or "（今日尚未运行 /report）")
    if morning_stale:
        morning_baseline = "⚠️ 今日 /report 尚未生成，以下为上一交易日基线，严禁当作今日裁决。\n" + morning_baseline
    gate_block = (
        "【本地异动门控：已触发深度复核】\n" + "\n".join(f"• {r}" for r in anomaly_reasons)
        if has_anomaly
        else (
            f"【本地异动门控：市场与美股持仓均平稳】"
            f"（持仓波动均 < ±{cfg.strategy.us_anomaly_move_pct:.1f}%，VIX 变动 < ±{cfg.strategy.us_anomaly_vix_change_pct:.1f}%，无挂单逼近触发）"
        )
    )
    cal_block = "\n" + " | ".join(calendar_alerts) if calendar_alerts else ""

    us_sys = build_agent_system_instruction(
        "你是【美股增量哨兵（Delta Sentinel）】。\n"
        "你的唯一职责：以今日晨报 `/report` 全文为绝对基线，结合最新美股行情/新闻/宏观，做一次开盘前二次核验！\n"
        "禁止对早报已决定的英股或正常无异动美股另起炉灶。\n"
        "严格二选一开头（禁用 Markdown # 和 **）：\n"
        "• 若今日早报之后无颠覆性美股新变量（或休市无新行情 / 本地异动门控平稳）：第一行直接写「✅ 维持早报计划」，"
        "随后仅用 2~3 条简写：美股开盘执行动作确认、止损核对与财报静默期提醒（控制在 380 字以内，绝不超 450 字）。\n"
        "• 若本地异动门控触发且突发新新闻、大幅跳空或宏观剧震真正改变赔率：第一行写「🚨 盘前增量修正」，"
        "直指要修正 /report 哪一条建议 + 最新数据依据 + 调整后指令（控制在 550 字以内，绝不超 600 字）。",
        cfg.strategy,
    )
    prompt = (
        f"【任务：美股增量哨兵】当前时段：{phase_desc}（UK {sched.uk_date} {sched.uk_weekday_label}{cal_block}）。\n"
        f"{gate_block}\n\n"
        f"=== 今日 /report 全文（必须对照逐条核验）===\n{morning_baseline}\n\n"
        f"=== 今日风控红线 ===\n{veto_block}\n\n"
        f"=== 最新美股事实与可用现金 ===\n"
        f"{build_wallet_context_text(portfolio, cfg.strategy, compact=True)}\n"
        f"【美股活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"{fact_sheet}"
        + (f"\n\n=== 异动标的实时 Google Search 归因 ===\n{us_grounding}" if us_grounding else "")
    )

    if has_anomaly:
        toolkit = MarketToolkit(
            market_data,
            portfolio,
            llm=llm,
            search_model=cfg.models.search_model,
            ops_notifier=ops_notifier,
            db=db,
            search_cache_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
        tools = toolkit.declarations()
        tool_executor = toolkit.as_executor()
        us_model = cfg.models.us_anomaly_model
        us_thinking = cfg.models.us_anomaly_thinking_level
    else:
        tools = None
        tool_executor = None
        us_model = cfg.models.us_sentinel_model
        us_thinking = cfg.models.us_thinking_level

    memo = await llm.generate(
        model=us_model,
        prompt=prompt,
        system_instruction=us_sys,
        thinking_level=us_thinking,
        tools=tools,
        tool_executor=tool_executor,
        temperature=0.15,
        usage_label="us_sentinel",
    )
    if not memo.startswith("⚠️"):
        # [用户授权调整] /us 触发异常修正（「🚨」）放宽至 600 字安全天花板、平稳日保持 450 字，并改按换行边界优雅裁剪。
        us_max_chars = 600 if (has_anomaly or memo.lstrip().startswith("🚨")) else 450
        if len(memo) > us_max_chars:
            memo = trim_text_at_line_boundary(memo, us_max_chars)
        await db.append_premarket_note(memo)
    return memo


async def generate_intraday_alert(
    cfg: AppConfig,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    *,
    symbol: str,
    current_price: float,
    move_pct: float,
    order_dist_pct: float | None = None,
    pending_order_text: str = "",
    currency: str = "",
    ops_notifier: OpsNotifier | None = None,
) -> str | None:
    """Event-driven intraday alert: 1-symbol grounding + DeepSeek push/no-push decision (<=250 chars)."""
    if not ds_helper_available(llm, cfg):
        return None
    sym = str(symbol or "").strip().upper()
    if not sym or current_price <= 0:
        return None

    try:
        (morning_text, _), guardrails = await asyncio.gather(
            db.get_morning_context(include_premarket=True),
            db.get_daily_guardrails(),
        )
    except Exception as exc:  # noqa: BLE001 - intraday alerting must never break the scheduler
        logger.warning("Intraday alert context load failed for %s: %s", sym, exc)
        return None

    grounding = ""
    try:
        grounding = await run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.search_model,
            scope="intraday",
            symbols=[sym],
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
            db=db,
            persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Intraday grounding failed for %s: %s", sym, exc)

    morning_slice = trim_text_at_line_boundary(str(morning_text or ""), 1200)
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（无）"
    distance_text = "N/A" if order_dist_pct is None else f"{order_dist_pct:.2f}%"
    numeric = (
        "【Python 实时数值（唯一真源，禁止改写）】\n"
        f"{sym} 现价 {current_price:.2f} {currency or ''} | 日内涨跌 {move_pct:+.2f}% | "
        f"距最近挂单价 {distance_text}\n"
        f"活动挂单：{pending_order_text or '无'}"
    )
    prompt = (
        f"{numeric}\n\n【今日 /report 原文节选】\n{morning_slice or '（今日尚未生成 /report）'}\n\n"
        f"【今日红线】\n{veto_block}\n\n"
        + (f"【盘中联网快讯】\n{grounding}\n\n" if grounding else "")
        + "请判断这次异动是否值得立即打断用户：只在具备可执行意义（趋势破坏、催化确认或与今日红线直接相关）时推送；"
        "正常波动、已被早报覆盖或与红线无关时返回 should_push=false。"
    )
    payload = await call_ds_json(
        cfg,
        llm,
        prompt=prompt,
        system_instruction=(
            "你是盘中异动播报员（只读）。只输出严格 JSON："
            '{"should_push": true/false, "alert_text": "<=250字纯文本"}。'
            "alert_text 必须只使用【Python 实时数值】中的现价/涨跌幅/挂单距离，禁止编造任何新数字，"
            "禁止 Markdown # 与 **；should_push=false 时 alert_text 可为空。"
        ),
        usage_label="intraday_alert",
    )
    if not payload or payload.get("should_push") is not True:
        return None
    text = sanitize_plain_text(str(payload.get("alert_text") or ""), 250)
    if len(text) < 10:
        return None
    return f"⚡️ 盘中异动 | {sym}\n{text}"
