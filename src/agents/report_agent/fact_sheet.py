"""Portfolio diff, benchmark comparison, lesson lookup, context-pack, and FactSheet assembly."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.tools import _format_us_deep_intel
from src.config import AppConfig
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_exposure_summary,
    compute_technical_snapshot,
    compute_window_return_pct,
    format_technical_line,
)
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.calendar_utils import get_market_calendar_alerts
from src.engine.portfolio_rules import (
    DisagreementItem,
    HoldingEvaluation,
    build_disagreement_matrix,
    evaluate_portfolio,
    format_disagreement_matrix_text,
    format_evaluations_text,
)
from src.storage.db import AdvisorDatabase, get_trading_cycle_id

logger = logging.getLogger("src.agents.report_agent")


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("src.agents.report_agent")
    return getattr(facade, name, default) if facade is not None else default

LONDON_TZ = ZoneInfo("Europe/London")
CONTEXT_PACK_VERSION = "1.0"
_PENDING_PROXIMITY_PCT = 1.5
_VIX_DANGER_LEVEL = 28.0
_CORE_DCA_HORIZON_DAYS = 20


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

    Kept to ~60 characters of numbers so the /report 2800-char hard cap is never squeezed.
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
                # v5.5.1 dual-philosophy anchors (ISA value / Invest right-side)
                "fcf_yield_pct": b.fcf_yield_pct,
                "target_high_price": b.target_high_price,
                "target_low_price": b.target_low_price,
                "peg_ratio": b.peg_ratio,
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

    calendar_alerts = _facade_attr("get_market_calendar_alerts", get_market_calendar_alerts)()

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
