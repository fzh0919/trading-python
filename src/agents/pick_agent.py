"""Aligned /pick & Weekly Radar: Inherits /report guardrails -> Screener -> Researcher -> Risk Critic (<=1000 chars)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available
from src.agents.llm_client import LLMBackend
from src.agents.prompts import (
    build_agent_system_instruction,
    build_wallet_context_text,
    format_preference_notes_block,
)
from src.agents.schemas import PickDecision, parse_or_repair_structured_json, render_pick_report
from src.agents.semantic_auditor import audit_generated_output, build_fact_baseline
from src.agents.tools import MarketToolkit, _format_us_deep_intel, run_forced_market_grounding
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.analytics import (
    compute_correlation_matrix,
    compute_exposure_summary,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    PortfolioData,
    T212Client,
    build_trade_context,
    compute_trade_behavior_stats,
    format_account_overview,
)
from src.engine.calendar_utils import get_market_calendar_alerts, get_us_market_schedule_uk
from src.engine.portfolio_rules import (
    HoldingEvaluation,
    apply_risk_override_guardrail,
    evaluate_portfolio,
    format_evaluations_text,
)
from src.engine.screener import ScreenerCandidate, run_quant_screener
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


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


def _lookup_ci(mapping: dict[str, Any], symbol: str) -> Any:
    """Case-insensitive lookup in a symbol-keyed dict."""
    if symbol in mapping:
        return mapping[symbol]
    sym_up = symbol.strip().upper()
    for k, v in mapping.items():
        if k.strip().upper() == sym_up:
            return v
    return None


def _extract_morning_audit_context(
    context_pack: dict[str, Any] | None,
    morning_report_full: str = "",
) -> tuple[str, dict[str, list[str]], list[dict[str, Any]]]:
    """Extract /report top_actions & pending_order_audits for /pick alignment and screener tagging."""
    report_dec = (
        context_pack.get("report_decision")
        if isinstance(context_pack, dict) and isinstance(context_pack.get("report_decision"), dict)
        else {}
    )
    top_actions = [
        item for item in (report_dec.get("top_actions") or []) if isinstance(item, dict)
    ]
    pending_audits = [
        item for item in (report_dec.get("pending_order_audits") or []) if isinstance(item, dict)
    ]

    lines: list[str] = []
    symbol_tags: dict[str, list[str]] = {}

    for act in top_actions:
        sym = str(act.get("symbol") or "").strip().upper()
        acct = str(act.get("account") or "").strip()
        verb = str(act.get("action") or "").strip().upper()
        pz = str(act.get("price_zone") or "").strip()
        reason = str(act.get("reason") or "").strip()
        if not sym or sym == "CASH":
            continue
        pz_part = f" @ {pz}" if pz else ""
        lines.append(f"• [持仓裁决] [{acct}] {verb} {sym}{pz_part} — {reason}")
        if verb in {"TRIM", "STOP", "HOLD"}:
            symbol_tags.setdefault(sym, []).append(f"持仓{verb}({acct})")

    for aud in pending_audits:
        sym = str(aud.get("symbol") or "").strip().upper()
        acct = str(aud.get("account") or "").strip()
        verdict = str(aud.get("verdict") or "").strip().upper()
        summary = str(aud.get("order_summary") or "").strip()
        reason = str(aud.get("reason") or "").strip()
        if not sym:
            continue
        lines.append(f"• [挂单裁决] [{acct}] {sym} ({summary}) →【{verdict}】{reason}")
        if verdict in {"CANCEL", "MODIFY", "KEEP"}:
            symbol_tags.setdefault(sym, []).append(f"挂单{verdict}({acct})")

    if lines:
        return "\n".join(lines), symbol_tags, top_actions

    # Fallback for legacy morning_reports rows without ContextPack.report_decision
    if morning_report_full:
        sec_lines: list[str] = []
        in_core = False
        for raw_line in morning_report_full.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("【今日核心行动与挂单裁决】"):
                in_core = True
                continue
            if in_core and stripped.startswith("【宏观"):
                break
            if in_core and stripped:
                sec_lines.append(stripped)
        if sec_lines:
            return "\n".join(sec_lines)[:900], symbol_tags, top_actions

    return "", symbol_tags, top_actions


_PREFILTER_SYSTEM = (
    "你是量化初筛名单筛选器（ID Filter），只做候选取舍，不写分析、不改任何数值。\n"
    "输入是 Python 量化引擎已算好的候选清单（原始分数、维度、优缺点、形态、重叠标记）。\n"
    "任务：为 ISA 长线赛道与 Invest 短线赛道各保留最值得深挖的标的，淘汰基本面/形态/风险明显更差的。\n"
    "硬约束：只能从输入清单里选择代码，严禁发明新代码；每轨保留不超过 {keep} 只。\n"
    "仅输出严格 JSON：{\"keep_isa\": [\"SYM\"], \"keep_invest\": [\"SYM\"], \"flags\": {\"被淘汰SYM\": \"<=15字原因\"}}，"
    "不要任何解释文字。"
)


def _prefilter_row(candidate: ScreenerCandidate) -> str:
    """Render one 0-token, network-free candidate row for the DeepSeek ID filter."""
    dims = " ".join(f"{k}={float(v):.0f}" for k, v in list(candidate.dimension_pct.items())[:3])
    pros = ";".join(str(x) for x in candidate.pros[:2])
    cons = ";".join(str(x) for x in candidate.cons[:2])
    patterns = ";".join(str(x) for x in candidate.pattern_signals[:2])
    state: list[str] = []
    if candidate.is_already_held:
        state.append("已持有")
    if candidate.overlap_warning:
        state.append("高重叠")
    return (
        f"{candidate.symbol} | {candidate.track} | 分 {candidate.score:.1f} | {dims or '无维度'} | "
        f"{candidate.asset_type}/{candidate.currency or '?'} | 优点 {pros or '-'} | 风险 {cons or '-'} | "
        f"形态 {patterns or '-'} | {'/'.join(state) or '无'}"
    )


def _resolve_keep_candidates(
    raw_keep: Any,
    candidates: list[ScreenerCandidate],
    keep_n: int,
) -> list[ScreenerCandidate]:
    """Map DS-returned tickers back onto the ORIGINAL ScreenerCandidate objects (raw order, capped)."""
    if not isinstance(raw_keep, list) or not candidates:
        return []
    wanted = {str(item or "").strip().upper() for item in raw_keep if str(item or "").strip()}
    if not wanted:
        return []
    kept = [c for c in candidates if c.symbol.strip().upper() in wanted]
    return kept[:keep_n]


async def _prefilter_candidates_with_ds(
    cfg: AppConfig,
    llm: LLMBackend,
    isa_cands: list[ScreenerCandidate],
    invest_cands: list[ScreenerCandidate],
) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
    """Narrow the wide quant funnel with a DeepSeek ID filter, always falling back to raw top-N objects."""
    keep_n = max(1, int(cfg.strategy.pick_prefilter_keep_per_track))
    fallback_isa = isa_cands[:keep_n]
    fallback_invest = invest_cands[:keep_n]
    if not cfg.models.ds_pick_prefilter_enabled or not ds_helper_available(llm, cfg):
        return fallback_isa, fallback_invest
    if len(isa_cands) + len(invest_cands) <= 12:
        return isa_cands, invest_cands

    lines = [
        "【ISA 长线赛道候选】",
        *(_prefilter_row(c) for c in isa_cands),
        "【Invest 短线赛道候选】",
        *(_prefilter_row(c) for c in invest_cands),
    ]
    payload = await call_ds_json(
        cfg,
        llm,
        prompt="\n".join(lines),
        system_instruction=_PREFILTER_SYSTEM.replace("{keep}", str(keep_n)),
        usage_label="pick_prefilter",
    )
    if not payload:
        return fallback_isa, fallback_invest

    keep_isa = _resolve_keep_candidates(payload.get("keep_isa"), isa_cands, keep_n)
    keep_invest = _resolve_keep_candidates(payload.get("keep_invest"), invest_cands, keep_n)
    if not keep_isa and not keep_invest:
        logger.warning("/pick DS prefilter returned no valid symbols; falling back to quant top-%d.", keep_n)
        return fallback_isa, fallback_invest
    flags = payload.get("flags")
    if isinstance(flags, dict) and flags:
        # Never forward淘汰标签 to the main model or Telegram: log only, to avoid polluting the source of truth.
        logger.debug("/pick DS prefilter flags: %s", flags)
    return (keep_isa or fallback_isa), (keep_invest or fallback_invest)


def _format_candidates_with_morning_tags(
    candidates: list[ScreenerCandidate],
    symbol_tags: dict[str, list[str]],
    empty_msg: str,
) -> str:
    if not candidates:
        return empty_msg
    lines: list[str] = []
    for c in candidates:
        line = c.summary_line
        tags = symbol_tags.get(c.symbol.strip().upper())
        if tags:
            line = f"{line} | ⚠️今日早报已裁决: {' / '.join(dict.fromkeys(tags))}"
        lines.append(line)
    return "\n".join(lines)


def _format_lessons_section(
    recent_lessons: list[dict[str, Any]],
    symbol_lessons: dict[str, list[dict[str, Any]]],
    horizon_days: int,
) -> str:
    """Format compact T+N reflection lessons for Researcher & Risk Critic prompts."""
    seen_keys: set[tuple[str, str, str, str]] = set()
    lines: list[str] = []

    def _append_row(row: dict[str, Any]) -> None:
        lesson = str(row.get("reflection_lesson") or "").strip()
        if not lesson:
            return
        sym = str(row.get("symbol") or "").strip().upper()
        acct = str(row.get("account") or "").strip()
        act = str(row.get("action") or "").strip().upper()
        dt = str(row.get("created_at_uk") or "").strip()
        key = (dt, acct, sym, act)
        if key in seen_keys:
            return
        seen_keys.add(key)
        alpha = row.get("alpha_pct")
        raw_ret = row.get("raw_return_pct")
        perf_parts: list[str] = []
        if raw_ret is not None:
            perf_parts.append(f"收益 {float(raw_ret):+.1f}%")
        if alpha is not None:
            perf_parts.append(f"Alpha {float(alpha):+.1f}%")
        perf = " / ".join(perf_parts) or "已结算"
        lines.append(f"• [{dt} {acct}] {act} {sym} ({perf}): {lesson[:180]}")

    for row in recent_lessons:
        _append_row(row)
    for rows in symbol_lessons.values():
        for row in rows:
            _append_row(row)

    if not lines:
        return f"（暂无已结算的 T+{horizon_days} 选股复盘教训；请严格按六重护栏与交易摩擦门槛审核）"
    return "\n".join(lines[:8])


async def _fetch_candidate_symbol_lessons(
    db: AdvisorDatabase,
    symbols: list[str],
    pack_symbols: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Load per-symbol lessons, reusing ContextPack slices when available."""
    out: dict[str, list[dict[str, Any]]] = {}
    to_query: list[str] = []
    for sym in symbols:
        pack_entry = _lookup_ci(pack_symbols, sym) if pack_symbols else None
        if isinstance(pack_entry, dict) and "lessons" in pack_entry:
            lessons = pack_entry.get("lessons") or []
            if lessons:
                out[sym] = list(lessons)
            continue
        to_query.append(sym)

    if not to_query:
        return out

    async def _load_one(sym: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            return sym, await db.get_symbol_lessons(sym, limit=2)
        except Exception as exc:
            logger.debug("Failed to load symbol lessons for %s: %s", sym, exc)
            return sym, []

    results = await asyncio.gather(*(_load_one(sym) for sym in to_query))
    for sym, rows in results:
        if rows:
            out[sym] = rows
    return out


def _build_holding_identity_section(
    tracked_syms: list[str],
    held_bundles: dict[str, TickerMarketBundle],
    isin_map: dict[str, str],
    pack_symbols: dict[str, Any],
) -> str:
    """Build compact identity & ex-dividend anchor lines only for symbols near ex-dividend."""
    lines: list[str] = []
    for sym in tracked_syms:
        b = held_bundles.get(sym)
        pack_entry = _lookup_ci(pack_symbols, sym) if pack_symbols else None
        anchor = ""
        ex_div_date = None
        days_to_ex_div = None
        if b is not None:
            anchor = b.identity_summary or b.build_identity_anchor(isin_override=isin_map.get(sym))
            ex_div_date = b.ex_dividend_date
            days_to_ex_div = b.days_to_ex_dividend
        if not anchor and isinstance(pack_entry, dict):
            anchor = str(pack_entry.get("identity_summary") or "").strip()
        if isinstance(pack_entry, dict) and ex_div_date is None:
            bsum = pack_entry.get("bundle_summary") or {}
            if isinstance(bsum, dict):
                ex_div_date = bsum.get("ex_dividend_date")
                days_to_ex_div = bsum.get("days_to_ex_dividend")
        if days_to_ex_div is None:
            continue
        try:
            days_val = int(days_to_ex_div)
        except (TypeError, ValueError):
            continue
        if not (-3 <= days_val <= 7):
            continue
        ex_note = ""
        if ex_div_date:
            ex_note = f" | 除息日 {ex_div_date}({days_val}天后)"
        lines.append(f"• {anchor or sym}{ex_note}")
    return "\n".join(lines)


def _collect_high_correlation_symbols(
    isa_cands: list[ScreenerCandidate],
    invest_cands: list[ScreenerCandidate],
    selected_actions: list[Any],
    held_syms: list[str],
    all_histories: dict[str, pd.DataFrame],
    cfg: AppConfig,
) -> set[str]:
    """Combine screener overlap warnings with deterministic check on ad-hoc LLM picks."""
    threshold = cfg.strategy.overlap_correlation_threshold
    screened_syms: set[str] = set()
    high_corr: set[str] = set()
    for c in (*isa_cands, *invest_cands):
        sym_up = c.symbol.strip().upper()
        screened_syms.add(sym_up)
        if c.overlap_warning or (c.max_portfolio_corr is not None and c.max_portfolio_corr >= threshold):
            high_corr.add(sym_up)

    held_up = {s.strip().upper() for s in held_syms}
    extra_check_syms: list[str] = []
    for item in selected_actions:
        sym = str(getattr(item, "symbol", "") or "").strip()
        sym_up = sym.upper()
        act_up = str(getattr(item, "action", "") or "").strip().upper()
        if act_up not in {"BUY", "ADD"} or not sym_up or sym_up == "CASH":
            continue
        if sym_up in screened_syms or sym_up in held_up:
            continue
        if _lookup_ci(all_histories, sym) is not None:
            extra_check_syms.append(sym)

    if extra_check_syms and held_syms:
        corr_input: dict[str, pd.DataFrame] = {}
        for hs in held_syms:
            df = _lookup_ci(all_histories, hs)
            if df is not None and not df.empty:
                corr_input[hs] = df
        target_keys: list[str] = []
        for sym in extra_check_syms:
            df = _lookup_ci(all_histories, sym)
            if df is not None and not df.empty:
                corr_input[sym] = df
                target_keys.append(sym)
        if target_keys:
            corr_matrix = compute_correlation_matrix(
                corr_input,
                [*held_syms, *target_keys],
                lookback_days=cfg.strategy.correlation_window_days,
                benchmark_symbol="SPY",
                uk_benchmark_symbol="ISF.L",
                eu_benchmark_symbol="EXS1.DE",
            )
            for sym in target_keys:
                _, max_corr = max_correlation_against_portfolio(corr_matrix, sym, held_syms)
                if max_corr is not None and max_corr >= threshold:
                    high_corr.add(sym.strip().upper())

    return high_corr


async def _post_screen_validate_decision(
    decision: PickDecision,
    *,
    portfolio: PortfolioData,
    macro: Any,
    cfg: AppConfig,
    market_data: MarketDataService,
    held_bundles: dict[str, TickerMarketBundle],
    held_histories: dict[str, pd.DataFrame],
    evals: list[HoldingEvaluation],
    isa_cands: list[ScreenerCandidate],
    invest_cands: list[ScreenerCandidate],
    guardrails: list[str],
    report_actions: list[dict[str, Any]] | None = None,
    llm: LLMBackend | None = None,
    ops_notifier: OpsNotifier | None = None,
    trade_ctx: dict[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Backfill native entry_price, apply deterministic risk override guardrail, and render."""
    all_bundles = dict(held_bundles)
    all_histories = dict(held_histories)

    missing_syms = [
        item.symbol.strip()
        for item in decision.selected_actions
        if item.symbol.strip()
        and item.symbol.strip().upper() != "CASH"
        and _lookup_ci(all_bundles, item.symbol) is None
    ]
    if missing_syms:
        unique_missing = list(dict.fromkeys(missing_syms))
        try:
            extra_hists, extra_bundles = await market_data.get_bundles(
                unique_missing,
                include_news=False,
                include_earnings=True,
                include_info=True,
            )
            all_histories.update(extra_hists)
            all_bundles.update(extra_bundles)
        except Exception as exc:
            logger.warning("Post-screen bundle fetch failed for %s: %s", unique_missing, exc)

    hydrated_actions = []
    for item in decision.selected_actions:
        if item.entry_price <= 0:
            resolved_price: float | None = None
            bundle = _lookup_ci(all_bundles, item.symbol)
            if bundle is not None and bundle.current_price is not None and bundle.current_price > 0:
                resolved_price = float(bundle.current_price)
            else:
                hist = _lookup_ci(all_histories, item.symbol)
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    try:
                        last_close = float(hist["Close"].dropna().iloc[-1])
                        if last_close > 0:
                            resolved_price = last_close
                    except (IndexError, TypeError, ValueError):
                        resolved_price = None
            if resolved_price is not None and resolved_price > 0:
                item = item.model_copy(update={"entry_price": round(resolved_price, 4)})
        hydrated_actions.append(item)

    hydrated_decision = decision.model_copy(update={"selected_actions": hydrated_actions})
    high_corr_syms = _collect_high_correlation_symbols(
        isa_cands,
        invest_cands,
        hydrated_decision.selected_actions,
        portfolio.unique_yahoo_symbols(),
        all_histories,
        cfg,
    )
    trade_behavior = compute_trade_behavior_stats(
        portfolio,
        recent_days=7,
        history_days=cfg.strategy.order_history_days,
        cooldown_days=cfg.strategy.rebuy_cooldown_days,
    )
    override_res = apply_risk_override_guardrail(
        hydrated_decision,
        portfolio=portfolio,
        macro=macro,
        strategy=cfg.strategy,
        bundle_map=all_bundles,
        evals=evals,
        high_correlation_symbols=high_corr_syms,
        daily_vetoes=guardrails,
        report_actions=report_actions,
        trade_behavior=trade_behavior,
    )
    resolved_decision = (
        override_res.decision
        if isinstance(override_res.decision, PickDecision)
        else hydrated_decision
    )
    final_text = render_pick_report(
        resolved_decision,
        max_chars=1000,
        risk_override=override_res,
    )
    parsed_picks = [
        {
            "account": item.account,
            "symbol": item.symbol,
            "action": item.action,
            "entry_price": item.entry_price if item.entry_price > 0 else None,
            "thesis": item.thesis,
            "horizon_days": cfg.strategy.settlement_holding_days,
            "benchmark_symbol": resolve_market_benchmark(item.symbol)[0],
        }
        for item in resolved_decision.selected_actions
    ]
    if llm is not None and not final_text.startswith("⚠️"):
        selected_syms = {
            item.symbol.strip().upper()
            for item in resolved_decision.selected_actions
            if item.symbol.strip() and item.symbol.strip().upper() != "CASH"
        }
        held_syms = {s.strip().upper() for s in portfolio.unique_yahoo_symbols()}
        rejected_syms = {
            c.symbol.strip().upper() for c in (*isa_cands, *invest_cands)
        } - selected_syms - held_syms
        await audit_generated_output(
            llm,
            cfg=cfg,
            scope="pick",
            fact_baseline=build_fact_baseline(
                scope="pick",
                portfolio=portfolio,
                evals=evals,
                trade_ctx=trade_ctx,
                guardrails=guardrails,
            ),
            rendered_text=final_text,
            strategy=cfg.strategy,
            ops_notifier=ops_notifier,
            headline=resolved_decision.headline,
            selected_symbols=selected_syms,
            known_symbols=rejected_syms,
        )
    return final_text, parsed_picks


async def generate_pick_radar(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    focus_query: str = "",
    is_weekly_auto: bool = False,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Run incremental opportunity scan synchronized with today's /report guardrails."""
    cfg.reload_pools_if_changed()
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 账户读取失败，无法生成选股雷达：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    tracked_syms = portfolio.unique_yahoo_symbols(include_pending=True)
    isin_map, created_at_map = _extract_holding_context(portfolio)

    # Step 1: Fetch held bundles/histories + macro + today's /report ContextPack & recent lessons first
    (
        (held_histories, held_bundles),
        macro,
        (morning_report_full, morning_conclusion),
        guardrails,
        context_pack,
        recent_lessons,
        pref_notes,
    ) = await asyncio.gather(
        market_data.get_bundles(
            tracked_syms,
            include_news=True,
            include_earnings=True,
            isin_map=isin_map or None,
        ),
        market_data.get_macro_snapshot(),
        db.get_morning_context(),
        db.get_daily_guardrails(),
        db.get_context_pack(allow_fallback=False),
        db.get_recent_lessons(limit=5),
        db.get_active_preference_notes(),
    )

    # Feed the latest GBP/USD mid so subsequent Invest wallet splits can be derived locally.
    if getattr(macro, "gbp_usd", None):
        set_rate = getattr(t212, "set_gbp_usd_rate", None)
        if callable(set_rate):
            set_rate(macro.gbp_usd)

    # Step 2: Load curated 150-stock active pool (if auditioned) & reuse held_histories in quant screener
    active_pool_rows: list[dict[str, Any]] = []
    if hasattr(db, "get_active_universe_pool"):
        try:
            active_pool_rows = await db.get_active_universe_pool()
        except Exception as exc:
            logger.debug("Failed loading active_universe_pool; falling back to config.yaml: %s", exc)
            active_pool_rows = []

    screener_kwargs: dict[str, Any] = {
        "portfolio": portfolio,
        "market_data": market_data,
        "universe": cfg.universe,
        "watchlist": cfg.watchlist,
        "strategy": cfg.strategy,
        "existing_histories": held_histories,
        "top_n_per_track": cfg.strategy.pick_wide_funnel_top_n,
    }
    if active_pool_rows:
        screener_kwargs["active_pool_rows"] = active_pool_rows

    isa_cands, invest_cands = await run_quant_screener(**screener_kwargs)
    # Wide funnel -> DeepSeek ID filter: only tickers are selected; the raw candidate objects stay untouched.
    isa_cands, invest_cands = await _prefilter_candidates_with_ds(cfg, llm, isa_cands, invest_cands)

    pack_symbols = (
        context_pack.get("symbols", {})
        if isinstance(context_pack, dict) and isinstance(context_pack.get("symbols"), dict)
        else {}
    )
    if not recent_lessons and isinstance(context_pack, dict):
        pack_recent = context_pack.get("recent_lessons")
        if isinstance(pack_recent, list):
            recent_lessons = pack_recent

    tech_map = {}
    for sym in tracked_syms:
        snap = compute_technical_snapshot(
            sym,
            held_histories.get(sym),
            position_created_at=created_at_map.get(sym),
            bundle=held_bundles.get(sym),
        )
        if snap is not None:
            tech_map[sym] = snap

    evals = evaluate_portfolio(portfolio, tech_map, held_bundles, cfg.strategy, histories=held_histories)
    evals_text = format_evaluations_text(evals)
    exposure_text = compute_exposure_summary(portfolio, held_bundles, histories=held_histories)
    trade_ctx = build_trade_context(
        portfolio,
        max_recent_lines=6,
        max_summary_lines=6,
        bundles=held_bundles,
        tech_map=tech_map,
        histories=held_histories,
        cooldown_days=cfg.strategy.rebuy_cooldown_days,
    )

    morning_audits_text, morning_symbol_tags, report_top_actions = _extract_morning_audit_context(
        context_pack,
        morning_report_full,
    )
    isa_cands_text = _format_candidates_with_morning_tags(
        isa_cands, morning_symbol_tags, "（ISA 候选池暂无达标标的）"
    )
    invest_cands_text = _format_candidates_with_morning_tags(
        invest_cands, morning_symbol_tags, "（Invest 候选池暂无达标标的）"
    )
    focus_line = (
        f"用户指定扫描方向：【{focus_query}】（若预设列表未覆盖，请务必调用 inspect_ticker 与 check_portfolio_correlation 实查该方向的具体股票或 LSE ETF！）"
        if focus_query.strip()
        else (
            "本周美股已收盘，请为下周挑选【增量高赔率机会或优胜劣汰换仓】。"
            if is_weekly_auto
            else "主动增量扫描（美/英/欧股与 LSE 英镑 ETF + 优胜劣汰换仓机会）。"
        )
    )
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（今日尚未运行 /report，请独立执行全部风控红线检查）"
    preference_block = format_preference_notes_block(pref_notes)
    us_sched = get_us_market_schedule_uk()
    calendar_alerts = get_market_calendar_alerts()
    calendar_note = f"｜{'；'.join(calendar_alerts)}" if calendar_alerts else ""
    market_phase_note = (
        f"当前英国时间 {us_sched.uk_date}（{us_sched.uk_weekday_label}），美股阶段：{us_sched.phase}"
        + ("（周末/休市：所有建议均为下个开盘日/限价挂单参考）" if us_sched.phase in {"WEEKEND", "HOLIDAY"} else "")
        + calendar_note
    )
    total_assets = max(portfolio.total_combined_value, 1.0)
    total_cash_pct = (portfolio.total_combined_cash / total_assets) * 100.0
    total_available = portfolio.isa.available_to_trade + portfolio.invest.available_to_trade
    exact_cash_summary = (
        f"双账户总净值 £{portfolio.total_combined_value:,.2f} | "
        f"总现金 £{portfolio.total_combined_cash:,.2f} (占总资产 {total_cash_pct:.1f}%) | "
        f"立即可用 £{total_available:,.2f} (ISA £{portfolio.isa.available_to_trade:,.2f} + Invest £{portfolio.invest.available_to_trade:,.2f}) | "
        f"挂单占用(未成交前仍计息) ISA £{portfolio.isa.reserved_cash:,.2f} / Invest £{portfolio.invest.reserved_cash:,.2f}\n"
        f"{build_wallet_context_text(portfolio, cfg.strategy, compact=True)}"
    )

    cand_syms = list(
        dict.fromkeys(
            [
                *(c.symbol for c in isa_cands[: cfg.strategy.pick_search_symbols_per_track]),
                *(c.symbol for c in invest_cands[: cfg.strategy.pick_search_symbols_per_track]),
            ]
        )
    )
    lesson_syms = list(dict.fromkeys([*cand_syms, *tracked_syms]))

    async def _empty_grounding() -> str:
        return ""

    async def _fetch_cand_bundles_cached() -> dict[str, TickerMarketBundle]:
        missing_cands = [s for s in cand_syms if s not in held_bundles]
        if not missing_cands:
            return {}
        try:
            _, bmap = await market_data.get_bundles(
                missing_cands,
                include_news=True,
                include_earnings=True,
            )
            return bmap
        except Exception as exc:
            logger.debug("Pre-populating candidate bundles failed in /pick: %s", exc)
            return {}

    async def _fetch_cand_deep_intel() -> dict[str, dict[str, Any]]:
        us_cands = [s for s in cand_syms if "." not in s and hasattr(market_data, "get_us_deep_intel_batch")]
        if not us_cands or not hasattr(market_data, "get_us_deep_intel_batch"):
            return {}
        try:
            return await market_data.get_us_deep_intel_batch(us_cands)
        except Exception as exc:
            logger.debug("Pre-populating candidate US deep intel failed in /pick: %s", exc)
            return {}

    grounding_coro = (
        run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.search_model,
            scope="pick",
            symbols=cand_syms,
            focus_query=focus_query,
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
            db=db,
            persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
        if cfg.models.enable_forced_search_grounding
        else _empty_grounding()
    )
    grounding_brief, symbol_lessons, extra_cand_bundles, cand_intel_map = await asyncio.gather(
        grounding_coro,
        _fetch_candidate_symbol_lessons(db, lesson_syms, pack_symbols),
        _fetch_cand_bundles_cached(),
        _fetch_cand_deep_intel(),
    )
    merged_cand_bundles = {**held_bundles, **extra_cand_bundles}
    cand_detail_lines: list[str] = []
    for sym in cand_syms:
        b = merged_cand_bundles.get(sym)
        parts: list[str] = []
        intel_str = _format_us_deep_intel(cand_intel_map.get(sym) or {})
        if intel_str:
            parts.append(intel_str)
        if b is not None and b.news:
            news_str = " / ".join(f"[{n['date']}] {n['title'][:70]}" for n in b.news[:2] if n.get("title"))
            if news_str:
                parts.append(f"新闻: {news_str}")
        if parts:
            cand_detail_lines.append(f"• {sym}: " + " | ".join(parts))
    cand_details_section = (
        "\n\n【Top 候选近期新闻与机构评级补充（0-Token 本地聚合）】：\n" + "\n".join(cand_detail_lines)
        if cand_detail_lines
        else ""
    )

    lessons_text = _format_lessons_section(
        recent_lessons,
        symbol_lessons,
        cfg.strategy.settlement_holding_days,
    )
    identity_text = _build_holding_identity_section(tracked_syms, held_bundles, isin_map, pack_symbols)
    pack_disagreement_text = (
        str(context_pack.get("disagreement_text") or "").strip()
        if isinstance(context_pack, dict)
        else ""
    )

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
        db=db,
        search_cache_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
    )

    # ---------- Stage 1: Incremental Opportunity Researcher ----------
    researcher_sys = build_agent_system_instruction(
        "你是增量选股投研员（Researcher Agent）。存量持仓体检与老挂单审计已由 `/report` 完成，你专注于结合今日风控红线与早报裁决，"
        "从全市场（美/英/欧个股与 LSE 英镑 ETF）中挖掘最值得新买入的配置或高确定性卖弱换强机会。\n"
        "对今日 `/report` 已裁决【CANCEL】或【MODIFY】的活动挂单，以及已裁决【TRIM】/【STOP】的持仓，默认必须严格遵循不得推翻；"
        "仅当你通过工具实查到早报生成之后的突发新公告或股价已显著回落至早报认可区间时，才可提出修正并写明最新依据。\n"
        f"请参考历史 T+{cfg.strategy.settlement_holding_days} 交易日选股复盘教训，避免对近期连续失效的板块或追高形态重复推荐。\n"
        "请给出不超过 3 条核心提案（ISA 新股或 LSE ETF / Invest 短线催化 / 必要时的卖弱换强），"
        "每条写清逻辑、介入区间、止损、目标；若上文初筛候选与实时 Google Search 摘要已提供充分事实，禁止无故重复调用工具；仅当用户指定新方向不在预设列表或核心数据缺失时，才调用 `inspect_ticker`、`check_portfolio_correlation` 或 `search_market_news`。",
        cfg.strategy,
        include_order_audit_rule=False,
    )
    morning_audit_block = (
        f"早报持仓与挂单裁决（无盘中新变量不得推翻）：\n{morning_audits_text}\n"
        if morning_audits_text
        else ""
    )
    researcher_prompt = (
        f"【任务：增量选股投研员】\n"
        f"时段状态：{market_phase_note}\n"
        f"方向：{focus_line}\n\n"
        f"=== 今日 /report 审计基线与风控红线（必须严格继承，除非有盘中新数据证明必须修正）===\n"
        f"早报基调：{morning_conclusion or '（暂无）'}\n"
        f"{morning_audit_block}"
        f"红线清单：\n{veto_block}\n\n"
        + (f"{preference_block}\n\n" if preference_block else "")
        + f"【历史 T+{cfg.strategy.settlement_holding_days} 选股复盘教训（禁止重蹈覆辙）】：\n{lessons_text}\n\n"
        f"【精确现金底座（禁止自行估算比例）】：{exact_cash_summary}\n"
        f"【宏观快照】：{macro.to_text()}\n"
        f"【行业与跨账户暴露】：\n{exposure_text}\n"
        + (f"【存量标的身份与除息日历锚点】：\n{identity_text}\n" if identity_text else "")
        + (f"【今日早报多空冲突摘要（ContextPack）】：\n{pack_disagreement_text}\n" if pack_disagreement_text else "")
        + f"【ISA 账户概览】：{format_account_overview(portfolio.isa).splitlines()[0]}\n"
        f"【Invest 账户概览】：{format_account_overview(portfolio.invest).splitlines()[0]}\n"
        f"【活动挂单】：\n{trade_ctx['pending_orders_text']}\n"
        f"【现有持仓触发信号】：\n{evals_text}\n\n"
        f"【Python 初筛 Top 候选 — ISA 长线赛道（含 LSE ETF 与美/英/欧龙头）】：\n{isa_cands_text}\n\n"
        f"【Python 初筛 Top 候选 — Invest 短线机动赛道】：\n{invest_cands_text}"
        + cand_details_section
        + (f"\n\n【实时 Google Search 候选与方向催化摘要】：\n{grounding_brief}" if grounding_brief else "")
    )

    research_rounds = 3 if focus_query.strip() else 2
    research_output = await llm.generate(
        model=cfg.models.pick_researcher_model,
        prompt=researcher_prompt,
        system_instruction=researcher_sys,
        thinking_level=cfg.models.pick_researcher_thinking_level,
        tools=toolkit.declarations(),
        tool_executor=toolkit.as_executor(),
        temperature=0.25,
        max_tool_rounds=research_rounds,
        usage_label="pick_researcher",
    )
    if research_output.startswith("⚠️"):
        return research_output

    # ---------- Stage 2: Risk & Bear Critic -> Structured PickDecision ----------
    critic_sys = build_agent_system_instruction(
        "你是独立风控官（Risk & Bear Critic）。请按下列六重护栏与今日 `/report` 红线严格审核投研员提案，"
        "直接输出最终结构化 JSON Decision（渲染后将限制在 1000 字内，结论置顶、废话为零）。\n\n"
        "【六重护栏】：\n"
        f"1. 盈亏比与摩擦：按目标空间扣除交易摩擦（ISA 美欧往返 {cfg.strategy.fx_fee_pct * 2:.2f}% / 英股买入 {cfg.strategy.uk_stamp_duty_pct:.1f}%印花税 + 买卖价差估算；"
        f"Invest 双钱包同币种 0% 换汇、跨币种单边 {cfg.strategy.fx_fee_pct:.2f}%）计算 R:R，低于 2:1 一律否决；"
        f"GBP {cfg.strategy.cash_interest_rate_pct:.1f}% / USD {cfg.strategy.usd_cash_interest_rate_pct:.1f}% 现金利息仅作等待期保底，不得作为否决理由。\n"
        "2. 穿透科技暴露：结合上下文“穿透后真实科技暴露(估算)”（ETF 隐含科技 + 直接科技个股），加仓后科技因子明显集中的须降级或改用低科技权重宽基。\n"
        f"3. 频率与冷却（模式 {cfg.strategy.trade_cadence_guard_mode}）：近 7 天参考 ≤{cfg.strategy.max_weekly_new_buys} 笔、"
        f"近 30 天 ≤{cfg.strategy.max_monthly_new_buys} 笔，清仓 {cfg.strategy.rebuy_cooldown_days} 天内同标的回补需谨慎；"
        + (
            "该约束由 Python 硬拦截，不要重复推荐已被拦截的标的。\n"
            if str(cfg.strategy.trade_cadence_guard_mode).lower() == "hard"
            else "若确有必要超额开仓或抢回补（强催化/避险切换），必须在该条 thesis 写明破例依据。\n"
        )
        + f"4. 相关性与行业红线：与当前持仓 60 日相关性 >= {cfg.strategy.overlap_correlation_threshold:.2f}、或违背 `/report` 今日红线名单的标的必须驳回。\n"
        "5. 财报/过热排雷：5日内财报、RSI>=72 追高全部否决或降级；结合历史复盘教训，近期连续失效板块/追高形态从严否决。\n"
        "6. 一致性、精确现金与零内部对话外泄：严格继承 `/report` 红线与挂单/持仓裁决"
        "（对早报已裁决 CANCEL/MODIFY 的活动挂单及 TRIM/STOP 的持仓，除非投研员提供早报之后的新催化或股价显著跌至早报认可区间的新证据，否则不得推翻或继续按早报否决的旧价位推荐；"
        "仅用提供的精确现金数字与比例，禁止自行瞎编比例）；"
        "若当前为周末/休市且有推荐标的，请在 headline 标明“下周一开盘/限价挂单参考”；"
        "禁止任何内部审查对话外泄：`headline` 严禁提及任何未入选、淘汰或被你否决的股票代码/方向（严禁出现“驳回XX”等字样），只写最终入选标的与现金安排；"
        "`rejected_or_frozen` 一律返回空数组 `[]`（不要列举任何未入选候选或否决理由，节省输出字数）；"
        f"若无个股达到 R:R≥2:1，且 ISA 现金占比 >{cfg.strategy.isa_cash_deploy_trigger_pct:.0f}%（或 Invest 现金 >{cfg.strategy.invest_cash_idle_warn_pct:.0f}%），"
        f"应优先输出低摩擦宽基 ETF（如 VWRP.L / VUAG.L）的分批定投方案：写明具体价位、单笔金额（ISA ETF ≥£{cfg.strategy.min_isa_etf_order_gbp:,.0f}）与节奏；"
        "若确实连宽基部署空间都没有，才让 `selected_actions` 为空数组 `[]`，并在 headline 说明“本轮无 R:R≥2:1 标的，现金待命 + 下一次部署触发条件”，"
        f"严禁默认写“现金拿住吃 {cfg.strategy.cash_interest_rate_pct:.1f}% 利息”。",
        cfg.strategy,
        include_order_audit_rule=False,
    )
    critic_prompt = (
        f"【时段状态】：{market_phase_note}\n"
        f"【今日 /report 基线】：{morning_conclusion or '（暂无）'}\n"
        + (f"【今日 /report 持仓与挂单裁决（默认遵循）】：\n{morning_audits_text}\n" if morning_audits_text else "")
        + f"【今日红线】：\n{veto_block}\n\n"
        + (f"{preference_block}\n\n" if preference_block else "")
        + f"【当前活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"【交易频率与冷却（Python 计算）】：{trade_ctx['trade_behavior_text']}\n\n"
        f"【历史 T+{cfg.strategy.settlement_holding_days} 选股复盘教训】：\n{lessons_text}\n\n"
        f"【精确现金底座】：{exact_cash_summary}\n"
        f"【双钱包明细与资金路径】：\n{build_wallet_context_text(portfolio, cfg.strategy)}\n"
        f"【行业暴露】：\n{exposure_text}\n\n"
        f"【投研员提案（待审）】：\n{research_output}"
    )

    raw_critic = await llm.generate(
        model=cfg.models.pick_model,
        prompt=critic_prompt,
        system_instruction=critic_sys,
        thinking_level=cfg.models.pick_thinking_level,
        response_schema=PickDecision,
        temperature=0.15,
        usage_label="pick_critic",
    )

    decision, resolved_critic_raw = await parse_or_repair_structured_json(
        raw_critic,
        PickDecision,
        llm=llm,
        repair_model=cfg.models.fallback_model,
        thinking_level="LOW",
        usage_label="pick_critic_repair",
        ops_notifier=ops_notifier,
        alert_source="pick_critic",
    )
    now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")

    if decision is not None:
        final_text, parsed_picks = await _post_screen_validate_decision(
            decision,
            portfolio=portfolio,
            macro=macro,
            cfg=cfg,
            market_data=market_data,
            held_bundles=held_bundles,
            held_histories=held_histories,
            evals=evals,
            isa_cands=isa_cands,
            invest_cands=invest_cands,
            guardrails=guardrails,
            report_actions=report_top_actions,
            llm=llm,
            ops_notifier=ops_notifier,
            trade_ctx=trade_ctx,
        )
    else:
        logger.warning("Pick Critic structured parse fell back to plain text.")
        final_text = resolved_critic_raw.strip()[:1000]
        parsed_picks = []

    if not final_text.startswith("⚠️"):
        await db.save_pick_radar(focus_query, research_output, final_text, now_uk)
        if parsed_picks:
            await db.log_recommendations(
                parsed_picks,
                default_horizon_days=cfg.strategy.settlement_holding_days,
            )

    return final_text
