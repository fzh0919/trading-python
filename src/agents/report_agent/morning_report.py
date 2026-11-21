"""Morning Report (/report) 3-step TradingAgents-Lite orchestrator and Single Source of Truth persistence."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction, build_wallet_context_text
from src.agents.report_agent.core_deployment import (
    _build_report_recommendations,
    _enrich_gate_candidate_bundles,
    _evaluate_deployment_gate_safely,
    _fetch_core_etf_candidates,
    _format_core_policy_ticket_block,
    _validate_core_deployment,
)
from src.agents.report_agent.debate import _build_debate_pm_block, _run_step2_debate
from src.agents.report_agent.fact_sheet import (
    LONDON_TZ,
    _CORE_DCA_HORIZON_DAYS,
    _account_change_pct,
    _assemble_fact_sheet_bundle,
    _compute_portfolio_diff_text,
    _format_benchmark_line,
    _lookup_benchmark_hist,
)
from src.agents.report_agent.sentinels import _daily_move_pct_from_hist
from src.agents.schemas import (
    MorningReportDecision,
    parse_or_repair_structured_json,
    render_morning_report,
    trim_text_at_line_boundary,
)
from src.agents.semantic_auditor import audit_generated_output, build_fact_baseline
from src.agents.tools import MarketToolkit, run_forced_market_grounding
from src.config import AppConfig
from src.data_providers.analytics import build_nav_row, compute_technical_snapshot
from src.data_providers.market_data import MarketDataService
from src.data_providers.market_data.models_and_parsing import _deserialize_ohlcv_json
from src.data_providers.t212_client import (
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import (
    get_uk_weekday_label,
    get_us_market_schedule_uk,
    is_uk_bank_holiday,
)
from src.engine.deployment_gate import format_deployment_gate_banner, reconcile_pm_pick_gate
from src.engine.portfolio_rules import (
    apply_risk_override_guardrail,
    build_report_size_notes,
    evaluate_portfolio,
    format_evaluations_text,
    resolve_invest_dynamic_rr_floor,
)
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger("src.agents.report_agent")


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("src.agents.report_agent")
    return getattr(facade, name, default) if facade is not None else default


def _current_cycle_id(db: AdvisorDatabase) -> str | None:
    """Best-effort current trading cycle id for same-day self-anchoring protection."""
    getter = getattr(db, "_cycle_id", None)
    if not callable(getter):
        return None
    try:
        return str(getter() or "").strip() or None
    except Exception:  # pragma: no cover - defensive only
        return None


async def _fetch_past_lessons(db: AdvisorDatabase, exclude_cycle_date: str | None) -> str:
    """Load the Memory Log, tolerating mocks/stubs that do not accept the new keyword."""
    fetch = getattr(db, "get_past_decision_context", None)
    if not callable(fetch):
        return ""
    try:
        return await fetch(limit=5, exclude_cycle_date=exclude_cycle_date)
    except TypeError:
        return await fetch(limit=5)


def _filter_fresh_cached_breadth_histories(
    cached_rows: dict[str, dict[str, Any]] | None,
    ref_hist: pd.DataFrame | None,
    *,
    max_bar_lag_days: int = 2,
    fallback_max_age_hours: int = 36,
    now_ms: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Filter local SQLite ``ohlcv_cache`` rows against the latest benchmark bar date.

    Rejects stale Sunday-night audition bars once a new weekday session has printed on SPY
    (e.g. Tuesday morning with Monday SPY bar vs Friday cache = 3 calendar days > 2), so
    breadth automatically falls back to today's freshly fetched ``pool_snaps`` instead of
    masking a Monday-Wednesday crash/V-bottom with last Friday's 150-stock cache.
    """
    if not cached_rows:
        return {}
    ref_bar_date = None
    if ref_hist is not None and not ref_hist.empty:
        try:
            ts = pd.Timestamp(ref_hist.index[-1])
            if not pd.isna(ts):
                ref_bar_date = ts.date()
        except Exception:
            ref_bar_date = None
    cur_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    fallback_cutoff_ms = cur_ms - int(fallback_max_age_hours) * 3_600_000

    out: dict[str, pd.DataFrame] = {}
    for sym_c, row_c in cached_rows.items():
        if not isinstance(row_c, dict):
            continue
        bar_date = None
        raw_bar = str(row_c.get("last_bar_date") or "").strip()[:10]
        if raw_bar:
            try:
                bar_date = datetime.strptime(raw_bar, "%Y-%m-%d").date()
            except ValueError:
                bar_date = None
        if ref_bar_date is not None and bar_date is not None:
            if (ref_bar_date - bar_date).days > int(max_bar_lag_days):
                continue
        elif ref_bar_date is None:
            updated_ms = int(row_c.get("updated_at_ms") or 0)
            if updated_ms <= 0 or updated_ms < fallback_cutoff_ms:
                continue
        df_c = _deserialize_ohlcv_json(str(row_c.get("payload_json") or ""))
        if df_c is None or df_c.empty:
            continue
        if ref_bar_date is not None and bar_date is None:
            try:
                df_ts = pd.Timestamp(df_c.index[-1])
                if pd.isna(df_ts) or (ref_bar_date - df_ts.date()).days > int(max_bar_lag_days):
                    continue
            except Exception:
                continue
        out[sym_c] = df_c
    return out


def _build_regime_anchor_text(
    cfg: AppConfig,
    gate_preview: dict[str, Any] | None,
    current_regime: str,
    *,
    vix: float | None = None,
) -> str:
    """Deterministic Regime / dynamic R:R anchor prepended to the shared Bull/Bear/PM context."""
    rr_floor, rr_tier = resolve_invest_dynamic_rr_floor(
        current_regime, None, cfg.strategy, vix=vix
    )
    regime_reason = ""
    breadth_line = ""
    isa_status = "未启用门控"
    invest_status = "未启用门控"
    if isinstance(gate_preview, dict):
        regime_reason = str(gate_preview.get("regime_reason") or "").strip()
        breadth = gate_preview.get("breadth") if isinstance(gate_preview.get("breadth"), dict) else {}
        if breadth:
            b_score = breadth.get("score")
            b_src = str(breadth.get("source") or "unavailable")
            b_cnt = int(breadth.get("pool_count") or 0)
            if b_score is not None:
                cnt_txt = f"，本地池{b_cnt}只" if b_cnt > 0 else ""
                breadth_line = f"市场宽度 = {float(b_score):.1f}（来源 {b_src}{cnt_txt}）"
            else:
                breadth_line = "市场宽度 = 降级未计入（中性放行，零额外外网请求）"
        isa_g = gate_preview.get("isa") or {}
        inv_g = gate_preview.get("invest") or {}
        if isinstance(inv_g, dict):
            # v5.5.7: the gate already resolved the VIX-inclusive floor — quote it verbatim.
            if inv_g.get("invest_min_rr_floor") is not None:
                rr_floor = float(inv_g.get("invest_min_rr_floor") or rr_floor)
            if inv_g.get("invest_rr_tier_label"):
                rr_tier = str(inv_g.get("invest_rr_tier_label"))
        if isinstance(isa_g, dict) and isa_g.get("status"):
            isa_status = f"{isa_g.get('status')}（{isa_g.get('reason') or '无补充理由'}）"
        if isinstance(inv_g, dict) and inv_g.get("status"):
            invest_status = f"{inv_g.get('status')}（{inv_g.get('reason') or '无补充理由'}）"
    if current_regime == "RISK_ON":
        rr_note = "，RISK_ON 右侧领涨 ≥1.6:1 / 常规 ≥2.0:1"
    elif current_regime == "RECOVERY":
        rr_note = "，RECOVERY 早春轻兵试探（仅 1 槽位/1.0%热力/0.6R，仅放行强 RS 领涨股）"
    else:
        rr_note = ""
    lines = [
        "=== 〇、Python 确定性市场状态与动态 R:R 门槛（唯一基准，禁止自行改判） ===",
        f"当日确定性市场状态 = {current_regime}（{regime_reason or '无补充理由'}）",
    ]
    if breadth_line:
        lines.append(breadth_line)
    lines.extend(
        [
            f"Invest 基础净 R:R 门槛 = ≥{rr_floor:.1f}:1（{rr_tier}{rr_note}）",
            f"ISA 门控 = {isa_status}",
            f"Invest 门控 = {invest_status}",
            (
                f"硬纪律：Bull/Bear/PM 必须以本段 Python 确定性 Regime（{current_regime}）为唯一基准，"
                "不得自行改判 Regime；若因突发宏观/事件需更谨慎，仅可显式声明"
                f"『在 {current_regime} 基础上主动收紧』，不得改称弱市或擅自抬高门槛。"
            ),
        ]
    )
    return "\n".join(lines)


def _build_morning_pm_prompt(
    *,
    cfg: AppConfig,
    debate_pm_block: str,
    calendar_alerts: list[str],
    past_lessons: str,
    wallet_text: str,
    diff_text: str,
    benchmark_line: str,
    trade_ctx: dict[str, Any],
    cash_flow_text: str,
    disagreement_text: str,
    core_policy_context: str,
    analyst_notes: str,
    auditor_notes: str,
    gate_preview: dict[str, Any] | None,
) -> tuple[str, str]:
    """Construct the PM system instruction and synthesis prompt for /report Step 3."""
    pm_role = (
        "你是最终投资组合经理（Portfolio Manager），负责拍板权衡「机会分析师（Bull）」与「独立风控审计官（Bear）」的冲突并输出结构化今日总纲（严格遵守 JSON Schema）。\n"
        "严格去重与排版准则（违反将导致重要信息被截断）：\n"
        "• 对【本地多空证据冲突对照】中列出的标的，必须正面裁决采纳 Bull 还是 Bear 依据，严禁和稀泥。\n"
        "• `top_actions`【只放现有持仓的 ADD/TRIM/STOP/HOLD】（1~3条）；所有活动挂单的撤改留在 `pending_order_audits`，严禁把 CANCEL_ORDER/MODIFY_ORDER 重复写进 `top_actions`！"
        "优先放实质调仓（ADD/TRIM/STOP）；若现有持仓全部观望无调仓，仅输出 1 条 CASH HOLD（或最多 1 条重点监控 HOLD），"
        "冲突标的的「维持观望裁决」请放入 `flagged_positions` 或 `hold_summary`，严禁用 3 条纯 HOLD 占满行动区。\n"
        "• `core_deployment`【仅对 Python 今日唯一签发的核心宽基政策票据做风控终审】（0~1条，仅 ISA，BUY）："
        "`symbol` 必须等于票据代码，`verdict` 只能填 APPROVE（按上限执行）/ SHRINK（缩减金额）/ DEFER（因重大宏观或事件暂缓）；"
        "票据为空时必须输出空数组 []，严禁自行挑选票据外的任何宽基代码（如 VUAG/IUKD）、严禁放大金额；"
        "已持仓子桶只能写进 top_actions ADD。\n"
        "• `size_hint` 必填数量口径：TRIM 写减仓比例（如 30%、1/3）或英镑金额（如 £2,300），ADD 写加仓金额或比例，STOP 写「全出」，HOLD/CASH 留空；严禁写股数（由 Python 按持仓确定性换算）。\n"
        "• `pending_order_audits` 逐一简洁裁决 KEEP/MODIFY/CANCEL，不要和上文重复抄写理由。\n"
        "• `flagged_positions` 只列出【未在 top_actions 或 pending_order_audits 中出现过】的其他异动/观察持仓（如暴跌、财报临近但暂不操作者），严禁再复述上方已给过动作的标的！\n"
        "• `daily_vetoes` 将写入数据库强制约束今日后续 `/pick` 和 `/us`，只写简洁可执行的红线（2~4条，每条<=50字）。\n"
        # [用户授权调整 v5.5.6] 篇幅以各 Schema 字段上限为准，渲染层 Python 控制 <=2800 字。
        "• 各字段严格遵守 JSON Schema 字数上限、逻辑清晰不缩写，严禁为凑字数重复或注水；最终渲染长度由 Python 控制在 ≤2800 字。"
        "严禁把「撤单」和「吃息」写成因果关系（如“撤X吃息待命”）：未成交限价挂单本就计息，撤单原因只能是追高/超频/盈亏比不足等交易逻辑。"
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
        + (f"{core_policy_context}\n\n" if core_policy_context else "")
        + f"=== 机会分析师报告 (Bull) ===\n{analyst_notes}\n\n"
        f"=== 独立风控审计官挑刺与挂单审计 (Bear) ===\n{auditor_notes}"
        + debate_pm_block
    )

    # v5.4.0: inject the deterministic gate bottom line into the PM prompt (advise mode only).
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
    return pm_sys, pm_prompt


async def _persist_morning_report_and_audit(
    *,
    cfg: AppConfig,
    db: AdvisorDatabase,
    llm: LLMBackend,
    portfolio: Any,
    histories: dict[str, pd.DataFrame],
    bundles: dict[str, Any],
    evals: list[Any],
    trade_ctx: dict[str, Any],
    fact_sheet: str,
    core_policy_context: str,
    resolved_decision: MorningReportDecision,
    final_report: str,
    short_conclusion: str,
    guardrails: list[str],
    context_pack: dict[str, Any],
    ops_notifier: OpsNotifier | None,
) -> None:
    """Persist non-fallback snapshot/NAV, morning report SOT, 5d-deduped ledger, and run read-only semantic audit."""
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
            nav_row = build_nav_row(portfolio, benchmark_close=bench_close)
            today_uk = str(nav_row.get("uk_date") or "")
            prev_row = await db.get_previous_nav_row(today_uk) if today_uk else None
            if prev_row is not None:
                nav_row = build_nav_row(
                    portfolio,
                    benchmark_close=bench_close,
                    prev_snapshot_ms=int(prev_row.get("created_at_ms") or 0),
                    prev_uk_date=str(prev_row.get("uk_date") or ""),
                )
            await db.upsert_nav_history(nav_row)
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
            all_recs = _build_report_recommendations(resolved_decision, portfolio, bundles, cfg)
            report_recs = [r for r in all_recs if str(r.get("source") or "") != "core_dca"]
            core_dca_recs = [r for r in all_recs if str(r.get("source") or "") == "core_dca"]
            if report_recs:
                # 5-day cross-source (report <-> pick) atomic dedupe inside SQLite write lock.
                await db.log_recommendations(
                    report_recs,
                    default_horizon_days=cfg.strategy.settlement_holding_days,
                    source="report",
                    dedupe_window_days=5,
                    dedupe_cross_source=True,
                )
            if core_dca_recs:
                # Strategic core DCA lives in its own ledger lane so it can never pollute
                # the 5-day tactical win-rate / Alpha / t-stat scoreboard.
                await db.log_recommendations(
                    core_dca_recs,
                    default_horizon_days=_CORE_DCA_HORIZON_DAYS,
                    source="core_dca",
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
                fact_sheet=fact_sheet + (f"\n\n{core_policy_context}" if core_policy_context else ""),
            ),
            rendered_text=final_report,
            strategy=cfg.strategy,
            ops_notifier=ops_notifier,
        )


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
    3) Portfolio Manager Pydantic Structured Synthesis + RiskOverride Guardrail -> <= 2800 chars
    """
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 双账户数据均读取失败：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    tracked_syms = portfolio.unique_yahoo_symbols(include_pending=True)
    held_syms = portfolio.unique_yahoo_symbols(include_pending=False)

    prev_snap, past_lessons, fact_bundle = await asyncio.gather(
        db.get_previous_snapshot(min_age_hours=4.0),
        _fetch_past_lessons(db, _current_cycle_id(db)),
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

    # v5.5.1: quotes/trends for the un-held core/diversifier ETF universe, so /report
    # can originate a new core sleeve (ISF/VERG etc.) without becoming a stock picker.
    gate_mode = str(getattr(cfg.strategy, "deployment_gate_mode", "off") or "off").strip().lower()
    core_etf_syms = (
        await _fetch_core_etf_candidates(market_data, cfg, bundles, histories)
        if gate_mode != "off"
        else []
    )

    # ---------- v5.4.0 deployment gate data pre-fetch (0 LLM calls, offline-tolerant) ----------
    gate_histories: dict[str, pd.DataFrame] = {}
    gate_candidates: list[dict[str, Any]] = []
    prev_gate_states: list[dict[str, Any]] = []
    pool_snaps: dict[str, Any] = {}
    cached_breadth_histories: dict[str, pd.DataFrame] = {}
    if gate_mode != "off":
        regime_bench = str(getattr(cfg.strategy, "regime_benchmark_symbol", "SPY") or "SPY").strip().upper()
        core_bench = str(getattr(cfg.strategy, "benchmark_symbol", "VWRP.L") or "VWRP.L").strip().upper()
        etf_upper = {s.strip().upper() for s in (cfg.universe.etf_symbols or []) if s} | {core_bench}
        active_pool_syms: list[str] = []
        all_active_equity_syms: list[str] = []
        if callable(getattr(db, "get_active_universe_pool", None)):
            try:
                active_rows = await db.get_active_universe_pool()
                for row in active_rows or []:
                    if isinstance(row, dict) and not row.get("is_etf"):
                        sym_r = str(row.get("symbol") or "").strip()
                        if sym_r and sym_r.upper() not in etf_upper:
                            all_active_equity_syms.append(sym_r)
                            if len(active_pool_syms) < 30:
                                active_pool_syms.append(sym_r)
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
        # Zero-extra-network local breadth cache read, anchored on the latest benchmark bar date
        # (<=2 calendar days lag; stale weekend audition bars fall back to today's fresh pool_snaps).
        if all_active_equity_syms and callable(getattr(db, "get_cached_ohlcv", None)):
            try:
                cached_rows = await db.get_cached_ohlcv(all_active_equity_syms[:150], period="2y")
                ref_hist = gate_histories.get(regime_bench)
                if ref_hist is None or ref_hist.empty:
                    ref_hist = histories.get(regime_bench)
                cached_breadth_histories = _filter_fresh_cached_breadth_histories(
                    cached_rows,
                    ref_hist,
                    max_bar_lag_days=2,
                )
            except Exception as exc:
                logger.debug("Deployment-gate local breadth cache read skipped: %s", exc)
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
        # candidates carry a bundle; fetch one lightweight batch for A1 pullbacks and
        # locally pattern-confirmed A2 breakouts (v5.5.9 M3 P1: gate_histories are
        # already in memory, so ordinary bull-tape risers cannot eat the batch cap).
        cand_bundles = await _enrich_gate_candidate_bundles(
            market_data,
            pool_syms,
            pool_snaps,
            bundles,
            tolerance_pct=float(getattr(cfg.strategy, "trigger_ma_tolerance_pct", 1.5) or 1.5),
            histories=gate_histories,
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

    # Feed the latest GBP/USD mid so the CURRENT in-memory portfolio and every downstream
    # wallet / gate / shared-context view share one rate (cold-start FX backfill, no HTTP).
    if getattr(macro, "gbp_usd", None):
        refresh_fx = getattr(t212, "refresh_portfolio_fx", None)
        if callable(refresh_fx):
            try:
                portfolio = refresh_fx(portfolio, macro.gbp_usd) or portfolio
            except Exception as exc:
                logger.debug("In-place portfolio FX refresh skipped: %s", exc)
                refresh_fx = None
        if not callable(refresh_fx):
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
        await _facade_attr("run_forced_market_grounding", run_forced_market_grounding)(
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

    us_sched = _facade_attr("get_us_market_schedule_uk", get_us_market_schedule_uk)()
    uk_bank_holiday = _facade_attr("is_uk_bank_holiday", is_uk_bank_holiday)()
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
    # Evaluate the deterministic gate BEFORE the debate so Bull/Bear/PM all see the same
    # single Python-issued core policy ticket (0 LLM calls, offline-tolerant).
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
            pool_snaps=pool_snaps,
            cached_breadth_histories=cached_breadth_histories,
        )
    core_policy_context = _format_core_policy_ticket_block(
        gate_preview, bundles, histories
    )

    # v5.5.1: re-evaluate holdings under the RESOLVED regime before the debate, so ISA value
    # holdings are never nagged by Invest-style volatility trims (and vice versa).
    current_regime = "NEUTRAL"
    if isinstance(gate_preview, dict):
        current_regime = str(gate_preview.get("regime") or "NEUTRAL").strip().upper()
    if current_regime in {"RISK_ON", "RECOVERY", "RISK_OFF"} or bool(gate_histories):
        eval_histories = {**gate_histories, **histories} if gate_histories else histories
        regime_evals = evaluate_portfolio(
            portfolio,
            tech_map,
            bundles,
            cfg.strategy,
            histories=eval_histories,
            regime=current_regime,
            macro=macro,
        )
        regime_eval_text = format_evaluations_text(regime_evals)
        base_eval_text = format_evaluations_text(evals)
        if regime_eval_text != base_eval_text and base_eval_text in fact_sheet:
            fact_sheet = fact_sheet.replace(base_eval_text, regime_eval_text, 1)
        evals = regime_evals
        evals_by_sym: dict[str, list[dict[str, Any]]] = {}
        for ev in evals:
            evals_by_sym.setdefault(ev.yahoo_symbol, []).append(ev.model_dump())
        for sym_key, entry in (context_pack.get("symbols") or {}).items():
            if isinstance(entry, dict):
                entry["evaluations"] = evals_by_sym.get(sym_key, [])
    context_pack["regime"] = current_regime
    regime_anchor = _build_regime_anchor_text(
        cfg,
        gate_preview,
        current_regime,
        vix=(macro.vix if (macro is not None and macro.vix is not None) else None),
    )
    shared_context = (
        f"时间：英国时间 {now_uk}（{get_uk_weekday_label()} | {dst_note}）\n\n"
        f"{regime_anchor}\n\n"
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
        + (f"\n\n{core_policy_context}" if core_policy_context else "")
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
        "3. 哪些持仓具备真实支撑值得考虑加仓或顺势持有（写明支撑位/目标价与依据）："
        "ISA 价值仓按「护城河/财务质量 + 估值安全边际（常规≥12%）」判断，优质股回撤是分批左侧加仓机会（写明基本面未证伪依据）；"
        "Invest 右侧仓按当日 Regime 动态 R:R（RISK_ON 领涨≥1.6 / NEUTRAL≥2.0 / 弱势≥2.5）与硬止损判断，绝不摊平；\n"
        f"4. 资金部署：若 ISA 现金占比 >{cfg.strategy.isa_cash_deploy_trigger_pct:.0f}% 且核心宽基未达标，"
        "只针对【今日核心宽基政策票据】中 Python 唯一签发的代码给出价位/节奏建议，"
        "严禁自行更换代码或放大金额；票据为空时明确建议不下新单；Invest 机动仓严格遵循右侧猎物门控，"
        "无达到当日动态 R:R 门槛的强信号时明确维持持币待命，严禁因现金占比高而勉强凑单；\n"
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
        "2. 破位与弱势持仓审计（双哲学分开审）：ISA 价值仓只看护城河/财务质量恶化与极端估值泡沫，"
        "不得因单纯跌破 MA200、RSI≥75 或浮盈 25% 催卖；Invest 右侧仓看趋势破位、前瞻 R:R 恶化与硬止损（含临近 5 日内财报）；"
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
    pm_sys, pm_prompt = _build_morning_pm_prompt(
        cfg=cfg,
        debate_pm_block=debate_pm_block,
        calendar_alerts=calendar_alerts,
        past_lessons=past_lessons,
        wallet_text=wallet_text,
        diff_text=diff_text,
        benchmark_line=benchmark_line,
        trade_ctx=trade_ctx,
        cash_flow_text=cash_flow_text,
        disagreement_text=disagreement_text,
        core_policy_context=core_policy_context,
        analyst_notes=analyst_notes,
        auditor_notes=auditor_notes,
        gate_preview=gate_preview,
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
    # [用户授权调整 v5.5.2] /report 安全天花板 2800 字（Prompt 控制 2100~2500 字）；IMP-12 接线时请保留 2800 上限。
    max_body_chars = 2800 - len(holiday_banner)
    if decision is not None:
        override_res = apply_risk_override_guardrail(
            decision,
            portfolio=portfolio,
            macro=macro,
            strategy=cfg.strategy,
            bundle_map=bundles,
            evals=evals,
            trade_behavior=trade_ctx.get("trade_behavior"),
            deployment_gate=gate_preview,
            tech_map=tech_map,
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
                pool_snaps=pool_snaps,
                cached_breadth_histories=cached_breadth_histories,
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

        # v5.5.1: deterministic rail for un-held core/diversifier ETF BUYs.
        resolved_decision, core_dep_notes = _validate_core_deployment(
            resolved_decision, portfolio, cfg, gate_dict, bundles
        )
        if core_dep_notes or override_res.decision is not resolved_decision:
            override_res = override_res.model_copy(
                update={
                    "decision": resolved_decision,
                    "applied": bool(override_res.applied or core_dep_notes),
                    "notes": [*override_res.notes, *core_dep_notes],
                }
            )

        context_pack["debate"] = debate.payload
        context_pack["report_decision"] = {
            "headline": resolved_decision.headline,
            "top_actions": [a.model_dump() for a in resolved_decision.top_actions],
            "pending_order_audits": [a.model_dump() for a in resolved_decision.pending_order_audits],
            "core_deployment": [a.model_dump() for a in resolved_decision.core_deployment],
            "daily_vetoes": list(resolved_decision.daily_vetoes),
        }
        final_report = holiday_banner + render_morning_report(
            resolved_decision,
            max_chars=max_body_chars,
            risk_override=override_res,
            gate_banner=gate_banner,
            size_notes=build_report_size_notes(
                resolved_decision.top_actions,
                portfolio,
                strategy=cfg.strategy,
            ),
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

    await _persist_morning_report_and_audit(
        cfg=cfg,
        db=db,
        llm=llm,
        portfolio=portfolio,
        histories=histories,
        bundles=bundles,
        evals=evals,
        trade_ctx=trade_ctx,
        fact_sheet=fact_sheet,
        core_policy_context=core_policy_context,
        resolved_decision=resolved_decision,
        final_report=final_report,
        short_conclusion=short_conclusion,
        guardrails=guardrails,
        context_pack=context_pack,
        ops_notifier=ops_notifier,
    )
    return final_report
