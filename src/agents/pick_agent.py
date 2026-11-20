"""Aligned /pick & Weekly Radar: Inherits /report guardrails -> Screener -> Researcher -> Risk Critic (<=1000 chars)."""

from __future__ import annotations

import asyncio
import logging
import re
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
    TechnicalSnapshot,
    classify_etf_distribution_policy,
    compute_correlation_matrix,
    compute_exposure_summary,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
    normalize_issuer_root,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
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
    _calibrate_target_price,
    _estimate_entry_friction_pct,
    _estimate_pending_order_gbp,
    _rescale_price_pair,
    apply_risk_override_guardrail,
    build_etf_to_equity_funding_note,
    evaluate_isa_value_gate,
    evaluate_portfolio,
    format_evaluations_text,
)
from src.engine.screener import ScreenerCandidate, run_quant_screener
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

_GUARDRAIL_TAG = "[🛡️风控护栏修正]"

# v5.5.1: taxable Invest must never buy accumulating core ETFs (offshore ERI reporting
# friction); only distributing share classes are allowed there. VERG.L / CUKX.L are
# ACCUMULATING share classes — their distributing equivalents are VERX.L / ISF.L.
_DISTRIBUTING_CORE_ETFS = {
    "VWRL", "VWRL.L",
    "VUSA", "VUSA.L",
    "IUSA", "IUSA.L",
    "ISF", "ISF.L",
    "VUKE", "VUKE.L",
    "VERX", "VERX.L",
    "VEUR", "VEUR.L",
    "IMEU", "IMEU.L",
}


def _format_gate_constraint_block(
    gate_ctx: dict[str, Any],
    portfolio: PortfolioData | None = None,
    *,
    strategy: StrategySection | None = None,
) -> str:
    """Render the deterministic deployment-gate constraints for Researcher/Critic prompts."""
    isa_g = gate_ctx.get("isa") if isinstance(gate_ctx.get("isa"), dict) else {}
    inv_g = gate_ctx.get("invest") if isinstance(gate_ctx.get("invest"), dict) else {}
    lines = ["【今日组合门控约束（确定性底线，严禁违反）】"]
    # v5.5.7: sync with today's /report approval + live pending BUY orders so /pick never
    # re-issues a core ticket that the morning report (or a resting order) already funded.
    approved_core_gbp = float(isa_g.get("approved_report_core_gbp") or 0.0)
    pending_buy_syms: set[str] = set()
    if portfolio is not None:
        for order in portfolio.all_pending_orders():
            if str(getattr(order, "side", "") or "").strip().upper() != "BUY":
                continue
            for raw in (getattr(order, "yahoo_symbol", ""), getattr(order, "ticker", "")):
                sym = str(raw or "").strip().upper()
                if sym:
                    pending_buy_syms.update({sym, sym.split(".")[0]})
    if isa_g.get("isa_satellite_saturated"):
        lines.append(
            f"• ISA 个股卫星层已饱和（持仓+挂单 {isa_g.get('committed_satellite_count')}只/"
            f"{float(isa_g.get('committed_satellite_pct') or 0.0):.1f}%）：ISA 赛道只允许宽基 ETF，"
            "严禁推荐新开 ISA 个股；核心宽基只能按 Python 政策票据代码执行。"
        )
    if isa_g.get("equity_open") is False:
        lines.append("• ISA 个股通道关闭（退潮期或结构饱和）：严禁推荐 ISA 新开个股。")
    if isa_g.get("etf_open") is False:
        lines.append("• ISA 宽基通道关闭（退潮期未达深跌采集阶梯）：严禁推荐 ISA 新开宽基 ETF。")
    if isa_g.get("prefer_core_diversifier"):
        plan = isa_g.get("core_allocation_plan") if isinstance(isa_g.get("core_allocation_plan"), dict) else {}
        tickets = plan.get("tickets") or []
        if approved_core_gbp > 0.0:
            lines.append(
                f"• 今日 /report 已批核心宽基定投 £{approved_core_gbp:,.0f}，/pick 不再重复签发核心宽基票据："
                "专注 Invest 右侧机会或必要换仓（同日双通道重复扣减现金为硬拦截）。"
            )
        elif tickets:
            t = tickets[0]
            ticket_sym = str(t.get("symbol") or "").strip().upper()
            already_pending = bool(
                ticket_sym and ticket_sym in pending_buy_syms
            )
            lines.append(
                f"• 今日核心宽基政策票据：{t.get('symbol')}（子桶 {t.get('sleeve_label')} "
                f"{float(t.get('current_pct') or 0.0):.1f}%→目标 {float(t.get('target_pct') or 0.0):.1f}%），"
                f"金额上限 £{float(t.get('recommended_gbp') or 0.0):,.0f}；"
                "只能按该代码执行，严禁自行换代码或超额；已持有子桶走 top_actions ADD。"
                + ("该标的已挂单执行中，勿重复建仓。" if already_pending else "")
            )
        else:
            lines.append(
                "• ISA 核心宽基仍有区域缺口但今日无 Python 政策票据：不得自行挑选宽基代码，"
                "现金不足/趋势破位/已有挂单时保持等待。"
            )
    lines.append(
        f"• Invest 状态 {inv_g.get('status')}：在险槽位 {inv_g.get('slots_used')}/{inv_g.get('slots_cap')}；"
        f"{inv_g.get('reason') or ''}"
    )
    # v5.5.9 M5: Invest distributing-ETF beta reservoir — admission ticket and the
    # zero-quota funding source, both Python-owned so the LLM never invents amounts.
    res = inv_g.get("reservoir_advice") if isinstance(inv_g.get("reservoir_advice"), dict) else {}
    reservoir_held_syms = [str(s) for s in (res.get("held_symbols") or []) if str(s).strip()]
    if res.get("eligible"):
        reserve_pct = float(
            getattr(strategy, "invest_reservoir_cash_reserve_pct", 30.0) if strategy else 30.0
        )
        lines.append(
            f"• Invest 派息宽基蓄水池票据：{res.get('ticket_symbol')} £{float(res.get('ticket_gbp') or 0.0):,.0f}"
            f"（仅吸收超出{reserve_pct:.0f}%流动预留带的溢出现金，零槽位零热力零Quota；"
            "严禁自行更换宽基代码或买入累积型 ETF）"
        )
    elif reservoir_held_syms:
        reservoir_total = float(inv_g.get("reservoir_total_gbp") or 0.0)
        lines.append(
            f"• Invest 持有派息宽基蓄水池 {','.join(reservoir_held_syms[:2])} £{reservoir_total:,.0f}"
            "（零Quota资金源：若建议建仓金额>可用现金，Python 将生成清宽基配对指令，严禁自行口算股数）"
        )
    if inv_g.get("invest_min_rr_floor") is not None:
        lines.append(
            f"• Invest 当日动态净 R:R 门槛 ≥{float(inv_g.get('invest_min_rr_floor') or 0.0):.1f}:1"
            f"（{inv_g.get('invest_rr_tier_label') or '动态门槛'}；含摩擦止损分母，Python 独立复算）。"
        )
    committed_heat = float(inv_g.get("committed_heat_pct") or 0.0)
    heat_cap = float(inv_g.get("heat_cap_pct") or 0.0)
    unit_risk = float(inv_g.get("unit_risk_gbp") or 0.0)
    if heat_cap > 0:
        risk_txt = f"，当前单笔风险预算 £{unit_risk:.0f}" if unit_risk > 0 else ""
        lines.append(
            f"• Invest 承诺热力 {committed_heat:.1f}%/{heat_cap:.1f}%"
            f"（含未成交新开买单 1R 承诺{risk_txt}）：触顶即须先上移止损/推保本降热。"
        )
    if inv_g.get("status") == "ROTATE":
        lines.append(
            f"• ROTATE 等风险置换：Invest 新开仓必须先在 thesis 写明以平掉/撤掉 "
            f"{inv_g.get('funding_from') or '现有未脱险弱仓'} 为置换资金源，严禁动用闲置现金净增槽位。"
        )
    scale = float(inv_g.get("risk_budget_scale") or 1.0)
    if 0.0 < scale < 1.0:
        scale_reason = str(inv_g.get("risk_scale_reason") or "").strip() or "连损恢复试探"
        pilot_floor = float(
            inv_g.get("min_invest_pilot_order_gbp")
            or getattr(strategy, "min_invest_pilot_order_gbp", 500.0)
            or 500.0
        )
        lines.append(
            f"• {scale_reason}：Invest 单笔风险预算按 {scale:.1f}R 轻兵试探执行（试探最小单笔地板 £{pilot_floor:,.0f}）。"
        )
    lines.append(f"• 市场状态 {gate_ctx.get('regime')}，全局门控 {gate_ctx.get('global_status')}。")
    return "\n".join(lines) + "\n"


def _is_etf_action(symbol: str, bundles: dict[str, TickerMarketBundle] | None, etf_set: set[str]) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    base = sym.split(".")[0]
    if sym in etf_set or base in etf_set:
        return True
    for candidate in (sym, symbol, base):
        bundle = (bundles or {}).get(candidate)
        if bundle is not None and str(getattr(bundle, "quote_type", "") or "").strip().upper() == "ETF":
            return True
    return False


def _apply_gate_channel_constraints(
    decision: PickDecision,
    portfolio: PortfolioData,
    gate_ctx: dict[str, Any],
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> tuple[PickDecision, list[str]]:
    """v5.4.0 post-guardrail hard interception: PM veto / account channels / dedupe / ROTATE.

    Never mutates the caller's decision object; returns the hardened copy and human-readable
    rejection notes (tagged so render_pick_report surfaces them).
    """
    inv_g = gate_ctx.get("invest") if isinstance(gate_ctx.get("invest"), dict) else {}
    isa_g = gate_ctx.get("isa") if isinstance(gate_ctx.get("isa"), dict) else {}
    pm_g = gate_ctx.get("pm") if isinstance(gate_ctx.get("pm"), dict) else {}
    funding_from = str(inv_g.get("funding_from") or "").strip().upper()
    etf_set = {str(s).strip().upper() for s in (gate_ctx.get("etf_set") or []) if str(s).strip()}
    core_policy = isa_g.get("core_policy") if isinstance(isa_g.get("core_policy"), dict) else {}
    core_bucket_by_symbol = (
        core_policy.get("bucket_by_symbol") if isinstance(core_policy.get("bucket_by_symbol"), dict) else {}
    )
    buy_actions = {"BUY", "ADD", "SWAP"}
    inv_status = str(inv_g.get("status") or "").strip().upper()
    inv_mode = str(inv_g.get("mode") or "").strip().lower()

    def _int_or_zero(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    # Invest new-equity budget for this single /pick run: a ROTATE swap gets exactly one
    # ticket, otherwise only the free risk slots (minus already-used ones) may be filled.
    if inv_status == "ROTATE" and funding_from:
        rem_invest_slots = 1
    else:
        rem_invest_slots = max(0, _int_or_zero(inv_g.get("slots_cap")) - _int_or_zero(inv_g.get("slots_used")))
    daily_cap = _int_or_zero(inv_g.get("daily_cap"))
    rem_daily_entries: int | None = (
        max(0, daily_cap - _int_or_zero(inv_g.get("today_new_equity_entries"))) if daily_cap > 0 else None
    )

    held_by_acct: dict[str, set[str]] = {"ISA": set(), "INVEST": set()}
    pending_buy_by_acct: dict[str, set[str]] = {"ISA": set(), "INVEST": set()}
    for pos in portfolio.all_positions():
        sym = pos.yahoo_symbol.upper()
        held_by_acct.setdefault(pos.account.strip().upper(), set()).update({sym, sym.split(".")[0]})
    for order in portfolio.all_pending_orders():
        if str(order.side or "").strip().upper() != "BUY":
            continue
        sym = order.yahoo_symbol.upper()
        pending_buy_by_acct.setdefault(order.account.strip().upper(), set()).update({sym, sym.split(".")[0]})

    other_acct = {"ISA": "INVEST", "INVEST": "ISA"}
    kept: list[Any] = []
    notes: list[str] = []
    kept_new_invest: set[str] = set()
    kept_isa_core = 0
    for item in decision.selected_actions:
        act_up = str(item.action or "").strip().upper()
        sym = item.symbol.strip().upper()
        base = sym.split(".")[0]
        acct = item.account.strip().upper()
        is_etf = False
        is_core_etf = False
        if act_up in buy_actions:
            is_etf = _is_etf_action(sym, bundles, etf_set)
            # Themed / dividend / bond ETFs are NOT core wide-base vehicles: in Invest they
            # must obey the same slot, daily-cap and regime channel as single-name equities.
            is_core_etf = bool(is_etf and (sym in etf_set or base in etf_set))
            if acct == "INVEST" and not is_core_etf and act_up in {"BUY", "ADD"}:
                existing_pending = pending_buy_by_acct.get("INVEST", set())
                if sym in existing_pending or base in existing_pending:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "Invest 已有该标的未成交买入挂单，严禁重复建仓叠加热力"
                    )
                    continue
            bundle = None
            if bundles:
                bundle = bundles.get(sym) or bundles.get(base)
            if acct == "INVEST" and is_etf:
                dist_policy = classify_etf_distribution_policy(sym, bundle)
                if is_core_etf and (
                    dist_policy != "DIST"
                    and sym not in _DISTRIBUTING_CORE_ETFS
                    and base not in _DISTRIBUTING_CORE_ETFS
                ):
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "应税 Invest 仅可买派息型核心宽基（离岸 ERI 报税摩擦），"
                        "仅可买派息型 VWRL.L/ISF.L/VERX.L/VEUR.L；"
                        "累计型 VWRP/VERG/CUKX 等严禁买入"
                    )
                    continue
                if not is_core_etf and dist_policy == "ACC":
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "应税 Invest 禁止买入累积型 ETF（CNX1/EQGB/IITU/IBTA 等离岸 ERI 报税摩擦）"
                    )
                    continue
            if acct == "INVEST" and is_core_etf:
                if (
                    str(gate_ctx.get("regime") or "").strip().upper() == "RISK_OFF"
                    and isa_g.get("etf_open") is not True
                ):
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "RISK_OFF 未达深跌阶梯，Invest 不得新建核心宽基，优先保留现金流"
                    )
                    continue
                if (
                    float(isa_g.get("deployable_pct") or 0.0) > 15.0
                    and float(isa_g.get("committed_core_etf_pct") or 0.0)
                    < float(core_policy.get("core_target_pct") or 60.0)
                ):
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "免税 ISA 仍有现金与核心缺口，核心宽基优先在 ISA 部署"
                    )
                    continue
            if acct == "ISA" and is_core_etf:
                if float(isa_g.get("approved_report_core_gbp") or 0.0) > 0.0:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "今日 /report 已安排核心宽基定投，避免同日双通道重复扣减现金"
                    )
                    continue
                bucket = core_bucket_by_symbol.get(sym) or core_bucket_by_symbol.get(base) or ""
                if bucket == "US_TILT" and core_policy.get("us_tilt_buy_frozen"):
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "美股与标普500敞口已超上限，禁止继续加码标普500宽基"
                    )
                    continue
                if kept_isa_core >= 1:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "单次 /pick 最多保留 1 条 ISA 核心宽基动作"
                    )
                    continue
            # PM veto is a hard stop for all new positions (kept as a defence-in-depth
            # check even though generate_pick_radar already returns early).
            if pm_g.get("allow") is False:
                notes.append(
                    f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                    f"PM 今日门控收紧：{pm_g.get('reason') or '暂停新开仓'}"
                )
                continue
            # Account-channel hard blocks (only in advise mode, because gate_ctx is None otherwise).
            if acct == "INVEST" and not is_core_etf:
                if inv_status == "RUN":
                    committed_heat = float(inv_g.get("committed_heat_pct") or 0.0)
                    heat_cap = float(inv_g.get("heat_cap_pct") or 0.0)
                    if heat_cap > 0 and committed_heat >= heat_cap:
                        notes.append(
                            f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                            f"Invest 承诺热力 {committed_heat:.2f}%/{heat_cap:.2f}% 已触顶，"
                            "严禁新开个股，须先上移止损/推保本平滑降热"
                        )
                        continue
                if inv_status not in {"RUN", "ROTATE"}:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        f"Invest {inv_status or 'UNKNOWN'}"
                        f"{('/' + inv_mode) if inv_mode else ''} 冻结新开个股/非核心 ETF"
                        f"（{inv_g.get('reason') or inv_status or '门控未开放'}）"
                    )
                    continue
            if acct == "ISA":
                if is_etf and isa_g.get("etf_open") is False:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "ISA 宽基通道关闭（退潮期未达深跌阶梯）"
                    )
                    continue
                if not is_etf and isa_g.get("equity_open") is False:
                    notes.append(
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        "ISA 个股通道关闭（退潮期或结构饱和）"
                    )
                    continue
            # Cross-account same-ticket dedupe applies to single-name equities only;
            # broad-base/core ETFs are intentionally allowed in both accounts.
            other = other_acct.get(acct, "")
            other_held = held_by_acct.get(other, set()) | pending_buy_by_acct.get(other, set())
            if other and not is_etf and (
                sym in other_held
                or base in other_held
                or normalize_issuer_root(sym) in {normalize_issuer_root(s) for s in other_held}
            ):
                note = (
                    f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                    f"跨账户同票去重：{item.symbol} 已在 {other} 持仓/挂单，请归口原持仓账户管理"
                )
                notes.append(note)
                continue
            if funding_from and acct == "INVEST" and inv_g.get("status") == "ROTATE":
                combined = f"{item.thesis} {item.risk_note}".upper()
                if funding_from and funding_from not in combined and funding_from.split(".")[0] not in combined:
                    note = (
                        f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                        f"ROTATE 等风险置换：Invest 新开仓须在 thesis 中写明先平/撤 {funding_from} 的置换资金源"
                    )
                    notes.append(note)
                    continue
            # One /pick run must never fan out 2~3 Invest equity tickets and blow through
            # both the risk-slot cap and the best-idea-only daily cap.
            if acct == "INVEST" and not is_core_etf and act_up in {"BUY", "SWAP"}:
                own_held = held_by_acct.get("INVEST", set()) | pending_buy_by_acct.get("INVEST", set())
                if (
                    sym not in own_held
                    and base not in own_held
                    and sym not in kept_new_invest
                    and base not in kept_new_invest
                ):
                    if rem_invest_slots <= 0:
                        notes.append(
                            f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                            f"Invest 本次 /pick 在险槽位额度已用尽（剩余 {rem_invest_slots} 笔）"
                        )
                        continue
                    if rem_daily_entries is not None and rem_daily_entries <= 0:
                        notes.append(
                            f"{_GUARDRAIL_TAG} [{item.account}] {item.symbol}({item.action}→拦截): "
                            "Invest 单日新开个股额度已用尽（Best Idea Only）"
                        )
                        continue
                    rem_invest_slots -= 1
                    if rem_daily_entries is not None:
                        rem_daily_entries -= 1
                    kept_new_invest.update({sym, base})
            if acct == "ISA" and is_core_etf:
                kept_isa_core += 1
        kept.append(item)

    if not notes:
        return decision, []
    hardened = decision.model_copy(
        update={
            "selected_actions": kept,
            "rejected_or_frozen": [*(decision.rejected_or_frozen or []), *notes],
        }
    )
    return hardened, notes


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
    *,
    portfolio: PortfolioData | None = None,
) -> tuple[str, dict[str, list[str]], list[dict[str, Any]]]:
    """Extract /report top_actions & pending_order_audits for /pick alignment and screener tagging.

    v5.5.7: verdicts already executed/repriced since 07:30 are dropped by diffing against the
    live portfolio (a filled or cancelled order must not reappear as today's open verdict).
    """
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
    live_pending: set[str] = set()
    live_held: set[str] = set()
    if portfolio is not None:
        for order in portfolio.all_pending_orders():
            for raw in (getattr(order, "yahoo_symbol", ""), getattr(order, "ticker", "")):
                s = str(raw or "").strip().upper()
                if s:
                    live_pending.update({s, s.split(".")[0]})
        for pos in portfolio.all_positions():
            for raw in (getattr(pos, "yahoo_symbol", ""), getattr(pos, "ticker", "")):
                s = str(raw or "").strip().upper()
                if s:
                    live_held.update({s, s.split(".")[0]})

    def _live(sym: str, pool: set[str]) -> bool:
        return bool(sym and (sym in pool or sym.split(".")[0] in pool))

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
        if portfolio is not None and verb in {"TRIM", "STOP", "HOLD"} and not _live(sym, live_held):
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
        if portfolio is not None and not _live(sym, live_pending):
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
    # v5.5.7: the ID filter now sees the odds/friction metrics that actually decide the ticket.
    metrics = candidate.filter_metrics or {}
    metric_bits: list[str] = []
    sector = str(metrics.get("sector") or "").strip()
    region = str(metrics.get("region") or "").strip()
    location = "/".join(x for x in (sector, region) if x)
    if location:
        metric_bits.append(location)
    if metrics.get("target_upside_pct") is not None:
        metric_bits.append(f"目标{float(metrics['target_upside_pct']):+.0f}%")
    if metrics.get("atr_pct") is not None:
        metric_bits.append(f"ATR14 {float(metrics['atr_pct']):.1f}%")
    if metrics.get("rs_60d_pct") is not None:
        metric_bits.append(f"60日RS {float(metrics['rs_60d_pct']):+.1f}%")
    if candidate.track != "ISA_LONG_TERM" and metrics.get("est_rr") is not None:
        metric_bits.append(f"预估R:R {float(metrics['est_rr']):.2f}:1")
    if candidate.track == "ISA_LONG_TERM":
        if metrics.get("mos_pct") is not None:
            metric_bits.append(f"安全边际 {float(metrics['mos_pct']):+.1f}%")
        if metrics.get("fcf_yield_pct") is not None:
            metric_bits.append(f"FCF {float(metrics['fcf_yield_pct']):.1f}%")
    metrics_str = f" | {' | '.join(metric_bits)}" if metric_bits else ""
    return (
        f"{candidate.symbol} | {candidate.track} | 分 {candidate.score:.1f} | {dims or '无维度'} | "
        f"{candidate.asset_type}/{candidate.currency or '?'}{metrics_str} | 优点 {pros or '-'} | 风险 {cons or '-'} | "
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


def _build_proposal_fact_card(
    research_output: str,
    *,
    isa_cands: list[ScreenerCandidate],
    invest_cands: list[ScreenerCandidate],
    merged_bundles: dict[str, TickerMarketBundle],
    tech_map: dict[str, Any],
    stage2_tech: dict[str, Any],
    toolkit: MarketToolkit,
    portfolio: PortfolioData,
    cfg: AppConfig,
    macro: Any,
    regime: str,
) -> tuple[str, set[str]]:
    """0-token one-hand fact card for the Critic, restricted to a strict symbol whitelist.

    The whitelist is the intersection of the Researcher's prose with symbols that really exist in
    this cycle's candidates / inspect cache / holdings, so words like ATR/RSI/VIX/ISA can never be
    mistaken for tickers.
    """
    candidate_corr: dict[str, tuple[float | None, str | None]] = {}
    candidate_syms: list[str] = []
    for cand in (*isa_cands, *invest_cands):
        sym_up = cand.symbol.strip().upper()
        if not sym_up:
            continue
        candidate_syms.append(sym_up)
        candidate_corr[sym_up] = (cand.most_correlated_holding, cand.max_portfolio_corr)
    inspect_syms = [str(s).strip().upper() for s in getattr(toolkit, "_inspect_cache", {}) or {}]
    held_syms = [str(s).strip().upper() for s in merged_bundles]
    whitelist = list(dict.fromkeys([*inspect_syms, *candidate_syms, *held_syms]))

    text_up = str(research_output or "").upper()
    matched: list[str] = []
    for sym in whitelist:
        base = sym.split(".")[0]
        tokens = {sym, base}
        hit = any(
            tok
            and re.search(rf"(?<![A-Z0-9]){re.escape(tok)}(?![A-Z0-9])", text_up)
            for tok in tokens
        )
        if hit:
            matched.append(sym)
        if len(matched) >= 5:
            break
    if not matched:
        return "", set()

    lines = ["【提案标的一手事实卡（Python 独立真源，直接对照核验）】"]
    crisis = str(regime or "").upper() == "RISK_OFF"
    for sym in matched:
        bundle = _lookup_ci(merged_bundles, sym)
        tech = _lookup_ci(stage2_tech, sym) or _lookup_ci(tech_map, sym)
        if bundle is None and tech is None:
            continue
        price = float(bundle.current_price) if (bundle and bundle.current_price) else (
            float(tech.last_close) if tech is not None else 0.0
        )
        ccy = str(getattr(bundle, "currency", "") or "")
        sector = str(getattr(bundle, "sector", "") or getattr(bundle, "industry", "") or "")[:16]
        region = "UK" if sym.endswith(".L") else ("EU" if sym.endswith((".DE", ".PA", ".AS", ".MI", ".MC", ".SW", ".F")) else "US")
        atr_txt = (
            f"{float(tech.atr14):.2f}({float(tech.atr_pct):.1f}%)"
            if (tech is not None and tech.atr14 is not None and tech.atr_pct is not None)
            else "N/A"
        )
        rsi_txt = f"{float(tech.rsi14):.0f}" if (tech is not None and tech.rsi14 is not None) else "N/A"
        ma50_txt = f"{float(tech.dist_ma50_pct):+.1f}%" if (tech is not None and tech.dist_ma50_pct is not None) else "N/A"
        ma200_txt = f"{float(tech.dist_ma200_pct):+.1f}%" if (tech is not None and tech.dist_ma200_pct is not None) else "N/A"
        corr_sym, corr_val = candidate_corr.get(sym, (None, None))
        if corr_val is None:
            corr_entry = (getattr(toolkit, "_corr_cache", {}) or {}).get(sym) or {}
            corr_val = corr_entry.get("max_correlation_60d")
            corr_sym = corr_entry.get("most_correlated_holding")
        corr_txt = f"{corr_sym} {float(corr_val):.2f}" if (corr_val is not None and corr_sym) else "N/A"
        earn_days = getattr(bundle, "days_to_earnings", None) if bundle is not None else None
        ex_days = getattr(bundle, "days_to_ex_dividend", None) if bundle is not None else None
        earn_txt = f"{int(earn_days)}天" if earn_days is not None else "无"
        ex_txt = f"{int(ex_days)}天" if ex_days is not None else "无"
        sell_txt = "N/A"
        if (
            bundle is not None
            and (bundle.analyst_count or 0) >= 3
            and bundle.target_mean_price is not None
            and bundle.target_mean_price > 0
            and price > 0
        ):
            mean_t = _calibrate_target_price(
                float(bundle.target_mean_price), price, symbol=sym, currency=ccy
            )
            if bundle.target_high_price is not None and float(bundle.target_high_price) > 0:
                high_t = _calibrate_target_price(
                    float(bundle.target_high_price), price, symbol=sym, currency=ccy
                )
                sell_txt = f"{mean_t:.2f}/高位{high_t:.2f}({int(bundle.analyst_count)}家)"
            else:
                sell_txt = f"{mean_t:.2f}({int(bundle.analyst_count)}家)"
        rs_txt = ""
        if tech is not None and (tech.rs_20d_pct is not None or tech.rs_60d_pct is not None):
            r20 = f"{float(tech.rs_20d_pct):+.1f}%" if tech.rs_20d_pct is not None else "N/A"
            r60 = f"{float(tech.rs_60d_pct):+.1f}%" if tech.rs_60d_pct is not None else "N/A"
            rs_txt = f" | RS20/60 {r20}/{r60}"
        isa_fric = _estimate_entry_friction_pct(sym, bundle, cfg.strategy, "ISA", None)
        inv_fric = _estimate_entry_friction_pct(
            sym, bundle, cfg.strategy, "INVEST", portfolio.invest
        )
        fric_txt = (
            f"ISA {isa_fric:.2f}%/Invest {inv_fric:.2f}%"
            if abs(isa_fric - inv_fric) >= 0.01
            else f"{isa_fric:.2f}%"
        )
        line = (
            f"• {sym} [{sector or '-'}/{region}] | 现价 {price:.2f} {ccy} | ATR14 {atr_txt} | RSI14 {rsi_txt}"
            f" | 距MA50/MA200 {ma50_txt}/{ma200_txt}{rs_txt} | 60日最大持仓相关 {corr_txt}"
            f" | 财报 {earn_txt} | 除息 {ex_txt} | 卖方均值 {sell_txt} | 摩擦 {fric_txt}"
        )
        if sym.endswith(".L") and not str(getattr(bundle, "quote_type", "") or "").upper() == "ETF":
            gate = evaluate_isa_value_gate(
                bundle,
                tech,
                macro,
                cfg.strategy,
                symbol=sym,
                crisis_mode=crisis,
            )
            line += f" | ISA价值门预判：{'通过' if gate.get('passed') else '未通过'}（{gate.get('summary')}）"
        lines.append(line)
    if len(lines) <= 1:
        return "", set()
    return "\n".join(lines), set(matched)


def _filter_audit_text_for_critic(audit_text: str, proposal_syms: set[str]) -> str:
    """Keep all live pending-order verdicts, plus only holding verdicts tied to this proposal."""
    if not audit_text:
        return ""
    kept: list[str] = []
    for line in audit_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "[挂单裁决]" in stripped:
            kept.append(stripped)
            continue
        if "[持仓裁决]" in stripped:
            upper = stripped.upper()
            if any(
                sym and re.search(rf"(?<![A-Z0-9]){re.escape(sym.split('.')[0])}(?![A-Z0-9])", upper)
                for sym in proposal_syms
            ):
                kept.append(stripped)
    return "\n".join(kept)


_ENTRY_GBP_RE = re.compile(r"£\s*([\d,]+(?:\.\d+)?)")


def _build_deterministic_cash_note(
    decision: PickDecision,
    portfolio: PortfolioData,
    sizing_map: dict[str, Any] | None,
    gate_ctx: dict[str, Any] | None,
) -> str:
    """Python-owned post-execution cash line so a Critic's pre-interception arithmetic can't leak."""
    sizing_map = sizing_map or {}
    isa_g = gate_ctx.get("isa") if isinstance((gate_ctx or {}).get("isa"), dict) else {}
    plan = isa_g.get("core_allocation_plan") if isinstance(isa_g.get("core_allocation_plan"), dict) else {}
    ticket_caps: dict[str, float] = {}
    for tk in plan.get("tickets") or []:
        if isinstance(tk, dict) and tk.get("symbol"):
            ticket_caps[str(tk.get("symbol")).strip().upper()] = float(tk.get("recommended_gbp") or 0.0)

    remaining = {
        "ISA": float(portfolio.isa.available_to_trade or 0.0),
        "INVEST": float(portfolio.invest.available_to_trade or 0.0),
    }
    pending_gbp: dict[tuple[str, str], float] = {}
    pos_by_sym = {p.yahoo_symbol.upper(): p for p in portfolio.all_positions()}
    for order in portfolio.all_pending_orders():
        if str(order.side or "").strip().upper() != "BUY":
            continue
        acct = str(order.account or "").strip().upper()
        sym = str(order.yahoo_symbol or order.ticker or "").strip().upper()
        if not sym:
            continue
        pending_gbp[(acct, sym)] = pending_gbp.get((acct, sym), 0.0) + float(
            _estimate_pending_order_gbp(order, pos_by_symbol=pos_by_sym) or 0.0
        )

    for item in decision.selected_actions:
        act = str(item.action or "").strip().upper()
        if act not in {"BUY", "ADD", "SWAP"}:
            continue
        acct = str(item.account or "").strip().upper()
        sym = str(item.symbol or "").strip().upper()
        if acct not in remaining or not sym:
            continue
        size = sizing_map.get(f"{acct}|{sym}") or {}
        suggested = float(size.get("suggested_gbp") or 0.0) if isinstance(size, dict) else 0.0
        planned_raw = getattr(item, "planned_amount_gbp", None)
        planned: float | None = None
        if planned_raw is not None and float(planned_raw) > 0:
            planned = float(planned_raw)
            if acct == "INVEST" and suggested > 0:
                planned = min(planned, suggested)
            elif acct == "ISA":
                cap = ticket_caps.get(sym) or ticket_caps.get(sym.split(".")[0]) or 0.0
                if cap > 0:
                    planned = min(planned, cap)
        if planned is None and suggested > 0:
            planned = suggested
        if planned is None:
            zone = str(getattr(item, "entry_zone", "") or "")
            match = _ENTRY_GBP_RE.search(zone)
            if match:
                try:
                    planned = float(match.group(1).replace(",", ""))
                except ValueError:
                    planned = None
        if planned is None or planned <= 0:
            continue
        existing = pending_gbp.get((acct, sym), 0.0)
        if existing <= 0:
            existing = pending_gbp.get((acct, sym.split(".")[0]), 0.0)
        deduction = max(0.0, float(planned) - float(existing))
        remaining[acct] = max(0.0, remaining[acct] - deduction)

    cash_line = (
        f"Python精确余现：ISA £{remaining['ISA']:,.0f} / Invest £{remaining['INVEST']:,.0f}"
        "（已扣本次执行额，不重复扣挂单）"
    )
    critic_note = str(decision.consistency_and_cash_note or "").strip()
    if not critic_note:
        return cash_line
    combined = f"{cash_line}；{critic_note[:45]}"
    if len(combined) > 95:
        # v5.5.7: the cash line is the non-negotiable tail — trim the critic declaration instead.
        tail_budget = 95 - len(cash_line) - 1
        combined = f"{cash_line}；{critic_note[:tail_budget]}" if tail_budget > 4 else cash_line
    return combined


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


_CORE_ETF_CN_ALIASES: dict[str, tuple[str, ...]] = {
    "ISF": ("富时100", "FTSE100", "FTSE 100"),
    "VUKE": ("富时100", "FTSE100", "FTSE 100"),
    "VUSA": ("标普500", "S&P500", "S&P 500"),
    "IUSA": ("标普500", "S&P500", "S&P 500"),
    "SPY": ("标普500", "S&P500", "S&P 500"),
    "VERX": ("标普500", "S&P500"),
    "VWRL": ("全球宽基", "全球ETF"),
    "VWRP": ("全球宽基", "全球ETF"),
    "VEUR": ("欧股宽基", "欧洲宽基"),
    "IMEU": ("欧股宽基", "欧洲宽基"),
}


def _reconcile_pick_headline(
    decision: PickDecision,
    *,
    original_actions: list[Any],
) -> PickDecision:
    """Keep PickDecision.headline consistent after Python guardrails intercept actions.

    Prevents semantic-audit violations where the LLM-written headline still promotes
    a ticker (or its Chinese broad-base alias) that was subsequently stripped by guardrails.
    """
    orig_syms = {
        str(getattr(a, "symbol", "") or "").strip().upper()
        for a in (original_actions or [])
        if str(getattr(a, "symbol", "") or "").strip()
        and str(getattr(a, "symbol", "") or "").strip().upper() != "CASH"
    }
    if not orig_syms:
        return decision

    kept_actions = list(decision.selected_actions or [])
    kept_syms = {
        str(getattr(a, "symbol", "") or "").strip().upper()
        for a in kept_actions
        if str(getattr(a, "symbol", "") or "").strip()
        and str(getattr(a, "symbol", "") or "").strip().upper() != "CASH"
    }
    intercepted_syms = orig_syms - kept_syms
    if not intercepted_syms:
        return decision

    if not kept_actions:
        return decision.model_copy(update={"headline": "本轮建议均被风控护栏拦截，现金待命"})

    headline_up = str(decision.headline or "").upper()
    kept_bases = {s.split(".")[0] for s in kept_syms}
    check_tokens: set[str] = set(intercepted_syms)
    for sym in intercepted_syms:
        base = sym.split(".")[0]
        if base and base not in kept_bases:
            check_tokens.add(base)

    leaked = any(
        re.search(rf"(?<![A-Z0-9]){re.escape(tok)}(?![A-Z0-9])", headline_up)
        for tok in check_tokens
    )
    if not leaked:
        # v5.5.7: also catch Chinese broad-base aliases leaking the intercepted core ETF.
        headline_raw = str(decision.headline or "")
        alias_tokens: set[str] = set()
        for sym in intercepted_syms:
            base = sym.split(".")[0]
            if base in kept_bases:
                continue
            for alias in _CORE_ETF_CN_ALIASES.get(base, ()):
                alias_tokens.add(alias)
        leaked = any(alias in headline_raw for alias in alias_tokens)
    if not leaked:
        return decision

    parts = [f"{a.account} {a.action} {a.symbol}" for a in kept_actions[:2]]
    rebuilt = f"风控筛后执行：{'、'.join(parts)}，其余现金待命"
    if len(rebuilt) > 55:
        rebuilt = rebuilt[:54] + "…"
    return decision.model_copy(update={"headline": rebuilt})


# v5.5.9 M6: shadow counterfactual triggers worth tracking (R:R floor family).
# RR_BELOW_FLOOR = computed R:R under the regime floor; RR_UNVERIFIED = LLM failed
# to give numeric stop/target so the floor could not even be verified — both are
# the same "R:R 门槛拦截" population for the avoided-loser / missed-winner study.
_SHADOW_RR_TRIGGERS: tuple[str, ...] = ("RR_BELOW_FLOOR", "RR_UNVERIFIED")
_SHADOW_TRIGGER_PRIORITY: tuple[str, ...] = ("RR_BELOW_FLOOR", "RR_UNVERIFIED", "CRITIC")
_SHADOW_MAX_ROWS_PER_PICK: int = 2


def _collect_guardrail_shadow_rows(
    hydrated_actions: list[Any],
    override_res: Any,
    kept_symbols: set[str],
    bundles: dict[str, Any],
    regime: str,
) -> list[dict[str, Any]]:
    """Build shadow-counterfactual rows for R:R-floor hard rejections (pure, offline-testable).

    Isolation guards:
    - ``kept_symbols`` (post-gate final selected actions) filters advise-mode
      advisories — only actions actually REMOVED from the ticket become shadows,
      so an advise-mode ticket that survives can never be double-counted.
    - Rows without a positive price anchor are skipped (no anchor -> no T+20).
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    overrides = list(getattr(override_res, "overrides", None) or [])
    by_key: dict[tuple[str, str], Any] = {}
    for rec in overrides:
        trigger = str(getattr(rec, "trigger_type", "") or "").strip().upper()
        if trigger not in _SHADOW_RR_TRIGGERS:
            continue
        key = (str(getattr(rec, "account", "") or "").strip().upper(), str(getattr(rec, "symbol", "") or "").strip().upper())
        if key in by_key:
            continue
        by_key[key] = rec
    for item in hydrated_actions:
        sym = str(item.symbol or "").strip().upper()
        acct = str(item.account or "").strip().upper()
        if not sym or sym == "CASH" or (acct, sym) not in by_key:
            continue
        if sym in kept_symbols:
            # advise/gate lanes kept the ticket alive — it is a REAL pick, not a shadow.
            continue
        rec = by_key[(acct, sym)]
        from_action = str(getattr(rec, "from_action", "") or "BUY").strip().upper()
        if from_action not in {"BUY", "ADD"}:
            continue
        bundle = _lookup_ci(bundles, sym)
        anchor = (
            float(bundle.current_price)
            if (
                bundle is not None
                and getattr(bundle, "current_price", None) is not None
                and float(bundle.current_price) > 0
            )
            else (float(item.entry_price) if item.entry_price and item.entry_price > 0 else None)
        )
        if anchor is None or anchor <= 0:
            continue
        if (acct, sym) in seen:
            continue
        seen.add((acct, sym))
        rows.append(
            {
                "account": item.account,
                "symbol": sym,
                "action": from_action,
                "entry_price": round(anchor, 4),
                "thesis": str(item.thesis or "")[:300],
                "horizon_days": 20,
                "benchmark_symbol": resolve_market_benchmark(sym)[0],
                "shadow_trigger": str(getattr(rec, "trigger_type", "") or "").strip().upper(),
                "shadow_reason": str(getattr(rec, "reason", "") or "").strip()[:200],
                "regime": str(regime or "").strip().upper()[:16],
            }
        )
    return rows


def _collect_critic_shadow_rows(
    proposal_syms: set[str],
    kept_symbols: set[str],
    held_symbols: set[str],
    isa_cands: list[Any],
    invest_cands: list[Any],
    bundles: dict[str, Any],
    regime: str,
) -> list[dict[str, Any]]:
    """Build shadow rows for Top candidates the Risk Critic dropped (pure, offline-testable).

    Only formal screener candidates count (not inspect-cache mentions or held
    symbols — a held-symbol mention is never a rejected new pick).
    """
    candidate_accounts: dict[str, str] = {}
    for cand in (*isa_cands, *invest_cands):
        sym_up = str(getattr(cand, "symbol", "") or "").strip().upper()
        if sym_up and sym_up not in candidate_accounts:
            # ScreenerCandidate carries a track tag, not an account label; derive
            # the account so ISA candidates never get misfiled into Invest.
            track = str(getattr(cand, "track", "") or "").strip().upper()
            candidate_accounts[sym_up] = "ISA" if track == "ISA_LONG_TERM" else "Invest"
    rows: list[dict[str, Any]] = []
    for sym in sorted(proposal_syms):
        sym_up = str(sym or "").strip().upper()
        if not sym_up or sym_up in kept_symbols or sym_up in held_symbols:
            continue
        if sym_up not in candidate_accounts:
            continue
        bundle = _lookup_ci(bundles, sym_up)
        anchor = (
            float(bundle.current_price)
            if (
                bundle is not None
                and getattr(bundle, "current_price", None) is not None
                and float(bundle.current_price) > 0
            )
            else None
        )
        if anchor is None or anchor <= 0:
            continue
        rows.append(
            {
                "account": candidate_accounts[sym_up],
                "symbol": sym_up,
                "action": "BUY",
                "entry_price": round(anchor, 4),
                "thesis": "",
                "horizon_days": 20,
                "benchmark_symbol": resolve_market_benchmark(sym_up)[0],
                "shadow_trigger": "CRITIC",
                "shadow_reason": "Critic终审未入选selected_actions(影子对照,不影响实盘)",
                "regime": str(regime or "").strip().upper()[:16],
            }
        )
    return rows


def _merge_shadow_rows_cap(
    guardrail_rows: list[dict[str, Any]],
    critic_rows: list[dict[str, Any]],
    cap: int = _SHADOW_MAX_ROWS_PER_PICK,
) -> list[dict[str, Any]]:
    """Merge the two shadow families by trigger priority and cap per-/pick sample size."""
    trigger_rank = {t: i for i, t in enumerate(_SHADOW_TRIGGER_PRIORITY)}
    for row in guardrail_rows + critic_rows:
        row.setdefault("shadow_trigger", "CRITIC")
    merged = sorted(
        guardrail_rows + critic_rows,
        key=lambda r: (trigger_rank.get(str(r.get("shadow_trigger")), 99), str(r.get("symbol", ""))),
    )
    seen: set[tuple[str, str]] = set()
    capped: list[dict[str, Any]] = []
    for row in merged:
        key = (str(row.get("account", "")).strip().upper(), str(row.get("symbol", "")).strip().upper())
        if key in seen:
            continue
        seen.add(key)
        capped.append(row)
        if len(capped) >= max(1, int(cap)):
            break
    return capped


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
    deployment_gate: dict[str, Any] | None = None,
    stage2_tech: dict[str, TechnicalSnapshot] | None = None,
    bench_histories: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Backfill native entry_price, apply deterministic risk override guardrail, and render.

    v5.5.9 M6 third return element: shadow-counterfactual rows for R:R-floor hard
    rejections (kept strictly out of the live pick ledger by the caller).
    """
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

    hydrated_decision = decision.model_copy(
        update={
            "selected_actions": [
                a.model_copy(update={"risk_note": ""}) for a in hydrated_actions
            ],
            # v5.5.7: Python owns these fields — never trust the model's leftovers.
            "rejected_or_frozen": [],
        }
    )
    # v5.5.1/v5.5.9 M2 排雷: new candidates have no held tech snapshot, and guardrail RS checks
    # must share the exact same RS20/RS60 computed during screening so screener and /pick never split.
    rs_lookup: dict[str, tuple[float | None, float | None]] = {}
    for cand in (*isa_cands, *invest_cands):
        c_sym = str(getattr(cand, "symbol", "") or "").strip().upper()
        if not c_sym:
            continue
        c_rs20 = getattr(cand, "rs_20d_pct", None)
        c_metrics = getattr(cand, "filter_metrics", None) or {}
        c_rs60 = c_metrics.get("rs_60d_pct") if isinstance(c_metrics, dict) else None
        if c_rs20 is not None or c_rs60 is not None:
            rs_lookup[c_sym] = (
                float(c_rs20) if c_rs20 is not None else None,
                float(c_rs60) if c_rs60 is not None else None,
            )
    if stage2_tech:
        for s_sym, s_snap in stage2_tech.items():
            if s_snap is None:
                continue
            s_up = str(s_sym or "").strip().upper()
            if s_up and (s_snap.rs_20d_pct is not None or s_snap.rs_60d_pct is not None):
                prev20, prev60 = rs_lookup.get(s_up, (None, None))
                rs_lookup[s_up] = (
                    s_snap.rs_20d_pct if s_snap.rs_20d_pct is not None else prev20,
                    s_snap.rs_60d_pct if s_snap.rs_60d_pct is not None else prev60,
                )

    bench_lookup: dict[str, pd.DataFrame] = {
        k: df
        for k, df in (bench_histories or {}).items()
        if df is not None and not df.empty
    }
    missing_benches: list[str] = []
    for item in decision.selected_actions:
        sym_clean = item.symbol.strip()
        if not sym_clean or sym_clean.upper() == "CASH":
            continue
        cands_b = resolve_market_benchmark(sym_clean)
        if not any(
            (_lookup_ci(bench_lookup, b_sym) is not None and not _lookup_ci(bench_lookup, b_sym).empty)
            or (_lookup_ci(all_histories, b_sym) is not None and not _lookup_ci(all_histories, b_sym).empty)
            for b_sym in cands_b
        ):
            if cands_b:
                missing_benches.append(cands_b[0])
    if missing_benches and callable(getattr(market_data, "get_histories", None)):
        unique_benches = list(dict.fromkeys(missing_benches))
        try:
            fetched_benches = await market_data.get_histories(unique_benches, period="1y")
            if isinstance(fetched_benches, dict):
                for b_k, b_df in fetched_benches.items():
                    if b_df is not None and not b_df.empty:
                        bench_lookup[b_k] = b_df
        except Exception as exc:
            logger.debug("Post-screen benchmark history lookup skipped for %s: %s", unique_benches, exc)

    def _bench_hist_for(sym_k: str) -> pd.DataFrame | None:
        for b_sym in resolve_market_benchmark(sym_k):
            b_df = _lookup_ci(bench_lookup, b_sym)
            if b_df is not None and not b_df.empty:
                return b_df
            b_df = _lookup_ci(all_histories, b_sym)
            if b_df is not None and not b_df.empty:
                return b_df
        return None

    all_tech_map: dict[str, TechnicalSnapshot] = {}
    if stage2_tech:
        for s_sym, s_snap in stage2_tech.items():
            if s_snap is not None:
                all_tech_map[s_sym] = s_snap
    for sym_k, hist_df in all_histories.items():
        snap_k = compute_technical_snapshot(
            sym_k,
            hist_df,
            bundle=_lookup_ci(all_bundles, sym_k),
            benchmark_hist=_bench_hist_for(sym_k),
        )
        if snap_k is not None:
            fb20, fb60 = rs_lookup.get(sym_k.strip().upper(), (None, None))
            upd: dict[str, Any] = {}
            if snap_k.rs_20d_pct is None and fb20 is not None:
                upd["rs_20d_pct"] = fb20
            if snap_k.rs_60d_pct is None and fb60 is not None:
                upd["rs_60d_pct"] = fb60
            if upd:
                snap_k = snap_k.model_copy(update=upd)
            all_tech_map[sym_k] = snap_k
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
        exempt_symbols={
            str(s).strip().upper()
            for s in ((deployment_gate or {}).get("etf_set") or [])
            if str(s).strip()
        }
        | {str(cfg.strategy.benchmark_symbol).strip().upper()},
    )
    # v5.4.0: gate-linked sizing — after a loss-streak cooldown the recovery pilot runs at
    # half risk (0.5R ≈ £42.5) so the suggested share count scales automatically.
    # v5.5.0: the scale is passed per-ticket and applied to Invest only — it must never
    # halve an ISA risk budget (ISA also gets the 0.5% NAV floor inside the sizer).
    gate_mode = str(getattr(cfg.strategy, "deployment_gate_mode", "advise") or "advise").strip().lower()
    gate_ctx = deployment_gate if (isinstance(deployment_gate, dict) and gate_mode == "advise") else None
    risk_budget_scale = 1.0
    if gate_ctx is not None:
        inv_g = gate_ctx.get("invest") if isinstance(gate_ctx.get("invest"), dict) else {}
        risk_budget_scale = float(inv_g.get("risk_budget_scale") or 1.0)
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
        deployment_gate=gate_ctx,
        risk_budget_scale=risk_budget_scale,
        tech_map=all_tech_map,
    )
    resolved_decision = (
        override_res.decision
        if isinstance(override_res.decision, PickDecision)
        else hydrated_decision
    )
    # v5.4.0: deterministic gate channel constraints (cross-account dedupe + ROTATE funding).
    final_decision = resolved_decision
    if gate_ctx is not None:
        final_decision, _gate_notes = _apply_gate_channel_constraints(
            resolved_decision, portfolio, gate_ctx, bundles=all_bundles
        )
    final_decision = _reconcile_pick_headline(
        final_decision,
        original_actions=hydrated_decision.selected_actions,
    )
    # v5.5.7: Python recomputes the post-execution cash line after every guardrail, so a Critic's
    # pre-interception arithmetic (and double-deducted pending orders) can never reach Telegram.
    final_decision = final_decision.model_copy(
        update={
            "consistency_and_cash_note": _build_deterministic_cash_note(
                final_decision,
                portfolio,
                override_res.sizing_map,
                gate_ctx,
            )
        }
    )
    # v5.5.9 M5: deterministic ETF-to-equity funding pre-instruction (清宽基换股).
    # Computed AFTER every guard so only fully vetted tickets can draw on the reservoir;
    # free cash >= ticket never touches it and sub-floor gaps defer to the plain FX path.
    reservoir_funding_note = build_etf_to_equity_funding_note(
        final_decision.selected_actions,
        portfolio,
        strategy=cfg.strategy,
        sizing_map=override_res.sizing_map,
    )
    final_text = render_pick_report(
        final_decision,
        max_chars=1000,
        risk_override=override_res,
        funding_note=reservoir_funding_note,
    )
    core_etf_set = {
        str(s).strip().upper()
        for s in ((deployment_gate or {}).get("etf_set") or [])
        if str(s).strip()
    }

    def _is_core_etf_symbol(symbol: str) -> bool:
        sym_up = str(symbol or "").strip().upper()
        return bool(
            sym_up
            and (
                sym_up in core_etf_set
                or sym_up.split(".")[0] in core_etf_set
                or normalize_issuer_root(sym_up)
                in {normalize_issuer_root(s) for s in core_etf_set}
            )
        )

    parsed_picks = []
    regime_label = str((gate_ctx or {}).get("regime") or "").strip().upper()
    for item in final_decision.selected_actions:
        is_core_dca = _is_core_etf_symbol(item.symbol) and str(item.action or "").strip().upper() in {
            "BUY",
            "ADD",
        }
        # v5.5.7: the T+5/T+20 settlement anchor is always the live market price — an LLM's
        # optimistic entry/limit can never pollute attribution.
        anchor_bundle = _lookup_ci(all_bundles, item.symbol)
        anchor_price = (
            float(anchor_bundle.current_price)
            if (
                anchor_bundle is not None
                and anchor_bundle.current_price is not None
                and float(anchor_bundle.current_price) > 0
            )
            else (item.entry_price if item.entry_price > 0 else None)
        )
        parsed_picks.append(
            {
                "account": item.account,
                "symbol": item.symbol,
                "action": item.action,
                "entry_price": anchor_price,
                "thesis": item.thesis,
                "horizon_days": 20 if is_core_dca else cfg.strategy.settlement_holding_days,
                "benchmark_symbol": resolve_market_benchmark(item.symbol)[0],
                "source": "core_dca" if is_core_dca else "pick",
                "fill_status": "ASSUMED" if is_core_dca else "",
                "regime": regime_label[:16],
            }
        )
    # v5.5.9 M6: shadow counterfactual rows for R:R-floor hard rejections. These are
    # advisory-lane telemetry only — the caller logs them through the isolated
    # shadow lane, never through log_recommendations.
    kept_symbols = {
        str(item.symbol or "").strip().upper()
        for item in final_decision.selected_actions
        if str(item.symbol or "").strip()
    }
    guardrail_shadow_rows = _collect_guardrail_shadow_rows(
        hydrated_actions,
        override_res,
        kept_symbols,
        all_bundles,
        regime_label,
    )
    if llm is not None and not final_text.startswith("⚠️"):
        selected_syms = {
            item.symbol.strip().upper()
            for item in final_decision.selected_actions
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
                candidate_bundles=all_bundles,
            ),
            rendered_text=final_text,
            strategy=cfg.strategy,
            ops_notifier=ops_notifier,
            headline=final_decision.headline,
            selected_symbols=selected_syms,
            known_symbols=rejected_syms,
        )
    return final_text, parsed_picks, guardrail_shadow_rows


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

    # Today's /report is the single source of truth: never substitute yesterday's red lines.
    baseline_text, baseline_conclusion, baseline_stale = await db.get_morning_context_with_status()
    if baseline_stale or not baseline_text.strip():
        return (
            "⛔ 今日 /report 基线尚未就绪（当前仅有上一交易日缓存），"
            "已拒绝用旧红线选股。请先运行 /report 后再执行 /pick。"
        )
    guardrails = await db.get_daily_guardrails()
    context_pack = await db.get_context_pack(allow_fallback=False)
    if not isinstance(context_pack, dict):
        return "⛔ 今日 /report ContextPack 缺失或不可读，/pick 暂停以避免无门控选股。请重跑 /report。"

    gate_mode = str(getattr(cfg.strategy, "deployment_gate_mode", "off") or "off").strip().lower()
    gate_ctx = context_pack.get("deployment_gate") if isinstance(context_pack, dict) else None
    gate_ctx = gate_ctx if isinstance(gate_ctx, dict) else None
    if gate_ctx is not None and gate_mode == "advise":
        gate_pm = gate_ctx.get("pm") if isinstance(gate_ctx.get("pm"), dict) else {}
        if gate_pm.get("allow") is False:
            pm_reason = str(gate_pm.get("reason") or "PM 主动收紧否决")
            return f"⛔ 今日组合门控已由 PM 收紧，暂停新开仓：{pm_reason}（如需复核请重跑 /report）"

    # Step 1: Fetch held bundles/histories + macro + recent lessons first
    (
        (held_histories, held_bundles),
        macro,
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
        db.get_recent_lessons(limit=5),
        db.get_active_preference_notes(),
    )
    morning_report_full, morning_conclusion = baseline_text, baseline_conclusion

    # Feed the latest GBP/USD mid so subsequent Invest wallet splits can be derived locally.
    # v5.5.7: mirror /report — re-derive the in-memory portfolio FX BEFORE any cash/wallet text
    # is built, otherwise the precise cash base still prints "GBP/USD 暂缺" on a cold start.
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

    # Step 2: Load curated 150-stock active pool (if auditioned) & reuse held_histories in quant screener
    active_pool_rows: list[dict[str, Any]] = []
    if hasattr(db, "get_active_universe_pool"):
        try:
            active_pool_rows = await db.get_active_universe_pool()
        except Exception as exc:
            logger.debug("Failed loading active_universe_pool; falling back to config.yaml: %s", exc)
            active_pool_rows = []

    # v5.5.7: channel short-circuits + cooldown pre-filter. A closed ISA equity channel or a
    # hard-mode 15-day re-buy cooldown must never burn Stage-2 IO / DeepSeek tokens.
    gate_isa = gate_ctx.get("isa") if isinstance(gate_ctx, dict) and isinstance(gate_ctx.get("isa"), dict) else {}
    has_focus = bool(focus_query.strip())
    enable_isa_equities = has_focus or (gate_isa.get("equity_open") is not False)
    enable_isa_etfs = has_focus or (gate_isa.get("etf_open") is not False)
    cooldown_excluded: set[str] = set()
    core_etf_exempt = {
        str(s).strip().upper()
        for s in ((gate_ctx or {}).get("etf_set") or [])
        if str(s).strip()
    } | {str(cfg.strategy.benchmark_symbol).strip().upper()}
    if str(getattr(cfg.strategy, "trade_cadence_guard_mode", "advisory")).lower() == "hard":
        try:
            early_behavior = compute_trade_behavior_stats(
                portfolio,
                recent_days=7,
                history_days=cfg.strategy.order_history_days,
                cooldown_days=cfg.strategy.rebuy_cooldown_days,
                exempt_symbols=core_etf_exempt,
            )
            cooldown_excluded = {
                str(row.get("ticker") or "").strip().upper()
                for row in (early_behavior.get("recently_closed") or [])
                if row.get("in_cooldown") and str(row.get("ticker") or "").strip()
            }
        except Exception as exc:
            logger.debug("Cooldown pre-filter skipped in /pick: %s", exc)

    screener_kwargs: dict[str, Any] = {
        "portfolio": portfolio,
        "market_data": market_data,
        "universe": cfg.universe,
        "watchlist": cfg.watchlist,
        "strategy": cfg.strategy,
        "existing_histories": held_histories,
        "top_n_per_track": cfg.strategy.pick_wide_funnel_top_n,
        "enable_isa_equities": enable_isa_equities,
        "enable_isa_etfs": enable_isa_etfs,
        "excluded_symbols": cooldown_excluded or None,
    }
    if active_pool_rows:
        screener_kwargs["active_pool_rows"] = active_pool_rows
    # v5.5.1: the ISA value / Invest right-side scoring needs the same macro + regime as the gate.
    if isinstance(macro, MacroSnapshot):
        screener_kwargs["macro"] = macro
    if gate_ctx is not None:
        screener_kwargs["regime_tag"] = str(gate_ctx.get("regime") or "")

    stage2_bundles: dict[str, TickerMarketBundle] = {}
    stage2_intel: dict[str, dict[str, Any]] = {}
    stage2_tech: dict[str, TechnicalSnapshot] = {}
    bench_histories: dict[str, pd.DataFrame] = {}
    screener_kwargs["out_bundles"] = stage2_bundles
    screener_kwargs["out_intel"] = stage2_intel
    screener_kwargs["out_tech"] = stage2_tech
    screener_kwargs["out_bench_histories"] = bench_histories
    isa_cands, invest_cands = await run_quant_screener(**screener_kwargs)
    # Wide funnel -> DeepSeek ID filter: only tickers are selected; the raw candidate objects stay untouched.
    isa_cands, invest_cands = await _prefilter_candidates_with_ds(cfg, llm, isa_cands, invest_cands)

    pack_symbols = (
        context_pack.get("symbols", {})
        if isinstance(context_pack, dict) and isinstance(context_pack.get("symbols"), dict)
        else {}
    )
    # v5.4.0: the deterministic deployment gate was parsed above before the screener.
    gate_block = ""
    if gate_ctx is not None and gate_mode == "advise":
        gate_block = _format_gate_constraint_block(gate_ctx, portfolio, strategy=cfg.strategy)
    if not recent_lessons and isinstance(context_pack, dict):
        pack_recent = context_pack.get("recent_lessons")
        if isinstance(pack_recent, list):
            recent_lessons = pack_recent

    def _held_bench_for(sym_h: str) -> pd.DataFrame | None:
        for b_sym in resolve_market_benchmark(sym_h):
            b_df = _lookup_ci(bench_histories, b_sym)
            if b_df is not None and not b_df.empty:
                return b_df
            b_df = _lookup_ci(held_histories, b_sym)
            if b_df is not None and not b_df.empty:
                return b_df
        return None

    tech_map = {}
    for sym in tracked_syms:
        snap = compute_technical_snapshot(
            sym,
            held_histories.get(sym),
            position_created_at=created_at_map.get(sym),
            bundle=held_bundles.get(sym),
            benchmark_hist=_held_bench_for(sym),
        )
        if snap is not None:
            tech_map[sym] = snap

    evals = evaluate_portfolio(
        portfolio,
        tech_map,
        held_bundles,
        cfg.strategy,
        histories=({**bench_histories, **held_histories} if bench_histories else held_histories),
        regime=str((gate_ctx or {}).get("regime") or "NEUTRAL"),
        macro=macro,
    )
    evals_text = format_evaluations_text(
        evals,
        collapse_isa_add=bool(
            isinstance(gate_ctx, dict)
            and isinstance(gate_ctx.get("isa"), dict)
            and gate_ctx["isa"].get("equity_open") is False
        ),
    )
    exposure_text = compute_exposure_summary(portfolio, held_bundles, histories=held_histories)
    trade_ctx = build_trade_context(
        portfolio,
        max_recent_lines=6,
        max_summary_lines=6,
        bundles=held_bundles,
        tech_map=tech_map,
        histories=held_histories,
        cooldown_days=cfg.strategy.rebuy_cooldown_days,
        exempt_symbols=core_etf_exempt,
    )

    morning_audits_text, morning_symbol_tags, report_top_actions = _extract_morning_audit_context(
        context_pack,
        morning_report_full,
        portfolio=portfolio,
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
    preference_block = (
        format_preference_notes_block(pref_notes)
        if bool(getattr(cfg.strategy, "user_preference_memory_enabled", True))
        else ""
    )
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

    grounding_coro = (
        run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.search_model,
            scope="pick",
            symbols=cand_syms,
            priority_symbols=cand_syms[:6],
            focus_query=focus_query,
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
            db=db,
            persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
        if cfg.models.enable_forced_search_grounding
        else _empty_grounding()
    )
    grounding_brief, symbol_lessons = await asyncio.gather(
        grounding_coro,
        _fetch_candidate_symbol_lessons(db, lesson_syms, pack_symbols),
    )
    merged_cand_bundles = {**held_bundles, **stage2_bundles}
    cand_intel_map = stage2_intel
    if grounding_brief:
        # v5.5.7: one explicit coverage manifest so the Researcher never re-runs search_market_news.
        grounding_brief = (
            f"已联网核查清单：[{', '.join(cand_syms) or '无'}]；"
            "以下结果为最新核查结论，严禁重复调用 search_market_news。\n" + grounding_brief
        )
    cand_detail_lines: list[str] = []
    for sym in cand_syms:
        b = merged_cand_bundles.get(sym)
        parts: list[str] = []
        intel_str = _format_us_deep_intel(cand_intel_map.get(sym) or {})
        if intel_str:
            parts.append(intel_str)
        if b is not None and b.news:
            name_tokens = {
                sym.upper(),
                sym.upper().split(".")[0],
                str(b.short_name or "").strip().upper(),
            }
            relevant_news = [
                n
                for n in b.news
                if n.get("title")
                and any(
                    tok and tok in str(n.get("title", "")).upper()
                    for tok in name_tokens
                )
            ]
            news_str = " / ".join(f"[{n['date']}] {n['title'][:70]}" for n in relevant_news[:2])
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

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
        db=db,
        search_cache_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        include_macro_tool=False,
    )

    # ---------- Stage 1: Incremental Opportunity Researcher ----------
    researcher_sys = build_agent_system_instruction(
        "你是增量选股投研员（Researcher Agent）。存量持仓体检与老挂单审计已由 `/report` 完成，你专注于结合今日风控红线与早报裁决，"
        "从全市场（美/英/欧个股与 LSE 英镑 ETF）中挖掘最值得新买入的配置或高确定性卖弱换强机会。\n"
        "对今日 `/report` 已裁决【CANCEL】或【MODIFY】的活动挂单，以及已裁决【TRIM】/【STOP】的持仓，默认必须严格遵循不得推翻；"
        "仅当你通过工具实查到早报生成之后的突发新公告或股价已显著回落至早报认可区间时，才可提出修正并写明最新依据。\n"
        f"请参考历史 T+{cfg.strategy.settlement_holding_days} 交易日选股复盘教训，避免对近期连续失效的板块或追高形态重复推荐。\n"
        "输出硬约束：总长 ≤1,200 字；共不超过 3 条核心提案（ISA 新股或 LSE ETF / Invest 短线催化 / 必要时的卖弱换强）。\n"
        "分轨授权 0 提案：某一轨若确实无达标票，直接写『ISA 轨 0 提案』或『Invest 轨 0 提案』即可，"
        "严禁长篇列举被淘汰代码或复述早报已维持的持仓项；另一轨的有效提案不受影响。\n"
        "每条提案必须按 5 要素落盘：①标的与账户；②介入区间；③止损位/证伪条件；④目标位与观察周期；"
        "⑤1 句校准式 Pre-Mortem（格式：最大单点失败路径是 X，但当前 Y 支撑/催化提供对冲；仅当 Z 触发时止损认输）。\n"
        "双轨终审标准："
        "ISA 价值轨核查护城河质量 + 估值安全边际（常规≥12%、危机日≥20%），允许 `stop_price=0`，"
        "但 `stop_condition` 必须写明基本面/护城河证伪条件，`target_price` 填保守合理估值锚；"
        "Invest 右侧轨核查当日 Regime 动态 R:R（以【今日组合门控约束】印出的 Python 确定性门槛为准）"
        "与止损幅≥max(1.5×ATR, 4%)，必填数值 `stop_price>0` 与 `target_price>0`；"
        "仅非 RISK_OFF 下确认强领涨股（MA200上方+BULL/PULLBACK+周线不空+RS20>0且RS60>0）可按双锚上限"
        "`max(卖方均值×1.15, 卖方高位×0.90, 入场+4×ATR)` 设目标，非领涨/弱势股及 RISK_OFF 仍死锁卖方均值 ±15%；"
        "若上文初筛候选与实时 Google Search 摘要已提供充分事实，禁止无故重复调用工具；仅当用户指定新方向不在预设列表或核心数据缺失时，才调用 `inspect_ticker`、`check_portfolio_correlation` 或 `search_market_news`。",
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
        f"{gate_block}"
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

    # v5.5.7: 0-token one-hand fact card so the Critic verifies real numbers instead of trusting
    # the Researcher's second-hand prose (whitelist intersection only — never a loose regex).
    fact_card_text, proposal_syms = _build_proposal_fact_card(
        research_output,
        isa_cands=isa_cands,
        invest_cands=invest_cands,
        merged_bundles=merged_cand_bundles,
        tech_map=tech_map,
        stage2_tech=stage2_tech,
        toolkit=toolkit,
        portfolio=portfolio,
        cfg=cfg,
        macro=macro,
        regime=str((gate_ctx or {}).get("regime") or "NEUTRAL"),
    )
    critic_audits_text = _filter_audit_text_for_critic(morning_audits_text, proposal_syms)

    # ---------- Stage 2: Risk & Bear Critic -> Structured PickDecision ----------
    critic_sys = build_agent_system_instruction(
        "你是独立风控官（Risk & Bear Critic）。请按下列【六重护栏（精炼版）】与今日 `/report` 红线审核投研员提案，"
        "直接输出最终结构化 JSON Decision（渲染后将限制在 1000 字内，结论置顶、废话为零）。\n\n"
        "【六重护栏（精炼版）】：\n"
        "1. 盈亏比与摩擦：Invest 必须给出标的本地货币的数值 `stop_price>0` 与 `target_price>0`；"
        "Python 会按含摩擦止损分母独立复算净 R:R，并以【今日组合门控约束】印出的当日确定性门槛与 ATR 噪声地板裁决；"
        "非 RISK_OFF 下的确认强领涨股（MA200上方+BULL/PULLBACK+周线不空+RS20>0且RS60>0）按双锚上限"
        "`max(卖方均值×1.15, 卖方高位×0.90, 入场+4×ATR)` 核验，非领涨/弱势/RISK_OFF 标的仍严锁卖方均值 ±15%；"
        "ISA 长线价值个股可填 `stop_price=0`，但必须在 `stop_condition` 写明基本面/护城河证伪条件，改核「质量门 + 安全边际（常规≥12% / 危机≥20%）」。\n"
        "2. 独立核验：必须对照【提案标的一手事实卡（Python 独立真源）】交叉核验投研员引用的现价/ATR/RSI/均线/财报/除息/卖方均值；"
        "发现事实冲突即降级或否决，事实卡未覆盖的标的不允许凭空补充数字。\n"
        "3. 穿透科技暴露与相关性：穿透后科技因子明显集中、或与持仓 60 日相关性达标、或违背今日红线名单的标的必须驳回。\n"
        "4. 财报/除息/过热排雷：5 日内财报、RSI 过热追高全部否决或降级；除息窗口新建仓须在 thesis 写明「抢息」或「等除息后再买」，"
        "Invest 应税轨按股息税后净额计入 R:R（ISA 股息免税可忽略）。\n"
        "5. 结构与槽位：ISA 个股通道关闭或卫星层饱和时只准按 Python 政策票据做宽基；"
        "Invest 新开仓须遵守在险槽位、承诺热力、单日额度与 ROTATE 置换资金源纪律。\n"
        "6. 一致性、精确现金与零内部对话外泄：严格继承 /report 红线与相关裁决；现金只用提供的精确数字，禁止口算；"
        "`headline` 严禁提及任何未入选、淘汰或被否决的标的/方向；`rejected_or_frozen` 一律返回空数组 `[]`；"
        "`consistency_and_cash_note` 只写一句 ≤45 字风控红线遵循声明（严禁自行计算余现，执行后余现由 Python 精确接管）；"
        "若当前为周末/休市且有推荐标的，请在 headline 标明「下周一开盘/限价挂单参考」。\n"
        f"若无个股达到双轨达标门槛，且 ISA 现金占比 >{cfg.strategy.isa_cash_deploy_trigger_pct:.0f}%，"
        "应优先按【今日组合门控约束】的 Python 核心政策票据做分批定投：写明票据代码、具体价位、"
        f"单笔金额（ISA ETF ≥£{cfg.strategy.min_isa_etf_order_gbp:,.0f} 且不超票据上限）与节奏，严禁自行换代码；"
        "核心宽基 stop_price/target_price 填 `null`；"
        "Invest 若无达标右侧个股（动态 R:R 未过门槛）直接令 selected_actions 为空数组 []；"
        "若确实连宽基部署空间都没有，才让 `selected_actions` 为空数组 `[]`，并在 headline 说明"
        "「本轮无达标机会（ISA 安全边际 / Invest 动态 R:R 未达标），现金待命 + 下一次部署触发条件」，"
        f"严禁默认写「现金拿住吃 {cfg.strategy.cash_interest_rate_pct:.1f}% 利息」。",
        cfg.strategy,
        include_order_audit_rule=False,
        compact_for_pick_critic=True,
    )
    critic_prompt = (
        f"{gate_block}"
        f"【时段状态】：{market_phase_note}\n"
        f"【今日 /report 基线】：{morning_conclusion or '（暂无）'}\n"
        + (
            f"【今日 /report 持仓与挂单裁决（默认遵循，仅列相关项）】：\n{critic_audits_text}\n"
            if critic_audits_text
            else ""
        )
        + f"【今日红线】：\n{veto_block}\n\n"
        + (f"{preference_block}\n\n" if preference_block else "")
        + f"【当前活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"【交易频率与冷却（Python 计算）】：{trade_ctx['trade_behavior_text']}\n\n"
        f"【历史 T+{cfg.strategy.settlement_holding_days} 选股复盘教训】：\n{lessons_text}\n\n"
        f"【精确现金底座】：{exact_cash_summary}\n"
        f"【行业暴露】：\n{exposure_text}\n\n"
        + (f"{fact_card_text}\n\n" if fact_card_text else "")
        + f"【投研员提案（待审）】：\n{research_output}"
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
        final_text, parsed_picks, guardrail_shadow_rows = await _post_screen_validate_decision(
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
            deployment_gate=gate_ctx,
            stage2_tech=stage2_tech,
            bench_histories=bench_histories,
        )
    else:
        logger.warning("Pick Critic structured parse failed; refusing to emit unguarded raw text.")
        return "⚠️ /pick 风控终审结构化解析失败，已拦截未经护栏核验的原始输出，请稍后重试 /pick。"

    if not final_text.startswith("⚠️"):
        await db.save_pick_radar(focus_query, research_output, final_text, now_uk)
        if parsed_picks:
            tactical_picks = [r for r in parsed_picks if r.get("source") != "core_dca"]
            core_picks = [r for r in parsed_picks if r.get("source") == "core_dca"]
            if tactical_picks:
                await db.log_recommendations(
                    tactical_picks,
                    default_horizon_days=cfg.strategy.settlement_holding_days,
                    source="pick",
                    dedupe_window_days=5,
                    dedupe_cross_source=True,
                )
            if core_picks:
                await db.log_recommendations(
                    core_picks,
                    default_horizon_days=20,
                    source="core_dca",
                    dedupe_same_day=True,
                )
        # v5.5.9 M6: shadow counterfactual lane — /pick Top candidates killed by the
        # R:R floor (deterministic guardrail) or dropped by the Critic, logged to a
        # status-isolated lane for the T+20 avoided-loser / missed-winner study.
        # Failure here must never affect the live pick flow.
        try:
            kept_syms = {
                str(item.symbol or "").strip().upper()
                for item in decision.selected_actions
                if str(item.symbol or "").strip()
            }
            held_syms_shadow = {s.strip().upper() for s in portfolio.unique_yahoo_symbols()}
            critic_shadow_rows = _collect_critic_shadow_rows(
                proposal_syms,
                kept_syms,
                held_syms_shadow,
                isa_cands,
                invest_cands,
                merged_cand_bundles,
                str((gate_ctx or {}).get("regime") or ""),
            )
            shadow_rows = _merge_shadow_rows_cap(guardrail_shadow_rows, critic_shadow_rows)
            if shadow_rows and hasattr(db, "log_shadow_rejections"):
                await db.log_shadow_rejections(shadow_rows)
        except Exception as exc:
            logger.warning("Shadow rejection logging skipped: %s", exc)

    return final_text
