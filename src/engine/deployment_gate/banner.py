"""Deployment gate banner formatting, main orchestrator, and 2-year offline replay."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.deployment_gate.common import (
    LONDON_TZ,
    _BANNER_MAX_CHARS,
    _GATE_VERSION,
    _close_series,
    _round2,
)
from src.engine.deployment_gate.core_policy import (
    CORE_UPPER_BAND_PCT,
    _build_etf_set,
    compute_core_allocation_plan,
)
from src.engine.deployment_gate.ledgers import (
    classify_pending_orders,
    compute_account_ledgers,
)
from src.engine.deployment_gate.regime import (
    _compute_raw_regime_metrics,
    evaluate_market_regime,
)
from src.engine.deployment_gate.triggers import scan_triggers_and_decide
from src.engine.portfolio_rules.dual_philosophy import resolve_invest_dynamic_rr_floor

# ---------------------------------------------------------------------------
# 6. <=110-char pure-text banner
# ---------------------------------------------------------------------------
def _fit_banner(line1: str, line2: str) -> str:
    banner = f"{line1}\n{line2}"
    if len(banner) <= _BANNER_MAX_CHARS:
        return banner
    budget = max(20, _BANNER_MAX_CHARS - len(line1) - 1)
    tail = line2
    while len(tail) > budget and ("，" in tail or ";" in tail or "；" in tail):
        cut = max(tail.rfind("，"), tail.rfind(";"), tail.rfind("；"))
        if cut <= 0:
            break
        tail = tail[:cut]
    if len(tail) > budget:
        tail = tail[: max(0, budget - 1)] + "…"
    return f"{line1}\n{tail}"


def _gpb_k(value: float) -> str:
    return f"£{value / 1000.0:.1f}k"


def _core_banner_label(isa: dict[str, Any]) -> str:
    plan = isa.get("core_allocation_plan") if isinstance(isa.get("core_allocation_plan"), dict) else {}
    tickets = plan.get("tickets") or []
    if tickets:
        t = tickets[0]
        return f"{t.get('symbol')}≤{_gpb_k(float(t.get('recommended_gbp') or 0.0))}"
    if float(isa.get("committed_core_etf_pct") or 0.0) >= CORE_UPPER_BAND_PCT:
        return "核心已达65%上限"
    return "ISF/VERG等低科技宽基" if isa.get("prefer_core_diversifier") else "VWRP"


def _reservoir_tag(invest: dict[str, Any]) -> str:
    """v5.5.9 M5 short advisory tag for the distributing-ETF beta reservoir.

    Appended LAST in the default WAIT banner line so the 110-char truncator always
    cuts it before anything else. The ISA RUN branch pins its own shorter line-1 tag
    instead (line 1 is never truncated). Never rendered in DEFEND/ROTATE/RUN
    branches, and RISK_OFF must never advertise new reservoir builds.
    """
    res = invest.get("reservoir_advice") if isinstance(invest.get("reservoir_advice"), dict) else {}
    if not isinstance(res, dict) or not res.get("eligible"):
        return ""
    sym = str(res.get("ticket_symbol") or "")
    ticket = float(res.get("ticket_gbp") or 0.0)
    if not sym or ticket <= 0:
        return ""
    return f"；Invest溢出£{ticket / 1000.0:.1f}k可停泊{sym}蓄水池(零Quota)"


def _reservoir_line1_tag(invest: dict[str, Any]) -> str:
    """Compact reservoir tag for the ISA RUN branch's untruncated first line."""
    res = invest.get("reservoir_advice") if isinstance(invest.get("reservoir_advice"), dict) else {}
    if not isinstance(res, dict) or not res.get("eligible"):
        return ""
    sym = str(res.get("ticket_symbol") or "")
    ticket = float(res.get("ticket_gbp") or 0.0)
    if not sym or ticket <= 0:
        return ""
    return f"·{sym.split('.')[0]}蓄水池£{ticket / 1000.0:.1f}k"


def format_deployment_gate_banner(gate_dict: dict[str, Any]) -> str:
    """Compose the <=110-char, <=2-line, pure-plain-text gate banner (no #, no **)."""
    isa = gate_dict.get("isa") or {}
    invest = gate_dict.get("invest") or {}
    regime = str(gate_dict.get("regime") or "UNKNOWN")
    global_status = str(gate_dict.get("global_status") or "UNKNOWN")
    fx_note = "汇率待校准" if gate_dict.get("fx_missing") else ""
    invest_status = str(invest.get("status") or "UNKNOWN")
    isa_status = str(isa.get("status") or "UNKNOWN")

    if invest_status == "DEFEND":
        heat = float(invest.get("heat_gbp_total") or 0.0)
        fin = "、".join(invest.get("financed_positions") or []) or "无"
        line1 = f"组合门控：DEFEND·渐进收缩｜市场转入退潮态｜Invest本金在险 £{heat:.0f}（{fin}保本保护中）"
        line2 = "→ ①撤未成交个股买单 ②清退无保护浮亏弱股，保留保本/逆势强股 ③ISA宽基不卖，深跌分档定投"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if invest_status == "RUN":
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        heat = float(invest.get("heat_gbp_total") or 0.0)
        fin = "、".join(invest.get("financed_positions") or []) or "无"
        avail = float(invest.get("available_to_trade_gbp") or 0.0)
        unit_risk = float(invest.get("unit_risk_gbp") or 85.0)
        committed = float(invest.get("committed_heat_pct") or 0.0)
        heat_cap = float(invest.get("heat_cap_pct") or 0.0)
        scale = float(invest.get("risk_budget_scale") or 1.0)
        if regime == "RECOVERY":
            line1 = f"组合门控：RUN·早春试探｜Invest槽位 {used}/{cap}（{scale:.1f}R轻兵，在险£{heat:.0f}）｜可用{_gpb_k(avail)}"
            line2 = (
                f"→ 单笔风险£{unit_risk:.0f}·热力{committed:.1f}%/{heat_cap:.1f}%；"
                "RECOVERY仅放行最强RS领涨股，可运行 /pick"
            )
        else:
            line1 = f"组合门控：RUN·右侧猎物｜Invest在险槽位 {used}/{cap}（{fin}已锁保本，本金在险£{heat:.0f}）｜可用{_gpb_k(avail)}"
            line2 = (
                f"→ 单笔风险£{unit_risk:.0f}·热力{committed:.1f}%/{heat_cap:.1f}%；"
                "已解锁右侧槽位（单日限最优选1笔），可运行 /pick 扫描突破龙头"
            )
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if invest_status == "ROTATE":
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        funding = str(invest.get("funding_from") or "")
        line1 = f"组合门控：ROTATE·等风险置换｜{regime}在险槽位 {used}/{cap} 已满｜观察池触发右侧强信号"
        line2 = f"→ 严禁动用闲置现金净增风险；若运行 /pick，须以平掉未脱险弱仓 {funding} 为前提等额置换"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if global_status == "UNKNOWN" or (isa_status == "UNKNOWN" and invest_status == "UNKNOWN"):
        return _fit_banner("组合门控：UNKNOWN｜账户快照或基准数据降级，保守不催买不乱喊减仓", "")

    if isa.get("isa_satellite_saturated"):
        d_pct = float(isa.get("deployable_pct") or 0.0)
        core_pct = float(isa.get("committed_core_etf_pct") or 0.0)
        count = int(isa.get("committed_satellite_count") or 0)
        sat_pct = float(isa.get("committed_satellite_pct") or 0.0)
        pending_eq = "、".join((isa.get("pending_new_equity_buys") or [])[:2]) or ""
        line1 = (
            f"组合门控：ISA 仅配宽基 / Invest {invest_status}｜ISA可用 {d_pct:.1f}%"
            f"（宽基{core_pct:.0f}%，个股含挂单{count}只/{sat_pct:.0f}%饱和）"
        )
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        unfin = "、".join((invest.get("unfinanced_positions") or [])[:2]) or "无"
        fin = "、".join((invest.get("financed_positions") or [])[:2]) or "无"
        parts = []
        if pending_eq:
            parts.append(f"审撤{pending_eq}买单")
        avail = float(isa.get("available_to_trade_gbp") or 0.0)
        core_label = _core_banner_label(isa)
        if avail >= 1000:
            parts.append(f"{_gpb_k(avail)}→{core_label}")
        parts.append(f"Invest槽位{used}/{cap}({unfin}未脱险,{fin}已锁保本)静待解锁")
        line2 = "→ " + "，".join(parts)
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if isa_status == "RUN":
        d_pct = float(isa.get("deployable_pct") or 0.0)
        core_pct = float(isa.get("committed_core_etf_pct") or 0.0)
        mode = str(isa.get("mode") or "dca")
        line1 = (
            f"组合门控：ISA RUN·{mode} / Invest {invest_status}"
            f"{_reservoir_line1_tag(invest)}｜ISA可用 {d_pct:.1f}%（宽基{core_pct:.0f}%）"
        )
        core_label = _core_banner_label(isa)
        line2 = f"→ 闲置现金分批归拢 {core_label}；Invest 无右侧猎物不催买（现金吃息仅作等待保底）"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    line1 = f"组合门控：ISA {isa_status} / Invest {invest_status}｜市场 {regime}"
    line2 = f"→ {str(isa.get('reason') or '')}" + _reservoir_tag(invest)
    return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))


# ---------------------------------------------------------------------------
# Orchestrator used by /report (and /pick via context pack)
# ---------------------------------------------------------------------------
def evaluate_deployment_gate(
    *,
    portfolio: PortfolioData,
    strategy: StrategySection,
    bundles: dict[str, TickerMarketBundle] | None = None,
    histories: dict[str, pd.DataFrame] | None = None,
    macro: MacroSnapshot | None = None,
    tech_map: dict[str, TechnicalSnapshot] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    universe_etf_symbols: list[str] | None = None,
    prev_gate_states: list[dict[str, Any]] | None = None,
    gbp_usd: float | None = None,
    now_uk: datetime | None = None,
    pool_snaps: dict[str, Any] | None = None,
    cached_breadth_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Full deterministic gate evaluation — 0 LLM calls, 0 network, 0 DB writes."""
    bundles = bundles or {}
    histories = histories or {}
    etf_set = _build_etf_set(strategy, universe_etf_symbols, bundles)
    fx = gbp_usd or getattr(portfolio, "gbp_usd_rate", None) or (macro.gbp_usd if macro is not None else None)

    eval_dt = now_uk or datetime.now(LONDON_TZ)
    today_date_str = eval_dt.strftime("%Y-%m-%d")

    prev = None
    for state in prev_gate_states or []:
        if not isinstance(state, dict) or not state.get("regime"):
            continue
        state_date = str(state.get("cycle_date") or str(state.get("evaluated_at_uk") or "")[:10]).strip()
        if state_date and state_date == today_date_str:
            continue  # same-day /report re-run must not inflate multi-day regime streaks
        prev = state
        break

    regime_info = evaluate_market_regime(
        histories,
        macro,
        strategy,
        prev,
        pool_snaps=pool_snaps,
        cached_breadth_histories=cached_breadth_histories,
    )
    classified = classify_pending_orders(portfolio, strategy, etf_set, bundles=bundles)
    ledgers = compute_account_ledgers(
        portfolio,
        classified,
        bundles,
        strategy,
        etf_set,
        tech_map=tech_map,
        now_uk=eval_dt,
        regime=regime_info["regime"],
    )
    decision = scan_triggers_and_decide(
        portfolio,
        bundles,
        strategy,
        ledgers,
        regime_info,
        histories,
        classified,
        candidates=candidates,
        gbp_usd=fx,
        now_uk=eval_dt,
    )
    core_policy = decision["isa"].get("core_policy") or {}
    macro_vix = macro.vix if (macro is not None and macro.vix is not None) else None
    decision["isa"]["core_allocation_plan"] = compute_core_allocation_plan(
        portfolio,
        strategy,
        decision["isa"],
        core_policy,
        etf_set,
        bundles,
        histories,
        vix=macro_vix,
    )
    # /report's approved core ticket is deducted from the same-day /pick core budget.
    decision["isa"].setdefault("approved_report_core_gbp", 0.0)
    # v5.5.7: publish the one deterministic Invest R:R floor (VIX included) so the morning
    # report, Researcher, Critic and Python guardrail all quote the same number.
    inv_rr_floor, inv_rr_tier = resolve_invest_dynamic_rr_floor(
        regime_info["regime"], None, strategy, vix=macro_vix
    )
    decision["invest"]["invest_min_rr_floor"] = round(float(inv_rr_floor), 2)
    decision["invest"]["invest_rr_tier_label"] = inv_rr_tier

    gate = {
        "version": _GATE_VERSION,
        "cycle_date": today_date_str,
        "evaluated_at_uk": eval_dt.strftime("%Y-%m-%d %H:%M"),
        "regime": regime_info["regime"],
        "raw_regime": regime_info["raw_regime"],
        "raw_streak_days": int(regime_info.get("raw_streak_days") or 0),
        "ease_streak_days": int(regime_info.get("ease_streak_days") or 0),
        "risk_on_exit_streak_days": int(regime_info.get("risk_on_exit_streak_days") or 0),
        "recovery_streak_days": int(regime_info.get("recovery_streak_days") or 0),
        "breadth": regime_info.get("breadth") or {},
        "regime_reason": regime_info["reason"],
        "regime_benchmark_symbol": regime_info["benchmark_symbol"],
        "regime_degraded_notes": regime_info["degraded_notes"],
        "etf_set": ledgers["etf_set"],
        "isa": decision["isa"],
        "invest": decision["invest"],
        "triggers": decision["triggers"],
        "global_status": decision["global_status"],
        "pm": decision["pm"],
        "gbp_usd": decision["gbp_usd"],
        "fx_missing": bool(decision["fx_missing"]),
    }
    gate["banner"] = format_deployment_gate_banner(gate)
    return gate


# ---------------------------------------------------------------------------
# 7. 2-year offline historical replay (anti-deadlock validation)
# ---------------------------------------------------------------------------
def replay_deployment_gate_history(
    bench_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    strategy: StrategySection | None = None,
    hy_oas_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Replay the regime state machine over a daily frame and report distribution stats.

    Detects the pathological deadlock where the system stays stuck in RISK_OFF for more
    than 60 straight days while the index has already made a fresh 60-day high.
    """
    strat = strategy or StrategySection()
    closes = _close_series(bench_df)
    vix_closes = _close_series(vix_df) if vix_df is not None else pd.Series(dtype=float)
    if hy_oas_series is None:
        hy_oas_series = pd.Series(dtype=float)
    vix_map = {str(idx.date()): float(v) for idx, v in vix_closes.items()} if len(vix_closes) else {}
    oas_map = {}
    if len(hy_oas_series):
        try:
            oas_map = {str(idx.date()): float(v) for idx, v in hy_oas_series.items()}
        except (TypeError, ValueError):
            oas_map = {}

    start = 200 if len(closes) > 200 else 0
    prev: dict[str, Any] | None = None
    state_counts = {"RISK_OFF": 0, "RECOVERY": 0, "NEUTRAL": 0, "RISK_ON": 0, "UNKNOWN": 0}
    transitions = 0
    last_regime = None
    risk_off_run = 0
    max_risk_off_run = 0
    deadlock_detected = False
    deadlock_details: list[dict[str, Any]] = []
    recovered_days_in_run = 0

    for i in range(start, len(closes)):
        window = closes.iloc[: i + 1]
        date_key = str(window.index[-1].date())
        vix = vix_map.get(date_key)
        hy_oas = oas_map.get(date_key)
        metrics = _compute_raw_regime_metrics(window, vix, hy_oas, strat)
        fake_macro = MacroSnapshot(fetched_at_utc=date_key, vix=vix, hy_oas_spread_pct=hy_oas)
        info = evaluate_market_regime(
            {str(strat.regime_benchmark_symbol).upper(): window}, fake_macro, strat, prev_gate_state=prev
        )
        regime = info["regime"]
        state_counts[regime] = state_counts.get(regime, 0) + 1
        if last_regime is not None and regime != last_regime:
            transitions += 1
        last_regime = regime

        if regime == "RISK_OFF":
            risk_off_run += 1
            # Pathological deadlock signature: benchmark fully stabilized back above MA200
            # (i.e. the crisis is over) yet the regime stays locked for far longer than the
            # exit confirmation window (e.g. the v2 "VIX<18 or bust" bug).
            if metrics["ma200"] is not None and metrics["price"] >= metrics["ma200"]:
                recovered_days_in_run += 1
            else:
                recovered_days_in_run = 0
        else:
            if risk_off_run > max_risk_off_run:
                max_risk_off_run = risk_off_run
            if risk_off_run > 60 and recovered_days_in_run > int(strat.regime_confirm_exit_days):
                deadlock_detected = True
                deadlock_details.append(
                    {
                        "run_days": risk_off_run,
                        "recovered_above_ma200_days": recovered_days_in_run,
                        "ended_at": date_key,
                    }
                )
            risk_off_run = 0
            recovered_days_in_run = 0
        prev = {
            "regime": regime,
            "raw_regime": info["raw_regime"],
            "raw_streak_days": info["raw_streak_days"],
            "ease_streak_days": info["ease_streak_days"],
            "risk_on_exit_streak_days": info.get("risk_on_exit_streak_days", 0),
            "recovery_streak_days": info.get("recovery_streak_days", 0),
        }

    if risk_off_run > max_risk_off_run:
        max_risk_off_run = risk_off_run
    if risk_off_run > 60 and recovered_days_in_run > int(strat.regime_confirm_exit_days):
        deadlock_detected = True
        deadlock_details.append(
            {
                "run_days": risk_off_run,
                "recovered_above_ma200_days": recovered_days_in_run,
                "ended_at": str(closes.index[-1].date()),
            }
        )

    total = sum(state_counts.values())
    return {
        "days_evaluated": total,
        "state_days": state_counts,
        "state_pct": {k: _round2((v / total) * 100.0) if total else 0.0 for k, v in state_counts.items()},
        "transitions": transitions,
        "max_risk_off_run_days": max_risk_off_run,
        "deadlock_detected": deadlock_detected,
        "deadlock_details": deadlock_details,
    }
