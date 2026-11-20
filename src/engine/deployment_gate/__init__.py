"""v5.4.0 Deterministic Deployment Gate (CIO Gate) — pure, offline, zero-network engine.

Connects the daily /report Single Source of Truth with the /pick radar by enforcing:

1. Cash-ledger identity (``available_to_trade`` already excludes reserved BUY cash —
   never subtract pending BUY value twice) and the pending-order three-way split
   (PROTECTIVE_SELL_STOP / CORE_ETF_BUY / NEW_EQUITY_BUY).
2. ISA Core-Satellite structure gates (satellite <=30% & <=8 names incl. pending buys;
   legacy <min_isa_equity_order_gbp crumbs are grandfathered, never force-liquidated).
3. Invest risk-slot engine with financed-risk unlock (protective STOP >= avg cost
   releases the slot), daily best-idea-only cap and epoch-anchored loss-streak cooldown.
4. FX-distortion-free market regime (SPY USD benchmark, VIX, HY OAS) with anti-deadlock
   1d-enter / 3d-exit confirmation hysteresis and step-wise RISK_OFF -> NEUTRAL -> RISK_ON.
5. A/B/C/D triggers, master DEFEND protocol, one-way PM veto clamp, <=110-char banner
   and a 2-year offline replay tool.

This module performs zero network requests and zero direct database writes.
"""

from __future__ import annotations

from src.engine.deployment_gate.common import (
    LONDON_TZ,
    REGIME_HY_OAS_RISK_ON_PCT,
    REGIME_HY_OAS_RISK_OFF_PCT_DEFAULT,
    ISA_EXTREME_LOW_CASH_PCT,
    TRIGGER_EARNINGS_EXCLUDE_DAYS,
    TRIGGER_EXDIV_EXCLUDE_DAYS,
    TRIGGER_RSI_EXCLUDE_ABOVE,
    GATE_EPOCH_PARSE_FAIL_NOTE,
    PENDING_PROTECTIVE_SELL_STOP,
    PENDING_CORE_ETF_BUY,
    PENDING_NEW_EQUITY_BUY,
    _GATE_VERSION,
    _BANNER_MAX_CHARS,
    _sym_up,
    _is_held_symbol,
    _find_position,
    _position_opened_today,
    _symbol_matches,
    _round2,
    _account_value_gbp,
    _parse_uk_midnight_ms,
    _parse_order_ts_ms,
    _close_series,
    _native_risk_to_gbp,
    _pending_buy_gbp,
    _pending_core_etf_gbp,
)
from src.engine.deployment_gate.core_policy import (
    _CORE_ETF_INDEX_FAMILIES,
    _CORE_DIVERSIFIER_FAMILIES,
    CORE_UPPER_BAND_PCT,
    CORE_ANCHOR_TARGET_PCT,
    CORE_ANCHOR_FLOOR_PCT,
    CORE_EU_TARGET_PCT,
    CORE_EU_CAP_PCT,
    CORE_UK_TARGET_PCT,
    CORE_UK_CAP_PCT,
    CORE_US_TILT_TARGET_PCT,
    PORTFOLIO_UK_LOOKTHROUGH_CAP_PCT,
    ISA_UK_LOOKTHROUGH_CAP_PCT,
    EQUITY_US_OVERWEIGHT_FREEZE_PCT,
    CORE_DAILY_BUDGET_MAX_PCT_ISA,
    CORE_DAILY_BUDGET_MAX_GBP,
    CORE_NEW_SLEEVE_MIN_GBP,
    CORE_MAX_TICKETS_PER_REPORT,
    CORE_SLEEVE_GLOBAL_ANCHOR,
    CORE_SLEEVE_EU,
    CORE_SLEEVE_UK,
    CORE_SLEEVE_US_TILT,
    _CORE_SLEEVE_ORDER,
    _CORE_FAMILY_TO_SLEEVE,
    _CORE_SLEEVE_ETF_PRIORITY,
    _CORE_SLEEVE_LABELS,
    _EU_INCL_UK_LOOKTHROUGH,
    core_sleeve_for_symbol,
    _build_etf_set,
    _is_etf_symbol,
    is_core_diversifier_symbol,
    core_etf_universe_symbols,
    compute_core_policy_gaps,
    _core_symbol_trend_metrics,
    compute_core_allocation_plan,
)
from src.engine.deployment_gate.ledgers import (
    classify_pending_orders,
    evaluate_tactical_quota_guards,
    compute_account_ledgers,
)
from src.engine.deployment_gate.regime import (
    _compute_raw_regime_metrics,
    compute_regime_breadth,
    evaluate_market_regime,
)
from src.engine.deployment_gate.triggers import (
    _core_etf_drawdown_step,
    _scan_a_equity_candidates,
    _today_external_inflow,
    scan_triggers_and_decide,
    reconcile_pm_pick_gate,
)
from src.engine.deployment_gate.banner import (
    _fit_banner,
    _gpb_k,
    _core_banner_label,
    format_deployment_gate_banner,
    evaluate_deployment_gate,
    replay_deployment_gate_history,
)

__all__ = [
    "LONDON_TZ",
    "REGIME_HY_OAS_RISK_ON_PCT",
    "REGIME_HY_OAS_RISK_OFF_PCT_DEFAULT",
    "ISA_EXTREME_LOW_CASH_PCT",
    "TRIGGER_EARNINGS_EXCLUDE_DAYS",
    "TRIGGER_EXDIV_EXCLUDE_DAYS",
    "TRIGGER_RSI_EXCLUDE_ABOVE",
    "GATE_EPOCH_PARSE_FAIL_NOTE",
    "PENDING_PROTECTIVE_SELL_STOP",
    "PENDING_CORE_ETF_BUY",
    "PENDING_NEW_EQUITY_BUY",
    "CORE_UPPER_BAND_PCT",
    "CORE_ANCHOR_TARGET_PCT",
    "CORE_ANCHOR_FLOOR_PCT",
    "CORE_EU_TARGET_PCT",
    "CORE_EU_CAP_PCT",
    "CORE_UK_TARGET_PCT",
    "CORE_UK_CAP_PCT",
    "CORE_US_TILT_TARGET_PCT",
    "PORTFOLIO_UK_LOOKTHROUGH_CAP_PCT",
    "ISA_UK_LOOKTHROUGH_CAP_PCT",
    "EQUITY_US_OVERWEIGHT_FREEZE_PCT",
    "CORE_DAILY_BUDGET_MAX_PCT_ISA",
    "CORE_DAILY_BUDGET_MAX_GBP",
    "CORE_NEW_SLEEVE_MIN_GBP",
    "CORE_MAX_TICKETS_PER_REPORT",
    "CORE_SLEEVE_GLOBAL_ANCHOR",
    "CORE_SLEEVE_EU",
    "CORE_SLEEVE_UK",
    "CORE_SLEEVE_US_TILT",
    "core_sleeve_for_symbol",
    "is_core_diversifier_symbol",
    "core_etf_universe_symbols",
    "compute_core_policy_gaps",
    "compute_core_allocation_plan",
    "classify_pending_orders",
    "evaluate_tactical_quota_guards",
    "compute_account_ledgers",
    "compute_regime_breadth",
    "evaluate_market_regime",
    "scan_triggers_and_decide",
    "reconcile_pm_pick_gate",
    "format_deployment_gate_banner",
    "evaluate_deployment_gate",
    "replay_deployment_gate_history",
]
