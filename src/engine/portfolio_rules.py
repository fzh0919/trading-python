"""Deterministic portfolio health, disagreement summary & risk-override guardrail engine."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.config import StrategySection
from src.data_providers.analytics import (
    _ETF_TECH_PENETRATION_PCT,
    TechnicalSnapshot,
    compute_drawdown_since_date,
    infer_etf_index_family,
    normalize_issuer_root,
)
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, PendingOrderItem, PortfolioData, PositionItem

# v5.4.0: single-source VIX danger threshold shared by apply_risk_override_guardrail
# and the deterministic deployment gate (src/engine/deployment_gate.py).
DEFAULT_VIX_DANGER_THRESHOLD: float = 28.0

# --- v5.5.1 Dual-Philosophy Constants (ISA Value vs Invest Right-Side) ---
# All of them are read through ``getattr(strategy, "<lower_name>", <CONST>)`` so the values
# stay overridable from config.yaml later without touching src/config.py today.
# Invest right-side dynamic R:R & noise-floor stops
INVEST_MIN_RR_RISK_ON_LEADER: float = 1.6
INVEST_MIN_RR_NEUTRAL: float = 2.0
INVEST_MIN_RR_WEAK_OR_RISK_OFF: float = 2.5
INVEST_STOP_MIN_ATR_MULT: float = 1.5
INVEST_STOP_MIN_PCT_FLOOR: float = 4.0
INVEST_TRAILING_DD_RISK_ON_PCT: float = -12.0
INVEST_TRAILING_DD_NEUTRAL_PCT: float = -8.5
INVEST_RSI_OVERBOUGHT_RISK_ON: float = 80.0
INVEST_FORWARD_RR_EXIT_WARN: float = 0.8
INVEST_RISK_ON_TARGET_HIGH_HAIRCUT: float = 0.90

# ISA Buffett value quality & margin-of-safety gates
ISA_VALUE_GUARD_MODE_DEFAULT: str = "hard"  # off / advisory / hard
ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT: bool = True
ISA_QUALITY_MIN_ROE_PCT: float = 10.0
ISA_QUALITY_MIN_PROFIT_MARGIN_PCT: float = 5.0
ISA_QUALITY_MAX_DEBT_TO_EQUITY: float = 180.0
ISA_QUALITY_MAX_DEBT_TO_EQUITY_NO_CR: float = 250.0
ISA_QUALITY_MIN_CURRENT_RATIO: float = 0.9
ISA_QUALITY_MIN_REV_GROWTH_PCT: float = -10.0
ISA_QUALITY_MIN_FCF_YIELD_PCT: float = -2.0
ISA_MOS_MIN_DISCOUNT_PCT: float = 12.0
ISA_MOS_CRISIS_DISCOUNT_PCT: float = 20.0
ISA_MOS_MIN_ERP_PCT: float = 1.0
ISA_MOS_CRISIS_MIN_ERP_PCT: float = 1.5
ISA_BUBBLE_FORWARD_PE: float = 45.0
ISA_BUBBLE_PEG_THRESHOLD: float = 2.8
ISA_VALUE_ADD_DRAWDOWN_PCT: float = -10.0
DEFAULT_RISK_FREE_YIELD_PCT: float = 4.2

# Banks / insurers / REITs / utilities legitimately run high leverage; never punish them here.
_LEVERAGE_EXEMPT_SECTOR_KEYWORDS: tuple[str, ...] = (
    "financial", "bank", "insurance", "real estate", "reit", "utilities", "utility",
)

# v5.5.0: only genuine broad-base vehicles keep the crisis-time BUY exemption. Themed,
# dividend, bond and commodity ETFs follow the same VIX freeze as single-name equities.
# v5.5.1: FTSE 100 / Europe broad-base ETFs are core *diversifiers* (they dilute US tech)
# and therefore also keep the VIX exemption and the ISA/ Invest core channel treatment.
_CORE_WIDE_BASE_FAMILIES = {
    "Global All-World/Developed (全球宽基)",
    "S&P 500 (美股标普500)",
    "FTSE 100 (英国富时100)",
    "Europe Equity (欧洲宽基)",
}


class HoldingEvaluation(BaseModel):
    account: str
    ticker: str
    yahoo_symbol: str
    weight_pct: float
    return_pct: float
    action_bias: str  # ADD_CANDIDATE | TRIM_ALERT | HOLD | REVIEW_STOP
    reasons: list[str] = Field(default_factory=list)
    friction_note: str = ""
    native_return_pct: float | None = None
    effective_stop_pct: float | None = None
    pending_buy_weight_pct: float = 0.0
    has_pending_stop: bool = False
    ex_div_guarded: bool = False
    # v5.3.2 benchmark-relative sizing (Python-computed, never left to the LLM)
    rr_ratio: float | None = None
    suggested_size_gbp: float | None = None
    min_order_gbp: float | None = None
    # v5.4.0: ISA legacy <min_isa_equity_order_gbp crumb currently inside a rebound/pruning window
    is_crumb_prune_candidate: bool = False
    # v5.5.1 dual-philosophy annotations (defaults keep old serialized payloads valid)
    philosophy_tag: str = ""                 # "ISA_VALUE" | "INVEST_RIGHT_SIDE"
    value_mos_pct: float | None = None       # ISA margin-of-safety discount (%)
    forward_rr: float | None = None          # Invest forward R:R at current price


class DisagreementItem(BaseModel):
    """Structured local bull-vs-bear evidence summary for a holding with conflicting signals."""

    account: str
    ticker: str
    yahoo_symbol: str
    conflict_type: str
    bull_points: list[str] = Field(default_factory=list)
    bear_points: list[str] = Field(default_factory=list)
    decision_hint: str = ""


class RiskOverrideRecord(BaseModel):
    """Audit record of a single deterministic post-LLM guardrail override."""

    account: str
    symbol: str
    from_action: str
    to_action: str
    trigger_type: str  # ISSUER_OVERWEIGHT | ACCOUNT_OVERWEIGHT | MACRO_VIX | EARNINGS_WINDOW | DAILY_VETO | CORRELATION_OVERLAP | REPORT_ACTION_CONFLICT | EX_DIV_WINDOW | RR_BELOW_FLOOR | RR_UNVERIFIED | FREQUENCY_CAP | REBUY_COOLDOWN | ISA_SATELLITE_SATURATED | ISA_QUALITY_FAIL | ISA_MOS_INSUFFICIENT | STOP_TOO_TIGHT
    reason: str


class RiskOverrideResult(BaseModel):
    """Result returned by apply_risk_override_guardrail without mutating caller input."""

    decision: Any
    applied: bool = False
    overrides: list[RiskOverrideRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _is_uk_listing(yahoo_symbol: str, inst_currency: str) -> bool:
    return yahoo_symbol.endswith(".L") or inst_currency.upper() in {"GBP", "GBX"}


def _us_sell_settled_in_gbp(wallet: AccountSnapshot | None) -> bool:
    """True when a recent US SELL was auto-converted back into the GBP wallet.

    T212 auto-converts USD sale proceeds to GBP unless the multi-currency USD wallet
    is actually funded. That means the next US buy must convert GBP->USD *again*, so a
    single-side FX estimate would silently understate round-trip friction.
    """
    for order in getattr(wallet, "recent_orders", None) or []:
        if str(getattr(order, "side", "") or "").strip().upper() != "SELL":
            continue
        sym = str(getattr(order, "yahoo_symbol", "") or "").strip().upper()
        ccy = str(getattr(order, "inst_currency", "") or "").strip().upper()
        if not sym or sym.endswith(".L") or ccy not in {"USD", ""}:
            continue
        if str(getattr(order, "wallet_currency", "") or "").strip().upper() == "GBP":
            return True
    return False


def _usd_retention_active(
    wallet: AccountSnapshot | None,
    usd_avail: float,
    *,
    min_usd: float,
) -> bool:
    """USD balance is only genuinely retained when it can fund a whole ticket."""
    if usd_avail < float(min_usd):
        return False
    if _us_sell_settled_in_gbp(wallet) and float(getattr(wallet, "usd_cash_total", 0.0) or 0.0) <= 0.01:
        return False
    return True


def wallet_friction_context(wallet: AccountSnapshot | None) -> tuple[bool, float, float]:
    """Return (dual_wallet_enabled, usd_available, gbp_available) for deterministic friction logic.

    When the GBP wallet is unknown (rate not yet injected), fall back to the API
    ``available_to_trade`` figure so we never overstate USD->GBP conversion costs.
    """
    if wallet is None or not getattr(wallet, "multi_currency_enabled", False):
        return False, 0.0, 0.0
    usd_avail = max(0.0, float(getattr(wallet, "usd_available_to_trade", 0.0) or 0.0))
    gbp_raw = getattr(wallet, "gbp_wallet_available", None)
    gbp_avail = (
        max(0.0, float(gbp_raw))
        if gbp_raw is not None
        else max(0.0, float(getattr(wallet, "available_to_trade", 0.0) or 0.0))
    )
    return True, usd_avail, gbp_avail


def _wallet_cash_basis_pct(
    pos: PositionItem,
    strategy: StrategySection,
    wallet: AccountSnapshot | None = None,
) -> float:
    """Cash-interest hurdle basis: 3.4% when spending an existing USD balance, else 4.0% GBP."""
    gbp_basis = float(strategy.cash_interest_rate_pct)
    if pos.account.upper() != "INVEST":
        return gbp_basis
    enabled, usd_avail, gbp_avail = wallet_friction_context(wallet)
    if not enabled:
        return gbp_basis
    usd_basis = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    if _is_uk_listing(pos.yahoo_symbol, pos.inst_currency):
        return usd_basis if gbp_avail <= 0.01 else gbp_basis
    if pos.inst_currency.upper() == "EUR":
        return gbp_basis
    return usd_basis if usd_avail > 0.01 else gbp_basis


def _round_trip_friction_pct(
    pos: PositionItem,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    wallet: AccountSnapshot | None = None,
) -> float:
    """Round-trip friction for the target position, asymmetric for the Invest dual wallet.

    Includes a conservative bid/ask spread estimate (full width, charged once per round trip).
    """
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    is_uk = _is_uk_listing(pos.yahoo_symbol, pos.inst_currency)
    is_invest = pos.account.upper() == "INVEST"
    enabled, usd_avail, gbp_avail = wallet_friction_context(wallet) if is_invest else (False, 0.0, 0.0)
    spread = float(
        getattr(strategy, "spread_estimate_uk_eu_pct", 0.20)
        if is_uk
        else getattr(strategy, "spread_estimate_us_pct", 0.05)
    )

    if is_invest and enabled:
        fx = float(strategy.fx_fee_pct)
        if is_uk:
            stamp = 0.0 if is_etf else float(strategy.uk_stamp_duty_pct)
            return stamp + (0.0 if gbp_avail > 0.01 else fx) + spread
        if pos.inst_currency.upper() == "EUR":
            return fx * 2.0 + spread
        usd_retained = _usd_retention_active(
            wallet, usd_avail, min_usd=max(100.0, float(pos.current_value or 0.0) * 0.5)
        )
        if not usd_retained and float(getattr(wallet, "usd_cash_total", 0.0) or 0.0) <= 0.01:
            if _us_sell_settled_in_gbp(wallet):
                # No real USD retention + proceeds auto-converted => GBP->USD->GBP round trip.
                return fx * 2.0 + spread
        return (0.0 if usd_retained else fx) + spread

    if is_uk and is_etf:
        return spread
    if is_uk:
        return float(strategy.uk_stamp_duty_pct) + spread
    return float(strategy.fx_fee_pct) * 2.0 + spread


def _friction_hint_base(
    pos: PositionItem,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    wallet: AccountSnapshot | None = None,
) -> str:
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    is_uk = _is_uk_listing(pos.yahoo_symbol, pos.inst_currency)
    enabled, usd_avail, gbp_avail = (
        wallet_friction_context(wallet) if pos.account.upper() == "INVEST" else (False, 0.0, 0.0)
    )
    fx = float(strategy.fx_fee_pct)
    if enabled:
        if is_uk and is_etf:
            if gbp_avail > 0.01:
                return "InvestLSE英镑ETF(GBP充足:0%换汇/0%印花税)"
            return f"InvestLSE英镑ETF(GBP不足需USD换回+{fx:.2f}%换汇)"
        if is_uk:
            if gbp_avail > 0.01:
                return f"Invest英股(GBP充足:0%换汇,买入{strategy.uk_stamp_duty_pct:.1f}%印花税)"
            return (
                f"Invest英股(GBP不足需USD换回+{fx:.2f}%换汇,"
                f"买入{strategy.uk_stamp_duty_pct:.1f}%印花税)"
            )
        if pos.inst_currency.upper() == "EUR":
            return f"Invest欧股({fx * 2:.2f}%往返换汇,暂不开EUR钱包)"
        if usd_avail > 0.01:
            return f"Invest美股(可用USD免换汇/新换单边{fx:.2f}%)"
        return f"Invest美股(USD不足需GBP换汇+{fx:.2f}%单边)"
    if is_uk and is_etf:
        return "LSE英镑ETF(0%换汇/0%印花税)"
    if is_uk:
        return f"英股(0%换汇, 买入{strategy.uk_stamp_duty_pct:.1f}%印花税)"
    if pos.inst_currency.upper() == "EUR":
        return f"欧股({strategy.fx_fee_pct:.2f}%单边换汇)"
    return f"美股({strategy.fx_fee_pct * 2:.2f}%往返换汇)"


def _friction_hint(
    pos: PositionItem,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    wallet: AccountSnapshot | None = None,
) -> str:
    """Friction label enriched with the conservative spread estimate used in R:R maths."""
    base = _friction_hint_base(pos, bundle, strategy, wallet)
    spread = float(
        getattr(strategy, "spread_estimate_uk_eu_pct", 0.20)
        if _is_uk_listing(pos.yahoo_symbol, pos.inst_currency)
        else getattr(strategy, "spread_estimate_us_pct", 0.05)
    )
    return f"{base}(含价差估算{spread:.2f}%)"


def compute_risk_based_size_gbp(
    *,
    account: str,
    strategy: StrategySection,
    effective_stop_pct: float | None = None,
    is_etf: bool = False,
    account_total_value_gbp: float | None = None,
    existing_position_value_gbp: float = 0.0,
    risk_budget_scale: float = 1.0,
) -> dict[str, Any]:
    """Risk-based ticket size: risk budget / |stop %|, capped by remaining single-name headroom.

    ``suggested_gbp`` never goes below zero; ``below_min_order`` flags an unfundable idea so the
    LLM layer must widen the stop or pick another vehicle instead of placing a fractional order.
    """
    account_up = (account or "").strip().upper()
    is_invest = account_up == "INVEST"
    if is_invest:
        min_order = float(strategy.min_invest_order_gbp)
        cap_pct = float(strategy.max_invest_single_weight_pct)
    else:
        min_order = float(strategy.min_isa_etf_order_gbp if is_etf else strategy.min_isa_equity_order_gbp)
        cap_pct = float(strategy.max_isa_single_weight_pct)

    base_budget = float(strategy.invest_risk_budget_gbp)
    nav_gbp = float(account_total_value_gbp) if account_total_value_gbp is not None else 0.0
    if is_invest:
        # 0.5% NAV lifetime self-calibrating budget; the loss-streak 0.5R pilot scale
        # applies to Invest only and must never shrink an ISA ticket.
        nav_scaled = nav_gbp * 0.005 if nav_gbp > 0 else 0.0
        risk_budget = max(base_budget, nav_scaled) * max(0.0, float(risk_budget_scale))
    else:
        risk_budget = max(base_budget, nav_gbp * 0.005) if nav_gbp > 0 else base_budget
    stop_pct = (
        abs(float(effective_stop_pct))
        if (effective_stop_pct is not None and float(effective_stop_pct) < 0)
        else None
    )
    budget_sized = (risk_budget / (stop_pct / 100.0)) if stop_pct else None

    headroom: float | None = None
    if account_total_value_gbp is not None and float(account_total_value_gbp) > 0:
        cap_value = float(account_total_value_gbp) * cap_pct / 100.0
        headroom = max(0.0, cap_value - float(existing_position_value_gbp or 0.0))

    candidates = [v for v in (budget_sized, headroom) if v is not None]
    suggested = min(candidates) if candidates else None

    return {
        "account": account_up or account,
        "min_order_gbp": round(min_order, 2),
        "risk_budget_gbp": round(risk_budget, 2),
        "budget_sized_gbp": round(budget_sized, 2) if budget_sized is not None else None,
        "headroom_gbp": round(headroom, 2) if headroom is not None else None,
        "suggested_gbp": round(suggested, 2) if suggested is not None else None,
        "below_min_order": bool(suggested is not None and suggested < min_order),
        "effective_stop_pct": round(-stop_pct, 2) if stop_pct else None,
        "single_name_cap_pct": cap_pct,
    }


def _native_return_pct(pos: PositionItem) -> float | None:
    if pos.avg_price > 0 and pos.cur_price > 0:
        return ((pos.cur_price / pos.avg_price) - 1.0) * 100.0
    return None


def _compute_adaptive_invest_stop_pct(
    tech: TechnicalSnapshot | None,
    strategy: StrategySection,
) -> float:
    """Compute ATR(14)-adaptive Invest stop threshold clamped to [-18%, -8%], falling back to config."""
    if tech is not None and tech.atr_pct is not None and tech.atr_pct > 0:
        return round(max(-18.0, min(-8.0, -2.8 * float(tech.atr_pct))), 2)
    return float(strategy.invest_stop_loss_alert_pct)


def _is_ex_div_guard(tech: TechnicalSnapshot | None, bundle: TickerMarketBundle | None) -> bool:
    if tech is not None and tech.is_ex_div_drop_guard:
        return True
    if bundle is not None and bundle.days_to_ex_dividend is not None:
        return -3 <= int(bundle.days_to_ex_dividend) <= 0
    return False


def _ex_div_buffer_pct(bundle: TickerMarketBundle | None) -> float:
    div_yield = float(bundle.dividend_yield_pct) if (bundle and bundle.dividend_yield_pct) else 2.5
    return max(2.0, min(6.0, div_yield))


def _estimate_pending_order_gbp(
    order: PendingOrderItem,
    pos: PositionItem | None = None,
    pos_by_symbol: dict[str, PositionItem] | None = None,
) -> float:
    """Estimate remaining pending order value in GBP without mixing native currencies."""
    ref_pos = pos or (pos_by_symbol.get(order.yahoo_symbol) if pos_by_symbol else None)
    if order.quantity > 0 and ref_pos is not None and ref_pos.quantity > 0 and ref_pos.current_value > 0:
        return (order.quantity / ref_pos.quantity) * ref_pos.current_value
    if order.value > 0:
        if order.currency == "GBp" or order.currency.upper() == "GBX":
            return order.value / 100.0
        if (
            ref_pos is not None
            and ref_pos.cur_price > 0
            and ref_pos.quantity > 0
            and ref_pos.current_value > 0
            and order.currency.upper() == ref_pos.inst_currency.upper()
        ):
            gbp_per_native_unit = ref_pos.current_value / (ref_pos.cur_price * ref_pos.quantity)
            return order.value * gbp_per_native_unit
        return order.value
    return 0.0


def _infer_account_total_gbp(pos: PositionItem, explicit_total: float | None = None) -> float | None:
    if explicit_total is not None and explicit_total > 0:
        return explicit_total
    if pos.weight_pct > 0 and pos.current_value > 0:
        return (pos.current_value / pos.weight_pct) * 100.0
    return None


def _compute_analyst_upside_pct(bundle: TickerMarketBundle | None) -> float | None:
    if (
        bundle is not None
        and (bundle.analyst_count or 0) >= 3
        and bundle.current_price is not None
        and bundle.current_price > 0
        and bundle.target_mean_price is not None
        and bundle.target_mean_price > 0
    ):
        return ((bundle.target_mean_price / bundle.current_price) - 1.0) * 100.0
    return None


# ---------------------------------------------------------------------------
# v5.3.4 BUY-side deterministic guards (R:R floor / ex-dividend entry / sizing)
# ---------------------------------------------------------------------------
_PICK_RR_TARGET_TOLERANCE_PCT = 15.0


def _pick_guard_mode(strategy: StrategySection) -> str:
    """Resolve the BUY-side R:R / ex-dividend guard strength (off / advisory / hard)."""
    mode = str(getattr(strategy, "pick_rr_guard_mode", "advisory") or "advisory").strip().lower()
    return mode if mode in {"off", "advisory", "hard"} else "advisory"


def _ex_div_yield_hint(bundle: TickerMarketBundle | None) -> str:
    div_yield = float(bundle.dividend_yield_pct) if (bundle and bundle.dividend_yield_pct) else 0.0
    if div_yield <= 0:
        return "股息率待核"
    return f"年化{div_yield:.1f}%/季息≈{div_yield / 4.0:.2f}%"


def _estimate_entry_friction_pct(
    symbol: str,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    account: str = "ISA",
    wallet: AccountSnapshot | None = None,
) -> float:
    """Round-trip friction for a NOT-yet-held symbol (mirrors ``_round_trip_friction_pct``)."""
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    inst_ccy = str(getattr(bundle, "currency", "") or "").upper()
    is_uk = symbol.upper().endswith(".L") or inst_ccy in {"GBP", "GBX"}
    is_invest = str(account or "").strip().upper() == "INVEST"
    enabled, usd_avail, gbp_avail = wallet_friction_context(wallet) if is_invest else (False, 0.0, 0.0)
    spread = float(
        getattr(strategy, "spread_estimate_uk_eu_pct", 0.20)
        if is_uk
        else getattr(strategy, "spread_estimate_us_pct", 0.05)
    )
    if is_invest and enabled:
        fx = float(strategy.fx_fee_pct)
        if is_uk:
            stamp = 0.0 if is_etf else float(strategy.uk_stamp_duty_pct)
            return stamp + (0.0 if gbp_avail > 0.01 else fx) + spread
        if inst_ccy == "EUR":
            return fx * 2.0 + spread
        usd_retained = _usd_retention_active(
            wallet, usd_avail, min_usd=float(strategy.min_invest_order_gbp)
        )
        if not usd_retained and float(getattr(wallet, "usd_cash_total", 0.0) or 0.0) <= 0.01:
            if _us_sell_settled_in_gbp(wallet):
                return fx * 2.0 + spread
        return (0.0 if usd_retained else fx) + spread
    if is_uk and is_etf:
        return spread
    if is_uk:
        return float(strategy.uk_stamp_duty_pct) + spread
    return float(strategy.fx_fee_pct) * 2.0 + spread


def _rescale_price_pair(price: float, ref: float) -> float:
    """Fix the classic GBp/GBP 100x slip between LLM-supplied prices and the quote currency."""
    if price <= 0 or ref <= 0:
        return price
    ratio = price / ref
    if ratio >= 50.0:
        return price / 100.0
    if ratio <= 0.02:
        return price * 100.0
    return price


def _resolve_pick_rr(
    item: Any,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    symbol: str = "",
    account: str = "",
    wallet: AccountSnapshot | None = None,
    tech: TechnicalSnapshot | None = None,
    regime: str = "NEUTRAL",
) -> dict[str, Any]:
    """Recompute BUY-side R:R in Python from structured prices instead of trusting LLM arithmetic.

    v5.5.1: accepts ``tech`` / ``regime`` so RISK_ON right-side leaders get a two-anchor target
    ceiling and every stop is checked against an ATR noise floor (``stop_too_tight``).
    """
    out: dict[str, Any] = {
        "entry": 0.0,
        "stop": 0.0,
        "target": 0.0,
        "risk_pct": None,
        "upside_pct": None,
        "rr": None,
        "friction_pct": None,
        "target_replaced": False,
        "price_rescaled": False,
        "stop_too_tight": False,
        "min_stop_pct": None,
    }
    entry = float(getattr(item, "entry_price", 0.0) or 0.0)
    if entry <= 0 and bundle is not None and bundle.current_price and bundle.current_price > 0:
        entry = float(bundle.current_price)
    if entry <= 0:
        return out
    # Align the LLM-supplied entry with the live quote currency before any R:R maths.
    if bundle is not None and bundle.current_price and float(bundle.current_price) > 0:
        rescaled_entry = _rescale_price_pair(entry, float(bundle.current_price))
        if rescaled_entry != entry:
            out["price_rescaled"] = True
            entry = rescaled_entry

    stop = float(getattr(item, "stop_price", 0.0) or 0.0)
    target = float(getattr(item, "target_price", 0.0) or 0.0)
    rescaled_stop = _rescale_price_pair(stop, entry)
    rescaled_target = _rescale_price_pair(target, entry)
    if rescaled_stop != stop or rescaled_target != target:
        out["price_rescaled"] = True
    stop, target = rescaled_stop, rescaled_target

    analyst_target = 0.0
    if (
        bundle is not None
        and (bundle.analyst_count or 0) >= 3
        and bundle.target_mean_price is not None
        and bundle.target_mean_price > 0
    ):
        analyst_target = _rescale_price_pair(float(bundle.target_mean_price), entry)

    is_right_side = bool(
        str(regime or "").upper() == "RISK_ON" and _is_right_side_leader(tech)
    )
    if is_right_side:
        # v5.5.1 bull-market two-anchor ceiling: never clamp a fresh RISK_ON leader back to a
        # stale analyst mean, but still cap LLM fantasy targets at the highest sane anchor.
        cand_ceilings: list[float] = []
        if analyst_target > 0:
            cand_ceilings.append(analyst_target * (1.0 + _PICK_RR_TARGET_TOLERANCE_PCT / 100.0))
        if bundle is not None and bundle.target_high_price and bundle.target_high_price > 0:
            th = _rescale_price_pair(float(bundle.target_high_price), entry)
            cand_ceilings.append(
                th * float(getattr(strategy, "invest_risk_on_target_high_haircut", INVEST_RISK_ON_TARGET_HIGH_HAIRCUT))
            )
        if tech is not None and tech.atr_pct is not None and tech.atr_pct > 0:
            cand_ceilings.append(entry * (1.0 + 4.0 * float(tech.atr_pct) / 100.0))
        ceiling = max(cand_ceilings) if cand_ceilings else 0.0
        if target <= 0 and analyst_target > 0:
            target = analyst_target
            out["target_replaced"] = True
        if ceiling > 0 and target > ceiling:
            target = ceiling
            out["target_replaced"] = True
        if analyst_target > 0 and target < analyst_target * (1.0 - _PICK_RR_TARGET_TOLERANCE_PCT / 100.0):
            target = analyst_target
            out["target_replaced"] = True
    elif analyst_target > 0:
        if target <= 0:
            target = analyst_target
            out["target_replaced"] = True
        elif abs(target - analyst_target) / analyst_target * 100.0 > _PICK_RR_TARGET_TOLERANCE_PCT:
            # LLM 自报目标价与分析师均值偏离过大 => 判为幻觉价位，改用 Python 侧均值复算。
            target = analyst_target
            out["target_replaced"] = True

    risk_pct = ((entry - stop) / entry) * 100.0 if 0 < stop < entry else None
    upside_pct = ((target / entry) - 1.0) * 100.0 if target > 0 else None
    friction_pct = _estimate_entry_friction_pct(
        symbol or str(getattr(item, "symbol", "") or ""),
        bundle,
        strategy,
        account or str(getattr(item, "account", "") or ""),
        wallet,
    )
    out.update(
        {
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
            "upside_pct": round(upside_pct, 2) if upside_pct is not None else None,
            "friction_pct": round(friction_pct, 2),
        }
    )
    # v5.5.1 Invest stop must clear the ATR noise floor — a razor-thin stop fakes a huge R:R.
    if risk_pct is not None and tech is not None and tech.atr_pct is not None and tech.atr_pct > 0:
        atr_mult = float(getattr(strategy, "invest_stop_min_atr_mult", INVEST_STOP_MIN_ATR_MULT))
        floor_pct = float(getattr(strategy, "invest_stop_min_pct_floor", INVEST_STOP_MIN_PCT_FLOOR))
        min_stop = round(max(atr_mult * float(tech.atr_pct), floor_pct), 2)
        out["min_stop_pct"] = min_stop
        if risk_pct < min_stop:
            out["stop_too_tight"] = True
    if risk_pct is not None and risk_pct > 0 and upside_pct is not None:
        out["rr"] = round((upside_pct - friction_pct) / risk_pct, 2)
    return out


def _is_leverage_exempt_sector(bundle: TickerMarketBundle | None) -> bool:
    """Financials / REITs / utilities run structurally high leverage — do not value-trap them."""
    if bundle is None:
        return False
    text = f"{bundle.sector or ''} {bundle.industry or ''}".lower()
    return any(kw in text for kw in _LEVERAGE_EXEMPT_SECTOR_KEYWORDS)


def evaluate_isa_value_gate(
    bundle: TickerMarketBundle | None,
    tech: TechnicalSnapshot | None,
    macro: MacroSnapshot | None,
    strategy: StrategySection,
    *,
    item: Any | None = None,
    symbol: str = "",
    crisis_mode: bool = False,
) -> dict[str, Any]:
    """ISA Buffett-style gate: quality (anti value-trap) × margin of safety (valuation).

    Replaces the tactical 2:1 price-stop R:R hurdle for ISA single names. A missing financial
    snapshot can never *falsify* quality (quality_pass=True) but always fails the safety margin
    (no valuation evidence => no buy).
    """
    min_roe = float(getattr(strategy, "isa_quality_min_roe_pct", ISA_QUALITY_MIN_ROE_PCT))
    min_margin = float(
        getattr(strategy, "isa_quality_min_profit_margin_pct", ISA_QUALITY_MIN_PROFIT_MARGIN_PCT)
    )
    max_de = float(getattr(strategy, "isa_quality_max_debt_to_equity", ISA_QUALITY_MAX_DEBT_TO_EQUITY))
    max_de_no_cr = float(
        getattr(strategy, "isa_quality_max_debt_to_equity_no_cr", ISA_QUALITY_MAX_DEBT_TO_EQUITY_NO_CR)
    )
    min_cr = float(getattr(strategy, "isa_quality_min_current_ratio", ISA_QUALITY_MIN_CURRENT_RATIO))
    min_rev = float(
        getattr(strategy, "isa_quality_min_rev_growth_pct", ISA_QUALITY_MIN_REV_GROWTH_PCT)
    )
    min_fcf = float(getattr(strategy, "isa_quality_min_fcf_yield_pct", ISA_QUALITY_MIN_FCF_YIELD_PCT))
    min_mos = float(getattr(strategy, "isa_mos_min_discount_pct", ISA_MOS_MIN_DISCOUNT_PCT))
    crisis_mos = float(getattr(strategy, "isa_mos_crisis_discount_pct", ISA_MOS_CRISIS_DISCOUNT_PCT))
    min_erp = float(getattr(strategy, "isa_mos_min_erp_pct", ISA_MOS_MIN_ERP_PCT))
    crisis_erp = float(getattr(strategy, "isa_mos_crisis_min_erp_pct", ISA_MOS_CRISIS_MIN_ERP_PCT))
    bubble_pe_th = float(getattr(strategy, "isa_bubble_forward_pe", ISA_BUBBLE_FORWARD_PE))
    bubble_peg_th = float(getattr(strategy, "isa_bubble_peg_threshold", ISA_BUBBLE_PEG_THRESHOLD))

    quality_fails: list[str] = []
    quality_pros: list[str] = []
    mos_reasons: list[str] = []

    if bundle is None:
        return {
            "quality_pass": True,
            "mos_pass": False,
            "passed": False,
            "bubble_risk": False,
            "crisis_mode": bool(crisis_mode),
            "mos_pct": None,
            "erp_pct": None,
            "fcf_yield_pct": None,
            "quality_fails": [],
            "quality_pros": [],
            "quality_note": "财务快照缺失待核",
            "mos_reasons": ["缺少现价/估值数据，无法确认安全边际"],
            "summary": "财务快照缺失待核 | 安全边际无法确认",
        }

    # --- Quality gate (anti value-trap) -------------------------------------------------
    if bundle.profit_margin_pct is not None and bundle.profit_margin_pct <= 0.0:
        quality_fails.append("净利率≤0%(亏损企业)")
    if bundle.roe_pct is not None and bundle.roe_pct <= 0.0:
        quality_fails.append("ROE≤0%(资本回报为负)")

    if not _is_leverage_exempt_sector(bundle) and bundle.debt_to_equity is not None:
        de = float(bundle.debt_to_equity)
        if bundle.current_ratio is not None:
            cr = float(bundle.current_ratio)
            if de > max_de and cr < min_cr:
                quality_fails.append(f"高杠杆且流动性偏紧(D/E={de:.0f}, CR={cr:.2f})")
        elif de > max_de_no_cr:
            quality_fails.append(f"非金融企业极高杠杆(D/E={de:.0f})")

    if bundle.revenue_growth_pct is not None and bundle.revenue_growth_pct <= min_rev:
        quality_fails.append("营收大幅萎缩(<= -10%)")
    if bundle.fcf_yield_pct is not None and bundle.fcf_yield_pct < min_fcf:
        quality_fails.append(f"自由现金流明显为负(FCF率={bundle.fcf_yield_pct:.1f}%)")

    if bundle.roe_pct is not None and bundle.roe_pct >= min_roe:
        quality_pros.append(f"ROE {bundle.roe_pct:.1f}%")
    if bundle.profit_margin_pct is not None and bundle.profit_margin_pct >= min_margin:
        quality_pros.append(f"净利率 {bundle.profit_margin_pct:.1f}%")
    if bundle.fcf_yield_pct is not None and bundle.fcf_yield_pct >= 3.0:
        quality_pros.append(f"FCF率 {bundle.fcf_yield_pct:.1f}%")

    quality_pass = len(quality_fails) == 0

    # --- Bubble risk ---------------------------------------------------------------------
    pe = None
    if bundle.forward_pe is not None and bundle.forward_pe > 0:
        pe = float(bundle.forward_pe)
    elif bundle.trailing_pe is not None and bundle.trailing_pe > 0:
        pe = float(bundle.trailing_pe)
    high_pe = bool(pe is not None and pe >= bubble_pe_th)
    growth_insufficient = bool(
        (bundle.peg_ratio is not None and bundle.peg_ratio >= bubble_peg_th)
        or (bundle.revenue_growth_pct is not None and bundle.revenue_growth_pct < 15.0)
        or (bundle.peg_ratio is None and bundle.revenue_growth_pct is None)
    )
    bubble_risk = bool(high_pe and growth_insufficient)

    # --- Margin of safety ----------------------------------------------------------------
    entry = float(getattr(item, "entry_price", 0.0) or 0.0)
    if entry <= 0:
        entry = float(bundle.current_price or 0.0)
    if entry > 0 and bundle.current_price and float(bundle.current_price) > 0:
        entry = _rescale_price_pair(entry, float(bundle.current_price))
    friction_pct = _estimate_entry_friction_pct(
        symbol or bundle.symbol, bundle, strategy, "ISA", None
    )

    target_mos_pct: float | None = None
    if (
        (bundle.analyst_count or 0) >= 3
        and bundle.target_mean_price is not None
        and bundle.target_mean_price > 0
        and entry > 0
    ):
        ref_target = _rescale_price_pair(float(bundle.target_mean_price), entry)
        item_target = float(getattr(item, "target_price", 0.0) or 0.0) if item is not None else 0.0
        if item_target > 0:
            llm_t = _rescale_price_pair(item_target, entry)
            if llm_t <= ref_target * 1.15:
                ref_target = min(llm_t, ref_target)
        target_mos_pct = round(((ref_target / entry) - 1.0) * 100.0 - friction_pct, 2)
    else:
        item_target = float(getattr(item, "target_price", 0.0) or 0.0) if item is not None else 0.0
        if item_target > 0 and entry > 0:
            llm_t = _rescale_price_pair(item_target, entry)
            target_mos_pct = round(
                min(((llm_t / entry) - 1.0) * 100.0, 25.0) - friction_pct, 2
            )

    rf_yield = DEFAULT_RISK_FREE_YIELD_PCT
    if macro is not None and macro.us_10y_yield is not None and macro.us_10y_yield > 0:
        rf_yield = float(macro.us_10y_yield)
    erp_pct: float | None = round((100.0 / pe) - rf_yield, 2) if pe is not None and pe > 0 else None
    fcf_yield_pct = bundle.fcf_yield_pct
    dd52 = tech.drawdown_52w_pct if tech is not None else None

    if bubble_risk:
        mos_pass = False
        mos_reasons.append("估值处于高PE泡沫区，缺乏安全边际")
    elif not crisis_mode:
        cond_target = bool(target_mos_pct is not None and target_mos_pct >= min_mos)
        cond_erp = bool(
            erp_pct is not None
            and erp_pct >= min_erp
            and (target_mos_pct is None or target_mos_pct >= 6.0)
            and (fcf_yield_pct is None or fcf_yield_pct >= 2.0)
        )
        cond_low_pe = bool(
            target_mos_pct is None
            and pe is not None
            and pe <= 20.0
            and (bundle.ev_ebitda is None or bundle.ev_ebitda <= 14.0)
        )
        mos_pass = bool(cond_target or cond_erp or cond_low_pe)
        if not mos_pass:
            parts = []
            if target_mos_pct is not None:
                parts.append(f"目标折价{target_mos_pct:+.1f}%(<{min_mos:.0f}%)")
            else:
                parts.append("缺分析师目标价")
            if erp_pct is not None:
                parts.append(f"ERP{erp_pct:+.1f}%(<{min_erp:.1f}%)")
            mos_reasons.append("常规安全边际不足：" + "，".join(parts))
    else:
        deep_discount = bool(target_mos_pct is not None and target_mos_pct >= crisis_mos)
        value_anchor = bool(
            (erp_pct is not None and erp_pct >= crisis_erp)
            or (fcf_yield_pct is not None and fcf_yield_pct >= 4.5)
            or (dd52 is not None and dd52 <= -15.0)
        )
        mos_pass = bool(deep_discount and value_anchor)
        if not mos_pass:
            if not deep_discount:
                got = f"{target_mos_pct:+.1f}%" if target_mos_pct is not None else "缺失"
                mos_reasons.append(f"危机深折价不足(目标折价{got}<{crisis_mos:.0f}%)")
            if not value_anchor:
                mos_reasons.append("危机模式缺少估值/现金流/深跌支撑")

    mos_bits: list[str] = []
    if target_mos_pct is not None:
        mos_bits.append(f"安全边际{target_mos_pct:+.1f}%")
    if erp_pct is not None:
        mos_bits.append(f"ERP{erp_pct:+.1f}%")
    if fcf_yield_pct is not None:
        mos_bits.append(f"FCF {fcf_yield_pct:.1f}%")
    quality_bit = "质量通过" if quality_pass else "质量门未过"
    summary = quality_bit + (" | " + " / ".join(mos_bits) if mos_bits else " | 无估值锚")

    return {
        "quality_pass": quality_pass,
        "mos_pass": mos_pass,
        "passed": bool(quality_pass and mos_pass),
        "bubble_risk": bubble_risk,
        "crisis_mode": bool(crisis_mode),
        "mos_pct": target_mos_pct,
        "erp_pct": erp_pct,
        "fcf_yield_pct": fcf_yield_pct,
        "quality_fails": quality_fails,
        "quality_pros": quality_pros,
        "quality_note": "；".join(quality_pros) if quality_pros else "",
        "mos_reasons": mos_reasons,
        "summary": summary,
    }


def _is_right_side_leader(tech: TechnicalSnapshot | None) -> bool:
    """True when a name is a clean Stage-2 style leader (above MA200, BULL/PULLBACK, weekly OK)."""
    if tech is None:
        return False
    above_ma200 = tech.dist_ma200_pct is not None and tech.dist_ma200_pct > 0.0
    return bool(
        above_ma200
        and tech.trend_regime in {"BULL", "PULLBACK"}
        and tech.weekly_bullish is not False
    )


def resolve_invest_dynamic_rr_floor(
    regime: str | None,
    tech: TechnicalSnapshot | None,
    strategy: StrategySection,
) -> tuple[float, str]:
    """Regime × structure dynamic R:R floor for Invest (1.6 / 2.0 / 2.5)."""
    reg = str(regime or "NEUTRAL").strip().upper()
    base_neutral = float(
        getattr(strategy, "invest_min_rr_neutral", None)
        or getattr(strategy, "min_rr_ratio", INVEST_MIN_RR_NEUTRAL)
        or INVEST_MIN_RR_NEUTRAL
    )
    rr_leader = float(
        getattr(strategy, "invest_min_rr_risk_on_leader", INVEST_MIN_RR_RISK_ON_LEADER)
    )
    rr_weak = float(getattr(strategy, "invest_min_rr_weak_or_risk_off", INVEST_MIN_RR_WEAK_OR_RISK_OFF))

    is_weak = bool(
        reg == "RISK_OFF"
        or (
            tech is not None
            and (
                tech.trend_regime == "WEAK"
                or (tech.dist_ma200_pct is not None and tech.dist_ma200_pct <= 0.0)
            )
        )
    )
    if is_weak:
        return rr_weak, "弱势/防御高赔率门槛"
    if reg == "RISK_ON" and _is_right_side_leader(tech):
        return min(rr_leader, base_neutral), "RISK_ON右侧强结构"
    return base_neutral, f"{reg or 'NEUTRAL'}标准门槛"


def compute_invest_forward_rr(
    pos: PositionItem,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    effective_stop_pct: float | None,
    regime: str = "NEUTRAL",
    wallet: AccountSnapshot | None = None,
) -> float | None:
    """Forward R:R from *today's* price to the remaining target, per unit of stop risk.

    A reading below ~0.8:1 means the leftover upside no longer pays for the downside risk,
    i.e. the position became a "鸡肋" candidate for profit-lock / ROTATE.
    """
    if pos.cur_price <= 0 or effective_stop_pct is None or effective_stop_pct >= 0:
        return None
    cur = (
        float(bundle.current_price)
        if (bundle is not None and bundle.current_price and bundle.current_price > 0)
        else float(pos.cur_price)
    )
    if cur <= 0:
        return None
    if (
        bundle is None
        or (bundle.analyst_count or 0) < 3
        or bundle.target_mean_price is None
        or bundle.target_mean_price <= 0
    ):
        return None
    mean_t = _rescale_price_pair(float(bundle.target_mean_price), cur)
    target = mean_t
    if (
        str(regime or "").upper() == "RISK_ON"
        and _is_right_side_leader(tech)
        and bundle.target_high_price is not None
        and bundle.target_high_price > 0
    ):
        high_t = _rescale_price_pair(float(bundle.target_high_price), cur) * float(
            getattr(strategy, "invest_risk_on_target_high_haircut", INVEST_RISK_ON_TARGET_HIGH_HAIRCUT)
        )
        target = max(mean_t, high_t)
    upside_pct = ((target / cur) - 1.0) * 100.0
    risk_pct = abs(float(effective_stop_pct))
    if risk_pct <= 0:
        return None
    friction_pct = _round_trip_friction_pct(pos, bundle, strategy, wallet) * 0.5
    return round((upside_pct - friction_pct) / risk_pct, 2)


def _shares_from_gbp(
    gbp_amount: float,
    price_local: float,
    currency: str,
    symbol: str,
    gbp_usd_rate: float | None = None,
) -> int | None:
    """Convert a GBP risk budget into an integer share count, or None when FX/currency is unknown."""
    if gbp_amount <= 0 or price_local <= 0:
        return None
    raw_ccy = str(currency or "")
    upper = raw_ccy.upper()
    if raw_ccy in {"GBp", "GBx"} or upper == "GBX":
        price_gbp = price_local / 100.0
    elif upper == "GBP":
        price_gbp = price_local
    elif upper == "USD" and gbp_usd_rate and float(gbp_usd_rate) > 0:
        price_gbp = price_local / float(gbp_usd_rate)
    else:
        # Missing currency must return None: blindly dividing a .L symbol by 100 would
        # turn a GBP-denominated ETF quote (VWRP.L ≈ £114) into £1.14.
        return None
    if price_gbp <= 0:
        return None
    return max(0, int(gbp_amount // price_gbp))


_SIZE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SIZE_FRAC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
_SIZE_GBP_RE = re.compile(r"£\s*([\d,]+(?:\.\d+)?)")
_FULL_EXIT_TOKENS = ("全出", "清仓", "全部卖出", "全卖", "全部清仓")


def _fmt_shares(shares: float) -> str:
    if shares <= 0:
        return ""
    if abs(shares - round(shares)) < 0.005:
        return f"≈{int(round(shares)):,}股"
    return f"≈{shares:,.2f}股"


def derive_action_size_note(action: str, size_hint: str, position: Any | None = None) -> str:
    """Deterministic share-count note for /report top_actions (never LLM arithmetic).

    TRIM percentages/fractions are applied to the existing holding; GBP amounts are converted
    with the position's own GBP-per-share mark, so GBp/GBP 100x mix-ups cannot leak in.
    ADD percentages stay as-is because their base (cash vs target weight) is ambiguous.
    """
    act = str(action or "").strip().upper()
    hint = str(size_hint or "").strip()
    if act not in {"ADD", "TRIM", "STOP"}:
        return ""
    if any(tok in hint for tok in _FULL_EXIT_TOKENS):
        return ""
    if act == "STOP":
        return "触发即全出" if not hint else ""
    if not hint:
        return ""
    qty = float(getattr(position, "quantity", 0.0) or 0.0) if position is not None else 0.0
    if act != "TRIM":
        return ""
    m = _SIZE_PCT_RE.search(hint)
    if m and qty > 0:
        return _fmt_shares(qty * float(m.group(1)) / 100.0)
    frac = _SIZE_FRAC_RE.search(hint)
    if frac and qty > 0:
        den = float(frac.group(2))
        if den > 0:
            return _fmt_shares(qty * float(frac.group(1)) / den)
    amt = _SIZE_GBP_RE.search(hint)
    if amt and position is not None and qty > 0:
        try:
            amount = float(amt.group(1).replace(",", ""))
        except ValueError:
            return ""
        value = float(getattr(position, "current_value", 0.0) or 0.0)
        if amount > 0 and value > 0:
            return _fmt_shares(amount / (value / qty))
    return ""


def build_report_size_notes(top_actions: list[Any] | None, portfolio: Any | None) -> dict[tuple[str, str], str]:
    """Map ``(ACCOUNT, SYMBOL)`` -> deterministic size note for :func:`render_morning_report`."""
    notes: dict[tuple[str, str], str] = {}
    if portfolio is None:
        return notes
    positions = list(portfolio.all_positions())
    for act in top_actions or []:
        account = str(getattr(act, "account", "") or "").strip().upper()
        symbol = str(getattr(act, "symbol", "") or "").strip().upper()
        if not symbol or symbol == "CASH":
            continue
        pos = next(
            (
                p
                for p in positions
                if str(getattr(p, "account", "")).strip().upper() == account
                and symbol
                in {
                    str(getattr(p, "yahoo_symbol", "")).strip().upper(),
                    str(getattr(p, "ticker", "")).strip().upper(),
                }
            ),
            None,
        )
        note = derive_action_size_note(getattr(act, "action", ""), getattr(act, "size_hint", ""), pos)
        if note:
            notes[(account, symbol)] = note
    return notes


def _build_pick_size_note(
    item: Any,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    account: str,
    account_total_value_gbp: float | None = None,
    existing_position_value_gbp: float = 0.0,
    gbp_usd_rate: float | None = None,
    wallet: AccountSnapshot | None = None,
    risk_budget_scale: float = 1.0,
    tech: TechnicalSnapshot | None = None,
    macro: MacroSnapshot | None = None,
    regime: str = "NEUTRAL",
    existing_position_weight_pct: float = 0.0,
) -> str:
    """Deterministic sizing hint: risk budget / stop distance for Invest, value batching for ISA."""
    acct_up = str(account or "").strip().upper()
    symbol = str(getattr(item, "symbol", "") or "")
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    stop_hint = float(getattr(item, "stop_price", 0.0) or 0.0)

    # v5.5.1 ISA value position: no tactical stop needed -> batch-in by valuation, not by ATR.
    if acct_up == "ISA" and not is_etf and stop_hint <= 0:
        gate = evaluate_isa_value_gate(
            bundle,
            tech,
            macro,
            strategy,
            item=item,
            symbol=symbol,
            crisis_mode=str(regime or "").upper() == "RISK_OFF",
        )
        if not gate.get("mos_pass"):
            return ""
        total = float(account_total_value_gbp or 0.0)
        if total <= 0:
            return f"ISA价值分批建议：{gate.get('summary')}，建议分批逢低建仓"
        cap_pct = float(getattr(strategy, "max_single_name_portfolio_pct", 12.0) or 12.0)
        headroom = max(0.0, total * cap_pct / 100.0 - float(existing_position_value_gbp or 0.0))
        if headroom <= 0:
            return ""
        step = min(headroom, max(float(strategy.min_isa_equity_order_gbp), total * 0.04))
        return (
            f"ISA价值分批建议 £{step:,.0f}"
            f"（单票余量£{headroom:,.0f}，{gate.get('summary')}，建议分批逢低建仓）"
        )

    info = _resolve_pick_rr(
        item,
        bundle,
        strategy,
        symbol=symbol,
        account=account,
        wallet=wallet,
        tech=tech,
        regime=regime,
    )
    entry = float(info.get("entry") or 0.0)
    stop = float(info.get("stop") or 0.0)
    risk_pct = info.get("risk_pct")
    if entry <= 0 or stop <= 0 or not risk_pct:
        return ""
    sizing = compute_risk_based_size_gbp(
        account=account,
        strategy=strategy,
        effective_stop_pct=-abs(float(risk_pct)),
        is_etf=is_etf,
        account_total_value_gbp=account_total_value_gbp,
        existing_position_value_gbp=float(existing_position_value_gbp or 0.0),
        risk_budget_scale=float(risk_budget_scale or 1.0),
    )
    suggested = sizing.get("suggested_gbp")
    if not suggested or float(suggested) <= 0:
        return ""
    cap_pct = float(sizing.get("single_name_cap_pct") or 0.0)
    parts = [
        f"以损定仓上限 £{float(suggested):,.0f}"
        f"（预算£{float(sizing.get('risk_budget_gbp') or 0.0):,.0f}÷止损{float(risk_pct):.1f}%，单仓≤{cap_pct:.0f}%）"
    ]
    if acct_up == "INVEST":
        min_rr, tier_label = resolve_invest_dynamic_rr_floor(regime, tech, strategy)
        parts.append(f"动态门槛{tier_label}≥{min_rr:.1f}:1")
    shares = _shares_from_gbp(
        float(suggested),
        entry,
        str(getattr(bundle, "currency", "") or ""),
        symbol,
        gbp_usd_rate,
    )
    if shares is not None and shares > 0:
        parts.append(f"≈{shares}股")
    if sizing.get("below_min_order"):
        parts.append(f"低于最小建仓额£{float(sizing.get('min_order_gbp') or 0.0):,.0f}，勿碎单")
    return "，".join(parts)


def evaluate_holding(
    pos: PositionItem,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    total_issuer_weight_pct: float | None = None,
    *,
    hist: pd.DataFrame | None = None,
    pending_orders: list[PendingOrderItem] | None = None,
    account_total_value: float | None = None,
    total_portfolio_value: float | None = None,
    wallet: AccountSnapshot | None = None,
    regime: str = "NEUTRAL",
    macro: MacroSnapshot | None = None,
) -> HoldingEvaluation:
    """Evaluate an existing position deterministically for Add / Trim / Stop / Hold.

    v5.5.1 splits the philosophy: ISA = Buffett value (quality × margin of safety, no short-term
    volatility sell signals), Invest = right-side momentum (regime-adaptive R:R, ATR noise floor,
    forward R:R decay).
    """
    reasons: list[str] = []
    add_score = 0
    trim_score = 0
    stop_flag = False

    is_invest = pos.account.upper() == "INVEST"
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    reg_up = str(regime or "NEUTRAL").strip().upper()
    philosophy_tag = "INVEST_RIGHT_SIDE" if is_invest else "ISA_VALUE"
    val_gate: dict[str, Any] | None = None
    if not is_invest and not is_etf:
        val_gate = evaluate_isa_value_gate(
            bundle,
            tech,
            macro,
            strategy,
            symbol=pos.yahoo_symbol,
            crisis_mode=(reg_up == "RISK_OFF"),
        )
    value_mos_pct = val_gate.get("mos_pct") if val_gate else None
    max_weight = strategy.max_invest_single_weight_pct if is_invest else strategy.max_isa_single_weight_pct
    earnings_soon = bool(
        bundle and bundle.days_to_earnings is not None and 0 <= bundle.days_to_earnings <= 5
    )
    ex_div_guard = _is_ex_div_guard(tech, bundle)
    ex_div_buf = _ex_div_buffer_pct(bundle) if ex_div_guard else 0.0
    native_ret = _native_return_pct(pos)
    effective_stop = _compute_adaptive_invest_stop_pct(tech, strategy) if is_invest else None

    # Active pending orders linkage (same account orders + cross-account issuer buy overlap)
    acct_key = pos.account.upper()
    pos_root = normalize_issuer_root(pos.ticker)
    pos_sym_up = pos.yahoo_symbol.upper()
    same_acct_orders = [
        o
        for o in (pending_orders or [])
        if o.account.upper() == acct_key
        and (o.yahoo_symbol.upper() == pos_sym_up or normalize_issuer_root(o.ticker) == pos_root)
    ]
    same_issuer_orders = [
        o
        for o in (pending_orders or [])
        if o.yahoo_symbol.upper() == pos_sym_up or normalize_issuer_root(o.ticker) == pos_root
    ]
    has_pending_stop = any(
        o.side.upper() == "SELL" or "STOP" in o.order_type.upper() for o in same_acct_orders
    )
    pending_buy_gbp = sum(
        _estimate_pending_order_gbp(o, pos) for o in same_acct_orders if o.side.upper() == "BUY"
    )
    pending_issuer_buy_gbp = sum(
        _estimate_pending_order_gbp(o, pos)
        for o in same_issuer_orders
        if o.side.upper() == "BUY"
    )

    inferred_acct_total = _infer_account_total_gbp(pos, account_total_value)
    pending_buy_weight_pct = (
        (pending_buy_gbp / inferred_acct_total) * 100.0
        if (pending_buy_gbp > 0 and inferred_acct_total and inferred_acct_total > 0)
        else 0.0
    )
    projected_acct_weight = pos.weight_pct + pending_buy_weight_pct
    projected_issuer_weight = total_issuer_weight_pct
    if total_issuer_weight_pct is not None and pending_issuer_buy_gbp > 0 and total_portfolio_value and total_portfolio_value > 0:
        projected_issuer_weight = total_issuer_weight_pct + (pending_issuer_buy_gbp / total_portfolio_value) * 100.0

    over_weight = False
    if not is_etf:
        if pos.weight_pct >= max_weight:
            over_weight = True
            trim_score += 2
            reasons.append(f"单只占{pos.account}仓位达{pos.weight_pct:.1f}%(≥上限{max_weight:.0f}%),存在集中度风险")
        elif pending_buy_weight_pct > 0 and projected_acct_weight >= max_weight:
            over_weight = True
            trim_score += 1
            reasons.append(
                f"计入在途BUY挂单后占{pos.account}仓位将达{projected_acct_weight:.1f}%(≥上限{max_weight:.0f}%),禁止继续追单"
            )

        if total_issuer_weight_pct is not None and total_issuer_weight_pct >= strategy.max_total_portfolio_weight_pct:
            over_weight = True
            trim_score += 2
            reasons.append(
                f"同发行人跨账户合并占全组合{total_issuer_weight_pct:.1f}%(≥上限{strategy.max_total_portfolio_weight_pct:.0f}%),禁止继续加码"
            )
        elif (
            projected_issuer_weight is not None
            and projected_issuer_weight >= strategy.max_total_portfolio_weight_pct
        ):
            over_weight = True
            trim_score += 1
            reasons.append(
                f"计入在途BUY挂单后跨账户合并将达{projected_issuer_weight:.1f}%(≥上限{strategy.max_total_portfolio_weight_pct:.0f}%),严禁加码"
            )

    # Invest stop-loss check with ATR(14) adaptive threshold, ex-div guard, and FX distortion check
    if is_invest and effective_stop is not None:
        atr_note = f"[ATR自适应{tech.atr_pct:.1f}%]" if (tech is not None and tech.atr_pct) else ""
        if pos.return_pct <= effective_stop:
            if ex_div_guard and pos.return_pct > (effective_stop - ex_div_buf):
                reasons.append(
                    f"处于除息窗口(账面{pos.return_pct:.1f}%略触止损线{effective_stop:.1f}%),暂免除权低开误触发"
                )
            else:
                stop_flag = True
                trim_score += 2
                reasons.append(
                    f"Invest短线仓浮亏{pos.return_pct:.1f}%(已触发止损复核线{effective_stop:.1f}%{atr_note})"
                )
        elif (
            native_ret is not None
            and pos.inst_currency.upper() not in {"GBP", "GBX"}
            and native_ret <= effective_stop
            and not (ex_div_guard and native_ret > (effective_stop - ex_div_buf))
        ):
            stop_flag = True
            trim_score += 2
            reasons.append(
                f"本币真实浮亏{native_ret:.1f}%已触及止损线{effective_stop:.1f}%{atr_note}(英镑账面{pos.return_pct:+.1f}%受汇率掩盖)"
            )

    isa_deep_dd = (not is_invest) and (pos.return_pct <= strategy.isa_deep_drawdown_review_pct)
    isa_quality_broken = bool(val_gate is not None and val_gate.get("quality_pass") is False)
    if isa_quality_broken:
        trim_score += 2
        fails_txt = "；".join(val_gate.get("quality_fails") or []) or "财务质量指标恶化"
        reasons.append(f"ISA基本面质量门预警({fails_txt})，须核查护城河是否证伪")
    if val_gate is not None and val_gate.get("bubble_risk"):
        # One of the 4 legitimate ISA sell/review cases: extreme valuation bubble.
        trim_score += 2
        reasons.append("ISA估值进入泡沫区(安全边际透支)，建议复核估值再平衡")
    if isa_deep_dd:
        trim_score += 1
        reasons.append(
            f"ISA长线仓浮亏{pos.return_pct:.1f}%(≤深套复核线{strategy.isa_deep_drawdown_review_pct:.0f}%),须先核查基本面是否证伪而非死扛摊平"
        )

    if is_invest and pos.return_pct >= strategy.invest_take_profit_review_pct:
        trim_score += 1
        reasons.append(f"Invest仓浮盈+{pos.return_pct:.1f}%,需评估催化剂是否已充分兑现以考虑分批止盈")

    if (
        native_ret is not None
        and pos.inst_currency.upper() not in {"GBP", "GBX"}
        and abs(pos.return_pct - native_ret) >= 5.0
        and (stop_flag or isa_deep_dd or pos.return_pct >= strategy.trailing_profit_activation_pct)
    ):
        reasons.append(f"[汇率校准: 本币{native_ret:+.1f}% vs 英镑{pos.return_pct:+.1f}%]")

    # Resolve post-entry trailing drawdown: prefer per-position hist + created_at when available
    recent_dd: float | None = None
    if hist is not None and pos.created_at:
        recent_dd = compute_drawdown_since_date(hist, pos.created_at, max_lookback_days=60)
    if recent_dd is None and tech is not None:
        recent_dd = tech.drawdown_60d_pct if tech.drawdown_60d_pct is not None else tech.drawdown_20d_pct

    # v5.5.1: trailing profit protection is an Invest (right-side) tool only. ISA value holdings
    # are never sold on short-term volatility; they exit on quality/bubble/concentration instead.
    trailing_triggered = False
    trailing_dd_thresh = float(strategy.trailing_drawdown_from_peak_pct)
    if is_invest:
        trailing_dd_thresh = (
            float(getattr(strategy, "invest_trailing_dd_risk_on_pct", INVEST_TRAILING_DD_RISK_ON_PCT))
            if reg_up == "RISK_ON"
            else float(getattr(strategy, "invest_trailing_dd_neutral_pct", INVEST_TRAILING_DD_NEUTRAL_PCT))
        )
    if (
        is_invest
        and pos.return_pct >= strategy.trailing_profit_activation_pct
        and recent_dd is not None
        and recent_dd <= trailing_dd_thresh
    ):
        if ex_div_guard and recent_dd > (trailing_dd_thresh - ex_div_buf):
            reasons.append(
                f"浮盈+{pos.return_pct:.1f}%但处于除息窗口(回撤{recent_dd:.1f}%),暂免移动止盈误触"
            )
        else:
            trailing_triggered = True
            trim_score += 2
            window_lbl = "开仓后波段高点" if pos.created_at else "近60日波段高点"
            reasons.append(
                f"浮盈+{pos.return_pct:.1f}%后自{window_lbl}回撤{recent_dd:.1f}%(≤{trailing_dd_thresh:.1f}%),触发移动止盈保护警戒"
            )

    if has_pending_stop and (stop_flag or trailing_triggered):
        stop_orders = [
            o for o in same_acct_orders if o.side.upper() == "SELL" or "STOP" in o.order_type.upper()
        ]
        prices = [
            f"{(o.stop_price or o.limit_price):.2f}"
            for o in stop_orders
            if (o.stop_price or o.limit_price) is not None
        ]
        price_hint = f"@{','.join(prices[:2])}" if prices else ""
        reasons.append(f"已有在途卖出/止损挂单保护{price_hint},请复核触发价是否充分")

    holding_above_ma20 = bool(
        tech is not None and tech.dist_ma20_pct is not None and tech.dist_ma20_pct >= 0.0
    )
    heavy_down_vol = bool(
        tech is not None
        and tech.volume_ratio_20d is not None
        and tech.volume_ratio_20d >= 1.8
        and tech.ret_1d_pct is not None
        and tech.ret_1d_pct <= -2.0
        and not ex_div_guard
    )
    invest_bull_ride = bool(
        is_invest and reg_up == "RISK_ON" and holding_above_ma20 and not heavy_down_vol
    )
    isa_value_add = False

    if tech is not None:
        rsi_ob_risk_on = float(
            getattr(strategy, "invest_rsi_overbought_risk_on", INVEST_RSI_OVERBOUGHT_RISK_ON)
        )
        if tech.rsi14 is not None and tech.rsi14 >= strategy.rsi_overbought_threshold:
            if invest_bull_ride and tech.rsi14 < rsi_ob_risk_on:
                # Let the winner run: never trim a RISK_ON leader on RSI alone.
                reasons.append(
                    f"RISK_ON强趋势RSI钝化(RSI14={tech.rsi14:.1f})且稳居MA20上方，"
                    "建议上移移动止损锁利，勿急于卖飞"
                )
            elif invest_bull_ride and tech.rsi14 >= rsi_ob_risk_on:
                trim_score += 1
                reasons.append(
                    f"RSI14={tech.rsi14:.1f}(≥{rsi_ob_risk_on:.0f}极端超买)，"
                    "仅提示上移移动止损，需叠加破位/乖离/前瞻R:R恶化再决策"
                )
            elif is_invest:
                trim_score += 2
                reasons.append(f"RSI14={tech.rsi14:.1f}(≥{strategy.rsi_overbought_threshold:.0f}超买区)")
            else:
                reasons.append(
                    f"RSI14={tech.rsi14:.1f}(≥{strategy.rsi_overbought_threshold:.0f})仅作情绪注记，"
                    "ISA价值仓不据此卖飞好公司"
                )

        if tech.dist_ma50_pct is not None and tech.dist_ma50_pct >= 18.0:
            if is_invest:
                trim_score += 1
                reasons.append(f"大幅正乖离MA50({tech.dist_ma50_pct:+.1f}%),短线追高风险大")
            else:
                reasons.append(
                    f"短线大幅正乖离MA50({tech.dist_ma50_pct:+.1f}%)，长线价值仓仅作注记"
                )

        if (
            tech.cyq_benefit_part_pct is not None
            and tech.cyq_benefit_part_pct >= 92.0
            and tech.rsi14 is not None
            and tech.rsi14 >= 70.0
        ):
            if is_invest:
                trim_score += 1
                reasons.append(
                    f"CYQ获利盘达{tech.cyq_benefit_part_pct:.0f}%且RSI={tech.rsi14:.1f},短期筹码拥挤"
                )
            else:
                reasons.append(
                    f"CYQ获利盘达{tech.cyq_benefit_part_pct:.0f}%且RSI={tech.rsi14:.1f}，仅作筹码注记"
                )
        elif (
            tech.cyq_regime == "上方套牢峰"
            and tech.cyq_benefit_part_pct is not None
            and tech.cyq_benefit_part_pct < 30.0
        ):
            reasons.append(f"CYQ上方套牢盘较重(仅{tech.cyq_benefit_part_pct:.0f}%获利),反弹留意抛压")

        if heavy_down_vol:
            if is_invest:
                trim_score += 1
                reasons.append(
                    f"放量下挫(量比{tech.volume_ratio_20d:.2f}, 单日{tech.ret_1d_pct:+.1f}%)"
                )
            else:
                reasons.append(
                    f"放量下挫(量比{tech.volume_ratio_20d:.2f}, 单日{tech.ret_1d_pct:+.1f}%)，"
                    "长线价值仓不据此减仓，须核查基本面是否证伪"
                )

        below_ma200 = tech.dist_ma200_pct is not None and tech.dist_ma200_pct <= 0.0
        mild_ex_div_dip = bool(
            ex_div_guard
            and tech.dist_ma200_pct is not None
            and tech.dist_ma200_pct >= -ex_div_buf
        )
        is_weak_trend = (tech.trend_regime == "WEAK" or (strategy.require_uptrend_for_add and below_ma200)) and not mild_ex_div_dip
        if mild_ex_div_dip and (tech.trend_regime == "WEAK" or below_ma200):
            reasons.append("除息日除权低开轻微跌破均线,暂免假破位判定")
        elif is_weak_trend:
            if is_invest:
                trim_score += 1
                reasons.append("股价处于MA200下方弱势结构,短线严禁盲目摊平")
            else:
                reasons.append("长线ISA处于MA200下方弱势区间(未达右侧趋势门槛),暂不触发加仓信号")

        # Add signals: strictly require uptrend (Price > MA200 when configured) and zero stop/overweight conflict
        uptrend_ok = (
            (tech.dist_ma200_pct > 0.0 if tech.dist_ma200_pct is not None else tech.trend_regime in {"BULL", "PULLBACK"})
            if strategy.require_uptrend_for_add
            else (tech.trend_regime != "WEAK")
        )
        can_add = (
            (not stop_flag)
            and (not isa_deep_dd)
            and (not over_weight)
            and (tech.trend_regime != "WEAK")
            and uptrend_ok
        )
        if can_add:
            near_ma50 = tech.dist_ma50_pct is not None and -4.5 <= tech.dist_ma50_pct <= 2.5
            near_ma200 = tech.dist_ma200_pct is not None and -3.0 <= tech.dist_ma200_pct <= 4.0
            healthy_dd = (
                tech.drawdown_52w_pct is not None
                and strategy.pullback_from_high_add_min_pct <= tech.drawdown_52w_pct <= strategy.pullback_from_high_add_max_pct
            )
            rsi_oversold = tech.rsi14 is not None and tech.rsi14 <= strategy.rsi_oversold_threshold

            if (near_ma50 or near_ma200) and tech.trend_regime in {"BULL", "PULLBACK"}:
                add_score += 2
                reasons.append("位于MA200上方且回踩中长期均线支撑区(MA50/MA200附近)")
            if healthy_dd:
                add_score += 1
                reasons.append(f"自52周高点健康回撤{tech.drawdown_52w_pct:.1f}%")
            if rsi_oversold:
                add_score += 1
                reasons.append(f"RSI14={tech.rsi14:.1f}进入超卖修复区")

            if tech.weekly_bullish is True and (near_ma50 or near_ma200 or healthy_dd):
                add_score += 1
                reasons.append("周线多头共振确认")
            elif tech.weekly_bullish is False:
                if add_score > 0:
                    add_score = max(0, add_score - 1)
                reasons.append("周线尚未转多,暂不激进加仓")

            if tech.adx14 is not None and tech.adx14 >= 25.0 and tech.trend_regime in {"BULL", "PULLBACK"} and add_score > 0:
                reasons.append(f"ADX14={tech.adx14:.1f}趋势强劲")
            if (
                tech.volume_ratio_20d is not None
                and tech.volume_ratio_20d <= 0.85
                and tech.trend_regime == "PULLBACK"
            ):
                reasons.append(f"缩量回踩(量比{tech.volume_ratio_20d:.2f})")
            if tech.cyq_regime in {"底部单峰密集", "单峰密集"} or (
                tech.cyq_benefit_part_pct is not None
                and 60.0 <= tech.cyq_benefit_part_pct <= 88.0
                and tech.trend_regime in {"BULL", "PULLBACK"}
                and add_score > 0
            ):
                reasons.append("CYQ筹码支撑结构良好")

        # v5.5.1 ISA value left-side add (four locks). The old "Price > MA200" gate
        # (`require_uptrend_for_add`) is deliberately waived for quality value names.
        if not is_invest and not is_etf and val_gate is not None:
            add_dd_thresh = float(
                getattr(strategy, "isa_value_add_drawdown_pct", ISA_VALUE_ADD_DRAWDOWN_PCT)
            )
            dd52_ok = bool(
                tech.drawdown_52w_pct is not None and tech.drawdown_52w_pct <= add_dd_thresh
            )
            cost_dd_ok = bool(pos.return_pct <= add_dd_thresh)
            four_locks = bool(
                (not over_weight)
                and (not stop_flag)
                and val_gate.get("quality_pass")
                and val_gate.get("mos_pass")
                and (dd52_ok or cost_dd_ok)
            )
            if four_locks:
                isa_value_add = True
                add_score = max(add_score, 2)
                reasons.append(
                    f"ISA价值左侧加仓候补({val_gate.get('summary')}，距高点/成本回撤达标；"
                    "须联网确认下跌源于市场情绪而非基本面/护城河恶化后分批加仓)"
                )

    # v5.4.0 ISA legacy crumb (grandfathered, < min_isa_equity_order_gbp) — only a *gentle*
    # rebound-window prune hint. Never force-liquidate in panic days or silent pullbacks.
    crumb_prune_candidate = False
    is_isa_legacy_crumb = (
        (not is_invest)
        and (not is_etf)
        and (0 < pos.current_value < strategy.min_isa_equity_order_gbp)
    )
    if is_isa_legacy_crumb and not stop_flag and not earnings_soon and tech is not None:
        rebound_window = bool(
            pos.return_pct >= strategy.isa_crumb_near_breakeven_floor_pct
            and (
                (tech.rsi14 is not None and tech.rsi14 >= 60.0)
                or (tech.drawdown_20d_pct is not None and tech.drawdown_20d_pct >= -2.0)
            )
        )
        if rebound_window:
            crumb_prune_candidate = True
            trim_score += 2
            reasons.append(
                f"历史小额散仓(£{pos.current_value:.0f}<£{strategy.min_isa_equity_order_gbp:.0f})"
                "反弹接近回本/阻力窗，可顺势归拢至核心宽基(零Quota损耗)"
            )

    # Hurdle gate v5.3.2: ISA judged on opportunity cost vs long-term trend, Invest on R:R.
    # Cash interest (4.0%/3.4%) is a parking-yield floor only — never a veto on long-term ISA holdings.
    upside_pct = _compute_analyst_upside_pct(bundle)
    friction_pct = _round_trip_friction_pct(pos, bundle, strategy, wallet)
    cash_floor_pct = _wallet_cash_basis_pct(pos, strategy, wallet)
    rr_ratio: float | None = None
    forward_rr: float | None = None
    if is_invest and effective_stop is not None and float(effective_stop) < 0 and upside_pct is not None:
        risk_pct = abs(float(effective_stop))
        rr_ratio = (upside_pct - friction_pct) / risk_pct if risk_pct > 0 else None
        min_rr, rr_tier = resolve_invest_dynamic_rr_floor(reg_up, tech, strategy)
        if rr_ratio is not None and rr_ratio < min_rr:
            add_score = 0
            reasons.append(
                f"风险回报比仅{rr_ratio:.2f}:1(<{min_rr:.1f}:1 {rr_tier}；目标空间{upside_pct:+.1f}%扣摩擦{friction_pct:.2f}% "
                f"vs 止损{risk_pct:.1f}%，现金停车{cash_floor_pct:.1f}%仅作等待期保底)"
            )
    elif not is_invest and upside_pct is not None:
        weak_long_trend = bool(
            tech is not None
            and tech.dist_ma200_pct is not None
            and tech.dist_ma200_pct < 0.0
            and tech.trend_regime == "WEAK"
        )
        if weak_long_trend and (upside_pct - friction_pct) <= 0.0:
            reasons.append(
                f"长期弱势(MA200下方)+目标空间{upside_pct:+.1f}%扣摩擦{friction_pct:.2f}%后不足,触发机会成本置换复核(不以现金利息否决)"
            )

    # v5.5.1 Invest forward R:R: leftover upside vs remaining downside risk (<0.8:1 = 鸡肋).
    if is_invest:
        forward_rr = compute_invest_forward_rr(
            pos,
            tech,
            bundle,
            strategy,
            effective_stop_pct=effective_stop,
            regime=reg_up,
            wallet=wallet,
        )
        forward_rr_warn = float(
            getattr(strategy, "invest_forward_rr_exit_warn", INVEST_FORWARD_RR_EXIT_WARN)
        )
        if (
            forward_rr is not None
            and forward_rr < forward_rr_warn
            and not stop_flag
            and (
                not holding_above_ma20
                or (tech is not None and tech.ret_20d_pct is not None and tech.ret_20d_pct <= 2.0)
            )
        ):
            trim_score += 1
            reasons.append(
                f"前瞻R:R恶化至{forward_rr:.2f}:1(<{forward_rr_warn:.1f}:1)，剩余赔率不足，"
                "可评估锁利或ROTATE置换高赔率标的"
            )

    sizing = compute_risk_based_size_gbp(
        account=pos.account,
        strategy=strategy,
        effective_stop_pct=effective_stop,
        is_etf=is_etf,
        account_total_value_gbp=inferred_acct_total,
        existing_position_value_gbp=pos.current_value,
    )

    if earnings_soon and bundle and bundle.earnings_date:
        reasons.append(f"⚠️ 距财报仅{bundle.days_to_earnings}天({bundle.earnings_date}),财报前谨慎重仓加码")
        if add_score > 0:
            add_score = min(max(0, add_score - 1), 1)

    if stop_flag or over_weight or (isa_deep_dd and not isa_value_add) or trim_score >= 2:
        add_score = 0

    if stop_flag:
        bias = "REVIEW_STOP"
    elif trim_score >= 2:
        bias = "TRIM_ALERT"
    elif add_score >= 2:
        bias = "ADD_CANDIDATE"
    else:
        bias = "HOLD"

    if not reasons:
        reasons.append("技术面与仓位均在常规区间,默认持有观望")

    return HoldingEvaluation(
        account=pos.account,
        ticker=pos.ticker,
        yahoo_symbol=pos.yahoo_symbol,
        weight_pct=pos.weight_pct,
        return_pct=pos.return_pct,
        action_bias=bias,
        reasons=reasons,
        friction_note=_friction_hint(pos, bundle, strategy, wallet),
        native_return_pct=round(native_ret, 2) if native_ret is not None else None,
        effective_stop_pct=effective_stop,
        pending_buy_weight_pct=round(pending_buy_weight_pct, 2),
        has_pending_stop=has_pending_stop,
        ex_div_guarded=ex_div_guard,
        rr_ratio=round(rr_ratio, 2) if rr_ratio is not None else None,
        suggested_size_gbp=sizing["suggested_gbp"],
        min_order_gbp=sizing["min_order_gbp"],
        is_crumb_prune_candidate=crumb_prune_candidate,
        philosophy_tag=philosophy_tag,
        value_mos_pct=round(float(value_mos_pct), 2) if value_mos_pct is not None else None,
        forward_rr=forward_rr,
    )


def _resolve_total_portfolio_gbp(data: PortfolioData) -> float:
    if data.total_combined_value > 0:
        return float(data.total_combined_value)
    acct_sum = float(data.isa.total_account_value or 0.0) + float(data.invest.total_account_value or 0.0)
    if acct_sum > 0:
        return acct_sum
    pos_sum = sum(float(p.current_value) for p in data.all_positions())
    return max(pos_sum, 1.0)


def evaluate_portfolio(
    data: PortfolioData,
    tech_map: dict[str, TechnicalSnapshot],
    bundle_map: dict[str, TickerMarketBundle],
    strategy: StrategySection,
    *,
    histories: dict[str, pd.DataFrame] | None = None,
    regime: str = "NEUTRAL",
    macro: MacroSnapshot | None = None,
) -> list[HoldingEvaluation]:
    """Evaluate all holdings across ISA & Invest with cross-account weight & pending-order context."""
    total_all = _resolve_total_portfolio_gbp(data)
    issuer_vals: dict[str, float] = {}
    for pos in data.all_positions():
        root = normalize_issuer_root(pos.ticker)
        issuer_vals[root] = issuer_vals.get(root, 0.0) + pos.current_value
    issuer_weights = {root: (val / total_all) * 100.0 for root, val in issuer_vals.items()}
    all_pending = data.all_pending_orders()
    account_totals = {
        "ISA": data.isa.total_account_value or sum(p.current_value for p in data.isa.positions),
        "INVEST": data.invest.total_account_value or sum(p.current_value for p in data.invest.positions),
    }

    evals = [
        evaluate_holding(
            pos,
            tech_map.get(pos.yahoo_symbol),
            bundle_map.get(pos.yahoo_symbol),
            strategy,
            total_issuer_weight_pct=issuer_weights.get(normalize_issuer_root(pos.ticker)),
            hist=histories.get(pos.yahoo_symbol) if histories else None,
            pending_orders=all_pending,
            account_total_value=account_totals.get(pos.account.upper()),
            total_portfolio_value=total_all,
            wallet=data.invest,
            regime=regime,
            macro=macro,
        )
        for pos in data.all_positions()
    ]

    # v5.4.0: at most `isa_crumb_pruning_max_per_report` crumb prune hints per report.
    # Extras are downgraded to HOLD — legacy crumbs are never force-liquidated en masse.
    pos_value_by_sym = {p.yahoo_symbol.upper(): p.current_value for p in data.all_positions()}
    crumb_alerts = [
        ev
        for ev in evals
        if ev.is_crumb_prune_candidate and ev.action_bias == "TRIM_ALERT"
    ]
    if len(crumb_alerts) > strategy.isa_crumb_pruning_max_per_report:
        crumb_alerts.sort(
            key=lambda ev: (
                -ev.return_pct,
                -(pos_value_by_sym.get(ev.yahoo_symbol.upper(), 0.0) or 0.0),
            ),
            reverse=True,
        )
        keep_ysyms = {ev.yahoo_symbol.upper() for ev in crumb_alerts[: strategy.isa_crumb_pruning_max_per_report]}
        evals = [
            (
                ev.model_copy(
                    update={
                        "action_bias": "HOLD",
                        "reasons": ["存量小额散仓静待反弹窗口分批归核，不低位急割"],
                        "is_crumb_prune_candidate": False,
                    }
                )
                if (ev.is_crumb_prune_candidate and ev.yahoo_symbol.upper() not in keep_ysyms)
                else ev
            )
            for ev in evals
        ]
    return evals


_CONFLICT_HINTS: dict[str, str] = {
    "STOP_VS_EX_DIV": "优先核实除权缺口，避免除息日误砍仓",
    "BULL_VS_OVERWEIGHT": "趋势向好但仓位触及上限，严禁加码并逢高控仓",
    "BULL_VS_EARNINGS_RISK": "多头结构遇财报窗口，暂停主动加仓等待业绩落地",
    "TREND_VS_CROWDED": "趋势多头但短线超买/获利盘拥挤，勿追高可分批兑现",
    "UPSIDE_VS_CASH_HURDLE": "技术形态尚可但盈亏比不足 2:1（目标空间扣摩擦后偏薄），不建议增配",
    "FX_DISTORTION": "账面受汇率波动干扰明显，按本币趋势与基本面单独核验",
    "SUPPORT_VS_WEAK_TREND": "估值/超卖有修复空间但右侧尚未转强，等待均线企稳",
    "MIXED_SIGNALS": "多空证据并存，PM须结合摩擦成本后独立裁决",
}


def build_disagreement_matrix(
    data: PortfolioData,
    tech_map: dict[str, TechnicalSnapshot],
    bundle_map: dict[str, TickerMarketBundle],
    strategy: StrategySection,
    evals: list[HoldingEvaluation] | None = None,
) -> list[DisagreementItem]:
    """Build deterministic 0-Token bull-vs-bear disagreement rows for holdings with conflicting signals."""
    total_all = _resolve_total_portfolio_gbp(data)
    issuer_vals: dict[str, float] = {}
    for pos in data.all_positions():
        root = normalize_issuer_root(pos.ticker)
        issuer_vals[root] = issuer_vals.get(root, 0.0) + pos.current_value
    issuer_weights = {root: (val / total_all) * 100.0 for root, val in issuer_vals.items()}
    eval_map = {(e.account.upper(), e.yahoo_symbol): e for e in (evals or [])}

    items: list[DisagreementItem] = []
    for pos in data.all_positions():
        tech = tech_map.get(pos.yahoo_symbol)
        bundle = bundle_map.get(pos.yahoo_symbol)
        ev = eval_map.get((pos.account.upper(), pos.yahoo_symbol))
        is_invest = pos.account.upper() == "INVEST"
        is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
        max_acct = strategy.max_invest_single_weight_pct if is_invest else strategy.max_isa_single_weight_pct
        issuer_w = issuer_weights.get(normalize_issuer_root(pos.ticker), pos.weight_pct)
        ex_div_guard = _is_ex_div_guard(tech, bundle)
        earnings_soon = bool(
            bundle and bundle.days_to_earnings is not None and 0 <= bundle.days_to_earnings <= 5
        )
        native_ret = _native_return_pct(pos)
        upside_pct = _compute_analyst_upside_pct(bundle)
        friction_pct = _round_trip_friction_pct(pos, bundle, strategy, data.invest)
        eff_stop = (ev.effective_stop_pct if ev else None) or _compute_adaptive_invest_stop_pct(tech, strategy)

        bull_points: list[str] = []
        bear_points: list[str] = []
        conflict_tags: set[str] = set()

        if tech is not None:
            if tech.trend_regime == "BULL":
                bull_points.append("日线多头排列" + ("+周线共振" if tech.weekly_bullish is True else ""))
            elif tech.trend_regime == "PULLBACK" and tech.dist_ma200_pct is not None and tech.dist_ma200_pct > 0:
                bull_points.append("MA200上方健康回踩")

            if tech.rsi14 is not None and tech.rsi14 <= strategy.rsi_oversold_threshold:
                bull_points.append(f"RSI14={tech.rsi14:.1f}超卖")
            if tech.cyq_regime in {"底部单峰密集", "单峰密集"}:
                bull_points.append(f"CYQ{tech.cyq_regime}支撑")
            elif tech.cyq_benefit_part_pct is not None and 60.0 <= tech.cyq_benefit_part_pct <= 88.0 and tech.trend_regime != "WEAK":
                bull_points.append(f"CYQ获利盘{tech.cyq_benefit_part_pct:.0f}%良性")

            if tech.rsi14 is not None and tech.rsi14 >= strategy.rsi_overbought_threshold:
                bear_points.append(f"RSI14={tech.rsi14:.1f}超买")
                conflict_tags.add("TREND_VS_CROWDED")
            if tech.dist_ma50_pct is not None and tech.dist_ma50_pct >= 18.0:
                bear_points.append(f"乖离MA50达{tech.dist_ma50_pct:+.1f}%")
                conflict_tags.add("TREND_VS_CROWDED")
            if tech.cyq_benefit_part_pct is not None and tech.cyq_benefit_part_pct >= 92.0:
                bear_points.append(f"获利盘{tech.cyq_benefit_part_pct:.0f}%拥挤")
                conflict_tags.add("TREND_VS_CROWDED")
            elif tech.cyq_regime == "上方套牢峰":
                bear_points.append("CYQ上方套牢阻力")
                conflict_tags.add("SUPPORT_VS_WEAK_TREND")
            if tech.trend_regime == "WEAK" or tech.weekly_bullish is False:
                bear_points.append("日/周线趋势偏弱")
                conflict_tags.add("SUPPORT_VS_WEAK_TREND")

        if upside_pct is not None:
            if is_invest and eff_stop is not None and float(eff_stop) < 0:
                rr = (upside_pct - friction_pct) / abs(float(eff_stop))
                if rr >= 2.0:
                    bull_points.append(f"盈亏比{rr:.1f}:1(目标空间{upside_pct:+.1f}%)")
                else:
                    bear_points.append(
                        f"盈亏比仅{rr:.1f}:1<2:1(目标空间{upside_pct:+.1f}%扣摩擦{friction_pct:.2f}%)"
                    )
                    conflict_tags.add("UPSIDE_VS_CASH_HURDLE")
            elif upside_pct >= friction_pct + 10.0:
                bull_points.append(f"目标空间{upside_pct:+.1f}%(扣摩擦{friction_pct:.2f}%仍厚)")
            elif upside_pct < friction_pct:
                bear_points.append(
                    f"目标空间{upside_pct:+.1f}%<往返摩擦{friction_pct:.2f}%(机会成本置换复核)"
                )
                conflict_tags.add("UPSIDE_VS_CASH_HURDLE")

        if not is_etf and (pos.weight_pct >= max_acct or issuer_w >= strategy.max_total_portfolio_weight_pct):
            bear_points.append(f"仓位触红线(单户{pos.weight_pct:.1f}%/合并{issuer_w:.1f}%)")
            conflict_tags.add("BULL_VS_OVERWEIGHT")

        if earnings_soon and bundle is not None:
            bear_points.append(f"{bundle.days_to_earnings}日内财报事件风险")
            conflict_tags.add("BULL_VS_EARNINGS_RISK")

        if is_invest and pos.return_pct <= eff_stop:
            bear_points.append(f"浮亏{pos.return_pct:.1f}%触止损线{eff_stop:.1f}%")
        elif (not is_invest) and pos.return_pct <= strategy.isa_deep_drawdown_review_pct:
            bear_points.append(f"ISA浮亏{pos.return_pct:.1f}%达深套复核线")

        if ex_div_guard:
            bull_points.append("除息窗口除权低开干扰")
            if bear_points:
                conflict_tags.add("STOP_VS_EX_DIV")

        if native_ret is not None and pos.inst_currency.upper() not in {"GBP", "GBX"}:
            if abs(pos.return_pct - native_ret) >= 5.0:
                note = f"汇率干扰(本币{native_ret:+.1f}% vs 英镑{pos.return_pct:+.1f}%)"
                if native_ret > pos.return_pct:
                    bull_points.append(note)
                else:
                    bear_points.append(note)
                conflict_tags.add("FX_DISTORTION")

        if not (bull_points and bear_points):
            continue

        priority = [
            "STOP_VS_EX_DIV",
            "BULL_VS_OVERWEIGHT",
            "BULL_VS_EARNINGS_RISK",
            "TREND_VS_CROWDED",
            "UPSIDE_VS_CASH_HURDLE",
            "FX_DISTORTION",
            "SUPPORT_VS_WEAK_TREND",
        ]
        conflict_type = next((tag for tag in priority if tag in conflict_tags), "MIXED_SIGNALS")
        items.append(
            DisagreementItem(
                account=pos.account,
                ticker=pos.ticker,
                yahoo_symbol=pos.yahoo_symbol,
                conflict_type=conflict_type,
                bull_points=bull_points[:3],
                bear_points=bear_points[:3],
                decision_hint=_CONFLICT_HINTS.get(conflict_type, _CONFLICT_HINTS["MIXED_SIGNALS"]),
            )
        )
    return items


def format_disagreement_matrix_text(items: list[DisagreementItem]) -> str:
    """Format disagreement items into compact plain-text rows for PM/ContextPack."""
    if not items:
        return "（当前持仓信号基本一致，无显著多空证据冲突）"
    high_priority = [it for it in items if it.conflict_type != "SUPPORT_VS_WEAK_TREND"]
    weak_items = [it for it in items if it.conflict_type == "SUPPORT_VS_WEAK_TREND"]
    lines = []
    for it in high_priority:
        bull = " / ".join(it.bull_points)
        bear = " / ".join(it.bear_points)
        lines.append(
            f"• [{it.account}] {it.ticker} ({it.conflict_type}) | 多方：{bull} ⚔️ 空方/风险：{bear} → 提示：{it.decision_hint}"
        )
    if weak_items:
        tickers_str = ", ".join(f"{it.ticker}[{it.account}]" for it in weak_items)
        lines.append(
            f"• [SUPPORT_VS_WEAK_TREND 合并 x{len(weak_items)}] {tickers_str} | "
            f"多方：估值/超卖或目标价有修复空间 ⚔️ 空方/风险：日/周线趋势偏弱或上方套牢 → 提示：{_CONFLICT_HINTS['SUPPORT_VS_WEAK_TREND']}"
        )
    return "\n".join(lines)


_VETO_BLOCK_KEYWORDS = ("禁止", "严禁", "勿", "不买", "不加", "暂停", "回避", "拦截", "冻结", "VETO", "NO ")
_NON_TICKER_TOKENS = {
    "ISA", "INVEST", "GBP", "GBX", "USD", "EUR", "ETF", "LSE", "NYSE", "VIX",
    "RSI", "ATR", "CYQ", "ADX", "MA20", "MA50", "MA200", "BUY", "ADD", "HOLD",
    "TRIM", "STOP", "KEEP", "MODIFY", "CANCEL", "CASH", "UK", "US", "EU", "PM",
}


def _symbol_matches(a: str, b: str) -> bool:
    sa = a.strip().upper()
    sb = b.strip().upper()
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    return sa.split(".")[0] == sb.split(".")[0]


def _extract_vetoed_symbols(daily_vetoes: list[str] | None) -> dict[str, str]:
    """Extract explicitly vetoed ticker symbols from daily_vetoes text lines."""
    vetoed: dict[str, str] = {}
    for line in daily_vetoes or []:
        text = str(line or "").strip()
        if not text:
            continue
        upper = text.upper()
        if not any(kw in text or kw in upper for kw in _VETO_BLOCK_KEYWORDS):
            continue
        for tok in re.findall(r"(?<![A-Z0-9.])([A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?)(?![A-Z0-9.])", upper):
            base = tok.split(".")[0]
            if tok in _NON_TICKER_TOKENS or base in _NON_TICKER_TOKENS or base.isdigit():
                continue
            vetoed.setdefault(tok, text)
            vetoed.setdefault(base, text)
    return vetoed


def _build_report_action_blocks(
    report_actions: list[Any] | None,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Index same-day /report TRIM/STOP directives by (ACCOUNT, SYMBOL_OR_ROOT) for hard reversal blocking."""
    blocks: dict[tuple[str, str], tuple[str, str]] = {}
    for item in report_actions or []:
        if isinstance(item, dict):
            raw_acct = str(item.get("account") or "").strip().upper()
            raw_sym = str(item.get("symbol") or "").strip().upper()
            raw_act = str(item.get("action") or "").strip().upper()
            raw_reason = str(item.get("reason") or "").strip()
        else:
            raw_acct = str(getattr(item, "account", "") or "").strip().upper()
            raw_sym = str(getattr(item, "symbol", "") or "").strip().upper()
            raw_act = str(getattr(item, "action", "") or "").strip().upper()
            raw_reason = str(getattr(item, "reason", "") or "").strip()

        if raw_act not in {"TRIM", "STOP"} or not raw_sym or raw_sym == "CASH":
            continue

        if raw_acct in {"双账户", "BOTH", "ALL", ""} or ("ISA" in raw_acct and "INVEST" in raw_acct):
            accts = ("ISA", "INVEST")
        elif "INVEST" in raw_acct:
            accts = ("INVEST",)
        elif "ISA" in raw_acct:
            accts = ("ISA",)
        else:
            accts = (raw_acct,)

        base = raw_sym.split(".")[0]
        root = normalize_issuer_root(base)
        keys = {k for k in (raw_sym, base, root) if k}
        for acct in accts:
            for key in keys:
                blocks.setdefault((acct, key), (raw_act, raw_reason))
    return blocks


def _cadence_guard_mode(strategy: StrategySection) -> str:
    """Resolve the cadence guard strength (off / advisory / hard)."""
    mode = str(getattr(strategy, "trade_cadence_guard_mode", "advisory") or "advisory").strip().lower()
    return mode if mode in {"off", "advisory", "hard"} else "advisory"


def _cadence_violation(
    behavior: dict[str, Any],
    sym_up: str,
    base: str,
    strategy: StrategySection,
    is_etf: bool = False,
) -> tuple[str, str] | None:
    """Detect a cadence/cooldown breach: weekly/monthly BUY caps or the re-buy cooldown.

    Returns ``None`` whenever the fetched order history is incomplete (``partial``) so that a
    short history window can never silently freeze all new deployment. This function only
    *detects* the breach — whether it blocks or merely advises is decided by
    ``trade_cadence_guard_mode``.

    v5.4.0: core wide-base ETF buys (VWRP.L etc.) are 100% quota-exempt — buying the
    benchmark or clearing legacy crumbs never consumes any individual-equity quota.
    """
    if is_etf:
        return None
    if not behavior or behavior.get("partial"):
        return None
    weekly_cap = int(getattr(strategy, "max_weekly_new_buys", 0) or 0)
    monthly_cap = int(getattr(strategy, "max_monthly_new_buys", 0) or 0)
    if weekly_cap > 0 and int(behavior.get("buys_7d") or 0) >= weekly_cap:
        return (
            "FREQUENCY_CAP",
            f"近 7 天已新开仓 {int(behavior.get('buys_7d') or 0)} 笔(≥上限{weekly_cap})，本周额度已用完",
        )
    if monthly_cap > 0 and int(behavior.get("buys_30d") or 0) >= monthly_cap:
        return (
            "FREQUENCY_CAP",
            f"近 30 天已新开仓 {int(behavior.get('buys_30d') or 0)} 笔(≥上限{monthly_cap})，本月额度已用完",
        )
    base_root = normalize_issuer_root(base)
    for row in behavior.get("recently_closed") or []:
        if not row.get("in_cooldown"):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if ticker == sym_up or ticker == base or normalize_issuer_root(ticker) == base_root:
            return (
                "REBUY_COOLDOWN",
                f"{ticker} 于 {float(row.get('days_since_sell') or 0):.0f} 天前清仓，处于 "
                f"{int(behavior.get('cooldown_days') or 0)} 天再买冷却期",
            )
    return None


def apply_risk_override_guardrail(
    decision: Any,
    *,
    portfolio: PortfolioData | None = None,
    macro: MacroSnapshot | None = None,
    strategy: StrategySection | None = None,
    bundle_map: dict[str, TickerMarketBundle] | None = None,
    evals: list[HoldingEvaluation] | None = None,
    high_correlation_symbols: set[str] | None = None,
    daily_vetoes: list[str] | None = None,
    report_actions: list[Any] | None = None,
    trade_behavior: dict[str, Any] | None = None,
    vix_danger_threshold: float = DEFAULT_VIX_DANGER_THRESHOLD,
    deployment_gate: dict[str, Any] | None = None,
    risk_budget_scale: float = 1.0,
    tech_map: dict[str, TechnicalSnapshot] | None = None,
) -> RiskOverrideResult:
    """Apply deterministic post-LLM risk override guardrails to MorningReportDecision or PickDecision."""
    if decision is None:
        return RiskOverrideResult(decision=None, applied=False)

    strat = strategy or StrategySection()
    bmap = bundle_map or {}
    macro_vix = macro.vix if (macro is not None and macro.vix is not None) else None
    macro_high_risk = macro_vix is not None and macro_vix >= vix_danger_threshold
    corr_set = {s.strip().upper() for s in (high_correlation_symbols or set()) if s and s.strip()}
    corr_bases = {s.split(".")[0] for s in corr_set}
    regime = str((deployment_gate or {}).get("regime") or "NEUTRAL").strip().upper()
    tech_lookup: dict[str, TechnicalSnapshot] = {}
    for t_sym, t_snap in (tech_map or {}).items():
        if not t_sym or t_snap is None:
            continue
        tech_lookup[t_sym.strip().upper()] = t_snap
        tech_lookup.setdefault(t_sym.strip().upper().split(".")[0], t_snap)

    merged_vetoes = list(daily_vetoes or [])
    if hasattr(decision, "daily_vetoes") and getattr(decision, "daily_vetoes"):
        merged_vetoes.extend(getattr(decision, "daily_vetoes"))
    vetoed_map = _extract_vetoed_symbols(merged_vetoes)
    report_action_blocks = _build_report_action_blocks(report_actions)

    # Precompute current + pending BUY weights by issuer root and by (account, symbol)
    issuer_weights: dict[str, float] = {}
    acct_weights: dict[tuple[str, str], float] = {}
    etf_roots: set[str] = set()
    account_totals: dict[str, float] = {}
    if portfolio is not None:
        total_all = _resolve_total_portfolio_gbp(portfolio)
        pos_by_sym = {p.yahoo_symbol: p for p in portfolio.all_positions()}
        account_totals = {
            "ISA": portfolio.isa.total_account_value or sum(p.current_value for p in portfolio.isa.positions),
            "INVEST": portfolio.invest.total_account_value or sum(p.current_value for p in portfolio.invest.positions),
        }
        issuer_vals: dict[str, float] = {}
        for p in portfolio.all_positions():
            root = normalize_issuer_root(p.ticker)
            issuer_vals[root] = issuer_vals.get(root, 0.0) + p.current_value
            acct_weights[(p.account.upper(), p.yahoo_symbol.upper())] = p.weight_pct
            acct_weights[(p.account.upper(), p.ticker.upper())] = p.weight_pct
            b = bmap.get(p.yahoo_symbol)
            if b and b.quote_type.upper() == "ETF":
                etf_roots.add(root)
        for o in portfolio.all_pending_orders():
            if o.side.upper() != "BUY":
                continue
            gbp_val = _estimate_pending_order_gbp(o, pos_by_symbol=pos_by_sym)
            if gbp_val <= 0:
                continue
            root = normalize_issuer_root(o.ticker)
            issuer_vals[root] = issuer_vals.get(root, 0.0) + gbp_val
            acct_total = account_totals.get(o.account.upper(), 0.0)
            if acct_total > 0:
                add_w = (gbp_val / acct_total) * 100.0
                for key_sym in {o.yahoo_symbol.upper(), o.ticker.upper()}:
                    if key_sym:
                        acct_weights[(o.account.upper(), key_sym)] = (
                            acct_weights.get((o.account.upper(), key_sym), 0.0) + add_w
                        )
        issuer_weights = {r: (v / total_all) * 100.0 for r, v in issuer_vals.items()}

    # Index bundles case-insensitively and by base symbol; also seed known UCITS ETF roots
    etf_roots.update(normalize_issuer_root(s) for s in _ETF_TECH_PENETRATION_PCT)
    core_etf_syms: set[str] = {str(strat.benchmark_symbol).strip().upper()}
    if isinstance(deployment_gate, dict):
        for etf_sym in deployment_gate.get("etf_set") or []:
            if etf_sym:
                etf_roots.add(normalize_issuer_root(str(etf_sym)))
                core_etf_syms.add(str(etf_sym).strip().upper())
    bundle_lookup: dict[str, TickerMarketBundle] = {}
    for sym, b in bmap.items():
        bundle_lookup[sym.upper()] = b
        bundle_lookup.setdefault(sym.upper().split(".")[0], b)
        if b.quote_type.upper() == "ETF":
            etf_roots.add(normalize_issuer_root(sym))

    # v5.4.0 zero-quota exemption: recompute equity-only BUY counts from raw order history so
    # that core-ETF buys (VWRP.L etc.) and SELL-side cleanups never consume quota. When
    # portfolio is None the caller-supplied trade_behavior is read verbatim (legacy contract).
    effective_behavior = dict(trade_behavior) if trade_behavior else {}
    benchmark_root = normalize_issuer_root(str(strat.benchmark_symbol))
    held_isa_syms: set[str] = set()
    held_isa_bases: set[str] = set()
    if portfolio is not None:
        for p in portfolio.isa.positions:
            held_isa_syms.add(p.yahoo_symbol.upper())
            held_isa_bases.add(p.yahoo_symbol.upper().split(".")[0])
        if effective_behavior:
            recent_all = [*portfolio.isa.recent_orders, *portfolio.invest.recent_orders]
            if recent_all:
                anchor_ms = max(
                    int(getattr(portfolio, "fetched_at_ms", 0) or 0),
                    max(o.filled_at_ms for o in recent_all),
                )

                def _order_is_etf(o: Any) -> bool:
                    root = normalize_issuer_root(str(getattr(o, "ticker", "") or ""))
                    sym = str(getattr(o, "yahoo_symbol", "") or "").upper()
                    if root in etf_roots or root == benchmark_root:
                        return True
                    if sym and (sym in etf_roots or normalize_issuer_root(sym) in etf_roots):
                        return True
                    return False

                equity_buys = [
                    o
                    for o in recent_all
                    if str(getattr(o, "side", "") or "").upper() == "BUY" and not _order_is_etf(o)
                ]
                effective_behavior["buys_7d"] = sum(
                    1 for o in equity_buys if anchor_ms - o.filled_at_ms <= 7 * 86_400_000
                )
                effective_behavior["buys_30d"] = sum(
                    1 for o in equity_buys if anchor_ms - o.filled_at_ms <= 30 * 86_400_000
                )

    stop_alerts: set[tuple[str, str]] = {
        (e.account.upper(), e.yahoo_symbol.upper())
        for e in (evals or [])
        if e.action_bias == "REVIEW_STOP"
    }
    cadence_advisories: list[str] = []

    def _core_dca_guard(
        sym_up: str,
        base: str,
        acct_up: str,
        bundle: TickerMarketBundle | None,
        guard_mode: str,
    ) -> tuple[str, str] | None:
        """v5.5.1 deterministic DCA guard replacing the old blank R:R exemption."""
        if guard_mode != "hard":
            return None
        isa_gate = (
            deployment_gate.get("isa")
            if isinstance(deployment_gate, dict) and isinstance(deployment_gate.get("isa"), dict)
            else {}
        )
        policy = isa_gate.get("core_policy") if isinstance(isa_gate.get("core_policy"), dict) else {}
        bucket_map = policy.get("bucket_by_symbol") if isinstance(policy.get("bucket_by_symbol"), dict) else {}
        bucket = bucket_map.get(sym_up) or bucket_map.get(base)
        if not bucket:
            family = infer_etf_index_family(sym_up, bundle)
            bucket = {
                "Global All-World/Developed (全球宽基)": "GLOBAL_ANCHOR",
                "S&P 500 (美股标普500)": "US_TILT",
                "Europe Equity (欧洲宽基)": "EU_DIVERSIFIER",
                "FTSE 100 (英国富时100)": "UK_DIVERSIFIER",
            }.get(family or "")
        committed_core_pct = float(
            policy.get("committed_core_pct")
            or isa_gate.get("committed_core_etf_pct")
            or 0.0
        )
        if committed_core_pct >= 65.0:
            return "ACCOUNT_OVERWEIGHT", "ISA 核心宽基已达65%硬上限，暂停新增宽基定投"
        buckets = policy.get("buckets") if isinstance(policy.get("buckets"), dict) else {}
        sleeve = buckets.get(bucket) if bucket else None
        if bucket == "US_TILT" and policy.get("us_tilt_buy_frozen"):
            return (
                "ACCOUNT_OVERWEIGHT",
                "美股与标普500敞口已超上限，禁止继续加码标普500宽基",
            )
        if isinstance(sleeve, dict):
            cap = float(sleeve.get("cap_pct") or 0.0)
            pct_now = float(sleeve.get("committed_pct") or 0.0)
            if cap > 0 and pct_now >= cap:
                return (
                    "ACCOUNT_OVERWEIGHT",
                    f"该核心子桶已达 {cap:.0f}% 硬上限，暂停继续加仓",
                )
        if acct_up == "ISA" and portfolio is not None and portfolio.isa.total_account_value > 0:
            isa_val = float(portfolio.isa.total_account_value)
            floor_gbp = (
                isa_val * float(getattr(strat, "isa_cash_band_low_pct", 5.0) or 0.0) / 100.0
                + float(getattr(strat, "min_isa_etf_order_gbp", 400.0) or 0.0)
            )
            if float(portfolio.isa.available_to_trade) < floor_gbp:
                return (
                    "ACCOUNT_OVERWEIGHT",
                    "ISA 可用现金逼近 5% 结构地板，暂停宽基定投",
                )
        if bucket in {"EU_DIVERSIFIER", "UK_DIVERSIFIER", "US_TILT"}:
            plan = (
                isa_gate.get("core_allocation_plan")
                if isinstance(isa_gate.get("core_allocation_plan"), dict)
                else {}
            )
            metrics_map = plan.get("symbol_metrics") if isinstance(plan.get("symbol_metrics"), dict) else {}
            metrics = metrics_map.get(sym_up) or metrics_map.get(base) or {}
            price = metrics.get("price")
            ma200 = metrics.get("ma200")
            dd52 = metrics.get("drawdown_52w_pct")
            rsi = metrics.get("rsi14")
            if (
                price is not None
                and ma200 is not None
                and float(price) < float(ma200) * 0.95
                and (dd52 is None or float(dd52) > -15.0)
            ):
                return (
                    "ACCOUNT_OVERWEIGHT",
                    "跌破MA200逾5%且未达-15%深跌阶梯，暂缓左侧接刀",
                )
            if rsi is not None and float(rsi) >= 72.0:
                return "ACCOUNT_OVERWEIGHT", "RSI14≥72 追高，暂停宽基买入"
        return None

    def _check_symbol_violation(
        account: str,
        symbol: str,
        action: str,
        item: Any | None = None,
        *,
        is_pick: bool = False,
    ) -> tuple[str, str] | None:
        sym_up = symbol.strip().upper()
        base = sym_up.split(".")[0]
        acct_up = account.strip().upper()
        act_up = action.strip().upper()
        if act_up not in {"BUY", "ADD", "SWAP"} or sym_up == "CASH":
            return None

        if sym_up in vetoed_map or base in vetoed_map:
            return "DAILY_VETO", f"违背今日早报红线({vetoed_map.get(sym_up) or vetoed_map.get(base)})"

        if report_action_blocks:
            root = normalize_issuer_root(base)
            hit = (
                report_action_blocks.get((acct_up, sym_up))
                or report_action_blocks.get((acct_up, base))
                or report_action_blocks.get((acct_up, root))
            )
            if hit is not None:
                rep_act, rep_reason = hit
                reason_tail = f"：{rep_reason}" if rep_reason else ""
                return "REPORT_ACTION_CONFLICT", f"今日早报已对该账户持仓裁决 {rep_act}{reason_tail}，禁止逆势买入/加仓"

        if sym_up in corr_set or base in corr_bases:
            return "CORRELATION_OVERLAP", f"与现有持仓相关性触发≥{strat.overlap_correlation_threshold:.2f}重叠拦截"

        if act_up in {"BUY", "ADD"}:
            root = normalize_issuer_root(base)
            b = bundle_lookup.get(sym_up) or bundle_lookup.get(base)
            benchmark_up = str(strat.benchmark_symbol).strip().upper()
            is_etf = bool(
                (b is not None and b.quote_type.upper() == "ETF")
                or (root in etf_roots)
                or (root == benchmark_root)
                or sym_up == benchmark_up
                or base == benchmark_up
            )
            # v5.4.0: core wide-base ETF buys are 100% quota-exempt (zero Quota).
            cadence_breach = _cadence_violation(effective_behavior, sym_up, base, strat, is_etf=is_etf)
            if cadence_breach is not None:
                guard_mode = _cadence_guard_mode(strat)
                if guard_mode == "hard":
                    return cadence_breach
                if guard_mode == "advisory":
                    _trig, _why = cadence_breach
                    note = (
                        f"[⚠️纪律提醒] [{account}] {symbol}: {_why}；"
                        "advisory 模式不拦截，若仍执行请在理由中写明催化/避险依据"
                    )
                    if note not in cadence_advisories:
                        cadence_advisories.append(note)

            if not is_etf:
                issuer_w = issuer_weights.get(root, 0.0)
                # v5.4.0 hard single-name cap across both accounts (12% of the whole portfolio).
                if issuer_w >= strat.max_single_name_portfolio_pct:
                    return (
                        "ISSUER_OVERWEIGHT",
                        f"同发行人跨账户占全组合{issuer_w:.1f}%(≥单票硬上限{strat.max_single_name_portfolio_pct:.0f}%)",
                    )
                if issuer_w >= strat.max_total_portfolio_weight_pct:
                    return (
                        "ISSUER_OVERWEIGHT",
                        f"同发行人跨账户占全组合{issuer_w:.1f}%(≥上限{strat.max_total_portfolio_weight_pct:.0f}%)",
                    )
                acct_w = acct_weights.get((acct_up, sym_up)) or acct_weights.get((acct_up, base), 0.0)
                acct_cap = strat.max_invest_single_weight_pct if acct_up == "INVEST" else strat.max_isa_single_weight_pct
                if acct_w >= acct_cap:
                    return (
                        "ACCOUNT_OVERWEIGHT",
                        f"占{account}单仓达{acct_w:.1f}%(≥上限{acct_cap:.0f}%)",
                    )
                # v5.4.0: ISA satellite layer saturated (holdings + pending equity buys over
                # 30% / 8 names) — block NEW ISA equity buys, wide-base ETFs stay allowed.
                if (
                    is_pick
                    and acct_up == "ISA"
                    and sym_up not in held_isa_syms
                    and base not in held_isa_bases
                    and isinstance(deployment_gate, dict)
                    and isinstance(deployment_gate.get("isa"), dict)
                    and deployment_gate["isa"].get("isa_satellite_saturated")
                ):
                    return (
                        "ISA_SATELLITE_SATURATED",
                        "ISA 个股卫星层已饱和，暂停新开个股，请归拢核心宽基",
                    )

            if (acct_up, sym_up) in stop_alerts or any(
                k[0] == acct_up and _symbol_matches(k[1], sym_up) for k in stop_alerts
            ):
                return "ACCOUNT_OVERWEIGHT", "标的已触发止损复核红线，严禁逆势加仓"

            if b and b.days_to_earnings is not None and 0 <= b.days_to_earnings <= 5:
                return "EARNINGS_WINDOW", f"距财报仅{b.days_to_earnings}天，禁止财报前主动加仓"

            # v5.3.4: BUY-side ex-dividend window + Python-recomputed R:R floor (pick only).
            guard_mode = _pick_guard_mode(strat)
            if is_pick and guard_mode != "off":
                wallet = (
                    portfolio.invest
                    if (portfolio is not None and acct_up == "INVEST")
                    else None
                )
                pick_core_etf = bool(
                    sym_up in core_etf_syms
                    or base in core_etf_syms
                    or normalize_issuer_root(sym_up)
                    in {normalize_issuer_root(s) for s in core_etf_syms}
                )
                if not pick_core_etf:
                    family = infer_etf_index_family(sym_up, b)
                    pick_core_etf = family in _CORE_WIDE_BASE_FAMILIES
                if pick_core_etf:
                    dca_breach = _core_dca_guard(sym_up, base, acct_up, b, guard_mode)
                    if dca_breach is not None:
                        return dca_breach
                ex_window = int(getattr(strat, "ex_div_entry_guard_days", 3) or 0)
                ex_days = int(b.days_to_ex_dividend) if (b and b.days_to_ex_dividend is not None) else None
                if ex_days is not None and ex_window > 0 and 0 < ex_days <= ex_window:
                    ex_msg = (
                        f"距除息仅{ex_days}天({b.ex_dividend_date or '日期待核'})，"
                        f"除权低开≈{_ex_div_yield_hint(b)}；"
                        "ISA 股息免税可抢息，Invest 应税须按股息税后净额核 R:R"
                    )
                    if guard_mode == "hard" and not pick_core_etf:
                        return "EX_DIV_WINDOW", ex_msg
                    advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {ex_msg}"
                    if advisory not in cadence_advisories:
                        cadence_advisories.append(advisory)

                if item is not None:
                    # Core wide-base / diversifier DCA is a structural allocation, not a
                    # tactical R:R trade: the 2:1 floor and numeric stop/target do not apply.
                    t_snap = tech_lookup.get(sym_up) or tech_lookup.get(base)
                    if pick_core_etf:
                        pass
                    elif acct_up == "ISA" and not is_etf:
                        # v5.5.1 ISA branch: quality gate + margin of safety REPLACE the 2:1 R:R
                        # hurdle entirely (a Buffett value name may legitimately carry stop_price=0).
                        isa_value_mode = str(
                            getattr(strat, "isa_value_guard_mode", ISA_VALUE_GUARD_MODE_DEFAULT) or ""
                        ).strip().lower()
                        if isa_value_mode != "off":
                            is_crisis = bool(macro_high_risk or regime == "RISK_OFF")
                            vg = evaluate_isa_value_gate(
                                b,
                                t_snap,
                                macro,
                                strat,
                                item=item,
                                symbol=sym_up,
                                crisis_mode=is_crisis,
                            )
                            if not vg["quality_pass"]:
                                vg_msg = f"ISA 质量门未通过：{'；'.join(vg['quality_fails'])}"
                                if isa_value_mode == "hard":
                                    return "ISA_QUALITY_FAIL", vg_msg
                                _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {vg_msg}"
                                if _adv not in cadence_advisories:
                                    cadence_advisories.append(_adv)
                            elif not vg["mos_pass"]:
                                mode_lbl = "危机深折价模式" if is_crisis else "常规模式"
                                vg_msg = (
                                    f"ISA 安全边际不足（{mode_lbl}）："
                                    f"{'；'.join(vg['mos_reasons']) or '估值证据不足'}"
                                )
                                if isa_value_mode == "hard":
                                    return "ISA_MOS_INSUFFICIENT", vg_msg
                                _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {vg_msg}"
                                if _adv not in cadence_advisories:
                                    cadence_advisories.append(_adv)
                    else:
                        # v5.5.1 Invest branch: regime × structure dynamic R:R + ATR noise floor.
                        rr_info = _resolve_pick_rr(
                            item,
                            b,
                            strat,
                            symbol=sym_up,
                            account=acct_up,
                            wallet=wallet,
                            tech=t_snap,
                            regime=regime,
                        )
                        if rr_info.get("stop_too_tight"):
                            st_msg = (
                                f"止损幅仅{float(rr_info.get('risk_pct') or 0.0):.1f}%，"
                                f"低于ATR噪声地板{float(rr_info.get('min_stop_pct') or 0.0):.1f}%，"
                                "禁止借窄止损虚增R:R"
                            )
                            if guard_mode == "hard":
                                return "STOP_TOO_TIGHT", st_msg
                            _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {st_msg}"
                            if _adv not in cadence_advisories:
                                cadence_advisories.append(_adv)
                        rr = rr_info.get("rr")
                        if rr is None:
                            rr_msg = (
                                "Python 无法复核 R:R（stop_price / target_price 未给出有效数字）；"
                                "hard 模式要求先给出数值止损与目标价"
                            )
                            if guard_mode == "hard":
                                return "RR_UNVERIFIED", rr_msg
                            advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {rr_msg}"
                            if advisory not in cadence_advisories:
                                cadence_advisories.append(advisory)
                        else:
                            min_rr, tier_label = resolve_invest_dynamic_rr_floor(regime, t_snap, strat)
                            if float(rr) < min_rr:
                                rr_msg = (
                                    f"Python 复算 R:R 仅{float(rr):.2f}:1(<{min_rr:.1f}:1 {tier_label}；"
                                    f"目标空间{float(rr_info['upside_pct'] or 0.0):+.1f}% "
                                    f"扣摩擦{float(rr_info['friction_pct'] or 0.0):.2f}% "
                                    f"vs 止损{float(rr_info['risk_pct'] or 0.0):.1f}%)"
                                )
                                if guard_mode == "hard":
                                    return "RR_BELOW_FLOOR", rr_msg
                                advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {rr_msg}"
                                if advisory not in cadence_advisories:
                                    cadence_advisories.append(advisory)

            if macro_high_risk and macro_vix is not None:
                core_etf_buy = bool(
                    sym_up in core_etf_syms
                    or base in core_etf_syms
                    or normalize_issuer_root(sym_up)
                    in {normalize_issuer_root(s) for s in core_etf_syms}
                )
                if not core_etf_buy:
                    family = infer_etf_index_family(sym_up, b)
                    core_etf_buy = family in _CORE_WIDE_BASE_FAMILIES
                # v5.5.1: ISA deep-margin-of-safety quality value names keep a staged
                # contrarian entry lane during a crisis; Invest stays 100% frozen.
                isa_crisis_value_exempt = False
                if (
                    acct_up == "ISA"
                    and not is_etf
                    and b is not None
                    and bool(
                        getattr(
                            strat,
                            "isa_value_crisis_buy_enabled",
                            ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT,
                        )
                    )
                ):
                    vg_crisis = evaluate_isa_value_gate(
                        b,
                        tech_lookup.get(sym_up) or tech_lookup.get(base),
                        macro,
                        strat,
                        item=item,
                        symbol=sym_up,
                        crisis_mode=True,
                    )
                    isa_crisis_value_exempt = bool(vg_crisis["passed"])
                if not core_etf_buy and not isa_crisis_value_exempt:
                    return (
                        "MACRO_VIX",
                        f"VIX={macro_vix:.1f}(≥{vix_danger_threshold:.0f}高危阈值)，暂停非深折价优质个股买入/加仓",
                    )

        return None

    overrides: list[RiskOverrideRecord] = []
    notes: list[str] = []

    # Branch 1: MorningReportDecision (has top_actions)
    if hasattr(decision, "top_actions"):
        new_actions = []
        for act in decision.top_actions:
            violation = _check_symbol_violation(act.account, act.symbol, act.action)
            if violation is not None:
                trig, why = violation
                orig_action = act.action
                new_reason = f"[🛡️风控护栏修正] {why}；原建议({orig_action})已降级HOLD"
                new_actions.append(act.model_copy(update={"action": "HOLD", "reason": new_reason}))
                overrides.append(
                    RiskOverrideRecord(
                        account=act.account,
                        symbol=act.symbol,
                        from_action=orig_action,
                        to_action="HOLD",
                        trigger_type=trig,
                        reason=why,
                    )
                )
                notes.append(f"[🛡️风控护栏修正] [{act.account}] {act.symbol}: {orig_action}→HOLD（{why}）")
            else:
                new_actions.append(act)
        copied = decision.model_copy(update={"top_actions": new_actions}) if overrides else decision.model_copy()
        notes.extend(cadence_advisories)
        return RiskOverrideResult(
            decision=copied,
            applied=bool(overrides),
            overrides=overrides,
            notes=notes,
        )

    # Branch 2: PickDecision (has selected_actions & rejected_or_frozen)
    if hasattr(decision, "selected_actions"):
        kept_actions = []
        new_rejected = list(getattr(decision, "rejected_or_frozen", []) or [])
        sizing_applied = False
        gbp_usd_rate = getattr(portfolio, "gbp_usd_rate", None) if portfolio is not None else None
        for item in decision.selected_actions:
            violation = _check_symbol_violation(
                item.account, item.symbol, item.action, item, is_pick=True
            )
            if violation is not None:
                trig, why = violation
                overrides.append(
                    RiskOverrideRecord(
                        account=item.account,
                        symbol=item.symbol,
                        from_action=item.action,
                        to_action="REJECTED",
                        trigger_type=trig,
                        reason=why,
                    )
                )
                note = f"[🛡️风控护栏修正] [{item.account}] {item.symbol}({item.action}→拦截): {why}"
                notes.append(note)
                new_rejected.append(note)
            else:
                keep_item = item
                if str(item.action).strip().upper() in {"BUY", "ADD"}:
                    acct_up = item.account.strip().upper()
                    sym_up = item.symbol.strip().upper()
                    sym_base = sym_up.split(".")[0]
                    existing_val = 0.0
                    if portfolio is not None:
                        item_root = normalize_issuer_root(sym_up)
                        for held in portfolio.all_positions():
                            if held.account.strip().upper() != acct_up:
                                continue
                            held_root = normalize_issuer_root(
                                str(getattr(held, "ticker", "") or held.yahoo_symbol)
                            )
                            if held_root == item_root:
                                existing_val += float(held.current_value or 0.0)
                    size_note = _build_pick_size_note(
                        item,
                        bundle_lookup.get(sym_up) or bundle_lookup.get(sym_base),
                        strat,
                        account=item.account,
                        account_total_value_gbp=account_totals.get(acct_up),
                        existing_position_value_gbp=existing_val,
                        gbp_usd_rate=gbp_usd_rate,
                        wallet=portfolio.invest if (portfolio is not None and acct_up == "INVEST") else None,
                        risk_budget_scale=float(risk_budget_scale or 1.0),
                        tech=tech_lookup.get(sym_up) or tech_lookup.get(sym_base),
                        macro=macro,
                        regime=regime,
                    )
                    if size_note:
                        # risk_note 是 Python 侧保留字段：模型必须返回空串，落地前由这里写入。
                        keep_item = item.model_copy(update={"risk_note": size_note})
                        sizing_applied = True
                kept_actions.append(keep_item)
        copied = (
            decision.model_copy(
                update={"selected_actions": kept_actions, "rejected_or_frozen": new_rejected}
            )
            if (overrides or sizing_applied)
            else decision.model_copy()
        )
        notes.extend(cadence_advisories)
        return RiskOverrideResult(
            decision=copied,
            applied=bool(overrides),
            overrides=overrides,
            notes=notes,
        )

    return RiskOverrideResult(decision=decision, applied=False)


_DEFAULT_HOLD_REASON = "技术面与仓位均在常规区间,默认持有观望"
_HIGH_PRIORITY_HOLD_KEYWORDS = (
    "距财报仅",
    "⚠️",
    "集中度风险",
    "≥上限",
    "止损",
    "深套复核线",
    "放量下挫",
    "超买区",
    "短期筹码拥挤",
    "大幅正乖离",
    "分批止盈",
)


def _is_routine_hold(ev: HoldingEvaluation) -> bool:
    if ev.action_bias != "HOLD":
        return False
    joined_reasons = "；".join(ev.reasons)
    return not any(kw in joined_reasons for kw in _HIGH_PRIORITY_HOLD_KEYWORDS)


def format_evaluations_text(evals: list[HoldingEvaluation], compact_hold: bool = True) -> str:
    if not evals:
        return "（当前无持仓需要触发器评估）"
    badge = {
        "ADD_CANDIDATE": "🟢 [加仓候选]",
        "TRIM_ALERT": "🟠 [减仓/止盈警戒]",
        "REVIEW_STOP": "🔴 [止损/证伪复核]",
        "HOLD": "⚪ [常规持有]",
    }
    lines: list[str] = []
    routine_holds: list[str] = []
    for ev in evals:
        if compact_hold and _is_routine_hold(ev):
            routine_holds.append(f"{ev.ticker}[{ev.account} {ev.weight_pct:.1f}%/{ev.return_pct:+.1f}%]")
            continue
        lines.append(
            f"• [{ev.account}] {ev.ticker} {badge.get(ev.action_bias, ev.action_bias)} "
            f"(占比 {ev.weight_pct:.1f}%, 浮盈 {ev.return_pct:+.1f}%, {ev.friction_note}): "
            + "；".join(ev.reasons)
        )
    if routine_holds:
        lines.append(f"• ⚪ [常规持有 x{len(routine_holds)} 只均无异常]：{', '.join(routine_holds)}")
    return "\n".join(lines)
