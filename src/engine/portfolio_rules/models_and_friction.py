"""Portfolio rule constants, Pydantic models, dual-wallet friction matrix, and risk sizing helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot, infer_etf_index_family
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, PendingOrderItem, PositionItem

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

# --- v5.5.9 M4 right-tail ladder (advisory layer; hard stops stay untouched) ---
# +15% profit -> raise stop to cost+friction (financed unlock: 0 heat / 0 slot);
# +25% leader standing on MA20 -> replace the mechanical trim nudge with a
# stop-lift advisory (lock floor +10% or trail MA20); NEUTRAL confirmed leaders
# keep a wider trailing buffer than laggards. All read via getattr(strategy,...).
INVEST_LADDER_BREAKEVEN_ADVISORY_PCT: float = 15.0
INVEST_LADDER_LOCK_FLOOR_PCT: float = 10.0
INVEST_TRAILING_DD_NEUTRAL_LEADER_PCT: float = -11.0


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


def is_invest_reservoir_symbol(yahoo_symbol: str, strategy: StrategySection) -> bool:
    """v5.5.9 M5: True when a symbol is an enabled Invest distributing-ETF reservoir member.

    Single source of truth shared by ledgers (reservoir holdings scan), sizing
    (ETF-to-equity funding sources) and triggers (reservoir admission tickets).
    Membership requires BOTH the master switch and the DIST whitelist; with the
    switch off the reservoir machinery (tickets + funding + whitelist tier) goes
    quiet. Whitelist symbols must stay a subset of the distributing core ETF set
    (UK ERI red line: taxable Invest never buys accumulating classes).
    """
    if not bool(getattr(strategy, "invest_reservoir_enabled", True)):
        return False
    sym = str(yahoo_symbol or "").strip().upper()
    if not sym:
        return False
    whitelist = {
        str(s).strip().upper()
        for s in (getattr(strategy, "invest_reservoir_etf_whitelist", None) or [])
        if str(s).strip()
    }
    if not whitelist:
        return False
    return sym in whitelist or sym.split(".")[0] in whitelist


def is_invest_core_beta_symbol(yahoo_symbol: str, strategy: StrategySection) -> bool:
    """v5.5.9 M5: True for any Invest-held core broad-base ETF (a beta vehicle).

    Tier 1 — an enabled reservoir whitelist member (the four DIST symbols the M5
    admission/funding machinery manages). Tier 2, switch-independent: a symbol in
    the curated static ETF index-family map whose family is a core wide-base family
    (legacy holdings bought through the ERI-guarded DIST channel, e.g. VUSA.L /
    IUSA.L / VUKE.L / IMEU.L). The inference is deliberately bundle-less so the
    "S&P 500" name fallback can never drag a sector ETF (e.g. IITU.L, S&P 500 Info
    Tech) into the core privilege; unknown symbols fail closed and keep the full
    tactical treatment. Admission tickets and funding sources stay whitelist-only.
    """
    if is_invest_reservoir_symbol(yahoo_symbol, strategy):
        return True
    sym = str(yahoo_symbol or "").strip().upper()
    if not sym:
        return False
    return infer_etf_index_family(sym, None) in _CORE_WIDE_BASE_FAMILIES


def emit_invest_core_beta_bypass_note(
    pos: PositionItem,
    strategy: StrategySection,
    reg_up: str,
    reasons: list[str],
) -> bool:
    """v5.5.9 M5: detect Invest core-beta ETF membership and emit its holding-eval note.

    Core broad-base ETFs are beta parking vehicles, not right-side tickets: returns
    True so evaluate_holding bypasses the ATR stop review, trailing take-profit, M4
    ladder, forward-R:R decay and the momentum technicals entirely. Whitelist members
    get the reservoir note; other curated core-family holdings (e.g. a legacy VUSA.L)
    get the plain core-beta note. Sector/thematic/bond ETFs outside the curated map
    keep the full tactical treatment (defence never gets looser) and ISA ETFs are
    untouched by construction. RISK_OFF tapes get a defensive review note only —
    core ETF sells never feed the loss-streak cooldown (ledgers filters ETFs).
    """
    if pos.account.upper() != "INVEST":
        return False
    if is_invest_reservoir_symbol(pos.yahoo_symbol, strategy):
        if str(reg_up or "").strip().upper() == "RISK_OFF":
            reasons.append(
                "退潮期派息宽基蓄水池仅作防御评估/减持腾资参考，绝不触发个股止损链、移动止盈或连损冷静期"
            )
        else:
            reasons.append(
                "Invest派息宽基蓄水池：Beta停泊零槽位零热力零Quota，旁路短线止损/移动止盈/阶梯（仅RISK_OFF复核防御处置）"
            )
        return True
    if is_invest_core_beta_symbol(pos.yahoo_symbol, strategy):
        if str(reg_up or "").strip().upper() == "RISK_OFF":
            reasons.append(
                "退潮期核心宽基ETF仅作防御评估/减持腾资参考，绝不触发个股止损链、移动止盈或连损冷静期"
            )
        else:
            reasons.append(
                "Invest核心宽基ETF：Beta车辆旁路短线止损/移动止盈/阶梯（仅RISK_OFF复核防御处置）"
            )
        return True
    return False


def emit_invest_ladder_advisories(
    pos: PositionItem,
    strategy: StrategySection,
    *,
    reg_up: str,
    stop_flag: bool,
    trailing_triggered: bool,
    holding_above_ma20: bool,
    heavy_down_vol: bool,
    ex_div_guard: bool,
    same_acct_orders: list[PendingOrderItem] | None,
    friction_pct: float,
    reasons: list[str],
) -> int:
    """v5.5.9 M4 right-tail ladder (advisory only; hard ATR stops and quota untouched).

    +15% -> suggest raising the stop to cost+friction so the ledger classifies the
    position as financed (0 heat / 0 slot); "already protected" is detected with the
    SAME rule as ledgers.py's breakeven unlock: protective STOP/STOP_LIMIT sell orders
    only (a high LIMIT take-profit order is NOT downside protection), covering >= 99%
    of shares with a quantity-weighted aligned stop >= cost+friction.
    +25% standing on MA20 without distribution -> stop-lift advisory replaces the old
    mechanical trim nudge; breakdown / heavy down-vol / an already-fired trailing
    warning (and every RISK_OFF tape) re-arm the old defensive nudge so weak names and
    crisis tapes keep byte-identical v5.5.8 behaviour. Returns the trim_score delta.
    """
    if pos.account.upper() != "INVEST":
        return 0
    # v5.5.9 M5: core broad-base ETFs (reservoir whitelist + curated core-family map)
    # are beta parking vehicles, not right-side tickets — the +15%/-breakeven /
    # +25%/-stop-lift ladder never applies to them.
    if is_invest_core_beta_symbol(pos.yahoo_symbol, strategy):
        return 0
    # Offense advisories (breakeven unlock / let-it-run stop lift) are frozen on RISK_OFF
    # tapes and on positions already in stop review; the defensive +25% take-profit
    # nudge below is NOT frozen anywhere (v5.5.8 parity — defence never gets looser).
    offense_allowed = bool(reg_up != "RISK_OFF" and not stop_flag)
    tp_review_pct = float(strategy.invest_take_profit_review_pct)

    if pos.return_pct >= tp_review_pct:
        if offense_allowed and holding_above_ma20 and not heavy_down_vol and not trailing_triggered:
            reasons.append(
                f"Invest仓浮盈+{pos.return_pct:.1f}%且稳居MA20上方无放量派发：无需因浮盈本身减仓，"
                f"建议将止损上移至max(成本+{float(getattr(strategy, 'invest_ladder_lock_floor_pct', INVEST_LADDER_LOCK_FLOOR_PCT)):.0f}%锁盈地板,沿MA20跟踪)让利润奔跑；"
                "仅叠加破位/放量长阴/移动止盈警戒/前瞻R:R恶化时再评估减仓或ROTATE置换"
            )
            return 0
        reasons.append(
            f"Invest仓浮盈+{pos.return_pct:.1f}%,需评估催化剂是否已充分兑现以考虑分批止盈"
        )
        return 1
    if not offense_allowed:
        return 0

    ladder_base_pct = float(
        getattr(strategy, "invest_ladder_breakeven_advisory_pct", INVEST_LADDER_BREAKEVEN_ADVISORY_PCT)
    )
    if not (ladder_base_pct <= pos.return_pct < tp_review_pct):
        return 0

    # "Already at breakeven" = the ledgers.py financed rule, replicated 1-to-1:
    # protective STOP/STOP_LIMIT sells covering >=99% of shares, quantity-weighted
    # aligned stop >= avg_price x (1 + round-trip friction). Plain high LIMIT
    # take-profit orders and partially covered stops do NOT silence the advisory.
    stop_orders = [
        o
        for o in (same_acct_orders or [])
        if o.side.strip().upper() == "SELL" and "STOP" in str(o.order_type or "").strip().upper()
    ]
    covered_qty = 0.0
    weighted_stop_num = 0.0
    priced_qty = 0.0
    for o in stop_orders:
        raw_stop = float(o.stop_price or o.limit_price or 0.0)
        if raw_stop <= 0:
            continue
        aligned = _rescale_price_pair(raw_stop, pos.avg_price)
        qty = float(o.quantity or 0.0)
        covered_qty += qty
        weighted_stop_num += aligned * qty
        priced_qty += qty
    covered_qty = min(float(pos.quantity), covered_qty)
    aligned_stop = (weighted_stop_num / priced_qty) if priced_qty > 0 else 0.0
    has_breakeven_stop = bool(
        pos.quantity > 0
        and covered_qty >= float(pos.quantity) * 0.99
        and pos.avg_price > 0
        and aligned_stop >= pos.avg_price * (1.0 + friction_pct / 100.0)
    )
    if has_breakeven_stop:
        return 0
    ex_div_note = "；除息窗口内可暂缓一日,防除权低开击穿保本止损" if ex_div_guard else ""
    reasons.append(
        f"浮盈+{pos.return_pct:.1f}%已过+{ladder_base_pct:.0f}%保本解锁线：建议将止损上移至成本×(1+往返摩擦{friction_pct:.1f}%)的保本位(须全额覆盖)，"
        f"该仓即转0热力/0槽位，释放新开仓火力{ex_div_note}"
    )
    return 0

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
    # v5.5.7: per-ticket deterministic sizing ("ACCT|SYMBOL" -> sizing dict) for exact cash notes.
    sizing_map: dict[str, Any] = Field(default_factory=dict)


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
    scale = max(0.0, float(risk_budget_scale))
    if is_invest:
        base_min_order = float(strategy.min_invest_order_gbp)
        pilot_min_order = float(getattr(strategy, "min_invest_pilot_order_gbp", 500.0) or 500.0)
        # In pilot-scaled states (RECOVERY 0.6R or post-loss 0.5R), relax Invest min ticket
        # to the £500 pilot floor so elevated-ATR stops aren't deadlocked by £700.
        min_order = min(base_min_order, pilot_min_order) if (0.0 < scale < 1.0) else base_min_order
        cap_pct = float(strategy.max_invest_single_weight_pct)
    else:
        min_order = float(strategy.min_isa_etf_order_gbp if is_etf else strategy.min_isa_equity_order_gbp)
        cap_pct = float(strategy.max_isa_single_weight_pct)

    base_budget = float(strategy.invest_risk_budget_gbp)
    nav_gbp = float(account_total_value_gbp) if account_total_value_gbp is not None else 0.0
    if is_invest:
        # v5.5.5: Invest-only 0.75% NAV self-calibrating budget (floor £85). The ISA branch
        # below stays exactly at 0.5% so the value track is byte-for-byte unaffected.
        invest_pct = float(getattr(strategy, "invest_risk_budget_pct", 0.75) or 0.75) / 100.0
        nav_scaled = nav_gbp * invest_pct if nav_gbp > 0 else 0.0
        risk_budget = max(base_budget, nav_scaled) * scale
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
