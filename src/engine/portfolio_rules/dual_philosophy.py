"""Dual-philosophy entry gates: ISA value/quality/MOS gate vs Invest dynamic R:R and forward R:R."""

from __future__ import annotations

from typing import Any

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, PositionItem
from src.engine.portfolio_rules.models_and_friction import (
    DEFAULT_RISK_FREE_YIELD_PCT,
    INVEST_MIN_RR_NEUTRAL,
    INVEST_MIN_RR_RISK_ON_LEADER,
    INVEST_MIN_RR_WEAK_OR_RISK_OFF,
    INVEST_RISK_ON_TARGET_HIGH_HAIRCUT,
    INVEST_STOP_MIN_ATR_MULT,
    INVEST_STOP_MIN_PCT_FLOOR,
    ISA_BUBBLE_FORWARD_PE,
    ISA_BUBBLE_PEG_THRESHOLD,
    ISA_MOS_CRISIS_DISCOUNT_PCT,
    ISA_MOS_CRISIS_MIN_ERP_PCT,
    ISA_MOS_MIN_DISCOUNT_PCT,
    ISA_MOS_MIN_ERP_PCT,
    ISA_QUALITY_MAX_DEBT_TO_EQUITY,
    ISA_QUALITY_MAX_DEBT_TO_EQUITY_NO_CR,
    ISA_QUALITY_MIN_CURRENT_RATIO,
    ISA_QUALITY_MIN_FCF_YIELD_PCT,
    ISA_QUALITY_MIN_PROFIT_MARGIN_PCT,
    ISA_QUALITY_MIN_REV_GROWTH_PCT,
    ISA_QUALITY_MIN_ROE_PCT,
    _LEVERAGE_EXEMPT_SECTOR_KEYWORDS,
    _rescale_price_pair,
    _round_trip_friction_pct,
    _us_sell_settled_in_gbp,
    _usd_retention_active,
    wallet_friction_context,
)

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


def _calibrate_target_price(
    price: float,
    ref: float,
    *,
    symbol: str = "",
    currency: str = "",
) -> float:
    """Calibrate vendor/LLM target prices against ref_price with wider .L GBp/GBP 100x guard."""
    if price <= 0 or ref <= 0:
        return price
    sym_up = str(symbol or "").strip().upper()
    ccy_raw = str(currency or "").strip()
    if sym_up.endswith(".L") or ccy_raw == "GBp" or ccy_raw.upper() == "GBX":
        ratio = price / ref
        if 0.005 <= ratio <= 0.05:
            return price * 100.0
        if 20.0 <= ratio <= 200.0:
            return price / 100.0
        return price
    return _rescale_price_pair(price, ref)


def is_strong_right_side_leader(
    tech: TechnicalSnapshot | None,
    *,
    regime: str | None = "NEUTRAL",
    rs_20d_pct: float | None = None,
    rs_60d_pct: float | None = None,
) -> bool:
    """Single source of truth for unlocking the Invest dual-anchor target ceiling (v5.5.9 M2).

    Unlocks ONLY when regime != RISK_OFF, structural Stage-2 leadership holds (Price > MA200,
    BULL/PULLBACK, weekly not bearish), AND both 20d & 60d benchmark-relative strength are
    strictly positive. Missing RS fails closed so weak/unverified names stay locked to mean.
    """
    reg = str(regime or "NEUTRAL").strip().upper().replace("-", "_")
    if reg == "RISK_OFF":
        return False
    if not _is_right_side_leader(tech):
        return False
    rs20 = rs_20d_pct if rs_20d_pct is not None else getattr(tech, "rs_20d_pct", None)
    rs60 = rs_60d_pct if rs_60d_pct is not None else getattr(tech, "rs_60d_pct", None)
    return bool(
        rs20 is not None
        and float(rs20) > 0.0
        and rs60 is not None
        and float(rs60) > 0.0
    )


def resolve_invest_target_anchors(
    *,
    ref_price: float,
    bundle: TickerMarketBundle | None,
    tech: TechnicalSnapshot | None,
    strategy: StrategySection,
    regime: str | None = "NEUTRAL",
    symbol: str = "",
    rs_20d_pct: float | None = None,
    rs_60d_pct: float | None = None,
) -> dict[str, Any]:
    """Unified Invest target-anchor resolver shared by scoring, /pick guardrail & holding eval.

    Returns both:
    - ``llm_ceiling``: upper bound on LLM-proposed targets (max of mean×1.15, high×0.90,
      entry+4×ATR for confirmed leaders; mean×1.15 otherwise).
    - ``deterministic_target``: target without synthetic 4×ATR (max of mean, high×0.90 for
      confirmed leaders; mean otherwise), preserving forward R:R deterioration warnings.
    """
    sym = (
        str(symbol or "").strip()
        or (str(bundle.symbol or "").strip() if bundle is not None else "")
        or (str(tech.symbol or "").strip() if tech is not None else "")
    )
    ccy = str(getattr(bundle, "currency", "") or "") if bundle is not None else ""
    is_leader = is_strong_right_side_leader(
        tech,
        regime=regime,
        rs_20d_pct=rs_20d_pct,
        rs_60d_pct=rs_60d_pct,
    )
    has_coverage = bool(bundle is not None and (bundle.analyst_count or 0) >= 3)
    analyst_mean: float | None = None
    if (
        has_coverage
        and ref_price > 0
        and bundle is not None
        and bundle.target_mean_price is not None
        and float(bundle.target_mean_price) > 0
    ):
        analyst_mean = _calibrate_target_price(
            float(bundle.target_mean_price), ref_price, symbol=sym, currency=ccy
        )

    target_high_adj: float | None = None
    haircut = float(
        getattr(strategy, "invest_risk_on_target_high_haircut", INVEST_RISK_ON_TARGET_HIGH_HAIRCUT)
    )
    if (
        has_coverage
        and ref_price > 0
        and bundle is not None
        and bundle.target_high_price is not None
        and float(bundle.target_high_price) > 0
    ):
        target_high_adj = (
            _calibrate_target_price(
                float(bundle.target_high_price), ref_price, symbol=sym, currency=ccy
            )
            * haircut
        )

    atr_anchor: float | None = None
    if (
        ref_price > 0
        and tech is not None
        and tech.atr_pct is not None
        and float(tech.atr_pct) > 0
    ):
        atr_anchor = ref_price * (1.0 + 4.0 * float(tech.atr_pct) / 100.0)

    if is_leader:
        cand_ceilings: list[float] = []
        if analyst_mean is not None and analyst_mean > 0:
            cand_ceilings.append(analyst_mean * (1.0 + _PICK_RR_TARGET_TOLERANCE_PCT / 100.0))
        if target_high_adj is not None and target_high_adj > 0:
            cand_ceilings.append(target_high_adj)
        if atr_anchor is not None and atr_anchor > 0:
            cand_ceilings.append(atr_anchor)
        llm_ceiling = max(cand_ceilings) if cand_ceilings else None
        det_cands = [v for v in (analyst_mean, target_high_adj) if v is not None and v > 0]
        deterministic_target = max(det_cands) if det_cands else None
    else:
        llm_ceiling = (
            analyst_mean * (1.0 + _PICK_RR_TARGET_TOLERANCE_PCT / 100.0)
            if (analyst_mean is not None and analyst_mean > 0)
            else None
        )
        deterministic_target = analyst_mean

    return {
        "is_leader": is_leader,
        "analyst_mean": analyst_mean,
        "target_high_adj": target_high_adj,
        "atr_anchor": atr_anchor,
        "llm_ceiling": llm_ceiling,
        "deterministic_target": deterministic_target,
    }


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
        "effective_risk_pct": None,
        "upside_pct": None,
        "rr": None,
        "friction_pct": None,
        "target_replaced": False,
        "price_rescaled": False,
        "entry_planned": False,
        "stop_too_tight": False,
        "min_stop_pct": None,
    }
    ref_price = (
        float(bundle.current_price)
        if (bundle is not None and bundle.current_price and float(bundle.current_price) > 0)
        else 0.0
    )
    entry = float(getattr(item, "entry_price", 0.0) or 0.0)
    if entry <= 0 and ref_price > 0:
        entry = ref_price
    if entry <= 0:
        return out
    # Align the LLM-supplied entry with the live quote currency before any R:R maths.
    if ref_price > 0:
        rescaled_entry = _rescale_price_pair(entry, ref_price)
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

    # v5.5.7: an optional planned limit centre may not be used to fake a better R:R —
    # rescale for GBp/GBP, clamp into [current*0.92, current*1.03] and require > stop.
    planned_raw = getattr(item, "planned_entry_price", None)
    if planned_raw is not None and float(planned_raw) > 0 and ref_price > 0:
        planned_entry = _rescale_price_pair(float(planned_raw), ref_price)
        lo, hi = ref_price * 0.92, ref_price * 1.03
        if lo <= planned_entry <= hi and (stop <= 0 or planned_entry > stop):
            if abs(planned_entry - entry) > 1e-9:
                out["price_rescaled"] = True
            entry = planned_entry
            out["entry_planned"] = True

    sym_resolved = symbol or str(getattr(item, "symbol", "") or "")
    anchors = resolve_invest_target_anchors(
        ref_price=entry,
        bundle=bundle,
        tech=tech,
        strategy=strategy,
        regime=regime,
        symbol=sym_resolved,
    )
    analyst_target = float(anchors["analyst_mean"] or 0.0)
    is_right_side = bool(anchors["is_leader"])
    out["is_right_side_leader"] = is_right_side
    if is_right_side:
        # v5.5.9 M2: non-RISK_OFF confirmed RS leaders use the unified two-anchor ceiling
        # instead of being dragged back to a stale analyst mean, while still capping hallucinations.
        ceiling = float(anchors["llm_ceiling"] or 0.0)
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
    effective_risk_pct = (
        round(float(risk_pct) + float(friction_pct), 2) if risk_pct is not None else None
    )
    out.update(
        {
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
            "effective_risk_pct": effective_risk_pct,
            "upside_pct": round(upside_pct, 2) if upside_pct is not None else None,
            "friction_pct": round(friction_pct, 2),
        }
    )
    # v5.5.1 Invest stop must clear the ATR noise floor — a razor-thin stop fakes a huge R:R.
    # v5.5.7: this check keeps using the PURE price stop (risk_pct); friction only enters the
    # net R:R and risk-based sizing denominator, so a narrow stop cannot bypass the ATR floor.
    if risk_pct is not None and tech is not None and tech.atr_pct is not None and tech.atr_pct > 0:
        atr_mult = float(getattr(strategy, "invest_stop_min_atr_mult", INVEST_STOP_MIN_ATR_MULT))
        floor_pct = float(getattr(strategy, "invest_stop_min_pct_floor", INVEST_STOP_MIN_PCT_FLOOR))
        min_stop = round(max(atr_mult * float(tech.atr_pct), floor_pct), 2)
        out["min_stop_pct"] = min_stop
        if risk_pct < min_stop:
            out["stop_too_tight"] = True
    if effective_risk_pct is not None and effective_risk_pct > 0 and upside_pct is not None:
        out["rr"] = round((upside_pct - friction_pct) / effective_risk_pct, 2)
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
    *,
    vix: float | None = None,
) -> tuple[float, str]:
    """Regime × structure × VIX dynamic R:R floor for Invest (1.6 / 2.0 / 2.5).

    v5.5.7: the VIX tighten line is folded in here so the morning report, Researcher prompt,
    Critic prompt and Python guardrail all read one and the same number (no more 2.0 vs 2.5
    split). A high-VIX tape tightens NEUTRAL to the 2.5 defensive floor.
    """
    reg = str(regime or "NEUTRAL").strip().upper().replace("-", "_")
    base_neutral = float(
        getattr(strategy, "invest_min_rr_neutral", None)
        or getattr(strategy, "min_rr_ratio", INVEST_MIN_RR_NEUTRAL)
        or INVEST_MIN_RR_NEUTRAL
    )
    rr_leader = float(
        getattr(strategy, "invest_min_rr_risk_on_leader", INVEST_MIN_RR_RISK_ON_LEADER)
    )
    rr_weak = float(getattr(strategy, "invest_min_rr_weak_or_risk_off", INVEST_MIN_RR_WEAK_OR_RISK_OFF))
    vix_threshold = float(
        getattr(strategy, "invest_vix_tighten_rr_threshold", 22.0) or 22.0
    )
    vix_high = bool(vix is not None and float(vix) >= vix_threshold)

    is_weak = bool(
        reg == "RISK_OFF"
        or vix_high
        or (
            tech is not None
            and (
                tech.trend_regime == "WEAK"
                or (tech.dist_ma200_pct is not None and tech.dist_ma200_pct <= 0.0)
            )
        )
    )
    if is_weak:
        if vix_high and reg != "RISK_OFF":
            return rr_weak, f"高波防御门槛(VIX≥{vix_threshold:.0f})"
        return rr_weak, "弱势/防御高赔率门槛"
    if reg == "RISK_ON" and _is_right_side_leader(tech):
        return min(rr_leader, base_neutral), "RISK_ON右侧强结构"
    if reg == "RECOVERY":
        return base_neutral, "RECOVERY早春试探门槛"
    return base_neutral, f"{reg or 'NEUTRAL'}标准门槛"


def resolve_holding_target_upside_pct(
    pos: PositionItem,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    is_invest: bool,
    regime: str = "NEUTRAL",
    rs_20d_pct: float | None = None,
    rs_60d_pct: float | None = None,
) -> float | None:
    """Resolve calibrated target upside % for a holding (Invest uses unified dual-anchor helper; ISA uses mean)."""
    ref_cur = (
        float(bundle.current_price)
        if (bundle is not None and bundle.current_price and bundle.current_price > 0)
        else (float(pos.cur_price) if is_invest else 0.0)
    )
    if ref_cur <= 0:
        return None
    anchors = resolve_invest_target_anchors(
        ref_price=ref_cur,
        bundle=bundle,
        tech=tech,
        strategy=strategy,
        regime=regime,
        symbol=pos.yahoo_symbol,
        rs_20d_pct=rs_20d_pct,
        rs_60d_pct=rs_60d_pct,
    )
    target = anchors["deterministic_target"] if is_invest else anchors["analyst_mean"]
    if target is None or float(target) <= 0:
        return None
    return ((float(target) / ref_cur) - 1.0) * 100.0


def compute_invest_forward_rr(
    pos: PositionItem,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    effective_stop_pct: float | None,
    regime: str = "NEUTRAL",
    wallet: AccountSnapshot | None = None,
    rs_20d_pct: float | None = None,
    rs_60d_pct: float | None = None,
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
    anchors = resolve_invest_target_anchors(
        ref_price=cur,
        bundle=bundle,
        tech=tech,
        strategy=strategy,
        regime=regime,
        symbol=pos.yahoo_symbol,
        rs_20d_pct=rs_20d_pct,
        rs_60d_pct=rs_60d_pct,
    )
    target = anchors["deterministic_target"]
    if target is None or float(target) <= 0:
        return None
    upside_pct = ((float(target) / cur) - 1.0) * 100.0
    risk_pct = abs(float(effective_stop_pct))
    if risk_pct <= 0:
        return None
    friction_pct = _round_trip_friction_pct(pos, bundle, strategy, wallet) * 0.5
    return round((upside_pct - friction_pct) / risk_pct, 2)
