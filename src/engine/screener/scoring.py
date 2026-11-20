"""Four-dimensional dual-philosophy candidate scoring and cross-sectional percentile calibration."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot
from src.engine.portfolio_rules import (
    DEFAULT_RISK_FREE_YIELD_PCT,
    _estimate_entry_friction_pct,
    _is_leverage_exempt_sector,
    evaluate_isa_value_gate,
    resolve_invest_dynamic_rr_floor,
    resolve_invest_target_anchors,
    wallet_friction_context,
)
from src.engine.screener.patterns import (
    ScreenerCandidate,
    _linear_scale,
    _normalize_regime_tag,
    compute_relative_strength,
    compute_vpvr_profile,
    detect_breakthrough_platform,
    detect_low_atr_contraction,
    detect_minervini_stage2,
    detect_vcp_contraction,
    detect_volume_ma_breakout,
)

def _safe_target_upside_pct(
    symbol: str,
    tech: TechnicalSnapshot,
    bundle: TickerMarketBundle | None,
) -> float | None:
    """Compute analyst consensus target upside with GBp/GBP 100x unit mismatch guard."""
    if bundle is None or bundle.target_mean_price is None or bundle.target_mean_price <= 0:
        return None
    if bundle.analyst_count is not None and bundle.analyst_count < 3:
        return None
    ref_price = (
        bundle.current_price
        if (bundle.current_price is not None and bundle.current_price > 0)
        else tech.last_close
    )
    if ref_price <= 0:
        return None

    target = float(bundle.target_mean_price)
    if symbol.strip().upper().endswith(".L"):
        ratio = target / ref_price
        if 0.005 <= ratio <= 0.05:
            target *= 100.0
        elif 20.0 <= ratio <= 200.0:
            target /= 100.0

    upside_pct = ((target / ref_price) - 1.0) * 100.0
    if not (-80.0 <= upside_pct <= 200.0):
        return None
    return upside_pct


def _resolve_configured_etfs(
    universe: UniverseSection | None = None,
    strategy: StrategySection | None = None,
) -> set[str]:
    etf_set: set[str] = set()
    for section in (universe, strategy):
        if section is None:
            continue
        raw = getattr(section, "etf_symbols", None)
        if raw:
            etf_set.update(str(s).strip().upper() for s in raw if str(s).strip())
    return etf_set


_EQ_REGION_SUFFIXES = (".DE", ".PA", ".AS", ".MI", ".MC", ".SW", ".BR", ".VI", ".F")


def _region_label(symbol: str) -> str:
    """Coarse listing region for the /pick candidate prefilter line (0-token, suffix-based)."""
    sym = str(symbol or "").strip().upper()
    if sym.endswith(".L"):
        return "UK"
    if sym.endswith(_EQ_REGION_SUFFIXES):
        return "EU"
    return "US"


def _calibrate_l_price(symbol: str, value: float | None, ref_price: float) -> float | None:
    """Fix the GBp/GBP 100x slip between a vendor price and the .L quote currency."""
    if value is None or value <= 0 or ref_price <= 0:
        return None
    price = float(value)
    if str(symbol or "").strip().upper().endswith(".L"):
        ratio = price / ref_price
        if 0.005 <= ratio <= 0.05:
            price *= 100.0
        elif 20.0 <= ratio <= 200.0:
            price /= 100.0
    return price


def _format_summary_line(
    *,
    symbol: str,
    track: str,
    asset_type: str,
    score: float,
    tech: TechnicalSnapshot,
    bundle: TickerMarketBundle | None,
    upside_pct: float | None,
    corr_holding: str | None,
    max_corr: float | None,
    rs_20d_pct: float | None,
    marginal_vol_delta_pct: float | None,
    regime_tag: str,
    dimension_pct: dict[str, float],
    pros: list[str],
    cons: list[str],
    rs_60d_pct: float | None = None,
    structure_levels: str = "",
    usd_wallet_ok: bool = False,
) -> str:
    pe_str = ""
    is_etf = asset_type == "ETF"
    if bundle and not is_etf:
        if bundle.forward_pe:
            pe_str = f" | 远期PE {bundle.forward_pe:.1f}x"
        elif bundle.trailing_pe:
            pe_str = f" | PE {bundle.trailing_pe:.1f}x"
    div_str = (
        f" | 股息 {bundle.dividend_yield_pct:.1f}%"
        if bundle and bundle.dividend_yield_pct and 0.0 < bundle.dividend_yield_pct <= 15.0
        else ""
    )
    roe_str = (
        f" | ROE {bundle.roe_pct:.0f}%"
        if bundle and not is_etf and bundle.roe_pct is not None
        else ""
    )
    upside_str = f" | 目标空间 {upside_pct:+.0f}%" if upside_pct is not None else ""
    cyq_str = (
        f" | CYQ获利 {tech.cyq_benefit_part_pct:.0f}%({tech.cyq_regime})"
        if not is_etf and tech.cyq_benefit_part_pct is not None and tech.cyq_regime
        else ""
    )
    ed_str = f" | 财报 {bundle.earnings_date}" if bundle and bundle.earnings_date else ""
    ex_div_str = ""
    if bundle is not None and bundle.days_to_ex_dividend is not None:
        try:
            ex_days = int(bundle.days_to_ex_dividend)
        except (TypeError, ValueError):
            ex_days = None
        if ex_days is not None and -3 <= ex_days <= 7:
            if ex_days > 0:
                ex_div_str = (
                    f" | 除息 {bundle.ex_dividend_date}({ex_days}天后)"
                    if bundle.ex_dividend_date
                    else f" | 除息({ex_days}天后)"
                )
            elif ex_days == 0:
                ex_div_str = " | 今日除息"
            else:
                ex_div_str = f" | 除息已过{abs(ex_days)}天"
    corr_str = f" | 最高相关({corr_holding})={max_corr:.2f}" if max_corr is not None and corr_holding else ""
    rs_str = f" | 20日RS {rs_20d_pct:+.1f}%" if rs_20d_pct is not None else ""
    rs60_str = f" | 60日RS {rs_60d_pct:+.1f}%" if rs_60d_pct is not None else ""
    mvol_str = f" | 增量波动 {marginal_vol_delta_pct:+.2f}%" if marginal_vol_delta_pct is not None else ""
    sector = str((bundle.sector or bundle.industry) if bundle else "") or ""
    region = _region_label(symbol)
    loc_str = f" | [{sector}/{region}]" if sector else f" | [{region}]"
    atr_bits: list[str] = []
    if tech.atr14 is not None:
        atr_bits.append(f"{float(tech.atr14):.2f}")
    if tech.atr_pct is not None:
        atr_bits.append(f"({float(tech.atr_pct):.1f}%)")
    atr_str = f" | ATR14 {' '.join(atr_bits)}" if atr_bits else ""
    weekly_bits: list[str] = []
    if tech.weekly_bullish is True:
        weekly_bits.append("周线多头")
    elif tech.weekly_bullish is False:
        weekly_bits.append("周线偏空")
    if tech.adx14 is not None:
        weekly_bits.append(f"ADX {float(tech.adx14):.0f}")
    weekly_str = f" | {'/'.join(weekly_bits)}" if weekly_bits else ""
    vol_str = (
        f" | 量比 {float(tech.volume_ratio_20d):.1f}x"
        if tech.volume_ratio_20d is not None
        else ""
    )
    cyq_cost_str = (
        f" | CYQ均成本 {float(tech.cyq_avg_cost):.2f}"
        if (not is_etf and tech.cyq_avg_cost is not None)
        else ""
    )
    structure_str = f" | 结构[{structure_levels}]" if structure_levels else ""
    usd_str = " | USD钱包可用" if usd_wallet_ok else ""
    sell_side_str = ""
    if (
        bundle is not None
        and (bundle.analyst_count or 0) >= 3
        and bundle.target_mean_price is not None
        and bundle.target_mean_price > 0
    ):
        ref_price = (
            float(bundle.current_price)
            if (bundle.current_price is not None and float(bundle.current_price) > 0)
            else float(tech.last_close)
        )
        mean = _calibrate_l_price(symbol, bundle.target_mean_price, ref_price)
        low = _calibrate_l_price(symbol, bundle.target_low_price, ref_price)
        high = _calibrate_l_price(symbol, bundle.target_high_price, ref_price)
        if mean is not None and ref_price > 0:
            mean_upside = ((mean / ref_price) - 1.0) * 100.0
            spread = (
                f"{low:.2f}~{mean:.2f}~{high:.2f}"
                if (low is not None and high is not None)
                else f"{mean:.2f}"
            )
            sell_side_str = (
                f" | 卖方({int(bundle.analyst_count)}家) {spread}({mean_upside:+.0f}%)"
            )
    dim_str = (
        f" | 4维 T{dimension_pct.get('trend_rs', 0.0) * 100:.0f}%/"
        f"P{dimension_pct.get('pattern_timing', 0.0) * 100:.0f}%/"
        f"V{dimension_pct.get('value_quality', 0.0) * 100:.0f}%/"
        f"F{dimension_pct.get('portfolio_fit', 0.0) * 100:.0f}%"
        if dimension_pct
        else ""
    )
    pros_str = f" | 优势: {', '.join(pros)}" if pros else ""
    cons_str = f" | 风险: {', '.join(cons)}" if cons else ""

    if (
        tech.ret_5d_pct is not None
        and tech.ret_20d_pct is not None
        and tech.dist_ma50_pct is not None
        and tech.dist_ma200_pct is not None
        and tech.rsi14 is not None
        and tech.drawdown_52w_pct is not None
    ):
        return (
            f"• [{track}][{asset_type}] {symbol}"
            f"{f' ({bundle.short_name})' if bundle and bundle.short_name else ''}{loc_str} "
            f"| 量化初筛分 {score:.0f}{dim_str} | 现价 {tech.last_close:.2f}"
            f"{f' {bundle.currency}' if bundle and bundle.currency else ''} | "
            f"5日 {tech.ret_5d_pct:+.1f}% / 20日 {tech.ret_20d_pct:+.1f}% | "
            f"距MA50 {tech.dist_ma50_pct:+.1f}% / MA200 {tech.dist_ma200_pct:+.1f}% | "
            f"RSI {tech.rsi14:.0f} | 距52w高点 {tech.drawdown_52w_pct:+.1f}%"
            f"{pe_str}{roe_str}{div_str}{upside_str}{rs_str}{rs60_str}{atr_str}{weekly_str}{vol_str}"
            f"{cyq_str}{cyq_cost_str}{structure_str}{sell_side_str}{ed_str}{ex_div_str}{corr_str}{mvol_str}{usd_str}"
            f"{pros_str}{cons_str}"
        )
    return f"• [{track}][{asset_type}] {symbol}{loc_str} | 分数 {score:.0f}{dim_str}{ex_div_str}{pros_str}{cons_str}"


def score_candidate(
    symbol: str,
    track: str,
    tech: TechnicalSnapshot,
    bundle: TickerMarketBundle | None,
    max_corr: float | None,
    corr_holding: str | None,
    is_held: bool,
    is_watchlist: bool,
    strategy: StrategySection,
    *,
    hist: pd.DataFrame | None = None,
    benchmark_hist: pd.DataFrame | None = None,
    weighted_corr: float | None = None,
    marginal_vol_delta_pct: float | None = None,
    regime_tag: str = "",
    is_recent_traded: bool = False,
    etf_symbols: Iterable[str] | None = None,
    deep_intel: dict[str, Any] | None = None,
    invest_wallet: AccountSnapshot | None = None,
    macro: MacroSnapshot | None = None,
) -> ScreenerCandidate:
    """Score a candidate across 4 orthogonal capped dimensions with asymmetric dual-track weights & factor attribution."""
    pros: list[str] = []
    cons: list[str] = []
    pattern_signals: list[str] = []

    sym_up = symbol.strip().upper()
    etf_set = {str(s).strip().upper() for s in (etf_symbols or ()) if str(s).strip()}
    is_etf = bool((bundle and bundle.quote_type.strip().upper() == "ETF") or sym_up in etf_set)
    asset_type = "ETF" if is_etf else "EQUITY"

    # ------------------------------------------------------------------
    # Dim 1: Trend & Relative Strength (0..30, midpoint 15.0)
    # ------------------------------------------------------------------
    dim1 = 15.0
    rs_info = compute_relative_strength(hist, benchmark_hist)
    rs_20d = rs_info.get("rs_20d_pct")
    rs_60d = rs_info.get("rs_60d_pct")
    if rs_20d is None and tech.rs_20d_pct is not None:
        rs_20d = float(tech.rs_20d_pct)
    if rs_60d is None and tech.rs_60d_pct is not None:
        rs_60d = float(tech.rs_60d_pct)
    rs_slope = rs_info.get("rs_slope_20d")
    minervini = detect_minervini_stage2(hist, tech, rs_60d_pct=rs_60d)
    minervini_phase = int(minervini.get("phase", 0))
    reg_norm = _normalize_regime_tag(regime_tag)
    # v5.5.7: the 24~30 top band on the Invest track is reserved for genuine right-side
    # leaders (20d & 60d RS positive and rising); ordinary bulls are capped lower.
    strong_rs = bool(
        track != "ISA_LONG_TERM"
        and (rs_20d or 0.0) > 0.0
        and (rs_60d or 0.0) > 0.0
        and (rs_slope or 0.0) > 0.0
    )

    if track == "ISA_LONG_TERM":
        # v5.5.1: ISA is a Buffett value track — momentum breakdowns are NOT auto-penalized;
        # a quality business on sale (drawdown) earns a margin-of-safety bonus instead.
        isa_quality_ok = not (
            (bundle is not None and bundle.profit_margin_pct is not None and bundle.profit_margin_pct <= 0.0)
            or (bundle is not None and bundle.roe_pct is not None and bundle.roe_pct <= 0.0)
            or (
                bundle is not None
                and bundle.revenue_growth_pct is not None
                and bundle.revenue_growth_pct <= -10.0
            )
        )
        if tech.trend_regime == "BULL":
            dim1 += 5.0
            pros.append("上升趋势(MA50>MA200)")
        elif tech.trend_regime == "PULLBACK":
            dim1 += 7.0
            pros.append("多头回踩支撑区")

        dd52 = tech.drawdown_52w_pct
        if dd52 is not None and -18.0 <= dd52 <= -4.0:
            dim1 += 3.0
        if dd52 is not None and -35.0 <= dd52 <= -10.0 and isa_quality_ok:
            dim1 += 5.0
            pros.append(f"优质股安全边际回撤({dd52:.0f}%)")
        if (not isa_quality_ok) and tech.trend_regime == "WEAK":
            dim1 -= 8.0
            cons.append("基本面偏弱+下跌趋势(防价值陷阱)")

        if tech.rsi14 is not None:
            if 32.0 <= tech.rsi14 <= 55.0:
                dim1 += 3.0
            elif tech.rsi14 >= 72.0:
                dim1 -= _linear_scale(tech.rsi14, 72.0, 85.0, 2.0, 6.0)

        if tech.volatility_60d_pct is not None and tech.volatility_60d_pct > 45.0:
            dim1 -= _linear_scale(tech.volatility_60d_pct, 45.0, 80.0, 2.0, 6.0)
            cons.append(f"高波动({tech.volatility_60d_pct:.0f}%)")
    else:
        if not strong_rs:
            # v5.5.7: trim the ordinary-bull baseline so mediocre momentum no longer saturates.
            dim1 -= 2.5
        if tech.trend_regime == "PULLBACK":
            dim1 += 6.0
            pros.append("强趋势回调买点")
        elif tech.trend_regime == "BULL":
            dim1 += 5.0
            pros.append("趋势延续")
        elif tech.trend_regime == "WEAK":
            dim1 -= 6.0
            cons.append("趋势偏弱")

        if tech.rsi14 is not None:
            if tech.rsi14 <= 38.0:
                # v5.5.9 M3: the oversold-reversal flavour belongs to the ISA value
                # track; Invest no longer pays a +4.5 bottom-fishing bonus here.
                dim1 += 1.5
            elif 40.0 <= tech.rsi14 <= 60.0 and (tech.ret_20d_pct or 0.0) > 0.0:
                dim1 += 3.0
                pros.append("动量健康")
            elif tech.rsi14 >= 70.0:
                if strong_rs and minervini["passes"]:
                    # v5.5.9 M3: genuine Stage-2 leaders are not punished for strength
                    # (RSI 70~80 is the daily norm for volume-backed breakouts); only
                    # extreme blow-offs (>=82) get a light caution so a parabolic top
                    # still surfaces a warning. ISA track is unreachable here —
                    # strong_rs is hard-wired False for ISA_LONG_TERM.
                    if tech.rsi14 >= 82.0:
                        dim1 -= _linear_scale(tech.rsi14, 82.0, 92.0, 1.0, 3.0)
                        cons.append("极端超买(≥82)轻提示")
                else:
                    dim1 -= _linear_scale(tech.rsi14, 70.0, 84.0, 2.0, 7.0)

        if bundle and bundle.days_to_earnings is not None and 0 <= bundle.days_to_earnings <= 5:
            dim1 -= 4.0
            cons.append(f"临近财报({bundle.days_to_earnings}天)")

    if minervini["passes"]:
        dim1 += 3.5
        pattern_signals.append("Minervini阶段2")
        pros.append(f"Minervini模板达标({minervini['criteria_passed']}/{minervini['criteria_total']})")
        if track != "ISA_LONG_TERM" and reg_norm == "RISK-ON" and (rs_20d or 0.0) > 0.0:
            # v5.5.1: RISK_ON reinforces Stage-2 leadership visibility on the Invest track.
            dim1 += 2.0
            pros.append("RISK_ON领涨结构强化")
    elif minervini_phase == 4 and track != "ISA_LONG_TERM":
        # Option 2A: Penalize Phase 4 downtrend heavily while preserving visibility + warning tag.
        # v5.5.1: ISA value track is exempt (a quality name in a downtrend can be a bargain).
        dim1 -= 8.0
        cons.append("[Phase4弱势]")

    if reg_norm == "RISK-OFF" and tech.trend_regime == "BULL" and (tech.rsi14 or 50.0) >= 62.0:
        dim1 -= 2.0
    elif reg_norm == "RISK-ON" and tech.trend_regime in {"BULL", "PULLBACK"}:
        dim1 += 1.0

    if tech.weekly_bullish is True:
        if tech.trend_regime in {"BULL", "PULLBACK"}:
            dim1 += 4.0
            pros.append("周/日线多头共振")
        else:
            dim1 += 2.0
            pros.append("周线多头")
    elif tech.weekly_bullish is False and track != "ISA_LONG_TERM":
        dim1 -= 4.0
        cons.append("周线偏空")

    if rs_20d is not None:
        dim1 += _linear_scale(rs_20d, -8.0, 10.0, -2.5, 3.5)
    if rs_slope is not None and rs_slope > 0.15 and (rs_20d or 0.0) > 0.0:
        dim1 += 1.5

    dim1_cap = 30.0 if (track == "ISA_LONG_TERM" or strong_rs) else 24.0
    dim1 = round(max(0.0, min(dim1_cap, dim1)), 2)

    # ------------------------------------------------------------------
    # Dim 2: Pattern & Volume-Price Structure Timing (0..25, midpoint 10.0)
    # ------------------------------------------------------------------
    dim2 = 10.0

    # Subgroup A: Contraction / coiling (mutually exclusive max)
    if hist is not None and not hist.empty and detect_vcp_contraction(hist, tech):
        dim2 += 7.0
        pattern_signals.append("VCP收缩")
        pros.append("VCP波动收缩蓄势")
    elif hist is not None and not hist.empty and detect_low_atr_contraction(hist, tech):
        dim2 += 5.0
        pattern_signals.append("低ATR收敛")
        pros.append("低ATR波动收敛")
    elif tech.bb_squeeze:
        dim2 += 3.5
        pros.append("布林带缩口蓄势")

    # Subgroup B: Breakout trigger (mutually exclusive max)
    if hist is not None and not hist.empty and detect_breakthrough_platform(hist):
        dim2 += 7.0
        pattern_signals.append("平台突破")
        pros.append("平台突破")
    elif hist is not None and not hist.empty and detect_volume_ma_breakout(hist, tech):
        dim2 += 5.5
        pattern_signals.append("放量站上均线")
        pros.append("放量站上均线")
    elif tech.volume_ratio_20d is not None and tech.volume_ratio_20d >= 1.5 and (tech.ret_1d_pct or 0.0) > 0.0:
        dim2 += 3.5
        pros.append(f"放量走强({tech.volume_ratio_20d:.1f}x)")

    # Subgroup C: Volume-Price distribution (ETFs use 60d VPVR POC; Stage 2 leaders exempt from >92% penalty)
    vpvr = compute_vpvr_profile(hist) if hist is not None and not hist.empty else None
    stage2_leader = bool(minervini["passes"] and (rs_slope or 0.0) > 0.0)

    if is_etf or not tech.cyq_regime:
        if vpvr is not None:
            if 0.0 <= vpvr["poc_dist_pct"] <= 8.0 and vpvr["overhead_vol_pct"] <= 35.0:
                dim2 += 5.0
                pros.append(f"VPVR支撑单峰(POC+{vpvr['poc_dist_pct']:.1f}%)")
            elif vpvr["poc_dist_pct"] < -1.0 and vpvr["overhead_vol_pct"] >= 65.0:
                dim2 -= 5.0
                cons.append(f"VPVR上方套牢({vpvr['overhead_vol_pct']:.0f}%)")
    else:
        if tech.cyq_regime == "底部单峰密集":
            dim2 += 6.0
            pros.append("CYQ底部单峰密集")
        elif tech.cyq_regime == "上方套牢峰":
            dim2 -= 6.0
            cons.append("CYQ上方套牢峰压制")
        elif (
            tech.cyq_regime == "获利盘高企"
            and (tech.cyq_benefit_part_pct or 0.0) >= 92.0
            and not stage2_leader
        ):
            dim2 -= 3.0
            cons.append(f"获利盘高企({tech.cyq_benefit_part_pct:.0f}%)")
        elif (
            tech.cyq_concentration_90_pct is not None
            and tech.cyq_concentration_90_pct <= 12.0
            and (tech.cyq_benefit_part_pct or 0.0) >= 45.0
        ):
            dim2 += 3.0
            pros.append(f"筹码集中度佳({tech.cyq_concentration_90_pct:.1f}%)")

    dim2 = round(max(0.0, min(25.0, dim2)), 2)

    # ------------------------------------------------------------------
    # Dim 3: Linear Valuation & Fundamental Quality (0..25, midpoint 10.0)
    # ------------------------------------------------------------------
    dim3 = 10.0
    upside_pct = _safe_target_upside_pct(symbol, tech, bundle)
    score_penalty = 0.0
    est_rr: float | None = None
    if upside_pct is not None:
        dim3 += _linear_scale(upside_pct, -15.0, 25.0, -4.0, 5.0)

    # v5.5.7/v5.5.9 M2 Invest forward R:R reachability soft gate: shares the exact same
    # resolve_invest_target_anchors helper as /pick and holding_eval. Confirmed RS leaders
    # in non-RISK_OFF regimes are evaluated against the two-anchor ceiling instead of a
    # stale sell-side mean; weak / non-RS / RISK_OFF names remain locked to analyst mean.
    if (
        track != "ISA_LONG_TERM"
        and not is_etf
        and upside_pct is not None
        and bundle is not None
        and (bundle.analyst_count or 0) >= 3
    ):
        ref_price = (
            float(bundle.current_price)
            if (bundle.current_price is not None and float(bundle.current_price) > 0)
            else float(tech.last_close)
        )
        anchors = resolve_invest_target_anchors(
            ref_price=ref_price,
            bundle=bundle,
            tech=tech,
            strategy=strategy,
            regime=reg_norm,
            symbol=sym_up,
            rs_20d_pct=rs_20d,
            rs_60d_pct=rs_60d,
        )
        eff_upside_pct = float(upside_pct)
        if anchors["is_leader"] and anchors["llm_ceiling"] is not None and ref_price > 0:
            eff_upside_pct = ((float(anchors["llm_ceiling"]) / ref_price) - 1.0) * 100.0
        atr_pct_val = float(tech.atr_pct) if (tech.atr_pct or 0.0) > 0 else 0.0
        atr_floor_pct = max(1.5 * atr_pct_val, 4.0) if atr_pct_val > 0 else 4.5
        est_friction_pct = _estimate_entry_friction_pct(
            sym_up, bundle, strategy, "INVEST", invest_wallet
        )
        est_rr = round(
            (eff_upside_pct - est_friction_pct) / (atr_floor_pct + est_friction_pct), 2
        )
        min_rr_floor, rr_tier = resolve_invest_dynamic_rr_floor(
            reg_norm,
            tech,
            strategy,
            vix=(macro.vix if (macro is not None and macro.vix is not None) else None),
        )
        reach_floor = 0.7 * float(min_rr_floor)
        if est_rr < reach_floor:
            deficit = reach_floor - est_rr
            score_penalty = min(15.0, max(8.0, 8.0 + 10.0 * deficit))
            cons.append(
                f"预估R:R仅{est_rr:.2f}:1(目标空间{eff_upside_pct:+.0f}%/门槛{min_rr_floor:.1f}:1 {rr_tier})"
            )

    if bundle and not is_etf:
        pe_val = (
            bundle.forward_pe
            if (bundle.forward_pe is not None and bundle.forward_pe > 0)
            else (bundle.trailing_pe if bundle.trailing_pe is not None and bundle.trailing_pe > 0 else None)
        )
        if pe_val is not None:
            if track == "ISA_LONG_TERM":
                if pe_val <= 22.0:
                    dim3 += _linear_scale(pe_val, 8.0, 22.0, 5.0, 2.5)
                else:
                    dim3 += _linear_scale(pe_val, 22.0, 55.0, 2.5, -5.0)
            else:
                if pe_val <= 28.0:
                    dim3 += _linear_scale(pe_val, 10.0, 28.0, 4.0, 1.5)
                else:
                    dim3 += _linear_scale(pe_val, 28.0, 65.0, 1.5, -5.0)

        if bundle.roe_pct is not None:
            dim3 += _linear_scale(bundle.roe_pct, 5.0, 25.0, -1.0, 3.0)
        if bundle.profit_margin_pct is not None:
            dim3 += _linear_scale(bundle.profit_margin_pct, 2.0, 22.0, -1.0, 2.0)
        if bundle.revenue_growth_pct is not None:
            dim3 += _linear_scale(bundle.revenue_growth_pct, 0.0, 25.0, -0.5, 2.0)

        # v5.5.1 ISA value quality: FCF yield + earnings yield spread, leverage exemption for
        # banks/insurers/REITs/utilities, and hard penalties for value traps.
        rf_yield = DEFAULT_RISK_FREE_YIELD_PCT
        if macro is not None and macro.us_10y_yield is not None and macro.us_10y_yield > 0:
            rf_yield = float(macro.us_10y_yield)
        if track == "ISA_LONG_TERM":
            if bundle.fcf_yield_pct is not None:
                dim3 += _linear_scale(bundle.fcf_yield_pct, -1.0, 7.0, -2.5, 4.0)
                if bundle.fcf_yield_pct >= 4.0:
                    pros.append(f"高自由现金流(FCF={bundle.fcf_yield_pct:.1f}%)")
            if pe_val is not None and pe_val > 0:
                erp = (100.0 / pe_val) - rf_yield
                dim3 += _linear_scale(erp, -1.5, 4.5, -2.0, 3.0)
            if (
                (bundle.profit_margin_pct is not None and bundle.profit_margin_pct <= 0.0)
                or (bundle.roe_pct is not None and bundle.roe_pct <= 0.0)
            ):
                dim3 -= 6.0
                cons.append("盈利为负(不符ISA质量门)")
            if bundle.revenue_growth_pct is not None and bundle.revenue_growth_pct <= -10.0:
                dim3 -= 4.0

        if bundle.debt_to_equity is not None and bundle.debt_to_equity >= 0.0:
            if not _is_leverage_exempt_sector(bundle):
                dim3 += _linear_scale(bundle.debt_to_equity, 30.0, 200.0, 1.0, -2.0)
                if bundle.debt_to_equity >= 180.0:
                    cons.append(f"高杠杆(D/E={bundle.debt_to_equity:.0f}%)")
        # IMP-18: Finnhub /stock/metric value multiples (only when the vendor supplied them).
        if bundle.price_to_book is not None and bundle.price_to_book > 0.0:
            if track == "ISA_LONG_TERM":
                dim3 += _linear_scale(bundle.price_to_book, 1.0, 6.0, 2.5, -1.5)
            else:
                dim3 += _linear_scale(bundle.price_to_book, 1.5, 8.0, 1.5, -1.5)
        if bundle.ev_ebitda is not None and bundle.ev_ebitda > 0.0:
            dim3 += _linear_scale(bundle.ev_ebitda, 8.0, 25.0, 1.0, -1.0)
    elif is_etf:
        dim3 += 2.0

    if bundle and bundle.dividend_yield_pct is not None and 0.0 < bundle.dividend_yield_pct <= 15.0:
        div = bundle.dividend_yield_pct
        if track == "ISA_LONG_TERM":
            dim3 += _linear_scale(div, 0.0, 3.5, 0.0, 3.5)
        else:
            dim3 += _linear_scale(div, 0.0, 4.0, 0.0, 2.0)

    wallet_enabled, usd_avail, gbp_avail = (
        wallet_friction_context(invest_wallet) if track != "ISA_LONG_TERM" else (False, 0.0, 0.0)
    )
    if sym_up.endswith(".L") and is_etf:
        if track == "ISA_LONG_TERM" or not wallet_enabled or gbp_avail > 0.01:
            dim3 += 6.0
            pros.append(
                "LSE英镑ETF(免换汇/免印花税)"
                if track == "ISA_LONG_TERM"
                else "LSE英镑ETF(Invest GBP充足:免换汇/免印花税)"
            )
        else:
            dim3 += 2.5
            pros.append(f"LSE英镑ETF(Invest GBP不足:USD换回+{strategy.fx_fee_pct:.2f}%换汇)")
    elif sym_up.endswith(".L"):
        if track == "ISA_LONG_TERM":
            dim3 += 2.5
            pros.append("英镑计价免换汇")
        else:
            # v5.5.7: stamp duty is a hard round-trip drag on short-term tactical entries —
            # cancel the old "no FX needed" bonus and charge the friction instead.
            dim3 -= 1.0
            cons.append(f"含{float(strategy.uk_stamp_duty_pct):.1f}%印花税(短线高摩擦)")

    if track != "ISA_LONG_TERM" and wallet_enabled and not sym_up.endswith(".L"):
        is_eur = sym_up.endswith((".DE", ".PA", ".AS", ".MI", ".MC", ".SW", ".BR", ".VI", ".F"))
        if is_eur:
            cons.append(f"欧股需换汇+{strategy.fx_fee_pct:.2f}%单边(往返{strategy.fx_fee_pct * 2:.2f}%)")
        elif usd_avail > 0.01:
            dim3 += 2.0
            pros.append("美股可用USD钱包免换汇")
        elif gbp_avail > 0.01:
            cons.append(f"美股USD不足需GBP换汇+{strategy.fx_fee_pct:.2f}%单边")

    if deep_intel and not is_etf:
        surprises = deep_intel.get("earnings_surprises") or []
        sp_vals = [
            float(s["surprise_pct"])
            for s in surprises
            if isinstance(s, dict) and isinstance(s.get("surprise_pct"), (int, float))
        ]
        if len(sp_vals) >= 4 and all(v > 0.0 for v in sp_vals[:4]):
            dim3 += 2.0
            pros.append("连续财报Beat")

        insider = deep_intel.get("insider_sentiment")
        if isinstance(insider, dict) and isinstance(insider.get("mspr"), (int, float)):
            mspr_val = float(insider["mspr"])
            if mspr_val > 0.0:
                dim3 += 1.5
                pros.append(f"内部人净增持(MSPR={mspr_val:+.1f})")
            elif mspr_val < -5.0:
                dim3 -= 1.0
                cons.append(f"内部人减持(MSPR={mspr_val:+.1f})")
            elif mspr_val < 0.0:
                cons.append(f"内部人减持(MSPR={mspr_val:+.1f})")

        recs = deep_intel.get("recommendations") or []
        if recs and isinstance(recs[0], dict):
            r0 = recs[0]
            sb = int(r0.get("strong_buy") or 0)
            buy = int(r0.get("buy") or 0)
            hold = int(r0.get("hold") or 0)
            sell = int(r0.get("sell") or 0)
            ss = int(r0.get("strong_sell") or 0)
            tot_r = sb + buy + hold + sell + ss
            if tot_r >= 3 and ((sb / tot_r) >= 0.45 or ((sb + buy) / tot_r >= 0.80 and sb >= 5)):
                dim3 += 1.5
                pros.append(f"机构强买共识({sb + buy}/{tot_r})")

    dim3 = round(max(0.0, min(25.0, dim3)), 2)

    # ------------------------------------------------------------------
    # Dim 4: Portfolio Fit, 3d Cross-Timezone Corr & Marginal Vol (0..20, midpoint 10.0)
    # ------------------------------------------------------------------
    dim4 = 10.0
    overlap_warning = False
    if max_corr is not None and not is_held:
        if max_corr >= strategy.overlap_correlation_threshold:
            overlap_warning = True
            dim4 -= 9.0
            cons.append(f"与持仓{corr_holding}相关性高达{max_corr:.2f}(高度重叠)")
        else:
            dim4 += _linear_scale(max_corr, 0.75, 0.25, 0.0, 6.0)

    if weighted_corr is not None and not is_held:
        dim4 += _linear_scale(weighted_corr, 0.75, 0.15, -1.5, 2.0)

    if marginal_vol_delta_pct is not None and not is_held:
        if marginal_vol_delta_pct <= -0.02:
            dim4 += _linear_scale(marginal_vol_delta_pct, -0.02, -0.35, 0.5, 2.5)
        elif marginal_vol_delta_pct >= 0.35:
            dim4 -= _linear_scale(marginal_vol_delta_pct, 0.35, 0.90, 0.5, 2.5)

    if is_watchlist:
        dim4 += 3.5
        pros.append("自选观察池")
    if is_recent_traded and not is_held:
        dim4 += 1.5
        pros.append("近90天实盘交易过")
    if is_held:
        cons.append("当前已持有")

    dim4 = round(max(0.0, min(20.0, dim4)), 2)

    dimension_scores = {
        "trend_rs": dim1,
        "pattern_timing": dim2,
        "value_quality": dim3,
        "portfolio_fit": dim4,
    }
    dimension_pct = {
        "trend_rs": round(dim1 / 30.0, 3),
        "pattern_timing": round(dim2 / 25.0, 3),
        "value_quality": round(dim3 / 25.0, 3),
        "portfolio_fit": round(dim4 / 20.0, 3),
    }

    # Asymmetric track-level weighting so ISA & Invest surface distinct candidate profiles
    if track == "ISA_LONG_TERM":
        # v5.5.1: value/quality dominant (15/10/50/25) — kills the ISA momentum bias.
        weighted_raw = (
            15.0 * dimension_pct["trend_rs"]
            + 10.0 * dimension_pct["pattern_timing"]
            + 50.0 * dimension_pct["value_quality"]
            + 25.0 * dimension_pct["portfolio_fit"]
        )
    else:
        weighted_raw = (
            35.0 * dimension_pct["trend_rs"]
            + 35.0 * dimension_pct["pattern_timing"]
            + 15.0 * dimension_pct["value_quality"]
            + 15.0 * dimension_pct["portfolio_fit"]
        )
    score = round(max(0.0, min(100.0, weighted_raw - score_penalty)), 1)
    reasons = [*pros, *cons]

    filter_metrics: dict[str, Any] = {
        "sector": (str(bundle.sector or bundle.industry or "")[:18] if bundle else ""),
        "region": _region_label(symbol),
        "target_upside_pct": (round(float(upside_pct), 1) if upside_pct is not None else None),
        "atr_pct": (round(float(tech.atr_pct), 2) if tech.atr_pct is not None else None),
        "rs_60d_pct": (round(float(rs_60d), 2) if rs_60d is not None else None),
        "est_rr": est_rr,
    }
    if track == "ISA_LONG_TERM" and not is_etf and bundle is not None:
        isa_gate = evaluate_isa_value_gate(
            bundle,
            tech,
            macro,
            strategy,
            symbol=sym_up,
            crisis_mode=(reg_norm == "RISK-OFF"),
        )
        filter_metrics["mos_pct"] = isa_gate.get("mos_pct")
        filter_metrics["fcf_yield_pct"] = isa_gate.get("fcf_yield_pct")

    rs_20d_round = round(rs_20d, 2) if rs_20d is not None else None
    rs_60d_round = round(rs_60d, 2) if rs_60d is not None else None
    structure_bits: list[str] = []
    if hist is not None and not hist.empty and "Low" in hist.columns:
        lows = pd.to_numeric(hist["Low"], errors="coerce").dropna()
        if len(lows) >= 20:
            structure_bits.append(f"20d低 {float(lows.tail(20).min()):.2f}")
        if len(lows) >= 60:
            structure_bits.append(f"60d低 {float(lows.tail(60).min()):.2f}")
    if tech.ma50 is not None:
        structure_bits.append(f"MA50 {float(tech.ma50):.2f}")
    if tech.ma200 is not None:
        structure_bits.append(f"MA200 {float(tech.ma200):.2f}")
    if tech.high_52w is not None:
        structure_bits.append(f"52w高 {float(tech.high_52w):.2f}")
    structure_levels = "/".join(structure_bits)

    usd_wallet_ok = False
    if invest_wallet is not None and track != "ISA_LONG_TERM":
        _, usd_avail, _ = wallet_friction_context(invest_wallet)
        rate = float(getattr(macro, "gbp_usd", 0.0) or 0.0) if macro is not None else 0.0
        usd_wallet_ok = bool(
            rate > 0 and usd_avail >= float(strategy.min_invest_order_gbp) * rate
        )
    summary_line = _format_summary_line(
        symbol=symbol,
        track=track,
        asset_type=asset_type,
        score=score,
        tech=tech,
        bundle=bundle,
        upside_pct=upside_pct,
        corr_holding=corr_holding,
        max_corr=max_corr,
        rs_20d_pct=rs_20d_round,
        marginal_vol_delta_pct=marginal_vol_delta_pct,
        regime_tag=regime_tag,
        dimension_pct=dimension_pct,
        pros=pros,
        cons=cons,
        rs_60d_pct=rs_60d_round,
        structure_levels=structure_levels,
        usd_wallet_ok=usd_wallet_ok,
    )

    return ScreenerCandidate(
        symbol=symbol,
        track=track,
        name=bundle.short_name if bundle else symbol,
        asset_type=asset_type,
        currency=bundle.currency if bundle else "",
        score=score,
        is_already_held=is_held,
        is_watchlist=is_watchlist,
        is_recent_traded=is_recent_traded,
        max_portfolio_corr=max_corr,
        most_correlated_holding=corr_holding,
        overlap_warning=overlap_warning,
        rs_20d_pct=rs_20d_round,
        marginal_vol_delta_pct=marginal_vol_delta_pct,
        regime_tag=regime_tag,
        dimension_scores=dimension_scores,
        dimension_pct=dimension_pct,
        pros=pros,
        cons=cons,
        pattern_signals=pattern_signals,
        reasons=reasons,
        summary_line=summary_line,
        filter_metrics=filter_metrics,
    )


def _apply_percentile_calibration(candidates: list[ScreenerCandidate]) -> list[ScreenerCandidate]:
    """Blend raw 4D score (85%) with cross-sectional pool percentile rank (15%) when pool >= 4."""
    if len(candidates) < 4:
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    raw_s = pd.Series([c.score for c in candidates], dtype=float)
    pct_s = raw_s.rank(method="average", pct=True) * 100.0
    calibrated: list[ScreenerCandidate] = []
    for idx, cand in enumerate(candidates):
        blended = round(max(0.0, min(100.0, 0.85 * cand.score + 0.15 * float(pct_s.iloc[idx]))), 1)
        old_tag = f"量化初筛分 {cand.score:.0f}"
        new_tag = f"量化初筛分 {blended:.0f}"
        if old_tag in cand.summary_line:
            new_summary = cand.summary_line.replace(old_tag, new_tag, 1)
        else:
            new_summary = cand.summary_line.replace(f"分数 {cand.score:.0f}", f"分数 {blended:.0f}", 1)
        calibrated.append(cand.model_copy(update={"score": blended, "summary_line": new_summary}))

    calibrated.sort(key=lambda c: c.score, reverse=True)
    return calibrated
