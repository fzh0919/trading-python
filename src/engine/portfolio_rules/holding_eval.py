"""Per-holding health evaluation, portfolio evaluation, disagreement matrix, and text formatting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_drawdown_since_date,
    compute_rs_20_60,
    normalize_issuer_root,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
)
from src.engine.portfolio_rules.dual_philosophy import (
    compute_invest_forward_rr,
    evaluate_isa_value_gate,
    is_strong_right_side_leader,
    resolve_holding_target_upside_pct,
    resolve_invest_dynamic_rr_floor,
)
from src.engine.portfolio_rules.models_and_friction import (
    INVEST_FORWARD_RR_EXIT_WARN,
    INVEST_RSI_OVERBOUGHT_RISK_ON,
    INVEST_TRAILING_DD_NEUTRAL_LEADER_PCT,
    INVEST_TRAILING_DD_NEUTRAL_PCT,
    INVEST_TRAILING_DD_RISK_ON_PCT,
    ISA_VALUE_ADD_DRAWDOWN_PCT,
    DisagreementItem,
    HoldingEvaluation,
    _compute_adaptive_invest_stop_pct,
    _compute_analyst_upside_pct,
    _estimate_pending_order_gbp,
    _ex_div_buffer_pct,
    _friction_hint,
    _infer_account_total_gbp,
    _is_ex_div_guard,
    _native_return_pct,
    _round_trip_friction_pct,
    _wallet_cash_basis_pct,
    compute_risk_based_size_gbp,
    emit_invest_core_beta_bypass_note,
    emit_invest_ladder_advisories,
)

def _evaluate_holding_technicals(
    *,
    pos: PositionItem,
    tech: TechnicalSnapshot,
    strategy: StrategySection,
    is_invest: bool,
    is_etf: bool,
    invest_bull_ride: bool,
    heavy_down_vol: bool,
    ex_div_guard: bool,
    ex_div_buf: float,
    stop_flag: bool,
    isa_deep_dd: bool,
    over_weight: bool,
    val_gate: dict[str, Any] | None,
    add_score: int,
    trim_score: int,
    reasons: list[str],
) -> tuple[int, int, bool]:
    """Evaluate technical overbought/breakdown/add signals and ISA four-lock left-side value adds."""
    isa_value_add = False
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

    return add_score, trim_score, isa_value_add


def evaluate_holding(
    pos: PositionItem,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    total_issuer_weight_pct: float | None = None,
    *,
    hist: pd.DataFrame | None = None,
    benchmark_hist: pd.DataFrame | None = None,
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
    is_reservoir = emit_invest_core_beta_bypass_note(pos, strategy, reg_up, reasons)
    tactical = is_invest and not is_reservoir
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
    if tactical and effective_stop is not None:
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

    # v5.5.9 M4: structure flags computed before the trailing block so the NEUTRAL
    # leader tier (below) and the +25% ladder branch can gate on them. Pure reads only.
    holding_above_ma20 = bool(tech is not None and tech.dist_ma20_pct is not None and tech.dist_ma20_pct >= 0.0)
    heavy_down_vol = bool(
        tech is not None
        and tech.volume_ratio_20d is not None
        and tech.volume_ratio_20d >= 1.8
        and tech.ret_1d_pct is not None
        and tech.ret_1d_pct <= -2.0
        and not ex_div_guard
    )

    # v5.5.9 M4: benchmark-relative RS resolved early (tech field first, else hist calc)
    # so the NEUTRAL leader trailing tier reuses the same M2 ruler as target anchors.
    rs_20d = tech.rs_20d_pct if tech is not None else None
    rs_60d = tech.rs_60d_pct if tech is not None else None
    if (rs_20d is None or rs_60d is None) and hist is not None and benchmark_hist is not None:
        calc_rs20, calc_rs60 = compute_rs_20_60(hist, benchmark_hist)
        rs_20d = calc_rs20 if rs_20d is None else rs_20d
        rs_60d = calc_rs60 if rs_60d is None else rs_60d

    # v5.5.1: trailing profit protection is an Invest (right-side) tool only. ISA value holdings
    # are never sold on short-term volatility; they exit on quality/bubble/concentration instead.
    trailing_triggered = False
    trailing_dd_thresh = float(strategy.trailing_drawdown_from_peak_pct)
    if is_invest:
        if reg_up == "RISK_ON":
            trailing_dd_thresh = float(
                getattr(strategy, "invest_trailing_dd_risk_on_pct", INVEST_TRAILING_DD_RISK_ON_PCT)
            )
        elif reg_up == "NEUTRAL" and holding_above_ma20 and is_strong_right_side_leader(
            tech, regime=reg_up, rs_20d_pct=rs_20d, rs_60d_pct=rs_60d
        ):
            # v5.5.9 M4: a confirmed NEUTRAL leader standing on MA20 keeps a wider
            # buffer (-11) against washouts; laggards / below-MA20 stay at -8.5.
            trailing_dd_thresh = float(
                getattr(
                    strategy,
                    "invest_trailing_dd_neutral_leader_pct",
                    INVEST_TRAILING_DD_NEUTRAL_LEADER_PCT,
                )
            )
        else:
            trailing_dd_thresh = float(
                getattr(strategy, "invest_trailing_dd_neutral_pct", INVEST_TRAILING_DD_NEUTRAL_PCT)
            )
    if (
        tactical
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

    invest_bull_ride = bool(
        is_invest and reg_up == "RISK_ON" and holding_above_ma20 and not heavy_down_vol
    )
    isa_value_add = False

    if tech is not None and not is_reservoir:
        add_score, trim_score, isa_value_add = _evaluate_holding_technicals(
            pos=pos,
            tech=tech,
            strategy=strategy,
            is_invest=is_invest,
            is_etf=is_etf,
            invest_bull_ride=invest_bull_ride,
            heavy_down_vol=heavy_down_vol,
            ex_div_guard=ex_div_guard,
            ex_div_buf=ex_div_buf,
            stop_flag=stop_flag,
            isa_deep_dd=isa_deep_dd,
            over_weight=over_weight,
            val_gate=val_gate,
            add_score=add_score,
            trim_score=trim_score,
            reasons=reasons,
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

    # Hurdle gate v5.3.2 / v5.5.9 M2: ISA uses pure analyst mean, Invest uses unified dual-anchor helper.
    # Cash interest (4.0%/3.4%) is a parking-yield floor only — never a veto on long-term ISA holdings.
    upside_pct = resolve_holding_target_upside_pct(
        pos, tech, bundle, strategy, is_invest=is_invest, regime=reg_up, rs_20d_pct=rs_20d, rs_60d_pct=rs_60d
    )
    friction_pct = _round_trip_friction_pct(pos, bundle, strategy, wallet)
    cash_floor_pct = _wallet_cash_basis_pct(pos, strategy, wallet)

    # v5.5.9 M4 right-tail ladder (models_and_friction.emit_invest_ladder_advisories):
    # +15% -> breakeven/slot-unlock advisory; +25% on MA20 -> stop-lift advisory replaces
    # the mechanical trim nudge. Weak names / breakdown / heavy volume keep the old nudge.
    trim_score += emit_invest_ladder_advisories(
        pos,
        strategy,
        reg_up=reg_up,
        stop_flag=stop_flag,
        trailing_triggered=trailing_triggered,
        holding_above_ma20=holding_above_ma20,
        heavy_down_vol=heavy_down_vol,
        ex_div_guard=ex_div_guard,
        same_acct_orders=same_acct_orders,
        friction_pct=friction_pct,
        reasons=reasons,
    )

    rr_ratio: float | None = None
    forward_rr: float | None = None
    if tactical and effective_stop is not None and float(effective_stop) < 0 and upside_pct is not None:
        risk_pct = abs(float(effective_stop))
        rr_ratio = (upside_pct - friction_pct) / risk_pct if risk_pct > 0 else None
        min_rr, rr_tier = resolve_invest_dynamic_rr_floor(
            reg_up, tech, strategy, vix=(macro.vix if (macro is not None and macro.vix is not None) else None)
        )
        if rr_ratio is not None and rr_ratio < min_rr:
            add_score = 0
            reasons.append(
                f"风险回报比仅{rr_ratio:.2f}:1(<{min_rr:.1f}:1 {rr_tier}；目标空间{upside_pct:+.1f}%扣摩擦{friction_pct:.2f}% "
                f"vs 止损{risk_pct:.1f}%，现金停车{cash_floor_pct:.1f}%仅作等待期保底)"
            )
    elif not is_invest and upside_pct is not None:
        weak_long_trend = bool(
            tech is not None and tech.dist_ma200_pct is not None and tech.dist_ma200_pct < 0.0 and tech.trend_regime == "WEAK"
        )
        if weak_long_trend and (upside_pct - friction_pct) <= 0.0:
            reasons.append(
                f"长期弱势(MA200下方)+目标空间{upside_pct:+.1f}%扣摩擦{friction_pct:.2f}%后不足,触发机会成本置换复核(不以现金利息否决)"
            )

    # v5.5.1/v5.5.9 M2 Invest forward R:R: leftover upside vs remaining downside risk (<0.8:1 = 鸡肋).
    if tactical:
        forward_rr = compute_invest_forward_rr(
            pos, tech, bundle, strategy,
            effective_stop_pct=effective_stop, regime=reg_up, wallet=wallet,
            rs_20d_pct=rs_20d, rs_60d_pct=rs_60d,
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

    def _bench_hist_for(sym: str) -> pd.DataFrame | None:
        if not histories:
            return None
        for b_sym in resolve_market_benchmark(sym):
            df_b = histories.get(b_sym)
            if df_b is not None and not df_b.empty:
                return df_b
        return None

    evals = [
        evaluate_holding(
            pos,
            tech_map.get(pos.yahoo_symbol),
            bundle_map.get(pos.yahoo_symbol),
            strategy,
            total_issuer_weight_pct=issuer_weights.get(normalize_issuer_root(pos.ticker)),
            hist=histories.get(pos.yahoo_symbol) if histories else None,
            benchmark_hist=_bench_hist_for(pos.yahoo_symbol),
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


def format_evaluations_text(
    evals: list[HoldingEvaluation],
    compact_hold: bool = True,
    *,
    collapse_isa_add: bool = False,
) -> str:
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
    folded_isa_adds: list[str] = []
    for ev in evals:
        if collapse_isa_add and ev.account.upper() == "ISA" and ev.action_bias == "ADD_CANDIDATE":
            # v5.5.7: a closed ISA equity channel must not spam 7 green add-candidate rows.
            folded_isa_adds.append(ev.ticker)
            continue
        if compact_hold and _is_routine_hold(ev):
            routine_holds.append(f"{ev.ticker}[{ev.account} {ev.weight_pct:.1f}%/{ev.return_pct:+.1f}%]")
            continue
        lines.append(
            f"• [{ev.account}] {ev.ticker} {badge.get(ev.action_bias, ev.action_bias)} "
            f"(占比 {ev.weight_pct:.1f}%, 浮盈 {ev.return_pct:+.1f}%, {ev.friction_note}): "
            + "；".join(ev.reasons)
        )
    if folded_isa_adds:
        lines.append(
            f"• 🟢 [ISA 加仓候选 x{len(folded_isa_adds)} 只，通道关闭仅登记不执行]：{', '.join(folded_isa_adds)}"
        )
    if routine_holds:
        lines.append(f"• ⚪ [常规持有 x{len(routine_holds)} 只均无异常]：{', '.join(routine_holds)}")
    return "\n".join(lines)
