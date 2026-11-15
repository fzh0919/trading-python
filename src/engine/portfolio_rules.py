"""Deterministic portfolio health, disagreement summary & risk-override guardrail engine."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.config import StrategySection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_drawdown_since_date,
    normalize_issuer_root,
)
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import PendingOrderItem, PortfolioData, PositionItem


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
    trigger_type: str  # ISSUER_OVERWEIGHT | ACCOUNT_OVERWEIGHT | MACRO_VIX | EARNINGS_WINDOW | DAILY_VETO | CORRELATION_OVERLAP
    reason: str


class RiskOverrideResult(BaseModel):
    """Result returned by apply_risk_override_guardrail without mutating caller input."""

    decision: Any
    applied: bool = False
    overrides: list[RiskOverrideRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _is_uk_listing(yahoo_symbol: str, inst_currency: str) -> bool:
    return yahoo_symbol.endswith(".L") or inst_currency.upper() in {"GBP", "GBX"}


def _round_trip_friction_pct(
    pos: PositionItem,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
) -> float:
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    is_uk = _is_uk_listing(pos.yahoo_symbol, pos.inst_currency)
    if is_uk and is_etf:
        return 0.0
    if is_uk:
        return float(strategy.uk_stamp_duty_pct)
    return float(strategy.fx_fee_pct) * 2.0


def _friction_hint(pos: PositionItem, bundle: TickerMarketBundle | None, strategy: StrategySection) -> str:
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    is_uk = _is_uk_listing(pos.yahoo_symbol, pos.inst_currency)
    if is_uk and is_etf:
        return "LSE英镑ETF(0%换汇/0%印花税)"
    if is_uk:
        return f"英股(0%换汇, 买入{strategy.uk_stamp_duty_pct:.1f}%印花税)"
    if pos.inst_currency.upper() == "EUR":
        return f"欧股({strategy.fx_fee_pct:.2f}%单边换汇)"
    return f"美股({strategy.fx_fee_pct * 2:.2f}%往返换汇)"


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
) -> HoldingEvaluation:
    """Evaluate an existing position deterministically for Add / Trim / Stop / Hold."""
    reasons: list[str] = []
    add_score = 0
    trim_score = 0
    stop_flag = False

    is_invest = pos.account.upper() == "INVEST"
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
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

    trailing_triggered = False
    if (
        pos.return_pct >= strategy.trailing_profit_activation_pct
        and recent_dd is not None
        and recent_dd <= strategy.trailing_drawdown_from_peak_pct
    ):
        if ex_div_guard and recent_dd > (strategy.trailing_drawdown_from_peak_pct - ex_div_buf):
            reasons.append(
                f"浮盈+{pos.return_pct:.1f}%但处于除息窗口(回撤{recent_dd:.1f}%),暂免移动止盈误触"
            )
        else:
            trailing_triggered = True
            trim_score += 2
            window_lbl = "开仓后波段高点" if pos.created_at else "近60日波段高点"
            reasons.append(
                f"浮盈+{pos.return_pct:.1f}%后自{window_lbl}回撤{recent_dd:.1f}%(≤{strategy.trailing_drawdown_from_peak_pct:.0f}%),触发移动止盈保护警戒"
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

    if tech is not None:
        if tech.rsi14 is not None and tech.rsi14 >= strategy.rsi_overbought_threshold:
            trim_score += 2 if is_invest else 1
            reasons.append(f"RSI14={tech.rsi14:.1f}(≥{strategy.rsi_overbought_threshold:.0f}超买区)")

        if tech.dist_ma50_pct is not None and tech.dist_ma50_pct >= 18.0:
            trim_score += 1
            reasons.append(f"大幅正乖离MA50({tech.dist_ma50_pct:+.1f}%),短线追高风险大")

        if (
            tech.cyq_benefit_part_pct is not None
            and tech.cyq_benefit_part_pct >= 92.0
            and tech.rsi14 is not None
            and tech.rsi14 >= 70.0
        ):
            trim_score += 1
            reasons.append(f"CYQ获利盘达{tech.cyq_benefit_part_pct:.0f}%且RSI={tech.rsi14:.1f},短期筹码拥挤")
        elif (
            tech.cyq_regime == "上方套牢峰"
            and tech.cyq_benefit_part_pct is not None
            and tech.cyq_benefit_part_pct < 30.0
        ):
            reasons.append(f"CYQ上方套牢盘较重(仅{tech.cyq_benefit_part_pct:.0f}%获利),反弹留意抛压")

        if (
            tech.volume_ratio_20d is not None
            and tech.volume_ratio_20d >= 1.8
            and tech.ret_1d_pct is not None
            and tech.ret_1d_pct <= -2.0
            and not ex_div_guard
        ):
            trim_score += 1
            reasons.append(f"放量下挫(量比{tech.volume_ratio_20d:.2f}, 单日{tech.ret_1d_pct:+.1f}%)")

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

    # 4% cash hurdle + round-trip friction gate
    upside_pct = _compute_analyst_upside_pct(bundle)
    if upside_pct is not None:
        hurdle_pct = float(strategy.cash_interest_rate_pct) + _round_trip_friction_pct(pos, bundle, strategy)
        if upside_pct < hurdle_pct:
            reasons.append(
                f"一致目标价空间仅{upside_pct:+.1f}%(低于{strategy.cash_interest_rate_pct:.1f}%现金+摩擦门槛{hurdle_pct:.1f}%)"
            )
            if add_score > 0:
                add_score = max(0, add_score - 1)

    if earnings_soon and bundle and bundle.earnings_date:
        reasons.append(f"⚠️ 距财报仅{bundle.days_to_earnings}天({bundle.earnings_date}),财报前谨慎重仓加码")
        if add_score > 0:
            add_score = min(max(0, add_score - 1), 1)

    if stop_flag or over_weight or isa_deep_dd or trim_score >= 2:
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
        friction_note=_friction_hint(pos, bundle, strategy),
        native_return_pct=round(native_ret, 2) if native_ret is not None else None,
        effective_stop_pct=effective_stop,
        pending_buy_weight_pct=round(pending_buy_weight_pct, 2),
        has_pending_stop=has_pending_stop,
        ex_div_guarded=ex_div_guard,
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

    return [
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
        )
        for pos in data.all_positions()
    ]


_CONFLICT_HINTS: dict[str, str] = {
    "STOP_VS_EX_DIV": "优先核实除权缺口，避免除息日误砍仓",
    "BULL_VS_OVERWEIGHT": "趋势向好但仓位触及上限，严禁加码并逢高控仓",
    "BULL_VS_EARNINGS_RISK": "多头结构遇财报窗口，暂停主动加仓等待业绩落地",
    "TREND_VS_CROWDED": "趋势多头但短线超买/获利盘拥挤，勿追高可分批兑现",
    "UPSIDE_VS_CASH_HURDLE": "技术形态尚可但预期空间跑不赢4%现金+摩擦，不建议增配",
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
        hurdle_pct = float(strategy.cash_interest_rate_pct) + _round_trip_friction_pct(pos, bundle, strategy)

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
            if upside_pct >= hurdle_pct + 4.0:
                bull_points.append(f"一致目标空间{upside_pct:+.1f}%")
            elif upside_pct < hurdle_pct:
                bear_points.append(f"目标空间{upside_pct:+.1f}%<现金门槛{hurdle_pct:.1f}%")
                conflict_tags.add("UPSIDE_VS_CASH_HURDLE")

        if not is_etf and (pos.weight_pct >= max_acct or issuer_w >= strategy.max_total_portfolio_weight_pct):
            bear_points.append(f"仓位触红线(单户{pos.weight_pct:.1f}%/合并{issuer_w:.1f}%)")
            conflict_tags.add("BULL_VS_OVERWEIGHT")

        if earnings_soon and bundle is not None:
            bear_points.append(f"{bundle.days_to_earnings}日内财报事件风险")
            conflict_tags.add("BULL_VS_EARNINGS_RISK")

        eff_stop = (ev.effective_stop_pct if ev else None) or _compute_adaptive_invest_stop_pct(tech, strategy)
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
    lines = []
    for it in items:
        bull = " / ".join(it.bull_points)
        bear = " / ".join(it.bear_points)
        lines.append(
            f"• [{it.account}] {it.ticker} ({it.conflict_type}) | 多方：{bull} ⚔️ 空方/风险：{bear} → 提示：{it.decision_hint}"
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
    vix_danger_threshold: float = 28.0,
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

    merged_vetoes = list(daily_vetoes or [])
    if hasattr(decision, "daily_vetoes") and getattr(decision, "daily_vetoes"):
        merged_vetoes.extend(getattr(decision, "daily_vetoes"))
    vetoed_map = _extract_vetoed_symbols(merged_vetoes)

    # Precompute current + pending BUY weights by issuer root and by (account, symbol)
    issuer_weights: dict[str, float] = {}
    acct_weights: dict[tuple[str, str], float] = {}
    etf_roots: set[str] = set()
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

    # Index bundles case-insensitively and by base symbol
    bundle_lookup: dict[str, TickerMarketBundle] = {}
    for sym, b in bmap.items():
        bundle_lookup[sym.upper()] = b
        bundle_lookup.setdefault(sym.upper().split(".")[0], b)
        if b.quote_type.upper() == "ETF":
            etf_roots.add(normalize_issuer_root(sym))

    stop_alerts: set[tuple[str, str]] = {
        (e.account.upper(), e.yahoo_symbol.upper())
        for e in (evals or [])
        if e.action_bias == "REVIEW_STOP"
    }

    def _check_symbol_violation(account: str, symbol: str, action: str) -> tuple[str, str] | None:
        sym_up = symbol.strip().upper()
        base = sym_up.split(".")[0]
        acct_up = account.strip().upper()
        act_up = action.strip().upper()
        if act_up not in {"BUY", "ADD", "SWAP"} or sym_up == "CASH":
            return None

        if sym_up in vetoed_map or base in vetoed_map:
            return "DAILY_VETO", f"违背今日早报红线({vetoed_map.get(sym_up) or vetoed_map.get(base)})"

        if sym_up in corr_set or base in corr_bases:
            return "CORRELATION_OVERLAP", f"与现有持仓相关性触发≥{strat.overlap_correlation_threshold:.2f}重叠拦截"

        if act_up in {"BUY", "ADD"}:
            root = normalize_issuer_root(base)
            b = bundle_lookup.get(sym_up) or bundle_lookup.get(base)
            is_etf = bool((b and b.quote_type.upper() == "ETF") or (root in etf_roots))

            if not is_etf:
                issuer_w = issuer_weights.get(root, 0.0)
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

            if (acct_up, sym_up) in stop_alerts or any(
                k[0] == acct_up and _symbol_matches(k[1], sym_up) for k in stop_alerts
            ):
                return "ACCOUNT_OVERWEIGHT", "标的已触发止损复核红线，严禁逆势加仓"

            if b and b.days_to_earnings is not None and 0 <= b.days_to_earnings <= 5:
                return "EARNINGS_WINDOW", f"距财报仅{b.days_to_earnings}天，禁止财报前主动加仓"

            if macro_high_risk and macro_vix is not None:
                return "MACRO_VIX", f"VIX={macro_vix:.1f}(≥{vix_danger_threshold:.0f}高危阈值)，暂停主动买入/加仓"

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
        for item in decision.selected_actions:
            violation = _check_symbol_violation(item.account, item.symbol, item.action)
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
                kept_actions.append(item)
        copied = (
            decision.model_copy(
                update={"selected_actions": kept_actions, "rejected_or_frozen": new_rejected}
            )
            if overrides
            else decision.model_copy()
        )
        return RiskOverrideResult(
            decision=copied,
            applied=bool(overrides),
            overrides=overrides,
            notes=notes,
        )

    return RiskOverrideResult(decision=decision, applied=False)


_DEFAULT_HOLD_REASON = "技术面与仓位均在常规区间,默认持有观望"


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
        is_routine_hold = (
            compact_hold
            and ev.action_bias == "HOLD"
            and len(ev.reasons) == 1
            and ev.reasons[0] == _DEFAULT_HOLD_REASON
        )
        if is_routine_hold:
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
