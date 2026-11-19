"""FX-distortion-free market regime evaluation with confirmation hysteresis."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.market_data import MacroSnapshot
from src.engine.portfolio_rules import DEFAULT_VIX_DANGER_THRESHOLD
from src.engine.deployment_gate.common import (
    REGIME_HY_OAS_RISK_ON_PCT,
    _close_series,
    _round2,
    _sym_up,
)

# ---------------------------------------------------------------------------
# 3. FX-distortion-free market regime with anti-deadlock confirmation
# ---------------------------------------------------------------------------
def _compute_raw_regime_metrics(
    closes: pd.Series,
    vix: float | None,
    hy_oas: float | None,
    strategy: StrategySection,
) -> dict[str, Any]:
    """Raw RISK_OFF / RISK_ON / NEUTRAL classification from one daily bar snapshot."""
    price = float(closes.iloc[-1])
    ma200 = float(closes.rolling(200, min_periods=200).mean().iloc[-1]) if len(closes) >= 200 else math.nan
    ma50 = float(closes.rolling(50, min_periods=50).mean().iloc[-1]) if len(closes) >= 50 else math.nan
    mom20 = None
    if len(closes) >= 21:
        mom20 = (price / float(closes.iloc[-21]) - 1.0) * 100.0
    high52 = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
    dd52 = (price / high52 - 1.0) * 100.0 if high52 > 0 else 0.0

    band = float(strategy.regime_ma200_band_pct)
    risk_off_flags: list[str] = []
    if not math.isnan(ma200) and ma200 > 0 and price < ma200 * (1.0 - band / 100.0):
        risk_off_flags.append(f"基准实质跌破MA200缓冲带{band:.0f}%")
    if vix is not None and vix >= DEFAULT_VIX_DANGER_THRESHOLD:
        risk_off_flags.append(f"VIX={vix:.1f}≥{DEFAULT_VIX_DANGER_THRESHOLD:.0f}")
    if dd52 <= -float(strategy.regime_drawdown_risk_off_pct):
        risk_off_flags.append(f"距52周高点回撤{dd52:.1f}%≥{strategy.regime_drawdown_risk_off_pct:.0f}%")
    if hy_oas is not None and hy_oas >= float(strategy.regime_hy_oas_risk_off_pct):
        risk_off_flags.append(f"HY OAS {hy_oas:.2f}%≥{strategy.regime_hy_oas_risk_off_pct:.1f}%")

    risk_on_ok = bool(
        (not math.isnan(ma200)) and price > ma200
        and (not math.isnan(ma50)) and ma50 > ma200
        and dd52 > -float(strategy.regime_drawdown_neutral_pct)
        and vix is not None and vix < float(strategy.regime_vix_risk_on)
        and hy_oas is not None and hy_oas < REGIME_HY_OAS_RISK_ON_PCT
    )
    if risk_off_flags:
        raw = "RISK_OFF"
    elif risk_on_ok:
        raw = "RISK_ON"
    else:
        raw = "NEUTRAL"
    return {
        "raw_regime": raw,
        "risk_off_flags": risk_off_flags,
        "price": _round2(price),
        "ma50": _round2(ma50) if not math.isnan(ma50) else None,
        "ma200": _round2(ma200) if not math.isnan(ma200) else None,
        "mom20_pct": _round2(mom20) if mom20 is not None else None,
        "drawdown_52w_pct": _round2(dd52),
        "vix": _round2(vix) if vix is not None else None,
        "hy_oas": _round2(hy_oas) if hy_oas is not None else None,
    }


def evaluate_market_regime(
    histories: dict[str, pd.DataFrame] | None,
    macro: MacroSnapshot | None,
    strategy: StrategySection,
    prev_gate_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anchored on the USD-native regime benchmark (SPY), never the GBP-denominated VWRP.L.

    Enter RISK_OFF fast (1 day), exit step-wise (3 days of crisis easing; VIX < 25 suffices —
    never requires VIX < 18) and enter RISK_ON only after 3 stable days, so a VIX stuck at
    18~25 in a bull run can never deadlock the system in RISK_OFF.
    """
    degraded: list[str] = []
    bench_sym = str(getattr(strategy, "regime_benchmark_symbol", "SPY") or "SPY").strip().upper()
    hist = (histories or {}).get(bench_sym)
    if hist is None or hist.empty:
        bench_sym = _sym_up(strategy.benchmark_symbol)
        hist = (histories or {}).get(bench_sym)
    closes = _close_series(hist)
    vix = float(macro.vix) if (macro is not None and macro.vix is not None) else None
    hy_oas = float(macro.hy_oas_spread_pct) if (macro is not None and macro.hy_oas_spread_pct is not None) else None
    stale = bool(macro is not None and macro.is_stale)

    if len(closes) < 60:
        return {
            "regime": "UNKNOWN",
            "raw_regime": "UNKNOWN",
            "confirmed": False,
            "reason": f"基准 {bench_sym} 行情不足(仅{len(closes)}根)，市场状态降级 UNKNOWN",
            "benchmark_symbol": bench_sym,
            "degraded_notes": ["基准行情数据不足"],
        }

    metrics = _compute_raw_regime_metrics(closes, vix, hy_oas, strategy)
    raw = metrics["raw_regime"]
    if stale and raw == "RISK_ON":
        raw = "NEUTRAL"
        metrics["raw_regime"] = "NEUTRAL"
        degraded.append("宏观数据过期，严禁盲目判 RISK_ON，保守降级 NEUTRAL")

    prev = prev_gate_state if isinstance(prev_gate_state, dict) else {}
    prev_raw = prev.get("raw_regime")
    prev_regime = prev.get("regime")
    prev_raw_streak = int(prev.get("raw_streak_days") or 0)
    prev_ease_streak = int(prev.get("ease_streak_days") or 0)

    raw_streak = (prev_raw_streak + 1) if (prev_raw == raw and prev_regime is not None) else 1

    price = metrics["price"]
    ma200 = metrics["ma200"]
    mom20 = metrics["mom20_pct"]
    # A 3-day dead-cat bounce while the index is still below MA200 must never release DEFEND.
    # Only when MA200 is genuinely unavailable (insufficient bars) may momentum stand in.
    trend_repaired = bool(
        (ma200 is not None and price >= ma200)
        or (ma200 is None and mom20 is not None and mom20 > 0)
    )
    crisis_eased = bool(
        (vix is None or vix < float(strategy.regime_vix_exit_risk_off))
        and (hy_oas is None or hy_oas < float(strategy.regime_hy_oas_risk_off_pct))
        and trend_repaired
    )
    ease_streak = (
        (prev_ease_streak + 1)
        if (prev_regime == "RISK_OFF" and crisis_eased)
        else (1 if crisis_eased else 0)
    )

    enter_days = int(strategy.regime_confirm_enter_days)
    exit_days = int(strategy.regime_confirm_exit_days)
    regime: str
    if prev_regime is None:
        if raw == "RISK_OFF":
            regime = "RISK_OFF"
        elif raw == "RISK_ON" and raw_streak >= exit_days:
            regime = "RISK_ON"
        else:
            regime = "NEUTRAL"
    elif raw == "RISK_OFF" and raw_streak >= enter_days and prev_regime in {"NEUTRAL", "RISK_ON"}:
        regime = "RISK_OFF"
    elif prev_regime == "RISK_OFF":
        # A -25% bear-market bottom that has repaired above MA200 for the full exit window
        # must not stay locked in RISK_OFF because the stale 52w-high drawdown flag echoes.
        flags = [str(f) for f in metrics.get("risk_off_flags") or []]
        only_stale_drawdown = bool(raw == "RISK_OFF" and flags and all("回撤" in f for f in flags))
        if raw != "RISK_OFF" and ease_streak >= exit_days:
            regime = "NEUTRAL"
        elif only_stale_drawdown and ease_streak >= exit_days:
            regime = "NEUTRAL"
        else:
            regime = "RISK_OFF"
    elif prev_regime == "RISK_ON":
        regime = "RISK_ON" if raw == "RISK_ON" else "NEUTRAL"
    else:  # NEUTRAL
        regime = "RISK_ON" if (raw == "RISK_ON" and raw_streak >= exit_days) else "NEUTRAL"

    reason = "；".join(metrics["risk_off_flags"]) if raw == "RISK_OFF" else (
        "强右侧趋势(MA50>MA200且回撤<8%且VIX<18)" if raw == "RISK_ON" else "常规回调/震荡分化"
    )
    return {
        "regime": regime,
        "raw_regime": raw,
        "confirmed": bool(regime == raw or (prev_regime == "RISK_OFF" and regime == "NEUTRAL")),
        "raw_streak_days": raw_streak,
        "ease_streak_days": ease_streak,
        "crisis_eased": crisis_eased,
        "reason": reason,
        "benchmark_symbol": bench_sym,
        "degraded_notes": degraded,
        **metrics,
    }
