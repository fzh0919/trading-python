"""FX-distortion-free four-state market regime evaluation with breadth fusion and hysteresis."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.market_data import MacroSnapshot
from src.engine.portfolio_rules import DEFAULT_VIX_DANGER_THRESHOLD
from src.engine.deployment_gate.common import (
    REGIME_BREADTH_MIN_CACHED_EQUITIES,
    REGIME_BREADTH_MIN_POOL_SNAPS,
    REGIME_BREADTH_MIN_REGIONS,
    REGIME_BREADTH_REGIONS,
    REGIME_HY_OAS_RISK_ON_PCT,
    _close_series,
    _round2,
    _sym_up,
)


def _lookup_history_ci(
    mapping: dict[str, pd.DataFrame] | None,
    symbol: str,
) -> pd.DataFrame | None:
    if not mapping:
        return None
    sym_up = _sym_up(symbol)
    if symbol in mapping:
        return mapping[symbol]
    if sym_up in mapping:
        return mapping[sym_up]
    for k, v in mapping.items():
        if _sym_up(k) == sym_up:
            return v
    return None


def _score_benchmark_closes(closes: pd.Series) -> float | None:
    if len(closes) < 60:
        return None
    price = float(closes.iloc[-1])
    ma50 = float(closes.rolling(50, min_periods=50).mean().iloc[-1])
    ma200 = float(closes.rolling(200, min_periods=200).mean().iloc[-1]) if len(closes) >= 200 else math.nan
    above_ma50 = price > ma50
    above_ma200 = (not math.isnan(ma200)) and price > ma200
    if above_ma200 and above_ma50:
        val = 90.0
    elif above_ma200:
        val = 60.0
    elif (not math.isnan(ma200)) and (not above_ma200) and (not above_ma50):
        val = 15.0
    else:
        val = 45.0
    if len(closes) >= 61:
        rets = closes.iloc[-61:].pct_change().dropna()
        if len(rets) >= 20:
            vol60 = float(rets.std()) * math.sqrt(252.0) * 100.0
            if math.isfinite(vol60) and vol60 > 25.0:
                val = max(0.0, val - 15.0)
    return val


def compute_regime_breadth(
    histories: dict[str, pd.DataFrame] | None,
    *,
    pool_snaps: dict[str, Any] | None = None,
    cached_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Zero-extra-network two-layer market breadth (0~100), degrading to None when unavailable.

    Layer 1 (65% weight): local equity participation — prefers fresh SQLite ``ohlcv_cache``
    histories (up to 150 active equities), falling back to ``pool_snaps`` already computed
    in /report.
    Layer 2 (35% weight): non-redundant regional benchmarks (US / UK / EU) from already-fetched
    2y histories. Never synthesizes a fake 50.0 default when both layers are absent.
    """
    pool_score: float | None = None
    pool_source = ""
    pool_count = 0
    above_ma50_pct: float | None = None
    bullish_pct: float | None = None

    if cached_histories:
        valid_eq = 0
        above_ma50_n = 0
        bullish_n = 0
        for df in cached_histories.values():
            closes = _close_series(df)
            if len(closes) < 60:
                continue
            price = float(closes.iloc[-1])
            ma50 = float(closes.iloc[-50:].mean())
            is_above_ma50 = price > ma50
            if len(closes) >= 200:
                ma200 = float(closes.iloc[-200:].mean())
                is_bullish = price > ma200
            else:
                is_bullish = is_above_ma50
            valid_eq += 1
            if is_above_ma50:
                above_ma50_n += 1
            if is_bullish:
                bullish_n += 1
        if valid_eq >= REGIME_BREADTH_MIN_CACHED_EQUITIES:
            above_ma50_pct = (above_ma50_n / valid_eq) * 100.0
            bullish_pct = (bullish_n / valid_eq) * 100.0
            pool_score = 0.5 * above_ma50_pct + 0.5 * bullish_pct
            pool_source = "ohlcv_cache"
            pool_count = valid_eq

    if pool_score is None and pool_snaps:
        snaps = [s for s in pool_snaps.values() if s is not None]
        if len(snaps) >= REGIME_BREADTH_MIN_POOL_SNAPS:
            above_ma50_n = sum(
                1 for s in snaps if getattr(s, "dist_ma50_pct", None) is not None and float(s.dist_ma50_pct) > 0.0
            )
            # Match ohlcv_cache layer 1-to-1: use price > MA200 (dist_ma200_pct > 0), falling back to
            # dist_ma50_pct > 0 only when MA200 is unavailable, preventing a 5~10pt systematic shift.
            bullish_n = sum(
                1
                for s in snaps
                if (
                    (getattr(s, "dist_ma200_pct", None) is not None and float(s.dist_ma200_pct) > 0.0)
                    or (
                        getattr(s, "dist_ma200_pct", None) is None
                        and getattr(s, "dist_ma50_pct", None) is not None
                        and float(s.dist_ma50_pct) > 0.0
                    )
                )
            )
            above_ma50_pct = (above_ma50_n / len(snaps)) * 100.0
            bullish_pct = (bullish_n / len(snaps)) * 100.0
            pool_score = 0.5 * above_ma50_pct + 0.5 * bullish_pct
            pool_source = "pool_snaps"
            pool_count = len(snaps)

    region_scores: dict[str, float] = {}
    for region, syms in REGIME_BREADTH_REGIONS:
        vals: list[float] = []
        for sym in syms:
            df = _lookup_history_ci(histories, sym)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                df = _lookup_history_ci(cached_histories, sym)
            closes = _close_series(df)
            sc = _score_benchmark_closes(closes)
            if sc is not None:
                vals.append(sc)
        if vals:
            region_scores[region] = sum(vals) / len(vals)

    bench_score: float | None = (
        (sum(region_scores.values()) / len(region_scores))
        if len(region_scores) >= REGIME_BREADTH_MIN_REGIONS
        else None
    )

    if pool_score is not None and bench_score is not None:
        combined: float | None = round(0.65 * pool_score + 0.35 * bench_score, 1)
        source = f"{pool_source}+benchmarks"
    elif pool_score is not None:
        combined = round(pool_score, 1)
        source = pool_source
    elif bench_score is not None:
        combined = round(bench_score, 1)
        source = "benchmarks_only"
    else:
        combined = None
        source = "unavailable"

    return {
        "score": combined,
        "source": source,
        "pool_count": pool_count,
        "pool_score": round(pool_score, 1) if pool_score is not None else None,
        "above_ma50_pct": round(above_ma50_pct, 1) if above_ma50_pct is not None else None,
        "bullish_pct": round(bullish_pct, 1) if bullish_pct is not None else None,
        "bench_score": round(bench_score, 1) if bench_score is not None else None,
        "regions": sorted(region_scores.keys()),
    }


# ---------------------------------------------------------------------------
# 3. FX-distortion-free four-state market regime with anti-deadlock confirmation
# ---------------------------------------------------------------------------
def _compute_raw_regime_metrics(
    closes: pd.Series,
    vix: float | None,
    hy_oas: float | None,
    strategy: StrategySection,
    *,
    breadth_score: float | None = None,
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

    breadth_min = float(getattr(strategy, "regime_breadth_risk_on_min", 55.0))
    breadth_ok = (breadth_score is None) or (float(breadth_score) >= breadth_min)
    risk_on_core_ok = bool(
        (not math.isnan(ma200)) and price > ma200
        and (not math.isnan(ma50)) and ma50 > ma200
        and dd52 > -float(strategy.regime_drawdown_neutral_pct)
        and vix is not None and vix < float(strategy.regime_vix_risk_on)
        and hy_oas is not None and hy_oas < REGIME_HY_OAS_RISK_ON_PCT
    )
    risk_on_ok = bool(risk_on_core_ok and breadth_ok)
    breadth_blocked_risk_on = bool((not risk_off_flags) and risk_on_core_ok and (not breadth_ok))
    if risk_off_flags:
        raw = "RISK_OFF"
    elif risk_on_ok:
        raw = "RISK_ON"
    else:
        raw = "NEUTRAL"
    return {
        "raw_regime": raw,
        "risk_off_flags": risk_off_flags,
        "breadth_blocked_risk_on": breadth_blocked_risk_on,
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
    *,
    pool_snaps: dict[str, Any] | None = None,
    cached_breadth_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Anchored on the USD-native regime benchmark (SPY), never the GBP-denominated VWRP.L.

    Four-state Regime 2.0:
    - Enter RISK_OFF fast (1 day).
    - Exit RISK_OFF to NEUTRAL when above MA200 for 3 eased days (VIX < 25, HY OAS < 5.5%);
      or enter RECOVERY early-spring pilot when still below MA200 but SPY > MA50 + VIX < 24
      + HY OAS < 5.2% (+ breadth >= 45 if available) holds for 3 days (filters 2-day dead-cat bounces).
    - Relapse RECOVERY -> RISK_OFF in 1 day if SPY < MA50 or VIX >= 26 or HY OAS >= 5.5%.
    - Enter RISK_ON after 3 stable days (VIX < 20, breadth >= 55 if available); exit RISK_ON -> NEUTRAL
      with 2-day hysteresis (except immediate 1-day drop when VIX >= 22, macro stale, or RISK_OFF).
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
            "breadth": {"score": None, "source": "unavailable", "pool_count": 0, "regions": []},
        }

    breadth = compute_regime_breadth(
        histories,
        pool_snaps=pool_snaps,
        cached_histories=cached_breadth_histories,
    )
    breadth_score = breadth.get("score")
    if breadth_score is None:
        degraded.append("市场宽度数据未命中本地池或多地基准，已中性降级放行")

    metrics = _compute_raw_regime_metrics(
        closes,
        vix,
        hy_oas,
        strategy,
        breadth_score=breadth_score,
    )
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
    prev_risk_on_exit_streak = int(prev.get("risk_on_exit_streak_days") or 0)
    prev_recovery_streak = int(prev.get("recovery_streak_days") or 0)

    raw_streak = (prev_raw_streak + 1) if (prev_raw == raw and prev_regime is not None) else 1

    price = metrics["price"]
    ma50 = metrics["ma50"]
    ma200 = metrics["ma200"]
    mom20 = metrics["mom20_pct"]
    # A 3-day dead-cat bounce while the index is still below MA200 must never release full NEUTRAL.
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
        if (prev_regime in {"RISK_OFF", "RECOVERY"} and crisis_eased)
        else (1 if crisis_eased else 0)
    )

    vix_recovery = float(getattr(strategy, "regime_vix_recovery", 24.0))
    vix_recovery_relapse = float(getattr(strategy, "regime_vix_recovery_relapse", 26.0))
    hy_oas_recovery = float(getattr(strategy, "regime_hy_oas_recovery_pct", 5.2))
    breadth_recovery_min = float(getattr(strategy, "regime_breadth_recovery_min", 45.0))
    above_ma50 = bool(
        (ma50 is not None and price > ma50)
        or (ma50 is None and mom20 is not None and mom20 > 0)
    )
    breadth_recovery_ok = (breadth_score is None) or (float(breadth_score) >= breadth_recovery_min)
    recovery_candidate = bool(
        (not stale)
        and (not trend_repaired)
        and above_ma50
        and (vix is not None and vix < vix_recovery)
        and (hy_oas is None or hy_oas < hy_oas_recovery)
        and breadth_recovery_ok
    )
    recovery_streak = (
        (prev_recovery_streak + 1)
        if (prev_regime == "RISK_OFF" and recovery_candidate)
        else 0
    )

    enter_days = int(strategy.regime_confirm_enter_days)
    exit_days = int(strategy.regime_confirm_exit_days)
    risk_on_exit_days = int(getattr(strategy, "regime_risk_on_exit_days", 2))
    vix_fast_exit = float(getattr(strategy, "regime_vix_risk_on_fast_exit", 22.0))
    risk_on_exit_streak = 0
    flags = [str(f) for f in metrics.get("risk_off_flags") or []]
    only_stale_drawdown = bool(raw == "RISK_OFF" and flags and all("回撤" in f for f in flags))

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
        # exits directly to NEUTRAL; while still below MA200, 3 days above MA50 with eased
        # volatility/credit unlocks the 0.6R / 1-slot RECOVERY early-spring pilot.
        if (raw != "RISK_OFF" or only_stale_drawdown) and ease_streak >= exit_days:
            regime = "NEUTRAL"
        elif recovery_candidate and recovery_streak >= exit_days:
            regime = "RECOVERY"
        else:
            regime = "RISK_OFF"
    elif prev_regime == "RECOVERY":
        recovery_relapse = bool(
            (vix is not None and vix >= vix_recovery_relapse)
            or (hy_oas is not None and hy_oas >= float(strategy.regime_hy_oas_risk_off_pct))
            or (
                (not trend_repaired)
                and (
                    (ma50 is not None and price < ma50)
                    or (ma50 is None and mom20 is not None and mom20 <= 0)
                )
            )
        )
        if recovery_relapse:
            regime = "RISK_OFF"
        elif (raw != "RISK_OFF" or only_stale_drawdown) and ease_streak >= exit_days:
            regime = "NEUTRAL"
        else:
            regime = "RECOVERY"
    elif prev_regime == "RISK_ON":
        if raw == "RISK_ON":
            regime = "RISK_ON"
        elif stale or (vix is not None and vix >= vix_fast_exit):
            regime = "NEUTRAL"
        else:
            cand_exit = prev_risk_on_exit_streak + 1
            if cand_exit >= risk_on_exit_days:
                regime = "NEUTRAL"
            else:
                regime = "RISK_ON"
                risk_on_exit_streak = cand_exit
    else:  # NEUTRAL
        regime = "RISK_ON" if (raw == "RISK_ON" and raw_streak >= exit_days) else "NEUTRAL"

    if regime != "RISK_OFF":
        recovery_streak = 0

    vix_risk_on = float(strategy.regime_vix_risk_on)
    breadth_risk_on_min = float(getattr(strategy, "regime_breadth_risk_on_min", 55.0))
    if regime == "RECOVERY":
        breadth_txt = f"，宽度{breadth_score:.1f}" if breadth_score is not None else ""
        rec_slots = int(getattr(strategy, "invest_slots_recovery", 1))
        rec_scale = float(getattr(strategy, "invest_recovery_risk_scale", 0.6) or 0.6)
        if prev_regime == "RECOVERY":
            vix_txt = f"VIX={vix:.1f}<急退线{vix_recovery_relapse:.0f}" if vix is not None else f"VIX<急退线{vix_recovery_relapse:.0f}"
            reason = (
                f"早春试探维持({bench_sym}稳居MA50上方，{vix_txt}{breadth_txt}，"
                f"{rec_slots}槽位/{rec_scale:.1f}R轻兵)"
            )
        else:
            reason = (
                f"早春试探({bench_sym}站上MA50且VIX<{vix_recovery:.0f}连稳{exit_days}天{breadth_txt}，"
                f"{rec_slots}槽位/{rec_scale:.1f}R轻兵)"
            )
    elif raw == "RISK_OFF":
        reason = "；".join(metrics["risk_off_flags"])
    elif raw == "RISK_ON":
        breadth_txt = f"且宽度{breadth_score:.1f}≥{breadth_risk_on_min:.0f}" if breadth_score is not None else ""
        reason = (
            f"强右侧趋势(MA50>MA200且回撤<{strategy.regime_drawdown_neutral_pct:.0f}%"
            f"且VIX<{vix_risk_on:.0f}{breadth_txt})"
        )
    elif metrics.get("breadth_blocked_risk_on") and breadth_score is not None:
        reason = f"窄宽度背离(宽度{breadth_score:.1f}<{breadth_risk_on_min:.0f})，压制 RISK_ON 维持 NEUTRAL"
    else:
        reason = "常规回调/震荡分化"
    if prev_regime == "RISK_ON" and regime == "RISK_ON" and risk_on_exit_streak > 0:
        reason = f"{reason}（RISK_ON 退档滞回第{risk_on_exit_streak}/{risk_on_exit_days}天，暂留强攻）"

    return {
        "regime": regime,
        "raw_regime": raw,
        "confirmed": bool(
            regime == raw
            or (prev_regime in {"RISK_OFF", "RECOVERY"} and regime in {"RECOVERY", "NEUTRAL"})
        ),
        "raw_streak_days": raw_streak,
        "ease_streak_days": ease_streak,
        "risk_on_exit_streak_days": risk_on_exit_streak,
        "recovery_streak_days": recovery_streak,
        "crisis_eased": crisis_eased,
        "recovery_candidate": recovery_candidate,
        "breadth": breadth,
        "reason": reason,
        "benchmark_symbol": bench_sym,
        "degraded_notes": degraded,
        **metrics,
    }
