"""Quantitative breakout patterns, Stage2/VCP/VPVR/RS indicators, market breadth, and marginal risk."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.data_providers.analytics import (
    TechnicalSnapshot,
    _normalize_adjusted_ohlcv,
    _resolve_col,
    compute_technical_snapshot,
)
from src.data_providers.market_data import normalize_history_index
from src.data_providers.t212_client import PortfolioData

def _normalize_regime_tag(value: str) -> str:
    """Normalize screener breadth tags (RISK-ON) and gate regimes (RISK_ON) to one vocabulary."""
    raw = str(value or "").strip().upper().replace("_", "-")
    if raw in {"RISK-ON", "RISKON"}:
        return "RISK-ON"
    if raw in {"RISK-OFF", "RISKOFF"}:
        return "RISK-OFF"
    return "NEUTRAL"


class ScreenerCandidate(BaseModel):
    symbol: str
    track: str  # ISA_LONG_TERM | INVEST_TACTICAL
    name: str = ""
    asset_type: str = "EQUITY"  # ETF | EQUITY
    currency: str = ""
    score: float
    is_already_held: bool = False
    is_watchlist: bool = False
    is_recent_traded: bool = False
    max_portfolio_corr: float | None = None
    most_correlated_holding: str | None = None
    overlap_warning: bool = False
    rs_20d_pct: float | None = None
    marginal_vol_delta_pct: float | None = None
    regime_tag: str = ""
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    dimension_pct: dict[str, float] = Field(default_factory=dict)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    pattern_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    summary_line: str = ""
    # v5.5.7: 0-token diagnostic metrics surfaced to the DeepSeek ID filter (never used in scoring).
    filter_metrics: dict[str, Any] = Field(default_factory=dict)


def _linear_scale(
    value: float,
    low_val: float,
    high_val: float,
    low_score: float,
    high_score: float,
) -> float:
    """Linearly map `value` from [low_val, high_val] to [low_score, high_score], clamped."""
    if high_val == low_val:
        return float(high_score)
    t = (float(value) - float(low_val)) / (float(high_val) - float(low_val))
    t = max(0.0, min(1.0, t))
    return float(low_score) + t * (float(high_score) - float(low_score))


def _extract_ohlcv_series(
    hist: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Return cleaned (open, high, low, close, volume) float series aligned by index."""
    if hist is None or hist.empty:
        return None
    norm = _normalize_adjusted_ohlcv(hist)
    close_col = _resolve_col(norm, "Close")
    if not close_col:
        return None
    open_col = _resolve_col(norm, "Open")
    high_col = _resolve_col(norm, "High")
    low_col = _resolve_col(norm, "Low")
    vol_col = _resolve_col(norm, "Volume")

    df = pd.DataFrame({"close": pd.to_numeric(norm[close_col], errors="coerce")})
    df["open"] = pd.to_numeric(norm[open_col], errors="coerce") if open_col else df["close"]
    df["high"] = pd.to_numeric(norm[high_col], errors="coerce") if high_col else df["close"]
    df["low"] = pd.to_numeric(norm[low_col], errors="coerce") if low_col else df["close"]
    df["volume"] = pd.to_numeric(norm[vol_col], errors="coerce") if vol_col else 0.0
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    if df.empty:
        return None
    norm_idx = normalize_history_index(df.index)
    if norm_idx is not None:
        df.index = norm_idx
        df = df[df.index.notna()]
        if df.empty:
            return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return (
        df["open"].fillna(df["close"]).astype(float),
        df["high"].fillna(df["close"]).astype(float),
        df["low"].fillna(df["close"]).astype(float),
        df["close"].astype(float),
        df["volume"].fillna(0.0).astype(float),
    )


def detect_breakthrough_platform(hist: pd.DataFrame | None, window: int = 60) -> bool:
    """Detect REF_stock-master platform breakout: tight MA60 base followed by volume breakout."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return False
    opens, _, _, closes, vols = extracted
    if len(closes) < window + 5 or float(vols.tail(window).sum()) <= 0.0:
        return False

    ma60 = closes.rolling(60, min_periods=60).mean()
    vol_ma5_prev = vols.rolling(5, min_periods=5).mean().shift(1)
    pct_chg = closes.pct_change() * 100.0

    o_arr = opens.tail(window).to_numpy(dtype=np.float64)
    c_arr = closes.tail(window).to_numpy(dtype=np.float64)
    v_arr = vols.tail(window).to_numpy(dtype=np.float64)
    m_arr = ma60.tail(window).to_numpy(dtype=np.float64)
    vm_arr = vol_ma5_prev.tail(window).to_numpy(dtype=np.float64)
    chg_arr = pct_chg.tail(window).to_numpy(dtype=np.float64)

    last_close = float(c_arr[-1])
    last_ma60 = float(m_arr[-1]) if not np.isnan(m_arr[-1]) else 0.0
    if last_ma60 <= 0.0 or last_close < last_ma60 * 0.98:
        return False

    for idx in range(len(c_arr)):
        ma_val = m_arr[idx]
        v_mean = vm_arr[idx]
        if np.isnan(ma_val) or ma_val <= 0.0 or np.isnan(v_mean) or v_mean <= 0.0:
            continue
        prev_c = c_arr[idx - 1] if idx > 0 else o_arr[idx]
        crossed = (o_arr[idx] < ma_val <= c_arr[idx]) or (prev_c < ma_val <= c_arr[idx])
        if not crossed:
            continue
        if c_arr[idx] <= o_arr[idx] or np.isnan(chg_arr[idx]) or chg_arr[idx] < 1.5:
            continue
        if (v_arr[idx] / v_mean) < 1.8:
            continue

        # Verify prior valid bars inside the window consolidated around MA60 (-5%..+20% below MA60)
        valid_prior = 0
        platform_ok = True
        for j in range(idx):
            prior_ma = m_arr[j]
            if np.isnan(prior_ma) or prior_ma <= 0.0:
                continue
            valid_prior += 1
            rel_gap = (prior_ma - c_arr[j]) / prior_ma
            if not (-0.05 < rel_gap < 0.20):
                platform_ok = False
                break
        if platform_ok and valid_prior >= 5:
            return True

    return False


def detect_low_atr_contraction(
    hist: pd.DataFrame | None,
    tech: TechnicalSnapshot,
    min_bars: int = 120,
) -> bool:
    """Detect low-ATR volatility compression (REF_stock-master low_atr + BB squeeze coiling)."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return False
    _, highs, lows, closes, _ = extracted
    if len(closes) < min_bars:
        return False
    if tech.atr_pct is None or not (0.0 < tech.atr_pct <= 3.0):
        return False
    if tech.ma50 is not None and tech.last_close < tech.ma50 * 0.98:
        return False

    c10 = closes.tail(10)
    c10_min = float(c10.min())
    c10_max = float(c10.max())
    if c10_min <= 0.0 or c10_max <= c10_min:
        return False
    range_10d = (c10_max - c10_min) / c10_min
    if range_10d > 0.10:
        return False

    if tech.bb_squeeze:
        return True

    prev_close = closes.shift(1)
    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14_pct = (tr.rolling(14, min_periods=14).mean() / closes) * 100.0
    valid_atr = atr14_pct.dropna()
    if len(valid_atr) < 60:
        return False
    base_atr_pct = float(valid_atr.tail(60).mean())
    curr_atr_pct = float(valid_atr.iloc[-1])
    return bool(base_atr_pct > 0.0 and curr_atr_pct <= base_atr_pct * 0.85)


def detect_volume_ma_breakout(hist: pd.DataFrame | None, tech: TechnicalSnapshot) -> bool:
    """Detect REF_stock-master enter.py pattern: up-day volume surge reclaiming MA20/MA50."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return False
    opens, _, _, closes, vols = extracted
    if len(closes) < 25 or float(vols.iloc[-1]) <= 0.0:
        return False

    last_open = float(opens.iloc[-1])
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    if prev_close <= 0.0:
        return False
    day_chg_pct = ((last_close / prev_close) - 1.0) * 100.0
    if day_chg_pct < 1.5 or last_close < last_open:
        return False

    vol_ma5_prev = float(vols.iloc[-6:-1].mean())
    vol_ratio_5d = (float(vols.iloc[-1]) / vol_ma5_prev) if vol_ma5_prev > 0.0 else 0.0
    if vol_ratio_5d < 1.8 and (tech.volume_ratio_20d is None or tech.volume_ratio_20d < 1.8):
        return False

    ma20 = closes.rolling(20, min_periods=20).mean()
    if not np.isnan(ma20.iloc[-1]) and not np.isnan(ma20.iloc[-2]):
        if (prev_close <= float(ma20.iloc[-2]) or last_open < float(ma20.iloc[-1])) and last_close >= float(ma20.iloc[-1]):
            return True

    if len(closes) >= 51:
        ma50 = closes.rolling(50, min_periods=50).mean()
        if not np.isnan(ma50.iloc[-1]) and not np.isnan(ma50.iloc[-2]):
            if (prev_close <= float(ma50.iloc[-2]) or last_open < float(ma50.iloc[-1])) and last_close >= float(ma50.iloc[-1]):
                return True

    return False


def detect_minervini_stage2(
    hist: pd.DataFrame | None,
    tech: TechnicalSnapshot,
    rs_60d_pct: float | None = None,
) -> dict[str, Any]:
    """Evaluate Minervini 8-criteria Trend Template & Weinstein Phase (1-4) cleanly in pure pandas."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return {"phase": 0, "passes": False, "criteria_passed": 0, "criteria_total": 0}
    _, highs, lows, closes, _ = extracted
    n_bars = len(closes)
    if n_bars < 50:
        return {"phase": 0, "passes": False, "criteria_passed": 0, "criteria_total": 0}

    price = float(closes.iloc[-1])
    sma50_s = closes.rolling(50, min_periods=50).mean()
    sma50 = float(sma50_s.iloc[-1])
    sma50_prev = float(sma50_s.iloc[-21]) if len(sma50_s.dropna()) >= 21 else float(sma50_s.dropna().iloc[0])

    # Fallback gracefully when a newly listed stock has <200 bars so Stage 1 doesn't drop it silently
    long_win = min(200, n_bars)
    mid_win = min(150, n_bars)
    sma150 = float(closes.rolling(mid_win, min_periods=mid_win).mean().iloc[-1])
    sma200_s = closes.rolling(long_win, min_periods=long_win).mean().dropna()
    if sma200_s.empty:
        return {"phase": 0, "passes": False, "criteria_passed": 0, "criteria_total": 0}
    sma200 = float(sma200_s.iloc[-1])
    sma200_prev = float(sma200_s.iloc[-21]) if len(sma200_s) >= 21 else float(sma200_s.iloc[0])

    w52_high = float(highs.tail(252).max())
    w52_low = float(lows.tail(252).min())

    # Phase 4 (Downtrend) checked first regardless of template
    if price < sma50 and price < sma200 and sma50 < sma200:
        phase = 4
    elif price > sma50 and sma50 > sma200 and sma50 >= sma50_prev and sma200 >= sma200_prev * 0.999:
        phase = 2
    elif price < sma50 and sma50 >= sma200 and price >= sma200 * 0.97:
        phase = 3  # Distribution / extended pullback near top
    else:
        phase = 1

    criteria = [
        price > sma150 and price > sma200,
        sma150 >= sma200,
        sma200 > sma200_prev,
        sma50 >= sma150 and sma50 > sma200,
        price > sma50,
        w52_low > 0.0 and price >= w52_low * 1.30,
        w52_high > 0.0 and price >= w52_high * 0.75,
    ]
    if rs_60d_pct is not None:
        criteria.append(rs_60d_pct > 0.0)
    elif tech.ret_60d_pct is not None:
        criteria.append(tech.ret_60d_pct > 0.0)

    passed = sum(1 for c in criteria if c)
    total = len(criteria)
    # Require >= 7 of 8 (or >= total - 1 when short RS absent) and Stage 2 alignment
    passes = passed >= max(6, total - 1) and phase == 2
    return {
        "phase": phase,
        "passes": bool(passes),
        "criteria_passed": passed,
        "criteria_total": total,
    }


def detect_vcp_contraction(
    hist: pd.DataFrame | None,
    tech: TechnicalSnapshot | None = None,
    window: int = 120,
) -> bool:
    """Detect Mark Minervini VCP: progressively tighter price contractions with volume dry-up."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return False
    _, highs, lows, closes, vols = extracted
    if len(closes) < 60:
        return False

    h_win = highs.tail(window).to_numpy(dtype=np.float64)
    l_win = lows.tail(window).to_numpy(dtype=np.float64)
    c_win = closes.tail(window).to_numpy(dtype=np.float64)
    v_win = vols.tail(window).to_numpy(dtype=np.float64)
    if float(np.sum(v_win[-40:])) <= 0.0:
        return False

    last_close = float(c_win[-1])
    win_high = float(np.max(h_win))
    if win_high <= 0.0 or last_close < win_high * 0.78:
        return False
    if tech is not None and tech.ma50 is not None and last_close < tech.ma50 * 0.96:
        return False

    # Divide the base into 3 sequential stages and require monotonically tightening depth
    seg = len(c_win) // 3
    if seg < 15:
        return False
    bounds = [(0, seg), (seg, 2 * seg), (2 * seg, len(c_win))]
    depths: list[float] = []
    for start, end in bounds:
        seg_hi = float(np.max(h_win[start:end]))
        seg_lo = float(np.min(l_win[start:end]))
        if seg_hi <= 0.0 or seg_hi <= seg_lo:
            return False
        depths.append(((seg_hi - seg_lo) / seg_hi) * 100.0)

    d1, d2, d3 = depths
    if d1 < 5.0:
        return False
    tightening = (d2 <= d1 * 0.88 and d3 <= d2 * 0.88) or (d3 <= d1 * 0.60 and d3 < d2)
    if not tightening or d3 > 12.0:
        return False

    # Volume dry-up validation during the final contraction stage
    v_recent = float(np.mean(v_win[-10:]))
    v_prior = float(np.mean(v_win[-50:-10])) if len(v_win) >= 50 else float(np.mean(v_win[:-10]))
    if v_prior <= 0.0:
        return False
    return bool(v_recent <= v_prior * 0.90)


def compute_relative_strength(
    hist: pd.DataFrame | None,
    benchmark_hist: pd.DataFrame | None,
) -> dict[str, float | None]:
    """Compute stock-vs-benchmark excess return over 20d/60d/120d and 20d RS slope."""
    empty: dict[str, float | None] = {
        "rs_20d_pct": None,
        "rs_60d_pct": None,
        "rs_120d_pct": None,
        "rs_slope_20d": None,
    }
    ext_s = _extract_ohlcv_series(hist)
    ext_b = _extract_ohlcv_series(benchmark_hist)
    if ext_s is None or ext_b is None:
        return empty
    _, _, _, s_close, _ = ext_s
    _, _, _, b_close, _ = ext_b

    aligned = pd.concat([s_close.rename("s"), b_close.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 22:
        return empty

    ratio = (aligned["s"] / aligned["b"]) * 100.0
    out = dict(empty)
    for horizon, key in ((20, "rs_20d_pct"), (60, "rs_60d_pct"), (120, "rs_120d_pct")):
        if len(ratio) >= horizon + 1:
            base = float(ratio.iloc[-(horizon + 1)])
            curr = float(ratio.iloc[-1])
            if base > 0.0:
                out[key] = ((curr / base) - 1.0) * 100.0

    # Linear regression slope (% per day) on normalized last-20d RS line
    tail20 = ratio.tail(20).to_numpy(dtype=np.float64)
    if len(tail20) == 20 and float(tail20[0]) > 0.0:
        norm_rs = (tail20 / float(tail20[0])) * 100.0
        x = np.arange(20, dtype=np.float64)
        slope = float(np.polyfit(x, norm_rs, 1)[0])
        out["rs_slope_20d"] = slope
    return out


def compute_vpvr_profile(
    hist: pd.DataFrame | None,
    window: int = 60,
    bins: int = 24,
) -> dict[str, float] | None:
    """Compute lightweight 60-day OHLCV Volume Profile (POC price, distance %, overhead volume %)."""
    extracted = _extract_ohlcv_series(hist)
    if extracted is None:
        return None
    _, highs, lows, closes, vols = extracted
    if len(closes) < 20:
        return None

    h_arr = highs.tail(window).to_numpy(dtype=np.float64)
    l_arr = lows.tail(window).to_numpy(dtype=np.float64)
    c_arr = closes.tail(window).to_numpy(dtype=np.float64)
    v_arr = vols.tail(window).to_numpy(dtype=np.float64)
    total_vol = float(np.sum(v_arr))
    if total_vol <= 0.0:
        return None

    p_min = float(np.min(l_arr))
    p_max = float(np.max(h_arr))
    last_close = float(c_arr[-1])
    if p_min <= 0.0 or p_max <= p_min or last_close <= 0.0:
        return None

    typical = (h_arr + l_arr + 2.0 * c_arr) / 4.0
    vol_hist, edges = np.histogram(typical, bins=bins, range=(p_min, p_max), weights=v_arr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    poc_idx = int(np.argmax(vol_hist))
    poc_price = float(centers[poc_idx])
    if poc_price <= 0.0:
        return None

    # Overhead volume above current price indicates resistance supply
    overhead_vol = float(np.sum(vol_hist[centers > last_close]))
    overhead_pct = (overhead_vol / total_vol) * 100.0
    poc_dist_pct = ((last_close / poc_price) - 1.0) * 100.0
    return {
        "poc_price": round(poc_price, 4),
        "poc_dist_pct": round(poc_dist_pct, 2),
        "overhead_vol_pct": round(overhead_pct, 1),
    }


def compute_market_breadth(
    stage1_tech: dict[str, TechnicalSnapshot | None],
    universe_syms: Iterable[str],
    histories: dict[str, pd.DataFrame] | None = None,
    benchmark_symbols: Iterable[str] = ("SPY", "ISF.L", "EXS1.DE"),
) -> dict[str, Any]:
    """Compute dual-anchored breadth: 50% universe internal breadth + 50% regional benchmark trend/vol state."""
    valid = [stage1_tech[s] for s in universe_syms if stage1_tech.get(s) is not None]
    total = len(valid)
    if total == 0 and not histories:
        return {"total": 0, "bullish_pct": 50.0, "above_ma50_pct": 50.0, "regime_tag": "TRANSITIONAL"}

    if total > 0:
        bullish = sum(1 for t in valid if t.trend_regime in {"BULL", "PULLBACK"})
        above_ma50 = sum(1 for t in valid if t.dist_ma50_pct is not None and t.dist_ma50_pct > 0.0)
        bullish_pct = (bullish / total) * 100.0
        above_ma50_pct = (above_ma50 / total) * 100.0
        univ_score = 0.5 * bullish_pct + 0.5 * above_ma50_pct
    else:
        bullish_pct = 50.0
        above_ma50_pct = 50.0
        univ_score = 50.0

    bench_scores: list[float] = []
    if histories:
        for b_sym in benchmark_symbols:
            df_b = histories.get(b_sym)
            if df_b is None or df_b.empty:
                continue
            b_tech = compute_technical_snapshot(b_sym, df_b)
            if b_tech is None:
                continue
            if b_tech.trend_regime in {"BULL", "PULLBACK"} and (b_tech.dist_ma50_pct or 0.0) > 0.0:
                b_val = 90.0
            elif (b_tech.dist_ma200_pct or -1.0) > 0.0:
                b_val = 60.0
            elif b_tech.trend_regime == "WEAK" and (b_tech.dist_ma50_pct or 0.0) < 0.0:
                b_val = 15.0
            else:
                b_val = 45.0
            if b_tech.volatility_60d_pct is not None and b_tech.volatility_60d_pct > 25.0:
                b_val = max(0.0, b_val - 15.0)
            bench_scores.append(b_val)

    combined_score = (
        0.5 * univ_score + 0.5 * (sum(bench_scores) / len(bench_scores))
        if bench_scores
        else univ_score
    )
    if combined_score >= 60.0:
        regime = "RISK-ON"
    elif combined_score <= 38.0:
        regime = "RISK-OFF"
    else:
        regime = "TRANSITIONAL"
    return {
        "total": total,
        "bullish_pct": round(bullish_pct, 1),
        "above_ma50_pct": round(above_ma50_pct, 1),
        "combined_score": round(combined_score, 1),
        "regime_tag": regime,
    }


def compute_portfolio_risk_metrics(
    histories: dict[str, pd.DataFrame],
    portfolio: PortfolioData,
    candidates: Iterable[str],
    *,
    lookback_days: int = 60,
    assumed_new_weight: float = 0.05,
) -> dict[str, dict[str, float | None]]:
    """Compute value-weighted 3d overlapping log-return corr & PSD daily-vol-scaled marginal vol delta."""
    held_values: dict[str, float] = {}
    for pos in portfolio.all_positions():
        sym = (pos.yahoo_symbol or "").strip()
        if sym and pos.current_value > 0.0:
            held_values[sym] = held_values.get(sym, 0.0) + float(pos.current_value)

    if not held_values:
        return {}

    close_series: dict[str, pd.Series] = {}
    vol_1d: dict[str, float] = {}
    for sym in dict.fromkeys([*held_values.keys(), *candidates]):
        ext = _extract_ohlcv_series(histories.get(sym))
        if ext is None:
            continue
        ser = ext[3]
        # Compute clean 1-day log-return volatility WITHOUT cross-holiday ffill distortion
        r1 = np.log(ser / ser.shift(1)).tail(max(lookback_days, 30)).dropna()
        if len(r1) < 15:
            continue
        std_1d = float(np.std(r1.to_numpy(dtype=np.float64), ddof=1))
        if std_1d > 0.0:
            close_series[sym] = ser
            vol_1d[sym] = std_1d

    held_valid = [s for s in held_values if s in close_series]
    if not held_valid:
        return {}

    raw_w = np.array([held_values[s] for s in held_valid], dtype=np.float64)
    w_sum = float(np.sum(raw_w))
    if w_sum <= 0.0:
        return {}
    w = raw_w / w_sum

    # Align across asynchronous US/UK/EU bank holidays (limit=1) for 3-day overlapping correlation ONLY
    held_px = pd.DataFrame({s: close_series[s] for s in held_valid}).sort_index().ffill(limit=1)
    held_log3 = np.log(held_px / held_px.shift(3)).tail(max(lookback_days, 45)).dropna()
    if len(held_log3) < 20:
        return {}

    out: dict[str, dict[str, float | None]] = {}
    held_set_local = set(held_valid)
    shrinkage = 0.20

    for cand in candidates:
        if cand not in close_series or cand in held_set_local:
            continue
        pair_px = pd.concat(
            [held_px, close_series[cand].rename("__cand__")],
            axis=1,
        ).sort_index().ffill(limit=1)
        pair_log3 = np.log(pair_px / pair_px.shift(3)).tail(max(lookback_days, 45)).dropna()
        if len(pair_log3) < 20:
            continue

        arr3 = pair_log3.to_numpy(dtype=np.float64)
        stds3 = np.std(arr3, axis=0, ddof=1)
        if np.any(stds3 <= 0.0):
            continue

        corr3 = np.corrcoef(arr3, rowvar=False)
        if corr3.ndim == 0:
            continue
        corr3 = np.nan_to_num(corr3, nan=0.0)
        np.fill_diagonal(corr3, 1.0)
        weighted_corr = float(np.dot(w, corr3[:-1, -1]))

        # Shrink 3d correlation matrix toward identity to guarantee positive definiteness,
        # then rescale by clean un-ffilled 1-day standard deviations D_1d.
        n_dim = corr3.shape[0]
        corr_shrunk = (1.0 - shrinkage) * corr3 + shrinkage * np.eye(n_dim, dtype=np.float64)
        d_vec = np.array([*(vol_1d[s] for s in held_valid), vol_1d[cand]], dtype=np.float64)
        cov_aug = np.outer(d_vec, d_vec) * corr_shrunk

        w_aug = np.concatenate([w * (1.0 - assumed_new_weight), np.array([assumed_new_weight])])
        aug_var = float(w_aug @ cov_aug @ w_aug)
        base_var = float(w @ cov_aug[:-1, :-1] @ w)
        if aug_var <= 0.0 or base_var <= 0.0:
            continue
        aug_vol_pct = float(np.sqrt(aug_var * 252.0) * 100.0)
        base_vol_pct = float(np.sqrt(base_var * 252.0) * 100.0)
        out[cand] = {
            "weighted_corr": round(weighted_corr, 3),
            "marginal_vol_delta_pct": round(aug_vol_pct - base_vol_pct, 3),
        }
    return out
