"""Quantitative multi-market pre-screener for /pick (0 LLM tokens before Agent review)."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    _normalize_adjusted_ohlcv,
    _resolve_col,
    compute_correlation_matrix,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
    resolve_market_benchmark,
)
from src.data_providers.market_data import (
    MarketDataService,
    TickerMarketBundle,
    normalize_history_index,
)
from src.data_providers.t212_client import PortfolioData


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
    corr_str = f" | 最高相关({corr_holding})={max_corr:.2f}" if max_corr is not None and corr_holding else ""
    rs_str = f" | 20日RS {rs_20d_pct:+.1f}%" if rs_20d_pct is not None else ""
    mvol_str = f" | 增量波动 {marginal_vol_delta_pct:+.2f}%" if marginal_vol_delta_pct is not None else ""
    reg_str = f" | 广度 {regime_tag}" if regime_tag else ""
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
            f"{f' ({bundle.short_name})' if bundle and bundle.short_name else ''} "
            f"| 量化初筛分 {score:.0f}{dim_str} | 现价 {tech.last_close:.2f}"
            f"{f' {bundle.currency}' if bundle and bundle.currency else ''} | "
            f"5日 {tech.ret_5d_pct:+.1f}% / 20日 {tech.ret_20d_pct:+.1f}% | "
            f"距MA50 {tech.dist_ma50_pct:+.1f}% / MA200 {tech.dist_ma200_pct:+.1f}% | "
            f"RSI {tech.rsi14:.0f} | 距52w高点 {tech.drawdown_52w_pct:+.1f}%"
            f"{pe_str}{roe_str}{div_str}{upside_str}{rs_str}{cyq_str}{ed_str}{corr_str}{mvol_str}{reg_str}"
            f"{pros_str}{cons_str}"
        )
    return f"• [{track}][{asset_type}] {symbol} | 分数 {score:.0f}{dim_str}{reg_str}{pros_str}{cons_str}"


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
    rs_slope = rs_info.get("rs_slope_20d")
    minervini = detect_minervini_stage2(hist, tech, rs_60d_pct=rs_60d)
    minervini_phase = int(minervini.get("phase", 0))

    if track == "ISA_LONG_TERM":
        if tech.trend_regime == "BULL":
            dim1 += 5.0
            pros.append("上升趋势(MA50>MA200)")
        elif tech.trend_regime == "PULLBACK":
            dim1 += 7.0
            pros.append("多头回踩支撑区")
        elif tech.trend_regime == "WEAK":
            dim1 -= 7.0
            cons.append("弱势低于MA200")

        if tech.drawdown_52w_pct is not None and -18.0 <= tech.drawdown_52w_pct <= -4.0:
            dim1 += 3.0
            pros.append(f"适度回撤{tech.drawdown_52w_pct:.1f}%")

        if tech.rsi14 is not None:
            if 32.0 <= tech.rsi14 <= 55.0:
                dim1 += 3.0
                pros.append(f"RSI={tech.rsi14:.0f}不追高")
            elif tech.rsi14 >= 68.0:
                dim1 -= _linear_scale(tech.rsi14, 68.0, 82.0, 2.0, 6.0)
                if tech.rsi14 >= 72.0:
                    cons.append(f"RSI={tech.rsi14:.0f}偏热")

        if tech.volatility_60d_pct is not None and tech.volatility_60d_pct > 45.0:
            dim1 -= _linear_scale(tech.volatility_60d_pct, 45.0, 80.0, 2.0, 6.0)
            cons.append(f"高波动({tech.volatility_60d_pct:.0f}%)")
    else:
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
                dim1 += 4.5
                pros.append(f"RSI={tech.rsi14:.0f}超卖弹性")
            elif 40.0 <= tech.rsi14 <= 60.0 and (tech.ret_20d_pct or 0.0) > 0.0:
                dim1 += 3.0
                pros.append("动量健康")
            elif tech.rsi14 >= 70.0:
                dim1 -= _linear_scale(tech.rsi14, 70.0, 84.0, 2.0, 7.0)
                if tech.rsi14 >= 74.0:
                    cons.append(f"RSI={tech.rsi14:.0f}短期过热")

        if bundle and bundle.days_to_earnings is not None and 0 <= bundle.days_to_earnings <= 5:
            dim1 -= 4.0
            cons.append(f"临近财报({bundle.days_to_earnings}天)")

    if minervini["passes"]:
        dim1 += 3.5
        pattern_signals.append("Minervini阶段2")
        pros.append(f"Minervini模板达标({minervini['criteria_passed']}/{minervini['criteria_total']})")
    elif minervini_phase == 4:
        # Option 2A: Penalize Phase 4 downtrend heavily while preserving visibility + warning tag
        dim1 -= 8.0
        cons.append("[Phase4弱势]")

    if regime_tag == "RISK-OFF" and tech.trend_regime == "BULL" and (tech.rsi14 or 50.0) >= 62.0:
        dim1 -= 2.0
    elif regime_tag == "RISK-ON" and tech.trend_regime in {"BULL", "PULLBACK"}:
        dim1 += 1.0

    if tech.weekly_bullish is True:
        if tech.trend_regime in {"BULL", "PULLBACK"}:
            dim1 += 4.0
            pros.append("周/日线多头共振")
        else:
            dim1 += 2.0
            pros.append("周线多头")
    elif tech.weekly_bullish is False:
        dim1 -= 4.0
        cons.append("周线偏空")

    if rs_20d is not None:
        dim1 += _linear_scale(rs_20d, -8.0, 10.0, -2.5, 3.5)
        if rs_20d >= 4.0:
            pros.append(f"RS强于基准({rs_20d:+.1f}%)")
        elif rs_20d <= -5.0:
            cons.append(f"RS跑输基准({rs_20d:+.1f}%)")
    if rs_slope is not None and rs_slope > 0.15 and (rs_20d or 0.0) > 0.0:
        dim1 += 1.5

    dim1 = round(max(0.0, min(30.0, dim1)), 2)

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
    if upside_pct is not None:
        dim3 += _linear_scale(upside_pct, -15.0, 25.0, -4.0, 5.0)
        if upside_pct >= 10.0:
            pros.append(f"目标价空间+{upside_pct:.0f}%")
        elif upside_pct <= -10.0:
            cons.append(f"高于一致目标价({upside_pct:+.0f}%)")

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
                    pros.append(f"估值合理(PE={pe_val:.1f}x)")
                else:
                    dim3 += _linear_scale(pe_val, 22.0, 55.0, 2.5, -5.0)
                    if pe_val >= 50.0:
                        cons.append(f"远期估值偏高(PE={pe_val:.1f}x)")
            else:
                if pe_val <= 28.0:
                    dim3 += _linear_scale(pe_val, 10.0, 28.0, 4.0, 1.5)
                    if pe_val <= 25.0:
                        pros.append(f"PE健康({pe_val:.1f}x)")
                else:
                    dim3 += _linear_scale(pe_val, 28.0, 65.0, 1.5, -5.0)
                    if pe_val >= 65.0:
                        cons.append(f"高估值(PE={pe_val:.1f}x)")

        if bundle.roe_pct is not None:
            dim3 += _linear_scale(bundle.roe_pct, 5.0, 25.0, -1.0, 3.0)
            if bundle.roe_pct >= 18.0:
                pros.append(f"高ROE({bundle.roe_pct:.0f}%)")
        if bundle.profit_margin_pct is not None:
            dim3 += _linear_scale(bundle.profit_margin_pct, 2.0, 22.0, -1.0, 2.0)
        if bundle.revenue_growth_pct is not None:
            dim3 += _linear_scale(bundle.revenue_growth_pct, 0.0, 25.0, -0.5, 2.0)
        if bundle.debt_to_equity is not None and bundle.debt_to_equity >= 0.0:
            dim3 += _linear_scale(bundle.debt_to_equity, 30.0, 200.0, 1.0, -2.0)
            if bundle.debt_to_equity >= 180.0:
                cons.append(f"高杠杆(D/E={bundle.debt_to_equity:.0f}%)")
    elif is_etf:
        dim3 += 2.0

    if bundle and bundle.dividend_yield_pct is not None and 0.0 < bundle.dividend_yield_pct <= 15.0:
        div = bundle.dividend_yield_pct
        if track == "ISA_LONG_TERM":
            dim3 += _linear_scale(div, 0.0, 3.5, 0.0, 3.5)
            if div >= 2.5:
                pros.append(f"股息率{div:.1f}%")
        else:
            dim3 += _linear_scale(div, 0.0, 4.0, 0.0, 2.0)
            if div >= 3.0:
                pros.append(f"股息保护{div:.1f}%")

    if sym_up.endswith(".L") and is_etf:
        dim3 += 6.0
        pros.append("LSE英镑ETF(免换汇/免印花税)")
    elif sym_up.endswith(".L"):
        dim3 += 2.5
        pros.append("英镑计价免换汇")

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
            if max_corr <= 0.55:
                pros.append(f"组合分散度佳(最高相关{max_corr:.2f})")

    if weighted_corr is not None and not is_held:
        dim4 += _linear_scale(weighted_corr, 0.75, 0.15, -1.5, 2.0)

    if marginal_vol_delta_pct is not None and not is_held:
        if marginal_vol_delta_pct <= -0.02:
            dim4 += _linear_scale(marginal_vol_delta_pct, -0.02, -0.35, 0.5, 2.5)
            pros.append(f"降低组合波动({marginal_vol_delta_pct:+.2f}%)")
        elif marginal_vol_delta_pct >= 0.35:
            dim4 -= _linear_scale(marginal_vol_delta_pct, 0.35, 0.90, 0.5, 2.5)
            cons.append(f"推高组合波动({marginal_vol_delta_pct:+.2f}%)")

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
        weighted_raw = (
            25.0 * dimension_pct["trend_rs"]
            + 15.0 * dimension_pct["pattern_timing"]
            + 35.0 * dimension_pct["value_quality"]
            + 25.0 * dimension_pct["portfolio_fit"]
        )
    else:
        weighted_raw = (
            35.0 * dimension_pct["trend_rs"]
            + 35.0 * dimension_pct["pattern_timing"]
            + 15.0 * dimension_pct["value_quality"]
            + 15.0 * dimension_pct["portfolio_fit"]
        )
    score = round(max(0.0, min(100.0, weighted_raw)), 1)
    reasons = [*pros, *cons]

    rs_20d_round = round(rs_20d, 2) if rs_20d is not None else None
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


async def run_quant_screener(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    universe: UniverseSection,
    watchlist: list[str],
    strategy: StrategySection,
    existing_histories: dict | None = None,
    top_n_per_track: int = 8,
) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
    """Two-stage quantitative pre-screener: 0-info OHLCV multi-bucket filter -> Stage 2 bundle enrichment."""
    held_syms = portfolio.unique_yahoo_symbols()
    held_set = {s.upper() for s in held_syms}
    watch_set = {s.upper() for s in watchlist}
    etf_set = _resolve_configured_etfs(universe, strategy)

    # Dynamically expand scan pools with symbols traded in the last 90 days
    cutoff_90d_ms = int(time.time() * 1000) - 90 * 86400 * 1000
    recent_isa: list[str] = []
    recent_invest: list[str] = []
    recent_traded_set: set[str] = set()
    for order in (*portfolio.isa.recent_orders, *portfolio.invest.recent_orders):
        sym = (order.yahoo_symbol or "").strip()
        if not sym:
            continue
        if order.filled_at_ms > 0 and order.filled_at_ms < cutoff_90d_ms:
            continue
        recent_traded_set.add(sym.upper())
        if order.account.upper() == "ISA":
            recent_isa.append(sym)
        else:
            recent_invest.append(sym)

    isa_pool = list(dict.fromkeys([*watchlist, *universe.isa_long_term, *recent_isa]))
    invest_pool = list(dict.fromkeys([*watchlist, *universe.invest_tactical, *recent_invest]))
    all_needed = list(dict.fromkeys([*held_syms, *isa_pool, *invest_pool]))

    benchmarks = ["SPY", "ISF.L"]
    if any(resolve_market_benchmark(sym)[0] == "EXS1.DE" for sym in all_needed):
        benchmarks.append("EXS1.DE")
    fetch_needed = list(dict.fromkeys([*all_needed, *benchmarks]))

    # Stage 1: Reuse already-fetched held histories & fetch remaining OHLCV without .info calls
    histories: dict[str, pd.DataFrame] = {}
    if existing_histories:
        for k, df in existing_histories.items():
            if df is not None and not df.empty:
                histories[k] = df

    missing_hist_syms = [s for s in fetch_needed if s not in histories]
    if missing_hist_syms:
        fetched_hists = await market_data.get_histories(missing_hist_syms, period="1y")
        for k, df in fetched_hists.items():
            if df is not None and not df.empty:
                histories[k] = df

    corr_matrix = compute_correlation_matrix(
        histories,
        all_needed,
        lookback_days=strategy.correlation_window_days,
        benchmark_symbol="SPY",
        uk_benchmark_symbol="ISF.L",
        eu_benchmark_symbol="EXS1.DE",
    )

    stage1_tech: dict[str, TechnicalSnapshot | None] = {
        sym: compute_technical_snapshot(sym, histories.get(sym))
        for sym in all_needed
    }

    breadth_info = compute_market_breadth(
        stage1_tech,
        [*isa_pool, *invest_pool],
        histories=histories,
        benchmark_symbols=benchmarks,
    )
    regime_tag = str(breadth_info.get("regime_tag", ""))

    corr_cache: dict[str, tuple[str | None, float | None]] = {
        sym: max_correlation_against_portfolio(corr_matrix, sym, held_syms)
        for sym in all_needed
    }
    risk_cache = compute_portfolio_risk_metrics(
        histories,
        portfolio,
        all_needed,
        lookback_days=strategy.correlation_window_days,
    )

    def _bench_for(sym: str) -> pd.DataFrame | None:
        bench_sym = resolve_market_benchmark(sym)[0]
        return histories.get(bench_sym)

    preselect_n = max(top_n_per_track * 2, top_n_per_track + 6)

    def _multi_bucket_shortlist(pool: list[str], track: str) -> list[str]:
        scored: list[ScreenerCandidate] = []
        tech_by_sym: dict[str, TechnicalSnapshot] = {}
        for sym in pool:
            tech = stage1_tech.get(sym)
            if tech is None:
                continue
            tech_by_sym[sym] = tech
            corr_sym, corr_val = corr_cache.get(sym, (None, None))
            r_meta = risk_cache.get(sym, {})
            scored.append(
                score_candidate(
                    symbol=sym,
                    track=track,
                    tech=tech,
                    bundle=None,
                    max_corr=corr_val,
                    corr_holding=corr_sym,
                    is_held=(sym.upper() in held_set),
                    is_watchlist=(sym.upper() in watch_set),
                    strategy=strategy,
                    hist=histories.get(sym),
                    benchmark_hist=_bench_for(sym),
                    weighted_corr=r_meta.get("weighted_corr"),
                    marginal_vol_delta_pct=r_meta.get("marginal_vol_delta_pct"),
                    regime_tag=regime_tag,
                    is_recent_traded=(sym.upper() in recent_traded_set),
                    etf_symbols=etf_set,
                )
            )

        if len(scored) <= preselect_n:
            scored.sort(key=lambda c: c.score, reverse=True)
            return [c.symbol for c in scored]

        picked: list[str] = []
        picked_set: set[str] = set()

        def _take(ordered: list[ScreenerCandidate], limit: int) -> None:
            for c in ordered:
                if len(picked) >= preselect_n or limit <= 0:
                    break
                if c.symbol not in picked_set:
                    picked.append(c.symbol)
                    picked_set.add(c.symbol)
                    limit -= 1

        # Bucket 1 (~40%): Momentum & breakout leaders (Trend+RS + Pattern)
        mom_quota = max(1, (preselect_n * 4) // 10)
        mom_sorted = sorted(
            scored,
            key=lambda c: (
                c.dimension_pct.get("trend_rs", 0.0) + c.dimension_pct.get("pattern_timing", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(mom_sorted, mom_quota)

        # Bucket 2 (~25%): Pullback & low-volatility base candidates (not Phase 4)
        pb_quota = max(1, preselect_n // 4)
        pb_pool = [
            c
            for c in scored
            if "[Phase4弱势]" not in c.cons
            and (
                tech_by_sym[c.symbol].trend_regime == "PULLBACK"
                or (tech_by_sym[c.symbol].rsi14 or 50.0) <= 50.0
                or "VCP收缩" in c.pattern_signals
                or "低ATR收敛" in c.pattern_signals
            )
        ]
        pb_pool.sort(
            key=lambda c: (
                c.dimension_pct.get("pattern_timing", 0.0) + c.dimension_pct.get("portfolio_fit", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(pb_pool, pb_quota)

        # Bucket 3 (~20%): Defensive low-volatility / LSE ETF / portfolio diversification bucket (0 extra IO)
        def_quota = max(1, preselect_n // 5)
        def_pool = [
            c
            for c in scored
            if "[Phase4弱势]" not in c.cons
            and not c.overlap_warning
            and (
                c.asset_type == "ETF"
                or (tech_by_sym[c.symbol].volatility_60d_pct is not None and tech_by_sym[c.symbol].volatility_60d_pct <= 28.0)
                or (c.marginal_vol_delta_pct is not None and c.marginal_vol_delta_pct <= 0.10)
                or (c.max_portfolio_corr is not None and c.max_portfolio_corr <= 0.55)
            )
        ]
        def_pool.sort(
            key=lambda c: (
                c.dimension_pct.get("portfolio_fit", 0.0) + c.dimension_pct.get("value_quality", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(def_pool, def_quota)

        # Bucket 4 (~15%, min 1): Watchlist & recent-traded priority protection (exclude severe overlaps)
        prio_quota = max(1, preselect_n // 6)
        prio_pool = sorted(
            [
                c
                for c in scored
                if (c.is_watchlist or c.is_recent_traded) and not c.overlap_warning
            ],
            key=lambda c: c.score,
            reverse=True,
        )
        _take(prio_pool, min(prio_quota, max(1, preselect_n - len(picked))))

        # Backfill remaining capacity by overall Stage 1 score
        overall_sorted = sorted(scored, key=lambda c: c.score, reverse=True)
        _take(overall_sorted, preselect_n - len(picked))
        return picked

    isa_shortlist = _multi_bucket_shortlist(isa_pool, "ISA_LONG_TERM")
    invest_shortlist = _multi_bucket_shortlist(invest_pool, "INVEST_TACTICAL")
    top_syms = list(dict.fromkeys([*isa_shortlist, *invest_shortlist]))

    # Stage 2: Enrich only shortlisted candidates with fundamentals/earnings/quote_type
    bundles: dict[str, TickerMarketBundle] = {}
    if top_syms:
        isin_map = {
            p.yahoo_symbol: p.isin
            for p in portfolio.all_positions()
            if p.yahoo_symbol and p.isin
        }
        bundle_hists, bundles = await market_data.get_bundles(
            top_syms,
            include_news=False,
            include_earnings=True,
            include_info=True,
            isin_map=isin_map or None,
        )
        for sym, df in bundle_hists.items():
            if (sym not in histories or histories[sym].empty) and df is not None and not df.empty:
                histories[sym] = df

    stage2_tech: dict[str, TechnicalSnapshot | None] = {
        sym: compute_technical_snapshot(sym, histories.get(sym), bundle=bundles.get(sym))
        for sym in top_syms
    }

    def _build_track(shortlist: list[str], track: str) -> list[ScreenerCandidate]:
        out: list[ScreenerCandidate] = []
        for sym in shortlist:
            tech = stage2_tech.get(sym) or stage1_tech.get(sym)
            if tech is None:
                continue
            corr_sym, corr_val = corr_cache.get(sym, (None, None))
            r_meta = risk_cache.get(sym, {})
            out.append(
                score_candidate(
                    symbol=sym,
                    track=track,
                    tech=tech,
                    bundle=bundles.get(sym),
                    max_corr=corr_val,
                    corr_holding=corr_sym,
                    is_held=(sym.upper() in held_set),
                    is_watchlist=(sym.upper() in watch_set),
                    strategy=strategy,
                    hist=histories.get(sym),
                    benchmark_hist=_bench_for(sym),
                    weighted_corr=r_meta.get("weighted_corr"),
                    marginal_vol_delta_pct=r_meta.get("marginal_vol_delta_pct"),
                    regime_tag=regime_tag,
                    is_recent_traded=(sym.upper() in recent_traded_set),
                    etf_symbols=etf_set,
                )
            )
        calibrated = _apply_percentile_calibration(out)
        return calibrated[:top_n_per_track]

    return _build_track(isa_shortlist, "ISA_LONG_TERM"), _build_track(invest_shortlist, "INVEST_TACTICAL")
