"""Quantitative multi-market pre-screener for /pick (0 LLM tokens before Agent review)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    _is_cross_timezone_pair,
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
from src.data_providers.t212_client import AccountSnapshot, PortfolioData
from src.engine.portfolio_rules import wallet_friction_context


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
            f"{pe_str}{roe_str}{div_str}{upside_str}{rs_str}{cyq_str}{ed_str}{ex_div_str}{corr_str}{mvol_str}{reg_str}"
            f"{pros_str}{cons_str}"
        )
    return f"• [{track}][{asset_type}] {symbol} | 分数 {score:.0f}{dim_str}{reg_str}{ex_div_str}{pros_str}{cons_str}"


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

        if tech.rsi14 is not None:
            if 32.0 <= tech.rsi14 <= 55.0:
                dim1 += 3.0
            elif tech.rsi14 >= 68.0:
                dim1 -= _linear_scale(tech.rsi14, 68.0, 82.0, 2.0, 6.0)

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
            elif 40.0 <= tech.rsi14 <= 60.0 and (tech.ret_20d_pct or 0.0) > 0.0:
                dim1 += 3.0
                pros.append("动量健康")
            elif tech.rsi14 >= 70.0:
                dim1 -= _linear_scale(tech.rsi14, 70.0, 84.0, 2.0, 7.0)

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
        if bundle.debt_to_equity is not None and bundle.debt_to_equity >= 0.0:
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
        if track == "ISA_LONG_TERM" or not wallet_enabled or gbp_avail > 0.01:
            dim3 += 2.5
            pros.append("英镑计价免换汇" if track == "ISA_LONG_TERM" else "英镑计价(Invest GBP充足:免换汇)")
        else:
            dim3 += 1.0
            cons.append(
                f"英股需USD换回GBP(+{strategy.fx_fee_pct:.2f}%换汇+"
                f"{strategy.uk_stamp_duty_pct:.1f}%印花税)"
            )

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
    active_pool_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
    """Two-stage quantitative pre-screener: 0-info OHLCV multi-bucket filter -> Stage 2 bundle enrichment."""
    held_syms = portfolio.unique_yahoo_symbols()
    held_set = {s.upper() for s in held_syms}
    watch_set = {s.upper() for s in watchlist}
    etf_set = _resolve_configured_etfs(universe, strategy)

    sector_hint_map: dict[str, str] = {}
    if active_pool_rows:
        for row in active_pool_rows:
            sym_u = str(row.get("symbol") or "").strip().upper()
            if not sym_u:
                continue
            if bool(row.get("is_etf")):
                etf_set.add(sym_u)
            sec = str(row.get("sector") or "").strip()
            if sec:
                sector_hint_map[sym_u] = sec

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

    if active_pool_rows:
        isa_base: list[str] = []
        invest_base: list[str] = []
        for row in active_pool_rows:
            sym_u = str(row.get("symbol") or "").strip().upper()
            if not sym_u:
                continue
            is_etf_row = bool(row.get("is_etf")) or (sym_u in etf_set)
            if bool(row.get("isa_eligible", True)):
                isa_base.append(sym_u)
            # Only share non-ETF vetted equities with Invest; keep LSE GBP ETFs ISA-focused unless already in invest_tactical
            if not is_etf_row and bool(row.get("invest_eligible", True)):
                invest_base.append(sym_u)
        for etf_sym in etf_set:
            if etf_sym not in isa_base:
                isa_base.append(etf_sym)
        isa_pool = list(dict.fromkeys([*watchlist, *isa_base, *recent_isa]))
        invest_pool = list(dict.fromkeys([*watchlist, *invest_base, *recent_invest]))
    else:
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
        against_symbols=held_syms or None,
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
                    invest_wallet=portfolio.invest,
                )
            )

        if len(scored) <= preselect_n:
            scored.sort(key=lambda c: c.score, reverse=True)
            return [c.symbol for c in scored]

        picked: list[str] = []
        picked_set: set[str] = set()
        sector_counts: dict[str, int] = {}
        max_per_sector = max(3, preselect_n // 4) if (sector_hint_map and len(scored) > preselect_n * 2) else preselect_n

        def _take(ordered: list[ScreenerCandidate], limit: int, *, apply_sector_cap: bool = True) -> None:
            for c in ordered:
                if len(picked) >= preselect_n or limit <= 0:
                    break
                if c.symbol in picked_set:
                    continue
                sec = sector_hint_map.get(c.symbol.upper(), "")
                if apply_sector_cap and sec and sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                picked.append(c.symbol)
                picked_set.add(c.symbol)
                if sec:
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
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
        _take(prio_pool, min(prio_quota, max(1, preselect_n - len(picked))), apply_sector_cap=False)

        # Backfill remaining capacity by overall Stage 1 score (respect sector cap first, relax if still short)
        overall_sorted = sorted(scored, key=lambda c: c.score, reverse=True)
        _take(overall_sorted, preselect_n - len(picked), apply_sector_cap=True)
        if len(picked) < preselect_n:
            _take(overall_sorted, preselect_n - len(picked), apply_sector_cap=False)
        return picked

    isa_shortlist = _multi_bucket_shortlist(isa_pool, "ISA_LONG_TERM")
    invest_shortlist = _multi_bucket_shortlist(invest_pool, "INVEST_TACTICAL")
    top_syms = list(dict.fromkeys([*isa_shortlist, *invest_shortlist]))

    # Stage 2: Enrich only shortlisted candidates with fundamentals/earnings/quote_type/US deep intel
    bundles: dict[str, TickerMarketBundle] = {}
    deep_intel_map: dict[str, dict[str, Any]] = {}
    if top_syms:
        isin_map = {
            p.yahoo_symbol: p.isin
            for p in portfolio.all_positions()
            if p.yahoo_symbol and p.isin
        }
        us_top_syms = [
            s
            for s in top_syms
            if "." not in s and "=" not in s and not s.startswith("^") and s.upper() not in etf_set
        ]

        async def _empty_intel() -> dict[str, dict[str, Any]]:
            return {}

        intel_coro = (
            market_data.get_us_deep_intel_batch(us_top_syms)
            if (us_top_syms and hasattr(market_data, "get_us_deep_intel_batch"))
            else _empty_intel()
        )
        (bundle_hists, bundles), deep_intel_map = await asyncio.gather(
            market_data.get_bundles(
                top_syms,
                include_news=False,
                include_earnings=True,
                include_info=True,
                isin_map=isin_map or None,
            ),
            intel_coro,
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
                    deep_intel=deep_intel_map.get(sym.upper()) or deep_intel_map.get(sym),
                    invest_wallet=portfolio.invest,
                )
            )
        calibrated = _apply_percentile_calibration(out)
        return calibrated[:top_n_per_track]

    return _build_track(isa_shortlist, "ISA_LONG_TERM"), _build_track(invest_shortlist, "INVEST_TACTICAL")


def _lookup_history_df(histories: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame | None:
    df = histories.get(symbol)
    if df is not None and not df.empty:
        return df
    return histories.get(symbol.upper())


def _audit_rolling_overlap_candidates(
    close_map: dict[str, pd.Series],
    candidate_syms: list[str],
    held_syms: list[str],
    threshold: float,
    roll_window: int = 60,
    eval_days: int = 120,
) -> list[dict[str, Any]]:
    """Identify non-held candidates whose rolling correlation with any held position stays >= threshold."""
    if not held_syms or not candidate_syms:
        return []
    held_set = {s.upper() for s in held_syms}
    eff_roll = max(20, min(60, int(roll_window or 60)))
    min_periods = max(20, eff_roll // 2)
    flagged: list[dict[str, Any]] = []

    for cand in candidate_syms:
        if cand.upper() in held_set or cand not in close_map:
            continue
        c_ser = close_map[cand]
        best_match: dict[str, Any] | None = None
        for held in held_syms:
            if held not in close_map or held.upper() == cand.upper():
                continue
            pair_px = pd.concat(
                [c_ser.rename("cand"), close_map[held].rename("held")],
                axis=1,
            ).sort_index().ffill(limit=1)
            step = 3 if _is_cross_timezone_pair(cand, held) else 1
            rets = np.log(pair_px / pair_px.shift(step)).dropna()
            if len(rets) < eff_roll + 15:
                continue
            roll_corr = rets["cand"].rolling(eff_roll, min_periods=min_periods).corr(rets["held"]).dropna()
            tail_corr = roll_corr.tail(eval_days)
            if len(tail_corr) < 15:
                continue
            latest_val = float(tail_corr.iloc[-1])
            median_val = float(tail_corr.median())
            persist_ratio = float((tail_corr >= threshold).mean())
            if latest_val >= threshold and persist_ratio >= 0.80:
                if best_match is None or median_val > best_match["corr_median_120d"]:
                    best_match = {
                        "symbol": cand,
                        "held_symbol": held,
                        "corr_latest": round(latest_val, 3),
                        "corr_median_120d": round(median_val, 3),
                        "persist_ratio": round(persist_ratio, 3),
                    }
        if best_match is not None:
            flagged.append(best_match)

    flagged.sort(key=lambda x: (-x["corr_median_120d"], x["symbol"]))
    return flagged


def _audit_zombie_candidates(
    close_map: dict[str, pd.Series],
    candidate_syms: list[str],
    min_streak_days: int = 60,
) -> list[dict[str, Any]]:
    """Identify candidates stuck in Phase 4 (Close < MA200 and MA50 < MA200) for >= min_streak_days."""
    zombies: list[dict[str, Any]] = []
    for sym in candidate_syms:
        closes = close_map.get(sym)
        if closes is None or len(closes) < 200 + min_streak_days - 1:
            continue
        ma50 = closes.rolling(50, min_periods=50).mean()
        ma200 = closes.rolling(200, min_periods=200).mean()
        cond = ((closes < ma200) & (ma50 < ma200)).fillna(False).to_numpy(dtype=bool)
        streak = 0
        for flag in reversed(cond):
            if flag:
                streak += 1
            else:
                break
        if streak >= min_streak_days:
            zombies.append(
                {
                    "symbol": sym,
                    "streak_days": streak,
                    "close": round(float(closes.iloc[-1]), 2),
                    "ma50": round(float(ma50.iloc[-1]), 2),
                    "ma200": round(float(ma200.iloc[-1]), 2),
                }
            )
    zombies.sort(key=lambda x: (-x["streak_days"], x["symbol"]))
    return zombies


def _backtest_breakout_thresholds(
    ohlcv_map: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]],
    candidate_syms: list[str],
    strategy: StrategySection,
    *,
    forward_bars: int = 60,
    cooldown_bars: int = 20,
) -> dict[str, Any]:
    """Run out-of-sample breakout simulation testing trailing TP (+20%/-12%) and adaptive ATR stop."""
    tp_act_pct = float(strategy.trailing_profit_activation_pct)
    tp_dd_pct = float(strategy.trailing_drawdown_from_peak_pct)
    fallback_stop_pct = float(strategy.invest_stop_loss_alert_pct)

    returns: list[float] = []
    tp_triggers = 0
    tp_wins = 0
    sl_triggers = 0
    horizon_exits = 0

    for sym in candidate_syms:
        ext = ohlcv_map.get(sym)
        if ext is None:
            continue
        _, highs, lows, closes, _ = ext
        n_bars = len(closes)
        if n_bars < 90:
            continue

        ma50 = closes.rolling(50, min_periods=50).mean()
        use_ma200 = n_bars >= 220
        ma200 = closes.rolling(200, min_periods=200).mean() if use_ma200 else None
        prev_c = closes.shift(1)
        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_c).abs(),
                (lows - prev_c).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()
        prior_high_20 = highs.shift(1).rolling(20, min_periods=20).max()

        start_idx = 200 if use_ma200 else 50
        i = start_idx
        while i < n_bars - 10:
            entry = float(closes.iloc[i])
            ma50_i = ma50.iloc[i]
            ph20_i = prior_high_20.iloc[i]
            if pd.isna(ma50_i) or pd.isna(ph20_i) or entry <= 0:
                i += 1
                continue
            if use_ma200 and ma200 is not None:
                ma200_i = ma200.iloc[i]
                if pd.isna(ma200_i) or float(ma50_i) <= float(ma200_i):
                    i += 1
                    continue
            if not (entry > float(ph20_i) and entry > float(ma50_i)):
                i += 1
                continue

            atr_i = atr14.iloc[i]
            if pd.notna(atr_i) and float(atr_i) > 0:
                atr_pct = (float(atr_i) / entry) * 100.0
                stop_pct = max(-18.0, min(-8.0, -2.8 * atr_pct))
            else:
                stop_pct = fallback_stop_pct

            peak_price = entry
            trailing_active = False
            exit_ret: float | None = None
            exit_reason = "HORIZON_END"
            end_j = min(n_bars - 1, i + forward_bars)

            for j in range(i + 1, end_j + 1):
                low_j = float(lows.iloc[j])
                high_j = float(highs.iloc[j])
                close_j = float(closes.iloc[j])

                low_ret = ((low_j / entry) - 1.0) * 100.0
                if not trailing_active and low_ret <= stop_pct:
                    exit_ret = stop_pct
                    exit_reason = "ATR_STOP"
                    break

                if high_j > peak_price:
                    peak_price = high_j
                peak_ret = ((peak_price / entry) - 1.0) * 100.0
                if peak_ret >= tp_act_pct:
                    trailing_active = True

                if trailing_active:
                    dd_from_peak = ((close_j / peak_price) - 1.0) * 100.0
                    if dd_from_peak <= tp_dd_pct:
                        exit_ret = ((close_j / entry) - 1.0) * 100.0
                        exit_reason = "TRAILING_TP"
                        break

            if exit_ret is None:
                exit_ret = ((float(closes.iloc[end_j]) / entry) - 1.0) * 100.0

            returns.append(exit_ret)
            if exit_reason == "TRAILING_TP":
                tp_triggers += 1
                if exit_ret > 0:
                    tp_wins += 1
            elif exit_reason == "ATR_STOP":
                sl_triggers += 1
            else:
                horizon_exits += 1

            i += cooldown_bars

    sample_count = len(returns)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "win_count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": None,
            "payoff_ratio": None,
            "trailing_tp_triggers": 0,
            "trailing_tp_win_rate_pct": 0.0,
            "atr_stop_triggers": 0,
            "horizon_exits": 0,
        }

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = round((len(wins) / sample_count) * 100.0, 1)
    avg_ret = round(float(np.mean(returns)), 2)
    payoff: float | None = None
    if wins and losses:
        avg_loss_abs = abs(float(np.mean(losses)))
        if avg_loss_abs > 1e-9:
            payoff = round(float(np.mean(wins)) / avg_loss_abs, 2)

    return {
        "sample_count": sample_count,
        "win_count": len(wins),
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "payoff_ratio": payoff,
        "trailing_tp_triggers": tp_triggers,
        "trailing_tp_win_rate_pct": round((tp_wins / tp_triggers) * 100.0, 1) if tp_triggers > 0 else 0.0,
        "atr_stop_triggers": sl_triggers,
        "horizon_exits": horizon_exits,
    }


def run_offline_universe_and_threshold_audit(
    histories: dict[str, pd.DataFrame],
    portfolio: PortfolioData,
    universe: UniverseSection,
    strategy: StrategySection,
    *,
    active_pool_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run 0-token offline universe redundancy, Phase 4 zombie, and TP/SL threshold backtest audit."""
    if active_pool_rows:
        pool_raw = [str(r.get("symbol") or "") for r in active_pool_rows] + list(universe.etf_symbols)
    else:
        pool_raw = (
            list(universe.isa_long_term)
            + list(universe.invest_tactical)
            + list(universe.etf_symbols)
        )
    universe_syms = list(
        dict.fromkeys(
            s.strip().upper()
            for s in pool_raw
            if str(s or "").strip()
        )
    )
    held_syms = list(
        dict.fromkeys(
            (p.yahoo_symbol or "").strip().upper()
            for p in portfolio.all_positions()
            if (p.yahoo_symbol or "").strip()
        )
    )

    ohlcv_map: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    close_map: dict[str, pd.Series] = {}
    for sym in dict.fromkeys(universe_syms + held_syms):
        ext = _extract_ohlcv_series(_lookup_history_df(histories, sym))
        if ext is not None:
            ohlcv_map[sym] = ext
            close_map[sym] = ext[3]

    overlap_candidates = _audit_rolling_overlap_candidates(
        close_map,
        universe_syms,
        held_syms,
        threshold=float(strategy.overlap_correlation_threshold),
        roll_window=int(strategy.correlation_window_days or 60),
        eval_days=120,
    )
    zombie_candidates = _audit_zombie_candidates(close_map, universe_syms, min_streak_days=60)
    backtest_stats = _backtest_breakout_thresholds(ohlcv_map, universe_syms, strategy)

    overlap_str = (
        ", ".join(f"{item['symbol']}(近{item['held_symbol']} {item['corr_median_120d']:.2f})" for item in overlap_candidates[:6])
        if overlap_candidates
        else "无"
    )
    if len(overlap_candidates) > 6:
        overlap_str += f" 等{len(overlap_candidates)}只"

    zombie_str = (
        ", ".join(f"{item['symbol']}({item['streak_days']}d)" for item in zombie_candidates[:6])
        if zombie_candidates
        else "无"
    )
    if len(zombie_candidates) > 6:
        zombie_str += f" 等{len(zombie_candidates)}只"

    payoff_str = f"{backtest_stats['payoff_ratio']:.2f}" if backtest_stats["payoff_ratio"] is not None else "N/A"
    avg_ret_str = (
        f"{backtest_stats['avg_return_pct']:+.2f}%"
        if backtest_stats["avg_return_pct"] is not None
        else "N/A"
    )
    summary_lines = [
        f"• 扫描池 {len(universe_syms)} 只候选 vs {len(held_syms)} 只持仓 (本地2y历史)",
        f"• 高重叠冗余(120d持续>={strategy.overlap_correlation_threshold:.2f}): {overlap_str}",
        f"• 长期弱势僵尸股(>=60d Phase4): {zombie_str}",
        (
            f"• 阈值回测(+{strategy.trailing_profit_activation_pct:.0f}%/{strategy.trailing_drawdown_from_peak_pct:.0f}%移动止盈 & ATR止损): "
            f"{backtest_stats['sample_count']} 样本 | 胜率 {backtest_stats['win_rate_pct']:.1f}% | "
            f"盈亏比 {payoff_str} | 均收益 {avg_ret_str} | "
            f"止盈 {backtest_stats['trailing_tp_triggers']} / 止损 {backtest_stats['atr_stop_triggers']}"
        ),
    ]

    return {
        "universe_count": len(universe_syms),
        "held_count": len(held_syms),
        "overlap_candidates": overlap_candidates,
        "zombie_candidates": zombie_candidates,
        "threshold_backtest": backtest_stats,
        "summary_text": "\n".join(summary_lines),
    }


def _approx_usd_multiplier(symbol: str, currency_hint: str = "") -> float:
    """Approximate FX multiplier to normalize daily turnover into USD for liquidity filtering."""
    curr = (currency_hint or "").strip()
    curr_up = curr.upper()
    if curr in {"GBp", "GBX"} or curr_up == "GBX":
        return 0.0125
    if curr_up == "GBP":
        return 1.25
    if curr_up == "EUR":
        return 1.08
    if curr_up == "DKK":
        return 0.145
    if curr_up == "CHF":
        return 1.12
    if curr_up == "SEK":
        return 0.095
    if curr_up == "USD":
        return 1.0
    sym_up = symbol.upper()
    if sym_up.endswith(".L"):
        return 0.0125
    if sym_up.endswith((".PA", ".DE", ".AS", ".MC", ".MI", ".BR", ".HE")):
        return 1.08
    if sym_up.endswith(".CO"):
        return 0.145
    return 1.0


def _match_t212_eligibility(
    symbol: str,
    elig_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Match Yahoo Finance symbol against SQLite T212 instrument metadata with punctuation variants."""
    if not elig_map:
        return None
    sym_u = symbol.strip().upper()
    if sym_u in elig_map:
        return elig_map[sym_u]
    if "." in sym_u:
        base, suffix = sym_u.rsplit(".", 1)
        alt1 = f"{base.replace('-', '.')}.{suffix}"
        alt2 = f"{base.replace('.', '-')}.{suffix}"
        for alt in (alt1, alt2):
            if alt in elig_map:
                return elig_map[alt]
    else:
        for alt in (sym_u.replace("-", "."), sym_u.replace(".", "-")):
            if alt in elig_map:
                return elig_map[alt]
    return None


def _is_uk_or_eu_symbol(symbol: str) -> bool:
    return symbol.strip().upper().endswith((".L", ".PA", ".DE", ".AS", ".MC", ".MI", ".BR", ".HE", ".CO"))


def _score_audition_equity(
    symbol: str,
    tech: TechnicalSnapshot,
    hist: pd.DataFrame,
    benchmark_hist: pd.DataFrame | None,
    fund: dict[str, Any] | None,
    avg_turnover_usd: float,
) -> float:
    """Score a vetted equity (0..100) for inclusion in the active 150-stock universe pool."""
    # 1) Profitability & Growth (0..35, midpoint 17.5)
    s_prof = 17.5
    if fund:
        roe = fund.get("roe_pct")
        if roe is not None:
            if roe >= 0:
                s_prof += _linear_scale(float(roe), 8.0, 32.0, 0.0, 7.5)
            else:
                s_prof -= _linear_scale(abs(float(roe)), 0.0, 25.0, 2.0, 7.5)
        margin = fund.get("profit_margin_pct")
        if margin is not None:
            if margin >= 0:
                s_prof += _linear_scale(float(margin), 4.0, 25.0, 0.0, 5.5)
            else:
                s_prof -= _linear_scale(abs(float(margin)), 0.0, 20.0, 1.5, 5.5)
        rev_g = fund.get("revenue_growth_pct")
        if rev_g is not None:
            s_prof += _linear_scale(float(rev_g), -5.0, 22.0, -3.5, 4.5)
    s_prof = max(0.0, min(35.0, s_prof))

    # 2) Balance Sheet & Valuation Discipline (0..25, midpoint 13.0)
    s_bal = 13.0
    if fund:
        dte = fund.get("debt_to_equity")
        if dte is not None:
            if dte <= 60.0:
                s_bal += 3.5
            elif dte <= 120.0:
                s_bal += 1.0
            elif dte > 180.0:
                s_bal -= _linear_scale(float(dte), 180.0, 350.0, 1.5, 5.0)
        pe = fund.get("forward_pe") or fund.get("trailing_pe")
        if pe is not None:
            if 8.0 <= pe <= 28.0:
                s_bal += 4.0
            elif 28.0 < pe <= 45.0:
                s_bal += 1.5
            elif pe > 65.0 or pe < 0.0:
                s_bal -= 3.5
        analysts = fund.get("analyst_count")
        if analysts is not None and analysts >= 10:
            s_bal += _linear_scale(float(analysts), 10.0, 35.0, 1.0, 3.0)
    s_bal = max(0.0, min(25.0, s_bal))

    # 3) Scale & Liquidity (0..20)
    s_liq = 8.0
    if avg_turnover_usd > 0:
        s_liq += _linear_scale(float(np.log10(max(avg_turnover_usd, 1e5))), 6.7, 9.0, 0.0, 6.0)
    mcap = fund.get("market_cap") if fund else None
    if mcap is not None and mcap > 0:
        s_liq += _linear_scale(float(np.log10(max(float(mcap), 1e8))), 9.0, 11.8, 0.0, 6.0)
    else:
        s_liq += 2.0
    s_liq = max(0.0, min(20.0, s_liq))

    # 4) Long-Term Trend & Relative Strength Health (0..20, midpoint 10.0)
    s_trend = 10.0
    if tech.trend_regime == "BULL":
        s_trend += 4.0
    elif tech.trend_regime == "PULLBACK":
        s_trend += 3.5
    elif tech.trend_regime == "WEAK":
        s_trend -= 4.0
    rs_info = compute_relative_strength(hist, benchmark_hist)
    rs_60d = rs_info.get("rs_60d_pct")
    if rs_60d is not None:
        s_trend += _linear_scale(float(rs_60d), -15.0, 15.0, -3.5, 4.0)
    if tech.ret_60d_pct is not None and tech.ret_60d_pct > 0:
        s_trend += 1.5
    s_trend = max(0.0, min(20.0, s_trend))

    return round(s_prof + s_bal + s_liq + s_trend, 2)


async def run_universe_audition(
    cfg: Any,
    t212: Any,
    market_data: MarketDataService,
    db: Any,
    *,
    sync_online_indices: bool = False,
    sync_t212_metadata: bool = True,
) -> dict[str, Any]:
    """0-Token multi-market audition: T212 tradability + 1y liquidity + cached fundamentals -> active top-N pool."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from src.engine.universe_seeds import resolve_base_universe_symbols

    universe_cfg = cfg.universe
    target_size = max(20, int(getattr(universe_cfg, "audition_target_size", 150) or 150))
    sector_cap = max(5, int(getattr(universe_cfg, "audition_sector_cap", 28) or 28))
    min_uk_eu_slots = max(0, int(getattr(universe_cfg, "audition_min_uk_eu_slots", 30) or 30))
    min_turnover_usd = max(0.0, float(getattr(universe_cfg, "audition_min_avg_dollar_vol_usd", 5_000_000.0) or 0.0))
    ttl_days = max(1, int(getattr(universe_cfg, "audition_fundamentals_ttl_days", 21) or 21))
    info_conc = max(1, int(getattr(universe_cfg, "audition_info_concurrency", 2) or 2))
    info_delay = max(0.0, float(getattr(universe_cfg, "audition_info_delay_seconds", 0.45) or 0.45))

    previous_rows = await db.get_active_universe_pool() if hasattr(db, "get_active_universe_pool") else []
    prev_equities = {str(r["symbol"]).upper() for r in previous_rows if not r.get("is_etf")}

    # Step 1: Resolve multi-market base pool & T212 account-level eligibility
    logger.info("Audition Step 1/4: resolving multi-market base universe & T212 eligibility...")
    base_data = await resolve_base_universe_symbols(cfg, db=db, sync_online=sync_online_indices)
    equities_map: dict[str, list[str]] = dict(base_data.get("equities") or {})
    etf_syms: list[str] = list(base_data.get("etfs") or [])

    elig_map = await db.get_t212_eligibility_by_yahoo() if hasattr(db, "get_t212_eligibility_by_yahoo") else {}
    latest_meta_ms = max((int(v.get("updated_at_ms") or 0) for v in elig_map.values()), default=0)
    stale_meta = (int(time.time() * 1000) - latest_meta_ms) > (20 * 3600 * 1000)
    if sync_t212_metadata and (not elig_map or stale_meta):
        if hasattr(t212, "sync_all_accounts_instruments_metadata"):
            try:
                await t212.sync_all_accounts_instruments_metadata()
                elig_map = await db.get_t212_eligibility_by_yahoo()
            except Exception as exc:
                logger.warning("Audition T212 metadata sync failed; continuing with existing table: %s", exc)
        elif hasattr(t212, "fetch_instruments_metadata"):
            try:
                await t212.fetch_instruments_metadata()
                elig_map = await db.get_t212_eligibility_by_yahoo()
            except Exception as exc:
                logger.warning("Audition T212 metadata sync failed: %s", exc)

    t212_filtered = bool(elig_map)
    tradable_equities: dict[str, dict[str, Any]] = {}
    for sym, idx_sources in equities_map.items():
        match = _match_t212_eligibility(sym, elig_map) if t212_filtered else None
        if t212_filtered and match is None:
            continue
        if match and str(match.get("inst_type") or "").upper() == "ETF":
            if sym not in etf_syms:
                etf_syms.append(sym)
            continue
        tradable_equities[sym] = {
            "symbol": sym,
            "index_sources": idx_sources,
            "currency_hint": str(match.get("currency_code") or "") if match else "",
            "name_hint": str(match.get("short_name") or "") if match else "",
            "isa_eligible": bool(match.get("isa_eligible", True)) if match else True,
            "invest_eligible": bool(match.get("invest_eligible", True)) if match else True,
        }

    tradable_etfs: list[dict[str, Any]] = []
    for sym in etf_syms:
        match = _match_t212_eligibility(sym, elig_map) if t212_filtered else None
        if t212_filtered and match is None:
            continue
        tradable_etfs.append(
            {
                "symbol": sym,
                "name": str(match.get("short_name") or sym) if match else sym,
                "currency": str(match.get("currency_code") or "GBP") if match else "GBP",
                "isa_eligible": bool(match.get("isa_eligible", True)) if match else True,
                "invest_eligible": bool(match.get("invest_eligible", True)) if match else True,
            }
        )

    # Step 2: Batched 1y OHLCV incremental update & local liquidity/structure pre-filter (0 .info calls)
    benchmarks = ["SPY", "ISF.L", "EXS1.DE"]
    eq_syms = list(tradable_equities.keys())
    logger.info(
        "Audition Step 2/4: fetching/updating 1y OHLCV for %d tradable equities (+%d ETFs)...",
        len(eq_syms),
        len(tradable_etfs),
    )
    histories = await market_data.get_histories([*eq_syms, *benchmarks], period="1y") if eq_syms else {}

    liquid_candidates: list[tuple[str, TechnicalSnapshot, pd.DataFrame, float]] = []
    for sym in eq_syms:
        hist = histories.get(sym)
        if hist is None or hist.empty or len(hist) < 120:
            continue
        ext = _extract_ohlcv_series(hist)
        if ext is None:
            continue
        _, _, _, closes, vols = ext
        if len(closes) < 120:
            continue
        meta = tradable_equities[sym]
        mult = _approx_usd_multiplier(sym, meta["currency_hint"])
        avg_turnover_usd = float((closes.tail(20) * vols.tail(20)).mean() * mult)
        if min_turnover_usd > 0 and avg_turnover_usd < min_turnover_usd:
            continue

        tech = compute_technical_snapshot(sym, hist)
        if tech is None:
            continue
        # Exclude severe long-term structural collapse before spending .info calls
        if (
            tech.ma200
            and tech.last_close < tech.ma200 * 0.55
            and (tech.ret_60d_pct or 0.0) < -25.0
        ):
            continue
        liquid_candidates.append((sym, tech, hist, avg_turnover_usd))

    # Step 3: Load cached fundamentals from SQLite & gently fetch missing/stale with 429 circuit breaker
    liquid_syms = [item[0] for item in liquid_candidates]
    logger.info(
        "Audition Step 3/4: %d/%d equities passed liquidity/structure pre-filter; loading fundamentals (cache TTL %dd)...",
        len(liquid_syms),
        len(eq_syms),
        ttl_days,
    )
    fund_map, fund_stats = await market_data.get_fundamentals_cached(
        liquid_syms,
        max_age_days=ttl_days,
        concurrency=info_conc,
        delay_seconds=info_delay,
        max_consecutive_429=3,
    )

    # Step 4: Score and select sector-balanced Top N (default 150) with UK/EU floor
    scored_rows: list[dict[str, Any]] = []
    for sym, tech, hist, avg_turnover_usd in liquid_candidates:
        meta = tradable_equities[sym]
        fund = fund_map.get(sym)
        if fund and str(fund.get("quote_type") or "").upper() == "ETF":
            continue
        bench_sym = resolve_market_benchmark(sym)[0]
        q_score = _score_audition_equity(
            sym,
            tech,
            hist,
            histories.get(bench_sym),
            fund,
            avg_turnover_usd,
        )
        sector = str((fund or {}).get("sector") or "").strip() or "Unknown"
        scored_rows.append(
            {
                "symbol": sym,
                "name": str((fund or {}).get("short_name") or meta["name_hint"] or sym),
                "exchange": str((fund or {}).get("exchange") or ""),
                "currency": str((fund or {}).get("currency") or meta["currency_hint"] or ""),
                "sector": sector,
                "industry": str((fund or {}).get("industry") or ""),
                "index_sources": meta["index_sources"],
                "quality_score": q_score,
                "avg_daily_turnover_usd": round(avg_turnover_usd, 1),
                "isa_eligible": meta["isa_eligible"],
                "invest_eligible": meta["invest_eligible"],
                "is_etf": False,
            }
        )

    scored_rows.sort(key=lambda r: (r["quality_score"], r["avg_daily_turnover_usd"]), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_set: set[str] = set()
    sector_counts: dict[str, int] = {}

    def _try_pick(row: dict[str, Any], *, enforce_cap: bool = True) -> bool:
        sym = row["symbol"]
        if sym in selected_set or len(selected) >= target_size:
            return False
        sec = row["sector"]
        cap = min(sector_cap, max(12, target_size // 5)) if sec == "Unknown" else sector_cap
        if enforce_cap and sector_counts.get(sec, 0) >= cap:
            return False
        selected.append(row)
        selected_set.add(sym)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        return True

    # Phase A: Reserve UK/EU floor slots first so FTSE + EuroStoxx remain well-represented
    if min_uk_eu_slots > 0:
        uk_eu_rows = [r for r in scored_rows if _is_uk_or_eu_symbol(r["symbol"])]
        for row in uk_eu_rows:
            if len(selected) >= min(min_uk_eu_slots, target_size):
                break
            _try_pick(row, enforce_cap=True)

    # Phase B: Fill remaining slots up to target_size respecting sector cap
    for row in scored_rows:
        if len(selected) >= target_size:
            break
        _try_pick(row, enforce_cap=True)

    # Phase C: Relax sector cap only if still below target_size
    if len(selected) < target_size:
        for row in scored_rows:
            if len(selected) >= target_size:
                break
            _try_pick(row, enforce_cap=False)

    etf_rows = [
        {
            "symbol": e["symbol"],
            "name": e["name"],
            "exchange": "LSE",
            "currency": e["currency"] or "GBP",
            "sector": "ETF",
            "industry": "UCITS ETF",
            "index_sources": ["CURATED_LSE_ETF"],
            "quality_score": 100.0,
            "avg_daily_turnover_usd": 0.0,
            "isa_eligible": e["isa_eligible"],
            "invest_eligible": e["invest_eligible"],
            "is_etf": True,
        }
        for e in tradable_etfs
    ]

    uk_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M")
    all_persisted = [*selected, *etf_rows]
    if all_persisted and hasattr(db, "replace_active_universe_pool"):
        await db.replace_active_universe_pool(all_persisted, updated_at_uk=uk_now)

    new_equities = {r["symbol"] for r in selected}
    added_syms = sorted(new_equities - prev_equities)
    removed_syms = sorted(prev_equities - new_equities) if prev_equities else []

    by_region = {"US": 0, "UK": 0, "EU": 0}
    for r in selected:
        sym = r["symbol"]
        if sym.endswith(".L"):
            by_region["UK"] += 1
        elif _is_uk_or_eu_symbol(sym):
            by_region["EU"] += 1
        else:
            by_region["US"] += 1

    logger.info(
        "Audition Step 4/4 complete: selected %d equities (US=%d, UK=%d, EU=%d) + %d LSE ETFs | fundamentals=%s",
        len(selected),
        by_region["US"],
        by_region["UK"],
        by_region["EU"],
        len(etf_rows),
        fund_stats,
    )

    return {
        "updated_at_uk": uk_now,
        "base_equities_count": len(equities_map),
        "t212_filtered": t212_filtered,
        "t212_tradable_count": len(tradable_equities),
        "liquidity_passed_count": len(liquid_candidates),
        "selected_equities_count": len(selected),
        "selected_etfs_count": len(etf_rows),
        "by_region": by_region,
        "by_sector": dict(sorted(sector_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "added_symbols": added_syms,
        "removed_symbols": removed_syms,
        "fundamentals_stats": fund_stats,
        "top_preview": [
            f"{r['symbol']}({r['quality_score']:.0f}, {r['sector']})"
            for r in selected[:10]
        ],
    }
