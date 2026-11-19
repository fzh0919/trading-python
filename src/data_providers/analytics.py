"""Deterministic pandas/numpy quantitative indicators, CYQ chip distribution, correlation matrix, and FactSheet builder."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import PortfolioData


class CYQMetrics(BaseModel):
    """Deterministic chip distribution (CYQ) summary computed from daily OHLCV."""

    benefit_part_pct: float
    avg_cost: float
    concentration_90_pct: float
    cost_90_low: float
    cost_90_high: float
    concentration_70_pct: float | None = None
    regime: str = "筹码均衡"  # 底部单峰密集 | 上方套牢峰 | 获利盘高企 | 筹码发散 | 筹码均衡


class TechnicalSnapshot(BaseModel):
    symbol: str
    last_close: float
    ret_1d_pct: float | None = None
    ret_5d_pct: float | None = None
    ret_20d_pct: float | None = None
    ret_60d_pct: float | None = None
    ma20: float | None = None
    ma50: float | None = None
    ma200: float | None = None
    dist_ma20_pct: float | None = None
    dist_ma50_pct: float | None = None
    dist_ma200_pct: float | None = None
    rsi14: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    drawdown_20d_pct: float | None = None
    drawdown_60d_pct: float | None = None
    drawdown_52w_pct: float | None = None
    volatility_60d_pct: float | None = None
    vol_60d_percentile: float | None = None  # IMP-17: 60d annualized vol ranked within its own 2y distribution
    trend_regime: str = "NEUTRAL"  # BULL | PULLBACK | WEAK | NEUTRAL

    # IMP-06: Multi-timeframe resonance, volatility/volume rulers, and trend strength
    atr14: float | None = None
    atr_pct: float | None = None
    volume_ratio_20d: float | None = None
    weekly_ma10: float | None = None
    weekly_ma30: float | None = None
    weekly_bullish: bool | None = None
    weekly_rsi14: float | None = None
    bb_width_pct: float | None = None
    bb_squeeze: bool = False
    adx14: float | None = None

    # IMP-06: Local 0-Token CYQ chip distribution metrics
    cyq_benefit_part_pct: float | None = None
    cyq_avg_cost: float | None = None
    cyq_concentration_90_pct: float | None = None
    cyq_cost_90_low: float | None = None
    cyq_cost_90_high: float | None = None
    cyq_regime: str | None = None

    # IMP-06: Ex-dividend window linkage for IMP-07 risk rules (prevents false stop/breakdown on ex-div drop)
    days_to_ex_dividend: int | None = None
    is_ex_div_window: bool = False      # True when -3 <= days_to_ex_dividend <= 3
    is_ex_div_drop_guard: bool = False  # True when -3 <= days_to_ex_dividend <= 0 (just went ex-div)


def compute_rsi_wilder(closes: pd.Series, period: int = 14) -> float | None:
    """Compute classic Wilder RSI(14) deterministically on closing prices."""
    s = closes.dropna().astype(float)
    if len(s) < period + 1:
        return None
    delta = s.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    last_loss = float(avg_loss.iloc[-1])
    last_gain = float(avg_gain.iloc[-1])
    if last_loss == 0.0 and last_gain == 0.0:
        return 50.0
    if last_loss == 0.0:
        return 100.0
    rs = last_gain / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _resolve_col(df: pd.DataFrame, target: str) -> str | None:
    """Case-insensitive column lookup for OHLCV DataFrames."""
    if target in df.columns:
        return target
    lower_target = target.lower()
    for col in df.columns:
        if str(col).lower() == lower_target:
            return str(col)
    return None


def _normalize_adjusted_ohlcv(hist: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC are proportionally adjusted for splits/dividends and free of GBp/GBP or unadjusted split jumps.

    1) If an 'Adj Close' column is present and differs from 'Close', scales Open/High/Low/Close proportionally.
    2) If a single-day corrupted tick (>4.0x or <0.25x jump that immediately reverts next session) is present, drops it.
    3) If a permanent unadjusted split or GBp<->GBP unit step (>4.0x or <0.25x) is present, back-adjusts prior OHLC
       so CYQ chip peaks and moving averages never mistake pre-split prices for overhead trapped supply.
    """
    close_col = _resolve_col(hist, "Close")
    if not close_col:
        return hist

    adj_col = _resolve_col(hist, "Adj Close") or _resolve_col(hist, "Adj_Close") or _resolve_col(hist, "AdjClose")
    high_col = _resolve_col(hist, "High")
    low_col = _resolve_col(hist, "Low")
    open_col = _resolve_col(hist, "Open")
    price_cols = [c for c in (close_col, open_col, high_col, low_col) if c]

    c_series = pd.to_numeric(hist[close_col], errors="coerce")
    valid_c = c_series.dropna()
    if len(valid_c) < 2:
        return hist

    c_vals = valid_c.to_numpy(dtype=np.float64, copy=False)
    ratios = np.ones(len(c_vals), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios[1:] = np.where(c_vals[:-1] > 0, c_vals[1:] / c_vals[:-1], 1.0)
    has_jump = bool(np.any((ratios >= 4.0) | (ratios <= 0.25)))

    needs_adj_scale = False
    if adj_col:
        adj_series = pd.to_numeric(hist[adj_col], errors="coerce")
        both = pd.DataFrame({"c": c_series, "a": adj_series}).dropna()
        both = both[(both["c"] > 0) & (both["a"] > 0)]
        if not both.empty:
            rel_diff = ((both["a"] - both["c"]).abs() / both["c"]).max()
            needs_adj_scale = bool(rel_diff > 1e-4)

    if not needs_adj_scale and not has_jump:
        return hist

    out = hist.copy()
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    if needs_adj_scale and adj_col:
        adj_vals = pd.to_numeric(out[adj_col], errors="coerce").astype(float)
        raw_c = out[close_col]
        valid_mask = (raw_c > 0) & (adj_vals > 0) & raw_c.notna() & adj_vals.notna()
        scale_factor = pd.Series(1.0, index=out.index, dtype=float)
        scale_factor.loc[valid_mask] = adj_vals.loc[valid_mask] / raw_c.loc[valid_mask]
        for col in price_cols:
            out[col] = out[col] * scale_factor

    # Recheck for unadjusted splits or 100x GBp/GBP glitches after Adj Close scaling
    out = out[out[close_col] > 0].copy()
    if len(out) < 2:
        return out

    # Step 1: Drop isolated 1-bar bad ticks (e.g. single day reported in GBp instead of GBP, reverting next bar)
    c_arr = out[close_col].to_numpy(dtype=np.float64, copy=True)
    keep_mask = np.ones(len(c_arr), dtype=bool)
    for i in range(1, len(c_arr) - 1):
        prev_c = c_arr[i - 1]
        curr_c = c_arr[i]
        next_c = c_arr[i + 1]
        if prev_c > 0 and curr_c > 0 and next_c > 0:
            r_in = curr_c / prev_c
            r_recover = next_c / prev_c
            if (r_in >= 4.0 or r_in <= 0.25) and (0.5 <= r_recover <= 2.0):
                keep_mask[i] = False
                c_arr[i] = prev_c
    if not np.all(keep_mask):
        out = out.iloc[keep_mask].copy()

    # Step 2: Back-adjust permanent step discontinuities (unadjusted splits / permanent unit switches)
    c_arr = out[close_col].to_numpy(dtype=np.float64)
    o_arr = out[open_col].to_numpy(dtype=np.float64) if open_col else c_arr
    cum_scale = np.ones(len(c_arr), dtype=np.float64)
    modified = False
    for i in range(1, len(c_arr)):
        prev_c = c_arr[i - 1] * cum_scale[i - 1]
        curr_c = c_arr[i]
        if prev_c <= 0 or curr_c <= 0:
            continue
        r = curr_c / prev_c
        if r >= 4.0 or r <= 0.25:
            curr_o = o_arr[i] if (open_col and not math.isnan(o_arr[i]) and o_arr[i] > 0) else curr_c
            r_open = curr_o / prev_c
            step = r_open if (r_open >= 4.0 or r_open <= 0.25) else r
            cum_scale[:i] *= step
            modified = True

    if modified:
        for col in price_cols:
            out[col] = out[col].to_numpy(dtype=np.float64) * cum_scale

    return out


def compute_cyq_metrics(
    hist: pd.DataFrame | None,
    cyq_days: int = 210,
    factor: int = 150,
) -> CYQMetrics | None:
    """Ported from REF_stock-master/instock/core/kline/cyq.py (CYQCalculator).

    Uses triangular volume price distribution across [Low, High] with relative-volume
    turnover decay (or explicit turnover % when present), automatic pre-adjustment/split
    continuity normalization, and adaptive price grid step for both low-priced GBP/EUR
    assets and high-priced GBp (pence) stocks.
    """
    if hist is None or hist.empty or factor < 10:
        return None

    norm_hist = _normalize_adjusted_ohlcv(hist)
    close_col = _resolve_col(norm_hist, "Close")
    high_col = _resolve_col(norm_hist, "High")
    low_col = _resolve_col(norm_hist, "Low")
    vol_col = _resolve_col(norm_hist, "Volume")
    open_col = _resolve_col(norm_hist, "Open")
    turnover_col = _resolve_col(norm_hist, "Turnover")

    if not close_col or not high_col or not low_col or not vol_col:
        return None

    work = pd.DataFrame(
        {
            "close": pd.to_numeric(norm_hist[close_col], errors="coerce"),
            "high": pd.to_numeric(norm_hist[high_col], errors="coerce"),
            "low": pd.to_numeric(norm_hist[low_col], errors="coerce"),
            "volume": pd.to_numeric(norm_hist[vol_col], errors="coerce"),
        },
        index=norm_hist.index,
    )
    if open_col:
        work["open"] = pd.to_numeric(norm_hist[open_col], errors="coerce").fillna(work["close"])
    else:
        work["open"] = work["close"]

    if turnover_col:
        work["turnover"] = pd.to_numeric(norm_hist[turnover_col], errors="coerce")

    # Filter out invalid prices or zero-volume suspended sessions
    work = work.dropna(subset=["close", "high", "low", "volume"])
    work = work[(work["close"] > 0) & (work["low"] > 0) & (work["high"] >= work["low"]) & (work["volume"] > 0)]
    if len(work) < 15:
        return None

    kdata = work.tail(cyq_days).copy()
    maxprice = float(kdata["high"].max())
    minprice = float(kdata["low"].min())
    if maxprice <= 0 or minprice <= 0 or maxprice < minprice:
        return None

    # Adaptive grid precision: prevents coarse quantization on low-priced GBP/USD stocks (< 2.0)
    price_span = maxprice - minprice
    raw_step = price_span / max(factor - 1, 1)
    accuracy = max(1e-4, raw_step) if price_span > 0 else max(1e-4, maxprice * 0.001)

    current_price = float(kdata["close"].iloc[-1])
    yrange = minprice + np.arange(factor, dtype=np.float64) * accuracy
    xdata = np.zeros(factor, dtype=np.float64)

    vol_ma20 = kdata["volume"].rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)
    opens = kdata["open"].to_numpy(dtype=np.float64)
    closes = kdata["close"].to_numpy(dtype=np.float64)
    highs = kdata["high"].to_numpy(dtype=np.float64)
    lows = kdata["low"].to_numpy(dtype=np.float64)
    vols = kdata["volume"].to_numpy(dtype=np.float64)
    has_turnover = "turnover" in kdata.columns and kdata["turnover"].notna().all()
    turnovers = kdata["turnover"].to_numpy(dtype=np.float64) if has_turnover else None

    for idx in range(len(kdata)):
        open_p = opens[idx]
        close_p = closes[idx]
        high_p = highs[idx]
        low_p = lows[idx]
        avg_p = (open_p + close_p + high_p + low_p) / 4.0

        if turnovers is not None:
            turnover_rate = float(np.clip(turnovers[idx] / 100.0, 0.001, 1.0))
        else:
            base_v = vol_ma20[idx] if vol_ma20[idx] > 0 else vols[idx]
            rel_v = (vols[idx] / base_v) if base_v > 0 else 1.0
            turnover_rate = float(np.clip(rel_v * 0.025, 0.003, 0.25))

        xdata *= 1.0 - turnover_rate

        h_idx = min(factor - 1, max(0, int((high_p - minprice) / accuracy)))
        l_idx = min(factor - 1, max(0, int((low_p - minprice) / accuracy + 0.99)))

        if high_p <= low_p + 1e-9 or h_idx <= l_idx:
            g_idx = min(factor - 1, max(0, int((avg_p - minprice) / accuracy)))
            xdata[g_idx] += ((factor - 1) * turnover_rate) / 2.0
        else:
            g0 = 2.0 / (high_p - low_p)
            grid_slice = yrange[l_idx : h_idx + 1]
            lower_mask = grid_slice <= avg_p
            weights = np.empty_like(grid_slice)

            denom_low = avg_p - low_p
            if abs(denom_low) < 1e-8:
                weights[lower_mask] = 1.0
            else:
                weights[lower_mask] = (grid_slice[lower_mask] - low_p) / denom_low

            upper_mask = ~lower_mask
            denom_high = high_p - avg_p
            if abs(denom_high) < 1e-8:
                weights[upper_mask] = 1.0
            else:
                weights[upper_mask] = (high_p - grid_slice[upper_mask]) / denom_high

            weights = np.clip(weights, 0.0, None)
            xdata[l_idx : h_idx + 1] += weights * g0 * turnover_rate

    total_chips = float(np.sum(xdata))
    if total_chips <= 1e-12:
        return None

    cumsum_chips = np.cumsum(xdata)

    def _get_cost_by_chip(target_chip: float) -> float:
        pos = int(np.searchsorted(cumsum_chips, target_chip, side="right"))
        pos = min(factor - 1, max(0, pos))
        return float(minprice + pos * accuracy)

    def _compute_band(pct: float) -> tuple[float, float, float]:
        low_q = (1.0 - pct) / 2.0
        high_q = (1.0 + pct) / 2.0
        p_low = _get_cost_by_chip(total_chips * low_q)
        p_high = _get_cost_by_chip(total_chips * high_q)
        conc = 0.0 if (p_low + p_high) <= 0 else ((p_high - p_low) / (p_low + p_high)) * 100.0
        return p_low, p_high, conc

    below_chips = float(np.sum(xdata[yrange <= current_price + 1e-9]))
    benefit_part_pct = float(np.clip((below_chips / total_chips) * 100.0, 0.0, 100.0))
    avg_cost = _get_cost_by_chip(total_chips * 0.5)
    c90_low, c90_high, conc_90_pct = _compute_band(0.90)
    _, _, conc_70_pct = _compute_band(0.70)

    # Check overhead trapped supply vs bottom single-peak accumulation
    overhead_mask = yrange > current_price * 1.015
    overhead_ratio_pct = (float(np.sum(xdata[overhead_mask])) / total_chips) * 100.0

    if benefit_part_pct <= 25.0 or (current_price < avg_cost * 0.97 and overhead_ratio_pct >= 55.0):
        regime = "上方套牢峰"
    elif conc_90_pct <= 12.0 and benefit_part_pct >= 50.0 and current_price <= avg_cost * 1.12:
        regime = "底部单峰密集"
    elif benefit_part_pct >= 88.0 and conc_90_pct <= 14.0:
        regime = "获利盘高企"
    elif conc_90_pct <= 12.0:
        regime = "单峰密集"
    elif conc_90_pct >= 20.0:
        regime = "筹码发散"
    else:
        regime = "筹码均衡"

    return CYQMetrics(
        benefit_part_pct=round(benefit_part_pct, 1),
        avg_cost=round(avg_cost, 4),
        concentration_90_pct=round(conc_90_pct, 2),
        cost_90_low=round(c90_low, 4),
        cost_90_high=round(c90_high, 4),
        concentration_70_pct=round(conc_70_pct, 2),
        regime=regime,
    )


def _compute_atr_and_adx(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> tuple[float | None, float | None]:
    """Compute Wilder ATR(14) in native instrument currency and ADX(14) trend strength using fast numpy arrays."""
    h_arr = highs.to_numpy(dtype=np.float64, copy=False)
    l_arr = lows.to_numpy(dtype=np.float64, copy=False)
    c_arr = closes.to_numpy(dtype=np.float64, copy=False)
    mask = ~(np.isnan(h_arr) | np.isnan(l_arr) | np.isnan(c_arr))
    h_arr = h_arr[mask]
    l_arr = l_arr[mask]
    c_arr = c_arr[mask]
    n = len(c_arr)
    if n < period + 1:
        return None, None

    prev_c = np.empty(n, dtype=np.float64)
    prev_c[0] = c_arr[0]
    prev_c[1:] = c_arr[:-1]

    tr = np.maximum(h_arr - l_arr, np.maximum(np.abs(h_arr - prev_c), np.abs(l_arr - prev_c)))
    atr_series = pd.Series(tr).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().to_numpy(dtype=np.float64)
    atr_val = float(atr_series[-1])
    if math.isnan(atr_val) or atr_val <= 0:
        return None, None

    up_move = np.empty(n, dtype=np.float64)
    down_move = np.empty(n, dtype=np.float64)
    up_move[0] = 0.0
    down_move[0] = 0.0
    up_move[1:] = h_arr[1:] - h_arr[:-1]
    down_move[1:] = l_arr[:-1] - l_arr[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    sm_plus = pd.Series(plus_dm).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().to_numpy(dtype=np.float64)
    sm_minus = pd.Series(minus_dm).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(atr_series > 1e-12, 100.0 * (sm_plus / atr_series), 0.0)
        mdi = np.where(atr_series > 1e-12, 100.0 * (sm_minus / atr_series), 0.0)
        denom = pdi + mdi
        dx = np.where(denom > 1e-12, 100.0 * (np.abs(pdi - mdi) / denom), np.nan)

    valid_dx = dx[~np.isnan(dx)]
    adx_val: float | None = None
    if len(valid_dx) >= 6:
        adx_s = pd.Series(valid_dx).ewm(alpha=1.0 / period, min_periods=min(period, len(valid_dx)), adjust=False).mean()
        raw_adx = float(adx_s.iloc[-1])
        if not math.isnan(raw_adx):
            adx_val = round(raw_adx, 1)

    return round(atr_val, 4), adx_val


def _compute_weekly_resonance(
    closes: pd.Series,
) -> tuple[float | None, float | None, bool | None, float | None]:
    """Sample 5-trading-day weekly bars anchored to latest close and compute weekly MA10/MA30 and weekly RSI14."""
    if len(closes) < 50:
        return None, None, None, None

    weekly = closes.iloc[::-5].iloc[::-1]
    if len(weekly) < 10:
        return None, None, None, None

    w_arr = weekly.to_numpy(dtype=np.float64, copy=False)
    w_ma10_raw = float(np.mean(w_arr[-10:]))
    w_ma10 = None if math.isnan(w_ma10_raw) else round(w_ma10_raw, 4)

    w_ma30: float | None = None
    if len(w_arr) >= 30:
        w_ma30_raw = float(np.mean(w_arr[-30:]))
        if not math.isnan(w_ma30_raw):
            w_ma30 = round(w_ma30_raw, 4)

    if w_ma10 is not None and w_ma30 is not None:
        w_bullish: bool | None = bool(w_ma10 > w_ma30 and float(w_arr[-1]) >= w_ma30)
    elif w_ma10 is not None:
        w_bullish = bool(float(w_arr[-1]) >= w_ma10)
    else:
        w_bullish = None

    w_rsi = compute_rsi_wilder(weekly, 14)
    if w_rsi is not None:
        w_rsi = round(w_rsi, 1)

    return w_ma10, w_ma30, w_bullish, w_rsi


def _compute_bollinger_squeeze(closes: pd.Series, window: int = 20) -> tuple[float | None, bool]:
    """Compute 20-day Bollinger Bandwidth (%) and detect volatility contraction squeeze."""
    if len(closes) < window:
        return None, False

    sma = closes.rolling(window).mean()
    std = closes.rolling(window).std(ddof=0)
    bw_series = ((4.0 * std) / sma.replace(0.0, np.nan)) * 100.0
    valid_bw = bw_series.dropna()
    if valid_bw.empty:
        return None, False

    curr_bw = float(valid_bw.iloc[-1])
    if math.isnan(curr_bw):
        return None, False

    tail_bw = valid_bw.tail(120)
    if len(tail_bw) >= 20:
        q20 = float(np.percentile(tail_bw.to_numpy(dtype=float), 20))
        is_squeeze = bool((curr_bw <= q20 and curr_bw <= 18.0) or curr_bw <= 6.0)
    else:
        is_squeeze = bool(curr_bw <= 6.0)

    return round(curr_bw, 2), is_squeeze


def compute_drawdown_since_date(
    hist: pd.DataFrame | None,
    created_at: str | None = None,
    max_lookback_days: int = 60,
) -> float | None:
    """Compute drawdown (%) from peak close since max(max_lookback_days ago, position created_at date).

    Prevents pre-entry historical highs from triggering false trailing-profit stop alerts (for IMP-07).
    """
    if hist is None or hist.empty:
        return None
    norm_hist = _normalize_adjusted_ohlcv(hist)
    close_col = _resolve_col(norm_hist, "Close")
    if not close_col:
        return None

    closes = pd.to_numeric(norm_hist[close_col], errors="coerce").dropna()
    if len(closes) < 2:
        return None

    window_closes = closes.tail(max_lookback_days).copy()
    last = float(window_closes.iloc[-1])
    if last <= 0:
        return None

    if created_at and str(created_at).strip():
        try:
            entry_ts = pd.to_datetime(str(created_at).strip(), utc=True).tz_localize(None).normalize()
            idx_dates = pd.to_datetime(window_closes.index).tz_localize(None).normalize()
            post_entry = window_closes[idx_dates >= entry_ts]
            if not post_entry.empty:
                window_closes = post_entry
        except Exception:
            pass

    peak = float(window_closes.max())
    return ((last / peak) - 1.0) * 100.0 if peak > 0 else None


def _compute_ex_div_flags(
    days_to_ex_div: int | None,
) -> tuple[int | None, bool, bool]:
    """Return (days_to_ex_dividend, is_ex_div_window [-3..3], is_ex_div_drop_guard [-3..0])."""
    if days_to_ex_div is None:
        return None, False, False
    try:
        d = int(days_to_ex_div)
    except (TypeError, ValueError):
        return None, False, False
    return d, bool(-3 <= d <= 3), bool(-3 <= d <= 0)


def compute_technical_snapshot(
    symbol: str,
    hist: pd.DataFrame | None,
    position_created_at: str | None = None,
    bundle: TickerMarketBundle | None = None,
) -> TechnicalSnapshot | None:
    """Compute moving averages, RSI14, ATR14, weekly resonance, BB squeeze, ADX14, CYQ, and ex-div flags."""
    if hist is None or hist.empty:
        return None
    norm_hist = _normalize_adjusted_ohlcv(hist)
    close_col = _resolve_col(norm_hist, "Close")
    if not close_col:
        return None
    closes = pd.to_numeric(norm_hist[close_col], errors="coerce").dropna().astype(float)
    closes = closes[closes > 0]
    if closes.empty:
        return None

    last = float(closes.iloc[-1])

    def _ret(lookback: int) -> float | None:
        if len(closes) <= lookback:
            return None
        prev = float(closes.iloc[-1 - lookback])
        return ((last / prev) - 1.0) * 100.0 if prev > 0 else None

    def _ma(window: int) -> float | None:
        if len(closes) < window:
            return None
        val = float(closes.rolling(window).mean().iloc[-1])
        return None if math.isnan(val) else val

    ma20 = _ma(20)
    ma50 = _ma(50)
    ma200 = _ma(200)

    def _dist(ma: float | None) -> float | None:
        return ((last / ma) - 1.0) * 100.0 if ma and ma > 0 else None

    dist20 = _dist(ma20)
    dist50 = _dist(ma50)
    dist200 = _dist(ma200)

    def _swing_dd(window: int) -> float | None:
        if len(closes) < min(window, 10):
            return None
        hi = float(closes.tail(window).max())
        return ((last / hi) - 1.0) * 100.0 if hi > 0 else None

    high_52w = float(closes.tail(252).max())
    low_52w = float(closes.tail(252).min())
    dd_20d = _swing_dd(20)
    dd_60d = (
        compute_drawdown_since_date(norm_hist, position_created_at, max_lookback_days=60)
        if position_created_at
        else _swing_dd(60)
    )
    dd_52w = ((last / high_52w) - 1.0) * 100.0 if high_52w > 0 else None

    daily_ret = closes.pct_change().dropna().tail(60)
    vol_60d = float(daily_ret.std(ddof=1) * np.sqrt(252) * 100.0) if len(daily_ret) >= 15 else None
    if vol_60d is not None and math.isnan(vol_60d):
        vol_60d = None

    # IMP-17: where does today's 60d vol sit inside its own 2y rolling distribution?
    vol_pctile: float | None = None
    if len(closes) >= 140:
        roll_vol = (closes.pct_change().rolling(60).std(ddof=1) * np.sqrt(252.0) * 100.0).dropna()
        if len(roll_vol) >= 60:
            current_vol = float(roll_vol.iloc[-1])
            if np.isfinite(current_vol) and current_vol > 0.0:
                vol_pctile = round(float((roll_vol <= current_vol).mean()) * 100.0, 1)

    rsi14 = compute_rsi_wilder(closes, 14)

    regime = "NEUTRAL"
    if ma50 is not None and ma200 is not None:
        if last >= ma50 and ma50 >= ma200:
            regime = "BULL"
        elif last < ma50 and last >= ma200 * 0.98:
            regime = "PULLBACK"
        elif last < ma200:
            regime = "WEAK"
    elif ma50 is not None:
        regime = "BULL" if last >= ma50 else "PULLBACK"

    # ATR(14) & ADX(14) when High/Low columns exist
    high_col = _resolve_col(norm_hist, "High")
    low_col = _resolve_col(norm_hist, "Low")
    atr14: float | None = None
    atr_pct: float | None = None
    adx14: float | None = None
    if high_col and low_col:
        highs = pd.to_numeric(norm_hist[high_col], errors="coerce")
        lows = pd.to_numeric(norm_hist[low_col], errors="coerce")
        atr14, adx14 = _compute_atr_and_adx(highs, lows, closes, period=14)
        if atr14 is not None and last > 0:
            atr_pct = round((atr14 / last) * 100.0, 2)

    # 20-day Volume Ratio (ignoring zero-volume suspended bars)
    vol_col = _resolve_col(norm_hist, "Volume")
    vol_ratio_20d: float | None = None
    if vol_col:
        vols = pd.to_numeric(norm_hist[vol_col], errors="coerce").dropna().astype(float)
        vols = vols[vols > 0]
        if len(vols) >= 5:
            last_vol = float(vols.iloc[-1])
            avg_vol_20 = float(vols.tail(20).mean())
            if avg_vol_20 > 0:
                vol_ratio_20d = round(last_vol / avg_vol_20, 2)

    # Weekly timeframe resonance & Bollinger squeeze
    w_ma10, w_ma30, w_bullish, w_rsi14 = _compute_weekly_resonance(closes)
    bb_width_pct, bb_squeeze = _compute_bollinger_squeeze(closes, window=20)

    # CYQ chip distribution metrics
    cyq = compute_cyq_metrics(norm_hist, cyq_days=210, factor=150)

    raw_ex_days = getattr(bundle, "days_to_ex_dividend", None) if bundle else None
    ex_days, is_ex_win, is_ex_guard = _compute_ex_div_flags(raw_ex_days)

    return TechnicalSnapshot(
        symbol=symbol,
        last_close=last,
        ret_1d_pct=_ret(1),
        ret_5d_pct=_ret(5),
        ret_20d_pct=_ret(20),
        ret_60d_pct=_ret(60),
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        dist_ma20_pct=dist20,
        dist_ma50_pct=dist50,
        dist_ma200_pct=dist200,
        rsi14=rsi14,
        high_52w=high_52w,
        low_52w=low_52w,
        drawdown_20d_pct=dd_20d,
        drawdown_60d_pct=dd_60d,
        drawdown_52w_pct=dd_52w,
        volatility_60d_pct=vol_60d,
        vol_60d_percentile=vol_pctile,
        trend_regime=regime,
        atr14=atr14,
        atr_pct=atr_pct,
        volume_ratio_20d=vol_ratio_20d,
        weekly_ma10=w_ma10,
        weekly_ma30=w_ma30,
        weekly_bullish=w_bullish,
        weekly_rsi14=w_rsi14,
        bb_width_pct=bb_width_pct,
        bb_squeeze=bb_squeeze,
        adx14=adx14,
        cyq_benefit_part_pct=cyq.benefit_part_pct if cyq else None,
        cyq_avg_cost=cyq.avg_cost if cyq else None,
        cyq_concentration_90_pct=cyq.concentration_90_pct if cyq else None,
        cyq_cost_90_low=cyq.cost_90_low if cyq else None,
        cyq_cost_90_high=cyq.cost_90_high if cyq else None,
        cyq_regime=cyq.regime if cyq else None,
        days_to_ex_dividend=ex_days,
        is_ex_div_window=is_ex_win,
        is_ex_div_drop_guard=is_ex_guard,
    )


def _extract_close_series(df: pd.DataFrame | None, lookback_days: int) -> pd.Series | None:
    """Extract normalized daily close series indexed by tz-naive calendar date."""
    if df is None or df.empty:
        return None
    norm_df = _normalize_adjusted_ohlcv(df)
    close_col = _resolve_col(norm_df, "Close")
    if not close_col:
        return None
    s = pd.to_numeric(norm_df[close_col], errors="coerce").dropna().astype(float)
    s = s[s > 0]
    if len(s) < 20:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.tail(lookback_days + 5)


def _extract_daily_returns(df: pd.DataFrame | None, lookback_days: int) -> pd.Series | None:
    s = _extract_close_series(df, lookback_days)
    if s is None:
        return None
    rets = s.pct_change().dropna().tail(lookback_days)
    return rets if len(rets) >= 15 else None


# LSE-listed ETFs whose underlying index is US / Global rather than domestic UK FTSE
_US_OR_GLOBAL_LSE_ETFS: set[str] = {
    "VUAG.L",
    "CSP1.L",
    "VUSA.L",
    "IUSA.L",
    "SPXP.L",
    "GSPX.L",
    "CNX1.L",
    "EQGB.L",
    "EQQQ.L",
    "XNAQ.L",
    "VWRP.L",
    "VWRL.L",
    "SWDA.L",
    "IWDA.L",
    "SSAC.L",
    "FWRG.L",
    "IWQU.L",
}

_EU_LSE_ETFS: set[str] = {
    "VERG.L",
    "VEUR.L",
    "IMEU.L",
}

_EU_SUFFIXES: tuple[str, ...] = (".DE", ".AS", ".PA", ".MI", ".MC", ".BR", ".SW", ".HE", ".ST", ".CO", ".OL")


def compute_window_return_pct(hist: pd.DataFrame | None, days: int) -> float | None:
    """Close-to-close return over the last ``days`` trading bars (None when data is short).

    Callers map calendar windows to bars: 5 bars ≈ 7 days, 21 bars ≈ 30 days.
    """
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        bars = int(days)
    except (TypeError, ValueError):
        return None
    if bars < 1:
        return None
    closes = _extract_close_series(hist, bars + 10)
    if closes is None or len(closes) < bars + 1:
        return None
    start = float(closes.iloc[-(bars + 1)])
    end = float(closes.iloc[-1])
    if start <= 0 or not (np.isfinite(start) and np.isfinite(end)):
        return None
    return ((end - start) / start) * 100.0


def _market_timezone_zone(symbol: str) -> str:
    """Classify trading venue closing timezone ('EU_UK' closes ~16:30 UK vs 'US' closes ~21:00 UK)."""
    clean = (symbol or "").strip().upper()
    if clean.endswith(".L") or clean.endswith(_EU_SUFFIXES) or clean in ("^FTSE", "^STOXX50E"):
        return "EU_UK"
    return "US"


def _is_cross_timezone_pair(sym_a: str, sym_b: str) -> bool:
    """True when one symbol trades on UK/European hours and the other on US hours (~4.5h close lag)."""
    return _market_timezone_zone(sym_a) != _market_timezone_zone(sym_b)


def resolve_market_benchmark(
    symbol: str,
    default_benchmark: str = "SPY",
    uk_benchmark: str = "ISF.L",
    eu_benchmark: str = "EXS1.DE",
) -> list[str]:
    """Return ordered list of preferred market benchmarks for OLS Beta stripping.

    - US equities and LSE-listed US/Global index ETFs (e.g. VUAG.L, CSP1.L, VWRP.L) anchor to SPY.
    - Domestic UK equities (.L) and UK index ETFs anchor to ISF.L (fallback ^FTSE -> SPY).
    - European equities (.DE/.AS/.PA/...) and Europe ETFs anchor to EXS1.DE / VERG.L / FEZ (fallback ISF.L -> SPY).
    """
    clean = (symbol or "").strip().upper()
    if clean in _US_OR_GLOBAL_LSE_ETFS:
        return [default_benchmark]
    if clean in _EU_LSE_ETFS or clean.endswith(_EU_SUFFIXES):
        candidates = [eu_benchmark, "VERG.L", "FEZ", "^STOXX50E", uk_benchmark, default_benchmark]
    elif clean.endswith(".L"):
        candidates = [uk_benchmark, "^FTSE", default_benchmark]
    else:
        candidates = [default_benchmark]
    return [c for c in dict.fromkeys(candidates) if c]


def _ols_residual_and_clone_flag(
    ser: pd.Series,
    bench_rets: pd.Series,
) -> tuple[pd.Series, bool]:
    """Run OLS regression of ser on bench_rets and return (residual_series, is_index_clone)."""
    pair = pd.concat([ser, bench_rets], axis=1, join="inner").dropna()
    if len(pair) < 15:
        return ser, False
    y = pair.iloc[:, 0].to_numpy(dtype=float)
    x = pair.iloc[:, 1].to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=1))
    if var_x <= 1e-12:
        return ser, False
    cov_xy = float(np.cov(y, x, ddof=1)[0, 1])
    beta = cov_xy / var_x
    alpha = float(np.mean(y) - beta * np.mean(x))
    resid = y - (alpha + beta * x)
    raw_std = float(np.std(y, ddof=1))
    resid_std = float(np.std(resid, ddof=1))
    is_clone = bool(beta > 0.7 and raw_std > 1e-12 and (resid_std / raw_std) < 0.22)
    return pd.Series(resid, index=pair.index), is_clone


def _compute_rect_corr(
    df: pd.DataFrame,
    row_syms: list[str],
    col_syms: list[str],
    min_periods: int = 15,
) -> pd.DataFrame:
    """Compute exact Pearson correlation matrix of shape (len(row_syms), len(col_syms)) without O(N^2) blowup."""
    r_valid = [s for s in row_syms if s in df.columns]
    c_valid = [s for s in col_syms if s in df.columns]
    if not r_valid or not c_valid:
        return pd.DataFrame(index=r_valid, columns=c_valid, dtype=float)
    if r_valid == c_valid:
        return df[r_valid].corr(min_periods=min_periods)

    sub_cols = df[c_valid]
    out = pd.DataFrame(index=r_valid, columns=c_valid, dtype=float)
    for r_sym in r_valid:
        ser_r = df[r_sym]
        mask_r = ser_r.notna()
        if int(mask_r.sum()) < min_periods:
            continue
        sub_masked = sub_cols.loc[mask_r]
        counts = sub_masked.notna().sum(axis=0)
        corrs = sub_masked.corrwith(ser_r.loc[mask_r])
        corrs = corrs.where(counts >= min_periods)
        out.loc[r_sym] = corrs
    return out


def compute_correlation_matrix(
    histories: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    lookback_days: int = 60,
    benchmark_symbol: str | None = "SPY",
    uk_benchmark_symbol: str | None = "ISF.L",
    eu_benchmark_symbol: str | None = "EXS1.DE",
    against_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix, stripping regional market Beta when benchmark history is available.

    - Calendar Alignment & Closing Time-Lag Compensation: aligns multi-market close series with ffill(limit=1) for
      asynchronous single-day US/UK/EU holidays and evaluates 2-day rolling returns alongside 1-day returns for
      cross-timezone pairs (16:30 UK close vs 21:00 US close) so cross-Atlantic overlap is not underestimated.
    - Pure index clones (where >95% of variance tracks their benchmark, or sharing the same ETF index family) keep raw
      correlation so duplicate index ETFs (e.g. VUAG.L vs CSP1.L or SPY) remain flagged as overlapping.
    - Individual stocks / active names use regional OLS market-residual correlation (US -> SPY, UK .L -> ISF.L,
      EU -> EXS1.DE/VERG.L, with graceful fallback to SPY) so market-wide selloffs do not create false overlap warnings.
    """
    target_syms = list(dict.fromkeys(symbols or list(histories.keys())))
    close_map: dict[str, pd.Series] = {}
    for sym in target_syms:
        cs = _extract_close_series(histories.get(sym), lookback_days)
        if cs is not None:
            close_map[sym] = cs

    if len(close_map) < 2:
        return pd.DataFrame()

    # Include available benchmarks in the aligned calendar so holiday ffill(limit=1) and 2-day returns are consistent
    bench_candidates: set[str] = set()
    if benchmark_symbol:
        for sym in list(close_map):
            for b_cand in resolve_market_benchmark(
                sym,
                default_benchmark=benchmark_symbol,
                uk_benchmark=uk_benchmark_symbol or benchmark_symbol,
                eu_benchmark=eu_benchmark_symbol or benchmark_symbol,
            ):
                if b_cand in histories and b_cand not in close_map:
                    b_cs = _extract_close_series(histories.get(b_cand), lookback_days)
                    if b_cs is not None:
                        close_map[b_cand] = b_cs
                        bench_candidates.add(b_cand)

    price_df = pd.DataFrame(close_map).sort_index()
    # Fill isolated 1-day asynchronous holidays (e.g. UK Bank Holiday vs US Independence Day) without bridging long gaps
    price_ffill = price_df.ffill(limit=1)
    ret1_df = price_ffill.pct_change(fill_method=None).tail(lookback_days)
    ret2_df = price_ffill.pct_change(periods=2, fill_method=None).tail(lookback_days)
    ret5_df = price_ffill.pct_change(periods=5, fill_method=None).tail(lookback_days)

    valid_syms = [s for s in target_syms if s in ret1_df.columns and ret1_df[s].dropna().shape[0] >= 15]
    if len(valid_syms) < 2:
        return pd.DataFrame()

    if against_symbols is not None:
        against_set = set(against_symbols)
        row_syms = [s for s in valid_syms if s in against_set]
        if not row_syms:
            return pd.DataFrame()
    else:
        row_syms = valid_syms
    col_syms = valid_syms

    raw_corr_1d = _compute_rect_corr(ret1_df, row_syms, col_syms, min_periods=15)
    raw_corr_2d = _compute_rect_corr(ret2_df, row_syms, col_syms, min_periods=15)
    raw_corr_5d = _compute_rect_corr(ret5_df, row_syms, col_syms, min_periods=15)

    # For cross-timezone pairs (UK/EU 16:30 close vs US 21:00 close), compensate for closing lag via 2d/5d rolling corr
    raw_corr = raw_corr_1d.copy()
    for a in row_syms:
        for b in col_syms:
            if a == b or not _is_cross_timezone_pair(a, b):
                continue
            best_v = raw_corr_1d.loc[a, b]
            for alt_corr in (raw_corr_2d, raw_corr_5d):
                if a in alt_corr.index and b in alt_corr.columns:
                    v_alt = alt_corr.loc[a, b]
                    if pd.notna(v_alt) and (pd.isna(best_v) or abs(float(v_alt)) > abs(float(best_v))):
                        best_v = v_alt
            raw_corr.loc[a, b] = best_v

    if benchmark_symbol is None:
        return raw_corr.round(2)

    resid1_map: dict[str, pd.Series] = {}
    resid2_map: dict[str, pd.Series] = {}
    is_index_clone: dict[str, bool] = {}
    clone_bench_map: dict[str, str] = {}

    for sym in valid_syms:
        ser1 = ret1_df[sym].dropna()
        ser2 = ret2_df[sym].dropna()
        pref_benches = resolve_market_benchmark(
            sym,
            default_benchmark=benchmark_symbol,
            uk_benchmark=uk_benchmark_symbol or benchmark_symbol,
            eu_benchmark=eu_benchmark_symbol or benchmark_symbol,
        )
        chosen_bench: str | None = None
        b_col: str | None = None
        for cand_b in pref_benches:
            if cand_b in ret1_df.columns and ret1_df[cand_b].dropna().shape[0] >= 15:
                chosen_bench = cand_b.upper()
                b_col = cand_b
                break

        if chosen_bench is None or b_col is None:
            resid1_map[sym] = ser1
            resid2_map[sym] = ser2
            is_index_clone[sym] = False
            continue

        if sym.upper() == chosen_bench:
            is_index_clone[sym] = True
            clone_bench_map[sym] = chosen_bench
            continue

        r1, clone1 = _ols_residual_and_clone_flag(ser1, ret1_df[b_col].dropna())
        r2, clone2 = _ols_residual_and_clone_flag(ser2, ret2_df[b_col].dropna())
        resid1_map[sym] = r1
        resid2_map[sym] = r2

        sym_fam = infer_etf_index_family(sym)
        bench_fam = infer_etf_index_family(chosen_bench)
        same_etf_family = bool(sym_fam and bench_fam and sym_fam == bench_fam)
        is_clone = bool(clone1 or (clone2 and _is_cross_timezone_pair(sym, chosen_bench)) or same_etf_family)
        is_index_clone[sym] = is_clone
        if is_clone:
            clone_bench_map[sym] = chosen_bench

    resid1_df = pd.DataFrame(resid1_map).dropna(how="all")
    resid2_df = pd.DataFrame(resid2_map).dropna(how="all")
    if resid1_df.empty:
        return raw_corr.round(2)

    resid_corr_1d = _compute_rect_corr(resid1_df, row_syms, col_syms, min_periods=15)
    resid_corr_2d = (
        _compute_rect_corr(resid2_df, row_syms, col_syms, min_periods=15)
        if not resid2_df.empty
        else pd.DataFrame()
    )

    combined = raw_corr.copy()
    for a in combined.index:
        for b in combined.columns:
            if a == b:
                continue
            if is_index_clone.get(a, False) and is_index_clone.get(b, False):
                if clone_bench_map.get(a) == clone_bench_map.get(b):
                    continue
            val = resid_corr_1d.loc[a, b] if (a in resid_corr_1d.index and b in resid_corr_1d.columns) else np.nan
            if _is_cross_timezone_pair(a, b) and a in resid_corr_2d.index and b in resid_corr_2d.columns:
                val2 = resid_corr_2d.loc[a, b]
                if pd.notna(val2) and (pd.isna(val) or abs(float(val2)) > abs(float(val))):
                    val = val2
            if pd.notna(val):
                combined.loc[a, b] = val
    return combined.round(2)


def max_correlation_against_portfolio(
    corr_matrix: pd.DataFrame,
    candidate_symbol: str,
    portfolio_symbols: list[str],
) -> tuple[str | None, float | None]:
    """Return (most_correlated_holding, corr_value) for overlap checking."""
    if corr_matrix.empty or candidate_symbol not in corr_matrix.columns:
        return None, None
    best_sym: str | None = None
    best_val: float = -1.0
    for held in portfolio_symbols:
        if held == candidate_symbol or held not in corr_matrix.index:
            continue
        val = corr_matrix.loc[held, candidate_symbol]
        if pd.notna(val) and float(val) > best_val:
            best_val = float(val)
            best_sym = held
    if best_sym is None:
        return None, None
    return best_sym, best_val


def format_technical_line(
    symbol: str,
    tech: TechnicalSnapshot | None,
    bundle: TickerMarketBundle | None = None,
) -> str:
    """Format a compact one-line quant + CYQ + multi-timeframe Fact Sheet for LLM prompt injection."""
    if tech is None:
        return f"• {symbol}: （暂无 Yahoo K 线技术面数据）"

    # Synchronize ex-dividend flags onto tech if bundle provides ex-dividend metadata
    if bundle is not None:
        b_ex_days = getattr(bundle, "days_to_ex_dividend", None)
        if b_ex_days is not None:
            ex_d, ex_win, ex_guard = _compute_ex_div_flags(b_ex_days)
            tech.days_to_ex_dividend = ex_d
            tech.is_ex_div_window = ex_win
            tech.is_ex_div_drop_guard = ex_guard

    curr_str = f" {bundle.currency}" if bundle and bundle.currency else ""
    parts: list[str] = [
        f"• {symbol} [{tech.trend_regime}] 现价 {tech.last_close:.2f}{curr_str}",
        f"5日 {tech.ret_5d_pct:+.1f}%" if tech.ret_5d_pct is not None else "",
        f"20日 {tech.ret_20d_pct:+.1f}%" if tech.ret_20d_pct is not None else "",
        f"距MA50 {tech.dist_ma50_pct:+.1f}%" if tech.dist_ma50_pct is not None else "",
        f"距MA200 {tech.dist_ma200_pct:+.1f}%" if tech.dist_ma200_pct is not None else "",
        f"RSI14 {tech.rsi14:.1f}" if tech.rsi14 is not None else "",
    ]

    if tech.weekly_bullish is not None:
        w_tag = "周多头" if tech.weekly_bullish else "周空头"
        w_rsi = f"(周RSI {tech.weekly_rsi14:.0f})" if tech.weekly_rsi14 is not None else ""
        parts.append(f"{w_tag}{w_rsi}")

    if tech.atr14 is not None:
        atr_p = f"({tech.atr_pct:.1f}%)" if tech.atr_pct is not None else ""
        parts.append(f"ATR14 {tech.atr14:.2f}{atr_p}")

    if tech.volume_ratio_20d is not None:
        vol_tag = "(放量)" if tech.volume_ratio_20d >= 1.5 else ("(缩量)" if tech.volume_ratio_20d <= 0.7 else "")
        parts.append(f"量比 {tech.volume_ratio_20d:.2f}x{vol_tag}")

    if tech.adx14 is not None:
        parts.append(f"ADX {tech.adx14:.0f}")

    if tech.bb_squeeze:
        bw_str = f"({tech.bb_width_pct:.1f}%)" if tech.bb_width_pct is not None else ""
        parts.append(f"BB挤压{bw_str}")

    if tech.cyq_benefit_part_pct is not None and tech.cyq_avg_cost is not None:
        reg_str = f"[{tech.cyq_regime}]" if tech.cyq_regime else ""
        conc_str = f"/集中{tech.cyq_concentration_90_pct:.1f}%" if tech.cyq_concentration_90_pct is not None else ""
        parts.append(
            f"筹码{reg_str} 获利{tech.cyq_benefit_part_pct:.0f}%/均成本{tech.cyq_avg_cost:.2f}{conc_str}"
        )

    if tech.drawdown_52w_pct is not None:
        parts.append(f"距52周高点 {tech.drawdown_52w_pct:+.1f}%")
    if tech.volatility_60d_pct is not None:
        vol_tail = (
            f"(2年{tech.vol_60d_percentile:.0f}分位)"
            if tech.vol_60d_percentile is not None
            else ""
        )
        parts.append(f"年化波动 {tech.volatility_60d_pct:.0f}%{vol_tail}")

    ex_div_rendered = False
    if bundle:
        if bundle.target_mean_price and tech.last_close > 0:
            upside_pct = ((bundle.target_mean_price / tech.last_close) - 1.0) * 100.0
            parts.append(f"目标均价 {bundle.target_mean_price:.2f}({upside_pct:+.1f}%)")
        if bundle.forward_pe:
            parts.append(f"远期PE {bundle.forward_pe:.1f}x")
        elif bundle.trailing_pe:
            parts.append(f"PE {bundle.trailing_pe:.1f}x")
        if bundle.dividend_yield_pct is not None and bundle.dividend_yield_pct >= 0.1:
            parts.append(f"股息率 {bundle.dividend_yield_pct:.2f}%")
        if bundle.earnings_date:
            days_s = f"({bundle.days_to_earnings}天后)" if bundle.days_to_earnings is not None else ""
            eps_val = getattr(bundle, "eps_estimate", None)
            eps_s = f" EPS预期 {eps_val:.2f}" if eps_val is not None else ""
            parts.append(f"财报 {bundle.earnings_date}{days_s}{eps_s}")
        ex_div_date = getattr(bundle, "ex_dividend_date", None)
        ex_days = getattr(bundle, "days_to_ex_dividend", None)
        if ex_days is None:
            ex_days = tech.days_to_ex_dividend
        if ex_div_date or ex_days is not None:
            date_prefix = f" {ex_div_date}" if ex_div_date else ""
            if ex_days is not None and -3 <= ex_days <= 0:
                day_lbl = "今日" if ex_days == 0 else f"{abs(ex_days)}天前"
                parts.append(f"除息{date_prefix}({day_lbl}⚠️刚除息自然扣息勿误判破位)")
            elif ex_days is not None and 1 <= ex_days <= 5:
                parts.append(f"除息{date_prefix}({ex_days}天后⚠️临近除息)")
            else:
                ex_s = f"({ex_days}天后)" if ex_days is not None else ""
                parts.append(f"除息{date_prefix}{ex_s}")
            ex_div_rendered = True
        if bundle.quote_type.upper() == "ETF":
            parts.append("[ETF]")

    if not ex_div_rendered and tech.days_to_ex_dividend is not None:
        ex_days = tech.days_to_ex_dividend
        if -3 <= ex_days <= 0:
            day_lbl = "今日" if ex_days == 0 else f"{abs(ex_days)}天前"
            parts.append(f"除息({day_lbl}⚠️刚除息自然扣息勿误判破位)")
        elif 1 <= ex_days <= 5:
            parts.append(f"除息({ex_days}天后⚠️临近除息)")

    return " | ".join(p for p in parts if p)


_ISSUER_ALIASES: dict[str, str] = {
    "FB": "META",
    "GOOGL": "GOOG",
    "BRK.A": "BRK-B",
    "BRK-A": "BRK-B",
}

# US mega-cap tech / semiconductor factor bucket used for look-through exposure hints.
_TECH_FACTOR_TICKERS: frozenset[str] = frozenset(
    {
        "MSFT",
        "AVGO",
        "NVDA",
        "AAPL",
        "GOOG",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "NFLX",
        "ORCL",
        "CRM",
        "ADBE",
        "INTC",
        "QCOM",
        "TXN",
        "MU",
        "PLTR",
        "ARM",
        "SMCI",
        "TSM",
        "ASML",
        # v5.5.1: high-beta US software / cyber-security / semi-cap leaders that were
        # silently missing from the factor bucket (PANW/CRWD/GOOGL look-through gaps).
        "PANW",
        "CRWD",
        "NOW",
        "SNOW",
        "DDOG",
        "NET",
        "ANET",
        "MRVL",
        "KLAC",
        "LRCX",
        "AMAT",
        "CDNS",
        "SNPS",
    }
)

# Static APPROXIMATE implied tech weight inside broad UCITS ETFs.
# These are published order-of-magnitude anchors, NOT a live look-through file:
# they only feed a coarse "how much tech do I really own" hint and must never
# be used for a hard risk block.
_ETF_TECH_PENETRATION_PCT: dict[str, float] = {
    # Global all-world / developed
    "VWRP": 25.0,
    "VWRP.L": 25.0,
    "VWRL": 25.0,
    "VWRL.L": 25.0,
    "SWDA": 25.0,
    "SWDA.L": 25.0,
    "IWDA": 24.0,
    "IWDA.L": 24.0,
    "SSAC": 24.0,
    "SSAC.L": 24.0,
    "FWRG": 24.0,
    "FWRG.L": 24.0,
    # S&P 500
    "VUAG": 32.0,
    "VUAG.L": 32.0,
    "CSP1": 32.0,
    "CSP1.L": 32.0,
    "VUSA": 32.0,
    "VUSA.L": 32.0,
    "IUSA": 32.0,
    "IUSA.L": 32.0,
    "SPY": 32.0,
    "VOO": 32.0,
    "IVV": 32.0,
    # NASDAQ 100
    "CNX1": 50.0,
    "CNX1.L": 50.0,
    "EQGB": 50.0,
    "EQGB.L": 50.0,
    "EQQQ": 50.0,
    "EQQQ.L": 50.0,
    "QQQ": 50.0,
    "QQQM": 50.0,
    # FTSE 100 (tech weight is negligible)
    "ISF": 1.0,
    "ISF.L": 1.0,
    "VUKE": 1.0,
    "VUKE.L": 1.0,
    "CUKX": 1.0,
    "CUKX.L": 1.0,
    "VUKG": 1.0,
    "VUKG.L": 1.0,
    # Europe broad (developed / ex-UK)
    "VERG": 8.0,
    "VERG.L": 8.0,
    "VERX": 8.0,
    "VERX.L": 8.0,
    "VEUR": 8.0,
    "VEUR.L": 8.0,
    "VEUA": 8.0,
    "VEUA.L": 8.0,
    "IMEU": 8.0,
    "IMEU.L": 8.0,
    "SMEA": 8.0,
    "SMEA.L": 8.0,
}

# Static APPROXIMATE regional look-through for broad UCITS ETFs (percent of ETF NAV).
# Kept deliberately coarse: it only produces a context line and the deterministic core
# policy gap maths never treats it as a hard risk threshold.
_ETF_REGIONAL_LOOKTHROUGH: dict[str, dict[str, float]] = {
    # Global all-world / developed
    "VWRP": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "VWRL": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "SSAC": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "FWRG": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    # Developed world (MSCI World)
    "SWDA": {"us": 71.0, "uk": 3.8, "eu_ex_uk": 13.5, "row": 11.7},
    "IWDA": {"us": 71.0, "uk": 3.8, "eu_ex_uk": 13.5, "row": 11.7},
    # S&P 500
    "VUAG": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUSA": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "CSP1": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "IUSA": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "SPXP": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "GSPX": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "SPY": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VOO": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "IVV": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    # Europe ex-UK (strictly zero UK overlap)
    "VERG": {"us": 0.0, "uk": 0.0, "eu_ex_uk": 100.0, "row": 0.0},
    "VERX": {"us": 0.0, "uk": 0.0, "eu_ex_uk": 100.0, "row": 0.0},
    # Europe including UK (~22% UK exposure)
    "VEUR": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "VEUA": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "IMEU": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "SMEA": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    # FTSE 100
    "ISF": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUKE": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "CUKX": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUKG": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
}

# Bond / gold ETFs are excluded from the equity-only regional denominator.
_NON_EQUITY_ETF_BASES: frozenset[str] = frozenset({"IGLT", "VGOV", "IBTA", "SGLN"})

# v5.5.1: distribution policy whitelists (LSE share classes, with/without ``.L``).
_KNOWN_DISTRIBUTING_ETFS: frozenset[str] = frozenset(
    {
        "VWRL", "VUSA", "IUSA", "ISF", "VUKE", "VERX", "VEUR", "IMEU",
        "VHYL", "IUKD", "IGLT", "VGOV", "VFEM", "MIDD",
    }
)
_KNOWN_ACCUMULATING_ETFS: frozenset[str] = frozenset(
    {
        "VWRP", "SWDA", "IWDA", "FWRG", "SSAC", "VUAG", "CSP1", "SPXP", "GSPX",
        "VERG", "VEUA", "SMEA", "CUKX", "VUKG", "CNX1", "EQGB", "IBTA", "IITU",
        "SMGB", "HEAL", "INRG", "EMIM", "WSML", "IWQU", "IWFQ", "IWMO", "VJPN", "SGLN",
    }
)
_LEVERAGED_ETP_TOKENS: tuple[str, ...] = (
    "2x", "3x", "-1x", "-2x", "-3x", "leveraged", "daily short",
    "inverse", "ultrashort", "covered call", "incomeshares",
)

# Underlying index family mapping for duplicate/clone ETF exposure aggregation
_ETF_INDEX_FAMILY_MAP: dict[str, str] = {
    # S&P 500 trackers
    "VUAG": "S&P 500 (美股标普500)",
    "VUAG.L": "S&P 500 (美股标普500)",
    "CSP1": "S&P 500 (美股标普500)",
    "CSP1.L": "S&P 500 (美股标普500)",
    "VUSA": "S&P 500 (美股标普500)",
    "VUSA.L": "S&P 500 (美股标普500)",
    "IUSA": "S&P 500 (美股标普500)",
    "IUSA.L": "S&P 500 (美股标普500)",
    "SPXP": "S&P 500 (美股标普500)",
    "SPXP.L": "S&P 500 (美股标普500)",
    "GSPX": "S&P 500 (美股标普500)",
    "GSPX.L": "S&P 500 (美股标普500)",
    "SPY": "S&P 500 (美股标普500)",
    "VOO": "S&P 500 (美股标普500)",
    "IVV": "S&P 500 (美股标普500)",
    # NASDAQ 100 trackers
    "CNX1": "NASDAQ 100 (美股纳指100)",
    "CNX1.L": "NASDAQ 100 (美股纳指100)",
    "EQGB": "NASDAQ 100 (美股纳指100)",
    "EQGB.L": "NASDAQ 100 (美股纳指100)",
    "EQQQ": "NASDAQ 100 (美股纳指100)",
    "EQQQ.L": "NASDAQ 100 (美股纳指100)",
    "XNAQ": "NASDAQ 100 (美股纳指100)",
    "XNAQ.L": "NASDAQ 100 (美股纳指100)",
    "QQQ": "NASDAQ 100 (美股纳指100)",
    "QQQM": "NASDAQ 100 (美股纳指100)",
    # All-World / Developed World broad trackers
    "VWRP": "Global All-World/Developed (全球宽基)",
    "VWRP.L": "Global All-World/Developed (全球宽基)",
    "VWRL": "Global All-World/Developed (全球宽基)",
    "VWRL.L": "Global All-World/Developed (全球宽基)",
    "SWDA": "Global All-World/Developed (全球宽基)",
    "SWDA.L": "Global All-World/Developed (全球宽基)",
    "IWDA": "Global All-World/Developed (全球宽基)",
    "IWDA.L": "Global All-World/Developed (全球宽基)",
    "FWRG": "Global All-World/Developed (全球宽基)",
    "FWRG.L": "Global All-World/Developed (全球宽基)",
    "SSAC": "Global All-World/Developed (全球宽基)",
    "SSAC.L": "Global All-World/Developed (全球宽基)",
    # FTSE 100 / UK Equity Dividend trackers
    "ISF": "FTSE 100 (英国富时100)",
    "ISF.L": "FTSE 100 (英国富时100)",
    "VUKE": "FTSE 100 (英国富时100)",
    "VUKE.L": "FTSE 100 (英国富时100)",
    "CUKX": "FTSE 100 (英国富时100)",
    "CUKX.L": "FTSE 100 (英国富时100)",
    "VUKG": "FTSE 100 (英国富时100)",
    "VUKG.L": "FTSE 100 (英国富时100)",
    # Europe broad trackers
    "VERG": "Europe Equity (欧洲宽基)",
    "VERG.L": "Europe Equity (欧洲宽基)",
    "VERX": "Europe Equity (欧洲宽基)",
    "VERX.L": "Europe Equity (欧洲宽基)",
    "VEUR": "Europe Equity (欧洲宽基)",
    "VEUR.L": "Europe Equity (欧洲宽基)",
    "VEUA": "Europe Equity (欧洲宽基)",
    "VEUA.L": "Europe Equity (欧洲宽基)",
    "IMEU": "Europe Equity (欧洲宽基)",
    "IMEU.L": "Europe Equity (欧洲宽基)",
    "SMEA": "Europe Equity (欧洲宽基)",
    "SMEA.L": "Europe Equity (欧洲宽基)",
    # UK Gilts
    "IGLT": "UK Gilts (英国国债)",
    "IGLT.L": "UK Gilts (英国国债)",
    "VGOV": "UK Gilts (英国国债)",
    "VGOV.L": "UK Gilts (英国国债)",
}


def normalize_issuer_root(ticker: str) -> str:
    """Normalize dual-class / shared-issuer tickers (e.g. GOOGL -> GOOG) for combined exposure checks."""
    clean = (ticker or "").strip().upper()
    return _ISSUER_ALIASES.get(clean, clean)


def infer_etf_index_family(
    symbol: str,
    bundle: TickerMarketBundle | None = None,
    name_hint: str = "",
) -> str | None:
    """Identify the underlying index family of an ETF to aggregate clone ETF risk exposure."""
    sym_clean = (symbol or "").strip().upper()
    combined_name = " ".join(
        filter(
            None,
            [
                name_hint,
                getattr(bundle, "short_name", "") if bundle else "",
                getattr(bundle, "long_name", "") if bundle else "",
            ],
        )
    ).lower()

    # v5.5.1: leveraged / inverse / covered-call ETPs must never inherit the broad-base
    # core privilege (e.g. "FTSE 100 3x Daily Leveraged" is NOT a core wide-base ETF).
    if any(token in combined_name for token in _LEVERAGED_ETP_TOKENS):
        return None

    if sym_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[sym_clean]
    base_clean = sym_clean.split(".")[0]
    if base_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[base_clean]

    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    if not is_etf and not any(k in combined_name for k in ("ucits", "etf", "vanguard", "ishares", "invesco", "spdr")):
        return None

    if "s&p 500" in combined_name or "s&p500" in combined_name or "sp500" in combined_name:
        return "S&P 500 (美股标普500)"
    if "nasdaq-100" in combined_name or "nasdaq 100" in combined_name:
        return "NASDAQ 100 (美股纳指100)"
    if "all-world" in combined_name or "msci world" in combined_name or "ftse all-world" in combined_name:
        return "Global All-World/Developed (全球宽基)"
    if "ftse 100" in combined_name:
        return "FTSE 100 (英国富时100)"
    if (
        "developed europe" in combined_name
        or "msci europe" in combined_name
        or "stoxx europe 600" in combined_name
        or "europe ex uk" in combined_name
    ):
        return "Europe Equity (欧洲宽基)"
    if "gilt" in combined_name:
        return "UK Gilts (英国国债)"
    return None


def classify_etf_distribution_policy(
    symbol: str,
    bundle: TickerMarketBundle | None = None,
) -> Literal["ACC", "DIST", "UNKNOWN"]:
    """Classify an ETF share class as accumulating / distributing / unknown.

    Taxable Invest accounts must only buy distributing classes: accumulating offshore
    funds trigger Excess Reportable Income (ERI) reporting friction in the UK.
    """
    sym_clean = (symbol or "").strip().upper()
    base_clean = sym_clean.split(".")[0]
    if base_clean in _KNOWN_DISTRIBUTING_ETFS or sym_clean in _KNOWN_DISTRIBUTING_ETFS:
        return "DIST"
    if base_clean in _KNOWN_ACCUMULATING_ETFS or sym_clean in _KNOWN_ACCUMULATING_ETFS:
        return "ACC"
    name = " ".join(
        filter(
            None,
            [
                str(getattr(bundle, "short_name", "") or ""),
                str(getattr(bundle, "long_name", "") or ""),
            ],
        )
    ).lower()
    if "(acc)" in name or "accumulating" in name or " acc " in f" {name} ":
        return "ACC"
    if "(dist)" in name or "distributing" in name or " dist " in f" {name} ":
        return "DIST"
    return "UNKNOWN"


def _regional_root(symbol: str) -> str:
    return (symbol or "").strip().upper().split(".")[0]


def _stock_region(symbol: str) -> str | None:
    """Map a single-name listing suffix to a coarse region (ETF-free helper)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    if sym.endswith(".L"):
        return "uk"
    if sym.endswith((".DE", ".PA", ".AS", ".MC", ".MI", ".SW")):
        return "eu_ex_uk"
    if "." not in sym:
        return "us"
    return None


def estimate_portfolio_regional_lookthrough(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None = None,
    *,
    include_pending_core_etfs: bool = True,
) -> dict[str, Any]:
    """Coarse US / UK / Europe-ex-UK split of the equity book (context only)."""
    bmap = bundles or {}
    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), 1.0)
    isa_val = max(float(portfolio.isa.total_account_value or 0.0), 1.0)

    region_values = {"us": 0.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0}
    isa_region_values = {"us": 0.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0}
    unmapped_value = 0.0
    excluded_value = 0.0

    def _add(acct: str, sym: str, value: float) -> None:
        nonlocal unmapped_value, excluded_value
        base = _regional_root(sym)
        if base in _NON_EQUITY_ETF_BASES:
            excluded_value += value
            return
        b = bmap.get(sym) or bmap.get(sym.strip().upper())
        is_etf = bool((b is not None and str(b.quote_type).upper() == "ETF"))
        row = _ETF_REGIONAL_LOOKTHROUGH.get(base) if (is_etf or base in _ETF_REGIONAL_LOOKTHROUGH) else None
        if row is not None:
            for key in region_values:
                share = float(row.get(key, 0.0)) / 100.0 * value
                region_values[key] += share
                if acct == "ISA":
                    isa_region_values[key] += share
            return
        region = _stock_region(sym)
        if region is None:
            unmapped_value += value
            return
        region_values[region] += value
        if acct == "ISA":
            isa_region_values[region] += value

    for pos in portfolio.all_positions():
        if pos.current_value > 0:
            _add(str(pos.account or "").strip().upper(), pos.yahoo_symbol or pos.ticker, float(pos.current_value))
    if include_pending_core_etfs:
        for order in portfolio.isa.pending_orders:
            if str(order.side or "").strip().upper() != "BUY":
                continue
            base = _regional_root(order.yahoo_symbol or order.ticker)
            if base not in _ETF_REGIONAL_LOOKTHROUGH:
                continue
            ccy = str(order.currency or "").strip()
            raw_value = float(order.value or 0.0)
            if ccy in {"GBp", "GBx", "GBX"}:
                value = raw_value / 100.0
            elif ccy.upper() in {"GBP", ""}:
                value = raw_value
            else:
                # Non-GBP pending value cannot be converted without an FX basis; skip
                # rather than silently mis-scaling the look-through percentages.
                continue
            if value > 0:
                _add("ISA", order.yahoo_symbol or order.ticker, value)

    equity_total = sum(region_values.values())

    def _pct_of(value: float, denom: float) -> float:
        return round(value / denom * 100.0, 2) if denom > 0 else 0.0

    return {
        "us_value": round(region_values["us"], 2),
        "uk_value": round(region_values["uk"], 2),
        "eu_ex_uk_value": round(region_values["eu_ex_uk"], 2),
        "row_value": round(region_values["row"], 2),
        "equity_value_gbp": round(equity_total, 2),
        "excluded_bond_gold_value": round(excluded_value, 2),
        "unmapped_value": round(unmapped_value, 2),
        "us_pct_nav": _pct_of(region_values["us"], total_all),
        "uk_pct_nav": _pct_of(region_values["uk"], total_all),
        "eu_ex_uk_pct_nav": _pct_of(region_values["eu_ex_uk"], total_all),
        "isa_us_pct": _pct_of(isa_region_values["us"], isa_val),
        "isa_uk_pct": _pct_of(isa_region_values["uk"], isa_val),
        "isa_eu_ex_uk_pct": _pct_of(isa_region_values["eu_ex_uk"], isa_val),
        "equity_us_share_pct": _pct_of(region_values["us"], equity_total),
        "equity_uk_share_pct": _pct_of(region_values["uk"], equity_total),
        "equity_eu_ex_uk_share_pct": _pct_of(region_values["eu_ex_uk"], equity_total),
        "include_pending_core_etfs": bool(include_pending_core_etfs),
    }


def estimate_penetration_tech_exposure(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> dict[str, Any]:
    """Estimate combined tech-factor exposure: direct tech holdings + static ETF look-through.

    Values are approximations for context only — never used as a hard blocking threshold.
    """
    bmap = bundles or {}
    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), 1.0)

    direct_by_ticker: dict[str, float] = {}
    direct_val = 0.0
    etf_implied_val = 0.0
    unmapped_etf_val = 0.0
    etf_rows: list[dict[str, Any]] = []

    for p in portfolio.all_positions():
        if p.current_value <= 0:
            continue
        root = normalize_issuer_root((p.ticker or "").strip().upper())
        if root in _TECH_FACTOR_TICKERS:
            direct_val += p.current_value
            direct_by_ticker[root] = direct_by_ticker.get(root, 0.0) + p.current_value
            continue
        sym = (p.yahoo_symbol or "").strip().upper()
        pct = _ETF_TECH_PENETRATION_PCT.get(sym) or _ETF_TECH_PENETRATION_PCT.get(sym.split(".")[0])
        if pct is None:
            b = bmap.get(p.yahoo_symbol)
            if b is not None and b.quote_type.upper() == "ETF":
                unmapped_etf_val += p.current_value
            continue
        implied = p.current_value * pct / 100.0
        etf_implied_val += implied
        etf_rows.append(
            {
                "symbol": p.yahoo_symbol,
                "account": p.account,
                "weight_pct": pct,
                "implied_value": round(implied, 2),
            }
        )

    combined = direct_val + etf_implied_val
    return {
        "direct_value": round(direct_val, 2),
        "direct_positions": {k: round(v, 2) for k, v in direct_by_ticker.items()},
        "etf_implied_value": round(etf_implied_val, 2),
        "etf_breakdown": etf_rows,
        "unmapped_etf_value": round(unmapped_etf_val, 2),
        "combined_value": round(combined, 2),
        "combined_pct": round((combined / total_all) * 100.0, 1),
        "total_value": round(total_all, 2),
    }


def compute_portfolio_cvar_and_risk_attribution(
    portfolio: PortfolioData,
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Compute PSD-shrunk portfolio annualized volatility, component risk contributions, and 95% historical 1d CVaR."""
    held_values: dict[str, float] = {}
    for pos in portfolio.all_positions():
        sym = (pos.yahoo_symbol or "").strip()
        if sym and pos.current_value > 0.0:
            held_values[sym] = held_values.get(sym, 0.0) + float(pos.current_value)

    if not held_values or not histories:
        return {}

    close_series: dict[str, pd.Series] = {}
    vol_1d: dict[str, float] = {}
    extract_days = max(int(lookback_days), 60) + 10
    for sym in held_values:
        cs = _extract_close_series(histories.get(sym), extract_days)
        if cs is None:
            continue
        # Clean 1-day log-return volatility WITHOUT cross-holiday ffill distortion
        r1 = np.log(cs / cs.shift(1)).tail(lookback_days).dropna()
        if len(r1) < 15:
            continue
        std_1d = float(np.std(r1.to_numpy(dtype=np.float64), ddof=1))
        if std_1d > 0.0:
            close_series[sym] = cs
            vol_1d[sym] = std_1d

    valid_syms = [s for s in held_values if s in close_series]
    if not valid_syms:
        return {}

    raw_w = np.array([held_values[s] for s in valid_syms], dtype=np.float64)
    w_sum = float(np.sum(raw_w))
    if w_sum <= 0.0:
        return {}
    w = raw_w / w_sum

    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), w_sum)

    px_df = pd.DataFrame({s: close_series[s] for s in valid_syms}).sort_index().ffill(limit=1)

    if len(valid_syms) == 1:
        sym0 = valid_syms[0]
        port_vol_pct = float(vol_1d[sym0] * np.sqrt(252.0) * 100.0)
        contributions = [
            {
                "symbol": sym0,
                "risk_pct": 100.0,
                "weight_pct": round((held_values[sym0] / total_all) * 100.0, 1),
                "invested_weight_pct": 100.0,
            }
        ]
    else:
        log3 = np.log(px_df / px_df.shift(3)).tail(lookback_days).dropna()
        n_dim = len(valid_syms)
        if len(log3) >= 20:
            arr3 = log3.to_numpy(dtype=np.float64)
            stds3 = np.std(arr3, axis=0, ddof=1)
            if np.any(stds3 <= 0.0):
                corr3 = np.eye(n_dim, dtype=np.float64)
            else:
                corr3 = np.corrcoef(arr3, rowvar=False)
                if corr3.ndim == 0:
                    corr3 = np.eye(n_dim, dtype=np.float64)
                else:
                    corr3 = np.nan_to_num(corr3, nan=0.0)
                    np.fill_diagonal(corr3, 1.0)
        else:
            corr3 = np.eye(n_dim, dtype=np.float64)

        shrinkage = 0.20
        corr_shrunk = (1.0 - shrinkage) * corr3 + shrinkage * np.eye(n_dim, dtype=np.float64)
        d_vec = np.array([vol_1d[s] for s in valid_syms], dtype=np.float64)
        cov_1d = np.outer(d_vec, d_vec) * corr_shrunk

        port_var = float(w @ cov_1d @ w)
        if port_var <= 0.0:
            return {}
        port_vol_pct = float(np.sqrt(port_var * 252.0) * 100.0)
        mrc_var = cov_1d @ w
        rc_pct_arr = (w * mrc_var) / port_var * 100.0

        contributions = []
        for idx, sym in enumerate(valid_syms):
            contributions.append(
                {
                    "symbol": sym,
                    "risk_pct": round(float(rc_pct_arr[idx]), 1),
                    "weight_pct": round((held_values[sym] / total_all) * 100.0, 1),
                    "invested_weight_pct": round(float(w[idx] * 100.0), 1),
                }
            )
        contributions.sort(key=lambda item: item["risk_pct"], reverse=True)

    # Historical 95% 1-day CVaR using aligned daily simple returns (excluding common no-move days)
    ret1_raw = px_df.pct_change(fill_method=None).tail(lookback_days).dropna(how="all")
    ret1_clean = ret1_raw.fillna(0.0)
    active_mask = (ret1_clean.abs().sum(axis=1) > 1e-12)
    ret1_clean = ret1_clean.loc[active_mask]

    cvar_95_pct: float | None = None
    cvar_95_gbp: float | None = None
    if len(ret1_clean) >= 20:
        port_rets = ret1_clean[valid_syms].to_numpy(dtype=np.float64) @ w
        q05 = float(np.quantile(port_rets, 0.05))
        tail = port_rets[port_rets <= q05]
        if len(tail) == 0:
            k = max(1, int(np.ceil(len(port_rets) * 0.05)))
            tail = np.sort(port_rets)[:k]
        cvar_val = float(np.mean(tail))
        cvar_95_pct = round(cvar_val * 100.0, 2)
        cvar_95_gbp = round(max(0.0, -cvar_val) * w_sum, 2)

    return {
        "portfolio_vol_pct": round(port_vol_pct, 2),
        "cvar_95_pct": cvar_95_pct,
        "cvar_95_gbp": cvar_95_gbp,
        "modeled_value_gbp": round(w_sum, 2),
        "risk_contributions": contributions,
        "top_risk_sources": contributions[:3],
    }


def compute_exposure_summary(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Deterministically compute cross-account duplicate issuers, clone ETF underlying index overlap, and sector weights."""
    by_issuer: dict[str, dict[str, float]] = {}
    by_etf_family: dict[str, list[tuple[str, str, float]]] = {}
    sector_vals: dict[str, float] = {}
    invest_sector_vals: dict[str, float] = {}

    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(raw_total, 1.0)
    invest_total = max(portfolio.invest.total_account_value, 1.0)

    for p in portfolio.all_positions():
        root = normalize_issuer_root(p.ticker)
        entry = by_issuer.setdefault(root, {"ISA": 0.0, "Invest": 0.0})
        entry[p.account] = entry.get(p.account, 0.0) + p.current_value

        b = bundles.get(p.yahoo_symbol)
        etf_family = infer_etf_index_family(p.yahoo_symbol, b, name_hint=p.name)
        is_etf = bool((b and b.quote_type.upper() == "ETF") or etf_family is not None)

        if etf_family:
            by_etf_family.setdefault(etf_family, []).append((p.account, p.ticker, p.current_value))

        if is_etf:
            sector = "ETF宽基/固收"
        elif b and b.sector:
            sector = b.sector
        else:
            sector = "Other/未分类"

        sector_vals[sector] = sector_vals.get(sector, 0.0) + p.current_value
        if p.account == "Invest":
            invest_sector_vals[sector] = invest_sector_vals.get(sector, 0.0) + p.current_value

    overlap_lines = []
    for root, accts in by_issuer.items():
        if accts.get("ISA", 0.0) > 0 and accts.get("Invest", 0.0) > 0:
            comb = accts["ISA"] + accts["Invest"]
            pct = (comb / total_all) * 100.0
            overlap_lines.append(
                f"• 跨账户重叠 {root}: ISA £{accts['ISA']:,.0f} + Invest £{accts['Invest']:,.0f} = 合计 £{comb:,.0f} (占总资产 {pct:.1f}%)"
            )

    clone_etf_lines = []
    for family, items in by_etf_family.items():
        distinct_tickers = {t for _, t, _ in items}
        if len(items) >= 2 and (len(distinct_tickers) >= 2 or len({a for a, _, _ in items}) >= 2):
            comb = sum(val for _, _, val in items)
            pct = (comb / total_all) * 100.0
            legs = " + ".join(f"{acct} {tkr} £{val:,.0f}" for acct, tkr, val in items)
            clone_etf_lines.append(
                f"• 克隆ETF底层暴露归并 [{family}]: {legs} = 合计 £{comb:,.0f} (占总资产 {pct:.1f}%)"
            )

    top_sectors = sorted(sector_vals.items(), key=lambda x: x[1], reverse=True)[:5]
    sector_str = " | ".join(
        f"{sec} {(val / total_all) * 100.0:.1f}%" for sec, val in top_sectors
    )
    top_inv_sectors = sorted(invest_sector_vals.items(), key=lambda x: x[1], reverse=True)[:3]
    inv_sector_str = " | ".join(
        f"{sec} {(val / invest_total) * 100.0:.1f}%" for sec, val in top_inv_sectors
    )

    lines = [
        f"• 总资产行业分布(前5): {sector_str or 'N/A'}",
        f"• Invest短线账户行业集中度(前3): {inv_sector_str or 'N/A'}",
    ]
    if overlap_lines:
        lines.extend(overlap_lines)
    else:
        lines.append("• 跨账户同名重叠: 无")
    if clone_etf_lines:
        lines.extend(clone_etf_lines)
    penetration = estimate_penetration_tech_exposure(portfolio, bundles)
    if penetration["combined_value"] > 0:
        lines.append(
            f"• 穿透后真实科技暴露(估算): ETF 隐含 £{penetration['etf_implied_value']:,.0f} + "
            f"直接科技个股 £{penetration['direct_value']:,.0f} = £{penetration['combined_value']:,.0f} "
            f"(占总资产 {penetration['combined_pct']:.1f}%；静态权重近似，非实时持仓穿透)"
        )
    if penetration.get("unmapped_etf_value", 0.0) > 0:
        lines.append(
            f"• 另有 £{penetration['unmapped_etf_value']:,.0f} ETF 未映射科技权重（未静默计入）"
        )
    regional = estimate_portfolio_regional_lookthrough(portfolio, bundles)
    if regional["equity_value_gbp"] > 0:
        lines.append(
            f"• 权益区域穿透(估算): 美股 {regional['equity_us_share_pct']:.1f}% | "
            f"英国 {regional['equity_uk_share_pct']:.1f}% | 欧陆(除英) {regional['equity_eu_ex_uk_share_pct']:.1f}%"
        )
    if histories:
        risk_info = compute_portfolio_cvar_and_risk_attribution(portfolio, histories)
        top_sources = risk_info.get("top_risk_sources") or []
        if top_sources:
            top_str = " | ".join(
                f"{item['symbol']} 占总风险 {item['risk_pct']:.0f}% (市值占 {item['weight_pct']:.0f}%)"
                for item in top_sources
            )
            lines.append(f"• 组合真实风险归因(前3波动源): {top_str}")
        cvar_pct = risk_info.get("cvar_95_pct")
        cvar_gbp = risk_info.get("cvar_95_gbp")
        if cvar_pct is not None and cvar_gbp is not None:
            lines.append(f"• 尾部压力测试(95%单日CVaR): 极端日均潜在回撤 {cvar_pct:+.2f}% (约 £{cvar_gbp:,.0f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IMP-17: Live account performance (NAV -> TWR), trade attribution & signal QC
# ---------------------------------------------------------------------------

_EXTERNAL_FLOW_SIGNS: dict[str, float] = {
    "DEPOSIT": 1.0,
    "WITHDRAW": -1.0,
    "WITHDRAWAL": -1.0,
}


def _event_uk_date(item: Any) -> str:
    """Best-effort UK calendar date (YYYY-MM-DD) from a T212 cash-event-like model."""
    for attr in ("uk_time_str", "date_time", "paid_on"):
        raw = str(getattr(item, attr, "") or "").strip()
        if len(raw) >= 10:
            return raw[:10]
    return ""


def build_nav_row(
    portfolio: PortfolioData,
    benchmark_close: float | None = None,
    uk_date: str | None = None,
) -> dict[str, Any]:
    """Build one lightweight daily NAV row (GBP-only scalars) for ``nav_history``.

    Only genuine external cash movements (DEPOSIT / WITHDRAW) count as flows.
    Cash interest, lending interest and platform fees are internal investment
    P&L / drag and must stay inside the TWR numerator; dividends likewise already
    sit inside NAV, so treating any of them as inflow would understate returns.
    """
    day = (uk_date or str(getattr(portfolio, "fetched_at_uk", "") or "").strip()[:10]).strip()[:10]
    total = float(getattr(portfolio, "total_combined_value", 0.0) or 0.0)
    cash = float(getattr(portfolio, "total_combined_cash", 0.0) or 0.0)

    net_flow = 0.0
    for acc in (portfolio.isa, portfolio.invest):
        for tx in getattr(acc, "recent_transactions", []) or []:
            if _event_uk_date(tx) != day:
                continue
            ccy = str(getattr(tx, "currency", "GBP") or "GBP").strip().upper()
            amount = float(getattr(tx, "amount", 0.0) or 0.0)
            if ccy == "USD":
                fx = float(getattr(portfolio, "gbp_usd_rate", None) or 0.0)
                if fx <= 0:
                    continue  # cannot convert safely -> never fabricate a GBP flow
                amount = amount / fx
            elif ccy not in ("GBP", ""):
                continue  # keep the NAV series strictly GBP-denominated
            sign = _EXTERNAL_FLOW_SIGNS.get(str(getattr(tx, "tx_type", "") or "").strip().upper())
            if sign is None:
                continue
            net_flow += sign * amount

    return {
        "uk_date": day,
        "total_value_gbp": round(total, 2),
        "cash_gbp": round(cash, 2),
        "invested_gbp": round(max(total - cash, 0.0), 2),
        "net_flow_gbp": round(net_flow, 2),
        "isa_value_gbp": round(float(getattr(portfolio.isa, "total_account_value", 0.0) or 0.0), 2),
        "invest_value_gbp": round(float(getattr(portfolio.invest, "total_account_value", 0.0) or 0.0), 2),
        "benchmark_close": round(float(benchmark_close), 6) if benchmark_close else None,
        "cash_rate_pct": float(getattr(portfolio, "cash_interest_rate_pct", 0.0) or 0.0),
    }


def compute_portfolio_performance(
    nav_rows: list[dict[str, Any]],
    benchmark_hist: pd.DataFrame | None = None,
    risk_free_annual_pct: float | None = None,
    min_periods: int = 15,
) -> dict[str, Any]:
    """Compute TWR-based account performance from persisted daily NAV rows (all GBP).

    TWR per period: ``r_t = (NAV_t - CF_t) / NAV_{t-1} - 1`` where CF_t is that day's
    *external* net flow. Cash drag and benchmark alpha/beta/IR are reported alongside.
    Callers must label the window: sub-1-year samples cannot be presented as
    annualized Sharpe/CAGR without a sample-size caveat.
    """
    empty: dict[str, Any] = {"ok": False, "samples": 0, "reason": "净值快照不足（至少需 2 天）"}
    if not nav_rows:
        return empty

    records: list[dict[str, Any]] = []
    for row in nav_rows:
        raw_date = str(row.get("uk_date") or "").strip()[:10]
        if len(raw_date) < 10:
            continue
        try:
            stamped = pd.Timestamp(raw_date)
            nav = float(row.get("total_value_gbp"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(nav) or nav <= 0.0:
            continue
        try:
            flow = float(row.get("net_flow_gbp") or 0.0)
        except (TypeError, ValueError):
            flow = 0.0
        try:
            cash = float(row.get("cash_gbp") or 0.0)
        except (TypeError, ValueError):
            cash = 0.0
        bench_raw = row.get("benchmark_close")
        try:
            bench = float(bench_raw) if bench_raw is not None else np.nan
        except (TypeError, ValueError):
            bench = np.nan
        records.append(
            {"date": stamped.normalize(), "nav": nav, "flow": flow, "cash": max(cash, 0.0), "bench": bench}
        )

    if len(records) < 2:
        return empty

    df = pd.DataFrame(records).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 2:
        return empty

    prev_nav = df["nav"].shift(1)
    valid = prev_nav > 0.0
    rets = pd.Series(np.nan, index=df.index, dtype=float)
    rets.loc[valid] = (df.loc[valid, "nav"] - df.loc[valid, "flow"]) / prev_nav.loc[valid] - 1.0
    rets = rets.dropna()
    if rets.empty:
        return empty

    arr = 1.0 + rets.to_numpy(dtype=float)
    curve = np.cumprod(arr)
    curve_full = np.concatenate(([1.0], curve))
    peaks = np.maximum.accumulate(curve_full)
    drawdown = curve_full / peaks - 1.0
    trough_k = int(np.argmin(drawdown))
    max_dd = float(drawdown[trough_k])

    def _date_at(k: int) -> pd.Timestamp:
        return df.index[0] if k == 0 else rets.index[k - 1]

    peak_k = int(np.argmax(curve_full[: trough_k + 1])) if trough_k > 0 else 0
    peak_value = float(curve_full[peak_k])
    recovery_k: int | None = None
    for k in range(trough_k + 1, len(curve_full)):
        if curve_full[k] >= peak_value - 1e-12:
            recovery_k = k
            break
    trough_date = _date_at(trough_k)
    recovery_days: int | None = None
    if recovery_k is not None:
        recovery_days = int((_date_at(recovery_k) - trough_date).days)

    total_return = float(curve[-1]) - 1.0
    span_days = max(int((df.index[-1] - df.index[0]).days), 1)
    years = span_days / 365.25
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if (years >= 0.25 and total_return > -1.0) else None

    mean_gap = span_days / max(len(df) - 1, 1)
    periods_per_year = float(np.clip(365.0 / max(mean_gap, 0.5), 12.0, 366.0))

    ann_vol: float | None = None
    if len(rets) >= min_periods:
        std = float(np.std(rets.to_numpy(dtype=float), ddof=1))
        if np.isfinite(std) and std > 0.0:
            ann_vol = std * math.sqrt(periods_per_year) * 100.0

    rf_pct = 0.0 if risk_free_annual_pct is None else float(risk_free_annual_pct)
    mar_period = (rf_pct / 100.0) / periods_per_year
    excess = rets.to_numpy(dtype=float) - mar_period
    downside = float(np.sqrt(np.mean(np.square(np.minimum(excess, 0.0)))))
    sortino: float | None = None
    if len(rets) >= min_periods and downside > 1e-12:
        sortino = float(np.mean(excess)) / downside * math.sqrt(periods_per_year)

    calmar: float | None = None
    if cagr is not None and max_dd < -1e-9:
        calmar = cagr / abs(max_dd)

    bench_rets: pd.Series | None = None
    bench_source = ""
    if df["bench"].notna().sum() >= max(5, int(len(df) * 0.6)):
        bench_series = df["bench"].astype(float).dropna()
        bench_rets = bench_series.pct_change().dropna()
        bench_source = "nav_benchmark_close"
    elif benchmark_hist is not None and not getattr(benchmark_hist, "empty", True):
        bench_col = _resolve_col(benchmark_hist, "Adj Close") or _resolve_col(benchmark_hist, "Close")
        if bench_col:
            series = pd.to_numeric(benchmark_hist[bench_col], errors="coerce").dropna().astype(float)
            series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            aligned = series.reindex(df.index).ffill(limit=3)
            bench_rets = aligned.pct_change().dropna()
            bench_source = "benchmark_hist"

    alpha_ann: float | None = None
    beta_val: float | None = None
    corr_val: float | None = None
    info_ratio: float | None = None
    tracking_error: float | None = None
    excess_ann: float | None = None
    if bench_rets is not None and not bench_rets.empty:
        pair = pd.concat([rets.rename("p"), bench_rets.rename("b")], axis=1, join="inner").dropna()
        if len(pair) >= max(10, min_periods):
            p_arr = pair["p"].to_numpy(dtype=float)
            b_arr = pair["b"].to_numpy(dtype=float)
            var_b = float(np.var(b_arr, ddof=1))
            if var_b > 1e-14:
                beta_val = float(np.cov(p_arr, b_arr, ddof=1)[0, 1]) / var_b
                alpha_ann = (float(np.mean(p_arr)) - beta_val * float(np.mean(b_arr))) * periods_per_year * 100.0
            std_p = float(np.std(p_arr, ddof=1))
            std_b = float(np.std(b_arr, ddof=1))
            if std_p > 1e-14 and std_b > 1e-14:
                corr_val = float(np.corrcoef(p_arr, b_arr)[0, 1])
            diff = p_arr - b_arr
            std_diff = float(np.std(diff, ddof=1))
            excess_ann = float(np.mean(diff)) * periods_per_year * 100.0
            if std_diff > 1e-14:
                tracking_error = std_diff * math.sqrt(periods_per_year) * 100.0
                info_ratio = float(np.mean(diff)) / std_diff * math.sqrt(periods_per_year)

    cash_drag: float | None = None
    invested_mask = df["nav"] > 0.0
    if invested_mask.any():
        cash_ratio = (df.loc[invested_mask, "cash"] / df.loc[invested_mask, "nav"]).clip(lower=0.0, upper=1.0)
        cash_drag = float(cash_ratio.mean()) * 100.0

    expected_bdays = int(np.busday_count(df.index[0].date(), (df.index[-1] + pd.Timedelta(days=1)).date()))

    return {
        "ok": True,
        "samples": int(len(df)),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "span_days": span_days,
        "coverage_pct": round(min(100.0, len(df) / max(expected_bdays, 1) * 100.0), 1),
        "periods_per_year": round(periods_per_year, 1),
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2) if cagr is not None else None,
        "ann_vol_pct": round(ann_vol, 2) if ann_vol is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "risk_free_annual_pct": round(rf_pct, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "drawdown_peak_date": _date_at(peak_k).strftime("%Y-%m-%d"),
        "drawdown_trough_date": trough_date.strftime("%Y-%m-%d"),
        "drawdown_recovery_days": recovery_days,
        "drawdown_unrecovered": bool(recovery_k is None and max_dd < -1e-9),
        "calmar": round(calmar, 2) if calmar is not None else None,
        "cash_drag_pct": round(cash_drag, 1) if cash_drag is not None else None,
        "bench_source": bench_source,
        "bench_beta": round(beta_val, 3) if beta_val is not None else None,
        "bench_corr": round(corr_val, 3) if corr_val is not None else None,
        "bench_alpha_annual_pct": round(alpha_ann, 2) if alpha_ann is not None else None,
        "bench_excess_annual_pct": round(excess_ann, 2) if excess_ann is not None else None,
        "bench_tracking_error_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "bench_info_ratio": round(info_ratio, 2) if info_ratio is not None else None,
    }


def compute_trade_attribution(portfolio: PortfolioData) -> dict[str, Any]:
    """Aggregate *actual* filled-order performance, grouped by wallet currency.

    Never cross-sums currencies (GBP/USD/GBp isolation). Sources are already-fetched
    T212 fields: ``realised_pnl`` per SELL fill, ``tax_total``, ``recent_dividends``
    and account ``total_fx_impact``. Holding days are approximated as
    "sell fill − earliest prior BUY fill of that ticker in the same account".
    """
    per_ccy: dict[str, dict[str, float]] = {}
    dividends_by_ccy: dict[str, float] = {}
    fx_impact = 0.0
    holding_days: list[float] = []
    sells_total = 0
    win_total = 0
    loss_total = 0

    for acc in (portfolio.isa, portfolio.invest):
        fx_impact += float(getattr(acc, "total_fx_impact", 0.0) or 0.0)
        fills_by_ticker: dict[str, list[Any]] = {}
        for order in getattr(acc, "recent_orders", []) or []:
            ticker = str(getattr(order, "ticker", "") or "")
            if ticker:
                fills_by_ticker.setdefault(ticker, []).append(order)

        for orders in fills_by_ticker.values():
            orders.sort(key=lambda o: int(getattr(o, "filled_at_ms", 0) or 0))
            first_buy_ms: int | None = None
            for order in orders:
                side = str(getattr(order, "side", "") or "").upper()
                stamped = int(getattr(order, "filled_at_ms", 0) or 0)
                if side == "BUY":
                    if first_buy_ms is None or (stamped and stamped < first_buy_ms):
                        first_buy_ms = stamped or first_buy_ms
                    continue
                if side != "SELL":
                    continue
                ccy = str(getattr(order, "wallet_currency", "GBP") or "GBP").strip().upper() or "GBP"
                slot = per_ccy.setdefault(
                    ccy,
                    {"sells": 0.0, "wins": 0.0, "losses": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
                     "realised": 0.0, "taxes": 0.0},
                )
                pnl = float(getattr(order, "realised_pnl", 0.0) or 0.0)
                slot["sells"] += 1.0
                slot["realised"] += pnl
                slot["taxes"] += float(getattr(order, "tax_total", 0.0) or 0.0)
                sells_total += 1
                if pnl > 0.01:
                    slot["wins"] += 1.0
                    slot["gross_win"] += pnl
                    win_total += 1
                elif pnl < -0.01:
                    slot["losses"] += 1.0
                    slot["gross_loss"] += abs(pnl)
                    loss_total += 1
                if first_buy_ms is not None and stamped > first_buy_ms:
                    holding_days.append((stamped - first_buy_ms) / 86_400_000.0)

        for dividend in getattr(acc, "recent_dividends", []) or []:
            ccy = str(getattr(dividend, "currency", "GBP") or "GBP").strip().upper() or "GBP"
            dividends_by_ccy[ccy] = dividends_by_ccy.get(ccy, 0.0) + float(
                getattr(dividend, "amount", 0.0) or 0.0
            )

    detail: dict[str, dict[str, Any]] = {}
    for ccy, slot in per_ccy.items():
        sells = int(slot["sells"])
        gross_loss = float(slot["gross_loss"])
        detail[ccy] = {
            "sells": sells,
            "wins": int(slot["wins"]),
            "losses": int(slot["losses"]),
            "win_rate_pct": round(slot["wins"] / sells * 100.0, 1) if sells else 0.0,
            "profit_factor": round(slot["gross_win"] / gross_loss, 2) if gross_loss > 1e-9 else None,
            "expectancy": round(slot["realised"] / sells, 2) if sells else 0.0,
            "realised": round(slot["realised"], 2),
            "taxes": round(slot["taxes"], 2),
            "avg_win": round(slot["gross_win"] / slot["wins"], 2) if slot["wins"] else None,
            "avg_loss": round(-gross_loss / slot["losses"], 2) if slot["losses"] else None,
        }

    return {
        "ok": sells_total > 0,
        "sells": sells_total,
        "wins": win_total,
        "losses": loss_total,
        "win_rate_pct": round(win_total / sells_total * 100.0, 1) if sells_total else 0.0,
        "by_currency": detail,
        "dividends_by_currency": {k: round(v, 2) for k, v in dividends_by_ccy.items() if abs(v) >= 0.005},
        "fx_impact_gbp": round(fx_impact, 2),
        "avg_holding_days": round(float(np.mean(holding_days)), 1) if holding_days else None,
        "holding_sample": len(holding_days),
    }


def compute_signal_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """T+N Alpha t-statistics and decay slope over settled recommendations.

    ``t = mean / (std / sqrt(n))``; ``significant`` only at |t| >= 2 with n >= 3.
    Small samples are never hidden: ``sample`` is always returned for honest labels.
    """
    horizons = (("t5", "alpha_pct"), ("t10", "t10_alpha_pct"), ("t20", "t20_alpha_pct"))
    stats: dict[str, dict[str, Any]] = {}
    for key, field in horizons:
        values: list[float] = []
        for row in rows:
            raw = row.get(field)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                values.append(val)
        count = len(values)
        if count == 0:
            stats[key] = {"n": 0, "mean_pct": None, "std_pct": None, "t_stat": None, "significant": False}
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if count >= 2 else 0.0
        t_stat: float | None = None
        if count >= 3 and std > 1e-12:
            t_stat = mean / (std / math.sqrt(count))
        stats[key] = {
            "n": count,
            "mean_pct": round(mean, 2),
            "std_pct": round(std, 2),
            "t_stat": round(t_stat, 2) if t_stat is not None else None,
            "significant": bool(t_stat is not None and abs(t_stat) >= 2.0),
        }

    decay: float | None = None
    if stats["t20"]["mean_pct"] is not None and stats["t5"]["mean_pct"] is not None:
        decay = round(float(stats["t20"]["mean_pct"]) - float(stats["t5"]["mean_pct"]), 2)
    return {
        "ok": stats["t5"]["n"] > 0,
        "sample": stats["t5"]["n"],
        "horizons": stats,
        "decay_t20_minus_t5_pct": decay,
    }


def compute_concentration_metrics(portfolio: PortfolioData) -> dict[str, Any]:
    """HHI / effective holdings / top-N weights computed on invested market value."""
    values: dict[str, float] = {}
    for pos in portfolio.all_positions():
        symbol = str(getattr(pos, "yahoo_symbol", "") or getattr(pos, "ticker", "") or "").strip().upper()
        value = float(getattr(pos, "current_value", 0.0) or 0.0)
        if not symbol or value <= 0.0:
            continue
        values[symbol] = values.get(symbol, 0.0) + value

    if not values:
        return {"ok": False, "positions": 0}

    invested = float(sum(values.values()))
    total = float(getattr(portfolio, "total_combined_value", 0.0) or 0.0) or invested
    weights = sorted((v / invested for v in values.values()), reverse=True)
    hhi = float(sum(w * w for w in weights))
    return {
        "ok": True,
        "positions": len(values),
        "invested_gbp": round(invested, 2),
        "cash_pct": round(max(total - invested, 0.0) / total * 100.0, 1) if total > 0 else None,
        "hhi": round(hhi, 4),
        "effective_n": round(1.0 / hhi, 2) if hhi > 0 else None,
        "top1_pct": round(weights[0] * 100.0, 1),
        "top3_pct": round(sum(weights[:3]) * 100.0, 1),
    }


# ---------------------------------------------------------------------------
# v5.5.1: core-policy benchmark (regional-tilt Beta vs stock-picking Alpha)
# ---------------------------------------------------------------------------
_CORE_POLICY_BENCHMARK_WEIGHTS: dict[str, float] = {
    "VWRP.L": 70.0,
    "VERG.L": 13.3,
    "ISF.L": 10.0,
    "VUAG.L": 6.7,
}


def compute_core_policy_benchmark_comparison(
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 60,
) -> dict[str, Any] | None:
    """Compare the four-bucket core policy mix against the default VWRP.L benchmark.

    Returns ``allocation_effect_pct`` (policy mix return minus VWRP.L return over the
    same window) and annualised tracking error, so the weekly / /perf review can tell
    a deliberate UK/EU regional tilt (Beta) apart from genuine stock-picking Alpha.
    """
    if not isinstance(histories, dict) or not histories:
        return None
    bench_sym = "VWRP.L"
    bench_hist = histories.get(bench_sym) if bench_sym in histories else histories.get("VWRP")
    bench_closes = _extract_close_series(bench_hist, int(lookback_days) + 5)
    if bench_closes is None or len(bench_closes) < 2:
        return None

    window = max(2, min(int(lookback_days), len(bench_closes) - 1))
    bench_ret = float(bench_closes.iloc[-1] / float(bench_closes.iloc[-1 - window]) - 1.0)

    returns_by_sym: dict[str, pd.Series] = {}
    used_weights: dict[str, float] = {}
    missing: list[str] = []
    for sym, weight in _CORE_POLICY_BENCHMARK_WEIGHTS.items():
        hist = histories.get(sym) if sym in histories else histories.get(sym.split(".")[0])
        closes = _extract_close_series(hist, window + 5)
        if closes is None or len(closes) < window + 1:
            if sym != bench_sym:
                missing.append(sym)
            continue
        rets = closes.pct_change().dropna().iloc[-window:]
        if rets.empty:
            missing.append(sym)
            continue
        returns_by_sym[sym] = rets.reset_index(drop=True)
        used_weights[sym] = float(weight)

    total_w = float(sum(used_weights.values()))
    if total_w <= 0 or bench_sym not in used_weights:
        return None
    norm = {sym: weight / total_w for sym, weight in used_weights.items()}

    min_len = min(len(series) for series in returns_by_sym.values())
    if min_len < 2:
        return None
    aligned = {sym: series.iloc[-min_len:].reset_index(drop=True) for sym, series in returns_by_sym.items()}
    policy_daily = sum(aligned[sym] * w for sym, w in norm.items())
    bench_daily = aligned[bench_sym]
    diff = policy_daily - bench_daily
    policy_ret = float((1.0 + policy_daily).prod() - 1.0)
    tracking_error = float(diff.std(ddof=1) * math.sqrt(252) * 100.0) if len(diff) > 1 else None
    return {
        "lookback_days": int(min_len),
        "weights_used": {sym: round(w * 100.0, 2) for sym, w in norm.items()},
        "missing_symbols": missing,
        "policy_return_pct": round(policy_ret * 100.0, 2),
        "benchmark_return_pct": round(bench_ret * 100.0, 2),
        "allocation_effect_pct": round((policy_ret - bench_ret) * 100.0, 2),
        "tracking_error_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "benchmark_symbol": bench_sym,
    }
