"""OHLCV adjustment self-healing, CYQ chip distribution, and technical snapshot indicators."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.data_providers.market_data import TickerMarketBundle

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

    # v5.5.9 M2: benchmark-relative strength (optional; populated when benchmark_hist is available)
    rs_20d_pct: float | None = None
    rs_60d_pct: float | None = None


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


def compute_rs_20_60(
    hist: pd.DataFrame | None,
    benchmark_hist: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    """Compute stock-vs-benchmark excess return over 20d and 60d without cross-layer imports."""
    s_close = _extract_close_series(hist, 100)
    b_close = _extract_close_series(benchmark_hist, 100)
    if s_close is None or b_close is None:
        return None, None
    aligned = pd.concat([s_close.rename("s"), b_close.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 21:
        return None, None
    ratio = (aligned["s"] / aligned["b"]) * 100.0
    rs20: float | None = None
    rs60: float | None = None
    if len(ratio) >= 21:
        base20 = float(ratio.iloc[-21])
        if base20 > 0.0:
            rs20 = round(((float(ratio.iloc[-1]) / base20) - 1.0) * 100.0, 2)
    if len(ratio) >= 61:
        base60 = float(ratio.iloc[-61])
        if base60 > 0.0:
            rs60 = round(((float(ratio.iloc[-1]) / base60) - 1.0) * 100.0, 2)
    return rs20, rs60


def compute_technical_snapshot(
    symbol: str,
    hist: pd.DataFrame | None,
    position_created_at: str | None = None,
    bundle: TickerMarketBundle | None = None,
    *,
    benchmark_hist: pd.DataFrame | None = None,
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
    rs20, rs60 = compute_rs_20_60(norm_hist, benchmark_hist) if benchmark_hist is not None else (None, None)

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
        rs_20d_pct=rs20,
        rs_60d_pct=rs60,
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
