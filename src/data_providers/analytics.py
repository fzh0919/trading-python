"""Deterministic pandas/numpy quantitative indicators, CYQ chip distribution, correlation matrix, and FactSheet builder."""

from __future__ import annotations

import math

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


def compute_correlation_matrix(
    histories: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    lookback_days: int = 60,
    benchmark_symbol: str | None = "SPY",
    uk_benchmark_symbol: str | None = "ISF.L",
    eu_benchmark_symbol: str | None = "EXS1.DE",
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

    raw_corr_1d = ret1_df[valid_syms].corr(min_periods=15)
    raw_corr_2d = ret2_df[valid_syms].corr(min_periods=15)
    raw_corr_5d = ret5_df[valid_syms].corr(min_periods=15)

    # For cross-timezone pairs (UK/EU 16:30 close vs US 21:00 close), compensate for closing lag via 2d/5d rolling corr
    raw_corr = raw_corr_1d.copy()
    for a in valid_syms:
        for b in valid_syms:
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

    resid_corr_1d = resid1_df.corr(min_periods=15)
    resid_corr_2d = resid2_df.corr(min_periods=15) if not resid2_df.empty else pd.DataFrame()

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
        parts.append(f"年化波动 {tech.volatility_60d_pct:.0f}%")

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
            parts.append(f"财报 {bundle.earnings_date}{days_s}")
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
    # Europe broad trackers
    "VERG": "Europe Equity (欧洲宽基)",
    "VERG.L": "Europe Equity (欧洲宽基)",
    "VEUR": "Europe Equity (欧洲宽基)",
    "VEUR.L": "Europe Equity (欧洲宽基)",
    "IMEU": "Europe Equity (欧洲宽基)",
    "IMEU.L": "Europe Equity (欧洲宽基)",
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
    if sym_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[sym_clean]
    base_clean = sym_clean.split(".")[0]
    if base_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[base_clean]

    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
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
    if "gilt" in combined_name:
        return "UK Gilts (英国国债)"
    return None


def compute_exposure_summary(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
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
    return "\n".join(lines)
