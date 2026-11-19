"""Schedule window constants and pure T+5/10/20 settlement, MFE/MAE, LSE unit-alignment, and fill-matching math."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.data_providers.analytics import _normalize_adjusted_ohlcv, resolve_market_benchmark
from src.engine.calendar_utils import has_market_closed_on_date, infer_market_from_symbol

LONDON_TZ = ZoneInfo("Europe/London")

MORNING_CATCHUP_END_MINUTES = 9 * 60 + 30  # 09:30 UK
DAILY_SETTLEMENT_START_MINUTES = 22 * 60  # 22:00 UK (after US/UK/EU close & Friday radar)
DAILY_SETTLEMENT_END_MINUTES = 23 * 60 + 30  # 23:30 UK catch-up cutoff
# IMP-18: evening EOD NAV snapshot so the TWR curve samples post-close, not only pre-market.
# Starts after prewarm (21:20) and the Friday weekly radar (21:30~21:35) to avoid job collisions,
# and stays before the 22:00 settlement job.
NAV_SNAPSHOT_START_MINUTES = 21 * 60 + 40  # 21:40 UK (US close 21:00 UK in both BST and GMT)
NAV_SNAPSHOT_END_MINUTES = 22 * 60  # 22:00 UK, hands over to the settlement job
WEEKLY_AUDIT_START_MINUTES = 10 * 60  # 10:00 UK Saturday idle window
WEEKLY_AUDIT_END_MINUTES = 22 * 60  # 22:00 UK Saturday cutoff
INTRADAY_SENTINEL_START_MINUTES = 8 * 60  # 08:00 UK (LSE open)
INTRADAY_SENTINEL_END_MINUTES = 21 * 60  # 21:00 UK (US close)
WEEKLY_WALLET_RECONCILE_START_MINUTES = 12 * 60  # 12:00 UK Saturday ledger reminder window


def _parse_recommendation_date(rec: dict[str, Any]) -> date:
    """Extract the UK calendar date of a recommendation record."""
    created_uk = str(rec.get("created_at_uk") or "").strip()
    if len(created_uk) >= 10:
        try:
            return date.fromisoformat(created_uk[:10])
        except ValueError:
            pass
    created_ms = int(rec.get("created_at_ms") or 0)
    return datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).astimezone(LONDON_TZ).date()


def _resolve_benchmark_candidates(symbol: str, stored_benchmark: Any) -> list[str]:
    """Order stored benchmark first, then regional defaults from analytics."""
    candidates: list[str] = []
    stored = str(stored_benchmark or "").strip().upper()
    if stored:
        candidates.append(stored)
    candidates.extend(resolve_market_benchmark(symbol))
    return list(dict.fromkeys(c for c in candidates if c))


def _extract_closed_daily_ohlc(
    df: pd.DataFrame | None,
    market: str,
    now_utc: datetime,
) -> pd.DataFrame:
    """Return sorted daily High/Low/Close DataFrame keyed by calendar date for closed sessions."""
    empty = pd.DataFrame(columns=["High", "Low", "Close"])
    if df is None or df.empty or "Close" not in df.columns:
        return empty

    norm = _normalize_adjusted_ohlcv(df)
    if norm is None or norm.empty or "Close" not in norm.columns:
        return empty

    closes = pd.to_numeric(norm["Close"], errors="coerce").dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return empty

    highs = pd.to_numeric(norm["High"], errors="coerce") if "High" in norm.columns else closes
    lows = pd.to_numeric(norm["Low"], errors="coerce") if "Low" in norm.columns else closes

    idx = pd.to_datetime(closes.index)
    by_date: dict[date, tuple[float, float, float]] = {}
    for ts, close_val in zip(idx, closes.to_numpy(dtype=float)):
        if not math.isfinite(close_val) or close_val <= 0:
            continue
        bar_date = ts.date()
        if not has_market_closed_on_date(bar_date, market=market, now_utc=now_utc):
            continue
        raw_high = highs.loc[ts] if ts in highs.index else close_val
        raw_low = lows.loc[ts] if ts in lows.index else close_val
        if isinstance(raw_high, pd.Series):
            raw_high = raw_high.iloc[-1]
        if isinstance(raw_low, pd.Series):
            raw_low = raw_low.iloc[-1]
        h_val = float(raw_high) if pd.notna(raw_high) and math.isfinite(float(raw_high)) and float(raw_high) > 0 else float(close_val)
        l_val = float(raw_low) if pd.notna(raw_low) and math.isfinite(float(raw_low)) and float(raw_low) > 0 else float(close_val)
        by_date[bar_date] = (max(h_val, float(close_val)), min(l_val, float(close_val)), float(close_val))

    if not by_date:
        return empty
    ordered_dates = sorted(by_date.keys())
    return pd.DataFrame(
        [by_date[d] for d in ordered_dates],
        index=ordered_dates,
        columns=["High", "Low", "Close"],
        dtype=float,
    )


def _extract_closed_daily_closes(
    df: pd.DataFrame | None,
    market: str,
    now_utc: datetime,
) -> pd.Series:
    """Return sorted daily close prices keyed by calendar date for sessions already closed."""
    ohlc = _extract_closed_daily_ohlc(df, market=market, now_utc=now_utc)
    if ohlc.empty:
        return pd.Series(dtype=float)
    return ohlc["Close"]


def _align_lse_unit_100x(value: float, reference: float, symbol: str = "") -> float:
    """Align 100x GBp/GBP unit mismatch on London (.L) instruments against a reference price."""
    if symbol and not symbol.strip().upper().endswith(".L"):
        return value
    if value <= 0 or reference <= 0:
        return value
    ratio = reference / value
    if 20.0 <= ratio <= 200.0:
        return value * 100.0
    if 0.005 <= ratio <= 0.05:
        return value * 0.01
    return value


def _align_entry_price_gbp_guard(
    symbol: str,
    entry_price: float,
    stock_closes: pd.Series,
    rec_date: date,
    exit_price: float,
) -> float | None:
    """Align GBp/GBP 100x unit mismatch on .L stocks; return None if unresolved."""
    eff_entry = float(entry_price)
    if symbol.endswith(".L"):
        prior_dates = [d for d in stock_closes.index if d <= rec_date]
        if prior_dates and (rec_date - prior_dates[-1]).days <= 5:
            hist_entry = float(stock_closes.loc[prior_dates[-1]])
            eff_entry = _align_lse_unit_100x(eff_entry, hist_entry, symbol)
        exit_ratio = exit_price / eff_entry
        if 0.005 <= exit_ratio <= 0.05 or 20.0 <= exit_ratio <= 200.0:
            return None
    return eff_entry


def _trade_direction(action: str) -> int:
    """+1 for long entries (BUY/ADD/SWAP), -1 for risk-reducing actions (TRIM/STOP/SELL).

    /pick SWAP is a rotation *into* a new long position, so its return direction is +1.
    """
    return -1 if str(action or "").strip().upper() in {"TRIM", "STOP", "SELL"} else 1


def _direction_adjusted_return_pct(action: str, entry: float, exit_price: float) -> float:
    """Long return for BUY/ADD/SWAP; avoided-loss return for TRIM/STOP/SELL."""
    if entry <= 0:
        return 0.0
    raw = ((exit_price - entry) / entry) * 100.0
    return raw * _trade_direction(action)


def _direction_adjusted_alpha_pct(action: str, entry: float, exit_price: float, bench_ret_pct: float) -> float:
    """Benchmark excess with the direction applied to the whole spread, not just the stock leg.

    Long: ``stock_ret - bench_ret``. Risk-reducing: ``bench_ret - stock_ret`` so that
    cutting a stock that then falls -3% while the benchmark rises +5% scores +8%.
    """
    if entry <= 0:
        return 0.0
    unadjusted_stock_ret = ((exit_price - entry) / entry) * 100.0
    return (unadjusted_stock_ret - float(bench_ret_pct)) * _trade_direction(action)


def _entry_fill_status(
    action: str,
    ohlc: pd.DataFrame,
    rec_date: date,
    target_date: date,
    eff_entry: float,
    *,
    tolerance_pct: float = 0.0,
    exclude_rec_date: bool = False,
) -> str:
    """Daily-bar approximation of whether a BUY/ADD entry was actually touched.

    We cannot know the exact order type from the ledger, so a BUY/ADD is treated as
    NOT_FILLED when the low never traded at/below the reference entry. TRIM/STOP are
    assumed executable at the reference price (they are risk-reducing actions).
    When the recommendation was generated after that session's close, its own day's
    low is already history and must be excluded (``exclude_rec_date=True``).
    """
    act = str(action or "").strip().upper()
    if act not in {"BUY", "ADD"}:
        return "ASSUMED"
    if ohlc.empty or eff_entry <= 0 or "Low" not in ohlc.columns:
        return "UNKNOWN"
    dates = ohlc.index
    if exclude_rec_date:
        window = ohlc[(dates > rec_date) & (dates <= target_date)]
    else:
        window = ohlc[(dates >= rec_date) & (dates <= target_date)]
    if window.empty:
        return "UNKNOWN"
    lows = pd.to_numeric(window["Low"], errors="coerce").dropna()
    if lows.empty:
        return "UNKNOWN"
    threshold = eff_entry * (1.0 + max(0.0, float(tolerance_pct)) / 100.0)
    return "FILLED" if float(lows.min()) <= threshold else "NOT_FILLED"


def _compute_mfe_mae_pct(
    ohlc: pd.DataFrame,
    rec_date: date,
    exit_date: date,
    eff_entry: float,
    *,
    direction: int = 1,
) -> tuple[float | None, float | None]:
    """Compute MFE/MAE% over (rec_date, exit_date] in the direction of the trade."""
    if ohlc.empty or eff_entry <= 0:
        return None, None
    dates = ohlc.index
    window = ohlc[(dates > rec_date) & (dates <= exit_date)]
    if window.empty:
        window = ohlc[dates == exit_date]
    if window.empty:
        return None, None
    max_high = float(window["High"].max())
    min_low = float(window["Low"].min())
    if not (math.isfinite(max_high) and math.isfinite(min_low)):
        return None, None
    if direction >= 0:
        mfe = round(((max_high - eff_entry) / eff_entry) * 100.0, 2)
        mae = round(((min_low - eff_entry) / eff_entry) * 100.0, 2)
    else:
        mfe = round(((eff_entry - min_low) / eff_entry) * 100.0, 2)
        mae = round(((eff_entry - max_high) / eff_entry) * 100.0, 2)
    return mfe, mae


def _match_recommendation_fill(
    rec: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
    recent_orders: list[Any],
    entry_price: float,
) -> tuple[float, float] | None:
    """Return (volume-weighted fill price, adverse slippage %) for same-account/symbol/side fills.

    Only fills inside [start_date, end_date] count. LSE fills are aligned to the recommendation's
    native unit via ``_align_lse_unit_100x`` so a pence/sterling mismatch can never fake a 100x slip.
    """
    sym = str(rec.get("symbol") or "").strip().upper()
    account = str(rec.get("account") or "").strip().upper()
    action = str(rec.get("action") or "BUY").strip().upper()
    side = "SELL" if _trade_direction(action) < 0 else "BUY"
    if not sym or entry_price <= 0:
        return None
    qty_sum = 0.0
    notional = 0.0
    for order in recent_orders or []:
        if str(getattr(order, "yahoo_symbol", "") or "").strip().upper() != sym:
            continue
        if str(getattr(order, "account", "") or "").strip().upper() != account:
            continue
        if str(getattr(order, "side", "") or "").strip().upper() != side:
            continue
        filled_ms = int(getattr(order, "filled_at_ms", 0) or 0)
        if filled_ms <= 0:
            continue
        fill_day = datetime.fromtimestamp(filled_ms / 1000.0, tz=timezone.utc).astimezone(LONDON_TZ).date()
        if fill_day < start_date or fill_day > end_date:
            continue
        qty = float(getattr(order, "quantity", 0.0) or 0.0)
        price = float(getattr(order, "price", 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        price = _align_lse_unit_100x(price, entry_price, sym)
        qty_sum += qty
        notional += qty * price
    if qty_sum <= 0:
        return None
    avg_fill = notional / qty_sum
    # Positive = worse than the recommended entry (BUY fills higher / SELL fills lower).
    slippage_pct = ((avg_fill - entry_price) / entry_price) * 100.0 * _trade_direction(action)
    return avg_fill, slippage_pct


def _compute_benchmark_return_pct(
    bench_candidates: list[str],
    get_closes_fn: Any,
    rec_date: date,
    exit_date: date,
) -> tuple[str, float] | None:
    """Find the first valid regional benchmark and return (benchmark_symbol, return_pct)."""
    for b_sym in bench_candidates:
        b_market = infer_market_from_symbol(b_sym)
        b_closes = get_closes_fn(b_sym, b_market)
        if b_closes.empty or b_closes.index[0] > rec_date or b_closes.index[-1] < exit_date:
            continue
        b_entry_dates = [d for d in b_closes.index if d <= rec_date]
        b_exit_dates = [d for d in b_closes.index if d <= exit_date]
        if not b_entry_dates or not b_exit_dates:
            continue
        b_entry = float(b_closes.loc[b_entry_dates[-1]])
        b_exit = float(b_closes.loc[b_exit_dates[-1]])
        if b_entry > 0:
            return b_sym, ((b_exit - b_entry) / b_entry) * 100.0
    return None


def _clean_reflection_lesson(raw_text: str, max_chars: int = 200) -> str:
    """Strip markdown artifacts and keep the T+N reflection compact (<80 tokens)."""
    text = str(raw_text or "").replace("#", "").replace("**", "")
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    for sep in ("。", "！", "？", "."):
        pos = clipped.rfind(sep)
        if pos >= max_chars // 2:
            return clipped[: pos + 1].strip()
    return clipped.rstrip()


def _normalize_quote(text: str) -> str:
    """Normalize whitespace so verbatim quote checks tolerate spacing differences only."""
    return re.sub(r"\s+", "", str(text or ""))


def _quote_is_verbatim(quote: str, corpus: list[str]) -> bool:
    """True only when the extracted quote appears verbatim inside one raw user utterance."""
    needle = _normalize_quote(quote)
    if len(needle) < 4:
        return False
    return any(needle in _normalize_quote(item) for item in corpus)
