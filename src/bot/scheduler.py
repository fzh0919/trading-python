"""Timezone-aware weekday scheduler (Europe/London + US market DST/holiday adaptation) and post-close settlement."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import (
    generate_intraday_alert,
    generate_morning_report,
    generate_us_sentinel,
)
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.analytics import (
    _CORE_POLICY_BENCHMARK_WEIGHTS,
    _compute_atr_and_adx,
    _normalize_adjusted_ohlcv,
    build_nav_row,
    compute_core_policy_benchmark_comparison,
    compute_window_return_pct,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import (
    AccountSnapshot,
    PortfolioData,
    T212Client,
    compute_trade_behavior_stats,
)
from src.engine.calendar_utils import (
    add_trading_days,
    get_us_market_schedule_uk,
    has_market_closed_on_date,
    infer_market_from_symbol,
    is_eu_market_holiday,
    is_london_weekday,
    is_uk_bank_holiday,
)
from src.engine.deployment_gate import core_sleeve_for_symbol
from src.engine.screener import run_offline_universe_and_threshold_audit, run_universe_audition
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
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


def _to_float(value: Any) -> float | None:
    """Coerce to a finite positive float, else None (used for price / prev-close overlays)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) and num > 0 else None


def _intraday_move_pct(
    hist: pd.DataFrame | None,
    price: float | None,
    *,
    today_uk: date,
    prev_close: float | None = None,
    symbol: str = "",
) -> float | None:
    """Intraday move vs prior close; `prev_close` (Finnhub/T212) overrides the daily-bar baseline."""
    px = _to_float(price)
    if px is None:
        return None
    base = _to_float(prev_close)
    if base is None:
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
        if closes.empty:
            return None
        try:
            last_bar_date = pd.Timestamp(closes.index[-1]).date()
        except (TypeError, ValueError):
            return None
        # Guard against multi-week stale caches masquerading as "intraday".
        if (today_uk - last_bar_date).days > 10:
            return None
        # A bar dated today is the running close -> the prior close is the bar before it.
        idx = -2 if (last_bar_date >= today_uk and len(closes) >= 2) else -1
        base = _to_float(closes.iloc[idx])
    if base is None:
        return None
    if symbol.strip().upper().endswith(".L"):
        px = _align_lse_unit_100x(px, base, symbol)
    return (px / base - 1.0) * 100.0


def _order_distance_pct(
    price: float | None,
    orders: list[Any],
    *,
    symbol: str = "",
) -> float | None:
    """Smallest absolute distance between the live price and any pending limit/stop trigger (%)."""
    px = _to_float(price)
    if px is None or not orders:
        return None
    distances: list[float] = []
    for order in orders:
        order_sym = symbol or str(getattr(order, "yahoo_symbol", "") or "")
        for field in ("limit_price", "stop_price"):
            trigger = _to_float(getattr(order, field, None))
            if trigger is None:
                continue
            if order_sym.strip().upper().endswith(".L"):
                trigger = _align_lse_unit_100x(trigger, px, order_sym)
            distances.append(abs(px - trigger) / px * 100.0)
    return min(distances) if distances else None


def _normalize_account_label(raw: Any) -> str:
    """Collapse any T212 account label to the two philosophy buckets used everywhere else."""
    label = str(raw or "").strip().upper()
    return "ISA" if label == "ISA" else "Invest"


def _has_protective_stop(orders: list[Any] | None, *, account: str | None = None) -> bool:
    """True when a live SELL STOP/STOP_LIMIT order protects this symbol (optionally per account)."""
    wanted = _normalize_account_label(account) if account else ""
    for order in orders or []:
        if wanted and _normalize_account_label(getattr(order, "account", "")) != wanted:
            continue
        if str(getattr(order, "side", "") or "").strip().upper() != "SELL":
            continue
        if "STOP" in str(getattr(order, "order_type", "") or "").strip().upper():
            return True
    return False


def _close_series_from_hist(hist: pd.DataFrame | None) -> pd.Series | None:
    """Case-tolerant numeric Close series (dropna) for drawdown / ATR helpers."""
    if hist is None or getattr(hist, "empty", True):
        return None
    col = None
    for candidate in hist.columns:
        if str(candidate).strip().lower() == "close":
            col = candidate
            break
    if col is None:
        return None
    closes = pd.to_numeric(hist[col], errors="coerce").dropna().astype(float)
    return closes if not closes.empty else None


def _drawdown_from_high_pct(hist: pd.DataFrame | None, window: int = 252) -> float | None:
    """Latest close vs the highest close in the trailing window (negative = below high).

    Both numbers come from the same daily series, so the LSE GBp/GBP 100x trap cannot leak in.
    """
    closes = _close_series_from_hist(hist)
    if closes is None:
        return None
    window_closes = closes.tail(max(2, int(window)))
    peak = float(window_closes.max())
    if peak <= 0:
        return None
    return (float(window_closes.iloc[-1]) / peak - 1.0) * 100.0


def _atr_pct_from_hist(hist: pd.DataFrame | None, period: int = 14) -> float | None:
    """ATR(period) as a percentage of the latest close (None when High/Low history is missing)."""
    if hist is None or getattr(hist, "empty", True):
        return None
    by_lower = {str(c).strip().lower(): c for c in hist.columns}
    high_col, low_col, close_col = by_lower.get("high"), by_lower.get("low"), by_lower.get("close")
    if high_col is None or low_col is None or close_col is None:
        return None
    highs = pd.to_numeric(hist[high_col], errors="coerce").dropna().astype(float)
    lows = pd.to_numeric(hist[low_col], errors="coerce").dropna().astype(float)
    closes = pd.to_numeric(hist[close_col], errors="coerce").dropna().astype(float)
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    atr, _ = _compute_atr_and_adx(
        highs.iloc[-n:].reset_index(drop=True),
        lows.iloc[-n:].reset_index(drop=True),
        closes.iloc[-n:].reset_index(drop=True),
        period=period,
    )
    last_close = float(closes.iloc[-1])
    if atr is None or atr <= 0 or last_close <= 0:
        return None
    return atr / last_close * 100.0


def _invest_adaptive_stop_pct(hist: pd.DataFrame | None, strategy: Any) -> float:
    """Naked-Invest adaptive stop line: 14-day ATR clamp (-8%..-18%) or the configured floor.

    Whichever of the ATR line / `invest_stop_loss_alert_pct` is closer to zero wins, so an
    unprotected position is flagged no later than either discipline would have cut it.
    """
    try:
        config_line = float(getattr(strategy, "invest_stop_loss_alert_pct", -12.0) or -12.0)
    except (TypeError, ValueError):
        config_line = -12.0
    if config_line > 0:
        config_line = -config_line
    atr_pct = _atr_pct_from_hist(hist)
    if atr_pct is None or atr_pct <= 0:
        return config_line
    atr_line = -min(max(2.8 * atr_pct, 8.0), 18.0)
    return max(config_line, atr_line)


def _format_pending_orders(orders: list[Any]) -> str:
    """Compact plain-text rendering of a symbol's live pending orders."""
    lines: list[str] = []
    for order in orders[:3]:
        side = str(getattr(order, "side", "") or "").strip()
        quantity = getattr(order, "quantity", None)
        limit_price = getattr(order, "limit_price", None)
        currency = str(getattr(order, "currency", "") or "").strip()
        account = str(getattr(order, "account", "") or "").strip()
        lines.append(f"{account} {side} {quantity} @ {limit_price} {currency}".strip())
    return "; ".join(lines)


FILL_QTY_TOLERANCE = 0.001
FILL_MATCH_GRACE_MS = 60_000
FILL_RECEIPT_MAX_CHARS = 450


def _pending_order_snapshot_row(order: Any) -> dict[str, Any]:
    """Flatten a live PendingOrderItem into a pending_order_snapshots row."""
    return {
        "account_label": str(getattr(order, "account", "") or "").strip(),
        "order_id": str(getattr(order, "id", "") or "").strip(),
        "yahoo_symbol": str(getattr(order, "yahoo_symbol", "") or "").strip().upper(),
        "side": str(getattr(order, "side", "") or "").strip().upper(),
        "quantity": float(getattr(order, "quantity", 0.0) or 0.0),
        "filled_quantity": float(getattr(order, "filled_quantity", 0.0) or 0.0),
        "limit_price": _to_float(getattr(order, "limit_price", None)),
        "stop_price": _to_float(getattr(order, "stop_price", None)),
        "order_status": str(getattr(order, "status", "") or ""),
        "currency": str(getattr(order, "currency", "") or ""),
    }


def _snapshot_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("account_label") or "").strip(),
        str(row.get("order_id") or "").strip(),
    )


def diff_pending_order_snapshots(
    previous: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
    *,
    supported_accounts: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Compare stored pending-order snapshots against this round's live orders.

    Accounts whose pending-order API is unavailable this round are excluded from the
    vanish/partial inference, so an HTTP 403 can never be mistaken for "all filled".
    """
    prev_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in previous or []:
        if not isinstance(row, dict):
            continue
        key = _snapshot_key(row)
        if key[0] and key[1]:
            prev_by_key[key] = row
    live_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in live_rows or []:
        if not isinstance(row, dict):
            continue
        key = _snapshot_key(row)
        if key[0] and key[1]:
            live_by_key[key] = row

    partial: list[dict[str, Any]] = []
    for key, live in live_by_key.items():
        prev = prev_by_key.get(key)
        if prev is None:
            continue
        prev_filled = float(prev.get("filled_quantity") or 0.0)
        live_filled = float(live.get("filled_quantity") or 0.0)
        if live_filled > prev_filled + FILL_QTY_TOLERANCE:
            partial.append(
                {
                    **live,
                    "first_seen_ms": int(prev.get("first_seen_ms") or 0),
                    "first_seen_uk": str(prev.get("first_seen_uk") or ""),
                    "previous_filled_quantity": prev_filled,
                }
            )

    supported = {str(a or "").strip() for a in (supported_accounts or set())}
    vanished = [
        prev
        for key, prev in prev_by_key.items()
        if key not in live_by_key and str(prev.get("account_label") or "").strip() in supported
    ]
    return {"partial": partial, "vanished": vanished, "live": list(live_by_key.values())}


def match_fill_history(
    order_row: dict[str, Any],
    recent_orders: list[Any],
    *,
    grace_ms: int = FILL_MATCH_GRACE_MS,
    excluded_fill_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Cross-verify a vanished pending order against T212 historical fills.

    Three lines of defence against false "filled" receipts after a cancel:

    1. When the original order id is known and history carries order ids, only an exact
       ``fill.order_id`` match counts — no fuzzy degradation.
    2. Legacy history rows without ``order_id`` may use the timestamp window, but a LIMIT
       order must still respect its physical price boundary (BUY fill <= limit*1.005,
       SELL fill >= limit*0.995).
    3. ``excluded_fill_ids`` lets the caller enforce one-to-one consumption so a single
       fill can never be claimed by two vanished orders in the same round.
    """
    account = str(order_row.get("account_label") or "").strip()
    symbol = str(order_row.get("yahoo_symbol") or "").strip().upper()
    side = str(order_row.get("side") or "").strip().upper()
    target_qty = float(order_row.get("quantity") or 0.0)
    first_seen_ms = int(order_row.get("first_seen_ms") or 0)
    order_id = str(order_row.get("order_id") or order_row.get("id") or "").strip()
    excluded = {str(i) for i in (excluded_fill_ids or set()) if str(i).strip()}
    if not account or not symbol or target_qty <= 0:
        return None
    window_start_ms = max(0, first_seen_ms - max(0, int(grace_ms))) if first_seen_ms > 0 else 0

    same_scope: list[Any] = []
    for fill in recent_orders or []:
        if str(getattr(fill, "account", "") or "").strip() != account:
            continue
        if str(getattr(fill, "yahoo_symbol", "") or "").strip().upper() != symbol:
            continue
        if str(getattr(fill, "side", "") or "").strip().upper() != side:
            continue
        same_scope.append(fill)

    history_has_order_ids = any(
        str(getattr(fill, "order_id", "") or "").strip() for fill in same_scope
    )
    if order_id and history_has_order_ids:
        # Strict mode: a cancel/expiry must return None rather than fall back to fuzzy matching.
        matched = [
            fill
            for fill in same_scope
            if str(getattr(fill, "order_id", "") or "").strip() == order_id
            and str(getattr(fill, "id", "") or "").strip() not in excluded
        ]
    else:
        limit_price = float(order_row.get("limit_price") or 0.0)
        matched = []
        for fill in same_scope:
            if str(getattr(fill, "id", "") or "").strip() in excluded:
                continue
            filled_at_ms = int(getattr(fill, "filled_at_ms", 0) or 0)
            if filled_at_ms <= 0 or filled_at_ms < window_start_ms:
                continue
            fill_price = float(getattr(fill, "price", 0.0) or 0.0)
            if limit_price > 0 and fill_price > 0:
                if side == "BUY" and fill_price > limit_price * 1.005:
                    continue
                if side == "SELL" and fill_price < limit_price * 0.995:
                    continue
            matched.append(fill)
    if not matched:
        return None

    qty = sum(abs(float(getattr(fill, "quantity", 0.0) or 0.0)) for fill in matched)
    if qty <= 0:
        return None
    notional = sum(
        abs(float(getattr(fill, "quantity", 0.0) or 0.0)) * abs(float(getattr(fill, "price", 0.0) or 0.0))
        for fill in matched
    )
    last_filled_ms = max(int(getattr(fill, "filled_at_ms", 0) or 0) for fill in matched)
    currency = (
        str(getattr(matched[-1], "inst_currency", "") or "")
        or str(getattr(matched[-1], "wallet_currency", "") or "")
        or str(order_row.get("currency") or "")
    )
    return {
        "matched_quantity": qty,
        "avg_price": (notional / qty) if qty > 0 else 0.0,
        "last_filled_at_ms": last_filled_ms,
        "currency": currency,
        "is_complete": qty >= target_qty * (1.0 - FILL_QTY_TOLERANCE),
        "fill_ids": [
            str(getattr(fill, "id", "") or "").strip()
            for fill in matched
            if str(getattr(fill, "id", "") or "").strip()
        ],
    }


def _format_uk_from_ms(ms: int) -> str:
    if int(ms or 0) <= 0:
        return ""
    return (
        datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)
        .astimezone(LONDON_TZ)
        .strftime("%Y-%m-%d %H:%M")
    )


def format_fill_receipt(
    order_row: dict[str, Any],
    match: dict[str, Any],
    *,
    partial: bool = False,
    cross_day: bool = False,
    usd_wallet_available: float | None = None,
    max_chars: int = FILL_RECEIPT_MAX_CHARS,
) -> str:
    """Render a plain-text fill receipt with an explicit wallet re-calibration nudge."""
    account = str(order_row.get("account_label") or "").strip()
    side = str(order_row.get("side") or "").strip().upper()
    symbol = str(order_row.get("yahoo_symbol") or "").strip().upper()
    order_qty = float(order_row.get("quantity") or 0.0)
    filled_qty = float(match.get("matched_quantity") or 0.0)
    avg_price = float(match.get("avg_price") or 0.0)
    currency = str(match.get("currency") or order_row.get("currency") or "").strip()
    filled_uk = _format_uk_from_ms(int(match.get("last_filled_at_ms") or 0))

    header = "⚠️ 挂单部分成交" if partial else "✅ 挂单成交回执"
    lines = [f"{header}（{account} {side} {symbol}）"]
    price_str = f"{avg_price:,.2f}" if avg_price > 0 else "成交价未回传"
    lines.append(f"• 成交: {filled_qty:g}/{order_qty:g} 股 @ {price_str} {currency}".strip())
    limit_price = order_row.get("limit_price")
    if limit_price is not None:
        try:
            lines.append(f"• 原挂单限价: {float(limit_price):,.2f} {currency}".strip())
        except (TypeError, ValueError):
            pass
    if partial:
        lines.append("• 状态: 部分成交，余量仍在挂单中，可继续等待或 /pos 复核。")
    elif not bool(match.get("is_complete", True)):
        lines.append("• 状态: 未足额成交，余量可能已撤单，请 /pos 复核。")
    else:
        lines.append("• 状态: 已全部成交。")
    if filled_uk:
        moment = f"跨日补发，成交时间 {filled_uk} UK" if cross_day else f"成交时间 {filled_uk} UK"
        lines.append(f"• {moment}")
    if account.upper() == "ISA":
        lines.append("• 现金与持仓已变：请用 /pos 复核实时仓位。")
    else:
        lines.append("• 现金与持仓已变：请 /usd 金额 或 /wallet gbp x usd y 校准钱包，再用 /pos 复核实时仓位。")
        # Read-only USD wallet hint: T212 may settle in USD or auto-convert, so never write the ledger.
        if (
            usd_wallet_available is not None
            and currency.strip().upper() == "USD"
            and avg_price > 0
            and filled_qty > 0
        ):
            notional = avg_price * filled_qty
            if side == "SELL":
                ref = float(usd_wallet_available) + notional
                lines.append(
                    f"• 若走 USD 钱包结算，参考余额约 ${ref:,.2f}（若回流 USD 钱包约 +${notional:,.2f}），"
                    f"核对后可发送 /usd {ref:,.2f}"
                )
            else:
                ref = float(usd_wallet_available) - notional
                lines.append(
                    f"• 若走 USD 钱包结算，参考余额约 ${ref:,.2f}（已扣成交额 ${notional:,.2f}），"
                    f"核对后可发送 /usd {ref:,.2f}"
                )
    return "\n".join(lines)[: max(0, int(max_chars))]


class TradingScheduler:
    """Deterministic minute-tick scheduler with per-job isolation and Telegram ops alerting."""

    def __init__(
        self,
        cfg: AppConfig,
        t212: T212Client,
        market_data: MarketDataService,
        llm: LLMBackend,
        db: AdvisorDatabase,
        telegram: TelegramClient,
        ops_notifier: OpsNotifier | None = None,
        execution_lock: asyncio.Lock | None = None,
        morning_retry_cooldown_s: float = 300.0,
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self.execution_lock: asyncio.Lock = execution_lock if execution_lock is not None else asyncio.Lock()
        self.current_job_name: str | None = None
        self.morning_retry_cooldown_s = max(0.0, float(morning_retry_cooldown_s))
        self._fired_dates: dict[str, str] = {}
        self._job_retry_after: dict[str, datetime] = {}
        self._morning_failures: dict[str, dict[str, Any]] = {}
        self._morning_missed_notified: set[str] = set()
        self.last_job_runs: dict[str, dict[str, str]] = {}
        self._last_intraday_scan_utc: datetime | None = None

    async def _is_job_locked(self, job_name: str, uk_date: str) -> bool:
        if self._fired_dates.get(job_name) == uk_date:
            return True
        if await self.db.has_scheduler_job_fired(job_name, uk_date):
            self._fired_dates[job_name] = uk_date
            return True
        return False

    async def _mark_job_locked(self, job_name: str, uk_date: str, uk_now_str: str) -> None:
        self._fired_dates[job_name] = uk_date
        await self.db.mark_scheduler_job_fired(job_name, uk_date, uk_now_str)

    async def _claim_job_run(
        self,
        job_name: str,
        uk_date: str,
        uk_now_str: str,
        *,
        now_utc: datetime | None = None,
        retry_cooldown_s: int = 0,
    ) -> bool:
        """Atomically check in-memory + SQLite persistence so restarts inside the trigger window never double-fire."""
        if retry_cooldown_s > 0:
            retry_after = self._job_retry_after.get(job_name)
            if retry_after is not None and (now_utc or datetime.now(timezone.utc)) < retry_after:
                return False
        if await self._is_job_locked(job_name, uk_date):
            return False
        await self._mark_job_locked(job_name, uk_date, uk_now_str)
        return True

    async def _release_job_run(
        self,
        job_name: str,
        uk_date: str,
        *,
        now_utc: datetime | None = None,
        retry_cooldown_s: int = 120,
    ) -> None:
        """Release a transiently failed one-shot claim so the remaining window can retry."""
        self._fired_dates.pop(job_name, None)
        try:
            await self.db.clear_scheduler_job_fired(job_name, uk_date)
        except Exception as exc:
            logger.warning("Failed to clear scheduler claim %s/%s: %s", job_name, uk_date, exc)
        if retry_cooldown_s > 0:
            base = now_utc or datetime.now(timezone.utc)
            self._job_retry_after[job_name] = base + timedelta(seconds=retry_cooldown_s)

    async def _execute_job(
        self,
        *,
        job_name: str,
        chat_id: str,
        uk_now_str: str,
        coro: Awaitable[str],
        send_to_tg: bool = True,
        send_failure_to_tg: bool = True,
    ) -> bool:
        async with self.execution_lock:
            self.current_job_name = job_name
            try:
                msg = await coro
                if msg.lstrip().startswith("⚠️"):
                    self.last_job_runs[job_name] = {"status": "FAILED", "uk_time": uk_now_str, "detail": msg[:120]}
                    if self.ops is not None:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_job_failed:{job_name}",
                            message=f"定时任务 {job_name} 返回异常结果：{msg[:200]}",
                        )
                    if send_to_tg and send_failure_to_tg and chat_id:
                        await self.tg.send_message(chat_id, msg)
                    return False

                detail = msg[:120] if not send_to_tg else ""
                if msg.lstrip().startswith("⛔"):
                    self.last_job_runs[job_name] = {"status": "BLOCKED", "uk_time": uk_now_str, "detail": msg[:120]}
                    if send_to_tg and chat_id and msg.strip():
                        await self.tg.send_message(chat_id, msg)
                    return True
                self.last_job_runs[job_name] = {"status": "OK", "uk_time": uk_now_str, "detail": detail}
                if send_to_tg and chat_id and msg.strip():
                    await self.tg.send_message(chat_id, msg)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job %s crashed", job_name)
                err_detail = f"{type(exc).__name__}: {exc}"
                self.last_job_runs[job_name] = {"status": "CRASHED", "uk_time": uk_now_str, "detail": err_detail[:120]}
                if self.ops is not None:
                    if job_name == "morning" and send_failure_to_tg and hasattr(self.ops, "notify_business_critical"):
                        await self.ops.notify_business_critical(
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            title="晨报定时任务异常崩溃",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                            bot_message="🚨 今日 /report 定时晨报生成异常中断，请发送 /report 重试或查看运维告警。",
                        )
                    else:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                        )
                return False
            finally:
                if self.current_job_name == job_name:
                    self.current_job_name = None

    async def _generate_settlement_reflection(
        self,
        *,
        rec: dict[str, Any],
        rec_date: date,
        exit_date: date,
        holding_days: int,
        entry_price: float,
        exit_price: float,
        raw_return_pct: float,
        benchmark_symbol: str,
        bench_return_pct: float,
        alpha_pct: float,
        direction: int = 1,
    ) -> str:
        """Generate a concise 2-4 sentence (<80 tokens) T+N post-mortem lesson using Hydrogen LOW."""
        symbol = str(rec.get("symbol") or "").upper()
        account = str(rec.get("account") or "ISA")
        action = str(rec.get("action") or "WATCH")
        thesis = str(rec.get("thesis") or "").strip()[:220] or "未记录"
        direction_label = "多头买入（价格涨=正确）" if direction >= 0 else "减仓/止损（价格跌=正确，避损口径）"

        system_instruction = (
            f"你是冷静客观的量化投研复盘教练。现在已知历史推荐在 T+{holding_days} 个交易日后的真实表现。\n"
            f"注意 T+{holding_days} 窗口可能短于中长线逻辑本身，不能仅凭短期波动全盘否定长线逻辑，但要直面短期择时或催化偏差。\n"
            "请用中文输出 2~4 句极简纯文本复盘教训（严格控制在 80 tokens / 120 汉字以内；"
            "严禁使用 #、**、列表符号或换行标题）：\n"
            f"1. 点明 T+{holding_days} 收益与相对 {benchmark_symbol} 的超额 Alpha 表现；\n"
            "2. 指出原推荐逻辑中哪部分被验证或证伪；\n"
            "3. 提炼 1 条下次同类选股或择时可直接复用的具体纪律。"
        )
        prompt = (
            f"标的: {symbol} ({account} | {action})\n"
            f"收益口径: {direction_label}\n"
            f"推荐日期: {rec_date.isoformat()} -> 结算日: {exit_date.isoformat()} (T+{holding_days})\n"
            f"入场价: {entry_price:.4g} -> 结算价: {exit_price:.4g}\n"
            f"原推荐逻辑: {thesis}\n"
            f"真实收益率: {raw_return_pct:+.2f}%\n"
            f"基准({benchmark_symbol})同期: {bench_return_pct:+.2f}%\n"
            f"超额 Alpha: {alpha_pct:+.2f}%"
        )
        raw_reply = await self.llm.generate(
            model=self.cfg.models.chat_model,
            prompt=prompt,
            system_instruction=system_instruction,
            thinking_level=self.cfg.models.chat_thinking_level,
            use_google_search=False,
            temperature=0.2,
            usage_label="settlement_reflection",
        )
        cleaned = _clean_reflection_lesson(raw_reply)
        if not cleaned or cleaned.lstrip().startswith("⚠️"):
            raise RuntimeError(cleaned or "empty reflection response")
        return cleaned

    def _has_audit_universe(self) -> bool:
        return bool(
            self.cfg.universe.audition_enabled
            or self.cfg.universe.isa_long_term
            or self.cfg.universe.invest_tactical
            or self.cfg.universe.etf_symbols
        )

    def _build_benchmark_dashboard(
        self,
        portfolio: PortfolioData,
        histories: dict[str, pd.DataFrame],
        *,
        previous_snapshot: PortfolioData | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """VWRP 7d/30d benchmark returns vs account net value, realised P/L and fee drag."""
        symbol = self.cfg.strategy.benchmark_symbol
        hist = histories.get(symbol)
        if hist is None or getattr(hist, "empty", True):
            hist = histories.get(symbol.split(".")[0])
        ret_7 = compute_window_return_pct(hist, 5)  # 5 trading bars ≈ 7 calendar days
        ret_30 = compute_window_return_pct(hist, 21)  # 21 trading bars ≈ 30 calendar days

        now = int(now_ms if now_ms is not None else datetime.now(timezone.utc).timestamp() * 1000)
        behavior = compute_trade_behavior_stats(
            portfolio,
            recent_days=7,
            history_days=self.cfg.strategy.order_history_days,
            cooldown_days=self.cfg.strategy.rebuy_cooldown_days,
            now_ms=now,
        )
        realized_by_ccy: dict[str, float] = {}
        for o in (*portfolio.isa.recent_orders, *portfolio.invest.recent_orders):
            if (o.side or "").upper() != "SELL":
                continue
            ccy = str(o.wallet_currency or "GBP").strip().upper() or "GBP"
            realized_by_ccy[ccy] = realized_by_ccy.get(ccy, 0.0) + float(o.realised_pnl or 0.0)

        account_change_pct: float | None = None
        baseline_age_days: float | None = None
        if previous_snapshot is not None:
            prev_val = float(previous_snapshot.total_combined_value or 0.0)
            cur_val = float(portfolio.total_combined_value or 0.0)
            if prev_val > 0 and cur_val > 0:
                account_change_pct = ((cur_val - prev_val) / prev_val) * 100.0
            base_ms = int(previous_snapshot.fetched_at_ms or 0)
            if base_ms > 0:
                baseline_age_days = round((now - base_ms) / 86400000.0, 1)

        if ret_7 is not None and account_change_pct is not None:
            alpha = account_change_pct - ret_7
            verdict = f"近7天{'跑赢' if alpha >= 0 else '跑输'}基准 {alpha:+.1f}pct"
        elif ret_7 is not None:
            verdict = f"近7天基准 {ret_7:+.1f}%（缺少账户基线快照，暂无法算超额）"
        else:
            verdict = "基准行情数据不足"

        dashboard: dict[str, Any] = {
            "benchmark_symbol": symbol,
            "benchmark_ret_7d_pct": round(ret_7, 2) if ret_7 is not None else None,
            "benchmark_ret_30d_pct": round(ret_30, 2) if ret_30 is not None else None,
            "account_total_value_gbp": round(float(portfolio.total_combined_value or 0.0), 2),
            "account_change_pct": round(account_change_pct, 2) if account_change_pct is not None else None,
            "baseline_age_days": baseline_age_days,
            "realized_pnl_by_currency": {k: round(v, 2) for k, v in realized_by_ccy.items() if abs(v) >= 0.01},
            "fees_pct_of_turnover_30d": behavior.get("fees_pct_of_turnover"),
            "buys_7d": behavior.get("buys_7d"),
            "buys_30d": behavior.get("buys_30d"),
            "verdict": verdict,
        }
        # v5.5.3: surface the four-bucket core policy mix vs VWRP.L so the weekly review can
        # separate a deliberate UK/EU regional tilt (Beta) from genuine stock-picking Alpha.
        core_cmp = compute_core_policy_benchmark_comparison(histories, lookback_days=60)
        dashboard["core_policy_benchmark"] = core_cmp
        dashboard["summary_text"] = self._format_benchmark_dashboard_text(dashboard)
        return dashboard

    @staticmethod
    def _format_benchmark_dashboard_text(dash: dict[str, Any]) -> str:
        """Render the dashboard as one compact plain-text line (no markdown)."""
        symbol = dash.get("benchmark_symbol") or "VWRP.L"
        ret_7 = dash.get("benchmark_ret_7d_pct")
        ret_30 = dash.get("benchmark_ret_30d_pct")
        parts = [
            f"{symbol} 近7天 {ret_7:+.1f}%" if ret_7 is not None else f"{symbol} 近7天 N/A",
            f"近30天 {ret_30:+.1f}%" if ret_30 is not None else "近30天 N/A",
        ]
        change = dash.get("account_change_pct")
        if change is not None:
            parts.append(f"账户净值 {float(change):+.1f}%")
        realized = dash.get("realized_pnl_by_currency") or {}
        if realized:
            parts.append("已实现 " + " ".join(f"{ccy} {float(amt):+.0f}" for ccy, amt in sorted(realized.items())))
        fees = dash.get("fees_pct_of_turnover_30d")
        if fees is not None:
            parts.append(f"税费占成交额 {float(fees):.2f}%")
        core = dash.get("core_policy_benchmark") or {}
        if core:
            te = core.get("tracking_error_pct")
            parts.append(
                f"四桶核心{int(core.get('lookback_days') or 60)}日 组合 {float(core['policy_return_pct']):+.1f}% "
                f"vs {core['benchmark_symbol']} {float(core['benchmark_return_pct']):+.1f}%"
                f"（倾斜 {float(core['allocation_effect_pct']):+.1f}pct"
                + (f" | 跟踪误差 {float(te):.1f}%" if te is not None else "")
                + "）"
            )
        parts.append(str(dash.get("verdict") or ""))
        return " | ".join(p for p in parts if p)

    async def _generate_weekly_diagnosis(
        self,
        audit: dict[str, Any],
        *,
        uk_date: str,
    ) -> str:
        """One-hop DeepSeek diagnosis over raw audit numbers + raw T+N settlement stats (no summary-of-summary)."""
        if not ds_helper_available(self.llm, self.cfg):
            return ""
        try:
            stats = await self.db.get_recommendation_stats()
            lessons = await self.db.get_recent_lessons(limit=8)
        except Exception as exc:
            logger.warning("Weekly diagnosis could not load settlement stats: %s", exc)
            return ""

        bt = audit.get("threshold_backtest") or {}
        bench_dash = str((audit.get("benchmark_dashboard") or {}).get("summary_text") or "N/A")
        cov = audit.get("fundamentals_coverage") or {}
        cov_bits: list[str] = []
        for market, item in (cov.get("markets") or {}).items():
            if not item.get("symbols"):
                continue
            cov_map = item.get("coverage_pct") or {}
            cov_bits.append(
                f"{market}({item['symbols']}只) "
                + "/".join(f"{k} {v if v is not None else 'N/A'}" for k, v in list(cov_map.items())[:3])
            )
        cov_text = " | ".join(cov_bits) if cov_bits else "N/A"
        low_cov = "、".join((cov.get("low_coverage") or [])[:5]) or "无"
        try:
            quality = await self.db.get_ops_quality_summary(days=7)
        except Exception as exc:
            logger.debug("Weekly diagnosis ops quality summary failed: %s", exc)
            quality = None
        sem = (quality or {}).get("semantic_audit") or {}
        degrade_txt = (
            f"语义质检 {int(sem.get('total') or 0)} 次 | 降级 {int((quality or {}).get('degraded_alerts') or 0)} 次"
            if quality
            else "N/A"
        )
        lines = [
            f"【本周体检 {uk_date}】池 {audit.get('universe_count', 0)} 只 | "
            f"重叠 {len(audit.get('overlap_candidates') or [])} | "
            f"僵尸 {len(audit.get('zombie_candidates') or [])}",
            f"【基准对照原始数字】{bench_dash[:300]}",
            f"【价值因子覆盖率原始数字】{cov_text[:300]} | 覆盖率<50%: {low_cov[:120]}",
            f"【近7天质检与降级】{degrade_txt}",
            f"【阈值回测原始数字】样本 {bt.get('sample_count', 0)} | {str(bt)[:600]}",
            f"【T+N 后验原始数字】总数 {stats.get('total_count', 0)} | 已结算 {stats.get('settled_count', 0)} | "
            f"原始胜率 {stats.get('win_rate_pct')} | Alpha 胜率 {stats.get('alpha_win_rate_pct')} | "
            f"T+10 Alpha {stats.get('avg_t10_alpha_pct')} | T+20 Alpha {stats.get('avg_t20_alpha_pct')} | "
            f"MFE {stats.get('avg_mfe_pct')} | MAE {stats.get('avg_mae_pct')}",
            "【近期已结算原话教训】",
        ]
        for row in lessons or []:
            lesson = str(row.get("reflection_lesson") or "").strip()
            if not lesson:
                continue
            lines.append(
                f"- {row.get('symbol')} {row.get('action')} 收益 {row.get('raw_return_pct')} "
                f"vs {row.get('benchmark_symbol')} Alpha {row.get('alpha_pct')}: {lesson[:120]}"
            )

        payload = await call_ds_json(
            self.cfg,
            self.llm,
            prompt="\n".join(lines),
            system_instruction=(
                "你是量化策略诊断官（只读）。根据原始回测、T+5/10/20 后验与基准对照数字，输出 3 条以内中文诊断："
                "①相对 VWRP 基准是跑赢还是跑输及主因（仓位/择时/摩擦）②参数/阈值与胜率结构建议 ③下周关注。"
                "禁止编造未给出的数字，禁止 Markdown # 与 **，总长 <=450 字。"
                '仅输出严格 JSON：{"diagnosis": "..."}'
            ),
            usage_label="weekly_quant_diagnosis",
        )
        if not payload:
            return ""
        text = sanitize_plain_text(str(payload.get("diagnosis") or ""), 450)
        if len(text) < 20:
            return ""
        return text

    async def _run_weekly_quant_audit_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Run 0-token offline universe audition + threshold audit and save silently to quant_health_audits."""
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")

        try:
            self.cfg.reload_pools_if_changed()
        except Exception as exc:
            logger.debug("Weekly quant audit config hot-reload failed: %s", exc)

        audition_res: dict[str, Any] | None = None
        if self.cfg.universe.audition_enabled and hasattr(self.db, "replace_active_universe_pool"):
            try:
                audition_res = await run_universe_audition(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.db,
                    sync_online_indices=True,
                    sync_t212_metadata=True,
                )
            except Exception as exc:
                logger.warning("Weekly universe audition failed; falling back to existing pool: %s", exc)

        portfolio: PortfolioData | None = None
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Weekly quant audit portfolio fetch failed; falling back to snapshot: %s", exc)
        if portfolio is None:
            try:
                portfolio = await self.db.get_latest_snapshot()
            except Exception:
                portfolio = None
        if portfolio is None:
            portfolio = PortfolioData(
                fetched_at_uk=uk_now_str,
                fetched_at_ms=int(base.timestamp() * 1000),
                isa=AccountSnapshot(ok=False, account_label="ISA"),
                invest=AccountSnapshot(ok=False, account_label="Invest"),
                is_fallback=True,
                fallback_source="empty_fallback",
            )

        active_pool_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_active_universe_pool"):
            try:
                active_pool_rows = await self.db.get_active_universe_pool()
            except Exception:
                active_pool_rows = []

        held_syms = portfolio.unique_yahoo_symbols(include_pending=False)
        if active_pool_rows:
            pool_syms = [str(r["symbol"]) for r in active_pool_rows] + list(self.cfg.universe.etf_symbols)
        else:
            pool_syms = (
                list(self.cfg.universe.isa_long_term)
                + list(self.cfg.universe.invest_tactical)
                + list(self.cfg.universe.etf_symbols)
            )
        all_syms = list(
            dict.fromkeys(
                s for s in (held_syms + pool_syms + [self.cfg.strategy.benchmark_symbol]) if s
            )
        )

        histories: dict[str, pd.DataFrame] = {}
        if all_syms:
            try:
                histories = await self.market_data.get_histories(all_syms, period="2y")
            except Exception as exc:
                logger.warning("Weekly quant audit histories fetch failed: %s", exc)

        audit_kwargs: dict[str, Any] = {
            "histories": histories,
            "portfolio": portfolio,
            "universe": self.cfg.universe,
            "strategy": self.cfg.strategy,
        }
        if active_pool_rows:
            audit_kwargs["active_pool_rows"] = active_pool_rows
        # ISA value-factor coverage health: read the cached fundamentals (never fetch online here).
        if all_syms and hasattr(self.db, "get_fundamentals_cache"):
            try:
                fundamentals_cache = await self.db.get_fundamentals_cache(all_syms)
                audit_kwargs["fundamentals_rows"] = list((fundamentals_cache or {}).values())
            except Exception as exc:
                logger.debug("Weekly fundamentals coverage cache read failed: %s", exc)
        audit = run_offline_universe_and_threshold_audit(**audit_kwargs)
        if audition_res:
            audit["audition"] = audition_res
        previous_snapshot: PortfolioData | None = None
        try:
            previous_snapshot = await self.db.get_previous_snapshot(min_age_hours=24 * 6)
        except Exception as exc:
            logger.debug("Weekly benchmark baseline snapshot unavailable: %s", exc)
        # Make sure all four core-policy sleeves are present before the comparison runs.
        missing_core_syms = [s for s in _CORE_POLICY_BENCHMARK_WEIGHTS if s not in histories]
        if missing_core_syms:
            try:
                extra_hist = await self.market_data.get_histories(missing_core_syms, period="2y")
                histories.update(extra_hist or {})
            except Exception as exc:
                logger.debug("Weekly core policy histories fetch failed: %s", exc)
        audit["benchmark_dashboard"] = self._build_benchmark_dashboard(
            portfolio,
            histories,
            previous_snapshot=previous_snapshot,
            now_ms=int(base.timestamp() * 1000),
        )
        diagnosis = await self._generate_weekly_diagnosis(audit, uk_date=uk_date)
        await self.db.save_quant_audit(
            audit_date=uk_date,
            summary_text=audit["summary_text"],
            payload=audit,
            created_at_uk=uk_now_str,
            ai_diagnosis=diagnosis,
        )
        return {**audit, "audit_date": uk_date, "ai_diagnosis": diagnosis}

    async def run_pool_audition(
        self,
        *,
        sync_online_indices: bool = True,
        sync_t212_metadata: bool = True,
        acquire_lock: bool = True,
    ) -> dict[str, Any]:
        """Refresh the curated 150-stock active universe pool (acquiring execution_lock unless already held)."""
        async def _run() -> dict[str, Any]:
            prev_job = self.current_job_name
            self.current_job_name = "refresh_pool"
            try:
                return await run_universe_audition(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.db,
                    sync_online_indices=sync_online_indices,
                    sync_t212_metadata=sync_t212_metadata,
                )
            finally:
                if self.current_job_name == "refresh_pool":
                    self.current_job_name = prev_job

        if acquire_lock:
            async with self.execution_lock:
                return await _run()
        return await _run()

    async def run_weekly_quant_audit(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Acquire execution_lock and silently execute weekly offline universe & threshold audit."""
        async with self.execution_lock:
            self.current_job_name = "weekly_quant_audit"
            try:
                return await self._run_weekly_quant_audit_impl(now_utc=now_utc)
            finally:
                if self.current_job_name == "weekly_quant_audit":
                    self.current_job_name = None

    async def _run_weekly_quant_audit_job(self, now_utc: datetime) -> str:
        audit = await self._run_weekly_quant_audit_impl(now_utc=now_utc)
        bt = audit.get("threshold_backtest") or {}
        summary = (
            f"体检池 {audit.get('universe_count', 0)} 只 | "
            f"重叠 {len(audit.get('overlap_candidates') or [])} | "
            f"僵尸 {len(audit.get('zombie_candidates') or [])} | "
            f"回测样本 {bt.get('sample_count', 0)}"
        )
        diagnosis = str(audit.get("ai_diagnosis") or "").strip()
        dash = str((audit.get("benchmark_dashboard") or {}).get("summary_text") or "").strip()
        benchmark_block = f"\n{dash}" if dash else ""
        if diagnosis:
            return (
                f"📊 周六策略体检周报（{audit.get('audit_date', '')}）\n"
                f"{diagnosis}{benchmark_block}\n\n{summary}"
            )
        # Silent fallback: keep the DB audit as the source of truth and never spam a bare one-liner.
        logger.info("Weekly quant audit completed without AI diagnosis: %s", summary)
        return ""

    async def _build_weekly_wallet_reconcile_message(self, now_utc: datetime | None = None) -> str:
        """Compose a lightweight weekly Invest GBP+USD ledger reminder (0-token, no AI calls)."""
        dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            return f"⚠️ 钱包对账提醒生成失败：{exc}"
        invest = getattr(portfolio, "invest", None)
        if invest is None or not getattr(invest, "ok", False):
            return f"⚠️ Invest 账户读取失败，本周钱包对账提醒跳过：{getattr(invest, 'error', '未知错误')}"

        records: dict[str, dict[str, Any]] = {}
        if hasattr(self.db, "get_wallet_balances"):
            try:
                records = await self.db.get_wallet_balances("Invest")
            except Exception as exc:
                logger.warning("Weekly wallet reconcile: failed loading balances: %s", exc)
        usd = records.get("USD") or {}
        usd_total = float(usd.get("available_amount") or 0.0) + float(usd.get("reserved_amount") or 0.0)
        updated_uk = str(usd.get("updated_at_uk") or "")
        updated_ms = int(usd.get("updated_at_ms") or 0)
        age_days = (
            max(0, int((int(dt_uk.timestamp() * 1000) - updated_ms) / 86_400_000)) if updated_ms > 0 else None
        )
        rate: float | None = getattr(self.t212, "gbp_usd_rate", None)
        if rate is None and hasattr(self.market_data, "get_macro_snapshot"):
            try:
                macro = await self.market_data.get_macro_snapshot()
                candidate = getattr(macro, "gbp_usd", None)
                if candidate is not None and float(candidate) > 0:
                    rate = float(candidate)
            except Exception as exc:
                logger.warning("Weekly wallet reconcile: macro fetch failed: %s", exc)

        lines = ["🧾 每周 Invest 双钱包对账提醒"]
        usd_line = f"• 本地 USD 记录: ${usd_total:,.2f}"
        if updated_uk:
            usd_line += f"（{updated_uk} UK"
            usd_line += f"，{age_days} 天前）" if age_days is not None else "）"
        lines.append(usd_line)
        lines.append(f"• T212 实时总现金: £{float(invest.total_cash or 0.0):,.2f}")
        if rate:
            derived_gbp = max(0.0, float(invest.total_cash or 0.0) - usd_total / rate)
            lines.append(f"• 推算 GBP 钱包: ≈ £{derived_gbp:,.2f}（GBP/USD {rate:.4f}）")
        lines.append(
            "提示：若本周有换汇、美股买卖回笼或手动换币，请回复 /usd 金额 校准；"
            "也可 /wallet gbp 金额 单报，或 /wallet gbp x usd y 双报核对。"
        )
        return "\n".join(lines)

    async def _run_weekly_wallet_reconcile_job(self, now_utc: datetime) -> str:
        return await self._build_weekly_wallet_reconcile_message(now_utc=now_utc)

    async def _extract_preference_notes(self, now_utc: datetime | None = None) -> int:
        """Single-hop DeepSeek extraction of explicit user preferences from today's raw chat utterances."""
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        if not ds_helper_available(self.llm, self.cfg):
            return 0
        day_start = dt_uk.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            queries = await self.db.get_user_queries_since(int(day_start.timestamp() * 1000), limit=60)
        except Exception as exc:
            logger.warning("Preference extraction could not load chat history: %s", exc)
            return 0
        corpus = [q for q in queries if str(q).strip()]
        if not corpus:
            return 0

        joined = "\n---\n".join(f"[{idx + 1}] {q}" for idx, q in enumerate(corpus))
        payload = await call_ds_json(
            self.cfg,
            self.llm,
            prompt=f"【今日用户原始发言（逐条独立，禁止合并改写）】\n{joined[:6000]}",
            system_instruction=(
                "你是用户偏好抽取器（只读）。只抽取用户主动声明的明确硬偏好/资金计划/禁忌"
                "（例如「年底前不碰军工」「明年3月前不加仓科技」）。没有明确偏好就返回空数组。"
                "quote 必须是某一条发言的逐字片段（不要改写、不要合并），rule 是 <=60 字可执行规则，ttl_days 1~30。\n"
                '仅输出严格 JSON：{"notes": [{"quote": "...", "rule": "...", "ttl_days": 14}]}'
            ),
            usage_label="preference_extract",
        )
        if not payload:
            return 0
        raw_notes = payload.get("notes")
        if not isinstance(raw_notes, list):
            return 0

        max_notes = max(1, int(self.cfg.strategy.preference_max_notes_per_day))
        default_ttl = max(1, int(self.cfg.strategy.preference_ttl_days))
        validated: list[dict[str, Any]] = []
        for item in raw_notes[: max_notes * 2]:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            rule = sanitize_plain_text(str(item.get("rule") or ""), 60)
            if not quote or not rule or not _quote_is_verbatim(quote, corpus):
                if quote and not _quote_is_verbatim(quote, corpus):
                    logger.info("Dropping non-verbatim preference quote: %s", quote[:40])
                continue
            try:
                ttl = int(item.get("ttl_days") or default_ttl)
            except (TypeError, ValueError):
                ttl = default_ttl
            ttl = min(30, max(1, ttl))
            validated.append(
                {
                    "source_cycle_date": uk_date,
                    "quote_text": sanitize_plain_text(quote, 120),
                    "rule_summary": rule,
                    "expires_on_uk": (dt_uk.date() + timedelta(days=ttl)).isoformat(),
                }
            )
            if len(validated) >= max_notes:
                break
        if not validated:
            return 0
        saved = await self.db.upsert_preference_notes(validated)
        if saved:
            await self.db.prune_expired_preference_notes(uk_date)
        return saved

    async def _load_recent_fills_by_account(self) -> tuple[dict[str, list[Any]], set[str]]:
        """Best-effort T212 recent fills per account; failures degrade to "not checked" (None)."""
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:  # noqa: BLE001 - settlement must never fail on this lookup
            logger.debug("Settlement recent-fill portfolio fetch failed: %s", exc)
            return {}, set()
        fills: dict[str, list[Any]] = {}
        available: set[str] = set()
        for acc in (portfolio.isa, portfolio.invest):
            if not getattr(acc, "ok", False) or not getattr(acc, "order_history_supported", False):
                continue
            label = str(getattr(acc, "account_label", "") or "").strip()
            if not label:
                continue
            fills[label] = list(getattr(acc, "recent_orders", None) or [])
            available.add(label)
        return fills, available

    async def _run_daily_settlement_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        # Read-only fill ledger: matching real fills never rewrites wallets, only annotates the pick.
        fills_by_account, fills_available = await self._load_recent_fills_by_account()
        pending = await self.db.get_pending_settlements(limit=50, only_with_entry_price=False)
        followups = (
            await self.db.get_settled_recommendations_for_followup(limit=50)
            if hasattr(self.db, "get_settled_recommendations_for_followup")
            else []
        )

        skipped = 0
        not_due = 0
        settled = 0
        not_filled = 0
        failed = 0
        followup_updated = 0
        mature_items: list[tuple[dict[str, Any], date, date, str, int, float, list[str]]] = []
        followup_items: list[tuple[dict[str, Any], date, str, float, list[str], date | None, date | None]] = []

        for rec in pending:
            rec_id = int(rec["id"])
            symbol = str(rec.get("symbol") or "").strip().upper()
            entry_raw = rec.get("entry_price")
            bench_candidates = _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))

            try:
                entry_val = float(entry_raw) if entry_raw is not None else 0.0
            except (TypeError, ValueError):
                entry_val = 0.0

            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=bench_candidates[0] if bench_candidates else "SPY",
                    settled_at_uk=uk_now_str,
                    raw_return_pct=None,
                    alpha_pct=None,
                    outcome_status="SKIPPED",
                    reflection_lesson="",
                )
                skipped += 1
                continue

            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            holding_days = max(1, int(rec.get("horizon_days") or self.cfg.strategy.settlement_holding_days or 5))
            target_date = add_trading_days(rec_date, holding_days, market=market)
            if not has_market_closed_on_date(target_date, market=market, now_utc=base):
                not_due += 1
                continue

            mature_items.append((rec, rec_date, target_date, market, holding_days, entry_val, bench_candidates))

        for rec in followups:
            symbol = str(rec.get("symbol") or "").strip().upper()
            try:
                entry_val = float(rec.get("entry_price") or 0.0)
            except (TypeError, ValueError):
                entry_val = 0.0
            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                continue
            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            bench_candidates = _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))

            t10_due_date: date | None = None
            if rec.get("t10_alpha_pct") is None:
                cand_t10 = add_trading_days(rec_date, 10, market=market)
                if has_market_closed_on_date(cand_t10, market=market, now_utc=base):
                    t10_due_date = cand_t10

            t20_due_date: date | None = None
            if rec.get("t20_alpha_pct") is None:
                cand_t20 = add_trading_days(rec_date, 20, market=market)
                if has_market_closed_on_date(cand_t20, market=market, now_utc=base):
                    t20_due_date = cand_t20

            if t10_due_date is not None or t20_due_date is not None:
                followup_items.append(
                    (rec, rec_date, market, entry_val, bench_candidates, t10_due_date, t20_due_date)
                )

        if mature_items or followup_items:
            fetch_symbols = list(
                dict.fromkeys(
                    [str(rec["symbol"]).strip().upper() for rec, *_ in mature_items]
                    + [b for *_, b_list in mature_items for b in b_list]
                    + [str(rec["symbol"]).strip().upper() for rec, *_ in followup_items]
                    + [b for _, _, _, _, b_list, _, _ in followup_items for b in b_list]
                )
            )
            try:
                histories = await self.market_data.get_histories(fetch_symbols, period="1y")
            except Exception as exc:
                logger.warning("Daily settlement history fetch failed: %s", exc)
                histories = {}
                failed += len(mature_items)
                mature_items = []
                followup_items = []

            ohlc_cache: dict[tuple[str, str], pd.DataFrame] = {}

            def _get_ohlc(sym: str, mkt: str) -> pd.DataFrame:
                key = (sym, mkt)
                if key not in ohlc_cache:
                    ohlc_cache[key] = _extract_closed_daily_ohlc(histories.get(sym), mkt, base)
                return ohlc_cache[key]

            def _get_closes(sym: str, mkt: str) -> pd.Series:
                ohlc = _get_ohlc(sym, mkt)
                return ohlc["Close"] if not ohlc.empty else pd.Series(dtype=float)

            for rec, rec_date, target_date, market, holding_days, entry_price, bench_candidates in mature_items:
                rec_id = int(rec["id"])
                symbol = str(rec["symbol"]).strip().upper()
                stock_ohlc = _get_ohlc(symbol, market)
                stock_closes = stock_ohlc["Close"] if not stock_ohlc.empty else pd.Series(dtype=float)
                exit_dates = [d for d in stock_closes.index if d >= target_date]
                if not exit_dates:
                    not_due += 1
                    continue
                exit_date = exit_dates[0]
                exit_price = float(stock_closes.loc[exit_date])

                eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, exit_price)
                if eff_entry is None:
                    logger.warning(
                        "Skipping settlement for %s (id=%s) due to unresolved GBp/GBP 100x mismatch",
                        symbol,
                        rec_id,
                    )
                    failed += 1
                    continue

                action = str(rec.get("action") or "BUY").strip().upper()
                direction = _trade_direction(action)
                # A recommendation generated after that session closed (e.g. Friday 21:30
                # /pick) cannot have been filled at the same day's low; start next session.
                exclude_rec_date = False
                rec_ms = int(rec.get("created_at_ms") or 0)
                if rec_ms > 0:
                    rec_dt_utc = datetime.fromtimestamp(rec_ms / 1000.0, tz=timezone.utc)
                    exclude_rec_date = has_market_closed_on_date(
                        rec_date, market=market, now_utc=rec_dt_utc
                    )
                fill_status = (
                    "ASSUMED"
                    if str(rec.get("source") or "").strip().lower() == "report"
                    else _entry_fill_status(
                        action,
                        stock_ohlc,
                        rec_date,
                        target_date,
                        eff_entry,
                        exclude_rec_date=exclude_rec_date,
                    )
                )
                account_label = str(rec.get("account") or "").strip()
                account_key = (
                    account_label
                    if account_label in fills_by_account
                    else _normalize_account_label(account_label)
                )
                actual_filled: int | None = None
                actual_fill_price: float | None = None
                execution_slippage_pct: float | None = None
                if account_key in fills_available:
                    matched_fill = _match_recommendation_fill(
                        rec,
                        start_date=rec_date,
                        end_date=exit_date,
                        recent_orders=fills_by_account.get(account_key) or [],
                        entry_price=eff_entry,
                    )
                    if matched_fill is not None:
                        actual_filled = 1
                        actual_fill_price = round(float(matched_fill[0]), 4)
                        execution_slippage_pct = round(float(matched_fill[1]), 3)
                        fill_status = "FILLED"
                    else:
                        actual_filled = 0
                if fill_status == "NOT_FILLED":
                    saved = await self.db.settle_recommendation(
                        rec_id,
                        benchmark_symbol=bench_candidates[0] if bench_candidates else "SPY",
                        settled_at_uk=uk_now_str,
                        raw_return_pct=None,
                        alpha_pct=None,
                        outcome_status="NOT_FILLED",
                        reflection_lesson="",
                        fill_status="NOT_FILLED",
                        actual_filled=actual_filled,
                        actual_fill_price=actual_fill_price,
                        execution_slippage_pct=execution_slippage_pct,
                    )
                    if saved:
                        not_filled += 1
                    continue

                unadjusted_stock_ret = ((exit_price - eff_entry) / eff_entry) * 100.0
                raw_return_pct = unadjusted_stock_ret * direction
                bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, exit_date)
                if bench_res is None:
                    logger.warning("Benchmark history unavailable for settling %s (%s -> %s)", symbol, rec_date, exit_date)
                    failed += 1
                    continue
                chosen_bench, bench_ret_pct = bench_res
                # direction=-1 => benchmark outperformance of an avoided loser is positive alpha.
                alpha_pct = _direction_adjusted_alpha_pct(action, eff_entry, exit_price, bench_ret_pct)
                mfe_pct, mae_pct = _compute_mfe_mae_pct(
                    stock_ohlc, rec_date, exit_date, eff_entry, direction=direction
                )

                # v5.5.3 dual-philosophy layering: an ISA value BUY/ADD at T+5 is noise, not a lesson.
                # Keep the T+5 outcome numeric only; the real post-mortem lands on the T+20 follow-up.
                isa_value_entry = (
                    _normalize_account_label(account_label) == "ISA"
                    and action in {"BUY", "ADD"}
                    and holding_days <= 5
                )
                if isa_value_entry:
                    lesson = ""
                else:
                    try:
                        lesson = await self._generate_settlement_reflection(
                            rec=rec,
                            rec_date=rec_date,
                            exit_date=exit_date,
                            holding_days=holding_days,
                            entry_price=eff_entry,
                            exit_price=exit_price,
                            raw_return_pct=raw_return_pct,
                            benchmark_symbol=chosen_bench,
                            bench_return_pct=bench_ret_pct,
                            alpha_pct=alpha_pct,
                            direction=direction,
                        )
                    except Exception as exc:
                        logger.warning("Settlement reflection failed for %s (id=%s): %s", symbol, rec_id, exc)
                        failed += 1
                        continue

                saved = await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=chosen_bench,
                    settled_at_uk=uk_now_str,
                    raw_return_pct=round(raw_return_pct, 2),
                    alpha_pct=round(alpha_pct, 2),
                    outcome_status="SETTLED",
                    reflection_lesson=lesson,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                    fill_status=fill_status,
                    actual_filled=actual_filled,
                    actual_fill_price=actual_fill_price,
                    execution_slippage_pct=execution_slippage_pct,
                )
                if saved:
                    settled += 1

            for rec, rec_date, market, entry_price, bench_candidates, t10_target, t20_target in followup_items:
                rec_id = int(rec["id"])
                symbol = str(rec["symbol"]).strip().upper()
                stock_ohlc = _get_ohlc(symbol, market)
                if stock_ohlc.empty:
                    continue
                stock_closes = stock_ohlc["Close"]
                action = str(rec.get("action") or "BUY").strip().upper()
                direction = _trade_direction(action)

                t10_ret: float | None = None
                t10_alpha: float | None = None
                t20_ret: float | None = None
                t20_alpha: float | None = None
                furthest_exit: date | None = None
                t20_ctx: tuple[str, float, float, date, float] | None = None

                for horizon_label, target_d in (("t10", t10_target), ("t20", t20_target)):
                    if target_d is None:
                        continue
                    ex_dates = [d for d in stock_closes.index if d >= target_d]
                    if not ex_dates:
                        continue
                    ex_date = ex_dates[0]
                    ex_price = float(stock_closes.loc[ex_date])
                    eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, ex_price)
                    if eff_entry is None:
                        continue
                    raw_ret = _direction_adjusted_return_pct(action, eff_entry, ex_price)
                    bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, ex_date)
                    if bench_res is None:
                        continue
                    b_sym, b_ret = bench_res
                    alpha_val = _direction_adjusted_alpha_pct(action, eff_entry, ex_price, b_ret)
                    if horizon_label == "t10":
                        t10_ret = round(raw_ret, 2)
                        t10_alpha = round(alpha_val, 2)
                    else:
                        t20_ret = round(raw_ret, 2)
                        t20_alpha = round(alpha_val, 2)
                        t20_ctx = (b_sym, b_ret, ex_price, ex_date, eff_entry)
                    if furthest_exit is None or ex_date > furthest_exit:
                        furthest_exit = ex_date

                if t10_alpha is None and t20_alpha is None:
                    continue

                mfe_new: float | None = None
                mae_new: float | None = None
                if furthest_exit is not None:
                    ref_exit_price = float(stock_closes.loc[furthest_exit])
                    eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, ref_exit_price)
                    if eff_entry is not None:
                        mfe_new, mae_new = _compute_mfe_mae_pct(
                            stock_ohlc, rec_date, furthest_exit, eff_entry, direction=direction
                        )
                        prev_mfe = rec.get("mfe_pct")
                        prev_mae = rec.get("mae_pct")
                        if mfe_new is not None and prev_mfe is not None:
                            mfe_new = max(float(prev_mfe), mfe_new)
                        if mae_new is not None and prev_mae is not None:
                            mae_new = min(float(prev_mae), mae_new)

                # T+20 补写：ISA 价值仓 T+5 刻意不写短期反思，在这里用 20 日真实表现补一条。
                t20_lesson: str | None = None
                if (
                    t20_alpha is not None
                    and t20_ctx is not None
                    and not str(rec.get("reflection_lesson") or "").strip()
                ):
                    b_sym, b_ret, t20_exit_price, t20_exit_date, t20_eff_entry = t20_ctx
                    try:
                        t20_lesson = await self._generate_settlement_reflection(
                            rec=rec,
                            rec_date=rec_date,
                            exit_date=t20_exit_date,
                            holding_days=20,
                            entry_price=t20_eff_entry,
                            exit_price=t20_exit_price,
                            raw_return_pct=float(t20_ret or 0.0),
                            benchmark_symbol=b_sym,
                            bench_return_pct=b_ret,
                            alpha_pct=float(t20_alpha),
                            direction=direction,
                        )
                    except Exception as exc:
                        logger.warning("T+20 reflection failed for %s (id=%s): %s", symbol, rec_id, exc)
                        t20_lesson = None

                updated = await self.db.update_recommendation_followup(
                    rec_id,
                    t10_return_pct=t10_ret,
                    t10_alpha_pct=t10_alpha,
                    t20_return_pct=t20_ret,
                    t20_alpha_pct=t20_alpha,
                    mfe_pct=mfe_new,
                    mae_pct=mae_new,
                    reflection_lesson=t20_lesson,
                )
                if updated:
                    followup_updated += 1

        weekly_audit_ran = False
        if self._has_audit_universe() and hasattr(self.db, "get_latest_quant_audit"):
            iso_year, iso_week, _ = dt_uk.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            latest_audit = await self.db.get_latest_quant_audit()
            audit_stale = latest_audit is None
            if latest_audit is not None:
                try:
                    audit_stale = (dt_uk.date() - date.fromisoformat(str(latest_audit["audit_date"])[:10])).days >= 7
                except ValueError:
                    audit_stale = True
            if audit_stale and await self._claim_job_run("weekly_quant_audit_catchup", week_key, uk_now_str):
                try:
                    await self._run_weekly_quant_audit_impl(now_utc=base)
                    weekly_audit_ran = True
                except Exception as exc:
                    logger.warning("Weekday catch-up weekly quant audit failed: %s", exc)

        nav_refreshed = ""
        dt_uk_now = base.astimezone(LONDON_TZ)
        if dt_uk_now.hour * 60 + dt_uk_now.minute >= NAV_SNAPSHOT_START_MINUTES:
            try:
                nav_refreshed = await self._run_nav_snapshot_job()
            except Exception as exc:
                logger.warning("Post-close NAV catch-up failed: %s", exc)

        pruned = await self.db.prune_old_records(retention_days=90, now_utc=base)
        preferences_saved = 0
        if self.cfg.strategy.user_preference_memory_enabled:
            try:
                preferences_saved = await self._extract_preference_notes(base)
            except Exception as exc:
                logger.warning("Preference extraction failed: %s", exc)
        return {
            "checked": len(pending),
            "settled": settled,
            "skipped": skipped,
            "not_filled": not_filled,
            "not_due": not_due,
            "failed": failed,
            "followup_updated": followup_updated,
            "weekly_audit_ran": weekly_audit_ran,
            "nav_refreshed": nav_refreshed,
            "pruned": pruned,
            "preferences_saved": preferences_saved,
        }

    async def run_daily_settlement(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Settle matured T+N recommendations with Alpha + Hydrogen LOW reflection and prune 90d logs."""
        async with self.execution_lock:
            self.current_job_name = "daily_settlement"
            try:
                return await self._run_daily_settlement_impl(now_utc=now_utc)
            finally:
                if self.current_job_name == "daily_settlement":
                    self.current_job_name = None

    async def _run_nav_snapshot_job(self) -> str:
        """Persist one post-close GBP NAV row so the TWR curve samples EOD, not pre-market.

        Reuses the /report row when the day already exists (``uk_date`` is the primary key),
        so this simply refreshes it with closing values. Fallback portfolios are never
        written (iron rule 2).
        """
        portfolio = await self.t212.get_portfolio(use_cache=False)
        if portfolio is None or getattr(portfolio, "is_fallback", False):
            return "skipped: portfolio fallback"
        if float(getattr(portfolio, "total_combined_value", 0.0) or 0.0) <= 0.0:
            return "skipped: zero net value"

        bench_symbol = str(getattr(self.cfg.strategy, "benchmark_symbol", "VWRP.L") or "VWRP.L")
        bench_close: float | None = None
        try:
            hist_map = await self.market_data.get_histories([bench_symbol], period="1mo")
            bench_hist = hist_map.get(bench_symbol)
            if bench_hist is not None and not getattr(bench_hist, "empty", True):
                close_col = "Adj Close" if "Adj Close" in bench_hist.columns else "Close"
                if close_col in bench_hist.columns:
                    bench_close = float(bench_hist[close_col].dropna().iloc[-1])
        except Exception as exc:
            logger.debug("NAV snapshot benchmark close unavailable: %s", exc)

        row = build_nav_row(portfolio, benchmark_close=bench_close)
        await self.db.upsert_nav_history(row)
        logger.info("NAV snapshot persisted: %s £%.2f", row.get("uk_date"), row.get("total_value_gbp", 0.0))
        return f"nav {row.get('uk_date')} £{row.get('total_value_gbp', 0.0):,.2f}"

    async def _run_daily_settlement_job(self, now_utc: datetime) -> str:
        summary = await self._run_daily_settlement_impl(now_utc=now_utc)
        pruned_total = sum(summary["pruned"].values())
        if summary["settled"] == 0 and summary["failed"] > 0:
            return (
                f"⚠️ 盘后结算异常：待处理 {summary['checked']} 条中 {summary['failed']} 条失败，"
                f"清理 {pruned_total} 条过期记录"
            )
        return (
            f"结算 {summary['settled']} 条 / 追评 {summary.get('followup_updated', 0)} / "
            f"未成交 {summary.get('not_filled', 0)} / 跳过 {summary['skipped']} / "
            f"未到期 {summary['not_due']} / 失败 {summary['failed']} / 清理 {pruned_total}"
        )

    async def _run_prewarm_impl(self, scope: str, now_utc: datetime | None = None) -> dict[str, Any]:
        """Silently warm 2y histories, macro snapshot, and daily 1x T212 instruments/dividends archive."""
        base = now_utc or datetime.now(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")

        try:
            self.cfg.reload_pools_if_changed()
        except Exception as exc:
            logger.debug("Prewarm config hot-reload failed: %s", exc)

        inst_synced = 0
        if hasattr(self.t212, "fetch_instruments_metadata"):
            try:
                if await self._claim_job_run("t212_metadata_sync", uk_date, uk_now_str):
                    inst_items = await self.t212.fetch_instruments_metadata()
                    if inst_items and hasattr(self.db, "upsert_t212_instruments"):
                        inst_synced = await self.db.upsert_t212_instruments(inst_items)
                    else:
                        inst_synced = len(inst_items or [])
            except Exception as exc:
                logger.warning("Prewarm T212 metadata sync failed (%s): %s", scope, exc)

        portfolio = None
        try:
            portfolio = await self.t212.get_portfolio(use_cache=False)
        except Exception as exc:
            logger.warning("Prewarm portfolio fetch failed (%s): %s", scope, exc)

        div_archived = 0
        if (
            portfolio is not None
            and not getattr(portfolio, "is_fallback", False)
            and hasattr(self.db, "upsert_dividends")
        ):
            live_divs = [
                d.model_dump()
                for acc in (portfolio.isa, portfolio.invest)
                if acc.ok and not getattr(acc, "is_fallback", False)
                for d in acc.recent_dividends
            ]
            if live_divs:
                try:
                    div_archived = await self.db.upsert_dividends(live_divs)
                except Exception as exc:
                    logger.warning("Prewarm dividend archive failed (%s): %s", scope, exc)

        held_syms = portfolio.unique_yahoo_symbols(include_pending=True) if portfolio is not None else []
        active_pool_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_active_universe_pool"):
            try:
                active_pool_rows = await self.db.get_active_universe_pool()
            except Exception:
                active_pool_rows = []
        if active_pool_rows:
            pool_syms = (
                list(self.cfg.watchlist)
                + [str(r["symbol"]) for r in active_pool_rows]
                + list(self.cfg.universe.etf_symbols)
                + ["SPY", "QQQ", "ISF.L", "EXS1.DE"]
            )
        else:
            pool_syms = (
                list(self.cfg.watchlist)
                + list(self.cfg.universe.isa_long_term)
                + list(self.cfg.universe.invest_tactical)
                + list(self.cfg.universe.etf_symbols)
                + ["SPY", "QQQ", "ISF.L", "EXS1.DE"]
            )
        all_syms = list(dict.fromkeys(s for s in (held_syms + pool_syms) if s))

        hist_map: dict[str, pd.DataFrame] = {}
        if all_syms:
            try:
                hist_map = await self.market_data.get_histories(all_syms, period="2y")
            except Exception as exc:
                logger.warning("Prewarm histories fetch failed (%s): %s", scope, exc)

        macro_ok = False
        if hasattr(self.market_data, "get_macro_snapshot"):
            try:
                macro = await self.market_data.get_macro_snapshot()
                macro_ok = not getattr(macro, "is_stale", False)
            except Exception as exc:
                logger.warning("Prewarm macro snapshot failed (%s): %s", scope, exc)

        return {
            "scope": scope,
            "uk_date": uk_date,
            "symbols_warmed": sum(1 for df in hist_map.values() if df is not None and not df.empty),
            "instruments_synced": inst_synced,
            "dividends_archived": div_archived,
            "macro_ok": macro_ok,
        }

    async def run_prewarm(self, scope: str, now_utc: datetime | None = None) -> dict[str, Any]:
        """Acquire execution_lock and silently prewarm market caches & T212 archives."""
        job_key = f"prewarm_{scope}"
        async with self.execution_lock:
            self.current_job_name = job_key
            try:
                return await self._run_prewarm_impl(scope, now_utc=now_utc)
            finally:
                if self.current_job_name == job_key:
                    self.current_job_name = None

    async def _run_prewarm_job(self, scope: str, now_utc: datetime) -> str:
        summary = await self._run_prewarm_impl(scope, now_utc=now_utc)
        return (
            f"预热({scope}): 行情 {summary['symbols_warmed']} 只 | "
            f"元数据 {summary['instruments_synced']} 条 | 分红 {summary['dividends_archived']} 条"
        )

    def _intraday_scan_due(self, now_utc: datetime) -> bool:
        """Throttle the 0-token intraday scan to `intraday_poll_minutes`."""
        last = self._last_intraday_scan_utc
        if last is None:
            return True
        interval = timedelta(minutes=max(1, int(self.cfg.strategy.intraday_poll_minutes)))
        try:
            return (now_utc - last) >= interval
        except TypeError:
            return True

    async def _collect_order_fill_receipts(self, now_utc: datetime, *, uk_date: str) -> list[tuple[str, str]]:
        """Diff the 0-token pending-order snapshot and return [(dedupe_key, receipt_text)].

        Reuses the portfolio already fetched by the intraday sentinel (cache-backed) and stays
        silent whenever either the pending-order or the order-history API is unavailable.
        """
        if not bool(getattr(self.cfg.strategy, "intraday_fill_receipt_enabled", True)):
            return []
        get_snapshots = getattr(self.db, "get_pending_order_snapshots", None)
        upsert_snapshots = getattr(self.db, "upsert_pending_order_snapshots", None)
        delete_snapshots = getattr(self.db, "delete_pending_order_snapshots", None)
        if not (callable(get_snapshots) and callable(upsert_snapshots) and callable(delete_snapshots)):
            return []
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Fill receipt portfolio fetch failed: %s", exc)
            return []

        accounts = [acc for acc in (portfolio.isa, portfolio.invest) if getattr(acc, "ok", False)]
        supported = {
            str(acc.account_label or "").strip()
            for acc in accounts
            if getattr(acc, "pending_orders_supported", False)
        }
        history_ok = any(getattr(acc, "order_history_supported", False) for acc in accounts)
        if not supported or not history_ok:
            # Cannot cross-verify a fill -> stay silent instead of risking a false receipt.
            return []
        history_accounts = {
            str(acc.account_label or "").strip()
            for acc in accounts
            if getattr(acc, "order_history_supported", False)
        }

        live_rows = [
            _pending_order_snapshot_row(order)
            for acc in accounts
            if str(acc.account_label or "").strip() in supported
            for order in (getattr(acc, "pending_orders", None) or [])
        ]
        try:
            previous = await get_snapshots()
        except Exception as exc:
            logger.warning("Fill receipt snapshot load failed: %s", exc)
            return []

        diff = diff_pending_order_snapshots(previous, live_rows, supported_accounts=supported)
        history = [*portfolio.isa.recent_orders, *portfolio.invest.recent_orders]
        now_ms = int(now_utc.timestamp() * 1000)
        receipts: list[tuple[str, str]] = []
        stale_keys: list[tuple[str, str]] = []
        claimed_fill_ids: set[str] = set()
        # Read-only USD wallet anchor for the receipt hint; a lookup failure must never block receipts.
        usd_wallet_available: float | None = None
        get_wallet_balances = getattr(self.db, "get_wallet_balances", None)
        if callable(get_wallet_balances):
            try:
                balances = await get_wallet_balances("Invest")
                usd_row = (balances or {}).get("USD") or {}
                usd_wallet_available = float(usd_row.get("available_amount") or 0.0)
            except Exception as exc:  # noqa: BLE001 - degrade silently to no hint
                logger.debug("Fill receipt USD wallet lookup failed: %s", exc)
                usd_wallet_available = None

        for row in diff["partial"]:
            key = _snapshot_key(row)
            dedupe = f"{key[0]}:{key[1]}:p{float(row.get('filled_quantity') or 0.0):g}"
            if await self._is_job_locked(f"fill_receipt:{dedupe}", uk_date):
                continue
            match = match_fill_history(row, history, excluded_fill_ids=claimed_fill_ids) or {
                "matched_quantity": float(row.get("filled_quantity") or 0.0),
                "avg_price": 0.0,
                "last_filled_at_ms": now_ms,
                "currency": str(row.get("currency") or ""),
                "is_complete": False,
            }
            claimed_fill_ids.update(match.get("fill_ids") or [])
            receipts.append((dedupe, format_fill_receipt(row, match, partial=True, usd_wallet_available=usd_wallet_available)))

        for row in diff["vanished"]:
            key = _snapshot_key(row)
            dedupe = f"{key[0]}:{key[1]}"
            if key[0] not in history_accounts:
                # No order history for this account this round -> cannot verify, keep the row.
                continue
            stale_keys.append(key)
            if await self._is_job_locked(f"fill_receipt:{dedupe}", uk_date):
                continue
            match = match_fill_history(row, history, excluded_fill_ids=claimed_fill_ids)
            if match is None:
                logger.info(
                    "Pending order %s %s vanished without a matching fill (likely cancelled)",
                    key[0],
                    key[1],
                )
                continue
            claimed_fill_ids.update(match.get("fill_ids") or [])
            cross_day = _format_uk_from_ms(int(match.get("last_filled_at_ms") or 0))[:10] != uk_date
            receipts.append(
                (
                    dedupe,
                    format_fill_receipt(
                        row,
                        match,
                        partial=not bool(match.get("is_complete", True)),
                        cross_day=cross_day,
                        usd_wallet_available=usd_wallet_available,
                    ),
                )
            )

        try:
            if live_rows:
                await upsert_snapshots(
                    live_rows,
                    seen_at_ms=now_ms,
                    seen_at_uk=now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                )
            if stale_keys:
                await delete_snapshots(stale_keys)
        except Exception as exc:
            logger.warning("Fill receipt snapshot persist failed: %s", exc)
        return receipts

    async def _run_intraday_sentinel_job(
        self,
        now_utc: datetime,
        *,
        chat_id: str,
        uk_date: str,
    ) -> list[str]:
        """Run one intraday scan under the execution lock; returns symbols that were actually pushed."""
        async with self.execution_lock:
            prev_job = self.current_job_name
            self.current_job_name = "intraday_sentinel"
            self._last_intraday_scan_utc = now_utc
            try:
                receipts = await self._collect_order_fill_receipts(now_utc, uk_date=uk_date)
            except Exception:
                logger.exception("Fill receipt check crashed; skipping receipts this round")
                receipts = []
            try:
                alerts = await self._run_intraday_sentinel_impl(now_utc, uk_date=uk_date)
            except Exception:
                logger.exception("Intraday sentinel scan crashed; skipping this round")
                alerts = []
            finally:
                if self.current_job_name == "intraday_sentinel":
                    self.current_job_name = prev_job

        uk_now_str = now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        pushed: list[str] = []
        for dedupe_key, text in receipts:
            try:
                delivered = await self.tg.send_message(chat_id, text)
            except Exception as exc:  # noqa: BLE001 - delivery failure must not break the loop
                logger.warning("Fill receipt delivery failed for %s: %s", dedupe_key, exc)
                continue
            if delivered is False:
                logger.warning("Fill receipt for %s was not delivered; keeping the dedupe slot free.", dedupe_key)
                continue
            # Mark only after a successful push so a failed delivery never mutes the receipt.
            await self._mark_job_locked(f"fill_receipt:{dedupe_key}", uk_date, uk_now_str)
            pushed.append(f"FILL:{dedupe_key}")

        for sym, text in alerts:
            try:
                delivered = await self.tg.send_message(chat_id, text)
            except Exception as exc:  # noqa: BLE001 - delivery failure must not break the loop
                logger.warning("Intraday alert delivery failed for %s: %s", sym, exc)
                continue
            if delivered is False:
                logger.warning("Intraday alert for %s was not delivered; keeping today's dedupe slot free.", sym)
                continue
            # Mark only after a successful push so a quiet DS verdict never burns the daily dedupe slot.
            await self._mark_job_locked(
                f"intraday_alert:{sym}",
                uk_date,
                now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
            )
            pushed.append(sym)
        if pushed:
            self.last_job_runs["intraday_sentinel"] = {
                "status": "OK",
                "uk_time": now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                "detail": ",".join(pushed)[:120],
            }
        return pushed

    async def _run_intraday_sentinel_impl(
        self,
        now_utc: datetime,
        *,
        uk_date: str,
    ) -> list[tuple[str, str]]:
        """0-token scan of open-market holdings/orders; only triggered symbols go to grounding + DS."""
        base = now_utc.astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Intraday sentinel portfolio fetch failed: %s", exc)
            return []
        if not portfolio.isa.ok and not portfolio.invest.ok:
            return []

        pending_by_symbol: dict[str, list[Any]] = {}
        for account in (portfolio.isa, portfolio.invest):
            for order in getattr(account, "pending_orders", None) or []:
                sym = str(getattr(order, "yahoo_symbol", "") or "").strip().upper()
                if sym:
                    pending_by_symbol.setdefault(sym, []).append(order)

        held_symbols = {
            str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            for pos in portfolio.all_positions()
            if str(getattr(pos, "yahoo_symbol", "") or "").strip()
        }
        candidates = sorted(held_symbols | set(pending_by_symbol))
        if not candidates:
            return []

        us_sched = get_us_market_schedule_uk(base)
        london_window = dt_uk.weekday() < 5 and time(8, 0) <= dt_uk.time() < time(16, 30)
        uk_holiday = is_uk_bank_holiday(base)
        eu_holiday = is_eu_market_holiday(base)
        market_open = {
            "US": us_sched.phase == "OPEN",
            "UK": london_window and not uk_holiday,
            "XETRA": london_window and not eu_holiday,
            "EURONEXT": london_window and not eu_holiday,
        }
        open_symbols = [s for s in candidates if market_open.get(infer_market_from_symbol(s), False)]
        if not open_symbols:
            return []

        try:
            histories, bundles = await self.market_data.get_bundles(
                open_symbols,
                include_news=False,
                include_earnings=False,
                include_info=False,
            )
        except Exception as exc:
            logger.warning("Intraday sentinel bundle fetch failed: %s", exc)
            return []

        # Finnhub fast US quote overlay: `include_info=False` bundles only carry the daily-close, which
        # can be 15+ minutes stale (h_cache TTL) or entirely missing right after the US open.
        quotes: dict[str, dict[str, Any]] = {}
        us_open_symbols = [s for s in open_symbols if infer_market_from_symbol(s) == "US"]
        if us_open_symbols:
            quote_results = await asyncio.gather(
                *(self.market_data.get_us_realtime_quote(s) for s in us_open_symbols),
                return_exceptions=True,
            )
            for sym, result in zip(us_open_symbols, quote_results):
                if isinstance(result, dict):
                    quotes[sym] = result
                elif isinstance(result, Exception):
                    logger.debug("Intraday realtime quote overlay failed for %s: %s", sym, result)

        position_by_symbol: dict[str, Any] = {}
        positions_by_symbol: dict[str, list[Any]] = {}
        for pos in portfolio.all_positions():
            sym = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            if not sym:
                continue
            positions_by_symbol.setdefault(sym, []).append(pos)
            position_by_symbol.setdefault(sym, pos)

        strategy = self.cfg.strategy
        invest_move_threshold = float(getattr(strategy, "intraday_move_alert_pct", 3.5) or 3.5)
        isa_move_threshold = float(getattr(strategy, "intraday_isa_move_alert_pct", 5.0) or 5.0)
        activation_pct = float(getattr(strategy, "trailing_profit_activation_pct", 20.0) or 20.0)
        trailing_dd_pct = float(getattr(strategy, "trailing_drawdown_from_peak_pct", -12.0) or -12.0)
        if trailing_dd_pct > 0:
            trailing_dd_pct = -trailing_dd_pct
        core_drawdown_steps = sorted(
            {abs(float(step)) for step in (getattr(strategy, "core_etf_drawdown_steps_pct", None) or []) if step},
            reverse=True,
        )

        today_uk = dt_uk.date()
        triggered: list[tuple[float, str, float, float, float | None, str, str, str]] = []
        for sym in open_symbols:
            bundle = bundles.get(sym)
            quote = quotes.get(sym) or {}
            position = position_by_symbol.get(sym)
            # Live price precedence: Finnhub realtime -> T212 position mark -> daily-bar close.
            bundle_px = _to_float(getattr(bundle, "current_price", None))
            price = (
                _to_float(quote.get("current_price"))
                or _to_float(getattr(position, "cur_price", None))
                or bundle_px
            )
            if price is None:
                continue
            if sym.endswith(".L") and bundle_px is not None:
                price = _align_lse_unit_100x(price, bundle_px, sym)
            move_pct = _intraday_move_pct(
                histories.get(sym),
                price,
                today_uk=today_uk,
                prev_close=_to_float(quote.get("prev_close")),
                symbol=sym,
            )
            order_dist = _order_distance_pct(price, pending_by_symbol.get(sym) or [], symbol=sym)

            positions = positions_by_symbol.get(sym) or []
            invest_positions = [
                p for p in positions if _normalize_account_label(getattr(p, "account", "")) == "Invest"
            ]
            isa_positions = [
                p for p in positions if _normalize_account_label(getattr(p, "account", "")) == "ISA"
            ]
            # Cross-account double holding follows the more sensitive Invest discipline.
            account_ctx = "Invest" if invest_positions else ("ISA" if isa_positions else "")
            move_threshold = invest_move_threshold if account_ctx != "ISA" else isa_move_threshold
            hit_move = move_pct is not None and abs(move_pct) >= move_threshold
            hit_order = order_dist is not None and order_dist <= strategy.intraday_order_near_pct

            reasons: list[str] = []
            priority = abs(move_pct or 0.0)
            if hit_move:
                direction = "上涨" if float(move_pct or 0.0) > 0 else "下跌"
                reasons.append(
                    f"日内{direction} {abs(float(move_pct or 0.0)):.1f}%（{account_ctx or '挂单'}阈值 {move_threshold:.1f}%）"
                )
            if hit_order:
                reasons.append(f"距最近挂单价 {float(order_dist or 0.0):.1f}%")

            if invest_positions:
                invest_pos = invest_positions[0]
                avg_price = _to_float(getattr(invest_pos, "avg_price", None))
                loss_pct = _to_float(getattr(invest_pos, "return_pct", None))
                cost_note = ""
                if avg_price is not None and avg_price > 0:
                    aligned_px = _align_lse_unit_100x(price, avg_price, sym) if sym.endswith(".L") else price
                    loss_pct = (aligned_px / avg_price - 1.0) * 100.0
                    cost_note = f"，成本 {avg_price:,.2f}"
                naked_invest = not _has_protective_stop(pending_by_symbol.get(sym), account="Invest")
                if naked_invest and loss_pct is not None:
                    stop_line = _invest_adaptive_stop_pct(histories.get(sym), strategy)
                    if loss_pct <= stop_line:
                        reasons.append(
                            "裸露止损/移动止盈逼近：Invest 无 STOP/STOP_LIMIT 挂单，"
                            f"浮亏 {loss_pct:.1f}%{cost_note} 已跌破自适应止损线 {stop_line:.1f}%"
                        )
                        priority = max(priority, abs(loss_pct))
                    elif loss_pct >= activation_pct:
                        dd60 = _drawdown_from_high_pct(histories.get(sym), window=60)
                        if dd60 is not None and dd60 <= trailing_dd_pct:
                            reasons.append(
                                "裸露止损/移动止盈逼近：Invest 无 STOP/STOP_LIMIT 挂单，"
                                f"浮盈 {loss_pct:.1f}%{cost_note} 后自近 60 日高点回撤 {dd60:.1f}%"
                            )
                            priority = max(priority, abs(dd60))

            if account_ctx == "ISA" and core_drawdown_steps:
                if core_sleeve_for_symbol(sym, bundles) is not None:
                    dd52 = _drawdown_from_high_pct(histories.get(sym), window=252)
                    if dd52 is not None:
                        crossed_steps = [step for step in core_drawdown_steps if dd52 <= -step]
                        if crossed_steps:
                            reasons.append(
                                f"ISA 核心宽基距 52 周高点回撤 {dd52:.1f}%，已触及 {crossed_steps[0]:.0f}% 阶梯"
                            )
                            priority = max(priority, abs(dd52))

            if not reasons:
                continue
            currency = str(
                getattr(bundle, "currency", "") or getattr(position, "inst_currency", "") or ""
            )
            triggered.append(
                (
                    priority,
                    sym,
                    price,
                    float(move_pct or 0.0),
                    order_dist,
                    currency,
                    account_ctx,
                    "；".join(reasons),
                )
            )

        triggered.sort(key=lambda item: item[0], reverse=True)
        max_alerts = 2
        alerts: list[tuple[str, str]] = []
        for _, sym, price, move_pct, order_dist, currency, account_ctx, trigger_reason in triggered:
            if len(alerts) >= max_alerts:
                logger.info("Intraday sentinel capped at %d alerts for this round", max_alerts)
                break
            if await self._is_job_locked(f"intraday_alert:{sym}", uk_date):
                continue
            try:
                text = await generate_intraday_alert(
                    self.cfg,
                    self.market_data,
                    self.llm,
                    self.db,
                    symbol=sym,
                    current_price=price,
                    move_pct=move_pct,
                    order_dist_pct=order_dist,
                    pending_order_text=_format_pending_orders(pending_by_symbol.get(sym) or []),
                    currency=currency,
                    account_context=account_ctx,
                    trigger_reason=trigger_reason,
                    ops_notifier=self.ops,
                )
            except Exception as exc:  # noqa: BLE001 - one symbol must never break the scan
                logger.warning("Intraday alert generation failed for %s: %s", sym, exc)
                text = None
            if text:
                alerts.append((sym, text))
        return alerts

    async def check_and_run_due_jobs(self, now_utc: datetime | None = None) -> list[str]:
        """Evaluate prewarm, morning catch-up, premarket sentinel, weekly radar, Saturday quant audit, and post-close settlement."""
        base = now_utc or datetime.now(timezone.utc)
        chat_id = self.cfg.env.telegram_chat_id.strip()
        if not chat_id:
            return []

        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        uk_min_of_day = dt_uk.hour * 60 + dt_uk.minute
        fired: list[str] = []

        # Saturday idle window (10:00 ~ 22:00 UK): silent 0-token offline universe & threshold health audit
        if dt_uk.weekday() == 5:
            if WEEKLY_AUDIT_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES and self._has_audit_universe():
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_quant_audit", week_key, uk_now_str):
                    fired.append("weekly_quant_audit")
                    logger.info("Running Saturday silent weekly quant health audit (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_quant_audit",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_quant_audit_job(base),
                        send_to_tg=True,
                    )

            # Saturday ledger reminder window (12:00 ~ 22:00 UK): weekly Idempotent GBP+USD reconcile nudge
            if WEEKLY_WALLET_RECONCILE_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES:
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_wallet_reconcile", week_key, uk_now_str):
                    fired.append("weekly_wallet_reconcile")
                    logger.info("Sending weekly wallet reconcile reminder (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_wallet_reconcile",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_wallet_reconcile_job(base),
                    )
            return fired

        if not is_london_weekday(base):
            return []

        # 0. Silent Morning Prewarm (-10m before morning_report, e.g. 07:20 ~ 07:29 UK)
        morning_target = self.cfg.schedule.morning_report_hour * 60 + self.cfg.schedule.morning_report_minute
        morning_prewarm_target = max(0, morning_target - 10)
        if morning_prewarm_target <= uk_min_of_day < morning_target:
            if await self._claim_job_run("prewarm_morning", uk_date, uk_now_str):
                fired.append("prewarm_morning")
                logger.info("Running silent morning prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_morning",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("morning", base),
                    send_to_tg=False,
                )

        # 1. Morning Report with N95 Catch-up & Retry (07:30 ~ 09:30 UK; lock scheduler_runs only on success)
        morning_catchup_end = max(morning_target + 5, MORNING_CATCHUP_END_MINUTES)
        if morning_target <= uk_min_of_day < morning_catchup_end:
            if not await self._is_job_locked("morning", uk_date):
                retry_info = self._morning_failures.get(uk_date)
                if retry_info is None or base >= retry_info["next_retry_utc"]:
                    fired.append("morning")
                    logger.info("Running scheduled morning report (%s)", uk_date)
                    ok = await self._execute_job(
                        job_name="morning",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_morning_report(
                            self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                        ),
                        send_failure_to_tg=False,
                    )
                    if ok:
                        await self._mark_job_locked("morning", uk_date, uk_now_str)
                        self._morning_failures.pop(uk_date, None)
                    else:
                        attempts = int(retry_info["attempts"]) + 1 if retry_info else 1
                        self._morning_failures[uk_date] = {
                            "attempts": attempts,
                            "next_retry_utc": base + timedelta(seconds=self.morning_retry_cooldown_s),
                        }
        elif uk_min_of_day >= morning_catchup_end:
            if (
                uk_date in self._morning_failures
                and uk_date not in self._morning_missed_notified
                and not await self._is_job_locked("morning", uk_date)
            ):
                self._morning_missed_notified.add(uk_date)
                if self.ops is not None and hasattr(self.ops, "notify_business_critical"):
                    await self.ops.notify_business_critical(
                        source="scheduler:morning",
                        alert_key=f"scheduler_morning_catchup_exhausted:{uk_date}",
                        title="晨报自动补发窗口已结束",
                        message=f"{uk_date} 早报在 09:30 UK 前多次补发仍未成功，请尽快检查网络或接口状态。",
                        bot_message="🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )
                else:
                    await self.tg.send_message(
                        chat_id,
                        "🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )

        # 2. US Pre-Market Sentinel & -10m Silent Prewarm (Mon-Fri, dynamic DST-adjusted minutes_to_open)
        us_sched = get_us_market_schedule_uk(base)
        lead = self.cfg.schedule.premarket_lead_minutes
        if us_sched.is_us_trading_day and not us_sched.is_us_holiday and lead < us_sched.minutes_to_open <= lead + 10:
            if await self._claim_job_run("prewarm_premarket", uk_date, uk_now_str):
                fired.append("prewarm_premarket")
                logger.info("Running silent premarket prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_premarket",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("premarket", base),
                    send_to_tg=False,
                )

        if us_sched.minutes_to_open <= lead and us_sched.minutes_to_open > lead - 5:
            if await self._claim_job_run(
                "premarket", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=60
            ):
                fired.append("premarket")
                if us_sched.is_us_holiday:
                    logger.info("US Market Holiday (%s): sending quiet notice.", us_sched.ny_date)
                    self.last_job_runs["premarket"] = {
                        "status": "HOLIDAY_MUTED",
                        "uk_time": uk_now_str,
                        "detail": us_sched.ny_date,
                    }
                    await self.tg.send_message(
                        chat_id,
                        "🔔 今日为美国股市法定休市日（Market Holiday），美股盘前哨兵自动静音，祝你今天轻松愉快！",
                    )
                elif us_sched.is_us_trading_day:
                    logger.info("Running scheduled US pre-market sentinel (%s)", uk_date)
                    ok = await self._execute_job(
                        job_name="premarket",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_us_sentinel(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            is_scheduled_premarket=True,
                            ops_notifier=self.ops,
                        ),
                    )
                    if not ok:
                        await self._release_job_run(
                            "premarket", uk_date, now_utc=base, retry_cooldown_s=60
                        )

        # 3. Friday Post-Close Weekly Opportunity Radar & -10m Silent Prewarm (respects dynamic early-close)
        if self.cfg.schedule.enable_weekly_radar and dt_uk.weekday() == 4 and us_sched.is_us_trading_day:
            after = self.cfg.schedule.weekly_radar_after_close_minutes
            mins_after_close = -us_sched.minutes_to_close
            prewarm_after = max(0, after - 10)
            if prewarm_after <= mins_after_close < after:
                if await self._claim_job_run("prewarm_weekly_radar", uk_date, uk_now_str):
                    fired.append("prewarm_weekly_radar")
                    logger.info("Running silent weekly radar prewarm (%s)", uk_date)
                    await self._execute_job(
                        job_name="prewarm_weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_prewarm_job("weekly_radar", base),
                        send_to_tg=False,
                    )
            if after <= mins_after_close < after + 30:
                if await self._claim_job_run(
                    "weekly_radar", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=120
                ):
                    fired.append("weekly_radar")
                    logger.info("Running scheduled Friday weekly opportunity radar (%s)", uk_date)
                    ok = await self._execute_job(
                        job_name="weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_pick_radar(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            focus_query="",
                            is_weekly_auto=True,
                            ops_notifier=self.ops,
                        ),
                    )
                    if not ok:
                        await self._release_job_run(
                            "weekly_radar", uk_date, now_utc=base, retry_cooldown_s=120
                        )

        # 3b. Evening EOD NAV snapshot (21:40~22:00 UK, weekdays): one post-close NAV row
        if (
            NAV_SNAPSHOT_START_MINUTES <= uk_min_of_day < NAV_SNAPSHOT_END_MINUTES
            and dt_uk.weekday() < 5
        ):
            if await self._claim_job_run(
                "nav_snapshot", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=60
            ):
                fired.append("nav_snapshot")
                logger.info("Running post-close NAV snapshot (%s)", uk_date)
                ok = await self._execute_job(
                    job_name="nav_snapshot",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_nav_snapshot_job(),
                    send_to_tg=False,
                )
                if not ok:
                    await self._release_job_run(
                        "nav_snapshot", uk_date, now_utc=base, retry_cooldown_s=60
                    )

        # 4. Daily Post-Close T+N Recommendation Settlement & 90-Day Pruning (22:00 ~ 23:30 UK)
        if DAILY_SETTLEMENT_START_MINUTES <= uk_min_of_day < DAILY_SETTLEMENT_END_MINUTES:
            if await self._claim_job_run("daily_settlement", uk_date, uk_now_str):
                fired.append("daily_settlement")
                logger.info("Running scheduled daily T+N settlement & DB pruning (%s)", uk_date)
                await self._execute_job(
                    job_name="daily_settlement",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_daily_settlement_job(base),
                    send_to_tg=False,
                )

        # 5. Intraday event-driven sentinel (08:00~21:00 UK; 0-token scan, never queues behind user commands)
        if (
            self.cfg.strategy.intraday_sentinel_enabled
            and INTRADAY_SENTINEL_START_MINUTES <= uk_min_of_day < INTRADAY_SENTINEL_END_MINUTES
            and not self.execution_lock.locked()
            and self._intraday_scan_due(base)
        ):
            pushed = await self._run_intraday_sentinel_job(base, chat_id=chat_id, uk_date=uk_date)
            if pushed:
                fired.append("intraday_sentinel")

        return fired

    async def run_forever(self, poll_interval_s: float = 30.0) -> None:
        logger.info("TradingScheduler started (Europe/London timezone aware).")
        while True:
            try:
                await self.check_and_run_due_jobs()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job loop error")
                if self.ops is not None:
                    await self.ops.notify(
                        severity="CRITICAL",
                        source="scheduler:loop",
                        alert_key="scheduler_loop_error",
                        message=f"调度主循环异常：{type(exc).__name__}: {exc}",
                    )
            await asyncio.sleep(poll_interval_s)
