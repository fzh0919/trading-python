"""Intraday move/stop-loss math, pending-order snapshot diffing, fill-history matching, and receipt formatting."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from src.bot.scheduler.settlement_math import LONDON_TZ, _align_lse_unit_100x
from src.data_providers.analytics import _compute_atr_and_adx

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
