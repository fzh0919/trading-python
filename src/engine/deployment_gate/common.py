"""Shared constants and pure helper utilities for the deployment gate."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.data_providers.t212_client import (
    AccountSnapshot,
    PendingOrderItem,
    PositionItem,
)
from src.engine.portfolio_rules import _estimate_pending_order_gbp

LONDON_TZ = ZoneInfo("Europe/London")

# Regime constants (design doc §5.2: RISK_ON requires HY OAS < 4.5%; RISK_OFF at >= 5.5%)
REGIME_HY_OAS_RISK_ON_PCT = 4.5
REGIME_HY_OAS_RISK_OFF_PCT_DEFAULT = 5.5
# ISA extreme low-cash defence line (design doc §6.1 D trigger)
ISA_EXTREME_LOW_CASH_PCT = 2.0
# A-trigger exclusion windows (calendar days)
TRIGGER_EARNINGS_EXCLUDE_DAYS = 5
TRIGGER_EXDIV_EXCLUDE_DAYS = 5
TRIGGER_RSI_EXCLUDE_ABOVE = 72.0
# Consecutive-loss: how far a sell may trail the matching BUY fill (ms)
GATE_EPOCH_PARSE_FAIL_NOTE = "gate_epoch_date 解析失败，按当日零点保守起算"

PENDING_PROTECTIVE_SELL_STOP = "PROTECTIVE_SELL_STOP"
PENDING_CORE_ETF_BUY = "CORE_ETF_BUY"
PENDING_NEW_EQUITY_BUY = "NEW_EQUITY_BUY"

_GATE_VERSION = "v5.4.0"
_BANNER_MAX_CHARS = 110


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def _sym_up(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _is_held_symbol(pos_by_sym: dict[str, Any], symbol: str) -> bool:
    """True when a symbol (or its base ticker) already exists in the given account book."""
    return _find_position(pos_by_sym, symbol) is not None


def _find_position(pos_by_sym: dict[str, Any], symbol: str) -> Any | None:
    """Find a position by exact Yahoo symbol or base ticker (case-insensitive)."""
    sym = _sym_up(symbol)
    if not sym:
        return None
    if symbol in pos_by_sym:
        return pos_by_sym[symbol]
    base = sym.split(".")[0]
    for key, pos in pos_by_sym.items():
        key_up = _sym_up(key)
        if key_up == sym or key_up.split(".")[0] == base:
            return pos
    return None


def _position_opened_today(pos: Any, today_start_ms: int) -> bool:
    """Best-effort check that a holding was opened today (for ADD vs NEW classification)."""
    created_ms = _parse_order_ts_ms(getattr(pos, "created_at", None))
    return created_ms is not None and created_ms >= today_start_ms


def _symbol_matches(a: str, b: str) -> bool:
    sa, sb = _sym_up(a), _sym_up(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    return sa.split(".")[0] == sb.split(".")[0]


def _round2(value: float | None, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _account_value_gbp(acct: AccountSnapshot) -> float:
    val = float(getattr(acct, "total_account_value", 0.0) or 0.0)
    if val > 0:
        return val
    pos_sum = sum(float(p.current_value) for p in (getattr(acct, "positions", None) or []))
    return max(pos_sum + float(getattr(acct, "total_cash", 0.0) or 0.0), 0.0)


def _parse_uk_midnight_ms(date_str: str) -> int:
    """Parse YYYY-MM-DD at UK 00:00 into epoch ms (Europe/London aware)."""
    try:
        dt = datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").replace(tzinfo=LONDON_TZ)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return int(datetime.now(LONDON_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _parse_order_ts_ms(value: Any) -> int | None:
    """Tolerant parser for pending-order created_at strings / ms ints."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _close_series(hist: pd.DataFrame | pd.Series | None) -> pd.Series:
    if isinstance(hist, pd.Series):
        closes = pd.to_numeric(hist, errors="coerce").dropna().astype(float)
        return closes[closes > 0]
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
    return closes[closes > 0]


def _native_risk_to_gbp(native_amount: float, pos: PositionItem) -> float:
    """Convert a native-currency risk amount into GBP using the position's own FX basis."""
    if native_amount <= 0:
        return 0.0
    # Keep the original casing: ``.upper()`` would turn LSE pence ("GBp"/"GBx") into "GBP"
    # and silently inflate the heat by 100x.
    raw_ccy = str(pos.inst_currency or "").strip()
    if raw_ccy in {"GBp", "GBx"} or raw_ccy.upper() == "GBX":
        return native_amount / 100.0
    if raw_ccy.upper() == "GBP":
        return native_amount
    if pos.current_value > 0 and pos.cur_price > 0 and pos.quantity > 0:
        gbp_per_native_unit = float(pos.current_value) / (float(pos.cur_price) * float(pos.quantity))
        return native_amount * gbp_per_native_unit
    return 0.0



def _pending_buy_gbp(
    orders: list[PendingOrderItem],
    pos_by_sym: dict[str, PositionItem],
) -> float:
    return sum(_estimate_pending_order_gbp(o, pos_by_symbol=pos_by_sym) for o in orders)


def _pending_core_etf_gbp(
    order: PendingOrderItem,
    pos_by_sym: dict[str, PositionItem],
) -> float:
    """GBP value of a pending core ETF BUY, anchored on the actual order notional.

    ``_estimate_pending_order_gbp`` scales by the held position ratio, which understates
    a standalone core DCA order (e.g. 25 VWRP shares vs a 100-share holding). Core policy
    maths must use the real committed cash, i.e. limit price x quantity.
    """
    value = float(order.value or 0.0)
    ccy = str(order.currency or "").strip()
    if value > 0:
        if ccy in {"GBp", "GBx"} or ccy.upper() == "GBX":
            return value / 100.0
        if ccy.upper() in {"GBP", ""}:
            return value
    return _estimate_pending_order_gbp(order, pos_by_symbol=pos_by_sym)
