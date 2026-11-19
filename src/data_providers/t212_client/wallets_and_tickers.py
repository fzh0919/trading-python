"""Invest dual-wallet reconciliation, T212->Yahoo ticker normalization, and HTTP header parsers."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from src.data_providers.t212_client.models import AccountSnapshot

LONDON_TZ = ZoneInfo("Europe/London")

_MANUAL_WALLET_SOURCES = {"manual", "manual_both", "user", "user_reported"}


def _wallet_amount(record: dict[str, Any] | None, key: str) -> float:
    try:
        return float((record or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pending_usd_buy_notional(pending_orders: list[Any] | None) -> float:
    """Sum the USD notional frozen by active BUY orders (limit > stop > API value)."""
    total = 0.0
    for order in pending_orders or []:
        if str(getattr(order, "side", "") or "").strip().upper() != "BUY":
            continue
        if str(getattr(order, "currency", "") or "").strip().upper() != "USD":
            continue
        qty = max(0.0, float(getattr(order, "quantity", 0.0) or 0.0))
        if qty <= 0:
            continue
        limit_price = float(getattr(order, "limit_price", 0.0) or 0.0)
        stop_price = float(getattr(order, "stop_price", 0.0) or 0.0)
        if limit_price > 0:
            notional = qty * limit_price
        elif stop_price > 0:
            notional = qty * stop_price
        else:
            notional = abs(float(getattr(order, "value", 0.0) or 0.0))
        total += max(0.0, notional)
    return total


def reconcile_invest_wallets(
    invest_snap: AccountSnapshot,
    wallet_records: dict[str, dict[str, Any]] | None,
    gbp_usd: float | None,
    *,
    mismatch_warn_gbp: float = 10.0,
    mismatch_warn_pct: float = 2.0,
    reconcile_warn_days: int = 7,
    now_ms: int | None = None,
) -> AccountSnapshot:
    """Attach an Invest GBP+USD wallet split without touching GBP accounting totals.

    Uses the locally stored USD anchor and derives GBP = API total cash - USD/GBPUSD.
    A user-reported GBP anchor (source manual*, at least as fresh as USD) is shown verbatim
    together with a mismatch warning when |GBP + USD/rate - API total cash| exceeds the
    larger of ``mismatch_warn_gbp`` and ``mismatch_warn_pct`` of total cash.
    """
    records = {str(k).strip().upper(): (v or {}) for k, v in (wallet_records or {}).items()}
    usd_rec = records.get("USD", {})
    usd_avail = max(0.0, _wallet_amount(usd_rec, "available_amount"))
    usd_reserved = max(0.0, _wallet_amount(usd_rec, "reserved_amount"))
    usd_total = usd_avail + usd_reserved
    usd_source = str(usd_rec.get("source") or "").strip().lower()
    usd_updated_uk = str(usd_rec.get("updated_at_uk") or "")
    usd_updated_ms = int(_wallet_amount(usd_rec, "updated_at_ms"))

    updates: dict[str, Any] = {
        "multi_currency_enabled": bool(invest_snap.ok),
        "usd_cash_total": round(usd_total, 2),
        "usd_available_to_trade": round(usd_avail, 2),
        "usd_reserved_cash": round(usd_reserved, 2),
        "wallet_updated_at_uk": usd_updated_uk,
    }
    if not invest_snap.ok:
        return invest_snap.model_copy(update=updates)

    warnings: list[str] = []
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    if reconcile_warn_days > 0 and usd_updated_ms > 0:
        anchor_days = (now - usd_updated_ms) / 86_400_000
        if anchor_days >= reconcile_warn_days:
            if usd_source in {"", "init", "default"}:
                warnings.append(
                    f"USD 钱包仍为默认 0 USD（{usd_updated_uk or '未知时间'} 起未校准），若持有 USD 请核对"
                )
            else:
                warnings.append(
                    f"USD 钱包锚自 {usd_updated_uk or '未知时间'} 起已 {int(anchor_days)} 天未校准"
                )

    try:
        rate = float(gbp_usd) if gbp_usd is not None else 0.0
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0.0:
        updates.update(
            {
                "gbp_wallet_total": None,
                "gbp_wallet_available": None,
                "gbp_wallet_reserved": None,
                "wallet_source": "pending_fx",
                "wallet_reconcile_warning": "；".join([*warnings, "缺少 GBP/USD 汇率，GBP 钱包待折算"]),
            }
        )
        return invest_snap.model_copy(update=updates)

    updates["gbp_usd_rate"] = round(rate, 4)

    # v5.5.2: active USD BUY orders freeze the USD wallet first. The API only exposes the
    # account-level reserved cash (GBP), so derive the USD slice with a triple clamp and
    # auto-release it as soon as the order disappears (cancel / fill).
    usd_available_effective = usd_avail
    if invest_snap.pending_orders_supported:
        pending_usd_buy = _pending_usd_buy_notional(getattr(invest_snap, "pending_orders", None))
        api_reserved_usd = max(0.0, float(invest_snap.reserved_cash or 0.0)) * rate
        if pending_usd_buy <= 0.0 or float(invest_snap.reserved_cash or 0.0) <= 0.01:
            auto_usd_reserved = 0.0
        else:
            auto_usd_reserved = min(usd_total, pending_usd_buy, api_reserved_usd)
        auto_usd_reserved = round(max(0.0, auto_usd_reserved), 2)
        usd_available_effective = round(max(0.0, usd_total - auto_usd_reserved), 2)
        updates["usd_reserved_cash"] = auto_usd_reserved
        updates["usd_available_to_trade"] = usd_available_effective

    gbp_rec = records.get("GBP", {})
    gbp_source = str(gbp_rec.get("source") or "").strip().lower()
    gbp_updated_ms = int(_wallet_amount(gbp_rec, "updated_at_ms"))
    gbp_is_manual = gbp_source in _MANUAL_WALLET_SOURCES and gbp_updated_ms >= usd_updated_ms

    if gbp_is_manual:
        gbp_avail = max(0.0, _wallet_amount(gbp_rec, "available_amount"))
        gbp_reserved = max(0.0, _wallet_amount(gbp_rec, "reserved_amount"))
        gbp_total = gbp_avail + gbp_reserved
        reported_total = gbp_total + usd_total / rate
        diff = reported_total - invest_snap.total_cash
        threshold = max(float(mismatch_warn_gbp), abs(invest_snap.total_cash) * float(mismatch_warn_pct) / 100.0)
        if abs(diff) > threshold:
            warnings.append(
                f"双报合计 £{reported_total:,.2f} 与 API 总现金 £{invest_snap.total_cash:,.2f} "
                f"偏差 £{diff:+,.2f}（告警阈值 £{threshold:,.2f}），请核对钱包余额"
            )
        updates.update(
            {
                "gbp_wallet_total": round(gbp_total, 2),
                "gbp_wallet_available": round(gbp_avail, 2),
                "gbp_wallet_reserved": round(gbp_reserved, 2),
                "wallet_source": "manual_both",
            }
        )
    else:
        derived_total = invest_snap.total_cash - usd_total / rate
        derived_avail = invest_snap.available_to_trade - usd_available_effective / rate
        derived_reserved = invest_snap.reserved_cash - float(updates["usd_reserved_cash"]) / rate
        if min(derived_total, derived_avail, derived_reserved) < -0.01:
            warnings.append("按 USD 锚推算的 GBP 钱包为负，USD 账本可能已因成交/换汇变化，请校准")
        updates.update(
            {
                "gbp_wallet_total": round(max(0.0, derived_total), 2),
                "gbp_wallet_available": round(max(0.0, derived_avail), 2),
                "gbp_wallet_reserved": round(max(0.0, derived_reserved), 2),
                "wallet_source": "derived_from_total",
            }
        )

    updates["wallet_reconcile_warning"] = "；".join(warnings)
    return invest_snap.model_copy(update=updates)


_T212_EXCHANGE_LETTER_TO_YAHOO: dict[str, str] = {
    "l": ".L",   # London Stock Exchange (e.g. VUAGl_EQ, VWRPl_EQ, IBTAl_EQ)
    "a": ".AS",  # Euronext Amsterdam (e.g. ASMLa_EQ)
    "d": ".DE",  # Deutsche Boerse / Xetra
    "p": ".PA",  # Euronext Paris
    "m": ".MI",  # Borsa Italiana
    "e": ".MC",  # Bolsa de Madrid
    "s": ".SW",  # SIX Swiss Exchange
    "b": ".BR",  # Euronext Brussels
    "v": ".VI",  # Vienna Stock Exchange
    "f": ".F",   # Frankfurt
}


def _split_t212_ticker(raw_ticker: str) -> tuple[str, str | None]:
    """Extract (clean_symbol, yahoo_exchange_suffix) from T212 instrument ticker."""
    if not raw_ticker:
        return "UNKNOWN", None
    s = str(raw_ticker).strip()

    # T212 EU/UK tickers commonly embed a lowercase exchange letter right before _EQ,
    # e.g. VUAGl_EQ -> (VUAG, .L), IBTAl_EQ -> (IBTA, .L), ASMLa_EQ -> (ASML, .AS)
    m = re.match(r"^([A-Z0-9][A-Z0-9._-]*?)([a-z])_EQ$", s)
    if m:
        return m.group(1), _T212_EXCHANGE_LETTER_TO_YAHOO.get(m.group(2))

    stripped = re.sub(r"_(?:[a-z0-9]{1,4}_)?EQ$", "", s, flags=re.IGNORECASE)
    if stripped != s:
        return stripped, None
    return re.sub(r"_[a-z0-9]+$", "", s, flags=re.IGNORECASE), None


_LEGACY_US_TICKER_ALIASES: dict[str, str] = {
    "FB": "META",
}
_NON_US_T212_SUFFIXES = ("_LN_EQ", "_UK_EQ", "_DE_EQ", "_FR_EQ", "_NL_EQ", "_CH_EQ")


def _normalize_legacy_us_ticker(
    raw_ticker: str,
    clean: str,
    inline_suffix: str | None = None,
    inst_currency: str = "",
) -> str:
    """Map legacy US ticker symbols (e.g. T212 FB_US_EQ -> META) without altering non-US symbols."""
    clean_up = (clean or "").strip().upper()
    alias = _LEGACY_US_TICKER_ALIASES.get(clean_up)
    if not alias or inline_suffix is not None:
        return clean
    raw_up = (raw_ticker or "").strip().upper()
    if raw_up.endswith(_NON_US_T212_SUFFIXES):
        return clean
    curr_up = (inst_currency or "").strip().upper()
    if raw_up.endswith("_US_EQ") or curr_up in {"", "USD"}:
        return alias
    return clean


def clean_t212_ticker(raw_ticker: str) -> str:
    """Normalize Trading 212 internal instrument codes like AAPL_US_EQ, VUAGl_EQ, ASMLa_EQ, BRK_B_US_EQ."""
    clean, inline_suffix = _split_t212_ticker(raw_ticker)
    return _normalize_legacy_us_ticker(raw_ticker, clean, inline_suffix)


def to_yahoo_symbol(raw_ticker: str, inst_currency: str = "USD", clean_ticker: str | None = None) -> str:
    """Convert Trading 212 ticker + currency metadata to Yahoo Finance ticker format."""
    parsed_clean, inline_suffix = _split_t212_ticker(raw_ticker)
    raw_upper = (raw_ticker or "").strip().upper()
    resolved_clean = _normalize_legacy_us_ticker(
        raw_ticker,
        clean_ticker or parsed_clean,
        inline_suffix,
        inst_currency,
    )
    clean = resolved_clean.strip().upper()
    curr = (inst_currency or "").strip()
    curr_upper = curr.upper()

    # LSE single-letter or dotted tickers (e.g. RR., BT.A) -> strip trailing dot, hyphenate class share
    yahoo_base = clean[:-1] if clean.endswith(".") else clean
    if "_" in yahoo_base:
        yahoo_base = yahoo_base.replace("_", "-")

    if inline_suffix:
        return yahoo_base if yahoo_base.endswith(inline_suffix) else f"{yahoo_base}{inline_suffix}"
    if raw_upper.endswith("_US_EQ"):
        return yahoo_base
    if raw_upper.endswith("_LN_EQ") or raw_upper.endswith("_UK_EQ"):
        return yahoo_base if yahoo_base.endswith(".L") else f"{yahoo_base}.L"
    if raw_upper.endswith("_DE_EQ"):
        return f"{yahoo_base}.DE"
    if raw_upper.endswith("_FR_EQ"):
        return f"{yahoo_base}.PA"
    if raw_upper.endswith("_NL_EQ"):
        return f"{yahoo_base}.AS"
    if raw_upper.endswith("_CH_EQ"):
        return f"{yahoo_base}.SW"

    # Generic _EQ suffix fallback based on currency
    if curr_upper in {"GBX", "GBP"} or curr == "GBp":
        return yahoo_base if yahoo_base.endswith(".L") else f"{yahoo_base}.L"
    if curr_upper == "EUR" and not any(yahoo_base.endswith(sfx) for sfx in (".DE", ".PA", ".AS", ".MI", ".MC")):
        return f"{yahoo_base}.DE"
    return yahoo_base


def _format_ms_to_uk(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(LONDON_TZ)
    return dt.strftime("%m-%d %H:%M")


def _parse_iso_ms(ts_str: str) -> int:
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return 0


def _is_transient_http_status(status_code: int) -> bool:
    return status_code in {408, 429} or (500 <= status_code < 600)


def _parse_ratelimit_remaining(headers: httpx.Headers) -> int | None:
    raw = headers.get("x-ratelimit-remaining")
    if raw is None:
        return None
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def _parse_ratelimit_reset_ts(headers: httpx.Headers) -> float | None:
    raw = headers.get("x-ratelimit-reset")
    if raw is None:
        return None
    try:
        ts = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    # Guard against millisecond Unix timestamps if any gateway emits ms
    if ts > 1e12:
        ts /= 1000.0
    return ts
