"""Async Trading 212 API v0 client for ISA & Invest accounts (read-only)."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


class PositionItem(BaseModel):
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str
    quantity: float
    avg_price: float
    cur_price: float
    current_value: float
    total_cost: float
    pnl: float
    fx_impact: float
    return_pct: float
    weight_pct: float = 0.0
    created_at: str | None = None
    isin: str | None = None
    cumulative_dividends_gbp: float = 0.0
    total_return_with_div_pct: float | None = None


class PendingOrderItem(BaseModel):
    id: str
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float = 0.0
    status: str = ""
    value: float
    limit_price: float | None = None
    stop_price: float | None = None
    currency: str
    time_in_force: str = ""
    created_at: str = ""


class HistoricalOrderItem(BaseModel):
    id: str
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str
    side: str
    order_type: str
    filled_at: str
    filled_at_ms: int
    uk_time_str: str
    quantity: float
    price: float
    net_value: float
    wallet_currency: str
    realised_pnl: float
    tax_total: float


class HistoricalTransactionItem(BaseModel):
    id: str
    account: str
    tx_type: str  # DEPOSIT | WITHDRAW | TRANSFER | FEE | INTEREST_ON_FREE_CASH | LENDING_INTEREST
    amount: float
    currency: str = "GBP"
    date_time: str
    date_time_ms: int
    uk_time_str: str


class HistoricalDividendItem(BaseModel):
    id: str
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str = ""
    dividend_type: str = ""
    amount: float
    currency: str = "GBP"
    gross_per_share: float = 0.0
    quantity: float = 0.0
    paid_on: str
    paid_on_ms: int
    uk_time_str: str


class AccountSnapshot(BaseModel):
    ok: bool = False
    error: str = ""
    account_label: str = ""
    account_currency: str = "GBP"
    total_account_value: float = 0.0
    investments_value: float = 0.0
    investments_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float | None = None
    total_cash: float = 0.0
    available_to_trade: float = 0.0
    reserved_cash: float = 0.0
    in_pies_cash: float = 0.0
    cash_ratio_pct: float = 0.0
    total_fx_impact: float = 0.0
    positions: list[PositionItem] = Field(default_factory=list)
    pending_orders: list[PendingOrderItem] = Field(default_factory=list)
    recent_orders: list[HistoricalOrderItem] = Field(default_factory=list)
    recent_transactions: list[HistoricalTransactionItem] = Field(default_factory=list)
    recent_dividends: list[HistoricalDividendItem] = Field(default_factory=list)
    pending_orders_supported: bool = False
    pending_orders_error: str = ""
    order_history_supported: bool = False
    transactions_supported: bool = False
    dividends_supported: bool = False
    cash_status_note: str = ""
    is_fallback: bool = False
    fallback_source: str = ""
    fallback_reason: str = ""
    snapshot_fetched_at_uk: str = ""


class PortfolioData(BaseModel):
    fetched_at_uk: str
    fetched_at_ms: int
    isa: AccountSnapshot
    invest: AccountSnapshot
    total_combined_value: float = 0.0
    total_combined_cash: float = 0.0
    cash_interest_rate_pct: float = 4.0
    is_fallback: bool = False
    fallback_source: str = ""
    fallback_note: str = ""

    def all_positions(self) -> list[PositionItem]:
        return [*self.isa.positions, *self.invest.positions]

    def all_pending_orders(self) -> list[PendingOrderItem]:
        return [*self.isa.pending_orders, *self.invest.pending_orders]

    def unique_yahoo_symbols(self, include_pending: bool = True) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.all_positions():
            seen.setdefault(p.yahoo_symbol, None)
        if include_pending:
            for o in self.all_pending_orders():
                if o.yahoo_symbol:
                    seen.setdefault(o.yahoo_symbol, None)
        return list(seen.keys())


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


class T212Client:
    """Async client for Trading 212 API v0 with connection pooling, rate-limit header pacing, and fallback."""

    def __init__(
        self,
        base_url: str = "https://live.trading212.com/api/v0",
        isa_key: str = "",
        isa_secret: str = "",
        invest_key: str = "",
        invest_secret: str = "",
        order_history_days: int = 30,
        order_history_max_pages: int = 3,
        cash_interest_rate_pct: float = 4.0,
        cache_ttl_seconds: int = 120,
        transport: httpx.AsyncBaseTransport | None = None,
        ops_notifier: Any = None,
        snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None = None,
        transient_backoff_s: float = 0.5,
        db: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.isa_key = isa_key.strip()
        self.isa_secret = isa_secret.strip()
        self.invest_key = invest_key.strip()
        self.invest_secret = invest_secret.strip()
        self.order_history_days = order_history_days
        self.order_history_max_pages = order_history_max_pages
        self.cash_interest_rate_pct = cash_interest_rate_pct
        self.cache_ttl_seconds = cache_ttl_seconds
        self._transport = transport
        self._ops = ops_notifier
        self._snapshot_loader = snapshot_loader
        self._transient_backoff_s = max(0.0, float(transient_backoff_s))
        self._db = db
        self._client: httpx.AsyncClient | None = None
        self._portfolio_lock = asyncio.Lock()
        self._cache: tuple[float, PortfolioData] | None = None
        self._last_good_portfolio: PortfolioData | None = None
        self._last_good_accounts: dict[str, tuple[str, int, AccountSnapshot]] = {}
        self._rate_limit_resets: dict[tuple[str, str], float] = {}
        self._instrument_map: dict[str, dict[str, Any]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def set_ops_notifier(self, ops_notifier: Any) -> None:
        self._ops = ops_notifier

    def set_snapshot_loader(self, snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None) -> None:
        self._snapshot_loader = snapshot_loader

    def set_db(self, db: Any) -> None:
        self._db = db

    def _resolve_ticker_metadata(
        self,
        raw_ticker: str,
        inst: dict[str, Any],
        default_currency: str,
        extra_currency: str = "",
    ) -> tuple[str, str, str, str | None]:
        """Resolve (clean_ticker, inst_currency, yahoo_symbol, isin) using cached T212 instrument metadata."""
        clean = clean_t212_ticker(raw_ticker)
        meta = self._instrument_map.get(raw_ticker) or {}
        inst_curr = str(
            inst.get("currency")
            or meta.get("currency_code")
            or ("USD" if raw_ticker.upper().endswith("_US_EQ") else "")
            or extra_currency
            or default_currency
        )
        yahoo_sym = str(meta.get("yahoo_symbol") or "").strip() or to_yahoo_symbol(raw_ticker, inst_curr, clean)
        raw_isin = inst.get("isin") or meta.get("isin") or ""
        isin = str(raw_isin).strip().upper() or None
        return clean, inst_curr, yahoo_sym, isin

    async def fetch_instruments_metadata(self, *, account_scope: str | None = None) -> list[dict[str, Any]]:
        """Fetch /equity/metadata/instruments (1 req / 50s bucket) and sync to SQLite archive if available."""
        scope = str(account_scope or "").strip().upper() or None
        if scope == "ISA":
            if not (self.isa_key and self.isa_secret):
                return []
            api_key, api_secret = self.isa_key, self.isa_secret
        elif scope == "INVEST":
            if not (self.invest_key and self.invest_secret):
                return []
            api_key, api_secret = self.invest_key, self.invest_secret
        else:
            if self.isa_key and self.isa_secret:
                api_key, api_secret = self.isa_key, self.isa_secret
            elif self.invest_key and self.invest_secret:
                api_key, api_secret = self.invest_key, self.invest_secret
            else:
                return []

        client = self._get_client()
        headers = self._auth_headers(api_key, api_secret)
        url = f"{self.base_url}/equity/metadata/instruments"
        try:
            resp = await self._request_with_retry(
                client,
                url,
                headers,
                default_retry_sleep_s=50.5,
                max_retries=2,
                rate_bucket="metadata_instruments",
                max_rate_wait_s=55.0,
            )
        except httpx.HTTPError as exc:
            logger.warning("T212 /equity/metadata/instruments failed: %s", exc)
            return []

        remaining = _parse_ratelimit_remaining(resp.headers)
        reset_ts = _parse_ratelimit_reset_ts(resp.headers)
        if remaining is not None and remaining <= 0 and reset_ts is None:
            self._rate_limit_resets[("metadata_instruments", urlparse(url).path)] = time.time() + 50.0
        elif scope is not None:
            # Proactively enforce T212 1 req / 50s cooldown before any subsequent account-scope call
            self._rate_limit_resets[("metadata_instruments", urlparse(url).path)] = time.time() + 52.0

        if resp.status_code != 200:
            logger.warning("T212 /equity/metadata/instruments returned HTTP %d", resp.status_code)
            return []

        raw_items = resp.json()
        if not isinstance(raw_items, list):
            return []

        now_ms = int(time.time() * 1000)
        parsed_items: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("ticker") or "").strip()
            if not ticker:
                continue
            currency_code = str(raw.get("currencyCode") or raw.get("currency") or "").strip()
            short_name = str(raw.get("shortName") or raw.get("name") or "").strip()
            isin = str(raw.get("isin") or "").strip().upper()
            inst_type = str(raw.get("type") or "").strip().upper()
            extended_hours = bool(raw.get("extendedHours", False))
            raw_sched = raw.get("workingScheduleId")
            working_schedule_id: int | None = None
            if raw_sched is not None and str(raw_sched).strip() != "":
                try:
                    working_schedule_id = int(raw_sched)
                except (TypeError, ValueError):
                    working_schedule_id = None
            clean_hint = short_name if re.fullmatch(r"[A-Za-z0-9._-]{1,10}", short_name) else None
            yahoo_sym = to_yahoo_symbol(ticker, currency_code or "USD", clean_ticker=clean_hint)
            item = {
                "ticker": ticker,
                "yahoo_symbol": yahoo_sym,
                "isin": isin,
                "currency_code": currency_code,
                "short_name": short_name,
                "inst_type": inst_type,
                "extended_hours": extended_hours,
                "working_schedule_id": working_schedule_id,
                "updated_at_ms": now_ms,
            }
            self._instrument_map[ticker] = item
            parsed_items.append(item)

        if parsed_items and self._db is not None and hasattr(self._db, "upsert_t212_instruments"):
            try:
                if scope is not None:
                    await self._db.upsert_t212_instruments(
                        parsed_items,
                        account_scope=scope,
                        full_snapshot=True,
                    )
                else:
                    await self._db.upsert_t212_instruments(parsed_items)
            except Exception as exc:
                logger.warning("Failed persisting T212 instrument metadata: %s", exc)
        return parsed_items

    async def sync_all_accounts_instruments_metadata(self) -> dict[str, int]:
        """Sync both ISA & Invest instrument lists sequentially, respecting T212's 1 req / 50s bucket."""
        isa_count = 0
        invest_count = 0
        if self.isa_key and self.isa_secret:
            isa_items = await self.fetch_instruments_metadata(account_scope="ISA")
            isa_count = len(isa_items)
        if self.invest_key and self.invest_secret:
            invest_items = await self.fetch_instruments_metadata(account_scope="INVEST")
            invest_count = len(invest_items)
        return {"isa_count": isa_count, "invest_count": invest_count}

    async def _notify_ops(
        self,
        severity: str,
        alert_key: str,
        message: str,
        *,
        title: str = "",
        business_blocking: bool = False,
        bot_message: str = "",
    ) -> None:
        if self._ops is None:
            return
        if business_blocking and hasattr(self._ops, "notify_business_critical"):
            await self._ops.notify_business_critical(
                source="t212",
                alert_key=alert_key,
                message=message,
                title=title,
                bot_message=bot_message,
            )
            return
        await self._ops.notify(severity=severity, source="t212", alert_key=alert_key, message=message)

    @staticmethod
    def _auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
        token = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("ascii")
        return {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        }

    @staticmethod
    def _compute_rate_limit_sleep(
        headers: httpx.Headers,
        default_sleep_s: float,
        max_wait_s: float = 15.0,
    ) -> float:
        reset_ts = _parse_ratelimit_reset_ts(headers)
        if reset_ts is not None:
            delta = reset_ts - time.time()
            if delta > 0:
                return min(max(delta + 0.05, 0.05), max_wait_s)
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                val = float(str(retry_after).strip())
                if val > 0:
                    return min(max(val, 0.05), max_wait_s)
            except ValueError:
                pass
        return min(max(default_sleep_s, 0.05), max_wait_s)

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        default_retry_sleep_s: float = 3.0,
        max_retries: int = 2,
        rate_bucket: str = "default",
        max_rate_wait_s: float = 15.0,
        transient_backoff_s: float | None = None,
    ) -> httpx.Response:
        backoff_base = self._transient_backoff_s if transient_backoff_s is None else transient_backoff_s
        endpoint_key = (rate_bucket, urlparse(url).path)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            pending_reset = self._rate_limit_resets.get(endpoint_key)
            if pending_reset is not None:
                wait_s = pending_reset - time.time()
                if wait_s > 0:
                    proactive_sleep = min(wait_s + 0.05, max_rate_wait_s)
                    logger.debug("T212 rate-limit pacing on %s (%s), sleeping %.2fs", url, rate_bucket, proactive_sleep)
                    await asyncio.sleep(proactive_sleep)
                self._rate_limit_resets.pop(endpoint_key, None)

            try:
                resp = await client.get(url, headers=headers, timeout=20.0)
                remaining = _parse_ratelimit_remaining(resp.headers)
                reset_ts = _parse_ratelimit_reset_ts(resp.headers)
                if remaining is not None and remaining <= 0 and reset_ts is not None:
                    self._rate_limit_resets[endpoint_key] = reset_ts
                elif resp.status_code == 200 and (remaining is None or remaining > 0):
                    self._rate_limit_resets.pop(endpoint_key, None)

                if resp.status_code == 429:
                    if attempt == max_retries:
                        return resp
                    sleep_s = self._compute_rate_limit_sleep(resp.headers, default_retry_sleep_s, max_wait_s=max_rate_wait_s)
                    self._rate_limit_resets.pop(endpoint_key, None)
                    logger.warning("T212 429 rate-limited on %s, waiting %.2fs", url, sleep_s)
                    await asyncio.sleep(sleep_s)
                    continue

                if resp.status_code == 408 or (500 <= resp.status_code < 600):
                    if attempt == max_retries:
                        return resp
                    backoff = min(backoff_base * (attempt + 1), 3.0)
                    logger.warning("T212 HTTP %d on %s, retrying in %.2fs", resp.status_code, url, backoff)
                    await asyncio.sleep(backoff)
                    continue

                return resp
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == max_retries:
                    raise
                await asyncio.sleep(min(backoff_base * (attempt + 1), 3.0))
        assert last_exc is not None
        raise last_exc

    async def _fetch_order_data(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        account_label: str,
        account_currency: str,
    ) -> tuple[list[PendingOrderItem], bool, str, list[HistoricalOrderItem], bool]:
        pending_orders: list[PendingOrderItem] = []
        pending_supported = False
        pending_error = ""
        recent_orders: list[HistoricalOrderItem] = []
        history_supported = False

        # 1. Active pending orders (/equity/orders)
        try:
            active_res = await self._request_with_retry(
                client,
                f"{self.base_url}/equity/orders",
                headers,
                default_retry_sleep_s=5.2,
                rate_bucket=account_label,
            )
            if active_res.status_code == 200:
                pending_supported = True
                raw_pending = active_res.json()
                if isinstance(raw_pending, list):
                    for o in raw_pending:
                        inst = o.get("instrument") or {}
                        raw_ticker = str(o.get("ticker") or inst.get("ticker") or "UNKNOWN")
                        # Note: T212 Order.currency is often the account settlement currency (e.g. GBP)
                        # whereas limitPrice/stopPrice are quoted in the instrument's trading currency.
                        clean, inst_curr, yahoo_sym, _ = self._resolve_ticker_metadata(
                            raw_ticker,
                            inst,
                            account_currency,
                            extra_currency=str(o.get("currency") or ""),
                        )
                        signed_qty = float(o.get("quantity") or 0)
                        raw_qty = abs(signed_qty)
                        filled_qty = abs(float(o.get("filledQuantity") or 0))
                        remaining_qty = max(raw_qty - filled_qty, 0.0) if (raw_qty > 0 and filled_qty > 0) else raw_qty
                        raw_val = abs(float(o.get("value") or 0))
                        filled_val = abs(float(o.get("filledValue") or 0))
                        remaining_val = max(raw_val - filled_val, 0.0) if (raw_val > 0 and filled_val > 0) else raw_val
                        order_status = str(o.get("status") or "")
                        side = str(o.get("side") or ("SELL" if signed_qty < 0 else "BUY"))
                        pending_orders.append(
                            PendingOrderItem(
                                id=str(o.get("id") or ""),
                                account=account_label,
                                raw_ticker=raw_ticker,
                                ticker=clean,
                                yahoo_symbol=yahoo_sym,
                                name=str(inst.get("name") or clean),
                                side=side,
                                order_type=str(o.get("type") or "LIMIT"),
                                quantity=remaining_qty,
                                filled_quantity=filled_qty,
                                status=order_status,
                                value=remaining_val,
                                limit_price=float(o["limitPrice"]) if o.get("limitPrice") is not None else None,
                                stop_price=float(o["stopPrice"]) if o.get("stopPrice") is not None else None,
                                currency=inst_curr,
                                time_in_force=str(o.get("timeInForce") or ""),
                                created_at=str(o.get("createdAt") or ""),
                            )
                        )
            elif active_res.status_code == 403:
                logger.info("%s /equity/orders returned 403 (Orders - Read disabled).", account_label)
                await self._notify_ops(
                    "WARN",
                    f"t212_orders_403:{account_label}",
                    f"Trading 212 [{account_label}] 活动挂单接口返回 403（可能缺少 Orders - Read 权限或被风控限制）。",
                )
            else:
                pending_error = f"HTTP {active_res.status_code}"
                logger.warning("%s /equity/orders HTTP %d", account_label, active_res.status_code)
                await self._notify_ops(
                    "WARN",
                    f"t212_orders_http:{account_label}",
                    f"Trading 212 [{account_label}] 活动挂单接口返回 HTTP {active_res.status_code}。",
                )
        except Exception as exc:
            pending_error = type(exc).__name__
            logger.warning("%s failed fetching pending orders: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_orders_exc:{account_label}",
                f"Trading 212 [{account_label}] 活动挂单拉取异常：{type(exc).__name__}: {exc}",
            )

        # 2. Historical filled orders (/equity/history/orders, 30d window)
        try:
            cutoff_ms = int(time.time() * 1000) - self.order_history_days * 86400 * 1000
            origin = re.sub(r"/api/v0/?$", "", self.base_url)
            next_url: str | None = f"{self.base_url}/equity/history/orders?limit=50"
            seen_ids: set[str] = set()

            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                hist_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if hist_res.status_code == 403:
                    logger.info("%s /equity/history/orders returned 403 (History - Orders disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_history_403:{account_label}",
                        f"Trading 212 [{account_label}] 历史成交接口返回 403（可能缺少 History - Orders 权限）。",
                    )
                    break
                if hist_res.status_code != 200:
                    logger.warning("%s /equity/history/orders page %d HTTP %d", account_label, page + 1, hist_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_history_http:{account_label}",
                        f"Trading 212 [{account_label}] 历史成交接口返回 HTTP {hist_res.status_code}。",
                    )
                    break

                history_supported = True
                body: dict[str, Any] = hist_res.json() or {}
                items = body.get("items") or []
                found_older = False

                for entry in items:
                    order = entry.get("order") or {}
                    fill = entry.get("fill") or None
                    if not fill or not fill.get("filledAt"):
                        continue
                    if fill.get("type") and fill.get("type") != "TRADE":
                        continue

                    raw_qty = fill.get("quantity") if fill.get("quantity") is not None else order.get("filledQuantity", 0)
                    qty = abs(float(raw_qty or 0))
                    if qty <= 0:
                        continue

                    filled_at_ms = _parse_iso_ms(str(fill["filledAt"]))
                    if filled_at_ms <= 0:
                        continue
                    if filled_at_ms < cutoff_ms:
                        found_older = True
                        continue

                    dedup_id = str(fill.get("id") or f"{order.get('id', '')}_{fill['filledAt']}")
                    if dedup_id in seen_ids:
                        continue
                    seen_ids.add(dedup_id)

                    inst = order.get("instrument") or {}
                    wallet = fill.get("walletImpact") or {}
                    raw_ticker = str(order.get("ticker") or inst.get("ticker") or "UNKNOWN")
                    wallet_currency = str(wallet.get("currency") or account_currency)
                    clean, inst_curr, yahoo_sym, _ = self._resolve_ticker_metadata(
                        raw_ticker,
                        inst,
                        wallet_currency,
                        extra_currency=str(order.get("currency") or ""),
                    )
                    signed_qty = fill.get("quantity") if fill.get("quantity") is not None else order.get("quantity", 0)
                    side = str(order.get("side") or ("SELL" if float(signed_qty or 0) < 0 else "BUY"))

                    tax_total = 0.0
                    for tax in (wallet.get("taxes") or []):
                        t_curr = tax.get("currency")
                        if not t_curr or t_curr == wallet_currency:
                            tax_total += abs(float(tax.get("quantity") or 0))

                    recent_orders.append(
                        HistoricalOrderItem(
                            id=dedup_id,
                            account=account_label,
                            raw_ticker=raw_ticker,
                            ticker=clean,
                            yahoo_symbol=yahoo_sym,
                            name=str(inst.get("name") or clean),
                            inst_currency=inst_curr,
                            side=side,
                            order_type=str(order.get("type") or "MARKET"),
                            filled_at=str(fill["filledAt"]),
                            filled_at_ms=filled_at_ms,
                            uk_time_str=_format_ms_to_uk(filled_at_ms),
                            quantity=qty,
                            price=float(fill.get("price") or 0),
                            net_value=abs(float(wallet.get("netValue") or 0)),
                            wallet_currency=wallet_currency,
                            realised_pnl=float(wallet.get("realisedProfitLoss") or 0),
                            tax_total=tax_total,
                        )
                    )

                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching order history: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_history_exc:{account_label}",
                f"Trading 212 [{account_label}] 历史成交拉取异常：{type(exc).__name__}: {exc}",
            )

        return pending_orders, pending_supported, pending_error, recent_orders, history_supported

    async def _fetch_cash_events(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        account_label: str,
        account_currency: str,
    ) -> tuple[list[HistoricalTransactionItem], bool, list[HistoricalDividendItem], bool]:
        """Fetch 30-day external deposits/withdrawals/interest (/equity/history/transactions) and dividends (/equity/history/dividends)."""
        recent_transactions: list[HistoricalTransactionItem] = []
        tx_supported = False
        recent_dividends: list[HistoricalDividendItem] = []
        div_supported = False
        cutoff_ms = int(time.time() * 1000) - self.order_history_days * 86400 * 1000
        origin = re.sub(r"/api/v0/?$", "", self.base_url)

        # 1. Transactions (/equity/history/transactions)
        try:
            next_url: str | None = f"{self.base_url}/equity/history/transactions?limit=50"
            seen_tx_ids: set[str] = set()
            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                tx_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if tx_res.status_code == 403:
                    logger.info("%s /equity/history/transactions returned 403 (History - Transactions disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_tx_403:{account_label}",
                        f"Trading 212 [{account_label}] 出入金流水接口返回 403（可能缺少 History - Transactions 权限）。",
                    )
                    break
                if tx_res.status_code == 404:
                    logger.debug("%s /equity/history/transactions returned 404; skipping.", account_label)
                    break
                if tx_res.status_code != 200:
                    logger.warning("%s /equity/history/transactions page %d HTTP %d", account_label, page + 1, tx_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_tx_http:{account_label}",
                        f"Trading 212 [{account_label}] 出入金流水接口返回 HTTP {tx_res.status_code}。",
                    )
                    break

                tx_supported = True
                body: dict[str, Any] = tx_res.json() or {}
                items = body.get("items") or []
                found_older = False
                for t in items:
                    dt_str = str(t.get("dateTime") or "")
                    dt_ms = _parse_iso_ms(dt_str)
                    if dt_ms <= 0:
                        continue
                    if dt_ms < cutoff_ms:
                        found_older = True
                        continue
                    tx_type = str(t.get("type") or "DEPOSIT").upper()
                    raw_amt = float(t.get("amount") or 0)
                    if tx_type in {"WITHDRAW", "FEE"}:
                        signed_amt = -abs(raw_amt)
                    elif tx_type in {"DEPOSIT", "INTEREST_ON_FREE_CASH", "LENDING_INTEREST"}:
                        signed_amt = abs(raw_amt)
                    else:
                        signed_amt = raw_amt
                    dedup_id = str(t.get("reference") or f"{tx_type}_{dt_str}_{signed_amt}")
                    if dedup_id in seen_tx_ids:
                        continue
                    seen_tx_ids.add(dedup_id)
                    recent_transactions.append(
                        HistoricalTransactionItem(
                            id=dedup_id,
                            account=account_label,
                            tx_type=tx_type,
                            amount=signed_amt,
                            currency=str(t.get("currency") or account_currency),
                            date_time=dt_str,
                            date_time_ms=dt_ms,
                            uk_time_str=_format_ms_to_uk(dt_ms),
                        )
                    )
                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching transactions: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_tx_exc:{account_label}",
                f"Trading 212 [{account_label}] 出入金流水拉取异常：{type(exc).__name__}: {exc}",
            )

        # 2. Dividends (/equity/history/dividends)
        try:
            next_url = f"{self.base_url}/equity/history/dividends?limit=50"
            seen_div_ids: set[str] = set()
            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                div_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if div_res.status_code == 403:
                    logger.info("%s /equity/history/dividends returned 403 (History - Dividends disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_div_403:{account_label}",
                        f"Trading 212 [{account_label}] 派息记录接口返回 403（可能缺少 History - Dividends 权限）。",
                    )
                    break
                if div_res.status_code == 404:
                    logger.debug("%s /equity/history/dividends returned 404; skipping.", account_label)
                    break
                if div_res.status_code != 200:
                    logger.warning("%s /equity/history/dividends page %d HTTP %d", account_label, page + 1, div_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_div_http:{account_label}",
                        f"Trading 212 [{account_label}] 派息记录接口返回 HTTP {div_res.status_code}。",
                    )
                    break

                div_supported = True
                body = div_res.json() or {}
                items = body.get("items") or []
                found_older = False
                for d in items:
                    paid_on = str(d.get("paidOn") or "")
                    paid_ms = _parse_iso_ms(paid_on)
                    if paid_ms <= 0:
                        continue
                    if paid_ms < cutoff_ms:
                        found_older = True
                        continue
                    inst = d.get("instrument") or {}
                    raw_ticker = str(d.get("ticker") or inst.get("ticker") or "UNKNOWN")
                    div_curr = str(d.get("currency") or account_currency)
                    meta = self._instrument_map.get(raw_ticker) or {}
                    inst_curr = str(d.get("tickerCurrency") or inst.get("currency") or meta.get("currency_code") or "")
                    clean = clean_t212_ticker(raw_ticker)
                    yahoo_sym = str(meta.get("yahoo_symbol") or "").strip() or to_yahoo_symbol(
                        raw_ticker, inst_curr or div_curr, clean
                    )
                    dedup_id = str(d.get("reference") or f"{raw_ticker}_{paid_on}")
                    if dedup_id in seen_div_ids:
                        continue
                    seen_div_ids.add(dedup_id)
                    recent_dividends.append(
                        HistoricalDividendItem(
                            id=dedup_id,
                            account=account_label,
                            raw_ticker=raw_ticker,
                            ticker=clean,
                            yahoo_symbol=yahoo_sym,
                            name=str(inst.get("name") or clean),
                            inst_currency=inst_curr,
                            dividend_type=str(d.get("type") or "DIVIDEND"),
                            amount=float(d.get("amount") or 0),
                            currency=div_curr,
                            gross_per_share=float(d.get("grossAmountPerShare") or 0),
                            quantity=float(d.get("quantity") or 0),
                            paid_on=paid_on,
                            paid_on_ms=paid_ms,
                            uk_time_str=_format_ms_to_uk(paid_ms),
                        )
                    )
                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching dividends: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_div_exc:{account_label}",
                f"Trading 212 [{account_label}] 派息记录拉取异常：{type(exc).__name__}: {exc}",
            )

        return recent_transactions, tx_supported, recent_dividends, div_supported

    async def fetch_single_account(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        api_secret: str,
        account_label: str,
    ) -> AccountSnapshot:
        snap, _ = await self._fetch_single_account_with_status(client, api_key, api_secret, account_label)
        return snap

    async def _fetch_single_account_with_status(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        api_secret: str,
        account_label: str,
    ) -> tuple[AccountSnapshot, bool]:
        """Fetch one account and return (snapshot, is_transient_error)."""
        if not api_key or not api_secret:
            return (
                AccountSnapshot(ok=False, account_label=account_label, error=f"{account_label} API credentials not configured."),
                False,
            )

        headers = self._auth_headers(api_key, api_secret)
        account_currency = "GBP"
        available_to_trade = 0.0
        in_pies = 0.0
        reserved = 0.0
        total_cash = 0.0
        cash_note = ""
        summary_inv_val: float | None = None
        summary_inv_cost: float | None = None
        summary_unrealized: float | None = None
        summary_realized: float | None = None
        summary_total_val: float | None = None
        summary_transient_err: str | None = None
        auth_alert_sent = False

        try:
            # 1. Account summary
            try:
                summary_res = await self._request_with_retry(
                    client,
                    f"{self.base_url}/equity/account/summary",
                    headers,
                    default_retry_sleep_s=5.2,
                    rate_bucket=account_label,
                )
                if summary_res.status_code == 200:
                    summary = summary_res.json() or {}
                    account_currency = str(summary.get("currency") or "GBP")
                    cash_obj = summary.get("cash") or {}
                    available_to_trade = float(cash_obj.get("availableToTrade") or 0)
                    in_pies = float(cash_obj.get("inPies") or 0)
                    reserved = float(cash_obj.get("reservedForOrders") or 0)
                    total_cash = available_to_trade + in_pies + reserved

                    inv = summary.get("investments") or {}
                    if inv.get("currentValue") is not None:
                        summary_inv_val = float(inv["currentValue"])
                    if inv.get("totalCost") is not None:
                        summary_inv_cost = float(inv["totalCost"])
                    if inv.get("unrealizedProfitLoss") is not None:
                        summary_unrealized = float(inv["unrealizedProfitLoss"])
                    if inv.get("realizedProfitLoss") is not None:
                        summary_realized = float(inv["realizedProfitLoss"])
                    if summary.get("totalValue") is not None:
                        summary_total_val = float(summary["totalValue"])
                else:
                    cash_note = f"(Account summary HTTP {summary_res.status_code})"
                    if summary_res.status_code in (401, 403):
                        auth_alert_sent = True
                        await self._notify_ops(
                            "CRITICAL",
                            f"t212_auth_failed:{account_label}",
                            f"Trading 212 [{account_label}] /equity/account/summary 返回 HTTP {summary_res.status_code}，API Key 鉴权失效。",
                            title=f"T212 {account_label} 鉴权失效",
                            business_blocking=True,
                            bot_message=f"🚨 Trading 212 [{account_label}] API Key 鉴权失效（HTTP {summary_res.status_code}），请及时更新只读密钥。",
                        )
                    if _is_transient_http_status(summary_res.status_code):
                        summary_transient_err = f"HTTP {summary_res.status_code}"
            except httpx.HTTPError as exc:
                summary_transient_err = f"{type(exc).__name__}: {exc}"
                cash_note = f"(Account summary {type(exc).__name__})"

            # 2. Positions
            pos_res = await self._request_with_retry(
                client,
                f"{self.base_url}/equity/positions",
                headers,
                default_retry_sleep_s=1.5,
                rate_bucket=account_label,
            )
            if pos_res.status_code != 200:
                if pos_res.status_code in (401, 403) and not auth_alert_sent:
                    await self._notify_ops(
                        "CRITICAL",
                        f"t212_auth_failed:{account_label}",
                        f"Trading 212 [{account_label}] /equity/positions 返回 HTTP {pos_res.status_code}，API Key 鉴权失效。",
                        title=f"T212 {account_label} 鉴权失效",
                        business_blocking=True,
                        bot_message=f"🚨 Trading 212 [{account_label}] API Key 鉴权失效（HTTP {pos_res.status_code}），今日风控与持仓同步已受阻。",
                    )
                return (
                    AccountSnapshot(
                        ok=False,
                        account_label=account_label,
                        error=f"{account_label} positions API HTTP {pos_res.status_code}: {pos_res.text[:200]}",
                    ),
                    _is_transient_http_status(pos_res.status_code),
                )

            raw_positions = pos_res.json() or []
            parsed_positions: list[PositionItem] = []
            calc_val = 0.0
            calc_cost = 0.0
            calc_pnl = 0.0
            total_fx_impact = 0.0

            for p in raw_positions:
                inst = p.get("instrument") or {}
                wallet = p.get("walletImpact") or {}
                raw_ticker = str(inst.get("ticker") or p.get("ticker") or "UNKNOWN")
                if wallet.get("currency"):
                    account_currency = str(wallet["currency"])
                clean, inst_curr, yahoo_sym, isin = self._resolve_ticker_metadata(
                    raw_ticker,
                    inst,
                    account_currency,
                )

                qty = float(p.get("quantity") or 0)
                avg_price = float(p.get("averagePricePaid") or 0)
                cur_price = float(p.get("currentPrice") or 0)

                gbx_scale = 0.01 if inst_curr in {"GBX", "GBp"} else 1.0
                current_value = (
                    float(wallet["currentValue"])
                    if wallet.get("currentValue") is not None
                    else cur_price * qty * gbx_scale
                )
                total_cost = (
                    float(wallet["totalCost"])
                    if wallet.get("totalCost") is not None
                    else avg_price * qty * gbx_scale
                )
                pnl = (
                    float(wallet["unrealizedProfitLoss"])
                    if wallet.get("unrealizedProfitLoss") is not None
                    else (current_value - total_cost)
                )
                fx_impact = float(wallet["fxImpact"]) if wallet.get("fxImpact") is not None else 0.0
                ret_pct = ((pnl / total_cost) * 100.0) if total_cost > 0 else 0.0

                raw_created_at = p.get("createdAt")
                created_at = str(raw_created_at).strip() if raw_created_at else None
                if created_at == "":
                    created_at = None

                calc_val += current_value
                calc_cost += total_cost
                calc_pnl += pnl
                total_fx_impact += fx_impact

                parsed_positions.append(
                    PositionItem(
                        account=account_label,
                        raw_ticker=raw_ticker,
                        ticker=clean,
                        yahoo_symbol=yahoo_sym,
                        name=str(inst.get("name") or clean),
                        inst_currency=inst_curr,
                        quantity=qty,
                        avg_price=avg_price,
                        cur_price=cur_price,
                        current_value=current_value,
                        total_cost=total_cost,
                        pnl=pnl,
                        fx_impact=fx_impact,
                        return_pct=ret_pct,
                        created_at=created_at,
                        isin=isin,
                    )
                )

            parsed_positions.sort(key=lambda x: x.current_value, reverse=True)

            # 3. Orders & Cash Events (Transactions / Dividends)
            pending_orders, pending_supported, pending_error, recent_orders, history_supported = await self._fetch_order_data(
                client, headers, account_label, account_currency
            )
            recent_transactions, tx_supported, recent_dividends, div_supported = await self._fetch_cash_events(
                client, headers, account_label, account_currency
            )

            # If summary had a transient error while positions succeeded, preserve cached cash if available
            if summary_transient_err is not None:
                mem_entry = self._last_good_accounts.get(account_label)
                if mem_entry is not None:
                    mem_uk, _, mem_acc = mem_entry
                    available_to_trade = mem_acc.available_to_trade
                    in_pies = mem_acc.in_pies_cash
                    reserved = mem_acc.reserved_cash
                    total_cash = mem_acc.total_cash
                    if summary_realized is None:
                        summary_realized = mem_acc.realized_pnl
                    cash_note = f"(摘要接口 {summary_transient_err}，现金沿用内存缓存 @ {mem_uk} UK)"

            investments_value = summary_inv_val if summary_inv_val is not None else calc_val
            investments_cost = summary_inv_cost if summary_inv_cost is not None else calc_cost
            unrealized_pnl = summary_unrealized if summary_unrealized is not None else calc_pnl
            total_account_value = summary_total_val if summary_total_val is not None else (investments_value + total_cash)
            cash_ratio = ((total_cash / total_account_value) * 100.0) if total_account_value > 0 else 0.0

            for pos in parsed_positions:
                pos.weight_pct = ((pos.current_value / total_account_value) * 100.0) if total_account_value > 0 else 0.0

            snap = AccountSnapshot(
                ok=True,
                account_label=account_label,
                account_currency=account_currency,
                total_account_value=total_account_value,
                investments_value=investments_value,
                investments_cost=investments_cost,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=summary_realized,
                total_cash=total_cash,
                available_to_trade=available_to_trade,
                reserved_cash=reserved,
                in_pies_cash=in_pies,
                cash_ratio_pct=cash_ratio,
                total_fx_impact=total_fx_impact,
                positions=parsed_positions,
                pending_orders=pending_orders,
                recent_orders=recent_orders,
                recent_transactions=recent_transactions,
                recent_dividends=recent_dividends,
                pending_orders_supported=pending_supported,
                pending_orders_error=pending_error,
                order_history_supported=history_supported,
                transactions_supported=tx_supported,
                dividends_supported=div_supported,
                cash_status_note=cash_note,
            )
            self._enrich_account_dividends(snap, None)
            return snap, False
        except httpx.HTTPError as exc:
            logger.warning("Transient HTTP error fetching %s account: %s", account_label, exc)
            return AccountSnapshot(ok=False, account_label=account_label, error=f"{type(exc).__name__}: {exc}"), True
        except Exception as exc:
            logger.exception("Error fetching %s account", account_label)
            return AccountSnapshot(ok=False, account_label=account_label, error=str(exc)), False

    @staticmethod
    def _annotate_account_fallback(
        acc: AccountSnapshot,
        source: str,
        orig_uk: str,
        reason: str,
    ) -> AccountSnapshot:
        fb = acc.model_copy(deep=True)
        src_label = "内存缓存" if source == "memory" else "数据库快照"
        tag = f"[⚠️ 5xx/断网回退{src_label} @ {orig_uk} UK]"
        fb.is_fallback = True
        fb.fallback_source = source
        fb.fallback_reason = reason[:160]
        fb.snapshot_fetched_at_uk = orig_uk
        fb.cash_status_note = f"{fb.cash_status_note} {tag}".strip() if fb.cash_status_note else tag
        return fb

    async def _resolve_account_fallback(
        self,
        account_label: str,
        reason: str,
        db_snapshot_cache: list[PortfolioData | None],
    ) -> tuple[AccountSnapshot, str, int, str] | None:
        """Try memory cache first, then optional DB snapshot_loader, for a failed account."""
        mem_entry = self._last_good_accounts.get(account_label)
        if mem_entry is not None:
            orig_uk, orig_ms, mem_acc = mem_entry
            return (
                self._annotate_account_fallback(mem_acc, "memory", orig_uk, reason),
                orig_uk,
                orig_ms,
                "memory",
            )

        if self._last_good_portfolio is not None:
            acc = self._last_good_portfolio.isa if account_label == "ISA" else self._last_good_portfolio.invest
            if acc.ok:
                orig_uk = acc.snapshot_fetched_at_uk or self._last_good_portfolio.fetched_at_uk
                orig_ms = self._last_good_portfolio.fetched_at_ms
                return (
                    self._annotate_account_fallback(acc, "memory", orig_uk, reason),
                    orig_uk,
                    orig_ms,
                    "memory",
                )

        if self._snapshot_loader is not None:
            if not db_snapshot_cache:
                try:
                    loaded = await self._snapshot_loader()
                except Exception as exc:
                    logger.warning("T212 snapshot_loader failed: %s", exc)
                    loaded = None
                db_snapshot_cache.append(loaded)
            db_snap = db_snapshot_cache[0]
            if db_snap is not None:
                acc = db_snap.isa if account_label == "ISA" else db_snap.invest
                if acc.ok:
                    orig_uk = acc.snapshot_fetched_at_uk or db_snap.fetched_at_uk
                    orig_ms = db_snap.fetched_at_ms
                    return (
                        self._annotate_account_fallback(acc, "db", orig_uk, reason),
                        orig_uk,
                        orig_ms,
                        "db",
                    )
        return None

    @staticmethod
    def _enrich_account_dividends(
        acc: AccountSnapshot,
        cum_map: dict[tuple[str, str], float] | None = None,
    ) -> None:
        """Populate cumulative_dividends_gbp & total_return_with_div_pct from DB archive and recent dividends."""
        if not acc.ok or not acc.positions:
            return
        recent_by_sym: dict[str, float] = {}
        for d in acc.recent_dividends:
            if d.yahoo_symbol and d.amount > 0:
                recent_by_sym[d.yahoo_symbol] = recent_by_sym.get(d.yahoo_symbol, 0.0) + float(d.amount)

        for pos in acc.positions:
            db_div = (cum_map or {}).get((acc.account_label, pos.yahoo_symbol), 0.0)
            rec_div = recent_by_sym.get(pos.yahoo_symbol, 0.0)
            cum_div = db_div if db_div > 0 else rec_div
            if cum_div > 0:
                pos.cumulative_dividends_gbp = round(float(cum_div), 4)
                if pos.total_cost > 0:
                    pos.total_return_with_div_pct = round(
                        ((pos.pnl + pos.cumulative_dividends_gbp) / pos.total_cost) * 100.0,
                        2,
                    )

    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        now = time.monotonic()
        if use_cache and self._cache is not None:
            cached_at, cached_data = self._cache
            if (now - cached_at) < self.cache_ttl_seconds:
                return cached_data

        async with self._portfolio_lock:
            now = time.monotonic()
            if use_cache and self._cache is not None:
                cached_at, cached_data = self._cache
                if (now - cached_at) < self.cache_ttl_seconds:
                    return cached_data

            if not self._instrument_map and self._db is not None and hasattr(self._db, "get_t212_instrument_map"):
                try:
                    self._instrument_map = await self._db.get_t212_instrument_map()
                except Exception as exc:
                    logger.debug("Failed loading cached T212 instrument map: %s", exc)

            client = self._get_client()
            (isa_snap, isa_transient), (invest_snap, invest_transient) = await asyncio.gather(
                self._fetch_single_account_with_status(client, self.isa_key, self.isa_secret, "ISA"),
                self._fetch_single_account_with_status(client, self.invest_key, self.invest_secret, "Invest"),
            )

            now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
            now_ms = int(time.time() * 1000)

            if isa_snap.ok:
                self._last_good_accounts["ISA"] = (now_uk, now_ms, isa_snap.model_copy(deep=True))
            if invest_snap.ok:
                self._last_good_accounts["Invest"] = (now_uk, now_ms, invest_snap.model_copy(deep=True))

            db_snapshot_cache: list[PortfolioData | None] = []
            fallback_meta: dict[str, tuple[str, int, str, str]] = {}

            if not isa_snap.ok and isa_transient:
                orig_err = isa_snap.error
                resolved = await self._resolve_account_fallback("ISA", orig_err, db_snapshot_cache)
                if resolved is not None:
                    isa_snap, orig_uk, orig_ms, src = resolved
                    fallback_meta["ISA"] = (orig_uk, orig_ms, src, orig_err)

            if not invest_snap.ok and invest_transient:
                orig_err = invest_snap.error
                resolved = await self._resolve_account_fallback("Invest", orig_err, db_snapshot_cache)
                if resolved is not None:
                    invest_snap, orig_uk, orig_ms, src = resolved
                    fallback_meta["Invest"] = (orig_uk, orig_ms, src, orig_err)

            isa_configured = bool(self.isa_key and self.isa_secret)
            invest_configured = bool(self.invest_key and self.invest_secret)
            live_count = int(isa_snap.ok and "ISA" not in fallback_meta) + int(
                invest_snap.ok and "Invest" not in fallback_meta
            )

            is_fallback = bool(fallback_meta)
            fallback_source = ""
            fallback_note = ""
            portfolio_uk = now_uk
            portfolio_ms = now_ms

            if fallback_meta:
                sources = {m[2] for m in fallback_meta.values()}
                if live_count == 0:
                    fallback_source = next(iter(sources)) if len(sources) == 1 else "mixed"
                    # Preserve historical snapshot timestamp when no live account succeeded
                    best_meta = max(fallback_meta.values(), key=lambda m: m[1])
                    portfolio_uk = best_meta[0]
                    portfolio_ms = best_meta[1]
                    fallback_note = f"⚠️ T212 实时拉取异常，已回退至 {portfolio_uk} UK 缓存快照 ({fallback_source})"
                else:
                    fallback_source = "mixed"
                    parts = [f"{lbl}@{m[0]} UK" for lbl, m in fallback_meta.items()]
                    fallback_note = f"⚠️ 部分账户实时拉取异常，已回退缓存快照: {', '.join(parts)}"

            cum_map: dict[tuple[str, str], float] = {}
            if self._db is not None:
                if not is_fallback:
                    live_divs = [
                        d.model_dump()
                        for acc in (isa_snap, invest_snap)
                        if acc.ok and not acc.is_fallback
                        for d in acc.recent_dividends
                    ]
                    if live_divs and hasattr(self._db, "upsert_dividends"):
                        try:
                            await self._db.upsert_dividends(live_divs)
                        except Exception as exc:
                            logger.warning("Failed archiving live T212 dividends: %s", exc)
                if hasattr(self._db, "get_cumulative_dividends_by_symbol"):
                    try:
                        cum_map = await self._db.get_cumulative_dividends_by_symbol()
                    except Exception as exc:
                        logger.debug("Failed loading cumulative dividends from DB: %s", exc)

            self._enrich_account_dividends(isa_snap, cum_map)
            self._enrich_account_dividends(invest_snap, cum_map)
            if isa_snap.ok and not isa_snap.is_fallback:
                self._last_good_accounts["ISA"] = (now_uk, now_ms, isa_snap.model_copy(deep=True))
            if invest_snap.ok and not invest_snap.is_fallback:
                self._last_good_accounts["Invest"] = (now_uk, now_ms, invest_snap.model_copy(deep=True))

            data = PortfolioData(
                fetched_at_uk=portfolio_uk,
                fetched_at_ms=portfolio_ms,
                isa=isa_snap,
                invest=invest_snap,
                total_combined_value=(
                    (isa_snap.total_account_value if isa_snap.ok else 0.0)
                    + (invest_snap.total_account_value if invest_snap.ok else 0.0)
                ),
                total_combined_cash=(
                    (isa_snap.total_cash if isa_snap.ok else 0.0)
                    + (invest_snap.total_cash if invest_snap.ok else 0.0)
                ),
                cash_interest_rate_pct=self.cash_interest_rate_pct,
                is_fallback=is_fallback,
                fallback_source=fallback_source,
                fallback_note=fallback_note,
            )

            for lbl, (orig_uk, _, src, orig_err) in fallback_meta.items():
                await self._notify_ops(
                    "WARN",
                    f"t212_fallback:{lbl}",
                    f"T212 {lbl} 账户实时拉取失败（{orig_err[:120]}），已回退至 {src} 快照 @ {orig_uk} UK。",
                )

            if not is_fallback and (isa_snap.ok or invest_snap.ok):
                self._cache = (now, data)
                self._last_good_portfolio = data.model_copy(deep=True)
                if isa_configured and not isa_snap.ok:
                    await self._notify_ops("WARN", "t212_isa_failed", f"T212 ISA 账户拉取失败：{isa_snap.error}")
                if invest_configured and not invest_snap.ok:
                    await self._notify_ops("WARN", "t212_invest_failed", f"T212 Invest 账户拉取失败：{invest_snap.error}")
            elif isa_snap.ok or invest_snap.ok:
                # Partial/full fallback: do not poison TTL cache, still alert if the other configured account failed without fallback
                if isa_configured and not isa_snap.ok:
                    await self._notify_ops("WARN", "t212_isa_failed", f"T212 ISA 账户拉取失败：{isa_snap.error}")
                if invest_configured and not invest_snap.ok:
                    await self._notify_ops("WARN", "t212_invest_failed", f"T212 Invest 账户拉取失败：{invest_snap.error}")
            elif isa_configured or invest_configured:
                await self._notify_ops(
                    "CRITICAL",
                    "t212_all_accounts_failed",
                    f"Trading 212 双账户全部读取失败！ISA: {isa_snap.error} | Invest: {invest_snap.error}",
                    title="T212 双账户全部读取失败",
                    business_blocking=True,
                    bot_message="🚨 Trading 212 双账户全部读取失败，今日持仓与风控不可用，请检查 API 密钥或查看运维告警。",
                )
            return data


def format_account_overview(acc: AccountSnapshot) -> str:
    if not acc.ok:
        return f"⚠️ {acc.account_label} 账户读取失败: {acc.error}"
    curr = acc.account_currency
    stock_only_pnl = acc.unrealized_pnl - acc.total_fx_impact
    ret_pct = ((acc.unrealized_pnl / acc.investments_cost) * 100.0) if acc.investments_cost > 0 else 0.0
    reserved_str = f", 挂单占用(未成交前仍计息): {acc.reserved_cash:.2f}" if acc.reserved_cash > 0 else ""
    note_str = f" {acc.cash_status_note}" if acc.cash_status_note else ""
    realized_str = f" | 累计已实现: {curr} {acc.realized_pnl:.2f}" if acc.realized_pnl is not None else ""
    fallback_head = (
        f" [⚠️ 回退缓存快照 @ {acc.snapshot_fetched_at_uk} UK]"
        if acc.is_fallback and acc.snapshot_fetched_at_uk
        else ""
    )

    lines = [
        f"• 账户总资产: {curr} {acc.total_account_value:,.2f}{fallback_head}",
        f"• 现金余额: {curr} {acc.total_cash:,.2f} (占 {acc.cash_ratio_pct:.1f}% | 可用: {acc.available_to_trade:,.2f}{reserved_str}){note_str}",
        f"• 持仓市值: {curr} {acc.investments_value:,.2f} (投入成本: {curr} {acc.investments_cost:,.2f})",
        (
            f"• 累计总浮盈: {acc.unrealized_pnl:+,.2f} {curr} ({ret_pct:+.2f}%) "
            f"[本币净盈亏 {stock_only_pnl:+,.2f} {curr} | 累计汇率影响 {acc.total_fx_impact:+,.2f} {curr}]{realized_str}"
        ),
    ]
    return "\n".join(lines)


def format_positions_text(acc: AccountSnapshot) -> str:
    if not acc.ok or not acc.positions:
        return "当前账户无股票持仓"
    curr = acc.account_currency
    lines = []
    for p in acc.positions:
        stock_net = p.pnl - p.fx_impact
        div_str = ""
        if p.cumulative_dividends_gbp > 0:
            tot_ret = (
                p.total_return_with_div_pct
                if p.total_return_with_div_pct is not None
                else (
                    ((p.pnl + p.cumulative_dividends_gbp) / p.total_cost) * 100.0
                    if p.total_cost > 0
                    else p.return_pct
                )
            )
            div_str = f" | 累计派息 +{p.cumulative_dividends_gbp:.2f} {curr} (含息总回报 {tot_ret:+.2f}%)"
        lines.append(
            f"• {p.ticker} ({p.name}) [{p.yahoo_symbol}]: 市值 {curr} {p.current_value:,.2f} (占 {p.weight_pct:.1f}%) | "
            f"均价 {p.avg_price:.2f}→现价 {p.cur_price:.2f} {p.inst_currency} | "
            f"浮盈 {p.pnl:+,.2f} {curr} ({p.return_pct:+.2f}%) "
            f"(本币 {stock_net:+,.2f} | 汇率 {p.fx_impact:+,.2f}) | {p.quantity:.4f}股{div_str}"
        )
    return "\n".join(lines)


def _compute_pending_order_quant_note(
    o: PendingOrderItem,
    *,
    display_curr: str,
    is_gbp_pence: bool,
    bundle: Any | None,
    tech: Any | None,
    hist: Any | None,
) -> str:
    """Compute quote-currency-aligned distance %, ATR multiple, 20d empirical touch rate, and chip/noise tags."""
    raw_trig = (
        o.stop_price
        if (o.stop_price is not None and "STOP" in (o.order_type or "").upper())
        else (o.limit_price if o.limit_price is not None else o.stop_price)
    )
    if raw_trig is None or raw_trig <= 0:
        return ""

    b_curr = getattr(bundle, "currency", None) if bundle is not None else None
    b_price = getattr(bundle, "current_price", None) if bundle is not None else None
    t_close = getattr(tech, "last_close", None) if tech is not None else None
    cur_price = float(b_price) if (b_price is not None and b_price > 0) else (
        float(t_close) if (t_close is not None and t_close > 0) else None
    )
    if cur_price is None or cur_price <= 0:
        return ""

    trig_aligned = float(raw_trig)
    cur_aligned = float(cur_price)

    # Guard against 100x GBp vs GBP mismatch on LSE (.L) symbols
    if o.yahoo_symbol.endswith(".L"):
        ratio_raw = trig_aligned / cur_aligned
        if is_gbp_pence and (o.currency or "").upper() == "GBP" and ratio_raw < 0.05:
            trig_aligned *= 100.0
        elif b_curr == "GBP" and display_curr == "GBp" and ratio_raw > 20.0:
            cur_aligned *= 100.0
        elif not is_gbp_pence and b_curr == "GBp" and ratio_raw < 0.05:
            trig_aligned *= 100.0

    scale_tech = 1.0
    if t_close is not None and t_close > 0:
        tech_ratio = cur_aligned / float(t_close)
        if 80.0 <= tech_ratio <= 125.0:
            scale_tech = 100.0
        elif 0.008 <= tech_ratio <= 0.0125:
            scale_tech = 0.01

    dist_pct = ((trig_aligned / cur_aligned) - 1.0) * 100.0
    atr_mult: float | None = None
    atr_raw = getattr(tech, "atr14", None) if tech is not None else None
    if atr_raw is not None and float(atr_raw) > 0:
        atr_aligned = float(atr_raw) * scale_tech
        atr_mult = abs(trig_aligned - cur_aligned) / atr_aligned

    touch_prob_20d_pct: float | None = None
    if hist is not None and getattr(hist, "empty", True) is False:
        try:
            import numpy as np
            import pandas as pd
            from src.data_providers.analytics import _normalize_adjusted_ohlcv, _resolve_col

            norm_df = _normalize_adjusted_ohlcv(hist)
            c_col = _resolve_col(norm_df, "Close")
            h_col = _resolve_col(norm_df, "High")
            l_col = _resolve_col(norm_df, "Low")
            if c_col and h_col and l_col:
                sub = norm_df[[c_col, h_col, l_col]].apply(pd.to_numeric, errors="coerce").dropna().tail(252)
                if len(sub) >= 25:
                    closes = sub[c_col].to_numpy(dtype=np.float64)
                    highs = sub[h_col].to_numpy(dtype=np.float64)
                    lows = sub[l_col].to_numpy(dtype=np.float64)
                    target_ratio = trig_aligned / cur_aligned
                    window = 20
                    n_bars = len(closes)
                    hits = 0
                    total_windows = 0
                    for i in range(0, n_bars - window):
                        base_c = closes[i]
                        if base_c <= 0:
                            continue
                        total_windows += 1
                        thresh = base_c * target_ratio
                        if target_ratio <= 1.0:
                            if float(np.min(lows[i + 1 : i + 1 + window])) <= thresh:
                                hits += 1
                        else:
                            if float(np.max(highs[i + 1 : i + 1 + window])) >= thresh:
                                hits += 1
                    if total_windows > 0:
                        touch_prob_20d_pct = (hits / total_windows) * 100.0
        except Exception as exc:
            logger.debug("Touch prob calc skipped for %s: %s", o.yahoo_symbol, exc)

    metrics_parts = [f"距现价 {dist_pct:+.1f}%"]
    if atr_mult is not None:
        metrics_parts.append(f"{atr_mult:.1f}倍ATR")
    if touch_prob_20d_pct is not None:
        metrics_parts.append(f"20日触价率 {touch_prob_20d_pct:.0f}%")

    tags: list[str] = []
    is_buy = (o.side or "").upper() == "BUY"
    is_stop = (o.stop_price is not None) or ("STOP" in (o.order_type or "").upper())
    if is_buy and atr_mult is not None and atr_mult > 4.0 and touch_prob_20d_pct is not None and touch_prob_20d_pct < 5.0:
        tags.append("[远端低概率单]")
    if is_stop and atr_mult is not None and atr_mult < 0.8:
        tags.append("[止损过窄易被日常波动扫出]")
    if is_buy and tech is not None:
        c_low_raw = getattr(tech, "cyq_cost_90_low", None)
        c_high_raw = getattr(tech, "cyq_cost_90_high", None)
        if c_low_raw is not None and c_high_raw is not None:
            c_low = float(c_low_raw) * scale_tech
            c_high = float(c_high_raw) * scale_tech
            if 0.0 < c_low <= trig_aligned <= c_high:
                tags.append("[位于90%筹码支撑带]")

    tag_str = f" {' '.join(tags)}" if tags else ""
    return f" | {' / '.join(metrics_parts)}{tag_str}"


def build_trade_context(
    data: PortfolioData,
    recent_days: int = 7,
    history_days: int = 30,
    max_recent_lines: int = 15,
    max_summary_lines: int = 12,
    bundles: dict[str, Any] | None = None,
    pending_us_only: bool = False,
    tech_map: dict[str, Any] | None = None,
    histories: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build pending orders, 7-day fills, and 30-day ticker behavior stats for AI prompts."""
    all_orders = sorted(
        [*data.isa.recent_orders, *data.invest.recent_orders],
        key=lambda o: o.filled_at_ms,
        reverse=True,
    )
    all_pending = [*data.isa.pending_orders, *data.invest.pending_orders]
    if pending_us_only:
        all_pending = [o for o in all_pending if "." not in o.yahoo_symbol]
    recent_cutoff_ms = int(time.time() * 1000) - recent_days * 86400 * 1000
    recent_7d = [o for o in all_orders if o.filled_at_ms >= recent_cutoff_ms]

    def _fmt_order(o: HistoricalOrderItem) -> str:
        side_lbl = "卖出" if o.side == "SELL" else "买入"
        cash_lbl = "回笼" if o.side == "SELL" else "投入"
        pnl_part = (
            f" | 已实现 {o.wallet_currency} {o.realised_pnl:+.2f}"
            if o.side == "SELL" and abs(o.realised_pnl) >= 0.01
            else ""
        )
        tax_part = f" | 税费 {o.wallet_currency} {o.tax_total:.2f}" if o.tax_total >= 0.01 else ""
        return (
            f"• [{o.uk_time_str} UK | {o.account}·{side_lbl}] {o.ticker} ({o.name})："
            f"{o.quantity:.4f}股 @ {o.price:.2f} {o.inst_currency} | "
            f"{cash_lbl} {o.wallet_currency} {o.net_value:.2f}{pnl_part}{tax_part} ({o.order_type})"
        )

    recent_text = (
        "\n".join(_fmt_order(o) for o in recent_7d[:max_recent_lines])
        if recent_7d
        else f"（近 {recent_days} 天无已成交买卖记录）"
    )

    stats: dict[str, dict[str, Any]] = {}
    total_buys = total_sells = 0
    total_buy_amt = total_sell_amt = total_realised = total_taxes = 0.0
    win_sells = loss_sells = 0
    sold_tickers: set[str] = set()

    for o in all_orders:
        key = f"{o.account}__{o.ticker}"
        if key not in stats:
            stats[key] = {
                "account": o.account,
                "ticker": o.ticker,
                "currency": o.wallet_currency,
                "buys": 0,
                "sells": 0,
                "buy_amt": 0.0,
                "sell_amt": 0.0,
                "realised": 0.0,
                "last_uk": o.uk_time_str,
                "last_ms": o.filled_at_ms,
            }
        st = stats[key]
        if o.filled_at_ms > st["last_ms"]:
            st["last_ms"] = o.filled_at_ms
            st["last_uk"] = o.uk_time_str

        if o.side == "SELL":
            total_sells += 1
            total_sell_amt += o.net_value
            total_realised += o.realised_pnl
            st["sells"] += 1
            st["sell_amt"] += o.net_value
            st["realised"] += o.realised_pnl
            sold_tickers.add(o.ticker)
            if o.realised_pnl > 0.01:
                win_sells += 1
            elif o.realised_pnl < -0.01:
                loss_sells += 1
        else:
            total_buys += 1
            total_buy_amt += o.net_value
            st["buys"] += 1
            st["buy_amt"] += o.net_value
        total_taxes += o.tax_total

    held_tickers = {p.ticker for p in data.all_positions()}
    closed_tickers = [t for t in sold_tickers if t not in held_tickers]

    stat_rows = sorted(stats.values(), key=lambda x: x["last_ms"], reverse=True)[:max_summary_lines]
    summary_lines = []
    for s in stat_rows:
        parts = []
        if s["buys"] > 0:
            parts.append(f"买{s['buys']}笔({s['currency']} {s['buy_amt']:.0f})")
        if s["sells"] > 0:
            parts.append(f"卖{s['sells']}笔({s['currency']} {s['sell_amt']:.0f})")
        pnl_s = f" | 已实现 {s['currency']} {s['realised']:+.2f}" if abs(s["realised"]) >= 0.01 else ""
        status_tag = " [已清仓]" if s["ticker"] in closed_tickers else ""
        summary_lines.append(f"• [{s['account']}] {s['ticker']}{status_tag}: {' / '.join(parts)}{pnl_s} | 最近 {s['last_uk']} UK")

    if all_orders:
        closed_note = f" | 期间已清仓退出: {', '.join(closed_tickers)}" if closed_tickers else ""
        header = (
            f"近 {history_days} 天共 {len(all_orders)} 笔成交（买入 {total_buys} 笔约 {total_buy_amt:.0f}，"
            f"卖出 {total_sells} 笔约 {total_sell_amt:.0f}；已实现净盈亏 {total_realised:+.2f}，"
            f"盈利平仓 {win_sells} 笔 / 亏损平仓 {loss_sells} 笔，累计税费 {total_taxes:.2f}）{closed_note}"
        )
        summary_text = header + "\n" + "\n".join(summary_lines)
    else:
        summary_text = f"（近 {history_days} 天无历史成交订单）"

    pending_lines = []
    for o in all_pending:
        b = bundles.get(o.yahoo_symbol) if bundles else None
        # Fix T212 /equity/orders returning account currency (GBP) instead of instrument quote currency (GBp/USD/EUR)
        display_curr = o.currency
        is_gbp_pence = (display_curr in {"GBX", "GBp"}) or (
            b is not None and b.currency == "GBp" and o.yahoo_symbol.endswith(".L")
        )
        if is_gbp_pence:
            display_curr = "GBp"
        elif b is not None and b.currency and not o.yahoo_symbol.endswith(".L") and display_curr == "GBP":
            display_curr = b.currency

        trig = []
        if o.limit_price is not None:
            gbp_eq = f"(£{o.limit_price / 100:.2f})" if is_gbp_pence else ""
            trig.append(f"限价 {o.limit_price:.2f}{gbp_eq}")
        if o.stop_price is not None:
            gbp_eq = f"(£{o.stop_price / 100:.2f})" if is_gbp_pence else ""
            trig.append(f"止损 {o.stop_price:.2f}{gbp_eq}")

        market_ctx = ""
        if b is not None and b.current_price is not None:
            tgt = f", 目标均价 {b.target_mean_price:.2f}" if b.target_mean_price else ""
            market_ctx = f" | 市场现价 {b.current_price:.2f} {b.currency or display_curr}{tgt}"

        quant_note = _compute_pending_order_quant_note(
            o,
            display_curr=display_curr,
            is_gbp_pence=is_gbp_pence,
            bundle=b,
            tech=tech_map.get(o.yahoo_symbol) if tech_map else None,
            hist=histories.get(o.yahoo_symbol) if histories else None,
        )

        qty_str = f"{o.quantity:.4f}股" if o.quantity > 0 else f"金额 {o.currency} {o.value:.2f}"
        if o.filled_quantity > 0 or o.status == "PARTIALLY_FILLED":
            qty_str += f"[部分成交,已成{o.filled_quantity:.4f}股]"
        side_lbl = "卖出" if o.side == "SELL" else "买入"
        pending_lines.append(
            f"• [{o.account}·挂单] {o.ticker} ({o.name}) [{o.yahoo_symbol}] {side_lbl} {qty_str} | "
            f"{o.order_type} {'/'.join(trig)} {display_curr} ({o.time_in_force}){market_ctx}{quant_note}"
        )

    pending_err_parts = [
        f"{acc.account_label}: {acc.pending_orders_error}"
        for acc in (data.isa, data.invest)
        if acc.ok and acc.pending_orders_error
    ]
    if pending_lines:
        if pending_err_parts:
            pending_lines.append(
                f"⚠️ 部分账户活动挂单接口临时异常（{', '.join(pending_err_parts)}），请勿据此判断该账户无挂单"
            )
        pending_text = "\n".join(pending_lines)
    elif pending_err_parts:
        pending_text = f"⚠️ 活动挂单接口临时异常（{', '.join(pending_err_parts)}），请勿据此判断挂单已撤销"
    else:
        pending_text = "（当前无未成交活动挂单）"

    history_cutoff_ms = int(time.time() * 1000) - history_days * 86400 * 1000
    all_tx = [
        tx
        for tx in (*data.isa.recent_transactions, *data.invest.recent_transactions)
        if tx.date_time_ms >= history_cutoff_ms
    ]
    all_divs = [
        dv
        for dv in (*data.isa.recent_dividends, *data.invest.recent_dividends)
        if dv.paid_on_ms >= history_cutoff_ms
    ]

    cash_flow_summary_text = ""
    if all_tx or all_divs:
        dep_sum = sum(tx.amount for tx in all_tx if tx.tx_type == "DEPOSIT")
        wd_sum = sum(tx.amount for tx in all_tx if tx.tx_type == "WITHDRAW")
        tr_sum = sum(tx.amount for tx in all_tx if tx.tx_type == "TRANSFER")
        fee_sum = sum(tx.amount for tx in all_tx if tx.tx_type == "FEE")
        int_sum = sum(tx.amount for tx in all_tx if tx.tx_type in {"INTEREST_ON_FREE_CASH", "LENDING_INTEREST"})
        div_sum = sum(dv.amount for dv in all_divs)

        cf_parts: list[str] = []
        if abs(dep_sum) >= 0.01:
            cf_parts.append(f"外部入金 {dep_sum:+,.2f} GBP")
        if abs(wd_sum) >= 0.01:
            cf_parts.append(f"外部出金 {wd_sum:+,.2f} GBP")
        if abs(tr_sum) >= 0.01:
            cf_parts.append(f"账户划转 {tr_sum:+,.2f} GBP")
        if abs(fee_sum) >= 0.01:
            cf_parts.append(f"平台费用 {fee_sum:+,.2f} GBP")
        if abs(div_sum) >= 0.01:
            cf_parts.append(f"派息入账 +{div_sum:,.2f} GBP({len(all_divs)}笔)")
        if abs(int_sum) >= 0.01:
            cf_parts.append(f"现金利息 +{int_sum:,.2f} GBP")

        tx_label_map = {
            "DEPOSIT": "入金",
            "WITHDRAW": "出金",
            "TRANSFER": "划转",
            "FEE": "扣费",
            "INTEREST_ON_FREE_CASH": "现金利息",
            "LENDING_INTEREST": "借券利息",
        }
        merged_events: list[tuple[int, str]] = []
        for tx in all_tx:
            lbl = tx_label_map.get(tx.tx_type, tx.tx_type)
            merged_events.append(
                (tx.date_time_ms, f"• [{tx.uk_time_str} UK | {tx.account}·{lbl}] {tx.amount:+,.2f} {tx.currency}")
            )
        for dv in all_divs:
            merged_events.append(
                (dv.paid_on_ms, f"• [{dv.uk_time_str} UK | {dv.account}·派息] {dv.ticker}: +{dv.amount:,.2f} {dv.currency}")
            )
        merged_events.sort(key=lambda item: item[0], reverse=True)
        cf_header = f"[资金流水与派息(近 {history_days} 天)] " + (" | ".join(cf_parts) if cf_parts else "（微量变动）")
        cf_lines = [line for _, line in merged_events[:6]]
        cash_flow_summary_text = cf_header + ("\n" + "\n".join(cf_lines) if cf_lines else "")

    return {
        "recent_trades_text": recent_text,
        "trade_summary_30d_text": summary_text,
        "pending_orders_text": pending_text,
        "cash_flow_summary_text": cash_flow_summary_text,
        "traded_tickers": ", ".join(dict.fromkeys(o.ticker for o in all_orders)) or "无",
    }
