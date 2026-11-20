"""Async Finnhub and FRED external REST API clients."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from src.data_providers.market_data.models_and_parsing import (
    _clean_text,
    _is_us_symbol,
    _safe_exc,
    _safe_float,
    _safe_int,
)

logger = logging.getLogger("src.data_providers.market_data")

_NY_TZ = ZoneInfo("America/New_York")

class FinnhubClient:
    """Optional free Finnhub client (60 calls/min) for US earnings, news, quotes & intel.

    v5.5.5: every HTTP GET flows through :meth:`_throttled_get` (semaphore + min interval)
    so concurrent `/pick` fan-out can never burst past the free-tier 60/min ceiling.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_concurrency: int = 2,
        min_interval_seconds: float | None = None,
        earnings_cache_ttl_seconds: int = 21600,
        retry_429_delay_seconds: float = 0.0,
    ) -> None:
        self.api_key = api_key.strip()
        self._transport = transport
        self.last_error: str | None = None
        self._max_concurrency = max(1, int(max_concurrency))
        # Mock transports (offline tests) must never introduce real sleeps; production defaults
        # to ~0.35s between dispatches -> <= 2 in flight and steady-state <= ~45 req/min.
        if min_interval_seconds is None:
            min_interval_seconds = 0.0 if transport is not None else 0.35
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._earnings_cache_ttl_seconds = max(0, int(earnings_cache_ttl_seconds))
        self._retry_429_delay_seconds = max(0.0, float(retry_429_delay_seconds))
        self._earnings_cache: dict[tuple[str, int], tuple[float, dict[str, tuple[str, str]]]] = {}
        # Loop-bound primitives are initialised lazily so repeated `asyncio.run` in tests
        # never reuses a Semaphore/Lock created on a closed event loop.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None
        self._rate_lock: asyncio.Lock | None = None
        self._last_request_ts = 0.0
        self._calendar_locks: dict[tuple[str, int], asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _ensure_loop_primitives(self) -> tuple[asyncio.Semaphore, asyncio.Lock]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - defensive, callers are always async
            loop = None
        if self._sem is None or self._rate_lock is None or self._loop is not loop:
            self._loop = loop
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._rate_lock = asyncio.Lock()
            self._last_request_ts = 0.0
            self._calendar_locks = {}
        return self._sem, self._rate_lock

    def _retry_after_seconds(self, res: httpx.Response) -> float:
        if self._min_interval_seconds <= 0.0:
            return self._retry_429_delay_seconds
        raw = res.headers.get("retry-after")
        try:
            value = float(raw) if raw is not None else 1.0
        except (TypeError, ValueError):
            value = 1.0
        return min(5.0, max(1.0, value))

    async def _throttled_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        """Semaphore-bounded, min-interval-throttled GET with a single HTTP 429 retry."""
        sem, rate_lock = self._ensure_loop_primitives()
        async with sem:
            async with rate_lock:
                if self._min_interval_seconds > 0.0:
                    wait = self._min_interval_seconds - (time.monotonic() - self._last_request_ts)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_request_ts = time.monotonic()
            res = await client.get(url, params=params, timeout=timeout)
        if res.status_code == 429:
            delay = self._retry_after_seconds(res)
            if delay > 0:
                await asyncio.sleep(delay)
            async with sem:
                res = await client.get(url, params=params, timeout=timeout)
        return res

    async def _fetch_company_news(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        days: int = 7,
        limit: int = 4,
    ) -> tuple[list[dict[str, str]], str | None]:
        if not self.enabled or not _is_us_symbol(symbol):
            return [], None
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=days)).isoformat()
        end = today.isoformat()
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/company-news",
                params={"symbol": symbol, "from": start, "to": end, "token": self.api_key},
                timeout=10.0,
            )
            if res.status_code != 200:
                return [], f"company-news HTTP {res.status_code} ({symbol})"
            items = res.json() or []
            out = []
            for item in items[:limit]:
                headline = str(item.get("headline") or "").strip()
                if not headline:
                    continue
                ts = item.get("datetime")
                date_str = (
                    datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                    if ts
                    else ""
                )
                out.append(
                    {
                        "date": date_str,
                        "source": str(item.get("source") or "Finnhub"),
                        "title": headline,
                    }
                )
            return out, None
        except Exception as exc:
            err = f"company-news {_safe_exc(exc)}"
            logger.debug("Finnhub news error for %s: %s", symbol, exc)
            return [], err

    async def get_company_news(
        self, client: httpx.AsyncClient, symbol: str, days: int = 7, limit: int = 4
    ) -> list[dict[str, str]]:
        out, err = await self._fetch_company_news(client, symbol, days=days, limit=limit)
        self.last_error = err
        return out

    def _calendar_lock(self, key: tuple[str, int]) -> asyncio.Lock:
        self._ensure_loop_primitives()
        lock = self._calendar_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._calendar_locks[key] = lock
        return lock

    async def _fetch_earnings_calendar(
        self, client: httpx.AsyncClient, days_ahead: int = 21
    ) -> tuple[dict[str, tuple[str, str]], str | None]:
        if not self.enabled:
            return {}, None
        today = datetime.now(timezone.utc).date()
        cache_key = (today.isoformat(), int(days_ahead))
        cached = self._earnings_cache.get(cache_key)
        if (
            cached is not None
            and self._earnings_cache_ttl_seconds > 0
            and (time.monotonic() - cached[0]) < self._earnings_cache_ttl_seconds
        ):
            return dict(cached[1]), None
        # In-flight merge: concurrent inspect_ticker/get_bundles fan-out shares one HTTP call.
        async with self._calendar_lock(cache_key):
            cached = self._earnings_cache.get(cache_key)
            if (
                cached is not None
                and self._earnings_cache_ttl_seconds > 0
                and (time.monotonic() - cached[0]) < self._earnings_cache_ttl_seconds
            ):
                return dict(cached[1]), None
            end = today + timedelta(days=days_ahead)
            try:
                res = await self._throttled_get(
                    client,
                    f"{self.BASE_URL}/calendar/earnings",
                    params={"from": today.isoformat(), "to": end.isoformat(), "token": self.api_key},
                    timeout=12.0,
                )
                if res.status_code != 200:
                    # Degraded response: never poisoned into the earnings cache (iron rule #2).
                    return {}, f"calendar/earnings HTTP {res.status_code}"
                rows = (res.json() or {}).get("earningsCalendar") or []
                mapping: dict[str, tuple[str, str]] = {}
                for r in rows:
                    sym = str(r.get("symbol") or "").upper()
                    dt_str = str(r.get("date") or "")
                    if not sym or not dt_str:
                        continue
                    eps = r.get("epsEstimate")
                    eps_note = f"EPS预期 {eps}" if eps is not None else ""
                    mapping.setdefault(sym, (dt_str, eps_note))
                self._earnings_cache[cache_key] = (time.monotonic(), dict(mapping))
                return mapping, None
            except Exception as exc:
                err = f"calendar/earnings {_safe_exc(exc)}"
                logger.debug("Finnhub earnings calendar error: %s", exc)
                return {}, err

    async def get_earnings_calendar(
        self, client: httpx.AsyncClient, days_ahead: int = 21
    ) -> dict[str, tuple[str, str]]:
        """Return {symbol: (date_str, eps_est_str)} for upcoming US earnings."""
        mapping, err = await self._fetch_earnings_calendar(client, days_ahead=days_ahead)
        self.last_error = err
        return mapping

    async def get_quote(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None:
        """Fetch fast US real-time quote from Finnhub /quote."""
        if not self.enabled or not _is_us_symbol(symbol):
            return None
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/quote",
                params={"symbol": symbol.strip().upper(), "token": self.api_key},
                timeout=8.0,
            )
            if res.status_code != 200:
                self.last_error = f"quote HTTP {res.status_code} ({symbol})"
                return None
            data = res.json() or {}
            price = _safe_float(data.get("c"))
            if price is None or price <= 0:
                return None
            self.last_error = None
            quote_payload: dict[str, Any] = {
                "symbol": symbol.strip().upper(),
                "current_price": price,
                "change": _safe_float(data.get("d")),
                "change_pct": _safe_float(data.get("dp")),
                "high": _safe_float(data.get("h")),
                "low": _safe_float(data.get("l")),
                "open": _safe_float(data.get("o")),
                "prev_close": _safe_float(data.get("pc")),
            }
            # v5.5.8: Finnhub /quote natively carries a Unix-seconds timestamp `t`.
            # Convert it to America/New_York session metadata so the /us anomaly gate
            # can reject stale (yesterday's) dp moves; when `t` is missing (legacy
            # stubs / older fixtures) no bar_date_et is written and callers keep
            # treating the quote as session-fresh for backward compatibility.
            t_val = _safe_float(data.get("t"))
            if t_val is not None and t_val > 0:
                ts_et = datetime.fromtimestamp(float(t_val), tz=_NY_TZ)
                quote_payload["bar_date_et"] = ts_et.strftime("%Y-%m-%d")
                quote_payload["quote_time_et"] = ts_et.strftime("%H:%M")
                quote_payload["source"] = "finnhub_quote"
            return quote_payload
        except Exception as exc:
            self.last_error = f"quote {_safe_exc(exc)}"
            logger.debug("Finnhub quote error for %s: %s", symbol, exc)
            return None

    async def get_profile_by_isin(self, client: httpx.AsyncClient, isin: str) -> dict[str, Any] | None:
        """Fallback free company profile lookup by ISIN via /stock/profile2."""
        clean_isin = _clean_text(isin).upper()
        if not self.enabled or not clean_isin:
            return None
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/stock/profile2",
                params={"isin": clean_isin, "token": self.api_key},
                timeout=10.0,
            )
            if res.status_code != 200:
                self.last_error = f"profile2 HTTP {res.status_code} ({clean_isin})"
                return None
            data = res.json() or {}
            if not data or not _clean_text(data.get("name")):
                return None
            self.last_error = None
            return {
                "symbol": _clean_text(data.get("ticker")),
                "name": _clean_text(data.get("name")),
                "exchange": _clean_text(data.get("exchange")),
                "currency": _clean_text(data.get("currency")),
                "industry": _clean_text(data.get("finnhubIndustry")),
                "country": _clean_text(data.get("country")),
                "isin": clean_isin,
            }
        except Exception as exc:
            self.last_error = f"profile2 {_safe_exc(exc)}"
            logger.debug("Finnhub profile2 error for %s: %s", clean_isin, exc)
            return None

    async def get_recommendation_trends(
        self, client: httpx.AsyncClient, symbol: str
    ) -> list[dict[str, Any]]:
        """Fetch analyst recommendation counts via /stock/recommendation."""
        if not self.enabled or not _is_us_symbol(symbol):
            return []
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/stock/recommendation",
                params={"symbol": symbol.strip().upper(), "token": self.api_key},
                timeout=10.0,
            )
            if res.status_code != 200:
                self.last_error = f"recommendation HTTP {res.status_code} ({symbol})"
                return []
            rows = res.json() or []
            if not isinstance(rows, list):
                return []
            self.last_error = None
            return [
                {
                    "period": str(r.get("period") or ""),
                    "strong_buy": _safe_int(r.get("strongBuy")) or 0,
                    "buy": _safe_int(r.get("buy")) or 0,
                    "hold": _safe_int(r.get("hold")) or 0,
                    "sell": _safe_int(r.get("sell")) or 0,
                    "strong_sell": _safe_int(r.get("strongSell")) or 0,
                }
                for r in rows[:2]
            ]
        except Exception as exc:
            self.last_error = f"recommendation {_safe_exc(exc)}"
            return []

    async def get_earnings_surprises(
        self, client: httpx.AsyncClient, symbol: str, limit: int = 4
    ) -> list[dict[str, Any]]:
        """Fetch historical quarterly EPS surprises via /stock/earnings."""
        if not self.enabled or not _is_us_symbol(symbol):
            return []
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/stock/earnings",
                params={"symbol": symbol.strip().upper(), "limit": limit, "token": self.api_key},
                timeout=10.0,
            )
            if res.status_code != 200:
                self.last_error = f"earnings HTTP {res.status_code} ({symbol})"
                return []
            rows = res.json() or []
            if not isinstance(rows, list):
                return []
            self.last_error = None
            return [
                {
                    "period": str(r.get("period") or ""),
                    "actual": _safe_float(r.get("actual")),
                    "estimate": _safe_float(r.get("estimate")),
                    "surprise_pct": _safe_float(r.get("surprisePercent")),
                }
                for r in rows[:limit]
            ]
        except Exception as exc:
            self.last_error = f"earnings {_safe_exc(exc)}"
            return []

    async def get_insider_sentiment(
        self, client: httpx.AsyncClient, symbol: str, days: int = 90
    ) -> dict[str, Any] | None:
        """Fetch US insider MSPR sentiment via /stock/insider-sentiment."""
        if not self.enabled or not _is_us_symbol(symbol):
            return None
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=days)).isoformat()
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/stock/insider-sentiment",
                params={
                    "symbol": symbol.strip().upper(),
                    "from": start,
                    "to": today.isoformat(),
                    "token": self.api_key,
                },
                timeout=10.0,
            )
            if res.status_code != 200:
                self.last_error = f"insider-sentiment HTTP {res.status_code} ({symbol})"
                return None
            rows = (res.json() or {}).get("data") or []
            if not rows:
                self.last_error = None
                return None
            latest = rows[-1]
            self.last_error = None
            return {
                "year": _safe_int(latest.get("year")),
                "month": _safe_int(latest.get("month")),
                "change": _safe_float(latest.get("change")),
                "mspr": _safe_float(latest.get("mspr")),
            }
        except Exception as exc:
            self.last_error = f"insider-sentiment {_safe_exc(exc)}"
            return None


    async def get_basic_financials(
        self, client: httpx.AsyncClient, symbol: str
    ) -> dict[str, Any] | None:
        """Fetch /stock/metric (Basic Financials) and return the raw ``metric`` dict.

        Free-tier endpoint; US symbols only (LSE/EU coverage is thin). Field names are
        resolved through alias candidates in :func:`apply_basic_financials` because the
        vendor's key naming has shifted between TTM / Annual / Quarterly variants.
        """
        clean = (symbol or "").strip().upper()
        if not self.enabled or not clean or not _is_us_symbol(clean):
            return None
        try:
            res = await self._throttled_get(
                client,
                f"{self.BASE_URL}/stock/metric",
                params={"symbol": clean, "metric": "all", "token": self.api_key},
                timeout=10.0,
            )
            if res.status_code != 200:
                self.last_error = f"metric HTTP {res.status_code} ({clean})"
                return None
            payload = res.json() or {}
            metrics = payload.get("metric") if isinstance(payload, dict) else None
            if not isinstance(metrics, dict) or not metrics:
                return None
            self.last_error = None
            return metrics
        except Exception as exc:
            self.last_error = f"metric {_safe_exc(exc)}"
            return None



class FredClient:
    """Optional free St. Louis Fed FRED client (120 calls/min) for official macro series."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str = "", transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = api_key.strip()
        self._transport = transport
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _fetch_series_stats(
        self, client: httpx.AsyncClient, series_id: str, limit: int = 25
    ) -> tuple[dict[str, float | None], str | None]:
        """Fetch recent FRED observations and compute latest value plus 5d/20d changes."""
        if not self.enabled:
            return {}, None
        try:
            res = await client.get(
                self.BASE_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": max(5, int(limit)),
                },
                timeout=10.0,
            )
            if res.status_code != 200:
                return {}, f"FRED {series_id} HTTP {res.status_code}"
            obs = (res.json() or {}).get("observations") or []
            vals: list[float] = []
            for row in obs:
                raw_val = str(row.get("value") or ".")
                if raw_val == ".":
                    continue
                parsed = _safe_float(raw_val)
                if parsed is not None:
                    vals.append(parsed)
            if not vals:
                return {}, None
            latest = vals[0]
            return {
                "latest": latest,
                "change_5d": (latest - vals[5]) if len(vals) >= 6 else None,
                "change_20d": (latest - vals[20]) if len(vals) >= 21 else None,
            }, None
        except Exception as exc:
            err = f"FRED {series_id} {_safe_exc(exc)}"
            logger.debug("FRED error for %s: %s", series_id, exc)
            return {}, err

    async def _fetch_latest_value(
        self, client: httpx.AsyncClient, series_id: str
    ) -> tuple[float | None, str | None]:
        stats, err = await self._fetch_series_stats(client, series_id, limit=25)
        return stats.get("latest"), err

    async def get_latest_value(self, client: httpx.AsyncClient, series_id: str) -> float | None:
        val, err = await self._fetch_latest_value(client, series_id)
        self.last_error = err
        return val
