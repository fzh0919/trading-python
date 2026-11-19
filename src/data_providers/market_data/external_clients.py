"""Async Finnhub and FRED external REST API clients."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.data_providers.market_data.models_and_parsing import (
    _clean_text,
    _is_us_symbol,
    _safe_exc,
    _safe_float,
    _safe_int,
)

logger = logging.getLogger("src.data_providers.market_data")

class FinnhubClient:
    """Optional free Finnhub client (60 calls/min) for US earnings, news, quotes & intel."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str = "", transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = api_key.strip()
        self._transport = transport
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

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
            res = await client.get(
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

    async def _fetch_earnings_calendar(
        self, client: httpx.AsyncClient, days_ahead: int = 21
    ) -> tuple[dict[str, tuple[str, str]], str | None]:
        if not self.enabled:
            return {}, None
        today = datetime.now(timezone.utc).date()
        end = today + timedelta(days=days_ahead)
        try:
            res = await client.get(
                f"{self.BASE_URL}/calendar/earnings",
                params={"from": today.isoformat(), "to": end.isoformat(), "token": self.api_key},
                timeout=12.0,
            )
            if res.status_code != 200:
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
            res = await client.get(
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
            return {
                "symbol": symbol.strip().upper(),
                "current_price": price,
                "change": _safe_float(data.get("d")),
                "change_pct": _safe_float(data.get("dp")),
                "high": _safe_float(data.get("h")),
                "low": _safe_float(data.get("l")),
                "open": _safe_float(data.get("o")),
                "prev_close": _safe_float(data.get("pc")),
            }
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
            res = await client.get(
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
            res = await client.get(
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
            res = await client.get(
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
            res = await client.get(
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
            res = await client.get(
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
