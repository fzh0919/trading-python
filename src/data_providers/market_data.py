"""Unified market & macro data provider: Yahoo Finance + Finnhub + FRED."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from pydantic import BaseModel, Field
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:  # pragma: no cover - defensive import across yfinance versions
    class YFRateLimitError(Exception):
        pass

logger = logging.getLogger(__name__)

_MINOR_UNIT_CURRENCIES = {"GBp", "GBX", "ZAc", "ILA"}
_PLACEHOLDER_STRINGS = {"none", "n/a", "na", "nan", "null", "-"}


def _clean_text(value: Any) -> str:
    """Return stripped text or empty string for missing / placeholder values."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in _PLACEHOLDER_STRINGS:
        return ""
    return text


def _safe_float(value: Any) -> float | None:
    """Parse finite float or return None."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _safe_int(value: Any) -> int | None:
    """Parse int safely or return None."""
    num = _safe_float(value)
    if num is None:
        return None
    return int(round(num))


def _ratio_to_pct(value: Any) -> float | None:
    """Convert decimal ratio (e.g. 0.165) to percentage (16.5)."""
    num = _safe_float(value)
    if num is None:
        return None
    return round(num * 100.0, 4)


def _is_us_symbol(symbol: str) -> bool:
    return "." not in symbol and "=" not in symbol and "^" not in symbol


def _parse_date_days_ahead(date_val: Any) -> tuple[str | None, int | None]:
    """Parse date string, datetime/date, or Unix epoch timestamp into (YYYY-MM-DD, days_ahead)."""
    if date_val is None:
        return None, None
    try:
        if isinstance(date_val, bool):
            return None, None
        if isinstance(date_val, (int, float)):
            if not math.isfinite(float(date_val)) or float(date_val) <= 0:
                return None, None
            dt = datetime.fromtimestamp(float(date_val), tz=timezone.utc).date()
        elif isinstance(date_val, datetime):
            dt = date_val.date()
        elif hasattr(date_val, "year") and hasattr(date_val, "month") and hasattr(date_val, "day"):
            dt = datetime(int(date_val.year), int(date_val.month), int(date_val.day)).date()
        else:
            dt_str = _clean_text(date_val)[:10]
            if not dt_str:
                return None, None
            dt = datetime.strptime(dt_str, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        delta = (dt - today).days
        return dt.isoformat(), delta
    except Exception:
        return None, None


def _resolve_dividend_yield_pct(
    info: dict[str, Any],
    current_price: float | None,
    currency: str,
) -> float | None:
    """Resolve dividend yield in percentage points without amplifying low-yield US stocks."""
    div_y = _safe_float(info.get("dividendYield"))
    trailing_ratio = _safe_float(info.get("trailingAnnualDividendYield"))
    div_rate = _safe_float(info.get("dividendRate"))
    price = (
        current_price
        or _safe_float(info.get("currentPrice"))
        or _safe_float(info.get("regularMarketPrice"))
        or _safe_float(info.get("previousClose"))
    )

    ref_pct: float | None = None
    if trailing_ratio is not None and 0.0 < trailing_ratio < 0.5:
        ref_pct = trailing_ratio * 100.0
    elif (
        div_rate is not None
        and div_rate > 0
        and price is not None
        and price > 0
        and currency not in _MINOR_UNIT_CURRENCIES
    ):
        cand_pct = (div_rate / price) * 100.0
        if 0.0 < cand_pct < 50.0:
            ref_pct = cand_pct

    if div_y is None or div_y < 0:
        return round(ref_pct, 4) if ref_pct is not None else None
    if div_y == 0.0:
        return round(ref_pct, 4) if ref_pct is not None else 0.0

    # Cross-check against reference yield to distinguish yfinance >=0.2.54 pct units vs legacy ratios
    if ref_pct is not None and ref_pct > 0:
        if abs((div_y * 100.0) - ref_pct) < abs(div_y - ref_pct):
            return round(div_y * 100.0, 4)
        return round(div_y, 4)

    # Current yfinance reports dividendYield directly in percentage points (e.g. 0.03 = 0.03%)
    return round(div_y, 4)


def normalize_history_index(index: pd.Index) -> pd.DatetimeIndex | None:
    """Convert date-like index to tz-naive midnight timestamps, preserving local exchange dates."""
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_localize(None).normalize()
    if pd.api.types.is_numeric_dtype(index):
        return None
    try:
        dt_idx = pd.to_datetime(index, errors="raise")
        return pd.DatetimeIndex(dt_idx).tz_localize(None).normalize()
    except Exception:
        try:
            # Handle object indices containing mixed UTC offsets across DST transitions
            return pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None).normalize() for x in index])
        except Exception:
            return None


def _sanitize_history_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize OHLCV DataFrame: coerce numeric columns, drop invalid Close rows, normalize & sort index."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    clean = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0]
    if clean.empty:
        return pd.DataFrame()
    norm_idx = normalize_history_index(clean.index)
    if norm_idx is not None:
        clean.index = norm_idx
        clean = clean[clean.index.notna()]
        if clean.empty:
            return pd.DataFrame()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    return clean


def _sanitize_intraday_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize intraday (e.g. 15m prepost) OHLCV DataFrame while preserving exact timestamps."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    clean = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0]
    if clean.empty:
        return pd.DataFrame()
    dt_idx = pd.to_datetime(clean.index, errors="coerce")
    clean.index = dt_idx
    clean = clean[clean.index.notna()]
    if clean.empty:
        return pd.DataFrame()
    return clean[~clean.index.duplicated(keep="last")].sort_index()


def _parse_eps_estimate(value: Any) -> float | None:
    """Extract numeric EPS estimate from float/int or strings like 'EPS预期 1.25'."""
    if value is None:
        return None
    direct = _safe_float(value)
    if direct is not None:
        return direct
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return _safe_float(m.group(0)) if m else None


def _serialize_ohlcv_df(symbol: str, period: str, df: pd.DataFrame) -> dict[str, Any] | None:
    """Serialize sanitized daily OHLCV DataFrame into SQLite ohlcv_cache payload."""
    clean = _sanitize_history_df(df)
    if clean.empty or "Close" not in clean.columns:
        return None
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in clean.columns]
    if "Close" not in cols:
        return None
    dates = [ts.strftime("%Y-%m-%d") for ts in pd.to_datetime(clean.index)]
    last_close = float(clean["Close"].iloc[-1])
    if not dates or not math.isfinite(last_close) or last_close <= 0:
        return None
    records = clean[cols].round(6).values.tolist()
    return {
        "symbol": symbol,
        "period": period,
        "last_bar_date": dates[-1],
        "last_close": last_close,
        "updated_at_ms": int(time.time() * 1000),
        "payload_json": json.dumps({"index": dates, "columns": cols, "data": records}),
    }


def _deserialize_ohlcv_json(payload_json: str) -> pd.DataFrame:
    """Deserialize persisted ohlcv_cache JSON string back to a sanitized DataFrame."""
    if not payload_json:
        return pd.DataFrame()
    try:
        obj = json.loads(payload_json)
        if not isinstance(obj, dict):
            return pd.DataFrame()
        index = obj.get("index")
        columns = obj.get("columns")
        data = obj.get("data")
        if not isinstance(index, list) or not isinstance(columns, list) or not isinstance(data, list):
            return pd.DataFrame()
        if not index or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data=data, index=pd.to_datetime(index, errors="coerce"), columns=columns)
        return _sanitize_history_df(df)
    except Exception:
        return pd.DataFrame()


class TickerMarketBundle(BaseModel):
    symbol: str
    quote_type: str = ""
    currency: str = ""
    short_name: str = ""
    long_name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    isin: str = ""
    identity_summary: str = ""
    current_price: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    market_cap: float | None = None
    dividend_yield_pct: float | None = None
    target_mean_price: float | None = None
    beta: float | None = None
    profit_margin_pct: float | None = None
    revenue_growth_pct: float | None = None
    roe_pct: float | None = None
    debt_to_equity: float | None = None
    analyst_count: int | None = None
    recommendation_key: str = ""
    earnings_date: str | None = None
    days_to_earnings: int | None = None
    eps_estimate: float | None = None
    ex_dividend_date: str | None = None
    days_to_ex_dividend: int | None = None
    news: list[dict[str, str]] = Field(default_factory=list)
    # History stored separately in analytics layer; serialized summary kept light
    has_history: bool = False

    def build_identity_anchor(self, isin_override: str | None = None) -> str:
        """Build deterministic identity anchor string to prevent LLM ticker/company confusion."""
        name = _clean_text(self.long_name) or _clean_text(self.short_name) or self.symbol
        qtype = _clean_text(self.quote_type).upper() or "EQUITY"
        exch = _clean_text(self.exchange)
        curr = _clean_text(self.currency)
        isin_val = _clean_text(isin_override) or _clean_text(self.isin)

        parts = [self.symbol, name]
        if exch:
            parts.append(exch)
        parts.append(qtype)
        if curr:
            parts.append(curr)

        if qtype != "ETF":
            sec = _clean_text(self.sector)
            ind = _clean_text(self.industry)
            if sec and ind and ind != sec:
                parts.append(f"{sec}/{ind}")
            elif sec:
                parts.append(sec)
            elif ind:
                parts.append(ind)
        if isin_val:
            parts.append(f"ISIN:{isin_val}")
        return " | ".join(parts)


class MacroSnapshot(BaseModel):
    fetched_at_utc: str
    gbp_usd: float | None = None
    eur_gbp: float | None = None
    us_10y_yield: float | None = None
    us_10y_change_20d_bps: float | None = None
    us_2y_yield: float | None = None
    yield_spread_10y_2y: float | None = None
    breakeven_10y_pct: float | None = None
    hy_oas_spread_pct: float | None = None
    hy_oas_change_20d_bps: float | None = None
    fed_funds_rate: float | None = None
    ecb_deposit_rate: float | None = None
    vix: float | None = None
    us_unemployment_pct: float | None = None
    spy_5d_pct: float | None = None
    qqq_5d_pct: float | None = None
    ftse_5d_pct: float | None = None
    is_stale: bool = False

    def to_text(self) -> str:
        parts = []
        if self.gbp_usd is not None:
            parts.append(f"GBP/USD: {self.gbp_usd:.4f}")
        if self.eur_gbp is not None:
            parts.append(f"EUR/GBP: {self.eur_gbp:.4f}")
        if self.us_10y_yield is not None:
            chg_s = (
                f"(20d {self.us_10y_change_20d_bps:+.0f}bp)"
                if self.us_10y_change_20d_bps is not None
                else ""
            )
            parts.append(f"10Y美债 {self.us_10y_yield:.2f}%{chg_s}")
        if self.us_2y_yield is not None:
            parts.append(f"美2Y国债: {self.us_2y_yield:.2f}%")
        if self.yield_spread_10y_2y is not None:
            parts.append(f"10Y-2Y利差: {self.yield_spread_10y_2y:+.2f}%")
        if self.breakeven_10y_pct is not None:
            parts.append(f"美10Y通胀预期: {self.breakeven_10y_pct:.2f}%")
        if self.hy_oas_spread_pct is not None:
            oas_chg = (
                f"(20d {self.hy_oas_change_20d_bps:+.0f}bp)"
                if self.hy_oas_change_20d_bps is not None
                else ""
            )
            parts.append(f"高收益债利差 {self.hy_oas_spread_pct:.2f}%{oas_chg}")
        if self.fed_funds_rate is not None:
            parts.append(f"Fed利率: {self.fed_funds_rate:.2f}%")
        if self.ecb_deposit_rate is not None:
            parts.append(f"ECB利率: {self.ecb_deposit_rate:.2f}%")
        if self.vix is not None:
            parts.append(f"VIX恐慌指数: {self.vix:.2f}")
        if self.us_unemployment_pct is not None:
            parts.append(f"美失业率: {self.us_unemployment_pct:.1f}%")
        idx_parts = []
        if self.spy_5d_pct is not None:
            idx_parts.append(f"标普(SPY)5日 {self.spy_5d_pct:+.2f}%")
        if self.qqq_5d_pct is not None:
            idx_parts.append(f"纳指(QQQ)5日 {self.qqq_5d_pct:+.2f}%")
        if self.ftse_5d_pct is not None:
            idx_parts.append(f"富时(ISF.L)5日 {self.ftse_5d_pct:+.2f}%")
        if idx_parts:
            parts.append(" | ".join(idx_parts))
        if not parts:
            return "（宏观与汇率快照暂不可用）"
        text = " | ".join(parts)
        return f"[陈旧缓存] {text}" if self.is_stale else text


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
            err = f"company-news {type(exc).__name__}: {exc}"
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
            err = f"calendar/earnings {type(exc).__name__}: {exc}"
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
            self.last_error = f"quote {type(exc).__name__}: {exc}"
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
            self.last_error = f"profile2 {type(exc).__name__}: {exc}"
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
            self.last_error = f"recommendation {type(exc).__name__}: {exc}"
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
            self.last_error = f"earnings {type(exc).__name__}: {exc}"
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
            self.last_error = f"insider-sentiment {type(exc).__name__}: {exc}"
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
            err = f"FRED {series_id} {type(exc).__name__}: {exc}"
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


class MarketDataService:
    """Async market data aggregator combining yfinance, Finnhub, and FRED with bounded TTL caching."""

    def __init__(
        self,
        finnhub_key: str = "",
        fred_key: str = "",
        ttl_seconds: int = 900,
        max_concurrency: int = 6,
        ops_notifier: Any = None,
        max_cache_entries: int = 256,
        deep_intel_ttl_seconds: int = 21600,
        db: Any = None,
    ) -> None:
        self.finnhub = FinnhubClient(finnhub_key)
        self.fred = FredClient(fred_key)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_cache_entries = max(1, int(max_cache_entries))
        self.deep_intel_ttl_seconds = max(60, int(deep_intel_ttl_seconds))
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ops = ops_notifier
        self._db = db

        self._hist_cache: OrderedDict[tuple[str, str], tuple[float, pd.DataFrame]] = OrderedDict()
        self._bundle_cache: OrderedDict[str, tuple[float, TickerMarketBundle]] = OrderedDict()
        self._deep_intel_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._macro_cache: tuple[float, MacroSnapshot] | None = None

        self._inflight_hist: dict[tuple[str, str], asyncio.Future[pd.DataFrame]] = {}
        self._inflight_bundle: dict[tuple[str, bool], asyncio.Future[tuple[pd.DataFrame, TickerMarketBundle]]] = {}
        self._inflight_macro: asyncio.Future[MacroSnapshot] | None = None

        self._finnhub_http: tuple[Any, httpx.AsyncClient] | None = None
        self._fred_http: tuple[Any, httpx.AsyncClient] | None = None
        self._init_yf_tz_cache()

    def set_ops_notifier(self, ops_notifier: Any) -> None:
        self._ops = ops_notifier

    def set_db(self, db: Any) -> None:
        self._db = db

    async def _notify_ops(self, severity: str, alert_key: str, message: str) -> None:
        if self._ops is not None:
            await self._ops.notify(severity=severity, source="market_data", alert_key=alert_key, message=message)

    @staticmethod
    def _init_yf_tz_cache() -> None:
        """Pre-create yfinance cache directory once to avoid thread race FileExistsError."""
        try:
            cache_dir = Path.home() / ".cache" / "py-yfinance"
            cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_dir))
        except Exception as exc:
            logger.debug("Skipping yfinance tz cache setup: %s", exc)

    async def _get_http_client(self, target: str) -> httpx.AsyncClient:
        """Lazily create/reuse httpx.AsyncClient, rebuilding automatically if test transport changes."""
        if target == "finnhub":
            transport = self.finnhub._transport
            if self._finnhub_http is not None:
                cached_tr, client = self._finnhub_http
                if cached_tr is transport and not client.is_closed:
                    return client
                await client.aclose()
            client = httpx.AsyncClient(transport=transport)
            self._finnhub_http = (transport, client)
            return client

        transport = self.fred._transport
        if self._fred_http is not None:
            cached_tr, client = self._fred_http
            if cached_tr is transport and not client.is_closed:
                return client
            await client.aclose()
        client = httpx.AsyncClient(transport=transport)
        self._fred_http = (transport, client)
        return client

    async def close(self) -> None:
        """Gracefully release internal HTTP client connection pools."""
        for attr in ("_finnhub_http", "_fred_http"):
            pair = getattr(self, attr, None)
            if pair is not None:
                _, client = pair
                if not client.is_closed:
                    await client.aclose()
                setattr(self, attr, None)

    def _prune_ordered_cache(self, cache: OrderedDict[Any, tuple[float, Any]], now: float, ttl: int) -> None:
        expired_keys = [k for k, (ts, _) in cache.items() if ttl <= 0 or (now - ts) >= ttl]
        for k in expired_keys:
            cache.pop(k, None)
        while len(cache) > self.max_cache_entries:
            cache.popitem(last=False)

    def _get_cached_hist(self, symbol: str, period: str, now: float, *, allow_stale: bool = False) -> pd.DataFrame | None:
        candidate_keys = [(symbol, period)]
        if period == "1y":
            candidate_keys.append((symbol, "2y"))
        elif allow_stale and period == "2y":
            candidate_keys.append((symbol, "1y"))
        for key in candidate_keys:
            entry = self._hist_cache.get(key)
            if entry is None:
                continue
            ts, df = entry
            if df.empty:
                continue
            if allow_stale or (self.ttl_seconds > 0 and (now - ts) < self.ttl_seconds):
                self._hist_cache.move_to_end(key)
                return df
        return None

    def _get_cached_hist_fallback(
        self,
        symbol: str,
        now: float,
        periods: tuple[str, ...] = ("2y", "1y"),
        *,
        allow_stale: bool = False,
    ) -> pd.DataFrame | None:
        """Return the first cached history DataFrame across periods without ambiguous DataFrame truthiness."""
        for period in periods:
            cached = self._get_cached_hist(symbol, period, now, allow_stale=allow_stale)
            if cached is not None:
                return cached
        return None

    def _set_cached_hist(self, symbol: str, period: str, now: float, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        key = (symbol, period)
        self._hist_cache[key] = (now, df)
        self._hist_cache.move_to_end(key)
        # Keep stale entries up to 2x TTL for error fallback unless capacity exceeded
        self._prune_ordered_cache(self._hist_cache, now, max(self.ttl_seconds * 4, 3600))

    def _get_cached_bundle(
        self,
        symbol: str,
        now: float,
        *,
        require_info: bool = True,
        allow_stale: bool = False,
    ) -> TickerMarketBundle | None:
        entry = self._bundle_cache.get(symbol)
        if entry is None:
            return None
        ts, bundle = entry
        if require_info and not bundle.short_name:
            return None
        if allow_stale or (self.ttl_seconds > 0 and (now - ts) < self.ttl_seconds):
            self._bundle_cache.move_to_end(symbol)
            return bundle
        return None

    def _set_cached_bundle(self, symbol: str, now: float, bundle: TickerMarketBundle) -> None:
        if not bundle.has_history and not bundle.short_name and bundle.current_price is None:
            return
        self._bundle_cache[symbol] = (now, bundle)
        self._bundle_cache.move_to_end(symbol)
        self._prune_ordered_cache(self._bundle_cache, now, max(self.ttl_seconds * 4, 3600))

    @staticmethod
    def _run_yf_call_with_retry(fn: Any, *, max_retries: int = 2, base_delay: float = 0.8) -> Any:
        """Retry transient yfinance rate limits or temporary failures with fresh Ticker instantiation."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except YFRateLimitError as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(tok in msg for tok in ("429", "too many requests", "timed out", "timeout", "temporarily"))
                if transient and attempt < max_retries:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        return None

    def _fetch_yf_sync(
        self,
        symbol: str,
        period: str = "1y",
        include_info: bool = True,
        include_news: bool = True,
        include_earnings: bool = True,
        preloaded_hist: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, TickerMarketBundle]:
        if preloaded_hist is not None and not preloaded_hist.empty:
            hist = _sanitize_history_df(preloaded_hist)
        else:
            def _load_hist() -> pd.DataFrame | None:
                return yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)

            hist = _sanitize_history_df(self._run_yf_call_with_retry(_load_hist))

        bundle = TickerMarketBundle(symbol=symbol, has_history=not hist.empty)
        if not hist.empty and "Close" in hist.columns:
            bundle.current_price = float(hist["Close"].iloc[-1])

        if not include_info:
            return hist, bundle

        info: dict[str, Any] = {}
        try:
            def _load_info() -> dict[str, Any]:
                t = yf.Ticker(symbol)
                data = t.info or {}
                return data if isinstance(data, dict) else {}

            info = self._run_yf_call_with_retry(_load_info) or {}
            bundle.quote_type = _clean_text(info.get("quoteType"))
            bundle.currency = _clean_text(info.get("currency"))
            bundle.short_name = _clean_text(info.get("shortName")) or _clean_text(info.get("longName")) or symbol
            bundle.long_name = _clean_text(info.get("longName")) or bundle.short_name
            bundle.exchange = _clean_text(info.get("exchange")) or _clean_text(info.get("fullExchangeName"))
            bundle.sector = _clean_text(info.get("sector"))
            bundle.industry = _clean_text(info.get("industry"))
            bundle.isin = _clean_text(info.get("isin")).upper()

            if bundle.current_price is None:
                bundle.current_price = (
                    _safe_float(info.get("currentPrice"))
                    or _safe_float(info.get("regularMarketPrice"))
                    or _safe_float(info.get("previousClose"))
                )
            bundle.trailing_pe = _safe_float(info.get("trailingPE"))
            bundle.forward_pe = _safe_float(info.get("forwardPE"))
            bundle.market_cap = _safe_float(info.get("marketCap"))
            bundle.dividend_yield_pct = _resolve_dividend_yield_pct(info, bundle.current_price, bundle.currency)
            bundle.target_mean_price = _safe_float(info.get("targetMeanPrice"))
            bundle.beta = _safe_float(info.get("beta"))
            bundle.profit_margin_pct = _ratio_to_pct(info.get("profitMargins"))
            bundle.revenue_growth_pct = _ratio_to_pct(info.get("revenueGrowth"))
            bundle.roe_pct = _ratio_to_pct(info.get("returnOnEquity"))
            bundle.debt_to_equity = _safe_float(info.get("debtToEquity"))
            bundle.analyst_count = _safe_int(info.get("numberOfAnalystOpinions"))
            bundle.recommendation_key = _clean_text(info.get("recommendationKey"))

            ex_div_str, ex_div_delta = _parse_date_days_ahead(info.get("exDividendDate"))
            if ex_div_str and (ex_div_delta is None or ex_div_delta >= -3):
                bundle.ex_dividend_date = ex_div_str
                bundle.days_to_ex_dividend = ex_div_delta
        except Exception as exc:
            logger.debug("yfinance info failed for %s: %s", symbol, exc)

        # Fallback earnings & ex-dividend date from yfinance calendar
        if include_earnings and bundle.quote_type.upper() != "ETF":
            try:
                cal = yf.Ticker(symbol).calendar
                if isinstance(cal, dict):
                    if cal.get("Earnings Date"):
                        ed_list = cal["Earnings Date"]
                        first_ed = ed_list[0] if isinstance(ed_list, (list, tuple)) and ed_list else ed_list
                        ed_str, delta = _parse_date_days_ahead(first_ed)
                        if ed_str and (delta is None or delta >= -2):
                            bundle.earnings_date = ed_str
                            bundle.days_to_earnings = delta
                    if bundle.ex_dividend_date is None and cal.get("Ex-Dividend Date"):
                        ex_str, ex_delta = _parse_date_days_ahead(cal["Ex-Dividend Date"])
                        if ex_str and (ex_delta is None or ex_delta >= -3):
                            bundle.ex_dividend_date = ex_str
                            bundle.days_to_ex_dividend = ex_delta
            except Exception:
                pass

        # Fallback news from yfinance (helpful for UK/EU stocks & ETFs)
        if include_news:
            try:
                raw_news = yf.Ticker(symbol).news or []
                parsed_news: list[dict[str, str]] = []
                for item in raw_news[:3]:
                    content = item.get("content") if isinstance(item.get("content"), dict) else item
                    title = str(content.get("title") or "").strip()
                    if not title:
                        continue
                    pub = str(content.get("pubDate") or "")[:10]
                    provider = ""
                    if isinstance(content.get("provider"), dict):
                        provider = str(content["provider"].get("displayName") or "")
                    parsed_news.append({"date": pub, "source": provider or "Yahoo", "title": title})
                bundle.news = parsed_news
            except Exception:
                pass

        bundle.identity_summary = bundle.build_identity_anchor()
        return hist, bundle

    @staticmethod
    def _extract_batch_frames(
        raw: pd.DataFrame | None,
        batch_syms: list[str],
        *,
        intraday: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Split yf.download output (MultiIndex or single-symbol flat DataFrame) per ticker."""
        out: dict[str, pd.DataFrame] = {}
        if raw is None or raw.empty:
            return out
        sanitizer = _sanitize_intraday_df if intraday else _sanitize_history_df

        if not isinstance(raw.columns, pd.MultiIndex):
            if len(batch_syms) == 1:
                cleaned = sanitizer(raw)
                if not cleaned.empty:
                    out[batch_syms[0]] = cleaned
            return out

        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        for sym in batch_syms:
            sub: pd.DataFrame | None = None
            if sym in lvl0:
                sub = raw[sym]
            elif sym in lvl1:
                sub = raw.xs(sym, axis=1, level=1)
            if sub is not None and not sub.empty:
                cleaned = sanitizer(sub)
                if not cleaned.empty:
                    out[sym] = cleaned
        return out

    def _download_batch_sync(self, batch_syms: list[str], period: str) -> dict[str, pd.DataFrame]:
        if not batch_syms:
            return {}
        try:
            def _do_download() -> pd.DataFrame:
                return yf.download(
                    tickers=batch_syms if len(batch_syms) > 1 else batch_syms[0],
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

            raw = self._run_yf_call_with_retry(_do_download, max_retries=1)
            return self._extract_batch_frames(raw, batch_syms)
        except Exception as exc:
            logger.debug("yfinance batch download failed (%d syms): %s", len(batch_syms), exc)
            return {}

    async def _download_histories_raw(self, symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
        """Download daily OHLCV for symbols in chunks of 30 with single-ticker fallback."""
        batch_map: dict[str, pd.DataFrame] = {}
        chunk_size = 30
        for idx in range(0, len(symbols), chunk_size):
            chunk = symbols[idx : idx + chunk_size]
            async with self._sem:
                chunk_res = await asyncio.to_thread(self._download_batch_sync, chunk, period)
            batch_map.update(chunk_res)

        missing = [s for s in symbols if s not in batch_map or batch_map[s].empty]
        if missing:
            async def _fallback_one(sym: str) -> tuple[str, pd.DataFrame]:
                try:
                    async with self._sem:
                        df, _ = await asyncio.to_thread(
                            self._fetch_yf_sync,
                            sym,
                            period,
                            False,
                            False,
                            False,
                        )
                    return sym, df
                except Exception as exc:
                    logger.debug("Fallback history fetch failed for %s: %s", sym, exc)
                    return sym, pd.DataFrame()

            fb_pairs = await asyncio.gather(*(_fallback_one(s) for s in missing))
            for sym, df in fb_pairs:
                if not df.empty:
                    batch_map[sym] = df
        return batch_map

    async def get_histories(self, symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
        """Batch-fetch daily OHLCV DataFrames with SQLite incremental merge, split detection & singleflight."""
        now = time.monotonic()
        unique_syms = list(dict.fromkeys(s.strip() for s in symbols if s and s.strip()))
        results: dict[str, pd.DataFrame] = {}
        wait_futures: dict[str, asyncio.Future[pd.DataFrame]] = {}
        to_fetch: list[str] = []
        owned_futures: dict[str, asyncio.Future[pd.DataFrame]] = {}

        loop = asyncio.get_running_loop()
        for sym in unique_syms:
            cached = self._get_cached_hist(sym, period, now)
            if cached is not None:
                results[sym] = cached
                continue
            inflight = self._inflight_hist.get((sym, period))
            if inflight is not None:
                wait_futures[sym] = inflight
                continue
            fut: asyncio.Future[pd.DataFrame] = loop.create_future()
            self._inflight_hist[(sym, period)] = fut
            owned_futures[sym] = fut
            to_fetch.append(sym)

        if to_fetch:
            try:
                batch_map: dict[str, pd.DataFrame] = {}
                local_df_map: dict[str, pd.DataFrame] = {}
                db_period_map: dict[str, str] = {}
                inc_fetch_syms: list[str] = []
                full_fetch_syms: list[str] = []
                to_persist: dict[str, tuple[str, pd.DataFrame]] = {}

                if self._db is not None and hasattr(self._db, "get_cached_ohlcv") and period in ("1y", "2y"):
                    db_cache: dict[str, dict[str, Any]] = {}
                    try:
                        db_cache = await self._db.get_cached_ohlcv(to_fetch, period=period)
                        if period == "1y":
                            miss_db = [s for s in to_fetch if s not in db_cache]
                            if miss_db:
                                db_cache.update(await self._db.get_cached_ohlcv(miss_db, period="2y"))
                    except Exception as exc:
                        logger.debug("SQLite ohlcv_cache lookup failed: %s", exc)
                        db_cache = {}

                    today_utc = datetime.now(timezone.utc).date()
                    for sym in to_fetch:
                        row = db_cache.get(sym)
                        if row:
                            local_df = _deserialize_ohlcv_json(str(row.get("payload_json") or ""))
                            if not local_df.empty:
                                local_df_map[sym] = local_df
                                db_period_map[sym] = str(row.get("period") or period)
                                last_bar_date = pd.Timestamp(local_df.index[-1]).date()
                                if (today_utc - last_bar_date).days <= 5:
                                    inc_fetch_syms.append(sym)
                                    continue
                        full_fetch_syms.append(sym)
                else:
                    full_fetch_syms = list(to_fetch)

                if inc_fetch_syms:
                    inc_map = await self._download_histories_raw(inc_fetch_syms, "1mo")
                    for sym in inc_fetch_syms:
                        local_df = local_df_map[sym]
                        inc_df = inc_map.get(sym, pd.DataFrame())
                        if inc_df.empty:
                            batch_map[sym] = local_df
                            continue
                        overlap_idx = local_df.index.intersection(inc_df.index)
                        if len(overlap_idx) == 0:
                            full_fetch_syms.append(sym)
                            continue
                        max_rel_diff = 0.0
                        for ts in overlap_idx:
                            c_old = float(local_df.loc[ts, "Close"])
                            c_new = float(inc_df.loc[ts, "Close"])
                            if c_old > 0:
                                rel_diff = abs(c_new - c_old) / c_old
                                if rel_diff > max_rel_diff:
                                    max_rel_diff = rel_diff
                        if max_rel_diff <= 0.005:
                            merged_df = _sanitize_history_df(pd.concat([local_df, inc_df]))
                            if not merged_df.empty:
                                batch_map[sym] = merged_df
                                to_persist[sym] = (db_period_map.get(sym, period), merged_df)
                            else:
                                full_fetch_syms.append(sym)
                        else:
                            logger.info(
                                "Split/corporate adjustment detected for %s (overlap diff=%.2f%% > 0.5%%); refetching full %s.",
                                sym,
                                max_rel_diff * 100.0,
                                period,
                            )
                            full_fetch_syms.append(sym)

                full_fetch_syms = list(dict.fromkeys(full_fetch_syms))
                if full_fetch_syms:
                    full_map = await self._download_histories_raw(full_fetch_syms, period)
                    for sym in full_fetch_syms:
                        df = full_map.get(sym, pd.DataFrame())
                        if not df.empty:
                            batch_map[sym] = df
                            to_persist[sym] = (period, df)
                        elif sym in local_df_map and not local_df_map[sym].empty:
                            batch_map[sym] = local_df_map[sym]

                if to_persist and self._db is not None and hasattr(self._db, "save_cached_ohlcv") and period in ("1y", "2y"):
                    entries = [
                        ent
                        for sym, (save_p, df) in to_persist.items()
                        if (ent := _serialize_ohlcv_df(sym, save_p, df)) is not None
                    ]
                    if entries:
                        try:
                            await self._db.save_cached_ohlcv(entries)
                        except Exception as exc:
                            logger.warning("Failed persisting ohlcv_cache entries: %s", exc)

                fetch_now = time.monotonic()
                for sym in to_fetch:
                    df = batch_map.get(sym, pd.DataFrame())
                    if not df.empty:
                        if sym in to_persist or self._db is None:
                            self._set_cached_hist(sym, period, fetch_now, df)
                        results[sym] = df
                    else:
                        stale = self._get_cached_hist(sym, period, fetch_now, allow_stale=True)
                        df = stale if stale is not None else pd.DataFrame()
                        results[sym] = df
                    fut = owned_futures[sym]
                    if not fut.done():
                        fut.set_result(df)
            except Exception as exc:
                for sym, fut in owned_futures.items():
                    if not fut.done():
                        fut.set_exception(exc)
                raise
            finally:
                for sym in to_fetch:
                    self._inflight_hist.pop((sym, period), None)

        if wait_futures:
            waited = await asyncio.gather(*wait_futures.values(), return_exceptions=True)
            for sym, val in zip(wait_futures.keys(), waited):
                if isinstance(val, pd.DataFrame):
                    results[sym] = val
                else:
                    results[sym] = pd.DataFrame()

        return results

    async def get_history(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        clean_sym = symbol.strip()
        if not clean_sym:
            return pd.DataFrame()
        hist_map = await self.get_histories([clean_sym], period=period)
        return hist_map.get(clean_sym, pd.DataFrame())

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
        isin_map: dict[str, str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle]]:
        now = time.monotonic()
        unique_syms = list(dict.fromkeys(s.strip() for s in symbols if s and s.strip()))
        histories: dict[str, pd.DataFrame] = {}
        bundles: dict[str, TickerMarketBundle] = {}
        failed_syms: list[str] = []

        if not include_info and not include_news and not include_earnings:
            histories = await self.get_histories(unique_syms, period="2y")
            for sym in unique_syms:
                df = histories.get(sym, pd.DataFrame())
                b = TickerMarketBundle(
                    symbol=sym,
                    isin=_clean_text((isin_map or {}).get(sym)).upper(),
                    has_history=not df.empty,
                    current_price=float(df["Close"].iloc[-1]) if not df.empty else None,
                )
                b.identity_summary = b.build_identity_anchor()
                bundles[sym] = b
                if df.empty:
                    failed_syms.append(f"{sym}(empty)")
            await self._emit_batch_failure_ops(unique_syms, failed_syms)
            return histories, bundles

        if self._db is not None and unique_syms:
            missing_h_syms = [
                s
                for s in unique_syms
                if self._get_cached_hist_fallback(s, now) is None
            ]
            if missing_h_syms:
                await self.get_histories(missing_h_syms, period="2y")
                now = time.monotonic()

        loop = asyncio.get_running_loop()

        async def _load_one(sym: str) -> None:
            h_cached = self._get_cached_hist_fallback(sym, now)
            b_cached = self._get_cached_bundle(sym, now, require_info=include_info)
            if h_cached is not None and b_cached is not None:
                histories[sym] = h_cached
                bundles[sym] = self._apply_isin_override(b_cached, isin_map)
                return

            inflight_key = (sym, include_info)
            inflight = self._inflight_bundle.get(inflight_key)
            if inflight is not None:
                try:
                    hist, bundle = await inflight
                    histories[sym] = hist
                    bundles[sym] = self._apply_isin_override(bundle, isin_map)
                    if hist.empty:
                        failed_syms.append(f"{sym}(empty)")
                except Exception as exc:
                    failed_syms.append(f"{sym}({type(exc).__name__})")
                return

            fut: asyncio.Future[tuple[pd.DataFrame, TickerMarketBundle]] = loop.create_future()
            self._inflight_bundle[inflight_key] = fut
            try:
                async with self._sem:
                    hist, bundle = await asyncio.to_thread(
                        self._fetch_yf_sync,
                        sym,
                        "2y",
                        include_info,
                        include_news,
                        include_earnings,
                        h_cached,
                    )
                fetch_now = time.monotonic()
                if not hist.empty:
                    self._set_cached_hist(sym, "2y", fetch_now, hist)
                else:
                    stale_h = self._get_cached_hist(sym, "2y", fetch_now, allow_stale=True)
                    if stale_h is not None:
                        hist = stale_h
                        bundle.has_history = True
                        if bundle.current_price is None:
                            bundle.current_price = float(stale_h["Close"].iloc[-1])
                    failed_syms.append(f"{sym}(empty)")

                if bundle.has_history or bundle.short_name or bundle.current_price is not None:
                    self._set_cached_bundle(sym, fetch_now, bundle)
                else:
                    stale_b = self._get_cached_bundle(sym, fetch_now, require_info=False, allow_stale=True)
                    if stale_b is not None:
                        bundle = stale_b

                histories[sym] = hist
                bundles[sym] = self._apply_isin_override(bundle, isin_map)
                if not fut.done():
                    fut.set_result((hist, bundle))
            except Exception as exc:
                failed_syms.append(f"{sym}({type(exc).__name__})")
                logger.warning("Yahoo Finance fetch failed for %s: %s", sym, exc)
                stale_h = self._get_cached_hist(sym, "2y", time.monotonic(), allow_stale=True)
                stale_b = self._get_cached_bundle(sym, time.monotonic(), require_info=False, allow_stale=True)
                if stale_h is not None:
                    histories[sym] = stale_h
                if stale_b is not None:
                    bundles[sym] = self._apply_isin_override(stale_b, isin_map)
                if not fut.done():
                    if stale_h is not None and stale_b is not None:
                        fut.set_result((stale_h, stale_b))
                    else:
                        fut.set_exception(exc)
            finally:
                self._inflight_bundle.pop(inflight_key, None)

        await asyncio.gather(*(_load_one(s) for s in unique_syms))
        await self._emit_batch_failure_ops(unique_syms, failed_syms)

        # Enrich stocks with Finnhub earnings calendar, news, and ISIN profile fallback
        if self.finnhub.enabled and (include_news or include_earnings or include_info):
            client = await self._get_http_client("finnhub")
            batch_errors: list[str] = []
            if include_earnings:
                cal, cal_err = await self.finnhub._fetch_earnings_calendar(client, days_ahead=21)
                if cal_err:
                    batch_errors.append(cal_err)
                for sym, b in bundles.items():
                    if b.quote_type.upper() == "ETF":
                        continue
                    if sym in cal:
                        ed_str, eps_raw = cal[sym]
                        ed_iso, delta = _parse_date_days_ahead(ed_str)
                        if ed_iso:
                            b.earnings_date = ed_iso
                            b.days_to_earnings = delta
                        eps_val = _parse_eps_estimate(eps_raw)
                        if eps_val is not None:
                            b.eps_estimate = eps_val

            if include_news:
                us_syms = [s for s in unique_syms if _is_us_symbol(s) and s in bundles]
                news_tasks = [self.finnhub._fetch_company_news(client, s, days=7, limit=3) for s in us_syms]
                news_results = await asyncio.gather(*news_tasks, return_exceptions=True)
                for sym, res in zip(us_syms, news_results):
                    if isinstance(res, tuple):
                        items, err = res
                        if err:
                            batch_errors.append(err)
                        if items:
                            bundles[sym].news = items

            if include_info:
                need_profile = [
                    (sym, b)
                    for sym, b in bundles.items()
                    if b.isin
                    and (
                        (not b.short_name or b.short_name.upper() == sym.upper())
                        or (b.quote_type.upper() != "ETF" and not b.sector)
                    )
                ]
                if need_profile:
                    prof_results = await asyncio.gather(
                        *(self.finnhub.get_profile_by_isin(client, b.isin) for _, b in need_profile),
                        return_exceptions=True,
                    )
                    prof_now = time.monotonic()
                    for (sym, b), prof in zip(need_profile, prof_results):
                        if not isinstance(prof, dict) or not prof:
                            continue
                        changed = False
                        prof_name = _clean_text(prof.get("name"))
                        if prof_name and (not b.short_name or b.short_name.upper() == sym.upper()):
                            b.short_name = prof_name
                            changed = True
                        if prof_name and (not b.long_name or b.long_name.upper() == sym.upper()):
                            b.long_name = prof_name
                            changed = True
                        prof_ind = _clean_text(prof.get("industry"))
                        if prof_ind and not b.sector:
                            b.sector = prof_ind
                            changed = True
                        if prof_ind and not b.industry:
                            b.industry = prof_ind
                            changed = True
                        prof_exch = _clean_text(prof.get("exchange"))
                        if prof_exch and not b.exchange:
                            b.exchange = prof_exch
                            changed = True
                        prof_curr = _clean_text(prof.get("currency"))
                        if prof_curr and not b.currency:
                            b.currency = prof_curr
                            changed = True
                        if changed:
                            b.identity_summary = b.build_identity_anchor()
                            self._set_cached_bundle(sym, prof_now, b)

            self.finnhub.last_error = batch_errors[0] if batch_errors else None
            if batch_errors:
                await self._notify_ops(
                    "WARN",
                    "finnhub_error",
                    f"Finnhub 美股新闻/财历接口异常：{batch_errors[0]}",
                )

        return histories, bundles

    @staticmethod
    def _apply_isin_override(
        bundle: TickerMarketBundle,
        isin_map: dict[str, str] | None,
    ) -> TickerMarketBundle:
        if not isin_map:
            return bundle
        override = _clean_text(isin_map.get(bundle.symbol)).upper()
        if not override or override == bundle.isin:
            return bundle
        return bundle.model_copy(
            update={
                "isin": override,
                "identity_summary": bundle.build_identity_anchor(isin_override=override),
            }
        )

    async def _emit_batch_failure_ops(self, unique_syms: list[str], failed_syms: list[str]) -> None:
        if not failed_syms:
            return
        if len(unique_syms) >= 2 and len(failed_syms) >= len(unique_syms):
            await self._notify_ops(
                "CRITICAL",
                "yfinance_all_failed",
                f"Yahoo Finance 行情批量全部拉取失败（{len(failed_syms)}/{len(unique_syms)}）：{', '.join(failed_syms[:5])}",
            )
        else:
            await self._notify_ops(
                "WARN",
                "yfinance_partial_failed",
                f"部分标的行情从 Yahoo Finance 拉取缺失（{len(failed_syms)}/{len(unique_syms)}）：{', '.join(failed_syms[:5])}",
            )

    async def get_us_realtime_quote(self, symbol: str) -> dict[str, Any] | None:
        """On-demand fast US real-time quote via Finnhub /quote (returns None if disabled or non-US)."""
        if not self.finnhub.enabled or not _is_us_symbol(symbol):
            return None
        client = await self._get_http_client("finnhub")
        return await self.finnhub.get_quote(client, symbol)

    def _download_prepost_15m_sync(self, batch_syms: list[str]) -> dict[str, pd.DataFrame]:
        if not batch_syms:
            return {}
        try:
            def _do_download() -> pd.DataFrame:
                return yf.download(
                    tickers=batch_syms if len(batch_syms) > 1 else batch_syms[0],
                    period="5d",
                    interval="15m",
                    prepost=True,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

            raw = self._run_yf_call_with_retry(_do_download, max_retries=1)
            return self._extract_batch_frames(raw, batch_syms, intraday=True)
        except Exception as exc:
            logger.debug("yfinance prepost 15m batch download failed (%d syms): %s", len(batch_syms), exc)
            return {}

    def _fetch_prepost_15m_single_sync(self, symbol: str) -> pd.DataFrame:
        try:
            def _load() -> pd.DataFrame | None:
                return yf.Ticker(symbol).history(
                    period="5d",
                    interval="15m",
                    prepost=True,
                    auto_adjust=True,
                )

            raw = self._run_yf_call_with_retry(_load, max_retries=1)
            return _sanitize_intraday_df(raw)
        except Exception as exc:
            logger.debug("yfinance prepost 15m single fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _extract_intraday_15m_quote(self, sym: str, df: pd.DataFrame | None) -> dict[str, Any] | None:
        clean = _sanitize_intraday_df(df)
        if clean.empty or "Close" not in clean.columns:
            return None
        closes = clean["Close"].astype(float)
        idx = pd.DatetimeIndex(closes.index)
        et_idx = idx.tz_convert(ZoneInfo("America/New_York")) if idx.tz is not None else idx
        current_price = float(closes.iloc[-1])
        if not math.isfinite(current_price) or current_price <= 0:
            return None

        dates = [ts.date() for ts in et_idx]
        unique_dates = sorted(set(dates))
        prev_close: float | None = None
        if len(unique_dates) >= 2:
            prev_date = unique_dates[-2]
            prev_mask = [d == prev_date for d in dates]
            prev_bars = closes[prev_mask]
            prev_et = [ts for ts, m in zip(et_idx, prev_mask) if m]
            reg_bars = [
                float(val)
                for val, ts in zip(prev_bars, prev_et)
                if 570 <= (ts.hour * 60 + ts.minute) <= 960
            ]
            if reg_bars:
                prev_close = reg_bars[-1]
            elif not prev_bars.empty:
                prev_close = float(prev_bars.iloc[-1])
        else:
            daily_df = self._get_cached_hist_fallback(sym, time.monotonic(), allow_stale=True)
            if daily_df is not None and not daily_df.empty and "Close" in daily_df.columns:
                d_closes = pd.to_numeric(daily_df["Close"], errors="coerce").dropna()
                if not d_closes.empty:
                    d_dates = [pd.Timestamp(ts).date() for ts in d_closes.index]
                    prior = [float(v) for v, d in zip(d_closes, d_dates) if d < unique_dates[0]]
                    if prior:
                        prev_close = prior[-1]
                    elif len(d_closes) >= 2:
                        prev_close = float(d_closes.iloc[-2])
            if prev_close is None:
                prev_close = float(closes.iloc[0])

        if prev_close is None or not math.isfinite(prev_close) or prev_close <= 0:
            return None
        change = current_price - prev_close
        change_pct = ((current_price / prev_close) - 1.0) * 100.0
        return {
            "symbol": sym,
            "current_price": current_price,
            "prev_close": prev_close,
            "change": round(change, 4),
            "change_pct": round(change_pct, 4),
            "source": "yfinance_prepost_15m",
        }

    async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest 15m pre/post-market quotes for US symbols and compute gap vs previous regular close."""
        us_syms = list(
            dict.fromkeys(
                s.strip().upper()
                for s in symbols
                if s and s.strip() and _is_us_symbol(s.strip())
            )
        )
        if not us_syms:
            return {}

        frame_map: dict[str, pd.DataFrame] = {}
        for idx in range(0, len(us_syms), 30):
            chunk = us_syms[idx : idx + 30]
            async with self._sem:
                chunk_res = await asyncio.to_thread(self._download_prepost_15m_sync, chunk)
            frame_map.update(chunk_res)

        missing = [s for s in us_syms if s not in frame_map or frame_map[s].empty]
        if missing:
            async def _fb(sym: str) -> tuple[str, pd.DataFrame]:
                async with self._sem:
                    df = await asyncio.to_thread(self._fetch_prepost_15m_single_sync, sym)
                return sym, df

            for sym, df in await asyncio.gather(*(_fb(s) for s in missing)):
                if not df.empty:
                    frame_map[sym] = df

        out: dict[str, dict[str, Any]] = {}
        for sym in us_syms:
            quote = self._extract_intraday_15m_quote(sym, frame_map.get(sym))
            if quote is not None:
                out[sym] = quote
        return out

    async def get_us_deep_intel(self, symbol: str) -> dict[str, Any]:
        """On-demand cached US analyst recommendation, EPS surprise & insider sentiment (rate-limit safe)."""
        clean_sym = symbol.strip().upper()
        if not clean_sym or not self.finnhub.enabled or not _is_us_symbol(clean_sym):
            return {}
        now = time.monotonic()
        cached = self._deep_intel_cache.get(clean_sym)
        if cached and (now - cached[0]) < self.deep_intel_ttl_seconds:
            self._deep_intel_cache.move_to_end(clean_sym)
            return cached[1]

        client = await self._get_http_client("finnhub")
        recs, surprises, insider = await asyncio.gather(
            self.finnhub.get_recommendation_trends(client, clean_sym),
            self.finnhub.get_earnings_surprises(client, clean_sym, limit=4),
            self.finnhub.get_insider_sentiment(client, clean_sym, days=90),
        )
        intel = {
            "symbol": clean_sym,
            "recommendations": recs,
            "earnings_surprises": surprises,
            "insider_sentiment": insider,
        }
        if recs or surprises or insider:
            self._deep_intel_cache[clean_sym] = (now, intel)
            self._deep_intel_cache.move_to_end(clean_sym)
            self._prune_ordered_cache(self._deep_intel_cache, now, self.deep_intel_ttl_seconds)
        return intel

    async def get_us_deep_intel_batch(
        self, symbols: list[str], max_concurrency: int = 4
    ) -> dict[str, dict[str, Any]]:
        """Batch-fetch Finnhub US deep intel reusing the 6h cache with bounded concurrency."""
        if not self.finnhub.enabled:
            return {}
        clean_syms = list(
            dict.fromkeys(
                s.strip().upper()
                for s in symbols
                if s and s.strip() and _is_us_symbol(s)
            )
        )
        if not clean_syms:
            return {}

        now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for sym in clean_syms:
            cached = self._deep_intel_cache.get(sym)
            if cached and (now - cached[0]) < self.deep_intel_ttl_seconds:
                self._deep_intel_cache.move_to_end(sym)
                if cached[1]:
                    out[sym] = cached[1]
            else:
                missing.append(sym)

        if missing:
            sem = asyncio.Semaphore(max(1, int(max_concurrency)))

            async def _fetch_one(sym: str) -> tuple[str, dict[str, Any]]:
                async with sem:
                    try:
                        return sym, await self.get_us_deep_intel(sym)
                    except Exception as exc:
                        logger.debug("get_us_deep_intel(%s) failed: %s", sym, exc)
                        return sym, {}

            results = await asyncio.gather(*(_fetch_one(sym) for sym in missing))
            for sym, intel in results:
                if intel:
                    out[sym] = intel
        return out

    async def get_macro_snapshot(self) -> MacroSnapshot:
        now = time.monotonic()
        if self._macro_cache and self.ttl_seconds > 0 and (now - self._macro_cache[0]) < self.ttl_seconds:
            return self._macro_cache[1]

        if self._inflight_macro is not None:
            return await self._inflight_macro

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[MacroSnapshot] = loop.create_future()
        self._inflight_macro = fut

        try:
            def _ret_5d(df: pd.DataFrame) -> float | None:
                if df.empty or "Close" not in df.columns:
                    return None
                closes = df["Close"].dropna()
                if len(closes) < 6:
                    return None
                prev = float(closes.iloc[-6])
                if prev <= 0:
                    return None
                return ((float(closes.iloc[-1]) / prev) - 1.0) * 100.0

            def _last_close(df: pd.DataFrame) -> float | None:
                if df.empty or "Close" not in df.columns:
                    return None
                closes = df["Close"].dropna()
                return float(closes.iloc[-1]) if not closes.empty else None

            fx_syms = ["GBPUSD=X", "EURGBP=X", "SPY", "QQQ", "ISF.L", "^VIX"]
            h_map: dict[str, pd.DataFrame] = {}

            async def _pull_hist(sym: str) -> None:
                try:
                    async with self._sem:
                        df, _ = await asyncio.to_thread(
                            self._fetch_yf_sync, sym, "1mo", False, False, False
                        )
                        h_map[sym] = df
                except Exception as exc:
                    logger.warning("Macro history fetch failed for %s: %s", sym, exc)

            async def _pull_fred() -> dict[str, dict[str, float | None]]:
                if not self.fred.enabled:
                    return {}
                client = await self._get_http_client("fred")
                series_ids = [
                    "DGS10",
                    "DFF",
                    "UNRATE",
                    "VIXCLS",
                    "DGS2",
                    "T10Y2Y",
                    "T10YIE",
                    "ECBDFR",
                    "BAMLH0A0HYM2",
                ]
                pairs = await asyncio.gather(
                    *(self.fred._fetch_series_stats(client, sid, limit=25) for sid in series_ids)
                )
                out: dict[str, dict[str, float | None]] = {}
                errs: list[str] = []
                for sid, (stats, err) in zip(series_ids, pairs):
                    out[sid] = stats
                    if err:
                        errs.append(err)
                self.fred.last_error = errs[0] if errs else None
                if (
                    errs
                    and out.get("DGS10", {}).get("latest") is None
                    and out.get("DFF", {}).get("latest") is None
                    and out.get("UNRATE", {}).get("latest") is None
                ):
                    await self._notify_ops(
                        "WARN",
                        "fred_error",
                        f"FRED 宏观利率接口异常（{errs[0]}），美债收益率/Fed利率暂缺。",
                    )
                return out

            _, fred_stats = await asyncio.gather(
                asyncio.gather(*(_pull_hist(s) for s in fx_syms)),
                _pull_fred(),
            )

            dgs10_stats = fred_stats.get("DGS10", {})
            us10y = dgs10_stats.get("latest")
            dgs10_chg20 = dgs10_stats.get("change_20d")
            us2y = fred_stats.get("DGS2", {}).get("latest")
            spread_10_2 = fred_stats.get("T10Y2Y", {}).get("latest")
            if spread_10_2 is None and us10y is not None and us2y is not None:
                spread_10_2 = round(us10y - us2y, 4)
            hy_oas_stats = fred_stats.get("BAMLH0A0HYM2", {})
            hy_oas_chg20 = hy_oas_stats.get("change_20d")

            snap = MacroSnapshot(
                fetched_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                gbp_usd=_last_close(h_map.get("GBPUSD=X", pd.DataFrame())),
                eur_gbp=_last_close(h_map.get("EURGBP=X", pd.DataFrame())),
                us_10y_yield=us10y,
                us_10y_change_20d_bps=round(dgs10_chg20 * 100.0, 1) if dgs10_chg20 is not None else None,
                us_2y_yield=us2y,
                yield_spread_10y_2y=spread_10_2,
                breakeven_10y_pct=fred_stats.get("T10YIE", {}).get("latest"),
                hy_oas_spread_pct=hy_oas_stats.get("latest"),
                hy_oas_change_20d_bps=round(hy_oas_chg20 * 100.0, 1) if hy_oas_chg20 is not None else None,
                fed_funds_rate=fred_stats.get("DFF", {}).get("latest"),
                ecb_deposit_rate=fred_stats.get("ECBDFR", {}).get("latest"),
                vix=_last_close(h_map.get("^VIX", pd.DataFrame())) or fred_stats.get("VIXCLS", {}).get("latest"),
                us_unemployment_pct=fred_stats.get("UNRATE", {}).get("latest"),
                spy_5d_pct=_ret_5d(h_map.get("SPY", pd.DataFrame())),
                qqq_5d_pct=_ret_5d(h_map.get("QQQ", pd.DataFrame())),
                ftse_5d_pct=_ret_5d(h_map.get("ISF.L", pd.DataFrame())),
            )
            if snap.gbp_usd is None and snap.vix is None and snap.spy_5d_pct is None:
                await self._notify_ops(
                    "CRITICAL",
                    "macro_data_unavailable",
                    "宏观与外汇（GBPUSD / VIX / SPY）核心行情全部获取失败，请检查网络或 Yahoo Finance 连通性。",
                )
                if self._macro_cache is not None:
                    stale_snap = self._macro_cache[1].model_copy(update={"is_stale": True})
                    if not fut.done():
                        fut.set_result(stale_snap)
                    return stale_snap
                if not fut.done():
                    fut.set_result(snap)
                return snap

            self._macro_cache = (time.monotonic(), snap)
            if not fut.done():
                fut.set_result(snap)
            return snap
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight_macro = None
