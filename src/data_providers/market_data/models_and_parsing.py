"""Market data models (TickerMarketBundle, MacroSnapshot), OHLCV sanitization/serialization, and parsers."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from src.redaction import redact_secrets

logger = logging.getLogger("src.data_providers.market_data")

_MINOR_UNIT_CURRENCIES = {"GBp", "GBX", "ZAc", "ILA"}
_PLACEHOLDER_STRINGS = {"none", "n/a", "na", "nan", "null", "-"}


def _safe_exc(exc: BaseException) -> str:
    """Format an exception for user-facing error text with credential fragments masked."""
    return f"{type(exc).__name__}: {redact_secrets(str(exc))}"

# IMP-17: yfinance-independent Yahoo transport (same vendor, no new key).
_YAHOO_CHART_HOSTS: tuple[str, ...] = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
_YAHOO_CHART_PERIODS: dict[str, str] = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "max": "max",
}
_YAHOO_CHART_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


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


# v5.5.1: FCF Yield sanity band. Cross-currency ADRs (JPY earnings / USD market cap)
# can produce absurd ratios; anything outside -50% ~ +50% is treated as bad data.
_FCF_YIELD_MIN_RATIO: float = -0.50
_FCF_YIELD_MAX_RATIO: float = 0.50


def _compute_safe_fcf_yield_pct(free_cashflow: Any, market_cap: Any) -> float | None:
    """Compute free-cash-flow yield (%) with currency-scale guards.

    Returns ``None`` when either input is missing/non-finite, when market cap is not
    positive, or when the raw ratio falls outside the sane ``-50% ~ +50%`` band.
    """
    fcf = _safe_float(free_cashflow)
    mcap = _safe_float(market_cap)
    if fcf is None or mcap is None or mcap <= 0.0:
        return None
    ratio = fcf / mcap
    if not math.isfinite(ratio) or ratio < _FCF_YIELD_MIN_RATIO or ratio > _FCF_YIELD_MAX_RATIO:
        return None
    return round(ratio * 100.0, 2)


def _pick_peg_ratio(info: dict[str, Any]) -> float | None:
    """Pick a sane PEG ratio from yfinance info (accepts 0 < PEG <= 50)."""
    peg_val = _safe_float(info.get("pegRatio"))
    if peg_val is None:
        peg_val = _safe_float(info.get("trailingPegRatio"))
    if peg_val is not None and 0.0 < peg_val <= 50.0:
        return round(peg_val, 3)
    return None


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


def _fetch_yahoo_chart_sync(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Direct Yahoo v8 chart fetch, used only when yfinance itself fails.

    Same vendor, independent transport: bypasses yfinance session/crumb/429 failures
    without adding an API key or a new vendor. Returns a standard OHLCV frame
    (Open/High/Low/Close/Volume/Adj Close) with a tz-naive daily index, or empty.
    """
    clean = (symbol or "").strip()
    if not clean:
        return pd.DataFrame()
    rng = _YAHOO_CHART_PERIODS.get(str(period or "").strip().lower(), "2y")
    params = {"range": rng, "interval": "1d", "includeAdjustedClose": "true", "events": "div,splits"}
    headers = {"User-Agent": _YAHOO_CHART_UA, "Accept": "application/json"}

    for host in _YAHOO_CHART_HOSTS:
        url = f"https://{host}/v8/finance/chart/{quote(clean, safe='')}"
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                response = client.get(url, params=params, headers=headers)
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception as exc:  # network / TLS / JSON errors: try the next host
            logger.debug("Yahoo chart fallback failed for %s via %s: %s", clean, host, exc)
            continue

        try:
            results = ((payload or {}).get("chart") or {}).get("result") or []
            if not results:
                continue
            node = results[0] or {}
            stamps = node.get("timestamp") or []
            quote_block = (((node.get("indicators") or {}).get("quote")) or [{}])[0] or {}
            if not stamps or not quote_block:
                continue
            index = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(None).normalize()
            frame = pd.DataFrame(
                {
                    "Open": quote_block.get("open"),
                    "High": quote_block.get("high"),
                    "Low": quote_block.get("low"),
                    "Close": quote_block.get("close"),
                    "Volume": quote_block.get("volume"),
                },
                index=index,
            )
            adj_block = (((node.get("indicators") or {}).get("adjclose")) or [{}])[0] or {}
            adj_values = adj_block.get("adjclose")
            if adj_values is not None and len(adj_values) == len(frame):
                frame["Adj Close"] = adj_values
            frame = frame.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
            if frame.empty:
                continue
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            logger.info("Yahoo chart fallback recovered %d bars for %s via %s", len(frame), clean, host)
            return frame
        except Exception as exc:
            logger.debug("Yahoo chart payload parse failed for %s via %s: %s", clean, host, exc)
            continue

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
    price_to_book: float | None = None
    ev_ebitda: float | None = None
    current_ratio: float | None = None
    metrics_source: str = ""
    analyst_count: int | None = None
    recommendation_key: str = ""
    earnings_date: str | None = None
    days_to_earnings: int | None = None
    eps_estimate: float | None = None
    ex_dividend_date: str | None = None
    days_to_ex_dividend: int | None = None
    # v5.5.1 dual-philosophy fundamentals (all optional -> old cached JSON stays valid)
    fcf_yield_pct: float | None = None
    target_high_price: float | None = None
    target_low_price: float | None = None
    peg_ratio: float | None = None
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


# IMP-18: Finnhub /stock/metric field aliases (TTM / Annual / Quarterly naming drifts).
_FINNHUB_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "price_to_book": ("pb", "pbAnnual", "priceToBookRatio"),
    "ev_ebitda": ("evEbitdaTTM", "evEbitda", "enterpriseValueOverEBITDATTM"),
    "current_ratio": ("currentRatio", "currentRatioAnnual", "currentRatioQuarterly"),
    "beta": ("beta",),
    # v5.5.1: P/FCF ratio -> converted to FCF yield % when yfinance has no free cashflow
    "pfcf": ("pfcfShareTTM", "pfcfShareAnnual", "priceToFreeCashFlowTTM"),
}


def _pick_metric(metrics: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    """Return the first finite positive numeric value found under any alias key."""
    for key in aliases:
        try:
            value = float(metrics.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def apply_basic_financials(bundle: TickerMarketBundle, metrics: dict[str, Any] | None) -> bool:
    """Merge Finnhub basic-financial metrics into a bundle. Returns True when anything changed.

    Only fills gaps (never overwrites a vendor value already present) and only accepts
    positive finite numbers, so a vendor ``0``/null never masquerades as a real ratio.
    """
    if not isinstance(metrics, dict) or not metrics:
        return False
    changed = False

    if bundle.price_to_book is None:
        pb = _pick_metric(metrics, _FINNHUB_METRIC_ALIASES["price_to_book"])
        if pb is not None:
            bundle.price_to_book = round(pb, 3)
            changed = True
    if bundle.ev_ebitda is None:
        ev = _pick_metric(metrics, _FINNHUB_METRIC_ALIASES["ev_ebitda"])
        if ev is not None:
            bundle.ev_ebitda = round(ev, 3)
            changed = True
    if bundle.current_ratio is None:
        ratio = _pick_metric(metrics, _FINNHUB_METRIC_ALIASES["current_ratio"])
        if ratio is not None:
            bundle.current_ratio = round(ratio, 3)
            changed = True
    if bundle.beta is None:
        beta = _pick_metric(metrics, _FINNHUB_METRIC_ALIASES["beta"])
        if beta is not None:
            bundle.beta = round(beta, 3)
            changed = True
    if bundle.fcf_yield_pct is None:
        pfcf = _pick_metric(metrics, _FINNHUB_METRIC_ALIASES["pfcf"])
        if pfcf is not None and pfcf >= 2.0:  # <=50% yield sanity guard
            bundle.fcf_yield_pct = round(100.0 / pfcf, 2)
            changed = True
    if changed:
        bundle.metrics_source = "finnhub_metric"
    return changed
