"""Unified market & macro data provider: Yahoo Finance + Finnhub + FRED."""

from __future__ import annotations

import logging
import time

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)

from src.data_providers.market_data.yahoo_mixin import YFRateLimitError, YahooHistoryMixin
from src.data_providers.market_data.models_and_parsing import (
    _MINOR_UNIT_CURRENCIES,
    _PLACEHOLDER_STRINGS,
    _safe_exc,
    _YAHOO_CHART_HOSTS,
    _YAHOO_CHART_PERIODS,
    _YAHOO_CHART_UA,
    _clean_text,
    _safe_float,
    _safe_int,
    _ratio_to_pct,
    _FCF_YIELD_MIN_RATIO,
    _FCF_YIELD_MAX_RATIO,
    _compute_safe_fcf_yield_pct,
    _pick_peg_ratio,
    _is_us_symbol,
    _parse_date_days_ahead,
    _resolve_dividend_yield_pct,
    normalize_history_index,
    _sanitize_history_df,
    _sanitize_intraday_df,
    _parse_eps_estimate,
    _serialize_ohlcv_df,
    _deserialize_ohlcv_json,
    _fetch_yahoo_chart_sync,
    TickerMarketBundle,
    MacroSnapshot,
    _FINNHUB_METRIC_ALIASES,
    _pick_metric,
    apply_basic_financials,
)
from src.data_providers.market_data.external_clients import (
    FinnhubClient,
    FredClient,
)
from src.data_providers.market_data.service import (
    MarketDataService,
)

__all__ = [
    "YFRateLimitError",
    "normalize_history_index",
    "TickerMarketBundle",
    "MacroSnapshot",
    "apply_basic_financials",
    "FinnhubClient",
    "FredClient",
    "MarketDataService",
]
