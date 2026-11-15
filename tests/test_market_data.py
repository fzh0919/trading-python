"""Offline unit tests for IMP-05 MarketDataService upgrades (batch OHLCV, dividend fix, identity, cache, Finnhub/FRED)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import time
from typing import Any
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd

from src.data_providers.market_data import (
    MarketDataService,
    TickerMarketBundle,
    YFRateLimitError,
)


def _sample_ohlcv(periods: int = 230, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2025-11-01", periods=periods, freq="B")
    close = start_price + np.linspace(0.0, 20.0, periods)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(periods, 1_500_000.0),
        },
        index=dates,
    )


class _DummyOps:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    async def notify(self, *, severity: str, source: str, alert_key: str, message: str) -> bool:
        self.events.append((severity, alert_key, message))
        return True


class TestMarketDataService(unittest.IsolatedAsyncioTestCase):
    async def test_dividend_yield_pct_calibration(self) -> None:
        svc = MarketDataService()
        hist = _sample_ohlcv(30, 120.0)

        cases = {
            # NVDA-like: yfinance >= 0.2.54 reports 0.03 meaning 0.03%, NOT 3.0%
            "NVDA": {
                "quoteType": "EQUITY",
                "currency": "USD",
                "shortName": "NVIDIA Corp",
                "dividendYield": 0.03,
                "trailingAnnualDividendYield": 0.0003,
            },
            # GOOGL-like without trailing ratio: 0.20 stays 0.20%, NOT 20.0%
            "GOOGL": {
                "quoteType": "EQUITY",
                "currency": "USD",
                "shortName": "Alphabet Inc",
                "dividendYield": 0.20,
            },
            # Legacy ratio format: 0.035 matched against trailingAnnualDividendYield=0.035 -> 3.5%
            "LEGACY": {
                "quoteType": "EQUITY",
                "currency": "USD",
                "shortName": "Legacy Corp",
                "dividendYield": 0.035,
                "trailingAnnualDividendYield": 0.035,
            },
            # LSE GBp minor unit: dividendRate in GBP shouldn't distort dividendYield=4.25%
            "AZN.L": {
                "quoteType": "EQUITY",
                "currency": "GBp",
                "shortName": "AstraZeneca PLC",
                "dividendYield": 4.25,
                "dividendRate": 3.5,
                "currentPrice": 11500.0,
            },
            # Missing dividendYield fallback to dividendRate / current_price
            "NODIVY": {
                "quoteType": "EQUITY",
                "currency": "USD",
                "shortName": "No DivY Corp",
                "dividendRate": 2.8,
            },
        }

        class _TickerStub:
            def __init__(self, sym: str) -> None:
                self.sym = sym

            def history(self, **kwargs: Any) -> pd.DataFrame:
                return hist

            @property
            def info(self) -> dict[str, Any]:
                return cases[self.sym]

            @property
            def calendar(self) -> dict[str, Any]:
                return {}

            @property
            def news(self) -> list[dict[str, Any]]:
                return []

        with patch("src.data_providers.market_data.yf.Ticker", side_effect=_TickerStub):
            _, bundles = await svc.get_bundles(
                list(cases.keys()), include_news=False, include_earnings=False
            )

        self.assertAlmostEqual(bundles["NVDA"].dividend_yield_pct or 0.0, 0.03, places=4)
        self.assertAlmostEqual(bundles["GOOGL"].dividend_yield_pct or 0.0, 0.20, places=4)
        self.assertAlmostEqual(bundles["LEGACY"].dividend_yield_pct or 0.0, 3.5, places=4)
        self.assertAlmostEqual(bundles["AZN.L"].dividend_yield_pct or 0.0, 4.25, places=4)
        self.assertAlmostEqual(bundles["NODIVY"].dividend_yield_pct or 0.0, 2.0, places=4)

    async def test_identity_anchor_ex_dividend_quality_and_toggle_flags(self) -> None:
        svc = MarketDataService()
        hist = _sample_ohlcv(20, 100.0)
        ex_dt = datetime.now(timezone.utc) + timedelta(days=7)
        cal_accessed = {"count": 0}
        news_accessed = {"count": 0}

        class _TickerStub:
            def __init__(self, sym: str) -> None:
                self.sym = sym

            def history(self, **kwargs: Any) -> pd.DataFrame:
                return hist

            @property
            def info(self) -> dict[str, Any]:
                if self.sym == "VWRP.L":
                    return {
                        "quoteType": "ETF",
                        "currency": "GBP",
                        "shortName": "Vanguard FTSE All-World",
                        "longName": "Vanguard FTSE All-World UCITS ETF",
                        "exchange": "LSE",
                        "sector": "N/A",
                        "industry": "nan",
                    }
                return {
                    "quoteType": "EQUITY",
                    "currency": "GBp",
                    "shortName": "AstraZeneca",
                    "longName": "AstraZeneca PLC",
                    "exchange": "LSE",
                    "sector": "Healthcare",
                    "industry": "Drug Manufacturers",
                    "beta": 0.45,
                    "profitMargins": 0.18,
                    "revenueGrowth": 0.14,
                    "returnOnEquity": 0.26,
                    "debtToEquity": 68.5,
                    "numberOfAnalystOpinions": 24,
                    "recommendationKey": "buy",
                    "exDividendDate": int(ex_dt.timestamp()),
                }

            @property
            def calendar(self) -> dict[str, Any]:
                cal_accessed["count"] += 1
                return {"Earnings Date": [ex_dt + timedelta(days=10)]}

            @property
            def news(self) -> list[dict[str, Any]]:
                news_accessed["count"] += 1
                return [{"title": "Headline", "pubDate": "2026-11-14"}]

        with patch("src.data_providers.market_data.yf.Ticker", side_effect=_TickerStub):
            _, bundles = await svc.get_bundles(
                ["AZN.L", "VWRP.L"],
                include_news=False,
                include_earnings=False,
                isin_map={"AZN.L": "GB0009895292"},
            )

        self.assertEqual(cal_accessed["count"], 0)
        self.assertEqual(news_accessed["count"], 0)

        azn = bundles["AZN.L"]
        self.assertEqual(azn.exchange, "LSE")
        self.assertEqual(azn.ex_dividend_date, ex_dt.date().isoformat())
        self.assertEqual(azn.days_to_ex_dividend, 7)
        self.assertEqual(azn.isin, "GB0009895292")
        self.assertIn("AstraZeneca PLC", azn.identity_summary)
        self.assertIn("ISIN:GB0009895292", azn.identity_summary)
        self.assertAlmostEqual(azn.beta or 0.0, 0.45, places=2)
        self.assertAlmostEqual(azn.profit_margin_pct or 0.0, 18.0, places=2)
        self.assertAlmostEqual(azn.revenue_growth_pct or 0.0, 14.0, places=2)
        self.assertAlmostEqual(azn.roe_pct or 0.0, 26.0, places=2)
        self.assertAlmostEqual(azn.debt_to_equity or 0.0, 68.5, places=2)
        self.assertEqual(azn.analyst_count, 24)
        self.assertEqual(azn.recommendation_key, "buy")

        vwrp = bundles["VWRP.L"]
        self.assertEqual(vwrp.sector, "")
        self.assertEqual(vwrp.industry, "")
        self.assertIn("ETF", vwrp.identity_summary)
        self.assertNotIn("N/A", vwrp.identity_summary)

    async def test_batch_get_histories_and_cache_reuse_in_get_bundles(self) -> None:
        svc = MarketDataService(ttl_seconds=900)
        nvda_df = _sample_ohlcv(220, 130.0)
        azn_df = _sample_ohlcv(220, 11000.0)
        asml_df = _sample_ohlcv(220, 750.0)
        multi = pd.concat({"NVDA": nvda_df, "AZN.L": azn_df}, axis=1)

        ticker_hist_calls: list[str] = []
        ticker_info_calls: list[str] = []

        class _TickerStub:
            def __init__(self, sym: str) -> None:
                self.sym = sym

            def history(self, **kwargs: Any) -> pd.DataFrame:
                ticker_hist_calls.append(self.sym)
                if self.sym == "ASML.AS":
                    return asml_df
                return nvda_df

            @property
            def info(self) -> dict[str, Any]:
                ticker_info_calls.append(self.sym)
                return {"quoteType": "EQUITY", "currency": "USD", "shortName": f"{self.sym} Name"}

        with (
            patch("src.data_providers.market_data.yf.download", return_value=multi) as dl_mock,
            patch("src.data_providers.market_data.yf.Ticker", side_effect=_TickerStub),
        ):
            hist_map = await svc.get_histories(["NVDA", "AZN.L", "ASML.AS"], period="1y")

            self.assertEqual(dl_mock.call_count, 1)
            self.assertEqual(ticker_info_calls, [])
            # Only ASML.AS missed batch download and fell back to single-ticker history
            self.assertEqual(ticker_hist_calls, ["ASML.AS"])
            self.assertEqual(len(hist_map["NVDA"]), 220)
            self.assertEqual(len(hist_map["AZN.L"]), 220)
            self.assertEqual(len(hist_map["ASML.AS"]), 220)

            # Now get_bundles for NVDA should reuse preloaded 1y history without calling ticker.history()
            ticker_hist_calls.clear()
            bundles_h, bundles = await svc.get_bundles(["NVDA"], include_news=False, include_earnings=False)
            self.assertEqual(ticker_hist_calls, [])
            self.assertEqual(ticker_info_calls, ["NVDA"])
            self.assertEqual(len(bundles_h["NVDA"]), 220)
            self.assertEqual(bundles["NVDA"].short_name, "NVDA Name")

            # Fetching a 1mo window uses separate (symbol, period) cache key and does not overwrite 1y
            nvda_1mo = _sample_ohlcv(20, 140.0)
            with patch("src.data_providers.market_data.yf.download", return_value=nvda_1mo):
                short_df = await svc.get_history("NVDA", period="1mo")
            self.assertEqual(len(short_df), 20)
            self.assertEqual(len(await svc.get_history("NVDA", period="1y")), 220)

    async def test_singleflight_dedup_concurrent_requests(self) -> None:
        svc = MarketDataService(ttl_seconds=900)
        df = _sample_ohlcv(50, 100.0)
        dl_calls = {"count": 0}
        info_calls = {"count": 0}

        def _slow_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
            dl_calls["count"] += 1
            time.sleep(0.03)
            return df

        class _TickerStub:
            def __init__(self, sym: str) -> None:
                self.sym = sym

            def history(self, **kwargs: Any) -> pd.DataFrame:
                time.sleep(0.03)
                return df

            @property
            def info(self) -> dict[str, Any]:
                info_calls["count"] += 1
                return {"quoteType": "EQUITY", "currency": "USD", "shortName": "MSFT"}

        with (
            patch("src.data_providers.market_data.yf.download", side_effect=_slow_download),
            patch("src.data_providers.market_data.yf.Ticker", side_effect=_TickerStub),
        ):
            h1, h2 = await asyncio.gather(
                svc.get_histories(["AAPL"]),
                svc.get_histories(["AAPL"]),
            )
            self.assertEqual(dl_calls["count"], 1)
            self.assertEqual(len(h1["AAPL"]), 50)
            self.assertEqual(len(h2["AAPL"]), 50)

            (_, b1), (_, b2) = await asyncio.gather(
                svc.get_bundles(["MSFT"], include_news=False, include_earnings=False),
                svc.get_bundles(["MSFT"], include_news=False, include_earnings=False),
            )
            self.assertEqual(info_calls["count"], 1)
            self.assertEqual(b1["MSFT"].short_name, "MSFT")
            self.assertEqual(b2["MSFT"].short_name, "MSFT")

    async def test_cache_lru_eviction_no_empty_poisoning_and_stale_fallback(self) -> None:
        svc = MarketDataService(ttl_seconds=900, max_cache_entries=2)
        df_a = _sample_ohlcv(10, 10.0)
        df_b = _sample_ohlcv(10, 20.0)
        df_c = _sample_ohlcv(10, 30.0)

        with (
            patch("src.data_providers.market_data.yf.download", return_value=pd.DataFrame()),
            patch(
                "src.data_providers.market_data.yf.Ticker.history",
                side_effect=[df_a, df_b, df_c, pd.DataFrame()],
            ),
        ):
            await svc.get_histories(["AAA"])
            await svc.get_histories(["BBB"])
            await svc.get_histories(["CCC"])
            self.assertEqual(list(svc._hist_cache.keys()), [("BBB", "1y"), ("CCC", "1y")])

            # Empty fetch must NOT poison cache for DDD
            res_empty = await svc.get_histories(["DDD"])
            self.assertTrue(res_empty["DDD"].empty)
            self.assertNotIn(("DDD", "1y"), svc._hist_cache)

        # Stale-while-error fallback when TTL expired and live Yahoo fails
        svc._hist_cache[("BBB", "1y")] = (time.monotonic() - 5000.0, df_b)
        with (
            patch("src.data_providers.market_data.yf.download", return_value=pd.DataFrame()),
            patch("src.data_providers.market_data.yf.Ticker.history", return_value=pd.DataFrame()),
        ):
            res_stale = await svc.get_histories(["BBB"])
            self.assertEqual(len(res_stale["BBB"]), 10)

    async def test_macro_failure_does_not_poison_cache_and_falls_back_to_stale(self) -> None:
        ops = _DummyOps()
        svc = MarketDataService(ttl_seconds=900, ops_notifier=ops)
        good_df = _sample_ohlcv(10, 1.28)

        with patch("src.data_providers.market_data.yf.Ticker.history", return_value=good_df):
            snap1 = await svc.get_macro_snapshot()
        self.assertIsNotNone(snap1.gbp_usd)
        self.assertFalse(snap1.is_stale)

        # Force TTL expiry, then simulate total Yahoo macro outage
        assert svc._macro_cache is not None
        svc._macro_cache = (time.monotonic() - 5000.0, snap1)
        with patch("src.data_providers.market_data.yf.Ticker.history", return_value=pd.DataFrame()):
            snap2 = await svc.get_macro_snapshot()

        self.assertTrue(snap2.is_stale)
        self.assertIn("[陈旧缓存]", snap2.to_text())
        self.assertAlmostEqual(snap2.gbp_usd or 0.0, snap1.gbp_usd or 0.0, places=4)
        self.assertTrue(any(sev == "CRITICAL" and key == "macro_data_unavailable" for sev, key, _ in ops.events))

    async def test_yf_rate_limit_retry_recreates_ticker_instance(self) -> None:
        svc = MarketDataService()
        good_df = _sample_ohlcv(15, 55.0)
        attempts = {"count": 0}

        class _RetryTicker:
            def __init__(self, sym: str) -> None:
                attempts["count"] += 1
                self.attempt_no = attempts["count"]

            def history(self, **kwargs: Any) -> pd.DataFrame:
                if self.attempt_no == 1:
                    raise YFRateLimitError()
                return good_df

            @property
            def info(self) -> dict[str, Any]:
                if self.attempt_no == 3:
                    raise YFRateLimitError()
                return {"quoteType": "EQUITY", "currency": "USD", "shortName": "NVIDIA Corp"}

        with (
            patch("src.data_providers.market_data.yf.download", return_value=pd.DataFrame()),
            patch("src.data_providers.market_data.yf.Ticker", side_effect=_RetryTicker),
            patch("src.data_providers.market_data.time.sleep", return_value=None),
        ):
            out = await svc.get_history("NVDA", period="1y")
            self.assertEqual(len(out), 15)
            self.assertEqual(attempts["count"], 2)

            # Clear cache or test info retry on a second symbol
            attempts["count"] = 2
            _, bundles = await svc.get_bundles(["AMD"], include_news=False, include_earnings=False)

        self.assertEqual(bundles["AMD"].short_name, "NVIDIA Corp")
        self.assertGreaterEqual(attempts["count"], 4)

    async def test_fred_extended_series_finnhub_error_preservation_and_deep_intel(self) -> None:
        ops = _DummyOps()
        finnhub_calls: list[str] = []

        def _finnhub_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            finnhub_calls.append(f"{path}?{request.url.query.decode()}")
            if path.endswith("/calendar/earnings"):
                return httpx.Response(200, json={"earningsCalendar": [{"symbol": "NVDA", "date": "2026-11-20"}]})
            if path.endswith("/company-news"):
                sym = request.url.params.get("symbol")
                if sym == "AAPL":
                    return httpx.Response(429, json={"error": "rate limit"})
                return httpx.Response(200, json=[{"headline": "NVDA AI demand", "source": "Reuters", "datetime": 1794600000}])
            if path.endswith("/quote"):
                return httpx.Response(200, json={"c": 145.5, "d": 2.5, "dp": 1.75, "h": 146.0, "l": 143.0, "o": 144.0, "pc": 143.0})
            if path.endswith("/stock/profile2"):
                return httpx.Response(200, json={"ticker": "AZN", "name": "AstraZeneca PLC", "exchange": "LSE", "currency": "GBP", "finnhubIndustry": "Pharmaceuticals", "country": "GB"})
            if path.endswith("/stock/recommendation"):
                return httpx.Response(200, json=[{"period": "2026-11-01", "strongBuy": 20, "buy": 15, "hold": 4, "sell": 0, "strongSell": 0}])
            if path.endswith("/stock/earnings"):
                return httpx.Response(200, json=[{"period": "2026-09-30", "actual": 0.81, "estimate": 0.75, "surprisePercent": 8.0}])
            if path.endswith("/stock/insider-sentiment"):
                return httpx.Response(200, json={"data": [{"year": 2026, "month": 10, "change": -1200, "mspr": -15.4}]})
            return httpx.Response(404)

        def _fred_handler(request: httpx.Request) -> httpx.Response:
            sid = request.url.params.get("series_id")
            vals = {
                "DGS10": "4.32",
                "DGS2": "4.10",
                "T10Y2Y": "0.22",
                "T10YIE": "2.35",
                "DFF": "4.50",
                "ECBDFR": "3.25",
                "UNRATE": "4.1",
                "VIXCLS": "16.4",
            }
            return httpx.Response(200, json={"observations": [{"value": vals.get(sid, ".")}]})

        svc = MarketDataService(finnhub_key="fh-test", fred_key="fred-test", ops_notifier=ops)
        svc.finnhub._transport = httpx.MockTransport(_finnhub_handler)
        svc.fred._transport = httpx.MockTransport(_fred_handler)

        hist = _sample_ohlcv(20, 140.0)

        class _TickerStub:
            def __init__(self, sym: str) -> None:
                self.sym = sym

            def history(self, **kwargs: Any) -> pd.DataFrame:
                return hist

            @property
            def info(self) -> dict[str, Any]:
                return {"quoteType": "EQUITY", "currency": "USD", "shortName": self.sym}

            @property
            def calendar(self) -> dict[str, Any]:
                return {}

            @property
            def news(self) -> list[dict[str, Any]]:
                return []

        try:
            with patch("src.data_providers.market_data.yf.Ticker", side_effect=_TickerStub):
                _, bundles = await svc.get_bundles(["AAPL", "NVDA"], include_news=True, include_earnings=True)
                macro = await svc.get_macro_snapshot()

            # Even though NVDA succeeded, AAPL 429 must remain recorded in finnhub.last_error
            self.assertIsNotNone(svc.finnhub.last_error)
            self.assertIn("429", svc.finnhub.last_error or "")
            self.assertEqual(bundles["NVDA"].earnings_date, "2026-11-20")
            self.assertEqual(bundles["NVDA"].news[0]["title"], "NVDA AI demand")

            self.assertAlmostEqual(macro.us_2y_yield or 0.0, 4.10, places=2)
            self.assertAlmostEqual(macro.yield_spread_10y_2y or 0.0, 0.22, places=2)
            self.assertAlmostEqual(macro.breakeven_10y_pct or 0.0, 2.35, places=2)
            self.assertAlmostEqual(macro.ecb_deposit_rate or 0.0, 3.25, places=2)
            self.assertIn("10Y-2Y利差: +0.22%", macro.to_text())
            self.assertIn("ECB利率: 3.25%", macro.to_text())

            quote = await svc.get_us_realtime_quote("NVDA")
            self.assertIsNotNone(quote)
            self.assertAlmostEqual(quote["current_price"], 145.5, places=2)

            client = await svc._get_http_client("finnhub")
            prof = await svc.finnhub.get_profile_by_isin(client, "GB0009895292")
            self.assertEqual(prof["name"], "AstraZeneca PLC")

            calls_before = len(finnhub_calls)
            intel1 = await svc.get_us_deep_intel("NVDA")
            intel2 = await svc.get_us_deep_intel("NVDA")
            self.assertEqual(len(finnhub_calls) - calls_before, 3)
            self.assertEqual(intel1, intel2)
            self.assertEqual(intel1["recommendations"][0]["strong_buy"], 20)
            self.assertAlmostEqual(intel1["earnings_surprises"][0]["surprise_pct"], 8.0, places=2)
            self.assertAlmostEqual(intel1["insider_sentiment"]["mspr"], -15.4, places=2)
        finally:
            await svc.close()


if __name__ == "__main__":
    unittest.main()
