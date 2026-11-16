"""Tests for Trading 212 ticker mapping, API client, retry/403 degradation, and trade summary."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    PortfolioData,
    PositionItem,
    T212Client,
    build_trade_context,
    clean_t212_ticker,
    format_account_overview,
    to_yahoo_symbol,
)
from src.storage.db import AdvisorDatabase


class TestT212Client(unittest.IsolatedAsyncioTestCase):
    def test_ticker_cleaning_and_yahoo_mapping(self) -> None:
        self.assertEqual(clean_t212_ticker("AAPL_US_EQ"), "AAPL")
        self.assertEqual(clean_t212_ticker("BRK_B_US_EQ"), "BRK_B")
        self.assertEqual(clean_t212_ticker("VUAGl_EQ"), "VUAG")
        self.assertEqual(clean_t212_ticker("VWRPl_EQ"), "VWRP")
        self.assertEqual(clean_t212_ticker("IBTAl_EQ"), "IBTA")
        self.assertEqual(clean_t212_ticker("ASMLa_EQ"), "ASML")
        self.assertEqual(clean_t212_ticker("RR_LN_EQ"), "RR")
        self.assertEqual(clean_t212_ticker("FB_US_EQ"), "META")
        self.assertEqual(clean_t212_ticker("FB_LN_EQ"), "FB")

        self.assertEqual(to_yahoo_symbol("AAPL_US_EQ", "USD"), "AAPL")
        self.assertEqual(to_yahoo_symbol("FB_US_EQ", "USD"), "META")
        self.assertEqual(to_yahoo_symbol("FB_US_EQ", "USD", clean_ticker="FB"), "META")
        self.assertEqual(to_yahoo_symbol("FB_LN_EQ", "GBX"), "FB.L")
        self.assertEqual(to_yahoo_symbol("BRK_B_US_EQ", "USD"), "BRK-B")
        self.assertEqual(to_yahoo_symbol("VUAGl_EQ", "GBP"), "VUAG.L")
        self.assertEqual(to_yahoo_symbol("VWRPl_EQ", "GBP"), "VWRP.L")
        self.assertEqual(to_yahoo_symbol("IBTAl_EQ", "USD"), "IBTA.L")
        self.assertEqual(to_yahoo_symbol("ASMLa_EQ", "EUR"), "ASML.AS")
        self.assertEqual(to_yahoo_symbol("SAPd_EQ", "EUR"), "SAP.DE")
        self.assertEqual(to_yahoo_symbol("MCp_EQ", "EUR"), "MC.PA")
        self.assertEqual(to_yahoo_symbol("RR_LN_EQ", "GBX"), "RR.L")
        self.assertEqual(to_yahoo_symbol("ASML_NL_EQ", "EUR"), "ASML.AS")

    async def test_fetch_portfolio_pagination_and_gbx_fallback(self) -> None:
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat().replace("+00:00", "Z")
        call_counts = {"summary": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            is_isa = "isa_key" in str(request.headers.get("Authorization", "")) or True
            # Distinguish via base64 prefix of key
            auth = request.headers.get("Authorization", "")
            is_invest = "aW52ZXN0" in auth  # base64("invest:...") starts with aW52ZXN0

            if path.endswith("/equity/account/summary"):
                call_counts["summary"] += 1
                if call_counts["summary"] == 1:
                    return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"error": "rate limit"})
                return httpx.Response(
                    200,
                    json={
                        "currency": "GBP",
                        "totalValue": 12000.0 if not is_invest else 5000.0,
                        "cash": {"availableToTrade": 2000.0, "reservedForOrders": 500.0, "inPies": 0.0},
                        "investments": {
                            "currentValue": 9500.0 if not is_invest else 2500.0,
                            "totalCost": 8500.0 if not is_invest else 2600.0,
                            "unrealizedProfitLoss": 1000.0 if not is_invest else -100.0,
                            "realizedProfitLoss": 300.0,
                        },
                    },
                )

            if path.endswith("/equity/positions"):
                if is_invest:
                    # Include a GBX position without walletImpact to test 0.01 scale fallback
                    return httpx.Response(
                        200,
                        json=[
                            {
                                "instrument": {"ticker": "RR_LN_EQ", "name": "Rolls-Royce", "currency": "GBX"},
                                "quantity": 100,
                                "averagePricePaid": 400.0,
                                "currentPrice": 500.0,
                            }
                        ],
                    )
                return httpx.Response(
                    200,
                    json=[
                        {
                            "instrument": {"ticker": "VUAG_EQ", "name": "Vanguard S&P 500", "currency": "GBP"},
                            "quantity": 50,
                            "averagePricePaid": 80.0,
                            "currentPrice": 90.0,
                            "walletImpact": {
                                "currency": "GBP",
                                "currentValue": 4500.0,
                                "totalCost": 4000.0,
                                "unrealizedProfitLoss": 500.0,
                                "fxImpact": 0.0,
                            },
                        },
                        {
                            "instrument": {"ticker": "NVDA_US_EQ", "name": "NVIDIA", "currency": "USD"},
                            "quantity": 40,
                            "averagePricePaid": 110.0,
                            "currentPrice": 135.0,
                            "walletImpact": {
                                "currency": "GBP",
                                "currentValue": 5000.0,
                                "totalCost": 4500.0,
                                "unrealizedProfitLoss": 500.0,
                                "fxImpact": -80.0,
                            },
                        },
                    ],
                )

            if path.endswith("/equity/orders"):
                if is_invest:
                    return httpx.Response(403, json={"error": "forbidden"})
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "ord-1",
                            "ticker": "MSFT_US_EQ",
                            "side": "BUY",
                            "type": "LIMIT",
                            "quantity": 2,
                            "limitPrice": 400.0,
                            "currency": "USD",
                            "timeInForce": "GTC",
                        },
                        {
                            "id": "ord-2",
                            "ticker": "NOW_US_EQ",
                            "side": "SELL",
                            "type": "STOP",
                            "quantity": 15,
                            "stopPrice": 93.80,
                            "currency": "GBP",
                            "timeInForce": "GTC",
                        },
                    ],
                )

            if path.endswith("/equity/history/orders"):
                if "cursor=p2" in str(request.url):
                    return httpx.Response(
                        200,
                        json={
                            "items": [
                                {
                                    "order": {"id": "o-old", "ticker": "AAPL_US_EQ", "side": "BUY", "quantity": 5},
                                    "fill": {"id": "f-old", "type": "TRADE", "filledAt": old_ts, "quantity": 5, "price": 180.0},
                                }
                            ]
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "order": {
                                    "id": "o-1",
                                    "ticker": "NVDA_US_EQ",
                                    "side": "BUY",
                                    "type": "LIMIT",
                                    "currency": "USD",
                                    "instrument": {"ticker": "NVDA_US_EQ", "name": "NVIDIA", "currency": "USD"},
                                },
                                "fill": {
                                    "id": "f-1",
                                    "type": "TRADE",
                                    "filledAt": recent_ts,
                                    "quantity": 10,
                                    "price": 125.0,
                                    "walletImpact": {
                                        "currency": "GBP",
                                        "netValue": 980.0,
                                        "realisedProfitLoss": 0.0,
                                        "taxes": [{"currency": "GBP", "quantity": 1.47}],
                                    },
                                },
                            },
                            {
                                "order": {"id": "o-split", "ticker": "NVDA_US_EQ"},
                                "fill": {"id": "f-split", "type": "STOCK_SPLIT", "filledAt": recent_ts, "quantity": 90},
                            },
                        ],
                        "nextPagePath": "/api/v0/equity/history/orders?cursor=p2",
                    },
                )

            return httpx.Response(404)

        client = T212Client(
            isa_key="isa",
            isa_secret="sec",
            invest_key="invest",
            invest_secret="sec",
            transport=httpx.MockTransport(handler),
        )
        portfolio = await asyncio.wait_for(client.get_portfolio(use_cache=False), timeout=10.0)

        self.assertTrue(portfolio.isa.ok)
        self.assertTrue(portfolio.invest.ok)
        self.assertEqual(portfolio.total_combined_value, 17000.0)
        self.assertEqual(portfolio.isa.positions[0].ticker, "NVDA")
        self.assertEqual(portfolio.isa.positions[1].yahoo_symbol, "VUAG.L")
        # Invest GBX 0.01 fallback: 100 * 500p = £500 currentValue, £400 cost -> £100 profit
        self.assertAlmostEqual(portfolio.invest.positions[0].current_value, 500.0, places=2)
        self.assertAlmostEqual(portfolio.invest.positions[0].pnl, 100.0, places=2)
        self.assertFalse(portfolio.invest.pending_orders_supported)  # 403 degraded cleanly
        self.assertEqual(len(portfolio.isa.pending_orders), 2)
        now_order = [o for o in portfolio.isa.pending_orders if o.ticker == "NOW"][0]
        self.assertEqual(now_order.currency, "USD")
        self.assertIn("MSFT", portfolio.unique_yahoo_symbols(include_pending=True))
        self.assertNotIn("MSFT", portfolio.unique_yahoo_symbols(include_pending=False))
        # Only TRADE fill within 30 days kept (STOCK_SPLIT & 45d-old fill filtered)
        self.assertEqual(len(portfolio.isa.recent_orders), 1)
        self.assertAlmostEqual(portfolio.isa.recent_orders[0].tax_total, 1.47, places=2)

        bundles = {
            "MSFT": TickerMarketBundle(symbol="MSFT", currency="USD", current_price=420.0, target_mean_price=480.0),
            "NOW": TickerMarketBundle(symbol="NOW", currency="USD", current_price=102.25),
        }
        ctx = build_trade_context(portfolio, bundles=bundles)
        self.assertIn("NVDA", ctx["recent_trades_text"])
        self.assertIn("MSFT", ctx["pending_orders_text"])
        self.assertIn("止损 93.80 USD", ctx["pending_orders_text"])
        self.assertNotIn("93.80 GBP", ctx["pending_orders_text"])
        self.assertIn("市场现价 420.00", ctx["pending_orders_text"])
        await client.close()

    async def test_connection_pool_reuse_close_and_position_created_at(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/equity/account/summary"):
                return httpx.Response(
                    200,
                    json={
                        "currency": "GBP",
                        "totalValue": 1000.0,
                        "cash": {"availableToTrade": 200.0, "reservedForOrders": 0.0, "inPies": 0.0},
                        "investments": {"currentValue": 800.0, "totalCost": 700.0, "unrealizedProfitLoss": 100.0},
                    },
                )
            if path.endswith("/equity/positions"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "instrument": {"ticker": "AAPL_US_EQ", "name": "Apple", "currency": "USD"},
                            "createdAt": "2025-09-10T14:30:00Z",
                            "quantity": 4,
                            "averagePricePaid": 200.0,
                            "currentPrice": 225.0,
                            "walletImpact": {"currency": "GBP", "currentValue": 700.0, "totalCost": 600.0, "unrealizedProfitLoss": 100.0},
                        },
                        {
                            "instrument": {"ticker": "MSFT_US_EQ", "name": "Microsoft", "currency": "USD"},
                            "createdAt": "",
                            "quantity": 1,
                            "averagePricePaid": 100.0,
                            "currentPrice": 100.0,
                            "walletImpact": {"currency": "GBP", "currentValue": 100.0, "totalCost": 100.0, "unrealizedProfitLoss": 0.0},
                        },
                    ],
                )
            if path.endswith("/equity/orders"):
                return httpx.Response(200, json=[])
            if path.endswith("/equity/history/orders"):
                return httpx.Response(200, json={"items": []})
            return httpx.Response(404)

        client = T212Client(
            isa_key="isa",
            isa_secret="sec",
            transport=httpx.MockTransport(handler),
        )
        p1 = await client.get_portfolio(use_cache=False)
        first_http_client = client._client
        self.assertIsNotNone(first_http_client)
        self.assertFalse(first_http_client.is_closed)

        p2 = await client.get_portfolio(use_cache=False)
        self.assertIs(client._client, first_http_client)
        self.assertEqual(p1.isa.positions[0].created_at, "2025-09-10T14:30:00Z")
        self.assertIsNone(p1.isa.positions[1].created_at)
        self.assertTrue(p2.isa.ok)

        await client.close()
        self.assertIsNone(client._client)
        self.assertTrue(first_http_client.is_closed)

        # Calling again after close transparently creates a fresh client
        p3 = await client.get_portfolio(use_cache=False)
        self.assertTrue(p3.isa.ok)
        self.assertIsNotNone(client._client)
        self.assertIsNot(client._client, first_http_client)
        await client.close()

    async def test_ratelimit_headers_429_and_proactive_pacing(self) -> None:
        summary_calls = 0
        history_times: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal summary_calls
            path = request.url.path
            if path.endswith("/equity/account/summary"):
                summary_calls += 1
                if summary_calls == 1:
                    # 429 with x-ratelimit-reset ~0.12s from now (should override longer Retry-After)
                    reset_ts = f"{time.time() + 0.12:.3f}"
                    return httpx.Response(
                        429,
                        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset_ts, "Retry-After": "9.0"},
                        json={"error": "rate limit"},
                    )
                return httpx.Response(
                    200,
                    headers={"x-ratelimit-remaining": "1", "x-ratelimit-reset": str(int(time.time() + 5))},
                    json={"currency": "GBP", "totalValue": 500.0, "cash": {"availableToTrade": 500.0}},
                )
            if path.endswith("/equity/positions"):
                return httpx.Response(200, json=[])
            if path.endswith("/equity/orders"):
                return httpx.Response(200, json=[])
            if path.endswith("/equity/history/orders"):
                history_times.append(time.monotonic())
                if "cursor=p2" not in str(request.url):
                    # Page 1 succeeds with remaining=0 -> page 2 must wait until x-ratelimit-reset
                    reset_ts = f"{time.time() + 0.15:.3f}"
                    return httpx.Response(
                        200,
                        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset_ts},
                        json={"items": [], "nextPagePath": "/api/v0/equity/history/orders?cursor=p2"},
                    )
                return httpx.Response(200, json={"items": []})
            return httpx.Response(404)

        client = T212Client(
            isa_key="isa",
            isa_secret="sec",
            transport=httpx.MockTransport(handler),
        )
        t0 = time.monotonic()
        portfolio = await asyncio.wait_for(client.get_portfolio(use_cache=False), timeout=5.0)
        elapsed = time.monotonic() - t0
        self.assertTrue(portfolio.isa.ok)
        self.assertEqual(summary_calls, 2)
        self.assertEqual(len(history_times), 2)
        # Proactive pacing waited for x-ratelimit-reset between page 1 and page 2
        self.assertGreaterEqual(history_times[1] - history_times[0], 0.10)
        # Total time stayed fast (< 2.0s), proving x-ratelimit-reset was used instead of 9.0s Retry-After
        self.assertLess(elapsed, 2.0)
        await client.close()

    async def test_5xx_memory_fallback_recovery_and_non_transient_401_guard(self) -> None:
        class _RecorderOps:
            def __init__(self) -> None:
                self.events: list[tuple[str, str, str]] = []

            async def notify(self, severity: str, source: str, alert_key: str, message: str) -> None:
                self.events.append((severity, alert_key, message))

        mode = {"status": 200}

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if mode["status"] == 503:
                return httpx.Response(503, json={"error": "upstream down"})
            if mode["status"] == 401:
                return httpx.Response(401, json={"error": "bad key"})
            if path.endswith("/equity/account/summary"):
                return httpx.Response(
                    200,
                    json={
                        "currency": "GBP",
                        "totalValue": 2500.0,
                        "cash": {"availableToTrade": 500.0, "reservedForOrders": 0.0, "inPies": 0.0},
                        "investments": {"currentValue": 2000.0, "totalCost": 1800.0, "unrealizedProfitLoss": 200.0},
                    },
                )
            if path.endswith("/equity/positions"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "instrument": {"ticker": "NVDA_US_EQ", "name": "NVIDIA", "currency": "USD"},
                            "createdAt": "2025-11-01T15:00:00Z",
                            "quantity": 10,
                            "averagePricePaid": 120.0,
                            "currentPrice": 140.0,
                            "walletImpact": {"currency": "GBP", "currentValue": 2000.0, "totalCost": 1800.0, "unrealizedProfitLoss": 200.0},
                        }
                    ],
                )
            if path.endswith("/equity/orders"):
                return httpx.Response(200, json=[])
            if path.endswith("/equity/history/orders"):
                return httpx.Response(200, json={"items": []})
            return httpx.Response(404)

        ops = _RecorderOps()
        client = T212Client(
            isa_key="isa",
            isa_secret="sec",
            cache_ttl_seconds=60,
            transport=httpx.MockTransport(handler),
            ops_notifier=ops,
            transient_backoff_s=0.01,
        )

        # 1. Initial healthy fetch
        live = await client.get_portfolio(use_cache=False)
        self.assertTrue(live.isa.ok)
        self.assertFalse(live.is_fallback)
        orig_ts = live.fetched_at_uk

        # 2. Upstream hits 503 -> retries & falls back to memory cache with explicit timestamp
        mode["status"] = 503
        fb = await asyncio.wait_for(client.get_portfolio(use_cache=False), timeout=8.0)
        self.assertTrue(fb.isa.ok)
        self.assertTrue(fb.is_fallback)
        self.assertEqual(fb.fallback_source, "memory")
        self.assertEqual(fb.fetched_at_uk, orig_ts)
        self.assertIn(orig_ts, fb.isa.cash_status_note)
        overview = format_account_overview(fb.isa)
        self.assertIn("回退缓存快照", overview)
        self.assertIn(orig_ts, overview)
        self.assertTrue(any(sev == "WARN" and key == "t212_fallback:ISA" for sev, key, _ in ops.events))

        # 3. Upstream recovers -> even use_cache=True fetches live immediately (fallback did not poison cache)
        mode["status"] = 200
        recovered = await client.get_portfolio(use_cache=True)
        self.assertTrue(recovered.isa.ok)
        self.assertFalse(recovered.is_fallback)

        # 4. 401 auth error is NOT transient -> must NOT fall back, must emit CRITICAL
        mode["status"] = 401
        failed = await client.get_portfolio(use_cache=False)
        self.assertFalse(failed.isa.ok)
        self.assertFalse(failed.is_fallback)
        self.assertTrue(any(sev == "CRITICAL" and key == "t212_all_accounts_failed" for sev, key, _ in ops.events))
        await client.close()

    async def test_cold_start_db_snapshot_fallback_and_summary_5xx_cash_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = AdvisorDatabase(Path(tmp) / "advisor.db")
            await db.init()
            saved = PortfolioData(
                fetched_at_uk="2026-11-14 07:30",
                fetched_at_ms=1763105400000,
                total_combined_value=12000.0,
                total_combined_cash=2500.0,
                isa=AccountSnapshot(
                    ok=True,
                    account_label="ISA",
                    total_account_value=12000.0,
                    investments_value=9500.0,
                    investments_cost=8500.0,
                    unrealized_pnl=1000.0,
                    total_cash=2500.0,
                    available_to_trade=2500.0,
                    cash_ratio_pct=20.83,
                    positions=[
                        PositionItem(
                            account="ISA",
                            raw_ticker="AAPL_US_EQ",
                            ticker="AAPL",
                            yahoo_symbol="AAPL",
                            name="Apple",
                            inst_currency="USD",
                            quantity=10,
                            avg_price=200.0,
                            cur_price=220.0,
                            current_value=9500.0,
                            total_cost=8500.0,
                            pnl=1000.0,
                            fx_impact=0.0,
                            return_pct=11.76,
                            weight_pct=79.17,
                            created_at="2025-05-01T10:00:00Z",
                        )
                    ],
                ),
                invest=AccountSnapshot(ok=False, account_label="Invest", error="Invest API credentials not configured."),
            )
            await db.save_snapshot(saved)

            # Cold-start client hits ConnectError on T212 -> falls back to SQLite latest snapshot
            async def failing_handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("network down", request=request)

            cold_client = T212Client(
                isa_key="isa",
                isa_secret="sec",
                transport=httpx.MockTransport(failing_handler),
                snapshot_loader=db.get_latest_snapshot,
                transient_backoff_s=0.01,
            )
            fb = await asyncio.wait_for(cold_client.get_portfolio(use_cache=False), timeout=8.0)
            self.assertTrue(fb.isa.ok)
            self.assertTrue(fb.is_fallback)
            self.assertEqual(fb.fallback_source, "db")
            self.assertEqual(fb.fetched_at_uk, "2026-11-14 07:30")
            self.assertIn("2026-11-14 07:30 UK", fb.isa.cash_status_note)
            self.assertEqual(fb.isa.positions[0].created_at, "2025-05-01T10:00:00Z")
            await cold_client.close()

            # Summary-only 502 after a good fetch preserves cached cash instead of zeroing to 0.0
            summary_fail = {"flag": False}

            async def partial_handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if path.endswith("/equity/account/summary"):
                    if summary_fail["flag"]:
                        return httpx.Response(502, json={"error": "bad gateway"})
                    return httpx.Response(
                        200,
                        json={
                            "currency": "GBP",
                            "totalValue": 3000.0,
                            "cash": {"availableToTrade": 1000.0, "reservedForOrders": 0.0, "inPies": 0.0},
                            "investments": {"currentValue": 2000.0, "totalCost": 1800.0, "unrealizedProfitLoss": 200.0},
                        },
                    )
                if path.endswith("/equity/positions"):
                    return httpx.Response(
                        200,
                        json=[
                            {
                                "instrument": {"ticker": "AAPL_US_EQ", "name": "Apple", "currency": "USD"},
                                "quantity": 10,
                                "averagePricePaid": 180.0,
                                "currentPrice": 200.0,
                                "walletImpact": {"currency": "GBP", "currentValue": 2000.0, "totalCost": 1800.0, "unrealizedProfitLoss": 200.0},
                            }
                        ],
                    )
                if path.endswith("/equity/orders"):
                    return httpx.Response(200, json=[])
                if path.endswith("/equity/history/orders"):
                    return httpx.Response(200, json={"items": []})
                return httpx.Response(404)

            c2 = T212Client(
                isa_key="isa",
                isa_secret="sec",
                transport=httpx.MockTransport(partial_handler),
                transient_backoff_s=0.01,
            )
            ok_snap = await c2.get_portfolio(use_cache=False)
            self.assertEqual(ok_snap.isa.total_cash, 1000.0)
            summary_fail["flag"] = True
            partial_snap = await asyncio.wait_for(c2.get_portfolio(use_cache=False), timeout=8.0)
            self.assertTrue(partial_snap.isa.ok)
            self.assertEqual(partial_snap.isa.total_cash, 1000.0)
            self.assertEqual(partial_snap.isa.total_account_value, 3000.0)
            self.assertIn("现金沿用内存缓存", partial_snap.isa.cash_status_note)
            await c2.close()

    async def test_isin_partially_filled_orders_5xx_indicator_and_cash_events(self) -> None:
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat().replace("+00:00", "Z")

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            auth = request.headers.get("Authorization", "")
            is_invest = "aW52ZXN0" in auth

            if path.endswith("/equity/account/summary"):
                return httpx.Response(
                    200,
                    json={
                        "currency": "GBP",
                        "totalValue": 5000.0,
                        "cash": {"availableToTrade": 1500.0, "reservedForOrders": 0.0, "inPies": 0.0},
                        "investments": {"currentValue": 3500.0, "totalCost": 3000.0, "unrealizedProfitLoss": 500.0},
                    },
                )
            if path.endswith("/equity/positions"):
                if is_invest:
                    return httpx.Response(200, json=[])
                return httpx.Response(
                    200,
                    json=[
                        {
                            "instrument": {
                                "ticker": "VUAGl_EQ",
                                "name": "Vanguard S&P 500",
                                "currency": "GBP",
                                "isin": "IE00BFMXXD54",
                            },
                            "createdAt": "2025-02-01T09:00:00Z",
                            "quantity": 40,
                            "averagePricePaid": 75.0,
                            "currentPrice": 87.5,
                            "walletImpact": {"currency": "GBP", "currentValue": 3500.0, "totalCost": 3000.0, "unrealizedProfitLoss": 500.0},
                        }
                    ],
                )
            if path.endswith("/equity/orders"):
                if is_invest:
                    # Simulate transient 503 on Invest /equity/orders (distinct from 403 disabled)
                    return httpx.Response(503, json={"error": "orders temporary unavailable"})
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "ord-partial",
                            "ticker": "NVDA_US_EQ",
                            "instrument": {"ticker": "NVDA_US_EQ", "name": "NVIDIA", "currency": "USD", "isin": "US67066G1040"},
                            "side": "BUY",
                            "type": "LIMIT",
                            "status": "PARTIALLY_FILLED",
                            "quantity": 10,
                            "filledQuantity": 4,
                            "limitPrice": 130.0,
                            "currency": "GBP",
                            "timeInForce": "GTC",
                        }
                    ],
                )
            if path.endswith("/equity/history/orders"):
                return httpx.Response(200, json={"items": []})
            if path.endswith("/equity/history/transactions"):
                if is_invest:
                    return httpx.Response(403, json={"error": "forbidden"})
                if "cursor=t2" in str(request.url):
                    return httpx.Response(
                        200,
                        json={
                            "items": [
                                {"type": "DEPOSIT", "amount": 500.0, "currency": "GBP", "dateTime": old_ts, "reference": "tx-old"}
                            ]
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {"type": "DEPOSIT", "amount": 1500.0, "currency": "GBP", "dateTime": recent_ts, "reference": "tx-1"},
                            {"type": "WITHDRAW", "amount": 200.0, "currency": "GBP", "dateTime": recent_ts, "reference": "tx-2"},
                            {"type": "INTEREST_ON_FREE_CASH", "amount": 6.5, "currency": "GBP", "dateTime": recent_ts, "reference": "tx-3"},
                        ],
                        "nextPagePath": "/api/v0/equity/history/transactions?cursor=t2",
                    },
                )
            if path.endswith("/equity/history/dividends"):
                if is_invest:
                    return httpx.Response(403, json={"error": "forbidden"})
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "ticker": "AAPL_US_EQ",
                                "instrument": {"ticker": "AAPL_US_EQ", "name": "Apple", "currency": "USD"},
                                "amount": 38.4,
                                "currency": "GBP",
                                "grossAmountPerShare": 0.25,
                                "quantity": 200,
                                "paidOn": recent_ts,
                                "reference": "div-1",
                                "type": "DIVIDENDS_PAID_BY_US_CORPORATIONS",
                            },
                            {
                                "ticker": "MSFT_US_EQ",
                                "amount": 15.0,
                                "currency": "GBP",
                                "paidOn": old_ts,
                                "reference": "div-old",
                            },
                        ]
                    },
                )
            return httpx.Response(404)

        client = T212Client(
            isa_key="isa",
            isa_secret="sec",
            invest_key="invest",
            invest_secret="sec",
            transport=httpx.MockTransport(handler),
            transient_backoff_s=0.01,
        )
        portfolio = await asyncio.wait_for(client.get_portfolio(use_cache=False), timeout=8.0)

        # 1. Position ISIN parsed cleanly
        self.assertEqual(portfolio.isa.positions[0].isin, "IE00BFMXXD54")

        # 2. PARTIALLY_FILLED order uses remaining quantity (10 - 4 = 6), Invest 503 recorded distinctly from 403
        self.assertEqual(len(portfolio.isa.pending_orders), 1)
        self.assertAlmostEqual(portfolio.isa.pending_orders[0].quantity, 6.0)
        self.assertAlmostEqual(portfolio.isa.pending_orders[0].filled_quantity, 4.0)
        self.assertEqual(portfolio.isa.pending_orders[0].status, "PARTIALLY_FILLED")
        self.assertEqual(portfolio.invest.pending_orders_error, "HTTP 503")

        # 3. 30d Transactions & Dividends parsed & old items filtered; Invest 403 degraded cleanly
        self.assertTrue(portfolio.isa.transactions_supported)
        self.assertTrue(portfolio.isa.dividends_supported)
        self.assertFalse(portfolio.invest.transactions_supported)
        self.assertFalse(portfolio.invest.dividends_supported)
        self.assertEqual(len(portfolio.isa.recent_transactions), 3)
        wd = [tx for tx in portfolio.isa.recent_transactions if tx.tx_type == "WITHDRAW"][0]
        self.assertAlmostEqual(wd.amount, -200.0)
        self.assertEqual(len(portfolio.isa.recent_dividends), 1)
        self.assertAlmostEqual(portfolio.isa.recent_dividends[0].amount, 38.4)

        ctx = build_trade_context(portfolio)
        self.assertIn("6.0000股[部分成交,已成4.0000股]", ctx["pending_orders_text"])
        self.assertIn("Invest: HTTP 503", ctx["pending_orders_text"])
        self.assertIn("外部入金 +1,500.00 GBP", ctx["cash_flow_summary_text"])
        self.assertIn("外部出金 -200.00 GBP", ctx["cash_flow_summary_text"])
        self.assertIn("派息入账 +38.40 GBP(1笔)", ctx["cash_flow_summary_text"])
        self.assertIn("现金利息 +6.50 GBP", ctx["cash_flow_summary_text"])
        self.assertIn("[资金流水与派息", ctx["cash_flow_summary_text"])
        self.assertNotIn("[资金流水与派息", ctx["trade_summary_30d_text"])
        await client.close()

        # Verify zero-bloat when no transactions/dividends and only 503 orders on empty account
        empty_p = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", pending_orders_error="HTTP 503"),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )
        empty_ctx = build_trade_context(empty_p)
        self.assertEqual(empty_ctx["cash_flow_summary_text"], "")
        self.assertNotIn("资金流水与派息", empty_ctx["trade_summary_30d_text"])
        self.assertIn("活动挂单接口临时异常", empty_ctx["pending_orders_text"])


if __name__ == "__main__":
    unittest.main()
