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
import numpy as np
import pandas as pd

from src.agents.prompts import build_wallet_context_text
from src.config import StrategySection
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalDividendItem,
    HistoricalOrderItem,
    HistoricalTransactionItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
    build_trade_context,
    clean_t212_ticker,
    compute_trade_behavior_stats,
    format_account_overview,
    format_positions_text,
    format_trade_behavior_text,
    reconcile_invest_wallets,
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
        reserved_overview = format_account_overview(
            AccountSnapshot(
                ok=True,
                account_label="ISA",
                total_account_value=2000.0,
                total_cash=2000.0,
                available_to_trade=1500.0,
                reserved_cash=500.0,
            )
        )
        self.assertIn("挂单占用(未成交前仍计息): 500.00", reserved_overview)
        self.assertNotIn("资金流水与派息", empty_ctx["trade_summary_30d_text"])
        self.assertIn("活动挂单接口临时异常", empty_ctx["pending_orders_text"])

    async def test_imp20_instruments_metadata_and_dividend_archive_total_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "t212_imp20.db")
            await db.init()

            # Pre-seed an older archived dividend payout (>30d ago, outside recent_dividends API window)
            await db.upsert_dividends(
                [
                    {
                        "id": "div-hist-1",
                        "account": "ISA",
                        "ticker": "VUAG",
                        "yahoo_symbol": "VUAG.L",
                        "amount_gbp": 20.0,
                        "gross_per_share": 0.4,
                        "paid_on": "2026-06-01T10:00:00Z",
                        "paid_on_ms": 1780308000000,
                    }
                ]
            )

            recent_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
            state = {"inst_calls": 0, "fail_positions": False}

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                auth = request.headers.get("Authorization", "")
                is_invest = "aW52ZXN0" in auth

                if path.endswith("/equity/metadata/instruments"):
                    state["inst_calls"] += 1
                    if state["inst_calls"] == 1:
                        return httpx.Response(429, headers={"Retry-After": "0.01"}, json={"error": "rate limit"})
                    return httpx.Response(
                        200,
                        headers={"x-ratelimit-remaining": "0"},
                        json=[
                            {
                                "ticker": "NVDA_US_EQ",
                                "isin": "US67066G1040",
                                "currencyCode": "USD",
                                "shortName": "NVDA",
                                "type": "STOCK",
                                "extendedHours": True,
                                "workingScheduleId": 55,
                            },
                            {
                                "ticker": "VUAGl_EQ",
                                "isin": "IE00BFMXXD54",
                                "currencyCode": "GBP",
                                "shortName": "VUAG",
                                "type": "ETF",
                                "extendedHours": False,
                                "workingScheduleId": 12,
                            },
                        ],
                    )

                if state["fail_positions"]:
                    return httpx.Response(503, json={"error": "upstream unavailable"})

                if is_invest:
                    return httpx.Response(404)

                if path.endswith("/equity/account/summary"):
                    return httpx.Response(
                        200,
                        json={
                            "currency": "GBP",
                            "totalValue": 5500.0,
                            "cash": {"availableToTrade": 1000.0, "reservedForOrders": 0.0, "inPies": 0.0},
                            "investments": {
                                "currentValue": 4500.0,
                                "totalCost": 4000.0,
                                "unrealizedProfitLoss": 500.0,
                            },
                        },
                    )
                if path.endswith("/equity/positions"):
                    return httpx.Response(
                        200,
                        json=[
                            {
                                "instrument": {"ticker": "VUAGl_EQ", "name": "Vanguard S&P 500"},
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
                            }
                        ],
                    )
                if path.endswith("/equity/history/dividends"):
                    return httpx.Response(
                        200,
                        json={
                            "items": [
                                {
                                    "ticker": "VUAGl_EQ",
                                    "amount": 30.0,
                                    "currency": "GBP",
                                    "grossAmountPerShare": 0.6,
                                    "quantity": 50,
                                    "paidOn": recent_ts,
                                    "reference": "div-recent-2",
                                    "type": "ORDINARY",
                                }
                            ]
                        },
                    )
                if path.endswith("/equity/orders") or path.endswith("/equity/history/orders") or path.endswith("/equity/history/transactions"):
                    return httpx.Response(200, json=[] if path.endswith("/equity/orders") else {"items": []})
                return httpx.Response(404)

            client = T212Client(
                isa_key="isa",
                isa_secret="sec",
                transport=httpx.MockTransport(handler),
                transient_backoff_s=0.01,
                db=db,
            )
            try:
                items = await client.fetch_instruments_metadata()
                self.assertEqual(len(items), 2)
                self.assertEqual(state["inst_calls"], 2)
                self.assertIn(("metadata_instruments", "/api/v0/equity/metadata/instruments"), client._rate_limit_resets)

                inst_map = await db.get_t212_instrument_map()
                self.assertEqual(inst_map["VUAGl_EQ"]["yahoo_symbol"], "VUAG.L")
                self.assertTrue(inst_map["NVDA_US_EQ"]["extended_hours"])

                portfolio = await client.get_portfolio(use_cache=False)
                self.assertFalse(portfolio.is_fallback)
                pos = portfolio.isa.positions[0]
                self.assertEqual(pos.yahoo_symbol, "VUAG.L")
                self.assertEqual(pos.isin, "IE00BFMXXD54")
                self.assertAlmostEqual(pos.cumulative_dividends_gbp, 50.0)
                self.assertAlmostEqual(pos.total_return_with_div_pct or 0.0, 13.75, places=2)

                pos_text = format_positions_text(portfolio.isa)
                self.assertIn("累计派息 +50.00 GBP (含息总回报 +13.75%)", pos_text)

                # On 503 fallback, dividend archive must NOT be written
                state["fail_positions"] = True
                upsert_calls = {"count": 0}
                orig_upsert = db.upsert_dividends

                async def _spy_upsert(div_items):
                    upsert_calls["count"] += 1
                    return await orig_upsert(div_items)

                db.upsert_dividends = _spy_upsert  # type: ignore[method-assign]
                fb_portfolio = await client.get_portfolio(use_cache=False)
                self.assertTrue(fb_portfolio.is_fallback)
                self.assertEqual(upsert_calls["count"], 0)
            finally:
                await client.close()

    def test_imp21_pending_order_atr_touch_prob_and_tags(self) -> None:
        from src.data_providers.analytics import compute_technical_snapshot
        from src.data_providers.market_data import TickerMarketBundle

        dates = pd.date_range("2025-11-01", periods=252, freq="B")
        # Controlled ~1% daily range oscillation around 100 -> ATR14 ~ 1.5
        base = 100.0 + np.sin(np.arange(252) / 6.0) * 1.5
        hist_us = pd.DataFrame(
            {
                "Open": base,
                "High": base + 0.8,
                "Low": base - 0.8,
                "Close": base,
                "Volume": np.full(252, 1_000_000.0),
            },
            index=dates,
        )
        # UK stock quoted in GBp (~500p), oscillating around 500p
        base_uk = 500.0 + np.sin(np.arange(252) / 6.0) * 8.0
        hist_uk = pd.DataFrame(
            {
                "Open": base_uk,
                "High": base_uk + 4.0,
                "Low": base_uk - 4.0,
                "Close": base_uk,
                "Volume": np.full(252, 500_000.0),
            },
            index=dates,
        )
        cur_us = float(hist_us["Close"].iloc[-1])
        cur_uk = float(hist_uk["Close"].iloc[-1])
        tech_us = compute_technical_snapshot("NVDA", hist_us)
        tech_uk = compute_technical_snapshot("AZN.L", hist_uk)
        assert tech_us is not None and tech_uk is not None

        # Place UK buy limit right inside CYQ 90% cost band
        uk_cyq_mid = round((float(tech_uk.cyq_cost_90_low or 495.0) + float(tech_uk.cyq_cost_90_high or 505.0)) / 2.0, 2)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-16 10:00",
            fetched_at_ms=1794823200000,
            isa=AccountSnapshot(
                ok=True,
                pending_orders=[
                    PendingOrderItem(
                        id="ord-far",
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        side="BUY",
                        order_type="LIMIT",
                        quantity=2.0,
                        value=160.0,
                        limit_price=round(cur_us * 0.80, 2),  # -20% (>10 ATR, 0% touch prob)
                        currency="USD",
                        time_in_force="GTC",
                    ),
                    PendingOrderItem(
                        id="ord-stop",
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        side="SELL",
                        order_type="STOP",
                        quantity=2.0,
                        value=200.0,
                        stop_price=round(cur_us - 0.3, 2),  # < 0.8 * ATR
                        currency="USD",
                        time_in_force="GTC",
                    ),
                ],
            ),
            invest=AccountSnapshot(
                ok=True,
                pending_orders=[
                    PendingOrderItem(
                        id="ord-cyq",
                        account="Invest",
                        raw_ticker="AZNl_EQ",
                        ticker="AZN",
                        yahoo_symbol="AZN.L",
                        name="AstraZeneca",
                        side="BUY",
                        order_type="LIMIT",
                        quantity=10.0,
                        value=50.0,
                        limit_price=uk_cyq_mid,
                        currency="GBP",  # T212 sometimes labels .L limit order currency as GBP while price is GBp
                        time_in_force="GTC",
                    ),
                ],
            ),
        )
        bundles = {
            "NVDA": TickerMarketBundle(symbol="NVDA", currency="USD", current_price=cur_us),
            "AZN.L": TickerMarketBundle(symbol="AZN.L", currency="GBp", current_price=cur_uk),
        }
        ctx = build_trade_context(
            portfolio,
            bundles=bundles,
            tech_map={"NVDA": tech_us, "AZN.L": tech_uk},
            histories={"NVDA": hist_us, "AZN.L": hist_uk},
        )
        p_txt = ctx["pending_orders_text"]
        self.assertIn("距现价 -20.0%", p_txt)
        self.assertIn("倍ATR", p_txt)
        self.assertIn("20日触价率 0%", p_txt)
        self.assertIn("[远端低概率单]", p_txt)
        self.assertIn("[止损过窄易被日常波动扫出]", p_txt)
        self.assertIn("[位于90%筹码支撑带]", p_txt)


class TestWalletSnapshotCompat(unittest.TestCase):
    """Phase 2/5 compatibility: legacy snapshot JSON, derived split, mismatch and clamping."""

    def test_legacy_snapshot_json_defaults_to_single_wallet(self) -> None:
        payload = {
            "fetched_at_uk": "2026-11-10 09:00",
            "fetched_at_ms": 1762765200000,
            "isa": {"ok": True, "account_label": "ISA", "total_cash": 100.0},
            "invest": {"ok": True, "account_label": "Invest", "total_cash": 200.0},
            "total_combined_value": 300.0,
            "total_combined_cash": 300.0,
        }
        data = PortfolioData.model_validate(payload)
        self.assertFalse(data.invest.multi_currency_enabled)
        self.assertIsNone(data.gbp_usd_rate)
        self.assertEqual(data.invest.usd_cash_total, 0.0)
        self.assertIsNone(data.invest.gbp_wallet_total)

    def test_derived_split_never_changes_gbp_totals(self) -> None:
        snap = AccountSnapshot(ok=True, total_cash=1000.0, available_to_trade=800.0, reserved_cash=200.0)
        updated_ms = int(time.time() * 1000)
        records = {
            "USD": {
                "available_amount": 250.0,
                "reserved_amount": 50.0,
                "updated_at_ms": updated_ms,
                "updated_at_uk": "2026-11-17 08:00",
                "source": "manual",
            }
        }
        out = reconcile_invest_wallets(snap, records, 1.25)
        self.assertEqual(out.usd_cash_total, 300.0)
        self.assertEqual(out.gbp_wallet_total, 760.0)  # 1000 - 300/1.25
        self.assertEqual(out.gbp_wallet_available, 600.0)  # 800 - 250/1.25
        self.assertEqual(out.gbp_wallet_reserved, 160.0)  # 200 - 50/1.25
        self.assertEqual(out.total_cash, 1000.0)
        self.assertEqual(out.wallet_source, "derived_from_total")

    def test_manual_dual_report_mismatch_and_negative_clamp(self) -> None:
        snap = AccountSnapshot(ok=True, total_cash=1000.0, available_to_trade=1000.0, reserved_cash=0.0)
        updated_ms = int(time.time() * 1000)
        dual = {
            "USD": {
                "available_amount": 250.0,
                "reserved_amount": 0.0,
                "updated_at_ms": updated_ms,
                "updated_at_uk": "2026-11-17 08:00",
                "source": "manual",
            },
            "GBP": {
                "available_amount": 500.0,
                "reserved_amount": 0.0,
                "updated_at_ms": updated_ms + 10,
                "updated_at_uk": "2026-11-17 08:01",
                "source": "manual",
            },
        }
        out = reconcile_invest_wallets(snap, dual, 1.25)
        self.assertEqual(out.wallet_source, "manual_both")
        self.assertIn("偏差", out.wallet_reconcile_warning)

        overshoot = {"USD": {**dual["USD"], "available_amount": 3000.0}}
        clamped = reconcile_invest_wallets(snap, overshoot, 1.25)
        self.assertEqual(clamped.gbp_wallet_total, 0.0)
        self.assertIn("负", clamped.wallet_reconcile_warning)

    def test_multi_currency_trade_context_does_not_mix_currencies(self) -> None:
        now_ms = int(time.time() * 1000)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-17 09:00",
            fetched_at_ms=now_ms,
            isa=AccountSnapshot(ok=True, account_label="ISA"),
            invest=AccountSnapshot(
                ok=True,
                account_label="Invest",
                recent_transactions=[
                    HistoricalTransactionItem(
                        id="t1",
                        account="Invest",
                        tx_type="DEPOSIT",
                        amount=500.0,
                        currency="GBP",
                        date_time="2026-11-17T08:00:00Z",
                        date_time_ms=now_ms,
                        uk_time_str="2026-11-17 08:00",
                    ),
                    HistoricalTransactionItem(
                        id="t2",
                        account="Invest",
                        tx_type="INTEREST_ON_FREE_CASH",
                        amount=3.4,
                        currency="USD",
                        date_time="2026-11-17T08:05:00Z",
                        date_time_ms=now_ms + 1000,
                        uk_time_str="2026-11-17 08:05",
                    ),
                ],
            ),
        )
        text = build_trade_context(portfolio, history_days=30)["cash_flow_summary_text"]
        self.assertIn("外部入金 +500.00 GBP", text)
        self.assertIn("现金利息 +3.40 USD", text)
        self.assertNotIn("503.40 GBP", text)

    def test_interest_detail_rows_are_filtered_but_summary_kept(self) -> None:
        """v5.5.6: the daily £1~£2 interest dribble must not crowd out real cash events."""
        now_ms = int(time.time() * 1000)

        def _tx(idx: int, tx_type: str, amount: float, when_ms: int) -> HistoricalTransactionItem:
            return HistoricalTransactionItem(
                id=f"tx{idx}",
                account="Invest",
                tx_type=tx_type,
                amount=amount,
                currency="GBP",
                date_time=datetime.fromtimestamp(when_ms / 1000, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                date_time_ms=when_ms,
                uk_time_str="2026-11-18 08:00",
            )

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-19 09:00",
            fetched_at_ms=now_ms,
            isa=AccountSnapshot(ok=True, account_label="ISA"),
            invest=AccountSnapshot(
                ok=True,
                account_label="Invest",
                recent_transactions=[
                    _tx(1, "DEPOSIT", 2000.0, now_ms - 5000),
                    _tx(2, "TRANSFER", 300.0, now_ms - 4000),
                    _tx(3, "INTEREST_ON_FREE_CASH", 1.23, now_ms - 3000),
                    _tx(4, "LENDING_INTEREST", 0.45, now_ms - 2000),
                ],
                recent_dividends=[
                    HistoricalDividendItem(
                        id="dv1",
                        account="Invest",
                        raw_ticker="ULVRl_EQ",
                        ticker="ULVRl_EQ",
                        yahoo_symbol="ULVR.L",
                        name="Unilever",
                        inst_currency="GBp",
                        amount=12.5,
                        currency="GBP",
                        paid_on="2026-11-18T08:00:00Z",
                        paid_on_ms=now_ms - 1000,
                        uk_time_str="2026-11-18 08:00",
                    )
                ],
            ),
        )
        text = build_trade_context(portfolio, history_days=30)["cash_flow_summary_text"]
        header, _, detail = text.partition("\n")
        # Summary line still carries the full interest total for macro reconciliation.
        self.assertIn("现金利息", header)
        self.assertIn("+1.68 GBP", header)  # 1.23 + 0.45
        # Detail rows keep real cash events, drop the interest noise.
        self.assertIn("入金", detail)
        self.assertIn("划转", detail)
        self.assertIn("派息", detail)
        self.assertNotIn("现金利息", detail)
        self.assertNotIn("借券利息", detail)


class TestPortfolioFxRefresh(unittest.TestCase):
    """v5.5.6: cold-start /report must backfill Invest wallets without another HTTP call."""

    def _pending_fx_portfolio(self, now_ms: int) -> PortfolioData:
        invest = AccountSnapshot(
            ok=True,
            account_label="Invest",
            total_cash=1000.0,
            available_to_trade=800.0,
            reserved_cash=200.0,
        )
        records = {
            "USD": {
                "available_amount": 250.0,
                "reserved_amount": 50.0,
                "updated_at_ms": now_ms,
                "updated_at_uk": "2026-11-19 08:00",
                "source": "manual",
            }
        }
        invest_pending = reconcile_invest_wallets(invest, records, None)
        return PortfolioData(
            fetched_at_uk="2026-11-19 08:01",
            fetched_at_ms=now_ms,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_cash=100.0),
            invest=invest_pending,
            total_combined_value=1100.0,
            total_combined_cash=1100.0,
            gbp_usd_rate=None,
        )

    def test_refresh_backfills_wallets_and_clears_pending_fx(self) -> None:
        now_ms = int(time.time() * 1000)
        portfolio = self._pending_fx_portfolio(now_ms)
        self.assertEqual(portfolio.invest.wallet_source, "pending_fx")
        self.assertIsNone(portfolio.invest.gbp_wallet_total)
        before = build_wallet_context_text(portfolio, StrategySection())
        self.assertIn("暂缺", before)
        self.assertIn("待汇率折算", before)

        client = T212Client(isa_key="a", isa_secret="b", invest_key="c", invest_secret="d")
        client._wallet_records = {
            "USD": {
                "available_amount": 250.0,
                "reserved_amount": 50.0,
                "updated_at_ms": now_ms,
                "updated_at_uk": "2026-11-19 08:00",
                "source": "manual",
            }
        }
        refreshed = client.refresh_portfolio_fx(portfolio, 1.25)

        self.assertEqual(refreshed.gbp_usd_rate, 1.25)
        self.assertEqual(refreshed.invest.gbp_wallet_total, 760.0)  # 1000 - 300/1.25
        self.assertEqual(refreshed.invest.wallet_source, "derived_from_total")
        self.assertEqual(refreshed.invest.total_cash, 1000.0)
        after = build_wallet_context_text(refreshed, StrategySection())
        self.assertIn("GBP/USD 1.2500", after)
        self.assertIn("£760.00", after)
        self.assertNotIn("暂缺", after)
        self.assertNotIn("待汇率折算", after)
        # Pure in-memory re-derivation: no HTTP client was ever constructed.
        self.assertIsNone(client._client)

    def test_refresh_without_rate_or_wallet_context_returns_portfolio_untouched(self) -> None:
        now_ms = int(time.time() * 1000)
        portfolio = self._pending_fx_portfolio(now_ms)
        client = T212Client(isa_key="a", isa_secret="b", invest_key="c", invest_secret="d")
        self.assertIs(client.refresh_portfolio_fx(portfolio, None), portfolio)

        client._wallet_records = {}
        client._db = None
        self.assertIs(client.refresh_portfolio_fx(portfolio, 1.25), portfolio)
        self.assertEqual(client.gbp_usd_rate, 1.25)


_NOW_MS = 1_763_000_000_000


def _hist_order(
    *,
    oid: str,
    side: str,
    ticker: str,
    account: str = "ISA",
    days_ago: float = 1.0,
    net_value: float = 800.0,
    wallet_currency: str = "GBP",
    tax_total: float = 0.0,
) -> HistoricalOrderItem:
    ms = _NOW_MS - int(days_ago * 86400 * 1000)
    return HistoricalOrderItem(
        id=oid,
        account=account,
        raw_ticker=f"{ticker}_US_EQ",
        ticker=ticker,
        yahoo_symbol=ticker,
        name=ticker,
        inst_currency=wallet_currency,
        side=side,
        order_type="MARKET",
        filled_at="2026-11-16T10:00:00Z",
        filled_at_ms=ms,
        uk_time_str="2026-11-16 10:00",
        quantity=1.0,
        price=net_value,
        net_value=net_value,
        wallet_currency=wallet_currency,
        realised_pnl=0.0,
        tax_total=tax_total,
    )


def _empty_portfolio(**kwargs: object) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 07:00",
        fetched_at_ms=_NOW_MS,
        isa=kwargs.get("isa") or AccountSnapshot(ok=True, account_label="ISA"),
        invest=kwargs.get("invest") or AccountSnapshot(ok=True, account_label="Invest"),
        gbp_usd_rate=kwargs.get("gbp_usd_rate"),
    )


class TestTradeBehaviorStats(unittest.TestCase):
    """Stage 1 (v5.3.2): 0-Token cadence / ticket size / fee / cooldown statistics."""

    def test_cadence_ticket_size_and_fee_ratio(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            recent_orders=[
                _hist_order(oid="b1", side="BUY", ticker="NVDA", days_ago=2, net_value=800.0, tax_total=4.0),
                _hist_order(oid="b2", side="BUY", ticker="MSFT", days_ago=20, net_value=1000.0, tax_total=5.0),
                _hist_order(oid="s1", side="SELL", ticker="AAPL", days_ago=3, net_value=1200.0),
                _hist_order(oid="s2", side="SELL", ticker="TSLA", days_ago=40, net_value=500.0),
            ],
        )
        stats = compute_trade_behavior_stats(_empty_portfolio(isa=isa), now_ms=_NOW_MS)
        self.assertEqual(stats["buys_7d"], 1)
        self.assertEqual(stats["buys_30d"], 2)
        self.assertEqual(stats["sells_7d"], 1)
        self.assertEqual(stats["sells_30d"], 1)  # 40-day-old fill outside the 30d window
        self.assertEqual(stats["avg_buy_amount_by_currency"]["GBP"], 900.0)
        self.assertAlmostEqual(stats["fees_pct_of_turnover"], 0.3, places=3)
        self.assertFalse(stats["partial"])

    def test_cooldown_list_skips_held_and_stale_tickers(self) -> None:
        held = PositionItem(
            account="ISA",
            raw_ticker="NVDA_US_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVIDIA",
            inst_currency="USD",
            quantity=2.0,
            avg_price=100.0,
            cur_price=110.0,
            current_value=220.0,
            total_cost=200.0,
            pnl=20.0,
            fx_impact=0.0,
            return_pct=10.0,
        )
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            positions=[held],
            recent_orders=[
                _hist_order(oid="s1", side="SELL", ticker="AAPL", days_ago=10, net_value=900.0),
                _hist_order(oid="s2", side="SELL", ticker="NVDA", days_ago=5, net_value=900.0),
                _hist_order(oid="s3", side="SELL", ticker="OLD", days_ago=40, net_value=900.0),
            ],
        )
        stats = compute_trade_behavior_stats(_empty_portfolio(isa=isa), now_ms=_NOW_MS, cooldown_days=15)
        closed = {row["ticker"]: row for row in stats["recently_closed"]}
        self.assertEqual(set(closed), {"AAPL"})
        self.assertTrue(closed["AAPL"]["in_cooldown"])
        self.assertAlmostEqual(closed["AAPL"]["days_since_sell"], 10.0, places=1)

    def test_partial_history_flag_and_empty_history(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            recent_orders=[_hist_order(oid="b1", side="BUY", ticker="NVDA", days_ago=5, net_value=750.0)],
        )
        stats = compute_trade_behavior_stats(_empty_portfolio(isa=isa), now_ms=_NOW_MS)
        self.assertTrue(stats["partial"])
        self.assertAlmostEqual(stats["coverage_days"], 5.0, places=1)

        empty = compute_trade_behavior_stats(_empty_portfolio(), now_ms=_NOW_MS)
        self.assertFalse(empty["partial"])
        self.assertEqual(empty["coverage_days"], 0.0)
        self.assertEqual(empty["buys_30d"], 0)

    def test_mixed_currency_never_inflates_gbp_total(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            recent_orders=[_hist_order(oid="b1", side="BUY", ticker="VUAG", days_ago=4, net_value=700.0)],
        )
        invest = AccountSnapshot(
            ok=True,
            account_label="Invest",
            recent_orders=[
                _hist_order(
                    oid="b2",
                    side="BUY",
                    ticker="MSFT",
                    account="Invest",
                    days_ago=6,
                    net_value=1000.0,
                    wallet_currency="USD",
                )
            ],
        )
        with_rate = compute_trade_behavior_stats(
            _empty_portfolio(isa=isa, invest=invest, gbp_usd_rate=1.25),
            now_ms=_NOW_MS,
        )
        self.assertEqual(with_rate["avg_buy_amount_by_currency"], {"GBP": 700.0, "USD": 1000.0})
        self.assertAlmostEqual(with_rate["avg_buy_gbp"], 750.0, places=2)

        without_rate = compute_trade_behavior_stats(_empty_portfolio(isa=isa, invest=invest), now_ms=_NOW_MS)
        self.assertIsNone(without_rate["avg_buy_gbp"])

    def test_format_text_exposes_cooldown_and_partial_warning(self) -> None:
        text = format_trade_behavior_text(
            {
                "buys_7d": 1,
                "buys_30d": 3,
                "sells_7d": 0,
                "sells_30d": 1,
                "avg_buy_amount_by_currency": {"GBP": 780.0},
                "avg_buy_gbp": 780.0,
                "fees_pct_of_turnover": 0.42,
                "recently_closed": [{"ticker": "AAPL", "account": "ISA", "days_since_sell": 4.0, "in_cooldown": True}],
                "cooldown_days": 15,
                "coverage_days": 12.0,
                "partial": True,
            }
        )
        self.assertIn("近7天买入 1 笔 / 近30天 3 笔", text)
        self.assertIn("税费占成交额 0.42%", text)
        self.assertIn("清仓 15 天冷却清单: AAPL(4天·冷却)", text)
        self.assertIn("频率统计不完整", text)
        self.assertEqual(format_trade_behavior_text({}), "")


class TestPendingOrderLimitUpside(unittest.TestCase):
    """Stage 1 (v5.3.2): pending BUY audits must anchor upside on the LIMIT price."""

    def test_limit_upside_anchored_on_limit_price_not_live_price(self) -> None:
        portfolio = _empty_portfolio(
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                pending_orders=[
                    PendingOrderItem(
                        id="o-dip",
                        account="ISA",
                        raw_ticker="MSFT_US_EQ",
                        ticker="MSFT",
                        yahoo_symbol="MSFT",
                        name="Microsoft",
                        side="BUY",
                        order_type="LIMIT",
                        quantity=1.0,
                        value=90.0,
                        limit_price=90.0,
                        currency="USD",
                        time_in_force="GTC",
                    ),
                    PendingOrderItem(
                        id="o-chase",
                        account="ISA",
                        raw_ticker="MSFT_US_EQ",
                        ticker="MSFT",
                        yahoo_symbol="MSFT",
                        name="Microsoft",
                        side="BUY",
                        order_type="LIMIT",
                        quantity=1.0,
                        value=102.0,
                        limit_price=102.0,
                        currency="USD",
                        time_in_force="GTC",
                    ),
                ],
            )
        )
        bundles = {
            "MSFT": TickerMarketBundle(
                symbol="MSFT",
                currency="USD",
                current_price=100.0,
                target_mean_price=120.0,
            )
        }
        text = build_trade_context(portfolio, bundles=bundles)["pending_orders_text"]
        # Limit 90 vs live 100 with a 120 target => +33.3% upside measured from the LIMIT price.
        self.assertIn("距现价 -10.0%", text)
        self.assertIn("限价空间 +33.3%", text)
        self.assertIn("[有效回踩埋伏单]", text)
        self.assertIn("[追价单]", text)
        # Live price sits 20% below target, so a live-price audit would wrongly show +20.0%.
        self.assertNotIn("限价空间 +20.0%", text)

if __name__ == "__main__":
    unittest.main()
