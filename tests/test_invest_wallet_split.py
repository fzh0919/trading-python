"""Offline tests for the Invest GBP+USD wallet split and multi-currency cash-flow isolation."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalDividendItem,
    HistoricalTransactionItem,
    PendingOrderItem,
    PortfolioData,
    T212Client,
    build_trade_context,
    format_account_overview,
    reconcile_invest_wallets,
)
from src.storage.db import AdvisorDatabase

_DAY_MS = 86_400_000


def _wallet_records(
    *,
    usd_available: float = 0.0,
    usd_reserved: float = 0.0,
    usd_source: str = "manual",
    usd_updated_ms: int | None = None,
    gbp_available: float | None = None,
    gbp_reserved: float = 0.0,
) -> dict[str, dict[str, object]]:
    now_ms = int(time.time() * 1000)
    records: dict[str, dict[str, object]] = {
        "USD": {
            "currency": "USD",
            "available_amount": usd_available,
            "reserved_amount": usd_reserved,
            "updated_at_uk": "2026-11-17 08:00",
            "updated_at_ms": now_ms if usd_updated_ms is None else usd_updated_ms,
            "source": usd_source,
            "note": "",
        }
    }
    if gbp_available is not None:
        records["GBP"] = {
            "currency": "GBP",
            "available_amount": gbp_available,
            "reserved_amount": gbp_reserved,
            "updated_at_uk": "2026-11-17 08:00",
            "updated_at_ms": now_ms,
            "source": "manual",
            "note": "",
        }
    return records


def _invest_snapshot(**overrides: object) -> AccountSnapshot:
    base: dict[str, object] = {
        "ok": True,
        "account_label": "Invest",
        "account_currency": "GBP",
        "total_account_value": 5000.0,
        "investments_value": 3000.0,
        "investments_cost": 2800.0,
        "total_cash": 2000.0,
        "available_to_trade": 1500.0,
        "reserved_cash": 500.0,
        "in_pies_cash": 0.0,
        "cash_ratio_pct": 40.0,
    }
    base.update(overrides)
    return AccountSnapshot(**base)


def _usd_buy_order(quantity: float, limit_price: float, *, side: str = "BUY", currency: str = "USD") -> PendingOrderItem:
    return PendingOrderItem(
        id=f"ord-{side}-{quantity:g}-{limit_price:g}",
        account="Invest",
        raw_ticker="NVDA_US_EQ",
        ticker="NVDA",
        yahoo_symbol="NVDA",
        name="NVIDIA",
        side=side,
        order_type="LIMIT",
        quantity=quantity,
        value=quantity * limit_price,
        limit_price=limit_price,
        currency=currency,
    )


class TestReconcileInvestWallets(unittest.TestCase):
    def test_derived_gbp_equals_total_cash_minus_usd(self) -> None:
        snap = reconcile_invest_wallets(_invest_snapshot(), _wallet_records(usd_available=600.0), 1.25)

        self.assertTrue(snap.multi_currency_enabled)
        self.assertEqual(snap.usd_cash_total, 600.0)
        self.assertEqual(snap.usd_available_to_trade, 600.0)
        self.assertEqual(snap.gbp_wallet_total, 1520.0)
        self.assertEqual(snap.gbp_wallet_available, 1020.0)
        self.assertEqual(snap.gbp_wallet_reserved, 500.0)
        self.assertEqual(snap.wallet_source, "derived_from_total")
        self.assertEqual(snap.gbp_usd_rate, 1.25)
        # GBP accounting invariants must stay untouched
        self.assertEqual(snap.total_cash, 2000.0)
        self.assertEqual(snap.available_to_trade, 1500.0)

    def test_negative_derived_gbp_is_clamped_with_calibration_warning(self) -> None:
        snap = reconcile_invest_wallets(_invest_snapshot(), _wallet_records(usd_available=3000.0), 1.25)

        self.assertEqual(snap.gbp_wallet_total, 0.0)
        self.assertEqual(snap.gbp_wallet_available, 0.0)
        self.assertIn("负", snap.wallet_reconcile_warning)
        self.assertIn("校准", snap.wallet_reconcile_warning)
        self.assertEqual(snap.total_cash, 2000.0)

    def test_missing_fx_rate_keeps_usd_only(self) -> None:
        snap = reconcile_invest_wallets(_invest_snapshot(), _wallet_records(usd_available=600.0), None)

        self.assertTrue(snap.multi_currency_enabled)
        self.assertEqual(snap.usd_cash_total, 600.0)
        self.assertIsNone(snap.gbp_wallet_total)
        self.assertEqual(snap.wallet_source, "pending_fx")
        self.assertIn("汇率", snap.wallet_reconcile_warning)


    def test_manual_both_within_tolerance_has_no_mismatch_warning(self) -> None:
        records = _wallet_records(usd_available=600.0, gbp_available=1520.0)
        snap = reconcile_invest_wallets(_invest_snapshot(), records, 1.25)

        self.assertEqual(snap.wallet_source, "manual_both")
        self.assertEqual(snap.gbp_wallet_total, 1520.0)
        self.assertNotIn("偏差", snap.wallet_reconcile_warning)

    def test_manual_both_mismatch_uses_max_of_gbp_and_pct_threshold(self) -> None:
        records = _wallet_records(usd_available=600.0, gbp_available=1200.0)
        snap = reconcile_invest_wallets(
            _invest_snapshot(),
            records,
            1.25,
            mismatch_warn_gbp=10.0,
            mismatch_warn_pct=2.0,
        )

        # reported = 1200 + 480 = 1680 vs 2000 -> diff -320 > max(10, 40)
        self.assertIn("偏差", snap.wallet_reconcile_warning)
        self.assertEqual(snap.gbp_wallet_total, 1200.0)

    def test_older_gbp_report_falls_back_to_derivation(self) -> None:
        stale_records = _wallet_records(usd_available=600.0, gbp_available=1200.0)
        stale_records["GBP"]["updated_at_ms"] = int(stale_records["USD"]["updated_at_ms"]) - _DAY_MS
        derived = reconcile_invest_wallets(_invest_snapshot(), stale_records, 1.25)

        self.assertEqual(derived.wallet_source, "derived_from_total")
        self.assertEqual(derived.gbp_wallet_total, 1520.0)
        self.assertEqual(derived.gbp_wallet_available, 1020.0)

    def test_stale_anchor_and_init_default_are_flagged(self) -> None:
        stale_ms = int(time.time() * 1000) - 10 * _DAY_MS
        stale = reconcile_invest_wallets(
            _invest_snapshot(),
            _wallet_records(usd_available=100.0, usd_updated_ms=stale_ms),
            1.25,
            reconcile_warn_days=7,
        )
        self.assertIn("未校准", stale.wallet_reconcile_warning)

        # A fresh default anchor must stay silent to avoid per-command alarm fatigue
        seeded_fresh = reconcile_invest_wallets(
            _invest_snapshot(),
            _wallet_records(usd_available=0.0, usd_source="init"),
            1.25,
        )
        self.assertEqual(seeded_fresh.wallet_reconcile_warning, "")
        self.assertEqual(seeded_fresh.gbp_wallet_total, 2000.0)

        seeded_stale = reconcile_invest_wallets(
            _invest_snapshot(),
            _wallet_records(usd_available=0.0, usd_source="init", usd_updated_ms=stale_ms),
            1.25,
            reconcile_warn_days=7,
        )
        self.assertIn("默认 0 USD", seeded_stale.wallet_reconcile_warning)

    def test_failed_snapshot_is_not_split(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(ok=False, error="boom"), _wallet_records(usd_available=10.0), 1.25
        )
        self.assertFalse(snap.multi_currency_enabled)
        self.assertIsNone(snap.gbp_wallet_total)

    def test_active_usd_buy_order_freezes_usd_wallet_first(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(
                pending_orders_supported=True,
                pending_orders=[_usd_buy_order(5.0, 100.0)],
            ),
            _wallet_records(usd_available=600.0),
            1.25,
        )

        # min(usd_total 600, pending 500, API reserved 500*1.25=625) = 500
        self.assertEqual(snap.usd_cash_total, 600.0)
        self.assertEqual(snap.usd_reserved_cash, 500.0)
        self.assertEqual(snap.usd_available_to_trade, 100.0)
        self.assertEqual(snap.gbp_wallet_total, 1520.0)
        self.assertEqual(snap.gbp_wallet_available, 1420.0)
        self.assertEqual(snap.gbp_wallet_reserved, 100.0)

    def test_pending_order_above_usd_balance_is_clamped_by_usd_total(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(
                pending_orders_supported=True,
                pending_orders=[_usd_buy_order(10.0, 100.0)],
            ),
            _wallet_records(usd_available=600.0),
            1.25,
        )

        self.assertEqual(snap.usd_reserved_cash, 600.0)
        self.assertEqual(snap.usd_available_to_trade, 0.0)
        self.assertEqual(snap.gbp_wallet_available, 1500.0)
        self.assertEqual(snap.gbp_wallet_reserved, 20.0)

    def test_cancelled_order_releases_manual_usd_reserve(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(pending_orders_supported=True, reserved_cash=500.0, pending_orders=[]),
            _wallet_records(usd_available=300.0, usd_reserved=300.0),
            1.25,
        )

        self.assertEqual(snap.usd_cash_total, 600.0)
        self.assertEqual(snap.usd_reserved_cash, 0.0)
        self.assertEqual(snap.usd_available_to_trade, 600.0)
        self.assertEqual(snap.gbp_wallet_available, 1020.0)
        self.assertEqual(snap.gbp_wallet_reserved, 500.0)

    def test_pending_api_unavailable_falls_back_to_static_anchor(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(pending_orders_supported=False, pending_orders=[]),
            _wallet_records(usd_available=480.0, usd_reserved=120.0),
            1.25,
        )

        self.assertEqual(snap.usd_reserved_cash, 120.0)
        self.assertEqual(snap.usd_available_to_trade, 480.0)
        self.assertEqual(snap.gbp_wallet_available, 1116.0)
        self.assertEqual(snap.gbp_wallet_reserved, 404.0)

    def test_non_usd_or_sell_orders_do_not_freeze_usd(self) -> None:
        snap = reconcile_invest_wallets(
            _invest_snapshot(
                pending_orders_supported=True,
                pending_orders=[
                    _usd_buy_order(5.0, 100.0, currency="GBP"),
                    _usd_buy_order(5.0, 100.0, side="SELL"),
                ],
            ),
            _wallet_records(usd_available=600.0),
            1.25,
        )

        self.assertEqual(snap.usd_reserved_cash, 0.0)
        self.assertEqual(snap.usd_available_to_trade, 600.0)


class TestWalletPresentation(unittest.TestCase):
    def test_formatter_renders_dual_wallet_line_only_when_enabled(self) -> None:
        snap = reconcile_invest_wallets(_invest_snapshot(), _wallet_records(usd_available=600.0), 1.25)
        snap = snap.model_copy(update={"cash_interest_rate_pct": 4.0, "usd_cash_interest_rate_pct": 3.4})

        text = format_account_overview(snap)
        self.assertIn("现金钱包拆解", text)
        self.assertIn("GBP 钱包 ≈ £1,520.00", text)
        self.assertIn("USD 钱包 $600.00", text)
        self.assertIn("≈ £480.00", text)
        self.assertIn("3.4%", text)
        self.assertIn("按 API 总现金推算", text)

        plain = format_account_overview(_invest_snapshot())
        self.assertNotIn("钱包拆解", plain)

    def test_pending_fx_formatter_shows_placeholder(self) -> None:
        snap = reconcile_invest_wallets(_invest_snapshot(), _wallet_records(usd_available=600.0), None)
        self.assertIn("GBP 钱包 ≈ 待汇率折算", format_account_overview(snap))


class TestMultiCurrencyTradeContext(unittest.TestCase):
    def _portfolio(self) -> PortfolioData:
        now_ms = int(time.time() * 1000)
        txs = [
            HistoricalTransactionItem(
                id="t1",
                account="Invest",
                tx_type="DEPOSIT",
                amount=1000.0,
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
        ]
        divs = [
            HistoricalDividendItem(
                id="d1",
                account="Invest",
                raw_ticker="NVDA_US_EQ",
                ticker="NVDA",
                yahoo_symbol="NVDA",
                name="NVIDIA",
                amount=12.5,
                currency="USD",
                paid_on="2026-11-17",
                paid_on_ms=now_ms,
                uk_time_str="2026-11-17 08:10",
            )
        ]
        return PortfolioData(
            fetched_at_uk="2026-11-17 09:00",
            fetched_at_ms=now_ms,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_cash=2000.0),
            invest=AccountSnapshot(
                ok=True,
                account_label="Invest",
                total_cash=2000.0,
                recent_transactions=txs,
                recent_dividends=divs,
            ),
            total_combined_value=12000.0,
            total_combined_cash=4000.0,
        )

    def test_usd_flows_never_inflate_gbp_totals(self) -> None:
        text = build_trade_context(self._portfolio(), history_days=30)["cash_flow_summary_text"]

        self.assertIn("外部入金 +1,000.00 GBP", text)
        self.assertIn("现金利息 +3.40 USD", text)
        self.assertIn("派息入账 +12.50 USD(1笔)", text)
        self.assertNotIn("1,003.40 GBP", text)
        self.assertNotIn("1,012.50 GBP", text)


class TestGbpUsdRateInjection(unittest.IsolatedAsyncioTestCase):
    async def test_first_rate_invalidates_cache_and_enables_derivation(self) -> None:
        client = T212Client()
        cached_portfolio = PortfolioData(
            fetched_at_uk="2026-11-17 09:00",
            fetched_at_ms=1,
            isa=AccountSnapshot(ok=True),
            invest=AccountSnapshot(ok=True),
        )
        client._cache = (time.monotonic(), cached_portfolio)

        client.set_gbp_usd_rate(1.25)
        self.assertIsNone(client._cache)
        self.assertEqual(client._gbp_usd_rate, 1.25)
        client.set_gbp_usd_rate(0.0)
        self.assertEqual(client._gbp_usd_rate, 1.25)


    async def test_get_portfolio_attaches_wallet_split_from_sqlite(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            is_invest = "aW52ZXN0" in request.headers.get("Authorization", "")
            if path.endswith("/equity/account/summary"):
                total = 5000.0 if is_invest else 10000.0
                invested = 3000.0 if is_invest else 8000.0
                return httpx.Response(
                    200,
                    json={
                        "currency": "GBP",
                        "totalValue": total,
                        "cash": {"availableToTrade": 1500.0, "reservedForOrders": 500.0, "inPies": 0.0},
                        "investments": {
                            "currentValue": invested,
                            "totalCost": invested - 200.0,
                            "unrealizedProfitLoss": 200.0,
                        },
                    },
                )
            if path.endswith("/equity/positions") or path.endswith("/equity/orders"):
                return httpx.Response(200, json=[])
            if path.startswith("/api/v0/equity/history/"):
                return httpx.Response(200, json={"items": []})
            return httpx.Response(404)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "wallet_client.db")
            await db.init()
            await db.upsert_wallet_balance("Invest", "USD", 600.0)

            client = T212Client(
                isa_key="isa",
                isa_secret="sec",
                invest_key="invest",
                invest_secret="sec",
                transport=httpx.MockTransport(handler),
                db=db,
                wallet_mismatch_warn_gbp=10.0,
            )
            portfolio = await asyncio.wait_for(
                client.get_portfolio(use_cache=False, gbp_usd_rate=1.25), timeout=10.0
            )

        self.assertTrue(portfolio.invest.ok)
        self.assertTrue(portfolio.invest.multi_currency_enabled)
        self.assertEqual(portfolio.invest.usd_cash_total, 600.0)
        self.assertAlmostEqual(portfolio.invest.gbp_wallet_total or 0.0, 1520.0, places=2)
        self.assertAlmostEqual(portfolio.invest.gbp_wallet_available or 0.0, 1020.0, places=2)
        self.assertEqual(portfolio.gbp_usd_rate, 1.25)
        # Combined cash stays in GBP; the USD wallet is not double-counted
        self.assertAlmostEqual(portfolio.total_combined_cash, 4000.0, places=2)
        self.assertFalse(portfolio.isa.multi_currency_enabled)


if __name__ == "__main__":
    unittest.main()

