"""Offline unit tests for IMP-17: NAV→TWR performance, trade attribution, signal QC, Yahoo fallback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_providers.analytics import (
    build_nav_row,
    compute_concentration_metrics,
    compute_portfolio_performance,
    compute_signal_quality,
    compute_trade_attribution,
)
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalDividendItem,
    HistoricalOrderItem,
    HistoricalTransactionItem,
    PortfolioData,
    PositionItem,
)
from src.storage.db import AdvisorDatabase


def _tx(amount: float, tx_type: str = "DEPOSIT", ccy: str = "GBP", day: str = "2026-11-17") -> HistoricalTransactionItem:
    return HistoricalTransactionItem(
        id=f"tx-{tx_type}-{amount}",
        account="ISA",
        tx_type=tx_type,
        amount=amount,
        currency=ccy,
        date_time=f"{day} 09:00:00",
        date_time_ms=1,
        uk_time_str=f"{day} 09:00",
    )


def _order(side: str, pnl: float, filled_ms: int, ticker: str = "NVDA", ccy: str = "GBP") -> HistoricalOrderItem:
    return HistoricalOrderItem(
        id=f"{ticker}-{side}-{filled_ms}",
        account="ISA",
        raw_ticker=f"{ticker}_US_EQ",
        ticker=ticker,
        yahoo_symbol=ticker,
        name=ticker,
        inst_currency="USD",
        side=side,
        order_type="MARKET",
        filled_at=f"2026-11-{10 + (filled_ms % 9):02d}T10:00:00Z",
        filled_at_ms=filled_ms,
        uk_time_str="2026-11-17 10:00",
        quantity=1.0,
        price=100.0,
        net_value=100.0,
        wallet_currency=ccy,
        realised_pnl=pnl,
        tax_total=0.5,
    )


def _portfolio(
    total: float = 1000.0,
    cash: float = 400.0,
    isa: AccountSnapshot | None = None,
    invest: AccountSnapshot | None = None,
) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 07:30",
        fetched_at_ms=1_760_000_000_000,
        isa=isa or AccountSnapshot(ok=True, account_label="ISA", total_account_value=total * 0.6),
        invest=invest or AccountSnapshot(ok=True, account_label="Invest", total_account_value=total * 0.4),
        total_combined_value=total,
        total_combined_cash=cash,
        cash_interest_rate_pct=4.0,
    )


def _nav_rows(pairs: list[tuple[str, float, float]]) -> list[dict[str, float | str]]:
    """Build NAV rows from (date, nav, flow) triples."""
    return [
        {
            "uk_date": day,
            "total_value_gbp": nav,
            "cash_gbp": 0.0,
            "invested_gbp": nav,
            "net_flow_gbp": flow,
            "isa_value_gbp": nav,
            "invest_value_gbp": 0.0,
            "benchmark_close": None,
            "cash_rate_pct": 4.0,
        }
        for day, nav, flow in pairs
    ]


class TestNavRowAndPerformance(unittest.TestCase):
    def test_build_nav_row_counts_only_external_gbp_flows(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=600.0,
            recent_transactions=[
                _tx(50.0, "DEPOSIT"),
                _tx(10.0, "WITHDRAW"),
                _tx(5.0, "FEE"),                    # internal drag: never an external flow
                _tx(12.0, "INTEREST_ON_FREE_CASH"),  # internal investment return
                _tx(999.0, "DEPOSIT", ccy="USD"),      # no FX rate -> cannot convert safely
                _tx(999.0, "DEPOSIT", day="2026-11-16"),  # ignored: not today
            ],
        )
        row = build_nav_row(_portfolio(total=1000.0, cash=400.0, isa=isa))
        self.assertEqual(row["uk_date"], "2026-11-17")
        self.assertAlmostEqual(row["net_flow_gbp"], 40.0, places=2)
        self.assertAlmostEqual(row["invested_gbp"], 600.0, places=2)
        self.assertEqual(row["cash_rate_pct"], 4.0)

    def test_usd_external_deposit_is_converted_to_gbp(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=600.0,
            recent_transactions=[_tx(125.0, "DEPOSIT", ccy="USD")],
        )
        portfolio = _portfolio(total=1000.0, cash=400.0, isa=isa)
        portfolio.gbp_usd_rate = 1.25
        row = build_nav_row(portfolio)
        self.assertAlmostEqual(row["net_flow_gbp"], 100.0, places=2)

    def test_cash_interest_counts_as_twr_return_not_external_deposit(self) -> None:
        interest_row = build_nav_row(
            _portfolio(
                cash=1000.0,
                isa=AccountSnapshot(
                    ok=True,
                    account_label="ISA",
                    total_account_value=600.0,
                    recent_transactions=[_tx(4.0, "INTEREST_ON_FREE_CASH")],
                ),
            )
        )
        self.assertAlmostEqual(interest_row["net_flow_gbp"], 0.0, places=2)
        perf = compute_portfolio_performance(
            _nav_rows([("2026-11-16", 1000.0, 0.0), ("2026-11-17", 1004.0, 0.0)])
        )
        self.assertTrue(perf["ok"])
        self.assertAlmostEqual(perf["total_return_pct"], 0.4, places=2)

    def test_twr_excludes_deposit_from_return(self) -> None:
        rows = _nav_rows(
            [
                ("2026-11-01", 100.0, 0.0),
                ("2026-11-02", 110.0, 0.0),
                ("2026-11-03", 160.0, 50.0),  # +50 deposit must NOT count as return
            ]
        )
        perf = compute_portfolio_performance(rows)
        self.assertTrue(perf["ok"])
        self.assertAlmostEqual(perf["total_return_pct"], 10.0, places=2)
        self.assertEqual(perf["samples"], 3)

    def test_max_drawdown_and_recovery_days(self) -> None:
        rows = _nav_rows(
            [
                ("2026-11-01", 100.0, 0.0),
                ("2026-11-02", 120.0, 0.0),
                ("2026-11-03", 90.0, 0.0),
                ("2026-11-04", 130.0, 0.0),
            ]
        )
        perf = compute_portfolio_performance(rows)
        self.assertAlmostEqual(perf["max_drawdown_pct"], -25.0, places=2)
        self.assertEqual(perf["drawdown_peak_date"], "2026-11-02")
        self.assertEqual(perf["drawdown_trough_date"], "2026-11-03")
        self.assertEqual(perf["drawdown_recovery_days"], 1)
        self.assertFalse(perf["drawdown_unrecovered"])

    def test_benchmark_beta_and_alpha_use_nav_benchmark_close(self) -> None:
        import pandas as pd

        dates = pd.bdate_range("2026-01-05", periods=16)
        closes = [100.0]
        for step in range(1, 16):
            closes.append(closes[-1] * (1.02 if step % 2 else 0.99))
        rows = []
        for day, close in zip(dates, closes):
            rows.append(
                {
                    "uk_date": day.strftime("%Y-%m-%d"),
                    "total_value_gbp": close,
                    "cash_gbp": 0.0,
                    "invested_gbp": close,
                    "net_flow_gbp": 0.0,
                    "isa_value_gbp": close,
                    "invest_value_gbp": 0.0,
                    "benchmark_close": close,
                    "cash_rate_pct": 4.0,
                }
            )
        perf = compute_portfolio_performance(rows)
        self.assertEqual(perf["bench_source"], "nav_benchmark_close")
        self.assertAlmostEqual(perf["bench_beta"], 1.0, places=2)
        self.assertAlmostEqual(perf["bench_alpha_annual_pct"], 0.0, places=2)

    def test_insufficient_samples_is_reported_not_guessed(self) -> None:
        perf = compute_portfolio_performance(_nav_rows([("2026-11-01", 100.0, 0.0)]))
        self.assertFalse(perf["ok"])
        self.assertEqual(perf["samples"], 0)
        self.assertIn("reason", perf)


class TestTradeAttributionAndSignalQuality(unittest.TestCase):
    def test_win_rate_profit_factor_and_expectancy(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=1000.0,
            recent_orders=[
                _order("BUY", 0.0, 1_000),
                _order("SELL", 100.0, 2_000),
                _order("SELL", -50.0, 3_000),
            ],
            recent_dividends=[
                HistoricalDividendItem(
                    id="d1",
                    account="ISA",
                    raw_ticker="NVDA_US_EQ",
                    ticker="NVDA",
                    yahoo_symbol="NVDA",
                    name="NVDA",
                    amount=12.5,
                    currency="GBP",
                    paid_on="2026-11-10",
                    paid_on_ms=1,
                    uk_time_str="2026-11-10 00:00",
                )
            ],
        )
        result = compute_trade_attribution(_portfolio(isa=isa, invest=AccountSnapshot(ok=True)))
        self.assertTrue(result["ok"])
        self.assertEqual(result["sells"], 2)
        self.assertAlmostEqual(result["win_rate_pct"], 50.0, places=1)
        gbp = result["by_currency"]["GBP"]
        self.assertAlmostEqual(gbp["profit_factor"], 2.0, places=2)
        self.assertAlmostEqual(gbp["expectancy"], 25.0, places=2)
        self.assertAlmostEqual(result["dividends_by_currency"]["GBP"], 12.5, places=2)
        self.assertIsNotNone(result["avg_holding_days"])

    def test_currency_isolation_is_never_summed(self) -> None:
        isa = AccountSnapshot(ok=True, recent_orders=[_order("SELL", 10.0, 1_000, ccy="GBP")])
        invest = AccountSnapshot(ok=True, recent_orders=[_order("SELL", 20.0, 2_000, ccy="USD")])
        result = compute_trade_attribution(_portfolio(isa=isa, invest=invest))
        self.assertEqual(set(result["by_currency"]), {"GBP", "USD"})
        self.assertAlmostEqual(result["by_currency"]["USD"]["realised"], 20.0, places=2)
        self.assertAlmostEqual(result["by_currency"]["GBP"]["realised"], 10.0, places=2)

    def test_signal_quality_t_stat_and_decay(self) -> None:
        rows = [
            {"alpha_pct": 1.0, "t10_alpha_pct": 0.5, "t20_alpha_pct": 0.0},
            {"alpha_pct": 2.0, "t10_alpha_pct": 1.5, "t20_alpha_pct": 1.0},
            {"alpha_pct": 3.0, "t10_alpha_pct": 2.5, "t20_alpha_pct": 2.0},
        ]
        quality = compute_signal_quality(rows)
        self.assertEqual(quality["sample"], 3)
        self.assertAlmostEqual(quality["horizons"]["t5"]["mean_pct"], 2.0, places=2)
        self.assertAlmostEqual(quality["horizons"]["t5"]["t_stat"], 3.46, places=2)
        self.assertTrue(quality["horizons"]["t5"]["significant"])
        self.assertAlmostEqual(quality["decay_t20_minus_t5_pct"], -1.0, places=2)

    def test_signal_quality_never_claims_significance_on_one_sample(self) -> None:
        quality = compute_signal_quality([{"alpha_pct": 5.0}])
        self.assertEqual(quality["horizons"]["t5"]["n"], 1)
        self.assertIsNone(quality["horizons"]["t5"]["t_stat"])
        self.assertFalse(quality["horizons"]["t5"]["significant"])

    def test_concentration_hhi_and_effective_n(self) -> None:
        isa = AccountSnapshot(
            ok=True,
            positions=[
                PositionItem(
                    account="ISA",
                    raw_ticker="A",
                    ticker="A",
                    yahoo_symbol="A",
                    name="A",
                    inst_currency="USD",
                    quantity=1,
                    avg_price=1,
                    cur_price=1,
                    current_value=500.0,
                    total_cost=1,
                    pnl=0,
                    fx_impact=0,
                    return_pct=0,
                ),
                PositionItem(
                    account="ISA",
                    raw_ticker="B",
                    ticker="B",
                    yahoo_symbol="B",
                    name="B",
                    inst_currency="USD",
                    quantity=1,
                    avg_price=1,
                    cur_price=1,
                    current_value=500.0,
                    total_cost=1,
                    pnl=0,
                    fx_impact=0,
                    return_pct=0,
                ),
            ],
        )
        metrics = compute_concentration_metrics(_portfolio(isa=isa, invest=AccountSnapshot(ok=True)))
        self.assertTrue(metrics["ok"])
        self.assertAlmostEqual(metrics["hhi"], 0.5, places=4)
        self.assertAlmostEqual(metrics["effective_n"], 2.0, places=2)
        self.assertAlmostEqual(metrics["top1_pct"], 50.0, places=1)


class TestNavHistoryStorage(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_is_idempotent_and_survives_prune(self) -> None:
        from datetime import datetime, timedelta, timezone

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "nav.db")
            await db.init()
            row = build_nav_row(_portfolio(total=1234.5, cash=234.5))
            await db.upsert_nav_history(row)
            await db.upsert_nav_history({**row, "total_value_gbp": 1300.0})

            rows = await db.get_nav_history(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["total_value_gbp"], 1300.0, places=2)

            await db.prune_old_records(
                retention_days=90, now_utc=datetime.now(timezone.utc) + timedelta(days=400)
            )
            rows_after = await db.get_nav_history(limit=10)
            self.assertEqual(len(rows_after), 1)


class TestYahooChartFallback(unittest.TestCase):
    """The fallback must never raise: it runs inside asyncio.gather for /report."""

    def test_returns_empty_frame_when_all_hosts_fail(self) -> None:
        from unittest.mock import patch

        import src.data_providers.market_data as md

        class _Resp:
            status_code = 503

            @staticmethod
            def json() -> dict:
                return {}

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            @staticmethod
            def get(*args, **kwargs) -> _Resp:
                return _Resp()

        with patch.object(md.httpx, "Client", _Client):
            frame = md._fetch_yahoo_chart_sync("AZN.L", "1y")
        self.assertTrue(frame.empty)

    def test_parses_ohlcv_and_adjclose(self) -> None:
        from unittest.mock import patch

        import src.data_providers.market_data as md

        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1_760_000_000, 1_760_086_400],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0, 11.0],
                                    "high": [10.5, 11.5],
                                    "low": [9.5, 10.5],
                                    "close": [10.2, 11.2],
                                    "volume": [1000, 2000],
                                }
                            ],
                            "adjclose": [{"adjclose": [10.0, 11.0]}],
                        },
                    }
                ]
            }
        }

        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return payload

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            @staticmethod
            def get(*args, **kwargs) -> _Resp:
                return _Resp()

        with patch.object(md.httpx, "Client", _Client):
            frame = md._fetch_yahoo_chart_sync("AZN.L", "1y")
        self.assertEqual(len(frame), 2)
        for column in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
            self.assertIn(column, frame.columns)
        self.assertAlmostEqual(float(frame["Close"].iloc[-1]), 11.2, places=4)


class TestPerfCommand(unittest.IsolatedAsyncioTestCase):
    async def test_perf_message_renders_from_local_data_only(self) -> None:
        from src.bot.handlers import AdvisorBot
        from src.config import AppConfig, EnvSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "perf.db")
            await db.init()
            for day, nav in (("2026-11-15", 1000.0), ("2026-11-16", 1010.0), ("2026-11-17", 1020.0)):
                await db.upsert_nav_history(
                    {
                        "uk_date": day,
                        "total_value_gbp": nav,
                        "cash_gbp": 100.0,
                        "invested_gbp": nav - 100.0,
                        "net_flow_gbp": 0.0,
                        "isa_value_gbp": nav * 0.6,
                        "invest_value_gbp": nav * 0.4,
                        "benchmark_close": nav,
                        "cash_rate_pct": 4.0,
                    }
                )

            class _FakeT212:
                @staticmethod
                async def get_portfolio(use_cache: bool = True) -> PortfolioData:
                    return _portfolio()

            class _FakeMarket:
                @staticmethod
                async def get_histories(symbols, period="1y"):
                    return {}

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="test-key")
            )
            bot = AdvisorBot(cfg, _FakeT212(), _FakeMarket(), None, db, None)
            message = await bot._build_perf_message()

        self.assertIn("实盘绩效看板", message)
        self.assertIn("TWR 累计收益", message)
        self.assertIn("+2.00%", message)
        self.assertNotIn("**", message)
        self.assertNotIn("#", message)


class TestBasicFinancialsMerge(unittest.TestCase):
    def test_alias_resolution_and_zero_rejection(self) -> None:
        from src.data_providers.market_data import _pick_metric

        self.assertEqual(_pick_metric({"pbAnnual": 3.2}, ("pb", "pbAnnual")), 3.2)
        self.assertIsNone(_pick_metric({"pb": 0}, ("pb",)))
        self.assertIsNone(_pick_metric({"pb": None}, ("pb",)))
        self.assertIsNone(_pick_metric({"pb": "n/a"}, ("pb",)))
        self.assertIsNone(_pick_metric({}, ("pb",)))

    def test_gap_filling_never_overwrites_existing_values(self) -> None:
        from src.data_providers.market_data import TickerMarketBundle, apply_basic_financials

        bundle = TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", beta=1.9)
        changed = apply_basic_financials(
            bundle,
            {"pb": 42.5, "evEbitdaTTM": 31.2, "currentRatioAnnual": 4.1, "beta": 0.5},
        )
        self.assertTrue(changed)
        self.assertEqual(bundle.price_to_book, 42.5)
        self.assertEqual(bundle.ev_ebitda, 31.2)
        self.assertEqual(bundle.current_ratio, 4.1)
        self.assertEqual(bundle.beta, 1.9)  # vendor value preserved, not overwritten
        self.assertEqual(bundle.metrics_source, "finnhub_metric")

    def test_empty_metrics_are_a_noop(self) -> None:
        from src.data_providers.market_data import TickerMarketBundle, apply_basic_financials

        bundle = TickerMarketBundle(symbol="NVDA")
        self.assertFalse(apply_basic_financials(bundle, None))
        self.assertFalse(apply_basic_financials(bundle, {}))
        self.assertFalse(apply_basic_financials(bundle, {"unrelated": 1}))
        self.assertIsNone(bundle.price_to_book)
        self.assertEqual(bundle.metrics_source, "")


class TestEveningNavSnapshotJob(unittest.IsolatedAsyncioTestCase):
    async def _run_job(self, db: AdvisorDatabase, portfolio: PortfolioData, with_benchmark: bool) -> str:
        import types

        from src.bot.scheduler import TradingScheduler

        frame = None
        if with_benchmark:
            import pandas as pd

            frame = pd.DataFrame(
                {"Close": [138.0, 139.5]},
                index=pd.to_datetime(["2026-11-16", "2026-11-17"]),
            )

        class _T212:
            @staticmethod
            async def get_portfolio(use_cache: bool = False) -> PortfolioData:
                return portfolio

        class _Market:
            @staticmethod
            async def get_histories(symbols, period="1mo"):
                return {"VWRP.L": frame} if frame is not None else {}

        stub = types.SimpleNamespace(
            t212=_T212(),
            market_data=_Market(),
            db=db,
            cfg=types.SimpleNamespace(strategy=types.SimpleNamespace(benchmark_symbol="VWRP.L")),
        )
        return await TradingScheduler._run_nav_snapshot_job(stub)

    async def test_persists_post_close_row_with_benchmark_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "eod.db")
            await db.init()
            result = await self._run_job(db, _portfolio(total=1500.0, cash=300.0), with_benchmark=True)
            self.assertIn("nav 2026-11-17", result)
            rows = await db.get_nav_history(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["total_value_gbp"], 1500.0, places=2)
            self.assertAlmostEqual(rows[0]["benchmark_close"], 139.5, places=4)

    async def test_fallback_portfolio_is_never_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "eod2.db")
            await db.init()
            fallback = PortfolioData(
                fetched_at_uk="2026-11-17 21:10",
                fetched_at_ms=1_760_000_100_000,
                isa=AccountSnapshot(ok=False),
                invest=AccountSnapshot(ok=False),
                total_combined_value=0.0,
                is_fallback=True,
                fallback_source="db",
            )
            result = await self._run_job(db, fallback, with_benchmark=False)
            self.assertIn("skipped", result)
            self.assertEqual(await db.get_nav_history(limit=10), [])



