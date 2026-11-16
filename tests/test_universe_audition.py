"""Pure offline unit tests for multi-market universe audition, N×H correlation, 429 circuit breaker, and /refresh_pool."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.bot.handlers import AdvisorBot
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.analytics import compute_correlation_matrix
from src.data_providers.market_data import MarketDataService, YFRateLimitError
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, PositionItem
from src.engine.screener import run_quant_screener, run_universe_audition
from src.engine.universe_seeds import _parse_wikipedia_constituents
from src.storage.db import AdvisorDatabase


def _make_ohlcv(
    n: int = 160,
    start_price: float = 100.0,
    drift: float = 0.001,
    volume: float = 1_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    rets = rng.normal(drift, 0.012, size=n)
    closes = start_price * np.cumprod(1.0 + rets)
    return pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(n, volume, dtype=float),
        },
        index=dates,
    )


class TestUniverseAudition(unittest.TestCase):
    def test_rect_correlation_matrix_matches_full_matrix(self) -> None:
        histories = {
            "SPY": _make_ohlcv(120, 500.0, 0.0005, seed=1),
            "ISF.L": _make_ohlcv(120, 800.0, 0.0003, seed=2),
            "MSFT": _make_ohlcv(120, 400.0, 0.0010, seed=3),
            "AZN.L": _make_ohlcv(120, 12000.0, 0.0006, seed=4),
            "NVDA": _make_ohlcv(120, 130.0, 0.0015, seed=5),
            "REL.L": _make_ohlcv(120, 3500.0, 0.0007, seed=6),
        }
        all_syms = ["MSFT", "AZN.L", "NVDA", "REL.L"]
        held = ["MSFT", "AZN.L"]

        full = compute_correlation_matrix(histories, all_syms, lookback_days=60)
        rect = compute_correlation_matrix(histories, all_syms, lookback_days=60, against_symbols=held)

        self.assertEqual(list(rect.index), held)
        self.assertEqual(list(rect.columns), all_syms)
        for h in held:
            for s in all_syms:
                if pd.isna(full.loc[h, s]):
                    self.assertTrue(pd.isna(rect.loc[h, s]))
                else:
                    self.assertAlmostEqual(float(rect.loc[h, s]), float(full.loc[h, s]), places=4)

    def test_wikipedia_parser_handles_us_and_uk_tables(self) -> None:
        us_html = """
        <table class="wikitable sortable">
          <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
          <tr><td>AAPL</td><td>Apple Inc.</td><td>IT</td></tr>
          <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
          <tr><td>MSFT[a]</td><td>Microsoft</td><td>IT</td></tr>
        """ + ("<tr><td>XOM</td><td>Exxon</td><td>Energy</td></tr>" * 25) + "</table>"
        us_syms = _parse_wikipedia_constituents(us_html, "SP500")
        self.assertIn("AAPL", us_syms)
        self.assertIn("BRK-B", us_syms)
        self.assertIn("MSFT", us_syms)

        uk_html = """
        <table class="wikitable sortable">
          <tr><th>Company</th><th>Ticker</th></tr>
          <tr><td>AstraZeneca</td><td>AZN</td></tr>
          <tr><td>BT Group</td><td>BT.A</td></tr>
          <tr><td>BAE Systems</td><td>BA.</td></tr>
        """ + ("<tr><td>Relx</td><td>REL</td></tr>" * 25) + "</table>"
        uk_syms = _parse_wikipedia_constituents(uk_html, "FTSE100")
        self.assertIn("AZN.L", uk_syms)
        self.assertIn("BA-.L", uk_syms)

    def test_db_eligibility_fundamentals_cache_and_active_pool(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                db = AdvisorDatabase(str(Path(tmp) / "test.db"))
                await db.init()

                await db.upsert_t212_instruments(
                    [
                        {"ticker": "AAPL_US_EQ", "yahoo_symbol": "AAPL", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "VWRPl_EQ", "yahoo_symbol": "VWRP.L", "currency_code": "GBP", "inst_type": "ETF"},
                    ],
                    account_scope="ISA",
                    full_snapshot=True,
                )
                await db.upsert_t212_instruments(
                    [
                        {"ticker": "AAPL_US_EQ", "yahoo_symbol": "AAPL", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "PLTR_US_EQ", "yahoo_symbol": "PLTR", "currency_code": "USD", "inst_type": "STOCK"},
                    ],
                    account_scope="INVEST",
                    full_snapshot=True,
                )

                elig = await db.get_t212_eligibility_by_yahoo()
                self.assertTrue(elig["AAPL"]["isa_eligible"])
                self.assertTrue(elig["AAPL"]["invest_eligible"])
                self.assertTrue(elig["VWRP.L"]["isa_eligible"])
                self.assertFalse(elig["VWRP.L"]["invest_eligible"])
                self.assertFalse(elig["PLTR"]["isa_eligible"])
                self.assertTrue(elig["PLTR"]["invest_eligible"])

                await db.upsert_fundamentals_cache(
                    [
                        {
                            "symbol": "AAPL",
                            "currency": "USD",
                            "sector": "Technology",
                            "market_cap": 3_000_000_000_000.0,
                            "roe_pct": 140.0,
                            "profit_margin_pct": 25.0,
                        }
                    ]
                )
                cached = await db.get_fundamentals_cache(["AAPL", "MSFT"], max_age_days=21)
                self.assertIn("AAPL", cached)
                self.assertNotIn("MSFT", cached)
                self.assertAlmostEqual(cached["AAPL"]["roe_pct"], 140.0)

                await db.replace_active_universe_pool(
                    [
                        {"symbol": "AAPL", "sector": "Technology", "quality_score": 92.0, "isa_eligible": True, "invest_eligible": True, "is_etf": False},
                        {"symbol": "VWRP.L", "sector": "ETF", "quality_score": 100.0, "isa_eligible": True, "invest_eligible": False, "is_etf": True},
                    ],
                    updated_at_uk="2026-11-16 12:00",
                )
                pool = await db.get_active_universe_pool()
                self.assertEqual(len(pool), 2)
                self.assertEqual(pool[0]["symbol"], "AAPL")
                self.assertTrue(pool[1]["is_etf"])

        asyncio.run(_run())

    def test_market_data_fundamentals_circuit_breaker_and_checkpoint(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                db = AdvisorDatabase(str(Path(tmp) / "test.db"))
                await db.init()
                await db.upsert_fundamentals_cache(
                    [{"symbol": "CACHED", "sector": "Healthcare", "roe_pct": 22.0}]
                )
                md = MarketDataService(db=db)

                class _FakeTicker:
                    def __init__(self, sym: str) -> None:
                        self.sym = sym

                    @property
                    def info(self) -> dict[str, Any]:
                        if self.sym == "GOOD1":
                            return {"quoteType": "EQUITY", "currency": "USD", "sector": "Technology", "returnOnEquity": 0.28, "marketCap": 5e10}
                        raise YFRateLimitError()

                with (
                    patch("src.data_providers.market_data.yf.Ticker", side_effect=_FakeTicker),
                    patch("src.data_providers.market_data.time.sleep", return_value=None),
                ):
                    res, stats = await md.get_fundamentals_cached(
                        ["CACHED", "GOOD1", "RL1", "RL2", "RL3", "SKIP1", "SKIP2"],
                        max_age_days=21,
                        concurrency=1,
                        delay_seconds=0.0,
                        max_consecutive_429=3,
                    )

                self.assertEqual(stats["cached"], 1)
                self.assertEqual(stats["fetched"], 1)
                self.assertTrue(stats["rate_limited"])
                self.assertEqual(stats["skipped"], 2)
                self.assertIn("GOOD1", res)

                # Verify GOOD1 was checkpointed to SQLite before exit
                persisted = await db.get_fundamentals_cache(["GOOD1"], max_age_days=21)
                self.assertIn("GOOD1", persisted)
                self.assertAlmostEqual(persisted["GOOD1"]["roe_pct"], 28.0)

        asyncio.run(_run())

    def test_universe_audition_and_screener_shared_pool_compliance(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                db = AdvisorDatabase(str(Path(tmp) / "test.db"))
                await db.init()

                # Populate T212 eligibility: US1/US2/UK1/EU1 in both; INVONLY only in Invest; VWRP.L only in ISA
                await db.upsert_t212_instruments(
                    [
                        {"ticker": "US1_US_EQ", "yahoo_symbol": "US1", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "US2_US_EQ", "yahoo_symbol": "US2", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "ILLIQ_US_EQ", "yahoo_symbol": "ILLIQ", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "UK1_UK_EQ", "yahoo_symbol": "UK1.L", "currency_code": "GBX", "inst_type": "STOCK"},
                        {"ticker": "EU1_DE_EQ", "yahoo_symbol": "EU1.DE", "currency_code": "EUR", "inst_type": "STOCK"},
                        {"ticker": "VWRPl_EQ", "yahoo_symbol": "VWRP.L", "currency_code": "GBP", "inst_type": "ETF"},
                    ],
                    account_scope="ISA",
                    full_snapshot=True,
                )
                await db.upsert_t212_instruments(
                    [
                        {"ticker": "US1_US_EQ", "yahoo_symbol": "US1", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "US2_US_EQ", "yahoo_symbol": "US2", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "INVONLY_US_EQ", "yahoo_symbol": "INVONLY", "currency_code": "USD", "inst_type": "STOCK"},
                        {"ticker": "UK1_UK_EQ", "yahoo_symbol": "UK1.L", "currency_code": "GBX", "inst_type": "STOCK"},
                        {"ticker": "EU1_DE_EQ", "yahoo_symbol": "EU1.DE", "currency_code": "EUR", "inst_type": "STOCK"},
                    ],
                    account_scope="INVEST",
                    full_snapshot=True,
                )

                await db.upsert_fundamentals_cache(
                    [
                        {"symbol": "US1", "sector": "Technology", "market_cap": 2e11, "roe_pct": 30.0, "profit_margin_pct": 22.0, "forward_pe": 22.0},
                        {"symbol": "US2", "sector": "Technology", "market_cap": 1.5e11, "roe_pct": 26.0, "profit_margin_pct": 18.0, "forward_pe": 24.0},
                        {"symbol": "INVONLY", "sector": "Industrials", "market_cap": 3e10, "roe_pct": 19.0, "profit_margin_pct": 14.0, "forward_pe": 20.0},
                        {"symbol": "UK1.L", "sector": "Healthcare", "market_cap": 1e11, "roe_pct": 24.0, "profit_margin_pct": 20.0, "forward_pe": 18.0},
                        {"symbol": "EU1.DE", "sector": "Industrials", "market_cap": 8e10, "roe_pct": 21.0, "profit_margin_pct": 15.0, "forward_pe": 19.0},
                    ]
                )

                class _FakeMD(MarketDataService):
                    async def get_histories(self, symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
                        out: dict[str, pd.DataFrame] = {}
                        for i, sym in enumerate(symbols):
                            if sym == "ILLIQ":
                                out[sym] = _make_ohlcv(160, 10.0, 0.0001, volume=1_000.0, seed=10)  # $10k/day -> filtered
                            elif sym == "UK1.L":
                                out[sym] = _make_ohlcv(160, 10000.0, 0.001, volume=100_000.0, seed=11)  # £100/sh * 100k = $12.5M/day
                            else:
                                out[sym] = _make_ohlcv(160, 100.0, 0.001, volume=500_000.0, seed=20 + i)
                        return out

                class _FakeT212:
                    async def sync_all_accounts_instruments_metadata(self) -> dict[str, int]:
                        return {"isa_count": 6, "invest_count": 5}

                cfg = AppConfig(
                    env=EnvSettings(),
                    universe=UniverseSection(
                        audition_enabled=True,
                        audition_target_size=20,
                        audition_sector_cap=10,
                        audition_min_uk_eu_slots=2,
                        audition_min_avg_dollar_vol_usd=1_000_000.0,
                        audition_indices=["SP500"],
                        etf_symbols=["VWRP.L"],
                        isa_long_term=["US1", "UK1.L", "EU1.DE", "ILLIQ"],
                        invest_tactical=["US2", "INVONLY", "NOT_ON_T212"],
                    ),
                )
                md = _FakeMD(db=db)
                fake_base = {
                    "equities": {
                        "US1": ["SP500"],
                        "US2": ["SP500"],
                        "ILLIQ": ["SP500"],
                        "INVONLY": ["SP500"],
                        "NOT_ON_T212": ["SP500"],
                        "UK1.L": ["FTSE100"],
                        "EU1.DE": ["EUROSTOXX50"],
                    },
                    "etfs": ["VWRP.L"],
                    "counts_by_index": {"SP500": 5},
                }

                with patch("src.engine.universe_seeds.resolve_base_universe_symbols", return_value=fake_base):
                    res = await run_universe_audition(
                        cfg,
                        _FakeT212(),
                        md,
                        db,
                        sync_online_indices=False,
                        sync_t212_metadata=False,
                    )

                self.assertEqual(res["t212_tradable_count"], 6)  # NOT_ON_T212 excluded
                self.assertEqual(res["liquidity_passed_count"], 5)  # ILLIQ excluded
                self.assertEqual(res["selected_equities_count"], 5)
                self.assertEqual(res["selected_etfs_count"], 1)

                active_rows = await db.get_active_universe_pool()
                portfolio = PortfolioData(
                    fetched_at_uk="2026-11-16 12:00",
                    fetched_at_ms=1000,
                    isa=AccountSnapshot(
                        ok=True,
                        account_label="ISA",
                        positions=[
                            PositionItem(
                                account="ISA",
                                ticker="HOLD1",
                                raw_ticker="HOLD_US_EQ",
                                yahoo_symbol="HOLD1",
                                name="Holding 1",
                                inst_currency="USD",
                                quantity=10.0,
                                avg_price=95.0,
                                cur_price=100.0,
                                total_cost=950.0,
                                current_value=1000.0,
                                pnl=50.0,
                                fx_impact=0.0,
                                return_pct=5.26,
                            )
                        ],
                    ),
                    invest=AccountSnapshot(ok=True, account_label="Invest"),
                )
                with patch.object(
                    md,
                    "get_bundles",
                    return_value=({}, {}),
                ):
                    isa_cands, invest_cands = await run_quant_screener(
                        portfolio=portfolio,
                        market_data=md,
                        universe=cfg.universe,
                        watchlist=[],
                        strategy=cfg.strategy,
                        active_pool_rows=active_rows,
                    )

                isa_syms = {c.symbol for c in isa_cands}
                invest_syms = {c.symbol for c in invest_cands}
                self.assertNotIn("INVONLY", isa_syms)  # ISA compliance gate
                self.assertIn("INVONLY", invest_syms)
                self.assertIn("VWRP.L", isa_syms)
                self.assertNotIn("VWRP.L", invest_syms)  # LSE ETFs stay ISA-focused
                self.assertIn("UK1.L", invest_syms)  # Shared vetted equity pool!
                self.assertIn("US2", isa_syms)  # Shared vetted equity pool!

        asyncio.run(_run())

    def test_telegram_refresh_pool_command(self) -> None:
        from src.bot.scheduler import TradingScheduler

        async def _run() -> None:
            sent: list[str] = []
            edits: list[str] = []

            class _FakeTG:
                async def send_typing(self, chat_id: str) -> None:
                    return None

                async def send_status(self, chat_id: str, text: str) -> int:
                    return 99

                async def edit_status(self, chat_id: str, msg_id: int, text: str) -> None:
                    edits.append(text)

                async def send_message(self, chat_id: str, text: str, **kwargs: Any) -> None:
                    sent.append(text)

            cfg = AppConfig(env=EnvSettings(telegram_chat_id="12345"))
            md = MarketDataService()
            tg = _FakeTG()
            sched = TradingScheduler(
                cfg,
                None,  # type: ignore[arg-type]
                md,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                tg,  # type: ignore[arg-type]
            )
            bot = AdvisorBot(
                cfg=cfg,
                t212=None,  # type: ignore[arg-type]
                market_data=md,
                llm=None,  # type: ignore[arg-type]
                db=None,  # type: ignore[arg-type]
                telegram=tg,  # type: ignore[arg-type]
                scheduler=sched,
            )
            self.assertIs(bot.execution_lock, sched.execution_lock)

            fake_res = {
                "updated_at_uk": "2026-11-16 15:00",
                "base_equities_count": 850,
                "t212_tradable_count": 720,
                "liquidity_passed_count": 480,
                "selected_equities_count": 150,
                "selected_etfs_count": 25,
                "by_region": {"US": 105, "UK": 30, "EU": 15},
                "by_sector": {"Technology": 28, "Healthcare": 22},
                "added_symbols": ["NVDA", "AZN.L"],
                "removed_symbols": [],
                "fundamentals_stats": {"cached": 400, "fetched": 80, "failed": 0, "skipped": 0, "rate_limited": False},
                "top_preview": ["NVDA(92, Technology)", "AZN.L(88, Healthcare)"],
            }
            with patch("src.bot.scheduler.run_universe_audition", return_value=fake_res):
                # Run with timeout to fail immediately on any re-entrant execution_lock deadlock
                await asyncio.wait_for(
                    bot.handle_update({"message": {"chat": {"id": 12345}, "text": "/refresh_pool"}}, wait=True),
                    timeout=3.0,
                )
            self.assertEqual(len(edits), 1)
            self.assertEqual(len(sent), 1)
            self.assertIn("精选个股 150 只", sent[0])
            self.assertIn("美股 105 | 英股 30 | 欧股 15", sent[0])
            self.assertFalse(sched.execution_lock.locked())
            self.assertIsNone(sched.current_job_name)

        asyncio.run(_run())

    def test_refresh_pool_cli_entrypoint(self) -> None:
        from src import refresh_pool

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "cli.db"
            cfg = AppConfig(
                env=EnvSettings(),
                universe=UniverseSection(audition_enabled=True, audition_indices=["SP500"]),
            )
            cfg.app.db_path = str(db_path)

            async def _fake_audition(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"selected_equities_count": 150, "selected_etfs_count": 25}

            with (
                patch("src.refresh_pool.load_config", return_value=cfg),
                patch("src.refresh_pool.run_universe_audition", side_effect=_fake_audition),
            ):
                rc = refresh_pool.main(["--no-online-indices", "--no-t212-sync"])
            self.assertEqual(rc, 0)
            self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()
