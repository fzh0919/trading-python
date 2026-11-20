"""v5.5.9 M6: shadow-counterfactual isolation, T+20 settlement, exit-rule A/B, regime attribution.

All offline. The isolation battery is the heart of this module: SHADOW_REJECTED /
SHADOW_SETTLED rows must never leak into live win-rate stats, recent-pick display,
dedupe windows, pending settlement queues or lessons.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import numpy as np
import pandas as pd

from src.agents.pick_agent import (
    _collect_critic_shadow_rows,
    _collect_guardrail_shadow_rows,
    _merge_shadow_rows_cap,
)
from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, StrategySection, UniverseSection
from src.data_providers.analytics import compute_regime_attribution
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, T212Client
from src.engine.portfolio_rules.models_and_friction import RiskOverrideRecord, RiskOverrideResult
from src.engine.screener.audit import (
    _backtest_exit_rule_ab,
    run_offline_universe_and_threshold_audit,
)
from src.storage.db import AdvisorDatabase

_NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)
_NOW_UK = "2026-11-20 22:00"


def _ohlcv_from_close(closes: np.ndarray, start: str = "2026-10-01", freq: str = "B") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq=freq)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes * 0.996,
            "High": closes * 1.012,
            "Low": closes * 0.988,
            "Close": closes,
            "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=dates,
    )


class _StubTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str | int, text: str) -> None:
        self.sent.append((str(chat_id), text))


class _StubT212(T212Client):
    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return PortfolioData(
            fetched_at_uk=_NOW_UK,
            fetched_at_ms=1795000000000,
            total_combined_value=10000.0,
            total_combined_cash=2500.0,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=10000.0, total_cash=2500.0),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )


class _FailLLM:
    """Any generate() call in the shadow lane is a bug — settlement must stay 0-token."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise AssertionError("shadow settlement must never call the LLM")


class _StubMarketData(MarketDataService):
    def __init__(self, histories: dict[str, pd.DataFrame]) -> None:
        self._hist = histories
        self.fetch_calls: list[list[str]] = []

    async def get_histories(self, symbols, period: str = "1y", **kwargs) -> dict[str, pd.DataFrame]:
        self.fetch_calls.append(list(symbols))
        return {s: self._hist[s] for s in symbols if s in self._hist}


def _cfg() -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, deepseek_api_key="", ops_notify_url="", telegram_chat_id="123"),
        universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
    )


def _shadow_row(symbol: str, trigger: str = "RR_BELOW_FLOOR", **overrides) -> dict:
    row = {
        "account": "Invest",
        "symbol": symbol,
        "action": "BUY",
        "entry_price": 100.0,
        "thesis": "counterfactual",
        "horizon_days": 20,
        "benchmark_symbol": "SPY",
        "shadow_trigger": trigger,
        "shadow_reason": f"{trigger} test row",
        "regime": "NEUTRAL",
    }
    row.update(overrides)
    return row


class _M6DbTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = AdvisorDatabase(Path(self._tmp.name) / "advisor.db")
        await self.db.init()


class TestM6SchemaMigration(_M6DbTestCase):
    async def test_shadow_columns_added_idempotently_without_touching_legacy_rows(self) -> None:
        legacy_path = Path(self._tmp.name) / "legacy.db"
        legacy_ddl = """
        CREATE TABLE pick_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_uk TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            account TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL,
            thesis TEXT DEFAULT '',
            outcome_status TEXT DEFAULT 'PENDING',
            source TEXT DEFAULT 'pick'
        );
        """
        async with aiosqlite.connect(legacy_path) as conn:
            await conn.executescript(legacy_ddl)
            await conn.execute(
                "INSERT INTO pick_recommendations (created_at_uk, created_at_ms, account, symbol, action) "
                "VALUES ('2026-01-05 09:00', 1767600000000, 'ISA', 'AZN.L', 'BUY')"
            )
            await conn.commit()

        db = AdvisorDatabase(legacy_path)
        await db.init()
        # A second init on the already-migrated DB must be a no-op (idempotency).
        await db.init()

        async with aiosqlite.connect(legacy_path) as conn:
            cur = await conn.execute("PRAGMA table_info(pick_recommendations)")
            cols = {str(row[1]) for row in await cur.fetchall()}
            cur2 = await conn.execute(
                "SELECT account, symbol, outcome_status FROM pick_recommendations"
            )
            rows = await cur2.fetchall()
        self.assertTrue({"regime", "shadow_trigger", "shadow_reason"}.issubset(cols))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "ISA")
        self.assertEqual(rows[0][1], "AZN.L")
        self.assertEqual(rows[0][2], "PENDING")


class TestM6ShadowIsolation(_M6DbTestCase):
    async def test_shadow_rows_never_leak_into_live_stats_display_or_queues(self) -> None:
        # One live settled winner, one live pending, two shadows.
        await self.db.log_recommendations(
            [
                {
                    "account": "Invest",
                    "symbol": "PLTR",
                    "action": "BUY",
                    "entry_price": 50.0,
                    "thesis": "live winner",
                    "horizon_days": 5,
                    "benchmark_symbol": "SPY",
                },
                {
                    "account": "ISA",
                    "symbol": "AZN.L",
                    "action": "BUY",
                    "entry_price": 30.0,
                    "thesis": "live pending",
                    "horizon_days": 5,
                    "benchmark_symbol": "^FTSE",
                },
            ],
            source="pick",
        )
        pending = await self.db.get_pending_settlements()
        self.assertEqual({r["symbol"] for r in pending}, {"PLTR", "AZN.L"})
        live_pltr = next(r for r in pending if r["symbol"] == "PLTR")
        self.assertTrue(
            await self.db.settle_recommendation(
                live_pltr["id"],
                benchmark_symbol="SPY",
                settled_at_uk=_NOW_UK,
                raw_return_pct=8.0,
                alpha_pct=2.0,
                outcome_status="SETTLED",
                reflection_lesson="lesson text",
            )
        )

        inserted = await self.db.log_shadow_rejections(
            [_shadow_row("NVDA", "RR_BELOW_FLOOR"), _shadow_row("MSFT", "CRITIC", account="ISA")],
            regime="NEUTRAL",
        )
        self.assertEqual(inserted, 2)

        stats = await self.db.get_recommendation_stats()
        self.assertEqual(stats["total_count"], 2)  # live rows only
        self.assertEqual(stats["pending_count"], 1)
        self.assertEqual(stats["settled_count"], 1)
        self.assertEqual(stats["win_count"], 1)
        self.assertEqual(stats["shadow_count"], 2)

        recent = await self.db.get_recent_recommendations(limit=10, exclude_core_dca=True)
        self.assertEqual({r["symbol"] for r in recent}, {"PLTR", "AZN.L"})
        recent_default = await self.db.get_recent_recommendations(limit=10)
        self.assertEqual({r["symbol"] for r in recent_default}, {"PLTR", "AZN.L"})

        live_pending = await self.db.get_pending_settlements()
        self.assertEqual({r["symbol"] for r in live_pending}, {"AZN.L"})

        shadow_pending = await self.db.get_pending_shadow_settlements()
        self.assertEqual({r["symbol"] for r in shadow_pending}, {"NVDA", "MSFT"})
        for row in shadow_pending:
            self.assertEqual(row["outcome_status"], "SHADOW_REJECTED")
            self.assertEqual(row["source"], "shadow")
            self.assertEqual(row["regime"], "NEUTRAL")

        # Cross-lane settlement guards: live updater rejects shadows and vice versa.
        shadow_id = shadow_pending[0]["id"]
        self.assertFalse(
            await self.db.settle_recommendation(
                shadow_id,
                benchmark_symbol="SPY",
                settled_at_uk=_NOW_UK,
                raw_return_pct=None,
                alpha_pct=None,
                outcome_status="SETTLED",
                reflection_lesson="",
            )
        )
        self.assertFalse(
            await self.db.settle_shadow_rejection(
                live_pending[0]["id"],
                benchmark_symbol="^FTSE",
                settled_at_uk=_NOW_UK,
                raw_return_pct=None,
                t20_return_pct=None,
                t20_alpha_pct=None,
            )
        )
        # The refused live settle must not have mutated the shadow row's status.
        still_shadow = await self.db.get_pending_shadow_settlements()
        self.assertEqual(len(still_shadow), 2)

        lessons = await self.db.get_symbol_lessons("NVDA")
        self.assertEqual(lessons, [])
        outcome_series = await self.db.get_settled_outcome_series()
        self.assertEqual({r["symbol"] for r in outcome_series}, {"PLTR"})

    async def test_shadow_row_never_suppresses_a_later_real_pick(self) -> None:
        # L1 regression: a same-symbol shadow inside the 5-day window must NOT
        # block a legitimate live recommendation (dedupe sees status, not symbol).
        await self.db.log_shadow_rejections([_shadow_row("PLTR")], regime="NEUTRAL")

        await self.db.log_recommendations(
            [
                {
                    "account": "Invest",
                    "symbol": "PLTR",
                    "action": "BUY",
                    "entry_price": 50.0,
                    "thesis": "real pick after a shadow rejection",
                    "horizon_days": 5,
                    "benchmark_symbol": "SPY",
                }
            ],
            source="pick",
            dedupe_window_days=5,
            dedupe_cross_source=True,
        )
        pending = await self.db.get_pending_settlements()
        self.assertEqual({r["symbol"] for r in pending}, {"PLTR"})
        shadow_pending = await self.db.get_pending_shadow_settlements()
        self.assertEqual(len(shadow_pending), 1)

    async def test_shadow_same_day_dedupe_keeps_sample_clean(self) -> None:
        first = await self.db.log_shadow_rejections([_shadow_row("NVDA", regime="RISK_ON")], regime="NEUTRAL")
        second = await self.db.log_shadow_rejections([_shadow_row("NVDA", regime="RISK_ON")], regime="NEUTRAL")
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        shadow_pending = await self.db.get_pending_shadow_settlements()
        self.assertEqual(len(shadow_pending), 1)
        self.assertEqual(shadow_pending[0]["regime"], "RISK_ON")


class TestM6ShadowStats(_M6DbTestCase):
    async def test_avoided_loser_missed_winner_neutral_and_by_regime(self) -> None:
        await self.db.log_shadow_rejections(
            [
                _shadow_row("WIN1", regime="RISK_ON"),
                _shadow_row("LOSE1", regime="NEUTRAL"),
                _shadow_row("MID1", regime="NEUTRAL"),
            ],
            regime="NEUTRAL",
        )
        pending = {r["symbol"]: r for r in await self.db.get_pending_shadow_settlements()}

        for sym, t20 in (("WIN1", 12.0), ("LOSE1", -5.0), ("MID1", 3.0)):
            self.assertTrue(
                await self.db.settle_shadow_rejection(
                    pending[sym]["id"],
                    benchmark_symbol="SPY",
                    settled_at_uk=_NOW_UK,
                    raw_return_pct=t20,
                    t20_return_pct=t20,
                    t20_alpha_pct=t20 - 1.0,
                )
            )
        # Re-settling is a no-op (idempotent status transition).
        self.assertFalse(
            await self.db.settle_shadow_rejection(
                pending["WIN1"]["id"],
                benchmark_symbol="SPY",
                settled_at_uk=_NOW_UK,
                raw_return_pct=0.0,
                t20_return_pct=0.0,
                t20_alpha_pct=0.0,
            )
        )

        stats = await self.db.get_shadow_rejection_stats()
        self.assertEqual(stats["settled"], 3)
        self.assertEqual(stats["avoided_loser"], 1)
        self.assertEqual(stats["missed_winner"], 1)
        self.assertEqual(stats["neutral"], 1)
        self.assertEqual(stats["avoided_loser_rate_pct"], 33.3)
        self.assertEqual(stats["missed_winner_rate_pct"], 33.3)
        self.assertAlmostEqual(stats["avg_t20_return_pct"], 3.33, places=2)
        self.assertTrue(stats["small_sample"])
        self.assertEqual(stats["by_regime"]["RISK_ON"]["missed_winner"], 1)
        self.assertEqual(stats["by_regime"]["NEUTRAL"]["avoided_loser"], 1)
        self.assertEqual(stats["by_regime"]["NEUTRAL"]["settled"], 2)
        # Live stats remain untouched by the shadow lane.
        live_stats = await self.db.get_recommendation_stats()
        self.assertEqual(live_stats["total_count"], 0)
        self.assertEqual(live_stats["shadow_count"], 3)


class TestM6ShadowCollectors(unittest.TestCase):
    def test_guardrail_collector_filters_trigger_family_and_kept_symbols(self) -> None:
        hydrated = [
            SimpleNamespace(symbol="PLTR", account="Invest", entry_price=50.0, thesis="t1"),
            SimpleNamespace(symbol="HELD.L", account="ISA", entry_price=0.0, thesis="t2"),
            SimpleNamespace(symbol="ADVD", account="Invest", entry_price=80.0, thesis="t3"),
        ]
        override_res = RiskOverrideResult(
            decision=None,
            overrides=[
                RiskOverrideRecord(
                    account="Invest", symbol="PLTR", from_action="BUY", to_action="WATCH",
                    trigger_type="RR_BELOW_FLOOR", reason="R:R 1.40 < 2.0 floor",
                ),
                RiskOverrideRecord(
                    account="Invest", symbol="ADVD", from_action="ADD", to_action="WATCH",
                    trigger_type="RR_UNVERIFIED", reason="no numeric stop/target",
                ),
                RiskOverrideRecord(
                    account="ISA", symbol="HELD.L", from_action="ADD", to_action="HOLD",
                    trigger_type="FREQUENCY_CAP", reason="weekly cap",
                ),
            ],
        )
        rows = _collect_guardrail_shadow_rows(
            hydrated, override_res, kept_symbols=set(), bundles={}, regime="NEUTRAL"
        )
        self.assertEqual({r["symbol"] for r in rows}, {"PLTR", "ADVD"})
        by_sym = {r["symbol"]: r for r in rows}
        self.assertEqual(by_sym["PLTR"]["shadow_trigger"], "RR_BELOW_FLOOR")
        self.assertEqual(by_sym["PLTR"]["benchmark_symbol"], "SPY")
        self.assertEqual(by_sym["PLTR"]["regime"], "NEUTRAL")
        self.assertEqual(by_sym["ADVD"]["entry_price"], 80.0)  # hydrated entry price fallback
        # FREQUENCY_CAP is not an R:R-floor trigger -> HELD.L never becomes a shadow.

        # Advise-mode guard: the ticket survived -> it is a real pick, not a shadow.
        rows_kept = _collect_guardrail_shadow_rows(
            hydrated, override_res, kept_symbols={"PLTR", "ADVD"}, bundles={}, regime=""
        )
        self.assertEqual(rows_kept, [])

        # No positive price anchor -> row skipped entirely.
        hydrated_no_px = [SimpleNamespace(symbol="PLTR", account="Invest", entry_price=0.0, thesis="")]
        rows_no_px = _collect_guardrail_shadow_rows(
            hydrated_no_px, override_res, kept_symbols=set(), bundles={}, regime=""
        )
        self.assertEqual(rows_no_px, [])

    def test_critic_collector_uses_candidates_only_and_bundles_for_anchor(self) -> None:
        isa_cands = [SimpleNamespace(symbol="CCC", track="ISA_LONG_TERM")]
        invest_cands = [SimpleNamespace(symbol="BBB", track="INVEST_TACTICAL")]
        bundles = {"BBB": SimpleNamespace(current_price=42.0)}
        rows = _collect_critic_shadow_rows(
            proposal_syms={"AAA", "BBB", "HELD1"},
            kept_symbols={"AAA"},
            held_symbols={"HELD1"},
            isa_cands=isa_cands,
            invest_cands=invest_cands,
            bundles=bundles,
            regime="RECOVERY",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "BBB")
        self.assertEqual(rows[0]["account"], "Invest")
        self.assertEqual(rows[0]["entry_price"], 42.0)
        self.assertEqual(rows[0]["shadow_trigger"], "CRITIC")
        self.assertEqual(rows[0]["regime"], "RECOVERY")

    def test_merge_cap_prioritises_rr_floor_and_caps_at_two(self) -> None:
        guardrail_rows = [
            _shadow_row("RRX", "RR_BELOW_FLOOR"),
            _shadow_row("RRU", "RR_UNVERIFIED"),
        ]
        critic_rows = [
            _shadow_row("CRI1", "CRITIC"),
            _shadow_row("CRI2", "CRITIC"),
        ]
        merged = _merge_shadow_rows_cap(guardrail_rows, critic_rows)
        self.assertEqual(len(merged), 2)
        self.assertEqual([r["shadow_trigger"] for r in merged], ["RR_BELOW_FLOOR", "RR_UNVERIFIED"])


class TestM6ShadowSettlementJob(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = AdvisorDatabase(Path(self._tmp.name) / "advisor.db")
        await self.db.init()
        # A US ticker whose 20-trading-day horizon has long matured by _NOW.
        pltr_closes = np.concatenate([np.full(25, 100.0), np.full(30, 112.0)])
        self.hist = {
            "PLTR": _ohlcv_from_close(pltr_closes, start="2026-10-01"),
            "SPY": _ohlcv_from_close(np.full(55, 500.0), start="2026-10-01"),
        }
        self.market = _StubMarketData(self.hist)
        self.llm = _FailLLM()
        self.sched = TradingScheduler(
            _cfg(), _StubT212(), self.market, self.llm, self.db, _StubTg()
        )

    async def test_t20_shadow_settlement_zero_token_and_isolated(self) -> None:
        self.assertEqual(
            await self.db.log_shadow_rejections(
                [
                    _shadow_row(
                        "PLTR",
                        created_at_uk="2026-10-09 12:00",
                        created_at_ms=1760000000000,
                        regime="RISK_ON",
                    )
                ],
                regime="NEUTRAL",
            ),
            1,
        )
        summary = await self.sched._run_daily_settlement_impl(now_utc=_NOW)

        self.assertEqual(summary["shadow_settled"], 1)
        self.assertEqual(summary["checked"], 0)  # no live rows were perturbed
        self.assertEqual(summary["settled"], 0)
        self.assertEqual(self.llm.calls, [])  # shadow lane is strictly 0-token

        shadow_pending = await self.db.get_pending_shadow_settlements()
        self.assertEqual(shadow_pending, [])
        stats = await self.db.get_shadow_rejection_stats()
        self.assertEqual(stats["settled"], 1)
        self.assertEqual(stats["missed_winner"], 1)  # +12% >= +10% -> 错杀大牛股
        self.assertEqual(stats["by_regime"]["RISK_ON"]["missed_winner"], 1)

        live_stats = await self.db.get_recommendation_stats()
        self.assertEqual(live_stats["total_count"], 0)
        self.assertEqual(live_stats["settled_count"], 0)
        self.assertEqual(live_stats["shadow_count"], 1)

        # The shared history batch must cover the shadow symbol exactly once
        # (zero extra fetches attributable to the shadow lane; the other calls
        # come from the weekly-audit catch-up and NAV bench on this stub).
        pltr_calls = sum("PLTR" in call for call in self.market.fetch_calls)
        self.assertEqual(pltr_calls, 1)


class TestM6ExitRuleAb(unittest.TestCase):
    def _entry_context(self, closes: np.ndarray, freq: str = "B") -> dict:
        df = _ohlcv_from_close(closes, start="2026-01-01", freq=freq)
        highs, lows, closes_s = df["High"], df["Low"], df["Close"]
        from src.engine.screener.audit import _backtest_entry_series

        ctx = _backtest_entry_series(highs, lows, closes_s)
        assert ctx is not None
        return {"df": df, "ctx": ctx}

    def test_paired_samples_and_symmetric_atr_stop_on_crash(self) -> None:
        # Steady rise to satisfy MA50>MA200, then a hard crash: both arms must take
        # the identical ATR stop on the identical entry, so net returns are equal.
        rise = np.linspace(100.0, 180.0, 240)
        crash = np.array([150.0, 110.0, 90.0, 70.0])
        closes = np.concatenate([rise, crash])
        df = _ohlcv_from_close(closes, start="2026-01-01")
        # A linear rise never beats the prior day's High(=Close*1.012), so stamp a
        # clean breakout bar (same convention as the v5.5.8 audit fixture).
        df.iloc[210, df.columns.get_loc("Close")] = 180.0
        df.iloc[210, df.columns.get_loc("High")] = 180.5
        ohlcv_map = {"CRASH1": (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])}
        res = _backtest_exit_rule_ab(ohlcv_map, ["CRASH1"], StrategySection(), forward_bars=60, cooldown_bars=10_000)
        self.assertEqual(res["sample_count"], 1)
        self.assertEqual(res["baseline"]["sample_count"], 1)
        self.assertEqual(res["v559_ladder"]["sample_count"], 1)
        self.assertEqual(res["baseline"]["avg_return_pct"], res["v559_ladder"]["avg_return_pct"])
        self.assertEqual(res["baseline"]["exit_mix"]["atr_stop"], 1)
        self.assertEqual(res["v559_ladder"]["exit_mix"]["atr_stop"], 1)
        # Friction (US round-trip 0.40%) deducted on both arms.
        self.assertLess(res["baseline"]["avg_return_pct"], 0.0)
        self.assertIn("friction_model", res)
        self.assertIn("survivorship_note", res)
        self.assertIn("simplification_note", res)
        self.assertTrue(res["small_sample"])

    def test_ladder_arms_floor_and_ma20_trail_on_slow_bleed_winner(self) -> None:
        # Single entry stamped right before the move; +40% spike; then a slow
        # bleed that trips the baseline trailing TP while the ladder arm either
        # banks the MA20 trail (higher) or holds the +10% lock floor.
        rise = np.linspace(100.0, 140.0, 240)
        spike = np.linspace(140.0, 195.0, 20)
        bleed = np.linspace(195.0, 150.0, 40)
        closes = np.concatenate([rise, spike, bleed])
        df = _ohlcv_from_close(closes, start="2026-01-01")
        df.iloc[200, df.columns.get_loc("Close")] = 140.0
        df.iloc[200, df.columns.get_loc("High")] = 140.5
        ohlcv_map = {"WIN1": (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])}
        res = _backtest_exit_rule_ab(ohlcv_map, ["WIN1"], StrategySection(), forward_bars=90, cooldown_bars=10_000)
        self.assertEqual(res["sample_count"], 1)
        self.assertGreater(res["v559_ladder"]["avg_return_pct"], res["baseline"]["avg_return_pct"])
        self.assertEqual(res["baseline"]["exit_mix"]["trailing_tp"], 1)
        self.assertTrue(
            res["v559_ladder"]["exit_mix"]["ma20_trail"] + res["v559_ladder"]["exit_mix"]["ladder_floor"] >= 1
        )

    def test_uk_symbol_stamp_duty_vs_lse_etf_friction_lanes(self) -> None:
        from src.engine.screener.audit import _ab_round_trip_friction_pct

        strat = StrategySection()
        self.assertAlmostEqual(_ab_round_trip_friction_pct("SHEL.L", set(), strat), 0.60, places=6)
        self.assertAlmostEqual(_ab_round_trip_friction_pct("VWRL.L", {"VWRL.L"}, strat), 0.10, places=6)
        self.assertAlmostEqual(_ab_round_trip_friction_pct("NVDA", set(), strat), 0.40, places=6)

    def test_audit_payload_gains_ab_block_and_summary_line_rules(self) -> None:
        rise = np.linspace(100.0, 180.0, 240)
        crash = np.array([150.0, 110.0, 90.0, 70.0])
        closes = np.concatenate([rise, crash])
        brk_df = _ohlcv_from_close(closes, start="2026-01-01")
        brk_df.iloc[210, brk_df.columns.get_loc("Close")] = 180.0
        brk_df.iloc[210, brk_df.columns.get_loc("High")] = 180.5
        histories = {"BRK1": brk_df}
        portfolio = PortfolioData(
            fetched_at_uk=_NOW_UK,
            fetched_at_ms=1795000000000,
            isa=AccountSnapshot(ok=True, account_label="ISA"),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )
        universe = UniverseSection(isa_long_term=["BRK1"], invest_tactical=[], etf_symbols=[])
        audit = run_offline_universe_and_threshold_audit(histories, portfolio, universe, StrategySection())
        self.assertIn("threshold_backtest", audit)  # legacy keys untouched
        self.assertIn("exit_rule_ab", audit)
        self.assertIn("退出规则A/B", audit["summary_text"])
        self.assertNotIn("#", audit["summary_text"])
        self.assertNotIn("**", audit["summary_text"])
        self.assertIn("幸存者偏差", audit["summary_text"])


class TestM6RegimeAttribution(unittest.TestCase):
    NAV_ROWS = [
        {"uk_date": "2026-11-01", "total_value_gbp": 100.0, "cash_gbp": 30.0, "net_flow_gbp": 0.0},
        {"uk_date": "2026-11-02", "total_value_gbp": 110.0, "cash_gbp": 30.0, "net_flow_gbp": 0.0},
        {"uk_date": "2026-11-03", "total_value_gbp": 104.0, "cash_gbp": 32.0, "net_flow_gbp": 0.0},
        {"uk_date": "2026-11-04", "total_value_gbp": 106.0, "cash_gbp": 32.0, "net_flow_gbp": 0.0},
        {"uk_date": "2026-11-05", "total_value_gbp": 120.0, "cash_gbp": 36.0, "net_flow_gbp": 5.0},
        {"uk_date": "2026-11-06", "total_value_gbp": 118.0, "cash_gbp": 36.0, "net_flow_gbp": 0.0},
    ]
    REGIME_ROWS = [
        {"uk_date": "2026-11-02", "regime": "RISK_ON"},
        {"uk_date": "2026-11-03", "regime": "RISK_ON"},
        {"uk_date": "2026-11-04", "regime": "NEUTRAL"},
        {"uk_date": "2026-11-05", "regime": "NEUTRAL"},
        {"uk_date": "2026-11-06", "regime": "RISK_OFF"},
    ]

    def test_buckets_tw_position_cash_and_flow_adjusted_twr(self) -> None:
        res = compute_regime_attribution(self.NAV_ROWS, self.REGIME_ROWS)
        self.assertTrue(res["ok"])
        self.assertEqual(res["attributed_days"], 5)
        self.assertEqual(res["unattributed_days"], 0)

        on = res["buckets"]["RISK_ON"]
        self.assertEqual(on["days"], 2)
        # (1 + 0.10) * (1 - 6/110) - 1 = 0.03999...
        self.assertAlmostEqual(on["twr_pct"], 4.0, delta=0.05)
        self.assertAlmostEqual(on["avg_cash_pct"], (30 / 110 + 32 / 104) * 50.0, delta=0.1)
        self.assertAlmostEqual(on["avg_position_pct"], 100.0 - (30 / 110 + 32 / 104) * 50.0, delta=0.1)

        neutral = res["buckets"]["NEUTRAL"]
        self.assertEqual(neutral["days"], 2)
        # (1 + 2/104) * (1 + 115/106) - 1 = +10.58%
        self.assertAlmostEqual(neutral["twr_pct"], 10.58, delta=0.02)

        off = res["buckets"]["RISK_OFF"]
        self.assertEqual(off["days"], 1)
        self.assertAlmostEqual(off["twr_pct"], (118.0 / 120.0 - 1.0) * 100.0, delta=0.05)
        self.assertAlmostEqual(off["avg_cash_pct"], 36.0 / 118.0 * 100.0, delta=0.1)
        self.assertNotIn("RECOVERY", res["buckets"])

    def test_not_ok_below_two_attributed_days(self) -> None:
        res = compute_regime_attribution(self.NAV_ROWS[:2], [])
        self.assertFalse(res["ok"])
        res_one = compute_regime_attribution(self.NAV_ROWS, self.REGIME_ROWS[:1])
        self.assertFalse(res_one["ok"])
        self.assertEqual(res_one["unattributed_days"], 4)
        res_empty = compute_regime_attribution([], [])
        self.assertFalse(res_empty["ok"])


class TestM6RegimeHistoryRepo(_M6DbTestCase):
    async def test_regime_history_parses_packs_and_skips_malformed(self) -> None:
        async with aiosqlite.connect(self.db.db_path) as conn:
            rows = [
                ("2026-11-17", '{"deployment_gate": {"regime": "RISK_OFF"}}'),
                ("2026-11-18", '{"regime": "RISK_ON", "deployment_gate": {"regime": "NEUTRAL"}}'),
                ("2026-11-19", '{"regime": "NEUTRAL"}'),
                ("2026-11-20", '{"regime": "RECOVERY"}'),
                ("2026-11-21", "not-json"),
                ("2026-11-22", '{"other": 1}'),
            ]
            for day, pack in rows:
                await conn.execute(
                    "INSERT INTO morning_reports (cycle_date, report_text, conclusion_text, "
                    "context_pack_json, updated_at_ms) VALUES (?, 'r', 'c', ?, 1)",
                    (day, pack),
                )
            await conn.commit()

        history = await self.db.get_regime_history(limit=10)
        labels = [h["regime"] for h in history]
        dates = [h["uk_date"] for h in history]
        self.assertEqual(labels, ["RISK_OFF", "RISK_ON", "NEUTRAL", "RECOVERY"])
        self.assertEqual(dates, sorted(dates))  # ascending for attribution joins
        self.assertNotIn("2026-11-21", dates)
        self.assertNotIn("2026-11-22", dates)


class TestM6PerfMessage(unittest.IsolatedAsyncioTestCase):
    async def test_perf_message_renders_regime_attribution_line(self) -> None:
        from src.bot.handlers import AdvisorBot

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "perf.db")
            await db.init()
            nav_days = [
                ("2026-11-15", 1000.0, 200.0, 0.0, "NEUTRAL"),
                ("2026-11-16", 1010.0, 200.0, 0.0, "NEUTRAL"),
                ("2026-11-17", 1020.0, 210.0, 0.0, "RISK_ON"),
                ("2026-11-18", 1030.0, 210.0, 0.0, "RISK_ON"),
            ]
            for day, nav, cash, flow, regime in nav_days:
                await db.upsert_nav_history(
                    {
                        "uk_date": day,
                        "total_value_gbp": nav,
                        "cash_gbp": cash,
                        "invested_gbp": nav - cash,
                        "net_flow_gbp": flow,
                        "isa_value_gbp": nav * 0.6,
                        "invest_value_gbp": nav * 0.4,
                        "benchmark_close": nav,
                        "cash_rate_pct": 4.0,
                    }
                )
                async with aiosqlite.connect(db.db_path) as conn:
                    await conn.execute(
                        "INSERT INTO morning_reports (cycle_date, report_text, conclusion_text, "
                        "context_pack_json, updated_at_ms) VALUES (?, 'r', 'c', ?, 1)",
                        (day, f'{{"regime": "{regime}"}}'),
                    )
                    await conn.commit()

            class _FakeT212:
                @staticmethod
                async def get_portfolio(use_cache: bool = True) -> PortfolioData:
                    return PortfolioData(
                        fetched_at_uk="2026-11-18 09:00",
                        fetched_at_ms=1795000000000,
                        isa=AccountSnapshot(ok=True, account_label="ISA"),
                        invest=AccountSnapshot(ok=True, account_label="Invest"),
                    )

            class _FakeMarket:
                @staticmethod
                async def get_histories(symbols, period: str = "1y"):
                    return {}

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="test-key")
            )
            bot = AdvisorBot(cfg, _FakeT212(), _FakeMarket(), None, db, None)
            message = await bot._build_perf_message()

        self.assertIn("分形态归因", message)
        self.assertIn("RISK_ON", message)
        self.assertIn("NEUTRAL", message)
        self.assertNotIn("**", message)
        self.assertNotIn("#", message)


class TestM6StatsMessageShadowLine(_M6DbTestCase):
    async def test_stats_message_appends_shadow_scoreboard_only_when_settled(self) -> None:
        from src.bot.handlers import AdvisorBot

        # No shadow rows yet -> the line must not render.
        cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="k"))
        bot = AdvisorBot(cfg, None, None, None, self.db, None)
        empty_msg = await bot._build_stats_message()
        self.assertNotIn("影子拦截复盘", empty_msg)

        await self.db.log_shadow_rejections(
            [_shadow_row("NVDA", created_at_uk="2026-10-09 12:00", created_at_ms=1760000000000)],
            regime="NEUTRAL",
        )
        pending = await self.db.get_pending_shadow_settlements()
        self.assertTrue(
            await self.db.settle_shadow_rejection(
                pending[0]["id"],
                benchmark_symbol="SPY",
                settled_at_uk=_NOW_UK,
                raw_return_pct=-6.0,
                t20_return_pct=-6.0,
                t20_alpha_pct=-4.0,
            )
        )
        msg = await bot._build_stats_message()
        self.assertIn("影子拦截复盘", msg)
        self.assertIn("护栏避雷正确 100.0%", msg)
        self.assertIn("n<30 仅参考", msg)
        self.assertNotIn("#", msg)
        self.assertNotIn("**", msg)


if __name__ == "__main__":
    unittest.main()
