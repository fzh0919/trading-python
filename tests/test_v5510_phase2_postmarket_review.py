"""Offline tests for v5.5.10 Phase 2: post-close ledger, 2x2 process review, and weekly closure."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pandas as pd

from src.agents.pick_agent import _format_lessons_section
from src.agents.report_agent.fact_sheet import _compact_lesson_record, _format_lesson_entry
from src.bot.handlers import AdvisorBot
from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, StrategySection
from src.data_providers.analytics.postmarket_review import (
    build_eod_attribution_ledger,
    classify_process_quadrant,
    format_postmarket_alert_text,
    format_postmarket_review_text,
    is_substantive_catalyst,
)
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
)
from src.storage.db import AdvisorDatabase

_NOW = datetime(2026, 11, 20, 22, 30, tzinfo=timezone.utc)
_UK_DATE = "2026-11-20"


def _ohlcv(closes: list[float], start: str = "2026-10-05") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "Open": [c * 0.998 for c in closes],
            "High": [c * 1.006 for c in closes],
            "Low": [c * 0.994 for c in closes],
            "Close": [float(c) for c in closes],
            "Volume": [1_000_000.0 for _ in closes],
        },
        index=dates,
    )


def _flat_then_move(last_move_pct: float, n: int = 30) -> pd.DataFrame:
    closes = [100.0] * (n - 1) + [100.0 * (1.0 + last_move_pct / 100.0)]
    return _ohlcv(closes)


def _pos(
    account: str,
    symbol: str,
    *,
    qty: float = 10.0,
    avg: float = 100.0,
    cur: float = 100.0,
    pnl: float = 0.0,
    fx: float = 0.0,
    currency: str = "USD",
    name: str = "",
    value: float | None = None,
) -> PositionItem:
    return PositionItem(
        account=account,
        raw_ticker=symbol,
        ticker=symbol,
        yahoo_symbol=symbol,
        name=name or symbol,
        inst_currency=currency,
        quantity=qty,
        avg_price=avg,
        cur_price=cur,
        current_value=float(value if value is not None else qty * cur),
        total_cost=qty * avg,
        pnl=pnl,
        fx_impact=fx,
        return_pct=((cur / avg) - 1.0) * 100.0 if avg > 0 else 0.0,
    )


def _order(
    account: str,
    symbol: str,
    *,
    side: str = "BUY",
    qty: float = 10.0,
    price: float = 100.0,
    uk_time: str = f"{_UK_DATE} 15:00",
    order_type: str = "MARKET",
    currency: str = "USD",
) -> HistoricalOrderItem:
    return HistoricalOrderItem(
        id=f"{symbol}-{side}-{price}",
        order_id=f"{symbol}-{side}-{price}",
        account=account,
        raw_ticker=symbol,
        ticker=symbol,
        yahoo_symbol=symbol,
        name=symbol,
        inst_currency=currency,
        side=side,
        order_type=order_type,
        filled_at=f"{uk_time}:00Z",
        filled_at_ms=1_765_000_000_000,
        uk_time_str=uk_time,
        quantity=qty,
        price=price,
        net_value=qty * price,
        wallet_currency="GBP",
        realised_pnl=0.0,
        tax_total=0.0,
    )


def _pending_stop(symbol: str, *, qty: float, stop: float) -> PendingOrderItem:
    return PendingOrderItem(
        id=f"{symbol}-stop",
        account="Invest",
        raw_ticker=symbol,
        ticker=symbol,
        yahoo_symbol=symbol,
        name=symbol,
        side="SELL",
        order_type="STOP_LIMIT",
        quantity=qty,
        filled_quantity=0.0,
        status="NEW",
        value=qty * stop,
        limit_price=stop,
        stop_price=stop,
        currency="USD",
    )


def _portfolio(
    *,
    isa_positions: list[PositionItem] | None = None,
    invest_positions: list[PositionItem] | None = None,
    invest_pending: list[PendingOrderItem] | None = None,
    isa_orders: list[HistoricalOrderItem] | None = None,
    invest_orders: list[HistoricalOrderItem] | None = None,
    fetched_at_uk: str = f"{_UK_DATE} 21:30",
    fetched_at_ms: int = 1_765_200_000_000,
    is_fallback: bool = False,
) -> PortfolioData:
    isa = AccountSnapshot(
        ok=True,
        account_label="ISA",
        total_account_value=10_000.0,
        total_cash=2_000.0,
        positions=list(isa_positions or []),
        recent_orders=list(isa_orders or []),
        order_history_supported=True,
    )
    invest = AccountSnapshot(
        ok=True,
        account_label="Invest",
        total_account_value=10_000.0,
        total_cash=2_000.0,
        positions=list(invest_positions or []),
        pending_orders=list(invest_pending or []),
        recent_orders=list(invest_orders or []),
        order_history_supported=True,
    )
    return PortfolioData(
        fetched_at_uk=fetched_at_uk,
        fetched_at_ms=fetched_at_ms,
        isa=isa,
        invest=invest,
        total_combined_value=20_000.0,
        total_combined_cash=4_000.0,
        is_fallback=is_fallback,
    )


def _cfg(**strategy_overrides: Any) -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, deepseek_api_key="", telegram_chat_id="123"),
        strategy=StrategySection(**strategy_overrides),
    )


class _StubLLM:
    def __init__(self, reply: str = "复盘：纪律执行正常，继续控制仓位与止损。") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.raise_exc: Exception | None = None
        self.deepseek_api_key = "test-key"

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.reply


class _StubT212(T212Client):
    def __init__(self, portfolio: PortfolioData) -> None:
        self.portfolio = portfolio
        self.calls = 0

    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        self.calls += 1
        return self.portfolio


class _StubMarket(MarketDataService):
    def __init__(self, histories: dict[str, pd.DataFrame] | None = None) -> None:
        super().__init__()
        self.histories = histories or {}
        self.calls: list[list[str]] = []

    async def get_histories(self, symbols: list[str], period: str = "1y", **kwargs: Any) -> dict[str, pd.DataFrame]:
        self.calls.append(list(symbols))
        return {s: self.histories[s] for s in symbols if s in self.histories}


class _StubTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str | int, text: str) -> None:
        self.sent.append((str(chat_id), text))


class _DbCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = AdvisorDatabase(Path(self._tmp.name) / "advisor.db")
        await self.db.init()


class TestStep1Storage(_DbCase):
    async def test_legacy_pick_table_gets_new_columns_idempotently(self) -> None:
        legacy_path = Path(self._tmp.name) / "legacy.db"
        async with aiosqlite.connect(legacy_path) as conn:
            await conn.executescript(
                """
                CREATE TABLE pick_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    account TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    thesis TEXT DEFAULT '',
                    horizon_days INTEGER DEFAULT 5,
                    benchmark_symbol TEXT DEFAULT '',
                    settled_at_uk TEXT DEFAULT '',
                    raw_return_pct REAL,
                    alpha_pct REAL,
                    outcome_status TEXT DEFAULT 'PENDING',
                    reflection_lesson TEXT DEFAULT '',
                    t10_return_pct REAL,
                    t10_alpha_pct REAL,
                    t20_return_pct REAL,
                    t20_alpha_pct REAL,
                    mfe_pct REAL,
                    mae_pct REAL,
                    source TEXT DEFAULT 'pick',
                    fill_status TEXT DEFAULT '',
                    actual_filled INTEGER DEFAULT NULL,
                    actual_fill_price REAL DEFAULT NULL,
                    execution_slippage_pct REAL DEFAULT NULL,
                    regime TEXT DEFAULT '',
                    shadow_trigger TEXT DEFAULT '',
                    shadow_reason TEXT DEFAULT ''
                );
                """
            )
            await conn.commit()
        legacy = AdvisorDatabase(legacy_path)
        await legacy.init()
        async with aiosqlite.connect(legacy_path) as conn:
            cur = await conn.execute("PRAGMA table_info(pick_recommendations)")
            cols = {str(row[1]) for row in await cur.fetchall()}
            cur = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_postmarket_reviews'"
            )
            table = await cur.fetchone()
        self.assertIn("process_quadrant", cols)
        self.assertIn("holding_catalyst", cols)
        self.assertIsNotNone(table)
        await legacy.init()  # idempotent second pass

    async def test_settle_and_followup_roundtrip_process_fields(self) -> None:
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1_762_000_000_000,
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                }
            ]
        )
        rec = (await self.db.get_recent_recommendations(limit=1))[0]
        saved = await self.db.settle_recommendation(
            int(rec["id"]),
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-20 22:00",
            raw_return_pct=9.0,
            alpha_pct=7.0,
            outcome_status="SETTLED",
            reflection_lesson="复盘：执行尚可。",
            mfe_pct=12.0,
            mae_pct=-2.0,
            actual_filled=1,
            execution_slippage_pct=2.2,
            process_quadrant="BAD_WIN",
            holding_catalyst="财报超预期",
        )
        self.assertTrue(saved)
        row = (await self.db.get_recent_recommendations(limit=1))[0]
        self.assertEqual(row["process_quadrant"], "BAD_WIN")
        self.assertEqual(row["holding_catalyst"], "财报超预期")
        updated = await self.db.update_recommendation_followup(
            int(rec["id"]),
            t20_return_pct=15.0,
            t20_alpha_pct=10.0,
            mfe_pct=18.0,
            mae_pct=-3.0,
            reflection_lesson="T+20 复盘：趋势延续。",
            process_quadrant="GOOD_WIN",
            holding_catalyst="上调指引",
        )
        self.assertTrue(updated)
        row = (await self.db.get_recent_lessons(limit=1))[0]
        self.assertEqual(row["process_quadrant"], "GOOD_WIN")
        self.assertEqual(row["holding_catalyst"], "上调指引")

    async def test_postmarket_review_upsert_is_idempotent_and_latest_readable(self) -> None:
        await self.db.upsert_daily_postmarket_review(
            {
                "uk_date": _UK_DATE,
                "created_at_uk": "2026-11-20 22:05",
                "created_at_ms": 1_765_000_000_000,
                "regime": "RISK_ON",
                "local_pnl_gbp": 10.0,
                "fx_pnl_gbp": 2.0,
                "total_mark_pnl_gbp": 12.0,
                "discipline_alert_count": 1,
                "summary_text": "第一版账单",
                "payload": {"event_cards": [{"symbol": "NVDA", "text": "财报超预期"}]},
            }
        )
        await self.db.upsert_daily_postmarket_review(
            {
                "uk_date": _UK_DATE,
                "created_at_uk": "2026-11-20 22:15",
                "created_at_ms": 1_765_000_100_000,
                "regime": "RISK_ON",
                "local_pnl_gbp": 20.0,
                "fx_pnl_gbp": 3.0,
                "total_mark_pnl_gbp": 23.0,
                "discipline_alert_count": 0,
                "summary_text": "第二版账单",
                "payload": {"event_cards": []},
            }
        )
        latest = await self.db.get_latest_postmarket_review()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["summary_text"], "第二版账单")
        self.assertEqual(latest["local_pnl_gbp"], 20.0)
        self.assertEqual(latest["payload"]["event_cards"], [])
        rows = await self.db.get_recent_postmarket_reviews(limit=7)
        self.assertEqual(len(rows), 1)

    async def test_postmarket_prune_keeps_latest_one(self) -> None:
        for day in ("2026-01-01", "2026-01-02"):
            await self.db.upsert_daily_postmarket_review(
                {
                    "uk_date": day,
                    "created_at_uk": f"{day} 22:00",
                    "created_at_ms": int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000),
                    "summary_text": day,
                    "payload": {},
                }
            )
        pruned = await self.db.prune_old_records(retention_days=90, now_utc=_NOW)
        self.assertEqual(pruned["daily_postmarket_reviews"], 1)
        rows = await self.db.get_recent_postmarket_reviews(limit=7)
        self.assertEqual([r["uk_date"] for r in rows], ["2026-01-02"])
        await self.db.upsert_daily_postmarket_review(
            {
                "uk_date": _UK_DATE,
                "created_at_uk": f"{_UK_DATE} 22:00",
                "created_at_ms": int(_NOW.timestamp() * 1000),
                "summary_text": _UK_DATE,
                "payload": {},
            }
        )
        pruned = await self.db.prune_old_records(retention_days=90, now_utc=_NOW)
        self.assertEqual(pruned["daily_postmarket_reviews"], 1)
        rows = await self.db.get_recent_postmarket_reviews(limit=7)
        self.assertEqual([r["uk_date"] for r in rows], [_UK_DATE])

    async def test_shadow_rows_never_enter_quadrant_counts(self) -> None:
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1_762_000_000_000,
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                }
            ]
        )
        rec = (await self.db.get_recent_recommendations(limit=1))[0]
        await self.db.settle_recommendation(
            int(rec["id"]),
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-20 22:00",
            raw_return_pct=5.0,
            alpha_pct=2.0,
            process_quadrant="GOOD_WIN",
        )
        await self.db.log_shadow_rejections(
            [
                {
                    "account": "Invest",
                    "symbol": "TSLA",
                    "action": "BUY",
                    "entry_price": 100.0,
                    "horizon_days": 20,
                    "benchmark_symbol": "SPY",
                    "shadow_trigger": "CRITIC",
                    "shadow_reason": "test",
                }
            ]
        )
        shadow = (await self.db.get_pending_shadow_settlements(limit=1))[0]
        await self.db.settle_shadow_rejection(
            int(shadow["id"]),
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-20 22:00",
            raw_return_pct=12.0,
            t20_return_pct=12.0,
            t20_alpha_pct=10.0,
        )
        stats = await self.db.get_recommendation_stats()
        self.assertEqual(stats["quadrant_counts"]["GOOD_WIN"], 1)
        self.assertEqual(sum(stats["quadrant_counts"].values()), 1)
        self.assertEqual(stats["shadow_count"], 1)

    async def test_snapshot_for_cycle_date_returns_earliest(self) -> None:
        early = _portfolio(fetched_at_uk=f"{_UK_DATE} 07:30")
        late = _portfolio(fetched_at_uk=f"{_UK_DATE} 21:30")
        async with self.db._connect(write=True) as conn:
            await conn.execute(
                "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_UK_DATE, f"{_UK_DATE} 07:30", 1_765_100_000_000, 20_000.0, early.model_dump_json()),
            )
            await conn.execute(
                "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_UK_DATE, f"{_UK_DATE} 21:30", 1_765_200_000_000, 20_000.0, late.model_dump_json()),
            )
            await conn.commit()
        snap = await self.db.get_snapshot_for_cycle_date(_UK_DATE)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.fetched_at_uk, f"{_UK_DATE} 07:30")


class TestStep2Ledger(unittest.IsolatedAsyncioTestCase):
    def test_local_vs_fx_split_for_unchanged_quantity(self) -> None:
        morning = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=100.0, pnl=100.0, fx=20.0)],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=110.0, pnl=160.0, fx=35.0)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(morning, eod, uk_date=_UK_DATE)
        self.assertAlmostEqual(ledger["totals"]["local_pnl_gbp"], 45.0, places=2)
        self.assertAlmostEqual(ledger["totals"]["fx_pnl_gbp"], 15.0, places=2)
        self.assertAlmostEqual(ledger["totals"]["total_mark_pnl_gbp"], 60.0, places=2)
        self.assertAlmostEqual(ledger["positions"][0]["local_ret_pct"], 10.0, places=2)
        self.assertFalse(ledger["positions"][0]["intraday_traded"])

    def test_intraday_trade_never_fabricates_unrealized_delta(self) -> None:
        morning = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=100.0, pnl=100.0, fx=20.0)],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=15, cur=110.0, pnl=300.0, fx=40.0)],
            isa_orders=[_order("ISA", "NVDA", qty=5, price=108.0)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(morning, eod, uk_date=_UK_DATE)
        self.assertEqual(ledger["totals"]["local_pnl_gbp"], 0.0)
        self.assertEqual(ledger["totals"]["fx_pnl_gbp"], 0.0)
        row = ledger["positions"][0]
        self.assertTrue(row["intraday_traded"])
        self.assertEqual(len(ledger["intraday_orders"]), 1)
        self.assertEqual(ledger["top_winners"], [])
        self.assertEqual(ledger["top_losers"], [])

    def test_gbp_100x_currency_mismatch_guard(self) -> None:
        morning = _portfolio(
            isa_positions=[_pos("ISA", "AZN.L", qty=10, cur=1000.0, pnl=0.0, currency="GBp")],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod = _portfolio(
            isa_positions=[_pos("ISA", "AZN.L", qty=10, cur=11.0, pnl=50.0, currency="GBP")],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(morning, eod, uk_date=_UK_DATE)
        row = ledger["positions"][0]
        self.assertTrue(row["currency_mismatch"])
        self.assertIsNone(row["local_ret_pct"])

        # Case-sensitive GBp vs GBP must flag mismatch even without a 100x price ratio.
        am_pence = _portfolio(
            isa_positions=[_pos("ISA", "AZN.L", qty=10, cur=100.0, pnl=0.0, currency="GBp")],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod_pounds = _portfolio(
            isa_positions=[_pos("ISA", "AZN.L", qty=10, cur=102.0, pnl=20.0, currency="GBP")],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger_case = build_eod_attribution_ledger(am_pence, eod_pounds, uk_date=_UK_DATE)
        self.assertTrue(ledger_case["positions"][0]["currency_mismatch"])
        self.assertIsNone(ledger_case["positions"][0]["local_ret_pct"])

    def test_same_day_roundtrip_unchanged_qty_marked_intraday_traded(self) -> None:
        morning = _portfolio(
            invest_positions=[_pos("Invest", "NVDA", qty=10, cur=100.0, pnl=0.0)],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod = _portfolio(
            invest_positions=[_pos("Invest", "NVDA", qty=10, cur=105.0, pnl=50.0)],
            invest_orders=[
                _order("Invest", "NVDA", side="SELL", price=102.0),
                _order("Invest", "NVDA", side="BUY", price=101.0),
            ],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(morning, eod, uk_date=_UK_DATE)
        self.assertTrue(ledger["positions"][0]["intraday_traded"])
        self.assertEqual(ledger["totals"]["local_pnl_gbp"], 0.0)
        self.assertEqual(len(ledger["intraday_orders"]), 2)

    def test_breakeven_stop_scan_and_core_beta_exemption(self) -> None:
        pos = _pos("Invest", "NVDA", qty=10, avg=100.0, cur=120.0)
        eod = _portfolio(invest_positions=[pos], fetched_at_uk=f"{_UK_DATE} 21:30")
        ledger = build_eod_attribution_ledger(eod, eod, uk_date=_UK_DATE)
        codes = {a["code"] for a in ledger["discipline_alerts"]}
        self.assertIn("MISSING_BREAKEVEN_STOP", codes)

        # Bare cost (100.0) without round-trip friction still triggers MISSING_BREAKEVEN_STOP
        # to stay 1-to-1 with ledgers.py / models_and_friction.py financed unlock rule.
        bare_cost_stop = _portfolio(
            invest_positions=[pos],
            invest_pending=[_pending_stop("NVDA", qty=10, stop=100.0)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            bare_cost_stop, bare_cost_stop, strategy=StrategySection(), uk_date=_UK_DATE
        )
        self.assertIn("MISSING_BREAKEVEN_STOP", {a["code"] for a in ledger["discipline_alerts"]})

        # Stop covering cost * (1 + round-trip friction) silences the advisory.
        full_stop = _portfolio(
            invest_positions=[pos],
            invest_pending=[_pending_stop("NVDA", qty=10, stop=100.5)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            full_stop, full_stop, strategy=StrategySection(), uk_date=_UK_DATE
        )
        self.assertNotIn("MISSING_BREAKEVEN_STOP", {a["code"] for a in ledger["discipline_alerts"]})

        half_stop = _portfolio(
            invest_positions=[pos],
            invest_pending=[_pending_stop("NVDA", qty=5, stop=100.5)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            half_stop, half_stop, strategy=StrategySection(), uk_date=_UK_DATE
        )
        self.assertIn("MISSING_BREAKEVEN_STOP", {a["code"] for a in ledger["discipline_alerts"]})

        core = _pos("Invest", "VUSA.L", qty=10, avg=100.0, cur=120.0, currency="GBP")
        core_port = _portfolio(invest_positions=[core], fetched_at_uk=f"{_UK_DATE} 21:30")
        ledger = build_eod_attribution_ledger(core_port, core_port, uk_date=_UK_DATE)
        self.assertNotIn("MISSING_BREAKEVEN_STOP", {a["code"] for a in ledger["discipline_alerts"]})

    def test_high_slippage_unplanned_trade_and_unprotected_loss(self) -> None:
        recs = [
            {
                "account": "ISA",
                "symbol": "NVDA",
                "action": "BUY",
                "entry_price": 100.0,
                "created_at_uk": "2026-11-20 09:00",
            }
        ]
        eod = _portfolio(
            invest_positions=[_pos("Invest", "AMD", qty=10, avg=100.0, cur=80.0)],
            isa_orders=[_order("ISA", "NVDA", qty=10, price=105.0)],
            invest_orders=[
                _order("Invest", "TSLA", qty=5, price=200.0),
                _order("Invest", "VUSA.L", qty=5, price=80.0, currency="GBP"),
            ],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            eod,
            eod,
            today_recs=recs,
            strategy=_cfg(),
            uk_date=_UK_DATE,
        )
        codes = {(a["code"], a["symbol"]) for a in ledger["discipline_alerts"]}
        self.assertIn(("HIGH_SLIPPAGE", "NVDA"), codes)
        self.assertIn(("UNPLANNED_TRADE", "TSLA"), codes)
        self.assertIn(("UNPROTECTED_LOSS", "AMD"), codes)
        self.assertNotIn(("UNPLANNED_TRADE", "VUSA.L"), codes)
        self.assertEqual(ledger["plan_vs_actual"]["high_slippage"], 1)
        self.assertEqual(ledger["plan_vs_actual"]["unplanned"], 1)

    def test_anomaly_gate_requires_move_atr_and_benchmark_excess(self) -> None:
        hist = _flat_then_move(-6.0)
        spy = _flat_then_move(0.5)
        morning = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=100.0, pnl=0.0)],
            fetched_at_uk=f"{_UK_DATE} 07:30",
        )
        eod = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=94.0, pnl=-60.0)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            morning, eod, histories={"NVDA": hist, "SPY": spy}, strategy=_cfg(), uk_date=_UK_DATE
        )
        self.assertEqual(len(ledger["anomaly_candidates"]), 1)
        self.assertEqual(ledger["anomaly_candidates"][0]["symbol"], "NVDA")

        quiet = build_eod_attribution_ledger(
            morning,
            _portfolio(
                isa_positions=[_pos("ISA", "NVDA", qty=10, cur=101.0, pnl=10.0)],
                fetched_at_uk=f"{_UK_DATE} 21:30",
            ),
            histories={"NVDA": hist, "SPY": spy},
            strategy=_cfg(),
            uk_date=_UK_DATE,
        )
        self.assertEqual(quiet["anomaly_candidates"], [])

        # Benchmark excess below 2.5% must fail the hard gate.
        spy_down = _flat_then_move(-5.5)
        excess_fail = build_eod_attribution_ledger(
            morning, eod, histories={"NVDA": hist, "SPY": spy_down}, strategy=_cfg(), uk_date=_UK_DATE
        )
        self.assertEqual(excess_fail["anomaly_candidates"], [])

        # 2*ATR above the absolute move must fail even when the move is > 3.5%.
        atr_hist = _ohlcv([100.0 + (3.0 if i % 2 else 0.0) for i in range(30)])
        atr_fail = build_eod_attribution_ledger(
            morning, eod, histories={"NVDA": atr_hist, "SPY": spy}, strategy=_cfg(), uk_date=_UK_DATE
        )
        self.assertEqual(atr_fail["anomaly_candidates"], [])

    def test_intraday_buy_drawdown_is_anomaly_candidate(self) -> None:
        hist = _flat_then_move(-4.0)
        spy = _flat_then_move(0.5)
        morning = _portfolio(fetched_at_uk=f"{_UK_DATE} 07:30")
        eod = _portfolio(
            invest_positions=[_pos("Invest", "TSLA", qty=10, cur=96.0)],
            invest_orders=[_order("Invest", "TSLA", qty=10, price=100.0)],
            fetched_at_uk=f"{_UK_DATE} 21:30",
        )
        ledger = build_eod_attribution_ledger(
            morning, eod, histories={"TSLA": hist, "SPY": spy}, strategy=_cfg(), uk_date=_UK_DATE
        )
        self.assertEqual(len(ledger["anomaly_candidates"]), 1)
        self.assertEqual(ledger["anomaly_candidates"][0]["reason"], "INTRADAY_BUY_DRAWDOWN")

    def test_formatting_never_emits_hash_or_bold_and_respects_caps(self) -> None:
        ledger = build_eod_attribution_ledger(
            _portfolio(isa_positions=[_pos("ISA", "NVDA", cur=110.0, pnl=100.0)]),
            _portfolio(isa_positions=[_pos("ISA", "NVDA", cur=110.0, pnl=150.0)]),
            uk_date=_UK_DATE,
        )
        ledger["discipline_alerts"] = [
            {"code": "HIGH_SLIPPAGE", "symbol": "NVDA", "message": "**追高**"}
        ]
        ledger["discipline_alert_count"] = 1
        text = format_postmarket_review_text(ledger, max_chars=120)
        alert = format_postmarket_alert_text(ledger, max_chars=80)
        self.assertLessEqual(len(text), 120)
        self.assertLessEqual(len(alert), 80)
        self.assertNotIn("#", text + alert)
        self.assertNotIn("**", text + alert)
        self.assertTrue(is_substantive_catalyst("英伟达上调全年营收指引。"))
        self.assertFalse(is_substantive_catalyst("无明确基本面公告，属市场/板块波动"))
        quiet_ledger = {"uk_date": _UK_DATE, "discipline_alerts": [], "discipline_alert_count": 0}
        quiet = format_postmarket_alert_text(
            quiet_ledger, event_cards=[{"symbol": "X", "text": "无明确基本面公告"}]
        )
        self.assertEqual(quiet, "")
        card_alert = format_postmarket_alert_text(
            quiet_ledger, event_cards=[{"symbol": "NVDA", "text": "财报超预期，上调指引"}], max_chars=250
        )
        self.assertIn("财报超预期", card_alert)
        self.assertLessEqual(len(card_alert), 250)


class TestStep2Classification(unittest.IsolatedAsyncioTestCase):
    def test_four_quadrants_and_missing_return(self) -> None:
        self.assertEqual(
            classify_process_quadrant(
                action="BUY",
                raw_return_pct=5.0,
                alpha_pct=1.0,
                fill_status="FILLED",
                actual_filled=1,
                execution_slippage_pct=1.0,
            ),
            "GOOD_WIN",
        )
        self.assertEqual(
            classify_process_quadrant(
                action="BUY",
                raw_return_pct=5.0,
                alpha_pct=1.0,
                fill_status="FILLED",
                actual_filled=1,
                execution_slippage_pct=2.0,
            ),
            "BAD_WIN",
        )
        self.assertEqual(
            classify_process_quadrant(
                action="BUY",
                raw_return_pct=-5.0,
                alpha_pct=-1.0,
                fill_status="FILLED",
                actual_filled=1,
                execution_slippage_pct=0.5,
            ),
            "GOOD_LOSS",
        )
        self.assertEqual(
            classify_process_quadrant(
                action="BUY",
                raw_return_pct=-5.0,
                alpha_pct=-1.0,
                fill_status="FILLED",
                actual_filled=1,
                execution_slippage_pct=2.0,
            ),
            "BAD_LOSS",
        )
        self.assertEqual(
            classify_process_quadrant(
                action="BUY",
                raw_return_pct=None,
                alpha_pct=None,
                fill_status="",
                actual_filled=None,
                execution_slippage_pct=None,
            ),
            "",
        )


class TestStep3Settlement(_DbCase):
    async def test_reflection_prompt_injects_process_context(self) -> None:
        llm = _StubLLM("复盘：方向正确但追高滑点偏大，下次限价分批。")
        sched = TradingScheduler(_cfg(), _StubT212(_portfolio()), _StubMarket(), llm, self.db, _StubTg())
        lesson = await sched._generate_settlement_reflection(
            rec={"symbol": "NVDA", "account": "Invest", "action": "BUY", "thesis": "AI 景气"},
            rec_date=datetime(2026, 11, 10).date(),
            exit_date=datetime(2026, 11, 20).date(),
            holding_days=5,
            entry_price=100.0,
            exit_price=105.0,
            raw_return_pct=5.0,
            benchmark_symbol="SPY",
            bench_return_pct=1.0,
            alpha_pct=4.0,
            mfe_pct=6.1,
            mae_pct=-1.2,
            actual_filled=1,
            actual_fill_price=102.0,
            execution_slippage_pct=2.0,
            regime="RISK_ON",
            process_quadrant="BAD_WIN",
            holding_catalyst="财报超预期，上调指引",
        )
        self.assertTrue(lesson)
        call = llm.calls[0]
        self.assertIn("市场状态: RISK_ON", call["prompt"])
        self.assertIn("持有期 MFE / MAE", call["prompt"])
        self.assertIn("实盘成交与滑点", call["prompt"])
        self.assertIn("2×2 过程象限: BAD_WIN", call["prompt"])
        self.assertIn("持有期客观催化: 财报超预期", call["prompt"])
        self.assertIn("GOOD_LOSS", call["system_instruction"])
        self.assertIn("BAD_WIN", call["system_instruction"])

    async def test_flash_lite_timeout_and_cache(self) -> None:
        llm = _StubLLM("英伟达财报超预期，上调全年指引。")
        sched = TradingScheduler(_cfg(), _StubT212(_portfolio()), _StubMarket(), llm, self.db, _StubTg())
        text = await sched._fetch_catalyst_with_flash_lite(
            "NVDA", window_label="T+5 settlement", reason_hint="raw +9%", db=self.db
        )
        self.assertIn("财报超预期", text)
        call = llm.calls[0]
        self.assertEqual(call["model"], _cfg().models.search_model)
        self.assertTrue(call["use_google_search"])
        self.assertEqual(call["thinking_level"], "LOW")
        self.assertEqual(call["round_timeout_seconds"], 30.0)
        again = await sched._fetch_catalyst_with_flash_lite(
            "NVDA", window_label="T+5 settlement", reason_hint="raw +9%", db=self.db
        )
        self.assertEqual(again, text)
        self.assertEqual(len(llm.calls), 1)

        boom = _StubLLM()
        boom.raise_exc = TimeoutError("hard timeout")
        sched_boom = TradingScheduler(_cfg(), _StubT212(_portfolio()), _StubMarket(), boom, self.db, _StubTg())
        self.assertEqual(
            await sched_boom._fetch_catalyst_with_flash_lite(
                "AMD", window_label="T+5", reason_hint="x", db=self.db
            ),
            "",
        )

    async def test_reflection_failure_still_settles_numerically(self) -> None:
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-02 15:00",
                    "created_at_ms": int(datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc).timestamp() * 1000),
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                    "thesis": "AI 景气",
                    "horizon_days": 5,
                    "benchmark_symbol": "SPY",
                }
            ]
        )
        closes = [100.0] * 30
        hist = _ohlcv(closes)
        spy = _ohlcv([500.0] * 30)
        llm = _StubLLM()
        llm.raise_exc = RuntimeError("Gemini transient")
        cfg = _cfg(postmarket_review_enabled=False, postmarket_event_search_enabled=False)
        sched = TradingScheduler(
            cfg, _StubT212(_portfolio()), _StubMarket({"NVDA": hist, "SPY": spy}), llm, self.db, _StubTg()
        )
        summary = await sched._run_daily_settlement_impl(now_utc=_NOW)
        self.assertEqual(summary["settled"], 1)
        row = (await self.db.get_recent_recommendations(limit=1))[0]
        self.assertEqual(row["outcome_status"], "SETTLED")
        self.assertEqual(row["reflection_lesson"], "")
        self.assertIn(row["process_quadrant"], {"GOOD_WIN", "GOOD_LOSS", "BAD_WIN", "BAD_LOSS"})

    async def test_postmarket_review_budget_cap_and_disabled_switch(self) -> None:
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        morning_positions = [
            _pos("ISA", sym, qty=10, cur=100.0, pnl=0.0) for sym in symbols
        ]
        eod_positions = [
            _pos("ISA", sym, qty=10, cur=94.0, pnl=-60.0) for sym in symbols
        ]
        morning = _portfolio(
            isa_positions=morning_positions,
            fetched_at_uk=f"{_UK_DATE} 07:30",
            fetched_at_ms=1_765_100_000_000,
        )
        eod = _portfolio(
            isa_positions=eod_positions,
            fetched_at_uk=f"{_UK_DATE} 21:30",
            fetched_at_ms=1_765_200_000_000,
        )
        async with self.db._connect(write=True) as conn:
            await conn.execute(
                "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_UK_DATE, f"{_UK_DATE} 07:30", 1_765_100_000_000, 20_000.0, morning.model_dump_json()),
            )
            await conn.commit()
        histories = {sym: _flat_then_move(-6.0) for sym in symbols}
        histories["SPY"] = _flat_then_move(0.5)
        llm = _StubLLM("公司发布重大合同公告。")
        sched = TradingScheduler(_cfg(), _StubT212(eod), _StubMarket(histories), llm, self.db, _StubTg())
        alert = await sched._run_postmarket_review(
            now_utc=_NOW, uk_date=_UK_DATE, search_budget=sched._postmarket_search_budget()
        )
        catalyst_calls = [c for c in llm.calls if c.get("usage_label") == "postmarket_catalyst"]
        self.assertEqual(len(catalyst_calls), 3)
        self.assertTrue(alert)
        latest = await self.db.get_latest_postmarket_review()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["uk_date"], _UK_DATE)

        disabled = TradingScheduler(
            _cfg(postmarket_event_search_enabled=False),
            _StubT212(eod),
            _StubMarket(histories),
            llm,
            self.db,
            _StubTg(),
        )
        self.assertEqual(disabled._postmarket_search_budget(), 0)
        before = len(llm.calls)
        await disabled._run_postmarket_review(now_utc=_NOW, uk_date=_UK_DATE, search_budget=0)
        self.assertEqual(len(llm.calls), before)

    async def test_fallback_portfolio_skips_postmarket_write(self) -> None:
        fallback = _portfolio(is_fallback=True, fetched_at_uk=f"{_UK_DATE} 21:30")
        llm = _StubLLM()
        sched = TradingScheduler(_cfg(), _StubT212(fallback), _StubMarket({}), llm, self.db, _StubTg())
        alert = await sched._run_postmarket_review(now_utc=_NOW, uk_date=_UK_DATE, search_budget=3)
        self.assertEqual(alert, "")
        self.assertIsNone(await self.db.get_latest_postmarket_review())
        self.assertEqual(llm.calls, [])

    async def test_disabled_postmarket_review_switch_skips_everything(self) -> None:
        llm = _StubLLM()
        sched = TradingScheduler(
            _cfg(postmarket_review_enabled=False),
            _StubT212(_portfolio()),
            _StubMarket({}),
            llm,
            self.db,
            _StubTg(),
        )
        alert = await sched._run_postmarket_review(now_utc=_NOW, uk_date=_UK_DATE, search_budget=3)
        self.assertEqual(alert, "")
        self.assertIsNone(await self.db.get_latest_postmarket_review())
        self.assertEqual(llm.calls, [])

    async def test_weekend_postmarket_review_is_skipped_and_never_replays_friday(self) -> None:
        friday_morning = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=100.0, pnl=0.0)],
            fetched_at_uk="2026-11-20 07:30",
            fetched_at_ms=1_765_100_000_000,
        )
        friday_close = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=94.0, pnl=-60.0)],
            fetched_at_uk="2026-11-21 22:10",
            fetched_at_ms=1_765_200_000_000,
        )
        async with self.db._connect(write=True) as conn:
            await conn.execute(
                "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                ("2026-11-20", "2026-11-20 07:30", 1_765_100_000_000, 20_000.0, friday_morning.model_dump_json()),
            )
            await conn.commit()
        sat_utc = datetime(2026, 11, 21, 22, 10, tzinfo=timezone.utc)
        llm = _StubLLM("公司发布重大合同公告。")
        sched = TradingScheduler(
            _cfg(),
            _StubT212(friday_close),
            _StubMarket({"NVDA": _flat_then_move(-6.0), "SPY": _flat_then_move(0.5)}),
            llm,
            self.db,
            _StubTg(),
        )
        alert = await sched._run_postmarket_review(
            now_utc=sat_utc, uk_date="2026-11-21", search_budget=3
        )
        self.assertEqual(alert, "")
        self.assertIsNone(await self.db.get_latest_postmarket_review())
        self.assertEqual(llm.calls, [])

    async def test_same_timestamp_fallback_refuses_to_self_diff(self) -> None:
        eod_ms = int(_NOW.timestamp() * 1000)
        eod = _portfolio(
            isa_positions=[_pos("ISA", "NVDA", qty=10, cur=105.0, pnl=50.0)],
            fetched_at_uk=f"{_UK_DATE} 22:30",
        )
        eod.fetched_at_ms = eod_ms
        # Only the EOD snapshot exists in DB — get_previous_snapshot falls back to it.
        async with self.db._connect(write=True) as conn:
            await conn.execute(
                "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_UK_DATE, f"{_UK_DATE} 22:30", eod_ms, 20_000.0, eod.model_dump_json()),
            )
            await conn.commit()
        llm = _StubLLM()
        sched = TradingScheduler(_cfg(), _StubT212(eod), _StubMarket({}), llm, self.db, _StubTg())
        alert = await sched._run_postmarket_review(now_utc=_NOW, uk_date=_UK_DATE, search_budget=3)
        self.assertEqual(alert, "")
        self.assertIsNone(await self.db.get_latest_postmarket_review())

    async def test_catalyst_cache_scopes_settlement_and_eod_separately(self) -> None:
        llm = _StubLLM("测试催化公告。")
        sched = TradingScheduler(_cfg(), _StubT212(_portfolio()), _StubMarket(), llm, self.db, _StubTg())
        await sched._fetch_catalyst_with_flash_lite(
            "NVDA",
            window_label="T+20 follow-up (2026-10-20 to 2026-11-18)",
            reason_hint="T+20 raw +12%",
            db=self.db,
            cache_scope="settle_t20_2026-10-20",
            uk_date=_UK_DATE,
        )
        await sched._fetch_catalyst_with_flash_lite(
            "NVDA",
            window_label=f"post-close anomaly ({_UK_DATE})",
            reason_hint="daily -5%",
            db=self.db,
            cache_scope="eod_anomaly",
            uk_date=_UK_DATE,
        )
        # Distinct scopes must not collide in grounding_news_cache.
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("2026-10-20 to 2026-11-18", llm.calls[0]["prompt"])
        self.assertIn(_UK_DATE, llm.calls[1]["prompt"])


class TestStep4Downstream(_DbCase):
    async def test_weekly_diagnosis_prompt_contains_new_observability(self) -> None:
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1_762_000_000_000,
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                }
            ]
        )
        rec = (await self.db.get_recent_recommendations(limit=1))[0]
        await self.db.settle_recommendation(
            int(rec["id"]),
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-20 22:00",
            raw_return_pct=5.0,
            alpha_pct=2.0,
            process_quadrant="GOOD_WIN",
        )
        await self.db.upsert_daily_postmarket_review(
            {
                "uk_date": _UK_DATE,
                "created_at_uk": f"{_UK_DATE} 22:00",
                "created_at_ms": 1_765_000_000_000,
                "regime": "RISK_ON",
                "local_pnl_gbp": 12.0,
                "fx_pnl_gbp": 3.0,
                "total_mark_pnl_gbp": 15.0,
                "discipline_alert_count": 2,
                "summary_text": "账单",
                "payload": {"event_cards": [{"symbol": "NVDA", "text": "财报超预期"}]},
            }
        )
        llm = _StubLLM('{"diagnosis": "参数建议：维持当前风控与仓位节奏，重点关注执行滑点与追高纪律。"}')
        cfg = _cfg()
        sched = TradingScheduler(cfg, _StubT212(_portfolio()), _StubMarket(), llm, self.db, _StubTg())
        diagnosis = await sched._generate_weekly_diagnosis(
            {
                "universe_count": 150,
                "overlap_candidates": [],
                "zombie_candidates": [],
                "threshold_backtest": {"sample_count": 40},
                "shadow_stats": {
                    "settled": 12,
                    "avoided_loser": 7,
                    "missed_winner": 2,
                    "avoided_loser_rate_pct": 58.3,
                    "missed_winner_rate_pct": 16.7,
                    "small_sample": True,
                },
                "exit_rule_ab": {
                    "sample_count": 40,
                    "baseline": {"win_rate_pct": 52.0, "avg_return_pct": 2.1, "max_single_loss_pct": -9.0},
                    "v559_ladder": {"win_rate_pct": 55.0, "avg_return_pct": 2.6, "max_single_loss_pct": -7.0},
                },
            },
            uk_date="2026-11-21",
        )
        self.assertIn("维持当前风控", diagnosis)
        prompt = llm.calls[0]["prompt"]
        for marker in (
            "影子护栏反事实统计",
            "退出规则离线 A/B 对照",
            "实盘执行采纳与 2x2 过程分布",
            "近 5 日盘后账单汇总",
            "GOOD_WIN 1",
        ):
            self.assertIn(marker, prompt)

    async def test_report_and_pick_lessons_carry_regime_and_mfe_mae(self) -> None:
        row = {
            "created_at_uk": "2026-11-10 15:00",
            "account": "Invest",
            "symbol": "NVDA",
            "action": "BUY",
            "raw_return_pct": 5.0,
            "alpha_pct": 3.2,
            "mfe_pct": 6.1,
            "mae_pct": -1.2,
            "regime": "RISK_ON",
            "process_quadrant": "GOOD_WIN",
            "reflection_lesson": "顺势突破，控制追高。",
        }
        compact = _compact_lesson_record(row)
        self.assertEqual(compact["regime"], "RISK_ON")
        self.assertEqual(compact["process_quadrant"], "GOOD_WIN")
        rendered = _format_lesson_entry("NVDA", compact)
        self.assertIn("RISK_ON", rendered)
        self.assertIn("MFE+6.1%/MAE-1.2%", rendered)
        self.assertIn("GOOD_WIN", rendered)
        pick_text = _format_lessons_section([compact], {}, 5)
        self.assertIn("RISK_ON", pick_text)
        self.assertIn("MFE+6.1%/MAE-1.2%", pick_text)
        self.assertIn("GOOD_WIN", pick_text)

    async def test_review_command_is_zero_token_and_plain_text(self) -> None:
        await self.db.upsert_daily_postmarket_review(
            {
                "uk_date": _UK_DATE,
                "created_at_uk": f"{_UK_DATE} 22:00",
                "created_at_ms": 1_765_000_000_000,
                "regime": "RISK_ON",
                "local_pnl_gbp": 12.0,
                "fx_pnl_gbp": 3.0,
                "total_mark_pnl_gbp": 15.0,
                "discipline_alert_count": 0,
                "summary_text": "盘后执行账单：股价 £+12.00 | 汇率 £+3.00",
                "payload": {"event_cards": []},
            }
        )
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1_762_000_000_000,
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                }
            ]
        )
        rec = (await self.db.get_recent_recommendations(limit=1))[0]
        await self.db.settle_recommendation(
            int(rec["id"]),
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-20 22:00",
            raw_return_pct=5.0,
            alpha_pct=3.0,
            reflection_lesson="复盘：守住纪律。",
            mfe_pct=6.1,
            mae_pct=-1.2,
            process_quadrant="GOOD_WIN",
        )
        llm = _StubLLM()
        bot = AdvisorBot(_cfg(), _StubT212(_portfolio()), _StubMarket(), llm, self.db, _StubTg())
        msg = await bot._build_review_message()
        self.assertIn(_UK_DATE, msg)
        self.assertIn("GOOD_WIN", msg)
        self.assertNotIn("#", msg)
        self.assertNotIn("**", msg)
        stats_msg = await bot._build_stats_message()
        self.assertIn("过程 2x2", stats_msg)
        self.assertIn("GOOD_WIN 1", stats_msg)
        self.assertFalse(bot._is_heavy_ai_command("/review"))
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
