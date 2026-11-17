"""Offline tests for the event-driven intraday sentinel (0-token scan + DeepSeek push gate)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.bot.scheduler import TradingScheduler, _intraday_move_pct, _order_distance_pct
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.market_data import MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
)
from src.storage.db import AdvisorDatabase

NOW = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)  # 15:00 UK: LSE open + US session OPEN
UK_DATE = "2026-11-17"


class _DsStubLLM:
    def __init__(self, reply: str | None = None) -> None:
        self.deepseek_api_key = "test-key"
        self.reply = reply or '{"should_push": false, "alert_text": ""}'
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.reply


class _NoKeyLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise AssertionError("sentinel must not call the LLM without a DeepSeek key")


class _StubTg(TelegramClient):
    def __init__(self, fail: bool = False) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    async def send_message(self, chat_id: str | int, text: str) -> bool:
        if self.fail:
            return False
        self.sent.append((str(chat_id), text))
        return True


class _StubT212(T212Client):
    def __init__(self, portfolio: PortfolioData) -> None:
        super().__init__()
        self._portfolio = portfolio

    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return self._portfolio


class _StubMarketData(MarketDataService):
    def __init__(self, histories: dict, bundles: dict, quotes: dict | None = None) -> None:
        super().__init__()
        self._histories = histories
        self._bundles = bundles
        self._quotes = quotes or {}
        self.calls: list[list[str]] = []
        self.quote_calls: list[str] = []

    async def get_histories(self, symbols: list[str], period: str = "1y") -> dict:
        return {s: self._histories[s] for s in symbols if s in self._histories}

    async def get_us_realtime_quote(self, symbol: str) -> dict | None:
        self.quote_calls.append(symbol)
        return self._quotes.get(symbol)

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
        isin_map: dict[str, str] | None = None,
    ):
        self.calls.append(list(symbols))
        hists = {s: self._histories[s] for s in symbols if s in self._histories}
        bundles = {s: self._bundles[s] for s in symbols if s in self._bundles}
        return hists, bundles


def _history(prev_close: float, last_close: float) -> pd.DataFrame:
    dates = pd.date_range(end="2026-11-17", periods=5, freq="B")
    closes = [prev_close] * 4 + [last_close]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000.0] * 5,
        },
        index=dates,
    )


def _position(symbol: str, currency: str, cur_price: float) -> PositionItem:
    return PositionItem(
        account="ISA",
        raw_ticker=f"{symbol}_EQ",
        ticker=symbol,
        yahoo_symbol=symbol,
        name=f"{symbol} Inc",
        inst_currency=currency,
        quantity=10.0,
        avg_price=cur_price * 0.9,
        cur_price=cur_price,
        current_value=cur_price * 10.0,
        total_cost=cur_price * 9.0,
        pnl=cur_price,
        fx_impact=0.0,
        return_pct=10.0,
    )


def _order(symbol: str, limit_price: float, currency: str) -> PendingOrderItem:
    return PendingOrderItem(
        id=f"ord-{symbol}",
        account="ISA",
        raw_ticker=f"{symbol}_EQ",
        ticker=symbol,
        yahoo_symbol=symbol,
        name=f"{symbol} Inc",
        side="BUY",
        order_type="LIMIT",
        quantity=5.0,
        value=limit_price * 5.0,
        limit_price=limit_price,
        currency=currency,
    )


def _portfolio(positions: list[PositionItem], orders: list[PendingOrderItem] | None = None) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 15:00",
        fetched_at_ms=1794927600000,
        total_combined_value=10000.0,
        total_combined_cash=2000.0,
        isa=AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=10000.0,
            total_cash=2000.0,
            positions=positions,
            pending_orders=orders or [],
        ),
        invest=AccountSnapshot(ok=True, account_label="Invest"),
    )


def _cfg() -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, deepseek_api_key="", ops_notify_url="", telegram_chat_id="123"),
        universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
    )


class _SentinelTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = AdvisorDatabase(Path(self._tmp.name) / "advisor.db")
        await self.db.init()
        self.cfg = _cfg()

    def _build(
        self,
        *,
        positions: list[PositionItem],
        orders: list[PendingOrderItem] | None = None,
        bundles: dict | None = None,
        histories: dict | None = None,
        quotes: dict | None = None,
        llm=None,
    ) -> tuple[TradingScheduler, _StubMarketData, _StubTg, object]:
        md = _StubMarketData(histories or {}, bundles or {}, quotes)
        tg = _StubTg()
        llm = llm or _DsStubLLM()
        sched = TradingScheduler(self.cfg, _StubT212(_portfolio(positions, orders)), md, llm, self.db, tg)
        return sched, md, tg, llm


def _bundle(symbol: str, price: float, currency: str) -> TickerMarketBundle:
    return TickerMarketBundle(symbol=symbol, current_price=price, currency=currency, quote_type="EQUITY")


class TestIntradaySentinel(_SentinelTestCase):
    async def test_move_trigger_pushes_and_marks_dedupe(self) -> None:
        llm = _DsStubLLM('{"should_push": true, "alert_text": "NVDA 日内 +3.60%，放量跌破早报支撑，建议减半仓观察。"}')
        sched, md, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 103.6)],
            bundles={"NVDA": _bundle("NVDA", 103.6, "USD")},
            histories={"NVDA": _history(100.0, 103.6)},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertIn("intraday_sentinel", fired)
        self.assertEqual(md.calls, [["NVDA"]])
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("盘中异动 | NVDA", tg.sent[0][1])
        self.assertTrue(any(c["usage_label"] == "intraday_alert" for c in llm.calls))
        self.assertTrue(await self.db.has_scheduler_job_fired("intraday_alert:NVDA", UK_DATE))

    async def test_should_push_false_is_silent_and_keeps_dedupe_slot(self) -> None:
        llm = _DsStubLLM('{"should_push": false, "alert_text": ""}')
        sched, _, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 103.6)],
            bundles={"NVDA": _bundle("NVDA", 103.6, "USD")},
            histories={"NVDA": _history(100.0, 103.6)},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(fired, [])
        self.assertEqual(tg.sent, [])
        self.assertFalse(await self.db.has_scheduler_job_fired("intraday_alert:NVDA", UK_DATE))

    async def test_quiet_market_makes_zero_llm_calls(self) -> None:
        llm = _DsStubLLM('{"should_push": true, "alert_text": "不该出现"}')
        sched, md, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 100.0)],
            bundles={"NVDA": _bundle("NVDA", 100.0, "USD")},
            histories={"NVDA": _history(100.0, 100.0)},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(fired, [])
        self.assertEqual(md.calls, [["NVDA"]])
        self.assertEqual(llm.calls, [])
        self.assertEqual(tg.sent, [])

    async def test_move_threshold_boundaries(self) -> None:
        for price, should_push in ((103.4, False), (103.6, True)):
            llm = _DsStubLLM('{"should_push": true, "alert_text": "触及异动阈值，建议复核早报支撑。"}')
            sched, _, tg, _ = self._build(
                positions=[_position("NVDA", "USD", price)],
                bundles={"NVDA": _bundle("NVDA", price, "USD")},
                histories={"NVDA": _history(100.0, price)},
                llm=llm,
            )
            await sched.check_and_run_due_jobs(NOW)
            self.assertEqual(len(tg.sent), 1 if should_push else 0, msg=f"price={price}")

    async def test_order_distance_thresholds(self) -> None:
        for limit_price, should_push in ((101.4, True), (101.6, False)):
            llm = _DsStubLLM('{"should_push": true, "alert_text": "现价逼近限价挂单，建议确认是否撤改。"}')
            sched, _, tg, _ = self._build(
                positions=[],
                orders=[_order("VWRP.L", limit_price, "GBP")],
                bundles={"VWRP.L": _bundle("VWRP.L", 100.0, "GBP")},
                histories={"VWRP.L": _history(100.0, 100.0)},
                llm=llm,
            )
            await sched.check_and_run_due_jobs(NOW)
            self.assertEqual(len(tg.sent), 1 if should_push else 0, msg=f"limit={limit_price}")

    async def test_window_boundaries(self) -> None:
        early = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc)
        late = datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc)
        for moment in (early, late):
            sched, md, tg, _ = self._build(
                positions=[_position("NVDA", "USD", 110.0)],
                bundles={"NVDA": _bundle("NVDA", 110.0, "USD")},
                histories={"NVDA": _history(100.0, 110.0)},
            )
            await sched.check_and_run_due_jobs(moment)
            self.assertEqual(md.calls, [], msg=f"moment={moment}")
            self.assertEqual(tg.sent, [])

        saturday = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
        sched, md, _, _ = self._build(
            positions=[_position("NVDA", "USD", 110.0)],
            bundles={"NVDA": _bundle("NVDA", 110.0, "USD")},
            histories={"NVDA": _history(100.0, 110.0)},
        )
        fired = await sched.check_and_run_due_jobs(saturday)
        self.assertNotIn("intraday_sentinel", fired)
        self.assertEqual(md.calls, [])

    async def test_us_symbol_skipped_before_us_open(self) -> None:
        before_us_open = datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc)  # 09:30 UK, US PRE
        sched, md, _, _ = self._build(
            positions=[_position("NVDA", "USD", 110.0)],
            bundles={"NVDA": _bundle("NVDA", 110.0, "USD")},
            histories={"NVDA": _history(100.0, 110.0)},
        )
        await sched.check_and_run_due_jobs(before_us_open)
        self.assertEqual(md.calls, [])

    async def test_lock_held_skips_scan_without_queueing(self) -> None:
        sched, md, _, _ = self._build(
            positions=[_position("NVDA", "USD", 110.0)],
            bundles={"NVDA": _bundle("NVDA", 110.0, "USD")},
            histories={"NVDA": _history(100.0, 110.0)},
        )
        async with sched.execution_lock:
            fired = await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(fired, [])
        self.assertEqual(md.calls, [])

    async def test_throttle_and_daily_dedupe(self) -> None:
        llm = _DsStubLLM('{"should_push": true, "alert_text": "盘中放量下跌，建议复核早报止损位。"}')
        sched, md, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 96.0)],
            bundles={"NVDA": _bundle("NVDA", 96.0, "USD")},
            histories={"NVDA": _history(100.0, 96.0)},
            llm=llm,
        )
        await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(len(md.calls), 1)
        self.assertEqual(len(tg.sent), 1)

        # 5 minutes later: throttled, no second scan.
        await sched.check_and_run_due_jobs(datetime(2026, 11, 17, 15, 5, tzinfo=timezone.utc))
        self.assertEqual(len(md.calls), 1)
        self.assertEqual(len(tg.sent), 1)

        # 31 minutes later: scan runs again but the per-symbol daily dedupe suppresses the push.
        await sched.check_and_run_due_jobs(datetime(2026, 11, 17, 15, 31, tzinfo=timezone.utc))
        self.assertEqual(len(md.calls), 2)
        self.assertEqual(len(tg.sent), 1)

    async def test_missing_ds_key_never_pushes(self) -> None:
        no_key = _NoKeyLLM()
        sched, md, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 110.0)],
            bundles={"NVDA": _bundle("NVDA", 110.0, "USD")},
            histories={"NVDA": _history(100.0, 110.0)},
            llm=no_key,
        )
        await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(len(md.calls), 1)
        self.assertEqual(no_key.calls, [])
        self.assertEqual(tg.sent, [])


    async def test_realtime_quote_overlays_stale_daily_bar(self) -> None:
        """Finnhub live price + prev_close must drive the move even when the daily bar is yesterday's."""
        stale_hist = pd.DataFrame(
            {"Close": [98.0, 100.0]},
            index=pd.date_range("2026-11-13", periods=2, freq="B"),  # last bar = Mon 16 (yesterday)
        )
        llm = _DsStubLLM('{"should_push": true, "alert_text": "NVDA 实时 +4.00%，放量突破早报压力位。"}')
        sched, md, tg, _ = self._build(
            positions=[_position("NVDA", "USD", 100.0)],
            bundles={"NVDA": _bundle("NVDA", 100.0, "USD")},
            histories={"NVDA": stale_hist},
            quotes={"NVDA": {"symbol": "NVDA", "current_price": 104.0, "prev_close": 100.0}},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertIn("intraday_sentinel", fired)
        self.assertEqual(md.quote_calls, ["NVDA"])
        self.assertEqual(len(tg.sent), 1)
        prompt = next(c["prompt"] for c in llm.calls if c["usage_label"] == "intraday_alert")
        self.assertIn("104.00", prompt)
        self.assertIn("+4.00%", prompt)

    async def test_stop_price_counts_as_order_proximity(self) -> None:
        llm = _DsStubLLM('{"should_push": true, "alert_text": "现价逼近止损触发价，建议确认风控计划。"}')
        stop_order = _order("VWRP.L", 120.0, "GBP")  # limit far away
        stop_order.stop_price = 100.8  # stop within 0.8%
        sched, _, tg, _ = self._build(
            positions=[],
            orders=[stop_order],
            bundles={"VWRP.L": _bundle("VWRP.L", 100.0, "GBP")},
            histories={"VWRP.L": _history(100.0, 100.0)},
            llm=llm,
        )
        await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("盘中异动 | VWRP.L", tg.sent[0][1])

    async def test_failed_delivery_does_not_burn_dedupe_slot(self) -> None:
        llm = _DsStubLLM('{"should_push": true, "alert_text": "盘中放量下跌，建议复核早报止损位。"}')
        md = _StubMarketData(
            {"NVDA": _history(100.0, 96.0)},
            {"NVDA": _bundle("NVDA", 96.0, "USD")},
        )
        tg = _StubTg(fail=True)
        sched = TradingScheduler(
            self.cfg,
            _StubT212(_portfolio([_position("NVDA", "USD", 96.0)])),
            md,
            llm,
            self.db,
            tg,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(fired, [])
        self.assertFalse(await self.db.has_scheduler_job_fired("intraday_alert:NVDA", UK_DATE))

        # Next scan window retries instead of being silently suppressed for the whole day.
        tg.fail = False
        await sched.check_and_run_due_jobs(datetime(2026, 11, 17, 15, 31, tzinfo=timezone.utc))
        self.assertEqual(len(tg.sent), 1)
        self.assertTrue(await self.db.has_scheduler_job_fired("intraday_alert:NVDA", UK_DATE))


class TestSentinelMath(unittest.TestCase):
    def test_move_pct_uses_yesterday_bar_when_live_price_given(self) -> None:
        fresh = _history(100.0, 104.0)
        self.assertAlmostEqual(
            _intraday_move_pct(fresh, 104.0, today_uk=date(2026, 11, 17)), 4.0, places=6
        )
        stale = pd.DataFrame(
            {"Close": [98.0, 100.0]},
            index=pd.date_range("2026-11-13", periods=2, freq="B"),  # last bar = Mon 16 (yesterday)
        )
        self.assertAlmostEqual(
            _intraday_move_pct(stale, 103.0, today_uk=date(2026, 11, 17)), 3.0, places=6
        )
        # Explicit Finnhub prev_close always wins over the daily-bar baseline.
        self.assertAlmostEqual(
            _intraday_move_pct(stale, 104.0, today_uk=date(2026, 11, 17), prev_close=100.0),
            4.0,
            places=6,
        )
        self.assertIsNone(_intraday_move_pct(None, 105.0, today_uk=date(2026, 11, 17)))
        self.assertIsNone(_intraday_move_pct(fresh, None, today_uk=date(2026, 11, 17)))
        ancient = pd.DataFrame(
            {"Close": [100.0]}, index=pd.date_range("2026-10-01", periods=1, freq="B")
        )
        self.assertIsNone(_intraday_move_pct(ancient, 105.0, today_uk=date(2026, 11, 17)))

    def test_order_distance_pct_includes_stop_price(self) -> None:
        orders = [_order("NVDA", 120.0, "USD")]
        orders[0].stop_price = 99.0
        self.assertAlmostEqual(_order_distance_pct(100.0, orders), 1.0, places=6)
        far_limit = [_order("NVDA", 101.0, "USD"), _order("NVDA", 99.0, "USD")]
        self.assertAlmostEqual(_order_distance_pct(100.0, far_limit), 1.0, places=6)
        self.assertIsNone(_order_distance_pct(100.0, []))
        self.assertIsNone(_order_distance_pct(100.0, [_order("NVDA", 0.0, "USD")]))
        self.assertIsNone(_order_distance_pct(None, orders))


if __name__ == "__main__":
    unittest.main()


