"""Offline v5.5.3 tests: chat context width, dual-philosophy sentinel, split attribution,
fill-ledger adoption, observability rollups and ISA value-factor coverage."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.agents.chat_agent import answer_user_chat
from src.bot.handlers import AdvisorBot
from src.bot.scheduler import (
    TradingScheduler,
    _match_recommendation_fill,
    _pending_order_snapshot_row,
    format_fill_receipt,
)
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.analytics import compute_account_split_performance, build_nav_row
from src.data_providers.market_data import MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalOrderItem,
    HistoricalTransactionItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
)
from src.engine.screener import compute_fundamentals_coverage
from src.storage.db import AdvisorDatabase

NOW = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)  # 15:00 UK, LSE + US open
UK_DATE = "2026-11-17"


# --------------------------------------------------------------------------- stubs


class _DsStubLLM:
    def __init__(self, reply: str | None = None) -> None:
        self.deepseek_api_key = "test-key"
        self.reply = reply or '{"should_push": false, "alert_text": ""}'
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.reply


class _StubTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str | int, text: str) -> bool:
        self.sent.append((str(chat_id), text))
        return True


class _StubT212(T212Client):
    def __init__(self, portfolio: PortfolioData) -> None:
        super().__init__()
        self._portfolio = portfolio

    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return self._portfolio


class _StubMarketData(MarketDataService):
    def __init__(self, histories: dict | None = None, bundles: dict | None = None) -> None:
        super().__init__()
        self._histories = histories or {}
        self._bundles = bundles or {}
        self.calls: list[list[str]] = []

    async def get_histories(self, symbols: list[str], period: str = "1y") -> dict:
        return {s: self._histories[s] for s in symbols if s in self._histories}

    async def get_us_realtime_quote(self, symbol: str) -> dict | None:
        return None

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


def _portfolio(
    isa_positions: list[PositionItem] | None = None,
    invest_positions: list[PositionItem] | None = None,
    *,
    isa_orders: list[PendingOrderItem] | None = None,
    invest_orders: list[PendingOrderItem] | None = None,
    isa_recent: list[HistoricalOrderItem] | None = None,
    invest_recent: list[HistoricalOrderItem] | None = None,
    pending_supported: bool = False,
    history_supported: bool = False,
    isa_transactions: list[HistoricalTransactionItem] | None = None,
    invest_transactions: list[HistoricalTransactionItem] | None = None,
) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 15:00",
        fetched_at_ms=1794927600000,
        total_combined_value=10000.0,
        total_combined_cash=2000.0,
        isa=AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=6000.0,
            total_cash=1000.0,
            positions=isa_positions or [],
            pending_orders=isa_orders or [],
            recent_orders=isa_recent or [],
            recent_transactions=isa_transactions or [],
            pending_orders_supported=pending_supported,
            order_history_supported=history_supported,
        ),
        invest=AccountSnapshot(
            ok=True,
            account_label="Invest",
            total_account_value=4000.0,
            total_cash=1000.0,
            positions=invest_positions or [],
            pending_orders=invest_orders or [],
            recent_orders=invest_recent or [],
            recent_transactions=invest_transactions or [],
            pending_orders_supported=pending_supported,
            order_history_supported=history_supported,
        ),
    )


def _position(
    symbol: str,
    price: float,
    *,
    account: str = "ISA",
    avg_price: float | None = None,
    currency: str = "USD",
) -> PositionItem:
    avg = price if avg_price is None else avg_price
    return PositionItem(
        account=account,
        raw_ticker=f"{symbol}_EQ",
        ticker=symbol,
        yahoo_symbol=symbol,
        name=symbol,
        inst_currency=currency,
        quantity=10.0,
        avg_price=avg,
        cur_price=price,
        current_value=price * 10.0,
        total_cost=avg * 10.0,
        pnl=(price - avg) * 10.0,
        fx_impact=0.0,
        return_pct=((price / avg) - 1.0) * 100.0 if avg else 0.0,
    )


def _history(prev_close: float, last_close: float, periods: int = 5) -> pd.DataFrame:
    dates = pd.date_range(end="2026-11-17", periods=periods, freq="B")
    closes = [prev_close] * (periods - 1) + [last_close]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes],
            "Close": closes,
            "Volume": [1_000_000.0] * periods,
        },
        index=dates,
    )


def _flat_history(peak: float, last: float, periods: int = 252) -> pd.DataFrame:
    closes = [peak] * (periods - 1) + [last]
    dates = pd.date_range(end="2026-11-17", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.002 for c in closes],
            "Low": [c * 0.998 for c in closes],
            "Close": closes,
            "Volume": [1_000_000.0] * periods,
        },
        index=dates,
    )


def _bundle(symbol: str, price: float, currency: str = "USD", quote_type: str = "EQUITY") -> TickerMarketBundle:
    return TickerMarketBundle(symbol=symbol, current_price=price, currency=currency, quote_type=quote_type)


def _cfg() -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, deepseek_api_key="", ops_notify_url="", telegram_chat_id="123"),
        universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
    )


class _DbCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = AdvisorDatabase(Path(self._tmp.name) / "advisor.db")
        await self.db.init()
        self.cfg = _cfg()

    def _scheduler(
        self,
        portfolio: PortfolioData,
        *,
        histories: dict | None = None,
        bundles: dict | None = None,
        llm=None,
    ) -> tuple[TradingScheduler, _StubMarketData, _StubTg, _DsStubLLM]:
        md = _StubMarketData(histories, bundles)
        tg = _StubTg()
        llm = llm or _DsStubLLM()
        sched = TradingScheduler(self.cfg, _StubT212(portfolio), md, llm, self.db, tg)
        return sched, md, tg, llm


# --------------------------------------------------------------- Phase 1: chat width


class TestChatMorningBlockWidth(_DbCase):
    async def test_morning_block_keeps_tail_beyond_2050(self) -> None:
        line = "• " + ("盘中与基本面跟踪" * 9)  # ~90 chars
        body = "\n".join(line for _ in range(31))
        morning = body + "\n【Invest 区块尾部标记】护栏修正保留"
        self.assertGreater(len(morning), 2050)
        self.assertLessEqual(len(morning), 2800)
        await self.db.save_morning_report(morning, "维持计划")

        llm = _DsStubLLM("OK")
        md = _StubMarketData()
        await answer_user_chat("今天怎么操作", self.cfg, _StubT212(_portfolio()), md, llm, self.db)
        prompt = llm.calls[-1]["prompt"]
        self.assertIn("【Invest 区块尾部标记】护栏修正保留", prompt)

        too_long = "\n".join(line for _ in range(60)) + "\n【超长尾部标记】"
        await self.db.save_morning_report(too_long, "维持计划")
        await answer_user_chat("今天怎么操作", self.cfg, _StubT212(_portfolio()), md, llm, self.db)
        self.assertNotIn("【超长尾部标记】", llm.calls[-1]["prompt"])


# ------------------------------------------------------------ Phase 1: sentinel


class TestDualPhilosophySentinel(_DbCase):
    async def test_invest_naked_stop_triggers_below_adaptive_line(self) -> None:
        portfolio = _portfolio(invest_positions=[_position("NVDA", 85.0, account="Invest", avg_price=100.0)])
        llm = _DsStubLLM('{"should_push": true, "alert_text": "NVDA 跌破裸仓自适应止损线，建议立即补挂止损或减仓。"}')
        sched, _, tg, _ = self._scheduler(
            portfolio,
            histories={"NVDA": _history(85.5, 85.0, periods=20)},
            bundles={"NVDA": _bundle("NVDA", 85.0)},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertIn("intraday_sentinel", fired)
        self.assertEqual(len(tg.sent), 1)
        prompt = next(c["prompt"] for c in llm.calls if c["usage_label"] == "intraday_alert")
        self.assertIn("裸露止损", prompt)
        self.assertIn("Invest", prompt)

    async def test_invest_with_protective_stop_does_not_fire(self) -> None:
        portfolio = _portfolio(
            invest_positions=[_position("NVDA", 85.0, account="Invest", avg_price=100.0)],
            invest_orders=[
                PendingOrderItem(
                    id="stop-1",
                    account="Invest",
                    raw_ticker="NVDA_EQ",
                    ticker="NVDA",
                    yahoo_symbol="NVDA",
                    name="NVDA",
                    side="SELL",
                    order_type="STOP",
                    quantity=10.0,
                    value=850.0,
                    stop_price=60.0,
                    currency="USD",
                )
            ],
        )
        llm = _DsStubLLM('{"should_push": true, "alert_text": "不该出现"}')
        sched, _, tg, _ = self._scheduler(
            portfolio,
            histories={"NVDA": _history(85.5, 85.0, periods=20)},
            bundles={"NVDA": _bundle("NVDA", 85.0)},
            llm=llm,
        )
        await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(tg.sent, [])
        self.assertEqual(llm.calls, [])

    async def test_isa_individual_38pct_filtered_by_5pct_threshold(self) -> None:
        portfolio = _portfolio(isa_positions=[_position("ULVR.L", 103.8, currency="GBp")])
        llm = _DsStubLLM('{"should_push": true, "alert_text": "不该出现"}')
        sched, md, tg, _ = self._scheduler(
            portfolio,
            histories={"ULVR.L": _history(100.0, 103.8)},
            bundles={"ULVR.L": _bundle("ULVR.L", 103.8, "GBp")},
            llm=llm,
        )
        fired = await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(md.calls, [["ULVR.L"]])
        self.assertEqual(tg.sent, [])
        self.assertEqual(llm.calls, [])
        self.assertNotIn("intraday_sentinel", fired)

    async def test_isa_core_etf_52w_drawdown_step_triggers(self) -> None:
        portfolio = _portfolio(isa_positions=[_position("VWRP.L", 89.0, currency="GBP")])
        llm = _DsStubLLM('{"should_push": true, "alert_text": "VWRP.L 宽基折价进入阶梯区，可分批补核心仓。"}')
        sched, _, tg, _ = self._scheduler(
            portfolio,
            histories={"VWRP.L": _flat_history(100.0, 89.0)},
            bundles={"VWRP.L": _bundle("VWRP.L", 89.0, "GBP", "ETF")},
            llm=llm,
        )
        await sched.check_and_run_due_jobs(NOW)
        self.assertEqual(len(tg.sent), 1)
        prompt = next(c["prompt"] for c in llm.calls if c["usage_label"] == "intraday_alert")
        self.assertIn("52 周高点", prompt)
        self.assertIn("ISA 核心宽基", prompt)


# ------------------------------------------------- Phase 1: USD fill receipt hint


class _WalletSpyDb(AdvisorDatabase):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.wallet_writes: list[tuple] = []

    async def upsert_wallet_balance(self, *args, **kwargs):  # type: ignore[override]
        self.wallet_writes.append((args, kwargs))
        return await super().upsert_wallet_balance(*args, **kwargs)


class TestUsdFillReceipt(_DbCase):
    async def test_usd_fill_receipt_readonly_reference(self) -> None:
        order_row = {
            "account_label": "Invest",
            "side": "BUY",
            "yahoo_symbol": "NVDA",
            "quantity": 10.0,
            "limit_price": 101.0,
            "currency": "USD",
        }
        match = {
            "matched_quantity": 10.0,
            "avg_price": 100.0,
            "last_filled_at_ms": 1794927600000,
            "currency": "USD",
            "is_complete": True,
        }
        text = format_fill_receipt(order_row, match, usd_wallet_available=2000.0)
        self.assertIn("参考余额约 $1,000.00", text)
        self.assertIn("/usd 1,000.00", text)
        self.assertLessEqual(len(text), 450)

        sell_match = dict(match, avg_price=110.0)
        sell_row = dict(order_row, side="SELL")
        sell_text = format_fill_receipt(sell_row, sell_match, usd_wallet_available=2000.0)
        self.assertIn("若回流 USD 钱包约 +$1,100.00", sell_text)

        isa_text = format_fill_receipt(dict(order_row, account_label="ISA"), match, usd_wallet_available=2000.0)
        self.assertNotIn("若走 USD 钱包结算", isa_text)

    async def test_fill_receipt_collection_never_writes_wallet(self) -> None:
        spy_db = _WalletSpyDb(Path(self._tmp.name) / "wallet_spy.db")
        await spy_db.init()
        await spy_db.upsert_wallet_balance(
            "Invest", "USD", 2000.0, source="manual", note="test anchor"
        )
        spy_db.wallet_writes.clear()

        order = PendingOrderItem(
            id="ord-usd-1",
            account="Invest",
            raw_ticker="NVDA_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA",
            side="BUY",
            order_type="LIMIT",
            quantity=5.0,
            value=500.0,
            limit_price=101.0,
            currency="USD",
        )
        row = _pending_order_snapshot_row(order)
        first_seen_ms = int(NOW.timestamp() * 1000) - 60_000
        await spy_db.upsert_pending_order_snapshots(
            [row],
            seen_at_ms=first_seen_ms,
            seen_at_uk="2026-11-17 14:59",
        )
        fill = HistoricalOrderItem(
            id="fill-usd-1",
            order_id="",
            account="Invest",
            raw_ticker="NVDA_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA",
            inst_currency="USD",
            side="BUY",
            order_type="LIMIT",
            filled_at="2026-11-17T15:01:00Z",
            filled_at_ms=first_seen_ms + 60_000,
            uk_time_str="2026-11-17 15:01",
            quantity=5.0,
            price=100.0,
            net_value=500.0,
            wallet_currency="USD",
            realised_pnl=0.0,
            tax_total=0.0,
        )
        portfolio = _portfolio(
            invest_recent=[fill],
            pending_supported=True,
            history_supported=True,
        )
        sched = TradingScheduler(self.cfg, _StubT212(portfolio), _StubMarketData(), _DsStubLLM(), spy_db, _StubTg())
        receipts = await sched._collect_order_fill_receipts(NOW, uk_date=UK_DATE)

        self.assertEqual(len(receipts), 1)
        self.assertIn("参考余额约 $1,500.00", receipts[0][1])
        self.assertEqual(spy_db.wallet_writes, [])
        balances = await spy_db.get_wallet_balances("Invest")
        self.assertAlmostEqual(balances["USD"]["available_amount"], 2000.0, places=2)


# ---------------------------------------------------- Phase 2: NAV split & /perf


class TestNavSplitAttribution(_DbCase):
    async def test_nav_history_split_columns_and_split_twr(self) -> None:
        rows = []
        for idx in range(6):
            day = (datetime(2026, 11, 10) + timedelta(days=idx)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "uk_date": day,
                    "total_value_gbp": 10000.0 + idx * 50.0,
                    "cash_gbp": 1000.0,
                    "invested_gbp": 9000.0,
                    "net_flow_gbp": 0.0,
                    "isa_value_gbp": 6000.0 + idx * 40.0,
                    "invest_value_gbp": 4000.0 + idx * 10.0,
                    "isa_net_flow_gbp": 0.0,
                    "invest_net_flow_gbp": 0.0,
                    "benchmark_close": 100.0 + idx,
                }
            )
        for row in rows:
            await self.db.upsert_nav_history(row)

        stored = await self.db.get_nav_history(limit=10)
        self.assertEqual(len(stored), 6)
        self.assertIn("isa_net_flow_gbp", stored[0])
        self.assertIn("invest_net_flow_gbp", stored[0])

        split = compute_account_split_performance(stored)
        self.assertTrue(split["ok"])
        self.assertTrue(split["isa"]["ok"])
        self.assertTrue(split["invest"]["ok"])
        self.assertGreater(split["isa"]["total_return_pct"], split["invest"]["total_return_pct"])

    async def test_build_nav_row_splits_external_flows(self) -> None:
        tx_isa = HistoricalTransactionItem(
            id="tx-isa",
            account="ISA",
            tx_type="DEPOSIT",
            amount=500.0,
            currency="GBP",
            date_time="2026-11-17T09:00:00Z",
            date_time_ms=1794900000000,
            uk_time_str="2026-11-17 09:00",
        )
        tx_invest = HistoricalTransactionItem(
            id="tx-inv",
            account="Invest",
            tx_type="WITHDRAW",
            amount=100.0,
            currency="GBP",
            date_time="2026-11-17T09:05:00Z",
            date_time_ms=1794900300000,
            uk_time_str="2026-11-17 09:05",
        )
        portfolio = _portfolio(isa_transactions=[tx_isa], invest_transactions=[tx_invest])
        row = build_nav_row(portfolio, benchmark_close=101.5, uk_date="2026-11-17")
        self.assertAlmostEqual(row["net_flow_gbp"], 400.0, places=2)
        self.assertAlmostEqual(row["isa_net_flow_gbp"], 500.0, places=2)
        self.assertAlmostEqual(row["invest_net_flow_gbp"], -100.0, places=2)

    async def test_negative_withdraw_is_never_flipped_into_an_inflow(self) -> None:
        # Production T212 payload: WITHDRAW amount is already -abs().
        tx = HistoricalTransactionItem(
            id="tx-wd-neg",
            account="ISA",
            tx_type="WITHDRAW",
            amount=-750.0,
            currency="GBP",
            date_time="2026-11-17T09:00:00Z",
            date_time_ms=1794900000000,
            uk_time_str="2026-11-17 09:00",
        )
        portfolio = _portfolio(isa_transactions=[tx])
        row = build_nav_row(portfolio, uk_date="2026-11-17")
        self.assertAlmostEqual(row["net_flow_gbp"], -750.0, places=2)

    def test_split_twr_smooths_a_large_invest_deposit(self) -> None:
        rows = [
            {
                "uk_date": "2026-11-10",
                "isa_value_gbp": 6000.0,
                "invest_value_gbp": 4000.0,
                "isa_net_flow_gbp": 0.0,
                "invest_net_flow_gbp": 0.0,
            },
            {
                "uk_date": "2026-11-11",
                "isa_value_gbp": 6000.0,
                "invest_value_gbp": 6000.0,  # 2000 deposit already included
                "isa_net_flow_gbp": 0.0,
                "invest_net_flow_gbp": 2000.0,
            },
            {
                "uk_date": "2026-11-12",
                "isa_value_gbp": 6000.0,
                "invest_value_gbp": 6030.0,
                "isa_net_flow_gbp": 0.0,
                "invest_net_flow_gbp": 0.0,
            },
        ]
        split = compute_account_split_performance(rows)
        self.assertTrue(split["invest"]["ok"])
        self.assertAlmostEqual(split["invest"]["total_return_pct"], 0.5, places=2)
        self.assertAlmostEqual(split["invest"]["max_drawdown_pct"], 0.0, places=2)

    async def test_perf_message_includes_split_and_core_policy(self) -> None:
        for idx in range(8):
            day = (datetime(2026, 11, 10) + timedelta(days=idx)).strftime("%Y-%m-%d")
            await self.db.upsert_nav_history(
                {
                    "uk_date": day,
                    "total_value_gbp": 10000.0 + idx * 30.0,
                    "cash_gbp": 800.0,
                    "invested_gbp": 9200.0,
                    "net_flow_gbp": 0.0,
                    "isa_value_gbp": 6000.0 + idx * 20.0,
                    "invest_value_gbp": 4000.0 + idx * 10.0,
                    "isa_net_flow_gbp": 0.0,
                    "invest_net_flow_gbp": 0.0,
                    "benchmark_close": 100.0 + idx * 0.5,
                }
            )

        dates = pd.date_range(end="2026-11-17", periods=120, freq="B")
        base = [float(i) for i in range(120)]
        histories = {
            sym: pd.DataFrame(
                {
                    "Open": [100 + v for v in base],
                    "High": [101 + v for v in base],
                    "Low": [99 + v for v in base],
                    "Close": [100 + v * 1.02 for v in base],
                    "Volume": [1_000_000.0] * 120,
                },
                index=dates,
            )
            for sym in ("VWRP.L", "VERG.L", "ISF.L", "VUAG.L")
        }
        md = _StubMarketData(histories)
        bot = AdvisorBot(self.cfg, _StubT212(_portfolio()), md, _DsStubLLM(), self.db, _StubTg())
        msg = await bot._build_perf_message()
        self.assertIn("分账户归因", msg)
        self.assertIn("ISA 价值", msg)
        self.assertIn("Invest 右侧", msg)
        self.assertIn("四桶核心政策", msg)
        self.assertIn("跟踪误差", msg)


# --------------------------------------- Phase 2: fill ledger + reflection layering


class TestFillLedgerAndReflection(_DbCase):
    async def test_match_recommendation_fill_slippage_and_units(self) -> None:
        rec = {"symbol": "NVDA", "account": "Invest", "action": "BUY"}
        fill = HistoricalOrderItem(
            id="f1",
            order_id="",
            account="Invest",
            raw_ticker="NVDA_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA",
            inst_currency="USD",
            side="BUY",
            order_type="LIMIT",
            filled_at="2026-11-17T15:01:00Z",
            filled_at_ms=int(NOW.timestamp() * 1000),
            uk_time_str="2026-11-17 15:01",
            quantity=10.0,
            price=101.0,
            net_value=1010.0,
            wallet_currency="USD",
            realised_pnl=0.0,
            tax_total=0.0,
        )
        matched = _match_recommendation_fill(
            rec,
            start_date=datetime(2026, 11, 17).date(),
            end_date=datetime(2026, 11, 18).date(),
            recent_orders=[fill],
            entry_price=100.0,
        )
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertAlmostEqual(matched[0], 101.0, places=4)
        self.assertAlmostEqual(matched[1], 1.0, places=4)

        sell_rec = dict(rec, action="TRIM")
        sell_fill = fill.model_copy(update={"side": "SELL", "price": 99.0})
        sell_matched = _match_recommendation_fill(
            sell_rec,
            start_date=datetime(2026, 11, 17).date(),
            end_date=datetime(2026, 11, 18).date(),
            recent_orders=[sell_fill],
            entry_price=100.0,
        )
        assert sell_matched is not None
        self.assertAlmostEqual(sell_matched[1], 1.0, places=4)

        lse_rec = {"symbol": "AZN.L", "account": "ISA", "action": "BUY"}
        lse_fill = fill.model_copy(
            update={"yahoo_symbol": "AZN.L", "account": "ISA", "price": 10600.0, "quantity": 2.0}
        )
        lse_matched = _match_recommendation_fill(
            lse_rec,
            start_date=datetime(2026, 11, 17).date(),
            end_date=datetime(2026, 11, 18).date(),
            recent_orders=[lse_fill],
            entry_price=100.0,
        )
        assert lse_matched is not None
        self.assertAlmostEqual(lse_matched[0], 106.0, places=4)
        self.assertAlmostEqual(lse_matched[1], 6.0, places=4)

        out_of_window = _match_recommendation_fill(
            rec,
            start_date=datetime(2026, 12, 1).date(),
            end_date=datetime(2026, 12, 2).date(),
            recent_orders=[fill],
            entry_price=100.0,
        )
        self.assertIsNone(out_of_window)

    async def test_settle_persists_fill_and_stats_split(self) -> None:
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1794313200000,
                    "account": "Invest",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                    "thesis": "动量突破",
                    "horizon_days": 5,
                    "benchmark_symbol": "SPY",
                },
                {
                    "created_at_uk": "2026-11-10 15:00",
                    "created_at_ms": 1794313200000,
                    "account": "ISA",
                    "symbol": "AZN.L",
                    "action": "ADD",
                    "entry_price": 100.0,
                    "thesis": "价值低吸",
                    "horizon_days": 5,
                    "benchmark_symbol": "ISF.L",
                },
            ]
        )
        pending = await self.db.get_pending_settlements()
        inv_rec = next(r for r in pending if r["symbol"] == "NVDA")
        isa_rec = next(r for r in pending if r["symbol"] == "AZN.L")
        await self.db.settle_recommendation(
            inv_rec["id"],
            benchmark_symbol="SPY",
            settled_at_uk="2026-11-17 22:00",
            raw_return_pct=4.0,
            alpha_pct=2.0,
            actual_filled=1,
            actual_fill_price=101.0,
            execution_slippage_pct=1.0,
            fill_status="FILLED",
        )
        await self.db.settle_recommendation(
            isa_rec["id"],
            benchmark_symbol="ISF.L",
            settled_at_uk="2026-11-17 22:00",
            raw_return_pct=6.0,
            alpha_pct=5.0,
            actual_filled=0,
            fill_status="ASSUMED",
        )

        settled_rows = await self.db.get_settled_recommendations_for_followup()
        inv_row = next(r for r in settled_rows if r["symbol"] == "NVDA")
        self.assertEqual(inv_row["actual_filled"], 1)
        self.assertAlmostEqual(inv_row["actual_fill_price"], 101.0, places=4)
        self.assertAlmostEqual(inv_row["execution_slippage_pct"], 1.0, places=4)
        self.assertEqual(inv_row["fill_status"], "FILLED")

        stats = await self.db.get_recommendation_stats()
        self.assertIn("by_account", stats)
        self.assertEqual(stats["by_account"]["Invest"]["settled_count"], 1)
        self.assertEqual(stats["by_account"]["ISA"]["settled_count"], 1)
        adoption = stats["adoption_stats"]
        self.assertEqual(adoption["checked_count"], 2)
        self.assertEqual(adoption["filled_count"], 1)
        self.assertAlmostEqual(adoption["adoption_rate_pct"], 50.0, places=1)
        self.assertAlmostEqual(adoption["avg_slippage_pct"], 1.0, places=3)

        bot = AdvisorBot(self.cfg, _StubT212(_portfolio()), _StubMarketData(), _DsStubLLM(), self.db, _StubTg())
        stats_msg = await bot._build_stats_message()
        self.assertIn("分账户表现", stats_msg)
        self.assertIn("ISA 价值", stats_msg)
        self.assertIn("Invest 右侧", stats_msg)
        self.assertIn("实盘采纳", stats_msg)
        self.assertIn("50.0%", stats_msg)

    async def test_isa_t5_no_reflection_then_t20_backfills(self) -> None:
        dates = pd.date_range("2026-10-01", "2026-12-15", freq="B")
        n = len(dates)
        stock = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0] * n,
                "Volume": [1_000_000.0] * n,
            },
            index=dates,
        )
        bench = stock.copy()
        await self.db.log_recommendations(
            [
                {
                    "created_at_uk": "2026-11-02 15:00",
                    "created_at_ms": int(datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc).timestamp() * 1000),
                    "account": "ISA",
                    "symbol": "NVDA",
                    "action": "BUY",
                    "entry_price": 100.0,
                    "thesis": "价值仓长期分批",
                    "horizon_days": 5,
                    "benchmark_symbol": "SPY",
                }
            ]
        )
        llm = _DsStubLLM("**T+20复盘**：长期价值逻辑经受住 20 日检验，下次继续分批而非追高。")
        sched, _, _, _ = self._scheduler(
            _portfolio(),
            histories={"NVDA": stock, "SPY": bench},
            llm=llm,
        )

        first = await sched.run_daily_settlement(
            now_utc=datetime(2026, 11, 10, 21, 30, tzinfo=timezone.utc)
        )
        self.assertEqual(first["settled"], 1)
        # ISA value T+5 must not train a 5-day lesson (other background jobs may call the LLM).
        self.assertEqual([c for c in llm.calls if c["usage_label"] == "settlement_reflection"], [])

        rows = await self.db.get_settled_recommendations_for_followup()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reflection_lesson"], "")

        second = await sched.run_daily_settlement(
            now_utc=datetime(2026, 12, 1, 21, 30, tzinfo=timezone.utc)
        )
        self.assertGreaterEqual(second["followup_updated"], 1)
        reflection_calls = [c for c in llm.calls if c["usage_label"] == "settlement_reflection"]
        self.assertEqual(len(reflection_calls), 1)
        self.assertIn("T+20", reflection_calls[0]["prompt"])

        lesson_rows = await self.db.get_symbol_lessons("NVDA", limit=1)
        self.assertEqual(len(lesson_rows), 1)
        lesson = lesson_rows[0]["reflection_lesson"]
        self.assertTrue(lesson)
        self.assertNotIn("#", lesson)
        self.assertNotIn("**", lesson)


# ------------------------------------------------- Phase 3: observability rollups


class TestObservabilityRollups(_DbCase):
    async def test_llm_reliability_summary(self) -> None:
        samples = [
            ("gemini-a", 1000, True),
            ("gemini-a", 2000, True),
            ("gemini-a", 5000, False),
            ("deepseek-b", 800, True),
            ("deepseek-b", 1200, True),
        ]
        for idx, (model, latency, success) in enumerate(samples):
            await self.db.log_llm_usage(
                {
                    "call_label": "report_pm" if idx % 2 == 0 else "pick_audit",
                    "model": model,
                    "thinking_level": "LOW",
                    "latency_ms": latency,
                    "success": success,
                }
            )
        summary = await self.db.get_llm_reliability_summary(days=7)
        total = summary["total"]
        self.assertEqual(total["calls"], 5)
        self.assertEqual(total["failed_calls"], 1)
        self.assertAlmostEqual(total["success_rate_pct"], 80.0, places=1)
        self.assertEqual(total["p50_latency_ms"], 1200)
        self.assertEqual(total["p95_latency_ms"], 5000)
        self.assertEqual(summary["by_model"]["gemini-a"]["calls"], 3)
        self.assertEqual(summary["by_model"]["gemini-a"]["p95_latency_ms"], 5000)
        self.assertIn("gemini-a", [name for name, item in summary["by_model"].items() if item["failed_calls"]])

    async def test_ops_quality_summary(self) -> None:
        await self.db.log_ops_event(
            severity="WARN",
            source="semantic_audit:report",
            alert_key="semantic_audit_report",
            message="[事实价不符] 早报引用价格与事实底座不一致",
            notified=False,
        )
        await self.db.log_ops_event(
            severity="WARN",
            source="semantic_audit:pick",
            alert_key="semantic_audit_pick",
            message="[建议价方向异常] 建议价高于现价",
            notified=False,
        )
        await self.db.log_ops_event(
            severity="WARN",
            source="llm",
            alert_key="report_debate_degraded",
            message="辩论降级为单边",
            notified=False,
        )
        summary = await self.db.get_ops_quality_summary(days=7)
        self.assertEqual(summary["semantic_audit"]["total"], 2)
        self.assertEqual(summary["semantic_audit"]["report"], 1)
        self.assertEqual(summary["semantic_audit"]["pick"], 1)
        self.assertEqual(summary["violation_tags"]["事实价不符"], 1)
        self.assertEqual(summary["violation_tags"]["建议价方向异常"], 1)
        self.assertEqual(summary["degraded_alerts"], 1)

    async def test_fundamentals_coverage_by_market(self) -> None:
        rows = [
            {
                "symbol": "AAPL",
                "quote_type": "EQUITY",
                "roe_pct": 35.0,
                "debt_to_equity": 1.2,
                "revenue_growth_pct": 8.0,
                "fcf_yield_pct": 3.1,
                "peg_ratio": 2.2,
                "target_mean_price": 210.0,
            },
            {
                "symbol": "MSFT",
                "quote_type": "EQUITY",
                "roe_pct": None,
                "debt_to_equity": 0.4,
                "revenue_growth_pct": None,
                "peg_ratio": 1.9,
            },
            {
                "symbol": "AZN.L",
                "quote_type": "EQUITY",
                "roe_pct": 18.0,
                "debt_to_equity": 0.6,
                "revenue_growth_pct": 5.0,
                "fcf_yield_pct": 4.0,
                "peg_ratio": 1.1,
                "target_high_price": 14000.0,
            },
            {"symbol": "ASML.AS", "quote_type": "EQUITY", "roe_pct": 22.0},
            {"symbol": "VWRP.L", "quote_type": "ETF", "roe_pct": None},
        ]
        coverage = compute_fundamentals_coverage(rows)
        self.assertTrue(coverage["ok"])
        self.assertEqual(coverage["skipped_etf"], 1)
        self.assertEqual(coverage["non_etf_count"], 4)
        us = coverage["markets"]["US"]
        self.assertEqual(us["symbols"], 2)
        self.assertAlmostEqual(us["coverage_pct"]["roe_pct"], 50.0, places=1)
        self.assertAlmostEqual(us["coverage_pct"]["revenue_growth_pct"], 50.0, places=1)
        self.assertAlmostEqual(us["coverage_pct"]["target_price"], 50.0, places=1)
        uk = coverage["markets"]["UK"]
        self.assertEqual(uk["symbols"], 1)
        self.assertAlmostEqual(uk["coverage_pct"]["target_price"], 100.0, places=1)
        eu = coverage["markets"]["EU"]
        self.assertEqual(eu["symbols"], 1)
        self.assertAlmostEqual(eu["coverage_pct"]["debt_to_equity"], 0.0, places=1)
        self.assertTrue(any("EU debt_to_equity 0%" in item for item in coverage["low_coverage"]))
        self.assertFalse(any("US peg_ratio" in item for item in coverage["low_coverage"]))


if __name__ == "__main__":
    unittest.main()
