"""Tests for UK/US DST market calendar and async TradingScheduler (fully mocked, with timeouts)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, PositionItem, T212Client
from src.engine.calendar_utils import (
    add_trading_days,
    get_market_calendar_alerts,
    get_us_market_schedule_uk,
    has_market_closed_on_date,
    infer_market_from_symbol,
    is_eu_early_close,
    is_eu_market_holiday,
    is_trading_day,
    is_uk_bank_holiday,
    is_uk_early_close,
    is_us_early_close,
    is_us_market_holiday,
    next_trading_day,
)
from src.storage.db import AdvisorDatabase


class _MockT212(T212Client):
    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return PortfolioData(
            fetched_at_uk="2026-11-13 09:00",
            fetched_at_ms=1794560400000,
            total_combined_value=10000.0,
            total_combined_cash=3000.0,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                total_account_value=10000.0,
                investments_value=7000.0,
                investments_cost=6500.0,
                unrealized_pnl=500.0,
                total_cash=3000.0,
                available_to_trade=3000.0,
                cash_ratio_pct=30.0,
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        inst_currency="USD",
                        quantity=10,
                        avg_price=120.0,
                        cur_price=135.0,
                        current_value=7000.0,
                        total_cost=6500.0,
                        pnl=500.0,
                        fx_impact=0.0,
                        return_pct=7.69,
                        weight_pct=70.0,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )


class _MockMarketData(MarketDataService):
    @staticmethod
    def _build_ohlcv(n: int = 220) -> pd.DataFrame:
        dates = pd.date_range("2025-10-01", periods=n, freq="B")
        closes = [100.0 + i * 0.2 for i in range(n)]
        return pd.DataFrame(
            {
                "Open": [c * 0.995 for c in closes],
                "High": [c * 1.01 for c in closes],
                "Low": [c * 0.99 for c in closes],
                "Close": closes,
                "Volume": [1_000_000.0 for _ in closes],
            },
            index=dates,
        )

    async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        return {s: self._build_ohlcv(220) for s in symbols if s and s.strip()}

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
        isin_map: dict[str, str] | None = None,
    ):
        histories = await self.get_histories(symbols, period="1y")
        bundles = {
            s: TickerMarketBundle(
                symbol=s,
                quote_type="EQUITY",
                currency="USD",
                short_name=s,
                isin=(isin_map or {}).get(s, ""),
                has_history=True,
            )
            for s in symbols
        }
        return histories, bundles

    async def get_macro_snapshot(self) -> MacroSnapshot:
        return MacroSnapshot(fetched_at_utc="2026-11-13 09:00 UTC", gbp_usd=1.27, vix=15.0)


class _StubLLM:
    async def generate(self, **kwargs) -> str:
        schema = kwargs.get("response_schema")
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "MorningReportDecision":
            return (
                '{"headline":"稳健观望为主","top_actions":[{"account":"ISA","symbol":"CASH","action":"HOLD","price_zone":"4%利息","reason":"赔率不足"}],'
                '"pending_order_audits":[],"daily_vetoes":["冻结芯片加仓"],"macro_and_cash":"美债偏高","account_diagnosis":"结构稳固",'
                '"flagged_positions":["NVDA 观望"],"hold_summary":"其余宽基继续持有","trade_review":"交易克制","earnings_watch":[]}'
            )
        if schema_name == "PickDecision":
            return (
                '{"headline":"空仓观望吃息","selected_actions":[],"rejected_or_frozen":["芯片超配"],'
                '"consistency_and_cash_note":"延续早报红线保留4%现金"}'
            )
        return "Stub scheduled step output"


class _StubTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str | int, text: str) -> None:
        self.sent.append((str(chat_id), text))


class TestCalendarAndScheduler(unittest.IsolatedAsyncioTestCase):
    def test_dst_normal_vs_mismatch_and_holidays(self) -> None:
        normal = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
        s_norm = get_us_market_schedule_uk(normal)
        self.assertFalse(s_norm.is_dst_mismatch)
        self.assertEqual(s_norm.open_time_uk, "14:30")
        self.assertEqual(s_norm.phase, "PRE")

        mismatch = datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc)
        s_mis = get_us_market_schedule_uk(mismatch)
        self.assertTrue(s_mis.is_dst_mismatch)
        self.assertEqual(s_mis.open_time_uk, "13:30")

        thanksgiving = datetime(2026, 11, 26, 14, 15, tzinfo=timezone.utc)
        self.assertTrue(is_us_market_holiday(thanksgiving))
        self.assertEqual(get_us_market_schedule_uk(thanksgiving).phase, "HOLIDAY")

        # England & Wales bank holidays (2026: Easter Monday, Early May, Spring, Summer, Dec 28 substitute)
        self.assertTrue(is_uk_bank_holiday(datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_uk_bank_holiday(datetime(2026, 5, 4, 8, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_uk_bank_holiday(datetime(2026, 5, 25, 8, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_uk_bank_holiday(datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_uk_bank_holiday(datetime(2026, 12, 28, 8, 0, tzinfo=timezone.utc)))
        self.assertFalse(is_uk_bank_holiday(datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)))

    async def test_scheduler_fires_and_respects_weekends_and_holidays(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                db = AdvisorDatabase(Path(tmpdir) / "test.db")
                await db.init()
                cfg = AppConfig(
                    env=EnvSettings(_env_file=None, telegram_chat_id="12345"),
                    universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
                )
                tg = _StubTg()
                sched = TradingScheduler(cfg, _MockT212(), _MockMarketData(), _StubLLM(), db, tg)

                # Weekend Saturday -> silent
                sat = datetime(2026, 11, 14, 7, 31, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(sat), [])

                # Friday 07:31 UK -> fires morning report; restarting scheduler in same window does NOT double-fire
                fri_morning = datetime(2026, 11, 13, 7, 31, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(fri_morning), ["morning"])
                self.assertEqual(len(tg.sent), 1)
                sched_restarted = TradingScheduler(cfg, _MockT212(), _MockMarketData(), _StubLLM(), db, tg)
                self.assertEqual(
                    await sched_restarted.check_and_run_due_jobs(datetime(2026, 11, 13, 7, 33, tzinfo=timezone.utc)),
                    [],
                )
                self.assertEqual(len(tg.sent), 1)

                # Friday 14:15 UK (15m before open) -> fires premarket
                fri_pre = datetime(2026, 11, 13, 14, 15, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(fri_pre), ["premarket"])

                # Friday 21:30 UK (30m after close) -> fires weekly_radar
                fri_radar = datetime(2026, 11, 13, 21, 30, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(fri_radar), ["weekly_radar"])

                # Thanksgiving Thursday 14:15 UTC -> holiday notice & no duplicate re-fire
                thx = datetime(2026, 11, 26, 14, 15, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(thx), ["premarket"])
                self.assertIn("休市日", tg.sent[-1][1])
                self.assertEqual(await sched.check_and_run_due_jobs(thx), [])

                # Black Friday (day after Thanksgiving, 13:00 ET / 18:00 UK early close) -> weekly_radar fires at 18:30 UK
                bf_early_radar = datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(bf_early_radar), ["weekly_radar"])
                self.assertEqual(
                    await sched.check_and_run_due_jobs(datetime(2026, 11, 27, 21, 30, tzinfo=timezone.utc)),
                    [],
                )

        await asyncio.wait_for(_run(), timeout=15.0)

    def test_early_close_eu_holidays_and_trading_day_math(self) -> None:
        # 1. US Early Closes (13:00 ET): Black Friday 2026, Xmas Eve 2026, Jul 3 2028
        bf_open = datetime(2026, 11, 27, 17, 50, tzinfo=timezone.utc)  # 12:50 ET
        bf_closed = datetime(2026, 11, 27, 18, 5, tzinfo=timezone.utc)  # 13:05 ET
        self.assertTrue(is_us_early_close(bf_open))
        sched_open = get_us_market_schedule_uk(bf_open)
        self.assertTrue(sched_open.is_us_early_close)
        self.assertEqual(sched_open.close_time_et, "13:00")
        self.assertEqual(sched_open.close_time_uk, "18:00")
        self.assertEqual(sched_open.phase, "OPEN")
        self.assertEqual(sched_open.minutes_to_close, 10)
        self.assertEqual(get_us_market_schedule_uk(bf_closed).phase, "CLOSED")

        xmas_eve_2026 = datetime(2026, 12, 24, 15, 0, tzinfo=timezone.utc)
        self.assertTrue(is_us_early_close(xmas_eve_2026))
        self.assertTrue(is_us_early_close(datetime(2028, 7, 3, 15, 0, tzinfo=timezone.utc)))
        # 2027-12-24 is Friday observed Christmas full holiday -> NOT an early close
        xmas_2027_obs = datetime(2027, 12, 24, 15, 0, tzinfo=timezone.utc)
        self.assertTrue(is_us_market_holiday(xmas_2027_obs))
        self.assertFalse(is_us_early_close(xmas_2027_obs))

        # 2. UK LSE Early Closes (12:30 UK): Christmas Eve & New Year's Eve
        lse_before = datetime(2026, 12, 24, 12, 25, tzinfo=timezone.utc)
        lse_after = datetime(2026, 12, 24, 12, 35, tzinfo=timezone.utc)
        self.assertTrue(is_uk_early_close(lse_before))
        self.assertTrue(is_uk_early_close(datetime(2026, 12, 31, 10, 0, tzinfo=timezone.utc)))
        self.assertFalse(has_market_closed_on_date(date(2026, 12, 24), "UK", now_utc=lse_before))
        self.assertTrue(has_market_closed_on_date(date(2026, 12, 24), "UK", now_utc=lse_after))

        # 3. EU Holidays & Xetra vs Euronext differences
        may_day_2026 = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)  # Friday 2026-05-01
        self.assertTrue(is_eu_market_holiday(may_day_2026, venue="EU"))
        self.assertTrue(is_eu_market_holiday(may_day_2026, venue="XETRA"))
        self.assertTrue(is_eu_market_holiday(may_day_2026, venue="EURONEXT"))
        self.assertFalse(is_us_market_holiday(may_day_2026))
        self.assertFalse(is_uk_bank_holiday(may_day_2026))

        self.assertTrue(is_eu_market_holiday(xmas_eve_2026, venue="XETRA"))
        self.assertFalse(is_eu_market_holiday(xmas_eve_2026, venue="EURONEXT"))
        self.assertTrue(is_eu_early_close(xmas_eve_2026, venue="EURONEXT"))
        self.assertFalse(is_eu_early_close(xmas_eve_2026, venue="XETRA"))
        # Euronext closes at 14:05 CET (= 13:05 UTC in winter) on Dec 24
        self.assertFalse(has_market_closed_on_date("2026-12-24", "EURONEXT", now_utc=datetime(2026, 12, 24, 13, 0, tzinfo=timezone.utc)))
        self.assertTrue(has_market_closed_on_date("2026-12-24", "EURONEXT", now_utc=datetime(2026, 12, 24, 13, 10, tzinfo=timezone.utc)))

        self.assertEqual(infer_market_from_symbol("VWRP.L"), "UK")
        self.assertEqual(infer_market_from_symbol("SAP.DE"), "XETRA")
        self.assertEqual(infer_market_from_symbol("ASML.AS"), "EURONEXT")
        self.assertEqual(infer_market_from_symbol("MC.PA"), "EURONEXT")
        self.assertEqual(infer_market_from_symbol("NVDA"), "US")

        # 4. add_trading_days & next_trading_day across Easter & May Day 2026
        # 2026-04-02 (Thu) -> Good Friday 2026-04-03 (US/UK/EU closed) -> Easter Mon 2026-04-06 (US open, UK/EU closed)
        self.assertEqual(add_trading_days("2026-04-02", 1, "US"), date(2026, 4, 6))
        self.assertEqual(add_trading_days("2026-04-02", 1, "UK"), date(2026, 4, 7))
        self.assertEqual(add_trading_days("2026-04-02", 1, "EU"), date(2026, 4, 7))
        self.assertEqual(add_trading_days(date(2026, 4, 2), 5, "US"), date(2026, 4, 10))
        self.assertEqual(add_trading_days(date(2026, 4, 2), 5, "UK"), date(2026, 4, 13))
        # May Day 2026-05-01 (Fri closed EU, open US/UK) + Early May UK Bank Holiday 2026-05-04 (Mon closed UK)
        self.assertEqual(add_trading_days("2026-04-28", 5, "US"), date(2026, 5, 5))
        self.assertEqual(add_trading_days("2026-04-28", 5, "UK"), date(2026, 5, 6))
        self.assertEqual(add_trading_days("2026-04-28", 5, "EU"), date(2026, 5, 6))
        # Negative step, zero step, and weekend next_trading_day
        self.assertEqual(add_trading_days("2026-04-06", 0, "US"), date(2026, 4, 6))
        self.assertEqual(add_trading_days("2026-04-06", -1, "US"), date(2026, 4, 2))
        self.assertFalse(is_trading_day("2026-05-01", "EU"))
        self.assertEqual(next_trading_day("2026-05-01", "EU", inclusive=True), date(2026, 5, 4))

        # 5. Alerts summary smoke check
        alerts = get_market_calendar_alerts(bf_open)
        self.assertTrue(any("美股半日市" in a for a in alerts))


if __name__ == "__main__":
    unittest.main()
