"""Tests for UK/US DST market calendar and async TradingScheduler (fully mocked, with timeouts)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.bot.handlers import AdvisorBot
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
        bundles: dict[str, TickerMarketBundle] = {}
        for s in symbols:
            currency = "GBP" if s.endswith(".L") else ("EUR" if s.endswith((".DE", ".AS", ".PA")) else "USD")
            hist = histories.get(s)
            curr_price = round(float(hist["Close"].iloc[-1]), 2) if hist is not None and not hist.empty else None
            b = TickerMarketBundle(
                symbol=s,
                quote_type="EQUITY",
                currency=currency,
                short_name=s,
                current_price=curr_price,
                isin=(isin_map or {}).get(s, ""),
                has_history=True,
            )
            b.identity_summary = b.build_identity_anchor(isin_override=(isin_map or {}).get(s))
            bundles[s] = b
        return histories, bundles

    async def get_us_deep_intel(self, symbol: str) -> dict:
        return {}

    async def get_us_realtime_quote(self, symbol: str) -> dict | None:
        return None

    async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {}

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

    async def test_imp16_morning_catchup_retry_and_lock_on_success_only(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "test.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="12345"),
                universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
            )
            tg = _StubTg()
            sched = TradingScheduler(
                cfg,
                _MockT212(),
                _MockMarketData(),
                _StubLLM(),
                db,
                tg,
                morning_retry_cooldown_s=300.0,
            )

            call_outcomes = [
                "⚠️ 家庭宽带重拨导致 T212 超时",
                "早报恢复补发成功正文",
            ]

            async def _fake_morning(*args, **kwargs) -> str:
                return call_outcomes.pop(0)

            with patch("src.bot.scheduler.generate_morning_report", side_effect=_fake_morning):
                # 07:31 UK fails -> NOT locked in scheduler_runs, no Telegram spam
                t1 = datetime(2026, 11, 13, 7, 31, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(t1), ["morning"])
                self.assertEqual(sched.last_job_runs["morning"]["status"], "FAILED")
                self.assertFalse(await db.has_scheduler_job_fired("morning", "2026-11-13"))
                self.assertEqual(len(tg.sent), 0)

                # 07:33 UK inside 300s cooldown -> skipped
                t2 = datetime(2026, 11, 13, 7, 33, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(t2), [])
                self.assertEqual(len(call_outcomes), 1)

                # 08:15 UK after cooldown & inside 09:30 catch-up window -> succeeds & locks DB
                t3 = datetime(2026, 11, 13, 8, 15, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(t3), ["morning"])
                self.assertEqual(sched.last_job_runs["morning"]["status"], "OK")
                self.assertTrue(await db.has_scheduler_job_fired("morning", "2026-11-13"))
                self.assertEqual(len(tg.sent), 1)
                self.assertEqual(tg.sent[0][1], "早报恢复补发成功正文")

                # Restarted scheduler at 08:45 UK must NOT double-fire
                sched2 = TradingScheduler(cfg, _MockT212(), _MockMarketData(), _StubLLM(), db, tg)
                self.assertEqual(
                    await sched2.check_and_run_due_jobs(datetime(2026, 11, 13, 8, 45, tzinfo=timezone.utc)),
                    [],
                )

            # Separate day: missed 07:30-07:35 due to downtime, boots at 08:50 UK -> catches up directly
            async def _fake_morning_ok(*args, **kwargs) -> str:
                return "重启后补发早报"

            with patch("src.bot.scheduler.generate_morning_report", side_effect=_fake_morning_ok):
                mon_late = datetime(2026, 11, 16, 8, 50, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(mon_late), ["morning"])
                self.assertTrue(await db.has_scheduler_job_fired("morning", "2026-11-16"))
                self.assertEqual(tg.sent[-1][1], "重启后补发早报")

            # Exhausted catch-up window after 09:30 UK emits one missed notice if failures happened earlier
            async def _fake_morning_fail(*args, **kwargs) -> str:
                return "⚠️ 持续断网"

            with patch("src.bot.scheduler.generate_morning_report", side_effect=_fake_morning_fail):
                tue_fail = datetime(2026, 11, 17, 7, 31, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(tue_fail), ["morning"])
                self.assertFalse(await db.has_scheduler_job_fired("morning", "2026-11-17"))
                tue_after = datetime(2026, 11, 17, 9, 35, tzinfo=timezone.utc)
                self.assertEqual(await sched.check_and_run_due_jobs(tue_after), [])
                self.assertIn("补发窗口", tg.sent[-1][1])
                prev_cnt = len(tg.sent)
                # Calling again at 10:00 UK does not duplicate the missed notice
                self.assertEqual(
                    await sched.check_and_run_due_jobs(datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)),
                    [],
                )
                self.assertEqual(len(tg.sent), prev_cnt)

    async def test_imp16_daily_settlement_t5_calendars_alpha_reflection_and_prune(self) -> None:
        class _SettlementMarketData(MarketDataService):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[list[str]] = []

            @staticmethod
            def _series_df(dates_str: list[str], closes: list[float]) -> pd.DataFrame:
                idx = pd.to_datetime(dates_str)
                return pd.DataFrame(
                    {
                        "Open": [c * 0.995 for c in closes],
                        "High": [c * 1.01 for c in closes],
                        "Low": [c * 0.99 for c in closes],
                        "Close": closes,
                        "Volume": [1_000_000.0 for _ in closes],
                    },
                    index=idx,
                )

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                self.calls.append(list(symbols))
                dates = [
                    "2026-11-05",
                    "2026-11-06",
                    "2026-11-09",
                    "2026-11-10",
                    "2026-11-11",
                    "2026-11-12",
                    "2026-11-13",
                ]
                out: dict[str, pd.DataFrame] = {}
                for sym in symbols:
                    if sym == "NVDA":
                        out[sym] = self._series_df(dates, [98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
                    elif sym == "SPY":
                        out[sym] = self._series_df(dates, [495.0, 500.0, 501.0, 503.0, 505.0, 508.0, 510.0])
                    elif sym == "AZN.L":
                        # Stock stored in GBp (10000 -> 10600, +6.0%), while rec entry_price was 100.0 GBP
                        out[sym] = self._series_df(dates, [9900.0, 10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10600.0])
                    elif sym == "ISF.L":
                        out[sym] = self._series_df(dates, [795.0, 800.0, 802.0, 804.0, 805.0, 806.0, 808.0])
                    elif sym == "AMD":
                        out[sym] = self._series_df(dates, [148.0, 150.0, 149.0, 147.0, 146.0, 145.0, 144.0])
                return out

        class _ReflectionLLM:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.fail_symbols: set[str] = set()

            async def generate(self, **kwargs) -> str:
                self.calls.append(dict(kwargs))
                prompt = kwargs.get("prompt", "")
                for sym in self.fail_symbols:
                    if f"标的: {sym} " in prompt:
                        return "⚠️ Gemini 瞬时不可用"
                return "## **T+5复盘**：顺势突破录得正超额Alpha，短期逻辑获验证；下次继续控制仓位与止损。"

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "test.db")
            await db.init()

            # Insert old records (>90 days before 2026-11-13) to test pruning while keeping latest snapshot
            old_ms = int(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
            recent_ms = int(datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
            async with db._connect(write=True) as conn:
                await conn.execute(
                    "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("2026-07-01", "2026-07-01 12:00", old_ms, 5000.0, "{}"),
                )
                await conn.execute(
                    "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("2026-11-10", "2026-11-10 12:00", recent_ms, 10000.0, "{}"),
                )
                await conn.execute(
                    "INSERT INTO ops_events (created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, 0)",
                    ("2026-07-01 12:00", old_ms, "WARN", "test", "old_k", "old msg"),
                )
                await conn.commit()

            base_ms = int(datetime(2026, 11, 6, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-11-06 21:30",
                        "created_at_ms": base_ms,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 100.0,
                        "thesis": "AI算力需求景气",
                        "horizon_days": 5,
                        "benchmark_symbol": "SPY",
                    },
                    {
                        "created_at_uk": "2026-11-06 16:40",
                        "created_at_ms": base_ms + 1000,
                        "account": "ISA",
                        "symbol": "AZN.L",
                        "action": "ADD",
                        "entry_price": 100.0,  # GBP vs GBp 10000 in history -> auto-aligned
                        "thesis": "医药防御低吸",
                        "horizon_days": 5,
                        "benchmark_symbol": "ISF.L",
                    },
                    {
                        "created_at_uk": "2026-11-06 21:30",
                        "created_at_ms": base_ms + 2000,
                        "account": "Invest",
                        "symbol": "AMD",
                        "action": "BUY",
                        "entry_price": 150.0,
                        "thesis": "芯片补涨",
                        "horizon_days": 5,
                        "benchmark_symbol": "SPY",
                    },
                    {
                        "created_at_uk": "2026-11-06 21:30",
                        "created_at_ms": base_ms + 3000,
                        "account": "ISA",
                        "symbol": "VWRP.L",
                        "action": "WATCH",
                        "entry_price": None,
                        "thesis": "无价格观察",
                    },
                    {
                        "created_at_uk": "2026-11-12 21:30",
                        "created_at_ms": recent_ms + 5000,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "ADD",
                        "entry_price": 108.0,
                        "thesis": "未满5个交易日",
                        "horizon_days": 5,
                        "benchmark_symbol": "SPY",
                    },
                ]
            )

            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="12345"))
            md = _SettlementMarketData()
            llm = _ReflectionLLM()
            llm.fail_symbols.add("AMD")  # transient failure for AMD should keep it PENDING
            tg = _StubTg()
            sched = TradingScheduler(cfg, _MockT212(), md, llm, db, tg)

            # At 18:00 UTC on 2026-11-13: LSE closed (16:30 UK), US still OPEN (21:00 UTC)!
            # Only AZN.L (UK) settles + VWRP.L (no price) skips; NVDA/AMD (US) remain PENDING!
            res_lse_close = await sched.run_daily_settlement(
                now_utc=datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
            )
            self.assertEqual(res_lse_close["settled"], 1)
            self.assertEqual(res_lse_close["skipped"], 1)
            self.assertEqual(res_lse_close["pruned"]["portfolio_snapshots"], 1)
            self.assertEqual(res_lse_close["pruned"]["ops_events"], 1)

            # Verify AZN.L settled accurately (+6.0% vs ISF.L +1.0% -> Alpha +5.0%) despite GBP vs GBp entry
            azn_lessons = await db.get_symbol_lessons("AZN.L", limit=1)
            self.assertEqual(len(azn_lessons), 1)
            self.assertAlmostEqual(azn_lessons[0]["raw_return_pct"], 6.0, places=2)
            self.assertAlmostEqual(azn_lessons[0]["alpha_pct"], 5.0, places=2)
            self.assertNotIn("#", azn_lessons[0]["reflection_lesson"])
            self.assertNotIn("**", azn_lessons[0]["reflection_lesson"])

            # At 22:00 UTC on 2026-11-13 via check_and_run_due_jobs -> daily_settlement runs
            fired = await sched.check_and_run_due_jobs(datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc))
            self.assertEqual(fired, ["daily_settlement"])
            self.assertTrue(await db.has_scheduler_job_fired("daily_settlement", "2026-11-13"))
            self.assertEqual(len(tg.sent), 0)  # background settlement must not spam Telegram chat

            # NVDA settled (+10.0% vs SPY +2.0% -> Alpha +8.0%), AMD failed reflection so stayed PENDING, newer NVDA not due
            nvda_lessons = await db.get_symbol_lessons("NVDA", limit=1)
            self.assertEqual(len(nvda_lessons), 1)
            self.assertAlmostEqual(nvda_lessons[0]["raw_return_pct"], 10.0, places=2)
            self.assertAlmostEqual(nvda_lessons[0]["alpha_pct"], 8.0, places=2)
            self.assertEqual(llm.calls[-1]["model"], cfg.models.chat_model)
            self.assertEqual(llm.calls[-1]["thinking_level"], cfg.models.chat_thinking_level)
            self.assertEqual(llm.calls[-1]["usage_label"], "settlement_reflection")

            still_pending = await db.get_pending_settlements()
            self.assertEqual([r["symbol"] for r in still_pending], ["AMD", "NVDA"])

            # Re-running check_and_run_due_jobs at 22:15 UTC does not re-fire
            self.assertEqual(
                await sched.check_and_run_due_jobs(datetime(2026, 11, 13, 22, 15, tzinfo=timezone.utc)),
                [],
            )

    async def test_imp16_shared_execution_lock_mutex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "test.db")
            await db.init()
            shared_lock = asyncio.Lock()
            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="12345"))
            sched = TradingScheduler(
                cfg,
                _MockT212(),
                _MockMarketData(),
                _StubLLM(),
                db,
                _StubTg(),
                execution_lock=shared_lock,
            )

            observed_job_inside = None
            orig_impl = sched._run_daily_settlement_impl

            async def _spy_impl(now_utc=None):
                nonlocal observed_job_inside
                observed_job_inside = sched.current_job_name
                return await orig_impl(now_utc=now_utc)

            sched._run_daily_settlement_impl = _spy_impl  # type: ignore[method-assign]

            async with shared_lock:
                task = asyncio.create_task(
                    sched.run_daily_settlement(now_utc=datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc))
                )
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
                self.assertIsNone(sched.current_job_name)

            res = await asyncio.wait_for(task, timeout=5.0)
            self.assertEqual(res["checked"], 0)
            self.assertEqual(observed_job_inside, "daily_settlement")
            self.assertIsNone(sched.current_job_name)
            self.assertFalse(shared_lock.locked())

    async def test_imp20_prewarm_windows_daily_metadata_sync_and_silence(self) -> None:
        from src.data_providers.t212_client import HistoricalDividendItem

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "prewarm.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="12345"),
                watchlist=["AAPL"],
                universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["AMD"]),
            )
            tg = _StubTg()

            class _PrewarmT212(_MockT212):
                def __init__(self) -> None:
                    super().__init__()
                    self.inst_sync_calls = 0

                async def fetch_instruments_metadata(self) -> list[dict]:
                    self.inst_sync_calls += 1
                    return [
                        {
                            "ticker": "NVDA_US_EQ",
                            "yahoo_symbol": "NVDA",
                            "isin": "US67066G1040",
                            "currency_code": "USD",
                            "short_name": "NVDA",
                            "inst_type": "STOCK",
                            "extended_hours": True,
                            "working_schedule_id": 55,
                            "updated_at_ms": 1794560400000,
                        }
                    ]

                async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
                    p = await super().get_portfolio(use_cache=use_cache)
                    p.isa.recent_dividends = [
                        HistoricalDividendItem(
                            id="div-prewarm-1",
                            account="ISA",
                            raw_ticker="VWRPl_EQ",
                            ticker="VWRP",
                            yahoo_symbol="VWRP.L",
                            name="Vanguard FTSE All-World",
                            amount=18.5,
                            paid_on="2026-11-12T10:00:00Z",
                            paid_on_ms=1794477600000,
                            uk_time_str="11-12 10:00",
                        )
                    ]
                    return p

            class _PrewarmMarketData(_MockMarketData):
                def __init__(self) -> None:
                    super().__init__()
                    self.warmed_batches: list[tuple[list[str], str]] = []
                    self.macro_calls = 0

                async def get_histories(self, symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
                    self.warmed_batches.append((list(symbols), period))
                    return await super().get_histories(symbols, period=period)

                async def get_macro_snapshot(self) -> MacroSnapshot:
                    self.macro_calls += 1
                    return await super().get_macro_snapshot()

            t212 = _PrewarmT212()
            md = _PrewarmMarketData()
            sched = TradingScheduler(cfg, t212, md, _StubLLM(), db, tg)

            # 1) 07:20 UK Friday -> triggers silent prewarm_morning & daily 1x metadata sync
            t_morning_prewarm = datetime(2026, 11, 13, 7, 20, tzinfo=timezone.utc)
            self.assertEqual(await sched.check_and_run_due_jobs(t_morning_prewarm), ["prewarm_morning"])
            self.assertEqual(len(tg.sent), 0)
            self.assertEqual(t212.inst_sync_calls, 1)
            self.assertEqual(md.macro_calls, 1)
            self.assertEqual(md.warmed_batches[0][1], "2y")
            for sym in ("NVDA", "AAPL", "VWRP.L", "AMD", "SPY"):
                self.assertIn(sym, md.warmed_batches[0][0])
            self.assertTrue(await db.has_scheduler_job_fired("prewarm_morning", "2026-11-13"))
            self.assertTrue(await db.has_scheduler_job_fired("t212_metadata_sync", "2026-11-13"))
            self.assertEqual(len(await db.get_t212_instrument_map()), 1)
            self.assertAlmostEqual((await db.get_cumulative_dividends_by_symbol())[("ISA", "VWRP.L")], 18.5)

            # Duplicate check in same window does NOT re-fire
            self.assertEqual(
                await sched.check_and_run_due_jobs(datetime(2026, 11, 13, 7, 25, tzinfo=timezone.utc)),
                [],
            )

            # 2) 14:05 UK Friday (-10m before 14:15 premarket sentinel) -> fires prewarm_premarket without duplicating metadata sync
            self.assertEqual(
                await sched.check_and_run_due_jobs(datetime(2026, 11, 13, 14, 5, tzinfo=timezone.utc)),
                ["prewarm_premarket"],
            )
            self.assertEqual(t212.inst_sync_calls, 1)
            self.assertEqual(len(tg.sent), 0)

            # 3) 21:20 UK Friday (-10m before 21:30 weekly_radar) -> fires prewarm_weekly_radar silently
            self.assertEqual(
                await sched.check_and_run_due_jobs(datetime(2026, 11, 13, 21, 20, tzinfo=timezone.utc)),
                ["prewarm_weekly_radar"],
            )
            self.assertEqual(len(tg.sent), 0)

            # Direct run_prewarm execution works cleanly
            manual_res = await sched.run_prewarm("manual", now_utc=datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc))
            self.assertGreaterEqual(manual_res["symbols_warmed"], 5)
            self.assertTrue(manual_res["macro_ok"])

    async def test_imp22_t10_t20_mfe_mae_followup_and_saturday_quant_audit(self) -> None:
        import numpy as np

        class _FollowupMarketData(MarketDataService):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[list[str], str]] = []

            @staticmethod
            def _make_range_df(
                dates: pd.DatetimeIndex,
                closes: list[float],
                highs: list[float] | None = None,
                lows: list[float] | None = None,
            ) -> pd.DataFrame:
                h = highs if highs is not None else [c * 1.01 for c in closes]
                l = lows if lows is not None else [c * 0.99 for c in closes]
                return pd.DataFrame(
                    {
                        "Open": [c * 0.998 for c in closes],
                        "High": h,
                        "Low": l,
                        "Close": closes,
                        "Volume": [1_000_000.0 for _ in closes],
                    },
                    index=dates,
                )

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                self.calls.append((list(symbols), period))
                # 25 business days starting 2026-10-05: enough for T+5, T+10, T+20
                dates = pd.date_range("2026-10-05", periods=25, freq="B")
                long_dates = pd.date_range("2025-07-01", periods=320, freq="B")
                out: dict[str, pd.DataFrame] = {}
                for sym in symbols:
                    if sym == "NVDA":
                        # Entry on 2026-10-05 (idx 0, close=100)
                        # T+5 (idx 5, 2026-10-12): close=105 (+5%), high peak=108 (+8%), low=98 (-2%)
                        # T+10 (idx 10, 2026-10-19): close=110 (+10%), high peak=115 (+15%)
                        # T+20 (idx 20, 2026-11-02): close=120 (+20%), high peak=125 (+25%), low=95 (-5%)
                        closes = [100.0 + i * 1.0 for i in range(25)]
                        highs = [c * 1.01 for c in closes]
                        lows = [c * 0.995 for c in closes]
                        highs[3] = 108.0
                        lows[2] = 98.0
                        highs[8] = 115.0
                        highs[18] = 125.0
                        lows[15] = 95.0
                        out[sym] = self._make_range_df(dates, closes, highs=highs, lows=lows)
                    elif sym == "SPY":
                        closes = [500.0 + i * 2.5 for i in range(25)]
                        out[sym] = self._make_range_df(dates, closes)
                    elif sym == "ZOMB.L":
                        z_closes = np.concatenate([np.linspace(120.0, 150.0, 120), np.linspace(150.0, 70.0, 200)])
                        out[sym] = self._make_range_df(
                            long_dates,
                            list(z_closes),
                            highs=list(z_closes * 1.01),
                            lows=list(z_closes * 0.99),
                        )
                    elif sym in ("VWRP.L", "ISF.L"):
                        b_closes = [100.0 + i * 0.1 for i in range(len(long_dates))]
                        out[sym] = self._make_range_df(long_dates, b_closes)
                return out

        class _SimpleReflectLLM:
            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, **kwargs) -> str:
                self.calls += 1
                return "T+5 正超额收益，严格跟踪止损纪律。"

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp22_sched.db")
            await db.init()

            rec_ms = int(datetime(2026, 10, 5, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-10-05 21:30",
                        "created_at_ms": rec_ms,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 100.0,
                        "thesis": "AI breakout",
                        "horizon_days": 5,
                        "benchmark_symbol": "SPY",
                    }
                ]
            )

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="12345"),
                universe=UniverseSection(isa_long_term=["VWRP.L", "ZOMB.L"]),
            )
            md = _FollowupMarketData()
            llm = _SimpleReflectLLM()
            tg = _StubTg()
            sched = TradingScheduler(cfg, _MockT212(), md, llm, db, tg)

            # 1) Settle at T+5 close (2026-10-12 22:00 UTC) -> settles T+5 with initial MFE (+8%) & MAE (-2%)
            res_t5 = await sched.run_daily_settlement(now_utc=datetime(2026, 10, 12, 22, 0, tzinfo=timezone.utc))
            self.assertEqual(res_t5["settled"], 1)
            self.assertEqual(llm.calls, 1)

            recs = await db.get_recent_recommendations(limit=5)
            self.assertAlmostEqual(recs[0]["raw_return_pct"], 5.0, places=2)
            self.assertAlmostEqual(recs[0]["mfe_pct"], 8.0, places=2)
            self.assertAlmostEqual(recs[0]["mae_pct"], -2.0, places=2)
            self.assertIsNone(recs[0]["t10_alpha_pct"])
            self.assertIsNone(recs[0]["t20_alpha_pct"])

            # 2) Run settlement after T+20 close (2026-11-03 22:00 UTC) -> 0 LLM calls! Updates T+10, T+20, and MFE/MAE
            res_t20 = await sched.run_daily_settlement(now_utc=datetime(2026, 11, 3, 22, 0, tzinfo=timezone.utc))
            self.assertEqual(res_t20["settled"], 0)
            self.assertEqual(res_t20["followup_updated"], 1)
            self.assertEqual(llm.calls, 1)  # 0 extra LLM calls for T+10 / T+20 follow-up

            recs_after = await db.get_recent_recommendations(limit=5)
            self.assertAlmostEqual(recs_after[0]["t10_return_pct"], 10.0, places=2)
            self.assertAlmostEqual(recs_after[0]["t10_alpha_pct"], 5.0, places=2)  # NVDA +10% vs SPY +5%
            self.assertAlmostEqual(recs_after[0]["t20_return_pct"], 20.0, places=2)
            self.assertAlmostEqual(recs_after[0]["t20_alpha_pct"], 10.0, places=2)  # NVDA +20% vs SPY +10%
            self.assertAlmostEqual(recs_after[0]["mfe_pct"], 25.0, places=2)
            self.assertAlmostEqual(recs_after[0]["mae_pct"], -5.0, places=2)

            # 3) Saturday idle window (2026-11-14 11:00 UTC) fires weekly_quant_audit silently once per week
            sat_early = datetime(2026, 11, 14, 7, 31, tzinfo=timezone.utc)
            self.assertEqual(await sched.check_and_run_due_jobs(sat_early), [])

            sat_idle = datetime(2026, 11, 14, 11, 0, tzinfo=timezone.utc)
            fired_sat = await sched.check_and_run_due_jobs(sat_idle)
            self.assertEqual(fired_sat, ["weekly_quant_audit"])
            self.assertEqual(len(tg.sent), 0)  # must not spam Telegram

            # Saturday 14:00 UK -> weekly Invest GBP+USD ledger reminder, once per ISO week
            fired_sat_noon = await sched.check_and_run_due_jobs(datetime(2026, 11, 14, 14, 0, tzinfo=timezone.utc))
            self.assertEqual(fired_sat_noon, ["weekly_wallet_reconcile"])
            self.assertEqual(len(tg.sent), 1)
            self.assertIn("双钱包对账提醒", tg.sent[-1][1])
            self.assertIn("/usd", tg.sent[-1][1])
            self.assertEqual(
                await sched.check_and_run_due_jobs(datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)), []
            )

            latest_audit = await db.get_latest_quant_audit()
            assert latest_audit is not None
            self.assertEqual(latest_audit["audit_date"], "2026-11-14")
            self.assertIn("ZOMB.L", latest_audit["summary_text"])


class TestWalletCommandParsing(unittest.TestCase):
    """Phase 3/5: /wallet & /usd argument validation before any DB write."""

    def test_usd_and_wallet_argument_validation(self) -> None:
        parse = AdvisorBot._parse_wallet_args
        self.assertEqual(parse("/usd", ["600"]), ("", {"USD": (600.0, 0.0)}))
        self.assertEqual(parse("/usd", ["600", "25"]), ("", {"USD": (600.0, 25.0)}))
        self.assertEqual(parse("/wallet", ["gbp", "1500"]), ("", {"GBP": (1500.0, 0.0)}))
        self.assertEqual(
            parse("/wallet", ["gbp", "1500", "usd", "600"]),
            ("", {"GBP": (1500.0, 0.0), "USD": (600.0, 0.0)}),
        )
        self.assertEqual(parse("/wallet", []), ("", {}))
        self.assertEqual(parse("/usd", []), ("", {}))
        self.assertIn("缺少金额", parse("/wallet", ["gbp"])[0])
        self.assertIn("格式不正确", parse("/wallet", ["usd", "abc"])[0])
        self.assertIn("负数", parse("/usd", ["-1"])[0])
        self.assertIn("用法", parse("/wallet", ["foo"])[0])


if __name__ == "__main__":
    unittest.main()
