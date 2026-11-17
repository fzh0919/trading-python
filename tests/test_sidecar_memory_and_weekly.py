"""Offline tests for Phase 4: weekly DeepSeek diagnosis + verbatim-anchored preference memory."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite

from src.agents.prompts import format_preference_notes_block
from src.bot.handlers import AdvisorBot
from src.bot.scheduler import TradingScheduler, _quote_is_verbatim
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, T212Client
from src.storage.db import AdvisorDatabase

LONDON_TZ = ZoneInfo("Europe/London")


class _DsStubLLM:
    """Stub exposing a DeepSeek key plus programmable replies keyed by usage_label."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.deepseek_api_key = "test-key"
        self.replies = replies or {}
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.replies.get(str(kwargs.get("usage_label")), "{}")


class _NoKeyLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise AssertionError("sidecar must not call the LLM without a DeepSeek key")


class _StubTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="stub")
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str | int, text: str) -> None:
        self.sent.append((str(chat_id), text))


class _StubT212(T212Client):
    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return PortfolioData(
            fetched_at_uk="2026-11-17 09:00",
            fetched_at_ms=1794900000000,
            total_combined_value=10000.0,
            total_combined_cash=2500.0,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=10000.0, total_cash=2500.0),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )


class _StubMarketData(MarketDataService):
    pass


def _cfg() -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, deepseek_api_key="", ops_notify_url="", telegram_chat_id="123"),
        universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
    )


class _DbTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "advisor.db"
        self.db = AdvisorDatabase(self.db_path)
        await self.db.init()
        self.cfg = _cfg()
        self.llm = _DsStubLLM()
        self.tg = _StubTg()
        self.sched = TradingScheduler(
            self.cfg, _StubT212(), _StubMarketData(), self.llm, self.db, self.tg
        )


class TestQuantAuditStorage(_DbTestCase):
    async def test_legacy_table_gets_ai_diagnosis_column(self) -> None:
        legacy_path = Path(self._tmp.name) / "legacy.db"
        async with aiosqlite.connect(legacy_path) as conn:
            await conn.executescript(
                """
                CREATE TABLE quant_health_audits (
                    audit_date TEXT PRIMARY KEY,
                    created_at_uk TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}'
                );
                """
            )
            await conn.execute(
                "INSERT INTO quant_health_audits VALUES ('2026-11-01','2026-11-01 12:00','老体检','{}')"
            )
            await conn.commit()
        legacy_db = AdvisorDatabase(legacy_path)
        await legacy_db.init()
        latest = await legacy_db.get_latest_quant_audit()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["summary_text"], "老体检")
        self.assertEqual(latest["ai_diagnosis"], "")

    async def test_ai_diagnosis_roundtrip(self) -> None:
        await self.db.save_quant_audit(
            audit_date="2026-11-14",
            summary_text="体检摘要",
            payload={"universe_count": 150},
            created_at_uk="2026-11-14 12:00",
            ai_diagnosis="把追踪止盈从 12% 收紧到 10%",
        )
        latest = await self.db.get_latest_quant_audit()
        self.assertEqual(latest["ai_diagnosis"], "把追踪止盈从 12% 收紧到 10%")
        self.assertEqual(latest["payload"]["universe_count"], 150)


class TestPreferenceNotesStorage(_DbTestCase):
    async def test_upsert_dedupes_and_ttl_filters(self) -> None:
        notes = [
            {
                "source_cycle_date": "2026-11-17",
                "quote_text": "不碰军工",
                "rule_summary": "年底前回避军工",
                "expires_on_uk": "2026-12-01",
            },
            {
                "source_cycle_date": "2026-11-17",
                "quote_text": "不碰军工",
                "rule_summary": "年底前回避军工",
                "expires_on_uk": "2026-12-01",
            },
        ]
        self.assertEqual(await self.db.upsert_preference_notes(notes), 1)
        active = await self.db.get_active_preference_notes("2026-11-20")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["quote_text"], "不碰军工")
        self.assertEqual(await self.db.get_active_preference_notes("2026-12-02"), [])
        self.assertEqual(await self.db.prune_expired_preference_notes("2026-12-02"), 1)
        self.assertEqual(await self.db.get_active_preference_notes("2026-12-02"), [])

    async def test_upsert_skips_invalid_entries(self) -> None:
        saved = await self.db.upsert_preference_notes(
            [
                {"quote_text": "", "rule_summary": "x", "expires_on_uk": "2026-12-01"},
                {"quote_text": "q", "rule_summary": "", "expires_on_uk": "2026-12-01"},
                {"quote_text": "q", "rule_summary": "r"},
                "not-a-dict",
            ]
        )
        self.assertEqual(saved, 0)

    async def test_user_queries_since_filters_by_cutoff(self) -> None:
        await self.db.append_chat_turn("今年年底前不碰军工板块", "好的", cycle_date="2026-11-17")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.assertIn("今年年底前不碰军工板块", await self.db.get_user_queries_since(now_ms - 600_000))
        self.assertEqual(await self.db.get_user_queries_since(now_ms + 600_000), [])


class TestPreferenceExtraction(_DbTestCase):
    async def test_only_verbatim_quotes_are_stored(self) -> None:
        await self.db.append_chat_turn(
            "今年年底前不碰军工板块，等财报确认再说", "记下了", cycle_date="2026-11-17"
        )
        self.llm.replies["preference_extract"] = (
            '{"notes": ['
            '{"quote": "不碰军工板块", "rule": "年底前回避军工", "ttl_days": 21},'
            '{"quote": "明年不加仓科技", "rule": "科技冻结", "ttl_days": 14}'
            "]}"
        )
        now = datetime.now(timezone.utc)
        saved = await self.sched._extract_preference_notes(now)
        self.assertEqual(saved, 1)
        active = await self.db.get_active_preference_notes("2026-11-18")
        self.assertEqual([note["quote_text"] for note in active], ["不碰军工板块"])
        uk_today = now.astimezone(LONDON_TZ).date()
        self.assertEqual(active[0]["expires_on_uk"], (uk_today + timedelta(days=21)).isoformat())
        self.assertEqual(self.llm.calls[0]["usage_label"], "preference_extract")
        self.assertIn("今年年底前不碰军工板块", self.llm.calls[0]["prompt"])

    async def test_no_chat_turns_makes_zero_calls(self) -> None:
        saved = await self.sched._extract_preference_notes(datetime.now(timezone.utc))
        self.assertEqual(saved, 0)
        self.assertEqual(self.llm.calls, [])

    async def test_missing_ds_key_is_noop(self) -> None:
        await self.db.append_chat_turn("年底前不碰军工", "好", cycle_date="2026-11-17")
        no_key = _NoKeyLLM()
        sched = TradingScheduler(self.cfg, _StubT212(), _StubMarketData(), no_key, self.db, self.tg)
        saved = await sched._extract_preference_notes(datetime.now(timezone.utc))
        self.assertEqual(saved, 0)
        self.assertEqual(no_key.calls, [])

    def test_quote_is_verbatim_helper(self) -> None:
        corpus = ["今年年底前不碰军工板块，等财报确认再说"]
        self.assertTrue(_quote_is_verbatim("不碰军工板块", corpus))
        self.assertTrue(_quote_is_verbatim("不碰 军工板块", corpus))
        self.assertFalse(_quote_is_verbatim("年底前回避军工", corpus))
        self.assertFalse(_quote_is_verbatim("军工", corpus))


class TestWeeklyDiagnosis(_DbTestCase):
    async def test_diagnosis_is_sanitized_and_capped(self) -> None:
        self.llm.replies["weekly_quant_diagnosis"] = (
            '{"diagnosis": "**参数建议**：追踪止盈 12% 收紧到 10%。\\n'
            '#胜率结构#：突破类 MFE 高但 T+20 回撤大。\\n下周关注：半导体财报。"}'
        )
        diagnosis = await self.sched._generate_weekly_diagnosis(
            {
                "universe_count": 150,
                "overlap_candidates": [],
                "zombie_candidates": [],
                "threshold_backtest": {"sample_count": 42},
            },
            uk_date="2026-11-14",
        )
        self.assertIn("参数建议", diagnosis)
        self.assertIn("T+20 回撤大", diagnosis)
        self.assertNotIn("**", diagnosis)
        self.assertNotIn("#", diagnosis)
        self.assertLessEqual(len(diagnosis), 450)
        self.assertEqual(self.llm.calls[0]["usage_label"], "weekly_quant_diagnosis")
        self.assertIn("T+20 Alpha", self.llm.calls[0]["prompt"])
        self.assertIn("阈值回测原始数字", self.llm.calls[0]["prompt"])

    async def test_too_short_diagnosis_degrades_to_empty(self) -> None:
        self.llm.replies["weekly_quant_diagnosis"] = '{"diagnosis": "太短"}'
        self.assertEqual(await self.sched._generate_weekly_diagnosis({}, uk_date="2026-11-14"), "")

    async def test_without_ds_key_no_call(self) -> None:
        no_key = _NoKeyLLM()
        sched = TradingScheduler(self.cfg, _StubT212(), _StubMarketData(), no_key, self.db, self.tg)
        self.assertEqual(await sched._generate_weekly_diagnosis({}, uk_date="2026-11-14"), "")
        self.assertEqual(no_key.calls, [])

    async def test_job_wraps_weekly_report(self) -> None:
        class _JobScheduler(TradingScheduler):
            async def _run_weekly_quant_audit_impl(self, now_utc=None):
                return {
                    "universe_count": 150,
                    "overlap_candidates": [],
                    "zombie_candidates": [],
                    "threshold_backtest": {"sample_count": 42},
                    "audit_date": "2026-11-14",
                    "ai_diagnosis": "参数建议：把追踪止盈从 12% 收紧到 10%。",
                }

        sched = _JobScheduler(self.cfg, _StubT212(), _StubMarketData(), self.llm, self.db, self.tg)
        msg = await sched._run_weekly_quant_audit_job(datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc))
        self.assertTrue(msg.startswith("📊 周六策略体检周报"))
        self.assertIn("追踪止盈", msg)
        self.assertIn("回测样本 42", msg)

    async def test_job_falls_back_without_diagnosis(self) -> None:
        class _JobScheduler(TradingScheduler):
            async def _run_weekly_quant_audit_impl(self, now_utc=None):
                return {
                    "universe_count": 150,
                    "overlap_candidates": [],
                    "zombie_candidates": [],
                    "threshold_backtest": {"sample_count": 0},
                    "audit_date": "2026-11-14",
                }

        sched = _JobScheduler(self.cfg, _StubT212(), _StubMarketData(), self.llm, self.db, self.tg)
        msg = await sched._run_weekly_quant_audit_job(datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc))
        # No diagnosis -> silent: nothing is pushed to Telegram, DB audit remains the source of truth.
        self.assertEqual(msg, "")


class TestPreferenceFormatting(unittest.IsolatedAsyncioTestCase):
    def test_block_has_quote_and_expiry(self) -> None:
        block = format_preference_notes_block(
            [
                {
                    "source_cycle_date": "2026-11-17",
                    "quote_text": "年底前不碰军工",
                    "rule_summary": "军工冻结",
                    "expires_on_uk": "2026-12-01",
                }
            ]
        )
        self.assertIn('[2026-11-17 · 原话: "年底前不碰军工"]', block)
        self.assertIn("军工冻结", block)
        self.assertIn("有效至 2026-12-01", block)
        self.assertIn("原话锚点", block)

    def test_empty_or_invalid_notes_render_nothing(self) -> None:
        self.assertEqual(format_preference_notes_block([]), "")
        self.assertEqual(format_preference_notes_block(None), "")
        self.assertEqual(format_preference_notes_block([{"quote_text": "", "rule_summary": "x"}]), "")

    def test_notes_are_capped(self) -> None:
        notes = [
            {
                "source_cycle_date": "2026-11-17",
                "quote_text": f"原话{i}",
                "rule_summary": f"规则{i}",
                "expires_on_uk": "2026-12-01",
            }
            for i in range(10)
        ]
        block = format_preference_notes_block(notes, max_notes=2)
        self.assertIn("原话0", block)
        self.assertIn("原话1", block)
        self.assertNotIn("原话2", block)


class TestStatsDiagnosisDisplay(_DbTestCase):
    async def test_stats_message_includes_ai_diagnosis(self) -> None:
        await self.db.save_quant_audit(
            audit_date="2026-11-14",
            summary_text="体检池 150 只",
            payload={},
            created_at_uk="2026-11-14 12:00",
            ai_diagnosis="参数建议：追踪止盈收紧到 10%",
        )
        bot = AdvisorBot(self.cfg, _StubT212(), _StubMarketData(), self.llm, self.db, self.tg)
        msg = await bot._build_stats_message()
        self.assertIn("【AI 策略诊断】", msg)
        self.assertIn("追踪止盈收紧到 10%", msg)
        self.assertIn("体检池 150 只", msg)


if __name__ == "__main__":
    unittest.main()



