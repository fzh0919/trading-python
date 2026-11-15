"""Unit & integration tests for v5.0.4: OpsNotifier, Gemini fallback, JSON repair, Grounding, /health, and Quick Keyboard."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.llm_client import GeminiClient
from src.agents.report_agent import generate_morning_report
from src.agents.schemas import MorningReportDecision, ReportActionItem, parse_or_repair_structured_json
from src.agents.tools import MarketToolkit, clear_grounding_cache, run_forced_market_grounding
from src.bot.handlers import BOT_COMMANDS, REMOVE_KEYBOARD_MARKUP, AdvisorBot
from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, EnvSettings
from src.data_providers.market_data import MacroSnapshot, MarketDataService
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, T212Client
from src.ops_notifier import OpsNotifier
from src.storage.db import AdvisorDatabase, get_trading_cycle_id
from src.version import APP_VERSION


class _StubTelegram(TelegramClient):
    def __init__(self, *, fail_delivery: bool = False) -> None:
        super().__init__(bot_token="fake-token")
        self.fail_delivery = fail_delivery
        self.sent: list[tuple[str, str, dict | None]] = []
        self.statuses: list[str] = []

    async def send_typing(self, chat_id) -> None:
        pass

    async def send_status(self, chat_id, text: str) -> int:
        self.statuses.append(text)
        return len(self.statuses)

    async def edit_status(self, chat_id, message_id, text: str) -> None:
        self.statuses.append(text)

    async def send_message(self, chat_id, text: str, reply_markup=None) -> bool:
        if self.fail_delivery:
            return False
        self.sent.append((str(chat_id), text, reply_markup))
        return True


class _FakeT212Healthy(T212Client):
    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return PortfolioData(
            fetched_at_uk="2026-11-16 09:00",
            fetched_at_ms=1794819600000,
            total_combined_value=12000.0,
            total_combined_cash=3000.0,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=8000.0, total_cash=2000.0, pending_orders_supported=True, order_history_supported=True),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=4000.0, total_cash=1000.0, pending_orders_supported=True, order_history_supported=False),
        )


class _FakeMarketHealthy(MarketDataService):
    async def get_macro_snapshot(self) -> MacroSnapshot:
        return MacroSnapshot(
            fetched_at_utc="2026-11-16 09:00 UTC",
            gbp_usd=1.28,
            vix=14.8,
            spy_5d_pct=0.9,
        )


class TestOpsNotifierAndLLMResilience(unittest.IsolatedAsyncioTestCase):
    async def test_ops_cooldown_suppression_and_offline_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "ops.db")
            await db.init()

            now = [100.0]
            tg = _StubTelegram(fail_delivery=False)
            ops = OpsNotifier(
                telegram=tg,
                chat_id="999",
                db=db,
                warn_cooldown_seconds=1800,
                critical_cooldown_seconds=120,
                clock=lambda: now[0],
            )

            # Startup notice delivers version and remove_keyboard cleanup
            ok = await ops.send_startup_notice(
                version=APP_VERSION,
                summary_lines=["test"],
                reply_markup=REMOVE_KEYBOARD_MARKUP,
            )
            self.assertTrue(ok)
            self.assertIn(APP_VERSION, tg.sent[-1][1])
            self.assertEqual(tg.sent[-1][2], REMOVE_KEYBOARD_MARKUP)

            # First WARN delivers; second within cooldown is suppressed; third after cooldown merges count
            self.assertTrue(await ops.notify(severity="WARN", source="finnhub", alert_key="finnhub_err", message="HTTP 502"))
            self.assertFalse(await ops.notify(severity="WARN", source="finnhub", alert_key="finnhub_err", message="HTTP 502 again"))
            now[0] += 1900.0
            self.assertTrue(await ops.notify(severity="WARN", source="finnhub", alert_key="finnhub_err", message="HTTP 503"))
            self.assertIn("已合并同类提示 1 次", tg.sent[-1][1])

            # Offline failure buffers alert and flushes when Telegram recovers
            tg.fail_delivery = True
            self.assertFalse(await ops.notify(severity="CRITICAL", source="t212", alert_key="t212_down", message="API timeout"))
            tg.fail_delivery = False
            self.assertTrue(await ops.flush_pending())
            self.assertIn("[系统恢复]", tg.sent[-1][1])
            self.assertIn("API timeout", tg.sent[-1][1])

            events = await db.get_recent_ops_events(limit=10)
            self.assertGreaterEqual(len(events), 4)

    async def test_gemini_transient_retry_fallback_and_total_outage_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "llm_resilience.db")
            await db.init()
            tg = _StubTelegram()
            ops = OpsNotifier(telegram=tg, chat_id="999", db=db, warn_cooldown_seconds=0, critical_cooldown_seconds=0)

            client = GeminiClient(
                api_key="dummy",
                usage_recorder=db.log_llm_usage,
                ops_notifier=ops,
                fallback_model="gemini-4-hydrogen",
                fallback_thinking_level="HIGH",
                max_transient_retries=1,
                retry_base_delay_s=0.0,
            )

            # 1) Primary argon fails persistently -> falls back to hydrogen HIGH + WARN ops alert + tracks search_queries
            call_models: list[str] = []

            async def _argon_fails_hydrogen_succeeds(*, model, contents, config):
                call_models.append(model)
                if model == "gemini-4-argon":
                    raise RuntimeError("503 Service Unavailable")
                return SimpleNamespace(
                    text="备援模型成功响应",
                    function_calls=[],
                    candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(web_search_queries=["q1", "q2"]))],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        cached_content_token_count=0,
                        candidates_token_count=50,
                        thoughts_token_count=20,
                        total_token_count=170,
                    ),
                )

            client._client = SimpleNamespace(  # type: ignore[assignment]
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=_argon_fails_hydrogen_succeeds))
            )
            res = await client.generate(model="gemini-4-argon", prompt="hi", thinking_level="HIGH", usage_label="report_pm")
            self.assertEqual(res, "备援模型成功响应")
            self.assertEqual(call_models, ["gemini-4-argon", "gemini-4-argon", "gemini-4-hydrogen"])
            self.assertTrue(any("[运维告警 | WARN]" in m[1] and "已自动降级" in m[1] for m in tg.sent))

            usage = await db.get_usage_summary(days=30)
            self.assertEqual(usage["today"]["search_queries"], 2)

            # 2) Both primary & fallback fail -> user sees unavailable + CRITICAL ops alert
            async def _all_fail(*, model, contents, config):
                raise RuntimeError("429 RESOURCE_EXHAUSTED")

            client._client = SimpleNamespace(  # type: ignore[assignment]
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=_all_fail))
            )
            outage_reply = await client.generate(model="gemini-4-argon", prompt="hi", usage_label="us_sentinel")
            self.assertIn("不可用", outage_reply)
            self.assertTrue(any("[运维告警 | CRITICAL]" in m[1] for m in tg.sent))

    async def test_json_auto_repair_and_failure_alerts(self) -> None:
        tg = _StubTelegram()
        ops = OpsNotifier(telegram=tg, chat_id="999", warn_cooldown_seconds=0, critical_cooldown_seconds=0)

        class _RepairLLM:
            def __init__(self, repaired_text: str) -> None:
                self.repaired_text = repaired_text
                self.calls = 0

            async def generate(self, **kwargs) -> str:
                self.calls += 1
                return self.repaired_text

        valid_json = MorningReportDecision(
            headline="修复成功测试",
            top_actions=[ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="N/A", reason="观望")],
            daily_vetoes=["保持现金"],
            macro_and_cash="利率4%",
            account_diagnosis="均衡",
            hold_summary="持有",
            trade_review="克制",
        ).model_dump_json()

        # Repair succeeds -> WARN alert
        llm_ok = _RepairLLM(valid_json)
        dec, _ = await parse_or_repair_structured_json(
            "broken non-json output",
            MorningReportDecision,
            llm=llm_ok,
            repair_model="gemini-4-hydrogen",
            ops_notifier=ops,
            alert_source="report_pm",
        )
        self.assertIsNotNone(dec)
        self.assertEqual(llm_ok.calls, 1)
        self.assertIn("[运维告警 | WARN]", tg.sent[-1][1])

        # Repair fails -> CRITICAL alert
        llm_bad = _RepairLLM("still broken")
        dec2, _ = await parse_or_repair_structured_json(
            "broken",
            MorningReportDecision,
            llm=llm_bad,
            repair_model="gemini-4-hydrogen",
            ops_notifier=ops,
            alert_source="report_pm",
        )
        self.assertIsNone(dec2)
        self.assertIn("[运维告警 | CRITICAL]", tg.sent[-1][1])

        # AI outage text skips repair
        llm_skip = _RepairLLM(valid_json)
        dec3, _ = await parse_or_repair_structured_json(
            "⚠️ Gemini AI 服务当前全部不可用",
            MorningReportDecision,
            llm=llm_skip,
            repair_model="gemini-4-hydrogen",
            ops_notifier=ops,
            alert_source="report_pm",
        )
        self.assertIsNone(dec3)
        self.assertEqual(llm_skip.calls, 0)


class TestForcedGroundingSchedulerHealthAndKeyboard(unittest.IsolatedAsyncioTestCase):
    async def test_forced_grounding_groups_uk_eu_us_and_caches(self) -> None:
        clear_grounding_cache()
        tg = _StubTelegram()
        ops = OpsNotifier(telegram=tg, chat_id="999", warn_cooldown_seconds=0)
        prompts_seen: list[str] = []

        class _GroundingLLM:
            async def generate(self, **kwargs) -> str:
                prompts_seen.append(kwargs["prompt"])
                self_search = kwargs.get("use_google_search")
                assert self_search is True
                return f"搜索摘要({len(prompts_seen)})"

        brief1 = await run_forced_market_grounding(
            llm=_GroundingLLM(),
            search_model="gemini-4-hydrogen",
            scope="report",
            symbols=["AZN.L", "VWRP.L", "ASML.AS", "NVDA"],
            ttl_seconds=900,
            ops_notifier=ops,
        )
        self.assertIn("【宏观与英/欧/美今日突发主线】", brief1)
        self.assertIn("【英股与 LSE ETF 联网新闻】", brief1)
        self.assertIn("【欧股联网新闻】", brief1)
        self.assertIn("【美股与全球 ETF 联网新闻】", brief1)
        self.assertEqual(len(prompts_seen), 4)

        # Cached second call hits memory cache without triggering new LLM calls
        brief2 = await run_forced_market_grounding(
            llm=_GroundingLLM(),
            search_model="gemini-4-hydrogen",
            scope="report",
            symbols=["NVDA", "ASML.AS", "VWRP.L", "AZN.L"],
            ttl_seconds=900,
            ops_notifier=ops,
        )
        self.assertEqual(brief1, brief2)
        self.assertEqual(len(prompts_seen), 4)

        # Verify batch_size=3 splits 4 US symbols into 2 search batches (#1 and #2)
        clear_grounding_cache()
        prompts_seen.clear()
        brief3 = await run_forced_market_grounding(
            llm=_GroundingLLM(),
            search_model="gemini-4-hydrogen",
            scope="report",
            symbols=["NVDA", "MSFT", "AAPL", "AMD"],
            ttl_seconds=900,
            ops_notifier=ops,
        )
        self.assertIn("【美股与全球 ETF 联网新闻 #1】", brief3)
        self.assertIn("【美股与全球 ETF 联网新闻 #2】", brief3)
        self.assertEqual(len(prompts_seen), 3)

    async def test_scheduler_crash_alert_quick_keyboard_help_and_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "e2e_v504.db")
            await db.init()

            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="test-key"))
            tg = _StubTelegram()
            ops = OpsNotifier(telegram=tg, chat_id="999", db=db, warn_cooldown_seconds=0, critical_cooldown_seconds=0)
            t212 = _FakeT212Healthy()
            md = _FakeMarketHealthy()

            class _BoomLLM:
                async def generate(self, **kwargs) -> str:
                    raise RuntimeError("Unexpected boom")

            sched = TradingScheduler(cfg, t212, md, _BoomLLM(), db, tg, ops_notifier=ops)
            # Monday 09:00 UTC -> triggers morning report which crashes on grounding/llm or executes
            # Force _execute_job directly with a raising coroutine
            await sched._execute_job(
                job_name="morning",
                chat_id="999",
                uk_now_str="2026-11-16 09:00",
                coro=asyncio.sleep(0, result="⚠️ Gemini 不可用"),
            )
            self.assertEqual(sched.last_job_runs["morning"]["status"], "FAILED")
            self.assertTrue(any("[运维告警 | CRITICAL]" in m[1] for m in tg.sent))

            bot = AdvisorBot(cfg, t212, md, _BoomLLM(), db, tg, ops_notifier=ops, scheduler=sched)

            # Left "/" menu keeps only the 6 core commands; ops commands live in /help
            self.assertEqual([c["command"] for c in BOT_COMMANDS], ["pos", "report", "pick", "us", "stats", "help"])

            # /help splits core vs ops commands and removes persistent bottom keyboard
            await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/help"}}, wait=True)
            help_msg, help_markup = tg.sent[-1][1], tg.sent[-1][2]
            self.assertIn("【核心投研指令", help_msg)
            self.assertIn("【运维与系统控制指令】", help_msg)
            self.assertIn("/health", help_msg)
            self.assertEqual(help_markup, REMOVE_KEYBOARD_MARKUP)

            # Trading cycle resets at 07:30 UK aligned with morning_report_hour/minute
            self.assertEqual(
                get_trading_cycle_id(datetime(2026, 11, 16, 7, 29, tzinfo=timezone.utc), reset_hour=7, reset_minute=30),
                "2026-11-15",
            )
            self.assertEqual(
                get_trading_cycle_id(datetime(2026, 11, 16, 7, 30, tzinfo=timezone.utc), reset_hour=7, reset_minute=30),
                "2026-11-16",
            )

            # /health (and typo /heath) returns diagnostics without # or **
            for cmd in ("/health", "/heath"):
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": cmd}}, wait=True)
                health_msg = tg.sent[-1][1]
                self.assertIn(f"v{APP_VERSION}", health_msg)
                self.assertIn("T212 ISA", health_msg)
                self.assertIn("Google Search 30日用量", health_msg)
                self.assertIn("最近 5 条运维事件", health_msg)
                self.assertNotIn("**", health_msg)
                self.assertNotIn("#", health_msg)

            # UK Bank Holiday annotation in morning report stays <= 1500 chars
            class _HolidayReportLLM:
                async def generate(self, **kwargs) -> str:
                    if kwargs.get("response_schema") is MorningReportDecision:
                        return MorningReportDecision(
                            headline="英国假日LSE休市，聚焦美股与现金",
                            top_actions=[
                                ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="N/A", reason="LSE休市观望")
                            ],
                            daily_vetoes=["保持现金"],
                            macro_and_cash="宏观" * 30,
                            account_diagnosis="诊断" * 40,
                            flagged_positions=[f"SYM{i}: " + ("摘要" * 15) for i in range(4)],
                            hold_summary="其余持有",
                            trade_review="操作复盘" * 10,
                        ).model_dump_json()
                    return "正常分析摘要"

            with patch("src.agents.report_agent.is_uk_bank_holiday", return_value=True):
                rep = await generate_morning_report(cfg, t212, md, _HolidayReportLLM(), db, ops_notifier=ops)
            self.assertLessEqual(len(rep), 1500)
            self.assertIn("英国 Bank Holiday（LSE 休市", rep)


if __name__ == "__main__":
    unittest.main()
