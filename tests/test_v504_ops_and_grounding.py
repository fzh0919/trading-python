"""Unit & integration tests for v5.0.4: OpsNotifier, Gemini fallback, JSON repair, Grounding, /health, and Quick Keyboard."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx

from src.agents.llm_client import GeminiClient
from src.agents.report_agent import generate_morning_report
from src.agents.schemas import MorningReportDecision, ReportActionItem, parse_or_repair_structured_json
from src.agents.tools import MarketToolkit, clear_grounding_cache, run_forced_market_grounding
from src.bot.handlers import BOT_COMMANDS, REMOVE_KEYBOARD_MARKUP, AdvisorBot
from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig, AppSection, EnvSettings
from src.data_providers.market_data import MacroSnapshot, MarketDataService
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, T212Client
from src.healthcheck import check_health, write_heartbeat
from src.ops_notifier import OpsNotifier, send_ops_notify_sync
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
        sys_seen: list[str] = []

        class _GroundingLLM:
            async def generate(self, **kwargs) -> str:
                prompts_seen.append(kwargs["prompt"])
                sys_seen.append(str(kwargs.get("system_instruction") or ""))
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
        self.assertTrue(all("primary English financial" in s and "Simplified Chinese" in s for s in sys_seen))
        self.assertTrue(any("Search Google for key macro catalysts" in p for p in prompts_seen))
        self.assertTrue(any("近期无重大新公告：" in p for p in prompts_seen))

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
                self.assertIn("Google Search 本月用量", health_msg)
                self.assertIn("最近 5 条运维事件", health_msg)
                self.assertNotIn("**", health_msg)
                self.assertNotIn("#", health_msg)

            # UK Bank Holiday annotation in morning report stays <= 1800 chars
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
            self.assertLessEqual(len(rep), 1800)
            self.assertIn("英国 Bank Holiday（LSE 休市", rep)


class TestHomelabOpsNotifyAndHealthcheck(unittest.IsolatedAsyncioTestCase):
    async def test_ops_notify_dual_routing_contract_bearer_and_resilience(self) -> None:
        env = EnvSettings(
            _env_file=None,
            ops_notify_url="http://ops-notify:8000/alert",
            ops_notify_token="secret-bearer-123",
        )
        self.assertEqual(env.ops_notify_url, "http://ops-notify:8000/alert")
        self.assertEqual(env.ops_notify_token, "secret-bearer-123")

        captured: list[tuple[dict[str, str], dict]] = []
        mode = {"status": 200, "raise_net": False}

        async def _handler(request: httpx.Request) -> httpx.Response:
            if mode["raise_net"]:
                raise httpx.ConnectError("ops-notify unreachable", request=request)
            headers = {k.lower(): v for k, v in request.headers.items()}
            body = json.loads(request.content.decode("utf-8"))
            captured.append((headers, body))
            if mode["status"] != 200:
                return httpx.Response(mode["status"], text="gateway error")
            return httpx.Response(200, json={"ok": True})

        tg = _StubTelegram()
        ops = OpsNotifier(
            telegram=tg,
            chat_id="999",
            db=None,
            warn_cooldown_seconds=1800,
            critical_cooldown_seconds=120,
            ops_notify_url=env.ops_notify_url,
            ops_notify_token=env.ops_notify_token,
            transport=httpx.MockTransport(_handler),
        )
        self.assertTrue(ops.ops_notify_enabled)

        # 1) Technical WARN -> goes ONLY to ops-notify with exact contract & Bearer header, NOT trading bot
        ok_tech = await ops.notify(
            severity="WARN",
            source="t212",
            alert_key="t212_rate_limit",
            title="T212 限流退避",
            message="429 rate-limited, retrying",
        )
        self.assertTrue(ok_tech)
        self.assertEqual(len(tg.sent), 0)
        self.assertEqual(len(captured), 1)
        hdrs, payload = captured[-1]
        self.assertEqual(hdrs.get("authorization"), "Bearer secret-bearer-123")
        self.assertEqual(payload["service"], "trading-python")
        self.assertEqual(payload["severity"], "WARN")
        self.assertEqual(payload["alert_key"], "t212_rate_limit")
        self.assertEqual(payload["title"], "T212 限流退避")
        self.assertEqual(payload["cooldown_sec"], 1800)
        self.assertIn("429 rate-limited", payload["message"])

        # 2) Strong business-blocking CRITICAL -> ops-notify CRITICAL + 1 concise trading bot alert
        ok_biz = await ops.notify_business_critical(
            source="t212",
            alert_key="t212_auth_failed:ISA",
            title="T212 ISA 鉴权失效",
            message="HTTP 401 unauthorized",
            bot_message="🚨 T212 ISA 鉴权失效，请更换只读 Key。",
        )
        self.assertTrue(ok_biz)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[-1][1]["severity"], "CRITICAL")
        self.assertEqual(captured[-1][1]["cooldown_sec"], 120)
        self.assertEqual(len(tg.sent), 1)
        self.assertEqual(tg.sent[-1][1], "🚨 T212 ISA 鉴权失效，请更换只读 Key。")

        # 3) ops-notify network failure must NOT raise and buffers for flush_pending() recovery
        mode["raise_net"] = True
        ok_fail = await ops.notify(
            severity="WARN",
            source="market_data",
            alert_key="finnhub_down",
            message="connection refused",
        )
        self.assertFalse(ok_fail)
        self.assertEqual(len(tg.sent), 1)

        mode["raise_net"] = False
        flushed = await ops.flush_pending()
        self.assertTrue(flushed)
        self.assertEqual(len(captured), 3)
        self.assertEqual(captured[-1][1]["alert_key"], "finnhub_down")

        # 4) Startup notice delivers both Telegram (with REMOVE_KEYBOARD_MARKUP) and ops-notify INFO
        ok_start = await ops.send_startup_notice(
            version=APP_VERSION,
            summary_lines=["ok"],
            reply_markup=REMOVE_KEYBOARD_MARKUP,
        )
        self.assertTrue(ok_start)
        self.assertEqual(tg.sent[-1][2], REMOVE_KEYBOARD_MARKUP)
        self.assertEqual(captured[-1][1]["severity"], "INFO")
        self.assertEqual(captured[-1][1]["alert_key"], "startup")

        await ops.close()

    async def test_t212_401_and_scheduler_morning_crash_route_to_business_blocking(self) -> None:
        captured: list[dict] = []

        async def _ops_handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"ok": True})

        async def _t212_401(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad token"})

        tg = _StubTelegram()
        ops = OpsNotifier(
            telegram=tg,
            chat_id="999",
            db=None,
            warn_cooldown_seconds=0,
            critical_cooldown_seconds=0,
            ops_notify_url="http://ops-notify:8000/alert",
            transport=httpx.MockTransport(_ops_handler),
        )

        # T212 401 triggers business-blocking CRITICAL to ops-notify + bot tip
        t212 = T212Client(
            isa_key="isa",
            isa_secret="sec",
            transport=httpx.MockTransport(_t212_401),
            ops_notifier=ops,
            transient_backoff_s=0.0,
        )
        data = await t212.get_portfolio(use_cache=False)
        self.assertFalse(data.isa.ok)
        self.assertTrue(any(p["alert_key"] == "t212_auth_failed:ISA" and p["severity"] == "CRITICAL" for p in captured))
        self.assertTrue(any("鉴权失效" in msg for _, msg, _ in tg.sent))
        await t212.close()

        # Scheduler morning crash sends CRITICAL to ops-notify + bot tip; non-morning crash stays ops-notify only
        tg.sent.clear()
        captured.clear()
        cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="test"))
        sched = TradingScheduler(cfg, t212, _FakeMarketHealthy(), SimpleNamespace(), SimpleNamespace(), tg, ops_notifier=ops)

        async def _crash():
            raise RuntimeError("boom")

        await sched._execute_job(job_name="premarket", chat_id="999", uk_now_str="2026-11-16 14:15", coro=_crash())
        self.assertEqual(len(tg.sent), 0)
        self.assertEqual(captured[-1]["alert_key"], "scheduler_crash:premarket")

        await sched._execute_job(job_name="morning", chat_id="999", uk_now_str="2026-11-16 07:30", coro=_crash())
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("/report", tg.sent[-1][1])
        self.assertEqual(captured[-1]["alert_key"], "scheduler_crash:morning")

        await ops.close()

    async def test_healthcheck_heartbeat_freshness_and_readonly_db_and_sync_crash_notify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hb_path = root / "advisor.heartbeat"
            db_path = root / "advisor.db"

            # 1) Missing heartbeat -> unhealthy
            ok, reason = check_health(hb_path, db_path)
            self.assertFalse(ok)
            self.assertIn("heartbeat missing", reason)

            # 2) Heartbeat written but DB missing -> unhealthy
            write_heartbeat(hb_path, now_ts=1000.0)
            os.utime(hb_path, (1000.0, 1000.0))
            ok, reason = check_health(hb_path, db_path, max_stale_seconds=90.0, now_ts=1010.0)
            self.assertFalse(ok)
            self.assertIn("database missing", reason)

            # 3) Valid initialized WAL DB + fresh heartbeat -> healthy
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("CREATE TABLE IF NOT EXISTS ops_events (id INTEGER PRIMARY KEY)")
                conn.commit()
            finally:
                conn.close()
            ok, reason = check_health(hb_path, db_path, max_stale_seconds=90.0, now_ts=1010.0)
            self.assertTrue(ok, msg=reason)
            self.assertEqual(reason, "ok")

            # 4) Stale heartbeat -> unhealthy
            ok, reason = check_health(hb_path, db_path, max_stale_seconds=90.0, now_ts=1200.0)
            self.assertFalse(ok)
            self.assertIn("heartbeat stale", reason)

        # 5) Sync fatal crash helper never raises even when unreachable
        with patch("src.ops_notifier.httpx.post", side_effect=httpx.ConnectError("down")) as mocked_post:
            self.assertFalse(send_ops_notify_sync("http://ops-notify:8000/alert", message="fatal", token="tok"))
            mocked_post.assert_called_once()
            call_kwargs = mocked_post.call_args.kwargs
            self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer tok")
            self.assertEqual(call_kwargs["json"]["severity"], "CRITICAL")
            self.assertEqual(call_kwargs["json"]["service"], "trading-python")


class TestImp11ToolsAndGrounding(unittest.IsolatedAsyncioTestCase):
    async def test_imp11_grounding_semaphore_and_expired_cache_cleanup(self) -> None:
        from src.agents import tools as tools_mod

        clear_grounding_cache()
        active = 0
        max_active = 0

        class _ConcurrencyLLM:
            async def generate(self, **kwargs) -> str:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1
                return "ok"

        # 21 US symbols -> 7 batches + 1 macro batch = 8 search calls, capped at Semaphore(3)
        syms = [f"SYM{i}" for i in range(21)]
        with patch("src.agents.tools._monotonic", return_value=1000.0):
            out1 = await run_forced_market_grounding(
                llm=_ConcurrencyLLM(),
                search_model="gemini-4-hydrogen",
                scope="report",
                symbols=syms,
                ttl_seconds=60,
            )
        self.assertIn("ok", out1)
        self.assertEqual(max_active, 3)
        self.assertEqual(len(tools_mod._GROUNDING_CACHE), 1)

        # Advance local monotonic clock past TTL=60s; new call should prune expired entry
        with patch("src.agents.tools._monotonic", return_value=1100.0):
            await run_forced_market_grounding(
                llm=_ConcurrencyLLM(),
                search_model="gemini-4-hydrogen",
                scope="pick",
                symbols=["NVDA"],
                ttl_seconds=60,
            )
        self.assertEqual(len(tools_mod._GROUNDING_CACHE), 1)
        self.assertTrue(any(k.startswith("pick:") for k in tools_mod._GROUNDING_CACHE))
        clear_grounding_cache()

    async def test_imp21_grounding_priority_symbols_1to1_and_batch_split(self) -> None:
        clear_grounding_cache()
        prompts_seen: list[str] = []
        active = 0
        max_active = 0

        class _PriorityLLM:
            async def generate(self, **kwargs) -> str:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                prompts_seen.append(str(kwargs.get("prompt") or ""))
                await asyncio.sleep(0.01)
                active -= 1
                return "ok"

        out = await run_forced_market_grounding(
            llm=_PriorityLLM(),
            search_model="gemini-4-hydrogen",
            scope="report",
            symbols=["NVDA", "AAPL", "MSFT", "AMD", "AZN.L"],
            priority_symbols=["NVDA", "AZN.L"],
            ttl_seconds=60,
        )
        self.assertIn("【重点标的1对1深搜 · NVDA】", out)
        self.assertIn("【重点标的1对1深搜 · AZN.L】", out)
        self.assertLessEqual(max_active, 3)

        # Priority symbols must NOT appear in the grouped 3-symbol prompts
        grouped_prompts = [
            p for p in prompts_seen
            if "1-to-1 Google Search deep dive" not in p and "key macro catalysts" not in p
        ]
        self.assertEqual(len(grouped_prompts), 1)  # Only AAPL, MSFT, AMD remain -> 1 US batch
        self.assertIn("AAPL, MSFT, AMD", grouped_prompts[0])
        self.assertNotIn("NVDA", grouped_prompts[0])
        self.assertNotIn("AZN.L", grouped_prompts[0])
        clear_grounding_cache()

    async def test_imp11_inspect_ticker_cyq_weekly_exdiv_intel_lessons_and_budget(self) -> None:
        import json
        import numpy as np
        import pandas as pd
        from src.data_providers.market_data import TickerMarketBundle
        from src.data_providers.t212_client import PositionItem

        dates = pd.bdate_range("2025-10-01", periods=220)
        closes = np.linspace(100.0, 145.0, len(dates)) + np.sin(np.arange(len(dates)) / 4.0) * 2.0
        hist_df = pd.DataFrame(
            {
                "Open": closes * 0.995,
                "High": closes * 1.01,
                "Low": closes * 0.99,
                "Close": closes,
                "Volume": np.full(len(dates), 2_000_000.0),
            },
            index=dates,
        )

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 08:00",
            fetched_at_ms=1_700_000_000_000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA Corporation",
                        inst_currency="USD",
                        isin="US67066G1040",
                        created_at="2026-09-01T10:00:00Z",
                        quantity=10.0,
                        avg_price=120.0,
                        cur_price=145.0,
                        current_value=1150.0,
                        total_cost=950.0,
                        pnl=200.0,
                        fx_impact=0.0,
                        return_pct=21.0,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )

        class _RichMarketData(MarketDataService):
            def __init__(self) -> None:
                self.seen_isin_maps: list[dict[str, str] | None] = []
                self.deep_intel_calls: list[str] = []

            async def get_bundles(self, symbols, include_news=True, include_earnings=True, isin_map=None, **_kwargs):
                self.seen_isin_maps.append(isin_map)
                sym = symbols[0]
                bundle = TickerMarketBundle(
                    symbol=sym,
                    short_name="NVIDIA Corp",
                    long_name="NVIDIA Corporation",
                    currency="USD",
                    quote_type="EQUITY",
                    sector="Technology",
                    isin=(isin_map or {}).get(sym) or "",
                    identity_summary=f"NVIDIA Corporation (NVDA | US67066G1040 | Technology) " + ("X" * 80),
                    current_price=145.0,
                    ex_dividend_date="2026-11-18",
                    days_to_ex_dividend=3,
                    news=[
                        {"date": f"2026-11-1{i}", "title": f"Headline {i} " + ("Y" * 180), "publisher": "Wire"}
                        for i in range(1, 7)
                    ],
                )
                return {sym: hist_df}, {sym: bundle}

            async def get_us_deep_intel(self, symbol: str):
                self.deep_intel_calls.append(symbol)
                return {
                    "symbol": symbol,
                    "recommendations": [{"period": "2026-11-01", "strong_buy": 20, "buy": 15, "hold": 3, "sell": 0, "strong_sell": 0}],
                    "earnings_surprises": [{"period": "2026-09-30", "surprise_pct": 8.4}],
                    "insider_sentiment": {"mspr": 12.5, "change": 4500},
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp11.db")
            await db.init()
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-11-01 09:00",
                        "symbol": "NVDA",
                        "account": "ISA",
                        "action": "BUY",
                        "entry_price": 130.0,
                        "benchmark_symbol": "SPY",
                    }
                ]
            )
            recent_recs = await db.get_recent_recommendations(limit=1)
            rec_id = recent_recs[0]["id"]
            await db.settle_recommendation(
                rec_id,
                benchmark_symbol="SPY",
                settled_at_uk="2026-11-08 21:10",
                raw_return_pct=5.2,
                alpha_pct=2.8,
                reflection_lesson="回踩MA20缩量企稳后买点胜率高。",
            )

            md = _RichMarketData()
            toolkit = MarketToolkit(md, portfolio, db=db)

            # 1) Held symbol passes ISIN, gets CYQ/weekly/ex_div/deep_intel/past_lessons & stays < 4000 chars
            res = await toolkit.execute("inspect_ticker", {"symbol": "NVDA"})
            self.assertEqual(md.seen_isin_maps[0], {"NVDA": "US67066G1040"})
            self.assertEqual(md.deep_intel_calls, ["NVDA"])
            self.assertIsNotNone(res["cyq"])
            self.assertIsNotNone(res["weekly_bullish"])
            self.assertEqual(res["ex_dividend_date"], "2026-11-18")
            self.assertEqual(res["days_to_ex_dividend"], 3)
            self.assertIn("筹码", res["summary"])
            self.assertIn("周多头", res["summary"])
            self.assertIn("机构评级", res["us_deep_intel"])
            self.assertEqual(len(res["past_lessons"]), 1)
            self.assertIn("回踩MA20", res["past_lessons"][0])
            self.assertEqual(len(res["news"]), 3)
            self.assertLess(len(json.dumps(res, ensure_ascii=False, default=str)), 4000)

            # 2) Unheld symbol passes isin_map=None gracefully
            res_unheld = await toolkit.execute("inspect_ticker", {"symbol": "MSFT"})
            self.assertIsNone(md.seen_isin_maps[1])
            self.assertEqual(res_unheld["past_lessons"], [])

            await db.close()

    async def test_imp11_check_portfolio_correlation_uses_get_histories_and_isf_l(self) -> None:
        import numpy as np
        import pandas as pd
        from src.data_providers.t212_client import PositionItem

        dates = pd.bdate_range("2026-06-01", periods=90)
        rng = np.random.default_rng(7)
        spy_c = 500.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, len(dates)))
        isf_c = 800.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.009, len(dates)))
        azn_c = 12000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, len(dates)))
        nvda_c = 130.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.018, len(dates)))

        def _df(c: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(
                {"Open": c * 0.995, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(len(c), 1_000_000.0)},
                index=dates,
            )

        hist_map = {"SPY": _df(spy_c), "ISF.L": _df(isf_c), "AZN.L": _df(azn_c), "NVDA": _df(nvda_c)}

        class _HistOnlyMarketData(MarketDataService):
            def __init__(self) -> None:
                self.called_histories: list[tuple[list[str], str]] = []
                self.called_bundles = False

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                self.called_histories.append((list(symbols), period))
                return {s: hist_map[s] for s in symbols if s in hist_map}

            async def get_bundles(self, *args, **kwargs):
                self.called_bundles = True
                raise AssertionError("check_portfolio_correlation must not call get_bundles")

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 08:00",
            fetched_at_ms=1_700_000_000_000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="AZNl_EQ",
                        ticker="AZN",
                        yahoo_symbol="AZN.L",
                        name="AstraZeneca",
                        inst_currency="GBp",
                        quantity=10.0,
                        avg_price=11500.0,
                        cur_price=12000.0,
                        current_value=1200.0,
                        total_cost=1150.0,
                        pnl=50.0,
                        fx_impact=0.0,
                        return_pct=4.3,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )
        md = _HistOnlyMarketData()
        toolkit = MarketToolkit(md, portfolio)
        res = await toolkit.execute("check_portfolio_correlation", {"symbol": "NVDA"})

        self.assertFalse(md.called_bundles)
        self.assertEqual(len(md.called_histories), 1)
        fetched_syms, period = md.called_histories[0]
        self.assertEqual(period, "1y")
        self.assertIn("SPY", fetched_syms)
        self.assertIn("ISF.L", fetched_syms)
        self.assertEqual(res["candidate"], "NVDA")
        self.assertEqual(res["most_correlated_holding"], "AZN.L")
        self.assertIsNotNone(res["max_correlation_60d"])


class TestImp15HealthDiagnostics(unittest.IsolatedAsyncioTestCase):
    async def test_imp15_health_reports_fallback_stale_unnotified_ops_gateway_and_monthly_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp15_health.db")
            await db.init()

            # Insert one usage row in current UK month and one in previous month (within 30d window)
            await db.log_llm_usage(
                {
                    "call_label": "report_pm",
                    "model": "gemini-4-argon",
                    "prompt_tokens": 1000,
                    "output_tokens": 200,
                    "search_queries": 17,
                }
            )
            await db.log_llm_usage(
                {
                    "call_label": "prev_month_search",
                    "model": "gemini-4-argon",
                    "prompt_tokens": 1000,
                    "output_tokens": 200,
                    "search_queries": 100,
                }
            )
            month_start_uk = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            prev_month_ms = int((month_start_uk - timedelta(days=2)).timestamp() * 1000)
            async with db._connect(write=True) as conn:
                await conn.execute(
                    "UPDATE llm_usage_logs SET created_at_ms = ? WHERE call_label = 'prev_month_search'",
                    (prev_month_ms,),
                )
                await conn.commit()

            # Log an unnotified suppressed ops event
            await db.log_ops_event(
                severity="WARN",
                source="t212:rate_limit",
                alert_key="t212_429",
                message="T212 触发 429 限流退避",
                notified=False,
                suppressed_count=4,
            )

            class _FallbackT212(T212Client):
                async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
                    return PortfolioData(
                        fetched_at_uk="2026-11-15 08:00",
                        fetched_at_ms=1794729600000,
                        is_fallback=True,
                        fallback_source="db_snapshot",
                        fallback_note="T212 HTTP 503，已使用冷启动快照兜底",
                        isa=AccountSnapshot(
                            ok=True,
                            account_label="ISA",
                            total_account_value=8000.0,
                            is_fallback=True,
                            fallback_source="db_snapshot",
                            fallback_reason="HTTP 503",
                            snapshot_fetched_at_uk="2026-11-14 18:00",
                        ),
                        invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=4000.0),
                    )

            class _StaleMarket(MarketDataService):
                async def get_macro_snapshot(self) -> MacroSnapshot:
                    return MacroSnapshot(
                        fetched_at_utc="2026-11-15 08:00 UTC",
                        gbp_usd=1.27,
                        vix=19.5,
                        spy_5d_pct=0.4,
                        is_stale=True,
                    )

            cfg = AppConfig(
                env=EnvSettings(
                    _env_file=None,
                    telegram_chat_id="999",
                    gemini_api_key="test-key",
                    ops_notify_url="http://ops-notify:8000/alert",
                )
            )
            tg = _StubTelegram()
            ops = OpsNotifier(
                telegram=tg,
                chat_id="999",
                db=db,
                ops_notify_url="http://ops-notify:8000/alert",
            )
            bot = AdvisorBot(cfg, _FallbackT212(), _StaleMarket(), object(), db, tg, ops_notifier=ops)
            try:
                msg = await bot._run_health_check()
                self.assertIn("⚠️ T212 ISA [降级缓存]", msg)
                self.assertNotIn("✅ T212 ISA", msg)
                self.assertIn("T212 组合降级告警", msg)
                self.assertIn("Yahoo 行情/宏观: ⚠️ 降级/陈旧缓存 (is_stale)", msg)
                self.assertNotIn("Yahoo 行情/宏观: ✅ 正常", msg)
                self.assertIn("近 24h 未送达/静默抑制告警 (1 条)", msg)
                self.assertIn("抑制合并4次", msg)
                self.assertIn("已启用 ops-notify", msg)
                self.assertIn("Google Search 本月用量: 17 / 5,000", msg)
                self.assertNotIn("117 / 5,000", msg)
                self.assertNotIn("**", msg)
                self.assertNotIn("#", msg)
            finally:
                await ops.close()


class TestImp17OpsRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_imp17_flush_pending_recovers_db_events_via_ops_notify_and_skips_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp17_ops.db")
            await db.init()

            # Pre-restart persisted ops events: INFO, cooldown-suppressed WARN, and 2 real unnotified alerts
            await db.log_ops_event(
                severity="INFO",
                source="lifecycle",
                alert_key="startup",
                message="old startup",
                notified=False,
                suppressed_count=0,
            )
            await db.log_ops_event(
                severity="WARN",
                source="t212",
                alert_key="t212_rate_limit",
                message="suppressed 429 repeat",
                notified=False,
                suppressed_count=3,
            )
            await db.log_ops_event(
                severity="WARN",
                source="market_data",
                alert_key="finnhub_down",
                message="finnhub timeout before power loss",
                notified=False,
                suppressed_count=0,
            )
            await db.log_ops_event(
                severity="CRITICAL",
                source="t212",
                alert_key="t212_auth_failed:ISA",
                message="ISA HTTP 401 before reboot",
                notified=False,
                suppressed_count=0,
            )

            captured: list[dict] = []
            mode = {"raise_net": True}

            async def _handler(request: httpx.Request) -> httpx.Response:
                if mode["raise_net"]:
                    raise httpx.ConnectError("ops-notify still starting", request=request)
                captured.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(200, json={"ok": True})

            tg = _StubTelegram()
            # Fresh OpsNotifier instance simulating process restart (empty in-memory buffers)
            ops = OpsNotifier(
                telegram=tg,
                chat_id="999",
                db=db,
                ops_notify_url="http://ops-notify:8000/alert",
                ops_notify_token="tok-17",
                transport=httpx.MockTransport(_handler),
            )
            try:
                # Failure keeps DB unnotified rows intact for retry
                self.assertFalse(await ops.flush_pending())
                self.assertEqual(len(captured), 0)
                self.assertEqual(len(await db.get_unnotified_ops_events(hours=24)), 3)

                # Recovery pushes merged ops-notify alert ONLY (no trading Bot spam) and marks delivered rows
                mode["raise_net"] = False
                self.assertTrue(await ops.flush_pending())
                self.assertEqual(len(tg.sent), 0)
                self.assertEqual(len(captured), 1)
                payload = captured[0]
                self.assertEqual(payload["severity"], "CRITICAL")
                self.assertEqual(payload["alert_key"], "ops_recovery_flush")
                self.assertEqual(payload["cooldown_sec"], 0)
                self.assertIn("finnhub timeout before power loss", payload["message"])
                self.assertIn("ISA HTTP 401 before reboot", payload["message"])
                self.assertNotIn("suppressed 429 repeat", payload["message"])
                self.assertNotIn("old startup", payload["message"])

                remaining = await db.get_unnotified_ops_events(hours=24)
                self.assertEqual(len(remaining), 1)
                self.assertEqual(remaining[0]["alert_key"], "t212_rate_limit")
                self.assertEqual(remaining[0]["suppressed_count"], 3)

                # Idempotent second flush sends nothing further
                self.assertTrue(await ops.flush_pending())
                self.assertEqual(len(captured), 1)
            finally:
                await ops.close()

    async def test_imp17_flush_pending_telegram_fallback_dedupes_in_memory_and_recovers_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp17_tg.db")
            await db.init()

            await db.log_ops_event(
                severity="WARN",
                source="market_data",
                alert_key="fred_down",
                message="FRED unreachable pre-restart",
                notified=False,
                suppressed_count=0,
            )
            await db.log_ops_event(
                severity="WARN",
                source="t212",
                alert_key="t212_rate_limit",
                message="suppressed warn should not spam",
                notified=False,
                suppressed_count=2,
            )

            tg = _StubTelegram(fail_delivery=True)
            ops = OpsNotifier(
                telegram=tg,
                chat_id="999",
                db=db,
                ops_notify_url="",
            )
            # In-process failure queues into _pending_buffer AND persists notified=0 in DB
            self.assertFalse(
                await ops.notify(
                    severity="CRITICAL",
                    source="scheduler",
                    alert_key="morning_crash",
                    message="morning report failed in process",
                )
            )
            tg.fail_delivery = False

            self.assertTrue(await ops.flush_pending())
            # 1 message from in-memory buffer + 1 message from pre-restart DB recovery (no duplicate for morning_crash!)
            self.assertEqual(len(tg.sent), 2)
            self.assertIn("morning report failed in process", tg.sent[0][1])
            self.assertIn("FRED unreachable pre-restart", tg.sent[1][1])
            self.assertNotIn("morning report failed in process", tg.sent[1][1])
            self.assertNotIn("suppressed warn should not spam", tg.sent[1][1])

            # Only the cooldown-suppressed row remains unnotified in DB
            remaining = await db.get_unnotified_ops_events(hours=24)
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["alert_key"], "t212_rate_limit")

            # Subsequent flush is a no-op
            self.assertTrue(await ops.flush_pending())
            self.assertEqual(len(tg.sent), 2)


class TestImp18MainAndTelegramClient(unittest.IsolatedAsyncioTestCase):
    async def test_imp18_telegram_client_429_retry_after_and_transient_backoff(self) -> None:
        attempts: list[str] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path.rsplit("/", 1)[-1])
            idx = len(attempts)
            if idx == 1:
                return httpx.Response(
                    429,
                    json={
                        "ok": False,
                        "error_code": 429,
                        "description": "Too Many Requests: retry after 2",
                        "parameters": {"retry_after": 0.02},
                    },
                )
            if idx == 2:
                raise httpx.ConnectError("broadband transient drop", request=request)
            if idx == 3:
                return httpx.Response(502, text="<html>Bad Gateway</html>")
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 101}})

        sleep_calls: list[float] = []
        orig_sleep = asyncio.sleep

        async def _spy_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await orig_sleep(0)

        tg = TelegramClient(
            "secret-token-18",
            transport=httpx.MockTransport(_handler),
            max_retries=3,
            base_backoff_s=0.01,
            max_retry_wait_s=1.0,
        )
        try:
            with patch("src.bot.telegram_client.asyncio.sleep", side_effect=_spy_sleep):
                ok = await tg.send_message("999", "hello imp-18")
            self.assertTrue(ok)
            self.assertEqual(len(attempts), 4)
            self.assertEqual(sleep_calls, [0.02, 0.02, 0.04])
        finally:
            await tg.close()

        # Non-retryable 400 fails in 1 shot without backoff
        bad_attempts = 0

        async def _bad_req(request: httpx.Request) -> httpx.Response:
            nonlocal bad_attempts
            bad_attempts += 1
            return httpx.Response(400, json={"ok": False, "error_code": 400, "description": "Bad Request"})

        tg_bad = TelegramClient(
            "secret-token-18",
            transport=httpx.MockTransport(_bad_req),
            max_retries=2,
            base_backoff_s=0.01,
        )
        try:
            self.assertFalse(await tg_bad.send_message("999", "bad"))
            self.assertEqual(bad_attempts, 1)
        finally:
            await tg_bad.close()

        # Persistent network outage exhausts retries and still raises httpx.HTTPError for _poll_loop tracking
        timeout_attempts = 0

        async def _always_timeout(request: httpx.Request) -> httpx.Response:
            nonlocal timeout_attempts
            timeout_attempts += 1
            raise httpx.ReadTimeout("timed out", request=request)

        tg_outage = TelegramClient(
            "secret-token-18",
            transport=httpx.MockTransport(_always_timeout),
            max_retries=1,
            base_backoff_s=0.0,
        )
        try:
            with self.assertRaises(httpx.ReadTimeout):
                await tg_outage.get_updates(offset=1, timeout_s=5)
            self.assertEqual(timeout_attempts, 2)
        finally:
            await tg_outage.close()

        # Exhausting 429 returns structured error payload and respects Retry-After header cap
        rl_attempts = 0
        rl_sleeps: list[float] = []

        async def _always_429(request: httpx.Request) -> httpx.Response:
            nonlocal rl_attempts
            rl_attempts += 1
            return httpx.Response(429, headers={"Retry-After": "999"}, text="rate limited")

        async def _spy_rl_sleep(delay: float) -> None:
            rl_sleeps.append(delay)
            await orig_sleep(0)

        tg_429 = TelegramClient(
            "secret-token-18",
            transport=httpx.MockTransport(_always_429),
            max_retries=1,
            base_backoff_s=0.01,
            max_retry_wait_s=0.25,
        )
        try:
            with patch("src.bot.telegram_client.asyncio.sleep", side_effect=_spy_rl_sleep):
                res = await tg_429._post("sendMessage", {"chat_id": "999", "text": "hi"})
            self.assertFalse(res.get("ok"))
            self.assertEqual(res.get("error_code"), 429)
            self.assertEqual(rl_attempts, 2)
            self.assertEqual(rl_sleeps, [0.25])
        finally:
            await tg_429.close()

    async def test_imp18_run_app_injects_snapshot_loader_flushes_ops_and_closes_all(self) -> None:
        import src.main as main_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "imp18_main.db"
            hb_path = Path(tmpdir) / "advisor.heartbeat"

            pre_db = AdvisorDatabase(db_path)
            await pre_db.init()
            saved_snap = PortfolioData(
                fetched_at_uk="2026-11-15 08:00",
                fetched_at_ms=1794729600000,
                isa=AccountSnapshot(
                    ok=True,
                    account_label="ISA",
                    currency="GBP",
                    total_account_value=5000.0,
                    total_cash=5000.0,
                ),
                invest=AccountSnapshot(
                    ok=True,
                    account_label="Invest",
                    currency="GBP",
                    total_account_value=3000.0,
                    total_cash=3000.0,
                ),
                total_combined_value=8000.0,
                total_combined_cash=8000.0,
            )
            await pre_db.save_snapshot(saved_snap)
            await pre_db.log_ops_event(
                severity="CRITICAL",
                source="t212",
                alert_key="pre_crash_alert",
                message="power loss before notify",
                notified=False,
                suppressed_count=0,
            )

            cfg = AppConfig(
                env=EnvSettings(
                    _env_file=None,
                    telegram_bot_token="fake-tg-token",
                    telegram_chat_id="999",
                    gemini_api_key="fake-gemini",
                    ops_notify_url="http://ops-notify:8000/alert",
                    ops_notify_token="tok-18",
                ),
                app=AppSection(db_path=str(db_path), log_level="WARNING"),
            )

            tg_requests: list[str] = []
            ops_payloads: list[dict] = []
            closed_resources: list[str] = []
            captured: dict[str, object] = {}

            async def _tg_handler(request: httpx.Request) -> httpx.Response:
                method = request.url.path.rsplit("/", 1)[-1]
                tg_requests.append(method)
                if method == "getUpdates":
                    return httpx.Response(200, json={"ok": True, "result": []})
                return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

            async def _ops_handler(request: httpx.Request) -> httpx.Response:
                ops_payloads.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(200, json={"ok": True})

            class _TrackedDB(AdvisorDatabase):
                async def close(self) -> None:
                    closed_resources.append("db")
                    await super().close()

            class _TrackedTG(TelegramClient):
                def __init__(self, bot_token: str, **kwargs) -> None:
                    super().__init__(
                        bot_token,
                        transport=httpx.MockTransport(_tg_handler),
                        base_backoff_s=0.0,
                    )

                async def close(self) -> None:
                    closed_resources.append("tg")
                    await super().close()

            class _TrackedOps(OpsNotifier):
                def __init__(self, **kwargs) -> None:
                    kwargs["transport"] = httpx.MockTransport(_ops_handler)
                    super().__init__(**kwargs)

                async def close(self) -> None:
                    closed_resources.append("ops")
                    await super().close()

            class _TrackedT212(T212Client):
                def __init__(self, **kwargs) -> None:
                    super().__init__(**kwargs)
                    captured["t212"] = self

                async def close(self) -> None:
                    closed_resources.append("t212")
                    await super().close()
                    raise RuntimeError("simulated t212 close hiccup")

            class _TrackedMarketData(MarketDataService):
                async def close(self) -> None:
                    closed_resources.append("market_data")
                    await super().close()

            class _FastScheduler(TradingScheduler):
                async def run_forever(self) -> None:
                    stop_event.set()
                    await asyncio.sleep(0)

            stop_event = asyncio.Event()

            with (
                patch.object(main_mod, "load_config", return_value=cfg),
                patch.object(main_mod, "AdvisorDatabase", _TrackedDB),
                patch.object(main_mod, "TelegramClient", _TrackedTG),
                patch.object(main_mod, "OpsNotifier", _TrackedOps),
                patch.object(main_mod, "T212Client", _TrackedT212),
                patch.object(main_mod, "MarketDataService", _TrackedMarketData),
                patch.object(main_mod, "TradingScheduler", _FastScheduler),
                patch.object(main_mod, "write_heartbeat", lambda: write_heartbeat(hb_path)),
            ):
                await main_mod.run_app(stop_event=stop_event)

            # Verify snapshot_loader injection works end-to-end against real DB
            t212_inst = captured["t212"]
            self.assertIsNotNone(t212_inst._snapshot_loader)
            loaded = await t212_inst._snapshot_loader()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.total_combined_value, 8000.0)

            # Verify startup flush_pending pushed the pre-restart CRITICAL event via ops-notify
            recovery_msgs = [p for p in ops_payloads if p.get("alert_key") == "ops_recovery_flush"]
            self.assertEqual(len(recovery_msgs), 1)
            self.assertIn("power loss before notify", recovery_msgs[0]["message"])
            self.assertEqual(await pre_db.get_unnotified_ops_events(hours=24), [])

            # Verify all resources closed even when t212.close() raised
            self.assertEqual(closed_resources, ["t212", "market_data", "ops", "tg", "db"])

    async def test_sqlite_grounding_cache_reuse_and_toolkit_controls(self) -> None:
        clear_grounding_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "tools_cache.db")
            await db.init()

            search_calls: list[str] = []

            class _CountingSearchLLM:
                async def generate(self, **kwargs) -> str:
                    search_calls.append(str(kwargs.get("prompt") or ""))
                    return f"NVDA 最新催化摘要 #{len(search_calls)}"

            llm = _CountingSearchLLM()
            per_sym: dict[str, str] = {}
            out1 = await run_forced_market_grounding(
                llm=llm,
                search_model="gemini-3.5-flash-lite",
                scope="report",
                symbols=["NVDA"],
                priority_symbols=["NVDA"],
                ttl_seconds=900,
                db=db,
                persist_ttl_seconds=10800,
                per_symbol_out=per_sym,
            )
            self.assertEqual(len(search_calls), 2)  # 1 macro + 1 priority NVDA
            self.assertIn("NVDA 最新催化摘要", per_sym["NVDA"])

            # Simulate container restart (clear in-memory cache) -> DB cache serves 0-LLM hits
            clear_grounding_cache()
            out2 = await run_forced_market_grounding(
                llm=llm,
                search_model="gemini-3.5-flash-lite",
                scope="report",
                symbols=["NVDA"],
                priority_symbols=["NVDA"],
                ttl_seconds=900,
                db=db,
                persist_ttl_seconds=10800,
            )
            self.assertEqual(len(search_calls), 2)
            self.assertEqual(out1, out2)

            # MarketToolkit with allow_search_tool=False omits search_market_news
            md_mock = SimpleNamespace()
            pf_mock = PortfolioData(
                fetched_at_uk="2026-11-16 07:30",
                fetched_at_ms=1,
                isa=AccountSnapshot(ok=True, account_label="ISA"),
                invest=AccountSnapshot(ok=True, account_label="Invest"),
            )
            tk_no_search = MarketToolkit(md_mock, pf_mock, llm=llm, db=db, allow_search_tool=False)
            self.assertNotIn("search_market_news", [d.name for d in tk_no_search.declarations()])

            # MarketToolkit search_market_news reuses DB symbol grounding cache with 0 LLM calls
            tk_with_search = MarketToolkit(md_mock, pf_mock, llm=llm, db=db, allow_search_tool=True)
            decls = tk_with_search.declarations()
            self.assertTrue(all(d.description.isascii() for d in decls))
            tool_res = await tk_with_search.execute("search_market_news", {"query": "NVDA earnings"})
            self.assertTrue(tool_res.get("cached"))
            self.assertIn("NVDA 最新催化摘要", tool_res.get("summary", ""))
            self.assertEqual(len(search_calls), 2)

            # Live search_market_news prompt uses English search instructions + Chinese output constraint
            live_res = await tk_with_search.execute("search_market_news", {"query": "AZN.L RNS trading update"})
            self.assertFalse(live_res.get("cached"))
            self.assertEqual(len(search_calls), 3)
            self.assertIn("Use Google Search", search_calls[-1])
            self.assertIn("Simplified Chinese", search_calls[-1])

            # bypass_cache=True forces fresh grounding search
            await run_forced_market_grounding(
                llm=llm,
                search_model="gemini-3.5-flash-lite",
                scope="us",
                symbols=["NVDA"],
                ttl_seconds=900,
                db=db,
                persist_ttl_seconds=10800,
                bypass_cache=True,
            )
            self.assertEqual(len(search_calls), 4)
        clear_grounding_cache()


if __name__ == "__main__":
    unittest.main()
