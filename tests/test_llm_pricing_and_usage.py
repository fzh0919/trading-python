"""Tests for config contract, mtime pool hot-reload, thinking-level normalization, and token/search telemetry."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from google.genai import types
from pydantic import ValidationError

from src.agents.chat_agent import _format_chat_history_compact
from src.agents.llm_client import (
    GeminiClient,
    _compact_tool_result,
    _function_declarations_to_openai_tools,
    normalize_thinking_level,
)
from src.agents.schemas import MorningReportDecision, parse_structured_json
from src.config import EnvSettings, ModelsSection, StrategySection, load_config
from src.ops_notifier import OpsNotifier
from src.storage.db import AdvisorDatabase


class TestConfigAndThinkingNormalization(unittest.TestCase):
    def test_thinking_normalization(self) -> None:
        self.assertEqual(normalize_thinking_level("gemini-4-argon", "MINIMAL"), "LOW")
        self.assertEqual(normalize_thinking_level("gemini-4-argon", "XHIGH"), "HIGH")
        self.assertEqual(normalize_thinking_level("gemini-4-carbon", "MINIMAL"), "LOW")
        self.assertEqual(normalize_thinking_level("gemini-4-carbon", "XHIGH"), "HIGH")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "MINIMAL"), "LOW")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "xhigh"), "XHIGH")
        self.assertEqual(normalize_thinking_level("gemini-3.5-flash-lite", "MINIMAL"), "MINIMAL")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "INVALID"), "")

    def test_models_and_strategy_defaults_and_legacy_migration(self) -> None:
        cfg = load_config("config.yaml", env_file=None)
        self.assertEqual(cfg.models.report_analyst_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.report_analyst_thinking_level, "HIGH")
        self.assertEqual(cfg.models.report_auditor_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.report_auditor_thinking_level, "HIGH")
        self.assertEqual(cfg.models.report_model, "gemini-4-carbon")
        self.assertEqual(cfg.models.report_thinking_level, "HIGH")
        self.assertEqual(cfg.models.us_sentinel_model, "deepseek-flash")
        self.assertEqual(cfg.models.us_thinking_level, "HIGH")
        self.assertEqual(cfg.models.us_anomaly_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.us_anomaly_thinking_level, "HIGH")
        self.assertEqual(cfg.models.pick_researcher_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.pick_researcher_thinking_level, "HIGH")
        self.assertEqual(cfg.models.pick_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.pick_thinking_level, "HIGH")
        self.assertEqual(cfg.models.chat_model, "deepseek-flash")
        self.assertEqual(cfg.models.chat_thinking_level, "HIGH")
        self.assertEqual(cfg.models.chat_deep_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.chat_deep_thinking_level, "HIGH")
        self.assertEqual(cfg.models.fallback_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.fallback_thinking_level, "HIGH")
        self.assertEqual(cfg.models.gemini_fallback_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.gemini_fallback_thinking_level, "MEDIUM")
        self.assertEqual(cfg.models.deepseek_fallback_model, "deepseek-flash")
        self.assertEqual(cfg.models.deepseek_fallback_thinking_level, "HIGH")
        self.assertEqual(cfg.models.report_priority_search_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.report_priority_search_thinking_level, "LOW")
        self.assertEqual(cfg.models.report_search_model, "gemini-3.5-flash-lite")
        self.assertEqual(cfg.models.search_model, "gemini-3.5-flash-lite")
        self.assertTrue(cfg.models.enable_forced_search_grounding)
        self.assertEqual(cfg.models.report_max_tool_rounds, 2)
        self.assertEqual(cfg.app.grounding_cache_ttl_seconds, 10800)
        self.assertEqual(cfg.models.google_search_free_quota_monthly, 5000)
        self.assertAlmostEqual(cfg.models.google_search_warn_pct, 85.0)

        self.assertAlmostEqual(cfg.strategy.us_anomaly_move_pct, 1.8)
        self.assertAlmostEqual(cfg.strategy.us_anomaly_vix_change_pct, 4.0)
        self.assertEqual(cfg.strategy.settlement_holding_days, 5)

        legacy = ModelsSection(premarket_model="gemini-3.8-flash", fast_thinking_level="LOW")
        self.assertEqual(legacy.report_analyst_model, "gemini-3.8-flash")
        self.assertEqual(legacy.report_analyst_thinking_level, "LOW")
        self.assertEqual(legacy.us_thinking_level, "LOW")
        self.assertEqual(legacy.chat_thinking_level, "LOW")

    def test_config_validation_and_symbol_normalization(self) -> None:
        with self.assertRaises(ValidationError):
            StrategySection(us_anomaly_move_pct=0.0)
        with self.assertRaises(ValidationError):
            StrategySection(settlement_holding_days=0)
        with self.assertRaises(ValidationError):
            ModelsSection(google_search_warn_pct=120.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "config.yaml"
            ypath.write_text(
                "watchlist:\n  - ' nvda '\n  - 'NVDA'\n  - ''\n  - 'aapl'\n"
                "universe:\n  isa_long_term: [' vwrp.l ', 'VWRP.L']\n  invest_tactical: ['tsla']\n",
                encoding="utf-8",
            )
            cfg = load_config(ypath, env_file=None)
            self.assertEqual(cfg.watchlist, ["NVDA", "AAPL"])
            self.assertEqual(cfg.universe.isa_long_term, ["VWRP.L"])
            self.assertEqual(cfg.universe.invest_tactical, ["TSLA"])

    def test_mtime_hot_reload_updates_pools_and_guards_broken_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "config.yaml"
            ypath.write_text(
                "watchlist:\n  - 'MSFT'\n"
                "universe:\n  isa_long_term: ['VWRP.L']\n  invest_tactical: ['NVDA']\n",
                encoding="utf-8",
            )
            cfg = load_config(ypath, env_file=None)
            self.assertFalse(cfg.reload_pools_if_changed())

            # Modify file and advance mtime explicitly to avoid same-tick filesystem timestamp granularity
            old_ns = ypath.stat().st_mtime_ns
            ypath.write_text(
                "watchlist:\n  - 'tsm'\n  - 'ASML.AS'\n"
                "universe:\n  isa_long_term: ['VWRP.L', 'VUAG.L']\n  invest_tactical: ['META']\n",
                encoding="utf-8",
            )
            os.utime(ypath, ns=(old_ns + 1_000_000, old_ns + 1_000_000))

            self.assertTrue(cfg.reload_pools_if_changed())
            self.assertEqual(cfg.watchlist, ["TSM", "ASML.AS"])
            self.assertEqual(cfg.universe.isa_long_term, ["VWRP.L", "VUAG.L"])
            self.assertEqual(cfg.universe.invest_tactical, ["META"])
            self.assertFalse(cfg.reload_pools_if_changed())

            # Write invalid YAML; reload must return False and preserve existing pools
            old_ns2 = ypath.stat().st_mtime_ns
            ypath.write_text("watchlist: [unclosed", encoding="utf-8")
            os.utime(ypath, ns=(old_ns2 + 1_000_000, old_ns2 + 1_000_000))
            self.assertFalse(cfg.reload_pools_if_changed())
            self.assertEqual(cfg.watchlist, ["TSM", "ASML.AS"])
            self.assertEqual(cfg.universe.invest_tactical, ["META"])

    def test_chat_history_compaction(self) -> None:
        turns = [(f"问题{i}", f"回答{i} " + ("内容" * 200)) for i in range(7)]
        compact = _format_chat_history_compact(turns, recent_full_turns=3)
        self.assertIn("[早期] 问: 问题0", compact)
        self.assertIn("[早期] 问: 问题3", compact)
        self.assertNotIn("[早期] 问: 问题4", compact)
        self.assertIn("👤 用户: 问题6", compact)


class TestGeminiClientTelemetryAndUsageDb(unittest.IsolatedAsyncioTestCase):
    async def test_client_accumulates_multiround_usage_and_db_aggregates(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                db = AdvisorDatabase(Path(tmpdir) / "usage.db")
                await db.init()

                client = GeminiClient(api_key="dummy-key", usage_recorder=db.log_llm_usage)
                calls_made: list[dict] = []

                async def _fake_generate_content(*, model, contents, config):
                    calls_made.append({"model": model, "config": config})
                    if len(calls_made) == 1:
                        return SimpleNamespace(
                            text="",
                            function_calls=[SimpleNamespace(name="inspect_ticker", args={"symbol": "NVDA"})],
                            candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
                            usage_metadata=SimpleNamespace(
                                prompt_token_count=4500,
                                cached_content_token_count=0,
                                candidates_token_count=40,
                                thoughts_token_count=120,
                                tool_use_prompt_token_count=0,
                                total_token_count=4660,
                            ),
                        )
                    return SimpleNamespace(
                        text="最终结论：观察 NVDA。",
                        function_calls=[],
                        candidates=[
                            SimpleNamespace(
                                grounding_metadata=SimpleNamespace(web_search_queries=["NVDA earnings"])
                            )
                        ],
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=5200,
                            cached_content_token_count=4000,
                            candidates_token_count=180,
                            thoughts_token_count=300,
                            tool_use_prompt_token_count=0,
                            total_token_count=5680,
                        ),
                    )

                # Patch async generate_content on the internal SDK client
                client._client = SimpleNamespace(  # type: ignore[assignment]
                    aio=SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate_content))
                )

                async def _fake_tool(name: str, args: dict) -> dict:
                    return {"symbol": args.get("symbol"), "price": 135.0}

                reply = await client.generate(
                    model="gemini-4-hydrogen",
                    prompt="检查 NVDA",
                    system_instruction="你是投顾",
                    thinking_level="XHIGH",
                    tool_executor=_fake_tool,
                    usage_label="pick_researcher",
                )
                self.assertIn("观察 NVDA", reply)
                self.assertEqual(len(calls_made), 2)

                summary = await db.get_usage_summary(days=30, now_utc=datetime.now(timezone.utc))
                self.assertEqual(summary["today"]["calls"], 1)
                self.assertEqual(summary["today"]["rounds"], 2)
                self.assertEqual(summary["today"]["prompt_tokens"], 9700)
                self.assertEqual(summary["today"]["cached_tokens"], 4000)
                self.assertEqual(summary["today"]["output_tokens"], 220)
                self.assertEqual(summary["today"]["thought_tokens"], 420)
                self.assertEqual(summary["today"]["search_queries"], 1)
                self.assertEqual(summary["today"]["cost_usd"], 0.0)
                self.assertEqual(summary["by_label_7d"][0]["label"], "pick_researcher")
                self.assertEqual(summary["by_model_nd"][0]["model"], "gemini-4-hydrogen")

        await asyncio.wait_for(_run(), timeout=10.0)

    async def test_hard_timeout_fails_over_without_retrying_primary(self) -> None:
        class _RecordingOps:
            def __init__(self) -> None:
                self.events: list[dict] = []

            async def notify(self, *, severity: str, source: str, alert_key: str, message: str) -> None:
                self.events.append(
                    {"severity": severity, "source": source, "alert_key": alert_key, "message": message}
                )

        ops = _RecordingOps()
        usage_records: list[dict] = []

        async def _record(rec: dict) -> None:
            usage_records.append(rec)

        client = GeminiClient(
            api_key="dummy-key",
            usage_recorder=_record,
            ops_notifier=ops,  # type: ignore[arg-type]
            fallback_model="gemini-4-hydrogen",
            fallback_thinking_level="HIGH",
            max_transient_retries=1,
            retry_base_delay_s=0.0,
            round_timeout_seconds=0.05,
        )
        attempts: list[str] = []

        async def _slow_primary_fast_fallback(*, model, contents, config):
            attempts.append(model)
            if model == "gemini-4-argon":
                await asyncio.sleep(0.25)
            return SimpleNamespace(
                text="超时后备援成功",
                function_calls=[],
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=50,
                    cached_content_token_count=0,
                    candidates_token_count=20,
                    thoughts_token_count=5,
                    total_token_count=75,
                ),
            )

        client._client = SimpleNamespace(  # type: ignore[assignment]
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=_slow_primary_fast_fallback))
        )
        reply = await client.generate(model="gemini-4-argon", prompt="hi", usage_label="report_pm")
        self.assertEqual(reply, "超时后备援成功")
        # Hard timeout no longer burns the transport retry budget: one attempt, then failover.
        self.assertEqual(attempts, ["gemini-4-argon", "gemini-4-hydrogen"])
        self.assertEqual(len(usage_records), 2)
        self.assertFalse(usage_records[0]["success"])
        self.assertTrue(usage_records[1]["success"])
        self.assertTrue(any(e["severity"] == "WARN" and "已自动降级" in e["message"] for e in ops.events))
        # v5.4.1 regression: the timeout log must not raise NameError and mask the real cause.
        fallback_msg = next(e["message"] for e in ops.events if "已自动降级" in e["message"])
        self.assertIn("TimeoutError", fallback_msg)
        self.assertNotIn("NameError", fallback_msg)

    async def test_hard_timeout_respects_explicit_retry_budget(self) -> None:
        ops_events: list[dict] = []

        class _RecordingOps:
            async def notify(self, *, severity: str, source: str, alert_key: str, message: str) -> None:
                ops_events.append({"severity": severity, "alert_key": alert_key, "message": message})

        async def _record(rec: dict) -> None:
            return None

        client = GeminiClient(
            api_key="dummy-key",
            usage_recorder=_record,
            ops_notifier=_RecordingOps(),  # type: ignore[arg-type]
            fallback_model="gemini-4-hydrogen",
            fallback_thinking_level="HIGH",
            max_transient_retries=2,
            timeout_max_retries=1,
            retry_base_delay_s=0.0,
            round_timeout_seconds=0.05,
        )
        attempts: list[str] = []

        async def _slow_primary_fast_fallback(*, model, contents, config):
            attempts.append(model)
            if model == "gemini-4-carbon":
                await asyncio.sleep(0.25)
            return SimpleNamespace(
                text="备援成功",
                function_calls=[],
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    cached_content_token_count=0,
                    candidates_token_count=5,
                    thoughts_token_count=0,
                    total_token_count=15,
                ),
            )

        client._client = SimpleNamespace(  # type: ignore[assignment]
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=_slow_primary_fast_fallback))
        )
        reply = await client.generate(model="gemini-4-carbon", prompt="hi", usage_label="report_auditor")
        self.assertEqual(reply, "备援成功")
        self.assertEqual(attempts, ["gemini-4-carbon", "gemini-4-carbon", "gemini-4-hydrogen"])
        self.assertTrue(any(e["alert_key"].startswith("llm_fallback:") for e in ops_events))

    async def test_concurrent_tool_calls_isolation_timeout_and_compaction(self) -> None:
        client = GeminiClient(
            api_key="dummy-key",
            round_timeout_seconds=5.0,
            tool_timeout_seconds=0.08,
            max_concurrent_tools=4,
            max_tool_result_chars=320,
        )
        round_contents: list[list] = []

        async def _fake_generate_content(*, model, contents, config):
            round_contents.append(list(contents))
            if len(round_contents) == 1:
                return SimpleNamespace(
                    text="",
                    function_calls=[
                        SimpleNamespace(name="inspect_ticker", args={"symbol": "NVDA"}),
                        SimpleNamespace(name="inspect_ticker", args={"symbol": "FAIL"}),
                        SimpleNamespace(name="inspect_ticker", args={"symbol": "MSFT"}),
                        SimpleNamespace(name="inspect_ticker", args={"symbol": "HANG"}),
                    ],
                    candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
                    usage_metadata=None,
                )
            return SimpleNamespace(text="多工具并发完成", function_calls=[], candidates=[], usage_metadata=None)

        client._client = SimpleNamespace(  # type: ignore[assignment]
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate_content))
        )

        active = 0
        peak_active = 0

        async def _tool_exec(name: str, args: dict) -> dict:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            try:
                sym = args["symbol"]
                if sym == "HANG":
                    await asyncio.sleep(0.3)
                    return {"symbol": sym}
                await asyncio.sleep(0.02)
                if sym == "FAIL":
                    raise RuntimeError("tool exploded")
                if sym == "NVDA":
                    return {
                        "symbol": "NVDA",
                        "summary": "A" * 500,
                        "news": [f"headline {i}: " + ("X" * 80) for i in range(10)],
                    }
                return {"symbol": sym, "summary": "compact ok"}
            finally:
                active -= 1

        reply = await client.generate(
            model="gemini-4-hydrogen",
            prompt="check all",
            tool_executor=_tool_exec,
            usage_label="report_analyst",
        )
        self.assertEqual(reply, "多工具并发完成")
        self.assertGreaterEqual(peak_active, 2)
        self.assertEqual(len(round_contents), 2)

        tool_user_msg = round_contents[1][-1]
        parts = tool_user_msg.parts
        self.assertEqual(len(parts), 4)

        nvda_res = parts[0].function_response.response["result"]
        self.assertEqual(nvda_res["symbol"], "NVDA")
        self.assertTrue(nvda_res.get("_truncated"))
        self.assertLessEqual(len(json.dumps(nvda_res, ensure_ascii=False)), 320)

        fail_res = parts[1].function_response.response["result"]
        self.assertEqual(fail_res, {"error": "tool exploded"})

        msft_res = parts[2].function_response.response["result"]
        self.assertEqual(msft_res, {"symbol": "MSFT", "summary": "compact ok"})

        hang_res = parts[3].function_response.response["result"]
        self.assertIn("timeout", hang_res["error"])

        # Direct unit check on normal-sized payloads remaining untouched
        small = {"symbol": "AAPL", "summary": "normal"}
        self.assertEqual(_compact_tool_result(small, 320), small)

    async def test_monthly_search_quota_alerts_use_this_month_bucket(self) -> None:
        class _StubTg:
            def __init__(self) -> None:
                self.bot_token = "tok"
                self.sent: list[tuple[str, str]] = []

            async def send_message(self, chat_id: str, text: str, **kwargs) -> bool:
                self.sent.append((chat_id, text))
                return True

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "quota.db")
            await db.init()

            # Insert a prior-month row to confirm this_month excludes prior-month search queries
            uk_tz = ZoneInfo("Europe/London")
            prev_month_dt = datetime.now(timezone.utc).astimezone(uk_tz).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=2)
            prev_ms = int(prev_month_dt.timestamp() * 1000)
            async with db._connect(write=True) as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_usage_logs (
                        cycle_date, created_at_uk, created_at_ms, call_label, model,
                        thinking_level, api_rounds, prompt_tokens, cached_tokens,
                        output_tokens, thought_tokens, total_tokens, estimated_cost_usd,
                        latency_ms, success, search_queries
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prev_month_dt.strftime("%Y-%m-%d"),
                        prev_month_dt.strftime("%Y-%m-%d %H:%M"),
                        prev_ms,
                        "report_grounding",
                        "gemini-4-hydrogen",
                        "LOW",
                        1,
                        100,
                        0,
                        20,
                        0,
                        120,
                        0.0,
                        50,
                        1,
                        999,
                    ),
                )
                await conn.commit()

            tg = _StubTg()
            ops = OpsNotifier(
                telegram=tg,  # type: ignore[arg-type]
                chat_id="12345",
                db=db,
                warn_cooldown_seconds=1800,
            )
            client = GeminiClient(
                api_key="dummy-key",
                usage_recorder=db.log_llm_usage,
                ops_notifier=ops,
                usage_summary_provider=db.get_usage_summary,
                search_free_quota_monthly=10,
                search_warn_pct=80.0,
            )

            query_batches = [["q1", "q2", "q3", "q4", "q5", "q6", "q7"], ["q8"], ["q9", "q10"]]

            async def _fake_search(*, model, contents, config):
                queries = query_batches.pop(0)
                return SimpleNamespace(
                    text="检索完成",
                    function_calls=[],
                    candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(web_search_queries=queries))],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        cached_content_token_count=0,
                        candidates_token_count=20,
                        thoughts_token_count=0,
                        total_token_count=120,
                    ),
                )

            client._client = SimpleNamespace(  # type: ignore[assignment]
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=_fake_search))
            )

            # 1) 7 / 10 (70% < 80%): prior month 999 queries must not trigger alert
            await client.generate(model="gemini-4-hydrogen", prompt="p1", use_google_search=True)
            self.assertEqual(len(tg.sent), 0)

            # 2) 8 / 10 (80%): warning alert triggered
            await client.generate(model="gemini-4-hydrogen", prompt="p2", use_google_search=True)
            self.assertEqual(len(tg.sent), 1)
            self.assertIn("8 / 10", tg.sent[0][1])

            # 3) 10 / 10 (100%): exceeded alert uses separate alert_key so it fires immediately
            await client.generate(model="gemini-4-hydrogen", prompt="p3", use_google_search=True)
            self.assertEqual(len(tg.sent), 2)
            self.assertIn("10 / 10", tg.sent[1][1])
            self.assertIn("免费额度上限", tg.sent[1][1])

    def test_candidate_chain_never_upgrades_hydrogen_to_argon(self) -> None:
        client = GeminiClient(api_key="dummy-key", fallback_model="gemini-4-hydrogen")
        self.assertEqual(
            client._candidate_chain("gemini-4-hydrogen", "HIGH"),
            [("gemini-4-hydrogen", "HIGH"), ("gemini-3.5-flash-lite", "LOW")],
        )
        self.assertEqual(
            client._candidate_chain("gemini-4-argon", "HIGH"),
            [("gemini-4-argon", "HIGH"), ("gemini-4-hydrogen", "HIGH")],
        )
        self.assertEqual(
            client._candidate_chain("gemini-4-carbon", "HIGH"),
            [("gemini-4-carbon", "HIGH"), ("gemini-4-hydrogen", "HIGH")],
        )

    def test_cross_vendor_candidate_chain_and_search_isolation(self) -> None:
        client = GeminiClient(
            api_key="dummy-gem",
            deepseek_api_key="dummy-ds",
            fallback_model="gemini-4-hydrogen",
            gemini_fallback_model="gemini-4-hydrogen",
            gemini_fallback_thinking_level="MEDIUM",
            deepseek_fallback_model="deepseek-flash",
            deepseek_fallback_thinking_level="HIGH",
        )
        self.assertEqual(
            client._candidate_chain("deepseek-flash", "HIGH"),
            [
                ("deepseek-flash", "HIGH"),
                ("gemini-4-hydrogen", "MEDIUM"),
                ("gemini-3.5-flash-lite", "LOW"),
            ],
        )
        self.assertEqual(
            client._candidate_chain("gemini-4-hydrogen", "HIGH"),
            [
                ("gemini-4-hydrogen", "HIGH"),
                ("gemini-3.5-flash-lite", "LOW"),
                ("deepseek-flash", "HIGH"),
            ],
        )
        # Google Search calls MUST never fall back to DeepSeek
        self.assertEqual(
            client._candidate_chain("gemini-4-hydrogen", "LOW", use_google_search=True),
            [
                ("gemini-4-hydrogen", "LOW"),
                ("gemini-3.5-flash-lite", "LOW"),
            ],
        )
        self.assertEqual(
            client._candidate_chain("gemini-3.5-flash-lite", "LOW", use_google_search=True),
            [
                ("gemini-3.5-flash-lite", "LOW"),
                ("gemini-4-hydrogen", "LOW"),
            ],
        )

    async def test_deepseek_tool_calls_telemetry_and_bidirectional_failover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "hybrid.db")
            await db.init()

            class _StubTg:
                bot_token = "dummy-token"

                def __init__(self) -> None:
                    self.sent: list[tuple[str, str]] = []

                async def send_message(self, chat_id: str, text: str) -> bool:
                    self.sent.append((chat_id, text))
                    return True

            tg = _StubTg()
            ops = OpsNotifier(telegram=tg, chat_id="999", db=db, warn_cooldown_seconds=0, critical_cooldown_seconds=0)
            client = GeminiClient(
                api_key="dummy-gem",
                deepseek_api_key="dummy-ds",
                usage_recorder=db.log_llm_usage,
                ops_notifier=ops,
                fallback_model="gemini-4-hydrogen",
                gemini_fallback_model="gemini-4-hydrogen",
                gemini_fallback_thinking_level="MEDIUM",
                deepseek_fallback_model="deepseek-flash",
                deepseek_fallback_thinking_level="HIGH",
                max_transient_retries=0,
                deepseek_max_retries=0,
                retry_base_delay_s=0.0,
            )

            # 1) Schema conversion lowercases Gemini types
            decls = [
                types.FunctionDeclaration(
                    name="inspect_ticker",
                    description="Fetch ticker info",
                    parameters={
                        "type": "OBJECT",
                        "properties": {"symbol": {"type": "STRING", "description": "Ticker"}},
                        "required": ["symbol"],
                    },
                )
            ]
            oai_tools = _function_declarations_to_openai_tools(decls)
            self.assertEqual(oai_tools[0]["function"]["parameters"]["type"], "object")
            self.assertEqual(oai_tools[0]["function"]["parameters"]["properties"]["symbol"]["type"], "string")

            # 2) DeepSeek primary multi-round tool call + reasoning_content preservation + telemetry
            ds_payloads: list[dict] = []

            async def _ds_tool_flow(payload: dict) -> dict:
                ds_payloads.append(payload)
                if len(ds_payloads) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "Need live price first",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "inspect_ticker", "arguments": '{"symbol": "NVDA"}'},
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1000,
                            "prompt_cache_hit_tokens": 400,
                            "completion_tokens": 180,
                            "completion_tokens_details": {"reasoning_tokens": 120},
                            "total_tokens": 1180,
                        },
                    }
                return {
                    "choices": [{"message": {"role": "assistant", "content": "NVDA 维持观察。"}}],
                    "usage": {
                        "prompt_tokens": 1300,
                        "prompt_cache_hit_tokens": 900,
                        "completion_tokens": 250,
                        "completion_tokens_details": {"reasoning_tokens": 150},
                        "total_tokens": 1550,
                    },
                }

            client._post_deepseek_completion = _ds_tool_flow  # type: ignore[method-assign]

            async def _exec(name: str, args: dict) -> dict:
                return {"symbol": args["symbol"], "price": 140.0}

            reply = await client.generate(
                model="deepseek-flash",
                prompt="NVDA 怎么看",
                system_instruction="你是量化顾问",
                thinking_level="HIGH",
                tools=decls,
                tool_executor=_exec,
                usage_label="chat",
            )
            self.assertEqual(reply, "NVDA 维持观察。")
            self.assertEqual(len(ds_payloads), 2)
            self.assertEqual(ds_payloads[0]["thinking"], {"type": "enabled"})
            self.assertEqual(ds_payloads[0]["reasoning_effort"], "high")
            self.assertNotIn("temperature", ds_payloads[0])
            # Second round must carry assistant reasoning_content and tool response
            second_msgs = ds_payloads[1]["messages"]
            self.assertEqual(second_msgs[2]["reasoning_content"], "Need live price first")
            self.assertEqual(second_msgs[3]["role"], "tool")
            self.assertEqual(second_msgs[3]["tool_call_id"], "call_1")

            summary = await db.get_usage_summary(days=30)
            self.assertEqual(summary["today"]["prompt_tokens"], 2300)
            self.assertEqual(summary["today"]["cached_tokens"], 1300)
            self.assertEqual(summary["today"]["thought_tokens"], 270)
            self.assertEqual(summary["today"]["output_tokens"], 160)

            # 3) DeepSeek 503 outage -> fails over cleanly to Gemini Hydrogen MEDIUM + WARN alert
            async def _ds_down(payload: dict) -> dict:
                raise RuntimeError("HTTP 503: upstream overloaded")

            gem_calls: list[tuple[str, str]] = []

            async def _gem_ok(*, model, contents, config):
                gem_calls.append((model, str(config.thinking_config.thinking_level.value)))
                return SimpleNamespace(
                    text="Gemini 备援回答成功",
                    function_calls=[],
                    candidates=[],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=200,
                        cached_content_token_count=0,
                        candidates_token_count=40,
                        thoughts_token_count=30,
                        total_token_count=270,
                    ),
                )

            client._post_deepseek_completion = _ds_down  # type: ignore[method-assign]
            client._client = SimpleNamespace(  # type: ignore[assignment]
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=_gem_ok))
            )
            fb_reply = await client.generate(
                model="deepseek-flash",
                prompt="hello",
                thinking_level="HIGH",
                usage_label="us_sentinel",
            )
            self.assertEqual(fb_reply, "Gemini 备援回答成功")
            self.assertEqual(gem_calls, [("gemini-4-hydrogen", "MEDIUM")])
            self.assertTrue(any("已自动降级到备援模型 gemini-4-hydrogen (MEDIUM)" in m[1] for m in tg.sent))

            # 4) Gemini structured JSON outage -> fails over to DeepSeek with JSON Schema injected
            async def _gem_down(*, model, contents, config):
                raise RuntimeError("503 Service Unavailable")

            ds_json_payloads: list[dict] = []

            async def _ds_json_ok(payload: dict) -> dict:
                ds_json_payloads.append(payload)
                valid_json = json.dumps(
                    {
                        "headline": "跨厂兜底早报成功",
                        "top_actions": [
                            {
                                "account": "ISA",
                                "symbol": "CASH",
                                "action": "HOLD",
                                "price_zone": "N/A",
                                "reason": "保持观察",
                            }
                        ],
                        "pending_order_audits": [],
                        "daily_vetoes": ["控制仓位"],
                        "macro_and_cash": "现金利息4%",
                        "account_diagnosis": "双账户正常",
                        "flagged_positions": [],
                        "hold_summary": "常规持有",
                        "trade_review": "无频繁交易",
                        "earnings_watch": [],
                    },
                    ensure_ascii=False,
                )
                return {
                    "choices": [{"message": {"role": "assistant", "content": valid_json}}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
                }

            client._client = SimpleNamespace(  # type: ignore[assignment]
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=_gem_down))
            )
            client._post_deepseek_completion = _ds_json_ok  # type: ignore[method-assign]
            pm_raw = await client.generate(
                model="gemini-4-hydrogen",
                prompt="综合早报",
                system_instruction="你是 PM",
                thinking_level="HIGH",
                response_schema=MorningReportDecision,
                usage_label="report_pm",
            )
            parsed = parse_structured_json(pm_raw, MorningReportDecision)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(parsed.headline, "跨厂兜底早报成功")
            self.assertEqual(ds_json_payloads[0]["response_format"], {"type": "json_object"})
            self.assertIn("JSON OUTPUT CONTRACT", ds_json_payloads[0]["messages"][0]["content"])
            self.assertIn("pending_order_audits", ds_json_payloads[0]["messages"][0]["content"])
            await client.close()
            await db.close()


class TestEgressGuards(unittest.TestCase):
    """A tampered base_url must never ship API keys or portfolio plaintext to a third party."""

    def test_defaults_pin_official_https_endpoints(self) -> None:
        cfg = load_config("config.yaml", env_file=None)
        self.assertEqual(cfg.env.deepseek_base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.env.t212_base_url, "https://live.trading212.com/api/v0")

    def test_plain_http_endpoint_is_rejected(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "http://relay.example.com/v1"}):
            with self.assertRaises(ValidationError):
                EnvSettings(_env_file=None)

    def test_unlisted_host_is_rejected_by_default(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://relay.example.com/v1"}):
            with self.assertRaises(ValidationError):
                EnvSettings(_env_file=None)

    def test_explicit_allowlist_unlocks_relay_and_trims_slash(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_BASE_URL": "https://relay.example.com/v1/",
                "LLM_EGRESS_ALLOWED_HOSTS": "relay.example.com, mirror.example.com",
            },
        ):
            env = EnvSettings(_env_file=None)
        self.assertEqual(env.deepseek_base_url, "https://relay.example.com/v1")

    def test_t212_endpoint_tampering_is_rejected(self) -> None:
        with patch.dict(os.environ, {"T212_BASE_URL": "https://evil.example.com/api/v0"}):
            with self.assertRaises(ValidationError):
                EnvSettings(_env_file=None)

    def test_demo_t212_endpoint_is_allowed(self) -> None:
        with patch.dict(os.environ, {"T212_BASE_URL": "https://demo.trading212.com/api/v0"}):
            env = EnvSettings(_env_file=None)
        self.assertEqual(env.t212_base_url, "https://demo.trading212.com/api/v0")

    def test_blank_endpoint_falls_back_to_official_default(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "", "T212_BASE_URL": ""}):
            env = EnvSettings(_env_file=None)
        self.assertEqual(env.deepseek_base_url, "https://api.deepseek.com")
        self.assertEqual(env.t212_base_url, "https://live.trading212.com/api/v0")


if __name__ == "__main__":
    unittest.main()
