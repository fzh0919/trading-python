"""Tests for config contract, mtime pool hot-reload, thinking-level normalization, and token/search telemetry."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from src.agents.chat_agent import _format_chat_history_compact
from src.agents.llm_client import GeminiClient, normalize_thinking_level
from src.config import ModelsSection, StrategySection, load_config
from src.storage.db import AdvisorDatabase


class TestConfigAndThinkingNormalization(unittest.TestCase):
    def test_thinking_normalization(self) -> None:
        self.assertEqual(normalize_thinking_level("gemini-4-argon", "MINIMAL"), "LOW")
        self.assertEqual(normalize_thinking_level("gemini-4-argon", "XHIGH"), "HIGH")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "MINIMAL"), "LOW")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "xhigh"), "XHIGH")
        self.assertEqual(normalize_thinking_level("gemini-3.5-flash-lite", "MINIMAL"), "MINIMAL")
        self.assertEqual(normalize_thinking_level("gemini-4-hydrogen", "INVALID"), "")

    def test_models_and_strategy_defaults_and_legacy_migration(self) -> None:
        cfg = load_config("config.yaml", env_file=None)
        self.assertEqual(cfg.models.report_analyst_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.report_analyst_thinking_level, "HIGH")
        self.assertEqual(cfg.models.report_model, "gemini-4-argon")
        self.assertEqual(cfg.models.report_thinking_level, "HIGH")
        self.assertEqual(cfg.models.us_sentinel_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.us_thinking_level, "MEDIUM")
        self.assertEqual(cfg.models.pick_researcher_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.pick_researcher_thinking_level, "HIGH")
        self.assertEqual(cfg.models.pick_model, "gemini-4-argon")
        self.assertEqual(cfg.models.pick_thinking_level, "HIGH")
        self.assertEqual(cfg.models.chat_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.chat_thinking_level, "LOW")
        self.assertEqual(cfg.models.chat_deep_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.chat_deep_thinking_level, "HIGH")
        self.assertEqual(cfg.models.fallback_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.fallback_thinking_level, "HIGH")
        self.assertEqual(cfg.models.report_search_model, "gemini-4-hydrogen")
        self.assertEqual(cfg.models.search_model, "gemini-3.5-flash-lite")
        self.assertTrue(cfg.models.enable_forced_search_grounding)
        self.assertEqual(cfg.models.report_max_tool_rounds, 6)
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


if __name__ == "__main__":
    unittest.main()
