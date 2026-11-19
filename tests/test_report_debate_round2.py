"""Offline tests for the /report Step 2 round-2 debate + dual neutral index (v5.4.2)."""

from __future__ import annotations

import asyncio
import unittest
import warnings
from typing import Any

from src.agents.report_agent import (
    STEP2_AUDITOR_FALLBACK,
    _build_debate_digest_prompt,
    _build_debate_pm_block,
    _build_factcheck_prompt,
    _build_rebuttal_prompt,
    _gather_named_notes,
    _run_step2_debate,
)
from src.config import AppConfig, EnvSettings


class _FakeToolkit:
    def declarations(self) -> list[Any]:
        return []

    def as_executor(self) -> Any:
        return None


class _DebateLLM:
    """Records every generate() call and returns per-usage_label canned text."""

    _R2R3_LABELS = {
        "report_bull_rebuttal",
        "report_bear_rebuttal",
        "report_debate_digest",
        "report_debate_factcheck",
    }

    def __init__(self, responses: dict[str, Any] | None = None, *, deepseek_key: str = "dummy-ds") -> None:
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []
        self.deepseek_api_key = deepseek_key
        self.r23_active = 0
        self.r23_max_active = 0

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        label = str(kwargs.get("usage_label") or "")
        tracked = label in self._R2R3_LABELS
        if tracked:
            self.r23_active += 1
            self.r23_max_active = max(self.r23_max_active, self.r23_active)
        try:
            value = self.responses.get(label)
            if callable(value):
                return await value(kwargs)
            if value is not None:
                return str(value)
            return f"{label}-text"
        finally:
            if tracked:
                self.r23_active -= 1


def _cfg(**overrides: Any) -> AppConfig:
    cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
    cfg.models.report_step2_timeout_seconds = 30.0
    cfg.models.report_step2_round1_seconds = 5.0
    cfg.models.report_step2_round2_seconds = 3.0
    cfg.models.report_step2_round3_seconds = 3.0
    for key, value in overrides.items():
        setattr(cfg.models, key, value)
    return cfg


async def _run(
    cfg: AppConfig,
    llm: _DebateLLM,
    *,
    fact_sheet: str = "FACT-BASE",
    shared_context: str = "SHARED-CTX",
) -> Any:
    return await _run_step2_debate(
        cfg=cfg,
        llm=llm,  # type: ignore[arg-type]
        toolkit=_FakeToolkit(),  # type: ignore[arg-type]
        shared_context=shared_context,
        fact_sheet=fact_sheet,
        disagreement_text="DISAGREE-MATRIX",
        analyst_sys="BULL-SYS",
        auditor_sys="BEAR-SYS",
        analyst_prompt="BULL-PROMPT",
        auditor_prompt="BEAR-PROMPT",
        report_rounds=2,
        ops_notifier=None,
    )


class TestDebatePrompts(unittest.TestCase):
    def test_rebuttal_prompt_contains_both_sides_and_hard_rules(self) -> None:
        prompt = _build_rebuttal_prompt(
            side="bull",
            own_notes="BULL-R1",
            opponent_notes="BEAR-R1",
            shared_context="FACTS",
            max_chars=2400,
        )
        self.assertIn("BULL-R1", prompt)
        self.assertIn("BEAR-R1", prompt)
        self.assertIn("FACTS", prompt)
        self.assertIn("不得编造", prompt)
        self.assertIn("不要给最终买卖结论", prompt)

    def test_pm_block_is_index_only(self) -> None:
        block = _build_debate_pm_block("DIGEST", "FACTCHECK")
        self.assertIn("不是裁决", block)
        self.assertIn("DIGEST", block)
        self.assertIn("FACTCHECK", block)
        self.assertEqual(_build_debate_pm_block("", ""), "")

    def test_digest_and_factcheck_prompts_are_bounded(self) -> None:
        digest = _build_debate_digest_prompt(
            bull_notes="B",
            bear_notes="S",
            bull_rebuttal="BR",
            bear_rebuttal="SR",
            disagreement_text="D",
            max_chars=900,
        )
        factcheck = _build_factcheck_prompt(
            fact_sheet="F", bull_rebuttal="BR", bear_rebuttal="SR", max_chars=400
        )
        self.assertIn("≤900 字", digest)
        self.assertIn("只做索引，不做裁决", digest)
        self.assertNotIn("关键原文", digest)
        self.assertIn("≤400 字", factcheck)
        self.assertIn("只标记、不判违规", factcheck)


class TestGatherNamedNotes(unittest.IsolatedAsyncioTestCase):
    async def test_failure_returns_empty_string_and_is_reported(self) -> None:
        async def _boom() -> str:
            raise RuntimeError("boom")

        async def _ok() -> str:
            return "OK"

        texts, degraded = await _gather_named_notes({"bad": _boom(), "good": _ok()}, timeout_s=2.0)
        self.assertEqual(texts, {"bad": "", "good": "OK"})
        self.assertEqual(degraded, ["bad"])


class TestStep2Debate(unittest.IsolatedAsyncioTestCase):
    async def test_six_calls_in_phase_order_and_prompt_symmetry(self) -> None:
        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": "BULL-R2",
                "report_bear_rebuttal": "BEAR-R2",
                "report_debate_digest": "DIGEST",
                "report_debate_factcheck": "FACTCHECK",
            }
        )
        result = await _run(_cfg(), llm)

        self.assertEqual(
            [c["usage_label"] for c in llm.calls],
            [
                "report_analyst",
                "report_auditor",
                "report_bull_rebuttal",
                "report_bear_rebuttal",
                "report_debate_digest",
                "report_debate_factcheck",
            ],
        )
        self.assertEqual(result.analyst_notes, "BULL-R1")
        self.assertEqual(result.auditor_notes, "BEAR-R1")
        self.assertEqual(result.digest, "DIGEST")
        self.assertEqual(result.factcheck, "FACTCHECK")
        self.assertEqual(result.degraded_roles, [])

        # R2 sees the opponent's R1 but is explicitly a cross-exam, not a fresh report.
        bull_r2 = llm.calls[2]["prompt"]
        bear_r2 = llm.calls[3]["prompt"]
        self.assertIn("BEAR-R1", bull_r2)
        self.assertIn("BULL-R1", bull_r2)
        self.assertIn("BULL-R1", bear_r2)
        self.assertIn("BEAR-R1", bear_r2)

        # R2/R3 run on DeepSeek with the configured budget + output cap.
        for call in llm.calls[2:]:
            self.assertEqual(call["model"], "deepseek-flash")
            self.assertEqual(call["thinking_level"], "HIGH")
        self.assertEqual(llm.calls[2]["max_output_tokens"], 16384)
        self.assertEqual(llm.calls[4]["max_output_tokens"], 8192)
        self.assertEqual(llm.calls[5]["max_output_tokens"], 8192)

        # PM payload stores the transcript but only the index is exposed for injection.
        payload = result.payload
        self.assertEqual(payload["r1"]["bull"], "BULL-R1")
        self.assertEqual(payload["r2"]["bull_rebuttal"], "BULL-R2")
        self.assertEqual(payload["r3"]["digest"], "DIGEST")

    async def test_r3_factcheck_uses_full_shared_context_and_digest_has_no_orphan_tail(self) -> None:
        """v5.5.6: R3 must verify against the SAME full base as Bull/Bear (no false [无据])."""
        shared = (
            "SHARED-FULL-CTX\n"
            "ISA 现金 £12,345.67｜Invest 现金 £8,765.43｜总净值 £210,000\n"
            "近 7 天 5 笔成交；15 天冷却名单：AMT、PANW\n"
            "活动挂单：ULVR.L BUY 4900 GBp\n"
        )
        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": "BULL-R2",
                "report_bear_rebuttal": "BEAR-R2",
                # Digest overflows with a lone Markdown heading left by tail trimming.
                "report_debate_digest": "1. 多方存活\n2. 空方存活\n**关键原文**",
                "report_debate_factcheck": "未发现无据数字",
            }
        )
        result = await _run(_cfg(), llm, fact_sheet="BARE-FACT-SHEET", shared_context=shared)
        factcheck_prompt = llm.calls[5]["prompt"]
        self.assertIn("ISA 现金 £12,345.67", factcheck_prompt)
        self.assertIn("Invest 现金 £8,765.43", factcheck_prompt)
        self.assertIn("15 天冷却名单：AMT、PANW", factcheck_prompt)
        self.assertNotIn("BARE-FACT-SHEET", factcheck_prompt)
        # The trimmed digest must not end on an orphan Markdown heading.
        self.assertTrue(result.digest.endswith("空方存活"), result.digest)
        self.assertNotIn("关键原文", result.digest)

    async def test_r3_factcheck_falls_back_to_fact_sheet_when_shared_context_empty(self) -> None:
        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": "BULL-R2",
                "report_bear_rebuttal": "BEAR-R2",
                "report_debate_digest": "DIGEST",
                "report_debate_factcheck": "FACTCHECK",
            }
        )
        await _run(_cfg(), llm, fact_sheet="BARE-FACT-SHEET", shared_context="")
        self.assertIn("BARE-FACT-SHEET", llm.calls[5]["prompt"])

    async def test_r1_degraded_skips_all_debate_rounds(self) -> None:
        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "⚠️ AI 模型服务当前全部不可用（主模型与备援均失败）：x",
            }
        )
        result = await _run(_cfg(), llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.auditor_notes, STEP2_AUDITOR_FALLBACK)
        self.assertEqual(result.degraded_roles, ["auditor"])
        self.assertEqual(result.digest, "")
        self.assertEqual(result.factcheck, "")
        self.assertEqual(result.payload["skipped"], "r1_degraded")

    async def test_r2_failure_skips_r3_and_keeps_r1(self) -> None:
        async def _boom(_: dict[str, Any]) -> str:
            raise RuntimeError("deepseek down")

        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": _boom,
            }
        )
        result = await _run(_cfg(), llm)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(result.analyst_notes, "BULL-R1")
        self.assertEqual(result.auditor_notes, "BEAR-R1")
        self.assertEqual(result.digest, "")
        self.assertEqual(result.payload["r2"]["bull_rebuttal"], "")
        self.assertEqual(result.payload["r3"], {})
        # Optional debate failures must not be reported as an R1 Step-2 degradation.
        self.assertEqual(result.degraded_roles, [])

    async def test_r3_digest_failure_keeps_factcheck(self) -> None:
        async def _boom(_: dict[str, Any]) -> str:
            raise RuntimeError("digest down")

        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": "BULL-R2",
                "report_bear_rebuttal": "BEAR-R2",
                "report_debate_digest": _boom,
                "report_debate_factcheck": "FACTCHECK",
            }
        )
        result = await _run(_cfg(), llm)
        self.assertEqual(result.digest, "")
        self.assertEqual(result.factcheck, "FACTCHECK")

    async def test_disabled_flag_restores_two_calls(self) -> None:
        llm = _DebateLLM({"report_analyst": "BULL-R1", "report_auditor": "BEAR-R1"})
        result = await _run(_cfg(report_debate_enabled=False), llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.digest, "")
        self.assertEqual(result.payload["skipped"], "debate_disabled")

    async def test_missing_deepseek_key_restores_two_calls(self) -> None:
        llm = _DebateLLM(
            {"report_analyst": "BULL-R1", "report_auditor": "BEAR-R1"},
            deepseek_key="",
        )
        result = await _run(_cfg(), llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.payload["skipped"], "deepseek_unavailable")

    async def test_r2_timeout_reclaims_tasks_and_returns_early(self) -> None:
        async def _slow(_: dict[str, Any]) -> str:
            await asyncio.sleep(30)
            return "late"

        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": _slow,
                "report_bear_rebuttal": _slow,
            }
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await _run(
            _cfg(report_step2_round2_seconds=0.1, report_debate_sequential=False), llm
        )
        elapsed = loop.time() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(result.digest, "")
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(result.payload["r3"], {})

    async def test_sequential_mode_keeps_one_r2_r3_call_in_flight(self) -> None:
        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": "BULL-R2",
                "report_bear_rebuttal": "BEAR-R2",
                "report_debate_digest": "DIGEST",
                "report_debate_factcheck": "FACTCHECK",
            }
        )
        result = await _run(_cfg(report_debate_sequential=True), llm)
        self.assertEqual(llm.r23_max_active, 1)
        self.assertEqual(result.digest, "DIGEST")
        self.assertEqual(result.factcheck, "FACTCHECK")
        self.assertEqual(result.degraded_roles, [])

    async def test_sequential_budget_exhaustion_skips_later_step_without_unawaited_warning(self) -> None:
        calls = {"n": 0}

        async def _slow(_: dict[str, Any]) -> str:
            calls["n"] += 1
            await asyncio.sleep(30)
            return "late"

        llm = _DebateLLM(
            {
                "report_analyst": "BULL-R1",
                "report_auditor": "BEAR-R1",
                "report_bull_rebuttal": _slow,
                "report_bear_rebuttal": _slow,
            }
        )
        with self.assertNoLogs(level="ERROR"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await _run(
                    _cfg(report_step2_round2_seconds=0.1, report_debate_sequential=True), llm
                )
        self.assertFalse(
            [w for w in caught if "never awaited" in str(w.message)],
            [str(w.message) for w in caught],
        )
        # Only R1 (x2) + the first R2 step ran; the second factory was never constructed.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(result.payload["r2"]["bull_rebuttal"], "")
        self.assertIn("bull_rebuttal", result.payload["degraded"])
        self.assertIn("bear_rebuttal", result.payload["degraded"])
        self.assertEqual(result.payload["r3"], {})
