"""Offline tests for /report Step 2 wall-clock budget and neutral degradation (v5.3.3)."""

from __future__ import annotations

import asyncio
import unittest

from src.agents.llm_client import DEGRADED_LLM_MARKERS, is_degraded_llm_output
from src.agents.report_agent import (
    STEP2_ANALYST_FALLBACK,
    STEP2_AUDITOR_FALLBACK,
    _gather_step2_notes,
)


class TestDegradedSentinel(unittest.TestCase):
    def test_detects_model_unavailable_sentinels(self) -> None:
        for marker in DEGRADED_LLM_MARKERS:
            self.assertTrue(is_degraded_llm_output(f"{marker}（主模型与备援均失败）：gemini-4-carbon: TimeoutError"))

    def test_real_analysis_text_is_not_degraded(self) -> None:
        self.assertFalse(is_degraded_llm_output("【Bull】NVDA 站上 MA200，回踩 118 可分批加仓。"))
        self.assertFalse(is_degraded_llm_output(""))
        self.assertFalse(is_degraded_llm_output(None))


class TestGatherStep2Notes(unittest.IsolatedAsyncioTestCase):
    async def test_both_branches_finish_within_budget(self) -> None:
        async def _fast(text: str) -> str:
            await asyncio.sleep(0)
            return text

        analyst, auditor, degraded = await _gather_step2_notes(
            _fast("BULL NOTES"), _fast("BEAR NOTES"), timeout_s=5.0
        )
        self.assertEqual((analyst, auditor), ("BULL NOTES", "BEAR NOTES"))
        self.assertEqual(degraded, [])

    async def test_slow_auditor_is_cancelled_and_analyst_kept(self) -> None:
        cancelled = asyncio.Event()

        async def _fast() -> str:
            return "BULL NOTES"

        async def _slow() -> str:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "BEAR NOTES"

        analyst, auditor, degraded = await _gather_step2_notes(_fast(), _slow(), timeout_s=0.1)
        self.assertEqual(analyst, "BULL NOTES")
        self.assertEqual(auditor, STEP2_AUDITOR_FALLBACK)
        self.assertEqual(degraded, ["auditor"])
        self.assertTrue(cancelled.is_set())

    async def test_both_branches_time_out(self) -> None:
        async def _slow() -> str:
            await asyncio.sleep(30)
            return "late"

        analyst, auditor, degraded = await _gather_step2_notes(_slow(), _slow(), timeout_s=0.1)
        self.assertEqual(analyst, STEP2_ANALYST_FALLBACK)
        self.assertEqual(auditor, STEP2_AUDITOR_FALLBACK)
        self.assertEqual(sorted(degraded), ["analyst", "auditor"])

    async def test_model_unavailable_sentinel_is_replaced_with_neutral_text(self) -> None:
        async def _sentinel() -> str:
            return f"{DEGRADED_LLM_MARKERS[0]}（gemini-4-carbon: TimeoutError）。请立即检查 API 配额。"

        async def _ok() -> str:
            return "BULL NOTES"

        analyst, auditor, degraded = await _gather_step2_notes(_ok(), _sentinel(), timeout_s=5.0)
        self.assertEqual(analyst, "BULL NOTES")
        self.assertEqual(auditor, STEP2_AUDITOR_FALLBACK)
        self.assertEqual(degraded, ["auditor"])
        self.assertNotIn("⚠️", auditor)

    async def test_zero_budget_disables_wall_clock_cut(self) -> None:
        async def _fast(text: str) -> str:
            return text

        analyst, auditor, degraded = await _gather_step2_notes(_fast("A"), _fast("B"), timeout_s=0)
        self.assertEqual((analyst, auditor), ("A", "B"))
        self.assertEqual(degraded, [])
