"""Offline tests for the DeepSeek sidecar: /pick ID prefilter, semantic linter, JSON/fallback helpers."""

from __future__ import annotations

import asyncio
import unittest

from src.agents.ds_sidecar import (
    call_ds_json,
    ds_helper_available,
    extract_json_object,
    sanitize_plain_text,
)
from src.agents.pick_agent import _prefilter_candidates_with_ds, _resolve_keep_candidates
from src.agents.semantic_auditor import audit_generated_output, build_fact_baseline
from src.config import AppConfig, EnvSettings
from src.data_providers.t212_client import AccountSnapshot, PortfolioData
from src.engine.screener import ScreenerCandidate


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
    """Stub with no DeepSeek credential: any call would be a bug (Gemini silent takeover guard)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise AssertionError("sidecar must not call the LLM without a DeepSeek key")


class _SlowLLM(_DsStubLLM):
    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        await asyncio.sleep(0.3)
        return '{"passed": true, "violations": []}'


class _RecordingOps:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def notify(self, **kwargs) -> bool:
        self.events.append(kwargs)
        return True


def _cfg(**models_overrides) -> AppConfig:
    cfg = AppConfig(env=EnvSettings(_env_file=None, deepseek_api_key="", ops_notify_url=""))
    for key, value in models_overrides.items():
        setattr(cfg.models, key, value)
    return cfg


def _cand(symbol: str, track: str, score: float) -> ScreenerCandidate:
    return ScreenerCandidate(
        symbol=symbol,
        track=track,
        name=f"{symbol} Holdings",
        score=score,
        currency="USD",
        dimension_pct={"trend_rs": 80.0, "value_quality": 65.0},
        pros=["营收增速稳健"],
        cons=["估值偏高"],
        pattern_signals=["VCP收缩"],
        summary_line=f"{symbol} | 量化分 {score:.1f}",
    )


def _portfolio() -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 09:00",
        fetched_at_ms=1794900000000,
        total_combined_value=10000.0,
        total_combined_cash=2500.0,
        isa=AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=6000.0,
            total_cash=1500.0,
            available_to_trade=1400.0,
            reserved_cash=100.0,
        ),
        invest=AccountSnapshot(
            ok=True,
            account_label="Invest",
            total_account_value=4000.0,
            total_cash=1000.0,
            available_to_trade=900.0,
            reserved_cash=100.0,
        ),
    )


class TestDsSidecarHelpers(unittest.IsolatedAsyncioTestCase):
    def test_availability_requires_real_key(self) -> None:
        cfg = _cfg()
        self.assertTrue(ds_helper_available(_DsStubLLM(), cfg))
        self.assertFalse(ds_helper_available(_NoKeyLLM(), cfg))
        self.assertFalse(ds_helper_available(None, cfg))

        cfg_with_env_key = AppConfig(
            env=EnvSettings(_env_file=None, deepseek_api_key="env-key", ops_notify_url="")
        )
        self.assertTrue(ds_helper_available(_NoKeyLLM(), cfg_with_env_key))

    def test_extract_json_object_handles_fences_and_noise(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object('前缀 {"passed": true} 后缀'), {"passed": True})
        self.assertEqual(extract_json_object('{"a": 1}{"b": 2}'), {"a": 1})
        self.assertIsNone(extract_json_object("⚠️ AI 模型服务当前全部不可用"))
        self.assertIsNone(extract_json_object("这不是 JSON"))
        self.assertIsNone(extract_json_object(""))

    async def test_call_ds_json_degrades_silently(self) -> None:
        cfg = _cfg()
        warning_stub = _DsStubLLM({"pick_prefilter": "⚠️ 模型不可用"})
        self.assertIsNone(
            await call_ds_json(cfg, warning_stub, prompt="p", usage_label="pick_prefilter")
        )
        self.assertIsNone(await call_ds_json(cfg, _NoKeyLLM(), prompt="p", usage_label="x"))

        class _BoomLLM(_DsStubLLM):
            async def generate(self, **kwargs) -> str:
                raise RuntimeError("network down")

        self.assertIsNone(await call_ds_json(cfg, _BoomLLM(), prompt="p", usage_label="x"))
        self.assertIsNone(
            await call_ds_json(cfg, _SlowLLM(), prompt="p", timeout_s=0.01, usage_label="x")
        )

    async def test_call_ds_json_uses_helper_model_and_thinking(self) -> None:
        cfg = _cfg()
        llm = _DsStubLLM({"pick_prefilter": '{"keep_isa": []}'})
        parsed = await call_ds_json(cfg, llm, prompt="p", usage_label="pick_prefilter")
        self.assertEqual(parsed, {"keep_isa": []})
        self.assertEqual(llm.calls[0]["model"], "deepseek-flash")
        self.assertEqual(llm.calls[0]["thinking_level"], "HIGH")




class TestPickPrefilter(unittest.IsolatedAsyncioTestCase):
    def _funnel(self) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
        isa = [_cand(f"ISA{i:02d}", "ISA_LONG_TERM", 90.0 - i) for i in range(1, 13)]
        invest = [_cand(f"INV{i:02d}", "INVEST_TACTICAL", 80.0 - i) for i in range(1, 13)]
        return isa, invest

    async def test_keep_original_objects_in_screener_order(self) -> None:
        cfg = _cfg()
        isa, invest = self._funnel()
        llm = _DsStubLLM(
            {
                "pick_prefilter": (
                    '{"keep_isa": ["ISA10", "ISA02"], "keep_invest": [], '
                    '"flags": {"ISA03": "估值过高"}}'
                )
            }
        )
        kept_isa, kept_invest = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual([c.symbol for c in kept_isa], ["ISA02", "ISA10"])
        self.assertIs(kept_isa[0], isa[1])
        self.assertIs(kept_isa[1], isa[9])
        # Empty invest list falls back to the raw quant top-8 instead of dropping the whole track.
        self.assertEqual([c.symbol for c in kept_invest], [c.symbol for c in invest[:8]])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("ISA01", llm.calls[0]["prompt"])
        self.assertNotIn("估值过高", llm.calls[0]["prompt"])

    async def test_keep_capped_per_track(self) -> None:
        cfg = _cfg()
        isa, invest = self._funnel()
        llm = _DsStubLLM(
            {
                "pick_prefilter": (
                    '{"keep_isa": ["ISA01","ISA02","ISA03","ISA04","ISA05","ISA06","ISA07","ISA08","ISA09","ISA10"],'
                    ' "keep_invest": ["INV01","INV02","INV03","INV04","INV05","INV06","INV07","INV08","INV09"]}'
                )
            }
        )
        kept_isa, kept_invest = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual(len(kept_isa), 8)
        self.assertEqual(len(kept_invest), 8)
        self.assertEqual(kept_isa[0].symbol, "ISA01")
        self.assertEqual(kept_invest[-1].symbol, "INV08")

    async def test_hallucinated_symbols_fall_back(self) -> None:
        cfg = _cfg()
        isa, invest = self._funnel()
        llm = _DsStubLLM({"pick_prefilter": '{"keep_isa": ["GHOST"], "keep_invest": ["FAKE"]}'})
        kept_isa, kept_invest = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual([c.symbol for c in kept_isa], [c.symbol for c in isa[:8]])
        self.assertEqual([c.symbol for c in kept_invest], [c.symbol for c in invest[:8]])

    async def test_invalid_json_and_exception_fall_back(self) -> None:
        cfg = _cfg()
        isa, invest = self._funnel()
        for reply in ("这不是 JSON", "{}", ""):
            kept_isa, kept_invest = await _prefilter_candidates_with_ds(
                cfg, _DsStubLLM({"pick_prefilter": reply}), isa, invest
            )
            self.assertEqual([c.symbol for c in kept_isa], [c.symbol for c in isa[:8]])
            self.assertEqual([c.symbol for c in kept_invest], [c.symbol for c in invest[:8]])

        class _BoomLLM(_DsStubLLM):
            async def generate(self, **kwargs) -> str:
                raise RuntimeError("network down")

        kept_isa, _ = await _prefilter_candidates_with_ds(cfg, _BoomLLM(), isa, invest)
        self.assertEqual([c.symbol for c in kept_isa], [c.symbol for c in isa[:8]])

    async def test_switch_off_makes_zero_calls(self) -> None:
        cfg = _cfg(ds_pick_prefilter_enabled=False)
        isa, invest = self._funnel()
        llm = _DsStubLLM({"pick_prefilter": '{"keep_isa": ["ISA01"], "keep_invest": ["INV01"]}'})
        kept_isa, kept_invest = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual(len(kept_isa), 8)
        self.assertEqual(len(kept_invest), 8)
        self.assertEqual(llm.calls, [])

    async def test_missing_key_makes_zero_calls(self) -> None:
        cfg = _cfg()
        isa, invest = self._funnel()
        llm = _NoKeyLLM()
        kept_isa, _ = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual([c.symbol for c in kept_isa], [c.symbol for c in isa[:8]])
        self.assertEqual(llm.calls, [])

    async def test_small_funnel_passes_through(self) -> None:
        cfg = _cfg()
        isa = [_cand(f"ISA{i}", "ISA_LONG_TERM", 70.0) for i in range(1, 6)]
        invest = [_cand(f"INV{i}", "INVEST_TACTICAL", 60.0) for i in range(1, 6)]
        llm = _DsStubLLM({"pick_prefilter": '{"keep_isa": ["ISA1"], "keep_invest": ["INV1"]}'})
        kept_isa, kept_invest = await _prefilter_candidates_with_ds(cfg, llm, isa, invest)
        self.assertEqual(len(kept_isa), 5)
        self.assertEqual(len(kept_invest), 5)
        self.assertEqual(llm.calls, [])

    def test_resolve_keep_candidates_ignores_non_list(self) -> None:
        isa = [_cand("A", "ISA_LONG_TERM", 1.0)]
        self.assertEqual(_resolve_keep_candidates("A", isa, 8), [])
        self.assertEqual(_resolve_keep_candidates(None, isa, 8), [])
        self.assertEqual(_resolve_keep_candidates(["a"], isa, 8)[0].symbol, "A")


class TestSemanticAuditor(unittest.IsolatedAsyncioTestCase):
    async def test_violations_are_reported_without_touching_text(self) -> None:
        cfg = _cfg()
        ops = _RecordingOps()
        llm = _DsStubLLM(
            {
                "semantic_audit_report": (
                    '{"passed": false, "violations": ["建议撤销 ULVR.L 挂单吃 4% 利息", '
                    '"**幻觉价位** 1234.5"]}'
                )
            }
        )
        rendered = "今日建议持有 VWRP.L，现金 4% 利息。"
        result = await audit_generated_output(
            llm,
            cfg=cfg,
            scope="report",
            fact_baseline="【原始数字底座】总现金 £2,500.00",
            rendered_text=rendered,
            strategy=cfg.strategy,
            ops_notifier=ops,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.skipped)
        self.assertEqual(result.violations[0], "建议撤销 ULVR.L 挂单吃 4% 利息")
        self.assertEqual(result.violations[1], "幻觉价位 1234.5")
        self.assertEqual(len(ops.events), 1)
        self.assertEqual(ops.events[0]["severity"], "WARN")
        self.assertEqual(ops.events[0]["alert_key"], "semantic_audit_report")
        self.assertEqual(rendered, "今日建议持有 VWRP.L，现金 4% 利息。")

    async def test_pick_headline_leak_is_deterministic(self) -> None:
        cfg = _cfg()
        llm = _DsStubLLM({"semantic_audit_pick": '{"passed": true, "violations": []}'})
        result = await audit_generated_output(
            llm,
            cfg=cfg,
            scope="pick",
            fact_baseline="【原始数字底座】",
            rendered_text="本轮只买 VWRP.L",
            strategy=cfg.strategy,
            headline="本轮不推荐 ULVR.L，继续观察 VWRP.L",
            selected_symbols={"VWRP.L"},
            known_symbols={"ULVR.L"},
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("ULVR.L" in item for item in result.violations))

    async def test_timeout_and_switch_off_are_silent(self) -> None:
        cfg = _cfg()
        ops = _RecordingOps()
        result = await audit_generated_output(
            _SlowLLM(),
            cfg=cfg,
            scope="report",
            fact_baseline="底座",
            rendered_text="正文",
            strategy=cfg.strategy,
            timeout_s=0.01,
            ops_notifier=ops,
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.skipped)
        self.assertEqual(ops.events, [])

        cfg_off = _cfg(ds_semantic_audit_enabled=False)
        no_key = _NoKeyLLM()
        result_off = await audit_generated_output(
            no_key,
            cfg=cfg_off,
            scope="report",
            fact_baseline="底座",
            rendered_text="正文",
            strategy=cfg.strategy,
        )
        self.assertTrue(result_off.skipped)
        self.assertEqual(no_key.calls, [])

    async def test_warning_output_is_not_audited(self) -> None:
        cfg = _cfg()
        llm = _DsStubLLM()
        result = await audit_generated_output(
            llm,
            cfg=cfg,
            scope="report",
            fact_baseline="底座",
            rendered_text="⚠️ 账户读取失败",
            strategy=cfg.strategy,
        )
        self.assertTrue(result.skipped)
        self.assertEqual(llm.calls, [])

    async def test_ops_alert_failure_never_breaks_audit(self) -> None:
        class _BrokenOps:
            async def notify(self, **kwargs) -> bool:
                raise RuntimeError("telegram down")

        cfg = _cfg()
        llm = _DsStubLLM(
            {"semantic_audit_report": '{"passed": false, "violations": ["幻觉价位 999"]}'}
        )
        result = await audit_generated_output(
            llm,
            cfg=cfg,
            scope="report",
            fact_baseline="底座",
            rendered_text="正文",
            strategy=cfg.strategy,
            ops_notifier=_BrokenOps(),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.violations, ["幻觉价位 999"])

    def test_build_fact_baseline_carries_truth_numbers(self) -> None:
        baseline = build_fact_baseline(
            scope="report",
            portfolio=_portfolio(),
            evals=[],
            trade_ctx={"pending_orders_text": "BUY VWRP.L @ 120.50 x 10"},
            guardrails=["Invest 半导体占比超限"],
            fact_sheet="NVDA | 现价 180.12 USD",
        )
        self.assertIn("总净值 £10,000.00", baseline)
        self.assertIn("总现金 £2,500.00", baseline)
        self.assertIn("ISA：可用 1,400.00", baseline)
        self.assertIn("未成交仍计息", baseline)
        self.assertIn("Invest 半导体占比超限", baseline)
        self.assertIn("NVDA | 现价 180.12 USD", baseline)

    def test_sanitize_plain_text_strips_markdown(self) -> None:
        self.assertEqual(sanitize_plain_text("**加粗** #标签 文本", 100), "加粗 标签 文本")
        self.assertEqual(len(sanitize_plain_text("x" * 50, 10)), 10)

    def test_fact_baseline_keeps_full_fact_sheet(self) -> None:
        filler = "\n".join(f"SYM{i:02d} | 现价 {100 + i}.00 USD" for i in range(400))
        self.assertGreater(len(filler), 4000)
        baseline = build_fact_baseline(scope="report", portfolio=_portfolio(), fact_sheet=filler)
        self.assertIn("SYM399 | 现价 499.00 USD", baseline)


if __name__ == "__main__":
    unittest.main()
