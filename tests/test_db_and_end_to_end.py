"""End-to-end integration tests for SQLite persistence and AdvisorBot commands (/pos, /report, /pick, /us, /stats, /clear)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.schemas import (
    MorningReportDecision,
    PendingOrderAuditItem,
    PickActionItem,
    PickDecision,
    ReportActionItem,
    render_morning_report,
    render_pick_report,
)
from src.agents.tools import clear_grounding_cache
from src.bot.handlers import AdvisorBot
from src.bot.telegram_client import TelegramClient, split_message_into_chunks
from src.config import AppConfig, EnvSettings, UniverseSection
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
)
from src.storage.db import AdvisorDatabase


class _FakeT212(T212Client):
    def __init__(self, data: PortfolioData) -> None:
        super().__init__()
        self._data = data

    async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
        return self._data


# TODO(IMP-08 / IMP-19): [接自 IMP-05 & IMP-06] screener.py / tools.py 改用 get_histories() 时，
# 须同步为 _FakeMarketData 补齐 async def get_histories(...) 并让测试 DataFrame 包含 Open/High/Low/Close/Volume。
class _FakeMarketData(MarketDataService):
    async def get_bundles(self, symbols: list[str], include_news: bool = True, include_earnings: bool = True):
        dates = pd.date_range("2025-10-01", periods=230, freq="B")
        histories = {}
        bundles = {}
        for idx, sym in enumerate(symbols):
            base = 100.0 + np.linspace(0, 25, 230) + np.sin(np.arange(230) / 5 + idx) * 2.0
            histories[sym] = pd.DataFrame({"Close": base}, index=dates)
            bundles[sym] = TickerMarketBundle(
                symbol=sym,
                quote_type="ETF" if sym.endswith(".L") and sym in {"VWRP.L", "VUAG.L"} else "EQUITY",
                currency="GBP" if sym.endswith(".L") else "USD",
                short_name=sym,
                forward_pe=20.0 if not sym.endswith(".L") else None,
                news=[{"date": "2026-11-13", "source": "Reuters", "title": f"{sym} steady momentum"}],
                has_history=True,
            )
        return histories, bundles

    async def get_macro_snapshot(self) -> MacroSnapshot:
        return MacroSnapshot(
            fetched_at_utc="2026-11-14 09:00 UTC",
            gbp_usd=1.2750,
            us_10y_yield=4.35,
            vix=15.2,
            spy_5d_pct=1.15,
        )


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        schema = kwargs.get("response_schema")
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "MorningReportDecision":
            return MorningReportDecision(
                headline="保留4%现金缓冲，建议撤销ULVR.L高价挂单",
                top_actions=[
                    ReportActionItem(
                        account="ISA",
                        symbol="ULVR.L",
                        action="CANCEL_ORDER",
                        price_zone="4900 GBp",
                        reason="扣印花税后上行空间不足以跑赢4%现金利息",
                    )
                ],
                pending_order_audits=[
                    PendingOrderAuditItem(
                        account="ISA",
                        symbol="ULVR.L",
                        verdict="CANCEL",
                        order_summary="BUY 4900 GBp",
                        reason="目标价空间有限且含0.5%印花税，建议撤单留现金",
                    )
                ],
                daily_vetoes=["Invest半导体占比超限，冻结新开芯片仓", "ULVR.L挂单建议撤单"],
                macro_and_cash="10Y美债偏高，保留GBP 4,000现金享4%年化利息。",
                account_diagnosis="ISA科技集中，NVDA仅在ISA持有。",
                flagged_positions=["NVDA: 持仓观察，勿盲目补仓"],
                hold_summary="其余长线底仓结构稳固，继续持有不动。",
                trade_review="近30日交易克制。",
                earnings_watch=[],
            ).model_dump_json()
        if schema_name == "PickDecision":
            return PickDecision(
                headline="延续早报红线，ISA分批补盲VWRP.L",
                selected_actions=[
                    PickActionItem(
                        account="ISA",
                        symbol="VWRP.L",
                        action="BUY",
                        thesis="全球宽基零换汇零印花税分散科技集中度",
                        entry_zone="110-112 GBP",
                        stop_condition="宏观看破MA200暂停",
                        target_window="长线持有1年以上",
                        entry_price=112.0,
                    )
                ],
                rejected_or_frozen=["半导体新股（继承早报红线冻结）"],
                consistency_and_cash_note="严格遵守今日/report半导体禁加红线，仅用ISA余粮低吸零摩擦宽基。",
            ).model_dump_json()
        prompt = kwargs.get("prompt", "")
        if kwargs.get("use_google_search"):
            return f"联网新闻摘要：{prompt[:60]}"
        if "增量选股投研员" in prompt:
            return "初选提案：建议 ISA 考察 VWRP.L 补齐非美敞口。"
        if "独立风控与挂单审计官" in prompt:
            return "审计意见：ULVR.L 挂单赔率不够建议撤单；冻结半导体加仓。"
        if "组合机会分析师" in prompt:
            return "分析师意见：宏观看平，NVDA 观察，关注现金利息。"
        if "美股增量哨兵" in prompt:
            return "✅ 维持早报计划\n美股或个股简报：继续按早报策略拿住现金吃4%利息。"
        return "单聊顾问回复：已同步晨报与哨兵意见，暂无新操作。"


class _RecordingTg(TelegramClient):
    def __init__(self) -> None:
        super().__init__(bot_token="fake")
        self.messages: list[str] = []
        self.markups: list[dict | None] = []
        self.statuses: list[str] = []

    async def send_typing(self, chat_id) -> None:
        pass

    async def send_status(self, chat_id, text: str) -> int:
        self.statuses.append(text)
        return len(self.statuses)

    async def edit_status(self, chat_id, message_id, text: str) -> None:
        self.statuses.append(text)

    async def send_message(self, chat_id, text: str, reply_markup=None) -> bool:
        self.messages.append(text)
        self.markups.append(reply_markup)
        return True


def _sample_portfolio() -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-14 09:00",
        fetched_at_ms=1794646800000,
        total_combined_value=15000.0,
        total_combined_cash=4000.0,
        isa=AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=10000.0,
            investments_value=7000.0,
            investments_cost=6200.0,
            unrealized_pnl=800.0,
            total_cash=3000.0,
            available_to_trade=1775.0,
            reserved_cash=1225.0,
            cash_ratio_pct=30.0,
            pending_orders=[
                PendingOrderItem(
                    id="p-ulvr",
                    account="ISA",
                    raw_ticker="ULVRl_EQ",
                    ticker="ULVR",
                    yahoo_symbol="ULVR.L",
                    name="Unilever",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=25.0,
                    value=1225.0,
                    limit_price=4900.0,
                    currency="GBP",
                    time_in_force="GTC",
                )
            ],
            positions=[
                PositionItem(
                    account="ISA",
                    raw_ticker="NVDA_US_EQ",
                    ticker="NVDA",
                    yahoo_symbol="NVDA",
                    name="NVIDIA",
                    inst_currency="USD",
                    quantity=20,
                    avg_price=110.0,
                    cur_price=135.0,
                    current_value=7000.0,
                    total_cost=6200.0,
                    pnl=800.0,
                    fx_impact=-50.0,
                    return_pct=12.9,
                    weight_pct=70.0,
                )
            ],
        ),
        invest=AccountSnapshot(
            ok=True,
            account_label="Invest",
            total_account_value=5000.0,
            investments_value=4000.0,
            investments_cost=4100.0,
            unrealized_pnl=-100.0,
            total_cash=1000.0,
            available_to_trade=1000.0,
            cash_ratio_pct=20.0,
        ),
    )


class TestDbAndEndToEnd(unittest.IsolatedAsyncioTestCase):
    def test_telegram_chunking(self) -> None:
        long_text = ("line of text\n" * 400).strip()
        chunks = split_message_into_chunks(long_text, max_length=1000)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))

    def test_renderers_enforce_strict_char_limits(self) -> None:
        verbose_report = MorningReportDecision(
            headline="核心动作置顶测试",
            top_actions=[
                ReportActionItem(account="ISA", symbol="ULVR.L", action="CANCEL_ORDER", price_zone="4900 GBp", reason="撤单")
            ],
            pending_order_audits=[],
            daily_vetoes=["冻结半导体"],
            macro_and_cash="宏观" * 200,
            account_diagnosis="诊断" * 200,
            flagged_positions=[f"HOLDING{i}: " + ("细节" * 40) for i in range(8)],
            hold_summary="常规持有",
            trade_review="操作复盘" * 50,
            earnings_watch=[],
        )
        rendered_r = render_morning_report(verbose_report, max_chars=1500)
        self.assertLessEqual(len(rendered_r), 1500)
        self.assertTrue(rendered_r.startswith("【今日核心行动与挂单裁决】"))
        self.assertIn("CANCEL_ORDER ULVR.L", rendered_r)

        dedup_report = MorningReportDecision(
            headline="测试去重与财报优先",
            top_actions=[
                ReportActionItem(account="ISA", symbol="ULVR.L", action="CANCEL_ORDER", price_zone="4900", reason="重复项应被滤掉"),
                ReportActionItem(account="Invest", symbol="AAPL", action="TRIM", price_zone="355", reason="估值透支"),
            ],
            pending_order_audits=[
                PendingOrderAuditItem(account="ISA", symbol="ULVR.L", verdict="CANCEL", order_summary="BUY 4900 GBp", reason="空间不足"),
            ],
            daily_vetoes=["冻结科技"],
            macro_and_cash="10Y高息。",
            account_diagnosis="Invest偏重。",
            flagged_positions=["AAPL [Invest]: 重复不该显示", "NVDA: 暴跌观望"],
            hold_summary="其余持有",
            trade_review="近30天平稳",
            earnings_watch=["NVDA(11-17)"],
        )
        rendered_dedup = render_morning_report(dedup_report, max_chars=1500)
        self.assertNotIn("CANCEL_ORDER ULVR.L", rendered_dedup)
        self.assertNotIn("AAPL [Invest]: 重复不该显示", rendered_dedup)
        self.assertIn("NVDA(11-17)", rendered_dedup)
        self.assertIn("NVDA: 暴跌观望", rendered_dedup)

        verbose_pick = PickDecision(
            headline="精选结论置顶",
            selected_actions=[
                PickActionItem(
                    account="ISA",
                    symbol="VWRP.L",
                    action="BUY",
                    thesis="分散" * 50,
                    entry_zone="110-112",
                    stop_condition="MA200",
                    target_window="1年",
                    entry_price=112.0,
                )
            ],
            rejected_or_frozen=["否决" * 80, "冻结" * 80],
            consistency_and_cash_note="现金" * 80,
        )
        rendered_p = render_pick_report(verbose_pick, max_chars=1000)
        self.assertLessEqual(len(rendered_p), 1000)
        self.assertTrue(rendered_p.startswith("【本轮精选行动（四要素直给）】"))

    async def test_bot_commands_end_to_end(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                db = AdvisorDatabase(Path(tmpdir) / "e2e.db")
                await db.init()

                cfg = AppConfig(
                    env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                    universe=UniverseSection(isa_long_term=["VWRP.L", "AZN.L"], invest_tactical=["AMD"]),
                )
                t212 = _FakeT212(_sample_portfolio())
                md = _FakeMarketData()
                llm = _RecordingLLM()
                tg = _RecordingTg()
                bot = AdvisorBot(cfg, t212, md, llm, db, tg)

                # 1. Unauthorized chat_id ignored
                await bot.handle_update({"message": {"chat": {"id": "111"}, "text": "/pos"}}, wait=True)
                self.assertEqual(tg.messages, [])

                # 2. /pos instant output (0 LLM calls)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/pos"}}, wait=True)
                self.assertEqual(len(llm.calls), 0)
                self.assertIn("NVDA", tg.messages[-1])
                self.assertIn("15,000.00", tg.messages[-1])

                clear_grounding_cache()
                # 3. /report runs forced grounding batches (macro + ULVR.L UK + NVDA US) + 3-step pipeline <= 1500 chars
                calls_before_report = len(llm.calls)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/report"}}, wait=True)
                report_calls = llm.calls[calls_before_report:]
                grounding_calls = [c for c in report_calls if c.get("usage_label") == "report_grounding"]
                pipeline_calls = [c for c in report_calls if c.get("usage_label") != "report_grounding"]
                self.assertEqual(len(grounding_calls), 3)
                self.assertTrue(all(c.get("use_google_search") for c in grounding_calls))
                self.assertTrue(all(c.get("model") == "gemini-4-hydrogen" for c in grounding_calls))
                self.assertEqual(
                    [(c["usage_label"], c["model"], c["thinking_level"]) for c in pipeline_calls],
                    [
                        ("report_analyst", "gemini-4-hydrogen", "HIGH"),
                        ("report_auditor", "gemini-4-argon", "HIGH"),
                        ("report_pm", "gemini-4-argon", "HIGH"),
                    ],
                )
                self.assertTrue(all(c.get("system_instruction") for c in pipeline_calls))
                self.assertIn("实时 Google Search", pipeline_calls[0]["prompt"])
                report_msg = tg.messages[-1]
                self.assertLessEqual(len(report_msg), 1500)
                self.assertTrue(report_msg.startswith("【今日核心行动与挂单裁决】"))
                self.assertIn("CANCEL", report_msg)
                guardrails = await db.get_daily_guardrails()
                self.assertIn("Invest半导体占比超限，冻结新开芯片仓", guardrails)

                # 4. /us Delta Sentinel receives full /report, filters out non-US orders (ULVR.L), uses us_sentinel_model
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/us"}}, wait=True)
                self.assertIn("美股或个股简报", tg.messages[-1])
                us_call = llm.calls[-1]
                self.assertEqual(us_call["usage_label"], "us_sentinel")
                self.assertEqual(us_call["model"], cfg.models.us_sentinel_model)
                self.assertEqual(us_call["thinking_level"], "MEDIUM")
                self.assertTrue(us_call.get("system_instruction"))
                self.assertIn("【活动挂单裁决】", us_call["prompt"])
                self.assertIn("【美股活动挂单】：\n（当前无未成交活动挂单）", us_call["prompt"])

                # 5. /pick inherits /report guardrails, runs forced grounding + 2 calls (Hydrogen HIGH -> Argon HIGH), <= 1000 chars
                calls_before = len(llm.calls)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/pick 英国ETF"}}, wait=True)
                pick_calls = llm.calls[calls_before:]
                pick_grounding = [c for c in pick_calls if c.get("usage_label") == "pick_grounding"]
                pick_pipeline = [c for c in pick_calls if c.get("usage_label") != "pick_grounding"]
                self.assertGreaterEqual(len(pick_grounding), 2)
                self.assertTrue(all(c.get("model") == "gemini-3.5-flash-lite" for c in pick_grounding))
                self.assertEqual(
                    [(c["usage_label"], c["model"], c["thinking_level"]) for c in pick_pipeline],
                    [
                        ("pick_researcher", "gemini-4-hydrogen", "HIGH"),
                        ("pick_critic", "gemini-4-argon", "HIGH"),
                    ],
                )
                self.assertIn("Invest半导体占比超限", pick_pipeline[0]["prompt"])
                self.assertIn("实时 Google Search", pick_pipeline[0]["prompt"])
                pick_msg = tg.messages[-1]
                self.assertLessEqual(len(pick_msg), 1000)
                self.assertTrue(pick_msg.startswith("【本轮精选行动（四要素直给）】"))
                self.assertIn("VWRP.L", pick_msg)

                # 6. /stats reflects saved structured pick
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/stats"}}, wait=True)
                self.assertIn("BUY VWRP.L", tg.messages[-1])

                # 6b. /usage reports token & search summary + active routing cleanly without USD estimates or Markdown #/**
                await db.log_llm_usage(
                    {
                        "call_label": "report_pm",
                        "model": "gemini-4-argon",
                        "thinking_level": "HIGH",
                        "api_rounds": 1,
                        "prompt_tokens": 2500,
                        "cached_tokens": 0,
                        "output_tokens": 400,
                        "thought_tokens": 900,
                        "total_tokens": 3800,
                        "estimated_cost_usd": 0.0,
                        "latency_ms": 1200,
                        "success": True,
                    }
                )
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/usage"}}, wait=True)
                usage_msg = tg.messages[-1]
                self.assertIn("Gemini API Token 与搜索用量看板", usage_msg)
                self.assertIn("report_pm", usage_msg)
                self.assertIn("gemini-4-argon", usage_msg)
                self.assertNotIn("$", usage_msg)
                self.assertNotIn("**", usage_msg)
                self.assertNotIn("#", usage_msg)

                # 7. Free chat receives BOTH /report AND /us sentinel supplement, omits bulky FactSheet, persists to chat_history, /clear resets
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "现在能加仓NVDA吗"}}, wait=True)
                chat_call = llm.calls[-1]
                self.assertEqual(chat_call["usage_label"], "chat")
                self.assertEqual(chat_call["model"], "gemini-4-hydrogen")
                self.assertEqual(chat_call["thinking_level"], "LOW")
                self.assertTrue(chat_call.get("system_instruction"))
                self.assertIn("【今日美股开盘前哨补充】", chat_call["prompt"])
                self.assertNotIn("【持仓量化技术面、目标价与财报雷达】", chat_call["prompt"])
                self.assertNotIn("【Python 规则引擎：加仓 / 减仓 / 止损复核信号】", chat_call["prompt"])
                turns = await db.get_chat_history()
                self.assertEqual(len(turns), 1)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/clear"}}, wait=True)
                self.assertEqual(await db.get_chat_history(), [])

        await asyncio.wait_for(_run(), timeout=15.0)


if __name__ == "__main__":
    unittest.main()
