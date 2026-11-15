"""End-to-end integration tests for SQLite persistence and AdvisorBot commands (/pos, /report, /pick, /us, /stats, /clear)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.schemas import (
    FlaggedPositionItem,
    MorningReportDecision,
    PendingOrderAuditItem,
    PickActionItem,
    PickDecision,
    ReportActionItem,
    _fit_max_chars,
    parse_structured_json,
    render_morning_report,
    render_pick_report,
    trim_text_at_line_boundary,
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


# TODO(IMP-11 / IMP-19): [接自 IMP-05 & IMP-06] tools.py 改用 get_histories() 与最终集成验收时复核 _FakeMarketData。
class _FakeMarketData(MarketDataService):
    async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        dates = pd.date_range("2025-10-01", periods=230, freq="B")
        histories = {}
        for sym in symbols:
            seed = zlib.crc32(sym.encode("utf-8"))
            freq_div = 3.5 + (seed % 9) * 0.7
            phase = ((seed >> 4) % 23) * 0.35
            base = 100.0 + np.linspace(0, 25, 230) + np.sin(np.arange(230) / freq_div + phase) * 2.0
            histories[sym] = pd.DataFrame(
                {
                    "Open": base * 0.996,
                    "High": base * 1.012,
                    "Low": base * 0.988,
                    "Close": base,
                    "Volume": np.full(230, 1_500_000.0),
                },
                index=dates,
            )
        return histories

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
        isin_map: dict[str, str] | None = None,
    ):
        histories = await self.get_histories(symbols, period="1y")
        bundles = {}
        for sym in symbols:
            bundles[sym] = TickerMarketBundle(
                symbol=sym,
                quote_type="ETF" if sym.endswith(".L") and sym in {"VWRP.L", "VUAG.L"} else "EQUITY",
                currency="GBP" if sym.endswith(".L") else "USD",
                short_name=sym,
                isin=(isin_map or {}).get(sym, ""),
                forward_pe=20.0 if not sym.endswith(".L") else None,
                news=[{"date": "2026-11-13", "source": "Reuters", "title": f"{sym} steady momentum"}]
                if include_news
                else [],
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

    def test_imp10_structured_flagged_positions_and_dedup(self) -> None:
        raw_json = (
            '{"headline":"结构化异动测试","top_actions":['
            '{"account":"ISA","symbol":"ULVR.L","action":"HOLD","price_zone":"4500 GBp","reason":"继续持有"}],'
            '"pending_order_audits":[{"account":"Invest","symbol":"RR.L","verdict":"KEEP","order_summary":"BUY 500 GBp","reason":"留存观察"}],'
            '"daily_vetoes":["控制半导体仓位"],"macro_and_cash":"利率稳定","account_diagnosis":"仓位均衡",'
            '"flagged_positions":[{"symbol":"ULVR","summary":"重复应剔除"},{"symbol":"RR.L","summary":"重复挂单应剔除"},'
            '{"symbol":"TSM","summary":"TSM: 晶圆放量上攻"},{"symbol":"NVDA","summary":"高位横盘"}],'
            '"hold_summary":"其余持有","trade_review":"交易平稳","earnings_watch":[]}'
        )
        parsed = parse_structured_json(raw_json, MorningReportDecision)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIsInstance(parsed.flagged_positions[0], FlaggedPositionItem)
        rendered = render_morning_report(parsed, max_chars=1500)
        self.assertNotIn("重复应剔除", rendered)
        self.assertNotIn("重复挂单应剔除", rendered)
        self.assertIn("• TSM: 晶圆放量上攻", rendered)
        self.assertNotIn("TSM: TSM:", rendered)
        self.assertIn("• NVDA: 高位横盘", rendered)

        # Legacy string coercion + empty sec3 must not emit orphan section header
        legacy_dec = MorningReportDecision(
            headline="全被去重测试",
            top_actions=[ReportActionItem(account="ISA", symbol="AAPL", action="TRIM", price_zone="230", reason="减仓")],
            daily_vetoes=["严守纪律"],
            macro_and_cash="现金4%",
            account_diagnosis="均衡",
            flagged_positions=["AAPL [Invest]: 应被去重", "2026年财报季观望"],
            hold_summary="",
            trade_review="",
            earnings_watch=[],
        )
        self.assertEqual(legacy_dec.flagged_positions[0].symbol, "AAPL")
        self.assertEqual(legacy_dec.flagged_positions[1].symbol, "")
        rendered_legacy = render_morning_report(legacy_dec, max_chars=1500)
        self.assertNotIn("应被去重", rendered_legacy)
        self.assertIn("• 2026年财报季观望", rendered_legacy)

        only_dedup = legacy_dec.model_copy(update={"flagged_positions": ["AAPL: 仅重复"]})
        rendered_no_sec3 = render_morning_report(only_dedup, max_chars=1500)
        self.assertNotIn("【财报预警、复盘与重点异动】", rendered_no_sec3)

    def test_imp10_risk_override_guardrail_rendering(self) -> None:
        from src.config import StrategySection
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        portfolio = _sample_portfolio()
        report_dec = MorningReportDecision(
            headline="测试风控护栏高亮",
            top_actions=[
                ReportActionItem(account="ISA", symbol="NVDA", action="ADD", price_zone="138", reason="趋势强劲"),
            ],
            daily_vetoes=["冻结半导体加仓"],
            macro_and_cash="VIX高企",
            account_diagnosis="ISA偏重",
            hold_summary="其余持有",
            trade_review="操作克制",
        )
        res_report = apply_risk_override_guardrail(
            report_dec,
            portfolio=portfolio,
            macro=MacroSnapshot(fetched_at_utc="2026-11-15T07:25:00Z", vix=31.0),
            strategy=StrategySection(),
        )
        self.assertTrue(res_report.applied)
        # Works both when passing res_report.decision and when passing original report_dec with risk_override
        rendered_r1 = render_morning_report(res_report.decision, max_chars=1500, risk_override=res_report)
        rendered_r2 = render_morning_report(report_dec, max_chars=1500, risk_override=res_report)
        for out in (rendered_r1, rendered_r2):
            self.assertLessEqual(len(out), 1500)
            self.assertIn("[🛡️风控护栏修正]", out)
            self.assertIn("ADD→HOLD", out)
            self.assertNotIn("**", out)
            self.assertNotIn("#", out)

        # PickDecision: 4 existing rejections + 2 guardrail rejections must ALL highlight the 2 guardrails in sec1
        pick_dec = PickDecision(
            headline="精选护栏测试",
            selected_actions=[
                PickActionItem(
                    account="ISA",
                    symbol="TSM",
                    action="BUY",
                    thesis="晶圆龙头",
                    entry_zone="180",
                    stop_condition="165",
                    target_window="210",
                    entry_price=180.0,
                ),
                PickActionItem(
                    account="Invest",
                    symbol="META",
                    action="ADD",
                    thesis="估值合理",
                    entry_zone="550",
                    stop_condition="510",
                    target_window="620",
                    entry_price=550.0,
                ),
            ],
            rejected_or_frozen=["旧驳回1", "旧驳回2", "旧驳回3", "旧驳回4"],
            consistency_and_cash_note="现金充足",
        )
        res_pick = apply_risk_override_guardrail(
            pick_dec,
            portfolio=portfolio,
            macro=MacroSnapshot(fetched_at_utc="2026-11-15T07:25:00Z", vix=18.0),
            strategy=StrategySection(),
            high_correlation_symbols={"META"},
            daily_vetoes=["禁止买入TSM等高波动标的"],
        )
        self.assertTrue(res_pick.applied)
        rendered_p = render_pick_report(res_pick.decision, max_chars=1000, risk_override=res_pick)
        self.assertLessEqual(len(rendered_p), 1000)
        self.assertEqual(rendered_p.count("[🛡️风控护栏修正]"), 2)
        self.assertIn("TSM(BUY→拦截)", rendered_p)
        self.assertIn("META(ADD→拦截)", rendered_p)
        self.assertIn("旧驳回1", rendered_p)
        self.assertNotIn("**", rendered_p)
        self.assertNotIn("#", rendered_p)

        # Fallback: even if caller passes res_pick.decision without risk_override, guardrail tags are still surfaced
        rendered_p_no_ro = render_pick_report(res_pick.decision, max_chars=1000)
        self.assertEqual(rendered_p_no_ro.count("[🛡️风控护栏修正]"), 2)

    def test_imp10_fit_max_chars_prevents_orphan_headers(self) -> None:
        # 1) Trimming lines inside tail section exposes a sub-header -> must be stripped
        sec_a = "【核心】\n• 行A"
        sec_b = "【次要】\n【子标题】\n• 第一条较长内容\n• 第二条较长内容"
        fitted = _fit_max_chars([sec_a, sec_b], max_chars=22)
        self.assertLessEqual(len(fitted), 22)
        self.assertFalse(fitted.splitlines()[-1].strip().startswith("【"))
        self.assertEqual(fitted, sec_a)

        # 2) Hard-slice of single section landing on or inside trailing 【...】 header
        single = "【核心行动】\n• 保留内容1234567890\n【活动挂单裁决】\n• 挂单明细"
        # Length up to end of "【活动挂单裁决】" is 6 + 16 + 9 = 31; max_chars=30 slices into "【活动挂单裁决"
        hard_sliced = _fit_max_chars([single], max_chars=30)
        self.assertLessEqual(len(hard_sliced), 30)
        self.assertNotIn("【活动", hard_sliced)
        self.assertTrue(hard_sliced.endswith("…"))

        # 3) Real render_morning_report with tight limit never leaves a dangling 【...】 line
        dec = MorningReportDecision(
            headline="紧凑裁剪测试",
            top_actions=[ReportActionItem(account="ISA", symbol="NVDA", action="HOLD", price_zone="130", reason="观察")],
            pending_order_audits=[
                PendingOrderAuditItem(account="ISA", symbol="ULVR.L", verdict="KEEP", order_summary="BUY 4500", reason="保留")
            ],
            daily_vetoes=["控制仓位"],
            macro_and_cash="宏观稳定",
            account_diagnosis="账户稳健",
            flagged_positions=["AMD: 观察"],
            hold_summary="其余持有",
            trade_review="操作克制",
        )
        for limit in (80, 120, 180, 250):
            out = render_morning_report(dec, max_chars=limit)
            self.assertLessEqual(len(out), limit)
            last_line = out.splitlines()[-1].strip()
            self.assertFalse(
                last_line.startswith("【") and last_line.endswith("】"),
                f"Orphan header at limit={limit}: {out!r}",
            )

    async def test_authorized_length_ceilings_us_chat_deep_and_tool_compaction(self) -> None:
        from src.agents.chat_agent import answer_user_chat
        from src.agents.report_agent import generate_us_sentinel
        from src.agents.tools import MarketToolkit

        # 1) render_morning_report default ceiling is now 1800 chars (1650-char report preserved intact)
        medium_report = MorningReportDecision(
            headline="1800字安全天花板验证",
            top_actions=[ReportActionItem(account="ISA", symbol="NVDA", action="HOLD", price_zone="138", reason="趋势完整")],
            daily_vetoes=["控制半导体仓位"],
            macro_and_cash="宏观" * 180,
            account_diagnosis="诊断" * 180,
            flagged_positions=[FlaggedPositionItem(symbol=f"SYM{i}", summary="观察" * 12) for i in range(4)],
            hold_summary="其余长线底仓持有",
            trade_review="近30天平稳",
        )
        rendered_default = render_morning_report(medium_report)
        self.assertGreater(len(rendered_default), 1000)
        self.assertLessEqual(len(rendered_default), 1800)
        self.assertIn("其余长线底仓持有", rendered_default)

        # 2) trim_text_at_line_boundary cleanly trims on newline without leaving dangling 【...】
        multiline = "✅ 维持早报计划\n• 要点1：正常\n【分段】\n• 要点2：" + ("长内容" * 150)
        trimmed = trim_text_at_line_boundary(multiline, 450)
        self.assertLessEqual(len(trimmed), 450)
        self.assertNotIn("【分段】", trimmed)

        # 3) /us sentinel: calm day <= 450, anomaly day (🚨) allows up to 600 chars
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "ceilings.db")
            await db.init()
            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
            t212 = _FakeT212(_sample_portfolio())
            md = _FakeMarketData()

            class _UsSentinelStubLLM:
                def __init__(self, reply: str) -> None:
                    self.reply = reply
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    return self.reply

            anomaly_520 = "🚨 盘前增量修正\n• NVDA 跳空低开5%，原定加仓暂停。\n• " + ("核心逻辑修正" * 80)
            self.assertGreater(len(anomaly_520), 450)
            self.assertLess(len(anomaly_520), 600)
            llm_anom = _UsSentinelStubLLM(anomaly_520)
            out_anom = await generate_us_sentinel(cfg, t212, md, llm_anom, db)
            self.assertEqual(out_anom, anomaly_520)

            calm_long = "✅ 维持早报计划\n• 观察1：盘前平稳。\n• 观察2：" + ("细节" * 240)
            llm_calm = _UsSentinelStubLLM(calm_long)
            out_calm = await generate_us_sentinel(cfg, t212, md, llm_calm, db)
            self.assertLessEqual(len(out_calm), 450)
            self.assertEqual(out_calm, "✅ 维持早报计划\n• 观察1：盘前平稳。")

            # 4) chat_agent passes >350 chars of /pick radar (up to 1050) & /deep prompt allows 1000~1200 chars
            long_pick = "【本轮精选行动】\n" + ("A" * 400) + "\nTAIL_MARKER_PRESERVED"
            await db.save_pick_radar("test", "research", long_pick, "2026-11-15 17:00")
            llm_chat = _UsSentinelStubLLM("OK")
            await answer_user_chat("/deep 分析一下刚才的 pick", cfg, t212, md, llm_chat, db)
            self.assertIn("TAIL_MARKER_PRESERVED", llm_chat.calls[-1]["prompt"])
            self.assertIn("1000~1200 字以内", llm_chat.calls[-1]["system_instruction"])

            await answer_user_chat("普通问题", cfg, t212, md, llm_chat, db)
            self.assertIn("500 字以内", llm_chat.calls[-1]["system_instruction"])

            # 5) MarketToolkit.inspect_ticker caps news to 3 items
            class _ManyNewsMarketData(_FakeMarketData):
                async def get_bundles(self, symbols, include_news=True, include_earnings=True, **kwargs):
                    hist, bmap = await super().get_bundles(symbols, include_news=include_news, include_earnings=include_earnings, **kwargs)
                    for b in bmap.values():
                        b.news = [{"date": f"2026-11-0{i}", "title": f"News {i}", "publisher": "Test"} for i in range(1, 9)]
                    return hist, bmap

            toolkit = MarketToolkit(_ManyNewsMarketData(), _sample_portfolio())
            res_tool = await toolkit.execute("inspect_ticker", {"symbol": "NVDA"})
            self.assertEqual(len(res_tool["news"]), 3)

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
                # 3. /report runs forced grounding batches (macro + ULVR.L UK + NVDA US) + 3-step pipeline <= 1800 chars
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
                self.assertLessEqual(len(report_msg), 1800)
                self.assertTrue(report_msg.startswith("【今日核心行动与挂单裁决】"))
                self.assertIn("CANCEL", report_msg)
                guardrails = await db.get_daily_guardrails()
                self.assertIn("Invest半导体占比超限，冻结新开芯片仓", guardrails)
                saved_pack = await db.get_context_pack(allow_fallback=False)
                self.assertIsNotNone(saved_pack)
                self.assertIn("NVDA", saved_pack["symbols"])
                self.assertIn("ULVR.L", saved_pack["symbols"])

                # 4. /us Delta Sentinel receives full /report, filters out non-US orders (ULVR.L), calm day uses LOW without extra search
                calls_before_us = len(llm.calls)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/us"}}, wait=True)
                self.assertIn("美股或个股简报", tg.messages[-1])
                us_calls = llm.calls[calls_before_us:]
                self.assertEqual(len(us_calls), 1)
                us_call = us_calls[-1]
                self.assertEqual(us_call["usage_label"], "us_sentinel")
                self.assertEqual(us_call["model"], cfg.models.us_sentinel_model)
                self.assertEqual(us_call["thinking_level"], "LOW")
                self.assertIsNone(us_call.get("tools"))
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

    async def test_imp12_context_pack_disagreement_guardrail_fallback_snapshot_and_cash_diff(self) -> None:
        from unittest.mock import patch
        from src.agents.report_agent import _compute_portfolio_diff_text, generate_morning_report
        from src.data_providers.t212_client import HistoricalDividendItem, HistoricalTransactionItem

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp12_report.db")
            await db.init()

            # Seed an earlier snapshot <4h ago + settled recommendation lesson for NVDA
            now_ms = int(1794646800000)
            prev_port = _sample_portfolio().model_copy(
                update={
                    "fetched_at_uk": "2026-11-14 07:30",
                    "fetched_at_ms": now_ms - 90 * 60 * 1000,
                    "total_combined_value": 14000.0,
                }
            )
            await db.save_snapshot(prev_port)

            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-11-01 10:00",
                        "created_at_ms": now_ms - 10 * 86400 * 1000,
                        "account": "ISA",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 120.0,
                        "thesis": "AI算力需求延续",
                        "benchmark_symbol": "SPY",
                    }
                ]
            )
            pending = await db.get_pending_settlements(limit=5)
            await db.settle_recommendation(
                pending[0]["id"],
                benchmark_symbol="SPY",
                settled_at_uk="2026-11-08 21:30",
                raw_return_pct=8.5,
                alpha_pct=5.2,
                reflection_lesson="回踩MA20介入胜率高，勿在RSI超买后追涨。",
            )

            # Current portfolio is fallback -> must NOT overwrite portfolio_snapshots, and has external DEPOSIT + dividend
            cur_port = _sample_portfolio()
            cur_port.is_fallback = True
            cur_port.fallback_source = "memory"
            cur_port.fallback_note = "T212 503 fallback"
            cur_port.isa.positions[0].isin = "US67066G1040"
            cur_port.isa.positions[0].created_at = "2026-01-10T14:30:00Z"
            cur_port.isa.recent_transactions = [
                HistoricalTransactionItem(
                    id="tx-dep",
                    account="ISA",
                    tx_type="DEPOSIT",
                    amount=800.0,
                    currency="GBP",
                    date_time="2026-11-14T08:00:00Z",
                    date_time_ms=now_ms - 30 * 60 * 1000,
                    uk_time_str="2026-11-14 08:00",
                )
            ]
            cur_port.isa.recent_dividends = [
                HistoricalDividendItem(
                    id="dv-1",
                    account="ISA",
                    raw_ticker="ULVRl_EQ",
                    ticker="ULVR",
                    yahoo_symbol="ULVR.L",
                    name="Unilever",
                    amount=45.0,
                    currency="GBP",
                    paid_on="2026-11-14T08:15:00Z",
                    paid_on_ms=now_ms - 20 * 60 * 1000,
                    uk_time_str="2026-11-14 08:15",
                )
            ]

            diff = _compute_portfolio_diff_text(cur_port, prev_port)
            self.assertIn("不足4h基准已回退最近快照", diff)
            self.assertIn("外部出入金 +800.00 GBP", diff)
            self.assertIn("剔除出入金后净值变动约 +200.00 GBP", diff)
            self.assertIn("区间派息入账 +45.00 GBP", diff)

            class _Imp12MarketData(_FakeMarketData):
                def __init__(self) -> None:
                    super().__init__()
                    self.seen_isin_maps: list[dict[str, str] | None] = []

                async def get_bundles(
                    self,
                    symbols: list[str],
                    include_news: bool = True,
                    include_earnings: bool = True,
                    include_info: bool = True,
                    isin_map: dict[str, str] | None = None,
                ):
                    self.seen_isin_maps.append(isin_map)
                    hist, bmap = await super().get_bundles(
                        symbols,
                        include_news=include_news,
                        include_earnings=include_earnings,
                        include_info=include_info,
                        isin_map=isin_map,
                    )
                    if "NVDA" in bmap:
                        bmap["NVDA"].long_name = "NVIDIA Corporation"
                        bmap["NVDA"].exchange = "NMS"
                        bmap["NVDA"].sector = "Technology"
                        bmap["NVDA"].ex_dividend_date = "2026-11-16"
                        bmap["NVDA"].days_to_ex_dividend = 1
                        bmap["NVDA"].identity_summary = bmap["NVDA"].build_identity_anchor(
                            isin_override=(isin_map or {}).get("NVDA")
                        )
                    return hist, bmap

                async def get_macro_snapshot(self) -> MacroSnapshot:
                    return MacroSnapshot(
                        fetched_at_utc="2026-11-14 09:00 UTC",
                        gbp_usd=1.2750,
                        us_10y_yield=4.35,
                        vix=31.5,
                        spy_5d_pct=-2.8,
                    )

            class _AggressiveReportLLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    if kwargs.get("response_schema") is MorningReportDecision:
                        return MorningReportDecision(
                            headline="激进加仓NVDA测试护栏拦截",
                            top_actions=[
                                ReportActionItem(
                                    account="ISA",
                                    symbol="NVDA",
                                    action="ADD",
                                    price_zone="135 USD",
                                    reason="动量强劲继续加仓",
                                )
                            ],
                            daily_vetoes=["控制半导体仓位"],
                            macro_and_cash="VIX偏高",
                            account_diagnosis="ISA集中度较高",
                            flagged_positions=[],
                            hold_summary="其余持有",
                            trade_review="操作克制",
                        ).model_dump_json()
                    return "分析师/审计摘要"

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
            )
            t212 = _FakeT212(cur_port)
            md = _Imp12MarketData()
            llm = _AggressiveReportLLM()

            with patch(
                "src.agents.report_agent.get_market_calendar_alerts",
                return_value=["🇺🇸 今日美股半日市（13:00 ET / 18:00 UK 提前收盘）"],
            ):
                report = await generate_morning_report(cfg, t212, md, llm, db)

            # 1) RiskOverride guardrail downgraded ADD NVDA under high VIX + overweight & highlighted tag
            self.assertIn("[🛡️风控护栏修正]", report)
            self.assertIn("HOLD NVDA", report)

            # 2) isin_map passed to get_bundles; Disagreement + calendar alerts injected into PM prompt
            self.assertTrue(any(m and m.get("NVDA") == "US67066G1040" for m in md.seen_isin_maps))
            pm_call = next(c for c in llm.calls if c.get("usage_label") == "report_pm")
            self.assertIn("Disagreement Matrix", pm_call["prompt"])
            self.assertIn("美股半日市", pm_call["prompt"])
            self.assertIn("外部出入金 +800.00 GBP", pm_call["prompt"])

            # 3) Fallback portfolio did NOT write a second row to portfolio_snapshots
            latest_snap = await db.get_latest_snapshot()
            self.assertIsNotNone(latest_snap)
            self.assertEqual(latest_snap.fetched_at_ms, prev_port.fetched_at_ms)

            # 4) AnalysisContextPack persisted with identity, technical (CYQ/weekly/ex_div), disagreements & lessons
            pack = await db.get_context_pack(allow_fallback=False)
            self.assertIsNotNone(pack)
            self.assertEqual(pack["pack_version"], "1.0")
            self.assertTrue(pack["is_fallback"])
            nvda_pack = pack["symbols"]["NVDA"]
            self.assertIn("ISIN:US67066G1040", nvda_pack["identity_summary"])
            self.assertIsNotNone(nvda_pack["technical"])
            self.assertIn("cyq_benefit_part_pct", nvda_pack["technical"])
            self.assertIn("weekly_bullish", nvda_pack["technical"])
            self.assertEqual(nvda_pack["technical"]["days_to_ex_dividend"], 1)
            self.assertEqual(len(nvda_pack["lessons"]), 1)
            self.assertIn("回踩MA20", nvda_pack["lessons"][0]["reflection_lesson"])

    async def test_imp12_us_sentinel_anomaly_gate_calm_vs_anomaly(self) -> None:
        from src.agents.report_agent import generate_us_sentinel

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp12_us_gate.db")
            await db.init()

            class _GateMarketData(_FakeMarketData):
                def __init__(self, nvda_jump_pct: float = 0.0, vix_jump_pct: float = 0.0) -> None:
                    super().__init__()
                    self.nvda_jump_pct = nvda_jump_pct
                    self.vix_jump_pct = vix_jump_pct

                async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                    dates = pd.date_range("2025-10-01", periods=230, freq="B")
                    out: dict[str, pd.DataFrame] = {}
                    for sym in symbols:
                        closes = np.full(230, 100.0)
                        if sym == "NVDA" and self.nvda_jump_pct != 0.0:
                            closes[-1] = 100.0 * (1.0 + self.nvda_jump_pct / 100.0)
                        elif sym == "^VIX":
                            closes = np.full(230, 16.0)
                            if self.vix_jump_pct != 0.0:
                                closes[-1] = 16.0 * (1.0 + self.vix_jump_pct / 100.0)
                        out[sym] = pd.DataFrame(
                            {
                                "Open": closes * 0.998,
                                "High": closes * 1.005,
                                "Low": closes * 0.995,
                                "Close": closes,
                                "Volume": np.full(230, 1_200_000.0),
                            },
                            index=dates,
                        )
                    return out

            class _GateLLM:
                def __init__(self, reply: str) -> None:
                    self.reply = reply
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    if kwargs.get("use_google_search"):
                        return "美股异动检索结果：NVDA 财报催化跳空。"
                    return self.reply

            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
            t212 = _FakeT212(_sample_portfolio())

            # 1) Calm gate: NVDA 0% move, VIX 0% move -> skips grounding, uses LOW, no tools, trims <= 450
            clear_grounding_cache()
            calm_llm = _GateLLM("✅ 维持早报计划\n• 观察1：盘前平稳。\n• 观察2：" + ("细节" * 240))
            calm_out = await generate_us_sentinel(cfg, t212, _GateMarketData(), calm_llm, db)
            self.assertLessEqual(len(calm_out), 450)
            self.assertEqual(len(calm_llm.calls), 1)
            self.assertEqual(calm_llm.calls[0]["thinking_level"], "LOW")
            self.assertIsNone(calm_llm.calls[0]["tools"])
            self.assertIn("市场与美股持仓均平稳", calm_llm.calls[0]["prompt"])

            # 2) Stock move anomaly (+4.5% >= 1.8%): triggers us_grounding + MEDIUM + tools + <= 600 ceiling
            clear_grounding_cache()
            anom_text = "🚨 盘前增量修正\n• NVDA 大幅异动，按规则重新核验。\n• " + ("归因分析" * 120)
            anom_llm = _GateLLM(anom_text)
            anom_out = await generate_us_sentinel(cfg, t212, _GateMarketData(nvda_jump_pct=4.5), anom_llm, db)
            self.assertLessEqual(len(anom_out), 600)
            self.assertGreater(len(anom_out), 450)
            grounding_calls = [c for c in anom_llm.calls if c.get("usage_label") == "us_grounding"]
            sentinel_call = next(c for c in anom_llm.calls if c.get("usage_label") == "us_sentinel")
            self.assertEqual(len(grounding_calls), 1)
            self.assertIn("NVDA", grounding_calls[0]["prompt"])
            self.assertEqual(sentinel_call["thinking_level"], "MEDIUM")
            self.assertIsNotNone(sentinel_call["tools"])
            self.assertIn("本地异动门控：已触发深度复核", sentinel_call["prompt"])

            # 3) VIX spike anomaly (+8.0% >= 4.0%) even with 0% stock move: triggers anomaly gate + MEDIUM
            clear_grounding_cache()
            vix_llm = _GateLLM("🚨 盘前增量修正\n• VIX 急升，暂停追高。")
            vix_out = await generate_us_sentinel(cfg, t212, _GateMarketData(vix_jump_pct=8.0), vix_llm, db)
            self.assertTrue(vix_out.startswith("🚨"))
            sentinel_vix = next(c for c in vix_llm.calls if c.get("usage_label") == "us_sentinel")
            self.assertEqual(sentinel_vix["thinking_level"], "MEDIUM")
            self.assertIn("VIX 日变动 +8.00%", sentinel_vix["prompt"])

    async def test_imp13_pick_reuse_lessons_context_and_post_screen_guardrails(self) -> None:
        from unittest.mock import patch
        from src.agents.pick_agent import generate_pick_radar
        from src.config import StrategySection

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp13_pick.db")
            await db.init()

            # Seed morning report with daily_vetoes + AnalysisContextPack (including disagreements & NVDA identity/lesson)
            now_ms = int(1794646800000)
            seeded_pack = {
                "pack_version": "1.0",
                "disagreement_text": "NVDA: 多空分歧（趋势多头 vs 获利盘过高，严禁盲目追高）",
                "symbols": {
                    "NVDA": {
                        "identity_summary": "NVDA | NVIDIA Corp | NASDAQ | EQUITY | USD | ISIN:US67066G1040",
                        "lessons": [
                            {
                                "created_at_uk": "2026-11-05 16:30",
                                "account": "ISA",
                                "symbol": "NVDA",
                                "action": "ADD",
                                "raw_return_pct": -3.1,
                                "alpha_pct": -4.2,
                                "reflection_lesson": "财报前两周追高易遇获利回吐，须等待回踩MA20。",
                            }
                        ],
                    }
                },
            }
            await db.save_morning_report(
                "【今日早报】",
                "谨慎持币吃息，禁止芯片追高",
                guardrails=["禁止买入AMD等高波动芯片"],
                context_pack=seeded_pack,
            )

            # Seed settled T+7 reflection lesson for AZN.L in candidate pool
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-11-01 16:30",
                        "created_at_ms": now_ms - 10 * 86400 * 1000,
                        "account": "ISA",
                        "symbol": "AZN.L",
                        "action": "BUY",
                        "entry_price": 12000.0,
                        "thesis": "医药防御配置",
                        "horizon_days": 7,
                        "benchmark_symbol": "ISF.L",
                    }
                ],
                default_horizon_days=7,
            )
            pending = await db.get_pending_settlements(limit=5)
            await db.settle_recommendation(
                pending[0]["id"],
                benchmark_symbol="ISF.L",
                settled_at_uk="2026-11-10 16:35",
                raw_return_pct=-4.5,
                alpha_pct=-5.1,
                reflection_lesson="英股医药除息前追高易回撤，需等放量企稳再介入。",
            )

            portfolio = _sample_portfolio()
            portfolio.isa.positions[0].isin = "US67066G1040"
            portfolio.isa.positions[0].created_at = "2026-01-10T10:00:00Z"

            class _Imp13MarketData(_FakeMarketData):
                def __init__(self) -> None:
                    super().__init__()
                    self.seen_isin_maps: list[dict[str, str] | None] = []
                    self.history_fetch_batches: list[list[str]] = []

                async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                    self.history_fetch_batches.append(list(symbols))
                    hists = await super().get_histories(symbols, period=period)
                    # Make ad-hoc META track NVDA identically so correlation overlap triggers >= 0.85
                    if "META" in hists:
                        nvda_df = (await super().get_histories(["NVDA"], period=period))["NVDA"]
                        hists["META"] = nvda_df.copy()
                    return hists

                async def get_bundles(
                    self,
                    symbols: list[str],
                    include_news: bool = True,
                    include_earnings: bool = True,
                    include_info: bool = True,
                    isin_map: dict[str, str] | None = None,
                ):
                    self.seen_isin_maps.append(isin_map)
                    histories = await super().get_histories(symbols, period="1y")
                    if "META" in histories:
                        histories["META"] = histories.get("NVDA") or (await super().get_histories(["NVDA"], period="1y"))["NVDA"].copy()
                    bundles: dict[str, TickerMarketBundle] = {}
                    for sym in symbols:
                        curr_px = 118.25 if sym == "VWRP.L" else (145.0 if sym == "AMD" else 520.0 if sym == "META" else 135.0)
                        b = TickerMarketBundle(
                            symbol=sym,
                            quote_type="ETF" if sym in {"VWRP.L", "VUAG.L"} else "EQUITY",
                            currency="GBP" if sym.endswith(".L") else "USD",
                            short_name=sym,
                            long_name=f"{sym} PLC" if sym.endswith(".L") else f"{sym} Corp",
                            exchange="LSE" if sym.endswith(".L") else "NASDAQ",
                            isin=(isin_map or {}).get(sym, ""),
                            current_price=curr_px,
                            ex_dividend_date="2026-11-20" if sym == "NVDA" else None,
                            days_to_ex_dividend=5 if sym == "NVDA" else None,
                            forward_pe=20.0 if sym not in {"VWRP.L", "VUAG.L"} else None,
                            news=[{"date": "2026-11-14", "source": "Reuters", "title": f"{sym} update"}] if include_news else [],
                            has_history=True,
                        )
                        b.identity_summary = b.build_identity_anchor(isin_override=(isin_map or {}).get(sym))
                        bundles[sym] = b
                    return histories, bundles

            class _Imp13LLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    schema = kwargs.get("response_schema")
                    if getattr(schema, "__name__", "") == "PickDecision":
                        # All three miss entry_price (0.0); AMD violates daily_vetoes, META violates corr overlap, VWRP.L survives
                        return PickDecision(
                            headline="精选低摩擦宽基，严守风控红线",
                            selected_actions=[
                                PickActionItem(
                                    account="Invest",
                                    symbol="AMD",
                                    action="BUY",
                                    thesis="芯片弹性大",
                                    entry_zone="140-146 USD",
                                    stop_condition="跌破132止损",
                                    target_window="170 USD",
                                    entry_price=0.0,
                                ),
                                PickActionItem(
                                    account="Invest",
                                    symbol="META",
                                    action="BUY",
                                    thesis="广告现金流稳健",
                                    entry_zone="510-525 USD",
                                    stop_condition="跌破480止损",
                                    target_window="580 USD",
                                    entry_price=0.0,
                                ),
                                PickActionItem(
                                    account="ISA",
                                    symbol="VWRP.L",
                                    action="BUY",
                                    thesis="全球宽基分散且免印花税换汇费",
                                    entry_zone="116-119 GBP",
                                    stop_condition="跌破MA200暂停",
                                    target_window="长线配置",
                                    entry_price=0.0,
                                ),
                            ],
                            rejected_or_frozen=["高估值软件股"],
                            consistency_and_cash_note="遵守早报芯片冻结红线，保留现金享4%利息。",
                        ).model_dump_json()
                    if kwargs.get("use_google_search"):
                        return "联网检索：VWRP.L 资金持续流入。"
                    return "投研员提案：建议考察 VWRP.L，同时关注 AMD 与 META。"

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                universe=UniverseSection(isa_long_term=["VWRP.L", "AZN.L"], invest_tactical=["AMD"]),
                strategy=StrategySection(settlement_holding_days=7, overlap_correlation_threshold=0.85),
            )
            t212 = _FakeT212(portfolio)
            md = _Imp13MarketData()
            llm = _Imp13LLM()

            clear_grounding_cache()
            captured_tech_kwargs: list[dict] = []
            captured_toolkit_dbs: list[object] = []
            from src.agents import pick_agent as _pick_mod
            orig_compute_tech = _pick_mod.compute_technical_snapshot
            orig_toolkit_cls = _pick_mod.MarketToolkit

            def _spy_compute_tech(*args, **kwargs):
                captured_tech_kwargs.append({"sym": args[0], **kwargs})
                return orig_compute_tech(*args, **kwargs)

            def _spy_toolkit(*args, **kwargs):
                captured_toolkit_dbs.append(kwargs.get("db"))
                return orig_toolkit_cls(*args, **kwargs)

            with (
                patch("src.agents.pick_agent.get_market_calendar_alerts", return_value=["🇬🇧 今日 LSE 半日市 12:30 收盘"]),
                patch("src.agents.pick_agent.compute_technical_snapshot", side_effect=_spy_compute_tech),
                patch("src.agents.pick_agent.MarketToolkit", side_effect=_spy_toolkit),
            ):
                out = await generate_pick_radar(cfg, t212, md, llm, db, focus_query="全球配置")

            # 1) Held ISIN, created_at & bundle forwarded; screener reused held_histories (no NVDA/ULVR.L re-fetch); MarketToolkit wired with db
            self.assertEqual(md.seen_isin_maps[0], {"NVDA": "US67066G1040"})
            self.assertEqual(captured_toolkit_dbs, [db])
            nvda_tech_call = next(c for c in captured_tech_kwargs if c["sym"] == "NVDA")
            self.assertEqual(nvda_tech_call["position_created_at"], "2026-01-10T10:00:00Z")
            self.assertIsNotNone(nvda_tech_call["bundle"])
            self.assertTrue(len(md.history_fetch_batches) >= 1)
            self.assertNotIn("NVDA", md.history_fetch_batches[0])
            self.assertNotIn("ULVR.L", md.history_fetch_batches[0])

            # 2) Researcher & Critic received calendar alerts, T+7 lessons, identity/ex-div & ContextPack disagreements
            research_call = next(c for c in llm.calls if c.get("usage_label") == "pick_researcher")
            critic_call = next(c for c in llm.calls if c.get("usage_label") == "pick_critic")
            self.assertIsNotNone(research_call.get("tools"))
            self.assertIn("🇬🇧 今日 LSE 半日市 12:30 收盘", research_call["prompt"])
            self.assertIn("英股医药除息前追高易回撤", research_call["prompt"])
            self.assertIn("财报前两周追高易遇获利回吐", research_call["prompt"])
            self.assertIn("NVDA: 多空分歧", research_call["prompt"])
            self.assertIn("ISIN:US67066G1040", research_call["prompt"])
            self.assertIn("除息日 2026-11-20", research_call["prompt"])
            self.assertIn("T+7", research_call["prompt"])
            self.assertIn("英股医药除息前追高易回撤", critic_call["prompt"])
            self.assertIn("T+7", critic_call["prompt"])

            # 3) Post-screen guardrail intercepted AMD (daily_vetoes) & META (ad-hoc corr >= 0.85), kept VWRP.L <= 1000 chars
            self.assertLessEqual(len(out), 1000)
            self.assertIn("[🛡️风控护栏修正]", out)
            self.assertIn("VWRP.L", out)
            self.assertIn("AMD(BUY→拦截)", out)
            self.assertIn("META(BUY→拦截)", out)

            # 4) Only surviving pick logged with native bundle.current_price (118.25), horizon_days=7, benchmark=SPY
            recs = await db.get_recent_recommendations(limit=10)
            vwrp_recs = [r for r in recs if r["symbol"] == "VWRP.L"]
            blocked_recs = [r for r in recs if r["symbol"] in {"AMD", "META"}]
            self.assertEqual(len(vwrp_recs), 1)
            self.assertEqual(blocked_recs, [])
            self.assertAlmostEqual(vwrp_recs[0]["entry_price"], 118.25, places=2)
            self.assertEqual(vwrp_recs[0]["horizon_days"], 7)
            self.assertEqual(vwrp_recs[0]["benchmark_symbol"], "SPY")


if __name__ == "__main__":
    unittest.main()
