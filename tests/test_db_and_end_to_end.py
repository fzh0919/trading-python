"""End-to-end integration tests for SQLite persistence and AdvisorBot commands (/pos, /report, /pick, /us, /stats, /clear)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
from src.bot.scheduler import TradingScheduler
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
            currency = "GBP" if sym.endswith(".L") else ("EUR" if sym.endswith((".DE", ".AS", ".PA")) else "USD")
            hist = histories.get(sym)
            curr_price = round(float(hist["Close"].iloc[-1]), 2) if hist is not None and not hist.empty else None
            b = TickerMarketBundle(
                symbol=sym,
                quote_type="ETF" if sym.endswith(".L") and sym in {"VWRP.L", "VUAG.L"} else "EQUITY",
                currency=currency,
                short_name=sym,
                current_price=curr_price,
                isin=(isin_map or {}).get(sym, ""),
                forward_pe=20.0 if not sym.endswith(".L") else None,
                news=[{"date": "2026-11-13", "source": "Reuters", "title": f"{sym} steady momentum"}]
                if include_news
                else [],
                has_history=True,
            )
            b.identity_summary = b.build_identity_anchor(isin_override=(isin_map or {}).get(sym))
            bundles[sym] = b
        return histories, bundles

    async def get_us_deep_intel(self, symbol: str) -> dict:
        return {}

    async def get_us_realtime_quote(self, symbol: str) -> dict | None:
        return None

    async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {}

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
        if kwargs.get("usage_label") == "settlement_reflection":
            return "T+5复盘：顺势动量验证正超额Alpha，短期逻辑获验证；继续严守止损与分散纪律。"
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
        self.assertNotIn("旧驳回1", rendered_p)
        self.assertNotIn("驳回/冻结", rendered_p)
        self.assertNotIn("**", rendered_p)
        self.assertNotIn("#", rendered_p)

        # Fallback: even if caller passes res_pick.decision without risk_override, guardrail tags are still surfaced
        rendered_p_no_ro = render_pick_report(res_pick.decision, max_chars=1000)
        self.assertEqual(rendered_p_no_ro.count("[🛡️风控护栏修正]"), 2)

        # Empty recommendations: suppress internal candidate rejections & suggest holding cash for interest
        empty_pick = PickDecision(
            headline="本轮无推荐，建议现金拿住吃4.0%利息",
            selected_actions=[],
            rejected_or_frozen=["AZN.L：短线追高", "META：科技红线"],
            consistency_and_cash_note="可用现金继续享 4.0% 年化利息。",
        )
        rendered_empty = render_pick_report(empty_pick, max_chars=1000)
        self.assertIn("无推荐", rendered_empty)
        self.assertIn("4.0%", rendered_empty)
        self.assertNotIn("AZN.L", rendered_empty)
        self.assertNotIn("META", rendered_empty)
        self.assertNotIn("驳回/冻结", rendered_empty)

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

        # 2) trim_text_at_line_boundary cleanly trims on newline without leaving dangling 【...】 or mid-clause breaks
        multiline = "✅ 维持早报计划\n• 要点1：正常\n【分段】\n• 要点2：" + ("长内容" * 150)
        trimmed = trim_text_at_line_boundary(multiline, 450)
        self.assertLessEqual(len(trimmed), 450)
        self.assertNotIn("【分段】", trimmed)

        single_long = "• 克隆ETF底层暴露归并 [S&P 500 (美股标普500)]：ISA VUAG £2,700 + Invest CSP1 £1,800；建议保持分散，暂不加码。"
        trimmed_single = trim_text_at_line_boundary(single_long, 56)
        self.assertLessEqual(len(trimmed_single), 56)
        self.assertFalse(trimmed_single.endswith("S&P…"))
        self.assertIn("；", single_long[len(trimmed_single) - 1 :])

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
                self.assertTrue(all("必须直白建议撤单（CANCEL）" in c["system_instruction"] for c in pipeline_calls))
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
                self.assertTrue(
                    all("必须直白建议撤单（CANCEL）" not in c["system_instruction"] for c in pick_pipeline)
                )
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
                self.assertIn("必须直白建议撤单（CANCEL）", chat_call["system_instruction"])
                # morning_report already embeds "• 今日风控红线：..." so chat skips duplicate 【今日风控红线】
                self.assertNotIn("【今日风控红线】", chat_call["prompt"])
                self.assertIn("Invest半导体占比超限，冻结新开芯片仓", chat_call["prompt"])
                self.assertEqual(chat_call["prompt"].count("Invest半导体占比超限，冻结新开芯片仓"), 1)
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

            # 2) isin_map passed to get_bundles; Disagreement + calendar alerts injected into PM prompt; cash flow printed once in analyst prompt
            self.assertTrue(any(m and m.get("NVDA") == "US67066G1040" for m in md.seen_isin_maps))
            analyst_call = next(c for c in llm.calls if c.get("usage_label") == "report_analyst")
            self.assertEqual(analyst_call["prompt"].count("[资金流水与派息"), 1)
            self.assertEqual(analyst_call["prompt"].count("外部入金 +800.00 GBP"), 1)
            pm_call = next(c for c in llm.calls if c.get("usage_label") == "report_pm")
            self.assertIn("Disagreement Matrix", pm_call["prompt"])
            self.assertIn("美股半日市", pm_call["prompt"])
            self.assertIn("外部出入金 +800.00 GBP", pm_call["prompt"])

            # 3) Fallback portfolio did NOT write a second row to portfolio_snapshots
            latest_snap = await db.get_latest_snapshot()
            self.assertIsNotNone(latest_snap)
            self.assertEqual(latest_snap.fetched_at_ms, prev_port.fetched_at_ms)

            # 4) AnalysisContextPack persisted with identity, technical (CYQ/weekly/ex_div), disagreements, lessons & report_decision
            pack = await db.get_context_pack(allow_fallback=False)
            self.assertIsNotNone(pack)
            self.assertEqual(pack["pack_version"], "1.0")
            self.assertTrue(pack["is_fallback"])
            self.assertIn("report_decision", pack)
            self.assertEqual(pack["report_decision"]["top_actions"][0]["symbol"], "NVDA")
            self.assertEqual(pack["report_decision"]["top_actions"][0]["action"], "HOLD")
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
            self.assertNotIn("高估值软件股", out)
            self.assertNotIn("驳回/冻结", out)

            # 4) Only surviving pick logged with native bundle.current_price (118.25), horizon_days=7, benchmark=SPY
            recs = await db.get_recent_recommendations(limit=10)
            vwrp_recs = [r for r in recs if r["symbol"] == "VWRP.L"]
            blocked_recs = [r for r in recs if r["symbol"] in {"AMD", "META"}]
            self.assertEqual(len(vwrp_recs), 1)
            self.assertEqual(blocked_recs, [])
            self.assertAlmostEqual(vwrp_recs[0]["entry_price"], 118.25, places=2)
            self.assertEqual(vwrp_recs[0]["horizon_days"], 7)
            self.assertEqual(vwrp_recs[0]["benchmark_symbol"], "SPY")

    async def test_pick_syncs_report_pending_audits_and_blocks_trim_reversal(self) -> None:
        from src.agents.pick_agent import generate_pick_radar
        from src.config import StrategySection
        from src.data_providers.t212_client import PendingOrderItem

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "sync_report_pick.db")
            await db.init()

            seeded_pack = {
                "pack_version": "1.0",
                "report_decision": {
                    "headline": "冻结科技，撤除高摩擦挂单",
                    "top_actions": [
                        {
                            "account": "ISA",
                            "symbol": "NVDA",
                            "action": "TRIM",
                            "price_zone": "168-170 USD",
                            "reason": "财报前借反抽减仓",
                        }
                    ],
                    "pending_order_audits": [
                        {
                            "account": "Invest",
                            "symbol": "SHEL.L",
                            "verdict": "CANCEL",
                            "order_summary": "BUY 3675 GBp",
                            "reason": "扣0.5%印花税后净空间不足以跑赢4%利息",
                        },
                        {
                            "account": "ISA",
                            "symbol": "AZN.L",
                            "verdict": "MODIFY",
                            "order_summary": "BUY 13400 GBp",
                            "reason": "RSI偏热，下调至12850p靠拢MA50",
                        },
                    ],
                    "daily_vetoes": ["严守4%现金门槛，不追高"],
                },
                "symbols": {},
            }
            await db.save_morning_report(
                "【今日核心行动与挂单裁决】\n• [Invest] SHEL.L →【CANCEL】",
                "冻结科技，撤除高摩擦挂单",
                guardrails=["严守4%现金门槛，不追高"],
                context_pack=seeded_pack,
            )

            portfolio = _sample_portfolio()
            portfolio.invest.pending_orders = [
                PendingOrderItem(
                    id="ord-shel",
                    account="Invest",
                    raw_ticker="SHELl_EQ",
                    ticker="SHEL",
                    yahoo_symbol="SHEL.L",
                    name="Shell PLC",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=10.0,
                    value=367.5,
                    limit_price=3675.0,
                    currency="GBp",
                    time_in_force="GTC",
                )
            ]

            class _SyncPickLLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    schema = kwargs.get("response_schema")
                    if getattr(schema, "__name__", "") == "PickDecision":
                        return PickDecision(
                            headline="严格延续早报风控",
                            selected_actions=[
                                PickActionItem(
                                    account="ISA",
                                    symbol="NVDA",
                                    action="BUY",
                                    thesis="尝试抄底",
                                    entry_zone="162 USD",
                                    stop_condition="150",
                                    target_window="190",
                                    entry_price=162.0,
                                ),
                                PickActionItem(
                                    account="ISA",
                                    symbol="VWRP.L",
                                    action="BUY",
                                    thesis="免税宽基长线定投",
                                    entry_zone="118 GBP",
                                    stop_condition="跌破MA200",
                                    target_window="长线",
                                    entry_price=118.0,
                                ),
                            ],
                            rejected_or_frozen=[],
                            consistency_and_cash_note="保留现金吃4%利息",
                        ).model_dump_json()
                    return "投研员提案：关注 VWRP.L，尊重早报对 SHEL.L 与 AZN.L 的挂单裁决。"

            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                universe=UniverseSection(isa_long_term=["VWRP.L", "AZN.L"], invest_tactical=["SHEL.L"]),
                strategy=StrategySection(overlap_correlation_threshold=0.95),
            )
            llm = _SyncPickLLM()
            clear_grounding_cache()
            out = await generate_pick_radar(cfg, _FakeT212(portfolio), _FakeMarketData(), llm, db)

            research_call = next(c for c in llm.calls if c.get("usage_label") == "pick_researcher")
            critic_call = next(c for c in llm.calls if c.get("usage_label") == "pick_critic")
            self.assertIn("[Invest] SHEL.L (BUY 3675 GBp) →【CANCEL】", research_call["prompt"])
            self.assertIn("[ISA] AZN.L (BUY 13400 GBp) →【MODIFY】", research_call["prompt"])
            self.assertIn("⚠️今日早报已裁决: 挂单MODIFY(ISA)", research_call["prompt"])
            self.assertIn("[Invest] SHEL.L (BUY 3675 GBp) →【CANCEL】", critic_call["prompt"])
            self.assertIn("SHEL.L", critic_call["prompt"])
            self.assertIn("当前活动挂单", critic_call["prompt"])
            self.assertIn("挂单占用(未成交前仍计息)", research_call["prompt"])
            self.assertIn("照常产生同等日息", research_call["system_instruction"])
            self.assertIn("NVDA(BUY→拦截)", out)
            self.assertIn("VWRP.L", out)

    async def test_imp14_chat_object_slicing_implicit_cache_deep_and_cross_cycle(self) -> None:
        from unittest.mock import patch
        from src.agents.chat_agent import answer_user_chat, decide_chat_objects
        from src.config import ModelsSection

        portfolio = _sample_portfolio()
        # Add Unilever short name / base ticker scenario & short 1-char ticker safety check
        portfolio.isa.positions.append(
            PositionItem(
                account="ISA",
                raw_ticker="ULVRl_EQ",
                ticker="ULVR",
                yahoo_symbol="ULVR.L",
                name="Unilever PLC",
                inst_currency="GBX",
                quantity=50,
                avg_price=4500.0,
                cur_price=4600.0,
                current_value=2300.0,
                total_cost=2250.0,
                pnl=50.0,
                fx_impact=0.0,
                return_pct=2.22,
                weight_pct=15.0,
            )
        )
        portfolio.invest.positions.append(
            PositionItem(
                account="Invest",
                raw_ticker="V_US_EQ",
                ticker="V",
                yahoo_symbol="V",
                name="Visa Inc.",
                inst_currency="USD",
                quantity=5,
                avg_price=280.0,
                cur_price=300.0,
                current_value=1180.0,
                total_cost=1100.0,
                pnl=80.0,
                fx_impact=0.0,
                return_pct=7.27,
                weight_pct=15.0,
            )
        )

        context_pack = {
            "pack_version": 1,
            "cycle_date": "2026-11-15",
            "exposure_summary": "半导体敞口偏高，英股消费品分散",
            "disagreement_text": "• NVDA: 多空分歧(趋势vs估值) | 逢回调看支撑",
            "recent_lessons": [
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "alpha_pct": -3.4,
                    "reflection_lesson": "高PE芯片追涨容易被震出",
                }
            ],
            "symbols": {
                "NVDA": {
                    "symbol": "NVDA",
                    "identity_summary": "NVDA | NVIDIA Corp | NASDAQ | Technology/Semiconductors | ISIN:US67066G1040",
                    "technical_line": "• NVDA [BULL] 现价 140.00 USD | 筹码[单峰] 获利78%/均成本128.50",
                    "technical": {"cyq_cost_90_low": 115.0, "cyq_cost_90_high": 145.0},
                    "bundle_summary": {
                        "short_name": "NVIDIA",
                        "long_name": "NVIDIA Corporation",
                        "currency": "USD",
                        "current_price": 140.0,
                        "roe_pct": 92.5,
                        "profit_margin_pct": 55.2,
                        "revenue_growth_pct": 78.0,
                        "recommendation_key": "strong_buy",
                        "news": [{"date": "2026-11-14", "title": "NVDA Blackwell demand strong"}],
                    },
                    "evaluations": [
                        {
                            "account": "ISA",
                            "action_bias": "HOLD",
                            "weight_pct": 40.0,
                            "native_return_pct": 16.67,
                            "effective_stop_pct": -12.5,
                            "reasons": ["趋势多头但单一仓位较重"],
                        }
                    ],
                    "disagreements": [
                        {
                            "account": "ISA",
                            "conflict_type": "趋势vs估值",
                            "bull_points": ["周线多头且ROE强劲"],
                            "bear_points": ["获利盘偏高注意兑现"],
                            "decision_hint": "持有观察勿追高",
                        }
                    ],
                    "lessons": [
                        {
                            "created_at_uk": "2026-11-01",
                            "action": "BUY",
                            "alpha_pct": 4.2,
                            "reflection_lesson": "财报后回踩MA20介入胜率高",
                        }
                    ],
                },
                "ULVR.L": {
                    "symbol": "ULVR.L",
                    "identity_summary": "ULVR.L | Unilever PLC | LSE | Consumer Defensive",
                    "technical_line": "• ULVR.L [RANGE] 现价 4600.00 GBp | 除息 2026-11-21(6天后)",
                    "bundle_summary": {"short_name": "Unilever", "currency": "GBp", "current_price": 4600.0},
                    "evaluations": [{"account": "ISA", "action_bias": "HOLD", "weight_pct": 15.0}],
                    "disagreements": [],
                    "lessons": [{"created_at_uk": "2026-10-20", "action": "BUY", "alpha_pct": 1.1, "reflection_lesson": "ULVR专属教训标记"}],
                },
                "V": {
                    "symbol": "V",
                    "identity_summary": "V | Visa Inc. | NYSE | Financial Services",
                    "technical_line": "• V [BULL] 现价 300.00 USD",
                    "bundle_summary": {"short_name": "Visa", "currency": "USD", "current_price": 300.0},
                },
            },
        }

        # 1) Pure 0-token symbol resolution: CJK adjacency, name alias, negation, short-code case sensitivity, follow-up
        res_nvda = decide_chat_objects("现在能加仓NVDA吗", portfolio, context_pack)
        self.assertEqual(res_nvda.decision_kind, "confirm")
        self.assertEqual(res_nvda.target_symbols, ("NVDA",))

        res_name_neg = decide_chat_objects("不要看NVDA，对比一下Unilever和V", portfolio, context_pack)
        self.assertEqual(res_name_neg.decision_kind, "compare")
        self.assertEqual(res_name_neg.target_symbols, ("ULVR.L", "V"))
        self.assertEqual(res_name_neg.denied_symbols, ("NVDA",))

        res_short_lower = decide_chat_objects("value style vs growth", portfolio, context_pack)
        self.assertEqual(res_short_lower.target_symbols, ())
        res_short_upper = decide_chat_objects("看看V现在能买吗", portfolio, context_pack)
        self.assertEqual(res_short_upper.target_symbols, ("V",))

        res_followup = decide_chat_objects(
            "它的止损设多少合适？",
            portfolio,
            context_pack,
            prev_cycle_turns=[("NVDA筹码怎么样？", "目前单峰获利78%")],
        )
        self.assertEqual(res_followup.decision_kind, "maintain")
        self.assertEqual(res_followup.target_symbols, ("NVDA",))

        # 2) Full answer_user_chat wiring: ContextPack slice, Implicit Cache prefix stability, /deep routing, prev cycle, calendar, db=db
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp14_chat.db")
            await db.init()

            await db.save_morning_report(
                "【今日核心行动与挂单裁决】\n• 保持4%现金缓冲",
                "晨报总结：严控芯片权重",
                ["冻结芯片追高"],
                context_pack=context_pack,
            )
            await db.save_pick_radar("英国ETF", "research", "PICK_RADAR_BODY_" + ("X" * 400) + "_END_PICK", "2026-11-15 17:00")
            # Seed a previous cycle chat turn before current 2026-11-15 cycle
            await db.append_chat_turn(
                user_query="上一周期问ULVR",
                bot_reply="上一周期建议等除息后再定",
                cycle_date="2020-01-01",
            )

            class _ChatSpyLLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    return "先说结论：NVDA 继续持有观察，不追高。"

            llm = _ChatSpyLLM()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models=ModelsSection(
                    chat_model="gemini-4-hydrogen",
                    chat_thinking_level="LOW",
                    chat_deep_model="gemini-4-argon",
                    chat_deep_thinking_level="HIGH",
                ),
            )
            t212 = _FakeT212(portfolio)
            md = _FakeMarketData()

            captured_toolkit_dbs: list[object] = []
            from src.agents import chat_agent as _chat_mod
            orig_toolkit = _chat_mod.MarketToolkit

            def _spy_toolkit(*args, **kwargs):
                captured_toolkit_dbs.append(kwargs.get("db"))
                return orig_toolkit(*args, **kwargs)

            with (
                patch("src.agents.chat_agent.get_market_calendar_alerts", return_value=["🇺🇸 今日美股半日市（13:00 ET / 18:00 UK 提前收盘）"]),
                patch("src.agents.chat_agent.MarketToolkit", side_effect=_spy_toolkit),
            ):
                # Turn 1: Normal query mentioning NVDA
                await answer_user_chat("NVDA现在筹码和分歧怎么样？", cfg, t212, md, llm, db)
                c1 = llm.calls[-1]
                p1 = c1["prompt"]

                self.assertEqual(c1["model"], "gemini-4-hydrogen")
                self.assertEqual(c1["thinking_level"], "LOW")
                self.assertEqual(c1["max_tool_rounds"], 3)
                self.assertIn("500 字以内", c1["system_instruction"])
                self.assertEqual(captured_toolkit_dbs[-1], db)

                # Static prefix order: morning report & guardrails & pick & baseline BEFORE live time / ISA / current query
                idx_report = p1.index("【今日早报及美股哨兵纪要】")
                idx_veto = p1.index("【今日风控红线】")
                idx_pick = p1.index("【最近一次选股雷达")
                idx_base = p1.index("【本周期组合基线与跨周期摘要】")
                idx_time = p1.index("当前英国时间 ")
                idx_isa = p1.index("【ISA 账户】")
                idx_slice = p1.index("【提问标的本地快照切片")
                idx_hist = p1.index("【今日对话记录")
                idx_query = p1.index("【用户当前问题】")
                self.assertTrue(idx_report < idx_veto < idx_pick < idx_base < idx_time < idx_isa < idx_slice < idx_hist < idx_query)

                # Local ContextPack slice injected ONLY for NVDA (not ULVR.L slice)
                self.assertIn("NVDA | NVIDIA Corp | NASDAQ", p1)
                self.assertIn("90%筹码带 115.00-145.00", p1)
                self.assertIn("ROE 92.5%", p1)
                self.assertIn("趋势多头但单一仓位较重", p1)
                self.assertIn("趋势vs估值", p1)
                self.assertIn("财报后回踩MA20介入胜率高", p1)
                self.assertNotIn("ULVR专属教训标记", p1)

                # Cross-cycle memory & calendar alert & /pick[:1050] preserved
                self.assertIn("上一周期问ULVR", p1)
                self.assertIn("🇺🇸 今日美股半日市", p1)
                self.assertIn("_END_PICK", p1)

                # Turn 2: Mutate live portfolio price & use /deep follow-up -> static prefix byte-identical!
                portfolio.isa.positions[0].cur_price = 148.5
                await answer_user_chat("/deep 它止损设多少合适？", cfg, t212, md, llm, db)
                c2 = llm.calls[-1]
                p2 = c2["prompt"]

                self.assertEqual(c2["model"], "gemini-4-argon")
                self.assertEqual(c2["thinking_level"], "HIGH")
                self.assertEqual(c2["max_tool_rounds"], 4)
                self.assertIn("1000~1200 字以内", c2["system_instruction"])
                self.assertEqual(p1[: p1.index("当前英国时间 ")], p2[: p2.index("当前英国时间 ")])
                # Follow-up recovered NVDA slice from Turn 1
                self.assertIn("上下文沿用", p2)
                self.assertIn("90%筹码带 115.00-145.00", p2)

    async def test_imp15_mutex_lock_usage_and_stats_dashboard(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                db = AdvisorDatabase(Path(tmpdir) / "imp15_e2e.db")
                await db.init()

                cfg = AppConfig(
                    env=EnvSettings(_env_file=None, telegram_chat_id="999", gemini_api_key="test-key"),
                    universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["NVDA"]),
                )
                t212 = _FakeT212(_sample_portfolio())
                md = _FakeMarketData()
                llm = _RecordingLLM()
                tg = _RecordingTg()
                sched = TradingScheduler(cfg, t212, md, llm, db, tg)
                bot = AdvisorBot(cfg, t212, md, llm, db, tg, scheduler=sched)
                self.assertIs(bot.execution_lock, sched.execution_lock)

                # 1. Shared mutex: while scheduler holds lock running morning report, manual /report is rejected
                #    while lightweight /pos continues immediately without blocking.
                started_evt = asyncio.Event()
                release_evt = asyncio.Event()

                async def _blocking_morning() -> str:
                    started_evt.set()
                    await release_evt.wait()
                    return "定时晨报已完成"

                sched_task = asyncio.create_task(
                    sched._execute_job(
                        job_name="morning",
                        chat_id="999",
                        uk_now_str="2026-11-15 07:30",
                        coro=_blocking_morning(),
                    )
                )
                await started_evt.wait()
                self.assertTrue(bot.execution_lock.locked())

                calls_before = len(llm.calls)
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/report"}}, wait=True)
                self.assertEqual(len(llm.calls), calls_before)
                self.assertIn("后台定时任务（morning）", tg.messages[-1])

                # Lightweight /pos still works while heavy lock is held
                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/pos"}}, wait=True)
                self.assertIn("ISA", tg.messages[-1])

                release_evt.set()
                await sched_task
                self.assertFalse(bot.execution_lock.locked())

                # 2. /usage displays natural-month Search quota, cache hit rate, thinking share, and zero $
                await db.log_llm_usage(
                    {
                        "call_label": "report_pm",
                        "model": "gemini-4-argon",
                        "thinking_level": "HIGH",
                        "api_rounds": 2,
                        "prompt_tokens": 10000,
                        "cached_tokens": 6000,
                        "output_tokens": 500,
                        "thought_tokens": 1500,
                        "total_tokens": 12000,
                        "search_queries": 42,
                        "success": True,
                    }
                )
                await db.log_llm_usage(
                    {
                        "call_label": "prev_month_search",
                        "model": "gemini-4-hydrogen",
                        "prompt_tokens": 2000,
                        "cached_tokens": 0,
                        "output_tokens": 100,
                        "thought_tokens": 100,
                        "total_tokens": 2200,
                        "search_queries": 100,
                        "success": True,
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

                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/usage"}}, wait=True)
                usage_msg = tg.messages[-1]
                self.assertIn("本月 (UK自然月)", usage_msg)
                self.assertIn("42 / 5,000", usage_msg)
                self.assertNotIn("142 / 5,000", usage_msg)
                self.assertIn("命中 60%", usage_msg)
                self.assertIn("思考占生成 75%", usage_msg)
                self.assertNotIn("$", usage_msg)
                self.assertNotIn("**", usage_msg)
                self.assertNotIn("#", usage_msg)

                # 3. /stats displays win rates, average alpha, lessons, live prices, and GBp/GBP 100x guard
                await db.log_recommendations(
                    [
                        {
                            "account": "Invest",
                            "symbol": "NVDA",
                            "action": "BUY",
                            "entry_price": 100.0,
                            "thesis": "AI需求强劲",
                        },
                        {
                            "account": "Invest",
                            "symbol": "AAPL",
                            "action": "BUY",
                            "entry_price": 200.0,
                            "thesis": "估值扩张",
                        },
                        {
                            "account": "Invest",
                            "symbol": "MSFT",
                            "action": "BUY",
                            "entry_price": 100.0,
                            "thesis": "云业务稳健",
                        },
                        {
                            "account": "ISA",
                            "symbol": "AZN.L",
                            "action": "BUY",
                            "entry_price": 12000.0,
                            "thesis": "英股医药龙头",
                        },
                    ]
                )
                pending = await db.get_pending_settlements(limit=10)
                nvda_rec = next(r for r in pending if r["symbol"] == "NVDA")
                aapl_rec = next(r for r in pending if r["symbol"] == "AAPL")
                await db.settle_recommendation(
                    nvda_rec["id"],
                    benchmark_symbol="SPY",
                    settled_at_uk="2026-11-12 21:00",
                    raw_return_pct=12.5,
                    alpha_pct=8.0,
                    reflection_lesson="放量突破均线后胜率高",
                )
                await db.settle_recommendation(
                    aapl_rec["id"],
                    benchmark_symbol="SPY",
                    settled_at_uk="2026-11-13 21:00",
                    raw_return_pct=-4.0,
                    alpha_pct=-6.0,
                    reflection_lesson="财报前追高易回撤",
                )

                await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/stats"}}, wait=True)
                stats_msg = tg.messages[-1]
                self.assertIn("原始胜率: 50.0% (1/2)", stats_msg)
                self.assertIn("平均收益: +4.25%", stats_msg)
                self.assertIn("超额 Alpha 胜率: 50.0% (1/2)", stats_msg)
                self.assertIn("平均 Alpha: +1.00%", stats_msg)
                self.assertIn("近期复盘教训", stats_msg)
                self.assertIn("放量突破均线后胜率高", stats_msg)
                self.assertIn("BUY MSFT @ 100.0 [待结算 | 现价", stats_msg)
                self.assertIn("BUY AZN.L @ 12000.0 [待结算 | 现价", stats_msg)
                self.assertIn("GBp/GBP量级差异", stats_msg)
                self.assertNotIn("$", stats_msg)
                self.assertNotIn("**", stats_msg)
                self.assertNotIn("#", stats_msg)

        await asyncio.wait_for(_run(), timeout=15.0)

    async def test_imp19_migration_backup_compliance_and_full_lifecycle_e2e(self) -> None:
        import json
        import os
        import sqlite3
        import tomllib
        from unittest.mock import patch

        import httpx
        import yaml

        import src.healthcheck as hc
        from src.config import AppSection
        from src.healthcheck import check_health, write_heartbeat
        from src.ops_notifier import OpsNotifier
        from src.version import APP_VERSION

        # 1) Static container, runtime, ignore rules, and version consistency audit (PROJECT_MIGRATION_RULES v1.2)
        repo_root = Path(__file__).resolve().parents[1]
        dockerfile_lines = [
            ln.strip()
            for ln in (repo_root / "Dockerfile").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        self.assertEqual(dockerfile_lines[0], "FROM python:3.12.13-slim-bookworm")
        self.assertEqual((repo_root / ".python-version").read_text(encoding="utf-8").strip(), "3.12.13")

        compose = yaml.safe_load((repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
        advisor = compose["services"]["advisor"]
        self.assertEqual(advisor["restart"], "unless-stopped")
        self.assertIn("TZ=Europe/London", advisor["environment"])
        self.assertEqual(advisor["mem_limit"], "2g")
        self.assertEqual(advisor["stop_grace_period"], "30s")
        self.assertEqual(advisor["logging"]["driver"], "json-file")
        self.assertEqual(advisor["logging"]["options"]["max-size"], "10m")
        self.assertEqual(advisor["logging"]["options"]["max-file"], "3")
        self.assertIn("src.healthcheck", " ".join(str(x) for x in advisor["healthcheck"]["test"]))
        self.assertIn("./config.yaml:/app/config.yaml:ro", advisor["volumes"])
        self.assertIn("./data:/app/data", advisor["volumes"])
        self.assertIn("homelab-net", advisor["networks"])
        self.assertTrue(compose["networks"]["homelab-net"]["external"])
        self.assertNotIn("ports", advisor)

        gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
        dockerignore_text = (repo_root / ".dockerignore").read_text(encoding="utf-8")
        for required_excl in (".env", "data/", "_ref/", "logs/", "__pycache__/", ".venv/"):
            self.assertIn(required_excl, gitignore_text)
            self.assertIn(required_excl, dockerignore_text)

        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["version"], APP_VERSION)

        class _Imp19MarketData(_FakeMarketData):
            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                dates = pd.date_range(end="2026-11-13", periods=230, freq="B")
                histories: dict[str, pd.DataFrame] = {}
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

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                legacy_db_path = root / "legacy_gcp.db"
                n95_stack_dir = root / "n95_stack"
                n95_db_path = n95_stack_dir / "data" / "advisor.db"
                n95_db_path.parent.mkdir(parents=True, exist_ok=True)
                hb_path = root / "advisor.heartbeat"
                backup_db_path = root / "daily_backup" / "advisor.db"
                backup_db_path.parent.mkdir(parents=True, exist_ok=True)

                # 2) Legacy v5.0.5 DB creation (missing IMP-03 migrated columns) + online WAL .backup migration
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                rec_ms = int(datetime(2026, 10, 28, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
                legacy_snap = _sample_portfolio().model_copy(
                    update={"total_combined_value": 14500.0, "fetched_at_ms": now_ms - 3_600_000}
                )
                src_conn = sqlite3.connect(legacy_db_path)
                try:
                    src_conn.executescript(
                        """
                        PRAGMA journal_mode=WAL;
                        CREATE TABLE portfolio_snapshots (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cycle_date TEXT NOT NULL,
                            fetched_at_uk TEXT NOT NULL,
                            fetched_at_ms INTEGER NOT NULL,
                            total_combined_value REAL NOT NULL,
                            payload_json TEXT NOT NULL
                        );
                        CREATE TABLE morning_reports (
                            cycle_date TEXT PRIMARY KEY,
                            report_text TEXT NOT NULL,
                            conclusion_text TEXT NOT NULL,
                            premarket_note TEXT DEFAULT '',
                            updated_at_ms INTEGER NOT NULL
                        );
                        CREATE TABLE pick_radars (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at_uk TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            focus_query TEXT DEFAULT '',
                            research_text TEXT DEFAULT '',
                            final_text TEXT NOT NULL
                        );
                        CREATE TABLE pick_recommendations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at_uk TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            account TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            action TEXT NOT NULL,
                            entry_price REAL,
                            thesis TEXT DEFAULT ''
                        );
                        CREATE TABLE chat_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cycle_date TEXT NOT NULL,
                            user_query TEXT NOT NULL,
                            bot_reply TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL
                        );
                        CREATE TABLE llm_usage_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at_uk TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            cycle_date TEXT NOT NULL,
                            call_label TEXT NOT NULL,
                            model TEXT NOT NULL,
                            thinking_level TEXT NOT NULL,
                            api_rounds INTEGER NOT NULL DEFAULT 1,
                            prompt_tokens INTEGER NOT NULL DEFAULT 0,
                            cached_tokens INTEGER NOT NULL DEFAULT 0,
                            output_tokens INTEGER NOT NULL DEFAULT 0,
                            thought_tokens INTEGER NOT NULL DEFAULT 0,
                            total_tokens INTEGER NOT NULL DEFAULT 0,
                            estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                            latency_ms INTEGER NOT NULL DEFAULT 0,
                            success INTEGER NOT NULL DEFAULT 1
                        );
                        CREATE TABLE ops_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at_uk TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            severity TEXT NOT NULL,
                            source TEXT NOT NULL,
                            alert_key TEXT NOT NULL,
                            message TEXT NOT NULL,
                            notified INTEGER NOT NULL DEFAULT 0,
                            suppressed_count INTEGER NOT NULL DEFAULT 0
                        );
                        CREATE TABLE scheduler_runs (
                            job_name TEXT NOT NULL,
                            uk_date TEXT NOT NULL,
                            fired_at_uk TEXT NOT NULL,
                            PRIMARY KEY (job_name, uk_date)
                        );
                        """
                    )
                    src_conn.execute(
                        "INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ("2026-11-15", "2026-11-15 08:00", now_ms - 3_600_000, 14500.0, legacy_snap.model_dump_json()),
                    )
                    src_conn.execute(
                        "INSERT INTO pick_recommendations (created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ("2026-10-28 15:00", rec_ms, "Invest", "NVDA", "BUY", 100.0, "AI算力需求延续"),
                    )
                    src_conn.execute(
                        "INSERT INTO ops_events (created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
                        ("2026-11-15 08:30", now_ms - 60_000, "CRITICAL", "t212", "legacy_unnotified", "legacy outage before migration"),
                    )
                    src_conn.commit()

                    # Perform online hot-backup while source WAL connection remains open
                    dst_conn = sqlite3.connect(n95_db_path)
                    try:
                        src_conn.backup(dst_conn)
                    finally:
                        dst_conn.close()
                finally:
                    src_conn.close()

                db = AdvisorDatabase(n95_db_path)
                await db.init()

                latest_snap = await db.get_latest_snapshot()
                self.assertIsNotNone(latest_snap)
                self.assertEqual(latest_snap.total_combined_value, 14500.0)
                pending_legacy = await db.get_pending_settlements()
                self.assertEqual(len(pending_legacy), 1)
                self.assertEqual(pending_legacy[0]["symbol"], "NVDA")

                write_heartbeat(hb_path)
                ok, reason = check_health(hb_path, n95_db_path)
                self.assertTrue(ok, msg=reason)

                # Verify relative data/advisor.db fallback when running check_health on host/local stack dir
                if not hc.DEFAULT_DB_PATH.exists():
                    orig_cwd = os.getcwd()
                    try:
                        os.chdir(n95_stack_dir)
                        clean_env = {k: v for k, v in os.environ.items() if k != "ADVISOR_DB_PATH"}
                        with patch.dict(os.environ, clean_env, clear=True):
                            ok_rel, reason_rel = check_health(hb_path)
                            self.assertTrue(ok_rel, msg=reason_rel)
                    finally:
                        os.chdir(orig_cwd)

                # 3) ops-notify end-to-end delivery + full business closed loop on migrated DB
                ops_requests: list[tuple[str, dict]] = []

                async def _ops_handler(request: httpx.Request) -> httpx.Response:
                    ops_requests.append(
                        (
                            request.headers.get("Authorization", ""),
                            json.loads(request.content.decode("utf-8")),
                        )
                    )
                    return httpx.Response(200, json={"ok": True})

                cfg = AppConfig(
                    env=EnvSettings(
                        _env_file=None,
                        telegram_chat_id="999",
                        gemini_api_key="test-key",
                        ops_notify_url="http://ops-notify:8000/alert",
                        ops_notify_token="n95-secret",
                    ),
                    app=AppSection(db_path=str(n95_db_path)),
                    universe=UniverseSection(isa_long_term=["VWRP.L", "AZN.L"], invest_tactical=["AMD"]),
                )
                tg = _RecordingTg()
                ops = OpsNotifier(
                    telegram=tg,
                    chat_id="999",
                    db=db,
                    ops_notify_url=cfg.env.ops_notify_url,
                    ops_notify_token=cfg.env.ops_notify_token,
                    transport=httpx.MockTransport(_ops_handler),
                )
                t212 = _FakeT212(_sample_portfolio())
                md = _Imp19MarketData()
                llm = _RecordingLLM()
                sched = TradingScheduler(cfg, t212, md, llm, db, tg, ops_notifier=ops)
                bot = AdvisorBot(cfg, t212, md, llm, db, tg, ops_notifier=ops, scheduler=sched)

                try:
                    self.assertTrue(await ops.flush_pending())
                    self.assertTrue(
                        await ops.notify(
                            severity="WARN",
                            source="imp19_rehearsal",
                            alert_key="imp19_test_alert",
                            title="N95 迁移演练测试",
                            message="验证 homelab-net ops-notify 连通",
                        )
                    )
                    self.assertEqual(len(ops_requests), 2)
                    self.assertTrue(all(auth == "Bearer n95-secret" for auth, _ in ops_requests))
                    self.assertTrue(all(p["service"] == "trading-python" for _, p in ops_requests))
                    self.assertEqual(ops_requests[0][1]["alert_key"], "ops_recovery_flush")
                    self.assertIn("legacy outage before migration", ops_requests[0][1]["message"])
                    self.assertEqual(ops_requests[1][1]["alert_key"], "imp19_test_alert")
                    self.assertEqual(tg.messages, [])

                    clear_grounding_cache()
                    await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/report"}}, wait=True)
                    report_msg = tg.messages[-1]
                    self.assertLessEqual(len(report_msg), 1800)
                    self.assertIsNotNone(await db.get_context_pack(allow_fallback=False))

                    await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/us"}}, wait=True)
                    self.assertIn("美股或个股简报", tg.messages[-1])

                    await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/pick 英国ETF"}}, wait=True)
                    self.assertIn("VWRP.L", tg.messages[-1])

                    settle_res = await sched.run_daily_settlement(
                        now_utc=datetime(2026, 11, 13, 22, 10, tzinfo=timezone.utc)
                    )
                    self.assertEqual(settle_res["settled"], 1)
                    self.assertEqual(settle_res["not_due"], 1)
                    nvda_lessons = await db.get_symbol_lessons("NVDA", limit=1)
                    self.assertEqual(len(nvda_lessons), 1)
                    self.assertEqual(nvda_lessons[0]["outcome_status"], "SETTLED")
                    self.assertIsNotNone(nvda_lessons[0]["alpha_pct"])
                    self.assertTrue(nvda_lessons[0]["reflection_lesson"])

                    await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/stats"}}, wait=True)
                    stats_msg = tg.messages[-1]
                    self.assertIn("已结算", stats_msg)
                    self.assertIn("近期复盘教训", stats_msg)
                    self.assertIn("BUY VWRP.L", stats_msg)

                    await bot.handle_update({"message": {"chat": {"id": "999"}, "text": "/health"}}, wait=True)
                    health_msg = tg.messages[-1]
                    self.assertIn(f"v{APP_VERSION}", health_msg)
                    self.assertIn("已启用 ops-notify", health_msg)
                    self.assertIn("SQLite & 告警: ✅ 正常", health_msg)

                    # Post-cycle online SQLite .backup simulation (N95 backup_daily.sh readiness)
                    live_conn = sqlite3.connect(n95_db_path)
                    try:
                        bkp_conn = sqlite3.connect(backup_db_path)
                        try:
                            live_conn.backup(bkp_conn)
                        finally:
                            bkp_conn.close()
                    finally:
                        live_conn.close()

                    ok_bkp, reason_bkp = check_health(hb_path, backup_db_path)
                    self.assertTrue(ok_bkp, msg=reason_bkp)
                finally:
                    await ops.close()
                    await t212.close()
                    await md.close()
                    await db.close()

        await asyncio.wait_for(_run(), timeout=20.0)

    async def test_imp20_us_sentinel_premarket_15m_gap_triggers_anomaly_gate(self) -> None:
        from unittest.mock import patch
        from src.agents.report_agent import generate_us_sentinel
        from src.engine.calendar_utils import UsMarketSchedule

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp20_us_prepost.db")
            await db.init()

            class _PreMarketGateMD(_FakeMarketData):
                def __init__(self) -> None:
                    super().__init__(finnhub_key="fh-test")

                async def get_us_realtime_quote(self, symbol: str) -> dict | None:
                    # Finnhub free /quote returns 0.0% during premarket before regular open
                    return {"symbol": symbol, "current_price": 100.0, "prev_close": 100.0, "change": 0.0, "change_pct": 0.0}

                async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict]:
                    return {
                        "NVDA": {
                            "symbol": "NVDA",
                            "current_price": 104.2,
                            "prev_close": 100.0,
                            "change": 4.2,
                            "change_pct": 4.2,
                            "source": "yfinance_prepost_15m",
                        }
                    }

            class _PreLLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    if kwargs.get("use_google_search"):
                        return "盘前异动检索：NVDA 盘前跳空 +4.2%。"
                    return "🚨 盘前增量修正\n• NVDA 盘前跳空 +4.2%，上调风控关注。"

            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
            t212 = _FakeT212(_sample_portfolio())
            llm = _PreLLM()
            pre_sched = UsMarketSchedule(
                ny_date="2026-11-13",
                uk_date="2026-11-13",
                uk_weekday_label="周五",
                open_time_uk="14:30",
                macro_time_uk="13:30",
                mid_morning_time_uk="15:00",
                fomc_time_uk="19:00",
                close_time_uk="21:00",
                is_dst_mismatch=False,
                is_us_trading_day=True,
                is_us_holiday=False,
                is_us_early_close=False,
                close_time_et="16:00",
                phase="PRE",
                minutes_to_open=15,
                minutes_since_open=-15,
                minutes_to_close=405,
            )

            clear_grounding_cache()
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=pre_sched):
                out = await generate_us_sentinel(cfg, t212, _PreMarketGateMD(), llm, db, is_scheduled_premarket=True)

            self.assertTrue(out.startswith("🚨"))
            sentinel_call = next(c for c in llm.calls if c.get("usage_label") == "us_sentinel")
            self.assertEqual(sentinel_call["thinking_level"], "MEDIUM")
            self.assertIn("本地异动门控：已触发深度复核", sentinel_call["prompt"])
            self.assertIn("+4.20%", sentinel_call["prompt"])

    async def test_imp21_morning_report_wires_deep_intel_cvar_touch_prob_and_priority_search(self) -> None:
        from unittest.mock import patch
        from src.agents.report_agent import generate_morning_report
        from src.agents.tools import clear_grounding_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp21_report.db")
            await db.init()

            class _Imp21MarketData(_FakeMarketData):
                async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                    hists = await super().get_histories(symbols, period=period)
                    # Give NVDA a +4.5% last-day move to trigger priority 1-to-1 Google Search
                    if "NVDA" in hists and not hists["NVDA"].empty:
                        df = hists["NVDA"].copy()
                        prev_c = float(df["Close"].iloc[-2])
                        spike = round(prev_c * 1.045, 2)
                        df.iloc[-1, df.columns.get_loc("Close")] = spike
                        df.iloc[-1, df.columns.get_loc("High")] = spike * 1.01
                        hists["NVDA"] = df
                    return hists

                async def get_bundles(self, symbols: list[str], **kwargs):
                    hists, bundles = await super().get_bundles(symbols, **kwargs)
                    if "NVDA" in bundles:
                        bundles["NVDA"].days_to_earnings = 4
                    return hists, bundles

                async def get_us_deep_intel_batch(self, symbols: list[str], max_concurrency: int = 4) -> dict[str, dict]:
                    return {
                        "NVDA": {
                            "recommendations": [{"period": "2026-11-01", "strong_buy": 25, "buy": 10, "hold": 2, "sell": 0, "strong_sell": 0}],
                            "earnings_surprises": [{"period": "2026-09-30", "surprise_pct": 9.5}],
                            "insider_sentiment": {"mspr": 14.2, "change": 600},
                        }
                    }

            portfolio = _sample_portfolio()
            t212 = _FakeT212(portfolio)
            md = _Imp21MarketData()
            llm = _RecordingLLM()
            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
            captured_grounding: dict[str, object] = {}

            async def _fake_grounding(**kwargs):
                captured_grounding.update(kwargs)
                return "【重点标的1对1深搜 · NVDA】\nNVDA 盘中上涨 +4.5% 临近财报。"

            clear_grounding_cache()
            try:
                with patch("src.agents.report_agent.run_forced_market_grounding", side_effect=_fake_grounding):
                    out = await generate_morning_report(cfg, t212, md, llm, db)
            finally:
                await md.close()
                await db.close()

            self.assertIn("保留4%现金缓冲", out)
            self.assertIn("NVDA", captured_grounding.get("priority_symbols") or [])
            analyst_call = next(c for c in llm.calls if c.get("usage_label") == "report_analyst")
            prompt_txt = analyst_call["prompt"]
            self.assertIn("组合真实风险归因(前3波动源)", prompt_txt)
            self.assertIn("尾部压力测试(95%单日CVaR)", prompt_txt)
            self.assertIn("美股机构评级、EPS惊喜与内部人动向", prompt_txt)
            self.assertIn("20日触价率", prompt_txt)

    async def test_imp22_stats_dashboard_renders_multiwindow_alpha_mfe_mae_and_weekly_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "imp22_stats.db")
            await db.init()

            rec_ms = int(datetime(2026, 10, 5, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-10-05 21:30",
                        "created_at_ms": rec_ms,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 100.0,
                        "thesis": "AI breakout",
                        "horizon_days": 5,
                        "benchmark_symbol": "SPY",
                    }
                ]
            )
            pending = await db.get_pending_settlements()
            rec_id = pending[0]["id"]
            await db.settle_recommendation(
                rec_id,
                benchmark_symbol="SPY",
                settled_at_uk="2026-10-12 22:00",
                raw_return_pct=5.2,
                alpha_pct=3.1,
                outcome_status="SETTLED",
                reflection_lesson="控制回撤继续跟踪。",
                mfe_pct=8.4,
                mae_pct=-2.1,
            )
            await db.update_recommendation_followup(
                rec_id,
                t10_return_pct=9.8,
                t10_alpha_pct=6.4,
                t20_return_pct=15.0,
                t20_alpha_pct=10.2,
                mfe_pct=17.5,
                mae_pct=-3.2,
            )
            await db.save_quant_audit(
                "2026-11-14",
                "• 扫描池 51 只候选 vs 10 只持仓\n• 长期弱势僵尸股(>=60d Phase4): 1只(TEST)",
                {"universe_count": 51},
                created_at_uk="2026-11-14 10:30",
            )

            cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
            bot = AdvisorBot(
                cfg=cfg,
                t212=_FakeT212(_sample_portfolio()),
                market_data=_FakeMarketData(),
                llm=_RecordingLLM(),
                db=db,
                telegram=_RecordingTg(),
            )
            try:
                stats_msg = await bot._build_stats_message()
            finally:
                await db.close()

            self.assertIn("多窗口 Alpha 轨迹: T+5 平均 +3.10% | T+10 平均 +6.40% | T+20 平均 +10.20%", stats_msg)
            self.assertIn("持仓期极值分布: 平均最大浮盈(MFE) +17.5% | 平均最大回撤(MAE) -3.2%", stats_msg)
            self.assertIn("【周末候选池与阈值体检】", stats_msg)
            self.assertIn("扫描池 51 只候选", stats_msg)


if __name__ == "__main__":
    unittest.main()
