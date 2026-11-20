"""v5.5.8 regression tests: daily-chat slimming, cross-cycle isolation, realtime calibration.

Covers the plan in docs/plans/2026-11-20-5.5.8-daily-chat-plan.md §5.2:
redline dedup, compare-hint fix + CN aliases (no index binding), /pick cross-cycle
isolation (afternoon /us never kills a same-day /pick), stale morning labelling,
lite-bundle realtime structure calibration (IBTA.L never counted as equity satellite),
/deep prefix strip + empty guard, head+tail reply preservation, plain-text guard,
byte-stable static prefix across turns and /us appends, and zero-extra-request
MacroSnapshot 1d moves.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from src.agents.chat_agent import (
    _build_intraday_structure_line,
    _clean_chat_plain_text,
    _format_chat_history_compact,
    _format_compact_positions_table,
    _format_context_pack_symbol_slice,
    _format_cycle_baseline_section,
    _has_complete_embedded_vetoes,
    _head_tail_preserve,
    answer_user_chat,
    decide_chat_objects,
)
from src.config import AppConfig, EnvSettings, ModelsSection, StrategySection
from src.data_providers.market_data import MacroSnapshot
from src.data_providers.market_data.service import MarketDataService
from src.data_providers.t212_client import (
    AccountSnapshot,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
    T212Client,
)
from src.storage.db import AdvisorDatabase
from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio


def _chat_cfg() -> AppConfig:
    return AppConfig(
        env=EnvSettings(_env_file=None, telegram_chat_id="999"),
        models=ModelsSection(
            chat_model="deepseek-flash",
            chat_thinking_level="HIGH",
            chat_deep_model="gemini-4-neon",
            chat_deep_thinking_level="MEDIUM",
        ),
    )


class _SpyLLM:
    def __init__(self, reply: str = "先说结论：继续持有观察。") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.reply


def _pos(
    account: str,
    yahoo_symbol: str,
    name: str,
    *,
    weight: float = 5.0,
    ret: float = 1.0,
    price: float = 100.0,
    currency: str = "USD",
    value: float = 500.0,
) -> PositionItem:
    ticker = yahoo_symbol.split(".")[0]
    return PositionItem(
        account=account,
        raw_ticker=f"{ticker}_EQ",
        ticker=ticker,
        yahoo_symbol=yahoo_symbol,
        name=name,
        inst_currency=currency,
        quantity=5.0,
        avg_price=price * 0.98,
        cur_price=price,
        current_value=value,
        total_cost=value * 0.97,
        pnl=value * 0.03,
        fx_impact=0.0,
        return_pct=ret,
        weight_pct=weight,
    )


def _isa_account(positions: list[PositionItem]) -> AccountSnapshot:
    total_value = 10000.0
    invested = sum(p.current_value for p in positions)
    return AccountSnapshot(
        ok=True,
        account_label="ISA",
        total_account_value=total_value,
        investments_value=invested,
        investments_cost=invested * 0.95,
        unrealized_pnl=invested * 0.05,
        total_cash=total_value - invested,
        available_to_trade=total_value - invested,
        reserved_cash=0.0,
        cash_ratio_pct=(total_value - invested) / total_value * 100.0,
        positions=positions,
    )


class TestV558UnitLevel(unittest.TestCase):
    def test_veto_dedup_with_semicolon_joined_morning_report(self) -> None:
        morning = (
            "【宏观、账户与今日风控红线】\n"
            "• 宏观与现金：平稳\n"
            "• 今日风控红线：冻结芯片追高；严控单日开仓"
        )
        guardrails = ["冻结芯片追高。", "严控单日开仓。"]
        self.assertTrue(_has_complete_embedded_vetoes(morning, guardrails))
        # No redline header at all -> still duplicates
        self.assertFalse(_has_complete_embedded_vetoes("【今日核心行动】\n• 保持", guardrails))
        # Truncated header -> duplicates
        self.assertFalse(
            _has_complete_embedded_vetoes("• 今日风控红线：冻结芯片追高；…", guardrails)
        )

    def test_single_ticker_with_haishi_word_is_confirm_not_compare(self) -> None:
        portfolio = _sample_portfolio()
        portfolio.isa.positions.append(
            _pos("ISA", "ISF.L", "iShares Core FTSE 100 UCITS ETF", weight=6.0, ret=0.8, price=7500.0, currency="GBp", value=600.0)
        )
        res = decide_chat_objects("ISF还是在1016成交了吗", portfolio, None)
        self.assertEqual(res.decision_kind, "confirm")
        self.assertEqual(res.target_symbols, ("ISF.L",))

    def test_cn_aliases_hit_but_generic_index_words_do_not_bind(self) -> None:
        portfolio = _sample_portfolio()
        # NVDA held; MSFT/VWRP.L known via ContextPack symbols
        context_pack = {
            "symbols": {
                "MSFT": {"identity_summary": "MSFT | Microsoft Corp | NASDAQ"},
                "VWRP.L": {
                    "identity_summary": "VWRP.L | Vanguard FTSE All-World UCITS ETF | LSE",
                    "bundle_summary": {"quote_type": "ETF", "short_name": "Vanguard All-World"},
                },
            }
        }
        res_nv = decide_chat_objects("英伟达现在怎么样", portfolio, context_pack)
        self.assertEqual(res_nv.decision_kind, "confirm")
        self.assertEqual(res_nv.target_symbols, ("NVDA",))

        res_ms = decide_chat_objects("微软的财报怎么看", portfolio, context_pack)
        self.assertEqual(res_ms.target_symbols, ("MSFT",))

        res_vwrp = decide_chat_objects("全球宽基该不该加仓", portfolio, context_pack)
        self.assertEqual(res_vwrp.target_symbols, ("VWRP.L",))

        # Generic index words must never collapse onto a single held ETF
        for query in ("标普今天涨了多少", "富时100怎么看", "纳指和欧股今天如何"):
            res_idx = decide_chat_objects(query, portfolio, context_pack)
            self.assertEqual(res_idx.target_symbols, (), msg=query)
            self.assertNotEqual(res_idx.decision_kind, "compare", msg=query)

        # compare still works with two resolved symbols (incl. CN alias)
        res_cmp = decide_chat_objects("对比一下英伟达和微软", portfolio, context_pack)
        self.assertEqual(res_cmp.decision_kind, "compare")
        self.assertEqual(set(res_cmp.target_symbols), {"NVDA", "MSFT"})

    def test_clean_chat_plain_text(self) -> None:
        raw = "**结论**：持有。\n# 今日盘面\n• 止损距离按 3*ATR 计算\n##小节标题未带空格不动"
        cleaned = _clean_chat_plain_text(raw)
        self.assertNotIn("**", cleaned)
        self.assertIn("结论：持有。", cleaned)
        self.assertNotIn("# 今日盘面", cleaned)
        self.assertIn("今日盘面", cleaned)
        self.assertIn("3*ATR", cleaned)
        self.assertIn("##小节标题未带空格不动", cleaned)  # heading strip needs whitespace
        self.assertIn("\n", cleaned)
        self.assertIn("•", cleaned)

    def test_head_tail_preserve_keeps_conclusion_and_execution_plan(self) -> None:
        short = "结论：持有。" * 20  # < 550 chars
        self.assertEqual(_head_tail_preserve(short), short)

        head_txt = "开头结论：" + "分" * 290
        tail_txt = "末尾执行预案：买入10股@99.5，止损92.0" + "尾" * 200
        long_reply = head_txt + "中" * 600 + tail_txt
        kept = _head_tail_preserve(long_reply)
        self.assertIn("…（中略）…", kept)
        self.assertIn("开头结论：", kept)
        self.assertIn("末尾执行预案：买入10股@99.5，止损92.0", kept)
        self.assertLessEqual(len(kept), 700)

    def test_history_compact_prev_cycle_bridge_only_on_first_turn(self) -> None:
        prev = [("昨天问EOG", "昨天建议148留单观察")]
        first = _format_chat_history_compact([], recent_full_turns=2, prev_cycle_turns=prev)
        self.assertIn("[上一周期] 问: 昨天问EOG", first)
        self.assertIn("昨天建议148留单观察", first)

        later = _format_chat_history_compact(
            [("今天问NVDA", "今天继续持有")], recent_full_turns=2, prev_cycle_turns=prev
        )
        self.assertNotIn("上一周期", later)

    def test_baseline_section_drops_exposure_and_weak_trend_merge_rows(self) -> None:
        context_pack = {
            "exposure_summary": "半导体敞口偏高",
            "disagreement_text": (
                "• [ISA] NVDA (STOP_VS_EX_DIV) | 多方：趋势完整 ⚔️ 空方：临近除息 → 提示：等除息后定\n"
                "• [SUPPORT_VS_WEAK_TREND 合并 x3] AMD[ISA], MSFT[ISA] | 多方：估值修复 ⚔️ 空方：趋势偏弱 → 提示：观察"
            ),
        }
        out = _format_cycle_baseline_section(context_pack, "【今日核心行动】\n• 保持现金缓冲")
        self.assertIn("组合底层敞口", out)
        self.assertIn("STOP_VS_EX_DIV", out)
        self.assertNotIn("SUPPORT_VS_WEAK_TREND", out)

        out_dedup = _format_cycle_baseline_section(
            context_pack, "【宏观、账户与今日风报红线】\n• 双账户与集中度：芯片偏高"
        )
        self.assertNotIn("组合底层敞口", out_dedup)

    def test_etf_slice_news_noise_filtered_but_equity_slice_kept(self) -> None:
        etf_entry = {
            "identity_summary": "ISF.L | iShares Core FTSE 100 UCITS ETF | LSE",
            "technical_line": "• ISF.L [RANGE] 现价 7500.00 GBp",
            "bundle_summary": {
                "quote_type": "ETF",
                "short_name": "iShares Core FTSE 100",
                "currency": "GBp",
                "current_price": 7500.0,
                "news": [
                    {"date": "2026-11-20", "title": "Kenya to switch on first oil rig pr"},
                    {"date": "2026-11-20", "title": "FTSE 100 opens lower as miners drag"},
                ],
            },
        }
        etf_out = _format_context_pack_symbol_slice("ISF.L", etf_entry)
        self.assertIn("FTSE 100 opens lower", etf_out)
        self.assertNotIn("Kenya", etf_out)

        equity_entry = {
            "identity_summary": "NVDA | NVIDIA Corp | NASDAQ",
            "technical_line": "• NVDA [BULL] 现价 140.00 USD",
            "bundle_summary": {
                "quote_type": "EQUITY",
                "currency": "USD",
                "current_price": 140.0,
                "news": [{"date": "2026-11-20", "title": "Unrelated macro headline but kept"}],
            },
        }
        equity_out = _format_context_pack_symbol_slice("NVDA", equity_entry)
        self.assertIn("Unrelated macro headline but kept", equity_out)

    def test_deep_compact_positions_table_routine_rows_one_line(self) -> None:
        positions = [
            _pos("ISA", "NVDA", "NVIDIA", weight=40.0, ret=25.0),
            _pos("ISA", "AMD", "AMD", weight=3.0, ret=1.2, price=110.0, currency="USD", value=300.0),
            _pos("ISA", "MSFT", "Microsoft", weight=4.0, ret=-1.0, price=420.0, currency="USD", value=400.0),
        ]
        acc = _isa_account(positions)
        out = _format_compact_positions_table(acc, ("NVDA",))
        self.assertIn("一行速览", out)
        self.assertIn("• AMD [3.0% | 浮盈 +1.2% | 现价 110.00 USD]", out)
        self.assertIn("• MSFT [4.0% | 浮盈 -1.0% | 现价 420.00 USD]", out)
        # focus position keeps full detail row
        self.assertIn("NVIDIA", out)
        self.assertNotIn("其余 2 只常规持仓（非本次提问焦点", out)

    def test_macro_snapshot_to_text_includes_1d_moves(self) -> None:
        snap = MacroSnapshot(
            fetched_at_utc="2026-11-20 15:00 UTC",
            spy_1d_pct=-0.82,
            qqq_1d_pct=-1.15,
            ftse_1d_pct=0.34,
            spy_5d_pct=1.2,
        )
        text = snap.to_text()
        self.assertIn("标普(SPY)当日 -0.82%", text)
        self.assertIn("纳指(QQQ)当日 -1.15%", text)
        self.assertIn("富时(ISF.L)当日 +0.34%", text)
        self.assertIn("标普(SPY)5日 +1.20%", text)


class TestV558MacroService1d(unittest.IsolatedAsyncioTestCase):
    async def test_macro_snapshot_1d_computed_with_zero_extra_requests(self) -> None:
        calls: list[str] = []

        class _CountingMD(MarketDataService):
            def _fetch_yf_sync(self, symbol: str, period: str = "2y", *args, **kwargs):
                calls.append(symbol)
                closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0]
                idx = pd.date_range("2026-11-10", periods=len(closes), freq="B")
                df = pd.DataFrame(
                    {
                        "Open": closes,
                        "High": closes,
                        "Low": closes,
                        "Close": closes,
                        "Volume": [1.0] * len(closes),
                    },
                    index=idx,
                )
                from src.data_providers.market_data import TickerMarketBundle

                return df, TickerMarketBundle(symbol=symbol, currency="USD", has_history=True)

        md = _CountingMD(finnhub_key="", fred_key="", ttl_seconds=900)
        snap = await md.get_macro_snapshot()
        # Exactly the 6 fixed macro symbols fetched — the 1d fields add zero requests.
        self.assertEqual(sorted(set(calls)), ["EURGBP=X", "GBPUSD=X", "ISF.L", "QQQ", "SPY", "^VIX"])
        self.assertAlmostEqual(snap.spy_1d_pct, (110.0 / 105.0 - 1.0) * 100.0, places=4)
        self.assertAlmostEqual(snap.qqq_1d_pct, (110.0 / 105.0 - 1.0) * 100.0, places=4)
        self.assertAlmostEqual(snap.ftse_1d_pct, (110.0 / 105.0 - 1.0) * 100.0, places=4)
        self.assertAlmostEqual(snap.spy_5d_pct, (110.0 / 101.0 - 1.0) * 100.0, places=4)


class TestV558ChatFlows(unittest.IsolatedAsyncioTestCase):
    async def test_redline_dedup_in_full_chat_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_dedup.db")
            await db.init()
            await db.save_morning_report(
                "【宏观、账户与今日风控红线】\n• 宏观与现金：平稳\n"
                "• 今日风控红线：冻结芯片追高；严控单日开仓",
                "晨报总结",
                ["冻结芯片追高。", "严控单日开仓。"],
            )
            llm = _SpyLLM()
            await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            prompt = llm.calls[-1]["prompt"]
            self.assertEqual(prompt.count("冻结芯片追高"), 1)
            self.assertNotIn("【今日风控红线】", prompt)

    async def test_pick_radar_cross_cycle_isolation_and_same_day_survival(self) -> None:
        marker = "PICK_MARKER_V558_KEEP"
        old_marker = "PICK_MARKER_V558_OLD"

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "v558_pick.db"
            db = AdvisorDatabase(db_path)
            await db.init()

            # Case 1: previous-cycle /pick is skipped entirely
            await db.save_pick_radar("focus", "research", old_marker, "2026-11-17 18:21")
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "UPDATE pick_radars SET created_at_ms = ?",
                    (int((time.time() - 3 * 86400) * 1000),),
                )
                conn.commit()
            finally:
                conn.close()
            llm = _SpyLLM()
            await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            self.assertNotIn(old_marker, llm.calls[-1]["prompt"])

            # Case 2: same-day /pick survives an afternoon /us premarket append
            # (append_premarket_note overwrites morning_reports.updated_at_ms — the
            # pick must NOT be judged stale by that timestamp).
            await db.save_pick_radar("focus2", "research", marker, "2026-11-20 10:00")
            await db.append_premarket_note("✅ 维持早报计划：美股盘前平稳")
            llm2 = _SpyLLM()
            await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm2, db)
            self.assertIn(marker, llm2.calls[-1]["prompt"])
            self.assertIn("【今日美股开盘前哨补充】", llm2.calls[-1]["prompt"])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Case 3: no morning report yet, but a same-cycle /pick exists -> injected
            db = AdvisorDatabase(Path(tmpdir) / "v558_pick_nofresh.db")
            await db.init()
            await db.save_pick_radar("focus3", "research", marker, "2026-11-20 11:00")
            llm3 = _SpyLLM()
            await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm3, db)
            self.assertIn(marker, llm3.calls[-1]["prompt"])

    async def test_stale_morning_report_gets_explicit_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_stale.db")
            await db.init()
            await db.save_morning_report(
                "【昨日早报正文】",
                "昨日结论",
                ["昨日红线。"],
                cycle_date="2026-11-17",
            )
            llm = _SpyLLM()
            await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            prompt = llm.calls[-1]["prompt"]
            self.assertIn("【上一周期陈旧早报（今日晨报尚未生成，仅作参考）】", prompt)
            self.assertNotIn("【今日早报及美股哨兵纪要】", prompt)

    async def test_morning_context_meta_splits_premarket_and_reports_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_meta.db")
            await db.init()
            static_txt, pre_note, conclusion, stale, cdate = await db.get_morning_context_meta()
            self.assertEqual((static_txt, pre_note, conclusion, stale), ("", "", "", False))

            await db.save_morning_report("正文", "结论", ["红线。"])
            await db.append_premarket_note("盘前平稳")
            static_txt, pre_note, conclusion, stale, cdate = await db.get_morning_context_meta()
            self.assertEqual(static_txt, "正文")
            self.assertEqual(pre_note, "盘前平稳")
            self.assertEqual(conclusion, "结论")
            self.assertFalse(stale)
            self.assertEqual(cdate, db._cycle_id())

            # Backward compatibility: legacy wrappers unchanged
            report, concl = await db.get_morning_context(include_premarket=True)
            self.assertIn("正文", report)
            self.assertIn("盘前平稳", report)
            ts, final_text = await db.get_latest_pick_radar()
            self.assertEqual((ts, final_text), ("", ""))

    async def test_realtime_calibration_counts_ibta_as_other_etf_not_satellite(self) -> None:
        positions = [
            _pos("ISA", "VWRP.L", "Vanguard FTSE All-World UCITS ETF", weight=36.0, ret=2.0, price=142.0, currency="GBp", value=3600.0),
            _pos("ISA", "IBTA.L", "iShares £ 0-1 yr UCITS ETF", weight=5.0, ret=0.5, price=52.0, currency="GBp", value=500.0),
            _pos("ISA", "NVDA", "NVIDIA", weight=10.0, ret=8.0, value=1000.0),
            _pos("ISA", "MSFT", "Microsoft", weight=9.0, ret=4.0, value=900.0),
        ]
        isa = _isa_account(positions)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-20 14:00",
            fetched_at_ms=int(time.time() * 1000),
            total_combined_value=10000.0,
            total_combined_cash=isa.total_cash,
            isa=isa,
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=2000.0, investments_value=0.0, investments_cost=0.0, unrealized_pnl=0.0, total_cash=2000.0, available_to_trade=2000.0, reserved_cash=0.0, cash_ratio_pct=100.0),
            gbp_usd_rate=1.27,
        )
        cfg = _chat_cfg()
        line = _build_intraday_structure_line(cfg, portfolio, None, "14:05")
        self.assertIn("【盘中实时结构校准（14:05 UK", line)
        # IBTA.L is a non-core ETF -> other_etf, NOT an equity satellite (2 not 3)
        self.assertIn("个股卫星 2 只", line)
        self.assertIn("ISA 核心宽基 36.0%", line)
        self.assertIn("Invest 在险 0 槽（保本 0 只）", line)
        self.assertIn("现金 100.0%", line)

    async def test_bare_fake_t212_without_refresh_portfolio_fx_is_safe(self) -> None:
        class _BareFakeT212:
            def __init__(self, data: PortfolioData) -> None:
                self._data = data

            async def get_portfolio(self, use_cache: bool = True) -> PortfolioData:
                return self._data

        portfolio = _sample_portfolio()
        portfolio.invest.gbp_usd_rate = None
        portfolio.gbp_usd_rate = None
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_bare.db")
            await db.init()
            llm = _SpyLLM()
            reply = await answer_user_chat(
                "普通问题", _chat_cfg(), _BareFakeT212(portfolio), _FakeMarketData(), llm, db
            )
            self.assertEqual(len(llm.calls), 1)
            self.assertIn("普通问题", llm.calls[-1]["prompt"])
            self.assertTrue(reply)

    async def test_deep_prefix_strip_empty_guard_and_history_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_deep.db")
            await db.init()
            llm = _SpyLLM()

            # Empty /deep: 0 LLM calls, no chat_history write
            reply = await answer_user_chat("   /deep  ", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            self.assertIn("/deep 用法提示", reply)
            self.assertEqual(len(llm.calls), 0)
            self.assertEqual(await db.get_chat_history(), [])

            # /deep with a question: prefix stripped everywhere
            await answer_user_chat("/deep 英伟达现在怎么样", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            call = llm.calls[-1]
            self.assertEqual(call["model"], "gemini-4-neon")
            self.assertIn("【用户当前问题】：\n英伟达现在怎么样", call["prompt"])
            self.assertNotIn("/deep", call["prompt"])
            turns = await db.get_chat_history()
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0][0], "英伟达现在怎么样")

    async def test_long_reply_tail_survives_into_next_turn_prompt(self) -> None:
        plan_tail = "【后续执行预案】买入10股@99.50，止损92.00，跌破MA20即撤离。"
        long_reply = "开头结论：可分批建仓。" + "论" * 700 + plan_tail + "尾" * 150
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_headtail.db")
            await db.init()
            llm = _SpyLLM(long_reply)
            await answer_user_chat("/deep 分析英伟达", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            llm2 = _SpyLLM()
            await answer_user_chat("刚才说的挂单参数再说一遍", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm2, db)
            prompt2 = llm2.calls[-1]["prompt"]
            self.assertIn("…（中略）…", prompt2)
            self.assertIn(plan_tail, prompt2)

    async def test_reply_markdown_is_cleaned_before_delivery_and_history(self) -> None:
        raw = "**结论**：持有。\n# 今日盘面\n• 止损按 3*ATR 计算"
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_plain.db")
            await db.init()
            llm = _SpyLLM(raw)
            reply = await answer_user_chat("普通问题", _chat_cfg(), _FakeT212(_sample_portfolio()), _FakeMarketData(), llm, db)
            self.assertNotIn("**", reply)
            self.assertNotIn("# ", reply)
            self.assertIn("3*ATR", reply)
            turns = await db.get_chat_history()
            self.assertNotIn("**", turns[0][1])

    async def test_static_prefix_byte_stable_across_turns_and_us_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_prefix.db")
            await db.init()
            await db.save_morning_report("早报正文", "结论", ["红线甲。"])
            await db.save_pick_radar("f", "r", "PICK_STABLE_MARKER", "2026-11-20 09:00")

            llm = _SpyLLM()
            t212 = _FakeT212(_sample_portfolio())
            md = _FakeMarketData()
            cfg = _chat_cfg()

            await answer_user_chat("现在能加仓NVDA吗", cfg, t212, md, llm, db)
            p1 = llm.calls[-1]["prompt"]
            self.assertIn("PICK_STABLE_MARKER", p1)
            self.assertNotIn("【今日美股开盘前哨补充】", p1)

            # Afternoon /us appends premarket_note (overwriting updated_at_ms)
            await db.append_premarket_note("✅ 盘前平稳，维持早报计划")
            await answer_user_chat("那ULVR呢", cfg, t212, md, llm, db)
            p2 = llm.calls[-1]["prompt"]
            self.assertIn("【今日美股开盘前哨补充】", p2)
            self.assertIn("PICK_STABLE_MARKER", p2)
            # Static prefix stays byte-identical through the time line anchor
            self.assertEqual(
                p1[: p1.index("当前英国时间 ")],
                p2[: p2.index("当前英国时间 ")],
            )

            # A third turn in the same cycle keeps the prefix identical too
            await answer_user_chat("再看看现金", cfg, t212, md, llm, db)
            p3 = llm.calls[-1]["prompt"]
            self.assertEqual(
                p1[: p1.index("当前英国时间 ")],
                p3[: p3.index("当前英国时间 ")],
            )

    async def test_deep_focused_query_compacts_routine_positions(self) -> None:
        positions = [
            _pos("ISA", "NVDA", "NVIDIA", weight=30.0, ret=12.0, value=3000.0),
        ] + [
            _pos("ISA", f"SYM{i}", f"Company {i}", weight=2.0 + i * 0.1, ret=1.0, price=50.0 + i, currency="USD", value=200.0 + i * 10)
            for i in range(1, 8)
        ]
        isa = _isa_account(positions)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-20 14:00",
            fetched_at_ms=int(time.time() * 1000),
            total_combined_value=10000.0,
            total_combined_cash=isa.total_cash,
            isa=isa,
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=2000.0, total_cash=2000.0, available_to_trade=2000.0, cash_ratio_pct=100.0),
            gbp_usd_rate=1.27,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_deepcompact.db")
            await db.init()
            llm = _SpyLLM()
            await answer_user_chat(
                "/deep 英伟达的加仓计划", _chat_cfg(), _FakeT212(portfolio), _FakeMarketData(), llm, db
            )
            prompt = llm.calls[-1]["prompt"]
            self.assertIn("一行速览", prompt)
            self.assertIn("• SYM1 [", prompt)
            # focus row keeps full detail; routine rows lose 4-decimal share counts
            self.assertIn("NVIDIA", prompt)
            self.assertNotIn("SYM7 (Company 7)", prompt)

    async def test_normal_chat_keeps_existing_compact_summary_behavior(self) -> None:
        positions = [
            _pos("ISA", "NVDA", "NVIDIA", weight=30.0, ret=12.0, value=3000.0),
        ] + [
            _pos("ISA", f"SYM{i}", f"Company {i}", weight=2.0, ret=1.0, value=200.0)
            for i in range(1, 4)
        ]
        isa = _isa_account(positions)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-20 14:00",
            fetched_at_ms=int(time.time() * 1000),
            total_combined_value=10000.0,
            total_combined_cash=isa.total_cash,
            isa=isa,
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=2000.0, total_cash=2000.0, available_to_trade=2000.0, cash_ratio_pct=100.0),
            gbp_usd_rate=1.27,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_normalcompact.db")
            await db.init()
            llm = _SpyLLM()
            await answer_user_chat(
                "英伟达现在怎么样", _chat_cfg(), _FakeT212(portfolio), _FakeMarketData(), llm, db
            )
            prompt = llm.calls[-1]["prompt"]
            self.assertIn("其余 3 只常规持仓", prompt)
            self.assertNotIn("一行速览", prompt)


if __name__ == "__main__":
    unittest.main()
