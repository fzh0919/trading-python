"""v5.5.8 /us Delta-Sentinel regression tests.

Covers the 2026-11-20 plan (docs/plans/2026-11-20-5.5.8-us-sentinel-plan.md) §5:
premarket freshness fuse (Yahoo 15m + Finnhub `t`), true-delta VIX gate, closed-market
fuse, 0-token order reconciliation, no-shell-row DB guard, generic-vs-event search cache
fix, single-tool tightening with light I/O, and the ✅<=450 / 🚨<=600 plain-text contract.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.prompts import (
    build_agent_system_instruction,
    clean_telegram_plain_text,
    has_complete_embedded_vetoes,
)
from src.agents.report_agent import generate_us_sentinel
from src.agents.tools import (
    LONDON_TZ as TOOLS_LONDON_TZ,
    MarketToolkit,
    _is_generic_symbol_news_query,
    clear_grounding_cache,
)
from src.config import AppConfig, EnvSettings, StrategySection
from src.data_providers.market_data import MacroSnapshot
from src.data_providers.t212_client import PendingOrderItem, PortfolioData
from src.engine.calendar_utils import UsMarketSchedule
from src.storage.db import AdvisorDatabase
from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio


def _sched(phase: str, ny_date: str = "2026-11-16") -> UsMarketSchedule:
    return UsMarketSchedule(
        ny_date=ny_date,
        uk_date="2026-11-16",
        uk_weekday_label="周一",
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
        phase=phase,
        minutes_to_open=15,
        minutes_since_open=-15,
        minutes_to_close=405,
    )


def _cfg() -> AppConfig:
    return AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))


class _StubLLM:
    def __init__(self, reply: str = "✅ 维持早报计划\n• 观察1：盘前平稳。") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if kwargs.get("use_google_search"):
            return "美股异动检索结果：NVDA 盘前异动归因（测试摘要）。"
        return self.reply


class _SentinelMarketData(_FakeMarketData):
    """Fake market data with per-call tracking for the /us freshness & I/O tests."""

    def __init__(
        self,
        *,
        finnhub_key: str = "",
        prepost: dict[str, dict] | None = None,
        realtime: dict[str, dict] | None = None,
        macro_vix: float = 15.2,
    ) -> None:
        super().__init__(finnhub_key=finnhub_key)
        self.prepost = prepost or {}
        self.realtime = realtime or {}
        self.macro_vix = macro_vix
        self.prepost_calls: list[list[str]] = []
        self.realtime_calls: list[str] = []
        self.bundles_calls: list[dict] = []
        self.deep_intel_calls: list[str] = []

    async def get_us_realtime_quote(self, symbol: str) -> dict | None:
        self.realtime_calls.append(symbol)
        return self.realtime.get(symbol)

    async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict]:
        self.prepost_calls.append(list(symbols))
        return {s: dict(self.prepost[s]) for s in symbols if s in self.prepost}

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        **kwargs,
    ):
        self.bundles_calls.append(
            {"symbols": list(symbols), "include_news": include_news, "include_earnings": include_earnings}
        )
        return await super().get_bundles(symbols, include_news=include_news, include_earnings=include_earnings, **kwargs)

    async def get_us_deep_intel_batch(self, symbols: list[str]) -> dict:
        self.deep_intel_calls.extend(symbols)
        return {}

    async def get_macro_snapshot(self) -> MacroSnapshot:
        return MacroSnapshot(
            fetched_at_utc="2026-11-16 09:00 UTC",
            gbp_usd=1.2750,
            us_10y_yield=4.35,
            vix=self.macro_vix,
            spy_5d_pct=1.15,
        )


def _portfolio_with_amzn_order() -> PortfolioData:
    base = _sample_portfolio()
    invest = base.invest.model_copy(
        update={
            "pending_orders": [
                PendingOrderItem(
                    id="p-amzn",
                    account="Invest",
                    raw_ticker="AMZN_US_EQ",
                    ticker="AMZN",
                    yahoo_symbol="AMZN",
                    name="Amazon",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=2.0,
                    value=360.0,
                    limit_price=180.0,
                    currency="USD",
                    time_in_force="GTC",
                )
            ]
        }
    )
    return base.model_copy(update={"invest": invest})


def _sentinel_call(llm: _StubLLM) -> dict:
    return next(c for c in llm.calls if c.get("usage_label") == "us_sentinel")


def _grounding_calls(llm: _StubLLM) -> list[dict]:
    return [c for c in llm.calls if c.get("usage_label") == "us_grounding"]


def _fresh_nvda_quote(pct: float = 4.5, *, source: str = "finnhub_quote") -> dict:
    return {
        "symbol": "NVDA",
        "current_price": round(100.0 * (1.0 + pct / 100.0), 2),
        "prev_close": 100.0,
        "change": round(pct, 2),
        "change_pct": round(pct, 4),
        "bar_date_et": "2026-11-16",
        "quote_time_et": "10:00",
        "session_tag": "REG",
        "source": source,
    }


class TestUsSentinelV558(unittest.IsolatedAsyncioTestCase):
    async def test_premarket_stale_quotes_do_not_trigger_gate_but_fresh_do(self) -> None:
        """PRE 阶段：隔日 Yahoo 15m / Finnhub t 报价不触发；当日新鲜或无 bar_date_et 桩正常触发。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_fresh.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            pre = _sched("PRE")
            ny = pre.ny_date
            stale_date = "2026-11-13"  # 上一交易日

            # 1) Yahoo 15m 隔日 K 线：dp +4% 但 bar_date_et != ny_date -> 平稳
            stale_15m = {
                "symbol": "NVDA",
                "current_price": 104.0,
                "prev_close": 100.0,
                "change": 4.0,
                "change_pct": 4.0,
                "bar_date_et": stale_date,
                "quote_time_et": "16:00",
                "session_tag": "POST",
                "source": "yfinance_prepost_15m",
            }
            clear_grounding_cache()
            llm = _StubLLM()
            md = _SentinelMarketData(prepost={"NVDA": stale_15m})
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=pre):
                await generate_us_sentinel(cfg, t212, md, llm, db)
            call = _sentinel_call(llm)
            self.assertEqual(call["model"], "deepseek-flash")
            self.assertIn("市场与美股持仓均平稳", call["prompt"])
            self.assertIn("盘前暂无新成交，沿用昨收 100.00", call["prompt"])
            self.assertEqual(_grounding_calls(llm), [])

            # 2) Finnhub 隔日 t 报价同样不触发
            clear_grounding_cache()
            llm_fh = _StubLLM()
            md_fh = _SentinelMarketData(
                finnhub_key="fh-test",
                realtime={"NVDA": {**_fresh_nvda_quote(4.0), "bar_date_et": stale_date, "quote_time_et": "16:00"}},
            )
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=pre):
                await generate_us_sentinel(cfg, t212, md_fh, llm_fh, db)
            self.assertEqual(_sentinel_call(llm_fh)["model"], "deepseek-flash")
            self.assertIn("盘前暂无新成交，沿用昨收 100.00", _sentinel_call(llm_fh)["prompt"])

            # 3) 当日新鲜 bar_date_et == ny_date：正常触发 + Prompt 注入实时价与时段标签
            clear_grounding_cache()
            llm_fresh = _StubLLM("🚨 盘前增量修正\n• NVDA 盘前跳空 +4%。")
            md_fresh = _SentinelMarketData(
                prepost={"NVDA": {**_fresh_nvda_quote(4.0), "session_tag": "PRE", "quote_time_et": "05:15", "source": "yfinance_prepost_15m"}},
            )
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=pre):
                out = await generate_us_sentinel(cfg, t212, md_fresh, llm_fresh, db)
            call_fresh = _sentinel_call(llm_fresh)
            self.assertEqual(call_fresh["model"], "gemini-4-hydrogen")
            self.assertIn("本地异动门控：已触发深度复核", call_fresh["prompt"])
            self.assertIn("美股盘前/盘中实时快照", call_fresh["prompt"])
            self.assertIn("实时 104.00 USD", call_fresh["prompt"])
            self.assertIn("时段 PRE 05:15 ET", call_fresh["prompt"])
            self.assertTrue(out.startswith("🚨"))
            self.assertEqual(len(_grounding_calls(llm_fresh)), 1)

            # 4) 无 bar_date_et 的单测桩保持向后兼容（沿用 imp20 语义）
            clear_grounding_cache()
            llm_stub = _StubLLM("🚨 盘前增量修正\n• NVDA 盘前跳空。")
            md_stub = _SentinelMarketData(
                prepost={"NVDA": {"symbol": "NVDA", "current_price": 104.2, "prev_close": 100.0, "change": 4.2, "change_pct": 4.2, "source": "yfinance_prepost_15m"}},
            )
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=pre):
                await generate_us_sentinel(cfg, t212, md_stub, llm_stub, db)
            self.assertEqual(_sentinel_call(llm_stub)["model"], "gemini-4-hydrogen")
            self.assertIn("+4.20%", _sentinel_call(llm_stub)["prompt"])

    async def test_vix_delta_gate_requires_increment_or_new_cross(self) -> None:
        """VIX 真 Delta 门控：早报已 >=28 时未超增量不触发；新突破 28 或增量达标才触发。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_vix.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            open_sched = _sched("OPEN")

            async def _seed(morning_vix: float) -> None:
                await db.save_morning_report(
                    "【今日早报】正文（含 VIX 基线）",
                    "结论",
                    guardrails=[],
                    context_pack={"macro": {"vix": morning_vix}, "symbols": {}},
                )

            # 1) 早报 VIX=28.4 -> 盘前 29.0（+2.1%，未超 ±4% 且未新突破）-> 平稳
            await _seed(28.4)
            clear_grounding_cache()
            llm_calm = _StubLLM()
            md_calm = _SentinelMarketData(macro_vix=29.0)
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, md_calm, llm_calm, db)
            call_calm = _sentinel_call(llm_calm)
            self.assertEqual(call_calm["model"], "deepseek-flash")
            self.assertIn("市场与美股持仓均平稳", call_calm["prompt"])
            self.assertEqual(_grounding_calls(llm_calm), [])

            # 2) 早报 VIX=25.0 -> 盘前 28.5（新突破 28）-> 触发
            db2 = AdvisorDatabase(Path(tmpdir) / "v558_vix_cross.db")
            await db2.init()
            await db2.save_morning_report(
                "【今日早报】正文",
                "结论",
                guardrails=[],
                context_pack={"macro": {"vix": 25.0}, "symbols": {}},
            )
            clear_grounding_cache()
            llm_cross = _StubLLM("🚨 盘前增量修正\n• VIX 早报后急升。")
            md_cross = _SentinelMarketData(macro_vix=28.5)
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                out = await generate_us_sentinel(cfg, t212, md_cross, llm_cross, db2)
            call_cross = _sentinel_call(llm_cross)
            self.assertEqual(call_cross["model"], "gemini-4-hydrogen")
            self.assertIn("新突破 28 警戒线", call_cross["prompt"])
            self.assertTrue(out.startswith("🚨"))

            # 3) 早报 VIX=25.0 -> 盘前 26.25（+5.0% 增量达标，未破 28）-> 触发
            db3 = AdvisorDatabase(Path(tmpdir) / "v558_vix_delta.db")
            await db3.init()
            await db3.save_morning_report(
                "【今日早报】正文",
                "结论",
                guardrails=[],
                context_pack={"macro": {"vix": 25.0}, "symbols": {}},
            )
            clear_grounding_cache()
            llm_delta = _StubLLM("🚨 盘前增量修正\n• VIX 增量跳升。")
            md_delta = _SentinelMarketData(macro_vix=26.25)
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, md_delta, llm_delta, db3)
            call_delta = _sentinel_call(llm_delta)
            self.assertEqual(call_delta["model"], "gemini-4-hydrogen")
            self.assertIn("VIX 较早报基线增量 +5.00%", call_delta["prompt"])

            # 4) 无早报包时保留原 VIX 日线变动兜底（冷启动兼容）
            db4 = AdvisorDatabase(Path(tmpdir) / "v558_vix_fallback.db")
            await db4.init()

            class _VixHistMD(_SentinelMarketData):
                async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                    out = await super().get_histories(symbols, period=period)
                    if "^VIX" in out:
                        closes = np.full(230, 16.0)
                        closes[-1] = 16.0 * 1.08
                        out["^VIX"] = pd.DataFrame(
                            {
                                "Open": closes * 0.998,
                                "High": closes * 1.005,
                                "Low": closes * 0.995,
                                "Close": closes,
                                "Volume": np.full(230, 1_200_000.0),
                            },
                            index=pd.date_range("2025-10-01", periods=230, freq="B"),
                        )
                    return out

            clear_grounding_cache()
            llm_fb = _StubLLM("🚨 盘前增量修正\n• VIX 日线急升。")
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, _VixHistMD(), llm_fb, db4)
            self.assertIn("VIX 日变动 +8.00%", _sentinel_call(llm_fb)["prompt"])

    async def test_weekend_and_holiday_fuse_skips_quotes_search_and_uses_ds_flash(self) -> None:
        """休市/周末熔断：不拉实时/盘前行情、不跑 us_grounding、锁定 deepseek-flash。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_closed.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())

            for phase in ("WEEKEND", "HOLIDAY"):
                clear_grounding_cache()
                llm = _StubLLM()
                md = _SentinelMarketData(
                    finnhub_key="fh-test",
                    realtime={"NVDA": _fresh_nvda_quote(4.5)},
                    prepost={"NVDA": _fresh_nvda_quote(4.5)},
                )
                sched = _sched(phase)
                with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=sched):
                    out = await generate_us_sentinel(cfg, t212, md, llm, db)
                call = _sentinel_call(llm)
                self.assertEqual(call["model"], "deepseek-flash")
                self.assertIsNone(call["tools"])
                self.assertIn("市场与美股持仓均平稳", call["prompt"])
                self.assertIn("休市中无实时成交", call["prompt"])
                self.assertEqual(md.prepost_calls, [], f"{phase} 不应拉取 15m 盘前行情")
                self.assertEqual(md.realtime_calls, [], f"{phase} 不应拉取 Finnhub 实时行情")
                self.assertEqual(_grounding_calls(llm), [], f"{phase} 不应跑 us_grounding")
                self.assertTrue(out.startswith("✅"))

    async def test_order_reconciliation_and_pending_symbol_search_merge(self) -> None:
        """0-Token 挂单对账：已撤销挂单标「无需再操作」，存活挂单并入异动日检索目标。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_recon.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_portfolio_with_amzn_order())
            open_sched = _sched("OPEN")

            await db.save_morning_report(
                "【今日早报】建议撤销 EOG 买单；AMZN 买单继续等待。",
                "结论",
                guardrails=["冻结双账户个股加仓"],
                context_pack={
                    "macro": {"vix": 15.2},
                    "symbols": {},
                    "pending_orders": [
                        {"account": "Invest", "yahoo_symbol": "EOG", "side": "BUY", "limit_price": 148.0, "stop_price": None},
                        {"account": "Invest", "yahoo_symbol": "AMZN", "side": "BUY", "limit_price": 180.0, "stop_price": None},
                    ],
                },
            )

            clear_grounding_cache()
            llm = _StubLLM("🚨 盘前增量修正\n• NVDA 异动，重新核验。")
            md = _SentinelMarketData(finnhub_key="fh-test", realtime={"NVDA": _fresh_nvda_quote(4.5)})
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                out = await generate_us_sentinel(cfg, t212, md, llm, db)
            call = _sentinel_call(llm)
            self.assertIn("早报后美股挂单变动对账", call["prompt"])
            self.assertIn("已撤销/已成交消失（无需再操作）：[Invest] EOG BUY 148.00 USD", call["prompt"])
            self.assertIn("当前仍有效美股挂单：[Invest] AMZN BUY 限价 180.00", call["prompt"])
            self.assertTrue(out.startswith("🚨"))

            # 活动挂单标的 AMZN 自动并入前置搜索目标（NVDA 1v1 深搜 + AMZN 批搜）
            grounding = _grounding_calls(llm)
            self.assertEqual(len(grounding), 2)
            joined = "\n".join(c["prompt"] for c in grounding)
            self.assertIn("NVDA", joined)
            self.assertIn("AMZN", joined)

    async def test_no_morning_report_never_writes_shell_row(self) -> None:
        """无早报日运行 /us 不得插入 report_text='' 空壳行（保留上一周期兜底早报）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_shell.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            open_sched = _sched("OPEN")

            yesterday = "2026-11-02"
            await db.save_morning_report(
                "上一周期早报正文（兜底基线）",
                "上一周期结论",
                guardrails=["旧周期红线"],
                cycle_date=yesterday,
            )

            clear_grounding_cache()
            llm = _StubLLM()
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, _SentinelMarketData(), llm, db)

            async with db._connect() as conn:
                cur = await conn.execute("SELECT cycle_date, report_text FROM morning_reports ORDER BY cycle_date")
                rows = await cur.fetchall()
            self.assertEqual([r[0] for r in rows], [yesterday])
            self.assertEqual(rows[0][1], "上一周期早报正文（兜底基线）")

            # 有当日早报时正常回写 premarket_note（防护不过度拦截）
            await db.save_morning_report(
                "【今日早报】正文",
                "今日结论",
                guardrails=["今日红线：冻结芯片追高"],
            )
            clear_grounding_cache()
            memo = "✅ 维持早报计划\n• 观察：平稳。"
            llm2 = _StubLLM(memo)
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, _SentinelMarketData(), llm2, db)
            note = await db.get_morning_context_meta()
            self.assertIn(memo, note[1])

    async def test_anomaly_day_single_tool_and_light_io_isolation(self) -> None:
        """异动日仅暴露 search_market_news + max_tool_rounds=1；轻量事实组装不拉英欧股/深度接口/新闻。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_tools.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            open_sched = _sched("OPEN")

            clear_grounding_cache()
            llm = _StubLLM("🚨 盘前增量修正\n• NVDA 异动。")
            md = _SentinelMarketData(finnhub_key="fh-test", realtime={"NVDA": _fresh_nvda_quote(4.5)})
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, md, llm, db)
            call = _sentinel_call(llm)

            tool_names = [d.name for d in (call.get("tools") or [])]
            self.assertEqual(tool_names, ["search_market_news"])
            self.assertEqual(call.get("max_tool_rounds"), 1)
            self.assertIn("仅当发现未覆盖的突发催化缺口时才可调用 1 次 search_market_news", call["prompt"])
            self.assertIn("禁止重复调 get_macro_snapshot", call["prompt"])

            # I/O 隔离：只拉美股、关新闻、零 Finnhub 深度接口
            self.assertEqual(len(md.bundles_calls), 1)
            self.assertEqual(md.bundles_calls[0]["symbols"], ["NVDA"])
            self.assertFalse(md.bundles_calls[0]["include_news"])
            self.assertTrue(md.bundles_calls[0]["include_earnings"])
            self.assertEqual(md.deep_intel_calls, [])

            # 内容瘦身：不再注入规则引擎加仓信号、Yahoo RSS 新闻与跨账户暴露
            self.assertNotIn("Python 规则引擎", call["prompt"])
            self.assertNotIn("近期公司新闻摘要", call["prompt"])
            self.assertNotIn("跨账户重叠与行业集中度", call["prompt"])
            self.assertIn("美股紧凑技术锚点", call["prompt"])

    async def test_embedded_vetoes_not_duplicated_in_us_prompt(self) -> None:
        """早报正文已内嵌完整红线时，/us Prompt 不再重复注入「今日风控红线」。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_veto.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            open_sched = _sched("OPEN")

            guardrails = ["冻结双账户个股加仓", "严禁芯片追高"]
            report_text = "【今日早报】\n【今日风控红线】\n• 冻结双账户个股加仓\n• 严禁芯片追高\n其余正文。"
            await db.save_morning_report(report_text, "结论", guardrails=guardrails)
            self.assertTrue(has_complete_embedded_vetoes(report_text, guardrails))

            clear_grounding_cache()
            llm = _StubLLM()
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                await generate_us_sentinel(cfg, t212, _SentinelMarketData(), llm, db)
            prompt = _sentinel_call(llm)["prompt"]
            self.assertNotIn("=== 今日风控红线 ===", prompt)
            self.assertEqual(prompt.count("今日风控红线"), 1)  # 仅早报正文内那一份

    async def test_length_ceiling_decoupled_from_gate_and_plain_text_clean(self) -> None:
        """字数契约：门控触发但结论为 ✅ 仍按 450 裁剪；🚨 按 600；输出剥离 ** 与行首 #。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_len.db")
            await db.init()
            cfg = _cfg()
            t212 = _FakeT212(_sample_portfolio())
            open_sched = _sched("OPEN")

            # 门控触发（NVDA 实时 +4.5%）但模型返回 ✅ -> 严格 450 上限
            clear_grounding_cache()
            calm_reply = "✅ 维持早报计划\n• 观察1：盘前平稳。\n• 观察2：" + ("细节" * 240)
            llm_calm = _StubLLM(calm_reply)
            md = _SentinelMarketData(finnhub_key="fh-test", realtime={"NVDA": _fresh_nvda_quote(4.5)})
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                out_calm = await generate_us_sentinel(cfg, t212, md, llm_calm, db)
            self.assertLessEqual(len(out_calm), 450)
            self.assertTrue(out_calm.startswith("✅"))
            self.assertIn("本地异动门控：已触发深度复核", _sentinel_call(llm_calm)["prompt"])

            # 🚨 结论按 600 上限，且纯文本清洗剥离 ** 与行首 # 标题、保留 1.4*ATR 与换行
            clear_grounding_cache()
            anom_reply = (
                "🚨 **盘前增量修正**\n"
                "# NVDA 异动速报\n"
                "• 止损上移至 1.4*ATR。\n"
                "• 归因：" + ("核心逻辑修正" * 110)
            )
            self.assertGreater(len(anom_reply), 600)
            llm_anom = _StubLLM(anom_reply)
            with patch("src.agents.report_agent.get_us_market_schedule_uk", return_value=open_sched):
                out_anom = await generate_us_sentinel(cfg, t212, md, llm_anom, db)
            self.assertLessEqual(len(out_anom), 600)
            self.assertTrue(out_anom.startswith("🚨"))
            self.assertNotIn("**", out_anom)
            self.assertNotIn("# NVDA", out_anom)
            self.assertIn("1.4*ATR", out_anom)

    async def test_search_market_news_generic_vs_event_cache_pierce(self) -> None:
        """缓存穿透精准修复：8 词泛新闻命中当日 sym 缓存；具体事件词穿透缓存联网。"""
        from src.agents.chat_agent import _clean_chat_plain_text, _has_complete_embedded_vetoes

        # 下沉后 chat_agent 保留原名别名（既有单测兼容护栏）
        self.assertIs(_clean_chat_plain_text, clean_telegram_plain_text)
        self.assertIs(_has_complete_embedded_vetoes, has_complete_embedded_vetoes)

        self.assertTrue(
            _is_generic_symbol_news_query('"AVGO" OR "Broadcom" stock news November 20 2026', "AVGO")
        )
        self.assertFalse(
            _is_generic_symbol_news_query('"NVDA California attorney general subpoena"', "NVDA")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "v558_cache.db")
            await db.init()
            uk_date = datetime.now(timezone.utc).astimezone(TOOLS_LONDON_TZ).date().isoformat()
            await db.upsert_grounding_news_cache(
                [
                    {"cache_key": f"grounding:{uk_date}:sym:AVGO", "scope": "us", "symbol": "AVGO", "summary_text": "AVGO 当日泛闻缓存摘要。"},
                    {"cache_key": f"grounding:{uk_date}:sym:NVDA", "scope": "us", "symbol": "NVDA", "summary_text": "NVDA 当日泛闻缓存摘要。"},
                ]
            )
            clear_grounding_cache()
            llm = _StubLLM()
            toolkit = MarketToolkit(_FakeMarketData(), _sample_portfolio(), llm=llm, db=db, search_cache_ttl_seconds=1800)

            # 8 词泛新闻查询：无视词数命中 sym:AVGO 当日缓存，0 次联网
            res_generic = await toolkit.execute(
                "search_market_news", {"query": '"AVGO" OR "Broadcom" stock news November 20 2026'}
            )
            self.assertTrue(res_generic.get("cached"))
            self.assertEqual(res_generic["summary"], "AVGO 当日泛闻缓存摘要。")
            self.assertEqual(llm.calls, [])

            # 含具体事件词（subpoena）：穿透缓存执行定向 Google Search
            res_event = await toolkit.execute(
                "search_market_news", {"query": '"NVDA California attorney general subpoena"'}
            )
            self.assertNotIn("cached", res_event)
            self.assertEqual(len(llm.calls), 1)
            self.assertTrue(all(c.get("use_google_search") for c in llm.calls))

    async def test_compact_us_sentinel_profile_drops_report_only_machinery(self) -> None:
        """精简画像：保留双账户哲学与挂单计息，删除四桶票据/配额/连损恢复/重审挂单长段落。"""
        strategy = StrategySection()
        sys_text = build_agent_system_instruction("ROLE", strategy, compact_for_us_sentinel=True)
        self.assertIn("美股增量哨兵精简版", sys_text)
        self.assertIn("不使用杠杆/期权/做空", sys_text)
        self.assertIn("巴菲特价值流派", sys_text)
        self.assertIn("右侧动量战术流派", sys_text)
        self.assertIn("照常计息", sys_text)
        self.assertIn("严禁另起炉灶", sys_text)
        # 删除项
        self.assertNotIn("四桶", sys_text)
        self.assertNotIn("配额", sys_text)
        self.assertNotIn("冷静", sys_text)
        self.assertNotIn("一律按挂单限价核算", sys_text)
        self.assertNotIn("碎单", sys_text)


if __name__ == "__main__":
    unittest.main()
