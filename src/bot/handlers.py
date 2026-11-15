"""Telegram command and free-text chat router with quick-action keyboard, /health diagnostics, and single-flight cancellation."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from src.agents.chat_agent import answer_user_chat, format_positions_instant
from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import generate_morning_report, generate_us_sentinel
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.storage.db import AdvisorDatabase
from src.version import APP_VERSION

if TYPE_CHECKING:
    from src.bot.scheduler import TradingScheduler
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

REMOVE_KEYBOARD_MARKUP: dict[str, Any] = {"remove_keyboard": True}

BOT_COMMANDS: list[dict[str, str]] = [
    {"command": "pos", "description": "实时双账户持仓、净值与挂单"},
    {"command": "report", "description": "全景深度晨报（含全市场联网检索）"},
    {"command": "pick", "description": "全市场选股雷达（可附方向，如 /pick 英股ETF）"},
    {"command": "us", "description": "美股盘前/盘中/收盘增量哨兵"},
    {"command": "stats", "description": "历史选股与调仓记录档案"},
    {"command": "help", "description": "查看核心与运维全部指令说明"},
]


def _normalize_command_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd_head = parts[0].split("@", maxsplit=1)[0]
        return f"{cmd_head} {parts[1]}" if len(parts) > 1 else cmd_head
    return text


def _format_uptime(elapsed_s: float) -> str:
    secs = max(0, int(elapsed_s))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}天{hours}小时{mins}分"
    if hours > 0:
        return f"{hours}小时{mins}分"
    return f"{mins}分"


class AdvisorBot:
    def __init__(
        self,
        cfg: AppConfig,
        t212: T212Client,
        market_data: MarketDataService,
        llm: LLMBackend,
        db: AdvisorDatabase,
        telegram: TelegramClient,
        ops_notifier: OpsNotifier | None = None,
        scheduler: TradingScheduler | None = None,
        started_monotonic: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self.scheduler = scheduler
        self._started_monotonic = started_monotonic if started_monotonic is not None else time.monotonic()
        self._started_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        self._active_task: asyncio.Task[None] | None = None
        self._active_status_msg_id: int | None = None

    def is_authorized(self, chat_id: str | int) -> bool:
        expected = self.cfg.env.telegram_chat_id.strip()
        return bool(expected) and str(chat_id) == expected

    async def cancel_active_task(self, chat_id: str | int) -> bool:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except asyncio.CancelledError:
                pass
            if self._active_status_msg_id:
                await self.tg.edit_status(chat_id, self._active_status_msg_id, "🛑 已取消上一条分析任务。")
                self._active_status_msg_id = None
            return True
        return False

    async def _send_reply(self, chat_id: str | int, text: str, *, remove_keyboard: bool = False) -> None:
        if remove_keyboard:
            await self.tg.send_message(chat_id, text, reply_markup=REMOVE_KEYBOARD_MARKUP)
        else:
            await self.tg.send_message(chat_id, text)

    @staticmethod
    def _build_help_text() -> str:
        return (
            f"🤖 Trading 212 × Gemini 量化投顾（v{APP_VERSION}）已就绪！\n\n"
            "【核心投研指令（见左侧 / 菜单栏）】\n"
            "• /pos — 瞬时拉取双账户实时净值、仓位与近7日成交（免等待 AI）\n"
            "• /report — 深度全景晨报（含全市场分域联网检索、30日复盘、逐仓与挂单审计）\n"
            "• /pick [方向] — 全市场选股雷达（美/英/欧股 + LSE ETF 联网催化 + 风控终审）\n"
            "• /us — 美股盘前/盘中/收盘/休市自适应增量盯盘简报\n"
            "• /stats — 查看近期历史选股与加减仓记录档案\n"
            "• 直接打字 — 全天单聊（自带今日晨报基线、实时查价与谷歌搜新闻工具）\n\n"
            "【运维与系统控制指令】\n"
            "• /health — 全链路实时自检（T212/行情/FRED/Finnhub/Gemini/DB 与近期告警）\n"
            "• /usage — 查看 Gemini Token 消耗、缓存命中率与本月 Google Search 免费额度进度\n"
            "• /clear — 清空今日对话记忆重新开始\n"
            "• /cancel — 立即中止正在运行的耗时分析"
        )

    async def handle_update(self, update: dict[str, Any], wait: bool = False) -> None:
        msg = update.get("message")
        if not msg or not msg.get("text"):
            return
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        text = _normalize_command_text(str(msg["text"]))

        if not self.is_authorized(chat_id):
            logger.warning("Rejected unauthorized Telegram chat_id=%s", chat_id)
            return

        lower = text.lower()
        if lower.startswith("/cancel"):
            cancelled = await self.cancel_active_task(chat_id)
            await self._send_reply(
                chat_id,
                "🛑 已中止正在执行的分析任务。" if cancelled else "ℹ️ 当前没有正在执行的任务。",
            )
            return

        if lower.startswith("/help") or lower.startswith("/start"):
            await self._send_reply(chat_id, self._build_help_text(), remove_keyboard=True)
            return

        if lower.startswith("/clear"):
            await self.db.clear_chat_history()
            await self._send_reply(chat_id, "🧹 今日对话记忆已清空，请开始新的探讨！")
            return

        if lower.startswith("/stats"):
            rows = await self.db.get_recent_recommendations(limit=10)
            if not rows:
                await self._send_reply(chat_id, "📭 暂无历史选股或调仓记录（运行一次 /pick 后会自动建档）。")
                return
            lines = ["📒 最近 10 条 /pick 与加减仓存档："]
            for r in rows:
                price_s = f" @ {r['entry_price']}" if r["entry_price"] is not None else ""
                lines.append(
                    f"• [{r['created_at_uk']}] [{r['account']}] {r['action']} {r['symbol']}{price_s}：{r['thesis']}"
                )
            await self._send_reply(chat_id, "\n".join(lines))
            return

        if lower.startswith("/usage"):
            summary = await self.db.get_usage_summary(days=30)
            await self._send_reply(chat_id, self._format_usage_message(summary))
            return

        # Cancel any still-running heavy command when user issues a new query
        if self._active_task and not self._active_task.done():
            await self.cancel_active_task(chat_id)

        self._active_task = asyncio.create_task(self._run_query_task(chat_id, text))
        if wait:
            await self._active_task

    async def _run_health_check(self) -> str:
        portfolio, macro, ops_events, usage_30d = await asyncio.gather(
            self.t212.get_portfolio(use_cache=True),
            self.market_data.get_macro_snapshot(),
            self.db.get_recent_ops_events(limit=5),
            self.db.get_usage_summary(days=30),
        )
        uptime_s = _format_uptime(time.monotonic() - self._started_monotonic)

        def _acc_line(label: str, acc: Any) -> str:
            if not acc.ok:
                return f"⚠️ {label}: 失败 ({str(acc.error)[:80]})"
            ord_s = "挂单✓" if acc.pending_orders_supported else "挂单403/关"
            hist_s = "历史✓" if acc.order_history_supported else "历史403/关"
            return f"✅ {label}: 净值 {acc.account_currency} {acc.total_account_value:,.2f} ({ord_s}, {hist_s})"

        yf_ok = macro.gbp_usd is not None and macro.spy_5d_pct is not None
        finnhub_status = (
            f"⚠️ 异常 ({self.market_data.finnhub.last_error})"
            if self.market_data.finnhub.last_error
            else ("✅ 已配置正常" if self.market_data.finnhub.enabled else "➖ 未配置")
        )
        fred_status = (
            f"⚠️ 异常 ({self.market_data.fred.last_error})"
            if self.market_data.fred.last_error
            else ("✅ 已配置正常" if self.market_data.fred.enabled else "➖ 未配置")
        )
        gemini_status = (
            f"✅ Key 已配置 | 主板 {self.cfg.models.report_model} | 备援 {self.cfg.models.fallback_model} | "
            f"早报搜索 {self.cfg.models.report_search_model} | 日常搜索 {self.cfg.models.search_model}"
            if self.cfg.env.gemini_api_key.strip()
            else "🚨 GEMINI_API_KEY 缺失"
        )
        # TODO(IMP-15): [接自 IMP-03 & IMP-04 & IMP-05] /health 须检查 portfolio.is_fallback/acc.is_fallback、macro.is_stale
        # 及 db.get_unnotified_ops_events() 未送达/抑制告警；Search 统计建议统一切换为 usage_30d["this_month"]。
        search_q_30d = usage_30d["last_nd"].get("search_queries", 0)

        lines = [
            f"🩺 系统全链路运维体检（v{APP_VERSION}）",
            f"• 启动时间: {self._started_uk} UK | 连续运行: {uptime_s}",
            f"• SQLite & 告警: ✅ 正常 | 强制联网扫描: {'开启' if self.cfg.models.enable_forced_search_grounding else '关闭'}",
            f"• Gemini: {gemini_status}",
            f"• Google Search 30日用量: {search_q_30d:,} / {self.cfg.models.google_search_free_quota_monthly:,} 免费额度",
            "",
            "【外部接口连通性】",
            f"• {_acc_line('T212 ISA', portfolio.isa)}",
            f"• {_acc_line('T212 Invest', portfolio.invest)}",
            f"• Yahoo 行情/宏观: {'✅ 正常' if yf_ok else '⚠️ 数据残缺'} (GBP/USD={macro.gbp_usd or 'N/A'}, VIX={macro.vix or 'N/A'})",
            f"• Finnhub 美股新闻/财历: {finnhub_status}",
            f"• FRED 利率/失业率: {fred_status}",
        ]

        if self.scheduler is not None and self.scheduler.last_job_runs:
            lines.append("\n【最近定时任务记录】")
            for job_name, info in self.scheduler.last_job_runs.items():
                detail = f" ({info['detail']})" if info.get("detail") else ""
                lines.append(f"• {job_name}: {info['status']} @ {info['uk_time']} UK{detail}")

        lines.append("\n【最近 5 条运维事件】")
        if not ops_events:
            lines.append("• 暂无运维异常记录。")
        else:
            for ev in ops_events:
                push_tag = "已推送" if ev["notified"] else "未推送/合并"
                sup = f", 合并{ev['suppressed_count']}次" if ev["suppressed_count"] > 0 else ""
                lines.append(
                    f"• [{ev['created_at_uk']}] [{ev['severity']}] {ev['source']}: {ev['message'][:80]} ({push_tag}{sup})"
                )

        return "\n".join(lines)

    async def _run_query_task(self, chat_id: str, text: str) -> None:
        status_msg_id: int | None = None
        try:
            await self.tg.send_typing(chat_id)
            lower = text.lower()

            if lower.startswith("/pos"):
                status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在拉取 T212 双账户实时仓位与挂单...")
                self._active_status_msg_id = status_msg_id
                reply = await format_positions_instant(self.cfg, self.t212)
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 持仓快照已就绪：")
                await self.tg.send_message(chat_id, reply)
                return

            if lower.startswith("/health") or lower.startswith("/heath"):
                status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在巡检 T212 / 行情 / Gemini / DB 全链路状态...")
                self._active_status_msg_id = status_msg_id
                reply = await self._run_health_check()
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 运维自检报告已生成：")
                await self.tg.send_message(chat_id, reply)
                return

            if lower.startswith("/report"):
                status_msg_id = await self.tg.send_status(
                    chat_id, "⏳ 正在并发拉取 T212、行情与英/欧/美全市场 Google 深度检索并生成晨报..."
                )
                self._active_status_msg_id = status_msg_id
                reply = await generate_morning_report(
                    self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                )
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 全景晨报分析完成：")
                await self.tg.send_message(chat_id, reply)
                return

            if lower.startswith("/us"):
                status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在扫描美股实时阶段与持仓异动...")
                self._active_status_msg_id = status_msg_id
                reply = await generate_us_sentinel(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.llm,
                    self.db,
                    is_scheduled_premarket=False,
                    ops_notifier=self.ops,
                )
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 美股盯盘简报完成：")
                await self.tg.send_message(chat_id, reply)
                return

            if lower.startswith("/pick"):
                focus_query = text[5:].strip()
                label = f"「{focus_query}」" if focus_query else "全市场（美/英/欧股 + LSE ETF + 持仓加减）"
                status_msg_id = await self.tg.send_status(
                    chat_id, f"⏳ 第 1/2 步：量化初筛 + 全市场联网催化扫描中（方向：{label}）..."
                )
                self._active_status_msg_id = status_msg_id
                reply = await generate_pick_radar(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.llm,
                    self.db,
                    focus_query=focus_query,
                    is_weekly_auto=False,
                    ops_notifier=self.ops,
                )
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 投研选股 + 风控官双审完成：")
                await self.tg.send_message(chat_id, reply)
                return

            # Free-form chat
            status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在核对实盘数据并思考回复...")
            self._active_status_msg_id = status_msg_id
            reply = await answer_user_chat(
                text, self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
            )
            await self.tg.edit_status(chat_id, status_msg_id, "✅ 回复如下：")
            await self.tg.send_message(chat_id, reply)
        except asyncio.CancelledError:
            logger.info("Task cancelled for query: %s", text[:40])
            raise
        except Exception as exc:
            logger.exception("Error processing message %s", text[:40])
            await self.tg.edit_status(chat_id, status_msg_id, "❌ 处理失败")
            if self.ops is not None:
                await self.ops.notify(
                    severity="CRITICAL",
                    source="bot:query_task",
                    alert_key="bot_handler_crash",
                    message=f"处理指令「{text[:40]}」时发生异常：{type(exc).__name__}: {exc}",
                )
            await self.tg.send_message(chat_id, f"⚠️ 执行出错：{exc}")
        finally:
            self._active_status_msg_id = None

    def _format_usage_message(self, summary: dict[str, Any]) -> str:
        def _fmt_bucket(title: str, b: dict[str, Any]) -> str:
            in_tok = b["prompt_tokens"]
            cached_tok = b["cached_tokens"]
            out_tok = b["output_tokens"]
            thought_tok = b["thought_tokens"]
            search_q = b.get("search_queries", 0)
            cache_pct = (cached_tok / in_tok * 100.0) if in_tok > 0 else 0.0
            search_str = f" | 搜索 {search_q}次" if search_q else ""
            return (
                f"• {title}：{b['calls']}次 ({b['rounds']}轮){search_str}\n"
                f"  输入 {in_tok:,} (缓存 {cached_tok:,}, {cache_pct:.0f}%) | "
                f"输出 {out_tok:,} + 思考 {thought_tok:,}"
            )

        m = self.cfg.models
        # TODO(IMP-15): [接自 IMP-03] 与 IMP-09 对齐，按自然月 free quota 改用 summary["this_month"].get("search_queries", 0)。
        search_30d = summary["last_nd"].get("search_queries", 0)
        free_quota = m.google_search_free_quota_monthly
        warn_at = int(free_quota * m.google_search_warn_pct / 100.0) if free_quota > 0 else 0
        search_badge = ""
        if free_quota > 0 and search_30d >= free_quota:
            search_badge = " 🚨 已超免费额度（超额按量计费）"
        elif free_quota > 0 and search_30d >= warn_at:
            search_badge = f" ⚠️ 接近 {m.google_search_warn_pct:.0f}% 预警线"

        lines = [
            f"📊 Gemini API Token 与搜索用量看板（周期 {summary['cycle_date']}）",
            _fmt_bucket("今日周期", summary["today"]),
            _fmt_bucket("近 7 天", summary["last_7d"]),
            _fmt_bucket(f"近 {summary['days']} 天", summary["last_nd"]),
            f"🔍 近 {summary['days']} 天 Google Search 查询: {search_30d:,} / {free_quota:,} 月免费额度{search_badge}",
        ]
        if summary["by_label_7d"]:
            lines.append("\n【近 7 天分节点消耗】")
            for item in summary["by_label_7d"]:
                lines.append(
                    f"• {item['label']}: {item['calls']}次 | "
                    f"In {item['prompt_tokens']:,} (缓存 {item['cached_tokens']:,}) / "
                    f"Out+Think {item['output_tokens'] + item['thought_tokens']:,}"
                )
        if summary["by_model_nd"]:
            lines.append(f"\n【近 {summary['days']} 天分模型统计】")
            for item in summary["by_model_nd"]:
                tot_tok = item["prompt_tokens"] + item["output_tokens"] + item["thought_tokens"]
                lines.append(
                    f"• {item['model']}: {item['calls']}次 ({item['rounds']}轮) | 总Token {tot_tok:,}"
                )
        if summary["last_nd"]["calls"] == 0:
            lines.append("\nℹ️ 暂无调用记录（运行 /report、/us、/pick 或单聊后会自动累计）。")

        lines.append(
            "\n【当前模型路由】\n"
            f"• 早报分析: {m.report_analyst_model} ({m.report_analyst_thinking_level}) | "
            f"风控/PM: {m.report_model} ({m.report_thinking_level})\n"
            f"• 美股哨兵: {m.us_sentinel_model} ({m.us_thinking_level}) | "
            f"单聊: {m.chat_model} ({m.chat_thinking_level}) / deep: {m.chat_deep_model} ({m.chat_deep_thinking_level})\n"
            f"• 选股初研: {m.pick_researcher_model} ({m.pick_researcher_thinking_level}) | "
            f"终审: {m.pick_model} ({m.pick_thinking_level})\n"
            f"• 备援模型: {m.fallback_model} ({m.fallback_thinking_level}) | "
            f"早报搜索: {m.report_search_model} | 单聊/日常搜索: {m.search_model}"
        )
        return "\n".join(lines)
