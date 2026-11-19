"""Telegram command and free-text chat router with quick-action keyboard, /health diagnostics, and single-flight cancellation."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from src.agents.chat_agent import answer_user_chat, format_positions_instant
from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import generate_morning_report, generate_us_sentinel
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.analytics import (
    _CORE_POLICY_BENCHMARK_WEIGHTS,
    compute_account_split_performance,
    compute_concentration_metrics,
    compute_core_policy_benchmark_comparison,
    compute_portfolio_performance,
    compute_signal_quality,
    compute_trade_attribution,
)
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client, reconcile_invest_wallets
from src.engine.screener import run_universe_audition
from src.redaction import redact_secrets
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
    {"command": "wallet", "description": "Invest GBP+USD 双钱包对账与校准"},
    {"command": "usd", "description": "上报 Invest USD 钱包余额（/usd 600 [25]）"},
    {"command": "report", "description": "全景深度晨报（含全市场联网检索）"},
    {"command": "pick", "description": "全市场选股雷达（可附方向，如 /pick 英股ETF）"},
    {"command": "us", "description": "美股盘前/盘中/收盘增量哨兵"},
    {"command": "deep", "description": "深度单聊（/deep 问题，4 轮多维推演）"},
    {"command": "stats", "description": "历史选股 T+N 胜率、Alpha 与 MFE/MAE 看板"},
    {"command": "perf", "description": "实盘绩效看板（TWR/回撤/Alpha-Beta-IR/胜率/HHI）"},
    {"command": "help", "description": "查看核心与运维全部指令说明"},
]


def _wallet_numbers(tokens: list[str]) -> list[float]:
    """Extract numeric amounts from user tokens, tolerating , £ $ prefixes."""
    out: list[float] = []
    for tok in tokens:
        cleaned = tok.replace(",", "").replace("£", "").replace("$", "").strip()
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


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
        execution_lock: asyncio.Lock | None = None,
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self.scheduler = scheduler
        self.execution_lock: asyncio.Lock = (
            execution_lock
            if execution_lock is not None
            else (getattr(scheduler, "execution_lock", None) or asyncio.Lock())
        )
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

    async def _deliver(self, chat_id: str | int, text: str, *, remove_keyboard: bool = False) -> bool:
        """Send a Telegram reply and raise a CRITICAL ops alert when the upstream drops it."""
        reply_markup = REMOVE_KEYBOARD_MARKUP if remove_keyboard else None
        try:
            delivered = await self.tg.send_message(chat_id, text, reply_markup=reply_markup)
        except Exception as exc:
            logger.warning("Telegram reply delivery raised (%s: %s)", type(exc).__name__, exc)
            delivered = False
        if not delivered:
            logger.error("Telegram reply undelivered (chat_id=%s, chars=%d)", chat_id, len(text))
            if self.ops is not None:
                try:
                    await self.ops.notify(
                        severity="CRITICAL",
                        source="telegram:delivery",
                        alert_key="telegram_reply_undelivered",
                        message=(
                            f"回复未送达 chat_id={chat_id}（{len(text)} 字符）：Telegram 上游 5xx 或网络中断。"
                            "请检查 getUpdates 轮询日志，恢复后手动重发该问题。"
                        ),
                    )
                except Exception as alert_exc:
                    logger.warning("Failed raising undelivered-reply alert: %s", alert_exc)
        return delivered

    async def _send_reply(self, chat_id: str | int, text: str, *, remove_keyboard: bool = False) -> bool:
        return await self._deliver(chat_id, text, remove_keyboard=remove_keyboard)

    @staticmethod
    def _build_help_text() -> str:
        return (
            f"🤖 Trading 212 × Gemini 量化投顾（v{APP_VERSION}）已就绪！\n\n"
            "【核心投研指令（见左侧 / 菜单栏）】\n"
            "• /pos — 瞬时拉取双账户实时净值、仓位、挂单与近7日成交（免等待 AI）\n"
            "• /report — 深度全景晨报（全市场分域联网检索、30日复盘、逐仓与挂单审计，≤2800字）\n"
            "• /pick [方向] — 全市场选股雷达（美/英/欧股 + LSE ETF 联网催化 + 风控终审，≤1000字）\n"
            "• /us — 美股盘前/盘中/收盘/休市自适应增量盯盘简报（平稳≤450字，异动≤600字）\n"
            "• /deep <问题> — 深度单聊升级版（1000~1200字、最多4轮工具的多维推演，Token 消耗更高）\n"
            "• /stats — 历史推荐 T+5/10/20 胜率与 Alpha、MFE/MAE 极值、近期教训与量化体检\n"
            "• /perf — 实盘绩效看板（TWR 剔入金收益、Sortino/最大回撤与修复天数、相对基准 Alpha-Beta-IR、\n"
            "  实盘胜率盈亏比期望、集中度 HHI、信号 t 值与衰减；0 Token 纯本地计算）\n"

            "• /wallet — Invest GBP+USD 双钱包对账（实时总现金、实时汇率、推算/实报 GBP 与校准差额）\n"
            "• /usd <金额> [挂单占用] — 上报 Invest USD 余额；/wallet gbp 1500 [usd 600] 单报或双报\n"
            "• /help — 显示本指令说明（/start 同义）；指令支持 /cmd@机器人名 形式，参数大小写不敏感\n"
            "• 直接打字 — 全天单聊（自带今日晨报基线、实时查价与谷歌搜新闻工具；/deep 前缀升级深度模式）\n\n"
            "【运维与系统控制指令（不在左侧菜单，直接输入即可）】\n"
            "• /refresh_pool — 手动触发多市场底池海选（0-Token 筛 150 只个股 + LSE ETF；附 offline 跳过联网指数同步）\n"
            "• /health — 全链路实时自检（T212/行情/FRED/Finnhub/Gemini/DB 与近期告警；/heath 为兼容别名）\n"
            "• /usage — 查看 Gemini Token 消耗、缓存命中率与本月 Google Search 免费额度进度\n"
            "• /clear — 清空今日对话记忆重新开始\n"
            "• /cancel — 立即中止正在运行的耗时分析"
        )

    # ---------- Invest GBP+USD wallet ledger (/wallet & /usd) ----------

    @staticmethod
    def _parse_wallet_args(head: str, args: list[str]) -> tuple[str, dict[str, tuple[float, float]]]:
        """Parse /usd and /wallet payloads into (error_message, {currency: (available, reserved)})."""
        if not args:
            return "", {}
        usage = "用法：/usd <可用金额> [挂单占用]；/wallet；/wallet gbp 1500；/wallet gbp 1500 usd 600"
        if head == "/usd":
            nums = _wallet_numbers(args)
            if not nums:
                return usage, {}
            avail = nums[0]
            reserved = nums[1] if len(nums) > 1 else 0.0
            if avail < 0 or reserved < 0:
                return "金额不能为负数。", {}
            return "", {"USD": (avail, reserved)}

        tokens = [a.lower() for a in args]
        if not any(t in {"gbp", "usd"} for t in tokens):
            return usage, {}
        updates: dict[str, tuple[float, float]] = {}
        for idx, tok in enumerate(tokens):
            ccy = tok.upper()
            if ccy not in {"GBP", "USD"}:
                continue
            if idx + 1 >= len(tokens):
                return f"{ccy} 缺少金额，例如 /wallet gbp 1500", {}
            nums = _wallet_numbers([args[idx + 1]])
            if not nums:
                return f"{ccy} 金额格式不正确：{args[idx + 1]}", {}
            if nums[0] < 0:
                return "金额不能为负数。", {}
            updates[ccy] = (nums[0], 0.0)
        if not updates:
            return usage, {}
        return "", updates

    async def _load_wallet_snapshot(self) -> tuple[Any, dict[str, dict[str, Any]], float | None]:
        """Load live Invest cash, local wallet anchors, and GBP/USD for ledger commands."""
        portfolio = await self.t212.get_portfolio(use_cache=True)
        records: dict[str, dict[str, Any]] = {}
        if hasattr(self.db, "get_wallet_balances"):
            try:
                records = await self.db.get_wallet_balances("Invest")
            except Exception as exc:
                logger.warning("Failed loading wallet balances: %s", exc)
        rate: float | None = None
        if hasattr(self.market_data, "get_macro_snapshot"):
            try:
                macro = await self.market_data.get_macro_snapshot()
                candidate = getattr(macro, "gbp_usd", None)
                if candidate is not None and float(candidate) > 0:
                    rate = float(candidate)
                    set_rate = getattr(self.t212, "set_gbp_usd_rate", None)
                    if callable(set_rate):
                        set_rate(rate)
            except Exception as exc:
                logger.warning("Failed fetching macro for wallet ledger: %s", exc)
        if rate is None:
            stored = getattr(self.t212, "gbp_usd_rate", None)
            if stored is not None and float(stored) > 0:
                rate = float(stored)
        return portfolio, records, rate

    def _format_wallet_status(
        self,
        portfolio: Any,
        records: dict[str, dict[str, Any]],
        rate: float | None,
        *,
        note: str = "",
    ) -> str:
        """Render the Invest GBP+USD ledger with Python-computed amounts and reconcile diff."""
        invest = portfolio.invest
        mismatch_floor = float(self.cfg.strategy.wallet_mismatch_warn_gbp)
        snap = reconcile_invest_wallets(
            invest,
            records,
            rate,
            mismatch_warn_gbp=mismatch_floor,
            reconcile_warn_days=int(self.cfg.strategy.wallet_reconcile_warn_days),
        )
        usd_rec = records.get("USD") or {}
        updated_ms = int(usd_rec.get("updated_at_ms") or 0)
        age_txt = ""
        if updated_ms > 0:
            age_days = max(0, int((time.time() * 1000 - updated_ms) / 86_400_000))
            age_txt = f"（{age_days} 天前）"

        lines = [f"💼 Invest 双钱包对账（{portfolio.fetched_at_uk} UK）"]
        lines.append(
            f"• T212 总现金(GBP): £{invest.total_cash:,.2f}"
            f"（可用 £{invest.available_to_trade:,.2f} / 挂单 £{invest.reserved_cash:,.2f}）"
        )
        lines.append(f"• 实时 GBP/USD: {rate:.4f}" if rate else "• 实时 GBP/USD: 暂缺（宏观快照未就绪）")
        usd_line = (
            f"• USD 钱包: ${snap.usd_cash_total:,.2f}"
            f"（可用 ${snap.usd_available_to_trade:,.2f} / 挂单占用 ${snap.usd_reserved_cash:,.2f}）"
        )
        if rate:
            usd_line += f" ≈ £{snap.usd_cash_total / rate:,.2f}"
        lines.append(usd_line)
        if snap.wallet_updated_at_uk:
            lines.append(f"• USD 记录更新: {snap.wallet_updated_at_uk} UK{age_txt}")
        if snap.gbp_wallet_total is None:
            lines.append("• GBP 钱包: 待汇率折算")
        else:
            src_lbl = "手动上报" if snap.wallet_source == "manual_both" else "按 API 总现金推算"
            gbp_reserved_txt = (
                f"，挂单占用 £{snap.gbp_wallet_reserved:,.2f}"
                if (snap.gbp_wallet_reserved or 0.0) > 0
                else ""
            )
            lines.append(
                f"• GBP 钱包: ≈ £{snap.gbp_wallet_total:,.2f}"
                f"（可用 £{snap.gbp_wallet_available or 0.0:,.2f}{gbp_reserved_txt}，{src_lbl}）"
            )
            threshold = max(mismatch_floor, abs(float(invest.total_cash)) * 0.02)
            diff = (snap.gbp_wallet_total + snap.usd_cash_total / rate) - float(invest.total_cash)
            lines.append(f"• 核对差额: £{diff:+,.2f}（告警阈值 £{threshold:,.2f}）")
        if snap.wallet_reconcile_warning:
            lines.append(f"⚠️ 校准提示: {snap.wallet_reconcile_warning}")
        if note:
            lines.append(note)
        lines.append(
            "提示：如有换汇或美股买卖回笼，回复 /usd 金额 校准；"
            "也可 /wallet gbp 1500 或 /wallet gbp 1500 usd 600 单报/双报。"
        )
        return "\n".join(lines)

    async def _handle_wallet_command(self, text: str) -> str:
        """Handle /wallet and /usd as lightweight, lock-free ledger commands."""
        parts = text.split()
        head = parts[0].lower().split("@", 1)[0]
        args = parts[1:]
        err, updates = self._parse_wallet_args(head, args)
        if err:
            return f"⚠️ {err}"

        portfolio, records, rate = await self._load_wallet_snapshot()
        invest = getattr(portfolio, "invest", None)
        if invest is None or not getattr(invest, "ok", False):
            return f"⚠️ Invest 账户读取失败，暂无法对账：{getattr(invest, 'error', '未知错误')}"

        note = ""
        if updates:
            total_cash = float(invest.total_cash or 0.0)
            threshold = max(float(self.cfg.strategy.wallet_mismatch_warn_gbp), abs(total_cash) * 0.02)
            usd_report = updates.get("USD")
            usd_amount = (usd_report[0] + usd_report[1]) if usd_report is not None else None
            if usd_amount is not None and rate:
                usd_gbp = float(usd_amount) / rate
                if usd_gbp > total_cash + threshold:
                    return (
                        f"⛔ 已阻断：USD ${usd_amount:,.2f} ÷ {rate:.4f} = £{usd_gbp:,.2f}，"
                        f"超出 Invest 总现金 £{total_cash:,.2f}（容差 £{threshold:,.2f}）。\n"
                        "请检查金额输入，或先 /pos 确认最新持仓快照后再报。"
                    )
            if usd_amount is not None and not rate:
                note = "（汇率暂缺：已先记录 USD 余额，GBP 拆解待汇率恢复后自动补算）"

            if set(updates) == {"USD"} and hasattr(self.db, "clear_manual_gbp_override"):
                # USD-only report restores GBP = TOTAL - USD auto derivation
                await self.db.clear_manual_gbp_override("Invest")

            if set(updates) == {"GBP"}:
                gbp_avail = updates["GBP"][0]
                if rate:
                    implied_usd = max(0.0, (total_cash - gbp_avail) * rate)
                    await self.db.upsert_wallet_balance(
                        "Invest", "USD", implied_usd, 0.0, source="manual_derived", note="由 GBP 反推"
                    )
                    await self.db.upsert_wallet_balance(
                        "Invest", "GBP", gbp_avail, 0.0, source="manual", note="Telegram 单报 GBP"
                    )
                    note = f"（已按 GBP £{gbp_avail:,.2f} 反推 USD ≈ ${implied_usd:,.2f}）"
                else:
                    await self.db.upsert_wallet_balance(
                        "Invest", "GBP", gbp_avail, 0.0, source="manual", note="Telegram 单报 GBP"
                    )
                    note = "（汇率暂缺：仅记录 GBP，USD 待下次双报或 /usd 校准）"
            else:
                # USD first, then GBP, so a dual report carries the newest GBP timestamp
                for ccy in ("USD", "GBP"):
                    if ccy not in updates:
                        continue
                    avail, reserved = updates[ccy]
                    await self.db.upsert_wallet_balance(
                        "Invest", ccy, avail, reserved, source="manual", note="Telegram 上报"
                    )
            records = await self.db.get_wallet_balances("Invest")

        return self._format_wallet_status(portfolio, records, rate, note=note)

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
            await self._send_reply(chat_id, await self._build_stats_message())
            return

        if lower.startswith("/perf"):
            await self._send_reply(chat_id, await self._build_perf_message())
            return

        if lower.startswith("/wallet") or lower.startswith("/usd"):
            await self._send_reply(chat_id, await self._handle_wallet_command(text))
            return

        if lower.startswith("/usage"):
            summary, reliability, quality = await asyncio.gather(
                self.db.get_usage_summary(days=30),
                self._safe_reliability_summary(),
                self._safe_ops_quality_summary(),
            )
            await self._send_reply(
                chat_id,
                self._format_usage_message(summary, reliability=reliability, quality=quality),
            )
            return

        # Cancel any still-running heavy AI command when the user issues another heavy query;
        # light commands (/pos /health) must stay lock-free and never kill a running report.
        if (
            self._active_task
            and not self._active_task.done()
            and self._is_heavy_ai_command(lower)
        ):
            await self.cancel_active_task(chat_id)

        self._active_task = asyncio.create_task(self._run_query_task(chat_id, text))
        if wait:
            await self._active_task

    @staticmethod
    def _is_heavy_ai_command(lower: str) -> bool:
        return not (
            lower.startswith("/pos")
            or lower.startswith("/health")
            or lower.startswith("/heath")
            or lower.startswith("/wallet")
            or lower.startswith("/usd")
        )

    @staticmethod
    def _compute_live_move_pct(symbol: str, entry_price: float, cur_price: float) -> tuple[float | None, bool]:
        """Compute native-currency return % with a 100x GBp/GBP unit mismatch guard for .L tickers."""
        if entry_price <= 0 or cur_price <= 0:
            return None, False
        ratio = cur_price / entry_price
        if symbol.strip().upper().endswith(".L") and (0.005 <= ratio <= 0.05 or 20.0 <= ratio <= 200.0):
            return None, True
        return (ratio - 1.0) * 100.0, False

    async def _build_perf_message(self) -> str:
        """0-Token live performance board: TWR curve, trade attribution and signal quality."""
        nav_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_nav_history"):
            try:
                nav_rows = await self.db.get_nav_history(limit=420)
            except Exception as exc:
                logger.warning("get_nav_history failed: %s", exc)

        bench_symbol = str(getattr(self.cfg.strategy, "benchmark_symbol", "VWRP.L") or "VWRP.L")
        portfolio = None
        if self.t212 is not None:
            try:
                portfolio = await self.t212.get_portfolio(use_cache=True)
            except Exception as exc:
                logger.warning("/perf portfolio fetch failed: %s", exc)

        bench_hist = None
        if nav_rows:
            try:
                bench_map = await self.market_data.get_histories([bench_symbol], period="1y")
                bench_hist = bench_map.get(bench_symbol)
            except Exception as exc:
                logger.warning("/perf benchmark fetch failed: %s", exc)

        rf_pct: float | None = None
        if portfolio is not None:
            rf_pct = float(getattr(portfolio, "cash_interest_rate_pct", 0.0) or 0.0)
        elif nav_rows and nav_rows[-1].get("cash_rate_pct") is not None:
            rf_pct = float(nav_rows[-1]["cash_rate_pct"])

        perf = compute_portfolio_performance(
            nav_rows, benchmark_hist=bench_hist, risk_free_annual_pct=rf_pct
        )

        lines = [f"📈 实盘绩效看板（基准 {bench_symbol}，GBP 口径）"]
        if not perf.get("ok"):
            reason = perf.get("reason") or "快照样本不足"
            lines.append(f"• 净值序列: {reason}")
            lines.append("• 说明: 每日 /report 盘前自动写入 1 条 NAV，累计 ≥2 天后开始出数。")
        else:
            lines.append(
                f"• 区间: {perf['first_date']} → {perf['last_date']} "
                f"（{perf['samples']} 快照 / 覆盖 {perf['coverage_pct']:.0f}%）"
            )
            lines.append(f"• TWR 累计收益（剔入金）: {perf['total_return_pct']:+.2f}%")
            risk_bits = []
            if perf.get("ann_vol_pct") is not None:
                risk_bits.append(f"年化波动 {perf['ann_vol_pct']:.1f}%")
            if perf.get("sortino") is not None:
                risk_bits.append(f"Sortino {perf['sortino']:.2f}（rf {perf['risk_free_annual_pct']:.1f}%）")
            if risk_bits:
                lines.append("• " + " | ".join(risk_bits))
            dd = (
                f"最大回撤 {perf['max_drawdown_pct']:.2f}%"
                f"（{perf['drawdown_peak_date']}→{perf['drawdown_trough_date']}）"
            )
            if perf.get("drawdown_recovery_days") is not None:
                dd += f"，{perf['drawdown_recovery_days']} 天修复"
            elif perf.get("drawdown_unrecovered"):
                dd += "，尚未修复"
            lines.append(f"• {dd}")
            if perf.get("cagr_pct") is not None:
                tail = f"，Calmar {perf['calmar']:.2f}" if perf.get("calmar") is not None else ""
                lines.append(f"• 年化收益 CAGR {perf['cagr_pct']:+.2f}%{tail}")
            else:
                lines.append("• 年化收益 CAGR: 样本 <3 个月，暂不出年化（避免误导）")
            if perf.get("bench_beta") is not None:
                ir_val = perf["bench_info_ratio"] if perf.get("bench_info_ratio") is not None else 0.0
                lines.append(
                    f"• 相对基准: Beta {perf['bench_beta']:.2f} | 年化 Alpha {perf['bench_alpha_annual_pct']:+.2f}% "
                    f"| 信息比率 {ir_val:.2f}"
                )
            else:
                lines.append("• 相对基准: 基准序列不足，暂无法回归")
            if perf.get("cash_drag_pct") is not None:
                lines.append(f"• 现金拖累: 平均现金占比 {perf['cash_drag_pct']:.1f}%")

        if nav_rows:
            split = compute_account_split_performance(
                nav_rows, benchmark_hist=bench_hist, risk_free_annual_pct=rf_pct
            )

            def _split_seg(label: str, item: dict[str, Any] | None) -> str:
                if not item or not item.get("ok"):
                    return f"{label} 样本不足"
                return (
                    f"{label} TWR {item['total_return_pct']:+.2f}% | 最大回撤 {item['max_drawdown_pct']:.2f}%"
                    f"（{item['samples']} 快照）"
                )

            lines.append(
                "• 分账户归因: "
                + _split_seg("ISA 价值", split.get("isa"))
                + " ／ "
                + _split_seg("Invest 右侧", split.get("invest"))
            )

            core_cmp = None
            try:
                core_hist_map = await self.market_data.get_histories(
                    list(_CORE_POLICY_BENCHMARK_WEIGHTS), period="1y"
                )
                core_cmp = compute_core_policy_benchmark_comparison(core_hist_map or {}, lookback_days=60)
            except Exception as exc:
                logger.warning("/perf core policy benchmark failed: %s", exc)
            if core_cmp:
                te = core_cmp.get("tracking_error_pct")
                te_s = f"{te:.1f}%" if te is not None else "N/A"
                lines.append(
                    f"• 四桶核心政策（{core_cmp['lookback_days']}日）: 组合 {core_cmp['policy_return_pct']:+.2f}% "
                    f"vs {core_cmp['benchmark_symbol']} {core_cmp['benchmark_return_pct']:+.2f}%"
                    f"（区域倾斜效应 {core_cmp['allocation_effect_pct']:+.2f}% | 跟踪误差 {te_s}）"
                )

        if portfolio is not None:
            attribution = compute_trade_attribution(portfolio)
            if attribution.get("ok"):
                ccy_bits = []
                for ccy, item in sorted(attribution["by_currency"].items()):
                    pf = f"{item['profit_factor']:.2f}" if item.get("profit_factor") is not None else "N/A"
                    ccy_bits.append(
                        f"{ccy} 胜率 {item['win_rate_pct']:.0f}%({item['wins']}/{item['sells']}) "
                        f"盈亏比 {pf} 期望 {item['expectancy']:+.2f}"
                    )
                lines.append("• 实盘交易: " + " | ".join(ccy_bits))
                if attribution.get("avg_holding_days") is not None:
                    lines.append(f"• 平均持仓: {attribution['avg_holding_days']:.1f} 天（近似，首笔买入起算）")
                divs = attribution.get("dividends_by_currency") or {}
                if divs:
                    div_s = " ".join(f"{c} {v:+.0f}" for c, v in sorted(divs.items()))
                    lines.append(f"• 累计派息: {div_s} | 汇率影响 £{attribution['fx_impact_gbp']:+.2f}")
            else:
                lines.append("• 实盘交易: 近 30 天无卖出成交，暂无胜率样本")

            concentration = compute_concentration_metrics(portfolio)
            if concentration.get("ok"):
                lines.append(
                    f"• 集中度: {concentration['positions']} 只 | HHI {concentration['hhi']:.3f} "
                    f"| 有效标的 {concentration['effective_n']:.1f} | Top1 {concentration['top1_pct']:.0f}% "
                    f"| Top3 {concentration['top3_pct']:.0f}%"
                )

        settled_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_settled_outcome_series"):
            try:
                settled_rows = await self.db.get_settled_outcome_series(limit=400)
            except Exception as exc:
                logger.warning("/perf settled series failed: %s", exc)
        if settled_rows:
            quality = compute_signal_quality(settled_rows)
            if quality.get("ok"):
                bits = []
                for key in ("t5", "t10", "t20"):
                    item = quality["horizons"][key]
                    if item.get("mean_pct") is None:
                        continue
                    t_s = f"t={item['t_stat']:.2f}" if item.get("t_stat") is not None else "t=N/A"
                    flag = "显著" if item["significant"] else "不显著"
                    bits.append(f"{key.upper()} {item['mean_pct']:+.2f}%({t_s},{flag})")
                lines.append("• 信号质检: " + " | ".join(bits))
                decay = quality.get("decay_t20_minus_t5_pct")
                decay_s = f"，T20−T5 衰减 {decay:+.2f}pct" if decay is not None else ""
                sample_note = "n<30 仅作参考，不构成统计证据" if quality["sample"] < 30 else "样本充足"
                lines.append(f"• 样本 n={quality['sample']}（{sample_note}）{decay_s}")

        return "\n".join(lines)

    async def _build_stats_message(self) -> str:
        async def _empty_audit() -> dict[str, Any] | None:
            return None

        audit_coro = (
            self.db.get_latest_quant_audit()
            if hasattr(self.db, "get_latest_quant_audit")
            else _empty_audit()
        )
        stats, lessons, rows, core_rows, latest_audit = await asyncio.gather(
            self.db.get_recommendation_stats(),
            self.db.get_recent_lessons(limit=3),
            self.db.get_recent_recommendations(limit=10, exclude_core_dca=True),
            self.db.get_recent_recommendations(limit=3, only_core_dca=True),
            audit_coro,
        )
        if (
            not rows
            and not core_rows
            and stats.get("total_count", 0) == 0
            and stats.get("core_dca_count", 0) == 0
            and not latest_audit
        ):
            return "📭 暂无历史选股或调仓记录（运行一次 /pick 后会自动建档）。"

        symbols = list(
            dict.fromkeys(
                str(r["symbol"]).strip()
                for r in (*rows, *core_rows)
                if r.get("symbol") and str(r["symbol"]).strip().upper() != "CASH"
            )
        )
        live_prices: dict[str, float] = {}
        if symbols:
            try:
                histories = await self.market_data.get_histories(symbols, period="1y")
            except Exception as exc:
                logger.warning("Failed fetching live prices for /stats: %s", exc)
                histories = {}
            for sym, df in histories.items():
                if df is not None and not df.empty and "Close" in df.columns:
                    raw_close = float(df["Close"].iloc[-1])
                    if math.isfinite(raw_close) and raw_close > 0:
                        live_prices[sym] = raw_close

        core_dca_cnt = int(stats.get("core_dca_count") or 0)
        core_suffix = f" | 核心定投归档 {core_dca_cnt} 条" if core_dca_cnt > 0 else ""
        lines = [
            "📊 历史选股后验胜率与 Alpha 看板",
            (
                f"• 样本总览: 共 {stats['total_count']} 条 "
                f"(已结算 {stats['settled_count']} | 待结算 {stats['pending_count']} | "
                f"未成交 {stats.get('not_filled_count', 0)} | 跳过 {stats['skipped_count']}){core_suffix}"
            ),
            "• 口径: BUY/ADD 为多头收益；TRIM/STOP/SWAP 为避损收益（价格跌=正确）",
        ]
        if stats["settled_count"] > 0:
            avg_raw = f"{stats['avg_raw_return_pct']:+.2f}%" if stats["avg_raw_return_pct"] is not None else "N/A"
            avg_alpha = f"{stats['avg_alpha_pct']:+.2f}%" if stats["avg_alpha_pct"] is not None else "N/A"
            t10_alpha = (
                f"{stats['avg_t10_alpha_pct']:+.2f}%"
                if stats.get("avg_t10_alpha_pct") is not None
                else "N/A"
            )
            t20_alpha = (
                f"{stats['avg_t20_alpha_pct']:+.2f}%"
                if stats.get("avg_t20_alpha_pct") is not None
                else "N/A"
            )
            mfe_s = f"{stats['avg_mfe_pct']:+.1f}%" if stats.get("avg_mfe_pct") is not None else "N/A"
            mae_s = f"{stats['avg_mae_pct']:+.1f}%" if stats.get("avg_mae_pct") is not None else "N/A"
            lines.append(
                f"• 原始胜率: {stats['win_rate_pct']:.1f}% ({stats['win_count']}/{stats['settled_count']}) | 平均收益: {avg_raw}"
            )
            lines.append(
                f"• 超额 Alpha 胜率: {stats['alpha_win_rate_pct']:.1f}% ({stats['alpha_win_count']}/{stats['settled_count']}) | 平均 Alpha: {avg_alpha}"
            )
            lines.append(
                f"• 多窗口 Alpha 轨迹: T+5 平均 {avg_alpha} | T+10 平均 {t10_alpha} | T+20 平均 {t20_alpha}"
            )
            lines.append(
                f"• 持仓期极值分布: 平均最大浮盈(MFE) {mfe_s} | 平均最大回撤(MAE) {mae_s}"
            )
        else:
            lines.append("• 胜率与 Alpha: 暂无已结算 (T+N) 样本，待盘后结算后自动生成。")

        split_accounts = stats.get("by_account") or {}
        if stats["settled_count"] > 0 and split_accounts:
            isa_a = split_accounts.get("ISA") or {}
            inv_a = split_accounts.get("Invest") or {}

            def _alpha_s(val: Any) -> str:
                return f"{float(val):+.2f}%" if val is not None else "N/A"

            lines.append(
                "• 分账户表现: "
                f"ISA 价值 {isa_a.get('settled_count', 0)} 笔 | 胜率 {isa_a.get('win_rate_pct', 0.0):.0f}% | "
                f"T+20 Alpha {_alpha_s(isa_a.get('avg_t20_alpha_pct'))} ／ "
                f"Invest 右侧 {inv_a.get('settled_count', 0)} 笔 | 胜率 {inv_a.get('win_rate_pct', 0.0):.0f}% | "
                f"T+5 Alpha {_alpha_s(inv_a.get('avg_alpha_pct'))} | "
                f"T+10 Alpha {_alpha_s(inv_a.get('avg_t10_alpha_pct'))}"
            )
        adoption = stats.get("adoption_stats") or {}
        if adoption.get("checked_count", 0) > 0:
            slip = adoption.get("avg_slippage_pct")
            slip_s = f"{float(slip):+.2f}%" if slip is not None else "N/A"
            lines.append(
                f"• 实盘采纳: {adoption.get('filled_count', 0)}/{adoption.get('checked_count', 0)} 条命中真实成交 "
                f"| 采纳率 {float(adoption.get('adoption_rate_pct') or 0.0):.1f}% | 平均执行滑点 {slip_s}"
            )

        if lessons:
            lines.append("\n【近期复盘教训】")
            for ls in lessons:
                raw_s = f"{ls['raw_return_pct']:+.1f}%" if ls.get("raw_return_pct") is not None else "N/A"
                alpha_s = f"{ls['alpha_pct']:+.1f}%" if ls.get("alpha_pct") is not None else "N/A"
                bench_s = ls.get("benchmark_symbol") or "基准"
                ts_s = ls.get("settled_at_uk") or ls.get("created_at_uk")
                lines.append(
                    f"• [{ts_s}] {ls['symbol']} ({ls['action']}, 收益 {raw_s} vs {bench_s} Alpha {alpha_s}): "
                    f"{str(ls['reflection_lesson'])[:120]}"
                )

        def _format_rec_line(r: dict[str, Any]) -> str:
            sym = str(r["symbol"])
            entry = r.get("entry_price")
            price_s = f" @ {entry}" if entry is not None else ""
            status = str(r.get("outcome_status") or "PENDING").upper()
            cur_p = live_prices.get(sym)
            if status == "SETTLED":
                raw_s = f"{r['raw_return_pct']:+.1f}%" if r.get("raw_return_pct") is not None else "N/A"
                alpha_s = f"{r['alpha_pct']:+.1f}%" if r.get("alpha_pct") is not None else "N/A"
                cur_s = f" | 现价 {cur_p:.2f}" if cur_p is not None else ""
                perf_s = f" [已结算: 收益 {raw_s}, Alpha {alpha_s}{cur_s}]"
            elif status == "NOT_FILLED":
                perf_s = " [未触价·未成交，已排除出胜率]"
            elif status == "SKIPPED":
                perf_s = " [已跳过结算]"
            elif cur_p is not None and entry is not None and float(entry) > 0:
                move_pct, unit_mismatch = self._compute_live_move_pct(sym, float(entry), cur_p)
                if unit_mismatch:
                    perf_s = f" [待结算 | 现价 {cur_p:.2f} (GBp/GBP量级差异)]"
                elif move_pct is not None:
                    perf_s = f" [待结算 | 现价 {cur_p:.2f} ({move_pct:+.1f}%)]"
                else:
                    perf_s = f" [待结算 | 现价 {cur_p:.2f}]"
            elif cur_p is not None:
                perf_s = f" [待结算 | 现价 {cur_p:.2f}]"
            else:
                perf_s = " [待结算]"
            src_key = str(r.get("source") or "pick").strip().lower()
            src_tag = {"report": "早报", "pick": "选股", "core_dca": "定投"}.get(src_key, src_key or "选股")
            return f"• [{r['created_at_uk']}] [{src_tag}] [{r['account']}] {r['action']} {sym}{price_s}{perf_s}：{r['thesis']}"

        if rows:
            lines.append("\n【最近 10 条 /pick 与加减仓表现】")
            for r in rows:
                lines.append(_format_rec_line(r))

        if core_rows:
            lines.append("\n【近期核心宽基定投归档（不计入战术胜率）】")
            for r in core_rows:
                lines.append(_format_rec_line(r))

        if latest_audit:
            audit_ts = str(latest_audit.get("created_at_uk") or latest_audit.get("audit_date") or "").strip()
            audit_header = f"\n【周末候选池与阈值体检】({audit_ts})" if audit_ts else "\n【周末候选池与阈值体检】"
            lines.append(audit_header)
            summary_txt = str(latest_audit.get("summary_text") or "").strip()
            if summary_txt:
                lines.append(summary_txt)
            diagnosis = str(latest_audit.get("ai_diagnosis") or "").strip()
            if diagnosis:
                lines.append("【AI 策略诊断】")
                lines.append(diagnosis)

        return "\n".join(lines)

    async def _run_health_check(self) -> str:
        portfolio, macro, ops_events, unnotified_events, usage_30d = await asyncio.gather(
            self.t212.get_portfolio(use_cache=True),
            self.market_data.get_macro_snapshot(),
            self.db.get_recent_ops_events(limit=5),
            self.db.get_unnotified_ops_events(hours=24, limit=10),
            self.db.get_usage_summary(days=30),
        )
        uptime_s = _format_uptime(time.monotonic() - self._started_monotonic)
        # Keep the T212 client's GBP/USD anchor warm for later wallet-split derivation.
        if getattr(macro, "gbp_usd", None):
            set_rate = getattr(self.t212, "set_gbp_usd_rate", None)
            if callable(set_rate):
                set_rate(macro.gbp_usd)

        def _acc_line(label: str, acc: Any) -> str:
            if not acc.ok:
                return f"⚠️ {label}: 失败 ({redact_secrets(str(acc.error))[:80]})"
            ord_s = "挂单✓" if acc.pending_orders_supported else "挂单403/关"
            hist_s = "历史✓" if acc.order_history_supported else "历史403/关"
            if getattr(acc, "is_fallback", False):
                fb_src = getattr(acc, "fallback_source", "") or "cache"
                fb_at = getattr(acc, "snapshot_fetched_at_uk", "")
                fb_reason = getattr(acc, "fallback_reason", "")
                meta_parts = [f"来源:{fb_src}"]
                if fb_at:
                    meta_parts.append(f"快照:{fb_at}")
                if fb_reason:
                    meta_parts.append(f"原因:{str(fb_reason)[:40]}")
                return (
                    f"⚠️ {label} [降级缓存]: 净值 {acc.account_currency} {acc.total_account_value:,.2f} "
                    f"({ord_s}, {hist_s}; {', '.join(meta_parts)})"
                )
            return f"✅ {label}: 净值 {acc.account_currency} {acc.total_account_value:,.2f} ({ord_s}, {hist_s})"

        yf_ok = macro.gbp_usd is not None and macro.spy_5d_pct is not None
        macro_stale = bool(getattr(macro, "is_stale", False))
        if macro_stale:
            yf_status = "⚠️ 降级/陈旧缓存 (is_stale)"
        elif not yf_ok:
            yf_status = "⚠️ 数据残缺"
        else:
            yf_status = "✅ 正常"

        finnhub_status = (
            f"⚠️ 异常 ({redact_secrets(str(self.market_data.finnhub.last_error))})"
            if self.market_data.finnhub.last_error
            else ("✅ 已配置正常" if self.market_data.finnhub.enabled else "➖ 未配置")
        )
        fred_status = (
            f"⚠️ 异常 ({redact_secrets(str(self.market_data.fred.last_error))})"
            if self.market_data.fred.last_error
            else ("✅ 已配置正常" if self.market_data.fred.enabled else "➖ 未配置")
        )
        gemini_status = (
            f"✅ Key 已配置 | Bear挑刺 {self.cfg.models.report_auditor_model} | "
            f"PM拍板 {self.cfg.models.report_model} | 备援 {self.cfg.models.fallback_model} | "
            f"早报搜索 {self.cfg.models.report_search_model} | 日常搜索 {self.cfg.models.search_model}"
            if self.cfg.env.gemini_api_key.strip()
            else "🚨 GEMINI_API_KEY 缺失"
        )
        ops_enabled = bool(self.ops is not None and getattr(self.ops, "ops_notify_enabled", False))
        ops_gateway_status = "✅ 已启用 ops-notify（双态分流）" if ops_enabled else "➖ 未配置（降级走本 Bot）"
        unnotified_cnt = len(unnotified_events)
        suppressed_total = sum(int(ev.get("suppressed_count") or 0) for ev in unnotified_events)
        if unnotified_cnt > 0:
            alert_health = f"⚠️ {unnotified_cnt} 条未送达/静默抑制(合并{suppressed_total}次)"
        else:
            alert_health = "✅ 正常"

        month_bucket = usage_30d.get("this_month") or usage_30d["last_nd"]
        search_q_month = month_bucket.get("search_queries", 0)

        lines = [
            f"🩺 系统全链路运维体检（v{APP_VERSION}）",
            f"• 启动时间: {self._started_uk} UK | 连续运行: {uptime_s}",
            f"• SQLite & 告警: {alert_health} | 运维网关: {ops_gateway_status}",
            f"• 强制联网扫描: {'开启' if self.cfg.models.enable_forced_search_grounding else '关闭'}",
            f"• Gemini: {gemini_status}",
            f"• Google Search 本月用量: {search_q_month:,} / {self.cfg.models.google_search_free_quota_monthly:,} 免费额度",
            "",
            "【外部接口连通性】",
            f"• {_acc_line('T212 ISA', portfolio.isa)}",
            f"• {_acc_line('T212 Invest', portfolio.invest)}",
        ]
        if getattr(portfolio, "is_fallback", False):
            fb_note = getattr(portfolio, "fallback_note", "") or getattr(portfolio, "fallback_source", "") or "已启用兜底快照"
            lines.append(f"• ⚠️ T212 组合降级告警: 当前非实时实盘数据（{str(fb_note)[:80]}）")
        lines.extend(
            [
                f"• Yahoo 行情/宏观: {yf_status} (GBP/USD={macro.gbp_usd or 'N/A'}, VIX={macro.vix or 'N/A'})",
                f"• Finnhub 美股新闻/财历: {finnhub_status}",
                f"• FRED 利率/失业率: {fred_status}",
            ]
        )

        if self.scheduler is not None and self.scheduler.last_job_runs:
            lines.append("\n【最近定时任务记录】")
            for job_name, info in self.scheduler.last_job_runs.items():
                detail = f" ({info['detail']})" if info.get("detail") else ""
                lines.append(f"• {job_name}: {info['status']} @ {info['uk_time']} UK{detail}")

        if unnotified_events:
            lines.append(f"\n【近 24h 未送达/静默抑制告警 ({unnotified_cnt} 条)】")
            for ev in unnotified_events[:5]:
                sup = f", 抑制合并{ev['suppressed_count']}次" if ev["suppressed_count"] > 0 else ""
                lines.append(
                    f"• [{ev['created_at_uk']}] [{ev['severity']}] {ev['source']}: {ev['message'][:80]} (待补推{sup})"
                )

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

        # v5.5.3 observability one-liner: read-only, best-effort, never allowed to break /health.
        try:
            rel = await self._safe_reliability_summary()
            qual = await self._safe_ops_quality_summary()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Health observability summary skipped: %s", exc)
            rel = qual = None
        health_bits: list[str] = []
        if rel:
            total_rel = rel.get("total") or {}
            rate = total_rel.get("success_rate_pct")
            rate_s = f"{float(rate):.1f}%" if rate is not None else "N/A"
            health_bits.append(f"近7天 LLM 成功率 {rate_s}（{int(total_rel.get('calls') or 0)} 次）")
        if qual:
            sem = qual.get("semantic_audit") or {}
            health_bits.append(
                f"语义质检 {int(sem.get('total') or 0)} 次 | 降级 {int(qual.get('degraded_alerts') or 0)} 次"
            )
        if health_bits:
            lines.append("• 近7天质量: " + " | ".join(health_bits))

        return "\n".join(lines)

    async def _run_query_task(self, chat_id: str, text: str) -> None:
        status_msg_id: int | None = None
        try:
            lower = text.lower()
            is_heavy = self._is_heavy_ai_command(lower)
            async with AsyncExitStack() as stack:
                if is_heavy:
                    if self.execution_lock.locked():
                        active_job = getattr(self.scheduler, "current_job_name", None)
                        job_hint = f"（{active_job}）" if active_job else ""
                        await self._deliver(
                            chat_id,
                            f"⏳ 后台定时任务{job_hint}或另一项 AI 分析正在执行中，为避免重复消耗额度与并发冲突，请稍候完成后再试。",
                        )
                        return
                    await stack.enter_async_context(self.execution_lock)

                await self.tg.send_typing(chat_id)

                if lower.startswith("/pos"):
                    status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在拉取 T212 双账户实时仓位与挂单...")
                    self._active_status_msg_id = status_msg_id
                    reply = await format_positions_instant(self.cfg, self.t212)
                    await self.tg.edit_status(chat_id, status_msg_id, "✅ 持仓快照已就绪：")
                    await self._deliver(chat_id, reply)
                    return

                if lower.startswith("/health") or lower.startswith("/heath"):
                    status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在巡检 T212 / 行情 / Gemini / DB 全链路状态...")
                    self._active_status_msg_id = status_msg_id
                    reply = await self._run_health_check()
                    await self.tg.edit_status(chat_id, status_msg_id, "✅ 运维自检报告已生成：")
                    await self._deliver(chat_id, reply)
                    return

                if lower.startswith("/report"):
                    status_msg_id = await self.tg.send_status(
                        chat_id, "⏳ 正在并发拉取 T212、行情与英/欧/美全市场 Google 深度检索并生成晨报..."
                    )
                    self._active_status_msg_id = status_msg_id
                    reply = await generate_morning_report(
                        self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                    )
                    await self.tg.edit_status(
                        chat_id,
                        status_msg_id,
                        "⚠️ 晨报未完成（未写入当日基线）" if str(reply).lstrip().startswith("⚠️") else "✅ 全景晨报分析完成：",
                    )
                    await self._deliver(chat_id, reply)
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
                    await self._deliver(chat_id, reply)
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
                    blocked = str(reply).lstrip().startswith(("⚠️", "⛔"))
                    await self.tg.edit_status(
                        chat_id,
                        status_msg_id,
                        "⛔ 选股未执行/已被门控拦截" if blocked else "✅ 投研选股 + 风控官双审完成：",
                    )
                    await self._deliver(chat_id, reply)
                    return

                if lower.startswith("/refresh_pool"):
                    sync_online = "offline" not in lower
                    status_msg_id = await self.tg.send_status(
                        chat_id,
                        "⏳ 正在执行多市场底池海选（T212 可买性交集 + 1y 流动性初筛 + 缓存基本面打分，约 1~4 分钟）...",
                    )
                    self._active_status_msg_id = status_msg_id
                    if self.scheduler is not None and hasattr(self.scheduler, "run_pool_audition"):
                        res = await self.scheduler.run_pool_audition(
                            sync_online_indices=sync_online,
                            sync_t212_metadata=True,
                            acquire_lock=False,
                        )
                    else:
                        res = await run_universe_audition(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.db,
                            sync_online_indices=sync_online,
                            sync_t212_metadata=True,
                        )
                    await self.tg.edit_status(chat_id, status_msg_id, "✅ 选股底池海选更新完成：")
                    await self._deliver(chat_id, self._format_audition_reply(res))
                    return

                # Free-form chat
                status_msg_id = await self.tg.send_status(chat_id, "⏳ 正在核对实盘数据并思考回复...")
                self._active_status_msg_id = status_msg_id
                reply = await answer_user_chat(
                    text, self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                )
                await self.tg.edit_status(chat_id, status_msg_id, "✅ 回复如下：")
                await self._deliver(chat_id, reply)
        except asyncio.CancelledError:
            logger.info("Task cancelled for query: %s", text[:40])
            raise
        except Exception as exc:
            logger.exception("Error processing message %s", text[:40])
            if status_msg_id is not None:
                await self.tg.edit_status(chat_id, status_msg_id, "❌ 处理失败")
            if self.ops is not None:
                await self.ops.notify(
                    severity="CRITICAL",
                    source="bot:query_task",
                    alert_key="bot_handler_crash",
                    message=f"处理指令「{text[:40]}」时发生异常：{type(exc).__name__}: {exc}",
                )
            await self._deliver(chat_id, f"⚠️ 执行出错：{exc}")
        finally:
            self._active_status_msg_id = None

    async def _safe_reliability_summary(self) -> dict[str, Any] | None:
        """Optional 7-day LLM reliability rollup; never let observability break /usage or /health."""
        getter = getattr(self.db, "get_llm_reliability_summary", None)
        if not callable(getter):
            return None
        try:
            return await getter(days=7)
        except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
            logger.warning("LLM reliability summary failed: %s", exc)
            return None

    async def _safe_ops_quality_summary(self) -> dict[str, Any] | None:
        getter = getattr(self.db, "get_ops_quality_summary", None)
        if not callable(getter):
            return None
        try:
            return await getter(days=7)
        except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
            logger.warning("Ops quality summary failed: %s", exc)
            return None

    def _format_usage_message(
        self,
        summary: dict[str, Any],
        reliability: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
    ) -> str:
        def _fmt_bucket(title: str, b: dict[str, Any]) -> str:
            in_tok = b["prompt_tokens"]
            cached_tok = b["cached_tokens"]
            out_tok = b["output_tokens"]
            thought_tok = b["thought_tokens"]
            search_q = b.get("search_queries", 0)
            cache_pct = (cached_tok / in_tok * 100.0) if in_tok > 0 else 0.0
            gen_tok = out_tok + thought_tok
            think_pct = (thought_tok / gen_tok * 100.0) if gen_tok > 0 else 0.0
            search_str = f" | 搜索 {search_q}次" if search_q else ""
            return (
                f"• {title}：{b['calls']}次 ({b['rounds']}轮){search_str}\n"
                f"  输入 {in_tok:,} (缓存 {cached_tok:,}, 命中 {cache_pct:.0f}%) | "
                f"输出 {out_tok:,} + 思考 {thought_tok:,} (思考占生成 {think_pct:.0f}%)"
            )

        m = self.cfg.models
        month_bucket = summary.get("this_month") or summary["last_nd"]
        search_month = month_bucket.get("search_queries", 0)
        free_quota = m.google_search_free_quota_monthly
        warn_at = int(free_quota * m.google_search_warn_pct / 100.0) if free_quota > 0 else 0
        quota_pct = (search_month / free_quota * 100.0) if free_quota > 0 else 0.0
        remaining_quota = max(free_quota - search_month, 0) if free_quota > 0 else 0
        search_badge = ""
        if free_quota > 0 and search_month >= free_quota:
            search_badge = " 🚨 已超免费额度（超额按量计费）"
        elif free_quota > 0 and search_month >= warn_at:
            search_badge = f" ⚠️ 接近 {m.google_search_warn_pct:.0f}% 预警线"

        lines = [
            f"📊 AI API Token 与搜索用量看板（周期 {summary['cycle_date']}）",
            _fmt_bucket("今日周期", summary["today"]),
            _fmt_bucket("近 7 天", summary["last_7d"]),
            _fmt_bucket("本月 (UK自然月)", month_bucket),
            _fmt_bucket(f"近 {summary['days']} 天", summary["last_nd"]),
            (
                f"🔍 本月 Google Search 查询: {search_month:,} / {free_quota:,} "
                f"({quota_pct:.1f}%, 剩余 {remaining_quota:,}){search_badge}"
            ),
            "ℹ️ 费用以 Google AI Studio 与 DeepSeek 官方账单为准，本看板仅统计纯 Token 与搜索配额。",
        ]
        if summary["by_label_7d"]:
            lines.append("\n【近 7 天分节点消耗】")
            for item in summary["by_label_7d"]:
                in_t = item["prompt_tokens"]
                cached_t = item["cached_tokens"]
                out_t = item["output_tokens"]
                thought_t = item["thought_tokens"]
                c_pct = (cached_t / in_t * 100.0) if in_t > 0 else 0.0
                gen_t = out_t + thought_t
                th_pct = (thought_t / gen_t * 100.0) if gen_t > 0 else 0.0
                lines.append(
                    f"• {item['label']}: {item['calls']}次 | 缓存命中 {c_pct:.0f}% | 思考占生成 {th_pct:.0f}% | "
                    f"In {in_t:,} (缓存 {cached_t:,}) / Out+Think {gen_t:,}"
                )
        if summary["by_model_nd"]:
            lines.append(f"\n【近 {summary['days']} 天分模型统计】")
            for item in summary["by_model_nd"]:
                in_t = item["prompt_tokens"]
                cached_t = item["cached_tokens"]
                out_t = item["output_tokens"]
                thought_t = item["thought_tokens"]
                tot_tok = in_t + out_t + thought_t
                c_pct = (cached_t / in_t * 100.0) if in_t > 0 else 0.0
                gen_t = out_t + thought_t
                th_pct = (thought_t / gen_t * 100.0) if gen_t > 0 else 0.0
                lines.append(
                    f"• {item['model']}: {item['calls']}次 ({item['rounds']}轮) | "
                    f"总Token {tot_tok:,} | 缓存命中 {c_pct:.0f}% | 思考占生成 {th_pct:.0f}%"
                )
        if summary["last_nd"]["calls"] == 0:
            lines.append("\nℹ️ 暂无调用记录（运行 /report、/us、/pick 或单聊后会自动累计）。")

        if reliability:
            total_rel = reliability.get("total") or {}
            calls = int(total_rel.get("calls") or 0)
            if calls <= 0:
                lines.append("\n【近 7 天模型可靠性】\n• 暂无调用样本")
            else:
                rate = total_rel.get("success_rate_pct")
                rate_s = f"{float(rate):.1f}%" if rate is not None else "N/A"
                p95 = total_rel.get("p95_latency_ms")
                p95_s = f"{float(p95) / 1000.0:.1f}s" if p95 is not None else "N/A"
                p50 = total_rel.get("p50_latency_ms")
                p50_s = f"{float(p50) / 1000.0:.1f}s" if p50 is not None else "N/A"
                lines.append("\n【近 7 天模型可靠性】")
                lines.append(
                    f"• 总体 {calls} 次 | 成功率 {rate_s}（失败 {int(total_rel.get('failed_calls') or 0)}）"
                    f" | P50 {p50_s} / P95 {p95_s}"
                )
                model_bits: list[str] = []
                for name, item in list((reliability.get("by_model") or {}).items())[:3]:
                    m_rate = item.get("success_rate_pct")
                    m_rate_s = f"{float(m_rate):.0f}%" if m_rate is not None else "N/A"
                    m_p95 = item.get("p95_latency_ms")
                    m_p95_s = f"{float(m_p95) / 1000.0:.1f}s" if m_p95 is not None else "N/A"
                    model_bits.append(f"{name} {m_rate_s}/P95 {m_p95_s}")
                if model_bits:
                    lines.append("• 分模型: " + " | ".join(model_bits))

        if quality:
            sem = quality.get("semantic_audit") or {}
            tags = quality.get("violation_tags") or {}
            sem_total = int(sem.get("total") or 0)
            degraded = int(quality.get("degraded_alerts") or 0)
            if sem_total <= 0 and degraded <= 0:
                lines.append("\n【近 7 天质检与降级】\n• 语义质检 0 次，模型/辩论降级 0 次")
            else:
                tag_s = "、".join(f"{k} {v}" for k, v in tags.items()) or "无明确违规范畴"
                lines.append("\n【近 7 天质检与降级】")
                lines.append(
                    f"• 语义质检触发 {sem_total} 次（report {int(sem.get('report') or 0)} / "
                    f"pick {int(sem.get('pick') or 0)}）| 主要类别: {tag_s} | 降级 {degraded} 次"
                )

        lines.append(
            "\n【当前模型路由】\n"
            f"• 早报分析: {m.report_analyst_model} ({m.report_analyst_thinking_level}) | "
            f"Bear挑刺: {m.report_auditor_model} ({m.report_auditor_thinking_level}) | "
            f"PM拍板: {m.report_model} ({m.report_thinking_level})\n"
            f"• 美股哨兵: 平稳 {m.us_sentinel_model} ({m.us_thinking_level}) / 异动 {m.us_anomaly_model} ({m.us_anomaly_thinking_level})\n"
            f"• 单聊: {m.chat_model} ({m.chat_thinking_level}) / deep: {m.chat_deep_model} ({m.chat_deep_thinking_level})\n"
            f"• 选股初研: {m.pick_researcher_model} ({m.pick_researcher_thinking_level}) | "
            f"终审: {m.pick_model} ({m.pick_thinking_level})\n"
            f"• 双向备援: Gemini←{m.deepseek_fallback_model} ({m.deepseek_fallback_thinking_level}) | "
            f"DeepSeek←{m.gemini_fallback_model} ({m.gemini_fallback_thinking_level}) | 同厂 {m.fallback_model}\n"
            f"• 早报搜索: 宏观与1v1 {m.report_priority_search_model} ({m.report_priority_search_thinking_level}) / "
            f"批量 {m.report_search_model} | 日常搜索: {m.search_model}"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_audition_reply(res: dict[str, Any]) -> str:
        reg = res.get("by_region") or {}
        fstats = res.get("fundamentals_stats") or {}
        sectors = list((res.get("by_sector") or {}).items())[:6]
        sec_str = " | ".join(f"{k}:{v}" for k, v in sectors) if sectors else "无"
        added = res.get("added_symbols") or []
        removed = res.get("removed_symbols") or []
        lines = [
            f"🎯 多市场选股底池海选完成 ({res.get('updated_at_uk', '')} UK)",
            f"• 漏斗: 基准池 {res.get('base_equities_count', 0)} → T212可买 {res.get('t212_tradable_count', 0)} → 流动性达标 {res.get('liquidity_passed_count', 0)} → 精选个股 {res.get('selected_equities_count', 0)} 只 + LSE ETF {res.get('selected_etfs_count', 0)} 只",
            f"• 区域分布: 美股 {reg.get('US', 0)} | 英股 {reg.get('UK', 0)} | 欧股 {reg.get('EU', 0)}",
            f"• 行业分布(前6): {sec_str}",
            f"• 基本面缓存: 命中 {fstats.get('cached', 0)} | 新抓 {fstats.get('fetched', 0)} | 失败 {fstats.get('failed', 0)}",
        ]
        if fstats.get("rate_limited"):
            lines.append(
                f"⚠️ Yahoo .info 触发连续 429 熔断保护，已安全跳过剩余 {fstats.get('skipped', 0)} 只（已抓数据已落库，稍后重跑可断点续传）。"
            )
        if added or removed:
            add_preview = ", ".join(added[:8]) + (f" 等{len(added)}只" if len(added) > 8 else "")
            rem_preview = ", ".join(removed[:8]) + (f" 等{len(removed)}只" if len(removed) > 8 else "")
            lines.append(f"• 池子变动: 新增 [{add_preview or '无'}] | 移出 [{rem_preview or '无'}]")
        preview = res.get("top_preview") or []
        if preview:
            lines.append(f"• 综合质地前列: {', '.join(preview[:6])}")
        return "\n".join(lines)
