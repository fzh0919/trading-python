"""Timezone-aware weekday scheduler (Europe/London + US market DST/holiday adaptation) and post-close settlement."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import generate_morning_report, generate_us_sentinel
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.analytics import _normalize_adjusted_ohlcv, resolve_market_benchmark
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.engine.calendar_utils import (
    add_trading_days,
    get_us_market_schedule_uk,
    has_market_closed_on_date,
    infer_market_from_symbol,
    is_london_weekday,
)
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

MORNING_CATCHUP_END_MINUTES = 9 * 60 + 30  # 09:30 UK
DAILY_SETTLEMENT_START_MINUTES = 22 * 60  # 22:00 UK (after US/UK/EU close & Friday radar)
DAILY_SETTLEMENT_END_MINUTES = 23 * 60 + 30  # 23:30 UK catch-up cutoff


def _parse_recommendation_date(rec: dict[str, Any]) -> date:
    """Extract the UK calendar date of a recommendation record."""
    created_uk = str(rec.get("created_at_uk") or "").strip()
    if len(created_uk) >= 10:
        try:
            return date.fromisoformat(created_uk[:10])
        except ValueError:
            pass
    created_ms = int(rec.get("created_at_ms") or 0)
    return datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).astimezone(LONDON_TZ).date()


def _resolve_benchmark_candidates(symbol: str, stored_benchmark: Any) -> list[str]:
    """Order stored benchmark first, then regional defaults from analytics."""
    candidates: list[str] = []
    stored = str(stored_benchmark or "").strip().upper()
    if stored:
        candidates.append(stored)
    candidates.extend(resolve_market_benchmark(symbol))
    return list(dict.fromkeys(c for c in candidates if c))


def _extract_closed_daily_closes(
    df: pd.DataFrame | None,
    market: str,
    now_utc: datetime,
) -> pd.Series:
    """Return sorted daily close prices keyed by calendar date for sessions already closed."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    norm = _normalize_adjusted_ohlcv(df)
    if norm is None or norm.empty or "Close" not in norm.columns:
        return pd.Series(dtype=float)

    closes = pd.to_numeric(norm["Close"], errors="coerce").dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return pd.Series(dtype=float)

    idx = pd.to_datetime(closes.index)
    by_date: dict[date, float] = {}
    for ts, val in zip(idx, closes.to_numpy(dtype=float)):
        if not math.isfinite(val) or val <= 0:
            continue
        bar_date = ts.date()
        if has_market_closed_on_date(bar_date, market=market, now_utc=now_utc):
            by_date[bar_date] = float(val)

    if not by_date:
        return pd.Series(dtype=float)
    ordered_dates = sorted(by_date.keys())
    return pd.Series([by_date[d] for d in ordered_dates], index=ordered_dates, dtype=float)


def _clean_reflection_lesson(raw_text: str, max_chars: int = 200) -> str:
    """Strip markdown artifacts and keep the T+N reflection compact (<80 tokens)."""
    text = str(raw_text or "").replace("#", "").replace("**", "")
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    for sep in ("。", "！", "？", "."):
        pos = clipped.rfind(sep)
        if pos >= max_chars // 2:
            return clipped[: pos + 1].strip()
    return clipped.rstrip()


class TradingScheduler:
    """Deterministic minute-tick scheduler with per-job isolation and Telegram ops alerting."""

    def __init__(
        self,
        cfg: AppConfig,
        t212: T212Client,
        market_data: MarketDataService,
        llm: LLMBackend,
        db: AdvisorDatabase,
        telegram: TelegramClient,
        ops_notifier: OpsNotifier | None = None,
        execution_lock: asyncio.Lock | None = None,
        morning_retry_cooldown_s: float = 300.0,
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self.execution_lock: asyncio.Lock = execution_lock if execution_lock is not None else asyncio.Lock()
        self.current_job_name: str | None = None
        self.morning_retry_cooldown_s = max(0.0, float(morning_retry_cooldown_s))
        self._fired_dates: dict[str, str] = {}
        self._morning_failures: dict[str, dict[str, Any]] = {}
        self._morning_missed_notified: set[str] = set()
        self.last_job_runs: dict[str, dict[str, str]] = {}

    async def _is_job_locked(self, job_name: str, uk_date: str) -> bool:
        if self._fired_dates.get(job_name) == uk_date:
            return True
        if await self.db.has_scheduler_job_fired(job_name, uk_date):
            self._fired_dates[job_name] = uk_date
            return True
        return False

    async def _mark_job_locked(self, job_name: str, uk_date: str, uk_now_str: str) -> None:
        self._fired_dates[job_name] = uk_date
        await self.db.mark_scheduler_job_fired(job_name, uk_date, uk_now_str)

    async def _claim_job_run(self, job_name: str, uk_date: str, uk_now_str: str) -> bool:
        """Atomically check in-memory + SQLite persistence so restarts inside the trigger window never double-fire."""
        if await self._is_job_locked(job_name, uk_date):
            return False
        await self._mark_job_locked(job_name, uk_date, uk_now_str)
        return True

    async def _execute_job(
        self,
        *,
        job_name: str,
        chat_id: str,
        uk_now_str: str,
        coro: Awaitable[str],
        send_to_tg: bool = True,
        send_failure_to_tg: bool = True,
    ) -> bool:
        async with self.execution_lock:
            self.current_job_name = job_name
            try:
                msg = await coro
                if msg.lstrip().startswith("⚠️"):
                    self.last_job_runs[job_name] = {"status": "FAILED", "uk_time": uk_now_str, "detail": msg[:120]}
                    if self.ops is not None:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_job_failed:{job_name}",
                            message=f"定时任务 {job_name} 返回异常结果：{msg[:200]}",
                        )
                    if send_to_tg and send_failure_to_tg and chat_id:
                        await self.tg.send_message(chat_id, msg)
                    return False

                detail = msg[:120] if not send_to_tg else ""
                self.last_job_runs[job_name] = {"status": "OK", "uk_time": uk_now_str, "detail": detail}
                if send_to_tg and chat_id:
                    await self.tg.send_message(chat_id, msg)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job %s crashed", job_name)
                err_detail = f"{type(exc).__name__}: {exc}"
                self.last_job_runs[job_name] = {"status": "CRASHED", "uk_time": uk_now_str, "detail": err_detail[:120]}
                if self.ops is not None:
                    if job_name == "morning" and send_failure_to_tg and hasattr(self.ops, "notify_business_critical"):
                        await self.ops.notify_business_critical(
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            title="晨报定时任务异常崩溃",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                            bot_message="🚨 今日 /report 定时晨报生成异常中断，请发送 /report 重试或查看运维告警。",
                        )
                    else:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                        )
                return False
            finally:
                if self.current_job_name == job_name:
                    self.current_job_name = None

    async def _generate_settlement_reflection(
        self,
        *,
        rec: dict[str, Any],
        rec_date: date,
        exit_date: date,
        holding_days: int,
        entry_price: float,
        exit_price: float,
        raw_return_pct: float,
        benchmark_symbol: str,
        bench_return_pct: float,
        alpha_pct: float,
    ) -> str:
        """Generate a concise 2-4 sentence (<80 tokens) T+N post-mortem lesson using Hydrogen LOW."""
        symbol = str(rec.get("symbol") or "").upper()
        account = str(rec.get("account") or "ISA")
        action = str(rec.get("action") or "WATCH")
        thesis = str(rec.get("thesis") or "").strip()[:220] or "未记录"

        system_instruction = (
            f"你是冷静客观的量化投研复盘教练。现在已知历史推荐在 T+{holding_days} 个交易日后的真实表现。\n"
            f"注意 T+{holding_days} 窗口可能短于中长线逻辑本身，不能仅凭短期波动全盘否定长线逻辑，但要直面短期择时或催化偏差。\n"
            "请用中文输出 2~4 句极简纯文本复盘教训（严格控制在 80 tokens / 120 汉字以内；"
            "严禁使用 #、**、列表符号或换行标题）：\n"
            f"1. 点明 T+{holding_days} 收益与相对 {benchmark_symbol} 的超额 Alpha 表现；\n"
            "2. 指出原推荐逻辑中哪部分被验证或证伪；\n"
            "3. 提炼 1 条下次同类选股或择时可直接复用的具体纪律。"
        )
        prompt = (
            f"标的: {symbol} ({account} | {action})\n"
            f"推荐日期: {rec_date.isoformat()} -> 结算日: {exit_date.isoformat()} (T+{holding_days})\n"
            f"入场价: {entry_price:.4g} -> 结算价: {exit_price:.4g}\n"
            f"原推荐逻辑: {thesis}\n"
            f"真实收益率: {raw_return_pct:+.2f}%\n"
            f"基准({benchmark_symbol})同期: {bench_return_pct:+.2f}%\n"
            f"超额 Alpha: {alpha_pct:+.2f}%"
        )
        raw_reply = await self.llm.generate(
            model=self.cfg.models.chat_model,
            prompt=prompt,
            system_instruction=system_instruction,
            thinking_level="LOW",
            use_google_search=False,
            temperature=0.2,
            usage_label="settlement_reflection",
        )
        cleaned = _clean_reflection_lesson(raw_reply)
        if not cleaned or cleaned.lstrip().startswith("⚠️"):
            raise RuntimeError(cleaned or "empty reflection response")
        return cleaned

    async def _run_daily_settlement_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        uk_now_str = base.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        pending = await self.db.get_pending_settlements(limit=50, only_with_entry_price=False)

        skipped = 0
        not_due = 0
        settled = 0
        failed = 0
        mature_items: list[tuple[dict[str, Any], date, date, str, int, float, list[str]]] = []

        for rec in pending:
            rec_id = int(rec["id"])
            symbol = str(rec.get("symbol") or "").strip().upper()
            entry_raw = rec.get("entry_price")
            bench_candidates = _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))

            try:
                entry_val = float(entry_raw) if entry_raw is not None else 0.0
            except (TypeError, ValueError):
                entry_val = 0.0

            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=bench_candidates[0] if bench_candidates else "SPY",
                    settled_at_uk=uk_now_str,
                    raw_return_pct=None,
                    alpha_pct=None,
                    outcome_status="SKIPPED",
                    reflection_lesson="",
                )
                skipped += 1
                continue

            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            holding_days = max(1, int(rec.get("horizon_days") or self.cfg.strategy.settlement_holding_days or 5))
            target_date = add_trading_days(rec_date, holding_days, market=market)
            if not has_market_closed_on_date(target_date, market=market, now_utc=base):
                not_due += 1
                continue

            mature_items.append((rec, rec_date, target_date, market, holding_days, entry_val, bench_candidates))

        if mature_items:
            fetch_symbols = list(
                dict.fromkeys(
                    [str(rec["symbol"]).strip().upper() for rec, *_ in mature_items]
                    + [b for *_, b_list in mature_items for b in b_list]
                )
            )
            try:
                histories = await self.market_data.get_histories(fetch_symbols, period="1y")
            except Exception as exc:
                logger.warning("Daily settlement history fetch failed: %s", exc)
                histories = {}
                failed += len(mature_items)
                mature_items = []

            closes_cache: dict[tuple[str, str], pd.Series] = {}

            def _get_closes(sym: str, mkt: str) -> pd.Series:
                key = (sym, mkt)
                if key not in closes_cache:
                    closes_cache[key] = _extract_closed_daily_closes(histories.get(sym), mkt, base)
                return closes_cache[key]

            for rec, rec_date, target_date, market, holding_days, entry_price, bench_candidates in mature_items:
                rec_id = int(rec["id"])
                symbol = str(rec["symbol"]).strip().upper()
                stock_closes = _get_closes(symbol, market)
                exit_dates = [d for d in stock_closes.index if d >= target_date]
                if not exit_dates:
                    not_due += 1
                    continue
                exit_date = exit_dates[0]
                exit_price = float(stock_closes.loc[exit_date])

                # Guard against GBp/GBP 100x unit mismatch on .L stocks
                eff_entry = entry_price
                if symbol.endswith(".L"):
                    prior_dates = [d for d in stock_closes.index if d <= rec_date]
                    if prior_dates and (rec_date - prior_dates[-1]).days <= 5:
                        hist_entry = float(stock_closes.loc[prior_dates[-1]])
                        unit_ratio = hist_entry / eff_entry
                        if 20.0 <= unit_ratio <= 200.0:
                            eff_entry *= 100.0
                        elif 0.005 <= unit_ratio <= 0.05:
                            eff_entry *= 0.01
                    exit_ratio = exit_price / eff_entry
                    if 0.005 <= exit_ratio <= 0.05 or 20.0 <= exit_ratio <= 200.0:
                        logger.warning(
                            "Skipping settlement for %s (id=%s) due to unresolved GBp/GBP 100x mismatch",
                            symbol,
                            rec_id,
                        )
                        failed += 1
                        continue

                raw_return_pct = ((exit_price - eff_entry) / eff_entry) * 100.0

                chosen_bench = ""
                bench_ret_pct = 0.0
                alpha_pct: float | None = None
                for b_sym in bench_candidates:
                    b_market = infer_market_from_symbol(b_sym)
                    b_closes = _get_closes(b_sym, b_market)
                    if b_closes.empty or b_closes.index[0] > rec_date or b_closes.index[-1] < exit_date:
                        continue
                    b_entry_dates = [d for d in b_closes.index if d <= rec_date]
                    b_exit_dates = [d for d in b_closes.index if d <= exit_date]
                    if not b_entry_dates or not b_exit_dates:
                        continue
                    b_entry = float(b_closes.loc[b_entry_dates[-1]])
                    b_exit = float(b_closes.loc[b_exit_dates[-1]])
                    if b_entry > 0:
                        bench_ret_pct = ((b_exit - b_entry) / b_entry) * 100.0
                        alpha_pct = raw_return_pct - bench_ret_pct
                        chosen_bench = b_sym
                        break

                if alpha_pct is None:
                    logger.warning("Benchmark history unavailable for settling %s (%s -> %s)", symbol, rec_date, exit_date)
                    failed += 1
                    continue

                try:
                    lesson = await self._generate_settlement_reflection(
                        rec=rec,
                        rec_date=rec_date,
                        exit_date=exit_date,
                        holding_days=holding_days,
                        entry_price=eff_entry,
                        exit_price=exit_price,
                        raw_return_pct=raw_return_pct,
                        benchmark_symbol=chosen_bench,
                        bench_return_pct=bench_ret_pct,
                        alpha_pct=alpha_pct,
                    )
                except Exception as exc:
                    logger.warning("Settlement reflection failed for %s (id=%s): %s", symbol, rec_id, exc)
                    failed += 1
                    continue

                saved = await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=chosen_bench,
                    settled_at_uk=uk_now_str,
                    raw_return_pct=round(raw_return_pct, 2),
                    alpha_pct=round(alpha_pct, 2),
                    outcome_status="SETTLED",
                    reflection_lesson=lesson,
                )
                if saved:
                    settled += 1

        pruned = await self.db.prune_old_records(retention_days=90, now_utc=base)
        return {
            "checked": len(pending),
            "settled": settled,
            "skipped": skipped,
            "not_due": not_due,
            "failed": failed,
            "pruned": pruned,
        }

    async def run_daily_settlement(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Settle matured T+N recommendations with Alpha + Hydrogen LOW reflection and prune 90d logs."""
        async with self.execution_lock:
            self.current_job_name = "daily_settlement"
            try:
                return await self._run_daily_settlement_impl(now_utc=now_utc)
            finally:
                if self.current_job_name == "daily_settlement":
                    self.current_job_name = None

    async def _run_daily_settlement_job(self, now_utc: datetime) -> str:
        summary = await self._run_daily_settlement_impl(now_utc=now_utc)
        pruned_total = sum(summary["pruned"].values())
        if summary["settled"] == 0 and summary["failed"] > 0:
            return (
                f"⚠️ 盘后结算异常：待处理 {summary['checked']} 条中 {summary['failed']} 条失败，"
                f"清理 {pruned_total} 条过期记录"
            )
        return (
            f"结算 {summary['settled']} 条 / 跳过 {summary['skipped']} / "
            f"未到期 {summary['not_due']} / 失败 {summary['failed']} / 清理 {pruned_total}"
        )

    async def check_and_run_due_jobs(self, now_utc: datetime | None = None) -> list[str]:
        """Evaluate morning catch-up, premarket sentinel, weekly radar, and post-close settlement."""
        base = now_utc or datetime.now(timezone.utc)
        if not is_london_weekday(base):
            return []

        chat_id = self.cfg.env.telegram_chat_id.strip()
        if not chat_id:
            return []

        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        uk_min_of_day = dt_uk.hour * 60 + dt_uk.minute
        fired: list[str] = []

        # 1. Morning Report with N95 Catch-up & Retry (07:30 ~ 09:30 UK; lock scheduler_runs only on success)
        morning_target = self.cfg.schedule.morning_report_hour * 60 + self.cfg.schedule.morning_report_minute
        morning_catchup_end = max(morning_target + 5, MORNING_CATCHUP_END_MINUTES)
        if morning_target <= uk_min_of_day < morning_catchup_end:
            if not await self._is_job_locked("morning", uk_date):
                retry_info = self._morning_failures.get(uk_date)
                if retry_info is None or base >= retry_info["next_retry_utc"]:
                    fired.append("morning")
                    logger.info("Running scheduled morning report (%s)", uk_date)
                    ok = await self._execute_job(
                        job_name="morning",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_morning_report(
                            self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                        ),
                        send_failure_to_tg=False,
                    )
                    if ok:
                        await self._mark_job_locked("morning", uk_date, uk_now_str)
                        self._morning_failures.pop(uk_date, None)
                    else:
                        attempts = int(retry_info["attempts"]) + 1 if retry_info else 1
                        self._morning_failures[uk_date] = {
                            "attempts": attempts,
                            "next_retry_utc": base + timedelta(seconds=self.morning_retry_cooldown_s),
                        }
        elif uk_min_of_day >= morning_catchup_end:
            if (
                uk_date in self._morning_failures
                and uk_date not in self._morning_missed_notified
                and not await self._is_job_locked("morning", uk_date)
            ):
                self._morning_missed_notified.add(uk_date)
                if self.ops is not None and hasattr(self.ops, "notify_business_critical"):
                    await self.ops.notify_business_critical(
                        source="scheduler:morning",
                        alert_key=f"scheduler_morning_catchup_exhausted:{uk_date}",
                        title="晨报自动补发窗口已结束",
                        message=f"{uk_date} 早报在 09:30 UK 前多次补发仍未成功，请尽快检查网络或接口状态。",
                        bot_message="🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )
                else:
                    await self.tg.send_message(
                        chat_id,
                        "🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )

        # 2. US Pre-Market Sentinel (Mon-Fri, dynamic DST-adjusted minutes_to_open == lead)
        us_sched = get_us_market_schedule_uk(base)
        lead = self.cfg.schedule.premarket_lead_minutes
        if us_sched.minutes_to_open <= lead and us_sched.minutes_to_open > lead - 5:
            if await self._claim_job_run("premarket", uk_date, uk_now_str):
                fired.append("premarket")
                if us_sched.is_us_holiday:
                    logger.info("US Market Holiday (%s): sending quiet notice.", us_sched.ny_date)
                    self.last_job_runs["premarket"] = {
                        "status": "HOLIDAY_MUTED",
                        "uk_time": uk_now_str,
                        "detail": us_sched.ny_date,
                    }
                    await self.tg.send_message(
                        chat_id,
                        "🔔 今日为美国股市法定休市日（Market Holiday），美股盘前哨兵自动静音，祝你今天轻松愉快！",
                    )
                elif us_sched.is_us_trading_day:
                    logger.info("Running scheduled US pre-market sentinel (%s)", uk_date)
                    await self._execute_job(
                        job_name="premarket",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_us_sentinel(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            is_scheduled_premarket=True,
                            ops_notifier=self.ops,
                        ),
                    )

        # 3. Friday Post-Close Weekly Opportunity Radar (respects dynamic early-close minutes_to_close e.g. 18:30 UK)
        if self.cfg.schedule.enable_weekly_radar and dt_uk.weekday() == 4 and us_sched.is_us_trading_day:
            after = self.cfg.schedule.weekly_radar_after_close_minutes
            mins_after_close = -us_sched.minutes_to_close
            if after <= mins_after_close < after + 5:
                if await self._claim_job_run("weekly_radar", uk_date, uk_now_str):
                    fired.append("weekly_radar")
                    logger.info("Running scheduled Friday weekly opportunity radar (%s)", uk_date)
                    await self._execute_job(
                        job_name="weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_pick_radar(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            focus_query="",
                            is_weekly_auto=True,
                            ops_notifier=self.ops,
                        ),
                    )

        # 4. Daily Post-Close T+N Recommendation Settlement & 90-Day Pruning (22:00 ~ 23:30 UK)
        if DAILY_SETTLEMENT_START_MINUTES <= uk_min_of_day < DAILY_SETTLEMENT_END_MINUTES:
            if await self._claim_job_run("daily_settlement", uk_date, uk_now_str):
                fired.append("daily_settlement")
                logger.info("Running scheduled daily T+N settlement & DB pruning (%s)", uk_date)
                await self._execute_job(
                    job_name="daily_settlement",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_daily_settlement_job(base),
                    send_to_tg=False,
                )

        return fired

    async def run_forever(self, poll_interval_s: float = 30.0) -> None:
        logger.info("TradingScheduler started (Europe/London timezone aware).")
        while True:
            try:
                await self.check_and_run_due_jobs()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job loop error")
                if self.ops is not None:
                    await self.ops.notify(
                        severity="CRITICAL",
                        source="scheduler:loop",
                        alert_key="scheduler_loop_error",
                        message=f"调度主循环异常：{type(exc).__name__}: {exc}",
                    )
            await asyncio.sleep(poll_interval_s)
