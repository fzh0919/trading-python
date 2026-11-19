"""TradingScheduler core lifecycle, per-job locking, dispatch loop, and minute-tick runner."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable

from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import generate_morning_report, generate_us_sentinel
from src.bot.scheduler.intraday_mixin import IntradaySentinelJobMixin
from src.bot.scheduler.settlement_math import (
    DAILY_SETTLEMENT_END_MINUTES,
    DAILY_SETTLEMENT_START_MINUTES,
    INTRADAY_SENTINEL_END_MINUTES,
    INTRADAY_SENTINEL_START_MINUTES,
    LONDON_TZ,
    MORNING_CATCHUP_END_MINUTES,
    NAV_SNAPSHOT_END_MINUTES,
    NAV_SNAPSHOT_START_MINUTES,
    WEEKLY_AUDIT_END_MINUTES,
    WEEKLY_AUDIT_START_MINUTES,
    WEEKLY_WALLET_RECONCILE_START_MINUTES,
)
from src.bot.scheduler.settlement_mixin import SettlementJobMixin
from src.bot.scheduler.weekly_audit_mixin import WeeklyAuditJobMixin
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.engine.calendar_utils import get_us_market_schedule_uk, is_london_weekday
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger("src.bot.scheduler")


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("src.bot.scheduler")
    return getattr(facade, name, default) if facade is not None else default


class TradingScheduler(SettlementJobMixin, WeeklyAuditJobMixin, IntradaySentinelJobMixin):
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
        self._job_retry_after: dict[str, datetime] = {}
        self._morning_failures: dict[str, dict[str, Any]] = {}
        self._morning_missed_notified: set[str] = set()
        self.last_job_runs: dict[str, dict[str, str]] = {}
        self._last_intraday_scan_utc: datetime | None = None

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

    async def _claim_job_run(
        self,
        job_name: str,
        uk_date: str,
        uk_now_str: str,
        *,
        now_utc: datetime | None = None,
        retry_cooldown_s: int = 0,
    ) -> bool:
        """Atomically check in-memory + SQLite persistence so restarts inside the trigger window never double-fire."""
        if retry_cooldown_s > 0:
            retry_after = self._job_retry_after.get(job_name)
            if retry_after is not None and (now_utc or datetime.now(timezone.utc)) < retry_after:
                return False
        if await self._is_job_locked(job_name, uk_date):
            return False
        await self._mark_job_locked(job_name, uk_date, uk_now_str)
        return True

    async def _release_job_run(
        self,
        job_name: str,
        uk_date: str,
        *,
        now_utc: datetime | None = None,
        retry_cooldown_s: int = 120,
    ) -> None:
        """Release a transiently failed one-shot claim so the remaining window can retry."""
        self._fired_dates.pop(job_name, None)
        try:
            await self.db.clear_scheduler_job_fired(job_name, uk_date)
        except Exception as exc:
            logger.warning("Failed to clear scheduler claim %s/%s: %s", job_name, uk_date, exc)
        if retry_cooldown_s > 0:
            base = now_utc or datetime.now(timezone.utc)
            self._job_retry_after[job_name] = base + timedelta(seconds=retry_cooldown_s)

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
                if msg.lstrip().startswith("⛔"):
                    self.last_job_runs[job_name] = {"status": "BLOCKED", "uk_time": uk_now_str, "detail": msg[:120]}
                    if send_to_tg and chat_id and msg.strip():
                        await self.tg.send_message(chat_id, msg)
                    return True
                self.last_job_runs[job_name] = {"status": "OK", "uk_time": uk_now_str, "detail": detail}
                if send_to_tg and chat_id and msg.strip():
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

    async def check_and_run_due_jobs(self, now_utc: datetime | None = None) -> list[str]:
        """Evaluate prewarm, morning catch-up, premarket sentinel, weekly radar, Saturday quant audit, and post-close settlement."""
        base = now_utc or datetime.now(timezone.utc)
        chat_id = self.cfg.env.telegram_chat_id.strip()
        if not chat_id:
            return []

        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        uk_min_of_day = dt_uk.hour * 60 + dt_uk.minute
        fired: list[str] = []

        # Saturday idle window (10:00 ~ 22:00 UK): silent 0-token offline universe & threshold health audit
        if dt_uk.weekday() == 5:
            if WEEKLY_AUDIT_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES and self._has_audit_universe():
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_quant_audit", week_key, uk_now_str):
                    fired.append("weekly_quant_audit")
                    logger.info("Running Saturday silent weekly quant health audit (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_quant_audit",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_quant_audit_job(base),
                        send_to_tg=True,
                    )

            # Saturday ledger reminder window (12:00 ~ 22:00 UK): weekly Idempotent GBP+USD reconcile nudge
            if WEEKLY_WALLET_RECONCILE_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES:
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_wallet_reconcile", week_key, uk_now_str):
                    fired.append("weekly_wallet_reconcile")
                    logger.info("Sending weekly wallet reconcile reminder (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_wallet_reconcile",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_wallet_reconcile_job(base),
                    )
            return fired

        if not is_london_weekday(base):
            return []

        # 0. Silent Morning Prewarm (-10m before morning_report, e.g. 07:20 ~ 07:29 UK)
        morning_target = self.cfg.schedule.morning_report_hour * 60 + self.cfg.schedule.morning_report_minute
        morning_prewarm_target = max(0, morning_target - 10)
        if morning_prewarm_target <= uk_min_of_day < morning_target:
            if await self._claim_job_run("prewarm_morning", uk_date, uk_now_str):
                fired.append("prewarm_morning")
                logger.info("Running silent morning prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_morning",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("morning", base),
                    send_to_tg=False,
                )

        # 1. Morning Report with N95 Catch-up & Retry (07:30 ~ 09:30 UK; lock scheduler_runs only on success)
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
                        coro=_facade_attr("generate_morning_report", generate_morning_report)(
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

        # 2. US Pre-Market Sentinel & -10m Silent Prewarm (Mon-Fri, dynamic DST-adjusted minutes_to_open)
        us_sched = get_us_market_schedule_uk(base)
        lead = self.cfg.schedule.premarket_lead_minutes
        if us_sched.is_us_trading_day and not us_sched.is_us_holiday and lead < us_sched.minutes_to_open <= lead + 10:
            if await self._claim_job_run("prewarm_premarket", uk_date, uk_now_str):
                fired.append("prewarm_premarket")
                logger.info("Running silent premarket prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_premarket",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("premarket", base),
                    send_to_tg=False,
                )

        if us_sched.minutes_to_open <= lead and us_sched.minutes_to_open > lead - 5:
            if await self._claim_job_run(
                "premarket", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=60
            ):
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
                    ok = await self._execute_job(
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
                    if not ok:
                        await self._release_job_run(
                            "premarket", uk_date, now_utc=base, retry_cooldown_s=60
                        )

        # 3. Friday Post-Close Weekly Opportunity Radar & -10m Silent Prewarm (respects dynamic early-close)
        if self.cfg.schedule.enable_weekly_radar and dt_uk.weekday() == 4 and us_sched.is_us_trading_day:
            after = self.cfg.schedule.weekly_radar_after_close_minutes
            mins_after_close = -us_sched.minutes_to_close
            prewarm_after = max(0, after - 10)
            if prewarm_after <= mins_after_close < after:
                if await self._claim_job_run("prewarm_weekly_radar", uk_date, uk_now_str):
                    fired.append("prewarm_weekly_radar")
                    logger.info("Running silent weekly radar prewarm (%s)", uk_date)
                    await self._execute_job(
                        job_name="prewarm_weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_prewarm_job("weekly_radar", base),
                        send_to_tg=False,
                    )
            if after <= mins_after_close < after + 30:
                if await self._claim_job_run(
                    "weekly_radar", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=120
                ):
                    fired.append("weekly_radar")
                    logger.info("Running scheduled Friday weekly opportunity radar (%s)", uk_date)
                    ok = await self._execute_job(
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
                    if not ok:
                        await self._release_job_run(
                            "weekly_radar", uk_date, now_utc=base, retry_cooldown_s=120
                        )

        # 3b. Evening EOD NAV snapshot (21:40~22:00 UK, weekdays): one post-close NAV row
        if (
            NAV_SNAPSHOT_START_MINUTES <= uk_min_of_day < NAV_SNAPSHOT_END_MINUTES
            and dt_uk.weekday() < 5
        ):
            if await self._claim_job_run(
                "nav_snapshot", uk_date, uk_now_str, now_utc=base, retry_cooldown_s=60
            ):
                fired.append("nav_snapshot")
                logger.info("Running post-close NAV snapshot (%s)", uk_date)
                ok = await self._execute_job(
                    job_name="nav_snapshot",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_nav_snapshot_job(),
                    send_to_tg=False,
                )
                if not ok:
                    await self._release_job_run(
                        "nav_snapshot", uk_date, now_utc=base, retry_cooldown_s=60
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

        # 5. Intraday event-driven sentinel (08:00~21:00 UK; 0-token scan, never queues behind user commands)
        if (
            self.cfg.strategy.intraday_sentinel_enabled
            and INTRADAY_SENTINEL_START_MINUTES <= uk_min_of_day < INTRADAY_SENTINEL_END_MINUTES
            and not self.execution_lock.locked()
            and self._intraday_scan_due(base)
        ):
            pushed = await self._run_intraday_sentinel_job(base, chat_id=chat_id, uk_date=uk_date)
            if pushed:
                fired.append("intraday_sentinel")

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
