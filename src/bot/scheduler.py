"""Timezone-aware weekday scheduler (Europe/London + US market DST/holiday adaptation)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable
from zoneinfo import ZoneInfo

from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import generate_morning_report, generate_us_sentinel
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.engine.calendar_utils import get_us_market_schedule_uk, is_london_weekday
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


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
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self._fired_dates: dict[str, str] = {}
        self.last_job_runs: dict[str, dict[str, str]] = {}

    async def _claim_job_run(self, job_name: str, uk_date: str, uk_now_str: str) -> bool:
        """Atomically check in-memory + SQLite persistence so restarts inside the trigger window never double-fire."""
        if self._fired_dates.get(job_name) == uk_date:
            return False
        if await self.db.has_scheduler_job_fired(job_name, uk_date):
            self._fired_dates[job_name] = uk_date
            return False
        self._fired_dates[job_name] = uk_date
        await self.db.mark_scheduler_job_fired(job_name, uk_date, uk_now_str)
        return True

    async def _execute_job(
        self,
        *,
        job_name: str,
        chat_id: str,
        uk_now_str: str,
        coro: Awaitable[str],
    ) -> None:
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
            else:
                self.last_job_runs[job_name] = {"status": "OK", "uk_time": uk_now_str, "detail": ""}
            await self.tg.send_message(chat_id, msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Scheduled job %s crashed", job_name)
            err_detail = f"{type(exc).__name__}: {exc}"
            self.last_job_runs[job_name] = {"status": "CRASHED", "uk_time": uk_now_str, "detail": err_detail[:120]}
            if self.ops is not None:
                await self.ops.notify(
                    severity="CRITICAL",
                    source=f"scheduler:{job_name}",
                    alert_key=f"scheduler_crash:{job_name}",
                    message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                )

    async def check_and_run_due_jobs(self, now_utc: datetime | None = None) -> list[str]:
        """Evaluate whether morning report, premarket sentinel, or weekly radar is due right now."""
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

        # 1. Morning Report (Mon-Fri configured morning time e.g. 07:30 UK, 5-minute catchup window)
        morning_target = self.cfg.schedule.morning_report_hour * 60 + self.cfg.schedule.morning_report_minute
        if morning_target <= uk_min_of_day < morning_target + 5:
            if await self._claim_job_run("morning", uk_date, uk_now_str):
                fired.append("morning")
                logger.info("Running scheduled morning report (%s)", uk_date)
                await self._execute_job(
                    job_name="morning",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=generate_morning_report(
                        self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                    ),
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

        # 3. Friday Post-Close Weekly Opportunity Radar
        # TODO(IMP-16): [接自 IMP-02 & IMP-03] 此处 minutes_to_close 已按美股半日市动态切至 18:30 UK，勿写死 21:30 UK；
        # 盘后结算需联合 infer_market_from_symbol + add_trading_days(..., cfg.strategy.settlement_holding_days) + has_market_closed_on_date，
        # 并调用 db.get_pending_settlements() / db.settle_recommendation() 与 db.prune_old_records(retention_days=90)。
        if self.cfg.schedule.enable_weekly_radar and dt_uk.weekday() == 4 and us_sched.is_us_trading_day:
            after = self.cfg.schedule.weekly_radar_after_close_minutes
            # minutes_to_close is negative after close; -minutes_to_close >= after
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
