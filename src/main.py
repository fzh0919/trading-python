"""Application entrypoint: starts SQLite, OpsNotifier, Telegram Long Polling, and Weekday Scheduler."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from contextlib import suppress

import httpx

from src.agents.llm_client import GeminiClient
from src.bot.handlers import BOT_COMMANDS, REMOVE_KEYBOARD_MARKUP, AdvisorBot
from src.bot.scheduler import TradingScheduler
from src.bot.telegram_client import TelegramClient
from src.config import load_config
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.ops_notifier import OpsNotifier
from src.storage.db import AdvisorDatabase
from src.version import APP_VERSION


async def run_app() -> None:
    started_mono = time.monotonic()
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.app.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Prevent httpx INFO logs from printing Telegram Bot Token in request URLs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger = logging.getLogger("trading-advisor")

    db = AdvisorDatabase(
        cfg.app.db_path,
        cycle_reset_hour=cfg.schedule.morning_report_hour,
        cycle_reset_minute=cfg.schedule.morning_report_minute,
    )
    await db.init()

    tg = TelegramClient(cfg.env.telegram_bot_token)
    ops = OpsNotifier(
        telegram=tg,
        chat_id=cfg.env.telegram_chat_id,
        db=db,
        warn_cooldown_seconds=cfg.app.ops_warn_cooldown_seconds,
    )

    # TODO(IMP-18): [接自 IMP-04] 创建 T212Client 时注入 snapshot_loader=db.get_latest_snapshot 实现冷启动 5xx 读库降级兜底。
    t212 = T212Client(
        base_url=cfg.env.t212_base_url,
        isa_key=cfg.env.t212_isa_key,
        isa_secret=cfg.env.t212_isa_secret,
        invest_key=cfg.env.t212_invest_key,
        invest_secret=cfg.env.t212_invest_secret,
        order_history_days=cfg.strategy.order_history_days,
        order_history_max_pages=cfg.strategy.order_history_max_pages,
        cash_interest_rate_pct=cfg.strategy.cash_interest_rate_pct,
        cache_ttl_seconds=cfg.app.cache_ttl_seconds,
        ops_notifier=ops,
    )
    market_data = MarketDataService(
        finnhub_key=cfg.env.finnhub_api_key,
        fred_key=cfg.env.fred_api_key,
        ttl_seconds=cfg.app.market_data_ttl_seconds,
        ops_notifier=ops,
    )
    llm = GeminiClient(
        cfg.env.gemini_api_key,
        usage_recorder=db.log_llm_usage,
        ops_notifier=ops,
        fallback_model=cfg.models.fallback_model,
        fallback_thinking_level=cfg.models.fallback_thinking_level,
    )
    scheduler = TradingScheduler(cfg, t212, market_data, llm, db, tg, ops_notifier=ops)
    bot = AdvisorBot(
        cfg,
        t212,
        market_data,
        llm,
        db,
        tg,
        ops_notifier=ops,
        scheduler=scheduler,
        started_monotonic=started_mono,
    )

    if cfg.env.telegram_bot_token:
        wh_res = await tg.delete_webhook(drop_pending_updates=False)
        logger.info("Telegram deleteWebhook -> Long Polling switch: %s", wh_res.get("ok"))
        try:
            cmd_res = await tg.set_my_commands(BOT_COMMANDS)
            logger.info("Telegram setMyCommands synced: %s", cmd_res.get("ok"))
        except Exception as exc:
            logger.warning("Telegram setMyCommands failed: %s", exc)
        await ops.send_startup_notice(
            version=APP_VERSION,
            summary_lines=[
                f"核心路由: {cfg.models.report_model} ({cfg.models.report_thinking_level}) | 备援: {cfg.models.fallback_model}",
                f"全市场强制联网搜索: {'开启' if cfg.models.enable_forced_search_grounding else '关闭'} "
                f"(早报 {cfg.models.report_search_model} | 日常 {cfg.models.search_model})",
                f"晨报日程: 工作日 {cfg.schedule.morning_report_hour:02d}:{cfg.schedule.morning_report_minute:02d} UK",
            ],
            reply_markup=REMOVE_KEYBOARD_MARKUP,
        )
    else:
        logger.warning("TELEGRAM_BOT_TOKEN is empty; Telegram polling will idle until configured.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop_event.set)

    async def _poll_loop() -> None:
        offset: int | None = None
        consecutive_poll_errors = 0
        outage_recorded = False
        while not stop_event.is_set():
            if not cfg.env.telegram_bot_token:
                await asyncio.sleep(5.0)
                continue
            try:
                updates = await tg.get_updates(offset=offset, timeout_s=25)
                if consecutive_poll_errors >= 2:
                    await ops.flush_pending()
                consecutive_poll_errors = 0
                outage_recorded = False
                for upd in updates:
                    offset = max(offset or 0, int(upd["update_id"]) + 1)
                    await bot.handle_update(upd)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                consecutive_poll_errors += 1
                logger.warning("Telegram polling transient error (%s: %s); retrying in 5s", type(exc).__name__, exc)
                if consecutive_poll_errors >= 2 and not outage_recorded:
                    outage_recorded = True
                    await ops.record_offline_incident(
                        source="telegram:polling",
                        alert_key="telegram_polling_outage",
                        message=f"Telegram 轮询连续断连 ({type(exc).__name__}: {exc})",
                    )
                await asyncio.sleep(5.0)
            except Exception as exc:
                consecutive_poll_errors += 1
                logger.exception("Telegram polling error; retrying in 5s")
                if not outage_recorded:
                    outage_recorded = True
                    await ops.record_offline_incident(
                        source="telegram:polling",
                        alert_key="telegram_polling_exception",
                        message=f"Telegram 轮询异常 ({type(exc).__name__}: {exc})",
                    )
                await asyncio.sleep(5.0)

    sched_task = asyncio.create_task(scheduler.run_forever())
    poll_task = asyncio.create_task(_poll_loop())

    logger.info("Trading 212 x Gemini AI Advisor v%s running...", APP_VERSION)
    try:
        await stop_event.wait()
    finally:
        for task in (sched_task, poll_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        # TODO(IMP-18): [接自 IMP-03 & IMP-04 & IMP-05] 停机清理一并释放 await t212.close()、await market_data.close()（及 await db.close()）。
        await tg.close()
        logger.info("Shutdown complete.")


def main() -> None:
    try:
        asyncio.run(run_app())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
