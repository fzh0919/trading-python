"""Unified Telegram operations notifier with severity routing, cooldown deduplication, and offline recovery."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.bot.telegram_client import TelegramClient
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


class OpsNotifier:
    """Centralized ops alerting channel over Telegram + SQLite audit log."""

    def __init__(
        self,
        *,
        telegram: TelegramClient | None = None,
        chat_id: str = "",
        db: AdvisorDatabase | None = None,
        warn_cooldown_seconds: int = 1800,
        critical_cooldown_seconds: int = 120,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.tg = telegram
        self.chat_id = chat_id.strip()
        self.db = db
        self.warn_cooldown_seconds = max(0, int(warn_cooldown_seconds))
        self.critical_cooldown_seconds = max(0, int(critical_cooldown_seconds))
        self._clock = clock or time.monotonic
        self._last_sent_at: dict[tuple[str, str], float] = {}
        self._suppressed_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._pending_buffer: list[str] = []

    def _now_uk(self) -> str:
        return datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")

    async def _persist(
        self,
        *,
        severity: str,
        source: str,
        alert_key: str,
        message: str,
        notified: bool,
        suppressed_count: int = 0,
    ) -> None:
        if self.db is None:
            return
        try:
            await self.db.log_ops_event(
                severity=severity,
                source=source,
                alert_key=alert_key,
                message=message,
                notified=notified,
                suppressed_count=suppressed_count,
            )
        except Exception as exc:
            logger.debug("Failed to persist ops event: %s", exc)

    async def _deliver_text(
        self,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        if self.tg is None or not self.chat_id or not getattr(self.tg, "bot_token", ""):
            return False
        try:
            if reply_markup is not None:
                res = await self.tg.send_message(self.chat_id, text, reply_markup=reply_markup)
            else:
                res = await self.tg.send_message(self.chat_id, text)
            return res is not False
        except Exception as exc:
            logger.warning("OpsNotifier failed sending Telegram message (%s: %s)", type(exc).__name__, exc)
            return False

    def _queue_offline(self, line: str) -> None:
        if len(self._pending_buffer) >= 20:
            self._pending_buffer.pop(0)
        self._pending_buffer.append(line)

    async def notify(
        self,
        *,
        severity: str,
        source: str,
        alert_key: str,
        message: str,
    ) -> bool:
        """Emit an ops alert with cooldown deduplication; buffers when Telegram is unreachable."""
        sev = severity.upper().strip() or "WARN"
        key_tuple = (sev, alert_key)
        cooldown = self.critical_cooldown_seconds if sev == "CRITICAL" else self.warn_cooldown_seconds
        now = self._clock()
        last = self._last_sent_at.get(key_tuple)

        if last is not None and cooldown > 0 and (now - last) < cooldown:
            self._suppressed_counts[key_tuple] += 1
            await self._persist(
                severity=sev,
                source=source,
                alert_key=alert_key,
                message=message,
                notified=False,
                suppressed_count=self._suppressed_counts[key_tuple],
            )
            return False

        prior_suppressed = self._suppressed_counts.pop(key_tuple, 0)
        self._last_sent_at[key_tuple] = now

        now_uk = self._now_uk()
        icon = "🚨" if sev == "CRITICAL" else "⚠️"
        sup_note = f"\n（距上次告警期间已合并同类提示 {prior_suppressed} 次）" if prior_suppressed > 0 else ""
        formatted = f"{icon} [运维告警 | {sev}] {source}\n时间: {now_uk} UK\n详情: {message}{sup_note}"

        sent = await self._deliver_text(formatted)
        if not sent:
            self._queue_offline(f"• [{now_uk}] [{sev}] {source}: {message[:180]}")

        await self._persist(
            severity=sev,
            source=source,
            alert_key=alert_key,
            message=message,
            notified=sent,
            suppressed_count=prior_suppressed,
        )
        return sent

    async def record_offline_incident(self, *, source: str, alert_key: str, message: str) -> None:
        """Record a transport outage locally without attempting Telegram HTTP calls."""
        now_uk = self._now_uk()
        self._queue_offline(f"• [{now_uk}] [WARN] {source}: {message[:180]}")
        await self._persist(
            severity="WARN",
            source=source,
            alert_key=alert_key,
            message=message,
            notified=False,
            suppressed_count=0,
        )

    # TODO(IMP-17): [接自 IMP-03] flush_pending() 须调用 db.get_unnotified_ops_events() 与 db.mark_ops_events_notified()
    # 捞取最近 24h 未送达的关键告警；注意区分冷却抑制写入的 (notified=0, suppressed_count>0) 记录避免重复刷屏。
    async def flush_pending(self) -> bool:
        """Flush buffered alerts after Telegram connectivity recovers."""
        if not self._pending_buffer:
            return True
        summary = "🔄 [系统恢复] Telegram 通道已恢复，以下为断连期间离线汇总：\n" + "\n".join(self._pending_buffer)
        sent = await self._deliver_text(summary)
        if sent:
            self._pending_buffer.clear()
        return sent

    async def send_startup_notice(
        self,
        *,
        version: str,
        summary_lines: list[str] | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        """Send startup readiness banner with active version and quick keyboard."""
        now_uk = self._now_uk()
        lines = [
            f"🟢 [系统上线] Trading 212 × Gemini 投顾 v{version} 已启动",
            f"启动时间: {now_uk} UK",
        ]
        if summary_lines:
            lines.extend(summary_lines)
        lines.append("可通过左侧 / 菜单调用核心投研指令；输入 /help 查看全部运维指令，/health 查看实时体检。")
        msg = "\n".join(lines)
        sent = await self._deliver_text(msg, reply_markup=reply_markup)
        await self._persist(
            severity="INFO",
            source="lifecycle",
            alert_key="startup",
            message=f"v{version} started",
            notified=sent,
        )
        return sent
