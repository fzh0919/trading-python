"""Unified ops alerting hub supporting ops-notify dual-mode routing, cooldown deduplication, and offline recovery."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx

from src.bot.telegram_client import TelegramClient
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")
_VALID_SEVERITIES = {"INFO", "WARN", "CRITICAL"}


def send_ops_notify_sync(
    url: str,
    *,
    service: str = "trading-python",
    severity: str = "CRITICAL",
    alert_key: str = "process_fatal_crash",
    title: str = "trading-python 进程异常退出",
    message: str,
    cooldown_sec: int = 0,
    token: str = "",
    timeout_s: float = 5.0,
) -> bool:
    """Best-effort synchronous POST to ops-notify for fatal top-level crash reporting."""
    target = url.strip()
    if not target:
        return False
    sev = severity.upper().strip()
    if sev not in _VALID_SEVERITIES:
        sev = "CRITICAL"
    headers = {"Content-Type": "application/json"}
    tok = token.strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    payload = {
        "service": service,
        "severity": sev,
        "alert_key": alert_key,
        "title": title,
        "message": message,
        "cooldown_sec": max(0, int(cooldown_sec)),
    }
    try:
        resp = httpx.post(target, json=payload, headers=headers, timeout=timeout_s)
        return 200 <= resp.status_code < 300
    except Exception as exc:
        logger.warning("Synchronous ops-notify delivery failed (%s: %s)", type(exc).__name__, exc)
        return False


class OpsNotifier:
    """Centralized ops alerting channel with ops-notify routing + Telegram fallback + SQLite audit log."""

    def __init__(
        self,
        *,
        telegram: TelegramClient | None = None,
        chat_id: str = "",
        db: AdvisorDatabase | None = None,
        warn_cooldown_seconds: int = 1800,
        critical_cooldown_seconds: int = 120,
        clock: Callable[[], float] | None = None,
        ops_notify_url: str = "",
        ops_notify_token: str = "",
        service_name: str = "trading-python",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.tg = telegram
        self.chat_id = chat_id.strip()
        self.db = db
        self.warn_cooldown_seconds = max(0, int(warn_cooldown_seconds))
        self.critical_cooldown_seconds = max(0, int(critical_cooldown_seconds))
        self._clock = clock or time.monotonic
        self.ops_notify_url = ops_notify_url.strip()
        self.ops_notify_token = ops_notify_token.strip()
        self.service_name = service_name.strip() or "trading-python"
        self._transport = transport
        self._http_client: httpx.AsyncClient | None = None
        self._last_sent_at: dict[tuple[str, str], float] = {}
        self._suppressed_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._pending_buffer: list[str] = []
        self._pending_ops_buffer: list[dict[str, Any]] = []

    @property
    def ops_notify_enabled(self) -> bool:
        return bool(self.ops_notify_url)

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(transport=self._transport)
        return self._http_client

    async def close(self) -> None:
        """Close the reusable async HTTP client for ops-notify."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

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

    def _build_ops_payload(
        self,
        *,
        severity: str,
        source: str,
        alert_key: str,
        title: str,
        message: str,
        cooldown_sec: int,
    ) -> dict[str, Any]:
        sev = severity.upper().strip()
        if sev not in _VALID_SEVERITIES:
            sev = "WARN"
        clean_title = title.strip() or f"{source} ({alert_key})"
        return {
            "service": self.service_name,
            "severity": sev,
            "alert_key": alert_key,
            "title": clean_title[:120],
            "message": message,
            "cooldown_sec": max(0, int(cooldown_sec)),
        }

    async def _deliver_ops_notify(self, payload: dict[str, Any]) -> bool:
        if not self.ops_notify_url:
            return False
        headers = {"Content-Type": "application/json"}
        if self.ops_notify_token:
            headers["Authorization"] = f"Bearer {self.ops_notify_token}"
        try:
            client = self._get_http_client()
            resp = await client.post(self.ops_notify_url, json=payload, headers=headers, timeout=5.0)
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(
                "ops-notify returned HTTP %d for alert_key=%s: %s",
                resp.status_code,
                payload.get("alert_key"),
                resp.text[:160],
            )
            return False
        except Exception as exc:
            logger.warning(
                "OpsNotifier failed posting to ops-notify (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def _queue_offline(self, line: str) -> None:
        if len(self._pending_buffer) >= 20:
            self._pending_buffer.pop(0)
        self._pending_buffer.append(line)

    def _queue_ops_offline(self, payload: dict[str, Any]) -> None:
        if len(self._pending_ops_buffer) >= 20:
            self._pending_ops_buffer.pop(0)
        self._pending_ops_buffer.append(payload)

    async def notify(
        self,
        *,
        severity: str,
        source: str,
        alert_key: str,
        message: str,
        title: str = "",
        business_blocking: bool = False,
        bot_message: str = "",
    ) -> bool:
        """Emit an ops alert with cooldown deduplication and ops-notify/Telegram dual-mode routing."""
        sev = severity.upper().strip() or "WARN"
        if sev not in _VALID_SEVERITIES:
            sev = "WARN"
        key_tuple = (sev, alert_key)
        cooldown = (
            self.critical_cooldown_seconds
            if sev == "CRITICAL"
            else (self.warn_cooldown_seconds if sev == "WARN" else 0)
        )
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
        sup_note = f"\n（距上次告警期间已合并同类提示 {prior_suppressed} 次）" if prior_suppressed > 0 else ""

        if self.ops_notify_enabled:
            ops_sev = "CRITICAL" if business_blocking else sev
            payload = self._build_ops_payload(
                severity=ops_sev,
                source=source,
                alert_key=alert_key,
                title=title or f"[{source}] {alert_key}",
                message=f"[{source} @ {now_uk} UK] {message}{sup_note}",
                cooldown_sec=cooldown,
            )
            ops_sent = await self._deliver_ops_notify(payload)
            if not ops_sent:
                self._queue_ops_offline(payload)

            bot_sent = False
            if business_blocking:
                tip = bot_message.strip() or (
                    f"🚨 [业务阻塞告警] {title.strip() or source}\n"
                    f"{message[:220]}"
                )
                bot_sent = await self._deliver_text(tip)
                if not bot_sent:
                    self._queue_offline(f"• [{now_uk}] [CRITICAL] {source}: {tip[:180]}")

            sent = ops_sent or bot_sent
        else:
            icon = "🚨" if sev == "CRITICAL" else "⚠️"
            formatted = f"{icon} [运维告警 | {sev}] {source}\n时间: {now_uk} UK\n详情: {message}{sup_note}"
            sent = await self._deliver_text(formatted)
            if not sent:
                self._queue_offline(f"• [{now_uk}] [{sev}] {source}: {message[:180]}")

        await self._persist(
            severity="CRITICAL" if business_blocking else sev,
            source=source,
            alert_key=alert_key,
            message=message,
            notified=sent,
            suppressed_count=prior_suppressed,
        )
        return sent

    async def notify_business_critical(
        self,
        *,
        source: str,
        alert_key: str,
        message: str,
        title: str = "",
        bot_message: str = "",
    ) -> bool:
        """Emit a business-blocking CRITICAL alert (ops-notify CRITICAL + single concise trading bot prompt)."""
        return await self.notify(
            severity="CRITICAL",
            source=source,
            alert_key=alert_key,
            message=message,
            title=title,
            business_blocking=True,
            bot_message=bot_message,
        )

    async def record_offline_incident(self, *, source: str, alert_key: str, message: str) -> None:
        """Record a transport outage locally without attempting Telegram HTTP calls."""
        now_uk = self._now_uk()
        if self.ops_notify_enabled:
            payload = self._build_ops_payload(
                severity="WARN",
                source=source,
                alert_key=alert_key,
                title=f"通信中断: {source}",
                message=f"[{source} @ {now_uk} UK] {message}",
                cooldown_sec=self.warn_cooldown_seconds,
            )
            ops_sent = await self._deliver_ops_notify(payload)
            if not ops_sent:
                self._queue_ops_offline(payload)
            await self._persist(
                severity="WARN",
                source=source,
                alert_key=alert_key,
                message=message,
                notified=ops_sent,
                suppressed_count=0,
            )
            return

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
    # 捞取最近 24h 未送达的关键告警；注意区分冷却抑制写入的 (notified=0, suppressed_count>0) 记录避免重复刷屏，
    # 且补推时遵循 ops_notify_url 双态分流（技术类仅推 ops-notify，避免刷屏交易 Bot）。
    async def flush_pending(self) -> bool:
        """Flush buffered alerts after connectivity recovers."""
        all_ok = True
        if self.ops_notify_enabled and self._pending_ops_buffer:
            remaining_ops: list[dict[str, Any]] = []
            for idx, payload in enumerate(self._pending_ops_buffer):
                if await self._deliver_ops_notify(payload):
                    continue
                remaining_ops.extend(self._pending_ops_buffer[idx:])
                all_ok = False
                break
            self._pending_ops_buffer = remaining_ops

        if self._pending_buffer:
            summary = "🔄 [系统恢复] Telegram 通道已恢复，以下为断连期间离线汇总：\n" + "\n".join(self._pending_buffer)
            sent = await self._deliver_text(summary)
            if sent:
                self._pending_buffer.clear()
            else:
                all_ok = False
        return all_ok

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
        bot_sent = await self._deliver_text(msg, reply_markup=reply_markup)
        ops_sent = False
        if self.ops_notify_enabled:
            ops_payload = self._build_ops_payload(
                severity="INFO",
                source="lifecycle",
                alert_key="startup",
                title=f"v{version} 启动上线",
                message=f"Trading 212 × Gemini 投顾 v{version} 已于 {now_uk} UK 启动就绪。",
                cooldown_sec=0,
            )
            ops_sent = await self._deliver_ops_notify(ops_payload)
        sent = bot_sent or ops_sent
        await self._persist(
            severity="INFO",
            source="lifecycle",
            alert_key="startup",
            message=f"v{version} started",
            notified=sent,
        )
        return sent
