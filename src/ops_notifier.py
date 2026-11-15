"""Unified ops alerting hub supporting ops-notify dual-mode routing, cooldown deduplication, and offline recovery."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
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
        self._pending_tg_keys: list[tuple[str, str, str] | None] = []
        self._pending_ops_keys: list[tuple[str, str, str] | None] = []

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

    @staticmethod
    def _event_key(source: str, alert_key: str, message: str) -> tuple[str, str, str]:
        return (str(source), str(alert_key), str(message)[:1000])

    def _queue_offline(self, line: str, event_key: tuple[str, str, str] | None = None) -> None:
        if len(self._pending_buffer) >= 20:
            self._pending_buffer.pop(0)
            if self._pending_tg_keys:
                self._pending_tg_keys.pop(0)
        self._pending_buffer.append(line)
        self._pending_tg_keys.append(event_key)

    def _queue_ops_offline(
        self,
        payload: dict[str, Any],
        event_key: tuple[str, str, str] | None = None,
    ) -> None:
        if len(self._pending_ops_buffer) >= 20:
            self._pending_ops_buffer.pop(0)
            if self._pending_ops_keys:
                self._pending_ops_keys.pop(0)
        self._pending_ops_buffer.append(payload)
        self._pending_ops_keys.append(event_key)

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
        ev_key = self._event_key(source, alert_key, message)

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
                self._queue_ops_offline(payload, ev_key)

            bot_sent = False
            if business_blocking:
                tip = bot_message.strip() or (
                    f"🚨 [业务阻塞告警] {title.strip() or source}\n"
                    f"{message[:220]}"
                )
                bot_sent = await self._deliver_text(tip)
                if not bot_sent:
                    self._queue_offline(f"• [{now_uk}] [CRITICAL] {source}: {tip[:180]}", ev_key)

            sent = ops_sent or bot_sent
        else:
            icon = "🚨" if sev == "CRITICAL" else "⚠️"
            formatted = f"{icon} [运维告警 | {sev}] {source}\n时间: {now_uk} UK\n详情: {message}{sup_note}"
            sent = await self._deliver_text(formatted)
            if not sent:
                self._queue_offline(f"• [{now_uk}] [{sev}] {source}: {message[:180]}", ev_key)

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
        ev_key = self._event_key(source, alert_key, message)
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
                self._queue_ops_offline(payload, ev_key)
            await self._persist(
                severity="WARN",
                source=source,
                alert_key=alert_key,
                message=message,
                notified=ops_sent,
                suppressed_count=0,
            )
            return

        self._queue_offline(f"• [{now_uk}] [WARN] {source}: {message[:180]}", ev_key)
        await self._persist(
            severity="WARN",
            source=source,
            alert_key=alert_key,
            message=message,
            notified=False,
            suppressed_count=0,
        )

    @staticmethod
    def _format_recovered_event_line(ev: dict[str, Any]) -> str:
        msg = str(ev.get("message") or "").replace("\n", " ").strip()
        return f"• [{ev['created_at_uk']}] [{ev['severity']}] {ev['source']} ({ev['alert_key']}): {msg[:180]}"

    @classmethod
    def _chunk_recovered_events(
        cls,
        events: list[dict[str, Any]],
        *,
        max_chars: int = 3400,
    ) -> list[tuple[list[int], list[str]]]:
        """Group recovered DB events into message-sized chunks preserving chronological order."""
        chunks: list[tuple[list[int], list[str]]] = []
        cur_ids: list[int] = []
        cur_lines: list[str] = []
        cur_len = 0
        for ev in events:
            line = cls._format_recovered_event_line(ev)
            added_len = len(line) + (1 if cur_lines else 0)
            if cur_lines and cur_len + added_len > max_chars:
                chunks.append((cur_ids, cur_lines))
                cur_ids = []
                cur_lines = []
                cur_len = 0
            cur_ids.append(int(ev["id"]))
            cur_lines.append(line)
            cur_len += len(line) + (1 if len(cur_lines) > 1 else 0)
        if cur_ids:
            chunks.append((cur_ids, cur_lines))
        return chunks

    async def _recover_unnotified_db_events(
        self,
        *,
        flushed_keys: Counter[tuple[str, str, str]],
        in_flight_keys: set[tuple[str, str, str]],
        can_deliver: bool,
    ) -> bool:
        """Recover recent unnotified CRITICAL/WARN DB events after restart without re-sending cooldown-suppressed rows."""
        if self.db is None:
            return True
        try:
            events = await self.db.get_unnotified_ops_events(
                hours=24,
                severities=("CRITICAL", "WARN"),
                limit=200,
            )
        except Exception as exc:
            logger.warning("OpsNotifier failed loading unnotified ops events (%s: %s)", type(exc).__name__, exc)
            return False

        if not events:
            return True

        # Match just-flushed in-memory alerts newest-first so older pre-restart identical events still recover.
        already_delivered_ids: list[int] = []
        matched_ids: set[int] = set()
        remaining_flushed = Counter(flushed_keys)
        for ev in reversed(events):
            key = self._event_key(str(ev["source"]), str(ev["alert_key"]), str(ev["message"]))
            if remaining_flushed[key] > 0:
                remaining_flushed[key] -= 1
                eid = int(ev["id"])
                matched_ids.add(eid)
                already_delivered_ids.append(eid)

        if already_delivered_ids:
            try:
                await self.db.mark_ops_events_notified(already_delivered_ids)
            except Exception as exc:
                logger.warning("OpsNotifier failed marking flushed ops events (%s: %s)", type(exc).__name__, exc)

        # Filter out cooldown-suppressed records (suppressed_count > 0) and items still in memory buffers.
        db_only_events: list[dict[str, Any]] = []
        for ev in events:
            eid = int(ev["id"])
            if eid in matched_ids:
                continue
            if int(ev.get("suppressed_count") or 0) > 0:
                continue
            key = self._event_key(str(ev["source"]), str(ev["alert_key"]), str(ev["message"]))
            if key in in_flight_keys:
                continue
            db_only_events.append(ev)

        if not db_only_events:
            return True
        if not can_deliver:
            return False

        all_ok = True
        events_by_id = {int(ev["id"]): ev for ev in db_only_events}
        for chunk_ids, chunk_lines in self._chunk_recovered_events(db_only_events):
            body = "\n".join(chunk_lines)
            if self.ops_notify_enabled:
                has_crit = any(str(events_by_id[eid].get("severity", "")).upper() == "CRITICAL" for eid in chunk_ids)
                payload = self._build_ops_payload(
                    severity="CRITICAL" if has_crit else "WARN",
                    source="ops_recovery",
                    alert_key="ops_recovery_flush",
                    title=f"重启/断连恢复：补推 {len(chunk_ids)} 条未送达告警",
                    message=body,
                    cooldown_sec=0,
                )
                delivered = await self._deliver_ops_notify(payload)
            else:
                summary = "🔄 [系统恢复] 补推近 24h 未送达关键告警：\n" + body
                delivered = await self._deliver_text(summary)

            if not delivered:
                all_ok = False
                break
            try:
                await self.db.mark_ops_events_notified(chunk_ids)
            except Exception as exc:
                logger.warning("OpsNotifier failed marking recovered ops events (%s: %s)", type(exc).__name__, exc)
                all_ok = False
        return all_ok

    async def flush_pending(self) -> bool:
        """Flush buffered in-memory alerts and recover recent unnotified DB events after connectivity/restart."""
        all_ok = True
        ops_flush_ok = True
        tg_flush_ok = True
        flushed_keys: Counter[tuple[str, str, str]] = Counter()
        in_flight_keys: set[tuple[str, str, str]] = set()

        if self.ops_notify_enabled and self._pending_ops_buffer:
            remaining_ops: list[dict[str, Any]] = []
            remaining_ops_keys: list[tuple[str, str, str] | None] = []
            for idx, payload in enumerate(self._pending_ops_buffer):
                key = self._pending_ops_keys[idx] if idx < len(self._pending_ops_keys) else None
                if await self._deliver_ops_notify(payload):
                    if key is not None:
                        flushed_keys[key] += 1
                    continue
                remaining_ops.extend(self._pending_ops_buffer[idx:])
                if idx < len(self._pending_ops_keys):
                    remaining_ops_keys.extend(self._pending_ops_keys[idx:])
                ops_flush_ok = False
                all_ok = False
                break
            self._pending_ops_buffer = remaining_ops
            self._pending_ops_keys = remaining_ops_keys
            in_flight_keys.update(k for k in remaining_ops_keys if k is not None)

        if self._pending_buffer:
            summary = "🔄 [系统恢复] Telegram 通道已恢复，以下为断连期间离线汇总：\n" + "\n".join(self._pending_buffer)
            sent = await self._deliver_text(summary)
            if sent:
                for key in self._pending_tg_keys:
                    if key is not None:
                        flushed_keys[key] += 1
                self._pending_buffer.clear()
                self._pending_tg_keys.clear()
            else:
                tg_flush_ok = False
                all_ok = False
                in_flight_keys.update(k for k in self._pending_tg_keys if k is not None)

        can_deliver_db = ops_flush_ok if self.ops_notify_enabled else tg_flush_ok
        if not await self._recover_unnotified_db_events(
            flushed_keys=flushed_keys,
            in_flight_keys=in_flight_keys,
            can_deliver=can_deliver_db,
        ):
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
