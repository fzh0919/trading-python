"""Pure-httpx async Telegram Bot client (Long Polling, chunked messaging, status updates)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TelegramPollError(httpx.HTTPError):
    """Upstream getUpdates failure (e.g. HTTP 502) surfaced to the polling supervisor."""

    def __init__(self, method: str, error_code: int | None, description: str) -> None:
        self.method = method
        self.error_code = error_code
        self.description = description
        super().__init__(f"{method} upstream failure (error_code={error_code}): {description}")


def _parse_json_dict(res: httpx.Response) -> dict[str, Any] | None:
    try:
        parsed = res.json()
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def split_message_into_chunks(text: str, max_length: int = 3800) -> list[str]:
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        slice_part = remaining[:max_length]
        split_idx = slice_part.rfind("\n")
        if split_idx < int(max_length * 0.5):
            split_idx = max_length
        chunks.append(remaining[:split_idx])
        remaining = remaining[split_idx:].lstrip("\n")
    return chunks


class TelegramClient:
    def __init__(
        self,
        bot_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_retries: int = 2,
        base_backoff_s: float = 0.5,
        max_retry_wait_s: float = 30.0,
        delivery_max_retries: int = 4,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._max_retries = max(0, int(max_retries))
        self._delivery_max_retries = max(self._max_retries, int(delivery_max_retries))
        self._base_backoff_s = max(0.0, float(base_backoff_s))
        self._max_retry_wait_s = max(0.0, float(max_retry_wait_s))

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _compute_backoff_s(self, attempt: int) -> float:
        return min(self._base_backoff_s * (2**attempt), self._max_retry_wait_s)

    def _extract_retry_after_s(
        self,
        res: httpx.Response,
        data: dict[str, Any] | None,
        attempt: int,
    ) -> float:
        raw_retry_after: Any = None
        if isinstance(data, dict):
            params = data.get("parameters")
            if isinstance(params, dict):
                raw_retry_after = params.get("retry_after")
        if raw_retry_after is None:
            raw_retry_after = res.headers.get("Retry-After")
        if raw_retry_after is not None:
            try:
                parsed = float(str(raw_retry_after).strip())
                if parsed >= 0:
                    return min(parsed, self._max_retry_wait_s)
            except (TypeError, ValueError):
                pass
        return self._compute_backoff_s(attempt)

    async def _post(
        self,
        method: str,
        payload: dict[str, Any],
        timeout: float = 20.0,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        if not self.bot_token:
            return {"ok": False, "description": "TELEGRAM_BOT_TOKEN missing"}
        client = self._get_client()
        url = f"{self.base_url}/{method}"
        retry_budget = self._max_retries if max_retries is None else max(0, int(max_retries))

        for attempt in range(retry_budget + 1):
            try:
                res = await client.post(url, json=payload, timeout=timeout)
                data = _parse_json_dict(res)

                is_rate_limited = res.status_code == 429 or (
                    isinstance(data, dict) and data.get("error_code") == 429
                )
                if is_rate_limited:
                    if attempt >= retry_budget:
                        return data or {
                            "ok": False,
                            "error_code": 429,
                            "description": (res.text or "Too Many Requests")[:200],
                        }
                    wait_s = self._extract_retry_after_s(res, data, attempt)
                    logger.warning("Telegram 429 rate-limited on %s, retrying in %.2fs", method, wait_s)
                    await asyncio.sleep(wait_s)
                    continue

                if 500 <= res.status_code < 600:
                    if attempt >= retry_budget:
                        return data or {
                            "ok": False,
                            "error_code": res.status_code,
                            "description": (res.text or f"HTTP {res.status_code}")[:200],
                        }
                    wait_s = self._compute_backoff_s(attempt)
                    logger.warning("Telegram HTTP %d on %s, retrying in %.2fs", res.status_code, method, wait_s)
                    await asyncio.sleep(wait_s)
                    continue

                if data is not None:
                    return data
                return {
                    "ok": False,
                    "error_code": res.status_code,
                    "description": (res.text or f"Invalid JSON response (HTTP {res.status_code})")[:200],
                }
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise
                wait_s = self._compute_backoff_s(attempt)
                safe_msg = str(exc).replace(self.bot_token, "***") if self.bot_token else str(exc)
                logger.warning(
                    "Telegram %s transient error (%s: %s), retrying in %.2fs",
                    method,
                    type(exc).__name__,
                    safe_msg,
                    wait_s,
                )
                await asyncio.sleep(wait_s)

        return {"ok": False, "description": "Telegram request exhausted retries"}

    async def delete_webhook(self, drop_pending_updates: bool = False) -> dict[str, Any]:
        return await self._post("deleteWebhook", {"drop_pending_updates": drop_pending_updates})

    @staticmethod
    def _is_poll_transport_failure(data: dict[str, Any]) -> bool:
        """True when a non-ok getUpdates payload means the transport/upstream is broken (not a client-side rejection)."""
        raw_code = data.get("error_code")
        if raw_code is None:
            return True
        try:
            code = int(raw_code)
        except (TypeError, ValueError):
            return True
        return code >= 500 or code in (401, 403)

    async def get_updates(self, offset: int | None = None, timeout_s: int = 25) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout_s, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        data = await self._post("getUpdates", payload, timeout=float(timeout_s + 10))
        if not data.get("ok"):
            logger.warning("Telegram getUpdates error: %s", data)
            if self._is_poll_transport_failure(data):
                raise TelegramPollError(
                    "getUpdates",
                    data.get("error_code"),
                    str(data.get("description") or "unknown upstream failure"),
                )
            return []
        return list(data.get("result") or [])

    async def send_typing(self, chat_id: str | int) -> None:
        try:
            await self._post("sendChatAction", {"chat_id": str(chat_id), "action": "typing"}, timeout=8.0)
        except Exception:
            pass

    async def send_status(self, chat_id: str | int, text: str) -> int | None:
        try:
            data = await self._post("sendMessage", {"chat_id": str(chat_id), "text": text})
            if data.get("ok") and data.get("result"):
                return int(data["result"]["message_id"])
        except Exception as exc:
            logger.debug("Failed sending status: %s", exc)
        return None

    async def edit_status(self, chat_id: str | int, message_id: int | None, text: str) -> None:
        if not message_id:
            return
        try:
            await self._post(
                "editMessageText",
                {"chat_id": str(chat_id), "message_id": message_id, "text": text},
            )
        except Exception:
            pass

    async def set_my_commands(self, commands: list[dict[str, str]]) -> dict[str, Any]:
        return await self._post("setMyCommands", {"commands": commands}, timeout=12.0)

    async def send_message(
        self,
        chat_id: str | int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        chunks = split_message_into_chunks(text, max_length=3800)
        if not chunks:
            return True
        all_ok = True
        for idx, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": str(chat_id),
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if reply_markup is not None and idx == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            data = await self._post(
                "sendMessage",
                payload,
                timeout=20.0,
                max_retries=self._delivery_max_retries,
            )
            if not data.get("ok"):
                all_ok = False
                logger.warning("Telegram sendMessage failed: %s", data)
        return all_ok
