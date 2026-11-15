"""Pure-httpx async Telegram Bot client (Long Polling, chunked messaging, status updates)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.bot_token = bot_token.strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _post(self, method: str, payload: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
        if not self.bot_token:
            return {"ok": False, "description": "TELEGRAM_BOT_TOKEN missing"}
        client = self._get_client()
        res = await client.post(f"{self.base_url}/{method}", json=payload, timeout=timeout)
        return res.json()

    async def delete_webhook(self, drop_pending_updates: bool = False) -> dict[str, Any]:
        return await self._post("deleteWebhook", {"drop_pending_updates": drop_pending_updates})

    async def get_updates(self, offset: int | None = None, timeout_s: int = 25) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout_s, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        data = await self._post("getUpdates", payload, timeout=float(timeout_s + 10))
        if not data.get("ok"):
            logger.warning("Telegram getUpdates error: %s", data)
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
            data = await self._post("sendMessage", payload, timeout=20.0)
            if not data.get("ok"):
                all_ok = False
                logger.warning("Telegram sendMessage failed: %s", data)
        return all_ok
