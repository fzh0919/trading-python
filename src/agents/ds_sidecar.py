"""DeepSeek V4.1 Flash sidecar helpers: gated availability, strict JSON extraction, hard-timeout calls.

All DeepSeek sidecar traffic funnels through this module so that:
- no sidecar call is attempted without a real DEEPSEEK_API_KEY (Gemini must never silently take over),
- every call has an independent short timeout and degrades to `None` instead of raising,
- structured JSON is extracted defensively and never trusted without Python-side revalidation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from src.agents.llm_client import LLMBackend
from src.config import AppConfig

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WARNING_PREFIXES = ("⚠️", "⚠")


def ds_helper_available(llm: LLMBackend | None, cfg: AppConfig) -> bool:
    """Return True only when the sidecar switch is on and a real DeepSeek credential exists."""
    if llm is None or not str(getattr(cfg.models, "ds_helper_model", "") or "").strip():
        return False
    key = getattr(llm, "deepseek_api_key", None)
    if key is None:
        key = getattr(cfg.env, "deepseek_api_key", "")
    return bool(str(key or "").strip())


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model reply, tolerating ```json fences and trailing prose."""
    raw = str(text or "").strip()
    if not raw or raw.startswith(_WARNING_PREFIXES):
        return None
    chunks = [match.group(1).strip() for match in _JSON_FENCE_RE.finditer(raw)]
    chunks.append(raw)
    for chunk in chunks:
        parsed = _decode_first_object(chunk)
        if parsed is not None:
            return parsed
    return None


def _decode_first_object(chunk: str) -> dict[str, Any] | None:
    start = chunk.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(chunk[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


async def call_ds_json(
    cfg: AppConfig,
    llm: LLMBackend | None,
    *,
    prompt: str,
    system_instruction: str = "",
    timeout_s: float | None = None,
    usage_label: str = "ds_sidecar",
) -> dict[str, Any] | None:
    """Call the DeepSeek sidecar model and return a parsed JSON object, or None on any failure."""
    if not ds_helper_available(llm, cfg):
        return None
    timeout = float(timeout_s if timeout_s is not None else cfg.models.ds_helper_timeout_seconds)
    timeout = max(0.05, timeout)
    try:
        raw = await asyncio.wait_for(
            llm.generate(
                model=cfg.models.ds_helper_model,
                prompt=prompt,
                system_instruction=system_instruction,
                thinking_level=cfg.models.ds_helper_thinking_level,
                temperature=0.1,
                usage_label=usage_label,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("DeepSeek sidecar %s timed out after %.1fs; degrading silently.", usage_label, timeout)
        return None
    except Exception as exc:  # noqa: BLE001 - sidecar failures must never break the caller
        logger.warning("DeepSeek sidecar %s failed (%s); degrading silently.", usage_label, exc)
        return None
    parsed = extract_json_object(raw)
    if parsed is None:
        logger.warning("DeepSeek sidecar %s returned non-JSON output; degrading silently.", usage_label)
    return parsed


def sanitize_plain_text(text: str, max_chars: int) -> str:
    """Strip Telegram-forbidden markdown markers and clamp to a hard character budget."""
    cleaned = str(text or "").replace("**", "").replace("#", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned[: max(0, int(max_chars))]
