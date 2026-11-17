"""Outbound text redaction (defence in depth for logs, alerts and /diag output).

Single source of truth for masking credential-looking fragments before they reach
Telegram, docker logs or a third-party ops gateway. Never used on values that must
stay numerically exact — it only rewrites query params and Authorization headers.
"""

from __future__ import annotations

import re

_MASK = "***"
_SECRET_QUERY_RE = re.compile(
    r"(?i)\b(token|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|pwd|sig|signature)=([^&\s\"'<>]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]{12,}=*")
_BASIC_RE = re.compile(r"(?i)\b(basic)\s+[A-Za-z0-9+/]{16,}=*")


def redact_secrets(text: str) -> str:
    """Mask ``token=`` / ``api_key=`` query params and ``Bearer`` / ``Basic`` header values."""
    raw = str(text or "")
    if not raw:
        return ""
    redacted = _SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}={_MASK}", raw)
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)} {_MASK}", redacted)
    redacted = _BASIC_RE.sub(lambda m: f"{m.group(1)} {_MASK}", redacted)
    return redacted
