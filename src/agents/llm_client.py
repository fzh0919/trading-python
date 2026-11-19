"""Async multi-provider LLM client (Gemini + DeepSeek) with ThinkingConfig, cross-vendor failover, Ops alerts, and telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

import httpx
from google import genai
from google.genai import types

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
UsageRecorder = Callable[[dict[str, Any]], Awaitable[None]]
UsageSummaryProvider = Callable[[], Awaitable[dict[str, Any]]]
_TRUNCATED_SUFFIX = "...(truncated)"


def normalize_thinking_level(model: str, thinking_level: str) -> str:
    """Clamp requested thinking_level to values supported by the target Gemini model.

    Prevents invalid levels (e.g. MINIMAL on Gemini 4) from triggering a 400 fallback
    that strips thinking_config and silently reverts to the model's high default.
    """
    lvl = (thinking_level or "").strip().upper()
    if lvl not in {"MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"}:
        return ""

    m = (model or "").strip().lower()
    if m.startswith("gemini-4-"):
        if lvl == "MINIMAL":
            return "LOW"
        if lvl == "XHIGH" and "hydrogen" not in m and "neon" not in m:
            return "HIGH"
        return lvl

    if "3.8-flash" in m or "3.7-flash" in m or "3.6-flash" in m or "3.1-pro" in m:
        if lvl == "MINIMAL":
            return "LOW"
        if lvl == "XHIGH":
            return "HIGH"
        return lvl

    if lvl == "XHIGH":
        return "HIGH"
    return lvl


def _is_deepseek_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("deepseek-")


def _gemini_schema_to_openai(schema: Any) -> dict[str, Any]:
    """Recursively convert Gemini Schema / dict (uppercase types) into OpenAI JSON Schema (lowercase types)."""
    if schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(schema, "model_dump"):
        raw = schema.model_dump(exclude_none=True)
    elif isinstance(schema, dict):
        raw = dict(schema)
    else:
        return {"type": "object", "properties": {}}

    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key == "type" and isinstance(val, str):
            out["type"] = val.lower()
        elif key == "properties" and isinstance(val, dict):
            out["properties"] = {pk: _gemini_schema_to_openai(pv) for pk, pv in val.items()}
        elif key == "items":
            out["items"] = _gemini_schema_to_openai(val)
        elif key in {"description", "required", "enum", "default", "title"}:
            out[key] = val
    if "type" not in out:
        out["type"] = "object"
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def _function_declarations_to_openai_tools(
    declarations: list[types.FunctionDeclaration],
) -> list[dict[str, Any]]:
    tools_out: list[dict[str, Any]] = []
    for decl in declarations:
        params_src = getattr(decl, "parameters_json_schema", None) or getattr(decl, "parameters", None)
        tools_out.append(
            {
                "type": "function",
                "function": {
                    "name": decl.name or "",
                    "description": decl.description or "",
                    "parameters": _gemini_schema_to_openai(params_src),
                },
            }
        )
    return tools_out


def _build_deepseek_thinking_params(thinking_level: str) -> tuple[str, dict[str, Any], bool]:
    """Map unified ThinkingLevelLiteral into DeepSeek thinking payload and normalized label."""
    lvl = (thinking_level or "HIGH").strip().upper()
    if lvl not in {"MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"}:
        lvl = "HIGH"
    if lvl == "MINIMAL":
        return "MINIMAL", {"thinking": {"type": "disabled"}}, False
    if lvl == "XHIGH":
        lvl = "HIGH"
    effort = lvl.lower()
    return lvl, {"thinking": {"type": "enabled"}, "reasoning_effort": effort}, True

_TRANSIENT_KEYWORDS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "rate limit",
    "quota",
    "unavailable",
    "overloaded",
    "deadline",
    "timed out",
    "timeout",
    "connection",
    "temporarily",
)


# Our own `asyncio.wait_for` ceiling (not a transport error): a hard timeout usually means the
# model is saturated, so retrying it costs minutes before failover can even start.
_HARD_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)

# Sentinel strings returned by `generate()` when every candidate model failed. They are meant for
# end-user surfaces (Telegram), never as analyst/auditor content inside an internal prompt.
DEGRADED_LLM_MARKERS: tuple[str, ...] = (
    "⚠️ AI 模型服务当前全部不可用",
    "⚠️ AI 模型密钥未配置",
    "⚠️ GEMINI_API_KEY 未配置",
)


def is_degraded_llm_output(text: str | None) -> bool:
    """True when `generate()` returned its model-unavailable sentinel instead of real content."""
    stripped = (text or "").strip()
    return any(stripped.startswith(marker) for marker in DEGRADED_LLM_MARKERS)


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= len(_TRUNCATED_SUFFIX):
        return text[:limit]
    return text[: limit - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX


def _json_length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _compact_item_strings(item: Any, per_str_limit: int) -> Any:
    if isinstance(item, str):
        return _truncate_text(item, per_str_limit)
    if isinstance(item, dict):
        return {
            k: _truncate_text(v, per_str_limit) if isinstance(v, str) else v
            for k, v in item.items()
        }
    return item


def _compact_tool_result(result: Any, max_chars: int) -> Any:
    """Compact oversized tool results to prevent multi-round input token snowballing."""
    if max_chars <= 0:
        return result
    raw_json = json.dumps(result, ensure_ascii=False, default=str)
    if len(raw_json) <= max_chars:
        return result

    if isinstance(result, dict):
        compacted: dict[str, Any] = dict(result)
        per_item_str_limit = max(80, min(280, max_chars // 6))
        for key, val in list(compacted.items()):
            if isinstance(val, list):
                trimmed_list = [_compact_item_strings(item, per_item_str_limit) for item in val]
                compacted[key] = trimmed_list
                while trimmed_list and _json_length(compacted) > max_chars:
                    trimmed_list.pop()
                    compacted[key] = trimmed_list

        compacted["_truncated"] = True
        if _json_length(compacted) <= max_chars:
            return compacted

        str_keys = sorted(
            (k for k, v in compacted.items() if isinstance(v, str) and len(v) > 40),
            key=lambda k: len(compacted[k]),
            reverse=True,
        )
        for key in str_keys:
            if _json_length(compacted) <= max_chars:
                break
            excess = _json_length(compacted) - max_chars
            cur_len = len(compacted[key])
            target_len = max(40, cur_len - excess - len(_TRUNCATED_SUFFIX))
            compacted[key] = _truncate_text(compacted[key], target_len)

        if _json_length(compacted) <= max_chars:
            return compacted

        preview_budget = max(24, max_chars - 44)
        fallback = {"_truncated": True, "preview": _truncate_text(raw_json, preview_budget)}
        if _json_length(fallback) <= max_chars:
            return fallback
        return _truncate_text(raw_json, max_chars)

    if isinstance(result, list):
        trimmed = [_compact_item_strings(item, max(60, max_chars // 4)) for item in result]
        while trimmed and _json_length(trimmed) > max_chars:
            trimmed.pop()
        if trimmed and _json_length(trimmed) <= max_chars:
            return trimmed
        return {"_truncated": True, "preview": _truncate_text(raw_json, max(24, max_chars - 44))}

    return _truncate_text(str(result), max_chars)


class LLMBackend(Protocol):
    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str = "",
        thinking_level: str = "MEDIUM",
        tools: list[types.FunctionDeclaration] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        use_google_search: bool = False,
        response_schema: Any = None,
        temperature: float = 0.25,
        max_tool_rounds: int = 4,
        usage_label: str = "",
        round_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str: ...


class GeminiClient:
    """Multi-provider async LLM client (Gemini primary/search + DeepSeek hybrid & cross-vendor failover)."""

    def __init__(
        self,
        api_key: str,
        usage_recorder: UsageRecorder | None = None,
        ops_notifier: OpsNotifier | None = None,
        fallback_model: str = "gemini-4-hydrogen",
        fallback_thinking_level: str = "HIGH",
        max_transient_retries: int = 2,
        retry_base_delay_s: float = 1.5,
        *,
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com",
        gemini_fallback_model: str = "gemini-4-hydrogen",
        gemini_fallback_thinking_level: str = "MEDIUM",
        deepseek_fallback_model: str = "deepseek-flash",
        deepseek_fallback_thinking_level: str = "HIGH",
        prefer_deepseek_over_flash_lite: bool = True,
        deepseek_round_timeout_seconds: float = 60.0,
        deepseek_max_retries: int = 1,
        round_timeout_seconds: float = 120.0,
        timeout_max_retries: int = 0,
        tool_timeout_seconds: float = 60.0,
        max_concurrent_tools: int = 6,
        max_tool_result_chars: int = 4000,
        usage_summary_provider: UsageSummaryProvider | None = None,
        search_free_quota_monthly: int = 5000,
        search_warn_pct: float = 85.0,
    ) -> None:
        self.api_key = api_key.strip()
        self._client: genai.Client | None = genai.Client(api_key=self.api_key) if self.api_key else None
        self.deepseek_api_key = (deepseek_api_key or "").strip()
        self.deepseek_base_url = (deepseek_base_url or "https://api.deepseek.com").strip().rstrip("/")
        self._http: httpx.AsyncClient | None = None
        self._usage_recorder = usage_recorder
        self._ops = ops_notifier
        self.fallback_model = fallback_model.strip() or "gemini-4-hydrogen"
        self.fallback_thinking_level = fallback_thinking_level.strip() or "HIGH"
        self.gemini_fallback_model = (gemini_fallback_model or "gemini-4-hydrogen").strip()
        self.gemini_fallback_thinking_level = (gemini_fallback_thinking_level or "MEDIUM").strip()
        self.deepseek_fallback_model = (deepseek_fallback_model or "deepseek-flash").strip()
        self.deepseek_fallback_thinking_level = (deepseek_fallback_thinking_level or "HIGH").strip()
        self.prefer_deepseek_over_flash_lite = bool(prefer_deepseek_over_flash_lite)
        self.max_transient_retries = max(0, int(max_transient_retries))
        self.timeout_max_retries = max(0, min(int(timeout_max_retries), self.max_transient_retries))
        self.deepseek_max_retries = max(0, int(deepseek_max_retries))
        self.retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self.round_timeout_seconds = max(0.0, float(round_timeout_seconds))
        self.deepseek_round_timeout_seconds = max(0.0, float(deepseek_round_timeout_seconds))
        self.tool_timeout_seconds = max(0.0, float(tool_timeout_seconds))
        self.max_concurrent_tools = max(1, int(max_concurrent_tools))
        self.max_tool_result_chars = max(0, int(max_tool_result_chars))
        self._usage_summary_provider = usage_summary_provider
        self.search_free_quota_monthly = max(0, int(search_free_quota_monthly))
        self.search_warn_pct = min(100.0, max(1.0, float(search_warn_pct)))

    def _ensure_http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            timeout = max(self.deepseek_round_timeout_seconds, 15.0)
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def set_usage_recorder(self, recorder: UsageRecorder | None) -> None:
        self._usage_recorder = recorder

    def set_ops_notifier(self, ops_notifier: OpsNotifier | None) -> None:
        self._ops = ops_notifier

    def set_usage_summary_provider(self, provider: UsageSummaryProvider | None) -> None:
        self._usage_summary_provider = provider

    @staticmethod
    def _build_thinking_config(model: str, thinking_level: str) -> tuple[types.ThinkingConfig | None, str]:
        lvl_str = normalize_thinking_level(model, thinking_level)
        if not lvl_str:
            return None, ""
        if lvl_str in types.ThinkingLevel.__members__:
            return types.ThinkingConfig(thinking_level=types.ThinkingLevel(lvl_str)), lvl_str
        # Support newer levels (e.g. XHIGH -> "xhigh") before SDK enum updates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return types.ThinkingConfig(thinking_level=lvl_str.lower()), lvl_str  # type: ignore[arg-type]

    def _candidate_chain(
        self,
        primary_model: str,
        primary_thinking: str,
        *,
        use_google_search: bool = False,
    ) -> list[tuple[str, str]]:
        raw_chain: list[tuple[str, str]] = [(primary_model, primary_thinking)]
        is_ds = _is_deepseek_model(primary_model)

        if is_ds:
            if self._client is not None:
                raw_chain.append((self.gemini_fallback_model, self.gemini_fallback_thinking_level))
                if self.gemini_fallback_model != "gemini-3.5-flash-lite":
                    raw_chain.append(("gemini-3.5-flash-lite", "LOW"))
        else:
            if self.fallback_model and self.fallback_model != primary_model:
                fb_think = "LOW" if (use_google_search and "flash-lite" not in self.fallback_model.lower()) else self.fallback_thinking_level
                raw_chain.append((self.fallback_model, fb_think))
            elif any(tag in (primary_model or "").lower() for tag in ("argon", "carbon", "neon")):
                raw_chain.append(("gemini-4-hydrogen", "HIGH"))
            elif "flash-lite" not in (primary_model or "").lower():
                # Avoid silently upgrading Hydrogen/Pro failures to more expensive Argon/Carbon.
                # v5.4.2: for non-search calls a configured DeepSeek key means a strong cross-vendor
                # reasoning fallback (HIGH + thinking) is available; prefer it over the weak
                # same-vendor 3.5-flash-lite. Search calls still fall back to flash-lite only.
                prefer_deepseek = (
                    self.prefer_deepseek_over_flash_lite
                    and not use_google_search
                    and bool(self.deepseek_api_key)
                    and self.deepseek_fallback_model != primary_model
                )
                if not prefer_deepseek:
                    raw_chain.append(("gemini-3.5-flash-lite", "LOW"))
            elif use_google_search:
                raw_chain.append(("gemini-4-hydrogen", "LOW"))

            if (
                not use_google_search
                and self.deepseek_api_key
                and self.deepseek_fallback_model
                and self.deepseek_fallback_model != primary_model
            ):
                raw_chain.append((self.deepseek_fallback_model, self.deepseek_fallback_thinking_level))

        deduped: list[tuple[str, str]] = []
        seen_models: set[str] = set()
        for cand_m, cand_t in raw_chain:
            key = (cand_m or "").strip()
            if not key or key in seen_models:
                continue
            seen_models.add(key)
            deduped.append((key, cand_t))
        return deduped

    async def _execute_tools_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        model_label: str,
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self.max_concurrent_tools)

        async def _run_one(call_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
            logger.info("LLM tool call [%s]: %s(%s)", model_label, call_name, call_args)
            try:
                async with sem:
                    coro = tool_executor(call_name, call_args)
                    if self.tool_timeout_seconds > 0:
                        result = await asyncio.wait_for(coro, timeout=self.tool_timeout_seconds)
                    else:
                        result = await coro
            except (TimeoutError, asyncio.TimeoutError):
                result = {"error": f"tool timeout after {self.tool_timeout_seconds:.1f}s"}
            except Exception as exc:
                result = {"error": str(exc)}
            compact = _compact_tool_result(result, self.max_tool_result_chars)
            return compact if isinstance(compact, dict) else {"result": compact}

        return list(await asyncio.gather(*(_run_one(n, a) for n, a in calls)))

    @staticmethod
    def _extract_search_query_count(response: Any) -> int:
        cands = getattr(response, "candidates", None) or []
        count = 0
        for cand in cands:
            gm = getattr(cand, "grounding_metadata", None)
            if gm is None:
                continue
            queries = getattr(gm, "web_search_queries", None) or []
            count += len(queries)
        return count

    async def _record_usage(
        self,
        *,
        usage_label: str,
        model: str,
        thinking_level: str,
        api_rounds: int,
        prompt_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        thought_tokens: int,
        total_tokens: int,
        latency_ms: int,
        success: bool,
        search_queries: int = 0,
    ) -> None:
        cache_pct = (cached_tokens / prompt_tokens * 100.0) if prompt_tokens > 0 else 0.0
        logger.info(
            "LLM usage [%s] model=%s think=%s rounds=%d in=%d(cached=%d,%.0f%%) out=%d thought=%d search=%d latency=%dms ok=%s",
            usage_label or "unknown",
            model,
            thinking_level or "DEFAULT",
            api_rounds,
            prompt_tokens,
            cached_tokens,
            cache_pct,
            output_tokens,
            thought_tokens,
            search_queries,
            latency_ms,
            success,
        )
        if self._usage_recorder is not None:
            try:
                await self._usage_recorder(
                    {
                        "call_label": usage_label or "unknown",
                        "model": model,
                        "thinking_level": thinking_level or "DEFAULT",
                        "api_rounds": api_rounds,
                        "prompt_tokens": prompt_tokens,
                        "cached_tokens": cached_tokens,
                        "output_tokens": output_tokens,
                        "thought_tokens": thought_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": 0.0,
                        "latency_ms": latency_ms,
                        "success": success,
                        "search_queries": search_queries,
                    }
                )
            except Exception as exc:
                logger.debug("Failed to persist LLM usage record: %s", exc)

        if search_queries > 0:
            await self._check_search_quota_alert()

    async def _check_search_quota_alert(self) -> None:
        """Alert via OpsNotifier when UK natural-month Google Search queries approach/exceed free quota."""
        if self._ops is None or self._usage_summary_provider is None or self.search_free_quota_monthly <= 0:
            return
        try:
            summary = await self._usage_summary_provider()
        except Exception as exc:
            logger.debug("Failed to query usage summary for Search quota check: %s", exc)
            return

        this_month = summary.get("this_month") or {}
        monthly_queries = int(this_month.get("search_queries", 0) or 0)
        warn_at = int(self.search_free_quota_monthly * self.search_warn_pct / 100.0)
        if monthly_queries >= self.search_free_quota_monthly:
            await self._emit_ops(
                severity="WARN",
                source="llm:search_quota",
                alert_key="google_search_quota_exceeded",
                message=(
                    f"本月（英国自然月）Google Search 查询量已达 {monthly_queries:,} / "
                    f"{self.search_free_quota_monthly:,} 免费额度上限，超出部分将按量计费。"
                ),
            )
        elif warn_at > 0 and monthly_queries >= warn_at:
            pct = monthly_queries / self.search_free_quota_monthly * 100.0
            await self._emit_ops(
                severity="WARN",
                source="llm:search_quota",
                alert_key="google_search_quota_warning",
                message=(
                    f"本月（英国自然月）Google Search 查询量已达 {monthly_queries:,} / "
                    f"{self.search_free_quota_monthly:,}（{pct:.1f}%，预警线 {self.search_warn_pct:.0f}%）。"
                ),
            )

    async def _emit_ops(self, *, severity: str, source: str, alert_key: str, message: str) -> None:
        if self._ops is None:
            return
        await self._ops.notify(severity=severity, source=source, alert_key=alert_key, message=message)

    async def _call_single_round(
        self,
        *,
        model: str,
        contents: list[Any],
        make_config: Callable[[types.ThinkingConfig | None], types.GenerateContentConfig],
        thinking_cfg: types.ThinkingConfig | None,
        effective_level: str,
        usage_label: str = "",
        round_timeout_seconds: float | None = None,
    ) -> tuple[Any, types.ThinkingConfig | None, str]:
        assert self._client is not None
        effective_timeout = (
            float(round_timeout_seconds)
            if round_timeout_seconds is not None and round_timeout_seconds > 0
            else self.round_timeout_seconds
        )

        async def _invoke_sdk(cfg_obj: types.GenerateContentConfig) -> Any:
            coro = self._client.aio.models.generate_content(  # type: ignore[union-attr]
                model=model,
                contents=contents,
                config=cfg_obj,
            )
            if effective_timeout > 0:
                return await asyncio.wait_for(coro, timeout=effective_timeout)
            return await coro

        cfg = make_config(thinking_cfg)
        for attempt in range(self.max_transient_retries + 1):
            try:
                resp = await _invoke_sdk(cfg)
                return resp, thinking_cfg, effective_level
            except Exception as exc:
                err_msg = str(exc) or type(exc).__name__
                if thinking_cfg is not None and ("thinking" in err_msg.lower() or "400" in err_msg):
                    if effective_level == "XHIGH":
                        logger.warning("Retrying %s with HIGH instead of XHIGH due to: %s", model, err_msg)
                        thinking_cfg, effective_level = self._build_thinking_config(model, "HIGH")
                    else:
                        logger.warning("Retrying %s without thinking_config due to: %s", model, err_msg)
                        thinking_cfg, effective_level = None, "DEFAULT"
                    cfg = make_config(thinking_cfg)
                    resp = await _invoke_sdk(cfg)
                    return resp, thinking_cfg, effective_level

                if _is_transient_llm_error(exc):
                    is_timeout = isinstance(exc, _HARD_TIMEOUT_ERRORS)
                    retry_budget = self.timeout_max_retries if is_timeout else self.max_transient_retries
                    if attempt < retry_budget:
                        delay = self.retry_base_delay_s * (2**attempt)
                        logger.warning(
                            "Transient Gemini error on %s (attempt %d/%d): %s; retrying in %.1fs",
                            model,
                            attempt + 1,
                            retry_budget + 1,
                            err_msg[:180],
                            delay,
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                        continue
                    if is_timeout:
                        logger.warning(
                            "Hard timeout on %s after %.0fs (%s); skipping further retries to fail over sooner",
                            model,
                            effective_timeout,
                            usage_label or "unknown",
                        )
                raise
        raise RuntimeError(f"Exhausted retries for {model}")

    async def _run_model_once(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        thinking_level: str,
        configured_tools: list[types.Tool],
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None,
        response_schema: Any,
        temperature: float,
        max_tool_rounds: int,
        usage_label: str,
        round_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str | None, str]:
        thinking_cfg, effective_level = self._build_thinking_config(model, thinking_level)

        def _make_config(t_cfg: types.ThinkingConfig | None) -> types.GenerateContentConfig:
            kwargs: dict[str, Any] = {"temperature": temperature}
            if max_output_tokens and max_output_tokens > 0:
                kwargs["max_output_tokens"] = int(max_output_tokens)
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            if configured_tools:
                kwargs["tools"] = configured_tools
            if response_schema is not None:
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = response_schema
            if t_cfg is not None:
                kwargs["thinking_config"] = t_cfg
            return types.GenerateContentConfig(**kwargs)

        contents: list[Any] = [prompt]
        t0 = time.monotonic()
        api_rounds = 0
        prompt_tokens = 0
        cached_tokens = 0
        output_tokens = 0
        thought_tokens = 0
        total_tokens = 0
        search_queries = 0
        success = False
        failure_reason = ""

        try:
            for round_idx in range(max_tool_rounds + 1):
                response, thinking_cfg, effective_level = await self._call_single_round(
                    model=model,
                    contents=contents,
                    make_config=_make_config,
                    thinking_cfg=thinking_cfg,
                    effective_level=effective_level,
                    usage_label=usage_label,
                    round_timeout_seconds=round_timeout_seconds,
                )
                api_rounds += 1
                search_queries += self._extract_search_query_count(response)
                meta = getattr(response, "usage_metadata", None)
                if meta is not None:
                    prompt_tokens += int(getattr(meta, "prompt_token_count", 0) or 0) + int(
                        getattr(meta, "tool_use_prompt_token_count", 0) or 0
                    )
                    cached_tokens += int(getattr(meta, "cached_content_token_count", 0) or 0)
                    output_tokens += int(getattr(meta, "candidates_token_count", 0) or 0)
                    thought_tokens += int(getattr(meta, "thoughts_token_count", 0) or 0)
                    total_tokens += int(getattr(meta, "total_token_count", 0) or 0)

                fn_calls = getattr(response, "function_calls", None) or []
                if not fn_calls or tool_executor is None or round_idx == max_tool_rounds:
                    text = (getattr(response, "text", None) or "").strip()
                    if text:
                        success = True
                        return text, ""
                    cands = getattr(response, "candidates", None) or []
                    fin_raw = getattr(cands[0], "finish_reason", None) if cands else None
                    fin_str = getattr(fin_raw, "name", None) or (str(fin_raw) if fin_raw is not None else "unknown")
                    failure_reason = (
                        f"模型返回空文本（finish_reason={fin_str}, out={output_tokens}, thought={thought_tokens}）"
                    )
                    return None, failure_reason

                if response.candidates and response.candidates[0].content:
                    contents.append(response.candidates[0].content)

                call_specs = [(c.name or "", dict(c.args or {})) for c in fn_calls]
                compacted_results = await self._execute_tools_batch(call_specs, tool_executor, model)
                fn_response_parts = [
                    types.Part.from_function_response(name=name, response={"result": res})
                    for (name, _), res in zip(call_specs, compacted_results)
                ]
                contents.append(types.Content(role="user", parts=fn_response_parts))

            failure_reason = "工具调用轮次超限"
            return None, failure_reason
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {str(exc)[:220]}"
            logger.warning("Gemini model %s failed [%s]: %s", model, usage_label or "unknown", failure_reason)
            return None, failure_reason
        finally:
            if api_rounds > 0 or not success:
                latency_ms = int((time.monotonic() - t0) * 1000)
                await self._record_usage(
                    usage_label=usage_label,
                    model=model,
                    thinking_level=effective_level,
                    api_rounds=api_rounds,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    output_tokens=output_tokens,
                    thought_tokens=thought_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    search_queries=search_queries,
                )

    async def _post_deepseek_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        http = self._ensure_http_client()
        url = f"{self.deepseek_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        resp = await http.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:220]}")
        return dict(resp.json())

    async def _call_deepseek_single_round(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        openai_tools: list[dict[str, Any]],
        response_schema: Any,
        temperature: float,
        thinking_level: str,
        round_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        eff_level, thinking_extra, thinking_enabled = _build_deepseek_thinking_params(thinking_level)
        effective_timeout = (
            float(round_timeout_seconds)
            if round_timeout_seconds is not None and round_timeout_seconds > 0
            else self.deepseek_round_timeout_seconds
        )

        async def _post(payload: dict[str, Any]) -> dict[str, Any]:
            coro = self._post_deepseek_completion(payload)
            if effective_timeout > 0:
                return await asyncio.wait_for(coro, timeout=effective_timeout)
            return await coro

        def _build_payload(extra_think: dict[str, Any], include_temp: bool) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if max_output_tokens and max_output_tokens > 0:
                body["max_tokens"] = int(max_output_tokens)
            if extra_think:
                body.update(extra_think)
            if include_temp:
                body["temperature"] = temperature
            if openai_tools:
                body["tools"] = openai_tools
                body["tool_choice"] = "auto"
            if response_schema is not None:
                body["response_format"] = {"type": "json_object"}
            return body

        payload = _build_payload(thinking_extra, include_temp=not thinking_enabled)
        for attempt in range(self.deepseek_max_retries + 1):
            try:
                data = await _post(payload)
                return data, eff_level
            except Exception as exc:
                err_msg = str(exc) or type(exc).__name__
                if thinking_extra and ("400" in err_msg or "thinking" in err_msg.lower() or "reasoning_effort" in err_msg.lower()):
                    logger.warning("Retrying DeepSeek %s without reasoning_effort due to: %s", model, err_msg[:180])
                    thinking_extra = {}
                    eff_level = "DEFAULT"
                    payload = _build_payload({}, include_temp=True)
                    data = await _post(payload)
                    return data, eff_level

                if attempt < self.deepseek_max_retries and _is_transient_llm_error(exc):
                    delay = self.retry_base_delay_s * (2**attempt)
                    logger.warning(
                        "Transient DeepSeek error on %s (attempt %d/%d): %s; retrying in %.1fs",
                        model,
                        attempt + 1,
                        self.deepseek_max_retries + 1,
                        err_msg[:180],
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"Exhausted DeepSeek retries for {model}")

    async def _run_deepseek_once(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        thinking_level: str,
        tools: list[types.FunctionDeclaration] | None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None,
        response_schema: Any,
        temperature: float,
        max_tool_rounds: int,
        usage_label: str,
        round_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str | None, str]:
        sys_text = system_instruction or ""
        if response_schema is not None:
            try:
                schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
            except Exception:
                schema_json = "{}"
            schema_contract = (
                "\n\n[JSON OUTPUT CONTRACT]\n"
                "Output strictly a single valid JSON object matching this JSON Schema (no markdown fences, no commentary):\n"
                f"{schema_json}"
            )
            sys_text = (sys_text + schema_contract).strip()

        messages: list[dict[str, Any]] = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        openai_tools = _function_declarations_to_openai_tools(tools) if (tools and response_schema is None) else []
        t0 = time.monotonic()
        api_rounds = 0
        prompt_tokens = 0
        cached_tokens = 0
        output_tokens = 0
        thought_tokens = 0
        total_tokens = 0
        effective_level = thinking_level or "HIGH"
        success = False
        failure_reason = ""

        try:
            for round_idx in range(max_tool_rounds + 1):
                data, effective_level = await self._call_deepseek_single_round(
                    model=model,
                    messages=messages,
                    openai_tools=openai_tools,
                    response_schema=response_schema,
                    temperature=temperature,
                    thinking_level=effective_level,
                    round_timeout_seconds=round_timeout_seconds,
                    max_output_tokens=max_output_tokens,
                )
                api_rounds += 1

                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens") or 0)
                c_hit = int(
                    usage.get("prompt_cache_hit_tokens")
                    or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                )
                comp_tok = int(usage.get("completion_tokens") or 0)
                reas_tok = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                out_tok = max(0, comp_tok - reas_tok)
                tot_tok = int(usage.get("total_tokens") or (p_tok + comp_tok))
                prompt_tokens += p_tok
                cached_tokens += c_hit
                output_tokens += out_tok
                thought_tokens += reas_tok
                total_tokens += tot_tok

                choices = data.get("choices") or []
                msg = (choices[0].get("message") if choices else None) or {}
                finish_reason = str((choices[0].get("finish_reason") if choices else None) or "unknown")
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls or tool_executor is None or round_idx == max_tool_rounds:
                    text = str(msg.get("content") or "").strip()
                    if text:
                        success = True
                        return text, ""
                    failure_reason = (
                        f"DeepSeek 返回空文本（finish_reason={finish_reason}, out={output_tokens}, thought={thought_tokens}）"
                    )
                    return None, failure_reason

                assistant_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                }
                if msg.get("reasoning_content"):
                    assistant_entry["reasoning_content"] = msg["reasoning_content"]
                messages.append(assistant_entry)

                call_specs: list[tuple[str, dict[str, Any]]] = []
                call_ids: list[str] = []
                for idx_tc, tc in enumerate(tool_calls):
                    fn_obj = tc.get("function") or {}
                    fn_name = str(fn_obj.get("name") or "")
                    raw_args = fn_obj.get("arguments")
                    parsed_args: dict[str, Any] = {}
                    if isinstance(raw_args, dict):
                        parsed_args = raw_args
                    elif isinstance(raw_args, str) and raw_args.strip():
                        try:
                            loaded = json.loads(raw_args)
                            if isinstance(loaded, dict):
                                parsed_args = loaded
                        except Exception:
                            parsed_args = {}
                    call_specs.append((fn_name, parsed_args))
                    call_ids.append(str(tc.get("id") or f"call_{round_idx}_{idx_tc}"))

                compacted_results = await self._execute_tools_batch(call_specs, tool_executor, model)
                for call_id, res in zip(call_ids, compacted_results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(res, ensure_ascii=False, default=str),
                        }
                    )

            failure_reason = "DeepSeek 工具调用轮次超限"
            return None, failure_reason
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {str(exc)[:220]}"
            logger.warning("DeepSeek model %s failed [%s]: %s", model, usage_label or "unknown", failure_reason)
            return None, failure_reason
        finally:
            if api_rounds > 0 or not success:
                latency_ms = int((time.monotonic() - t0) * 1000)
                await self._record_usage(
                    usage_label=usage_label,
                    model=model,
                    thinking_level=effective_level,
                    api_rounds=api_rounds,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    output_tokens=output_tokens,
                    thought_tokens=thought_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    search_queries=0,
                )

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str = "",
        thinking_level: str = "MEDIUM",
        tools: list[types.FunctionDeclaration] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        use_google_search: bool = False,
        response_schema: Any = None,
        temperature: float = 0.25,
        max_tool_rounds: int = 4,
        usage_label: str = "",
        round_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        if use_google_search and self._client is None:
            await self._emit_ops(
                severity="CRITICAL",
                source=f"llm:{usage_label or model}",
                alert_key="gemini_api_key_missing",
                message="GEMINI_API_KEY 未配置，Google Search 联网检索不可用。",
            )
            return "⚠️ GEMINI_API_KEY 未配置，无法调用大模型分析。"

        if self._client is None and not self.deepseek_api_key:
            await self._emit_ops(
                severity="CRITICAL",
                source=f"llm:{usage_label or model}",
                alert_key="gemini_api_key_missing",
                message="GEMINI_API_KEY 与 DEEPSEEK_API_KEY 均未配置，所有 AI 分析不可用。",
            )
            return "⚠️ GEMINI_API_KEY 未配置，无法调用大模型分析。"

        configured_tools: list[types.Tool] = []
        if response_schema is None:
            # Keep structured-output finalizer tool-free for universal Gemini compatibility
            if tools:
                configured_tools.append(types.Tool(function_declarations=tools))
            elif use_google_search:
                configured_tools.append(types.Tool(google_search=types.GoogleSearch()))

        raw_candidates = self._candidate_chain(model, thinking_level, use_google_search=use_google_search)
        candidates: list[tuple[str, str]] = []
        skipped_errors: list[str] = []
        for cand_m, cand_t in raw_candidates:
            if _is_deepseek_model(cand_m):
                if use_google_search:
                    continue
                if not self.deepseek_api_key:
                    skipped_errors.append(f"{cand_m}: DEEPSEEK_API_KEY 未配置")
                    continue
            else:
                if self._client is None:
                    skipped_errors.append(f"{cand_m}: GEMINI_API_KEY 未配置")
                    continue
            candidates.append((cand_m, cand_t))

        if not candidates:
            await self._emit_ops(
                severity="CRITICAL",
                source=f"llm:{usage_label or model}",
                alert_key="gemini_api_key_missing",
                message=f"无可用模型凭证（{' | '.join(skipped_errors)}）。",
            )
            return "⚠️ AI 模型密钥未配置，无法调用大模型分析。"

        errors: list[str] = list(skipped_errors)
        used_fallback = bool(skipped_errors)

        for idx, (cand_model, cand_think) in enumerate(candidates):
            # A per-call ceiling is only for the primary candidate: fallbacks keep the
            # client defaults so one slow primary cannot stall the whole failover chain.
            cand_timeout = round_timeout_seconds if idx == 0 else None
            if _is_deepseek_model(cand_model):
                text, err_reason = await self._run_deepseek_once(
                    model=cand_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    thinking_level=cand_think,
                    tools=tools,
                    tool_executor=tool_executor,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tool_rounds=max_tool_rounds,
                    usage_label=usage_label,
                    round_timeout_seconds=cand_timeout,
                    max_output_tokens=max_output_tokens,
                )
            else:
                text, err_reason = await self._run_model_once(
                    model=cand_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    thinking_level=cand_think,
                    configured_tools=configured_tools,
                    tool_executor=tool_executor,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tool_rounds=max_tool_rounds,
                    usage_label=usage_label,
                    round_timeout_seconds=cand_timeout,
                    max_output_tokens=max_output_tokens,
                )
            if text is not None:
                if idx > 0 or used_fallback:
                    err_summary = " | ".join(errors)[:360] if errors else "未知异常"
                    await self._emit_ops(
                        severity="WARN",
                        source=f"llm:{usage_label or model}",
                        alert_key=f"llm_fallback:{model}->{cand_model}",
                        message=(
                            f"主模型 {model} 调用异常（{err_summary}），"
                            f"已自动降级到备援模型 {cand_model} ({cand_think}) 完成响应。"
                        ),
                    )
                return text
            errors.append(f"{cand_model}: {err_reason}")

        joined_errs = " | ".join(errors)
        await self._emit_ops(
            severity="CRITICAL",
            source=f"llm:{usage_label or model}",
            alert_key=f"llm_unavailable:{usage_label or model}",
            message=f"所有可用 AI 模型均不可用（{joined_errs}）。请立即检查 API 配额、账单状态或服务商异常。",
        )
        return f"⚠️ AI 模型服务当前全部不可用（主模型与备援均失败，已发送运维告警）：{errors[-1]}"
