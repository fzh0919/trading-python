"""Async Gemini API client with ThinkingConfig, retry/fallback degradation, Ops alerts, and token telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

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
        if lvl == "XHIGH" and "hydrogen" not in m:
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
    ) -> str: ...


class GeminiClient:
    """Wrapper over google-genai async SDK with transient retry, fallback model switching, and telemetry."""

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
        round_timeout_seconds: float = 120.0,
        tool_timeout_seconds: float = 60.0,
        max_concurrent_tools: int = 6,
        max_tool_result_chars: int = 4000,
        usage_summary_provider: UsageSummaryProvider | None = None,
        search_free_quota_monthly: int = 5000,
        search_warn_pct: float = 85.0,
    ) -> None:
        self.api_key = api_key.strip()
        self._client: genai.Client | None = genai.Client(api_key=self.api_key) if self.api_key else None
        self._usage_recorder = usage_recorder
        self._ops = ops_notifier
        self.fallback_model = fallback_model.strip() or "gemini-4-hydrogen"
        self.fallback_thinking_level = fallback_thinking_level.strip() or "HIGH"
        self.max_transient_retries = max(0, int(max_transient_retries))
        self.retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self.round_timeout_seconds = max(0.0, float(round_timeout_seconds))
        self.tool_timeout_seconds = max(0.0, float(tool_timeout_seconds))
        self.max_concurrent_tools = max(1, int(max_concurrent_tools))
        self.max_tool_result_chars = max(0, int(max_tool_result_chars))
        self._usage_summary_provider = usage_summary_provider
        self.search_free_quota_monthly = max(0, int(search_free_quota_monthly))
        self.search_warn_pct = min(100.0, max(1.0, float(search_warn_pct)))

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

    def _candidate_chain(self, primary_model: str, primary_thinking: str) -> list[tuple[str, str]]:
        chain: list[tuple[str, str]] = [(primary_model, primary_thinking)]
        if self.fallback_model and self.fallback_model != primary_model:
            chain.append((self.fallback_model, self.fallback_thinking_level))
        elif primary_model != "gemini-4-argon":
            chain.append(("gemini-4-argon", "MEDIUM"))
        return chain

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
    ) -> tuple[Any, types.ThinkingConfig | None, str]:
        assert self._client is not None

        async def _invoke_sdk(cfg_obj: types.GenerateContentConfig) -> Any:
            coro = self._client.aio.models.generate_content(  # type: ignore[union-attr]
                model=model,
                contents=contents,
                config=cfg_obj,
            )
            if self.round_timeout_seconds > 0:
                return await asyncio.wait_for(coro, timeout=self.round_timeout_seconds)
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

                if attempt < self.max_transient_retries and _is_transient_llm_error(exc):
                    delay = self.retry_base_delay_s * (2**attempt)
                    logger.warning(
                        "Transient Gemini error on %s (attempt %d/%d): %s; retrying in %.1fs",
                        model,
                        attempt + 1,
                        self.max_transient_retries + 1,
                        err_msg[:180],
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
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
    ) -> tuple[str | None, str]:
        thinking_cfg, effective_level = self._build_thinking_config(model, thinking_level)

        def _make_config(t_cfg: types.ThinkingConfig | None) -> types.GenerateContentConfig:
            kwargs: dict[str, Any] = {"temperature": temperature}
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
                    failure_reason = "模型返回空文本（可能受限流或安全策略拦截）"
                    return None, failure_reason

                if response.candidates and response.candidates[0].content:
                    contents.append(response.candidates[0].content)

                sem = asyncio.Semaphore(self.max_concurrent_tools)

                async def _execute_single_tool(call: Any) -> types.Part:
                    call_name = call.name or ""
                    call_args = dict(call.args or {})
                    logger.info("Gemini tool call [%s]: %s(%s)", model, call_name, call_args)
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
                    return types.Part.from_function_response(name=call_name, response={"result": compact})

                fn_response_parts = list(await asyncio.gather(*(_execute_single_tool(c) for c in fn_calls)))
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
    ) -> str:
        if self._client is None:
            await self._emit_ops(
                severity="CRITICAL",
                source=f"llm:{usage_label or model}",
                alert_key="gemini_api_key_missing",
                message="GEMINI_API_KEY 未配置，所有 AI 分析不可用。",
            )
            return "⚠️ GEMINI_API_KEY 未配置，无法调用大模型分析。"

        configured_tools: list[types.Tool] = []
        if response_schema is None:
            # Keep structured-output finalizer tool-free for universal Gemini compatibility
            if tools:
                configured_tools.append(types.Tool(function_declarations=tools))
            elif use_google_search:
                configured_tools.append(types.Tool(google_search=types.GoogleSearch()))

        candidates = self._candidate_chain(model, thinking_level)
        errors: list[str] = []

        for idx, (cand_model, cand_think) in enumerate(candidates):
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
            )
            if text is not None:
                if idx > 0:
                    await self._emit_ops(
                        severity="WARN",
                        source=f"llm:{usage_label or model}",
                        alert_key=f"llm_fallback:{model}->{cand_model}",
                        message=(
                            f"主模型 {model} 调用异常（{errors[0]}），"
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
            message=f"所有 Gemini 模型均不可用（{joined_errs}）。请立即检查 API 配额、账单状态或 Google 服务异常。",
        )
        return f"⚠️ Gemini AI 服务当前全部不可用（主模型与备援均失败，已发送运维告警）：{errors[-1]}"
