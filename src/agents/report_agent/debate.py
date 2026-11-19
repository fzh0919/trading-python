"""Step 2 Bull/Bear parallel analysis, Round 2 cross-rebuttal, and Round 3 digest/factcheck."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agents.ds_sidecar import ds_helper_available
from src.agents.llm_client import LLMBackend, is_degraded_llm_output
from src.agents.prompts import build_agent_system_instruction
from src.agents.schemas import trim_text_at_line_boundary
from src.agents.tools import MarketToolkit
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger("src.agents.report_agent")

STEP2_ANALYST_FALLBACK = (
    "（本轮 Bull 机会分析未返回有效结果：模型调用超时或不可用。PM 请仅依据本地 FactSheet、"
    "多空证据对照与量化指标，按保守口径裁决，不得编造新的价位依据；无证据支撑则维持 HOLD。）"
)
STEP2_AUDITOR_FALLBACK = (
    "（本轮 Bear 独立审计未返回有效结果：模型调用超时或不可用。PM 请按保守口径自行裁决："
    "今日不放行超出既有计划的新增加仓，既有挂单维持原状不撤。）"
)


async def _gather_step2_notes(
    analyst_coro: Awaitable[str],
    auditor_coro: Awaitable[str],
    *,
    timeout_s: float,
) -> tuple[str, str, list[str]]:
    """Run Bull/Bear concurrently under one wall-clock budget.

    Timed-out tasks are cancelled and their notes replaced with neutral placeholders, and the
    `generate()` model-unavailable sentinel is swapped out too: OpsNotifier wording must never
    reach the PM prompt where it would read like a Bear audit opinion.
    """
    tasks: dict[str, asyncio.Task[str]] = {
        "analyst": asyncio.ensure_future(analyst_coro),
        "auditor": asyncio.ensure_future(auditor_coro),
    }
    degraded: list[str] = []
    if timeout_s > 0:
        _, pending = await asyncio.wait(set(tasks.values()), timeout=timeout_s)
        if pending:
            for role, task in tasks.items():
                if task in pending:
                    degraded.append(role)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Report Step 2 wall-clock budget %.0fs exceeded; degraded roles: %s",
                timeout_s,
                ", ".join(degraded),
            )
    else:
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    fallbacks = {"analyst": STEP2_ANALYST_FALLBACK, "auditor": STEP2_AUDITOR_FALLBACK}
    notes: dict[str, str] = {}
    for role, task in tasks.items():
        text = ""
        if not task.cancelled():
            try:
                text = (task.result() or "").strip()
            except Exception as exc:  # defensive: generate() already converts its own failures
                logger.warning("Report Step 2 %s task raised: %s", role, exc)
        if not text or is_degraded_llm_output(text):
            if role not in degraded:
                degraded.append(role)
            text = fallbacks[role]
        notes[role] = text
    return notes["analyst"], notes["auditor"], degraded


# ---------- v5.4.2: Step 2 round-2 debate (DeepSeek) + dual neutral index ----------

_DEBATE_NEUTRAL_SYSTEM = (
    "你是只读的中立辩论记录员/数字核验员，只做信息索引与事实比对，"
    "不做投资裁决、不给买卖建议、不新增任何事实或价位。"
)


@dataclass
class Step2DebateResult:
    """R1/R2/R3 debate output handed to the PM (full transcript goes to context_pack only)."""

    analyst_notes: str
    auditor_notes: str
    digest: str = ""
    factcheck: str = ""
    degraded_roles: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


async def _gather_named_notes(
    coros: dict[str, Awaitable[str]],
    *,
    timeout_s: float,
) -> tuple[dict[str, str], list[str]]:
    """Run named coroutines concurrently under one wall-clock budget.

    Unlike `_gather_step2_notes`, a failed round returns an empty string (no neutral
    placeholder): optional debate rounds must never leak ops wording into the PM prompt.
    """
    tasks = {name: asyncio.ensure_future(coro) for name, coro in coros.items()}
    degraded: list[str] = []
    if timeout_s > 0:
        _, pending = await asyncio.wait(set(tasks.values()), timeout=timeout_s)
        if pending:
            degraded.extend(name for name, task in tasks.items() if task in pending)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Report debate budget %.0fs exceeded; degraded roles: %s",
                timeout_s,
                ", ".join(degraded),
            )
    else:
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    texts: dict[str, str] = {}
    for name, task in tasks.items():
        text = ""
        if not task.cancelled():
            try:
                text = (task.result() or "").strip()
            except Exception as exc:  # defensive: generate() already converts its own failures
                logger.warning("Report debate %s task raised: %s", name, exc)
        if not text or is_degraded_llm_output(text):
            if name not in degraded:
                degraded.append(name)
            text = ""
        texts[name] = text
    return texts, degraded


async def _run_named_notes_sequential(
    coro_factories: dict[str, Callable[[float], Awaitable[str]]],
    *,
    timeout_s: float,
    step_ceiling_s: float | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Run named steps strictly one-by-one under a shared phase budget (v5.5.5).

    Lazy factories are mandatory: when an earlier step consumes the whole phase budget we must
    never construct a coroutine that is then abandoned (that would raise
    ``RuntimeWarning: coroutine was never awaited``). Failed/timed-out steps degrade to an empty
    string exactly like :func:`_gather_named_notes`, without aborting the morning PM decision.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    ceiling = float(step_ceiling_s) if step_ceiling_s is not None else float(timeout_s)
    texts: dict[str, str] = {}
    degraded: list[str] = []
    for name, factory in coro_factories.items():
        remaining = float(timeout_s) - (loop.time() - started)
        # Windows' monotonic clock is quantised (~15.6 ms); treat sub-centisecond leftovers
        # as exhausted so we never build a coroutine that instantly degrades.
        if remaining <= 0.01:
            degraded.append(name)
            texts[name] = ""
            continue
        step_budget = min(ceiling, remaining)
        task = asyncio.ensure_future(factory(step_budget))
        try:
            _, pending = await asyncio.wait({task}, timeout=step_budget)
            if pending:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                degraded.append(name)
                texts[name] = ""
                logger.warning(
                    "Report debate sequential step %s exceeded %.1fs; degraded",
                    name,
                    step_budget,
                )
                continue
            text = (task.result() or "").strip()
        except Exception as exc:  # defensive: generate() already converts its own failures
            logger.warning("Report debate %s task raised: %s", name, exc)
            text = ""
        if not text or is_degraded_llm_output(text):
            if name not in degraded:
                degraded.append(name)
            text = ""
        texts[name] = text
    return texts, degraded


def _build_rebuttal_prompt(
    *,
    side: str,
    own_notes: str,
    opponent_notes: str,
    shared_context: str,
    max_chars: int,
) -> str:
    if side == "bull":
        role = "Bull（机会方）"
        opponent = "Bear（风控审计官）"
        focus = "逐条反驳 Bear 的风险论断，指出哪些担忧已被事实底座化解、哪些被夸大，并守住你首轮的多方立论。"
    else:
        role = "Bear（风控审计官）"
        opponent = "Bull（机会方）"
        focus = "逐条攻击 Bull 的支撑位、目标价与加仓理由，指出其最坏情况与证据缺口，并守住你首轮的空方立论。"
    return (
        f"【任务：{role} 第二轮交叉质询（≤{max_chars} 字）】\n"
        "你在首轮已提交过一份草稿；本轮只做交叉质询，不重复首轮已说过的内容。\n"
        f"{focus}\n\n"
        f"=== 你的首轮草稿 ===\n{own_notes}\n\n"
        f"=== 对手 {opponent} 的首轮草稿 ===\n{opponent_notes}\n\n"
        "=== 唯一事实底座（与首轮相同） ===\n"
        f"{shared_context}\n\n"
        "硬规则：\n"
        "1. 开头先写「本轮最关键的 3 个反驳点」；\n"
        "2. 逐条回应对手的具体论点，必须引用底座中的数字/事实，不得编造；\n"
        "3. 不得新增首轮未出现的价位或新闻；\n"
        "4. 不要给最终买卖结论（PM 负责裁决）。"
    )


def _build_rebuttal_system(side: str, cfg: AppConfig, max_chars: int) -> str:
    if side == "bull":
        role = (
            f"你是 Bull 阵营的第二轮交叉质询者（≤{max_chars} 字）。"
            "基于同一份事实底座，逐条反驳 Bear 首轮的论点；不要重写研报、不要给最终买卖结论。"
        )
    else:
        role = (
            f"你是 Bear 阵营的第二轮交叉质询者（≤{max_chars} 字）。"
            "基于同一份事实底座，逐条攻击 Bull 首轮的支撑位/目标价/加仓理由；不要重写研报、不要给最终买卖结论。"
        )
    return build_agent_system_instruction(role, cfg.strategy)


def _build_debate_digest_prompt(
    *,
    bull_notes: str,
    bear_notes: str,
    bull_rebuttal: str,
    bear_rebuttal: str,
    disagreement_text: str,
    max_chars: int,
) -> str:
    return (
        f"【任务：中立辩论摘要（≤{max_chars} 字；只做索引，不做裁决）】\n"
        "你只负责整理这场 Bull/Bear 辩论，不给买卖建议、不新增事实。\n\n"
        f"=== Bull 首轮 ===\n{bull_notes}\n\n"
        f"=== Bear 首轮 ===\n{bear_notes}\n\n"
        f"=== Bull 对 Bear 的质询 ===\n{bull_rebuttal}\n\n"
        f"=== Bear 对 Bull 的质询 ===\n{bear_rebuttal}\n\n"
        f"=== 本地多空证据冲突对照 ===\n{disagreement_text}\n\n"
        "按以下四段输出（每段 ≤3 条，直接给索引，不要复述双方原话）：\n"
        "1. 存活的多方论点；\n"
        "2. 存活的空方论点；\n"
        "3. 已被驳倒的论点及原因；\n"
        "4. 仍无证据、需 PM 亲自裁决的冲突。"
    )


def _strip_trailing_orphan_markdown(text: str) -> str:
    """Drop trailing blank lines and lone Markdown/【】 headings left behind by char trimming."""
    lines = text.splitlines()
    while lines:
        stripped = lines[-1].strip()
        if not stripped:
            lines.pop()
            continue
        is_heading = (
            (len(stripped) > 4 and stripped.startswith("**") and stripped.endswith("**"))
            or stripped.startswith("#")
            or (len(stripped) >= 3 and stripped.startswith("【") and stripped.endswith("】"))
        )
        if not is_heading:
            break
        lines.pop()
    return "\n".join(lines)


def _build_factcheck_prompt(
    *,
    fact_sheet: str,
    bull_rebuttal: str,
    bear_rebuttal: str,
    max_chars: int,
) -> str:
    return (
        f"【任务：只读数字核验（≤{max_chars} 字）】\n"
        "下面给出唯一事实底座与两份第二轮质询。只核对质询中出现的价格、持仓成本、止损价、挂单价、百分比"
        "能否在事实底座中找到依据。\n\n"
        f"=== 只读事实底座 ===\n{fact_sheet}\n\n"
        f"=== Bull 质询 ===\n{bull_rebuttal}\n\n"
        f"=== Bear 质询 ===\n{bear_rebuttal}\n\n"
        "输出格式：逐条 `[无据] 标的 / 数字 / 出自 Bull|Bear`；没有发现则输出「未发现无据数字」。\n"
        "硬规则：只标记、不判违规；底座中找不到不等于错误，由 PM 判断；不得新增事实。"
    )


def _build_debate_pm_block(digest: str, factcheck: str) -> str:
    if not digest and not factcheck:
        return ""
    block = "\n\n=== 第二轮交叉质询与中立索引（DeepSeek，仅供索引，不是裁决） ===\n"
    if digest:
        block += f"【中立摘要】\n{digest}\n"
    if factcheck:
        block += f"【证据核验】\n{factcheck}\n"
    return block


async def _notify_debate_degraded(ops_notifier: OpsNotifier | None, roles: list[str]) -> None:
    if ops_notifier is None or not roles:
        return
    await ops_notifier.notify(
        severity="WARN",
        source="report:debate",
        alert_key="report_debate_degraded",
        message=f"/report 第二轮辩论降级：{'/'.join(roles)} 未返回有效结论，PM 将按首轮材料裁决。",
    )


async def _run_step2_debate(
    *,
    cfg: AppConfig,
    llm: LLMBackend,
    toolkit: MarketToolkit,
    shared_context: str,
    fact_sheet: str,
    disagreement_text: str,
    analyst_sys: str,
    auditor_sys: str,
    analyst_prompt: str,
    auditor_prompt: str,
    report_rounds: int,
    ops_notifier: OpsNotifier | None,
) -> Step2DebateResult:
    """Three-phase Step 2: R1 Gemini debate → R2 DeepSeek cross-exam → R3 dual neutral."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    total_budget = float(cfg.models.report_step2_timeout_seconds)
    r1_budget = min(float(cfg.models.report_step2_round1_seconds), total_budget)

    analyst_notes, auditor_notes, degraded_roles = await _gather_step2_notes(
        llm.generate(
            model=cfg.models.report_analyst_model,
            prompt=analyst_prompt,
            system_instruction=analyst_sys,
            thinking_level=cfg.models.report_analyst_thinking_level,
            tools=toolkit.declarations(),
            tool_executor=toolkit.as_executor(),
            temperature=0.2,
            max_tool_rounds=report_rounds,
            usage_label="report_analyst",
        ),
        llm.generate(
            model=cfg.models.report_auditor_model,
            prompt=auditor_prompt,
            system_instruction=auditor_sys,
            thinking_level=cfg.models.report_auditor_thinking_level,
            tools=toolkit.declarations(),
            tool_executor=toolkit.as_executor(),
            temperature=0.15,
            max_tool_rounds=report_rounds,
            usage_label="report_auditor",
        ),
        timeout_s=r1_budget,
    )

    payload: dict[str, Any] = {
        "enabled": bool(cfg.models.report_debate_enabled),
        "degraded": list(degraded_roles),
        "r1": {"bull": analyst_notes, "bear": auditor_notes},
        "r2": {},
        "r3": {},
        "elapsed": {"r1_s": round(loop.time() - started, 1)},
    }
    result = Step2DebateResult(
        analyst_notes=analyst_notes,
        auditor_notes=auditor_notes,
        degraded_roles=list(degraded_roles),
        payload=payload,
    )

    if degraded_roles:
        payload["skipped"] = "r1_degraded"
        return result
    if not cfg.models.report_debate_enabled or not ds_helper_available(llm, cfg):
        payload["skipped"] = "debate_disabled" if not cfg.models.report_debate_enabled else "deepseek_unavailable"
        return result

    remaining = total_budget - (loop.time() - started)
    r2_budget = min(float(cfg.models.report_step2_round2_seconds), max(0.0, remaining))
    if r2_budget <= 0:
        payload["skipped"] = "no_budget_r2"
        return result

    debate_round_timeout = max(30.0, r2_budget - 15.0)
    max_chars = int(cfg.models.report_debate_max_chars)
    r2_started = loop.time()
    r2_factories: dict[str, Callable[[float], Awaitable[str]]] = {
        "bull_rebuttal": lambda step_budget: llm.generate(
            model=cfg.models.report_debate_model,
            prompt=_build_rebuttal_prompt(
                side="bull",
                own_notes=analyst_notes,
                opponent_notes=auditor_notes,
                shared_context=shared_context,
                max_chars=max_chars,
            ),
            system_instruction=_build_rebuttal_system("bull", cfg, max_chars),
            thinking_level=cfg.models.report_debate_thinking_level,
            temperature=0.25,
            usage_label="report_bull_rebuttal",
            round_timeout_seconds=step_budget,
            max_output_tokens=cfg.models.report_debate_max_output_tokens,
        ),
        "bear_rebuttal": lambda step_budget: llm.generate(
            model=cfg.models.report_debate_model,
            prompt=_build_rebuttal_prompt(
                side="bear",
                own_notes=auditor_notes,
                opponent_notes=analyst_notes,
                shared_context=shared_context,
                max_chars=max_chars,
            ),
            system_instruction=_build_rebuttal_system("bear", cfg, max_chars),
            thinking_level=cfg.models.report_debate_thinking_level,
            temperature=0.2,
            usage_label="report_bear_rebuttal",
            round_timeout_seconds=step_budget,
            max_output_tokens=cfg.models.report_debate_max_output_tokens,
        ),
    }
    if bool(getattr(cfg.models, "report_debate_sequential", True)):
        r2_texts, r2_degraded = await _run_named_notes_sequential(
            r2_factories,
            timeout_s=r2_budget,
            step_ceiling_s=debate_round_timeout,
        )
    else:
        r2_texts, r2_degraded = await _gather_named_notes(
            {name: factory(debate_round_timeout) for name, factory in r2_factories.items()},
            timeout_s=r2_budget,
        )
    r2_texts = {name: trim_text_at_line_boundary(text, max_chars) if text else "" for name, text in r2_texts.items()}
    payload["r2"] = r2_texts
    payload["elapsed"]["r2_s"] = round(loop.time() - r2_started, 1)
    if r2_degraded:
        payload["degraded"].extend(r2_degraded)
        await _notify_debate_degraded(ops_notifier, r2_degraded)
        return result

    remaining = total_budget - (loop.time() - started)
    r3_budget = min(float(cfg.models.report_step2_round3_seconds), max(0.0, remaining))
    if r3_budget <= 0:
        payload["skipped"] = "no_budget_r3"
        return result

    neutral_timeout = max(20.0, r3_budget - 10.0)
    digest_chars = int(cfg.models.report_neutral_digest_max_chars)
    factcheck_chars = int(cfg.models.report_neutral_factcheck_max_chars)
    r3_started = loop.time()
    r3_factories: dict[str, Callable[[float], Awaitable[str]]] = {
        "digest": lambda step_budget: llm.generate(
            model=cfg.models.report_neutral_model,
            prompt=_build_debate_digest_prompt(
                bull_notes=analyst_notes,
                bear_notes=auditor_notes,
                bull_rebuttal=r2_texts.get("bull_rebuttal", ""),
                bear_rebuttal=r2_texts.get("bear_rebuttal", ""),
                disagreement_text=disagreement_text,
                max_chars=digest_chars,
            ),
            system_instruction=_DEBATE_NEUTRAL_SYSTEM,
            thinking_level=cfg.models.report_neutral_thinking_level,
            temperature=0.1,
            usage_label="report_debate_digest",
            round_timeout_seconds=step_budget,
            max_output_tokens=8192,
        ),
        "factcheck": lambda step_budget: llm.generate(
            model=cfg.models.report_neutral_model,
            prompt=_build_factcheck_prompt(
                fact_sheet=shared_context or fact_sheet,
                bull_rebuttal=r2_texts.get("bull_rebuttal", ""),
                bear_rebuttal=r2_texts.get("bear_rebuttal", ""),
                max_chars=factcheck_chars,
            ),
            system_instruction=_DEBATE_NEUTRAL_SYSTEM,
            thinking_level=cfg.models.report_neutral_thinking_level,
            temperature=0.0,
            usage_label="report_debate_factcheck",
            round_timeout_seconds=step_budget,
            max_output_tokens=8192,
        ),
    }
    if bool(getattr(cfg.models, "report_debate_sequential", True)):
        r3_texts, r3_degraded = await _run_named_notes_sequential(
            r3_factories,
            timeout_s=r3_budget,
            step_ceiling_s=neutral_timeout,
        )
    else:
        r3_texts, r3_degraded = await _gather_named_notes(
            {name: factory(neutral_timeout) for name, factory in r3_factories.items()},
            timeout_s=r3_budget,
        )
    result.digest = (
        _strip_trailing_orphan_markdown(trim_text_at_line_boundary(r3_texts.get("digest", ""), digest_chars))
        if r3_texts.get("digest")
        else ""
    )
    result.factcheck = (
        trim_text_at_line_boundary(r3_texts.get("factcheck", ""), factcheck_chars)
        if r3_texts.get("factcheck")
        else ""
    )
    payload["r3"] = {"digest": result.digest, "factcheck": result.factcheck}
    payload["elapsed"]["r3_s"] = round(loop.time() - r3_started, 1)
    if r3_degraded:
        payload["degraded"].extend(r3_degraded)
        await _notify_debate_degraded(ops_notifier, r3_degraded)
    return result
