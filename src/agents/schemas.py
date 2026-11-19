"""Pydantic decision schemas & inverted-pyramid Telegram renderers (adapted from TradingAgents)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.engine.portfolio_rules import RiskOverrideResult

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_GUARDRAIL_TAG = "[🛡️风控护栏修正]"
_ADVISORY_TAG = "[⚠️纪律提醒]"


class ReportActionItem(BaseModel):
    account: str = Field(description="ISA 或 Invest 或 双账户")
    symbol: str = Field(description="现有持仓标的代码，若今日持仓观望无调仓可写 CASH（挂单撤改勿放此处）")
    action: str = Field(description="仅限持仓动作：ADD / TRIM / STOP / HOLD（撤改挂单只放 pending_order_audits，禁止重复放此处）")
    price_zone: str = Field(description="建议触发价位或区间，如 '355-358 USD' 或 '4500 GBp'")
    size_hint: str = Field(
        default="",
        description=(
            "数量口径（v5.4.1）：ADD/TRIM 必填比例或英镑金额，如 '30%'、'1/3'、'£2,300'；"
            "STOP 填 '全出'；HOLD/CASH 留空。严禁写股数（由 Python 按持仓确定性换算）"
        ),
    )
    reason: str = Field(
        description=(
            "一句话核心依据（不超过60字）；Invest 调仓须标明资金路径，"
            "如「USD钱包免换汇」或「GBP不足需USD换回+0.15%」"
        )
    )


class CoreDeploymentItem(BaseModel):
    """v5.5.1: PM verdict on Python's deterministic core-wide-base policy ticket."""

    account: str = Field(default="ISA", description="仅限 ISA（应税 Invest 仅允许派息型核心宽基，且不在本字段新建仓）")
    symbol: str = Field(description="必须等于 Python 今日签发的 policy_ticket.symbol（严禁换代码）")
    verdict: Literal["APPROVE", "SHRINK", "DEFER"] = Field(
        default="APPROVE",
        description="APPROVE=按上限执行；SHRINK=缩减金额；DEFER=因重大宏观/事件暂缓（不生成买入指令）",
    )
    action: Literal["BUY"] = Field(default="BUY", description="固定为 BUY")
    approved_gbp: float | None = Field(
        default=None,
        description="PM 批准的英镑金额；DEFER 时为 0；留空则从 size_hint 提取或回退 Python 建议金额",
    )
    price_zone: str = Field(description="建议买入价位/区间，如 '35.2-35.6 GBP'")
    size_hint: str = Field(description="建议金额（如 '£1,600'）；必填，且不得超过 Python 今日预算上限")
    reason: str = Field(description="一句话批准/缩减/暂缓依据（不超过60字）")


class PendingOrderAuditItem(BaseModel):
    account: str = Field(description="ISA 或 Invest")
    symbol: str = Field(description="挂单标的代码")
    verdict: Literal["KEEP", "MODIFY", "CANCEL"] = Field(description="KEEP（保留）| MODIFY（调低/改价）| CANCEL（建议撤单）")
    order_summary: str = Field(description="极简挂单方向与价位，如 'BUY 1250 GBp' 或 'STOP 93.80 USD'")
    reason: str = Field(
        description=(
            "结合现价、均线、目标价及印花税/换汇成本给出的一句话审计结论（不超过60字）；"
            "Invest 按实际资金来源计（同币种钱包 0% / 跨币种单边 0.15%）"
        )
    )


class FlaggedPositionItem(BaseModel):
    symbol: str = Field(description="标的代码，如 NVDA、ULVR.L")
    summary: str = Field(description="一句话异动/观察摘要（不超过60字）")


def _coerce_flagged_position_str(raw: str) -> FlaggedPositionItem | None:
    text = raw.strip().lstrip("•-* ").strip()
    if not text:
        return None
    m = re.match(
        r"^([A-Za-z0-9]{1,12}(?:\.[A-Za-z]{1,4})?)(?:\s*[\[\(][^\]\)]*[\]\)])?\s*(?:[:：—\-]\s*|\s+|$)(.*)$",
        text,
    )
    if m:
        cand_sym = m.group(1).strip().upper()
        rest = m.group(2).strip()
        if any(c.isalpha() for c in cand_sym) and not rest.startswith("/"):
            return FlaggedPositionItem(symbol=cand_sym, summary=rest)
    return FlaggedPositionItem(symbol="", summary=text)


def _as_flagged_position_item(item: Any) -> FlaggedPositionItem | None:
    if isinstance(item, FlaggedPositionItem):
        return FlaggedPositionItem(symbol=item.symbol.strip(), summary=item.summary.strip())
    if isinstance(item, dict):
        return FlaggedPositionItem(
            symbol=str(item.get("symbol") or "").strip(),
            summary=str(item.get("summary") or "").strip(),
        )
    if isinstance(item, str):
        return _coerce_flagged_position_str(item)
    return _coerce_flagged_position_str(str(item))


class PickGateDecision(BaseModel):
    """v5.4.0 PM-level pick gate — deterministic rules are the bottom line, PM may only tighten."""

    allow: bool = Field(default=True, description="是否允许本轮 /pick 新开仓位（false 为 PM 主动收紧否决）")
    mode: str = Field(default="none", description="run=可运行选股 | rotate=须等风险置换 | none=无可用通道")
    funding_from: str = Field(default="", description="ROTATE 模式的合法腾挪资金源（平掉/撤掉的标的代码），缺省为空")
    reason: str = Field(default="", description="一句话说明允许或收紧的原因（不超过45字）")


class MorningReportDecision(BaseModel):
    headline: str = Field(description="一句话定调今日核心策略与现金态度（不超过55字）")
    top_actions: list[ReportActionItem] = Field(
        default_factory=list,
        description="今日现有持仓最优先的 1~3 条调仓动作（不含撤单/改单）；若持仓全部不动，给1条 CASH HOLD 即可",
    )
    pending_order_audits: list[PendingOrderAuditItem] = Field(
        default_factory=list,
        description="对所有未成交活动挂单逐一做出 KEEP / MODIFY / CANCEL 独立审计（没有挂单则为空数组）",
    )
    core_deployment: list[CoreDeploymentItem] = Field(
        default_factory=list,
        description=(
            "0~2 条未持有的核心/分散宽基 ETF 新买入（仅 ISA，BUY）；"
            "核心宽基已达标、现金不足或趋势破位时留空数组；优先低科技 FTSE100/欧洲宽基分散美股科技"
        ),
    )
    daily_vetoes: list[str] = Field(
        default_factory=list,
        description="2~4条今日硬性风控红线（供后续/pick与/us严格继承，每条不超过50字）",
    )
    macro_and_cash: str = Field(
        description=(
            "宏观利率/VIX/汇率精炼要点，并说明相对基准 VWRP 的配置态度与现金部署计划"
            "（现金占比、触发阈值、下一步具体价位与金额；不超过130字）"
        )
    )
    account_diagnosis: str = Field(description="ISA与Invest净值、FX影响、行业/跨账户集中度痛点（不超过180字）")
    flagged_positions: list[FlaggedPositionItem] = Field(
        default_factory=list,
        description="仅列出【未在 top_actions 与 pending_order_audits 中出现过】的其他异动/破位/观察持仓，每条<=60字（最多4条，严禁与上方重复）",
    )
    hold_summary: str = Field(description="一句话合并概括其余常规持有不动仓位（不超过80字）")
    trade_review: str = Field(description="近7-30天实盘操作与摩擦一句话短评（不超过100字）")
    earnings_watch: list[str] = Field(
        default_factory=list,
        description="近7日财报预警与应对预案（每条<=45字，无则为空数组）",
    )
    pick_gate: PickGateDecision = Field(
        default_factory=PickGateDecision,
        description=(
            "今日 /pick 组合门控收紧权（仅可收紧：仅在发现重大宏观/事件风险时把 allow 置 false 并写明理由；"
            "严禁把组合门控的 WAIT/DEFEND 升为 RUN；ROTATE 时必须给出 funding_from 合法置换源）"
        ),
    )

    @field_validator("flagged_positions", mode="before")
    @classmethod
    def _normalize_flagged_positions(cls, value: Any) -> list[FlaggedPositionItem]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized: list[FlaggedPositionItem] = []
        for item in value:
            coerced = _as_flagged_position_item(item)
            if coerced is not None:
                normalized.append(coerced)
        return normalized


class PickActionItem(BaseModel):
    account: str = Field(description="ISA 或 Invest")
    symbol: str = Field(description="标的代码，如 VWRP.L、SHEL.L、TSM")
    action: Literal["BUY", "ADD", "TRIM", "SWAP"] = Field(description="BUY / ADD / TRIM / SWAP")
    thesis: str = Field(
        description=(
            "交易逻辑与催化剂，说明相对基准 VWRP 的配置理由、目标空间/盈亏比(R:R)与摩擦承受度（不超过55字）；"
            "Invest 动作用须标明资金路径，如「用 USD 钱包免换汇」或「GBP 不足需 USD 换回含 0.15% 损耗」"
        )
    )
    entry_zone: str = Field(description="建议买入或加减仓价位区间与建议金额")
    stop_condition: str = Field(description="硬止损位或证伪条件")
    target_window: str = Field(description="目标位或观察周期")
    entry_price: float = Field(default=0.0, description="当前参考现价数字（用于数据库记录，未知填0）")
    stop_price: float = Field(
        default=0.0,
        description=(
            "数值硬止损价（标的本地货币）。Invest 动作必填 >0 且止损宽度≥max(1.5×ATR, 4%)；"
            "ISA 长线价值个股可填 0（须在 stop_condition 写明基本面/护城河证伪条件）；"
            "Python 用它复算 R:R 与以损定仓，禁止只写在 stop_condition 文本里"
        ),
    )
    target_price: float = Field(
        default=0.0,
        description=(
            "数值目标价 / 保守内在价值锚（标的本地货币 >0）。"
            "Invest 非牛市偏离分析师均值>15% 会回拨均值，RISK_ON 右侧强股放宽至高目标上沿；ISA 填保守合理估值锚"
        ),
    )
    risk_note: str = Field(
        default="",
        description="内部保留字段，模型输出时请一律返回空字符串 ''（由 Python 写入 R:R 与仓位校验结论）",
    )


class PickDecision(BaseModel):
    headline: str = Field(
        description=(
            "一句话定调本轮最终选股结论与现金部署计划（不超过55字；严禁提及任何未入选、内部讨论或被否决的候选标的/方向；"
            "若无推荐，写明“本轮无达标机会（ISA 安全边际 / Invest 动态 R:R 未达标），现金待命”并给出基准 VWRP 相对态度或下一次部署触发条件）"
        )
    )
    selected_actions: list[PickActionItem] = Field(
        default_factory=list,
        description="通过风控官六重护栏的最终推荐（0~3条）；若均未达标请返回空数组",
    )
    rejected_or_frozen: list[str] = Field(
        default_factory=list,
        description="内部保留字段，模型输出时请一律返回空数组 []（严禁在最终回复中列举未入选候选股或内部审查过程）",
    )
    consistency_and_cash_note: str = Field(
        description="一句话说明执行建议后的双账户剩余现金与占总资产精确比例，及对早报红线的继承（不超过85字）"
    )


def parse_structured_json(raw_text: str, model_cls: type[T]) -> T | None:
    """Parse raw JSON (or fenced JSON block) returned by LLM into a Pydantic model."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return model_cls.model_validate_json(cleaned)
    except Exception:
        # Try locating first JSON object brace block
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                return model_cls.model_validate(data)
            except Exception:
                return None
    return None


async def parse_or_repair_structured_json(
    raw_text: str,
    model_cls: type[T],
    *,
    llm: Any,
    repair_model: str,
    thinking_level: str = "LOW",
    usage_label: str = "json_repair",
    ops_notifier: Any = None,
    alert_source: str = "structured_json",
) -> tuple[T | None, str]:
    """Parse structured JSON or run a single repair pass with ops alerting on fallback/failure."""
    parsed = parse_structured_json(raw_text, model_cls)
    if parsed is not None:
        return parsed, raw_text

    if not raw_text or raw_text.lstrip().startswith("⚠️"):
        return None, raw_text

    logger.warning("%s initial structured JSON parse failed; attempting single repair pass.", alert_source)
    repair_prompt = (
        f"请将以下模型输出严格转换为符合 `{model_cls.__name__}` Schema 的合法 JSON，"
        "不得编造原文本中不存在的价格或标的，仅做结构化清洗与字段对齐：\n\n"
        f"{raw_text[:4000]}"
    )
    repaired_raw = await llm.generate(
        model=repair_model,
        prompt=repair_prompt,
        system_instruction="你是严格的数据格式修复器，只输出合法 JSON。",
        thinking_level=thinking_level,
        response_schema=model_cls,
        temperature=0.0,
        usage_label=usage_label,
    )
    repaired = parse_structured_json(repaired_raw, model_cls)
    if repaired is not None:
        if ops_notifier is not None:
            await ops_notifier.notify(
                severity="WARN",
                source=alert_source,
                alert_key=f"json_repaired:{alert_source}",
                message=f"{alert_source} 初次结构化 JSON 解析异常，已自动修复恢复正常渲染。",
            )
        return repaired, repaired_raw

    if ops_notifier is not None:
        await ops_notifier.notify(
            severity="CRITICAL",
            source=alert_source,
            alert_key=f"json_repair_failed:{alert_source}",
            message=f"{alert_source} 结构化 JSON 自动修复仍失败，已退回文本截断保底（可能缺失风控红线/存档）。",
        )
    return None, raw_text


def _is_orphan_header_line(line: str, *, allow_unclosed: bool = False) -> bool:
    stripped = line.strip()
    if not stripped.startswith("【"):
        return False
    if stripped.endswith("】") and len(stripped) >= 3:
        return True
    return allow_unclosed and "】" not in stripped


def _strip_trailing_orphan_headers(section: str, *, allow_unclosed: bool = False) -> str:
    lines = section.splitlines()
    while lines and (
        not lines[-1].strip()
        or _is_orphan_header_line(lines[-1], allow_unclosed=allow_unclosed)
    ):
        lines.pop()
    return "\n".join(lines)


def _fit_max_chars(sections: list[str], max_chars: int) -> str:
    """Keep top priority sections intact and gracefully trim tail sections without leaving orphan headers."""
    if max_chars <= 0:
        return ""
    kept = [
        cleaned
        for s in sections
        if s and (cleaned := _strip_trailing_orphan_headers(s))
    ]
    if not kept:
        return ""
    text = "\n\n".join(kept)
    if len(text) <= max_chars:
        return text

    # Drop optional/lower sections progressively from the end while keeping sections[0] (top actions)
    while len(kept) > 1 and len("\n\n".join(kept)) > max_chars:
        last = kept[-1]
        lines = last.splitlines()
        if len(lines) > 2:
            trimmed = _strip_trailing_orphan_headers("\n".join(lines[:-1]))
            if trimmed:
                kept[-1] = trimmed
            else:
                kept.pop()
        else:
            kept.pop()

    joined = "\n\n".join(kept)
    if len(joined) <= max_chars:
        return joined
    if max_chars == 1:
        return joined[:1]

    sliced = joined[: max_chars - 1].rstrip()
    sliced_lines = sliced.splitlines()
    while len(sliced_lines) > 1 and (
        not sliced_lines[-1].strip()
        or _is_orphan_header_line(sliced_lines[-1], allow_unclosed=True)
    ):
        sliced_lines.pop()
    result = "\n".join(sliced_lines).rstrip() or sliced
    return result + "…"


def _sanitize_plain_output(text: str) -> str:
    """Enforce the Telegram plain-text rule even if the model disobeys the prompt."""
    return str(text or "").replace("**", "").replace("#", "")


def trim_text_at_line_boundary(text: str, max_chars: int) -> str:
    """Trim multi-line text at newline boundaries, then sentence/clause boundaries before hard slicing."""
    cleaned = text.strip()
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    lines = cleaned.splitlines()
    while len(lines) > 1 and len("\n".join(lines)) > max_chars:
        lines.pop()
        while lines and (not lines[-1].strip() or _is_orphan_header_line(lines[-1])):
            lines.pop()
    candidate = "\n".join(lines).rstrip()
    if candidate and len(candidate) <= max_chars:
        return candidate
    if max_chars == 1:
        return cleaned[:1]

    target = candidate or cleaned
    min_cut = max(1, max_chars // 2)
    head_full = target[:max_chars]
    for sep in ("。", "！", "？"):
        pos = head_full.rfind(sep)
        if pos >= min_cut:
            return target[: pos + 1].rstrip()
    head = target[: max_chars - 1]
    for sep in ("；", ";", " | ", "，", ",", "：", ":", " "):
        pos = head.rfind(sep)
        if pos >= min_cut:
            cut = target[:pos].rstrip()
            if cut:
                return cut + "…"
    return head.rstrip() + "…"


def _format_guardrail_note(text: str) -> str:
    cleaned = text.strip().lstrip("•-* ").strip()
    if not cleaned:
        return ""
    if _GUARDRAIL_TAG in cleaned or _ADVISORY_TAG in cleaned:
        return cleaned
    return f"{_GUARDRAIL_TAG} {cleaned}"


def _extract_override_notes(risk_override: Any) -> list[str]:
    if risk_override is None:
        return []
    notes = list(getattr(risk_override, "notes", None) or [])
    formatted: list[str] = []
    seen: set[str] = set()
    for note in notes:
        fn = _format_guardrail_note(str(note))
        if fn and fn not in seen:
            seen.add(fn)
            formatted.append(fn)
    if formatted:
        return formatted
    for ov in getattr(risk_override, "overrides", None) or []:
        acct = str(getattr(ov, "account", "") or "").strip()
        sym = str(getattr(ov, "symbol", "") or "").strip()
        from_act = str(getattr(ov, "from_action", "") or "").strip()
        to_act = str(getattr(ov, "to_action", "") or "").strip()
        reason = str(getattr(ov, "reason", "") or "").strip()
        prefix = f"[{acct}] {sym}" if acct else sym
        raw = f"{prefix}: {from_act}→{to_act}（{reason}）" if reason else f"{prefix}: {from_act}→{to_act}"
        fn = _format_guardrail_note(raw)
        if fn and fn not in seen:
            seen.add(fn)
            formatted.append(fn)
    return formatted


# [用户授权调整] /report 渲染默认安全天花板由 1500 → 1800 → 2200 → 2800（v5.5.2 用户授权；Prompt 软约束 2100~2500 字）。
def render_morning_report(
    decision: MorningReportDecision,
    max_chars: int = 2800,
    *,
    risk_override: RiskOverrideResult | None = None,
    gate_banner: str = "",
    size_notes: dict[tuple[str, str], str] | None = None,
) -> str:
    """Render MorningReportDecision as inverted-pyramid Telegram report strictly <= max_chars with zero overlap.

    v5.4.0: when ``gate_banner`` is non-empty it is pinned as the FIRST line of section 2
    (【宏观、账户与今日风控红线】) so the deterministic deployment-gate conclusion can never
    be trimmed away by the tail-char truncator.

    v5.4.1: ``size_notes`` carries Python-derived share counts keyed by (ACCOUNT, SYMBOL); the
    model only ever supplies ``size_hint`` (percentage / GBP amount / 全出), never share counts.

    v5.5.1: section 1 is deterministically split into 🏛️ ISA 价值账户 / ⚡ Invest 右侧账户
    blocks inside the SAME Telegram message (no extra message, no protocol change), and the
    default ceiling is raised to 2200 chars (explicit ``max_chars`` callers still win).

    v5.5.2: default ceiling raised to 2800 chars (still far below Telegram's 3800 split line)
    so the report can read naturally instead of using telegraph-style abbreviations.
    """
    ro_decision = getattr(risk_override, "decision", None) if risk_override is not None else None
    if isinstance(ro_decision, MorningReportDecision):
        decision = ro_decision

    audited_orders = {(a.account.strip().upper(), a.symbol.strip().upper()) for a in decision.pending_order_audits}
    filtered_actions = [
        act
        for act in decision.top_actions
        if not (
            act.action.strip().upper() in {"CANCEL_ORDER", "MODIFY_ORDER", "KEEP_ORDER"}
            and (act.account.strip().upper(), act.symbol.strip().upper()) in audited_orders
        )
    ]

    notes = size_notes or {}
    action_lines = [f"📌 今日定调：{decision.headline}"]

    def _acct_bucket(raw_account: Any) -> str:
        up = str(raw_account or "").strip().upper().replace(" ", "")
        if "INVEST" in up:
            return "INVEST"
        if "ISA" in up:
            return "ISA"
        return "OTHER"

    def _action_line(idx: int, act: Any) -> str:
        pz = f" @ {act.price_zone}" if act.price_zone else ""
        act_up = act.action.strip().upper()
        hint = act.size_hint.strip() if act_up in {"ADD", "TRIM", "STOP"} else ""
        note = notes.get((act.account.strip().upper(), act.symbol.strip().upper()), "")
        if act_up in {"ADD", "TRIM"}:
            size_txt = f" · {hint}" if hint else " · 数量未定"
        else:
            size_txt = f" · {hint}" if hint else ""
        note_txt = f" ({note})" if note else ""
        return f"{idx}. [{act.account}] {act.action} {act.symbol}{pz}{size_txt}{note_txt} — {act.reason}"

    def _audit_line(aud: Any) -> str:
        return f"• [{aud.account}] {aud.symbol} ({aud.order_summary}) →【{aud.verdict}】{aud.reason}"

    action_buckets: dict[str, list[Any]] = {"ISA": [], "INVEST": [], "OTHER": []}
    for act in filtered_actions[:4]:
        action_buckets[_acct_bucket(getattr(act, "account", ""))].append(act)
    audit_buckets: dict[str, list[Any]] = {"ISA": [], "INVEST": [], "OTHER": []}
    for aud in decision.pending_order_audits:
        audit_buckets[_acct_bucket(getattr(aud, "account", ""))].append(aud)

    # Global / cross-account rows (e.g. CASH HOLD, 双账户) stay un-split right under the headline.
    for idx, act in enumerate(action_buckets["OTHER"], 1):
        action_lines.append(_action_line(idx, act))
    for aud in audit_buckets["OTHER"]:
        action_lines.append(_audit_line(aud))

    exec_core_items = [
        item
        for item in decision.core_deployment
        if str(getattr(item, "verdict", "APPROVE") or "APPROVE").strip().upper() in {"APPROVE", "SHRINK"}
    ]

    isa_block: list[str] = [
        _action_line(idx, act) for idx, act in enumerate(action_buckets["ISA"], 1)
    ]
    if exec_core_items:
        isa_block.append("【核心宽基部署】")
        for item in exec_core_items[:1]:
            verdict = str(getattr(item, "verdict", "APPROVE") or "APPROVE").strip().upper()
            verdict_tag = " · 缩减" if verdict == "SHRINK" else ""
            isa_block.append(
                f"• [{item.account}] BUY {item.symbol} @ {item.price_zone} · {item.size_hint}"
                f"{verdict_tag} — {item.reason}"
            )
    for aud in audit_buckets["ISA"]:
        isa_block.append(_audit_line(aud))

    invest_block: list[str] = [
        _action_line(idx, act) for idx, act in enumerate(action_buckets["INVEST"], 1)
    ]
    for aud in audit_buckets["INVEST"]:
        invest_block.append(_audit_line(aud))

    if isa_block or invest_block:
        if isa_block:
            action_lines.append("🏛️ ISA 价值账户")
            action_lines.extend(isa_block)
        else:
            action_lines.append("🏛️ ISA 价值账户：今日持仓观望，核心定投/安全边际待命")
        if invest_block:
            action_lines.append("⚡ Invest 右侧账户")
            action_lines.extend(invest_block)
        else:
            action_lines.append("⚡ Invest 右侧账户：今日无右侧新动作，现金吃息待命")

    # Guardrail / discipline notes always sit at the very END of section 1 so tail trimming
    # of later sections can never hide a hard Python override.
    override_notes = _extract_override_notes(risk_override)
    for note in override_notes[:4]:
        action_lines.append(f"• {note}")

    sec1 = "【今日核心行动与挂单裁决】\n" + "\n".join(action_lines)

    veto_str = (
        "；".join(decision.daily_vetoes[:4])
        if decision.daily_vetoes
        else "相对基准 VWRP 跟踪，严守 ISA 安全边际 / Invest 动态 R:R 与最小建仓额，不追高。"
    )
    sec2 = (
        "【宏观、账户与今日风控红线】\n"
        + (f"{gate_banner.strip()}\n" if gate_banner.strip() else "")
        + f"• 宏观与现金：{decision.macro_and_cash}\n"
        + f"• 双账户与集中度：{decision.account_diagnosis}\n"
        + f"• 今日风控红线：{veto_str}"
    )

    covered_syms: set[str] = set()
    for act in filtered_actions:
        sym_up = act.symbol.strip().upper()
        if sym_up and sym_up != "CASH":
            covered_syms.add(sym_up)
            covered_syms.add(sym_up.split(".")[0])
    for aud in decision.pending_order_audits:
        sym_up = aud.symbol.strip().upper()
        if sym_up and sym_up != "CASH":
            covered_syms.add(sym_up)
            covered_syms.add(sym_up.split(".")[0])
    for dep in exec_core_items:
        sym_up = dep.symbol.strip().upper()
        if sym_up and sym_up != "CASH":
            covered_syms.add(sym_up)
            covered_syms.add(sym_up.split(".")[0])

    pos_lines: list[str] = []
    if decision.earnings_watch:
        pos_lines.append("• 近7日财报预警：" + " | ".join(decision.earnings_watch[:4]))
    if decision.trade_review:
        pos_lines.append(f"• 操作复盘：{decision.trade_review}")
    for raw_item in decision.flagged_positions[:5]:
        item = _as_flagged_position_item(raw_item)
        if item is None:
            continue
        sym = item.symbol.strip()
        summary = item.summary.strip()
        if not sym and not summary:
            continue
        if sym:
            sym_up = sym.upper()
            sym_base = sym_up.split(".")[0]
            if "/" not in sym_up and (sym_up in covered_syms or sym_base in covered_syms):
                continue
            if not summary:
                pos_lines.append(f"• {sym}")
            elif re.match(rf"^{re.escape(sym)}\s*[:：]", summary, flags=re.IGNORECASE):
                pos_lines.append(f"• {summary}")
            else:
                pos_lines.append(f"• {sym}: {summary}")
        else:
            pos_lines.append(f"• {summary}")
    if decision.hold_summary:
        pos_lines.append(f"• 常规持有：{decision.hold_summary}")

    sec3 = "【财报预警、复盘与重点异动】\n" + "\n".join(pos_lines) if pos_lines else ""
    return _sanitize_plain_output(_fit_max_chars([sec1, sec2, sec3], max_chars=max_chars))


def render_pick_report(
    decision: PickDecision,
    max_chars: int = 1000,
    *,
    risk_override: RiskOverrideResult | None = None,
) -> str:
    """Render PickDecision as inverted-pyramid Telegram pick brief strictly <= max_chars.

    Always renders the caller-supplied `decision` as the source of truth so post-override
    steps (such as deployment-gate channel filtering) are never clobbered; `risk_override`
    is used solely to surface guardrail notes.
    """
    guardrail_notes = _extract_override_notes(risk_override)
    seen_guardrails = set(guardrail_notes)
    for rej in decision.rejected_or_frozen:
        if _GUARDRAIL_TAG in rej:
            fn = _format_guardrail_note(rej)
            if fn and fn not in seen_guardrails:
                seen_guardrails.add(fn)
                guardrail_notes.append(fn)

    sec1_lines = [f"📌 本轮结论：{decision.headline}"]
    if not decision.selected_actions:
        sec1_lines.append("• ⚪ 本轮无 R:R≥2:1 标的，现金待命待回踩（非永久空仓）。")
    else:
        for idx, item in enumerate(decision.selected_actions[:3], 1):
            sec1_lines.append(
                f"{idx}. [{item.account}] {item.action} {item.symbol}\n"
                f"   逻辑：{item.thesis}\n"
                f"   区间：{item.entry_zone} | 止损：{item.stop_condition} | 目标：{item.target_window}"
            )
            if str(getattr(item, "risk_note", "") or "").strip():
                sec1_lines.append(f"   校验：{item.risk_note.strip()}")
    sec1 = "【本轮精选行动（四要素直给）】\n" + "\n".join(sec1_lines)

    # v5.5.5: Python-determined guardrail rows live in their own audited section so the
    # semantic linter never mistakes them for a rejected-name leak inside the conclusion block.
    sec_guard = ""
    if guardrail_notes:
        guard_lines = [f"• {gnote}" for gnote in guardrail_notes[:4]]
        sec_guard = "【风控护栏记录】\n" + "\n".join(guard_lines)

    sec2_lines = []
    if decision.consistency_and_cash_note:
        sec2_lines.append(f"• 基线与现金：{decision.consistency_and_cash_note}")
    sec2 = "【现金约束与风控基线】\n" + "\n".join(sec2_lines) if sec2_lines else ""

    return _sanitize_plain_output(_fit_max_chars([sec1, sec_guard, sec2], max_chars=max_chars))
