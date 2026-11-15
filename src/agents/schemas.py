"""Pydantic decision schemas & inverted-pyramid Telegram renderers (adapted from TradingAgents)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ReportActionItem(BaseModel):
    account: str = Field(description="ISA 或 Invest 或 双账户")
    symbol: str = Field(description="现有持仓标的代码，若今日持仓观望无调仓可写 CASH（挂单撤改勿放此处）")
    action: str = Field(description="仅限持仓动作：ADD / TRIM / STOP / HOLD（撤改挂单只放 pending_order_audits，禁止重复放此处）")
    price_zone: str = Field(description="建议触发价位或区间，如 '355-358 USD' 或 '4500 GBp'")
    reason: str = Field(description="一句话核心依据（不超过35字）")


class PendingOrderAuditItem(BaseModel):
    account: str = Field(description="ISA 或 Invest")
    symbol: str = Field(description="挂单标的代码")
    verdict: Literal["KEEP", "MODIFY", "CANCEL"] = Field(description="KEEP（保留）| MODIFY（调低/改价）| CANCEL（建议撤单）")
    order_summary: str = Field(description="极简挂单方向与价位，如 'BUY 1250 GBp' 或 'STOP 93.80 USD'")
    reason: str = Field(description="结合现价、均线、目标价及印花税/换汇成本给出的一句话审计结论（不超过38字）")


class MorningReportDecision(BaseModel):
    headline: str = Field(description="一句话定调今日核心策略与现金态度（不超过45字）")
    top_actions: list[ReportActionItem] = Field(
        default_factory=list,
        description="今日现有持仓最优先的 1~3 条调仓动作（不含撤单/改单）；若持仓全部不动，给1条 CASH HOLD 即可",
    )
    pending_order_audits: list[PendingOrderAuditItem] = Field(
        default_factory=list,
        description="对所有未成交活动挂单逐一做出 KEEP / MODIFY / CANCEL 独立审计（没有挂单则为空数组）",
    )
    daily_vetoes: list[str] = Field(
        default_factory=list,
        description="2~4条今日硬性风控红线（供后续/pick与/us严格继承，每条不超过35字）",
    )
    macro_and_cash: str = Field(description="宏观利率/VIX/汇率与4%现金安全垫精炼要点（不超过90字）")
    account_diagnosis: str = Field(description="ISA与Invest净值、FX影响、行业/跨账户集中度痛点（不超过130字）")
    flagged_positions: list[str] = Field(
        default_factory=list,
        description="仅列出【未在 top_actions 与 pending_order_audits 中出现过】的其他异动/破位/观察持仓，每条<=40字（最多4条，严禁与上方重复）",
    )
    hold_summary: str = Field(description="一句话合并概括其余常规持有不动仓位（不超过55字）")
    trade_review: str = Field(description="近7-30天实盘操作与摩擦一句话短评（不超过70字）")
    earnings_watch: list[str] = Field(
        default_factory=list,
        description="近7日财报预警与应对预案（每条<=30字，无则为空数组）",
    )


class PickActionItem(BaseModel):
    account: str = Field(description="ISA 或 Invest")
    symbol: str = Field(description="标的代码，如 VWRP.L、SHEL.L、TSM")
    action: Literal["BUY", "ADD", "TRIM", "SWAP"] = Field(description="BUY / ADD / TRIM / SWAP")
    thesis: str = Field(description="交易逻辑与催化，含扣费与4%利息对比优势（不超过55字）")
    entry_zone: str = Field(description="建议买入或加减仓价位区间与建议金额")
    stop_condition: str = Field(description="硬止损位或证伪条件")
    target_window: str = Field(description="目标位或观察周期")
    entry_price: float = Field(default=0.0, description="当前参考现价数字（用于数据库记录，未知填0）")


class PickDecision(BaseModel):
    headline: str = Field(description="一句话定调本轮选股/换仓结论与现金安排（不超过55字）")
    selected_actions: list[PickActionItem] = Field(
        default_factory=list,
        description="通过风控官四重护栏的最终推荐（0~3条）；若均未达标请返回空数组",
    )
    rejected_or_frozen: list[str] = Field(
        default_factory=list,
        description="1~3条【本轮选股中新探测但被风控驳回】的标的或方向及原因（每条<=35字；勿重复早报已撤的老挂单）",
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


def _fit_max_chars(sections: list[str], max_chars: int) -> str:
    """Keep top priority sections intact and gracefully trim tail sections if over max_chars."""
    text = "\n\n".join(s for s in sections if s.strip())
    if len(text) <= max_chars:
        return text

    # Drop optional/lower sections progressively from the end while keeping sections[0] (top actions)
    kept = [s for s in sections if s.strip()]
    while len(kept) > 1 and len("\n\n".join(kept)) > max_chars:
        last = kept[-1]
        lines = last.splitlines()
        if len(lines) > 2:
            kept[-1] = "\n".join(lines[:-1])
        else:
            kept.pop()

    joined = "\n\n".join(kept)
    if len(joined) > max_chars:
        return joined[: max_chars - 1].rstrip() + "…"
    return joined


# TODO(IMP-10): [接自 IMP-07] 配合 portfolio_rules.apply_risk_override_guardrail() 返回的 RiskOverrideResult.notes/overrides，
# 在 render_morning_report / render_pick_report 中统一高亮展示 [🛡️风控护栏修正] 标记，并升级 flagged_positions 结构化去重。
def render_morning_report(decision: MorningReportDecision, max_chars: int = 1500) -> str:
    """Render MorningReportDecision as inverted-pyramid Telegram report strictly <= max_chars with zero overlap."""
    audited_orders = {(a.account.strip().upper(), a.symbol.strip().upper()) for a in decision.pending_order_audits}
    filtered_actions = [
        act
        for act in decision.top_actions
        if not (
            act.action.strip().upper() in {"CANCEL_ORDER", "MODIFY_ORDER", "KEEP_ORDER"}
            and (act.account.strip().upper(), act.symbol.strip().upper()) in audited_orders
        )
    ]

    action_lines = [f"📌 今日定调：{decision.headline}"]
    for idx, act in enumerate(filtered_actions[:4], 1):
        pz = f" @ {act.price_zone}" if act.price_zone else ""
        action_lines.append(f"{idx}. [{act.account}] {act.action} {act.symbol}{pz} — {act.reason}")

    if decision.pending_order_audits:
        action_lines.append("【活动挂单裁决】")
        for aud in decision.pending_order_audits:
            action_lines.append(
                f"• [{aud.account}] {aud.symbol} ({aud.order_summary}) →【{aud.verdict}】{aud.reason}"
            )
    sec1 = "【今日核心行动与挂单裁决】\n" + "\n".join(action_lines)

    veto_str = "；".join(decision.daily_vetoes[:4]) if decision.daily_vetoes else "严守4%现金门槛，不追高。"
    sec2 = (
        "【宏观、账户与今日风控红线】\n"
        f"• 宏观与现金：{decision.macro_and_cash}\n"
        f"• 双账户与集中度：{decision.account_diagnosis}\n"
        f"• 今日风控红线：{veto_str}"
    )

    covered_syms = {
        act.symbol.strip().upper().split(".")[0] for act in filtered_actions
    } | {
        aud.symbol.strip().upper().split(".")[0] for aud in decision.pending_order_audits
    }
    pos_lines: list[str] = []
    if decision.earnings_watch:
        pos_lines.append("• 近7日财报预警：" + " | ".join(decision.earnings_watch[:4]))
    if decision.trade_review:
        pos_lines.append(f"• 操作复盘：{decision.trade_review}")
    for line in decision.flagged_positions[:5]:
        # Skip position blurbs that merely repeat symbols already given explicit actions/audits above
        prefix = (
            line.strip()
            .lstrip("•-* ")
            .split(":")[0]
            .split("：")[0]
            .split("[")[0]
            .split("(")[0]
            .split("—")[0]
            .strip()
            .upper()
        )
        head_token = prefix.split()[0].split(".")[0] if prefix else ""
        if head_token and head_token in covered_syms and "/" not in head_token:
            continue
        pos_lines.append(f"• {line}")
    if decision.hold_summary:
        pos_lines.append(f"• 常规持有：{decision.hold_summary}")

    sec3 = "【财报预警、复盘与重点异动】\n" + "\n".join(pos_lines)
    return _fit_max_chars([sec1, sec2, sec3], max_chars=max_chars)


def render_pick_report(decision: PickDecision, max_chars: int = 1000) -> str:
    """Render PickDecision as inverted-pyramid Telegram pick brief strictly <= max_chars."""
    sec1_lines = [f"📌 本轮结论：{decision.headline}"]
    if not decision.selected_actions:
        sec1_lines.append("• ⚪ 今日无高赔率新机会通过风控审核，建议保留现金继续享 4.0% 年化利息。")
    else:
        for idx, item in enumerate(decision.selected_actions[:3], 1):
            sec1_lines.append(
                f"{idx}. [{item.account}] {item.action} {item.symbol}\n"
                f"   逻辑：{item.thesis}\n"
                f"   区间：{item.entry_zone} | 止损：{item.stop_condition} | 目标：{item.target_window}"
            )
    sec1 = "【本轮精选行动（四要素直给）】\n" + "\n".join(sec1_lines)

    sec2_lines = []
    if decision.consistency_and_cash_note:
        sec2_lines.append(f"• 基线与现金：{decision.consistency_and_cash_note}")
    for rej in decision.rejected_or_frozen[:4]:
        sec2_lines.append(f"• 驳回/冻结：{rej}")
    sec2 = "【风控审查与现金约束】\n" + "\n".join(sec2_lines) if sec2_lines else ""

    return _fit_max_chars([sec1, sec2], max_chars=max_chars)
