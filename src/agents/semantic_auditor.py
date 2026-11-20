"""Read-only semantic linter sidecar for /report and /pick rendered output (DeepSeek Flash).

Design invariants:
- Zero mutation: the rendered Telegram text is never rewritten here.
- Deterministic first: hard rules (eg. /pick headline leaking a rejected ticker) are checked in Python.
- Silent degradation: DS timeout/exception/non-JSON -> `skipped=True`, no alerts, no blocking.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend
from src.config import AppConfig, StrategySection

logger = logging.getLogger(__name__)

_TICKER_TOKEN_RE = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b")
_SCOPE_LABELS = {"report": "/report 早报", "pick": "/pick 选股雷达"}

_SYSTEM = (
    "你是只读合规语义质检员（Semantic Linter）。你只做一件事：核对渲染文本是否违反下列语义红线，绝不改写文本。\n"
    "红线：\n"
    "1. 幻觉事实价：文本以【事实口吻】陈述现价/持仓成本/止损价/挂单价/目标价，但数值不在【原始数字底座】中，"
    "或 GBp 与 GBP 混用导致 100 倍量级错位。\n"
    "   • 底座只含「已存在的真实数字」；今日新建议的价位天然不在底座中，不得据此判违规。\n"
    "   • ADD/BUY/TRIM/减仓/清仓/撤单改挂 后紧跟的价位属于【建议执行价】，只校验方向："
    "买入建议价明显高于底座现价（>3% 且未写明突破追买理由）或卖出建议价低于底座现价才算异常。\n"
    "   • 底座中若没有该标的现价，禁止据此判违规。\n"
    "2. 违背挂单计息：建议撤销/放弃限价买单以便吃现金利息，或声称未成交挂单不再计息。\n"
    "3. 内部驳回外泄：仅当「📌 本轮结论：」标题行或推荐执行条目出现未入选/被否决/被冻结的标的代码或方向时算违规；"
    "独立「【风控护栏记录】」区中由 Python 确定的「风控护栏修正」或「→拦截」行属于合规审计披露，严禁判为违规。\n"
    "4. 与今日红线冲突：建议与 daily_vetoes 明确冲突的加仓或开仓。\n"
    "只报告有明确文本证据的违规，每条 <=60 字并引用原句片段；没有违规就返回空数组。\n"
    "违规条目加前缀分类：[事实价不符] 或 [建议价方向异常]。\n"
    "仅输出严格 JSON：{\"passed\": true/false, \"violations\": [\"...\"]}"
)


class SemanticAuditResult(BaseModel):
    passed: bool = True
    violations: list[str] = Field(default_factory=list)
    skipped: bool = False


def _fmt_num(value: Any, fmt: str = ",.2f") -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "N/A"


def build_fact_baseline(
    *,
    scope: str,
    portfolio: Any,
    evals: list[Any] | None = None,
    trade_ctx: dict[str, str] | None = None,
    guardrails: list[str] | None = None,
    fact_sheet: str = "",
    candidate_bundles: dict[str, Any] | None = None,
) -> str:
    """Assemble a deterministic numeric baseline (Python/SQLite truth) for the linter to compare against."""
    lines = [f"【原始数字底座（只含已存在的事实价） | {_SCOPE_LABELS.get(scope, scope)}】"]
    lines.append(
        f"账户：总净值 £{_fmt_num(getattr(portfolio, 'total_combined_value', None))}"
        f" | 总现金 £{_fmt_num(getattr(portfolio, 'total_combined_cash', None))}"
    )
    for label in ("isa", "invest"):
        acct = getattr(portfolio, label, None)
        if acct is None:
            continue
        acct_val = float(getattr(acct, "total_account_value", 0.0) or 0.0)
        available = float(getattr(acct, "available_to_trade", 0.0) or 0.0)
        total_cash = float(getattr(acct, "total_cash", 0.0) or 0.0)
        reserved = float(getattr(acct, "reserved_cash", 0.0) or 0.0)
        if acct_val > 0:
            avail_pct = (available / acct_val) * 100.0
            cash_pct = (total_cash / acct_val) * 100.0
            line = (
                f"{label.upper()}：可用 {_fmt_num(available)} (可用占比 {avail_pct:.1f}%)"
                f" | 挂单占用 {_fmt_num(reserved)}"
                f" | 总现金 {_fmt_num(total_cash)} (总现金占比 {cash_pct:.1f}%)"
            )
        else:
            line = (
                f"{label.upper()}：可用 {_fmt_num(available)}"
                f" | 挂单占用 {_fmt_num(reserved)}"
                f" | 总现金 {_fmt_num(total_cash)}"
            )
        lines.append(line)
    for label in ("isa", "invest"):
        acct = getattr(portfolio, label, None)
        if acct is None:
            continue
        positions = getattr(acct, "positions", None) or []
        if not positions:
            continue
        display_label = "ISA" if label == "isa" else "Invest"
        lines.append(f"{display_label} 全持仓成本/现价表（跨账户同名标的逐一列出）：")
        for pos in positions:
            sym = str(getattr(pos, "yahoo_symbol", "") or getattr(pos, "ticker", "") or "").strip()
            if not sym:
                continue
            ccy = str(getattr(pos, "inst_currency", "") or "").strip()
            lines.append(
                f"• {display_label} {sym} | 均价 {_fmt_num(getattr(pos, 'avg_price', None))} {ccy}"
                f" | 现价 {_fmt_num(getattr(pos, 'cur_price', None))} {ccy}"
                f" | 收益率 {_fmt_num(getattr(pos, 'return_pct', None), '+.2f')}%"
            )
    pending_text = str((trade_ctx or {}).get("pending_orders_text") or "").strip()
    if pending_text:
        lines.append("活动挂单（未成交仍计息）：\n" + pending_text[:1200])
    if scope == "pick" and candidate_bundles:
        # v5.5.7/v5.5.9 M2: anchor this round's new picks (price + calibrated sell-side mean/high)
        # so the linter catches GBp/GBP 100x slips without false-flagging dual-anchor target citations.
        from src.engine.portfolio_rules import _calibrate_target_price

        anchor_lines: list[str] = []
        for sym, bundle in list(candidate_bundles.items())[:12]:
            price = getattr(bundle, "current_price", None)
            if price is None or float(price) <= 0:
                continue
            ccy = str(getattr(bundle, "currency", "") or "")
            line = f"• {sym} | 现价 {_fmt_num(price)} {ccy} | 币种 {ccy or '待核'}"
            if (getattr(bundle, "analyst_count", 0) or 0) >= 3:
                mean_raw = getattr(bundle, "target_mean_price", None)
                high_raw = getattr(bundle, "target_high_price", None)
                if mean_raw is not None and float(mean_raw) > 0:
                    mean_c = _calibrate_target_price(
                        float(mean_raw), float(price), symbol=str(sym), currency=ccy
                    )
                    line += f" | 卖方均值 {_fmt_num(mean_c)}"
                if high_raw is not None and float(high_raw) > 0:
                    high_c = _calibrate_target_price(
                        float(high_raw), float(price), symbol=str(sym), currency=ccy
                    )
                    line += f" | 卖方高位 {_fmt_num(high_c)}"
            anchor_lines.append(line)
        if anchor_lines:
            lines.append("本次入选新标的现价锚点（含币种）：\n" + "\n".join(anchor_lines))
    if evals:
        try:
            from src.engine.portfolio_rules import format_evaluations_text

            eval_text = format_evaluations_text(list(evals), compact_hold=True)
        except Exception:  # noqa: BLE001 - baseline building must never break the caller
            eval_text = ""
        if eval_text:
            lines.append("持仓规则：\n" + eval_text[:1200])
    if guardrails:
        lines.append("今日红线：\n" + "\n".join(f"• {g}" for g in guardrails))
    sheet = str(fact_sheet or "").strip()
    if sheet:
        # Keep the whole FactSheet: with ~20 holdings the tail prices fall beyond 4k chars and a truncated
        # baseline makes the linter report false "hallucinated price" violations.
        lines.append("FactSheet（唯一数值真源）：\n" + sheet[:12000])
    return "\n".join(lines)


def _detect_pick_headline_leak(
    headline: str,
    selected_symbols: set[str],
    rejected_symbols: set[str],
) -> list[str]:
    """0-token hard rule: a rejected candidate ticker must never surface in the /pick headline."""
    if not headline or not rejected_symbols:
        return []
    violations: list[str] = []
    for match in _TICKER_TOKEN_RE.finditer(headline.upper()):
        token = match.group(1)
        if token in rejected_symbols and token not in selected_symbols and token not in violations:
            violations.append(f"/pick headline 提及未入选标的 {token}，疑似内部驳回外泄")
    return violations


def _strategy_hint(strategy: StrategySection) -> str:
    return (
        "【策略基准】现金利息 GBP "
        f"{_fmt_num(getattr(strategy, 'cash_interest_rate_pct', None), '.1f')}% / USD "
        f"{_fmt_num(getattr(strategy, 'usd_cash_interest_rate_pct', None), '.1f')}%，"
        f"换汇 {_fmt_num(getattr(strategy, 'fx_fee_pct', None), '.2f')}%/边，"
        f"英股印花税 {_fmt_num(getattr(strategy, 'uk_stamp_duty_pct', None), '.1f')}%"
    )


async def _emit_violations(result: SemanticAuditResult, *, scope: str, ops_notifier: Any) -> None:
    if not result.violations:
        return
    full_text = sanitize_plain_text("；".join(result.violations), max(400, sum(len(v) for v in result.violations) + 16))
    joined = sanitize_plain_text("；".join(result.violations), 400)
    logger.warning("Semantic audit (%s) flagged %d violation(s): %s", scope, len(result.violations), full_text)
    if ops_notifier is None:
        return
    try:
        await ops_notifier.notify(
            severity="WARN",
            source=f"semantic_audit:{scope}",
            alert_key=f"semantic_audit_{scope}",
            message=joined,
        )
    except Exception as exc:  # noqa: BLE001 - alerting must never break the main pipeline
        logger.warning("Semantic audit (%s) failed to emit ops alert: %s", scope, exc)


async def audit_generated_output(
    llm: LLMBackend | None,
    *,
    cfg: AppConfig,
    scope: str,
    fact_baseline: str,
    rendered_text: str,
    strategy: StrategySection,
    timeout_s: float | None = None,
    ops_notifier: Any = None,
    headline: str = "",
    selected_symbols: set[str] | None = None,
    known_symbols: set[str] | None = None,
) -> SemanticAuditResult:
    """Audit rendered text against 4 semantic red lines. Read-only: never edits or blocks the output."""
    text = str(rendered_text or "")
    if not text.strip() or text.lstrip().startswith("⚠️"):
        return SemanticAuditResult(skipped=True)
    if not cfg.models.ds_semantic_audit_enabled:
        return SemanticAuditResult(skipped=True)

    violations: list[str] = []
    if scope == "pick":
        violations = _detect_pick_headline_leak(
            headline,
            {str(s).strip().upper() for s in (selected_symbols or set())},
            {str(s).strip().upper() for s in (known_symbols or set())},
        )

    if not ds_helper_available(llm, cfg):
        result = SemanticAuditResult(passed=not violations, violations=violations, skipped=True)
        await _emit_violations(result, scope=scope, ops_notifier=ops_notifier)
        return result

    prompt = (
        f"{_strategy_hint(strategy)}\n\n"
        f"{fact_baseline}\n\n"
        "=== 待质检渲染文本（Telegram 原文，禁止改写）===\n"
        f"{text}\n\n"
        "请对照原始数字底座逐条核对上述 4 条红线（严格区分【事实价】与【今日建议执行价】），只输出 JSON。"
    )
    payload = await call_ds_json(
        cfg,
        llm,
        prompt=prompt,
        system_instruction=_SYSTEM,
        timeout_s=timeout_s,
        usage_label=f"semantic_audit_{scope}",
    )
    if payload is None:
        result = SemanticAuditResult(passed=not violations, violations=violations, skipped=True)
        await _emit_violations(result, scope=scope, ops_notifier=ops_notifier)
        return result

    raw_violations = payload.get("violations")
    if isinstance(raw_violations, list):
        for item in raw_violations[:6]:
            cleaned = sanitize_plain_text(str(item), 60)
            if cleaned and cleaned not in violations:
                violations.append(cleaned)

    result = SemanticAuditResult(passed=not violations, violations=violations, skipped=False)
    await _emit_violations(result, scope=scope, ops_notifier=ops_notifier)
    return result

