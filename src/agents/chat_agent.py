"""Interactive Telegram daily chat agent (/pos instant snapshot + free-form advisor chat)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction
from src.agents.tools import MarketToolkit
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import (
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import get_uk_weekday_label, get_us_market_schedule_uk, is_uk_bank_holiday
from src.storage.db import AdvisorDatabase


async def format_positions_instant(cfg: AppConfig, t212: T212Client) -> str:
    """Instant deterministic /pos portfolio & 30d trade snapshot without waiting for LLM."""
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 数据获取失败：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    trade_ctx = build_trade_context(portfolio, max_recent_lines=5, max_summary_lines=6)
    lines = [
        f"💼 T212 双账户实时快照（{portfolio.fetched_at_uk} UK | 闲置现金利息 {cfg.strategy.cash_interest_rate_pct:.1f}%）",
        f"合计总净值: GBP {portfolio.total_combined_value:,.2f} | 合计现金: GBP {portfolio.total_combined_cash:,.2f}",
        "",
        "【Stocks ISA（长线稳健仓）】",
        format_account_overview(portfolio.isa),
        format_positions_text(portfolio.isa),
        "",
        "【Invest（短线机动仓）】",
        format_account_overview(portfolio.invest),
        format_positions_text(portfolio.invest),
        "",
        "【当前活动挂单】",
        trade_ctx["pending_orders_text"],
        "",
        "【近7天实盘成交】",
        trade_ctx["recent_trades_text"],
    ]
    return "\n".join(lines)


def _format_chat_history_compact(history_turns: list[tuple[str, str]], recent_full_turns: int = 3) -> str:
    """Keep recent turns concise while compacting older turns to control prompt bloat."""
    if not history_turns:
        return "（今日暂无前序对话）"
    split_idx = max(0, len(history_turns) - recent_full_turns)
    lines: list[str] = []
    for u, a in history_turns[:split_idx]:
        a_one_line = " ".join(a.splitlines()).strip()[:100]
        lines.append(f"• [早期] 问: {u[:60]} → 答: {a_one_line}")
    for u, a in history_turns[split_idx:]:
        lines.append(f"👤 用户: {u}\n🤖 顾问: {a[:380]}")
    return "\n\n".join(lines)


async def answer_user_chat(
    user_query: str,
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Answer arbitrary Telegram user questions using compact T212 context, on-demand tools, guardrails, and daily memory."""
    portfolio = await t212.get_portfolio(use_cache=True)
    trade_ctx = build_trade_context(portfolio, history_days=7, max_recent_lines=3, max_summary_lines=0)

    # TODO(IMP-14): [接自 IMP-01 & IMP-02 & IMP-03 & IMP-12] 接入 db.get_context_pack()（含 symbols[sym].identity_summary/technical/technical_line/bundle_summary/evaluations/disagreements/lessons）局部标的情报切片、
    # db.get_previous_cycle_chat_history() 跨周期摘要、calendar_utils.get_market_calendar_alerts()，并在消息以 /deep 开头时按 cfg.models.chat_deep_model / chat_deep_thinking_level 升档。
    (morning_full_text, morning_conclusion), guardrails, (pick_ts, pick_text), history_turns = await asyncio.gather(
        db.get_morning_context(include_premarket=True),
        db.get_daily_guardrails(),
        db.get_latest_pick_radar(),
        db.get_chat_history(limit=cfg.strategy.max_daily_chat_turns),
    )

    history_block = _format_chat_history_compact(history_turns)
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（无）"
    sched = get_us_market_schedule_uk()
    uk_hol_note = "｜🇬🇧 今日英国 Bank Holiday（LSE 休市）" if is_uk_bank_holiday() else ""
    morning_block = morning_full_text[:2050] if morning_full_text else (morning_conclusion or "（暂无）")
    reset_str = f"{cfg.schedule.morning_report_hour:02d}:{cfg.schedule.morning_report_minute:02d} UK"
    # [用户授权调整] /deep 单聊 Prompt 软约束放宽至 1000~1200 字，普通单聊保持 500 字以内（均不做代码截断）；IMP-14 升级模型路由时请保留。
    is_deep_query = user_query.strip().lower().startswith("/deep")
    length_rule = "1000~1200 字以内（深度解析可展开多维证据与执行预案）" if is_deep_query else "500 字以内"

    chat_sys = build_agent_system_instruction(
        "你是用户的全天候私人量化投资顾问。\n"
        "交互与成本纪律：\n"
        "• 上来第一句先给明确结论与建议，后面再简要解释。\n"
        "• 上下文已包含最新持仓占比/浮盈、活动挂单、今日早报、美股哨兵与风控红线；若已有信息足以回答（如仓位诊断、挂单看法、策略解读），请直接一轮作答，严禁无故调用工具。\n"
        "• 仅当用户询问持仓外的新标的，或确实缺少近期技术面/相关性/突发公告时，才按需调用 `inspect_ticker`、`check_portfolio_correlation` 或 `search_market_news`；禁止连续重复调用相同工具。\n"
        f"• 直白客观（不迎合已有挂单或前序决定），禁用 Markdown # 和 **，用 • 和【】排版，控制在 {length_rule}。",
        cfg.strategy,
    )

    prompt = (
        f"当前英国时间 {sched.uk_date} ({get_uk_weekday_label()}{uk_hol_note})，美股状态：{sched.phase}（常规开盘 {sched.open_time_uk} UK）。\n\n"
        f"【ISA 账户】\n{format_account_overview(portfolio.isa)}\n{format_positions_text(portfolio.isa)}\n\n"
        f"【Invest 账户】\n{format_account_overview(portfolio.invest)}\n{format_positions_text(portfolio.invest)}\n\n"
        f"【当前挂单与近7日成交】\n{trade_ctx['pending_orders_text']}\n{trade_ctx['recent_trades_text']}\n\n"
        f"【今日早报及美股哨兵纪要】：\n{morning_block}\n\n"
        f"【今日风控红线】：\n{veto_block}\n"
        # [用户授权调整] 放宽 /pick 雷达切片至 1050 字符以完整保留精选行动与驳回原因；IMP-14 请保留。
        + (f"【最近一次选股雷达 ({pick_ts} UK)】：\n{pick_text[:1050]}\n\n" if pick_text else "\n")
        + f"【今日对话记录（{reset_str} 晨报周期自动重置）】：\n{history_block}\n\n"
        + f"【用户当前问题】：\n{user_query}"
    )

    # TODO(IMP-14): [接自 IMP-11] 传入 db=db 以激活 inspect_ticker 的历史反思教训查询。
    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
    )
    reply = await llm.generate(
        model=cfg.models.chat_model,
        prompt=prompt,
        system_instruction=chat_sys,
        thinking_level=cfg.models.chat_thinking_level,
        tools=toolkit.declarations(),
        tool_executor=toolkit.as_executor(),
        temperature=0.25,
        max_tool_rounds=3,
        usage_label="chat",
    )

    if not reply.startswith("⚠️"):
        await db.append_chat_turn(
            user_query=user_query,
            bot_reply=reply,
            max_turns=cfg.strategy.max_daily_chat_turns,
        )
    return reply
