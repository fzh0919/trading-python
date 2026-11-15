"""Aligned /pick & Weekly Radar: Inherits /report guardrails -> Screener -> Researcher -> Risk Critic (<=1000 chars)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction
from src.agents.schemas import PickDecision, parse_or_repair_structured_json, render_pick_report
from src.agents.tools import MarketToolkit, run_forced_market_grounding
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.analytics import compute_exposure_summary, compute_technical_snapshot
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import (
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import get_us_market_schedule_uk, is_uk_bank_holiday
from src.engine.portfolio_rules import evaluate_portfolio, format_evaluations_text
from src.engine.screener import run_quant_screener
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


async def generate_pick_radar(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    focus_query: str = "",
    is_weekly_auto: bool = False,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Run incremental opportunity scan synchronized with today's /report guardrails."""
    cfg.reload_pools_if_changed()
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 账户读取失败，无法生成选股雷达：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    # TODO(IMP-13): [接自 IMP-02 & IMP-04 & IMP-05 & IMP-06] 传 isin_map、pos.created_at 与 bundle；复用 held_histories 传入 run_quant_screener(..., existing_histories=...)；
    # 注入 db.get_recent_lessons()/get_symbol_lessons()、cfg.strategy.settlement_holding_days、bundle.identity_summary 与 get_market_calendar_alerts()。
    tracked_syms = portfolio.unique_yahoo_symbols(include_pending=True)
    ((held_histories, held_bundles), macro, (isa_cands, invest_cands), (_, morning_conclusion), guardrails) = await asyncio.gather(
        market_data.get_bundles(tracked_syms, include_news=True, include_earnings=True),
        market_data.get_macro_snapshot(),
        run_quant_screener(
            portfolio=portfolio,
            market_data=market_data,
            universe=cfg.universe,
            watchlist=cfg.watchlist,
            strategy=cfg.strategy,
            top_n_per_track=8,
        ),
        db.get_morning_context(),
        db.get_daily_guardrails(),
    )

    tech_map = {}
    for sym in tracked_syms:
        snap = compute_technical_snapshot(sym, held_histories.get(sym))
        if snap is not None:
            tech_map[sym] = snap

    # TODO(IMP-13): [接自 IMP-07] evaluate_portfolio 可选传入 histories=held_histories；
    evals = evaluate_portfolio(portfolio, tech_map, held_bundles, cfg.strategy)
    evals_text = format_evaluations_text(evals)
    exposure_text = compute_exposure_summary(portfolio, held_bundles)
    trade_ctx = build_trade_context(portfolio, max_recent_lines=6, max_summary_lines=6, bundles=held_bundles)

    isa_cands_text = "\n".join(c.summary_line for c in isa_cands) or "（ISA 候选池暂无达标标的）"
    invest_cands_text = "\n".join(c.summary_line for c in invest_cands) or "（Invest 候选池暂无达标标的）"
    focus_line = (
        f"用户指定扫描方向：【{focus_query}】（若预设列表未覆盖，请务必调用 inspect_ticker 与 check_portfolio_correlation 实查该方向的具体股票或 LSE ETF！）"
        if focus_query.strip()
        else (
            "本周美股已收盘，请为下周挑选【增量高赔率机会或优胜劣汰换仓】。"
            if is_weekly_auto
            else "主动增量扫描（美/英/欧股与 LSE 英镑 ETF + 优胜劣汰换仓机会）。"
        )
    )
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（今日尚未运行 /report，请独立执行全部风控红线检查）"
    us_sched = get_us_market_schedule_uk()
    uk_hol_note = "｜🇬🇧 英国 Bank Holiday（LSE 休市，英股/ETF 参考下一开盘日）" if is_uk_bank_holiday() else ""
    market_phase_note = (
        f"当前英国时间 {us_sched.uk_date}（{us_sched.uk_weekday_label}），美股阶段：{us_sched.phase}"
        + ("（周末/休市：所有建议均为下个开盘日/限价挂单参考）" if us_sched.phase in {"WEEKEND", "HOLIDAY"} else "")
        + uk_hol_note
    )
    total_assets = max(portfolio.total_combined_value, 1.0)
    total_cash_pct = (portfolio.total_combined_cash / total_assets) * 100.0
    total_available = portfolio.isa.available_to_trade + portfolio.invest.available_to_trade
    exact_cash_summary = (
        f"双账户总净值 £{portfolio.total_combined_value:,.2f} | "
        f"总现金 £{portfolio.total_combined_cash:,.2f} (占总资产 {total_cash_pct:.1f}%) | "
        f"立即可用 £{total_available:,.2f} (ISA £{portfolio.isa.available_to_trade:,.2f} + Invest £{portfolio.invest.available_to_trade:,.2f}) | "
        f"挂单锁定 ISA £{portfolio.isa.reserved_cash:,.2f} / Invest £{portfolio.invest.reserved_cash:,.2f}"
    )

    cand_syms = list(
        dict.fromkeys([*(c.symbol for c in isa_cands[:6]), *(c.symbol for c in invest_cands[:6])])
    )
    grounding_brief = (
        await run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.search_model,
            scope="pick",
            symbols=cand_syms,
            focus_query=focus_query,
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
        )
        if cfg.models.enable_forced_search_grounding
        else ""
    )

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
    )

    # ---------- Stage 1: Incremental Opportunity Researcher ----------
    researcher_sys = build_agent_system_instruction(
        "你是增量选股投研员（Researcher Agent）。存量持仓体检与老挂单审计已由 `/report` 完成，你专注于结合今日风控红线，"
        "从全市场（美/英/欧个股与 LSE 英镑 ETF）中挖掘最值得新买入的配置或高确定性卖弱换强机会。\n"
        "请给出不超过 3 条核心提案（ISA 新股或 LSE ETF / Invest 短线催化 / 必要时的卖弱换强），"
        "每条写清逻辑、介入区间、止损、目标；若上文初筛候选与实时 Google Search 摘要已提供充分事实，禁止无故重复调用工具；仅当用户指定新方向不在预设列表或核心数据缺失时，才调用 `inspect_ticker`、`check_portfolio_correlation` 或 `search_market_news`。",
        cfg.strategy,
    )
    researcher_prompt = (
        f"【任务：增量选股投研员】\n"
        f"时段状态：{market_phase_note}\n"
        f"方向：{focus_line}\n\n"
        f"=== 今日 /report 审计基线与风控红线（必须严格继承，除非有盘中新数据证明必须修正）===\n"
        f"早报基调：{morning_conclusion or '（暂无）'}\n"
        f"红线清单：\n{veto_block}\n\n"
        f"【精确现金底座（禁止自行估算比例）】：{exact_cash_summary}\n"
        f"【宏观快照】：{macro.to_text()}\n"
        f"【行业与跨账户暴露】：\n{exposure_text}\n"
        f"【ISA 账户概览】：{format_account_overview(portfolio.isa).splitlines()[0]}\n"
        f"【Invest 账户概览】：{format_account_overview(portfolio.invest).splitlines()[0]}\n"
        f"【活动挂单】：\n{trade_ctx['pending_orders_text']}\n"
        f"【现有持仓触发信号】：\n{evals_text}\n\n"
        f"【Python 初筛 Top 候选 — ISA 长线赛道（含 LSE ETF 与美/英/欧龙头）】：\n{isa_cands_text}\n\n"
        f"【Python 初筛 Top 候选 — Invest 短线机动赛道】：\n{invest_cands_text}"
        + (f"\n\n【实时 Google Search 候选与方向催化摘要】：\n{grounding_brief}" if grounding_brief else "")
    )

    research_output = await llm.generate(
        model=cfg.models.pick_researcher_model,
        prompt=researcher_prompt,
        system_instruction=researcher_sys,
        thinking_level=cfg.models.pick_researcher_thinking_level,
        tools=toolkit.declarations(),
        tool_executor=toolkit.as_executor(),
        temperature=0.25,
        max_tool_rounds=4,
        usage_label="pick_researcher",
    )
    if research_output.startswith("⚠️"):
        return research_output

    # ---------- Stage 2: Risk & Bear Critic -> Structured PickDecision ----------
    critic_sys = build_agent_system_instruction(
        "你是独立风控官（Risk & Bear Critic）。请按四重护栏与今日 `/report` 红线严格审核投研员提案，"
        "直接输出最终结构化 JSON Decision（渲染后将限制在 1000 字内，结论置顶、废话为零）。\n\n"
        "【四重一票否决硬护栏】：\n"
        f"1. 现金与 {cfg.strategy.cash_interest_rate_pct:.1f}% 基准 + 交易摩擦（美欧往返 {cfg.strategy.fx_fee_pct * 2:.2f}% / 英股买入 {cfg.strategy.uk_stamp_duty_pct:.1f}%印花税）：跑不赢现金利息的平庸机会直接枪毙。\n"
        f"2. 相关性与行业红线：与当前持仓 60 日相关性 >= {cfg.strategy.overlap_correlation_threshold:.2f}、或违背 `/report` 今日红线名单的标的必须驳回。\n"
        "3. 财报/过热排雷：5日内财报、RSI>=72 追高全部否决或降级。\n"
        "4. 一致性、精确现金与去重：严格继承 `/report` 红线（仅用提供的精确现金数字与比例，禁止自行瞎编比例）；"
        "若当前为周末/休市，请在 headline 标明“下周一开盘/限价挂单参考”；"
        "`rejected_or_frozen` 只列本轮选股新否决的候选股或方向，不要把早报已裁决的老挂单再复述一遍；"
        "若无优质机会，直接让 `selected_actions` 为空数组拿 4% 利息。",
        cfg.strategy,
    )
    critic_prompt = (
        f"【时段状态】：{market_phase_note}\n"
        f"【今日 /report 基线】：{morning_conclusion or '（暂无）'}\n"
        f"【今日红线】：\n{veto_block}\n\n"
        f"【精确现金底座】：{exact_cash_summary}\n"
        f"【行业暴露】：\n{exposure_text}\n\n"
        f"【投研员提案（待审）】：\n{research_output}"
    )

    raw_critic = await llm.generate(
        model=cfg.models.pick_model,
        prompt=critic_prompt,
        system_instruction=critic_sys,
        thinking_level=cfg.models.pick_thinking_level,
        response_schema=PickDecision,
        temperature=0.15,
        usage_label="pick_critic",
    )

    decision, resolved_critic_raw = await parse_or_repair_structured_json(
        raw_critic,
        PickDecision,
        llm=llm,
        repair_model=cfg.models.fallback_model,
        thinking_level="LOW",
        usage_label="pick_critic_repair",
        ops_notifier=ops_notifier,
        alert_source="pick_critic",
    )
    now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")

    # TODO(IMP-13): [接自 IMP-07] decision 非空时调用 apply_risk_override_guardrail(decision, portfolio=portfolio, macro=macro, strategy=cfg.strategy, bundle_map=held_bundles, high_correlation_symbols=..., daily_vetoes=morning_vetoes) 做后置强校验。
    if decision is not None:
        final_text = render_pick_report(decision, max_chars=1000)
        parsed_picks = [
            {
                "account": item.account,
                "symbol": item.symbol,
                "action": item.action,
                "entry_price": item.entry_price if item.entry_price > 0 else None,
                "thesis": item.thesis,
            }
            for item in decision.selected_actions
        ]
    else:
        logger.warning("Pick Critic structured parse fell back to plain text.")
        final_text = resolved_critic_raw.strip()[:1000]
        parsed_picks = []

    if not final_text.startswith("⚠️"):
        await db.save_pick_radar(focus_query, research_output, final_text, now_uk)
        if parsed_picks:
            await db.log_recommendations(parsed_picks)

    return final_text
