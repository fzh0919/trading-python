"""Morning Report (/report) TradingAgents-Lite 3-Step Pipeline & /us Sentinel."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction
from src.agents.schemas import (
    MorningReportDecision,
    parse_or_repair_structured_json,
    render_morning_report,
)
from src.agents.tools import MarketToolkit, run_forced_market_grounding
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_exposure_summary,
    compute_technical_snapshot,
    format_technical_line,
)
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import (
    PortfolioData,
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import get_uk_weekday_label, get_us_market_schedule_uk, is_uk_bank_holiday
from src.engine.portfolio_rules import evaluate_portfolio, format_evaluations_text
from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")


def _compute_portfolio_diff_text(current: PortfolioData, previous: PortfolioData | None) -> str:
    if previous is None:
        return "（首次建档，暂无前一日净值基准；以 30 日实盘成交作依据）"

    def _map_pos(data: PortfolioData) -> dict[str, dict]:
        out = {}
        for acc in (data.isa, data.invest):
            if not acc.ok:
                continue
            for p in acc.positions:
                out[f"{acc.account_label}:{p.ticker}"] = {
                    "qty": p.quantity,
                    "account": acc.account_label,
                    "ticker": p.ticker,
                }
        return out

    prev_map = _map_pos(previous)
    cur_map = _map_pos(current)
    changes: list[str] = []

    val_diff = current.total_combined_value - previous.total_combined_value
    if abs(val_diff) >= 0.01:
        changes.append(
            f"总净值: {previous.total_combined_value:,.2f} → {current.total_combined_value:,.2f} GBP ({val_diff:+,.2f})"
        )
    for k, cur in cur_map.items():
        if k not in prev_map:
            changes.append(f"新增 [{cur['account']}] {cur['ticker']} ({cur['qty']:.4f}股)")
        else:
            dq = cur["qty"] - prev_map[k]["qty"]
            if abs(dq) > 1e-4:
                changes.append(f"{'加仓' if dq > 0 else '减仓'} [{cur['account']}] {cur['ticker']} ({dq:+.4f}股)")
    for k, prev in prev_map.items():
        if k not in cur_map:
            changes.append(f"清仓 [{prev['account']}] {prev['ticker']}")
    return "; ".join(changes) if changes else f"对比 {previous.fetched_at_uk} 无仓位数量变动"


def _extract_conclusion_fallback(report_text: str) -> str:
    m = re.search(r"(?:今日定调|核心结论与策略)[：:]*\s*([\s\S]{0,600})", report_text)
    if m:
        return m.group(1).strip()[:500]
    return report_text[:500].strip()


async def build_portfolio_fact_sheet(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    cfg: AppConfig,
    us_only: bool = False,
) -> tuple[str, dict[str, TechnicalSnapshot], dict[str, TickerMarketBundle], MacroSnapshot]:
    """Gather deterministic data for held positions AND pending orders + exposure breakdown."""
    all_tracked = portfolio.unique_yahoo_symbols(include_pending=True)
    held_all = portfolio.unique_yahoo_symbols(include_pending=False)
    # For US sentinel mode, focus technicals/news strictly on US tickers while keeping macro & exposure context
    tracked_syms = [s for s in all_tracked if "." not in s and "=" not in s and "^" not in s] if us_only else all_tracked
    held_syms_set = set(held_all)

    fetch_syms = all_tracked if (us_only and len(all_tracked) > len(tracked_syms)) else tracked_syms
    # TODO(IMP-12): [接自 IMP-04 & IMP-05 & IMP-06] get_bundles 传 isin_map，compute_technical_snapshot 传入 pos.created_at (兼容 None) 与 bundle；
    # 整合 bundle.identity_summary / CYQ / 周线 / 除息 / Disagreement 组装 AnalysisContextPack 并存入 save_morning_report(..., context_pack=...)；
    # /us 哨兵接入 cfg.strategy.us_anomaly_move_pct / us_anomaly_vix_change_pct 门控及 get_market_calendar_alerts()。
    (histories, bundles), macro = await asyncio.gather(
        market_data.get_bundles(fetch_syms, include_news=True, include_earnings=True),
        market_data.get_macro_snapshot(),
    )
    tech_map: dict[str, TechnicalSnapshot] = {}
    for sym in tracked_syms:
        snap = compute_technical_snapshot(sym, histories.get(sym))
        if snap is not None:
            tech_map[sym] = snap

    # TODO(IMP-12): [接自 IMP-07] 传入 histories=histories 到 evaluate_portfolio(..., histories=histories)，
    # 并调用 build_disagreement_matrix() / format_disagreement_matrix_text() 写入 fact_sheet 与 Step 3 PM / ContextPack。
    evals = evaluate_portfolio(portfolio, tech_map, bundles, cfg.strategy)
    if us_only:
        evals = [e for e in evals if "." not in e.yahoo_symbol]
    eval_text = format_evaluations_text(evals)
    exposure_text = compute_exposure_summary(portfolio, bundles)

    held_lines = [format_technical_line(sym, tech_map.get(sym), bundles.get(sym)) for sym in tracked_syms if sym in held_syms_set]
    pending_only_syms = [sym for sym in tracked_syms if sym not in held_syms_set]
    pending_tech_lines = [format_technical_line(sym, tech_map.get(sym), bundles.get(sym)) for sym in pending_only_syms]

    news_lines = []
    for sym in tracked_syms:
        b = bundles.get(sym)
        if b and b.news:
            titles = " / ".join(f"[{n['date']}] {n['title']}" for n in b.news[:2])
            news_lines.append(f"• {sym}: {titles}")

    fact_sheet = (
        f"【实时宏观与汇率（确定性数据源）】\n{macro.to_text()}\n\n"
        f"【跨账户重叠与行业集中度（Python 实算）】\n{exposure_text}\n\n"
        f"【持仓量化技术面、目标价与财报雷达】\n"
        + ("\n".join(held_lines) if held_lines else "（无持仓技术数据）")
        + (
            "\n\n【未成交活动挂单标的实时技术面与估值（供审计挂单）】\n" + "\n".join(pending_tech_lines)
            if pending_tech_lines
            else ""
        )
        + "\n\n【Python 规则引擎：加仓 / 减仓 / 止损复核信号】\n"
        + eval_text
        + "\n\n【近期公司新闻摘要】\n"
        + ("\n".join(news_lines) if news_lines else "（近7天无重要突发新闻或暂无抓取）")
    )
    return fact_sheet, tech_map, bundles, macro


async def generate_morning_report(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """TradingAgents-Lite 3-step pipeline:
    1) Deterministic FactSheet + Concurrent Multi-Region Google Search Grounding
    2) Concurrent Analyst (Bull/Opportunity, HIGH, <=6 tool rounds) & Risk Auditor (Bear, Argon HIGH, <=6 tool rounds)
    3) Portfolio Manager Pydantic Structured Synthesis (with auto-repair) -> <= 1500 chars
    """
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 双账户数据均读取失败：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    tracked_syms = portfolio.unique_yahoo_symbols(include_pending=True)
    grounding_coro = (
        run_forced_market_grounding(
            llm=llm,
            search_model=cfg.models.report_search_model,
            scope="report",
            symbols=tracked_syms,
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
        )
        if cfg.models.enable_forced_search_grounding
        else asyncio.sleep(0, result="")
    )

    prev_snap, past_lessons, (fact_sheet, _, bundles, _), grounding_brief = await asyncio.gather(
        db.get_previous_snapshot(min_age_hours=4.0),
        db.get_past_decision_context(limit=5),
        build_portfolio_fact_sheet(portfolio, market_data, cfg),
        grounding_coro,
    )

    trade_ctx = build_trade_context(
        portfolio,
        history_days=cfg.strategy.order_history_days,
        max_recent_lines=10,
        max_summary_lines=10,
        bundles=bundles,
    )
    diff_text = _compute_portfolio_diff_text(portfolio, prev_snap)
    us_sched = get_us_market_schedule_uk()
    uk_bank_holiday = is_uk_bank_holiday()
    now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
    dst_note = (
        f"今日美股 {us_sched.open_time_uk} UK 开盘（错位周）"
        if us_sched.is_dst_mismatch
        else f"今日美股 {us_sched.open_time_uk} UK 开盘"
    )
    if us_sched.phase in {"HOLIDAY", "WEEKEND"}:
        dst_note = "今日美股休市"
    if uk_bank_holiday:
        dst_note += " | 🇬🇧 今日英国 Bank Holiday（LSE 休市，英股/LSE ETF 暂不成交，按下一开盘日预案）"

    shared_context = (
        f"时间：英国时间 {now_uk}（{get_uk_weekday_label()} | {dst_note}）\n\n"
        f"=== 一、双账户状态 ===\n"
        f"【Stocks ISA】\n{format_account_overview(portfolio.isa)}\n持仓：\n{format_positions_text(portfolio.isa)}\n\n"
        f"【Invest】\n{format_account_overview(portfolio.invest)}\n持仓：\n{format_positions_text(portfolio.invest)}\n\n"
        f"【净值变动】{diff_text}\n\n"
        f"【当前活动挂单（含市场现价/目标价）】\n{trade_ctx['pending_orders_text']}\n\n"
        f"【近7天实盘成交】\n{trade_ctx['recent_trades_text']}\n\n"
        f"【近{cfg.strategy.order_history_days}天行为统计】\n{trade_ctx['trade_summary_30d_text']}\n\n"
        f"=== 二、量化事实底座 ===\n{fact_sheet}"
        + (f"\n\n=== 三、实时 Google Search 全市场（英/欧/美/ETF/宏观）深度新闻 ===\n{grounding_brief}" if grounding_brief else "")
    )

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.report_search_model,
        ops_notifier=ops_notifier,
    )

    # ---------- Step 2A: Opportunity & Portfolio Analyst ----------
    analyst_sys = build_agent_system_instruction(
        "你是组合机会分析师（Bull/Opportunity 视角）。请基于用户提供的唯一真实事实数据，用干货要点输出完整内部研报草稿供 PM 拍板（1200 字内）：\n"
        "1. 宏观利率/汇率主线与今日整体交易气候；\n"
        "2. 双账户净值、FX 拆解与近 7/30 天实盘复盘要点；\n"
        "3. 哪些持仓具备真实技术支撑（回踩 MA50/MA200 且基本面未坏）值得考虑加仓或顺势持有（写明支撑位/目标价与依据）；\n"
        "4. 近 7 日财报事件应对预案。\n"
        "切勿平铺常规正常持仓，仅点名关键标的；以上 FactSheet 与实时 Google Search 摘要已涵盖持仓及挂单的量价、目标价、财报日与最新新闻，"
        "对数据已完整且无异常的持仓切勿重复调用工具；仅当某只活动挂单、临近财报股、异常波动或英欧美标的出现疑点/信息冲突时，主动调用 inspect_ticker 或 search_market_news 深入核查。",
        cfg.strategy,
    )
    analyst_prompt = f"【任务：组合机会分析师（Bull 视角，深度推理并保留关键数字/支撑位，1200 字内）】\n{shared_context}"

    # ---------- Step 2B: Ruthless Risk & Pending-Order Auditor ----------
    auditor_sys = build_agent_system_instruction(
        "你是独立风控与挂单审计官（Bear/Conservative 视角）。你的唯一任务是冷酷挑刺与风险一票否决，提供完整内部审计草稿供 PM 拍板（1200 字内）：\n"
        "1. 行业集中度与跨账户重叠审计：基于 Python 算出的行业权重与跨账户重叠，明确哪些行业/标的是否已超载、今天必须冻结禁止加码（列出红线）。\n"
        "2. 破位与弱势持仓审计：跌破 MA200 或触及止损线（尤其 Invest 短线仓）或临近 5 日内财报的标的，明确减仓/硬止损/严禁补仓指令；"
        "常规持仓无需重复调工具，但如发现未披露利空、监管传闻或挂单定价值得怀疑，务必调用 search_market_news 或 inspect_ticker 二次交叉验证。\n"
        "3. 【活动挂单零情面独立审计】：逐一核查【当前活动挂单】！"
        "无论挂单是用户手动下单还是前序 AI 建议，结合当前现价、均线位置、分析师目标价、"
        f"{cfg.strategy.uk_stamp_duty_pct:.1f}% 印花税/{cfg.strategy.fx_fee_pct * 2:.2f}% 往返换汇费以及 {cfg.strategy.cash_interest_rate_pct:.1f}% 现金利息门槛——"
        "只要限价过高、向上空间不足或逆势风险大，必须敢于直白裁决【CANCEL 建议撤单】或【MODIFY 建议下调限价至具体数字】！",
        cfg.strategy,
    )
    auditor_prompt = f"【任务：独立风控与挂单审计官（Bear 视角，深度审计并保留完整依据，1200 字内）】\n{shared_context}"

    report_rounds = max(1, int(cfg.models.report_max_tool_rounds))
    analyst_notes, auditor_notes = await asyncio.gather(
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
            model=cfg.models.report_model,
            prompt=auditor_prompt,
            system_instruction=auditor_sys,
            thinking_level=cfg.models.report_thinking_level,
            tools=toolkit.declarations(),
            tool_executor=toolkit.as_executor(),
            temperature=0.15,
            max_tool_rounds=report_rounds,
            usage_label="report_auditor",
        ),
    )

    # ---------- Step 3: Portfolio Manager Structured Synthesis ----------
    pm_sys = build_agent_system_instruction(
        "你是最终投资组合经理（Portfolio Manager），负责拍板权衡「机会分析师」与「独立风控审计官」的冲突并输出结构化今日总纲（严格遵守 JSON Schema）。\n"
        "严格去重与排版准则（违反将导致重要信息被截断）：\n"
        "• `top_actions`【只放现有持仓的 ADD/TRIM/STOP/HOLD】（1~3条）；所有活动挂单的撤改留在 `pending_order_audits`，严禁把 CANCEL_ORDER/MODIFY_ORDER 重复写进 `top_actions`！\n"
        "• `pending_order_audits` 逐一简洁裁决 KEEP/MODIFY/CANCEL，不要和上文重复抄写理由。\n"
        "• `flagged_positions` 只列出【未在 top_actions 或 pending_order_audits 中出现过】的其他异动/观察持仓（如暴跌、财报临近但暂不操作者），严禁再复述上方已给过动作的标的！\n"
        "• `daily_vetoes` 将写入数据库强制约束今日后续 `/pick` 和 `/us`，只写简洁可执行的红线（2~4条，每条<=35字）。\n"
        "• 全文必须精炼直白，控制在 1250 字左右（绝不可超 1500 字），确保财报预警与操作复盘完整展示。",
        cfg.strategy,
    )
    pm_prompt = (
        f"=== 历史决策回顾 (Memory Log) ===\n{past_lessons}\n\n"
        f"=== 账户速览 ===\nISA 现金 {portfolio.isa.account_currency} {portfolio.isa.total_cash:,.2f} ({portfolio.isa.cash_ratio_pct:.1f}% | 可用 {portfolio.isa.available_to_trade:,.2f}) | "
        f"Invest 现金 {portfolio.invest.account_currency} {portfolio.invest.total_cash:,.2f} ({portfolio.invest.cash_ratio_pct:.1f}%)\n"
        f"活动挂单：\n{trade_ctx['pending_orders_text']}\n\n"
        f"=== 机会分析师报告 ===\n{analyst_notes}\n\n"
        f"=== 独立风控审计官挑刺与挂单审计 ===\n{auditor_notes}"
    )

    raw_pm = await llm.generate(
        model=cfg.models.report_model,
        prompt=pm_prompt,
        system_instruction=pm_sys,
        thinking_level=cfg.models.report_thinking_level,
        response_schema=MorningReportDecision,
        temperature=0.15,
        usage_label="report_pm",
    )

    decision, resolved_pm_raw = await parse_or_repair_structured_json(
        raw_pm,
        MorningReportDecision,
        llm=llm,
        repair_model=cfg.models.fallback_model,
        thinking_level="LOW",
        usage_label="report_pm_repair",
        ops_notifier=ops_notifier,
        alert_source="report_pm",
    )
    holiday_banner = "🇬🇧 今日英国 Bank Holiday（LSE 休市，英股/LSE ETF 暂不交易）\n\n" if uk_bank_holiday else ""
    max_body_chars = 1500 - len(holiday_banner)
    # TODO(IMP-12): [接自 IMP-07] decision 非空时先调用 apply_risk_override_guardrail(decision, portfolio=portfolio, macro=macro, strategy=cfg.strategy, bundle_map=bundles, evals=evals) 取修正后 decision 再渲染。
    if decision is not None:
        final_report = holiday_banner + render_morning_report(decision, max_chars=max_body_chars)
        guardrails = list(decision.daily_vetoes)
        actions_summary = "; ".join(
            f"[{a.account}] {a.action} {a.symbol}({a.price_zone})" for a in decision.top_actions[:4]
        )
        short_conclusion = f"{decision.headline} | 重点动作: {actions_summary}"
    else:
        logger.warning("Morning PM structured parse fell back to plain text.")
        final_report = holiday_banner + resolved_pm_raw[:max_body_chars]
        guardrails = []
        short_conclusion = _extract_conclusion_fallback(final_report)

    # TODO(IMP-12): [接自 IMP-04] 当 portfolio.is_fallback 为 True 时必须跳过 db.save_snapshot(portfolio)，防降级缓存污染最新快照；
    # 同时在 _compute_portfolio_diff_text() 与早报中消费 recent_transactions / recent_dividends / cash_flow_summary_text。
    await db.save_snapshot(portfolio)
    if not final_report.startswith("⚠️"):
        await db.save_morning_report(final_report, short_conclusion, guardrails=guardrails)
    return final_report


async def generate_us_sentinel(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    is_scheduled_premarket: bool = False,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Delta US Sentinel (/us & premarket): checks latest US data against FULL /report + vetoes, <= 450 chars."""
    sched = get_us_market_schedule_uk()
    portfolio = await t212.get_portfolio(use_cache=not is_scheduled_premarket)
    fact_sheet, _, bundles, _ = await build_portfolio_fact_sheet(portfolio, market_data, cfg, us_only=True)
    (morning_report_full, morning_conclusion), guardrails = await asyncio.gather(
        db.get_morning_context(include_premarket=False),
        db.get_daily_guardrails(),
    )

    trade_ctx = build_trade_context(
        portfolio,
        max_recent_lines=5,
        max_summary_lines=5,
        bundles=bundles,
        pending_us_only=True,
    )
    phase_desc = {
        "PRE": f"美股盘前（{sched.open_time_uk} UK 开盘，距开盘 {max(sched.minutes_to_open, 0)} 分钟）",
        "OPEN": f"美股盘中（已开盘 {max(sched.minutes_since_open, 0)} 分钟）",
        "CLOSED": f"美股已收盘（{sched.close_time_uk} UK 收盘）",
        "HOLIDAY": "美股法定休市日",
        "WEEKEND": "周末美股休市",
    }.get(sched.phase, sched.phase)

    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（今日尚未生成 /report 红线）"
    morning_baseline = morning_report_full[:1500] if morning_report_full else (morning_conclusion or "（今日尚未运行 /report）")
    us_sys = build_agent_system_instruction(
        "你是【美股增量哨兵（Delta Sentinel）】。\n"
        "你的唯一职责：以今日晨报 `/report` 全文为绝对基线，结合最新美股行情/新闻/宏观，做一次开盘前二次核验！\n"
        "禁止对早报已决定的英股或正常无异动美股另起炉灶。\n"
        "严格二选一开头并限制在 380 字以内（绝对不超 450 字，禁用 Markdown # 和 **）：\n"
        "• 若今日早报之后无颠覆性美股新变量（或休市无新行情）：第一行直接写「✅ 维持早报计划」，"
        "随后仅用 2~3 条简写：美股开盘执行动作确认、止损核对与财报静默期提醒。\n"
        "• 若盘前突发新新闻/跳空/宏观剧震真正改变赔率：第一行写「🚨 盘前增量修正」，"
        "直指要修正 /report 哪一条建议 + 最新数据依据 + 调整后指令。",
        cfg.strategy,
    )
    prompt = (
        f"【任务：美股增量哨兵】当前时段：{phase_desc}（UK {sched.uk_date} {sched.uk_weekday_label}）。\n\n"
        f"=== 今日 /report 全文（必须对照逐条核验）===\n{morning_baseline}\n\n"
        f"=== 今日风控红线 ===\n{veto_block}\n\n"
        f"=== 最新美股事实与可用现金（ISA {portfolio.isa.available_to_trade:,.2f} GBP | Invest {portfolio.invest.available_to_trade:,.2f} GBP）===\n"
        f"【美股活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"{fact_sheet}"
    )

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
    )
    memo = await llm.generate(
        model=cfg.models.us_sentinel_model,
        prompt=prompt,
        system_instruction=us_sys,
        thinking_level=cfg.models.us_thinking_level,
        tools=toolkit.declarations(),
        tool_executor=toolkit.as_executor(),
        temperature=0.15,
        usage_label="us_sentinel",
    )
    if not memo.startswith("⚠️"):
        if len(memo) > 450:
            memo = memo[:449].rstrip() + "…"
        await db.append_premarket_note(memo)
    return memo
