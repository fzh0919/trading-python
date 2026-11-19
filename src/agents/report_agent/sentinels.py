"""US pre-open/mid-session anomaly sentinel (/us) and intraday event flash alert generator."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction, build_wallet_context_text
from src.agents.report_agent.fact_sheet import (
    _PENDING_PROXIMITY_PCT,
    _VIX_DANGER_LEVEL,
    _assemble_fact_sheet_bundle,
    _is_us_symbol,
)
from src.agents.schemas import trim_text_at_line_boundary
from src.agents.tools import MarketToolkit, run_forced_market_grounding
from src.config import AppConfig
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData, T212Client, build_trade_context
from src.engine.calendar_utils import get_us_market_schedule_uk
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger("src.agents.report_agent")


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("src.agents.report_agent")
    return getattr(facade, name, default) if facade is not None else default


def _daily_move_pct_from_hist(hist: pd.DataFrame | None) -> float | None:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
    closes = closes[closes > 0]
    if len(closes) < 2:
        return None
    prev = float(closes.iloc[-2])
    last = float(closes.iloc[-1])
    if prev <= 0:
        return None
    return ((last / prev) - 1.0) * 100.0


def _evaluate_us_anomaly_gate(
    *,
    portfolio: PortfolioData,
    cfg: AppConfig,
    histories: dict[str, pd.DataFrame],
    bundles: dict[str, TickerMarketBundle],
    macro: MacroSnapshot,
    vix_hist: pd.DataFrame | None,
    morning_context_pack: dict[str, Any] | None,
    realtime_quotes: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Evaluate local 0-LLM Anomaly Gate for /us sentinel (moves, order proximity, VIX delta)."""
    move_threshold = float(cfg.strategy.us_anomaly_move_pct)
    vix_chg_threshold = float(cfg.strategy.us_anomaly_vix_change_pct)
    quotes = realtime_quotes or {}

    anomaly_syms: list[str] = []
    reasons: list[str] = []

    us_held_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=False) if _is_us_symbol(s)]
    for sym in us_held_syms:
        move_pct: float | None = None
        quote = quotes.get(sym)
        if quote:
            q_pct = quote.get("change_pct")
            if q_pct is not None:
                move_pct = float(q_pct)
            else:
                q_cur = quote.get("current_price")
                q_prev = quote.get("prev_close")
                if q_cur and q_prev and float(q_prev) > 0:
                    move_pct = ((float(q_cur) / float(q_prev)) - 1.0) * 100.0
        if move_pct is None:
            move_pct = _daily_move_pct_from_hist(histories.get(sym))

        if move_pct is not None and abs(move_pct) >= move_threshold:
            anomaly_syms.append(sym)
            reasons.append(f"{sym} 日内/最新日线涨跌 {move_pct:+.2f}%（达异动阈值 ±{move_threshold:.1f}%）")

    for order in portfolio.all_pending_orders():
        sym = order.yahoo_symbol
        if not sym or not _is_us_symbol(sym):
            continue
        quote = quotes.get(sym) or {}
        b = bundles.get(sym)
        hist_sym = histories.get(sym)
        hist_last = None
        if hist_sym is not None and not hist_sym.empty and "Close" in hist_sym.columns:
            c_ser = pd.to_numeric(hist_sym["Close"], errors="coerce").dropna()
            if not c_ser.empty:
                hist_last = float(c_ser.iloc[-1])
        cur_price = quote.get("current_price") or (b.current_price if b else None) or hist_last
        if cur_price is None or cur_price <= 0:
            continue
        for label, trig in (("限价", order.limit_price), ("止损", order.stop_price)):
            if trig is None or trig <= 0:
                continue
            dist_pct = ((cur_price - trig) / trig) * 100.0
            crossed = (
                (order.side == "BUY" and cur_price <= trig)
                or (order.side == "SELL" and label == "止损" and cur_price <= trig)
                or (order.side == "SELL" and label == "限价" and cur_price >= trig)
            )
            if crossed or abs(dist_pct) <= _PENDING_PROXIMITY_PCT:
                if sym not in anomaly_syms:
                    anomaly_syms.append(sym)
                reasons.append(
                    f"[{order.account}] {sym} {order.side} 挂单{label} {trig:.2f} 临近/触及现价 {cur_price:.2f}（距 {dist_pct:+.2f}%）"
                )

    vix_chg_pct = _daily_move_pct_from_hist(vix_hist)
    if vix_chg_pct is None and macro.vix is not None and morning_context_pack:
        base_vix = (morning_context_pack.get("macro") or {}).get("vix")
        if base_vix is not None and float(base_vix) > 0:
            vix_chg_pct = ((macro.vix / float(base_vix)) - 1.0) * 100.0

    if vix_chg_pct is not None and abs(vix_chg_pct) >= vix_chg_threshold:
        reasons.append(f"VIX 日变动 {vix_chg_pct:+.2f}%（达阈值 ±{vix_chg_threshold:.1f}%）")
    if macro.vix is not None and macro.vix >= _VIX_DANGER_LEVEL:
        reasons.append(f"VIX 处于高危水平 {macro.vix:.2f}（>= {_VIX_DANGER_LEVEL:.0f}）")

    return bool(reasons), anomaly_syms, reasons


async def generate_us_sentinel(
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    is_scheduled_premarket: bool = False,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Delta US Sentinel (/us & premarket): local Anomaly Gate -> calm LOW <=450 / anomaly search+MEDIUM <=600."""
    sched = _facade_attr("get_us_market_schedule_uk", get_us_market_schedule_uk)()
    portfolio = await t212.get_portfolio(use_cache=not is_scheduled_premarket)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 数据获取失败，无法执行美股哨兵：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    us_held_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=False) if _is_us_symbol(s)]
    us_tracked_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=True) if _is_us_symbol(s)]

    async def _fetch_vix_hist() -> dict[str, pd.DataFrame]:
        try:
            return await market_data.get_histories(["^VIX"], period="1mo")
        except Exception as exc:
            logger.debug("VIX history fetch failed in /us gate: %s", exc)
            return {}

    async def _fetch_us_quotes() -> dict[str, dict[str, Any]]:
        if not us_tracked_syms:
            return {}
        out: dict[str, dict[str, Any]] = {}
        if getattr(getattr(market_data, "finnhub", None), "enabled", False):
            pairs = await asyncio.gather(
                *(market_data.get_us_realtime_quote(sym) for sym in us_tracked_syms),
                return_exceptions=True,
            )
            for sym, res in zip(us_tracked_syms, pairs):
                if isinstance(res, dict) and res:
                    out[sym] = res

        if sched.phase == "PRE":
            need_prepost = list(us_tracked_syms)
        else:
            need_prepost = [
                sym
                for sym in us_tracked_syms
                if sym not in out
                or out[sym].get("change_pct") is None
                or abs(float(out[sym].get("change_pct") or 0.0)) < 1e-9
            ]

        if need_prepost and hasattr(market_data, "get_us_prepost_quotes"):
            try:
                prepost_map = await market_data.get_us_prepost_quotes(need_prepost)
                for sym, pp in (prepost_map or {}).items():
                    if isinstance(pp, dict) and pp.get("current_price"):
                        out[sym] = {**(out.get(sym) or {}), **pp}
            except Exception as exc:
                logger.debug("US prepost quotes fallback failed in /us sentinel: %s", exc)
        return out

    (
        fact_bundle,
        (morning_report_full, morning_conclusion, morning_stale),
        (guardrails, guardrails_stale),
        morning_pack,
        vix_hist_map,
        realtime_quotes,
    ) = await asyncio.gather(
        _assemble_fact_sheet_bundle(portfolio, market_data, cfg, us_only=True, db=db),
        db.get_morning_context_with_status(include_premarket=False),
        db.get_daily_guardrails_with_status(),
        db.get_context_pack(allow_fallback=False),
        _fetch_vix_hist(),
        _fetch_us_quotes(),
    )
    (
        fact_sheet,
        tech_map,
        bundles,
        macro,
        histories,
        _evals,
        _disagreements,
        _disagreement_text,
        calendar_alerts,
        _context_pack,
    ) = fact_bundle

    has_anomaly, anomaly_syms, anomaly_reasons = _evaluate_us_anomaly_gate(
        portfolio=portfolio,
        cfg=cfg,
        histories=histories,
        bundles=bundles,
        macro=macro,
        vix_hist=vix_hist_map.get("^VIX"),
        morning_context_pack=morning_pack,
        realtime_quotes=realtime_quotes,
    )

    us_grounding = ""
    if has_anomaly and cfg.models.enable_forced_search_grounding:
        search_targets = anomaly_syms or us_held_syms or us_tracked_syms
        if search_targets:
            us_grounding = await _facade_attr("run_forced_market_grounding", run_forced_market_grounding)(
                llm=llm,
                search_model=cfg.models.search_model,
                scope="us",
                symbols=search_targets,
                ttl_seconds=cfg.app.market_data_ttl_seconds,
                ops_notifier=ops_notifier,
                db=db,
                persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
                bypass_cache=True,
            )

    trade_ctx = build_trade_context(
        portfolio,
        max_recent_lines=5,
        max_summary_lines=5,
        bundles=bundles,
        pending_us_only=True,
        tech_map=tech_map,
        histories=histories,
    )
    phase_desc = {
        "PRE": f"美股盘前（{sched.open_time_uk} UK 开盘，距开盘 {max(sched.minutes_to_open, 0)} 分钟）",
        "OPEN": f"美股盘中（已开盘 {max(sched.minutes_since_open, 0)} 分钟）",
        "CLOSED": f"美股已收盘（{sched.close_time_uk} UK 收盘）",
        "HOLIDAY": "美股法定休市日",
        "WEEKEND": "周末美股休市",
    }.get(sched.phase, sched.phase)

    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（今日尚未生成 /report 红线）"
    if guardrails_stale:
        veto_block = "（以下为上一交易日旧红线，仅作参考，严禁当作今日裁决）\n" + veto_block
    # [用户授权调整 v5.5.2] 随 /report 放宽至 2800 同步放宽基线切片；IMP-12 请保留。
    morning_baseline = morning_report_full[:2800] if morning_report_full else (morning_conclusion or "（今日尚未运行 /report）")
    if morning_stale:
        morning_baseline = "⚠️ 今日 /report 尚未生成，以下为上一交易日基线，严禁当作今日裁决。\n" + morning_baseline
    gate_block = (
        "【本地异动门控：已触发深度复核】\n" + "\n".join(f"• {r}" for r in anomaly_reasons)
        if has_anomaly
        else (
            f"【本地异动门控：市场与美股持仓均平稳】"
            f"（持仓波动均 < ±{cfg.strategy.us_anomaly_move_pct:.1f}%，VIX 变动 < ±{cfg.strategy.us_anomaly_vix_change_pct:.1f}%，无挂单逼近触发）"
        )
    )
    cal_block = "\n" + " | ".join(calendar_alerts) if calendar_alerts else ""

    us_sys = build_agent_system_instruction(
        "你是【美股增量哨兵（Delta Sentinel）】。\n"
        "你的唯一职责：以今日晨报 `/report` 全文为绝对基线，结合最新美股行情/新闻/宏观，做一次开盘前二次核验！\n"
        "禁止对早报已决定的英股或正常无异动美股另起炉灶。\n"
        "严格二选一开头（禁用 Markdown # 和 **）：\n"
        "• 若今日早报之后无颠覆性美股新变量（或休市无新行情 / 本地异动门控平稳）：第一行直接写「✅ 维持早报计划」，"
        "随后仅用 2~3 条简写：美股开盘执行动作确认、止损核对与财报静默期提醒（控制在 380 字以内，绝不超 450 字）。\n"
        "• 若本地异动门控触发且突发新新闻、大幅跳空或宏观剧震真正改变赔率：第一行写「🚨 盘前增量修正」，"
        "直指要修正 /report 哪一条建议 + 最新数据依据 + 调整后指令（控制在 550 字以内，绝不超 600 字）。",
        cfg.strategy,
    )
    prompt = (
        f"【任务：美股增量哨兵】当前时段：{phase_desc}（UK {sched.uk_date} {sched.uk_weekday_label}{cal_block}）。\n"
        f"{gate_block}\n\n"
        f"=== 今日 /report 全文（必须对照逐条核验）===\n{morning_baseline}\n\n"
        f"=== 今日风控红线 ===\n{veto_block}\n\n"
        f"=== 最新美股事实与可用现金 ===\n"
        f"{build_wallet_context_text(portfolio, cfg.strategy, compact=True)}\n"
        f"【美股活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"{fact_sheet}"
        + (f"\n\n=== 异动标的实时 Google Search 归因 ===\n{us_grounding}" if us_grounding else "")
    )

    if has_anomaly:
        toolkit = MarketToolkit(
            market_data,
            portfolio,
            llm=llm,
            search_model=cfg.models.search_model,
            ops_notifier=ops_notifier,
            db=db,
            search_cache_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
        tools = toolkit.declarations()
        tool_executor = toolkit.as_executor()
        us_model = cfg.models.us_anomaly_model
        us_thinking = cfg.models.us_anomaly_thinking_level
    else:
        tools = None
        tool_executor = None
        us_model = cfg.models.us_sentinel_model
        us_thinking = cfg.models.us_thinking_level

    memo = await llm.generate(
        model=us_model,
        prompt=prompt,
        system_instruction=us_sys,
        thinking_level=us_thinking,
        tools=tools,
        tool_executor=tool_executor,
        temperature=0.15,
        usage_label="us_sentinel",
    )
    if not memo.startswith("⚠️"):
        # [用户授权调整] /us 触发异常修正（「🚨」）放宽至 600 字安全天花板、平稳日保持 450 字，并改按换行边界优雅裁剪。
        us_max_chars = 600 if (has_anomaly or memo.lstrip().startswith("🚨")) else 450
        if len(memo) > us_max_chars:
            memo = trim_text_at_line_boundary(memo, us_max_chars)
        await db.append_premarket_note(memo)
    return memo


async def generate_intraday_alert(
    cfg: AppConfig,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    *,
    symbol: str,
    current_price: float,
    move_pct: float,
    order_dist_pct: float | None = None,
    pending_order_text: str = "",
    currency: str = "",
    account_context: str = "",
    trigger_reason: str = "",
    ops_notifier: OpsNotifier | None = None,
) -> str | None:
    """Event-driven intraday alert: 1-symbol grounding + DeepSeek push/no-push decision (<=250 chars)."""
    if not ds_helper_available(llm, cfg):
        return None
    sym = str(symbol or "").strip().upper()
    if not sym or current_price <= 0:
        return None

    try:
        (morning_text, _), guardrails = await asyncio.gather(
            db.get_morning_context(include_premarket=True),
            db.get_daily_guardrails(),
        )
    except Exception as exc:  # noqa: BLE001 - intraday alerting must never break the scheduler
        logger.warning("Intraday alert context load failed for %s: %s", sym, exc)
        return None

    grounding = ""
    try:
        grounding = await _facade_attr("run_forced_market_grounding", run_forced_market_grounding)(
            llm=llm,
            search_model=cfg.models.search_model,
            scope="intraday",
            symbols=[sym],
            ttl_seconds=cfg.app.market_data_ttl_seconds,
            ops_notifier=ops_notifier,
            db=db,
            persist_ttl_seconds=cfg.app.grounding_cache_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Intraday grounding failed for %s: %s", sym, exc)

    morning_slice = trim_text_at_line_boundary(str(morning_text or ""), 1800)
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（无）"
    distance_text = "N/A" if order_dist_pct is None else f"{order_dist_pct:.2f}%"
    account_tag = str(account_context or "").strip()
    philosophy_map = {
        "ISA": "ISA 巴菲特价值仓：忽略短期噪音，仅基本面破坏或深折价机会才打扰用户",
        "Invest": "Invest 右侧动量仓：重点关注止损纪律与趋势破坏，破位需立即提醒",
    }
    philosophy_text = philosophy_map.get(account_tag, "未标注账户：按通用趋势/风控纪律判断")
    numeric = (
        "【Python 实时数值（唯一真源，禁止改写）】\n"
        f"{sym} 现价 {current_price:.2f} {currency or ''} | 日内涨跌 {move_pct:+.2f}% | "
        f"距最近挂单价 {distance_text}\n"
        f"活动挂单：{pending_order_text or '无'}\n"
        f"账户哲学：{philosophy_text}"
        + (f"\n触发原因：{trigger_reason}" if trigger_reason else "")
    )
    prompt = (
        f"{numeric}\n\n【今日 /report 原文节选】\n{morning_slice or '（今日尚未生成 /report）'}\n\n"
        f"【今日红线】\n{veto_block}\n\n"
        + (f"【盘中联网快讯】\n{grounding}\n\n" if grounding else "")
        + "请判断这次异动是否值得立即打断用户：只在具备可执行意义（趋势破坏、催化确认或与今日红线直接相关）时推送；"
        "正常波动、已被早报覆盖或与红线无关时返回 should_push=false。"
        "ISA 价值仓请严格执行噪音过滤，只有基本面破坏或深折价机会才推送；Invest 右侧仓优先止损纪律与趋势破坏。"
    )
    payload = await call_ds_json(
        cfg,
        llm,
        prompt=prompt,
        system_instruction=(
            "你是盘中异动播报员（只读）。只输出严格 JSON："
            '{"should_push": true/false, "alert_text": "<=250字纯文本"}。'
            "alert_text 必须只使用【Python 实时数值】中的现价/涨跌幅/挂单距离，禁止编造任何新数字，"
            "禁止 Markdown # 与 **；should_push=false 时 alert_text 可为空。"
        ),
        usage_label="intraday_alert",
    )
    if not payload or payload.get("should_push") is not True:
        return None
    text = sanitize_plain_text(str(payload.get("alert_text") or ""), 250)
    if len(text) < 10:
        return None
    return f"⚡️ 盘中异动 | {sym}\n{text}"
