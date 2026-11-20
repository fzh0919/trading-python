"""US pre-open/mid-session anomaly sentinel (/us) and intraday event flash alert generator."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend
from src.agents.prompts import (
    build_agent_system_instruction,
    build_wallet_context_text,
    clean_telegram_plain_text,
    has_complete_embedded_vetoes,
)
from src.agents.report_agent.fact_sheet import (
    _PENDING_PROXIMITY_PCT,
    _VIX_DANGER_LEVEL,
    _is_us_symbol,
)
from src.agents.schemas import trim_text_at_line_boundary
from src.agents.tools import MarketToolkit, run_forced_market_grounding
from src.config import AppConfig
from src.data_providers.analytics import compute_technical_snapshot, format_technical_line
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData, T212Client, build_trade_context
from src.engine.calendar_utils import get_market_calendar_alerts, get_us_market_schedule_uk
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


def _quote_is_fresh(quote: dict[str, Any] | None, ny_date: str) -> bool:
    """A quote is session-fresh when it carries no bar_date_et (legacy stubs) or matches the US trading date."""
    if not quote:
        return True
    bar_date = str(quote.get("bar_date_et") or "").strip()
    return (not bar_date) or (not ny_date) or bar_date == ny_date


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
    schedule_phase: str = "",
    ny_date: str = "",
) -> tuple[bool, list[str], list[str]]:
    """Evaluate local 0-LLM Delta Anomaly Gate for /us sentinel (fresh moves, order proximity, VIX delta).

    v5.5.8 upgrades over the legacy gate:
    - WEEKEND/HOLIDAY fuse: market-closed phases short-circuit to calm (no quote pulls,
      no searches, no Hydrogen escalation).
    - Pre/open-session freshness: stale (yesterday's) dp/15m moves are never reported as
      today's pre-market anomaly; the morning report already covered that close move.
    - VIX delta gate: with a fresh morning pack the trigger compares against the
      report-time VIX (delta >= threshold, or a *new* cross above the danger level)
      instead of re-alarming on a static level the morning report already priced in.
    """
    if schedule_phase in {"WEEKEND", "HOLIDAY"}:
        return False, [], []

    move_threshold = float(cfg.strategy.us_anomaly_move_pct)
    vix_chg_threshold = float(cfg.strategy.us_anomaly_vix_change_pct)
    quotes = realtime_quotes or {}
    in_preopen = schedule_phase in {"PRE", "OPEN"}

    anomaly_syms: list[str] = []
    reasons: list[str] = []

    us_held_syms = [s for s in portfolio.unique_yahoo_symbols(include_pending=False) if _is_us_symbol(s)]
    for sym in us_held_syms:
        move_pct: float | None = None
        quote = quotes.get(sym)
        used_quote = False
        if quote:
            # Outside PRE/OPEN the latest completed-session dp is the regular daily move;
            # inside PRE/OPEN only session-fresh quotes count (stale = yesterday's noise).
            if _quote_is_fresh(quote, ny_date) or not in_preopen:
                q_pct = quote.get("change_pct")
                if q_pct is not None:
                    move_pct = float(q_pct)
                else:
                    q_cur = quote.get("current_price")
                    q_prev = quote.get("prev_close")
                    if q_cur and q_prev and float(q_prev) > 0:
                        move_pct = ((float(q_cur) / float(q_prev)) - 1.0) * 100.0
                used_quote = move_pct is not None
        if not used_quote:
            if schedule_phase == "PRE" and morning_context_pack:
                # Morning report already covered yesterday's close move for this symbol;
                # without a fresh pre-market trade there is no *increment* to re-report.
                continue
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

    morning_vix: float | None = None
    if morning_context_pack:
        base_vix = (morning_context_pack.get("macro") or {}).get("vix")
        if base_vix is not None and float(base_vix) > 0:
            morning_vix = float(base_vix)

    if morning_vix is not None and macro.vix is not None:
        delta_vix_pct = ((macro.vix / morning_vix) - 1.0) * 100.0
        if abs(delta_vix_pct) >= vix_chg_threshold:
            reasons.append(f"VIX 较早报基线增量 {delta_vix_pct:+.2f}%（达阈值 ±{vix_chg_threshold:.1f}%）")
        if macro.vix >= _VIX_DANGER_LEVEL and morning_vix < _VIX_DANGER_LEVEL:
            reasons.append(
                f"VIX 早报后新突破 {_VIX_DANGER_LEVEL:.0f} 警戒线：{morning_vix:.2f} → {macro.vix:.2f}"
            )
    else:
        vix_chg_pct = _daily_move_pct_from_hist(vix_hist)
        if vix_chg_pct is not None and abs(vix_chg_pct) >= vix_chg_threshold:
            reasons.append(f"VIX 日变动 {vix_chg_pct:+.2f}%（达阈值 ±{vix_chg_threshold:.1f}%）")
        if macro.vix is not None and macro.vix >= _VIX_DANGER_LEVEL:
            reasons.append(f"VIX 处于高危水平 {macro.vix:.2f}（>= {_VIX_DANGER_LEVEL:.0f}）")

    return bool(reasons), anomaly_syms, reasons


async def _fetch_us_sentinel_facts_data(
    market_data: MarketDataService,
    us_tracked_syms: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle], MacroSnapshot, list[str]]:
    """v5.5.8 /us: light I/O — US bundles only (no .L/EU quotes, no RSS news, no 30x
    Finnhub deep-intel calls, no T+5 lessons) plus macro snapshot and calendar alerts."""
    get_cal = _facade_attr("get_market_calendar_alerts", get_market_calendar_alerts)

    async def _empty_bundles() -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle]]:
        return {}, {}

    bundles_coro = (
        market_data.get_bundles(us_tracked_syms, include_news=False, include_earnings=True)
        if us_tracked_syms
        else _empty_bundles()
    )
    (histories, bundles), macro = await asyncio.gather(
        bundles_coro,
        market_data.get_macro_snapshot(),
    )
    calendar_alerts = get_cal()
    return histories, bundles, macro, list(calendar_alerts or [])


def build_us_sentinel_facts(
    *,
    cfg: AppConfig,
    us_tracked_syms: list[str],
    bundles: dict[str, TickerMarketBundle],
    macro: MacroSnapshot,
    calendar_alerts: list[str],
    anomaly_syms: list[str],
    us_pending_syms: list[str],
    tech_map: dict[str, Any],
    fetched_utc_label: str,
) -> str:
    """v5.5.8 /us: compact deterministic facts — timestamped macro, technical anchors only
    for anomaly ∪ active-order ∪ near-event names, and a one-line rollup for the rest.

    Replaces the full `_assemble_fact_sheet_bundle` output (cross-account exposure,
    Python rule-engine ADD/CUT signals, disagreement matrix, static Finnhub
    ratings/EPS/insider lines, Yahoo RSS news, identity anchors are all dropped).
    """
    anomaly_set = set(anomaly_syms or [])
    pending_set = set(us_pending_syms or [])
    anchor_lines: list[str] = []
    calm_syms: list[str] = []
    for sym in us_tracked_syms:
        b = bundles.get(sym)
        near_event = False
        if b is not None:
            dte = b.days_to_earnings
            dxd = b.days_to_ex_dividend
            near_event = (
                (dte is not None and 0 <= int(dte) <= 7)
                or (dxd is not None and -3 <= int(dxd) <= 3)
            )
        if sym in anomaly_set or sym in pending_set or near_event:
            anchor_lines.append(format_technical_line(sym, tech_map.get(sym), b))
        else:
            calm_syms.append(sym)

    macro_block = (
        f"【实时宏观与汇率（{fetched_utc_label} UTC 已最新拉取，禁止重复调 get_macro_snapshot）】\n"
        f"{macro.to_text()}"
    )
    cal_block = (
        "\n\n【英美欧市场日历提醒】\n" + "\n".join(f"• {a}" for a in calendar_alerts)
        if calendar_alerts
        else ""
    )
    tech_block = "\n".join(anchor_lines) if anchor_lines else "（当前无美股技术数据）"
    calm_line = (
        f"\n其余平稳美股（无盘前异动/无挂单/无近7日财报与除息事件）：{', '.join(calm_syms)}"
        if calm_syms
        else ""
    )
    return (
        macro_block
        + cal_block
        + f"\n\n【美股紧凑技术锚点（仅异动/挂单/近事件标的）】\n"
        + tech_block
        + calm_line
    )


def _build_us_realtime_snapshot_block(
    *,
    us_tracked_syms: list[str],
    realtime_quotes: dict[str, dict[str, Any]],
    bundles: dict[str, TickerMarketBundle],
    schedule_phase: str,
    ny_date: str,
) -> str:
    """0-Token pre-open/intraday realtime price truth block (single source of live prices)."""
    if not us_tracked_syms:
        return ""
    lines: list[str] = []
    market_closed = schedule_phase in {"WEEKEND", "HOLIDAY"}
    for sym in us_tracked_syms:
        quote = realtime_quotes.get(sym) or {}
        b = bundles.get(sym)
        if market_closed:
            px = b.current_price if b is not None else None
            px_str = f"{float(px):.2f}" if px else "N/A"
            lines.append(f"• {sym}：休市中无实时成交，最近收盘 {px_str} USD")
            continue
        cur = quote.get("current_price")
        if cur and _quote_is_fresh(quote, ny_date):
            pct = quote.get("change_pct")
            pct_str = f"{float(pct):+.2f}%" if pct is not None else "N/A"
            sess = str(quote.get("session_tag") or schedule_phase or "").strip() or "N/A"
            qt = str(quote.get("quote_time_et") or "").strip()
            time_part = f" {qt} ET" if qt else ""
            lines.append(f"• {sym} 实时 {float(cur):.2f} USD｜较昨收 {pct_str}｜时段 {sess}{time_part}")
        else:
            prev = quote.get("prev_close")
            if prev is None and b is not None:
                prev = b.current_price
            prev_str = f"{float(prev):.2f}" if prev else "N/A"
            lines.append(f"• {sym}：盘前暂无新成交，沿用昨收 {prev_str} USD（严禁把昨收旧涨跌当今日异动）")
    return "【美股盘前/盘中实时快照（唯一实时价真源，与早报数字冲突时以此为准）】\n" + "\n".join(lines)


def _fmt_order_price(value: Any) -> str:
    try:
        return f"{float(value):.2f}" if value is not None and float(value) > 0 else "?"
    except (TypeError, ValueError):
        return "?"


def _order_distance_suffix(order: Any, quote: dict[str, Any] | None, bundle: TickerMarketBundle | None) -> str:
    cur = (quote or {}).get("current_price")
    if not cur and bundle is not None:
        cur = bundle.current_price
    if not cur or float(cur) <= 0:
        return ""
    parts: list[str] = []
    for label, trig in (("限价", getattr(order, "limit_price", None)), ("止损", getattr(order, "stop_price", None))):
        if trig is None or float(trig) <= 0:
            continue
        dist_pct = ((float(cur) - float(trig)) / float(trig)) * 100.0
        parts.append(f"距{label} {dist_pct:+.2f}%（现价 {float(cur):.2f}）")
    return "（" + "，".join(parts) + "）" if parts else ""


def _build_us_order_reconciliation_block(
    morning_pack: dict[str, Any] | None,
    portfolio: PortfolioData,
    realtime_quotes: dict[str, dict[str, Any]],
    bundles: dict[str, TickerMarketBundle],
) -> str:
    """0-Token pending-order reconciliation vs the morning report (stops the model from
    telling the user to cancel an order that was already cancelled/filled)."""

    def _key(account: Any, sym: Any, side: Any, limit_price: Any, stop_price: Any) -> tuple:
        def _round(v: Any) -> float | None:
            try:
                return round(float(v), 4) if v is not None else None
            except (TypeError, ValueError):
                return None

        return (str(account or ""), str(sym or "").upper(), str(side or "").upper(), _round(limit_price), _round(stop_price))

    current = [
        o for o in portfolio.all_pending_orders() if o.yahoo_symbol and _is_us_symbol(o.yahoo_symbol)
    ]
    cur_map = {_key(o.account, o.yahoo_symbol, o.side, o.limit_price, o.stop_price): o for o in current}
    lines: list[str] = []

    morning_orders = morning_pack.get("pending_orders") if isinstance(morning_pack, dict) else None
    if isinstance(morning_orders, list) and morning_orders:
        m_map: dict[tuple, dict[str, Any]] = {}
        for o in morning_orders:
            if not isinstance(o, dict) or not _is_us_symbol(str(o.get("yahoo_symbol") or "")):
                continue
            m_map[_key(o.get("account"), o.get("yahoo_symbol"), o.get("side"), o.get("limit_price"), o.get("stop_price"))] = o
        for k, o in m_map.items():
            if k not in cur_map:
                price_part = _fmt_order_price(o.get("limit_price") or o.get("stop_price"))
                lines.append(f"• 已撤销/已成交消失（无需再操作）：[{k[0]}] {k[1]} {k[2]} {price_part} USD")
        for k, o in cur_map.items():
            if k not in m_map:
                lines.append(
                    f"• 新增挂单（早报后新挂）：[{o.account}] {o.yahoo_symbol} {o.side} {_fmt_order_price(o.limit_price or o.stop_price)} USD"
                )
        for k, o in cur_map.items():
            if k in m_map:
                suffix = _order_distance_suffix(o, realtime_quotes.get(o.yahoo_symbol), bundles.get(o.yahoo_symbol))
                lines.append(
                    f"• 当前仍有效美股挂单：[{o.account}] {o.yahoo_symbol} {o.side} 限价 {_fmt_order_price(o.limit_price)}"
                    f"（止损 {_fmt_order_price(o.stop_price)}）{suffix}"
                )
    else:
        morning_syms: set[str] = set()
        if isinstance(morning_pack, dict):
            for sym, entry in (morning_pack.get("symbols") or {}).items():
                if not isinstance(entry, dict) or not _is_us_symbol(str(sym)):
                    continue
                if not entry.get("held_accounts"):
                    morning_syms.add(str(sym).upper())
        cur_syms = {o.yahoo_symbol.upper() for o in current}
        for sym in sorted(morning_syms - cur_syms):
            lines.append(f"• 已撤销/已成交消失（无需再操作）：早报记录的美股挂单 {sym} 当前已不存在")
        for sym in sorted(cur_syms - morning_syms):
            lines.append(f"• 新增挂单（早报后新挂）：{sym}")
        for sym in sorted(cur_syms & morning_syms):
            order = next((o for o in current if o.yahoo_symbol.upper() == sym), None)
            suffix = _order_distance_suffix(order, realtime_quotes.get(sym), bundles.get(sym)) if order else ""
            lines.append(f"• 当前仍有效美股挂单：{sym}{suffix}")

    if not lines:
        return ""
    return "【早报后美股挂单变动对账（严格以本清单为准，已撤销/成交的不重复提醒）】\n" + "\n".join(lines)


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
    us_pending_syms = [
        o.yahoo_symbol
        for o in portfolio.all_pending_orders()
        if o.yahoo_symbol and _is_us_symbol(o.yahoo_symbol)
    ]
    market_closed = sched.phase in {"WEEKEND", "HOLIDAY"}

    async def _fetch_vix_hist() -> dict[str, pd.DataFrame]:
        try:
            return await market_data.get_histories(["^VIX"], period="1mo")
        except Exception as exc:
            logger.debug("VIX history fetch failed in /us gate: %s", exc)
            return {}

    async def _empty_hist_map() -> dict[str, pd.DataFrame]:
        return {}

    async def _fetch_us_quotes() -> dict[str, dict[str, Any]]:
        # WEEKEND/HOLIDAY fuse: no realtime pulls on closed markets.
        if market_closed or not us_tracked_syms:
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
                        base = out.get(sym) or {}
                        if base and _quote_is_fresh(base, sched.ny_date) and not _quote_is_fresh(pp, sched.ny_date):
                            # Finnhub is session-fresh but the 15m bar is from a prior
                            # day: keep Finnhub price fields, let stale prepost only
                            # fill in what is missing (never the reverse).
                            out[sym] = {**pp, **base}
                        else:
                            out[sym] = {**base, **pp}
            except Exception as exc:
                logger.debug("US prepost quotes fallback failed in /us sentinel: %s", exc)
        return out

    (
        (morning_report_full, morning_conclusion, morning_stale),
        (guardrails, guardrails_stale),
        morning_pack,
        vix_hist_map,
        realtime_quotes,
        (histories, bundles, macro, calendar_alerts),
    ) = await asyncio.gather(
        db.get_morning_context_with_status(include_premarket=False),
        db.get_daily_guardrails_with_status(),
        db.get_context_pack(allow_fallback=False),
        _empty_hist_map() if market_closed else _fetch_vix_hist(),
        _fetch_us_quotes(),
        _fetch_us_sentinel_facts_data(market_data, us_tracked_syms),
    )

    # v5.5.8: mirror /report /pick — refresh the Invest GBP wallet conversion when the
    # portfolio was fetched before the FX rate was known (pure in-memory re-derivation).
    if (
        getattr(getattr(portfolio, "invest", None), "gbp_usd_rate", None) is None
        and macro.gbp_usd
        and hasattr(t212, "refresh_portfolio_fx")
    ):
        try:
            portfolio = t212.refresh_portfolio_fx(portfolio, macro.gbp_usd)
        except Exception as exc:
            logger.debug("refresh_portfolio_fx failed in /us sentinel: %s", exc)

    tech_map: dict[str, Any] = {}
    for sym in us_tracked_syms:
        snap = compute_technical_snapshot(sym, histories.get(sym), bundle=bundles.get(sym))
        if snap is not None:
            tech_map[sym] = snap

    has_anomaly, anomaly_syms, anomaly_reasons = _evaluate_us_anomaly_gate(
        portfolio=portfolio,
        cfg=cfg,
        histories=histories,
        bundles=bundles,
        macro=macro,
        vix_hist=vix_hist_map.get("^VIX"),
        morning_context_pack=morning_pack,
        realtime_quotes=realtime_quotes,
        schedule_phase=sched.phase,
        ny_date=sched.ny_date,
    )

    us_grounding = ""
    search_targets: list[str] = []
    if has_anomaly and cfg.models.enable_forced_search_grounding:
        # Anomaly names first, then active-US-order names (e.g. AMZN pending but flat)
        # so the model never has to re-search an order target on its second round.
        base_targets = anomaly_syms or us_held_syms or us_tracked_syms
        search_targets = list(dict.fromkeys([*base_targets, *us_pending_syms]))[:6]

        def _anomaly_rank(sym: str) -> float:
            quote = realtime_quotes.get(sym) or {}
            pct = quote.get("change_pct")
            if isinstance(pct, (int, float)):
                return abs(float(pct))
            hist_move = _daily_move_pct_from_hist(histories.get(sym))
            return abs(hist_move) if hist_move is not None else 0.0

        def _order_proximity_pct(sym: str) -> float | None:
            quote = realtime_quotes.get(sym) or {}
            cur = quote.get("current_price")
            if not cur:
                b = bundles.get(sym)
                cur = b.current_price if b is not None else None
            if not cur or float(cur) <= 0:
                return None
            best: float | None = None
            for order in portfolio.all_pending_orders():
                if order.yahoo_symbol != sym:
                    continue
                for trig in (order.limit_price, order.stop_price):
                    if trig and float(trig) > 0:
                        dist = abs(((float(cur) - float(trig)) / float(trig)) * 100.0)
                        best = dist if best is None else min(best, dist)
            return best

        def _is_priority(sym: str) -> bool:
            if _anomaly_rank(sym) >= 3.0:
                return True
            prox = _order_proximity_pct(sym)
            return prox is not None and prox <= _PENDING_PROXIMITY_PCT

        prio_syms = sorted(
            (s for s in dict.fromkeys([*anomaly_syms, *us_pending_syms]) if _is_priority(s)),
            key=_anomaly_rank,
            reverse=True,
        )[:2]
        if search_targets:
            us_grounding = await _facade_attr("run_forced_market_grounding", run_forced_market_grounding)(
                llm=llm,
                search_model=cfg.models.search_model,
                scope="us",
                symbols=search_targets,
                priority_symbols=prio_syms or None,
                ttl_seconds=900,
                ops_notifier=ops_notifier,
                db=db,
                # 30-min freshness window: scheduled pre-open runs always refresh,
                # while repeat /us taps within the window reuse the cached search.
                persist_ttl_seconds=1800,
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
            f"（持仓波动均 < ±{cfg.strategy.us_anomaly_move_pct:.1f}%，VIX 增量 < ±{cfg.strategy.us_anomaly_vix_change_pct:.1f}%，无挂单逼近触发）"
        )
    )
    cal_block = "\n" + " | ".join(calendar_alerts) if calendar_alerts else ""
    # v5.5.8: the morning report already embeds the full redlines — never paste them twice.
    veto_section = (
        ""
        if (
            not guardrails_stale
            and has_complete_embedded_vetoes(morning_report_full, guardrails)
        )
        else f"=== 今日风控红线 ===\n{veto_block}\n\n"
    )
    now_utc_label = datetime.now(timezone.utc).strftime("%H:%M")
    facts_text = build_us_sentinel_facts(
        cfg=cfg,
        us_tracked_syms=us_tracked_syms,
        bundles=bundles,
        macro=macro,
        calendar_alerts=calendar_alerts,
        anomaly_syms=anomaly_syms,
        us_pending_syms=us_pending_syms,
        tech_map=tech_map,
        fetched_utc_label=now_utc_label,
    )
    realtime_block = _build_us_realtime_snapshot_block(
        us_tracked_syms=us_tracked_syms,
        realtime_quotes=realtime_quotes,
        bundles=bundles,
        schedule_phase=sched.phase,
        ny_date=sched.ny_date,
    )
    recon_block = _build_us_order_reconciliation_block(morning_pack, portfolio, realtime_quotes, bundles)

    us_sys = build_agent_system_instruction(
        "你是【美股增量哨兵（Delta Sentinel）】。\n"
        "你的唯一职责：以今日晨报 `/report` 全文为绝对基线，结合最新美股行情/新闻/宏观，做一次开盘前二次核验！\n"
        "禁止对早报已决定的英股或正常无异动美股另起炉灶。\n"
        "严格二选一开头（禁用 Markdown # 和 **）：\n"
        "• 若今日早报之后无颠覆性美股新变量（或休市无新行情 / 本地异动门控平稳）：第一行直接写「✅ 维持早报计划」，"
        "随后仅用 2~3 条简写：美股开盘执行动作确认（严格以『当前仍有效挂单』为准，已撤销/已成交的不重复提醒）、"
        "止损/异动核对；仅当存在 7 天内财报或 3 天内除息时补充提醒，无则略过，严禁拿 15 天以上的远期财报硬凑静默期"
        "（控制在 380 字以内，绝不超 450 字）。\n"
        "• 若本地异动门控触发且突发新新闻、大幅跳空或宏观剧震真正改变赔率：第一行写「🚨 盘前增量修正」，"
        "直指要修正 /report 哪一条建议 + 最新数据依据 + 调整后指令（控制在 550 字以内，绝不超 600 字）。",
        cfg.strategy,
        compact_for_us_sentinel=True,
    )
    prompt = (
        f"【任务：美股增量哨兵】当前时段：{phase_desc}（UK {sched.uk_date} {sched.uk_weekday_label}{cal_block}）。\n"
        f"{gate_block}\n\n"
        + (f"{realtime_block}\n\n" if realtime_block else "")
        + (f"{recon_block}\n\n" if recon_block else "")
        + f"=== 今日 /report 全文（必须对照逐条核验）===\n{morning_baseline}\n\n"
        + veto_section
        + f"=== 最新美股事实与可用现金 ===\n"
        f"{build_wallet_context_text(portfolio, cfg.strategy, compact=True)}\n"
        f"【美股活动挂单】：\n{trade_ctx['pending_orders_text']}\n\n"
        f"{facts_text}"
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
            # Align the tool cache window with /us's 30-min freshness: always hits the
            # just-finished /us pre-search, never serves a 2.5h-old report cache entry.
            search_cache_ttl_seconds=min(int(cfg.app.grounding_cache_ttl_seconds), 1800),
        )
        # Plan option A: keep exactly one escape-hatch search tool + a single round —
        # 90% of anomaly days answer in one pass, no wasted second Hydrogen round.
        tools = toolkit.declarations(allowed_tools={"search_market_news"})
        tool_executor = toolkit.as_executor()
        us_model = cfg.models.us_anomaly_model
        us_thinking = cfg.models.us_anomaly_thinking_level
        prompt += (
            f"\n\n⚠️ 工具使用纪律：实时宏观快照、美股盘前价与异动标的联网检索已于 {now_utc_label} UTC 刚刚完成；"
            "仅当发现未覆盖的突发催化缺口时才可调用 1 次 search_market_news，否则直接输出最终简报。"
        )
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
        max_tool_rounds=1,
        temperature=0.15,
        usage_label="us_sentinel",
    )
    if not memo.startswith("⚠️"):
        memo = clean_telegram_plain_text(memo)
        # [用户授权调整] /us 严格兑现「✅<=450 / 🚨<=600」契约：仅看结论前缀，与门控状态解耦。
        us_max_chars = 600 if memo.lstrip().startswith("🚨") else 450
        if len(memo) > us_max_chars:
            memo = trim_text_at_line_boundary(memo, us_max_chars)
        # v5.5.8: never insert a report_text='' shell row on days without a morning
        # report — that would wipe the previous cycle's fallback baseline for chat /pick.
        if not morning_stale and bool(morning_report_full or morning_conclusion):
            await db.append_premarket_note(memo)
    logger.info(
        "us_sentinel completed [phase=%s gate=%s reasons=%d model=%s prompt_chars=%d search_targets=%d reply_chars=%d]",
        sched.phase,
        "anomaly" if has_anomaly else "calm",
        len(anomaly_reasons),
        us_model,
        len(prompt),
        len(search_targets),
        len(memo),
    )
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
