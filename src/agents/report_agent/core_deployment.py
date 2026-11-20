"""Deployment gate safe wrapper, report recommendation ledger, and core-ETF deployment validation."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from src.agents.report_agent.fact_sheet import _CORE_DCA_HORIZON_DAYS
from src.agents.schemas import MorningReportDecision
from src.config import AppConfig
from src.data_providers.analytics import (
    TechnicalSnapshot,
    normalize_issuer_root,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.deployment_gate import (
    CORE_UPPER_BAND_PCT,
    core_etf_universe_symbols,
    evaluate_deployment_gate,
)
from src.engine.deployment_gate.common import (
    TRIGGER_A1_MIN_DIST_MA50_PCT,
    TRIGGER_A2_MAX_DIST_MA50_PCT,
    TRIGGER_A2_RSI_EXCLUDE_ABOVE,
)
from src.engine.screener.patterns import (
    detect_breakthrough_platform,
    detect_minervini_stage2,
    detect_vcp_contraction,
)

logger = logging.getLogger("src.agents.report_agent")

def _evaluate_deployment_gate_safely(
    cfg: AppConfig,
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame],
    macro: MacroSnapshot,
    tech_map: dict[str, TechnicalSnapshot],
    gate_histories: dict[str, pd.DataFrame],
    gate_candidates: list[dict[str, Any]],
    prev_gate_states: list[dict[str, Any]],
    *,
    pool_snaps: dict[str, Any] | None = None,
    cached_breadth_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any] | None:
    """Deterministic gate evaluation that must never break the /report pipeline."""
    try:
        return evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=cfg.strategy,
            bundles=bundles,
            histories={**gate_histories, **histories},
            macro=macro,
            tech_map=tech_map,
            candidates=gate_candidates,
            universe_etf_symbols=cfg.universe.etf_symbols,
            prev_gate_states=prev_gate_states,
            gbp_usd=getattr(macro, "gbp_usd", None),
            pool_snaps=pool_snaps,
            cached_breadth_histories=cached_breadth_histories,
        )
    except Exception as exc:  # noqa: BLE001 - gate failure must never block the report
        logger.warning("Deployment gate evaluation failed, banner omitted: %s", exc)
        return None


def _build_report_recommendations(
    decision: MorningReportDecision,
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    cfg: AppConfig,
) -> list[dict[str, Any]]:
    """Turn /report ADD/TRIM/STOP decisions into the same T+N ledger as /pick.

    HOLD/CASH are intentionally excluded. Report actions are marked source='report' and
    fill_status='ASSUMED' because price_zone may be a trigger zone rather than a fill price.
    """
    price_by_sym = {
        p.yahoo_symbol.upper(): float(p.cur_price)
        for p in portfolio.all_positions()
        if p.yahoo_symbol and float(getattr(p, "cur_price", 0.0) or 0.0) > 0
    }
    account_ok = {
        "ISA": bool(portfolio.isa.ok and not portfolio.isa.is_fallback),
        "INVEST": bool(portfolio.invest.ok and not portfolio.invest.is_fallback),
    }
    out: list[dict[str, Any]] = []
    for act in decision.top_actions:
        action = str(act.action or "").strip().upper()
        if action not in {"ADD", "TRIM", "STOP"}:
            continue
        sym = str(act.symbol or "").strip().upper()
        if not sym or sym == "CASH":
            continue
        acct = str(act.account or "").strip().upper()
        if acct == "ISA" and not account_ok["ISA"]:
            continue
        if acct == "INVEST" and not account_ok["INVEST"]:
            continue
        bundle = bundles.get(sym) or bundles.get(sym.split(".")[0])
        price = None
        if bundle is not None and bundle.current_price is not None and float(bundle.current_price) > 0:
            price = float(bundle.current_price)
        elif sym in price_by_sym:
            price = price_by_sym[sym]
        if price is None or price <= 0:
            continue
        out.append(
            {
                "account": act.account,
                "symbol": sym,
                "action": action,
                "entry_price": round(price, 4),
                "thesis": f"{act.reason} | {act.price_zone}"[:300],
                "horizon_days": max(1, int(cfg.strategy.settlement_holding_days)),
                "benchmark_symbol": resolve_market_benchmark(sym)[0],
                "source": "report",
                "fill_status": "ASSUMED",
            }
        )
    # v5.5.1: strategic core DCA is a separate ledger lane (source='core_dca') with a
    # 20-day horizon so 5-day tactical win-rate/Alpha is never polluted by broad ETF noise.
    for dep in getattr(decision, "core_deployment", []) or []:
        verdict = str(getattr(dep, "verdict", "APPROVE") or "APPROVE").strip().upper()
        if verdict not in {"APPROVE", "SHRINK"}:
            continue
        sym = str(getattr(dep, "symbol", "") or "").strip().upper()
        if not sym or sym == "CASH" or not account_ok["ISA"]:
            continue
        bundle = bundles.get(sym) or bundles.get(sym.split(".")[0])
        price = None
        if bundle is not None and bundle.current_price is not None and float(bundle.current_price) > 0:
            price = float(bundle.current_price)
        if price is None or price <= 0:
            continue
        amount = getattr(dep, "approved_gbp", None)
        size_txt = f"£{float(amount):,.0f}" if amount is not None else getattr(dep, "size_hint", "")
        out.append(
            {
                "account": str(getattr(dep, "account", "ISA") or "ISA").strip().upper(),
                "symbol": sym,
                "action": "BUY",
                "entry_price": round(price, 4),
                "thesis": f"核心宽基DCA | {getattr(dep, 'reason', '')} | {size_txt} | {getattr(dep, 'price_zone', '')}"[:300],
                "horizon_days": _CORE_DCA_HORIZON_DAYS,
                "benchmark_symbol": resolve_market_benchmark(sym)[0],
                "source": "core_dca",
                "fill_status": "ASSUMED",
            }
        )
    return out


async def _fetch_core_etf_candidates(
    market_data: MarketDataService,
    cfg: AppConfig,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame],
) -> list[str]:
    """Ensure the report has quotes/trends for the un-held core/diversifier ETF universe.

    One lightweight batch (news off), merged in place. Never fatal: PM can still use
    held data if the ETF quote batch fails.
    """
    core_syms = core_etf_universe_symbols(
        cfg.strategy,
        list(getattr(cfg.universe, "etf_symbols", None) or []),
        bundles,
    )
    missing = [s for s in core_syms if s not in bundles]
    if not missing or not callable(getattr(market_data, "get_bundles", None)):
        return core_syms
    try:
        fetched = await market_data.get_bundles(
            missing, include_news=False, include_earnings=True
        )
        fetched_hist = fetched[0] if isinstance(fetched, tuple) else {}
        fetched_bundles = fetched[1] if isinstance(fetched, tuple) else fetched
        if isinstance(fetched_hist, dict):
            histories.update({k: v for k, v in fetched_hist.items() if v is not None})
        if isinstance(fetched_bundles, dict):
            bundles.update({k: v for k, v in fetched_bundles.items() if v is not None})
    except Exception as exc:
        logger.debug("Report core ETF candidate fetch skipped: %s", exc)
    return core_syms


def _format_core_policy_ticket_block(
    gate_preview: dict[str, Any] | None,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame],
) -> str:
    """Render Python's deterministic core policy ticket for Bull/Bear/PM prompts.

    The PM never picks an ETF: it may only APPROVE / SHRINK / DEFER the single ticket.
    """
    isa = (gate_preview or {}).get("isa") if isinstance(gate_preview, dict) else {}
    plan = (isa or {}).get("core_allocation_plan") if isinstance(isa, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    tickets = plan.get("tickets") or []
    held_actions = plan.get("held_actions") or []
    lines: list[str] = []
    header = (
        "【今日核心宽基政策缺口票据（Python 确定性算定；PM 仅可 APPROVE / SHRINK / DEFER，"
        "严禁换代码或超额）】"
    )
    if tickets:
        lines.append(header)
        for t in tickets[:1]:
            sym = str(t.get("symbol") or "").strip().upper()
            ccy = str(t.get("currency") or getattr(bundles.get(sym), "currency", "") or "").strip()
            is_gbpence = ccy in {"GBp", "GBX"}

            def _fmt_quote(value: Any, *, gbp_ref: bool = False) -> str:
                txt = f"{float(value):g} {ccy}".strip() if ccy else f"{float(value):g}"
                if gbp_ref and is_gbpence:
                    txt += f" (≈£{float(value) / 100.0:.2f})"
                return txt

            parts = [
                f"• {sym}｜子桶 {t.get('sleeve_label')} {float(t.get('current_pct') or 0.0):.1f}%→目标 "
                f"{float(t.get('target_pct') or 0.0):.1f}%（上限 {float(t.get('cap_pct') or 0.0):.1f}%）",
                f"今日预算上限 £{float(t.get('recommended_gbp') or 0.0):,.0f}",
            ]
            price = t.get("price")
            if price is not None:
                parts.append(f"现价 {_fmt_quote(price, gbp_ref=True)}")
            if t.get("ma50") is not None:
                parts.append(f"MA50 {_fmt_quote(t['ma50'])}")
            if t.get("ma200") is not None:
                parts.append(f"MA200 {_fmt_quote(t['ma200'])}")
            if t.get("drawdown_52w_pct") is not None:
                parts.append(f"距52w高 {float(t['drawdown_52w_pct']):+.1f}%")
            if t.get("rsi14") is not None:
                parts.append(f"RSI14 {float(t['rsi14']):.0f}")
            lines.append(" | ".join(parts))
            if str(t.get("execution_style") or "").upper() == "PULLBACK_LIMIT":
                lines.append("  执行方式：PULLBACK_LIMIT（回踩 MA20 或现价 -2%~-3% 限价挂单，严禁现价追高）")
            lines.append(
                "  PM 输出：symbol 必须为该代码；APPROVE 按上限执行 / SHRINK 缩减金额 / DEFER 今日暂缓；"
                f"金额不得超过 £{float(t.get('recommended_gbp') or 0.0):,.0f}"
            )
    else:
        reasons = "；".join(str(n) for n in (plan.get("deferred_notes") or [])[:3])
        lines.append(
            "【今日核心宽基政策票据】今日无未持仓核心宽基新建票据，core_deployment 必须输出空数组 []。"
        )
        lines.append(f"  原因：{reasons or '核心已达标/现金预算不足/已有未成交挂单'}")
    if held_actions:
        held_txt = "、".join(
            f"{a.get('symbol')}({a.get('sleeve')})" for a in held_actions[:3] if a.get("symbol")
        )
        if held_txt:
            lines.append(
                f"• 子桶已持有 {held_txt}：只能写进 top_actions ADD，严禁写入 core_deployment"
            )
    return "\n".join(lines)


def _extract_gbp_amount(text: Any) -> float | None:
    match = re.search(r"£\s*([\d,]+(?:\.\d+)?)", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _validate_core_deployment(
    decision: MorningReportDecision,
    portfolio: PortfolioData,
    cfg: AppConfig,
    gate_dict: dict[str, Any] | None,
    bundles: dict[str, TickerMarketBundle],
) -> tuple[MorningReportDecision, list[str]]:
    """One-way clamp of PM core deployment against Python's deterministic policy ticket.

    The PM may only APPROVE / SHRINK / DEFER the single ticket Python issued today. It can
    never swap the symbol or enlarge the amount; any violation is intercepted or clamped.
    """
    if not decision.core_deployment:
        return decision, []
    notes: list[str] = []
    if not isinstance(gate_dict, dict) or not gate_dict.get("etf_set"):
        return decision.model_copy(update={"core_deployment": []}), [
            "[🛡️风控护栏修正] core_deployment 已清空：组合门控未运行或缺少核心宽基集合"
        ]
    isa_gate = gate_dict.get("isa") if isinstance(gate_dict.get("isa"), dict) else {}
    if isa_gate.get("etf_open") is False:
        return decision.model_copy(update={"core_deployment": []}), [
            "[🛡️风控护栏修正] core_deployment 已清空：ISA 宽基通道关闭（退潮期未达深跌阶梯）"
        ]

    min_order = float(getattr(cfg.strategy, "min_isa_etf_order_gbp", 400.0) or 400.0)
    isa_val = float(portfolio.isa.total_account_value or 0.0)
    cash_floor = isa_val * float(getattr(cfg.strategy, "isa_cash_band_low_pct", 5.0) or 0.0) / 100.0
    available = float(portfolio.isa.available_to_trade or 0.0)
    committed_core_pct = float(isa_gate.get("committed_core_etf_pct") or 0.0)
    plan = isa_gate.get("core_allocation_plan") if isinstance(isa_gate.get("core_allocation_plan"), dict) else None

    if plan is not None:
        tickets = plan.get("tickets") or []
        if not tickets:
            notes.append(
                "[🛡️风控护栏修正] core_deployment 已清空：今日无未持仓核心宽基政策票据"
                "（严禁 PM 自行挑选宽基代码）"
            )
            return decision.model_copy(update={"core_deployment": []}), notes
        ticket = tickets[0] if isinstance(tickets[0], dict) else {}
        ticket_sym = str(ticket.get("symbol") or "").strip().upper()
        ticket_cap = float(ticket.get("recommended_gbp") or 0.0)
        kept: list[Any] = []
        for idx, item in enumerate(decision.core_deployment):
            sym = str(item.symbol or "").strip().upper()
            verdict = str(getattr(item, "verdict", "APPROVE") or "APPROVE").strip().upper()
            acct = str(item.account or "ISA").strip().upper()
            if idx > 0:
                notes.append(f"[🛡️风控护栏修正] {sym or '空代码'}(BUY→拦截): 单次仅可审批 1 张核心票据")
                continue
            if verdict not in {"APPROVE", "SHRINK", "DEFER"}:
                notes.append(f"[🛡️风控护栏修正] {sym}(BUY→拦截): 非法 verdict={verdict}")
                continue
            if verdict == "DEFER":
                notes.append(f"[🛡️风控护栏修正] {sym}(BUY→暂缓): PM 判定 DEFER，今日不下单、不入 T+N 账本")
                continue
            if acct != "ISA":
                notes.append(f"[🛡️风控护栏修正] [{acct}] {sym}(BUY→拦截): 核心宽基新建仓仅限 ISA")
                continue
            if not ticket_sym or sym != ticket_sym:
                notes.append(
                    f"[🛡️风控护栏修正] {sym or '空代码'}(BUY→拦截): 仅可审批 Python 票据 "
                    f"{ticket_sym or '（无票据）'}，严禁换代码（如 VUAG/IUKD 等）"
                )
                continue
            amount = getattr(item, "approved_gbp", None)
            try:
                amount_f = float(amount) if amount is not None else None
            except (TypeError, ValueError):
                amount_f = None
            if amount_f is None:
                amount_f = _extract_gbp_amount(item.size_hint)
            if amount_f is None:
                amount_f = ticket_cap
            if amount_f <= 0:
                notes.append(f"[🛡️风控护栏修正] {sym}(BUY→拦截): 批准金额为 0")
                continue
            if amount_f < min_order:
                notes.append(
                    f"[🛡️风控护栏修正] {sym}(BUY→拦截): £{amount_f:,.0f}"
                    f" 低于 ISA ETF 最小建仓额 £{min_order:,.0f}"
                )
                continue
            if ticket_cap > 0 and amount_f > ticket_cap:
                notes.append(
                    f"[🛡️风控护栏修正] {sym}(BUY→钳位): £{amount_f:,.0f} 超 Python 票据上限 "
                    f"£{ticket_cap:,.0f}，已向下钳位"
                )
                amount_f = ticket_cap
            if committed_core_pct >= CORE_UPPER_BAND_PCT:
                notes.append(
                    f"[🛡️风控护栏修正] {sym}(BUY→拦截): ISA 核心宽基已达 "
                    f"{CORE_UPPER_BAND_PCT:.0f}% 硬上限"
                )
                continue
            if isa_val > 0 and available - amount_f < cash_floor:
                notes.append(
                    f"[🛡️风控护栏修正] {sym}(BUY→拦截): 成交后可用现金将跌破 "
                    f"{cfg.strategy.isa_cash_band_low_pct:.0f}% 结构地板 £{cash_floor:,.0f}"
                )
                continue
            kept.append(
                item.model_copy(
                    update={
                        "approved_gbp": round(amount_f, 2),
                        "size_hint": f"£{amount_f:,.0f}",
                        "action": "BUY",
                        "account": "ISA",
                    }
                )
            )
            gate_dict["isa"]["approved_report_core_gbp"] = round(amount_f, 2)
            gate_dict["isa"]["approved_core_symbol"] = ticket_sym
            break
        if len(kept) == len(decision.core_deployment) and not notes:
            return decision, []
        return decision.model_copy(update={"core_deployment": kept}), notes

    # Legacy gate dicts (offline unit fixtures): keep the structural rail without a ticket.
    allowed = {str(s).strip().upper() for s in gate_dict.get("etf_set") or [] if str(s).strip()}
    allowed_roots = {normalize_issuer_root(s) for s in allowed}
    held_isa: set[str] = set()
    for pos in portfolio.isa.positions:
        sym = str(pos.yahoo_symbol or "").strip().upper()
        if sym:
            held_isa.add(sym)
            held_isa.add(sym.split(".")[0])
    for order in portfolio.isa.pending_orders:
        if str(order.side or "").strip().upper() != "BUY":
            continue
        sym = str(order.yahoo_symbol or "").strip().upper()
        if sym:
            held_isa.add(sym)
            held_isa.add(sym.split(".")[0])
    kept_legacy: list[Any] = []
    seen: set[str] = set()
    for item in decision.core_deployment:
        sym = str(item.symbol or "").strip().upper()
        base = sym.split(".")[0]
        acct = str(item.account or "ISA").strip().upper()
        verdict = str(getattr(item, "verdict", "APPROVE") or "APPROVE").strip().upper()
        if verdict not in {"APPROVE", "SHRINK"}:
            notes.append(f"[🛡️风控护栏修正] {sym}(BUY→暂缓): PM 判定 DEFER，今日不下单")
            continue
        if len(kept_legacy) >= 2:
            notes.append(f"[🛡️风控护栏修正] {sym}(BUY→拦截): core_deployment 单次最多 2 条")
            continue
        if acct != "ISA":
            notes.append(f"[🛡️风控护栏修正] [{acct}] {sym}(BUY→拦截): 核心宽基新建仓仅限 ISA")
            continue
        if not sym or sym in held_isa or base in held_isa:
            notes.append(f"[🛡️风控护栏修正] {sym or '空代码'}(BUY→拦截): 已持仓请用 top_actions ADD")
            continue
        if sym not in allowed and normalize_issuer_root(sym) not in allowed_roots:
            notes.append(
                f"[🛡️风控护栏修正] {sym}(BUY→拦截): 不属于核心/分散宽基集合"
                "（仅 VWRP/VUAG/SWDA/CSP1/ISF/VERG 等）"
            )
            continue
        if sym in seen:
            notes.append(f"[🛡️风控护栏修正] {sym}(BUY→拦截): core_deployment 重复标的")
            continue
        bundle = bundles.get(sym) or bundles.get(sym.split(".")[0])
        price = float(bundle.current_price) if (bundle is not None and bundle.current_price) else 0.0
        if price <= 0:
            notes.append(f"[🛡️风控护栏修正] {sym}(BUY→拦截): 缺少实时行情，无法核对价位")
            continue
        amount = _extract_gbp_amount(item.size_hint)
        if amount is not None and 0 < amount < min_order:
            notes.append(
                f"[🛡️风控护栏修正] {sym}(BUY→拦截): £{amount:,.0f}"
                f" 低于 ISA ETF 最小建仓额 £{min_order:,.0f}"
            )
            continue
        seen.add(sym)
        kept_legacy.append(item)
    if len(kept_legacy) == len(decision.core_deployment) and not notes:
        return decision, []
    return decision.model_copy(update={"core_deployment": kept_legacy}), notes


def _looks_like_a2_breakout(
    snap: Any,
    tolerance_pct: float,
    a2_max_dist_pct: float,
) -> bool:
    """Cheap snapshot-shape test for the v5.5.9 M3 A2 breakout pre-filter.

    Mirrors the A2 band gates cheaply (no pattern IO): above MA200, BULL/PULLBACK,
    weekly not bearish, MA50 extension inside (tol, +14%], RSI below the 82 extreme
    line. In a bull tape this shape alone can match dozens of ordinary risers, so the
    caller MUST layer the local pattern confirmation on top before spending a bundle
    slot — never let plain uptrends eat the batch cap (v5.5.9 M3 P1 starvation fix).
    """
    dist = getattr(snap, "dist_ma50_pct", None)
    if dist is None:
        return False
    d200 = getattr(snap, "dist_ma200_pct", None)
    if d200 is None or float(d200) <= 0.0:
        return False
    if not (float(tolerance_pct) < float(dist) <= float(a2_max_dist_pct)):
        return False
    if str(getattr(snap, "trend_regime", "") or "") not in {"BULL", "PULLBACK"}:
        return False
    if getattr(snap, "weekly_bullish", None) is False:
        return False
    rsi = getattr(snap, "rsi14", None)
    if rsi is not None and float(rsi) >= TRIGGER_A2_RSI_EXCLUDE_ABOVE:
        return False
    return True


def _gate_candidate_needs_bundle(
    snap: Any,
    sym: str,
    tolerance_pct: float,
    a2_max_dist_pct: float,
    histories: dict[str, pd.DataFrame] | None,
) -> bool:
    """Bundle-allocation policy for un-held A-trigger candidates (v5.5.9 M3 P1).

    - A1 pullback band (TRIGGER_A1_MIN_DIST_MA50_PCT .. tol): always covered, matching
      the scan's band byte-for-byte so the earnings/ex-div mines see the full range.
    - A2 shape: only admitted when the LOCAL 0-network pattern confirmation passes —
      Minervini Stage-2 plus a VCP contraction or a platform breakout. The pre-filter's
      minervini call omits RS, which is a strict superset of the scan's with-RS call
      (>=6 of 7 vs >=7 of 8 criteria), so a scan-admitted A2 name can never be starved
      here; the shape-only false positives are what get cut instead.
    - A2 shape without a local history is skipped: the scan's A2 gates are unsatisfiable
      without hist anyway (fail-closed there, no bundle wasted here).
    """
    dist = getattr(snap, "dist_ma50_pct", None)
    if dist is None:
        return False
    d = float(dist)
    if TRIGGER_A1_MIN_DIST_MA50_PCT <= d <= float(tolerance_pct):
        return True
    if not _looks_like_a2_breakout(snap, float(tolerance_pct), float(a2_max_dist_pct)):
        return False
    hist = (histories or {}).get(sym)
    if hist is None or hist.empty:
        return False
    if not detect_minervini_stage2(hist, snap).get("passes"):
        return False
    return bool(detect_vcp_contraction(hist, snap) or detect_breakthrough_platform(hist))


async def _enrich_gate_candidate_bundles(
    market_data: MarketDataService,
    pool_syms: list[str],
    pool_snaps: dict[str, Any],
    bundles: dict[str, TickerMarketBundle],
    *,
    tolerance_pct: float = 1.5,
    a2_max_dist_pct: float = TRIGGER_A2_MAX_DIST_MA50_PCT,
    histories: dict[str, pd.DataFrame] | None = None,
    max_symbols: int = 8,
) -> dict[str, TickerMarketBundle]:
    """Attach lightweight earnings/ex-div bundles to un-held A1 pullback / A2 breakout candidates.

    Deployment-gate A-trigger scans un-held candidates only, so they normally have no
    bundle; without this the 5-day earnings and ex-dividend exclusion windows are blind.
    v5.5.9 M3 P1: the A2 lane is pattern-confirmed locally (0 network, using the caller's
    in-memory gate histories) so ordinary bull-tape risers can never consume the batch
    cap and starve genuine breakout names — the scan stays fail-closed on missing
    bundles, so bundle allocation IS the A2 channel's availability.
    One batched call, news disabled (0 LLM, earnings calendar only).
    """
    cand_bundles: dict[str, TickerMarketBundle] = dict(bundles)
    hit_syms = [
        sym
        for sym in pool_syms
        if sym in pool_snaps
        and sym not in bundles
        and _gate_candidate_needs_bundle(
            pool_snaps[sym], sym, float(tolerance_pct), float(a2_max_dist_pct), histories
        )
    ][: max(0, int(max_symbols))]
    if not hit_syms or not callable(getattr(market_data, "get_bundles", None)):
        return cand_bundles
    try:
        fetched = await market_data.get_bundles(
            hit_syms, include_news=False, include_earnings=True
        )
        fetched_map = fetched[1] if isinstance(fetched, tuple) else fetched
        if isinstance(fetched_map, dict):
            for sym, bundle in fetched_map.items():
                if bundle is not None:
                    cand_bundles[sym] = bundle
    except Exception as exc:
        logger.debug("Deployment-gate candidate bundle fetch skipped: %s", exc)
    return cand_bundles
