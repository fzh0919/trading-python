"""Four-bucket deterministic core policy gaps and allocation planning."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_technical_snapshot,
    estimate_portfolio_regional_lookthrough,
    infer_etf_index_family,
)
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.deployment_gate.common import (
    _account_value_gbp,
    _pending_core_etf_gbp,
    _round2,
    _sym_up,
)

# Only true broad-base vehicles keep the core-ETF privilege. Global/S&P 500 are the
# US-heavy core; FTSE 100 and Europe are accepted as *core diversifiers* so the ISA
# 60% target can be met without mechanically increasing US tech exposure. Themed,
# high-dividend, bond and commodity ETFs stay demoted to other_etf_gbp.
_CORE_ETF_INDEX_FAMILIES = {
    "Global All-World/Developed (全球宽基)",
    "S&P 500 (美股标普500)",
    "FTSE 100 (英国富时100)",
    "Europe Equity (欧洲宽基)",
}
_CORE_DIVERSIFIER_FAMILIES = {
    "FTSE 100 (英国富时100)",
    "Europe Equity (欧洲宽基)",
}

# v5.5.1 Architecture A: deterministic four-bucket core policy (percent of ISA NAV).
CORE_UPPER_BAND_PCT = 65.0          # ISA core hard upper band (>=65% freezes all new core buys)
CORE_ANCHOR_TARGET_PCT = 42.0       # global anchor target (ISA %)
CORE_ANCHOR_FLOOR_PCT = 32.0        # global anchor floor after US-tilt offset
CORE_EU_TARGET_PCT = 8.0            # Europe broad target (ISA %)
CORE_EU_CAP_PCT = 10.0              # Europe broad hard cap (ISA %)
CORE_UK_TARGET_PCT = 6.0            # UK broad target (ISA %)
CORE_UK_CAP_PCT = 8.0               # UK broad hard cap (ISA %)
CORE_US_TILT_TARGET_PCT = 4.0       # S&P 500 incremental target/cap (ISA %)
PORTFOLIO_UK_LOOKTHROUGH_CAP_PCT = 12.0   # whole-portfolio UK look-through cap (home-bias guard)
ISA_UK_LOOKTHROUGH_CAP_PCT = 14.0         # ISA UK look-through cap
EQUITY_US_OVERWEIGHT_FREEZE_PCT = 65.0    # freeze S&P 500 adds above this US share of equity
CORE_DAILY_BUDGET_MAX_PCT_ISA = 3.0       # single-day core DCA budget (ISA %, ~£1,720)
CORE_DAILY_BUDGET_MAX_GBP = 1600.0        # single-day core DCA absolute cap
CORE_NEW_SLEEVE_MIN_GBP = 1000.0          # new un-held sleeve minimum ticket
CORE_MAX_TICKETS_PER_REPORT = 1           # one new core ticket per /report (time diversification)

CORE_SLEEVE_GLOBAL_ANCHOR = "GLOBAL_ANCHOR"
CORE_SLEEVE_EU = "EU_DIVERSIFIER"
CORE_SLEEVE_UK = "UK_DIVERSIFIER"
CORE_SLEEVE_US_TILT = "US_TILT"
_CORE_SLEEVE_ORDER = (
    CORE_SLEEVE_EU,
    CORE_SLEEVE_UK,
    CORE_SLEEVE_GLOBAL_ANCHOR,
    CORE_SLEEVE_US_TILT,
)
_CORE_FAMILY_TO_SLEEVE = {
    "Global All-World/Developed (全球宽基)": CORE_SLEEVE_GLOBAL_ANCHOR,
    "S&P 500 (美股标普500)": CORE_SLEEVE_US_TILT,
    "Europe Equity (欧洲宽基)": CORE_SLEEVE_EU,
    "FTSE 100 (英国富时100)": CORE_SLEEVE_UK,
}
_CORE_SLEEVE_ETF_PRIORITY: dict[str, list[str]] = {
    CORE_SLEEVE_EU: ["VERG.L", "VEUA.L", "VEUR.L", "IMEU.L"],
    CORE_SLEEVE_UK: ["CUKX.L", "VUKG.L", "ISF.L", "VUKE.L"],
    CORE_SLEEVE_GLOBAL_ANCHOR: ["VWRP.L", "FWRG.L", "SWDA.L", "VWRL.L"],
}
_CORE_SLEEVE_LABELS = {
    CORE_SLEEVE_GLOBAL_ANCHOR: "全球锚",
    CORE_SLEEVE_EU: "欧洲宽基",
    CORE_SLEEVE_UK: "英国富时100",
    CORE_SLEEVE_US_TILT: "标普500",
}
# Coarse UK look-through occupancy of a Europe-including-UK vehicle.
_EU_INCL_UK_LOOKTHROUGH = 0.22


def core_sleeve_for_symbol(
    symbol: str,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> str | None:
    """Map a symbol to one of the four deterministic core policy buckets."""
    sym = _sym_up(symbol)
    if not sym:
        return None
    bundle = (bundles or {}).get(sym) or (bundles or {}).get(symbol)
    return _CORE_FAMILY_TO_SLEEVE.get(infer_etf_index_family(sym, bundle) or "")


def _build_etf_set(
    strategy: StrategySection,
    universe_etf_symbols: list[str] | None,
    bundles: dict[str, TickerMarketBundle] | None,
) -> set[str]:
    """Core wide-base ETF membership = benchmark symbol + genuine broad-base universe ETFs.

    Other ETF-type instruments (e.g. a short-bond IBTA, a UK dividend or a thematic
    ETF) are excluded from the satellite equity bucket via ``_is_etf_symbol`` but do
    NOT inflate the core wide-base percentage.
    """
    etf_set = {_sym_up(strategy.benchmark_symbol)}
    for sym in universe_etf_symbols or []:
        sym_up = _sym_up(sym)
        if not sym_up:
            continue
        bundle = (bundles or {}).get(sym_up) or (bundles or {}).get(sym)
        family = infer_etf_index_family(sym_up, bundle)
        if family in _CORE_ETF_INDEX_FAMILIES:
            etf_set.add(sym_up)
    return etf_set


def _is_etf_symbol(symbol: str, etf_set: set[str], bundles: dict[str, TickerMarketBundle] | None) -> bool:
    sym = _sym_up(symbol)
    if sym in etf_set:
        return True
    b = (bundles or {}).get(sym)
    return bool(b is not None and str(getattr(b, "quote_type", "") or "").upper() == "ETF")


def is_core_diversifier_symbol(
    symbol: str,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> bool:
    """True for broad-base FTSE 100 / Europe ETFs that diversify the US-heavy core."""
    sym = _sym_up(symbol)
    if not sym:
        return False
    bundle = (bundles or {}).get(sym) or (bundles or {}).get(symbol)
    return infer_etf_index_family(sym, bundle) in _CORE_DIVERSIFIER_FAMILIES


def core_etf_universe_symbols(
    strategy: StrategySection,
    universe_etf_symbols: list[str] | None,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> list[str]:
    """Sorted core wide-base universe (global/S&P + FTSE100/Europe diversifiers)."""
    return sorted(_build_etf_set(strategy, universe_etf_symbols, bundles))


def compute_core_policy_gaps(
    portfolio: PortfolioData,
    strategy: StrategySection,
    etf_set: set[str],
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> dict[str, Any]:
    """Deterministic four-bucket ISA core policy gaps (no LLM, no network, no DB)."""
    isa = portfolio.isa
    isa_val = _account_value_gbp(isa)
    pos_by_sym = {p.yahoo_symbol: p for p in portfolio.all_positions()}
    held_gbp: dict[str, float] = {s: 0.0 for s in _CORE_SLEEVE_ORDER}
    pending_gbp: dict[str, float] = {s: 0.0 for s in _CORE_SLEEVE_ORDER}
    held_symbols: dict[str, list[str]] = {s: [] for s in _CORE_SLEEVE_ORDER}
    pending_symbols: dict[str, list[str]] = {s: [] for s in _CORE_SLEEVE_ORDER}
    bucket_by_symbol: dict[str, str] = {}

    for sym in etf_set:
        sleeve = core_sleeve_for_symbol(sym, bundles)
        if sleeve:
            bucket_by_symbol[_sym_up(sym)] = sleeve

    for pos in isa.positions:
        sym = _sym_up(pos.yahoo_symbol)
        sleeve = core_sleeve_for_symbol(sym, bundles)
        if not sleeve:
            continue
        held_gbp[sleeve] += float(pos.current_value or 0.0)
        held_symbols[sleeve].append(sym)
        bucket_by_symbol[sym] = sleeve
        bucket_by_symbol.setdefault(sym.split(".")[0], sleeve)

    for order in isa.pending_orders:
        if str(order.side or "").strip().upper() != "BUY":
            continue
        sym = _sym_up(order.yahoo_symbol)
        sleeve = core_sleeve_for_symbol(sym, bundles)
        if not sleeve:
            continue
        pending_gbp[sleeve] += _pending_core_etf_gbp(order, pos_by_sym)
        pending_symbols[sleeve].append(sym)
        bucket_by_symbol[sym] = sleeve
        bucket_by_symbol.setdefault(sym.split(".")[0], sleeve)

    def _pct(gbp: float) -> float:
        return _round2((gbp / isa_val) * 100.0) if isa_val > 0 else 0.0

    committed_pct = {s: _pct(held_gbp[s] + pending_gbp[s]) for s in _CORE_SLEEVE_ORDER}
    committed_core_pct = _round2(sum(committed_pct.values()))
    us_tilt_pct = committed_pct[CORE_SLEEVE_US_TILT]

    regional = estimate_portfolio_regional_lookthrough(portfolio, bundles)
    equity_us_share = float(regional.get("equity_us_share_pct") or 0.0)
    isa_uk_pct = float(regional.get("isa_uk_pct") or 0.0)
    portfolio_uk_pct = float(regional.get("uk_pct_nav") or 0.0)

    us_tilt_buy_frozen = bool(
        us_tilt_pct >= CORE_US_TILT_TARGET_PCT
        or equity_us_share > EQUITY_US_OVERWEIGHT_FREEZE_PCT
    )
    excess_us_tilt_pct = _round2(max(0.0, us_tilt_pct - CORE_US_TILT_TARGET_PCT))
    effective_anchor_target_pct = _round2(
        max(CORE_ANCHOR_FLOOR_PCT, CORE_ANCHOR_TARGET_PCT - excess_us_tilt_pct)
    )

    anchor_gap_pct = _round2(max(0.0, effective_anchor_target_pct - committed_pct[CORE_SLEEVE_GLOBAL_ANCHOR]))
    eu_gap_pct = _round2(max(0.0, min(CORE_EU_TARGET_PCT, CORE_EU_CAP_PCT) - committed_pct[CORE_SLEEVE_EU]))
    uk_gap_pct = _round2(max(0.0, min(CORE_UK_TARGET_PCT, CORE_UK_CAP_PCT) - committed_pct[CORE_SLEEVE_UK]))
    us_tilt_gap_pct = (
        0.0 if us_tilt_buy_frozen else _round2(max(0.0, CORE_US_TILT_TARGET_PCT - us_tilt_pct))
    )

    isa_uk_headroom_pct = _round2(max(0.0, ISA_UK_LOOKTHROUGH_CAP_PCT - isa_uk_pct))
    portfolio_uk_headroom_pct = _round2(max(0.0, PORTFOLIO_UK_LOOKTHROUGH_CAP_PCT - portfolio_uk_pct))
    uk_headroom_pct = min(isa_uk_headroom_pct, portfolio_uk_headroom_pct)

    core_cap_pct = min(float(strategy.isa_core_etf_target_pct), CORE_UPPER_BAND_PCT)
    remaining_core_pct = _round2(max(0.0, core_cap_pct - committed_core_pct))
    sleeve_gap_sum_pct = _round2(anchor_gap_pct + eu_gap_pct + uk_gap_pct + us_tilt_gap_pct)
    core_total_gap_pct = _round2(min(sleeve_gap_sum_pct, remaining_core_pct))
    core_total_gap_gbp = _round2(core_total_gap_pct / 100.0 * isa_val)

    if equity_us_share > 70.0:
        priority = [CORE_SLEEVE_EU, CORE_SLEEVE_UK, CORE_SLEEVE_GLOBAL_ANCHOR, CORE_SLEEVE_US_TILT]
    else:
        gap_by_sleeve = {
            CORE_SLEEVE_GLOBAL_ANCHOR: anchor_gap_pct,
            CORE_SLEEVE_EU: eu_gap_pct,
            CORE_SLEEVE_UK: uk_gap_pct,
            CORE_SLEEVE_US_TILT: us_tilt_gap_pct,
        }
        priority = sorted(_CORE_SLEEVE_ORDER, key=lambda s: gap_by_sleeve.get(s, 0.0), reverse=True)

    def _bucket(sleeve: str, target: float, cap: float, gap: float) -> dict[str, Any]:
        return {
            "sleeve": sleeve,
            "label": _CORE_SLEEVE_LABELS[sleeve],
            "held_gbp": _round2(held_gbp[sleeve]),
            "pending_gbp": _round2(pending_gbp[sleeve]),
            "committed_gbp": _round2(held_gbp[sleeve] + pending_gbp[sleeve]),
            "committed_pct": committed_pct[sleeve],
            "target_pct": target,
            "cap_pct": cap,
            "gap_pct": gap,
            "gap_gbp": _round2(gap / 100.0 * isa_val),
            "held_symbols": sorted(set(held_symbols[sleeve])),
            "pending_symbols": sorted(set(pending_symbols[sleeve])),
            "frozen": bool(sleeve == CORE_SLEEVE_US_TILT and us_tilt_buy_frozen),
        }

    buckets = {
        CORE_SLEEVE_GLOBAL_ANCHOR: _bucket(
            CORE_SLEEVE_GLOBAL_ANCHOR, effective_anchor_target_pct, CORE_UPPER_BAND_PCT, anchor_gap_pct
        ),
        CORE_SLEEVE_EU: _bucket(CORE_SLEEVE_EU, CORE_EU_TARGET_PCT, CORE_EU_CAP_PCT, eu_gap_pct),
        CORE_SLEEVE_UK: _bucket(CORE_SLEEVE_UK, CORE_UK_TARGET_PCT, CORE_UK_CAP_PCT, uk_gap_pct),
        CORE_SLEEVE_US_TILT: _bucket(
            CORE_SLEEVE_US_TILT, CORE_US_TILT_TARGET_PCT, CORE_US_TILT_TARGET_PCT, us_tilt_gap_pct
        ),
    }

    return {
        "isa_value_gbp": _round2(isa_val),
        "committed_core_pct": committed_core_pct,
        "core_target_pct": _round2(core_cap_pct),
        "core_upper_band_pct": CORE_UPPER_BAND_PCT,
        "anchor_target_pct": effective_anchor_target_pct,
        "anchor_floor_pct": CORE_ANCHOR_FLOOR_PCT,
        "excess_us_tilt_pct": excess_us_tilt_pct,
        "us_tilt_buy_frozen": us_tilt_buy_frozen,
        "equity_us_share_pct": _round2(equity_us_share),
        "isa_uk_lookthrough_pct": _round2(isa_uk_pct),
        "portfolio_uk_lookthrough_pct": _round2(portfolio_uk_pct),
        "isa_uk_headroom_pct": isa_uk_headroom_pct,
        "portfolio_uk_headroom_pct": portfolio_uk_headroom_pct,
        "uk_headroom_pct": uk_headroom_pct,
        "uk_headroom_gbp": _round2(uk_headroom_pct / 100.0 * isa_val),
        "anchor_gap_pct": anchor_gap_pct,
        "eu_gap_pct": eu_gap_pct,
        "uk_gap_pct": uk_gap_pct,
        "us_tilt_gap_pct": us_tilt_gap_pct,
        "core_total_gap_pct": core_total_gap_pct,
        "core_total_gap_gbp": core_total_gap_gbp,
        "priority": priority,
        "buckets": buckets,
        "bucket_by_symbol": bucket_by_symbol,
        "prefer_core_diversifier": bool(eu_gap_pct > 0.0 or uk_gap_pct > 0.0),
    }


def _core_symbol_trend_metrics(
    symbol: str,
    bundles: dict[str, TickerMarketBundle] | None,
    histories: dict[str, pd.DataFrame] | None,
) -> dict[str, Any]:
    """Price / MA / RSI / 52w-drawdown snapshot for a core ETF (best effort)."""
    sym = _sym_up(symbol)
    bundle = (bundles or {}).get(sym) or (bundles or {}).get(symbol)
    hmap = histories or {}
    hist = hmap.get(sym) if sym in hmap else hmap.get(symbol)
    price = float(bundle.current_price) if (bundle is not None and bundle.current_price) else None
    snap: TechnicalSnapshot | None = None
    try:
        snap = compute_technical_snapshot(sym, hist, bundle=bundle)
    except Exception:
        snap = None
    if snap is not None and float(getattr(snap, "last_close", 0.0) or 0.0) > 0:
        price = float(snap.last_close)
    metrics = {
        "symbol": sym,
        "price": _round2(price) if price else None,
        "ma20": _round2(getattr(snap, "ma20", None)) if snap and snap.ma20 else None,
        "ma50": _round2(getattr(snap, "ma50", None)) if snap and snap.ma50 else None,
        "ma200": _round2(getattr(snap, "ma200", None)) if snap and snap.ma200 else None,
        "dist_ma50_pct": _round2(getattr(snap, "dist_ma50_pct", None)) if snap and snap.dist_ma50_pct is not None else None,
        "rsi14": _round2(getattr(snap, "rsi14", None)) if snap and snap.rsi14 is not None else None,
        "drawdown_52w_pct": _round2(getattr(snap, "drawdown_52w_pct", None)) if snap and snap.drawdown_52w_pct is not None else None,
    }
    return metrics


def compute_core_allocation_plan(
    portfolio: PortfolioData,
    strategy: StrategySection,
    isa_ledger: dict[str, Any],
    core_policy: dict[str, Any],
    etf_set: set[str],
    bundles: dict[str, TickerMarketBundle] | None = None,
    histories: dict[str, pd.DataFrame] | None = None,
    *,
    vix: float | None = None,
) -> dict[str, Any]:
    """Deterministic daily core DCA budget + at most one un-held ticket.

    The LLM never chooses the symbol: it may only APPROVE / SHRINK / DEFER this ticket.
    """
    isa = portfolio.isa
    isa_val = _account_value_gbp(isa)
    min_order = float(getattr(strategy, "min_isa_etf_order_gbp", 400.0) or 400.0)
    plan: dict[str, Any] = {
        "enabled": False,
        "daily_budget_gbp": 0.0,
        "tickets": [],
        "held_actions": [],
        "sleeve_summary": [],
        "deferred_notes": [],
        "symbol_metrics": {},
        "cash_floor_gbp": 0.0,
        "deployable_above_floor_gbp": 0.0,
        "core_total_gap_gbp": _round2(float(core_policy.get("core_total_gap_gbp") or 0.0)),
    }
    if not isinstance(core_policy, dict) or not core_policy.get("buckets"):
        plan["deferred_notes"].append("核心政策缺口未计算，今日不签发核心宽基票据")
        return plan
    if not isa_ledger.get("ok"):
        plan["deferred_notes"].append("ISA 账户快照降级，暂停核心宽基票据")
        return plan
    if isa_ledger.get("etf_open") is not True:
        plan["deferred_notes"].append("ISA 宽基通道关闭（退潮期未达深跌阶梯）")
        return plan
    if str(isa_ledger.get("status") or "") != "RUN" or str(isa_ledger.get("mode") or "") not in {
        "dca",
        "inflow",
        "ladder",
    }:
        plan["deferred_notes"].append(f"ISA 今日门控 {isa_ledger.get('status')}/{isa_ledger.get('mode')}，不催买")
        return plan
    core_cap_pct = min(float(strategy.isa_core_etf_target_pct), CORE_UPPER_BAND_PCT)
    if float(core_policy.get("committed_core_pct") or 0.0) >= core_cap_pct:
        plan["deferred_notes"].append(f"ISA 核心宽基已达 {core_cap_pct:.0f}% 上限，暂停新增")
        return plan
    core_gap_gbp = float(core_policy.get("core_total_gap_gbp") or 0.0)
    if core_gap_gbp < min_order:
        plan["deferred_notes"].append("四桶核心缺口不足最小建仓额，今日不签发票据")
        return plan

    cash_floor_gbp = isa_val * float(strategy.isa_cash_band_low_pct) / 100.0
    deployable_above_floor = max(0.0, float(isa.available_to_trade) - cash_floor_gbp)
    plan["cash_floor_gbp"] = _round2(cash_floor_gbp)
    plan["deployable_above_floor_gbp"] = _round2(deployable_above_floor)
    if deployable_above_floor < min_order:
        plan["deferred_notes"].append(
            f"守住 {strategy.isa_cash_band_low_pct:.0f}% 现金地板后可部署仅 £{deployable_above_floor:,.0f}，暂停定投"
        )
        return plan

    vix_scale = 0.5 if (vix is not None and float(vix) >= 30.0 and isa_ledger.get("mode") != "ladder") else 1.0
    raw_daily_cap = min(CORE_DAILY_BUDGET_MAX_GBP, isa_val * CORE_DAILY_BUDGET_MAX_PCT_ISA / 100.0) * vix_scale
    daily_budget = min(
        deployable_above_floor,
        core_gap_gbp,
        max(min_order, raw_daily_cap),
    )
    plan["daily_budget_gbp"] = _round2(daily_budget)
    plan["vix_scale"] = vix_scale

    priority = list(core_policy.get("priority") or _CORE_SLEEVE_ORDER)
    buckets = core_policy.get("buckets") or {}
    for sym in sorted(etf_set):
        plan["symbol_metrics"][_sym_up(sym)] = _core_symbol_trend_metrics(sym, bundles, histories)

    for sleeve in priority:
        bucket = buckets.get(sleeve) or {}
        if not bucket:
            continue
        summary = {
            "sleeve": sleeve,
            "label": bucket.get("label", _CORE_SLEEVE_LABELS.get(sleeve, sleeve)),
            "committed_pct": bucket.get("committed_pct", 0.0),
            "target_pct": bucket.get("target_pct", 0.0),
            "cap_pct": bucket.get("cap_pct", 0.0),
            "gap_pct": bucket.get("gap_pct", 0.0),
            "gap_gbp": bucket.get("gap_gbp", 0.0),
            "frozen": bool(bucket.get("frozen")),
        }
        plan["sleeve_summary"].append(summary)
        gap_gbp = float(bucket.get("gap_gbp") or 0.0)
        if bucket.get("frozen") or gap_gbp < min_order:
            continue
        pending_symbols = list(bucket.get("pending_symbols") or [])
        if pending_symbols:
            plan["deferred_notes"].append(
                f"{summary['label']}已有未成交买单 {'/'.join(pending_symbols[:2])}，今日不叠单"
            )
            continue
        held_symbols = list(bucket.get("held_symbols") or [])
        if held_symbols:
            plan["held_actions"].append(
                {
                    "symbol": held_symbols[0],
                    "sleeve": sleeve,
                    "gap_gbp": _round2(gap_gbp),
                    "instruction": "走 top_actions ADD，不写 core_deployment",
                }
            )
            continue
        if len(plan["tickets"]) >= CORE_MAX_TICKETS_PER_REPORT:
            break
        candidate = ""
        for sym in _CORE_SLEEVE_ETF_PRIORITY.get(sleeve, []):
            if _sym_up(sym) in etf_set:
                candidate = _sym_up(sym)
                break
        if not candidate:
            plan["deferred_notes"].append(
                f"{summary['label']}缺口 {gap_gbp:,.0f}，但候选池未配置对应宽基代码"
            )
            continue
        metrics = plan["symbol_metrics"].get(candidate) or {}
        execution_style = "MARKET"
        if sleeve != CORE_SLEEVE_GLOBAL_ANCHOR:
            price = metrics.get("price")
            ma200 = metrics.get("ma200")
            dd52 = metrics.get("drawdown_52w_pct")
            if (
                price is not None
                and ma200 is not None
                and float(price) < float(ma200) * 0.95
                and (dd52 is None or float(dd52) > -15.0)
            ):
                plan["deferred_notes"].append(
                    f"{candidate} 跌破MA200逾5%且未达-15%深跌阶梯，暂缓左侧接刀"
                )
                continue
            rsi = metrics.get("rsi14")
            dist_ma50 = metrics.get("dist_ma50_pct")
            if (rsi is not None and float(rsi) >= 72.0) or (
                dist_ma50 is not None and float(dist_ma50) > 5.0
            ):
                execution_style = "PULLBACK_LIMIT"
        if (
            gap_gbp >= CORE_NEW_SLEEVE_MIN_GBP
            and daily_budget < CORE_NEW_SLEEVE_MIN_GBP
        ):
            plan["deferred_notes"].append(
                f"新子桶 {candidate} 至少 £{CORE_NEW_SLEEVE_MIN_GBP:,.0f} 起投，当前预算 £{daily_budget:,.0f} 暂缓碎买"
            )
            continue
        recommended = min(daily_budget, gap_gbp)
        recommended = math.floor(recommended / 50.0) * 50.0
        if 400.0 <= gap_gbp <= 1150.0 and daily_budget >= gap_gbp:
            recommended = gap_gbp
        if recommended < min_order:
            plan["deferred_notes"].append(f"{candidate} 建议金额 £{recommended:,.0f} 低于最小建仓额")
            continue
        if sleeve in {CORE_SLEEVE_EU, CORE_SLEEVE_UK}:
            uk_occupancy = 1.0 if sleeve == CORE_SLEEVE_UK else (
                _EU_INCL_UK_LOOKTHROUGH if candidate.startswith(("VEUR", "VEUA", "IMEU", "SMEA")) else 0.0
            )
            uk_headroom_gbp = float(core_policy.get("uk_headroom_gbp") or 0.0)
            if uk_occupancy > 0 and uk_headroom_gbp < recommended * uk_occupancy:
                capped = math.floor(uk_headroom_gbp / uk_occupancy / 50.0) * 50.0
                if capped < min_order:
                    plan["deferred_notes"].append(
                        f"{candidate} 触及英国穿透上限 {ISA_UK_LOOKTHROUGH_CAP_PCT:.0f}%，今日不新增"
                    )
                    continue
                recommended = capped
        plan["tickets"].append(
            {
                "symbol": candidate,
                "sleeve": sleeve,
                "sleeve_label": summary["label"],
                "is_held": False,
                "current_pct": summary["committed_pct"],
                "target_pct": summary["target_pct"],
                "cap_pct": summary["cap_pct"],
                "gap_gbp": _round2(gap_gbp),
                "recommended_gbp": _round2(recommended),
                "daily_budget_gbp": _round2(daily_budget),
                "execution_style": execution_style,
                "price": metrics.get("price"),
                "ma20": metrics.get("ma20"),
                "ma50": metrics.get("ma50"),
                "ma200": metrics.get("ma200"),
                "rsi14": metrics.get("rsi14"),
                "dist_ma50_pct": metrics.get("dist_ma50_pct"),
                "drawdown_52w_pct": metrics.get("drawdown_52w_pct"),
                "reason": (
                    f"{summary['label']} {summary['committed_pct']:.1f}%→目标 {summary['target_pct']:.1f}%，"
                    f"今日预算上限 £{daily_budget:,.0f}"
                ),
            }
        )
        break

    plan["enabled"] = bool(plan["tickets"])
    return plan
