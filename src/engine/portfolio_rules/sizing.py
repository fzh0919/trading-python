"""Deterministic GBP-to-shares position sizing notes for /report actions and /pick candidates."""

from __future__ import annotations

import math
import re
from typing import Any

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot
from src.engine.portfolio_rules.dual_philosophy import (
    _resolve_pick_rr,
    evaluate_isa_value_gate,
    resolve_invest_dynamic_rr_floor,
)
from src.engine.portfolio_rules.models_and_friction import (
    _estimate_pending_order_gbp,
    compute_risk_based_size_gbp,
    is_invest_core_beta_symbol,
    is_invest_reservoir_symbol,
)

def _shares_from_gbp(
    gbp_amount: float,
    price_local: float,
    currency: str,
    symbol: str,
    gbp_usd_rate: float | None = None,
) -> int | None:
    """Convert a GBP risk budget into an integer share count, or None when FX/currency is unknown."""
    if gbp_amount <= 0 or price_local <= 0:
        return None
    raw_ccy = str(currency or "")
    upper = raw_ccy.upper()
    if raw_ccy in {"GBp", "GBx"} or upper == "GBX":
        price_gbp = price_local / 100.0
    elif upper == "GBP":
        price_gbp = price_local
    elif upper == "USD" and gbp_usd_rate and float(gbp_usd_rate) > 0:
        price_gbp = price_local / float(gbp_usd_rate)
    else:
        # Missing currency must return None: blindly dividing a .L symbol by 100 would
        # turn a GBP-denominated ETF quote (VWRP.L ≈ £114) into £1.14.
        return None
    if price_gbp <= 0:
        return None
    return max(0, int(gbp_amount // price_gbp))


_SIZE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SIZE_FRAC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
_SIZE_GBP_RE = re.compile(r"£\s*([\d,]+(?:\.\d+)?)")
_FULL_EXIT_TOKENS = ("全出", "清仓", "全部卖出", "全卖", "全部清仓")


def _fmt_shares(shares: float) -> str:
    if shares <= 0:
        return ""
    if abs(shares - round(shares)) < 0.005:
        return f"≈{int(round(shares)):,}股"
    return f"≈{shares:,.2f}股"


def derive_action_size_note(action: str, size_hint: str, position: Any | None = None) -> str:
    """Deterministic share-count note for /report top_actions (never LLM arithmetic).

    TRIM percentages/fractions are applied to the existing holding; GBP amounts are converted
    with the position's own GBP-per-share mark, so GBp/GBP 100x mix-ups cannot leak in.
    ADD percentages stay as-is because their base (cash vs target weight) is ambiguous.
    """
    act = str(action or "").strip().upper()
    hint = str(size_hint or "").strip()
    if act not in {"ADD", "TRIM", "STOP"}:
        return ""
    if any(tok in hint for tok in _FULL_EXIT_TOKENS):
        return ""
    if act == "STOP":
        return "触发即全出" if not hint else ""
    if not hint:
        return ""
    qty = float(getattr(position, "quantity", 0.0) or 0.0) if position is not None else 0.0
    if act != "TRIM":
        return ""
    m = _SIZE_PCT_RE.search(hint)
    if m and qty > 0:
        return _fmt_shares(qty * float(m.group(1)) / 100.0)
    frac = _SIZE_FRAC_RE.search(hint)
    if frac and qty > 0:
        den = float(frac.group(2))
        if den > 0:
            return _fmt_shares(qty * float(frac.group(1)) / den)
    amt = _SIZE_GBP_RE.search(hint)
    if amt and position is not None and qty > 0:
        try:
            amount = float(amt.group(1).replace(",", ""))
        except ValueError:
            return ""
        value = float(getattr(position, "current_value", 0.0) or 0.0)
        if amount > 0 and value > 0:
            return _fmt_shares(amount / (value / qty))
    return ""


def build_report_size_notes(
    top_actions: list[Any] | None,
    portfolio: Any | None,
    strategy: StrategySection | None = None,
) -> dict[tuple[str, str], str]:
    """Map ``(ACCOUNT, SYMBOL)`` -> deterministic size note for :func:`render_morning_report`.

    v5.5.9 M5: with ``strategy`` supplied, an Invest equity BUY/ADD whose ``size_hint``
    parses to a GBP amount exceeding the free liquid cash (while a distributing-ETF
    reservoir is held) additionally carries the compact ETF-to-equity funding note.
    Without ``strategy`` the behaviour is byte-identical to v5.5.8.
    """
    notes: dict[tuple[str, str], str] = {}
    if portfolio is None:
        return notes
    positions = list(portfolio.all_positions())
    for act in top_actions or []:
        account = str(getattr(act, "account", "") or "").strip().upper()
        symbol = str(getattr(act, "symbol", "") or "").strip().upper()
        if not symbol or symbol == "CASH":
            continue
        pos = next(
            (
                p
                for p in positions
                if str(getattr(p, "account", "")).strip().upper() == account
                and symbol
                in {
                    str(getattr(p, "yahoo_symbol", "")).strip().upper(),
                    str(getattr(p, "ticker", "")).strip().upper(),
                }
            ),
            None,
        )
        note = derive_action_size_note(getattr(act, "action", ""), getattr(act, "size_hint", ""), pos)
        if strategy is not None:
            funding = _report_reservoir_funding_suffix(act, portfolio, strategy)
            if funding:
                note = f"{note}；{funding}" if note else funding
        if note:
            notes[(account, symbol)] = note
    return notes


# ---------------------------------------------------------------------------
# v5.5.9 M5: ETF-to-equity funding (清宽基换股) — deterministic, advisory, zero quota.
# ---------------------------------------------------------------------------
def _invest_reservoir_holdings(
    portfolio: Any,
    strategy: StrategySection,
) -> list[tuple[str, float, Any]]:
    """Invest reservoir ``(symbol, value_gbp, position)`` rows sorted by value desc."""
    held: list[tuple[str, float, Any]] = []
    invest_snap = getattr(portfolio, "invest", None)
    for pos in getattr(invest_snap, "positions", None) or []:
        sym = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
        if sym and is_invest_reservoir_symbol(sym, strategy):
            held.append((sym, float(getattr(pos, "current_value", 0.0) or 0.0), pos))
    held.sort(key=lambda r: -r[1])
    return held


def _reservoir_proceeds_per_share_gbp(pos: Any) -> float | None:
    """GBP proceeds per reservoir share from the position's own price/currency mark.

    LSE distributing ETFs quote either GBP (VWRL.L ≈ £1xx) or pence (ISF.L ≈ 1,0xx GBp);
    mixing them up would fake a 100x share count, so unknown currencies return None
    and the funding note simply drops the share estimate instead of guessing.
    """
    price = float(getattr(pos, "cur_price", 0.0) or 0.0)
    ccy = str(getattr(pos, "inst_currency", "") or "").strip()
    upper = ccy.upper()
    if price <= 0:
        return None
    if ccy in {"GBp", "GBx"} or upper == "GBX":
        return price / 100.0
    if upper == "GBP":
        return price
    return None


def _reservoir_funding_need_gbp(ticket_gbp: float, free_gbp: float, strategy: StrategySection) -> float:
    """Gap between a ticket and free cash, grossed up by the spread/FX buffer (ceil)."""
    gap = max(0.0, float(ticket_gbp) - float(free_gbp))
    if gap <= 0:
        return 0.0
    buffer_pct = float(getattr(strategy, "invest_reservoir_funding_buffer_pct", 2.0) or 0.0)
    return math.ceil(gap * (1.0 + buffer_pct / 100.0))


def _reservoir_funding_min_need_gbp(strategy: StrategySection) -> float:
    return float(getattr(strategy, "invest_reservoir_funding_min_need_gbp", 200.0) or 0.0)


def _pick_funding_source(
    need_gbp: float,
    reservoir: list[tuple[str, float, Any]],
) -> tuple[str, float, Any] | None:
    """Largest reservoir holding that can fully cover the gap, else None."""
    for cand_sym, cand_value, cand_pos in reservoir:
        if cand_value + 1e-9 >= need_gbp:
            return cand_sym, cand_value, cand_pos
    return None


def _reservoir_shares_txt(need_gbp: float, cand_pos: Any) -> str:
    proceeds = _reservoir_proceeds_per_share_gbp(cand_pos)
    if proceeds is None or proceeds <= 0:
        return ""
    shares = int(math.ceil(need_gbp / proceeds))
    if shares <= 0:
        return ""
    return f"(≈{shares:,}股)"


def _reservoir_fx_txt(target_symbol: str, strategy: StrategySection) -> str:
    sym = str(target_symbol or "").strip().upper()
    if sym.endswith(".L"):
        return "同币种0%换汇"
    fx = float(getattr(strategy, "fx_fee_pct", 0.15) or 0.15)
    if sym.endswith((".DE", ".PA", ".AS", ".MC", ".MI", ".SW")):
        return f"GBP→EUR单边{fx:.2f}%换汇"
    return f"GBP→USD单边{fx:.2f}%换汇"


def build_etf_to_equity_funding_note(
    actions: list[Any] | None,
    portfolio: Any | None,
    *,
    strategy: StrategySection,
    sizing_map: dict[str, Any] | None = None,
) -> str:
    """v5.5.9 M5 deterministic paired pre-instruction for /pick kept actions.

    Fires only when a kept Invest equity BUY/ADD ticket exceeds the free liquid cash
    while a distributing-ETF reservoir is held: Python computes the gap (grossed up by
    the 2% spread/FX buffer) and emits ``[资金腾挪·零Quota] 先卖蓄水池 → 再建仓``.
    Free cash >= ticket never touches the reservoir (zero-churn); the £200 noise floor
    leaves sub-floor gaps to the cheaper plain FX path; SWAP/ROTATE funding stays owned
    by the weak-position replacement machinery. At most one note per /pick run.
    """
    if portfolio is None or not bool(getattr(strategy, "invest_reservoir_enabled", True)):
        return ""
    reservoir = _invest_reservoir_holdings(portfolio, strategy)
    if not reservoir:
        return ""
    invest_snap = getattr(portfolio, "invest", None)
    if invest_snap is None:
        return ""
    running_free = float(getattr(invest_snap, "available_to_trade", 0.0) or 0.0)
    sizing_map = sizing_map or {}
    pos_by_sym = {
        str(getattr(p, "yahoo_symbol", "") or "").strip().upper(): p
        for p in portfolio.all_positions()
    }
    pending_by_sym: dict[str, float] = {}
    for order in getattr(invest_snap, "pending_orders", None) or []:
        if str(getattr(order, "side", "") or "").strip().upper() != "BUY":
            continue
        sym = str(getattr(order, "yahoo_symbol", "") or "").strip().upper()
        if not sym:
            continue
        pending_by_sym[sym] = pending_by_sym.get(sym, 0.0) + float(
            _estimate_pending_order_gbp(order, pos_by_symbol=pos_by_sym) or 0.0
        )

    for item in actions or []:
        act_up = str(getattr(item, "action", "") or "").strip().upper()
        if act_up not in {"BUY", "ADD"}:
            continue
        acct = str(getattr(item, "account", "") or "").strip().upper()
        if acct != "INVEST":
            continue
        sym = str(getattr(item, "symbol", "") or "").strip().upper()
        if not sym or is_invest_core_beta_symbol(sym, strategy):
            continue
        size = sizing_map.get(f"{acct}|{sym}") or sizing_map.get(f"{acct}|{sym.split('.')[0]}")
        if isinstance(size, dict) and bool(size.get("is_etf")):
            continue  # core/themed ETF tickets never draw on the reservoir
        suggested = float(size.get("suggested_gbp") or 0.0) if isinstance(size, dict) else 0.0
        planned_raw = getattr(item, "planned_amount_gbp", None)
        planned: float | None = None
        if planned_raw is not None and float(planned_raw) > 0:
            planned = float(planned_raw)
            if suggested > 0:
                planned = min(planned, suggested)
        elif suggested > 0:
            planned = suggested
        else:
            zone = str(getattr(item, "entry_zone", "") or "")
            m = _SIZE_GBP_RE.search(zone)
            if m:
                try:
                    planned = float(m.group(1).replace(",", ""))
                except ValueError:
                    planned = None
        if planned is None or planned <= 0:
            continue
        existing = pending_by_sym.get(sym, 0.0) or pending_by_sym.get(sym.split(".")[0], 0.0)
        effective_ticket = max(0.0, float(planned) - float(existing))
        if effective_ticket <= 0:
            continue
        need = _reservoir_funding_need_gbp(effective_ticket, running_free, strategy)
        if need >= _reservoir_funding_min_need_gbp(strategy):
            source = _pick_funding_source(need, reservoir)
            if source is not None:
                cand_sym, _cand_value, cand_pos = source
                buffer_pct = float(
                    getattr(strategy, "invest_reservoir_funding_buffer_pct", 2.0) or 0.0
                )
                return (
                    f"[资金腾挪·零Quota] 先卖出 Invest {cand_sym} £{need:,.0f}"
                    f"{_reservoir_shares_txt(need, cand_pos)} 释放现金 → 再建仓 {sym} "
                    f"£{effective_ticket:,.0f}"
                    f"（含{buffer_pct:.0f}%价差/汇率缓冲；{_reservoir_fx_txt(sym, strategy)}；"
                    "应税部分处置涉CGT自行核查）"
                )
        running_free = max(0.0, running_free - effective_ticket)
    return ""


def _report_reservoir_funding_suffix(
    act: Any,
    portfolio: Any,
    strategy: StrategySection,
) -> str:
    """Compact /report funding suffix for one Invest equity BUY/ADD top action.

    The /report PM only supplies ``size_hint`` text (no deterministic sizing map), so
    the paired instruction is emitted ONLY when the hint parses to a GBP amount —
    Python never invents a ticket size. Unparseable hints defer to the precise /pick run.
    """
    act_up = str(getattr(act, "action", "") or "").strip().upper()
    if act_up not in {"BUY", "ADD"}:
        return ""
    # v5.5.9 M5 P1-2 fix: the paired instruction moves INVEST money only — an ISA
    # ticket must never be told to sell the taxable Invest reservoir (cross-account,
    # cross-tax-scheme funding is impossible by construction).
    account = str(getattr(act, "account", "") or "").strip().upper()
    if account != "INVEST":
        return ""
    symbol = str(getattr(act, "symbol", "") or "").strip().upper()
    # Core broad-base ETF tickets (reservoir whitelist + curated core-family map)
    # never draw on the reservoir either — they ARE the parking machinery.
    if not symbol or is_invest_core_beta_symbol(symbol, strategy):
        return ""
    reservoir = _invest_reservoir_holdings(portfolio, strategy)
    if not reservoir:
        return ""
    free = float(getattr(getattr(portfolio, "invest", None), "available_to_trade", 0.0) or 0.0)
    m = _SIZE_GBP_RE.search(str(getattr(act, "size_hint", "") or ""))
    if not m:
        return ""
    try:
        amount = float(m.group(1).replace(",", ""))
    except ValueError:
        return ""
    if amount <= 0:
        return ""
    need = _reservoir_funding_need_gbp(amount, free, strategy)
    if need < _reservoir_funding_min_need_gbp(strategy):
        return ""
    source = _pick_funding_source(need, reservoir)
    if source is None:
        return ""
    cand_sym, _cand_value, cand_pos = source
    return (
        f"资金腾挪·零Quota：先卖Invest {cand_sym} £{need:,.0f}"
        f"{_reservoir_shares_txt(need, cand_pos)}释放现金"
    )


def _resolve_pick_sizing(
    item: Any,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    account: str,
    account_total_value_gbp: float | None = None,
    existing_position_value_gbp: float = 0.0,
    gbp_usd_rate: float | None = None,
    wallet: AccountSnapshot | None = None,
    risk_budget_scale: float = 1.0,
    tech: TechnicalSnapshot | None = None,
    macro: MacroSnapshot | None = None,
    regime: str = "NEUTRAL",
    existing_position_weight_pct: float = 0.0,
) -> dict[str, Any]:
    """Resolve the deterministic sizing note plus the raw sizing/R:R dicts for guardrails."""
    acct_up = str(account or "").strip().upper()
    symbol = str(getattr(item, "symbol", "") or "")
    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    stop_hint = float(getattr(item, "stop_price", 0.0) or 0.0)

    # v5.5.1 ISA value position: no tactical stop needed -> batch-in by valuation, not by ATR.
    if acct_up == "ISA" and not is_etf and stop_hint <= 0:
        gate = evaluate_isa_value_gate(
            bundle,
            tech,
            macro,
            strategy,
            item=item,
            symbol=symbol,
            crisis_mode=str(regime or "").upper() == "RISK_OFF",
        )
        if not gate.get("mos_pass"):
            return {"note": "", "size": None, "rr": None, "is_etf": is_etf}
        total = float(account_total_value_gbp or 0.0)
        if total <= 0:
            return {
                "note": f"ISA价值分批建议：{gate.get('summary')}，建议分批逢低建仓",
                "size": None,
                "rr": None,
                "is_etf": is_etf,
            }
        cap_pct = float(getattr(strategy, "max_single_name_portfolio_pct", 12.0) or 12.0)
        headroom = max(0.0, total * cap_pct / 100.0 - float(existing_position_value_gbp or 0.0))
        if headroom <= 0:
            return {"note": "", "size": None, "rr": None, "is_etf": is_etf}
        step = min(headroom, max(float(strategy.min_isa_equity_order_gbp), total * 0.04))
        return {
            "note": (
                f"ISA价值分批建议 £{step:,.0f}"
                f"（单票余量£{headroom:,.0f}，{gate.get('summary')}，建议分批逢低建仓）"
            ),
            "size": None,
            "rr": None,
            "is_etf": is_etf,
        }

    info = _resolve_pick_rr(
        item,
        bundle,
        strategy,
        symbol=symbol,
        account=account,
        wallet=wallet,
        tech=tech,
        regime=regime,
    )
    entry = float(info.get("entry") or 0.0)
    stop = float(info.get("stop") or 0.0)
    risk_pct = info.get("risk_pct")
    effective_risk_pct = info.get("effective_risk_pct")
    friction_pct = float(info.get("friction_pct") or 0.0)
    if entry <= 0 or stop <= 0 or not risk_pct or not effective_risk_pct:
        return {"note": "", "size": None, "rr": info, "is_etf": is_etf}
    sizing = compute_risk_based_size_gbp(
        account=account,
        strategy=strategy,
        effective_stop_pct=-abs(float(effective_risk_pct)),
        is_etf=is_etf,
        account_total_value_gbp=account_total_value_gbp,
        existing_position_value_gbp=float(existing_position_value_gbp or 0.0),
        risk_budget_scale=float(risk_budget_scale or 1.0),
    )
    suggested = sizing.get("suggested_gbp")
    if not suggested or float(suggested) <= 0:
        return {"note": "", "size": sizing, "rr": info, "is_etf": is_etf}
    cap_pct = float(sizing.get("single_name_cap_pct") or 0.0)
    parts = [
        f"以损定仓上限 £{float(suggested):,.0f}"
        f"（预算£{float(sizing.get('risk_budget_gbp') or 0.0):,.0f}÷含摩擦止损{float(effective_risk_pct):.1f}%"
        f"=纯止损{float(risk_pct):.1f}%+摩擦{friction_pct:.2f}%，单仓≤{cap_pct:.0f}%）"
    ]
    if acct_up == "INVEST":
        min_rr, tier_label = resolve_invest_dynamic_rr_floor(
            regime,
            tech,
            strategy,
            vix=(macro.vix if (macro is not None and macro.vix is not None) else None),
        )
        parts.append(f"动态门槛{tier_label}≥{min_rr:.1f}:1")
    shares = _shares_from_gbp(
        float(suggested),
        entry,
        str(getattr(bundle, "currency", "") or ""),
        symbol,
        gbp_usd_rate,
    )
    if shares is not None and shares > 0:
        parts.append(f"≈{shares}股(上限)")
    planned_amount = getattr(item, "planned_amount_gbp", None)
    if planned_amount is not None and float(planned_amount) > float(suggested) * 1.05:
        parts.append(f"建议金额£{float(planned_amount):,.0f}超上限，按£{float(suggested):,.0f}钳位")
    if sizing.get("below_min_order"):
        parts.append(f"低于最小建仓额£{float(sizing.get('min_order_gbp') or 0.0):,.0f}，勿碎单")
    return {"note": "，".join(parts), "size": sizing, "rr": info, "is_etf": is_etf}


def _build_pick_size_note(
    item: Any,
    bundle: TickerMarketBundle | None,
    strategy: StrategySection,
    *,
    account: str,
    account_total_value_gbp: float | None = None,
    existing_position_value_gbp: float = 0.0,
    gbp_usd_rate: float | None = None,
    wallet: AccountSnapshot | None = None,
    risk_budget_scale: float = 1.0,
    tech: TechnicalSnapshot | None = None,
    macro: MacroSnapshot | None = None,
    regime: str = "NEUTRAL",
    existing_position_weight_pct: float = 0.0,
) -> str:
    """Deterministic sizing hint: risk budget / stop distance for Invest, value batching for ISA."""
    return str(
        _resolve_pick_sizing(
            item,
            bundle,
            strategy,
            account=account,
            account_total_value_gbp=account_total_value_gbp,
            existing_position_value_gbp=existing_position_value_gbp,
            gbp_usd_rate=gbp_usd_rate,
            wallet=wallet,
            risk_budget_scale=risk_budget_scale,
            tech=tech,
            macro=macro,
            regime=regime,
            existing_position_weight_pct=existing_position_weight_pct,
        ).get("note")
        or ""
    )
