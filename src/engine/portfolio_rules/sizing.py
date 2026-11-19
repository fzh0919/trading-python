"""Deterministic GBP-to-shares position sizing notes for /report actions and /pick candidates."""

from __future__ import annotations

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
from src.engine.portfolio_rules.models_and_friction import compute_risk_based_size_gbp

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


def build_report_size_notes(top_actions: list[Any] | None, portfolio: Any | None) -> dict[tuple[str, str], str]:
    """Map ``(ACCOUNT, SYMBOL)`` -> deterministic size note for :func:`render_morning_report`."""
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
        if note:
            notes[(account, symbol)] = note
    return notes


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
