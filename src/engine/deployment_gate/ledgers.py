"""Pending-order classification, tactical quota guards, and account ledgers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import (
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
)
from src.engine.portfolio_rules import (
    _compute_adaptive_invest_stop_pct,
    _rescale_price_pair,
    _round_trip_friction_pct,
)
from src.engine.deployment_gate.common import (
    LONDON_TZ,
    PENDING_CORE_ETF_BUY,
    PENDING_NEW_EQUITY_BUY,
    PENDING_PROTECTIVE_SELL_STOP,
    _account_value_gbp,
    _find_position,
    _is_held_symbol,
    _native_risk_to_gbp,
    _parse_order_ts_ms,
    _parse_uk_midnight_ms,
    _pending_buy_gbp,
    _pending_core_etf_gbp,
    _position_opened_today,
    _round2,
    _sym_up,
)
from src.engine.deployment_gate.core_policy import (
    _is_etf_symbol,
    compute_core_policy_gaps,
    is_core_diversifier_symbol,
)

# ---------------------------------------------------------------------------
# 1. Pending-order three-way classification
# ---------------------------------------------------------------------------
def classify_pending_orders(
    portfolio: PortfolioData,
    strategy: StrategySection,
    etf_set: set[str],
    pos_by_sym: dict[str, PositionItem] | None = None,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> dict[str, Any]:
    """Split every active pending order into PROTECTIVE_SELL_STOP / CORE_ETF_BUY / NEW_EQUITY_BUY.

    Protective STOP/STOP_LIMIT sell orders are positive risk shields: they never block
    the gate and never trigger any "pending order pauses new entries" logic.
    """
    by_class: dict[str, list[PendingOrderItem]] = {
        PENDING_PROTECTIVE_SELL_STOP: [],
        PENDING_CORE_ETF_BUY: [],
        PENDING_NEW_EQUITY_BUY: [],
    }
    by_symbol: dict[str, str] = {}
    for order in portfolio.all_pending_orders():
        side = str(order.side or "").strip().upper()
        order_type = str(order.order_type or "").strip().upper()
        sym = _sym_up(order.yahoo_symbol)
        if side == "SELL" and "STOP" in order_type:
            klass = PENDING_PROTECTIVE_SELL_STOP
        elif side == "BUY" and _is_etf_symbol(sym, etf_set, bundles):
            klass = PENDING_CORE_ETF_BUY
        elif side == "BUY":
            klass = PENDING_NEW_EQUITY_BUY
        else:
            continue
        by_class[klass].append(order)
        by_symbol[sym] = klass
    return {"by_class": by_class, "by_symbol": by_symbol}


# ---------------------------------------------------------------------------
# 2. Account ledgers + tactical quota guards
# ---------------------------------------------------------------------------
def evaluate_tactical_quota_guards(
    portfolio: PortfolioData,
    strategy: StrategySection,
    etf_set: set[str],
    bundles: dict[str, TickerMarketBundle] | None = None,
    *,
    invest_financed_positions: list[str] | None = None,
    now_uk: datetime | None = None,
) -> dict[str, Any]:
    """Epoch-anchored daily single-shot cap + consecutive-loss cooldown / half-size recovery.

    Only Invest tactical sell orders whose *opening* BUY fill also happened on/after
    ``gate_epoch_date`` count towards the loss streak — legacy pre-epoch positions that
    get closed are exempt ("历史包袱出清").
    """
    now = now_uk or datetime.now(LONDON_TZ)
    now_ms = int(now.timestamp() * 1000)
    today_start_ms = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    epoch_ms = _parse_uk_midnight_ms(strategy.gate_epoch_date)
    lookback_ms = int(strategy.consecutive_loss_lookback_days) * 86_400_000
    cooldown_ms = int(strategy.consecutive_loss_cooldown_days) * 86_400_000

    # --- Same-day best-idea-only: distinct NEW equity entries (ADDs do not consume the cap)
    today_new_symbols: set[str] = set()
    today_fills: list[dict[str, Any]] = []
    invest_pos_by_sym = {p.yahoo_symbol: p for p in portfolio.invest.positions}
    for order in portfolio.invest.recent_orders:
        if str(order.side or "").strip().upper() != "BUY":
            continue
        if _is_etf_symbol(order.yahoo_symbol, etf_set, bundles):
            continue
        if order.filled_at_ms < today_start_ms:
            continue
        today_fills.append(
            {"symbol": order.yahoo_symbol, "filled_at_ms": order.filled_at_ms, "value": order.net_value}
        )
        pos_filled = _find_position(invest_pos_by_sym, order.yahoo_symbol)
        is_new_entry = pos_filled is None or _position_opened_today(pos_filled, today_start_ms)
        if is_new_entry:
            today_new_symbols.add(_sym_up(order.yahoo_symbol))
    today_pending: list[str] = []
    classified = classify_pending_orders(portfolio, strategy, etf_set, bundles=bundles)
    for order in classified["by_class"][PENDING_NEW_EQUITY_BUY]:
        if order.account.strip().upper() != "INVEST":
            continue
        created_ms = _parse_order_ts_ms(order.created_at)
        if created_ms is None or created_ms < today_start_ms:
            continue
        sym_pending = _sym_up(order.yahoo_symbol)
        pos_pending = _find_position(invest_pos_by_sym, order.yahoo_symbol)
        is_new_entry = pos_pending is None or _position_opened_today(pos_pending, today_start_ms)
        if is_new_entry and sym_pending not in today_new_symbols:
            today_new_symbols.add(sym_pending)
            today_pending.append(order.yahoo_symbol)
    today_equity_entries = len(today_new_symbols)

    daily_cap = int(strategy.max_new_equity_entries_per_day)
    daily_cap_reached = today_equity_entries >= daily_cap if daily_cap > 0 else False

    # --- Consecutive-loss chain (epoch-anchored, tactical entries only)
    loss_limit = int(strategy.consecutive_loss_limit)
    buys_by_sym: dict[str, list[HistoricalOrderItem]] = {}
    for order in portfolio.invest.recent_orders:
        if str(order.side or "").strip().upper() == "BUY" and not _is_etf_symbol(
            order.yahoo_symbol, etf_set, bundles
        ):
            buys_by_sym.setdefault(_sym_up(order.yahoo_symbol), []).append(order)
    for rows in buys_by_sym.values():
        rows.sort(key=lambda r: r.filled_at_ms)

    tactical_sells: list[HistoricalOrderItem] = []
    for order in portfolio.invest.recent_orders:
        if str(order.side or "").strip().upper() != "SELL":
            continue
        if _is_etf_symbol(order.yahoo_symbol, etf_set, bundles):
            continue
        if order.filled_at_ms < epoch_ms:
            continue  # pre-epoch close -> exempt
        rows = buys_by_sym.get(_sym_up(order.yahoo_symbol)) or []
        opened_after_epoch = any(epoch_ms <= r.filled_at_ms < order.filled_at_ms for r in rows)
        if not opened_after_epoch:
            # No BUY inside the 30-day order-history window. That is only genuine legacy
            # when the sell itself happened before the window aged past the epoch; a sell
            # well after gate_epoch_date with no visible BUY is a post-epoch position whose
            # opening fill simply fell out of the API history and must still count.
            held_over_history_window = (
                order.filled_at_ms - epoch_ms
                >= int(strategy.order_history_days) * 86_400_000
            )
            if not held_over_history_window:
                continue  # legacy cleanup, exempt
        tactical_sells.append(order)
    tactical_sells.sort(key=lambda r: r.filled_at_ms, reverse=True)
    # Count each symbol once: multiple partial trims of the same loser must not fake a streak.
    latest_sell_by_sym: dict[str, HistoricalOrderItem] = {}
    for sell in tactical_sells:
        latest_sell_by_sym.setdefault(_sym_up(sell.yahoo_symbol), sell)
    tactical_sells = sorted(latest_sell_by_sym.values(), key=lambda r: r.filled_at_ms, reverse=True)

    consecutive_losses = 0
    last_loss_ms: int | None = None
    for sell in tactical_sells:
        if now_ms - sell.filled_at_ms > lookback_ms:
            break
        if sell.realised_pnl < 0:
            consecutive_losses += 1
            if last_loss_ms is None:
                # tactical_sells is sorted newest-first; anchor cooldown to the MOST RECENT loss
                last_loss_ms = sell.filled_at_ms
        else:
            break

    in_loss_cooldown = False
    risk_budget_scale = 1.0
    if consecutive_losses >= loss_limit and last_loss_ms is not None:
        if now_ms - last_loss_ms < cooldown_ms:
            in_loss_cooldown = True
        else:
            # Only a financed tactical position opened AFTER last_loss_ms clears the 0.5R pilot;
            # pre-existing financed positions (e.g. live AMZN opened earlier) do not cancel it.
            post_streak_financed = False
            inv_pos_by_sym = {_sym_up(p.yahoo_symbol): p for p in portfolio.invest.positions}
            for fin_sym in invest_financed_positions or []:
                fsym = _sym_up(fin_sym)
                if any(r.filled_at_ms > last_loss_ms for r in (buys_by_sym.get(fsym) or [])):
                    post_streak_financed = True
                    break
                p_obj = inv_pos_by_sym.get(fsym)
                created_ms = _parse_order_ts_ms(getattr(p_obj, "created_at", None)) if p_obj is not None else None
                if created_ms is not None and created_ms > last_loss_ms:
                    post_streak_financed = True
                    break
            if not post_streak_financed:
                risk_budget_scale = float(strategy.recovery_pilot_risk_scale)

    return {
        "gate_epoch_date": strategy.gate_epoch_date,
        "gate_epoch_ms": epoch_ms,
        "today_new_equity_entries": today_equity_entries,
        "today_fills": today_fills,
        "today_pending_new_buys": today_pending,
        "daily_cap": daily_cap,
        "daily_cap_reached": daily_cap_reached,
        "tactical_losses_recent": consecutive_losses,
        "last_loss_ms": last_loss_ms,
        "in_loss_cooldown": in_loss_cooldown,
        "risk_budget_scale": _round2(risk_budget_scale),
    }


def compute_account_ledgers(
    portfolio: PortfolioData,
    classified: dict[str, Any],
    bundles: dict[str, TickerMarketBundle] | None,
    strategy: StrategySection,
    etf_set: set[str],
    *,
    tech_map: dict[str, TechnicalSnapshot] | None = None,
    now_uk: datetime | None = None,
) -> dict[str, Any]:
    """Compute ISA core-satellite ledger and Invest Portfolio-Heat / financed-slot ledger."""
    by_class = classified["by_class"]
    pos_by_sym = {p.yahoo_symbol: p for p in portfolio.all_positions()}

    # ---------------- ISA ledger ----------------
    isa = portfolio.isa
    isa_val = _account_value_gbp(isa)
    isa_ok = bool(isa.ok and not isa.is_fallback and isa_val > 0)
    core_etf_gbp = 0.0
    core_diversifier_gbp = 0.0
    other_etf_gbp = 0.0
    satellite_gbp = 0.0
    held_equity_syms: list[str] = []
    crumb_count = 0
    min_crumb = float(getattr(strategy, "min_isa_equity_order_gbp", 1200.0) or 1200.0)
    for pos in isa.positions:
        sym = _sym_up(pos.yahoo_symbol)
        if sym in etf_set:
            core_etf_gbp += float(pos.current_value)  # benchmark + core universe = core wide base
            if is_core_diversifier_symbol(sym, bundles):
                core_diversifier_gbp += float(pos.current_value)
        elif _is_etf_symbol(sym, etf_set, bundles):
            other_etf_gbp += float(pos.current_value)  # ETF but not core (e.g. short-bond IBTA)
        else:
            satellite_gbp += float(pos.current_value)
            held_equity_syms.append(sym)
            if 0 < float(pos.current_value) < min_crumb:
                crumb_count += 1

    isa_pending_core = [o for o in by_class[PENDING_CORE_ETF_BUY] if o.account.strip().upper() == "ISA"]
    isa_pending_equity = [o for o in by_class[PENDING_NEW_EQUITY_BUY] if o.account.strip().upper() == "ISA"]
    core_pending_gbp = sum(_pending_core_etf_gbp(o, pos_by_sym) for o in isa_pending_core)
    core_diversifier_pending_gbp = sum(
        _pending_core_etf_gbp(o, pos_by_sym)
        for o in isa_pending_core
        if is_core_diversifier_symbol(_sym_up(o.yahoo_symbol), bundles)
    )
    equity_pending_gbp = _pending_buy_gbp(isa_pending_equity, pos_by_sym)

    def _pct(gbp: float) -> float:
        return _round2((gbp / isa_val) * 100.0) if isa_val > 0 else 0.0

    committed_satellite_count = len(held_equity_syms) + len(
        {
            _sym_up(o.yahoo_symbol)
            for o in isa_pending_equity
            if _sym_up(o.yahoo_symbol) not in {_sym_up(s) for s in held_equity_syms}
        }
    )
    committed_satellite_pct = _pct(satellite_gbp + equity_pending_gbp)
    isa_satellite_saturated = bool(
        committed_satellite_pct >= float(strategy.isa_satellite_max_total_pct)
        or committed_satellite_count >= int(strategy.isa_satellite_max_count)
    )
    committed_core_gbp = core_etf_gbp + core_pending_gbp
    committed_diversifier_gbp = core_diversifier_gbp + core_diversifier_pending_gbp
    diversifier_share_pct = (
        _round2(committed_diversifier_gbp / committed_core_gbp * 100.0)
        if committed_core_gbp > 0
        else 0.0
    )
    core_policy = compute_core_policy_gaps(portfolio, strategy, etf_set, bundles)
    prefer_core_diversifier = bool(isa_ok and core_policy.get("prefer_core_diversifier"))
    isa_ledger = {
        "ok": isa_ok,
        "total_account_value_gbp": _round2(isa_val),
        "total_cash_gbp": _round2(isa.total_cash),
        "available_to_trade_gbp": _round2(isa.available_to_trade),
        "reserved_cash_gbp": _round2(isa.reserved_cash),
        "total_cash_pct": _pct(isa.total_cash),
        "deployable_pct": _pct(isa.available_to_trade),  # already net of reserved BUY cash!
        "core_etf_pct": _pct(core_etf_gbp),
        "core_diversifier_pct": _pct(committed_diversifier_gbp),
        "core_diversifier_share_of_core_pct": diversifier_share_pct,
        "prefer_core_diversifier": prefer_core_diversifier,
        "core_policy": core_policy,
        "other_etf_pct": _pct(other_etf_gbp),
        "committed_core_etf_pct": _pct(committed_core_gbp),
        "satellite_equity_pct": _pct(satellite_gbp),
        "committed_satellite_pct": committed_satellite_pct,
        "committed_satellite_count": committed_satellite_count,
        "held_equity_count": len(held_equity_syms),
        "legacy_crumb_count": crumb_count,
        "isa_satellite_saturated": isa_satellite_saturated,
        "pending_core_etf_buys": [_sym_up(o.yahoo_symbol) for o in isa_pending_core],
        "pending_new_equity_buys": [_sym_up(o.yahoo_symbol) for o in isa_pending_equity],
    }

    # ---------------- Invest ledger (Portfolio Heat & Financed slots) ----------------
    invest = portfolio.invest
    invest_val = _account_value_gbp(invest)
    invest_ok = bool(invest.ok and not invest.is_fallback and invest_val > 0)
    financed: list[str] = []
    unfinanced: list[str] = []
    heat_breakdown: dict[str, float] = {}
    total_heat = 0.0
    slots_used = 0

    protective_by_sym: dict[str, list[PendingOrderItem]] = {}
    for order in by_class[PENDING_PROTECTIVE_SELL_STOP]:
        if order.account.strip().upper() == "INVEST":
            protective_by_sym.setdefault(_sym_up(order.yahoo_symbol), []).append(order)

    for pos in invest.positions:
        sym = _sym_up(pos.yahoo_symbol)
        if _is_etf_symbol(sym, etf_set, bundles):
            continue
        bundle = (bundles or {}).get(pos.yahoo_symbol)
        friction_pct = _round_trip_friction_pct(pos, bundle, strategy, invest)
        stop_orders = protective_by_sym.get(sym) or []
        is_financed = False
        heat_gbp = 0.0
        if stop_orders:
            covered_qty = 0.0
            weighted_stop_num = 0.0
            aligned_stop = 0.0
            for order in stop_orders:
                if not (order.stop_price or order.limit_price):
                    continue
                raw_stop = float(order.stop_price or order.limit_price or 0.0)
                aligned = _rescale_price_pair(raw_stop, pos.avg_price)
                qty = float(order.quantity or 0.0)
                covered_qty += qty
                weighted_stop_num += aligned * qty
                aligned_stop = aligned  # single protective order => itself
            covered_qty = min(float(pos.quantity), covered_qty)
            if covered_qty > 0:
                aligned_stop = weighted_stop_num / max(
                    sum(float(o.quantity or 0.0) for o in stop_orders if (o.stop_price or o.limit_price)),
                    1e-9,
                )
            # Breakeven unlock requires the protective stop to cover ~all shares: a 60%
            # stop on a 100% position still leaves 40% naked downside risk.
            if (
                pos.quantity > 0
                and covered_qty >= float(pos.quantity) * 0.99
                and pos.avg_price > 0
                and aligned_stop >= pos.avg_price * (1.0 + friction_pct / 100.0)
            ):
                is_financed = True
                heat_gbp = 0.0
            else:
                covered_risk = max(0.0, float(pos.avg_price) - aligned_stop) * covered_qty
                uncovered_qty = max(0.0, float(pos.quantity) - covered_qty)
                if uncovered_qty > 0:
                    tech = (tech_map or {}).get(pos.yahoo_symbol)
                    stop_pct = _compute_adaptive_invest_stop_pct(tech, strategy)
                    uncovered_risk = (
                        float(pos.avg_price) * abs(float(stop_pct)) / 100.0 * uncovered_qty
                    )
                else:
                    uncovered_risk = 0.0
                heat_gbp = _native_risk_to_gbp(covered_risk + uncovered_risk, pos)
        else:
            tech = (tech_map or {}).get(pos.yahoo_symbol)
            stop_pct = _compute_adaptive_invest_stop_pct(tech, strategy)
            native_risk = float(pos.avg_price) * abs(float(stop_pct)) / 100.0 * float(pos.quantity)
            heat_gbp = _native_risk_to_gbp(native_risk, pos)
        heat_gbp = _round2(max(0.0, heat_gbp))
        heat_breakdown[sym] = heat_gbp
        total_heat += heat_gbp
        if is_financed:
            financed.append(sym)
        else:
            unfinanced.append(sym)
            slots_used += 1

    invest_pos_by_sym = {p.yahoo_symbol: p for p in invest.positions}
    invest_pending_all = list(
        dict.fromkeys(
            _sym_up(o.yahoo_symbol)
            for o in by_class[PENDING_NEW_EQUITY_BUY]
            if o.account.strip().upper() == "INVEST"
        )
    )
    invest_pending_new: list[str] = []
    invest_pending_add: list[str] = []
    for order in by_class[PENDING_NEW_EQUITY_BUY]:
        if order.account.strip().upper() != "INVEST":
            continue
        sym_pending = _sym_up(order.yahoo_symbol)
        if _is_held_symbol(invest_pos_by_sym, sym_pending):
            invest_pending_add.append(sym_pending)
        else:
            invest_pending_new.append(sym_pending)
    invest_pending_new = list(dict.fromkeys(invest_pending_new))
    invest_pending_add = list(dict.fromkeys(invest_pending_add))
    slots_used += len(invest_pending_new)
    invest_ledger = {
        "ok": invest_ok,
        "total_account_value_gbp": _round2(invest_val),
        "total_cash_gbp": _round2(invest.total_cash),
        "available_to_trade_gbp": _round2(invest.available_to_trade),
        "cash_pct": _round2((invest.total_cash / invest_val) * 100.0) if invest_val > 0 else 0.0,
        "financed_positions": financed,
        "unfinanced_positions": unfinanced,
        "heat_gbp_total": _round2(total_heat),
        "heat_breakdown": heat_breakdown,
        "slots_used": slots_used,
        "pending_equity_buys_all": invest_pending_all,
        "pending_new_equity_buys": invest_pending_new,
        "pending_add_equity_buys": invest_pending_add,
        "pending_protective_stops": sorted({_sym_up(o.yahoo_symbol) for o in by_class[PENDING_PROTECTIVE_SELL_STOP]}),
    }

    guards = evaluate_tactical_quota_guards(
        portfolio,
        strategy,
        etf_set,
        bundles,
        invest_financed_positions=financed,
        now_uk=now_uk,
    )
    invest_ledger.update(guards)
    return {"isa": isa_ledger, "invest": invest_ledger, "etf_set": sorted(etf_set)}
