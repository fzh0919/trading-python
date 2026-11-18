"""v5.4.0 Deterministic Deployment Gate (CIO Gate) — pure, offline, zero-network engine.

Connects the daily /report Single Source of Truth with the /pick radar by enforcing:

1. Cash-ledger identity (``available_to_trade`` already excludes reserved BUY cash —
   never subtract pending BUY value twice) and the pending-order three-way split
   (PROTECTIVE_SELL_STOP / CORE_ETF_BUY / NEW_EQUITY_BUY).
2. ISA Core-Satellite structure gates (satellite <=30% & <=8 names incl. pending buys;
   legacy <min_isa_equity_order_gbp crumbs are grandfathered, never force-liquidated).
3. Invest risk-slot engine with financed-risk unlock (protective STOP >= avg cost
   releases the slot), daily best-idea-only cap and epoch-anchored loss-streak cooldown.
4. FX-distortion-free market regime (SPY USD benchmark, VIX, HY OAS) with anti-deadlock
   1d-enter / 3d-exit confirmation hysteresis and step-wise RISK_OFF -> NEUTRAL -> RISK_ON.
5. A/B/C/D triggers, master DEFEND protocol, one-way PM veto clamp, <=110-char banner
   and a 2-year offline replay tool.

This module performs zero network requests and zero direct database writes.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot, infer_etf_index_family, normalize_issuer_root
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
)
from src.engine.portfolio_rules import (
    DEFAULT_VIX_DANGER_THRESHOLD,
    _compute_adaptive_invest_stop_pct,
    _estimate_pending_order_gbp,
    _rescale_price_pair,
    _round_trip_friction_pct,
)

LONDON_TZ = ZoneInfo("Europe/London")

# Regime constants (design doc §5.2: RISK_ON requires HY OAS < 4.5%; RISK_OFF at >= 5.5%)
REGIME_HY_OAS_RISK_ON_PCT = 4.5
REGIME_HY_OAS_RISK_OFF_PCT_DEFAULT = 5.5
# ISA extreme low-cash defence line (design doc §6.1 D trigger)
ISA_EXTREME_LOW_CASH_PCT = 2.0
# A-trigger exclusion windows (calendar days)
TRIGGER_EARNINGS_EXCLUDE_DAYS = 5
TRIGGER_EXDIV_EXCLUDE_DAYS = 5
TRIGGER_RSI_EXCLUDE_ABOVE = 72.0
# Consecutive-loss: how far a sell may trail the matching BUY fill (ms)
GATE_EPOCH_PARSE_FAIL_NOTE = "gate_epoch_date 解析失败，按当日零点保守起算"

PENDING_PROTECTIVE_SELL_STOP = "PROTECTIVE_SELL_STOP"
PENDING_CORE_ETF_BUY = "CORE_ETF_BUY"
PENDING_NEW_EQUITY_BUY = "NEW_EQUITY_BUY"

_GATE_VERSION = "v5.4.0"
_BANNER_MAX_CHARS = 110


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def _sym_up(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _is_held_symbol(pos_by_sym: dict[str, Any], symbol: str) -> bool:
    """True when a symbol (or its base ticker) already exists in the given account book."""
    return _find_position(pos_by_sym, symbol) is not None


def _find_position(pos_by_sym: dict[str, Any], symbol: str) -> Any | None:
    """Find a position by exact Yahoo symbol or base ticker (case-insensitive)."""
    sym = _sym_up(symbol)
    if not sym:
        return None
    if symbol in pos_by_sym:
        return pos_by_sym[symbol]
    base = sym.split(".")[0]
    for key, pos in pos_by_sym.items():
        key_up = _sym_up(key)
        if key_up == sym or key_up.split(".")[0] == base:
            return pos
    return None


def _position_opened_today(pos: Any, today_start_ms: int) -> bool:
    """Best-effort check that a holding was opened today (for ADD vs NEW classification)."""
    created_ms = _parse_order_ts_ms(getattr(pos, "created_at", None))
    return created_ms is not None and created_ms >= today_start_ms


def _symbol_matches(a: str, b: str) -> bool:
    sa, sb = _sym_up(a), _sym_up(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    return sa.split(".")[0] == sb.split(".")[0]


def _round2(value: float | None, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _account_value_gbp(acct: AccountSnapshot) -> float:
    val = float(getattr(acct, "total_account_value", 0.0) or 0.0)
    if val > 0:
        return val
    pos_sum = sum(float(p.current_value) for p in (getattr(acct, "positions", None) or []))
    return max(pos_sum + float(getattr(acct, "total_cash", 0.0) or 0.0), 0.0)


def _parse_uk_midnight_ms(date_str: str) -> int:
    """Parse YYYY-MM-DD at UK 00:00 into epoch ms (Europe/London aware)."""
    try:
        dt = datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").replace(tzinfo=LONDON_TZ)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return int(datetime.now(LONDON_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _parse_order_ts_ms(value: Any) -> int | None:
    """Tolerant parser for pending-order created_at strings / ms ints."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _close_series(hist: pd.DataFrame | pd.Series | None) -> pd.Series:
    if isinstance(hist, pd.Series):
        closes = pd.to_numeric(hist, errors="coerce").dropna().astype(float)
        return closes[closes > 0]
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
    return closes[closes > 0]


def _native_risk_to_gbp(native_amount: float, pos: PositionItem) -> float:
    """Convert a native-currency risk amount into GBP using the position's own FX basis."""
    if native_amount <= 0:
        return 0.0
    # Keep the original casing: ``.upper()`` would turn LSE pence ("GBp"/"GBx") into "GBP"
    # and silently inflate the heat by 100x.
    raw_ccy = str(pos.inst_currency or "").strip()
    if raw_ccy in {"GBp", "GBx"} or raw_ccy.upper() == "GBX":
        return native_amount / 100.0
    if raw_ccy.upper() == "GBP":
        return native_amount
    if pos.current_value > 0 and pos.cur_price > 0 and pos.quantity > 0:
        gbp_per_native_unit = float(pos.current_value) / (float(pos.cur_price) * float(pos.quantity))
        return native_amount * gbp_per_native_unit
    return 0.0


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
# Soft target: once at least ~1/3 of the ISA core sleeve is in FTSE100/Europe,
# stop nudging towards diversifiers. This is a preference, never a hard block.
_CORE_DIVERSIFIER_SOFT_SHARE_PCT = 33.0


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


def _pending_buy_gbp(
    orders: list[PendingOrderItem],
    pos_by_sym: dict[str, PositionItem],
) -> float:
    return sum(_estimate_pending_order_gbp(o, pos_by_symbol=pos_by_sym) for o in orders)


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
    core_pending_gbp = _pending_buy_gbp(isa_pending_core, pos_by_sym)
    core_diversifier_pending_gbp = _pending_buy_gbp(
        [
            o
            for o in isa_pending_core
            if is_core_diversifier_symbol(_sym_up(o.yahoo_symbol), bundles)
        ],
        pos_by_sym,
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
    prefer_core_diversifier = bool(
        isa_ok
        and committed_core_gbp / isa_val * 100.0 < float(strategy.isa_core_etf_target_pct)
        and diversifier_share_pct < _CORE_DIVERSIFIER_SOFT_SHARE_PCT
    )
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


# ---------------------------------------------------------------------------
# 3. FX-distortion-free market regime with anti-deadlock confirmation
# ---------------------------------------------------------------------------
def _compute_raw_regime_metrics(
    closes: pd.Series,
    vix: float | None,
    hy_oas: float | None,
    strategy: StrategySection,
) -> dict[str, Any]:
    """Raw RISK_OFF / RISK_ON / NEUTRAL classification from one daily bar snapshot."""
    price = float(closes.iloc[-1])
    ma200 = float(closes.rolling(200, min_periods=200).mean().iloc[-1]) if len(closes) >= 200 else math.nan
    ma50 = float(closes.rolling(50, min_periods=50).mean().iloc[-1]) if len(closes) >= 50 else math.nan
    mom20 = None
    if len(closes) >= 21:
        mom20 = (price / float(closes.iloc[-21]) - 1.0) * 100.0
    high52 = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
    dd52 = (price / high52 - 1.0) * 100.0 if high52 > 0 else 0.0

    band = float(strategy.regime_ma200_band_pct)
    risk_off_flags: list[str] = []
    if not math.isnan(ma200) and ma200 > 0 and price < ma200 * (1.0 - band / 100.0):
        risk_off_flags.append(f"基准实质跌破MA200缓冲带{band:.0f}%")
    if vix is not None and vix >= DEFAULT_VIX_DANGER_THRESHOLD:
        risk_off_flags.append(f"VIX={vix:.1f}≥{DEFAULT_VIX_DANGER_THRESHOLD:.0f}")
    if dd52 <= -float(strategy.regime_drawdown_risk_off_pct):
        risk_off_flags.append(f"距52周高点回撤{dd52:.1f}%≥{strategy.regime_drawdown_risk_off_pct:.0f}%")
    if hy_oas is not None and hy_oas >= float(strategy.regime_hy_oas_risk_off_pct):
        risk_off_flags.append(f"HY OAS {hy_oas:.2f}%≥{strategy.regime_hy_oas_risk_off_pct:.1f}%")

    risk_on_ok = bool(
        (not math.isnan(ma200)) and price > ma200
        and (not math.isnan(ma50)) and ma50 > ma200
        and dd52 > -float(strategy.regime_drawdown_neutral_pct)
        and vix is not None and vix < float(strategy.regime_vix_risk_on)
        and hy_oas is not None and hy_oas < REGIME_HY_OAS_RISK_ON_PCT
    )
    if risk_off_flags:
        raw = "RISK_OFF"
    elif risk_on_ok:
        raw = "RISK_ON"
    else:
        raw = "NEUTRAL"
    return {
        "raw_regime": raw,
        "risk_off_flags": risk_off_flags,
        "price": _round2(price),
        "ma50": _round2(ma50) if not math.isnan(ma50) else None,
        "ma200": _round2(ma200) if not math.isnan(ma200) else None,
        "mom20_pct": _round2(mom20) if mom20 is not None else None,
        "drawdown_52w_pct": _round2(dd52),
        "vix": _round2(vix) if vix is not None else None,
        "hy_oas": _round2(hy_oas) if hy_oas is not None else None,
    }


def evaluate_market_regime(
    histories: dict[str, pd.DataFrame] | None,
    macro: MacroSnapshot | None,
    strategy: StrategySection,
    prev_gate_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anchored on the USD-native regime benchmark (SPY), never the GBP-denominated VWRP.L.

    Enter RISK_OFF fast (1 day), exit step-wise (3 days of crisis easing; VIX < 25 suffices —
    never requires VIX < 18) and enter RISK_ON only after 3 stable days, so a VIX stuck at
    18~25 in a bull run can never deadlock the system in RISK_OFF.
    """
    degraded: list[str] = []
    bench_sym = str(getattr(strategy, "regime_benchmark_symbol", "SPY") or "SPY").strip().upper()
    hist = (histories or {}).get(bench_sym)
    if hist is None or hist.empty:
        bench_sym = _sym_up(strategy.benchmark_symbol)
        hist = (histories or {}).get(bench_sym)
    closes = _close_series(hist)
    vix = float(macro.vix) if (macro is not None and macro.vix is not None) else None
    hy_oas = float(macro.hy_oas_spread_pct) if (macro is not None and macro.hy_oas_spread_pct is not None) else None
    stale = bool(macro is not None and macro.is_stale)

    if len(closes) < 60:
        return {
            "regime": "UNKNOWN",
            "raw_regime": "UNKNOWN",
            "confirmed": False,
            "reason": f"基准 {bench_sym} 行情不足(仅{len(closes)}根)，市场状态降级 UNKNOWN",
            "benchmark_symbol": bench_sym,
            "degraded_notes": ["基准行情数据不足"],
        }

    metrics = _compute_raw_regime_metrics(closes, vix, hy_oas, strategy)
    raw = metrics["raw_regime"]
    if stale and raw == "RISK_ON":
        raw = "NEUTRAL"
        metrics["raw_regime"] = "NEUTRAL"
        degraded.append("宏观数据过期，严禁盲目判 RISK_ON，保守降级 NEUTRAL")

    prev = prev_gate_state if isinstance(prev_gate_state, dict) else {}
    prev_raw = prev.get("raw_regime")
    prev_regime = prev.get("regime")
    prev_raw_streak = int(prev.get("raw_streak_days") or 0)
    prev_ease_streak = int(prev.get("ease_streak_days") or 0)

    raw_streak = (prev_raw_streak + 1) if (prev_raw == raw and prev_regime is not None) else 1

    price = metrics["price"]
    ma200 = metrics["ma200"]
    mom20 = metrics["mom20_pct"]
    # A 3-day dead-cat bounce while the index is still below MA200 must never release DEFEND.
    # Only when MA200 is genuinely unavailable (insufficient bars) may momentum stand in.
    trend_repaired = bool(
        (ma200 is not None and price >= ma200)
        or (ma200 is None and mom20 is not None and mom20 > 0)
    )
    crisis_eased = bool(
        (vix is None or vix < float(strategy.regime_vix_exit_risk_off))
        and (hy_oas is None or hy_oas < float(strategy.regime_hy_oas_risk_off_pct))
        and trend_repaired
    )
    ease_streak = (
        (prev_ease_streak + 1)
        if (prev_regime == "RISK_OFF" and crisis_eased)
        else (1 if crisis_eased else 0)
    )

    enter_days = int(strategy.regime_confirm_enter_days)
    exit_days = int(strategy.regime_confirm_exit_days)
    regime: str
    if prev_regime is None:
        if raw == "RISK_OFF":
            regime = "RISK_OFF"
        elif raw == "RISK_ON" and raw_streak >= exit_days:
            regime = "RISK_ON"
        else:
            regime = "NEUTRAL"
    elif raw == "RISK_OFF" and raw_streak >= enter_days and prev_regime in {"NEUTRAL", "RISK_ON"}:
        regime = "RISK_OFF"
    elif prev_regime == "RISK_OFF":
        # A -25% bear-market bottom that has repaired above MA200 for the full exit window
        # must not stay locked in RISK_OFF because the stale 52w-high drawdown flag echoes.
        flags = [str(f) for f in metrics.get("risk_off_flags") or []]
        only_stale_drawdown = bool(raw == "RISK_OFF" and flags and all("回撤" in f for f in flags))
        if raw != "RISK_OFF" and ease_streak >= exit_days:
            regime = "NEUTRAL"
        elif only_stale_drawdown and ease_streak >= exit_days:
            regime = "NEUTRAL"
        else:
            regime = "RISK_OFF"
    elif prev_regime == "RISK_ON":
        regime = "RISK_ON" if raw == "RISK_ON" else "NEUTRAL"
    else:  # NEUTRAL
        regime = "RISK_ON" if (raw == "RISK_ON" and raw_streak >= exit_days) else "NEUTRAL"

    reason = "；".join(metrics["risk_off_flags"]) if raw == "RISK_OFF" else (
        "强右侧趋势(MA50>MA200且回撤<8%且VIX<18)" if raw == "RISK_ON" else "常规回调/震荡分化"
    )
    return {
        "regime": regime,
        "raw_regime": raw,
        "confirmed": bool(regime == raw or (prev_regime == "RISK_OFF" and regime == "NEUTRAL")),
        "raw_streak_days": raw_streak,
        "ease_streak_days": ease_streak,
        "crisis_eased": crisis_eased,
        "reason": reason,
        "benchmark_symbol": bench_sym,
        "degraded_notes": degraded,
        **metrics,
    }


# ---------------------------------------------------------------------------
# 4. Triggers and account-level decisions
# ---------------------------------------------------------------------------
def _core_etf_drawdown_step(
    histories: dict[str, pd.DataFrame] | None,
    strategy: StrategySection,
) -> dict[str, Any]:
    """A-trigger: benchmark 52w-high drawdown ladder (first crossing of -5/-10/-15/-20%).

    Considers both the GBP-quoted core benchmark (VWRP.L) and the USD-native regime
    benchmark (SPY) and triggers on the deeper side. In a sterling crisis the GBP
    quote can mask a genuine -15% USD drawdown, leaving the ladder permanently idle.
    """
    primary_sym = _sym_up(strategy.benchmark_symbol)
    out: dict[str, Any] = {
        "symbol": primary_sym,
        "hit": None,
        "drawdown_52w_pct": None,
        "drawdown_52w_pct_by_symbol": {},
        "first_cross": False,
    }
    samples: list[tuple[str, pd.Series, float, float]] = []
    for raw_sym in (
        primary_sym,
        _sym_up(strategy.regime_benchmark_symbol),
    ):
        if not raw_sym:
            continue
        closes = _close_series((histories or {}).get(raw_sym))
        if len(closes) < 2:
            continue
        price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        high52 = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
        if high52 <= 0:
            continue
        dd_now = (price / high52 - 1.0) * 100.0
        dd_prev = (prev_price / high52 - 1.0) * 100.0
        out["drawdown_52w_pct_by_symbol"][raw_sym] = _round2(dd_now)
        samples.append((raw_sym, closes, dd_now, dd_prev))
    if not samples:
        return out
    chosen_sym, _closes, dd_now, dd_prev = min(samples, key=lambda row: row[2])
    out["symbol"] = chosen_sym
    out["drawdown_52w_pct"] = _round2(dd_now)
    for step in sorted((float(s) for s in strategy.core_etf_drawdown_steps_pct), reverse=True):
        if dd_now <= -step and dd_prev > -step:
            out["hit"] = step
            out["first_cross"] = True
            break
    return out


def _scan_a_equity_candidates(
    candidates: list[dict[str, Any]] | None,
    portfolio: PortfolioData,
    strategy: StrategySection,
    histories: dict[str, pd.DataFrame] | None,
    *,
    etf_set: set[str] | None = None,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> list[dict[str, Any]]:
    """A-trigger: right-side uptrend equities pulling back to MA50, cross-account deduped."""
    held_or_pending = {
        _sym_up(p.yahoo_symbol) for p in portfolio.all_positions()
    } | {_sym_up(o.yahoo_symbol) for o in portfolio.all_pending_orders()}
    held_or_pending_roots = {normalize_issuer_root(s) for s in held_or_pending if s}
    known_etfs = etf_set or set()
    hits: list[dict[str, Any]] = []
    tol = float(strategy.trigger_ma_tolerance_pct)
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        sym = _sym_up(str(cand.get("symbol") or ""))
        if not sym or sym in held_or_pending or normalize_issuer_root(sym) in held_or_pending_roots:
            continue
        tech = cand.get("tech")
        bundle = cand.get("bundle") or (bundles or {}).get(sym)
        if _is_etf_symbol(sym, known_etfs, bundles) or (
            bundle is not None and str(getattr(bundle, "quote_type", "") or "").upper() == "ETF"
        ):
            continue
        if not isinstance(tech, TechnicalSnapshot):
            continue
        if bundle is not None:
            days_earn = getattr(bundle, "days_to_earnings", None)
            if days_earn is not None and 0 <= int(days_earn) <= TRIGGER_EARNINGS_EXCLUDE_DAYS:
                continue
        ex_days = None
        if bundle is not None and getattr(bundle, "days_to_ex_dividend", None) is not None:
            ex_days = int(getattr(bundle, "days_to_ex_dividend"))
        if ex_days is None and getattr(tech, "days_to_ex_dividend", None) is not None:
            ex_days = int(getattr(tech, "days_to_ex_dividend"))
        if ex_days is None and bool(getattr(tech, "is_ex_div_window", False)):
            ex_days = 0
        if ex_days is not None and 0 <= ex_days <= TRIGGER_EXDIV_EXCLUDE_DAYS:
            continue
        rsi = getattr(tech, "rsi14", None)
        if rsi is not None and float(rsi) >= TRIGGER_RSI_EXCLUDE_ABOVE:
            continue
        dist_ma50 = getattr(tech, "dist_ma50_pct", None)
        pullback = dist_ma50 is not None and abs(float(dist_ma50)) <= tol
        if not pullback:
            continue
        ma50 = getattr(tech, "ma50", None)
        ma200 = getattr(tech, "ma200", None)
        ma200_rising = None
        hist = (histories or {}).get(cand.get("symbol"))
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = _close_series(hist)
            if len(closes) >= 220:
                ma200_now = float(closes.rolling(200).mean().iloc[-1])
                ma200_prev = float(closes.rolling(200).mean().iloc[-21])
                ma200_rising = ma200_now > ma200_prev
        uptrend = bool(
            (ma50 is not None and ma200 is not None and float(ma50) > float(ma200))
            or (ma200_rising is True and getattr(tech, "trend_regime", "") in {"BULL", "PULLBACK"})
        )
        if ma200_rising is not None:
            uptrend = bool(
                ma200_rising
                and ((ma50 is not None and ma200 is not None and float(ma50) > float(ma200))
                     or getattr(tech, "trend_regime", "") in {"BULL", "PULLBACK"})
            )
        if not uptrend:
            continue
        hits.append({"symbol": sym, "dist_ma50_pct": _round2(dist_ma50), "account": str(cand.get("account") or "Invest")})
        if len(hits) >= int(strategy.trigger_max_symbols):
            break
    return hits


def _today_external_inflow(
    portfolio: PortfolioData,
    strategy: StrategySection,
    now_uk: datetime | None,
) -> dict[str, Any]:
    now = now_uk or datetime.now(LONDON_TZ)
    today_start_ms = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    per_acct: dict[str, float] = {}
    for acct in (portfolio.isa, portfolio.invest):
        net = 0.0
        for tx in acct.recent_transactions:
            if tx.date_time_ms < today_start_ms:
                continue
            if tx.tx_type == "DEPOSIT":
                net += float(tx.amount)
            elif tx.tx_type == "WITHDRAW":
                net -= float(tx.amount)
        per_acct[acct.account_label.upper()] = _round2(net)
    triggered: list[str] = []
    for label, net in per_acct.items():
        acct = portfolio.isa if label == "ISA" else portfolio.invest
        acct_val = _account_value_gbp(acct)
        if net >= float(strategy.inflow_trigger_min_gbp) and acct_val > 0 and net >= acct_val * float(strategy.inflow_trigger_pct) / 100.0:
            triggered.append(label)
    return {"per_account_net_gbp": per_acct, "triggered_accounts": triggered}


def scan_triggers_and_decide(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None,
    strategy: StrategySection,
    ledgers: dict[str, Any],
    regime_info: dict[str, Any],
    histories: dict[str, pd.DataFrame] | None,
    classified: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    gbp_usd: float | None = None,
    now_uk: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate A/B/C/D triggers and produce per-account + global gate decisions."""
    isa = ledgers["isa"]
    invest = ledgers["invest"]
    regime = regime_info.get("regime", "UNKNOWN")

    a_core = _core_etf_drawdown_step(histories, strategy)
    a_equity = _scan_a_equity_candidates(
        candidates,
        portfolio,
        strategy,
        histories,
        etf_set=set(ledgers.get("etf_set") or []),
        bundles=bundles,
    )

    b_dca = bool(
        isa["ok"]
        and isa["deployable_pct"] > float(strategy.isa_cash_band_high_pct) + float(strategy.min_deploy_excess_pct)
        and isa["committed_core_etf_pct"] < float(strategy.isa_core_etf_target_pct)
    )
    inflow = _today_external_inflow(portfolio, strategy, now_uk)
    c_inflow = bool(inflow["triggered_accounts"])

    d_defend = bool(
        regime == "RISK_OFF"
        and (
            invest["pending_equity_buys_all"]
            or isa["pending_new_equity_buys"]
            or invest["slots_used"] > 0
        )
    )
    isa_extreme_low_cash = bool(
        isa["ok"] and isa["total_cash_pct"] < ISA_EXTREME_LOW_CASH_PCT
    )
    # Persistent deep drawdown (>=-10%) during RISK_OFF must keep the core-ETF channel open
    # even after the first-cross day has passed, otherwise a whole sideways bottom is missed.
    persistent_deep_drawdown = bool(
        isa["ok"]
        and a_core.get("drawdown_52w_pct") is not None
        and float(a_core["drawdown_52w_pct"]) <= -10.0
        and isa["deployable_pct"] > float(strategy.isa_cash_band_high_pct)
        and isa["committed_core_etf_pct"] < float(strategy.isa_core_etf_target_pct)
        and len(isa["pending_core_etf_buys"]) == 0
    )

    # ---------- ISA decision ----------
    if not isa["ok"]:
        isa_status, isa_mode = "UNKNOWN", "none"
        isa_reason = "ISA 账户快照降级/不可用，不催买不乱喊减仓"
    else:
        ladder_hit = a_core.get("hit")
        if regime == "RISK_OFF":
            if ladder_hit is not None and float(ladder_hit) >= 10.0:
                isa_status, isa_mode = "RUN", "ladder"
                isa_reason = f"退潮期 ISA 个股通道关闭，宽基深跌{-float(ladder_hit):.0f}%阶梯逆向播种"
            elif persistent_deep_drawdown:
                isa_status, isa_mode = "RUN", "ladder"
                isa_reason = (
                    f"退潮期宽基持续深回撤{float(a_core['drawdown_52w_pct']):.1f}%"
                    "且尚未挂出接针单，允许补挂核心宽基分批买单"
                )
            else:
                isa_status, isa_mode = "WAIT", "none"
                isa_reason = "退潮期冻结新开个股，宽基未达深跌阶梯暂缓"
        elif c_inflow and "ISA" in inflow["triggered_accounts"]:
            isa_status, isa_mode = "RUN", "inflow"
            isa_reason = f"外部净入金 £{inflow['per_account_net_gbp']['ISA']:,.0f} 达标，引导分批配置核心宽基"
        elif b_dca:
            isa_status, isa_mode = "RUN", "dca"
            if isa.get("prefer_core_diversifier"):
                isa_reason = (
                    f"ISA 可用现金 {isa['deployable_pct']:.1f}%>上沿且宽基仅 {isa['committed_core_etf_pct']:.0f}%（含挂单），"
                    "优先用 FTSE100/欧洲宽基分散美股科技，勿只加 VWRP"
                )
            else:
                isa_reason = f"ISA 可用现金 {isa['deployable_pct']:.1f}%>上沿且宽基仅 {isa['committed_core_etf_pct']:.0f}%（含挂单），分批归拢 VWRP"
        elif ladder_hit is not None:
            isa_status, isa_mode = "RUN", "ladder"
            isa_reason = f"核心宽基距52周高点回撤首次触及 -{float(ladder_hit):.0f}% 阶梯"
        else:
            isa_status, isa_mode = "WAIT", "none"
            isa_reason = "现金带与宽基占比均在区间内"
        if isa["isa_satellite_saturated"]:
            isa_reason = f"ISA 个股卫星层饱和(持仓+挂单 {isa['committed_satellite_count']}只/{isa['committed_satellite_pct']:.1f}%)，仅配宽基：" + isa_reason
    isa["status"] = isa_status
    isa["mode"] = isa_mode
    isa["reason"] = isa_reason

    # ---------- Invest decision (never dca on idle cash) ----------
    slots_cap = {
        "RISK_ON": int(strategy.invest_slots_risk_on),
        "NEUTRAL": int(strategy.invest_slots_neutral),
        "RISK_OFF": int(strategy.invest_slots_risk_off),
        "UNKNOWN": int(strategy.invest_slots_neutral),
    }.get(regime, int(strategy.invest_slots_neutral))
    invest["slots_cap"] = slots_cap

    if not invest["ok"]:
        invest_status, invest_mode = "UNKNOWN", "none"
        invest_reason = "Invest 账户快照降级/不可用"
    elif regime == "RISK_OFF":
        invest_status, invest_mode = "DEFEND", "defend"
        invest_reason = "市场退潮：0 槽位冻结新开个股，启动四步渐进防守"
    elif invest["in_loss_cooldown"]:
        invest_status, invest_mode = "WAIT", "cooldown"
        invest_reason = (
            f"gate_epoch_date 后近{strategy.consecutive_loss_lookback_days}天连损"
            f"{invest['tactical_losses_recent']}笔，冷静{strategy.consecutive_loss_cooldown_days}个交易日"
        )
    elif invest["daily_cap_reached"]:
        invest_status, invest_mode = "WAIT", "daily_cap"
        invest_reason = (
            f"今日已新开/新挂 {invest['today_new_equity_entries']} 笔个股"
            f"(上限{strategy.max_new_equity_entries_per_day})，Best Idea Only 单日止步"
        )
    elif invest["slots_used"] >= slots_cap and slots_cap > 0:
        if a_equity:
            invest_status, invest_mode = "ROTATE", "rotate"
            funding_from = ""
            if invest["pending_new_equity_buys"]:
                funding_from = invest["pending_new_equity_buys"][0]
            elif invest.get("pending_add_equity_buys"):
                funding_from = invest["pending_add_equity_buys"][0]
            elif invest["unfinanced_positions"]:
                funding_from = invest["unfinanced_positions"][0]
            invest["funding_from"] = funding_from
            invest_reason = (
                f"在险槽位 {invest['slots_used']}/{slots_cap} 已满但右侧强信号命中，"
                f"须先腾挪 {funding_from} 等风险置换"
            )
        else:
            invest_status, invest_mode = "WAIT", "slots_full"
            invest_reason = (
                f"在险槽位 {invest['slots_used']}/{slots_cap} 已满"
                f"({','.join(invest['unfinanced_positions']) or '无'} 未脱险)，静待推保本解锁"
            )
    elif a_equity:
        invest_status, invest_mode = "RUN", "targeted"
        invest_reason = f"右侧猎物触发({','.join(h['symbol'] for h in a_equity)})，在险槽位 {invest['slots_used']}/{slots_cap}"
    else:
        invest_status, invest_mode = "WAIT", "none"
        invest_reason = "无右侧猎物，现金吃息等待(绝不因现金多催买)"
    invest["status"] = invest_status
    invest["mode"] = invest_mode
    invest["reason"] = invest_reason

    if d_defend or isa_extreme_low_cash:
        invest["defend_note"] = (
            "①优先撤除未成交个股买单 ②保护保本仓与已挂止损逆势仓，只收紧止盈 ③清退无保护浮亏弱股"
        )
        if isa_extreme_low_cash:
            invest["defend_note"] += " ④ISA 现金跌破2%结构垫，停止一切非必要支出/买入"

    triggers = {
        "A": {
            "core_etf_step": a_core,
            "equity_candidates": a_equity,
        },
        "B": {"isa_core_dca": b_dca},
        "C": inflow,
        "D": {
            "risk_off_defend": d_defend,
            "isa_extreme_low_cash": isa_extreme_low_cash,
        },
    }

    # ---------- Global merge: DEFEND > RUN > ROTATE > WAIT > UNKNOWN ----------
    statuses = {isa_status, invest_status}
    if "DEFEND" in statuses:
        global_status = "DEFEND"
    elif "RUN" in statuses:
        global_status = "RUN"
    elif "ROTATE" in statuses:
        global_status = "ROTATE"
    elif "WAIT" in statuses:
        global_status = "WAIT"
    else:
        global_status = "UNKNOWN"

    # ---------- Default pick gate (deterministic baseline, PM may only tighten) ----------
    isa_equity_open = bool(
        isa["ok"] and regime != "RISK_OFF" and not isa["isa_satellite_saturated"]
    )
    # In RISK_OFF the ISA wide-base channel only opens on a valid drawdown ladder step;
    # otherwise the gate WAIT text and the actual permission would contradict each other.
    isa_etf_open = bool(
        isa["ok"]
        and (regime != "RISK_OFF" or a_core.get("hit") is not None or persistent_deep_drawdown)
    )
    isa["equity_open"] = isa_equity_open
    isa["etf_open"] = isa_etf_open
    invest_open = invest_status == "RUN"
    invest_rotate = invest_status == "ROTATE"
    allow = isa_equity_open or isa_etf_open or invest_open or invest_rotate
    if invest_rotate:
        pm_mode, pm_funding = "rotate", invest.get("funding_from", "")
        pm_reason = invest_reason
    elif allow:
        pm_mode, pm_funding = "run", ""
        pm_reason = (
            "Invest 已解锁右侧槽位" if invest_open else
            ("ISA 个股饱和，仅限宽基 ETF" if isa["isa_satellite_saturated"] else "存在可用开仓通道")
        )
    else:
        pm_mode, pm_funding = "none", ""
        pm_reason = "当前无任何可用开仓通道"

    return {
        "isa": isa,
        "invest": invest,
        "regime": regime,
        "triggers": triggers,
        "global_status": global_status,
        "pm": {
            "allow": allow,
            "mode": pm_mode,
            "funding_from": pm_funding,
            "reason": pm_reason,
            "pm_overridden": False,
        },
        "gbp_usd": _round2(gbp_usd) if gbp_usd else None,
        "fx_missing": gbp_usd is None,
    }


# ---------------------------------------------------------------------------
# 5. One-way PM veto clamp
# ---------------------------------------------------------------------------
def reconcile_pm_pick_gate(
    gate_dict: dict[str, Any],
    pm_pick_gate: dict[str, Any] | None,
    whitelist_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """Enforce the one-way PM contract on the final pick gate.

    PM may only tighten (allow=false); Python can never be upgraded from WAIT / DEFEND /
    UNKNOWN into RUN. A ROTATE without a legal funding source degrades to WAIT, and any
    funding symbol outside the whitelist is stripped.
    """
    pm = dict(gate_dict.get("pm") or {})
    whitelist = {_sym_up(s) for s in (whitelist_symbols or set()) if s}
    pm["pm_overridden"] = False

    if not isinstance(pm_pick_gate, dict) or not pm_pick_gate:
        return pm

    wants_allow = bool(pm_pick_gate.get("allow", True))
    if not wants_allow:
        pm["allow"] = False
        pm["mode"] = "none"
        pm["funding_from"] = ""
        pm["reason"] = str(pm_pick_gate.get("reason") or "PM 主动收紧否决")
        return pm

    # PM wants allow=True — deterministic bottom line decides.
    base_mode = pm.get("mode", "none")
    if base_mode == "none":
        pm["allow"] = False
        pm["mode"] = "none"
        pm["funding_from"] = ""
        pm["reason"] = (
            f"组合门控底线 {gate_dict.get('global_status', 'WAIT')} 态，PM 无权升级为 RUN"
        )
        pm["pm_overridden"] = True
        return pm
    if base_mode == "rotate":
        funding = str(pm_pick_gate.get("funding_from") or pm.get("funding_from") or "").strip()
        funding_up = _sym_up(funding)
        if not funding_up or (whitelist and funding_up not in whitelist):
            pm["allow"] = False
            pm["mode"] = "none"
            pm["funding_from"] = ""
            pm["reason"] = "ROTATE 缺合法置换源(未平弱仓/未撤旧买单)，自动降级 WAIT"
            pm["pm_overridden"] = True
            return pm
        pm["allow"] = True
        pm["mode"] = "rotate"
        pm["funding_from"] = funding_up
        pm["reason"] = str(pm_pick_gate.get("reason") or pm.get("reason") or "等风险置换")
        return pm
    # base_mode == "run"
    pm["allow"] = True
    pm["mode"] = "run"
    pm["funding_from"] = ""
    pm["reason"] = str(pm_pick_gate.get("reason") or pm.get("reason") or "允许运行 /pick")
    return pm


# ---------------------------------------------------------------------------
# 6. <=110-char pure-text banner
# ---------------------------------------------------------------------------
def _fit_banner(line1: str, line2: str) -> str:
    banner = f"{line1}\n{line2}"
    if len(banner) <= _BANNER_MAX_CHARS:
        return banner
    budget = max(20, _BANNER_MAX_CHARS - len(line1) - 1)
    tail = line2
    while len(tail) > budget and ("，" in tail or ";" in tail or "；" in tail):
        cut = max(tail.rfind("，"), tail.rfind(";"), tail.rfind("；"))
        if cut <= 0:
            break
        tail = tail[:cut]
    if len(tail) > budget:
        tail = tail[: max(0, budget - 1)] + "…"
    return f"{line1}\n{tail}"


def _gpb_k(value: float) -> str:
    return f"£{value / 1000.0:.1f}k"


def format_deployment_gate_banner(gate_dict: dict[str, Any]) -> str:
    """Compose the <=110-char, <=2-line, pure-plain-text gate banner (no #, no **)."""
    isa = gate_dict.get("isa") or {}
    invest = gate_dict.get("invest") or {}
    regime = str(gate_dict.get("regime") or "UNKNOWN")
    global_status = str(gate_dict.get("global_status") or "UNKNOWN")
    fx_note = "汇率待校准" if gate_dict.get("fx_missing") else ""
    invest_status = str(invest.get("status") or "UNKNOWN")
    isa_status = str(isa.get("status") or "UNKNOWN")

    if invest_status == "DEFEND":
        heat = float(invest.get("heat_gbp_total") or 0.0)
        fin = "、".join(invest.get("financed_positions") or []) or "无"
        line1 = f"组合门控：DEFEND·渐进收缩｜市场转入退潮态｜Invest本金在险 £{heat:.0f}（{fin}保本保护中）"
        line2 = "→ ①撤未成交个股买单 ②清退无保护浮亏弱股，保留保本/逆势强股 ③ISA宽基不卖，深跌分档定投"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if invest_status == "RUN":
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        heat = float(invest.get("heat_gbp_total") or 0.0)
        fin = "、".join(invest.get("financed_positions") or []) or "无"
        avail = float(invest.get("available_to_trade_gbp") or 0.0)
        line1 = f"组合门控：RUN·右侧猎物｜Invest在险槽位 {used}/{cap}（{fin}已锁保本，本金在险£{heat:.0f}）｜可用{_gpb_k(avail)}"
        line2 = "→ 已解锁右侧开仓槽位（单日限最优选1笔，风险预算£85），可运行 /pick 扫描突破龙头"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if invest_status == "ROTATE":
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        funding = str(invest.get("funding_from") or "")
        line1 = f"组合门控：ROTATE·等风险置换｜{regime}在险槽位 {used}/{cap} 已满｜观察池触发右侧强信号"
        line2 = f"→ 严禁动用闲置现金净增风险；若运行 /pick，须以平掉未脱险弱仓 {funding} 为前提等额置换"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if global_status == "UNKNOWN" or (isa_status == "UNKNOWN" and invest_status == "UNKNOWN"):
        return _fit_banner("组合门控：UNKNOWN｜账户快照或基准数据降级，保守不催买不乱喊减仓", "")

    if isa.get("isa_satellite_saturated"):
        d_pct = float(isa.get("deployable_pct") or 0.0)
        core_pct = float(isa.get("committed_core_etf_pct") or 0.0)
        count = int(isa.get("committed_satellite_count") or 0)
        sat_pct = float(isa.get("committed_satellite_pct") or 0.0)
        pending_eq = "、".join((isa.get("pending_new_equity_buys") or [])[:2]) or ""
        line1 = (
            f"组合门控：ISA 仅配宽基 / Invest {invest_status}｜ISA可用 {d_pct:.1f}%"
            f"（宽基{core_pct:.0f}%，个股含挂单{count}只/{sat_pct:.0f}%饱和）"
        )
        used = int(invest.get("slots_used") or 0)
        cap = int(invest.get("slots_cap") or 0)
        unfin = "、".join((invest.get("unfinanced_positions") or [])[:2]) or "无"
        fin = "、".join((invest.get("financed_positions") or [])[:2]) or "无"
        parts = []
        if pending_eq:
            parts.append(f"审撤{pending_eq}买单")
        avail = float(isa.get("available_to_trade_gbp") or 0.0)
        core_label = "ISF/VERG等低科技宽基" if isa.get("prefer_core_diversifier") else "VWRP"
        if avail >= 1000:
            parts.append(f"{_gpb_k(avail)}归拢{core_label}")
        parts.append(f"Invest槽位{used}/{cap}({unfin}未脱险,{fin}已锁保本)静待解锁")
        line2 = "→ " + "，".join(parts)
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    if isa_status == "RUN":
        d_pct = float(isa.get("deployable_pct") or 0.0)
        core_pct = float(isa.get("committed_core_etf_pct") or 0.0)
        mode = str(isa.get("mode") or "dca")
        line1 = f"组合门控：ISA RUN·{mode} / Invest {invest_status}｜ISA可用 {d_pct:.1f}%（宽基{core_pct:.0f}%）"
        core_label = "ISF/VERG等低科技宽基" if isa.get("prefer_core_diversifier") else "VWRP"
        line2 = f"→ 闲置现金分批归拢 {core_label}；Invest 无右侧猎物不催买（现金吃息仅作等待保底）"
        return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))

    line1 = f"组合门控：ISA {isa_status} / Invest {invest_status}｜市场 {regime}"
    line2 = f"→ {str(isa.get('reason') or '')}"
    return _fit_banner(line1, line2 + (f"({fx_note})" if fx_note else ""))


# ---------------------------------------------------------------------------
# Orchestrator used by /report (and /pick via context pack)
# ---------------------------------------------------------------------------
def evaluate_deployment_gate(
    *,
    portfolio: PortfolioData,
    strategy: StrategySection,
    bundles: dict[str, TickerMarketBundle] | None = None,
    histories: dict[str, pd.DataFrame] | None = None,
    macro: MacroSnapshot | None = None,
    tech_map: dict[str, TechnicalSnapshot] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    universe_etf_symbols: list[str] | None = None,
    prev_gate_states: list[dict[str, Any]] | None = None,
    gbp_usd: float | None = None,
    now_uk: datetime | None = None,
) -> dict[str, Any]:
    """Full deterministic gate evaluation — 0 LLM calls, 0 network, 0 DB writes."""
    bundles = bundles or {}
    histories = histories or {}
    etf_set = _build_etf_set(strategy, universe_etf_symbols, bundles)
    fx = gbp_usd or getattr(portfolio, "gbp_usd_rate", None) or (macro.gbp_usd if macro is not None else None)

    eval_dt = now_uk or datetime.now(LONDON_TZ)
    today_date_str = eval_dt.strftime("%Y-%m-%d")

    prev = None
    for state in prev_gate_states or []:
        if not isinstance(state, dict) or not state.get("regime"):
            continue
        state_date = str(state.get("cycle_date") or str(state.get("evaluated_at_uk") or "")[:10]).strip()
        if state_date and state_date == today_date_str:
            continue  # same-day /report re-run must not inflate multi-day regime streaks
        prev = state
        break

    regime_info = evaluate_market_regime(histories, macro, strategy, prev)
    classified = classify_pending_orders(portfolio, strategy, etf_set, bundles=bundles)
    ledgers = compute_account_ledgers(
        portfolio, classified, bundles, strategy, etf_set, tech_map=tech_map, now_uk=eval_dt
    )
    decision = scan_triggers_and_decide(
        portfolio,
        bundles,
        strategy,
        ledgers,
        regime_info,
        histories,
        classified,
        candidates=candidates,
        gbp_usd=fx,
        now_uk=eval_dt,
    )

    gate = {
        "version": _GATE_VERSION,
        "cycle_date": today_date_str,
        "evaluated_at_uk": eval_dt.strftime("%Y-%m-%d %H:%M"),
        "regime": regime_info["regime"],
        "raw_regime": regime_info["raw_regime"],
        "raw_streak_days": int(regime_info.get("raw_streak_days") or 0),
        "ease_streak_days": int(regime_info.get("ease_streak_days") or 0),
        "regime_reason": regime_info["reason"],
        "regime_benchmark_symbol": regime_info["benchmark_symbol"],
        "regime_degraded_notes": regime_info["degraded_notes"],
        "etf_set": ledgers["etf_set"],
        "isa": decision["isa"],
        "invest": decision["invest"],
        "triggers": decision["triggers"],
        "global_status": decision["global_status"],
        "pm": decision["pm"],
        "gbp_usd": decision["gbp_usd"],
        "fx_missing": bool(decision["fx_missing"]),
    }
    gate["banner"] = format_deployment_gate_banner(gate)
    return gate


# ---------------------------------------------------------------------------
# 7. 2-year offline historical replay (anti-deadlock validation)
# ---------------------------------------------------------------------------
def replay_deployment_gate_history(
    bench_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    strategy: StrategySection | None = None,
    hy_oas_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Replay the regime state machine over a daily frame and report distribution stats.

    Detects the pathological deadlock where the system stays stuck in RISK_OFF for more
    than 60 straight days while the index has already made a fresh 60-day high.
    """
    strat = strategy or StrategySection()
    closes = _close_series(bench_df)
    vix_closes = _close_series(vix_df) if vix_df is not None else pd.Series(dtype=float)
    if hy_oas_series is None:
        hy_oas_series = pd.Series(dtype=float)
    vix_map = {str(idx.date()): float(v) for idx, v in vix_closes.items()} if len(vix_closes) else {}
    oas_map = {}
    if len(hy_oas_series):
        try:
            oas_map = {str(idx.date()): float(v) for idx, v in hy_oas_series.items()}
        except (TypeError, ValueError):
            oas_map = {}

    start = 200 if len(closes) > 200 else 0
    prev: dict[str, Any] | None = None
    state_counts = {"RISK_OFF": 0, "NEUTRAL": 0, "RISK_ON": 0, "UNKNOWN": 0}
    transitions = 0
    last_regime = None
    risk_off_run = 0
    max_risk_off_run = 0
    deadlock_detected = False
    deadlock_details: list[dict[str, Any]] = []
    recovered_days_in_run = 0

    for i in range(start, len(closes)):
        window = closes.iloc[: i + 1]
        date_key = str(window.index[-1].date())
        vix = vix_map.get(date_key)
        hy_oas = oas_map.get(date_key)
        metrics = _compute_raw_regime_metrics(window, vix, hy_oas, strat)
        fake_macro = MacroSnapshot(fetched_at_utc=date_key, vix=vix, hy_oas_spread_pct=hy_oas)
        info = evaluate_market_regime(
            {str(strat.regime_benchmark_symbol).upper(): window}, fake_macro, strat, prev_gate_state=prev
        )
        regime = info["regime"]
        state_counts[regime] = state_counts.get(regime, 0) + 1
        if last_regime is not None and regime != last_regime:
            transitions += 1
        last_regime = regime

        if regime == "RISK_OFF":
            risk_off_run += 1
            # Pathological deadlock signature: benchmark fully stabilized back above MA200
            # (i.e. the crisis is over) yet the regime stays locked for far longer than the
            # exit confirmation window (e.g. the v2 "VIX<18 or bust" bug).
            if metrics["ma200"] is not None and metrics["price"] >= metrics["ma200"]:
                recovered_days_in_run += 1
            else:
                recovered_days_in_run = 0
        else:
            if risk_off_run > max_risk_off_run:
                max_risk_off_run = risk_off_run
            if risk_off_run > 60 and recovered_days_in_run > int(strat.regime_confirm_exit_days):
                deadlock_detected = True
                deadlock_details.append(
                    {
                        "run_days": risk_off_run,
                        "recovered_above_ma200_days": recovered_days_in_run,
                        "ended_at": date_key,
                    }
                )
            risk_off_run = 0
            recovered_days_in_run = 0
        prev = {
            "regime": regime,
            "raw_regime": info["raw_regime"],
            "raw_streak_days": info["raw_streak_days"],
            "ease_streak_days": info["ease_streak_days"],
        }

    if risk_off_run > max_risk_off_run:
        max_risk_off_run = risk_off_run
    if risk_off_run > 60 and recovered_days_in_run > int(strat.regime_confirm_exit_days):
        deadlock_detected = True
        deadlock_details.append(
            {
                "run_days": risk_off_run,
                "recovered_above_ma200_days": recovered_days_in_run,
                "ended_at": str(closes.index[-1].date()),
            }
        )

    total = sum(state_counts.values())
    return {
        "days_evaluated": total,
        "state_days": state_counts,
        "state_pct": {k: _round2((v / total) * 100.0) if total else 0.0 for k, v in state_counts.items()},
        "transitions": transitions,
        "max_risk_off_run_days": max_risk_off_run,
        "deadlock_detected": deadlock_detected,
        "deadlock_details": deadlock_details,
    }
