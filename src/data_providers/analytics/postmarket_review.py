"""Deterministic post-close execution ledger and 2x2 process classification (0-Token).

This module is deliberately pure Python/pandas: it never calls an LLM or any network
source.  It splits the day's mark-to-market move into local-price P&L vs FX P&L,
scans plan-vs-actual discipline, and produces a hard-gated shortlist of anomaly
candidates for the optional Flash Lite catalyst lookup.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.data_providers.analytics.correlation import resolve_market_benchmark
from src.data_providers.analytics.technical import (
    _compute_atr_and_adx,
    _normalize_adjusted_ohlcv,
    _resolve_col,
)

_NO_NEWS_PHRASE = "无明确基本面公告"
_CORE_BETA_FALLBACK = {
    "VUSA.L",
    "VUAG.L",
    "VWRP.L",
    "IUSA.L",
    "CSP1.L",
    "VUKE.L",
    "IMEU.L",
    "ISF.L",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _plain(text: Any) -> str:
    """Strip Markdown tokens forbidden by the Telegram contract."""
    return str(text or "").replace("#", "").replace("**", "").strip()


def _clamp_text(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _clamp_lines(lines: list[str], max_chars: int) -> str:
    limit = max(1, int(max_chars))
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if len(candidate) <= limit:
            kept.append(line)
            continue
        remaining = limit - len("\n".join(kept))
        if remaining > 8 and kept:
            kept.append(_clamp_text(line, remaining - 1))
        break
    return _plain("\n".join(kept))


def is_substantive_catalyst(text: Any) -> bool:
    """True when Flash Lite returned a real announcement rather than the mandated no-news phrase."""
    clean = _plain(text)
    if len(clean) < 4 or clean.startswith("⚠️"):
        return False
    return _NO_NEWS_PHRASE not in clean


def classify_process_quadrant(
    *,
    action: str,
    raw_return_pct: float | None,
    alpha_pct: float | None,
    fill_status: str,
    actual_filled: int | None,
    execution_slippage_pct: float | None,
) -> str:
    """Classify a settled recommendation into GOOD_WIN / GOOD_LOSS / BAD_WIN / BAD_LOSS.

    Process quality is driven only by actual execution degradation (>+1.5% adverse
    slippage on a filled order).  A missing fill record is treated as a theoretical
    signal (GOOD).  Result quality is direction-aware through ``raw_return_pct``,
    which settlement already signs by trade direction.  Missing returns stay
    unclassified instead of being forced into a quadrant.
    """
    if raw_return_pct is None:
        return ""
    try:
        raw_val = float(raw_return_pct)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(raw_val):
        return ""
    process_bad = False
    if actual_filled is not None and int(actual_filled) == 1 and execution_slippage_pct is not None:
        try:
            process_bad = float(execution_slippage_pct) > 1.5
        except (TypeError, ValueError):
            process_bad = False
    result = "WIN" if raw_val >= 0.0 else "LOSS"
    return f"{'BAD' if process_bad else 'GOOD'}_{result}"


def _normalize_account(label: Any) -> str:
    clean = str(label or "").strip().upper()
    return "ISA" if clean == "ISA" else ("Invest" if clean else "")


def _core_beta(symbol: str, strategy: Any) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    try:
        from src.engine.portfolio_rules.models_and_friction import is_invest_core_beta_symbol

        if strategy is not None:
            return bool(is_invest_core_beta_symbol(sym, strategy))
    except Exception:
        pass
    return sym in _CORE_BETA_FALLBACK


def _normalize_currency_code(raw: Any) -> str:
    """Normalise currency codes while preserving case-sensitive GBp (pence) vs GBP (pounds)."""
    text = str(raw or "").strip()
    if text == "GBp":
        return "GBp"
    return text.upper()


def _breakeven_friction_pct(pos: Any, strategy: Any, invest_acc: Any) -> float:
    """Match ledgers.py / models_and_friction.py round-trip breakeven friction 1-to-1."""
    if strategy is not None:
        try:
            from src.engine.portfolio_rules.models_and_friction import _round_trip_friction_pct

            return max(0.0, float(_round_trip_friction_pct(pos, None, strategy, invest_acc)))
        except Exception:
            pass
    # Conservative standalone fallback when no StrategySection is supplied.
    symbol = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
    ccy = _normalize_currency_code(getattr(pos, "inst_currency", ""))
    if symbol.endswith(".L") or ccy in ("GBp", "GBP"):
        return 0.70
    return 0.35


def _align_price(price: float, ref: float) -> float:
    """Repair a 100x GBp/GBP slip between an order/stop price and its reference quote."""
    if price <= 0 or ref <= 0:
        return price
    ratio = price / ref
    if ratio >= 50.0:
        return price / 100.0
    if ratio <= 0.02:
        return price * 100.0
    return price


def _looks_like_100x_move(symbol: str, ratio: float) -> bool:
    return (
        str(symbol or "").upper().endswith(".L")
        and math.isfinite(ratio)
        and ((0.005 <= ratio <= 0.05) or (20.0 <= ratio <= 200.0))
    )


def _position_return_pct(pos: Any) -> float | None:
    avg = _safe_float(getattr(pos, "avg_price", 0.0))
    cur = _safe_float(getattr(pos, "cur_price", 0.0))
    if avg <= 0 or cur <= 0:
        return None
    return ((cur / avg) - 1.0) * 100.0


def _daily_move_pct(hist: pd.DataFrame | None) -> float | None:
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        norm = _normalize_adjusted_ohlcv(hist)
        close_col = _resolve_col(norm, "Close")
        if not close_col:
            return None
        closes = pd.to_numeric(norm[close_col], errors="coerce").dropna().astype(float)
        closes = closes[closes > 0]
        if len(closes) < 2:
            return None
        prev = float(closes.iloc[-2])
        last = float(closes.iloc[-1])
        if prev <= 0 or not math.isfinite(prev) or not math.isfinite(last):
            return None
        return ((last / prev) - 1.0) * 100.0
    except Exception:
        return None


def _atr_pct(hist: pd.DataFrame | None) -> float | None:
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        norm = _normalize_adjusted_ohlcv(hist)
        close_col = _resolve_col(norm, "Close")
        high_col = _resolve_col(norm, "High")
        low_col = _resolve_col(norm, "Low")
        if not close_col:
            return None
        frame = norm[[c for c in (high_col, low_col, close_col) if c]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(frame) < 20:
            return None
        frame = frame[frame[close_col] > 0]
        if len(frame) < 20:
            return None
        highs = frame[high_col] if high_col else frame[close_col]
        lows = frame[low_col] if low_col else frame[close_col]
        closes = frame[close_col]
        atr14, _ = _compute_atr_and_adx(highs, lows, closes)
        last = float(closes.iloc[-1])
        if atr14 is None or last <= 0:
            return None
        return (float(atr14) / last) * 100.0
    except Exception:
        return None


def _benchmark_daily_return(
    symbol: str,
    histories: dict[str, pd.DataFrame] | None,
    strategy: Any,
) -> float | None:
    if not histories:
        return None
    candidates = list(resolve_market_benchmark(symbol))
    if strategy is not None:
        for attr in ("benchmark_symbol", "regime_benchmark_symbol"):
            raw = str(getattr(strategy, attr, "") or "").strip().upper()
            if raw:
                candidates.append(raw)
    for bench in dict.fromkeys(candidates):
        if bench in histories:
            move = _daily_move_pct(histories.get(bench))
            if move is not None:
                return move
    return None


def _order_date(order: Any) -> str:
    raw = str(getattr(order, "uk_time_str", "") or "").strip()
    if len(raw) >= 10:
        return raw[:10]
    raw = str(getattr(order, "filled_at", "") or "").strip()
    return raw[:10] if len(raw) >= 10 else ""


def _order_dict(order: Any, fallback_account: str = "") -> dict[str, Any]:
    return {
        "account": _normalize_account(getattr(order, "account", "") or fallback_account),
        "symbol": str(getattr(order, "yahoo_symbol", "") or "").strip().upper(),
        "side": str(getattr(order, "side", "") or "").strip().upper(),
        "order_type": str(getattr(order, "order_type", "") or "").strip().upper(),
        "quantity": _safe_float(getattr(order, "quantity", 0.0)),
        "price": _safe_float(getattr(order, "price", 0.0)),
        "net_value": _safe_float(getattr(order, "net_value", 0.0)),
        "currency": str(getattr(order, "inst_currency", "") or "").strip(),
        "order_id": str(getattr(order, "id", "") or getattr(order, "order_id", "") or ""),
        "filled_at": str(getattr(order, "uk_time_str", "") or getattr(order, "filled_at", "") or ""),
    }


def _today_orders(eod_snap: Any, uk_date: str) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for label, acc in (("ISA", getattr(eod_snap, "isa", None)), ("Invest", getattr(eod_snap, "invest", None))):
        if acc is None:
            continue
        for order in getattr(acc, "recent_orders", None) or []:
            if uk_date and _order_date(order) != uk_date:
                continue
            item = _order_dict(order, fallback_account=label)
            if item["symbol"]:
                orders.append(item)
    return orders


def _add_alert(
    alerts: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    code: str,
    severity: str,
    account: str,
    symbol: str,
    message: str,
) -> None:
    key = (code, account, symbol)
    if key in seen:
        return
    seen.add(key)
    alerts.append(
        {
            "code": code,
            "severity": severity,
            "account": account,
            "symbol": symbol,
            "message": _plain(message),
        }
    )


def _scan_discipline(
    *,
    eod_snap: Any,
    orders: list[dict[str, Any]],
    today_recs: list[dict[str, Any]] | None,
    histories: dict[str, pd.DataFrame] | None,
    strategy: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    recs = list(today_recs or [])

    recs_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in recs:
        key = (
            _normalize_account(rec.get("account")),
            str(rec.get("symbol") or "").strip().upper(),
        )
        if key[1]:
            recs_by_key.setdefault(key, []).append(rec)

    matched_orders = 0
    slips: list[float] = []
    high_slippage = 0
    unplanned = 0

    for order in orders:
        if order["side"] != "BUY":
            continue
        key = (order["account"], order["symbol"])
        candidates = recs_by_key.get(key, [])
        plan_recs = [r for r in candidates if str(r.get("action") or "").upper() in {"BUY", "ADD"}]
        if not plan_recs:
            if not _core_beta(order["symbol"], strategy):
                unplanned += 1
                _add_alert(
                    alerts,
                    seen,
                    code="UNPLANNED_TRADE",
                    severity="WARNING",
                    account=order["account"],
                    symbol=order["symbol"],
                    message=f"计划外建仓：{order['account']} {order['symbol']} 当日买入无对应近两日建议",
                )
            continue
        rec = plan_recs[0]
        entry = _safe_float(rec.get("entry_price"))
        fill = _safe_float(order.get("price"))
        if entry <= 0 or fill <= 0:
            continue
        aligned_fill = _align_price(fill, entry)
        adverse = ((aligned_fill / entry) - 1.0) * 100.0
        matched_orders += 1
        slips.append(adverse)
        if adverse > 1.0:
            high_slippage += 1
            _add_alert(
                alerts,
                seen,
                code="HIGH_SLIPPAGE",
                severity="WARNING",
                account=order["account"],
                symbol=order["symbol"],
                message=f"追高滑点：{order['symbol']} 建议 {entry:.4g} 实际 {aligned_fill:.4g} 劣化 {adverse:+.2f}%",
            )

    invest_acc = getattr(eod_snap, "invest", None)
    if invest_acc is not None:
        pending = list(getattr(invest_acc, "pending_orders", None) or [])
        breakeven_pct = _safe_float(
            getattr(strategy, "invest_ladder_breakeven_advisory_pct", 15.0), 15.0
        )
        default_stop_pct = _safe_float(getattr(strategy, "invest_stop_loss_alert_pct", -12.0), -12.0)
        for pos in getattr(invest_acc, "positions", None) or []:
            symbol = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            if not symbol or _core_beta(symbol, strategy):
                continue
            qty = _safe_float(getattr(pos, "quantity", 0.0))
            if qty <= 0:
                continue
            avg = _safe_float(getattr(pos, "avg_price", 0.0))
            ret = _position_return_pct(pos)
            stops = [
                o
                for o in pending
                if str(getattr(o, "yahoo_symbol", "") or "").strip().upper() == symbol
                and str(getattr(o, "side", "") or "").strip().upper() == "SELL"
                and "STOP" in str(getattr(o, "order_type", "") or "").upper()
                and _safe_float(getattr(o, "stop_price", 0.0) or getattr(o, "limit_price", 0.0)) > 0
            ]

            if ret is not None and ret >= breakeven_pct:
                friction_pct = _breakeven_friction_pct(pos, strategy, invest_acc)
                breakeven_floor = avg * (1.0 + friction_pct / 100.0)
                covered = 0.0
                weighted_stop_num = 0.0
                priced_qty = 0.0
                for stop in stops:
                    raw_stop = _safe_float(
                        getattr(stop, "stop_price", 0.0) or getattr(stop, "limit_price", 0.0)
                    )
                    if raw_stop <= 0:
                        continue
                    aligned_stop = _align_price(raw_stop, avg)
                    remaining = max(
                        0.0,
                        _safe_float(getattr(stop, "quantity", 0.0))
                        - _safe_float(getattr(stop, "filled_quantity", 0.0)),
                    )
                    if remaining <= 0:
                        continue
                    covered += remaining
                    weighted_stop_num += aligned_stop * remaining
                    priced_qty += remaining
                weighted_stop = (weighted_stop_num / priced_qty) if priced_qty > 0 else 0.0
                has_breakeven = (
                    covered + 1e-9 >= qty * 0.99
                    and avg > 0
                    and weighted_stop + 1e-9 >= breakeven_floor
                )
                if not has_breakeven:
                    _add_alert(
                        alerts,
                        seen,
                        code="MISSING_BREAKEVEN_STOP",
                        severity="WARNING",
                        account="Invest",
                        symbol=symbol,
                        message=(
                            f"漏推保本：{symbol} 浮盈 {ret:+.1f}% 但未覆盖保本位"
                            f"（成本×(1+{friction_pct:.2f}%)= {breakeven_floor:.4g}，止损覆盖 {min(covered, qty):.2f}/{qty:.2f} 股）"
                        ),
                    )

            if ret is not None:
                atr_pct = _atr_pct((histories or {}).get(symbol))
                stop_pct = (
                    round(max(-18.0, min(-8.0, -2.8 * atr_pct)), 2)
                    if atr_pct is not None and atr_pct > 0
                    else default_stop_pct
                )
                if ret <= stop_pct and not stops:
                    _add_alert(
                        alerts,
                        seen,
                        code="UNPROTECTED_LOSS",
                        severity="HIGH",
                        account="Invest",
                        symbol=symbol,
                        message=(
                            f"裸露亏损仓：{symbol} 浮亏 {ret:+.1f}% 已破止损线 "
                            f"{stop_pct:.1f}% 且无 SELL STOP 挂单"
                        ),
                    )

    avg_slip = round(sum(slips) / len(slips), 3) if slips else None
    return alerts, {
        "checked_orders": len(orders),
        "matched_orders": matched_orders,
        "high_slippage": high_slippage,
        "unplanned": unplanned,
        "avg_adverse_slippage_pct": avg_slip,
    }


def _scan_anomalies(
    *,
    eod_snap: Any,
    position_rows: dict[tuple[str, str], dict[str, Any]],
    orders: list[dict[str, Any]],
    histories: dict[str, pd.DataFrame] | None,
    strategy: Any,
) -> list[dict[str, Any]]:
    min_move = _safe_float(getattr(strategy, "postmarket_anomaly_min_move_pct", 3.5), 3.5)
    buy_keys = {
        (order["account"], order["symbol"])
        for order in orders
        if order["side"] == "BUY"
    }
    by_symbol: dict[str, dict[str, Any]] = {}
    for account_label, acc in (("ISA", getattr(eod_snap, "isa", None)), ("Invest", getattr(eod_snap, "invest", None))):
        if acc is None:
            continue
        for pos in getattr(acc, "positions", None) or []:
            symbol = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            if not symbol or _core_beta(symbol, strategy):
                continue
            qty = _safe_float(getattr(pos, "quantity", 0.0))
            if qty <= 0:
                continue
            key = (account_label, symbol)
            row = position_rows.get(key) or {}
            hist = (histories or {}).get(symbol)
            intraday = bool(row.get("intraday_traded"))
            move = _daily_move_pct(hist) if intraday else row.get("local_ret_pct")
            if move is None:
                move = _daily_move_pct(hist)
            if move is None:
                continue
            atr_pct = _atr_pct(hist)
            threshold = 4.5 if atr_pct is None else max(min_move, 2.0 * atr_pct)
            bench_move = _benchmark_daily_return(symbol, histories, strategy)
            excess = (move - bench_move) if bench_move is not None else None
            abnormal = (
                abs(move) >= threshold
                and excess is not None
                and abs(excess) >= 2.5
            )
            drawdown_after_buy = bool(key in buy_keys) and move <= -3.0
            if not (abnormal or drawdown_after_buy):
                continue
            reason = "INTRADAY_BUY_DRAWDOWN" if drawdown_after_buy else "ABNORMAL_MOVE"
            market_cap = _safe_float(getattr(pos, "current_value", 0.0))
            candidate = {
                "symbol": symbol,
                "account": account_label,
                "local_ret_pct": round(float(move), 2),
                "atr_pct": round(float(atr_pct), 2) if atr_pct is not None else None,
                "threshold_pct": round(float(threshold), 2),
                "benchmark_move_pct": round(float(bench_move), 2) if bench_move is not None else None,
                "excess_pct": round(float(excess), 2) if excess is not None else None,
                "position_value_gbp": round(market_cap, 2),
                "intraday_traded": intraday,
                "reason": reason,
                "event_reason_hint": (
                    f"本地日涨跌 {move:+.1f}%"
                    + (f"，相对基准超额 {excess:+.1f}%" if excess is not None else "")
                    + (f"，持仓市值 £{market_cap:,.0f}" if market_cap > 0 else "")
                    + ("，当日有实盘买入且收盘回撤" if drawdown_after_buy else "")
                ),
            }
            previous = by_symbol.get(symbol)
            if previous is None or abs(candidate["local_ret_pct"]) > abs(previous["local_ret_pct"]):
                by_symbol[symbol] = candidate
    return sorted(by_symbol.values(), key=lambda x: abs(x["local_ret_pct"]), reverse=True)


def build_eod_attribution_ledger(
    morning_snap: Any,
    eod_snap: Any,
    *,
    regime: str = "",
    today_recs: list[dict[str, Any]] | None = None,
    histories: dict[str, pd.DataFrame] | None = None,
    strategy: Any = None,
    uk_date: str = "",
) -> dict[str, Any]:
    """Build the 0-Token post-close ledger from the morning baseline and EOD snapshots."""
    eod_date = uk_date or str(getattr(eod_snap, "fetched_at_uk", "") or "")[:10]
    ledger: dict[str, Any] = {
        "uk_date": eod_date,
        "regime": str(regime or "").strip().upper(),
        "accounts": {
            "ISA": {"local_pnl_gbp": 0.0, "fx_pnl_gbp": 0.0, "total_mark_pnl_gbp": 0.0},
            "Invest": {"local_pnl_gbp": 0.0, "fx_pnl_gbp": 0.0, "total_mark_pnl_gbp": 0.0},
        },
        "totals": {"local_pnl_gbp": 0.0, "fx_pnl_gbp": 0.0, "total_mark_pnl_gbp": 0.0},
        "positions": [],
        "top_winners": [],
        "top_losers": [],
        "intraday_orders": [],
        "discipline_alerts": [],
        "discipline_alert_count": 0,
        "plan_vs_actual": {},
        "anomaly_candidates": [],
    }
    if morning_snap is None or eod_snap is None:
        return ledger

    def _index(snap: Any) -> dict[tuple[str, str], Any]:
        out: dict[tuple[str, str], Any] = {}
        for label, acc in (("ISA", getattr(snap, "isa", None)), ("Invest", getattr(snap, "invest", None))):
            if acc is None:
                continue
            for pos in getattr(acc, "positions", None) or []:
                sym = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
                if not sym:
                    continue
                account = _normalize_account(getattr(pos, "account", "") or label)
                out[(account, sym)] = pos
        return out

    morning_pos = _index(morning_snap)
    eod_pos = _index(eod_snap)
    orders = _today_orders(eod_snap, eod_date)
    orders_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for order in orders:
        orders_by_key.setdefault((order["account"], order["symbol"]), []).append(order)

    position_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(set(morning_pos) | set(eod_pos)):
        account, symbol = key
        am = morning_pos.get(key)
        eod = eod_pos.get(key)
        qty_am = _safe_float(getattr(am, "quantity", 0.0)) if am is not None else 0.0
        qty_eod = _safe_float(getattr(eod, "quantity", 0.0)) if eod is not None else 0.0
        currency_am = _normalize_currency_code(getattr(am, "inst_currency", "")) if am is not None else ""
        currency_eod = _normalize_currency_code(getattr(eod, "inst_currency", "")) if eod is not None else ""
        name = str(getattr(eod, "name", "") or getattr(am, "name", "") or symbol)
        row: dict[str, Any] = {
            "account": account,
            "symbol": symbol,
            "name": name,
            "currency": currency_eod or currency_am,
            "qty_am": round(qty_am, 6),
            "qty_eod": round(qty_eod, 6),
            "intraday_traded": False,
            "local_pnl_gbp": 0.0,
            "fx_pnl_gbp": 0.0,
            "total_mark_pnl_gbp": 0.0,
            "local_ret_pct": None,
            "currency_mismatch": False,
        }
        if (
            am is not None
            and eod is not None
            and abs(qty_eod - qty_am) <= 1e-6
            and not orders_by_key.get(key)
        ):
            delta_pnl = _safe_float(getattr(eod, "pnl", 0.0)) - _safe_float(getattr(am, "pnl", 0.0))
            delta_fx = _safe_float(getattr(eod, "fx_impact", 0.0)) - _safe_float(getattr(am, "fx_impact", 0.0))
            delta_local = delta_pnl - delta_fx
            row["local_pnl_gbp"] = round(delta_local, 2)
            row["fx_pnl_gbp"] = round(delta_fx, 2)
            row["total_mark_pnl_gbp"] = round(delta_local + delta_fx, 2)
            price_am = _safe_float(getattr(am, "cur_price", 0.0))
            price_eod = _safe_float(getattr(eod, "cur_price", 0.0))
            currency_ok = bool(currency_am) and currency_am == currency_eod
            ratio = (price_eod / price_am) if price_am > 0 and price_eod > 0 else 0.0
            if not currency_ok or _looks_like_100x_move(symbol, ratio):
                row["currency_mismatch"] = True
            elif price_am > 0 and price_eod > 0:
                row["local_ret_pct"] = round(((price_eod / price_am) - 1.0) * 100.0, 2)
            agg = ledger["accounts"][account]
            agg["local_pnl_gbp"] = round(agg["local_pnl_gbp"] + delta_local, 2)
            agg["fx_pnl_gbp"] = round(agg["fx_pnl_gbp"] + delta_fx, 2)
            agg["total_mark_pnl_gbp"] = round(agg["total_mark_pnl_gbp"] + delta_local + delta_fx, 2)
            total = ledger["totals"]
            total["local_pnl_gbp"] = round(total["local_pnl_gbp"] + delta_local, 2)
            total["fx_pnl_gbp"] = round(total["fx_pnl_gbp"] + delta_fx, 2)
            total["total_mark_pnl_gbp"] = round(total["total_mark_pnl_gbp"] + delta_local + delta_fx, 2)
        else:
            row["intraday_traded"] = True
            row["orders"] = orders_by_key.get(key, [])
            ledger["intraday_orders"].extend(row["orders"])
        position_rows[key] = row
        ledger["positions"].append(row)

    unchanged = [
        row
        for row in ledger["positions"]
        if not row["intraday_traded"] and row["local_pnl_gbp"] != 0
    ]
    winners = sorted(
        [r for r in unchanged if r["local_pnl_gbp"] > 0],
        key=lambda r: r["local_pnl_gbp"],
        reverse=True,
    )[:3]
    losers = sorted(
        [r for r in unchanged if r["local_pnl_gbp"] < 0],
        key=lambda r: r["local_pnl_gbp"],
    )[:3]
    ledger["top_winners"] = [
        {"symbol": r["symbol"], "account": r["account"], "local_pnl_gbp": r["local_pnl_gbp"]}
        for r in winners
    ]
    ledger["top_losers"] = [
        {"symbol": r["symbol"], "account": r["account"], "local_pnl_gbp": r["local_pnl_gbp"]}
        for r in losers
    ]

    alerts, plan_vs_actual = _scan_discipline(
        eod_snap=eod_snap,
        orders=orders,
        today_recs=today_recs,
        histories=histories,
        strategy=strategy,
    )
    ledger["discipline_alerts"] = alerts
    ledger["discipline_alert_count"] = len(alerts)
    ledger["plan_vs_actual"] = plan_vs_actual
    ledger["anomaly_candidates"] = _scan_anomalies(
        eod_snap=eod_snap,
        position_rows=position_rows,
        orders=orders,
        histories=histories,
        strategy=strategy,
    )
    return ledger


def _fmt_gbp(value: Any) -> str:
    return f"£{_safe_float(value):+,.2f}"


def _event_card_lines(event_cards: list[Any] | None) -> list[str]:
    lines: list[str] = []
    for card in event_cards or []:
        if isinstance(card, dict):
            symbol = str(card.get("symbol") or "").strip().upper()
            text = _plain(card.get("text") or card.get("catalyst") or "")
        else:
            symbol, text = "", _plain(card)
        if not is_substantive_catalyst(text):
            continue
        lines.append(f"• {symbol}: {text}" if symbol else f"• {text}")
    return lines


def format_postmarket_review_text(
    ledger: dict[str, Any],
    *,
    event_cards: list[Any] | None = None,
    max_chars: int = 900,
) -> str:
    """Render the daily 0-Token ledger as compact plain text (no # or **)."""
    uk_date = str(ledger.get("uk_date") or "").strip()
    regime = str(ledger.get("regime") or "UNKNOWN").strip().upper()
    totals = ledger.get("totals") or {}
    lines = [
        f"🧾 盘后执行账单 {uk_date} | {regime}",
        (
            f"• 盯市拆解: 股价 {_fmt_gbp(totals.get('local_pnl_gbp'))} | "
            f"汇率 {_fmt_gbp(totals.get('fx_pnl_gbp'))} | "
            f"合计 {_fmt_gbp(totals.get('total_mark_pnl_gbp'))}"
        ),
    ]
    accounts = ledger.get("accounts") or {}
    account_bits = []
    for label in ("ISA", "Invest"):
        item = accounts.get(label) or {}
        if any(abs(_safe_float(item.get(k))) > 0.005 for k in ("local_pnl_gbp", "fx_pnl_gbp")):
            account_bits.append(
                f"{label} 股价 {_fmt_gbp(item.get('local_pnl_gbp'))} / 汇率 {_fmt_gbp(item.get('fx_pnl_gbp'))}"
            )
    if account_bits:
        lines.append("• 分账户: " + " ｜ ".join(account_bits))

    winners = ledger.get("top_winners") or []
    if winners:
        lines.append(
            "• 贡献赢家: "
            + " | ".join(f"{w.get('symbol')} {_fmt_gbp(w.get('local_pnl_gbp'))}" for w in winners)
        )
    losers = ledger.get("top_losers") or []
    if losers:
        lines.append(
            "• 拖累输家: "
            + " | ".join(f"{w.get('symbol')} {_fmt_gbp(w.get('local_pnl_gbp'))}" for w in losers)
        )
    intraday = ledger.get("intraday_orders") or []
    if intraday:
        lines.append(f"• 当日成交 {len(intraday)} 笔（股数变动仓位不伪造未实现盈亏差）")

    alert_count = int(ledger.get("discipline_alert_count") or 0)
    alerts = ledger.get("discipline_alerts") or []
    if alert_count > 0:
        bits = "；".join(
            f"{a.get('code')} {a.get('symbol')}" for a in alerts[:3]
        )
        lines.append(f"• 纪律核对: {alert_count} 项（{bits}）")

    lines.extend(_event_card_lines(event_cards))
    return _clamp_lines(lines, max_chars)


def format_postmarket_alert_text(
    ledger: dict[str, Any],
    *,
    event_cards: list[Any] | None = None,
    max_chars: int = 250,
) -> str:
    """Render the <=250-char alert. Empty when the day is quiet or only no-news cards exist."""
    alerts = ledger.get("discipline_alerts") or []
    cards = _event_card_lines(event_cards)
    if not alerts and not cards:
        return ""
    lines = [f"⚠️ 盘后纪律提醒 {str(ledger.get('uk_date') or '').strip()}"]
    if alerts:
        bits = []
        for alert in alerts[:3]:
            bits.append(f"{alert.get('code')} {alert.get('symbol')}")
        first_reason = str(alerts[0].get("message") or "").strip()[:70]
        lines.append(f"纪律: {'；'.join(bits)}" + (f" | {first_reason}" if first_reason else ""))
    for card in cards[:2]:
        lines.append(card)
    return _clamp_lines(lines, max_chars)
