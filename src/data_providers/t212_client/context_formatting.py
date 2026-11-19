"""Account overview formatting, pending-order quantitative notes, and trade-behavior context builders."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.data_providers.t212_client.models import (
    AccountSnapshot,
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
)

logger = logging.getLogger("src.data_providers.t212_client")

def _format_wallet_split_line(acc: AccountSnapshot) -> str:
    """Render the Invest GBP+USD wallet split with Python-computed amounts (no LLM arithmetic)."""
    gbp_rate = acc.cash_interest_rate_pct
    usd_rate = acc.usd_cash_interest_rate_pct
    usd_equiv_str = ""
    if acc.gbp_usd_rate and acc.gbp_usd_rate > 0:
        usd_equiv_str = f" ≈ £{acc.usd_cash_total / acc.gbp_usd_rate:,.2f}"
    usd_reserved_str = f", 挂单 ${acc.usd_reserved_cash:,.2f}" if acc.usd_reserved_cash > 0 else ""
    usd_part = (
        f"USD 钱包 ${acc.usd_cash_total:,.2f} "
        f"(可用 ${acc.usd_available_to_trade:,.2f}{usd_reserved_str}{usd_equiv_str}, {usd_rate:.1f}%)"
    )
    if acc.gbp_wallet_total is None:
        gbp_part = "GBP 钱包 ≈ 待汇率折算"
    else:
        gbp_reserved_str = (
            f", 挂单 £{acc.gbp_wallet_reserved:,.2f}" if (acc.gbp_wallet_reserved or 0.0) > 0 else ""
        )
        gbp_part = (
            f"GBP 钱包 ≈ £{acc.gbp_wallet_total:,.2f} "
            f"(可用 £{acc.gbp_wallet_available or 0.0:,.2f}{gbp_reserved_str}, {gbp_rate:.1f}%)"
        )
    src_map = {
        "manual_both": "手动双报",
        "derived_from_total": "按 API 总现金推算",
        "pending_fx": "待汇率",
    }
    src = src_map.get(acc.wallet_source, acc.wallet_source)
    src_str = ""
    if src:
        src_str = f"，{src}"
        if acc.wallet_updated_at_uk:
            src_str += f" @ {acc.wallet_updated_at_uk}"
    line = f"• 现金钱包拆解: {gbp_part} + {usd_part}{src_str}"
    if acc.wallet_reconcile_warning:
        line += f"\n  ⚠️ 钱包校准: {acc.wallet_reconcile_warning}"
    return line


def format_account_overview(acc: AccountSnapshot) -> str:
    if not acc.ok:
        return f"⚠️ {acc.account_label} 账户读取失败: {acc.error}"
    curr = acc.account_currency
    stock_only_pnl = acc.unrealized_pnl - acc.total_fx_impact
    ret_pct = ((acc.unrealized_pnl / acc.investments_cost) * 100.0) if acc.investments_cost > 0 else 0.0
    reserved_str = f", 挂单占用(未成交前仍计息): {acc.reserved_cash:.2f}" if acc.reserved_cash > 0 else ""
    note_str = f" {acc.cash_status_note}" if acc.cash_status_note else ""
    realized_str = f" | 累计已实现: {curr} {acc.realized_pnl:.2f}" if acc.realized_pnl is not None else ""
    fallback_head = (
        f" [⚠️ 回退缓存快照 @ {acc.snapshot_fetched_at_uk} UK]"
        if acc.is_fallback and acc.snapshot_fetched_at_uk
        else ""
    )

    lines = [
        f"• 账户总资产: {curr} {acc.total_account_value:,.2f}{fallback_head}",
        f"• 现金余额: {curr} {acc.total_cash:,.2f} (占 {acc.cash_ratio_pct:.1f}% | 可用: {acc.available_to_trade:,.2f}{reserved_str}){note_str}",
    ]
    if acc.multi_currency_enabled:
        lines.append(_format_wallet_split_line(acc))
    lines.extend(
        [
            f"• 持仓市值: {curr} {acc.investments_value:,.2f} (投入成本: {curr} {acc.investments_cost:,.2f})",
            (
                f"• 累计总浮盈: {acc.unrealized_pnl:+,.2f} {curr} ({ret_pct:+.2f}%) "
                f"[本币净盈亏 {stock_only_pnl:+,.2f} {curr} | 累计汇率影响 {acc.total_fx_impact:+,.2f} {curr}]{realized_str}"
            ),
        ]
    )
    return "\n".join(lines)


def format_positions_text(acc: AccountSnapshot) -> str:
    if not acc.ok or not acc.positions:
        return "当前账户无股票持仓"
    curr = acc.account_currency
    lines = []
    for p in acc.positions:
        stock_net = p.pnl - p.fx_impact
        div_str = ""
        if p.cumulative_dividends_gbp > 0:
            tot_ret = (
                p.total_return_with_div_pct
                if p.total_return_with_div_pct is not None
                else (
                    ((p.pnl + p.cumulative_dividends_gbp) / p.total_cost) * 100.0
                    if p.total_cost > 0
                    else p.return_pct
                )
            )
            div_str = f" | 累计派息 +{p.cumulative_dividends_gbp:.2f} {curr} (含息总回报 {tot_ret:+.2f}%)"
        lines.append(
            f"• {p.ticker} ({p.name}) [{p.yahoo_symbol}]: 市值 {curr} {p.current_value:,.2f} (占 {p.weight_pct:.1f}%) | "
            f"均价 {p.avg_price:.2f}→现价 {p.cur_price:.2f} {p.inst_currency} | "
            f"浮盈 {p.pnl:+,.2f} {curr} ({p.return_pct:+.2f}%) "
            f"(本币 {stock_net:+,.2f} | 汇率 {p.fx_impact:+,.2f}) | {p.quantity:.4f}股{div_str}"
        )
    return "\n".join(lines)


def _compute_pending_order_quant_note(
    o: PendingOrderItem,
    *,
    display_curr: str,
    is_gbp_pence: bool,
    bundle: Any | None,
    tech: Any | None,
    hist: Any | None,
) -> str:
    """Compute quote-currency-aligned distance %, ATR multiple, 20d empirical touch rate, and chip/noise tags."""
    raw_trig = (
        o.stop_price
        if (o.stop_price is not None and "STOP" in (o.order_type or "").upper())
        else (o.limit_price if o.limit_price is not None else o.stop_price)
    )
    if raw_trig is None or raw_trig <= 0:
        return ""
    is_buy = (o.side or "").upper() == "BUY"
    is_limit_buy = is_buy and o.limit_price is not None and float(o.limit_price) > 0

    b_curr = getattr(bundle, "currency", None) if bundle is not None else None
    b_price = getattr(bundle, "current_price", None) if bundle is not None else None
    t_close = getattr(tech, "last_close", None) if tech is not None else None
    cur_price = float(b_price) if (b_price is not None and b_price > 0) else (
        float(t_close) if (t_close is not None and t_close > 0) else None
    )
    if cur_price is None or cur_price <= 0:
        return ""

    trig_aligned = float(raw_trig)
    cur_aligned = float(cur_price)

    # Guard against 100x GBp vs GBP mismatch on LSE (.L) symbols
    if o.yahoo_symbol.endswith(".L"):
        ratio_raw = trig_aligned / cur_aligned
        if is_gbp_pence and (o.currency or "").upper() == "GBP" and ratio_raw < 0.05:
            trig_aligned *= 100.0
        elif b_curr == "GBP" and display_curr == "GBp" and ratio_raw > 20.0:
            cur_aligned *= 100.0
        elif not is_gbp_pence and b_curr == "GBp" and ratio_raw < 0.05:
            trig_aligned *= 100.0

    scale_tech = 1.0
    if t_close is not None and t_close > 0:
        tech_ratio = cur_aligned / float(t_close)
        if 80.0 <= tech_ratio <= 125.0:
            scale_tech = 100.0
        elif 0.008 <= tech_ratio <= 0.0125:
            scale_tech = 0.01

    dist_pct = ((trig_aligned / cur_aligned) - 1.0) * 100.0
    atr_mult: float | None = None
    atr_raw = getattr(tech, "atr14", None) if tech is not None else None
    if atr_raw is not None and float(atr_raw) > 0:
        atr_aligned = float(atr_raw) * scale_tech
        atr_mult = abs(trig_aligned - cur_aligned) / atr_aligned

    touch_prob_20d_pct: float | None = None
    if hist is not None and getattr(hist, "empty", True) is False:
        try:
            import numpy as np
            import pandas as pd
            from src.data_providers.analytics import _normalize_adjusted_ohlcv, _resolve_col

            norm_df = _normalize_adjusted_ohlcv(hist)
            c_col = _resolve_col(norm_df, "Close")
            h_col = _resolve_col(norm_df, "High")
            l_col = _resolve_col(norm_df, "Low")
            if c_col and h_col and l_col:
                sub = norm_df[[c_col, h_col, l_col]].apply(pd.to_numeric, errors="coerce").dropna().tail(252)
                if len(sub) >= 25:
                    closes = sub[c_col].to_numpy(dtype=np.float64)
                    highs = sub[h_col].to_numpy(dtype=np.float64)
                    lows = sub[l_col].to_numpy(dtype=np.float64)
                    target_ratio = trig_aligned / cur_aligned
                    window = 20
                    n_bars = len(closes)
                    hits = 0
                    total_windows = 0
                    for i in range(0, n_bars - window):
                        base_c = closes[i]
                        if base_c <= 0:
                            continue
                        total_windows += 1
                        thresh = base_c * target_ratio
                        if target_ratio <= 1.0:
                            if float(np.min(lows[i + 1 : i + 1 + window])) <= thresh:
                                hits += 1
                        else:
                            if float(np.max(highs[i + 1 : i + 1 + window])) >= thresh:
                                hits += 1
                    if total_windows > 0:
                        touch_prob_20d_pct = (hits / total_windows) * 100.0
        except Exception as exc:
            logger.debug("Touch prob calc skipped for %s: %s", o.yahoo_symbol, exc)

    # Anchor the R:R / upside view on the ORDER LIMIT PRICE, never on the live price:
    # a low-ball rebound order must not be judged as "no upside" just because price dropped.
    limit_upside_pct: float | None = None
    target_mean = getattr(bundle, "target_mean_price", None) if bundle is not None else None
    denom = 1.0 + dist_pct / 100.0
    if (
        is_limit_buy
        and b_price is not None
        and float(b_price) > 0
        and target_mean is not None
        and float(target_mean) > 0
        and abs(denom) > 1e-6
    ):
        cur_upside_pct = ((float(target_mean) / float(b_price)) - 1.0) * 100.0
        limit_upside_pct = (((1.0 + cur_upside_pct / 100.0) / denom) - 1.0) * 100.0

    metrics_parts = [f"距现价 {dist_pct:+.1f}%"]
    if limit_upside_pct is not None:
        metrics_parts.append(f"限价空间 {limit_upside_pct:+.1f}%")
    if atr_mult is not None:
        metrics_parts.append(f"{atr_mult:.1f}倍ATR")
    if touch_prob_20d_pct is not None:
        metrics_parts.append(f"20日触价率 {touch_prob_20d_pct:.0f}%")

    tags: list[str] = []
    is_stop = (o.stop_price is not None) or ("STOP" in (o.order_type or "").upper())
    if is_limit_buy and -12.0 <= dist_pct <= -3.0:
        tags.append("[有效回踩埋伏单]")
    if is_limit_buy and dist_pct > 0.5:
        tags.append("[追价单]")
    if is_buy and atr_mult is not None and atr_mult > 4.0 and touch_prob_20d_pct is not None and touch_prob_20d_pct < 5.0:
        tags.append("[远端低概率单]")
    if is_stop and atr_mult is not None and atr_mult < 0.8:
        tags.append("[止损过窄易被日常波动扫出]")
    if is_buy and tech is not None:
        c_low_raw = getattr(tech, "cyq_cost_90_low", None)
        c_high_raw = getattr(tech, "cyq_cost_90_high", None)
        if c_low_raw is not None and c_high_raw is not None:
            c_low = float(c_low_raw) * scale_tech
            c_high = float(c_high_raw) * scale_tech
            if 0.0 < c_low <= trig_aligned <= c_high:
                tags.append("[位于90%筹码支撑带]")

    tag_str = f" {' '.join(tags)}" if tags else ""
    return f" | {' / '.join(metrics_parts)}{tag_str}"


def compute_trade_behavior_stats(
    data: PortfolioData,
    *,
    recent_days: int = 7,
    history_days: int = 30,
    cooldown_days: int = 15,
    now_ms: int | None = None,
    exempt_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """0-Token trade cadence, ticket size, fee ratio and post-exit cooldown statistics.

    ``partial`` is True when the fetched order history covers less than ``history_days``;
    callers must treat cadence caps as advisory (never a hard block) in that case.
    Amounts stay in each fill's wallet currency — USD never inflates GBP totals.

    v5.5.7: zero-quota core wide-base ETF buys are excluded via ``exempt_symbols`` and buys of the
    same (account, ticker, UK date) are merged to one ticket so a partially filled limit order
    cannot inflate the weekly/monthly buy count.
    """
    now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
    orders = sorted(
        [*data.isa.recent_orders, *data.invest.recent_orders],
        key=lambda o: o.filled_at_ms,
        reverse=True,
    )
    recent_cutoff = now - max(1, recent_days) * 86400 * 1000
    history_cutoff = now - max(1, history_days) * 86400 * 1000
    exempt_up = {str(s).strip().upper() for s in (exempt_symbols or set()) if str(s).strip()}

    def _is_exempt(o: Any) -> bool:
        sym = str(getattr(o, "yahoo_symbol", "") or "").strip().upper()
        tick = str(getattr(o, "ticker", "") or "").strip().upper()
        for candidate in (sym, tick):
            if candidate and (candidate in exempt_up or candidate.split(".")[0] in exempt_up):
                return True
        return False

    def _ticket_day(o: Any) -> str:
        uk_date = str(getattr(o, "uk_time_str", "") or "").strip()[:10]
        return uk_date or str(int(getattr(o, "filled_at_ms", 0) or 0) // 86400000)

    buys_7d = buys_30d = sells_7d = sells_30d = 0
    buy_keys_7d: set[tuple[str, str, str]] = set()
    buy_keys_30d: set[tuple[str, str, str]] = set()
    buy_amt_by_ccy: dict[str, float] = {}
    buy_cnt_by_ccy: dict[str, int] = {}
    turnover_30d = taxes_30d = 0.0
    last_sell_ms: dict[tuple[str, str], int] = {}
    oldest_ms: int | None = None

    for o in orders:
        if oldest_ms is None or o.filled_at_ms < oldest_ms:
            oldest_ms = o.filled_at_ms
        is_buy = (o.side or "").upper() != "SELL"
        ccy = str(o.wallet_currency or "GBP").strip().upper() or "GBP"
        in_30d = o.filled_at_ms >= history_cutoff
        in_7d = o.filled_at_ms >= recent_cutoff
        quota_exempt = bool(is_buy and _is_exempt(o))
        ticket_key = (
            (o.account or "").upper(),
            (o.ticker or o.yahoo_symbol or "").upper(),
            _ticket_day(o),
        )
        counted_buy = bool(is_buy and not quota_exempt)
        new_buy_30d = counted_buy and ticket_key not in buy_keys_30d
        new_buy_7d = counted_buy and ticket_key not in buy_keys_7d
        if in_30d:
            turnover_30d += abs(float(o.net_value or 0.0))
            taxes_30d += abs(float(o.tax_total or 0.0))
            if counted_buy and new_buy_30d:
                buys_30d += 1
                buy_keys_30d.add(ticket_key)
            elif not is_buy:
                sells_30d += 1
        if in_7d:
            if counted_buy and new_buy_7d:
                buys_7d += 1
                buy_keys_7d.add(ticket_key)
            elif not is_buy:
                sells_7d += 1
        if counted_buy and in_30d:
            buy_amt_by_ccy[ccy] = buy_amt_by_ccy.get(ccy, 0.0) + float(o.net_value or 0.0)
            buy_cnt_by_ccy[ccy] = buy_cnt_by_ccy.get(ccy, 0) + 1
        if not is_buy:
            key = ((o.account or "").upper(), (o.ticker or "").upper())
            prev = last_sell_ms.get(key)
            if prev is None or o.filled_at_ms > prev:
                last_sell_ms[key] = o.filled_at_ms

    avg_buy_by_ccy = {
        ccy: round(amt / buy_cnt_by_ccy[ccy], 2)
        for ccy, amt in buy_amt_by_ccy.items()
        if buy_cnt_by_ccy.get(ccy, 0) > 0
    }

    rate = float(data.gbp_usd_rate) if (data.gbp_usd_rate or 0.0) > 0 else None
    gbp_total = 0.0
    gbp_complete = True
    for ccy, amt in buy_amt_by_ccy.items():
        if ccy in {"GBP", "GBX"}:
            gbp_total += amt
        elif ccy == "USD" and rate is not None:
            gbp_total += amt / rate
        else:
            gbp_complete = False
    avg_buy_gbp = round(gbp_total / buys_30d, 2) if (gbp_complete and buys_30d > 0) else None

    held = {(p.ticker or "").upper() for p in data.all_positions()}
    cooldown_window = max(int(cooldown_days), int(history_days))
    recently_closed: list[dict[str, Any]] = []
    for (acct, ticker), ms in last_sell_ms.items():
        if not ticker or ticker in held:
            continue
        days_since = (now - ms) / 86400000.0
        if days_since > cooldown_window:
            continue
        recently_closed.append(
            {
                "ticker": ticker,
                "account": acct,
                "days_since_sell": round(days_since, 1),
                "in_cooldown": days_since <= float(cooldown_days),
            }
        )
    recently_closed.sort(key=lambda r: r["days_since_sell"])

    coverage_days = round((now - oldest_ms) / 86400000.0, 1) if oldest_ms is not None else 0.0
    partial = bool(oldest_ms is not None and coverage_days < float(history_days) - 0.5)

    return {
        "buys_7d": buys_7d,
        "buys_30d": buys_30d,
        "sells_7d": sells_7d,
        "sells_30d": sells_30d,
        "avg_buy_amount_by_currency": avg_buy_by_ccy,
        "avg_buy_gbp": avg_buy_gbp,
        "taxes_30d": round(taxes_30d, 2),
        "turnover_30d": round(turnover_30d, 2),
        "fees_pct_of_turnover": round((taxes_30d / turnover_30d) * 100.0, 3) if turnover_30d > 0 else 0.0,
        "recently_closed": recently_closed[:12],
        "cooldown_days": int(cooldown_days),
        "coverage_days": coverage_days,
        "partial": partial,
    }


def format_trade_behavior_text(behavior: dict[str, Any]) -> str:
    """Render the structured cadence/cooldown stats as one compact plain-text line."""
    if not behavior:
        return ""
    parts = [
        f"近7天买入 {behavior.get('buys_7d', 0)} 笔 / 近30天 {behavior.get('buys_30d', 0)} 笔",
        f"卖出 {behavior.get('sells_7d', 0)} 笔 / {behavior.get('sells_30d', 0)} 笔",
    ]
    avg_by_ccy = behavior.get("avg_buy_amount_by_currency") or {}
    avg_parts = [f"{ccy} {float(amt):,.0f}" for ccy, amt in sorted(avg_by_ccy.items()) if float(amt) > 0]
    if avg_parts:
        parts.append("平均单笔买入 " + " / ".join(avg_parts))
    avg_gbp = behavior.get("avg_buy_gbp")
    if avg_gbp is not None and float(avg_gbp) > 0:
        parts.append(f"折合 £{float(avg_gbp):,.0f}")
    parts.append(f"税费占成交额 {float(behavior.get('fees_pct_of_turnover') or 0.0):.2f}%")
    closed = behavior.get("recently_closed") or []
    if closed:
        cooldown = int(behavior.get("cooldown_days") or 0)
        items = "、".join(
            f"{row['ticker']}({float(row['days_since_sell']):.0f}天{'·冷却' if row.get('in_cooldown') else ''})"
            for row in closed[:6]
        )
        parts.append(f"清仓 {cooldown} 天冷却清单: {items}")
    if behavior.get("partial"):
        parts.append(
            f"⚠️历史仅覆盖 {behavior.get('coverage_days', 0)} 天，"
            f"7/30天频次暂不硬拦（{int(behavior.get('cooldown_days') or 0)}天清仓冷却硬生效）"
        )
    return " | ".join(parts)


def build_trade_context(
    data: PortfolioData,
    recent_days: int = 7,
    history_days: int = 30,
    max_recent_lines: int = 15,
    max_summary_lines: int = 12,
    bundles: dict[str, Any] | None = None,
    pending_us_only: bool = False,
    tech_map: dict[str, Any] | None = None,
    histories: dict[str, Any] | None = None,
    cooldown_days: int = 15,
    exempt_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """Build pending orders, 7-day fills, and 30-day ticker behavior stats for AI prompts."""
    all_orders = sorted(
        [*data.isa.recent_orders, *data.invest.recent_orders],
        key=lambda o: o.filled_at_ms,
        reverse=True,
    )
    all_pending = [*data.isa.pending_orders, *data.invest.pending_orders]
    if pending_us_only:
        all_pending = [o for o in all_pending if "." not in o.yahoo_symbol]
    recent_cutoff_ms = int(time.time() * 1000) - recent_days * 86400 * 1000
    recent_7d = [o for o in all_orders if o.filled_at_ms >= recent_cutoff_ms]

    def _fmt_order(o: HistoricalOrderItem) -> str:
        side_lbl = "卖出" if o.side == "SELL" else "买入"
        cash_lbl = "回笼" if o.side == "SELL" else "投入"
        pnl_part = (
            f" | 已实现 {o.wallet_currency} {o.realised_pnl:+.2f}"
            if o.side == "SELL" and abs(o.realised_pnl) >= 0.01
            else ""
        )
        tax_part = f" | 税费 {o.wallet_currency} {o.tax_total:.2f}" if o.tax_total >= 0.01 else ""
        return (
            f"• [{o.uk_time_str} UK | {o.account}·{side_lbl}] {o.ticker} ({o.name})："
            f"{o.quantity:.4f}股 @ {o.price:.2f} {o.inst_currency} | "
            f"{cash_lbl} {o.wallet_currency} {o.net_value:.2f}{pnl_part}{tax_part} ({o.order_type})"
        )

    recent_text = (
        "\n".join(_fmt_order(o) for o in recent_7d[:max_recent_lines])
        if recent_7d
        else f"（近 {recent_days} 天无已成交买卖记录）"
    )

    stats: dict[str, dict[str, Any]] = {}
    total_buys = total_sells = 0
    total_buy_amt = total_sell_amt = total_realised = total_taxes = 0.0
    win_sells = loss_sells = 0
    sold_tickers: set[str] = set()

    for o in all_orders:
        key = f"{o.account}__{o.ticker}"
        if key not in stats:
            stats[key] = {
                "account": o.account,
                "ticker": o.ticker,
                "currency": o.wallet_currency,
                "buys": 0,
                "sells": 0,
                "buy_amt": 0.0,
                "sell_amt": 0.0,
                "realised": 0.0,
                "last_uk": o.uk_time_str,
                "last_ms": o.filled_at_ms,
            }
        st = stats[key]
        if o.filled_at_ms > st["last_ms"]:
            st["last_ms"] = o.filled_at_ms
            st["last_uk"] = o.uk_time_str

        if o.side == "SELL":
            total_sells += 1
            total_sell_amt += o.net_value
            total_realised += o.realised_pnl
            st["sells"] += 1
            st["sell_amt"] += o.net_value
            st["realised"] += o.realised_pnl
            sold_tickers.add(o.ticker)
            if o.realised_pnl > 0.01:
                win_sells += 1
            elif o.realised_pnl < -0.01:
                loss_sells += 1
        else:
            total_buys += 1
            total_buy_amt += o.net_value
            st["buys"] += 1
            st["buy_amt"] += o.net_value
        total_taxes += o.tax_total

    held_tickers = {p.ticker for p in data.all_positions()}
    closed_tickers = [t for t in sold_tickers if t not in held_tickers]

    stat_rows = sorted(stats.values(), key=lambda x: x["last_ms"], reverse=True)[:max_summary_lines]
    summary_lines = []
    for s in stat_rows:
        parts = []
        if s["buys"] > 0:
            parts.append(f"买{s['buys']}笔({s['currency']} {s['buy_amt']:.0f})")
        if s["sells"] > 0:
            parts.append(f"卖{s['sells']}笔({s['currency']} {s['sell_amt']:.0f})")
        pnl_s = f" | 已实现 {s['currency']} {s['realised']:+.2f}" if abs(s["realised"]) >= 0.01 else ""
        status_tag = " [已清仓]" if s["ticker"] in closed_tickers else ""
        summary_lines.append(f"• [{s['account']}] {s['ticker']}{status_tag}: {' / '.join(parts)}{pnl_s} | 最近 {s['last_uk']} UK")

    if all_orders:
        closed_note = f" | 期间已清仓退出: {', '.join(closed_tickers)}" if closed_tickers else ""
        header = (
            f"近 {history_days} 天共 {len(all_orders)} 笔成交（买入 {total_buys} 笔约 {total_buy_amt:.0f}，"
            f"卖出 {total_sells} 笔约 {total_sell_amt:.0f}；已实现净盈亏 {total_realised:+.2f}，"
            f"盈利平仓 {win_sells} 笔 / 亏损平仓 {loss_sells} 笔，累计税费 {total_taxes:.2f}）{closed_note}"
        )
        summary_text = header + "\n" + "\n".join(summary_lines)
    else:
        summary_text = f"（近 {history_days} 天无历史成交订单）"

    pending_lines = []
    for o in all_pending:
        b = bundles.get(o.yahoo_symbol) if bundles else None
        # Fix T212 /equity/orders returning account currency (GBP) instead of instrument quote currency (GBp/USD/EUR)
        display_curr = o.currency
        is_gbp_pence = (display_curr in {"GBX", "GBp"}) or (
            b is not None and b.currency == "GBp" and o.yahoo_symbol.endswith(".L")
        )
        if is_gbp_pence:
            display_curr = "GBp"
        elif b is not None and b.currency and not o.yahoo_symbol.endswith(".L") and display_curr == "GBP":
            display_curr = b.currency

        trig = []
        if o.limit_price is not None:
            gbp_eq = f"(£{o.limit_price / 100:.2f})" if is_gbp_pence else ""
            trig.append(f"限价 {o.limit_price:.2f}{gbp_eq}")
        if o.stop_price is not None:
            gbp_eq = f"(£{o.stop_price / 100:.2f})" if is_gbp_pence else ""
            trig.append(f"止损 {o.stop_price:.2f}{gbp_eq}")

        market_ctx = ""
        if b is not None and b.current_price is not None:
            tgt = f", 目标均价 {b.target_mean_price:.2f}" if b.target_mean_price else ""
            market_ctx = f" | 市场现价 {b.current_price:.2f} {b.currency or display_curr}{tgt}"

        quant_note = _compute_pending_order_quant_note(
            o,
            display_curr=display_curr,
            is_gbp_pence=is_gbp_pence,
            bundle=b,
            tech=tech_map.get(o.yahoo_symbol) if tech_map else None,
            hist=histories.get(o.yahoo_symbol) if histories else None,
        )

        qty_str = f"{o.quantity:.4f}股" if o.quantity > 0 else f"金额 {o.currency} {o.value:.2f}"
        if o.filled_quantity > 0 or o.status == "PARTIALLY_FILLED":
            qty_str += f"[部分成交,已成{o.filled_quantity:.4f}股]"
        side_lbl = "卖出" if o.side == "SELL" else "买入"
        pending_lines.append(
            f"• [{o.account}·挂单] {o.ticker} ({o.name}) [{o.yahoo_symbol}] {side_lbl} {qty_str} | "
            f"{o.order_type} {'/'.join(trig)} {display_curr} ({o.time_in_force}){market_ctx}{quant_note}"
        )

    pending_err_parts = [
        f"{acc.account_label}: {acc.pending_orders_error}"
        for acc in (data.isa, data.invest)
        if acc.ok and acc.pending_orders_error
    ]
    if pending_lines:
        if pending_err_parts:
            pending_lines.append(
                f"⚠️ 部分账户活动挂单接口临时异常（{', '.join(pending_err_parts)}），请勿据此判断该账户无挂单"
            )
        pending_text = "\n".join(pending_lines)
    elif pending_err_parts:
        pending_text = f"⚠️ 活动挂单接口临时异常（{', '.join(pending_err_parts)}），请勿据此判断挂单已撤销"
    else:
        pending_text = "（当前无未成交活动挂单）"

    history_cutoff_ms = int(time.time() * 1000) - history_days * 86400 * 1000
    all_tx = [
        tx
        for tx in (*data.isa.recent_transactions, *data.invest.recent_transactions)
        if tx.date_time_ms >= history_cutoff_ms
    ]
    all_divs = [
        dv
        for dv in (*data.isa.recent_dividends, *data.invest.recent_dividends)
        if dv.paid_on_ms >= history_cutoff_ms
    ]

    cash_flow_summary_text = ""
    if all_tx or all_divs:
        def _sum_by_currency(items: list[Any], tx_types: set[str] | None = None) -> dict[str, float]:
            """Aggregate amounts per currency so USD flows never inflate GBP totals."""
            sums: dict[str, float] = {}
            for item in items:
                if tx_types is not None and getattr(item, "tx_type", "") not in tx_types:
                    continue
                ccy = str(getattr(item, "currency", "") or "GBP").strip().upper() or "GBP"
                sums[ccy] = sums.get(ccy, 0.0) + float(getattr(item, "amount", 0.0) or 0.0)
            return sums

        def _fmt_sums(label: str, sums: dict[str, float]) -> list[str]:
            return [
                f"{label} {amount:+,.2f} {ccy}"
                for ccy, amount in sums.items()
                if abs(amount) >= 0.01
            ]

        cf_parts: list[str] = []
        cf_parts.extend(_fmt_sums("外部入金", _sum_by_currency(all_tx, {"DEPOSIT"})))
        cf_parts.extend(_fmt_sums("外部出金", _sum_by_currency(all_tx, {"WITHDRAW"})))
        cf_parts.extend(_fmt_sums("账户划转", _sum_by_currency(all_tx, {"TRANSFER"})))
        cf_parts.extend(_fmt_sums("平台费用", _sum_by_currency(all_tx, {"FEE"})))
        div_sums = _sum_by_currency(all_divs)
        div_parts = [f"+{amount:,.2f} {ccy}" for ccy, amount in div_sums.items() if abs(amount) >= 0.01]
        if div_parts:
            cf_parts.append(f"派息入账 {' '.join(div_parts)}({len(all_divs)}笔)")
        cf_parts.extend(
            _fmt_sums(
                "现金利息",
                _sum_by_currency(all_tx, {"INTEREST_ON_FREE_CASH", "LENDING_INTEREST"}),
            )
        )

        tx_label_map = {
            "DEPOSIT": "入金",
            "WITHDRAW": "出金",
            "TRANSFER": "划转",
            "FEE": "扣费",
            "INTEREST_ON_FREE_CASH": "现金利息",
            "LENDING_INTEREST": "借券利息",
        }
        merged_events: list[tuple[int, str]] = []
        # Top summary line keeps the full 现金利息 totals; the detail list skips the daily
        # £1~£2 interest dribble so real deposits/withdrawals/dividends get the 6 slots.
        interest_noise = {"INTEREST_ON_FREE_CASH", "LENDING_INTEREST"}
        for tx in all_tx:
            if str(getattr(tx, "tx_type", "") or "") in interest_noise:
                continue
            lbl = tx_label_map.get(tx.tx_type, tx.tx_type)
            merged_events.append(
                (tx.date_time_ms, f"• [{tx.uk_time_str} UK | {tx.account}·{lbl}] {tx.amount:+,.2f} {tx.currency}")
            )
        for dv in all_divs:
            merged_events.append(
                (dv.paid_on_ms, f"• [{dv.uk_time_str} UK | {dv.account}·派息] {dv.ticker}: +{dv.amount:,.2f} {dv.currency}")
            )
        merged_events.sort(key=lambda item: item[0], reverse=True)
        cf_header = f"[资金流水与派息(近 {history_days} 天)] " + (" | ".join(cf_parts) if cf_parts else "（微量变动）")
        cf_lines = [line for _, line in merged_events[:6]]
        cash_flow_summary_text = cf_header + ("\n" + "\n".join(cf_lines) if cf_lines else "")

    trade_behavior = compute_trade_behavior_stats(
        data,
        recent_days=recent_days,
        history_days=history_days,
        cooldown_days=cooldown_days,
        exempt_symbols=exempt_symbols,
    )

    return {
        "recent_trades_text": recent_text,
        "trade_summary_30d_text": summary_text,
        "pending_orders_text": pending_text,
        "cash_flow_summary_text": cash_flow_summary_text,
        "traded_tickers": ", ".join(dict.fromkeys(o.ticker for o in all_orders)) or "无",
        "trade_behavior": trade_behavior,
        "trade_behavior_text": format_trade_behavior_text(trade_behavior),
    }
