"""NAV time-weighted return (TWR), account-split performance, trade attribution, and core policy benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.data_providers.analytics.technical import (
    _extract_close_series,
    _resolve_col,
)
from src.data_providers.t212_client import PortfolioData


# ---------------------------------------------------------------------------
# IMP-17: Live account performance (NAV -> TWR), trade attribution & signal QC
# ---------------------------------------------------------------------------

_EXTERNAL_FLOW_SIGNS: dict[str, float] = {
    "DEPOSIT": 1.0,
    "WITHDRAW": -1.0,
    "WITHDRAWAL": -1.0,
}


def _event_uk_date(item: Any) -> str:
    """Best-effort UK calendar date (YYYY-MM-DD) from a T212 cash-event-like model."""
    for attr in ("uk_time_str", "date_time", "paid_on"):
        raw = str(getattr(item, attr, "") or "").strip()
        if len(raw) >= 10:
            return raw[:10]
    return ""


def build_nav_row(
    portfolio: PortfolioData,
    benchmark_close: float | None = None,
    uk_date: str | None = None,
) -> dict[str, Any]:
    """Build one lightweight daily NAV row (GBP-only scalars) for ``nav_history``.

    Only genuine external cash movements (DEPOSIT / WITHDRAW) count as flows.
    Cash interest, lending interest and platform fees are internal investment
    P&L / drag and must stay inside the TWR numerator; dividends likewise already
    sit inside NAV, so treating any of them as inflow would understate returns.
    """
    day = (uk_date or str(getattr(portfolio, "fetched_at_uk", "") or "").strip()[:10]).strip()[:10]
    total = float(getattr(portfolio, "total_combined_value", 0.0) or 0.0)
    cash = float(getattr(portfolio, "total_combined_cash", 0.0) or 0.0)

    net_flow = 0.0
    isa_net_flow = 0.0
    invest_net_flow = 0.0
    for acc in (portfolio.isa, portfolio.invest):
        acc_flow = 0.0
        for tx in getattr(acc, "recent_transactions", []) or []:
            if _event_uk_date(tx) != day:
                continue
            ccy = str(getattr(tx, "currency", "GBP") or "GBP").strip().upper()
            amount = float(getattr(tx, "amount", 0.0) or 0.0)
            if ccy == "USD":
                fx = float(getattr(portfolio, "gbp_usd_rate", None) or 0.0)
                if fx <= 0:
                    continue  # cannot convert safely -> never fabricate a GBP flow
                amount = amount / fx
            elif ccy not in ("GBP", ""):
                continue  # keep the NAV series strictly GBP-denominated
            sign = _EXTERNAL_FLOW_SIGNS.get(str(getattr(tx, "tx_type", "") or "").strip().upper())
            if sign is None:
                continue
            acc_flow += sign * amount
        net_flow += acc_flow
        # v5.5.3: split the same external-flow stream per philosophical account for split TWR.
        if str(getattr(acc, "account_label", "") or "").strip().upper() == "ISA":
            isa_net_flow += acc_flow
        else:
            invest_net_flow += acc_flow

    return {
        "uk_date": day,
        "total_value_gbp": round(total, 2),
        "cash_gbp": round(cash, 2),
        "invested_gbp": round(max(total - cash, 0.0), 2),
        "net_flow_gbp": round(net_flow, 2),
        "isa_value_gbp": round(float(getattr(portfolio.isa, "total_account_value", 0.0) or 0.0), 2),
        "invest_value_gbp": round(float(getattr(portfolio.invest, "total_account_value", 0.0) or 0.0), 2),
        "isa_net_flow_gbp": round(isa_net_flow, 2),
        "invest_net_flow_gbp": round(invest_net_flow, 2),
        "benchmark_close": round(float(benchmark_close), 6) if benchmark_close else None,
        "cash_rate_pct": float(getattr(portfolio, "cash_interest_rate_pct", 0.0) or 0.0),
    }


def compute_portfolio_performance(
    nav_rows: list[dict[str, Any]],
    benchmark_hist: pd.DataFrame | None = None,
    risk_free_annual_pct: float | None = None,
    min_periods: int = 15,
) -> dict[str, Any]:
    """Compute TWR-based account performance from persisted daily NAV rows (all GBP).

    TWR per period: ``r_t = (NAV_t - CF_t) / NAV_{t-1} - 1`` where CF_t is that day's
    *external* net flow. Cash drag and benchmark alpha/beta/IR are reported alongside.
    Callers must label the window: sub-1-year samples cannot be presented as
    annualized Sharpe/CAGR without a sample-size caveat.
    """
    empty: dict[str, Any] = {"ok": False, "samples": 0, "reason": "净值快照不足（至少需 2 天）"}
    if not nav_rows:
        return empty

    records: list[dict[str, Any]] = []
    for row in nav_rows:
        raw_date = str(row.get("uk_date") or "").strip()[:10]
        if len(raw_date) < 10:
            continue
        try:
            stamped = pd.Timestamp(raw_date)
            nav = float(row.get("total_value_gbp"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(nav) or nav <= 0.0:
            continue
        try:
            flow = float(row.get("net_flow_gbp") or 0.0)
        except (TypeError, ValueError):
            flow = 0.0
        try:
            cash = float(row.get("cash_gbp") or 0.0)
        except (TypeError, ValueError):
            cash = 0.0
        bench_raw = row.get("benchmark_close")
        try:
            bench = float(bench_raw) if bench_raw is not None else np.nan
        except (TypeError, ValueError):
            bench = np.nan
        records.append(
            {"date": stamped.normalize(), "nav": nav, "flow": flow, "cash": max(cash, 0.0), "bench": bench}
        )

    if len(records) < 2:
        return empty

    df = pd.DataFrame(records).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 2:
        return empty

    prev_nav = df["nav"].shift(1)
    valid = prev_nav > 0.0
    rets = pd.Series(np.nan, index=df.index, dtype=float)
    rets.loc[valid] = (df.loc[valid, "nav"] - df.loc[valid, "flow"]) / prev_nav.loc[valid] - 1.0
    rets = rets.dropna()
    if rets.empty:
        return empty

    arr = 1.0 + rets.to_numpy(dtype=float)
    curve = np.cumprod(arr)
    curve_full = np.concatenate(([1.0], curve))
    peaks = np.maximum.accumulate(curve_full)
    drawdown = curve_full / peaks - 1.0
    trough_k = int(np.argmin(drawdown))
    max_dd = float(drawdown[trough_k])

    def _date_at(k: int) -> pd.Timestamp:
        return df.index[0] if k == 0 else rets.index[k - 1]

    peak_k = int(np.argmax(curve_full[: trough_k + 1])) if trough_k > 0 else 0
    peak_value = float(curve_full[peak_k])
    recovery_k: int | None = None
    for k in range(trough_k + 1, len(curve_full)):
        if curve_full[k] >= peak_value - 1e-12:
            recovery_k = k
            break
    trough_date = _date_at(trough_k)
    recovery_days: int | None = None
    if recovery_k is not None:
        recovery_days = int((_date_at(recovery_k) - trough_date).days)

    total_return = float(curve[-1]) - 1.0
    span_days = max(int((df.index[-1] - df.index[0]).days), 1)
    years = span_days / 365.25
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if (years >= 0.25 and total_return > -1.0) else None

    mean_gap = span_days / max(len(df) - 1, 1)
    periods_per_year = float(np.clip(365.0 / max(mean_gap, 0.5), 12.0, 366.0))

    ann_vol: float | None = None
    if len(rets) >= min_periods:
        std = float(np.std(rets.to_numpy(dtype=float), ddof=1))
        if np.isfinite(std) and std > 0.0:
            ann_vol = std * math.sqrt(periods_per_year) * 100.0

    rf_pct = 0.0 if risk_free_annual_pct is None else float(risk_free_annual_pct)
    mar_period = (rf_pct / 100.0) / periods_per_year
    excess = rets.to_numpy(dtype=float) - mar_period
    downside = float(np.sqrt(np.mean(np.square(np.minimum(excess, 0.0)))))
    sortino: float | None = None
    if len(rets) >= min_periods and downside > 1e-12:
        sortino = float(np.mean(excess)) / downside * math.sqrt(periods_per_year)

    calmar: float | None = None
    if cagr is not None and max_dd < -1e-9:
        calmar = cagr / abs(max_dd)

    bench_rets: pd.Series | None = None
    bench_source = ""
    if df["bench"].notna().sum() >= max(5, int(len(df) * 0.6)):
        bench_series = df["bench"].astype(float).dropna()
        bench_rets = bench_series.pct_change().dropna()
        bench_source = "nav_benchmark_close"
    elif benchmark_hist is not None and not getattr(benchmark_hist, "empty", True):
        bench_col = _resolve_col(benchmark_hist, "Adj Close") or _resolve_col(benchmark_hist, "Close")
        if bench_col:
            series = pd.to_numeric(benchmark_hist[bench_col], errors="coerce").dropna().astype(float)
            series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            aligned = series.reindex(df.index).ffill(limit=3)
            bench_rets = aligned.pct_change().dropna()
            bench_source = "benchmark_hist"

    alpha_ann: float | None = None
    beta_val: float | None = None
    corr_val: float | None = None
    info_ratio: float | None = None
    tracking_error: float | None = None
    excess_ann: float | None = None
    if bench_rets is not None and not bench_rets.empty:
        pair = pd.concat([rets.rename("p"), bench_rets.rename("b")], axis=1, join="inner").dropna()
        if len(pair) >= max(10, min_periods):
            p_arr = pair["p"].to_numpy(dtype=float)
            b_arr = pair["b"].to_numpy(dtype=float)
            var_b = float(np.var(b_arr, ddof=1))
            if var_b > 1e-14:
                beta_val = float(np.cov(p_arr, b_arr, ddof=1)[0, 1]) / var_b
                alpha_ann = (float(np.mean(p_arr)) - beta_val * float(np.mean(b_arr))) * periods_per_year * 100.0
            std_p = float(np.std(p_arr, ddof=1))
            std_b = float(np.std(b_arr, ddof=1))
            if std_p > 1e-14 and std_b > 1e-14:
                corr_val = float(np.corrcoef(p_arr, b_arr)[0, 1])
            diff = p_arr - b_arr
            std_diff = float(np.std(diff, ddof=1))
            excess_ann = float(np.mean(diff)) * periods_per_year * 100.0
            if std_diff > 1e-14:
                tracking_error = std_diff * math.sqrt(periods_per_year) * 100.0
                info_ratio = float(np.mean(diff)) / std_diff * math.sqrt(periods_per_year)

    cash_drag: float | None = None
    invested_mask = df["nav"] > 0.0
    if invested_mask.any():
        cash_ratio = (df.loc[invested_mask, "cash"] / df.loc[invested_mask, "nav"]).clip(lower=0.0, upper=1.0)
        cash_drag = float(cash_ratio.mean()) * 100.0

    expected_bdays = int(np.busday_count(df.index[0].date(), (df.index[-1] + pd.Timedelta(days=1)).date()))

    return {
        "ok": True,
        "samples": int(len(df)),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "span_days": span_days,
        "coverage_pct": round(min(100.0, len(df) / max(expected_bdays, 1) * 100.0), 1),
        "periods_per_year": round(periods_per_year, 1),
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2) if cagr is not None else None,
        "ann_vol_pct": round(ann_vol, 2) if ann_vol is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "risk_free_annual_pct": round(rf_pct, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "drawdown_peak_date": _date_at(peak_k).strftime("%Y-%m-%d"),
        "drawdown_trough_date": trough_date.strftime("%Y-%m-%d"),
        "drawdown_recovery_days": recovery_days,
        "drawdown_unrecovered": bool(recovery_k is None and max_dd < -1e-9),
        "calmar": round(calmar, 2) if calmar is not None else None,
        "cash_drag_pct": round(cash_drag, 1) if cash_drag is not None else None,
        "bench_source": bench_source,
        "bench_beta": round(beta_val, 3) if beta_val is not None else None,
        "bench_corr": round(corr_val, 3) if corr_val is not None else None,
        "bench_alpha_annual_pct": round(alpha_ann, 2) if alpha_ann is not None else None,
        "bench_excess_annual_pct": round(excess_ann, 2) if excess_ann is not None else None,
        "bench_tracking_error_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "bench_info_ratio": round(info_ratio, 2) if info_ratio is not None else None,
    }


def compute_account_split_performance(
    nav_rows: list[dict[str, Any]],
    benchmark_hist: pd.DataFrame | None = None,
    risk_free_annual_pct: float | None = None,
) -> dict[str, Any]:
    """Per-philosophy TWR: ISA value book vs Invest momentum book on the same math.

    Each account reuses ``compute_portfolio_performance`` with its own NAV leg
    (``*_value_gbp + *_net_flow_gbp``) and its own external-flow stream, so ISA
    capital injections never masquerade as Invest alpha (or vice versa).
    """
    out: dict[str, Any] = {"ok": False, "isa": None, "invest": None}
    if not nav_rows:
        return out

    def _leg(prefix: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in nav_rows:
            try:
                value = float(row.get(f"{prefix}_value_gbp") or 0.0)
                flow = float(row.get(f"{prefix}_net_flow_gbp") or 0.0)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "uk_date": row.get("uk_date"),
                    "total_value_gbp": value + flow,
                    "net_flow_gbp": flow,
                    "cash_gbp": 0.0,
                    "benchmark_close": row.get("benchmark_close"),
                }
            )
        return rows

    isa_perf = compute_portfolio_performance(
        _leg("isa"), benchmark_hist=benchmark_hist, risk_free_annual_pct=risk_free_annual_pct
    )
    invest_perf = compute_portfolio_performance(
        _leg("invest"), benchmark_hist=benchmark_hist, risk_free_annual_pct=risk_free_annual_pct
    )
    out["isa"] = isa_perf
    out["invest"] = invest_perf
    out["ok"] = bool(isa_perf.get("ok") or invest_perf.get("ok"))
    return out


def compute_trade_attribution(portfolio: PortfolioData) -> dict[str, Any]:
    """Aggregate *actual* filled-order performance, grouped by wallet currency.

    Never cross-sums currencies (GBP/USD/GBp isolation). Sources are already-fetched
    T212 fields: ``realised_pnl`` per SELL fill, ``tax_total``, ``recent_dividends``
    and account ``total_fx_impact``. Holding days are approximated as
    "sell fill − earliest prior BUY fill of that ticker in the same account".
    """
    per_ccy: dict[str, dict[str, float]] = {}
    dividends_by_ccy: dict[str, float] = {}
    fx_impact = 0.0
    holding_days: list[float] = []
    sells_total = 0
    win_total = 0
    loss_total = 0

    for acc in (portfolio.isa, portfolio.invest):
        fx_impact += float(getattr(acc, "total_fx_impact", 0.0) or 0.0)
        fills_by_ticker: dict[str, list[Any]] = {}
        for order in getattr(acc, "recent_orders", []) or []:
            ticker = str(getattr(order, "ticker", "") or "")
            if ticker:
                fills_by_ticker.setdefault(ticker, []).append(order)

        for orders in fills_by_ticker.values():
            orders.sort(key=lambda o: int(getattr(o, "filled_at_ms", 0) or 0))
            first_buy_ms: int | None = None
            for order in orders:
                side = str(getattr(order, "side", "") or "").upper()
                stamped = int(getattr(order, "filled_at_ms", 0) or 0)
                if side == "BUY":
                    if first_buy_ms is None or (stamped and stamped < first_buy_ms):
                        first_buy_ms = stamped or first_buy_ms
                    continue
                if side != "SELL":
                    continue
                ccy = str(getattr(order, "wallet_currency", "GBP") or "GBP").strip().upper() or "GBP"
                slot = per_ccy.setdefault(
                    ccy,
                    {"sells": 0.0, "wins": 0.0, "losses": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
                     "realised": 0.0, "taxes": 0.0},
                )
                pnl = float(getattr(order, "realised_pnl", 0.0) or 0.0)
                slot["sells"] += 1.0
                slot["realised"] += pnl
                slot["taxes"] += float(getattr(order, "tax_total", 0.0) or 0.0)
                sells_total += 1
                if pnl > 0.01:
                    slot["wins"] += 1.0
                    slot["gross_win"] += pnl
                    win_total += 1
                elif pnl < -0.01:
                    slot["losses"] += 1.0
                    slot["gross_loss"] += abs(pnl)
                    loss_total += 1
                if first_buy_ms is not None and stamped > first_buy_ms:
                    holding_days.append((stamped - first_buy_ms) / 86_400_000.0)

        for dividend in getattr(acc, "recent_dividends", []) or []:
            ccy = str(getattr(dividend, "currency", "GBP") or "GBP").strip().upper() or "GBP"
            dividends_by_ccy[ccy] = dividends_by_ccy.get(ccy, 0.0) + float(
                getattr(dividend, "amount", 0.0) or 0.0
            )

    detail: dict[str, dict[str, Any]] = {}
    for ccy, slot in per_ccy.items():
        sells = int(slot["sells"])
        gross_loss = float(slot["gross_loss"])
        detail[ccy] = {
            "sells": sells,
            "wins": int(slot["wins"]),
            "losses": int(slot["losses"]),
            "win_rate_pct": round(slot["wins"] / sells * 100.0, 1) if sells else 0.0,
            "profit_factor": round(slot["gross_win"] / gross_loss, 2) if gross_loss > 1e-9 else None,
            "expectancy": round(slot["realised"] / sells, 2) if sells else 0.0,
            "realised": round(slot["realised"], 2),
            "taxes": round(slot["taxes"], 2),
            "avg_win": round(slot["gross_win"] / slot["wins"], 2) if slot["wins"] else None,
            "avg_loss": round(-gross_loss / slot["losses"], 2) if slot["losses"] else None,
        }

    return {
        "ok": sells_total > 0,
        "sells": sells_total,
        "wins": win_total,
        "losses": loss_total,
        "win_rate_pct": round(win_total / sells_total * 100.0, 1) if sells_total else 0.0,
        "by_currency": detail,
        "dividends_by_currency": {k: round(v, 2) for k, v in dividends_by_ccy.items() if abs(v) >= 0.005},
        "fx_impact_gbp": round(fx_impact, 2),
        "avg_holding_days": round(float(np.mean(holding_days)), 1) if holding_days else None,
        "holding_sample": len(holding_days),
    }


def compute_signal_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """T+N Alpha t-statistics and decay slope over settled recommendations.

    ``t = mean / (std / sqrt(n))``; ``significant`` only at |t| >= 2 with n >= 3.
    Small samples are never hidden: ``sample`` is always returned for honest labels.
    """
    horizons = (("t5", "alpha_pct"), ("t10", "t10_alpha_pct"), ("t20", "t20_alpha_pct"))
    stats: dict[str, dict[str, Any]] = {}
    for key, field in horizons:
        values: list[float] = []
        for row in rows:
            raw = row.get(field)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                values.append(val)
        count = len(values)
        if count == 0:
            stats[key] = {"n": 0, "mean_pct": None, "std_pct": None, "t_stat": None, "significant": False}
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if count >= 2 else 0.0
        t_stat: float | None = None
        if count >= 3 and std > 1e-12:
            t_stat = mean / (std / math.sqrt(count))
        stats[key] = {
            "n": count,
            "mean_pct": round(mean, 2),
            "std_pct": round(std, 2),
            "t_stat": round(t_stat, 2) if t_stat is not None else None,
            "significant": bool(t_stat is not None and abs(t_stat) >= 2.0),
        }

    decay: float | None = None
    if stats["t20"]["mean_pct"] is not None and stats["t5"]["mean_pct"] is not None:
        decay = round(float(stats["t20"]["mean_pct"]) - float(stats["t5"]["mean_pct"]), 2)
    return {
        "ok": stats["t5"]["n"] > 0,
        "sample": stats["t5"]["n"],
        "horizons": stats,
        "decay_t20_minus_t5_pct": decay,
    }


def compute_concentration_metrics(portfolio: PortfolioData) -> dict[str, Any]:
    """HHI / effective holdings / top-N weights computed on invested market value."""
    values: dict[str, float] = {}
    for pos in portfolio.all_positions():
        symbol = str(getattr(pos, "yahoo_symbol", "") or getattr(pos, "ticker", "") or "").strip().upper()
        value = float(getattr(pos, "current_value", 0.0) or 0.0)
        if not symbol or value <= 0.0:
            continue
        values[symbol] = values.get(symbol, 0.0) + value

    if not values:
        return {"ok": False, "positions": 0}

    invested = float(sum(values.values()))
    total = float(getattr(portfolio, "total_combined_value", 0.0) or 0.0) or invested
    weights = sorted((v / invested for v in values.values()), reverse=True)
    hhi = float(sum(w * w for w in weights))
    return {
        "ok": True,
        "positions": len(values),
        "invested_gbp": round(invested, 2),
        "cash_pct": round(max(total - invested, 0.0) / total * 100.0, 1) if total > 0 else None,
        "hhi": round(hhi, 4),
        "effective_n": round(1.0 / hhi, 2) if hhi > 0 else None,
        "top1_pct": round(weights[0] * 100.0, 1),
        "top3_pct": round(sum(weights[:3]) * 100.0, 1),
    }


# ---------------------------------------------------------------------------
# v5.5.1: core-policy benchmark (regional-tilt Beta vs stock-picking Alpha)
# ---------------------------------------------------------------------------
_CORE_POLICY_BENCHMARK_WEIGHTS: dict[str, float] = {
    "VWRP.L": 70.0,
    "VERG.L": 13.3,
    "ISF.L": 10.0,
    "VUAG.L": 6.7,
}


def compute_core_policy_benchmark_comparison(
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 60,
) -> dict[str, Any] | None:
    """Compare the four-bucket core policy mix against the default VWRP.L benchmark.

    Returns ``allocation_effect_pct`` (policy mix return minus VWRP.L return over the
    same window) and annualised tracking error, so the weekly / /perf review can tell
    a deliberate UK/EU regional tilt (Beta) apart from genuine stock-picking Alpha.
    """
    if not isinstance(histories, dict) or not histories:
        return None
    bench_sym = "VWRP.L"
    bench_hist = histories.get(bench_sym) if bench_sym in histories else histories.get("VWRP")
    bench_closes = _extract_close_series(bench_hist, int(lookback_days) + 5)
    if bench_closes is None or len(bench_closes) < 2:
        return None

    window = max(2, min(int(lookback_days), len(bench_closes) - 1))
    bench_ret = float(bench_closes.iloc[-1] / float(bench_closes.iloc[-1 - window]) - 1.0)

    returns_by_sym: dict[str, pd.Series] = {}
    used_weights: dict[str, float] = {}
    missing: list[str] = []
    for sym, weight in _CORE_POLICY_BENCHMARK_WEIGHTS.items():
        hist = histories.get(sym) if sym in histories else histories.get(sym.split(".")[0])
        closes = _extract_close_series(hist, window + 5)
        if closes is None or len(closes) < window + 1:
            if sym != bench_sym:
                missing.append(sym)
            continue
        rets = closes.pct_change().dropna().iloc[-window:]
        if rets.empty:
            missing.append(sym)
            continue
        returns_by_sym[sym] = rets.reset_index(drop=True)
        used_weights[sym] = float(weight)

    total_w = float(sum(used_weights.values()))
    if total_w <= 0 or bench_sym not in used_weights:
        return None
    norm = {sym: weight / total_w for sym, weight in used_weights.items()}

    min_len = min(len(series) for series in returns_by_sym.values())
    if min_len < 2:
        return None
    aligned = {sym: series.iloc[-min_len:].reset_index(drop=True) for sym, series in returns_by_sym.items()}
    policy_daily = sum(aligned[sym] * w for sym, w in norm.items())
    bench_daily = aligned[bench_sym]
    diff = policy_daily - bench_daily
    policy_ret = float((1.0 + policy_daily).prod() - 1.0)
    tracking_error = float(diff.std(ddof=1) * math.sqrt(252) * 100.0) if len(diff) > 1 else None
    return {
        "lookback_days": int(min_len),
        "weights_used": {sym: round(w * 100.0, 2) for sym, w in norm.items()},
        "missing_symbols": missing,
        "policy_return_pct": round(policy_ret * 100.0, 2),
        "benchmark_return_pct": round(bench_ret * 100.0, 2),
        "allocation_effect_pct": round((policy_ret - bench_ret) * 100.0, 2),
        "tracking_error_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "benchmark_symbol": bench_sym,
    }


_REGIME_BUCKET_ORDER: tuple[str, ...] = ("RISK_ON", "NEUTRAL", "RECOVERY", "RISK_OFF")


def compute_regime_attribution(
    nav_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """v5.5.9 M6: descriptive per-regime attribution over the persisted daily NAV series.

    Each day's label comes from the morning report's context pack (the regime that
    actually governed that day's positioning — never a hindsight reconstruction).
    Daily TWR uses the exact same flow-adjusted formula as
    ``compute_portfolio_performance``; a regime bucket's compounded return is the
    product of its own daily returns, so the buckets multiply back into the whole
    window's TWR. Descriptive only — no significance claims, no annualisation.
    Days without a recorded regime stay visible as ``unattributed_days``.
    """
    empty = {"ok": False, "buckets": {}, "unattributed_days": 0, "attributed_days": 0}
    if not nav_rows:
        return empty

    regime_by_date: dict[str, str] = {}
    for row in regime_rows or []:
        day = str(row.get("uk_date") or "").strip()[:10]
        label = str(row.get("regime") or "").strip().upper()
        if len(day) == 10 and label:
            regime_by_date[day] = label

    records: list[dict[str, Any]] = []
    for row in nav_rows:
        raw_date = str(row.get("uk_date") or "").strip()[:10]
        if len(raw_date) < 10:
            continue
        try:
            nav = float(row.get("total_value_gbp"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(nav) or nav <= 0.0:
            continue
        try:
            flow = float(row.get("net_flow_gbp") or 0.0)
        except (TypeError, ValueError):
            flow = 0.0
        try:
            cash = max(float(row.get("cash_gbp") or 0.0), 0.0)
        except (TypeError, ValueError):
            cash = 0.0
        records.append({"date": raw_date, "nav": nav, "flow": flow, "cash": cash})

    if len(records) < 2:
        return empty

    # Drop the first record for return math parity with compute_portfolio_performance.
    daily: list[dict[str, Any]] = []
    for prev, cur in zip(records, records[1:]):
        if prev["nav"] <= 0:
            continue
        r = (cur["nav"] - cur["flow"]) / prev["nav"] - 1.0
        label = regime_by_date.get(cur["date"], "")
        daily.append(
            {
                "date": cur["date"],
                "ret": r,
                "cash_pct": (cur["cash"] / cur["nav"]) * 100.0,
                "regime": label,
            }
        )

    buckets: dict[str, dict[str, Any]] = {}
    for label in _REGIME_BUCKET_ORDER:
        rows = [d for d in daily if d["regime"] == label]
        if not rows:
            continue
        twr = float(np.prod([1.0 + d["ret"] for d in rows]) - 1.0)
        avg_cash = float(np.mean([d["cash_pct"] for d in rows]))
        buckets[label] = {
            "days": len(rows),
            "twr_pct": round(twr * 100.0, 2),
            "avg_cash_pct": round(avg_cash, 1),
            "avg_position_pct": round(100.0 - avg_cash, 1),
        }

    attributed = [d for d in daily if d["regime"]]
    if len(attributed) < 2:
        return {
            "ok": False,
            "buckets": buckets,
            "unattributed_days": len(daily) - len(attributed),
            "attributed_days": len(attributed),
        }
    return {
        "ok": True,
        "buckets": buckets,
        "unattributed_days": len(daily) - len(attributed),
        "attributed_days": len(attributed),
    }
