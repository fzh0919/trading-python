"""Offline universe overlap/zombie diagnostics, breakout threshold backtest, and fundamentals coverage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import _is_cross_timezone_pair
from src.data_providers.t212_client import PortfolioData
from src.engine.screener.patterns import _extract_ohlcv_series

def _lookup_history_df(histories: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame | None:
    df = histories.get(symbol)
    if df is not None and not df.empty:
        return df
    return histories.get(symbol.upper())


def _audit_rolling_overlap_candidates(
    close_map: dict[str, pd.Series],
    candidate_syms: list[str],
    held_syms: list[str],
    threshold: float,
    roll_window: int = 60,
    eval_days: int = 120,
) -> list[dict[str, Any]]:
    """Identify non-held candidates whose rolling correlation with any held position stays >= threshold."""
    if not held_syms or not candidate_syms:
        return []
    held_set = {s.upper() for s in held_syms}
    eff_roll = max(20, min(60, int(roll_window or 60)))
    min_periods = max(20, eff_roll // 2)
    flagged: list[dict[str, Any]] = []

    for cand in candidate_syms:
        if cand.upper() in held_set or cand not in close_map:
            continue
        c_ser = close_map[cand]
        best_match: dict[str, Any] | None = None
        for held in held_syms:
            if held not in close_map or held.upper() == cand.upper():
                continue
            pair_px = pd.concat(
                [c_ser.rename("cand"), close_map[held].rename("held")],
                axis=1,
            ).sort_index().ffill(limit=1)
            step = 3 if _is_cross_timezone_pair(cand, held) else 1
            rets = np.log(pair_px / pair_px.shift(step)).dropna()
            if len(rets) < eff_roll + 15:
                continue
            roll_corr = rets["cand"].rolling(eff_roll, min_periods=min_periods).corr(rets["held"]).dropna()
            tail_corr = roll_corr.tail(eval_days)
            if len(tail_corr) < 15:
                continue
            latest_val = float(tail_corr.iloc[-1])
            median_val = float(tail_corr.median())
            persist_ratio = float((tail_corr >= threshold).mean())
            if latest_val >= threshold and persist_ratio >= 0.80:
                if best_match is None or median_val > best_match["corr_median_120d"]:
                    best_match = {
                        "symbol": cand,
                        "held_symbol": held,
                        "corr_latest": round(latest_val, 3),
                        "corr_median_120d": round(median_val, 3),
                        "persist_ratio": round(persist_ratio, 3),
                    }
        if best_match is not None:
            flagged.append(best_match)

    flagged.sort(key=lambda x: (-x["corr_median_120d"], x["symbol"]))
    return flagged


def _audit_zombie_candidates(
    close_map: dict[str, pd.Series],
    candidate_syms: list[str],
    min_streak_days: int = 60,
) -> list[dict[str, Any]]:
    """Identify candidates stuck in Phase 4 (Close < MA200 and MA50 < MA200) for >= min_streak_days."""
    zombies: list[dict[str, Any]] = []
    for sym in candidate_syms:
        closes = close_map.get(sym)
        if closes is None or len(closes) < 200 + min_streak_days - 1:
            continue
        ma50 = closes.rolling(50, min_periods=50).mean()
        ma200 = closes.rolling(200, min_periods=200).mean()
        cond = ((closes < ma200) & (ma50 < ma200)).fillna(False).to_numpy(dtype=bool)
        streak = 0
        for flag in reversed(cond):
            if flag:
                streak += 1
            else:
                break
        if streak >= min_streak_days:
            zombies.append(
                {
                    "symbol": sym,
                    "streak_days": streak,
                    "close": round(float(closes.iloc[-1]), 2),
                    "ma50": round(float(ma50.iloc[-1]), 2),
                    "ma200": round(float(ma200.iloc[-1]), 2),
                }
            )
    zombies.sort(key=lambda x: (-x["streak_days"], x["symbol"]))
    return zombies


def _backtest_breakout_thresholds(
    ohlcv_map: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]],
    candidate_syms: list[str],
    strategy: StrategySection,
    *,
    forward_bars: int = 60,
    cooldown_bars: int = 20,
) -> dict[str, Any]:
    """Run out-of-sample breakout simulation testing trailing TP (+20%/-12%) and adaptive ATR stop."""
    tp_act_pct = float(strategy.trailing_profit_activation_pct)
    tp_dd_pct = float(strategy.trailing_drawdown_from_peak_pct)
    fallback_stop_pct = float(strategy.invest_stop_loss_alert_pct)

    returns: list[float] = []
    tp_triggers = 0
    tp_wins = 0
    sl_triggers = 0
    horizon_exits = 0

    for sym in candidate_syms:
        ext = ohlcv_map.get(sym)
        if ext is None:
            continue
        _, highs, lows, closes, _ = ext
        n_bars = len(closes)
        if n_bars < 90:
            continue

        ma50 = closes.rolling(50, min_periods=50).mean()
        use_ma200 = n_bars >= 220
        ma200 = closes.rolling(200, min_periods=200).mean() if use_ma200 else None
        prev_c = closes.shift(1)
        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_c).abs(),
                (lows - prev_c).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()
        prior_high_20 = highs.shift(1).rolling(20, min_periods=20).max()

        start_idx = 200 if use_ma200 else 50
        i = start_idx
        while i < n_bars - 10:
            entry = float(closes.iloc[i])
            ma50_i = ma50.iloc[i]
            ph20_i = prior_high_20.iloc[i]
            if pd.isna(ma50_i) or pd.isna(ph20_i) or entry <= 0:
                i += 1
                continue
            if use_ma200 and ma200 is not None:
                ma200_i = ma200.iloc[i]
                if pd.isna(ma200_i) or float(ma50_i) <= float(ma200_i):
                    i += 1
                    continue
            if not (entry > float(ph20_i) and entry > float(ma50_i)):
                i += 1
                continue

            atr_i = atr14.iloc[i]
            if pd.notna(atr_i) and float(atr_i) > 0:
                atr_pct = (float(atr_i) / entry) * 100.0
                stop_pct = max(-18.0, min(-8.0, -2.8 * atr_pct))
            else:
                stop_pct = fallback_stop_pct

            peak_price = entry
            trailing_active = False
            exit_ret: float | None = None
            exit_reason = "HORIZON_END"
            end_j = min(n_bars - 1, i + forward_bars)

            for j in range(i + 1, end_j + 1):
                low_j = float(lows.iloc[j])
                high_j = float(highs.iloc[j])
                close_j = float(closes.iloc[j])

                low_ret = ((low_j / entry) - 1.0) * 100.0
                if not trailing_active and low_ret <= stop_pct:
                    exit_ret = stop_pct
                    exit_reason = "ATR_STOP"
                    break

                if high_j > peak_price:
                    peak_price = high_j
                peak_ret = ((peak_price / entry) - 1.0) * 100.0
                if peak_ret >= tp_act_pct:
                    trailing_active = True

                if trailing_active:
                    dd_from_peak = ((close_j / peak_price) - 1.0) * 100.0
                    if dd_from_peak <= tp_dd_pct:
                        exit_ret = ((close_j / entry) - 1.0) * 100.0
                        exit_reason = "TRAILING_TP"
                        break

            if exit_ret is None:
                exit_ret = ((float(closes.iloc[end_j]) / entry) - 1.0) * 100.0

            returns.append(exit_ret)
            if exit_reason == "TRAILING_TP":
                tp_triggers += 1
                if exit_ret > 0:
                    tp_wins += 1
            elif exit_reason == "ATR_STOP":
                sl_triggers += 1
            else:
                horizon_exits += 1

            i += cooldown_bars

    sample_count = len(returns)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "win_count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": None,
            "payoff_ratio": None,
            "trailing_tp_triggers": 0,
            "trailing_tp_win_rate_pct": 0.0,
            "atr_stop_triggers": 0,
            "horizon_exits": 0,
        }

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = round((len(wins) / sample_count) * 100.0, 1)
    avg_ret = round(float(np.mean(returns)), 2)
    payoff: float | None = None
    if wins and losses:
        avg_loss_abs = abs(float(np.mean(losses)))
        if avg_loss_abs > 1e-9:
            payoff = round(float(np.mean(wins)) / avg_loss_abs, 2)

    return {
        "sample_count": sample_count,
        "win_count": len(wins),
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "payoff_ratio": payoff,
        "trailing_tp_triggers": tp_triggers,
        "trailing_tp_win_rate_pct": round((tp_wins / tp_triggers) * 100.0, 1) if tp_triggers > 0 else 0.0,
        "atr_stop_triggers": sl_triggers,
        "horizon_exits": horizon_exits,
    }


_FUNDAMENTAL_COVERAGE_FIELDS: tuple[str, ...] = (
    "roe_pct",
    "debt_to_equity",
    "revenue_growth_pct",
    "fcf_yield_pct",
    "peg_ratio",
    "target_price",
)


def compute_fundamentals_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-market ISA value-factor coverage audit over cached fundamentals (non-ETF only).

    A field counts as covered only when it is a finite number (0 is a valid value).
    Any field below 50% coverage in a market is surfaced as a risk warning so the
    value screen cannot quietly degrade into a momentum screen on missing data.
    """
    markets: dict[str, list[dict[str, Any]]] = {"US": [], "UK": [], "EU": []}

    def _market_of(sym: str) -> str:
        s = sym.strip().upper()
        if s.endswith(".L"):
            return "UK"
        if s.endswith((".AS", ".DE", ".PA")):
            return "EU"
        return "US"

    def _has(field: str, row: dict[str, Any]) -> bool:
        if field == "target_price":
            raw_values = (row.get("target_mean_price"), row.get("target_high_price"))
        else:
            raw_values = (row.get(field),)
        for raw in raw_values:
            if raw is None or raw == "":
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return True
        return False

    skipped_etf = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        if str(row.get("quote_type") or "").strip().upper() == "ETF":
            skipped_etf += 1
            continue
        markets[_market_of(sym)].append(row)

    out: dict[str, Any] = {
        "ok": True,
        "non_etf_count": 0,
        "skipped_etf": skipped_etf,
        "markets": {},
        "low_coverage": [],
    }
    for market, items in markets.items():
        count = len(items)
        out["non_etf_count"] += count
        coverage: dict[str, float | None] = {}
        for field in _FUNDAMENTAL_COVERAGE_FIELDS:
            covered = sum(1 for row in items if _has(field, row))
            coverage[field] = round(covered / count * 100.0, 1) if count > 0 else None
        low = [
            f"{market} {field} {pct:.0f}%"
            for field, pct in coverage.items()
            if pct is not None and pct < 50.0
        ]
        out["markets"][market] = {"symbols": count, "coverage_pct": coverage}
        out["low_coverage"].extend(low)

    return out


def run_offline_universe_and_threshold_audit(
    histories: dict[str, pd.DataFrame],
    portfolio: PortfolioData,
    universe: UniverseSection,
    strategy: StrategySection,
    *,
    active_pool_rows: list[dict[str, Any]] | None = None,
    fundamentals_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run 0-token offline universe redundancy, Phase 4 zombie, and TP/SL threshold backtest audit."""
    if active_pool_rows:
        pool_raw = [str(r.get("symbol") or "") for r in active_pool_rows] + list(universe.etf_symbols)
    else:
        pool_raw = (
            list(universe.isa_long_term)
            + list(universe.invest_tactical)
            + list(universe.etf_symbols)
        )
    universe_syms = list(
        dict.fromkeys(
            s.strip().upper()
            for s in pool_raw
            if str(s or "").strip()
        )
    )
    held_syms = list(
        dict.fromkeys(
            (p.yahoo_symbol or "").strip().upper()
            for p in portfolio.all_positions()
            if (p.yahoo_symbol or "").strip()
        )
    )

    ohlcv_map: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    close_map: dict[str, pd.Series] = {}
    for sym in dict.fromkeys(universe_syms + held_syms):
        ext = _extract_ohlcv_series(_lookup_history_df(histories, sym))
        if ext is not None:
            ohlcv_map[sym] = ext
            close_map[sym] = ext[3]

    overlap_candidates = _audit_rolling_overlap_candidates(
        close_map,
        universe_syms,
        held_syms,
        threshold=float(strategy.overlap_correlation_threshold),
        roll_window=int(strategy.correlation_window_days or 60),
        eval_days=120,
    )
    zombie_candidates = _audit_zombie_candidates(close_map, universe_syms, min_streak_days=60)
    backtest_stats = _backtest_breakout_thresholds(ohlcv_map, universe_syms, strategy)

    overlap_str = (
        ", ".join(f"{item['symbol']}(近{item['held_symbol']} {item['corr_median_120d']:.2f})" for item in overlap_candidates[:6])
        if overlap_candidates
        else "无"
    )
    if len(overlap_candidates) > 6:
        overlap_str += f" 等{len(overlap_candidates)}只"

    zombie_str = (
        ", ".join(f"{item['symbol']}({item['streak_days']}d)" for item in zombie_candidates[:6])
        if zombie_candidates
        else "无"
    )
    if len(zombie_candidates) > 6:
        zombie_str += f" 等{len(zombie_candidates)}只"

    payoff_str = f"{backtest_stats['payoff_ratio']:.2f}" if backtest_stats["payoff_ratio"] is not None else "N/A"
    avg_ret_str = (
        f"{backtest_stats['avg_return_pct']:+.2f}%"
        if backtest_stats["avg_return_pct"] is not None
        else "N/A"
    )
    summary_lines = [
        f"• 扫描池 {len(universe_syms)} 只候选 vs {len(held_syms)} 只持仓 (本地2y历史)",
        f"• 高重叠冗余(120d持续>={strategy.overlap_correlation_threshold:.2f}): {overlap_str}",
        f"• 长期弱势僵尸股(>=60d Phase4): {zombie_str}",
        (
            f"• 阈值回测(+{strategy.trailing_profit_activation_pct:.0f}%/{strategy.trailing_drawdown_from_peak_pct:.0f}%移动止盈 & ATR止损): "
            f"{backtest_stats['sample_count']} 样本 | 胜率 {backtest_stats['win_rate_pct']:.1f}% | "
            f"盈亏比 {payoff_str} | 均收益 {avg_ret_str} | "
            f"止盈 {backtest_stats['trailing_tp_triggers']} / 止损 {backtest_stats['atr_stop_triggers']}"
        ),
    ]

    fundamentals_coverage: dict[str, Any] | None = None
    if fundamentals_rows is not None:
        fundamentals_coverage = compute_fundamentals_coverage(fundamentals_rows)
        market_bits: list[str] = []
        for market, item in (fundamentals_coverage.get("markets") or {}).items():
            if not item.get("symbols"):
                continue
            cov = item.get("coverage_pct") or {}
            avg_cov = [v for v in cov.values() if v is not None]
            avg_s = f"{sum(avg_cov) / len(avg_cov):.0f}%" if avg_cov else "N/A"
            market_bits.append(f"{market} {item['symbols']}只均值 {avg_s}")
        if market_bits:
            summary_lines.append("• 价值因子覆盖率: " + " | ".join(market_bits))
        if fundamentals_coverage.get("low_coverage"):
            summary_lines.append(
                "• 覆盖率风险(<50%): " + "、".join(fundamentals_coverage["low_coverage"][:6])
            )

    return {
        "universe_count": len(universe_syms),
        "held_count": len(held_syms),
        "overlap_candidates": overlap_candidates,
        "zombie_candidates": zombie_candidates,
        "threshold_backtest": backtest_stats,
        "fundamentals_coverage": fundamentals_coverage,
        "summary_text": "\n".join(summary_lines),
    }
