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


def _backtest_entry_series(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
) -> dict[str, Any] | None:
    """Shared PIT entry-series precompute for both offline backtests (M6).

    Identical rolling windows (MA50/MA200/ATR14/prior 20d high/MA20) so the
    baseline threshold backtest and the v5.5.9 exit-rule A/B can never drift
    onto different entry filters — the A/B delta must stay exit-side only.
    """
    n_bars = len(closes)
    if n_bars < 90:
        return None
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
    ma20 = closes.rolling(20, min_periods=20).mean()
    return {
        "n_bars": n_bars,
        "ma50": ma50,
        "ma200": ma200,
        "use_ma200": use_ma200,
        "atr14": atr14,
        "prior_high_20": prior_high_20,
        "ma20": ma20,
        "start_idx": 200 if use_ma200 else 50,
    }


def _breakout_entry_ok(
    entry: float,
    ctx: dict[str, Any],
    i: int,
) -> bool:
    """Breakout entry predicate shared by both backtest variants."""
    ma50_i = ctx["ma50"].iloc[i]
    ph20_i = ctx["prior_high_20"].iloc[i]
    if pd.isna(ma50_i) or pd.isna(ph20_i) or entry <= 0:
        return False
    if ctx["use_ma200"] and ctx["ma200"] is not None:
        ma200_i = ctx["ma200"].iloc[i]
        if pd.isna(ma200_i) or float(ma50_i) <= float(ma200_i):
            return False
    return bool(entry > float(ph20_i) and entry > float(ma50_i))


def _atr_stop_pct_or_fallback(
    ctx: dict[str, Any],
    i: int,
    entry: float,
    fallback_stop_pct: float,
) -> float:
    """Adaptive ATR stop (noise floor band [-18%, -8%]), identical in both variants."""
    atr_i = ctx["atr14"].iloc[i]
    if pd.notna(atr_i) and float(atr_i) > 0:
        atr_pct = (float(atr_i) / entry) * 100.0
        return max(-18.0, min(-8.0, -2.8 * atr_pct))
    return fallback_stop_pct


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
        ctx = _backtest_entry_series(highs, lows, closes)
        if ctx is None:
            continue

        i = ctx["start_idx"]
        while i < ctx["n_bars"] - 10:
            entry = float(closes.iloc[i])
            if not _breakout_entry_ok(entry, ctx, i):
                i += 1
                continue

            stop_pct = _atr_stop_pct_or_fallback(ctx, i, entry, fallback_stop_pct)

            peak_price = entry
            trailing_active = False
            exit_ret: float | None = None
            exit_reason = "HORIZON_END"
            end_j = min(ctx["n_bars"] - 1, i + forward_bars)

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


_AB_SPREAD_PCT_PER_SIDE: float = 0.05


def _ab_round_trip_friction_pct(sym: str, etf_syms: set[str], strategy: StrategySection) -> float:
    """Approximate round-trip friction per symbol market for the offline A/B.

    Reads the live config (fx_fee_pct / uk_stamp_duty_pct) so the offline model
    can never silently diverge from the platform's real cost basis; only the
    spread (0.05%/side for liquid names) is an explicit offline approximation.
    """
    fx = float(getattr(strategy, "fx_fee_pct", 0.15) or 0.15)
    stamp = float(getattr(strategy, "uk_stamp_duty_pct", 0.50) or 0.50)
    spread_rt = _AB_SPREAD_PCT_PER_SIDE * 2.0
    sym_up = str(sym or "").strip().upper()
    if sym_up.endswith(".L"):
        return spread_rt if sym_up in etf_syms else stamp + spread_rt
    return fx * 2.0 + spread_rt


def _summarise_net_returns(returns: list[float]) -> dict[str, Any]:
    """Win rate / payoff / avg / worst-single-trade over net (friction-deducted) returns."""
    sample_count = len(returns)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": None,
            "payoff_ratio": None,
            "max_single_loss_pct": None,
        }
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    payoff: float | None = None
    if wins and losses:
        avg_loss_abs = abs(float(np.mean(losses)))
        if avg_loss_abs > 1e-9:
            payoff = round(float(np.mean(wins)) / avg_loss_abs, 2)
    return {
        "sample_count": sample_count,
        "win_rate_pct": round((len(wins) / sample_count) * 100.0, 1),
        "avg_return_pct": round(float(np.mean(returns)), 2),
        "payoff_ratio": payoff,
        "max_single_loss_pct": round(float(min(returns)), 2),
    }


def _backtest_exit_rule_ab(
    ohlcv_map: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]],
    candidate_syms: list[str],
    strategy: StrategySection,
    *,
    etf_syms: set[str] | None = None,
    forward_bars: int = 60,
    cooldown_bars: int = 20,
) -> dict[str, Any]:
    """Paired offline A/B: v5.5.8-style trailing exit vs the v5.5.9 M4 ladder exit.

    Both variants share the exact same breakout entries (same scan, same cooldown)
    and the same round-trip friction deduction, so the delta isolates the exit rule
    alone. The v5.5.9 arm is a deliberately conservative simplification of the live
    advisory stack: +breakeven floor after ``invest_ladder_breakeven_advisory_pct``
    (+15%), then ``max(cost x (1 + lock floor), MA20 close-trail)`` once the peak
    crosses ``invest_take_profit_review_pct`` (+25%). Dual-anchor targets, RECOVERY
    0.5R scaling and regime gating are intentionally NOT simulated (no historical
    VIX/HY breadth locally — fabricating them would be hindsight bias).
    """
    etf_set = {s.strip().upper() for s in (etf_syms or set())}
    tp_act_pct = float(strategy.trailing_profit_activation_pct)
    tp_dd_pct = float(strategy.trailing_drawdown_from_peak_pct)
    fallback_stop_pct = float(strategy.invest_stop_loss_alert_pct)
    breakeven_act_pct = float(
        getattr(strategy, "invest_ladder_breakeven_advisory_pct", 15.0) or 15.0
    )
    lock_floor_pct = float(getattr(strategy, "invest_ladder_lock_floor_pct", 10.0) or 10.0)
    ma20_trail_act_pct = float(
        getattr(strategy, "invest_take_profit_review_pct", 25.0) or 25.0
    )

    baseline_rets: list[float] = []
    ladder_rets: list[float] = []
    exit_mix = {
        "baseline": {"trailing_tp": 0, "atr_stop": 0, "horizon": 0},
        "v559_ladder": {"ladder_floor": 0, "ma20_trail": 0, "atr_stop": 0, "horizon": 0},
    }

    for sym in candidate_syms:
        ext = ohlcv_map.get(sym)
        if ext is None:
            continue
        _, highs, lows, closes, _ = ext
        ctx = _backtest_entry_series(highs, lows, closes)
        if ctx is None:
            continue
        friction_pct = _ab_round_trip_friction_pct(sym, etf_set, strategy)

        i = ctx["start_idx"]
        while i < ctx["n_bars"] - 10:
            entry = float(closes.iloc[i])
            if not _breakout_entry_ok(entry, ctx, i):
                i += 1
                continue

            stop_pct = _atr_stop_pct_or_fallback(ctx, i, entry, fallback_stop_pct)
            end_j = min(ctx["n_bars"] - 1, i + forward_bars)

            # --- baseline arm: pure trailing TP (+act / -dd) + ATR stop ---
            peak_price = entry
            trailing_active = False
            exit_ret: float | None = None
            exit_reason = "horizon"
            for j in range(i + 1, end_j + 1):
                low_j = float(lows.iloc[j])
                high_j = float(highs.iloc[j])
                close_j = float(closes.iloc[j])
                low_ret = ((low_j / entry) - 1.0) * 100.0
                if not trailing_active and low_ret <= stop_pct:
                    exit_ret = stop_pct
                    exit_reason = "atr_stop"
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
                        exit_reason = "trailing_tp"
                        break
            if exit_ret is None:
                exit_ret = ((float(closes.iloc[end_j]) / entry) - 1.0) * 100.0
            baseline_rets.append(exit_ret - friction_pct)
            exit_mix["baseline"][exit_reason] += 1

            # --- v5.5.9 ladder arm: ATR stop -> +15% breakeven floor -> +25% lock/MA20 ---
            # friction-adjusted breakeven: exit level = entry x (1 + round-trip friction),
            # exactly the ledgers.py financed-unlock convention.
            breakeven_floor_pct = friction_pct
            floor_armed = False
            lock_armed = False
            peak_price = entry
            exit_ret = None
            exit_reason = "horizon"
            for j in range(i + 1, end_j + 1):
                low_j = float(lows.iloc[j])
                high_j = float(highs.iloc[j])
                close_j = float(closes.iloc[j])

                stop_level_pct = stop_pct
                if floor_armed:
                    stop_level_pct = max(stop_pct, breakeven_floor_pct)
                if lock_armed:
                    stop_level_pct = max(stop_level_pct, lock_floor_pct)
                low_ret = ((low_j / entry) - 1.0) * 100.0
                if low_ret <= stop_level_pct:
                    exit_ret = stop_level_pct
                    exit_reason = "ladder_floor" if floor_armed else "atr_stop"
                    break

                if high_j > peak_price:
                    peak_price = high_j
                peak_ret = ((peak_price / entry) - 1.0) * 100.0
                if not floor_armed and peak_ret >= breakeven_act_pct:
                    floor_armed = True
                if not lock_armed and peak_ret >= ma20_trail_act_pct:
                    lock_armed = True

                if lock_armed:
                    ma20_j = ctx["ma20"].iloc[j]
                    if pd.notna(ma20_j):
                        trail_level = float(max(entry * (1.0 + lock_floor_pct / 100.0), float(ma20_j)))
                        if close_j <= trail_level:
                            exit_ret = ((close_j / entry) - 1.0) * 100.0
                            exit_reason = "ma20_trail"
                            break

            if exit_ret is None:
                exit_ret = ((float(closes.iloc[end_j]) / entry) - 1.0) * 100.0
            ladder_rets.append(exit_ret - friction_pct)
            exit_mix["v559_ladder"][exit_reason] += 1

            i += cooldown_bars

    baseline_stats = _summarise_net_returns(baseline_rets)
    ladder_stats = _summarise_net_returns(ladder_rets)
    delta: dict[str, Any] = {}
    for key in ("win_rate_pct", "avg_return_pct", "payoff_ratio", "max_single_loss_pct"):
        b_val = baseline_stats.get(key)
        v_val = ladder_stats.get(key)
        if b_val is None or v_val is None:
            delta[key] = None
        else:
            delta[key] = round(float(v_val) - float(b_val), 2)

    return {
        "sample_count": baseline_stats["sample_count"],
        "baseline": {**baseline_stats, "exit_mix": exit_mix["baseline"]},
        "v559_ladder": {**ladder_stats, "exit_mix": exit_mix["v559_ladder"]},
        "delta": delta,
        "friction_model": {
            "fx_fee_pct": float(getattr(strategy, "fx_fee_pct", 0.15) or 0.15),
            "uk_stamp_duty_pct": float(getattr(strategy, "uk_stamp_duty_pct", 0.50) or 0.50),
            "spread_pct_per_side": _AB_SPREAD_PCT_PER_SIDE,
            "note": "往返摩擦按标的币种/印花税近似扣除，两臂同表同扣，delta 只反映退出侧",
        },
        "survivorship_note": (
            "样本取自当前静态现役底池回溯2年，已退市/暴雷/被剔除标的不在样本内，"
            "整体收益倾向被高估，结论只可用于规则间相对比较"
        ),
        "simplification_note": (
            "v5.5.9臂仅模拟+15%保本地板与+25%后max(锁盈地板,MA20收盘跟踪)的简化阶梯，"
            "未含双锚目标价/RECOVERY缩放/形态门控，属保守近似"
        ),
        "small_sample": baseline_stats["sample_count"] < 30,
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
    # v5.5.9 M6: paired exit-rule A/B on the same local 2y pool (same entries,
    # same friction — delta is exit-side only). Purely observational: nothing
    # here tunes config or writes back into the screener.
    etf_set = {str(s or "").strip().upper() for s in universe.etf_symbols if str(s or "").strip()}
    exit_rule_ab = _backtest_exit_rule_ab(ohlcv_map, universe_syms, strategy, etf_syms=etf_set)

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
    ab_n = int(exit_rule_ab.get("sample_count") or 0)
    if ab_n > 0:
        ab_base = exit_rule_ab["baseline"]
        ab_ladder = exit_rule_ab["v559_ladder"]

        def _ab_val(item: dict[str, Any], key: str) -> str:
            raw = item.get(key)
            if raw is None:
                return "N/A"
            return f"{float(raw):+.2f}%" if key == "avg_return_pct" else f"{float(raw):.1f}%"

        small_note = "（n<30 仅参考，不构成统计证据）" if exit_rule_ab.get("small_sample") else ""
        ab_line = (
            f"• 退出规则A/B(基线移动止盈 vs v5.5.9阶梯保本+MA20, 同批入场配对净扣摩擦): {ab_n} 样本 | "
            f"基线 胜率 {_ab_val(ab_base, 'win_rate_pct')} 均收益 {_ab_val(ab_base, 'avg_return_pct')} "
            f"最大单亏 {_ab_val(ab_base, 'max_single_loss_pct')} vs "
            f"阶梯 胜率 {_ab_val(ab_ladder, 'win_rate_pct')} 均收益 {_ab_val(ab_ladder, 'avg_return_pct')} "
            f"最大单亏 {_ab_val(ab_ladder, 'max_single_loss_pct')}{small_note}"
            "（幸存者偏差: 静态现役底池回溯, 仅作规则间相对比较）"
        )
    else:
        ab_line = "• 退出规则A/B: 本地样本不足, 暂无对比"
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
        ab_line,
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
        "exit_rule_ab": exit_rule_ab,
        "fundamentals_coverage": fundamentals_coverage,
        "summary_text": "\n".join(summary_lines),
    }
