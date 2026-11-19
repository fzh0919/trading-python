"""Cross-timezone de-beta correlation matrix, clone detection, and market benchmark resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_providers.analytics.exposure import infer_etf_index_family
from src.data_providers.analytics.technical import (
    _extract_close_series,
    _extract_daily_returns,
    _normalize_adjusted_ohlcv,
    _resolve_col,
)

# LSE-listed ETFs whose underlying index is US / Global rather than domestic UK FTSE
_US_OR_GLOBAL_LSE_ETFS: set[str] = {
    "VUAG.L",
    "CSP1.L",
    "VUSA.L",
    "IUSA.L",
    "SPXP.L",
    "GSPX.L",
    "CNX1.L",
    "EQGB.L",
    "EQQQ.L",
    "XNAQ.L",
    "VWRP.L",
    "VWRL.L",
    "SWDA.L",
    "IWDA.L",
    "SSAC.L",
    "FWRG.L",
    "IWQU.L",
}

_EU_LSE_ETFS: set[str] = {
    "VERG.L",
    "VEUR.L",
    "IMEU.L",
}

_EU_SUFFIXES: tuple[str, ...] = (".DE", ".AS", ".PA", ".MI", ".MC", ".BR", ".SW", ".HE", ".ST", ".CO", ".OL")


def compute_window_return_pct(hist: pd.DataFrame | None, days: int) -> float | None:
    """Close-to-close return over the last ``days`` trading bars (None when data is short).

    Callers map calendar windows to bars: 5 bars ≈ 7 days, 21 bars ≈ 30 days.
    """
    if hist is None or getattr(hist, "empty", True):
        return None
    try:
        bars = int(days)
    except (TypeError, ValueError):
        return None
    if bars < 1:
        return None
    closes = _extract_close_series(hist, bars + 10)
    if closes is None or len(closes) < bars + 1:
        return None
    start = float(closes.iloc[-(bars + 1)])
    end = float(closes.iloc[-1])
    if start <= 0 or not (np.isfinite(start) and np.isfinite(end)):
        return None
    return ((end - start) / start) * 100.0


def _market_timezone_zone(symbol: str) -> str:
    """Classify trading venue closing timezone ('EU_UK' closes ~16:30 UK vs 'US' closes ~21:00 UK)."""
    clean = (symbol or "").strip().upper()
    if clean.endswith(".L") or clean.endswith(_EU_SUFFIXES) or clean in ("^FTSE", "^STOXX50E"):
        return "EU_UK"
    return "US"


def _is_cross_timezone_pair(sym_a: str, sym_b: str) -> bool:
    """True when one symbol trades on UK/European hours and the other on US hours (~4.5h close lag)."""
    return _market_timezone_zone(sym_a) != _market_timezone_zone(sym_b)


def resolve_market_benchmark(
    symbol: str,
    default_benchmark: str = "SPY",
    uk_benchmark: str = "ISF.L",
    eu_benchmark: str = "EXS1.DE",
) -> list[str]:
    """Return ordered list of preferred market benchmarks for OLS Beta stripping.

    - US equities and LSE-listed US/Global index ETFs (e.g. VUAG.L, CSP1.L, VWRP.L) anchor to SPY.
    - Domestic UK equities (.L) and UK index ETFs anchor to ISF.L (fallback ^FTSE -> SPY).
    - European equities (.DE/.AS/.PA/...) and Europe ETFs anchor to EXS1.DE / VERG.L / FEZ (fallback ISF.L -> SPY).
    """
    clean = (symbol or "").strip().upper()
    if clean in _US_OR_GLOBAL_LSE_ETFS:
        return [default_benchmark]
    if clean in _EU_LSE_ETFS or clean.endswith(_EU_SUFFIXES):
        candidates = [eu_benchmark, "VERG.L", "FEZ", "^STOXX50E", uk_benchmark, default_benchmark]
    elif clean.endswith(".L"):
        candidates = [uk_benchmark, "^FTSE", default_benchmark]
    else:
        candidates = [default_benchmark]
    return [c for c in dict.fromkeys(candidates) if c]


def _ols_residual_and_clone_flag(
    ser: pd.Series,
    bench_rets: pd.Series,
) -> tuple[pd.Series, bool]:
    """Run OLS regression of ser on bench_rets and return (residual_series, is_index_clone)."""
    pair = pd.concat([ser, bench_rets], axis=1, join="inner").dropna()
    if len(pair) < 15:
        return ser, False
    y = pair.iloc[:, 0].to_numpy(dtype=float)
    x = pair.iloc[:, 1].to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=1))
    if var_x <= 1e-12:
        return ser, False
    cov_xy = float(np.cov(y, x, ddof=1)[0, 1])
    beta = cov_xy / var_x
    alpha = float(np.mean(y) - beta * np.mean(x))
    resid = y - (alpha + beta * x)
    raw_std = float(np.std(y, ddof=1))
    resid_std = float(np.std(resid, ddof=1))
    is_clone = bool(beta > 0.7 and raw_std > 1e-12 and (resid_std / raw_std) < 0.22)
    return pd.Series(resid, index=pair.index), is_clone


def _compute_rect_corr(
    df: pd.DataFrame,
    row_syms: list[str],
    col_syms: list[str],
    min_periods: int = 15,
) -> pd.DataFrame:
    """Compute exact Pearson correlation matrix of shape (len(row_syms), len(col_syms)) without O(N^2) blowup."""
    r_valid = [s for s in row_syms if s in df.columns]
    c_valid = [s for s in col_syms if s in df.columns]
    if not r_valid or not c_valid:
        return pd.DataFrame(index=r_valid, columns=c_valid, dtype=float)
    if r_valid == c_valid:
        return df[r_valid].corr(min_periods=min_periods)

    sub_cols = df[c_valid]
    out = pd.DataFrame(index=r_valid, columns=c_valid, dtype=float)
    for r_sym in r_valid:
        ser_r = df[r_sym]
        mask_r = ser_r.notna()
        if int(mask_r.sum()) < min_periods:
            continue
        sub_masked = sub_cols.loc[mask_r]
        counts = sub_masked.notna().sum(axis=0)
        corrs = sub_masked.corrwith(ser_r.loc[mask_r])
        corrs = corrs.where(counts >= min_periods)
        out.loc[r_sym] = corrs
    return out


def compute_correlation_matrix(
    histories: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    lookback_days: int = 60,
    benchmark_symbol: str | None = "SPY",
    uk_benchmark_symbol: str | None = "ISF.L",
    eu_benchmark_symbol: str | None = "EXS1.DE",
    against_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix, stripping regional market Beta when benchmark history is available.

    - Calendar Alignment & Closing Time-Lag Compensation: aligns multi-market close series with ffill(limit=1) for
      asynchronous single-day US/UK/EU holidays and evaluates 2-day rolling returns alongside 1-day returns for
      cross-timezone pairs (16:30 UK close vs 21:00 US close) so cross-Atlantic overlap is not underestimated.
    - Pure index clones (where >95% of variance tracks their benchmark, or sharing the same ETF index family) keep raw
      correlation so duplicate index ETFs (e.g. VUAG.L vs CSP1.L or SPY) remain flagged as overlapping.
    - Individual stocks / active names use regional OLS market-residual correlation (US -> SPY, UK .L -> ISF.L,
      EU -> EXS1.DE/VERG.L, with graceful fallback to SPY) so market-wide selloffs do not create false overlap warnings.
    """
    target_syms = list(dict.fromkeys(symbols or list(histories.keys())))
    close_map: dict[str, pd.Series] = {}
    for sym in target_syms:
        cs = _extract_close_series(histories.get(sym), lookback_days)
        if cs is not None:
            close_map[sym] = cs

    if len(close_map) < 2:
        return pd.DataFrame()

    # Include available benchmarks in the aligned calendar so holiday ffill(limit=1) and 2-day returns are consistent
    bench_candidates: set[str] = set()
    if benchmark_symbol:
        for sym in list(close_map):
            for b_cand in resolve_market_benchmark(
                sym,
                default_benchmark=benchmark_symbol,
                uk_benchmark=uk_benchmark_symbol or benchmark_symbol,
                eu_benchmark=eu_benchmark_symbol or benchmark_symbol,
            ):
                if b_cand in histories and b_cand not in close_map:
                    b_cs = _extract_close_series(histories.get(b_cand), lookback_days)
                    if b_cs is not None:
                        close_map[b_cand] = b_cs
                        bench_candidates.add(b_cand)

    price_df = pd.DataFrame(close_map).sort_index()
    # Fill isolated 1-day asynchronous holidays (e.g. UK Bank Holiday vs US Independence Day) without bridging long gaps
    price_ffill = price_df.ffill(limit=1)
    ret1_df = price_ffill.pct_change(fill_method=None).tail(lookback_days)
    ret2_df = price_ffill.pct_change(periods=2, fill_method=None).tail(lookback_days)
    ret5_df = price_ffill.pct_change(periods=5, fill_method=None).tail(lookback_days)

    valid_syms = [s for s in target_syms if s in ret1_df.columns and ret1_df[s].dropna().shape[0] >= 15]
    if len(valid_syms) < 2:
        return pd.DataFrame()

    if against_symbols is not None:
        against_set = set(against_symbols)
        row_syms = [s for s in valid_syms if s in against_set]
        if not row_syms:
            return pd.DataFrame()
    else:
        row_syms = valid_syms
    col_syms = valid_syms

    raw_corr_1d = _compute_rect_corr(ret1_df, row_syms, col_syms, min_periods=15)
    raw_corr_2d = _compute_rect_corr(ret2_df, row_syms, col_syms, min_periods=15)
    raw_corr_5d = _compute_rect_corr(ret5_df, row_syms, col_syms, min_periods=15)

    # For cross-timezone pairs (UK/EU 16:30 close vs US 21:00 close), compensate for closing lag via 2d/5d rolling corr
    raw_corr = raw_corr_1d.copy()
    for a in row_syms:
        for b in col_syms:
            if a == b or not _is_cross_timezone_pair(a, b):
                continue
            best_v = raw_corr_1d.loc[a, b]
            for alt_corr in (raw_corr_2d, raw_corr_5d):
                if a in alt_corr.index and b in alt_corr.columns:
                    v_alt = alt_corr.loc[a, b]
                    if pd.notna(v_alt) and (pd.isna(best_v) or abs(float(v_alt)) > abs(float(best_v))):
                        best_v = v_alt
            raw_corr.loc[a, b] = best_v

    if benchmark_symbol is None:
        return raw_corr.round(2)

    resid1_map: dict[str, pd.Series] = {}
    resid2_map: dict[str, pd.Series] = {}
    is_index_clone: dict[str, bool] = {}
    clone_bench_map: dict[str, str] = {}

    for sym in valid_syms:
        ser1 = ret1_df[sym].dropna()
        ser2 = ret2_df[sym].dropna()
        pref_benches = resolve_market_benchmark(
            sym,
            default_benchmark=benchmark_symbol,
            uk_benchmark=uk_benchmark_symbol or benchmark_symbol,
            eu_benchmark=eu_benchmark_symbol or benchmark_symbol,
        )
        chosen_bench: str | None = None
        b_col: str | None = None
        for cand_b in pref_benches:
            if cand_b in ret1_df.columns and ret1_df[cand_b].dropna().shape[0] >= 15:
                chosen_bench = cand_b.upper()
                b_col = cand_b
                break

        if chosen_bench is None or b_col is None:
            resid1_map[sym] = ser1
            resid2_map[sym] = ser2
            is_index_clone[sym] = False
            continue

        if sym.upper() == chosen_bench:
            is_index_clone[sym] = True
            clone_bench_map[sym] = chosen_bench
            continue

        r1, clone1 = _ols_residual_and_clone_flag(ser1, ret1_df[b_col].dropna())
        r2, clone2 = _ols_residual_and_clone_flag(ser2, ret2_df[b_col].dropna())
        resid1_map[sym] = r1
        resid2_map[sym] = r2

        sym_fam = infer_etf_index_family(sym)
        bench_fam = infer_etf_index_family(chosen_bench)
        same_etf_family = bool(sym_fam and bench_fam and sym_fam == bench_fam)
        is_clone = bool(clone1 or (clone2 and _is_cross_timezone_pair(sym, chosen_bench)) or same_etf_family)
        is_index_clone[sym] = is_clone
        if is_clone:
            clone_bench_map[sym] = chosen_bench

    resid1_df = pd.DataFrame(resid1_map).dropna(how="all")
    resid2_df = pd.DataFrame(resid2_map).dropna(how="all")
    if resid1_df.empty:
        return raw_corr.round(2)

    resid_corr_1d = _compute_rect_corr(resid1_df, row_syms, col_syms, min_periods=15)
    resid_corr_2d = (
        _compute_rect_corr(resid2_df, row_syms, col_syms, min_periods=15)
        if not resid2_df.empty
        else pd.DataFrame()
    )

    combined = raw_corr.copy()
    for a in combined.index:
        for b in combined.columns:
            if a == b:
                continue
            if is_index_clone.get(a, False) and is_index_clone.get(b, False):
                if clone_bench_map.get(a) == clone_bench_map.get(b):
                    continue
            val = resid_corr_1d.loc[a, b] if (a in resid_corr_1d.index and b in resid_corr_1d.columns) else np.nan
            if _is_cross_timezone_pair(a, b) and a in resid_corr_2d.index and b in resid_corr_2d.columns:
                val2 = resid_corr_2d.loc[a, b]
                if pd.notna(val2) and (pd.isna(val) or abs(float(val2)) > abs(float(val))):
                    val = val2
            if pd.notna(val):
                combined.loc[a, b] = val
    return combined.round(2)


def max_correlation_against_portfolio(
    corr_matrix: pd.DataFrame,
    candidate_symbol: str,
    portfolio_symbols: list[str],
) -> tuple[str | None, float | None]:
    """Return (most_correlated_holding, corr_value) for overlap checking."""
    if corr_matrix.empty or candidate_symbol not in corr_matrix.columns:
        return None, None
    best_sym: str | None = None
    best_val: float = -1.0
    for held in portfolio_symbols:
        if held == candidate_symbol or held not in corr_matrix.index:
            continue
        val = corr_matrix.loc[held, candidate_symbol]
        if pd.notna(val) and float(val) > best_val:
            best_val = float(val)
            best_sym = held
    if best_sym is None:
        return None, None
    return best_sym, best_val
