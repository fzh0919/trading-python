"""Two-stage multi-bucket quantitative screener orchestration for /pick."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_correlation_matrix,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
    resolve_market_benchmark,
)
from src.data_providers.market_data import (
    MacroSnapshot,
    MarketDataService,
    TickerMarketBundle,
)
from src.data_providers.t212_client import PortfolioData
from src.engine.screener.patterns import (
    ScreenerCandidate,
    _normalize_regime_tag,
    compute_market_breadth,
    compute_portfolio_risk_metrics,
)
from src.engine.screener.scoring import (
    _apply_percentile_calibration,
    _resolve_configured_etfs,
    score_candidate,
)

async def run_quant_screener(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    universe: UniverseSection,
    watchlist: list[str],
    strategy: StrategySection,
    existing_histories: dict | None = None,
    top_n_per_track: int = 8,
    active_pool_rows: list[dict[str, Any]] | None = None,
    macro: MacroSnapshot | None = None,
    regime_tag: str = "",
) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
    """Two-stage quantitative pre-screener: 0-info OHLCV multi-bucket filter -> Stage 2 bundle enrichment."""
    held_syms = portfolio.unique_yahoo_symbols()
    held_set = {s.upper() for s in held_syms}
    watch_set = {s.upper() for s in watchlist}
    etf_set = _resolve_configured_etfs(universe, strategy)

    sector_hint_map: dict[str, str] = {}
    if active_pool_rows:
        for row in active_pool_rows:
            sym_u = str(row.get("symbol") or "").strip().upper()
            if not sym_u:
                continue
            if bool(row.get("is_etf")):
                etf_set.add(sym_u)
            sec = str(row.get("sector") or "").strip()
            if sec:
                sector_hint_map[sym_u] = sec

    # Dynamically expand scan pools with symbols traded in the last 90 days
    cutoff_90d_ms = int(time.time() * 1000) - 90 * 86400 * 1000
    recent_isa: list[str] = []
    recent_invest: list[str] = []
    recent_traded_set: set[str] = set()
    for order in (*portfolio.isa.recent_orders, *portfolio.invest.recent_orders):
        sym = (order.yahoo_symbol or "").strip()
        if not sym:
            continue
        if order.filled_at_ms > 0 and order.filled_at_ms < cutoff_90d_ms:
            continue
        recent_traded_set.add(sym.upper())
        if order.account.upper() == "ISA":
            recent_isa.append(sym)
        else:
            recent_invest.append(sym)

    if active_pool_rows:
        isa_base: list[str] = []
        invest_base: list[str] = []
        for row in active_pool_rows:
            sym_u = str(row.get("symbol") or "").strip().upper()
            if not sym_u:
                continue
            is_etf_row = bool(row.get("is_etf")) or (sym_u in etf_set)
            if bool(row.get("isa_eligible", True)):
                isa_base.append(sym_u)
            # Only share non-ETF vetted equities with Invest; keep LSE GBP ETFs ISA-focused unless already in invest_tactical
            if not is_etf_row and bool(row.get("invest_eligible", True)):
                invest_base.append(sym_u)
        for etf_sym in etf_set:
            if etf_sym not in isa_base:
                isa_base.append(etf_sym)
        isa_pool = list(dict.fromkeys([*watchlist, *isa_base, *recent_isa]))
        invest_pool = list(dict.fromkeys([*watchlist, *invest_base, *recent_invest]))
    else:
        isa_pool = list(dict.fromkeys([*watchlist, *universe.isa_long_term, *recent_isa]))
        invest_pool = list(dict.fromkeys([*watchlist, *universe.invest_tactical, *recent_invest]))

    all_needed = list(dict.fromkeys([*held_syms, *isa_pool, *invest_pool]))

    benchmarks = ["SPY", "ISF.L"]
    if any(resolve_market_benchmark(sym)[0] == "EXS1.DE" for sym in all_needed):
        benchmarks.append("EXS1.DE")
    fetch_needed = list(dict.fromkeys([*all_needed, *benchmarks]))

    # Stage 1: Reuse already-fetched held histories & fetch remaining OHLCV without .info calls
    histories: dict[str, pd.DataFrame] = {}
    if existing_histories:
        for k, df in existing_histories.items():
            if df is not None and not df.empty:
                histories[k] = df

    missing_hist_syms = [s for s in fetch_needed if s not in histories]
    if missing_hist_syms:
        fetched_hists = await market_data.get_histories(missing_hist_syms, period="1y")
        for k, df in fetched_hists.items():
            if df is not None and not df.empty:
                histories[k] = df

    corr_matrix = compute_correlation_matrix(
        histories,
        all_needed,
        lookback_days=strategy.correlation_window_days,
        benchmark_symbol="SPY",
        uk_benchmark_symbol="ISF.L",
        eu_benchmark_symbol="EXS1.DE",
        against_symbols=held_syms or None,
    )

    stage1_tech: dict[str, TechnicalSnapshot | None] = {
        sym: compute_technical_snapshot(sym, histories.get(sym))
        for sym in all_needed
    }

    breadth_info = compute_market_breadth(
        stage1_tech,
        [*isa_pool, *invest_pool],
        histories=histories,
        benchmark_symbols=benchmarks,
    )
    regime_tag = (
        _normalize_regime_tag(regime_tag)
        if str(regime_tag or "").strip()
        else _normalize_regime_tag(breadth_info.get("regime_tag", ""))
    )

    corr_cache: dict[str, tuple[str | None, float | None]] = {
        sym: max_correlation_against_portfolio(corr_matrix, sym, held_syms)
        for sym in all_needed
    }
    risk_cache = compute_portfolio_risk_metrics(
        histories,
        portfolio,
        all_needed,
        lookback_days=strategy.correlation_window_days,
    )

    def _bench_for(sym: str) -> pd.DataFrame | None:
        bench_sym = resolve_market_benchmark(sym)[0]
        return histories.get(bench_sym)

    preselect_n = max(top_n_per_track * 2, top_n_per_track + 6)

    def _multi_bucket_shortlist(pool: list[str], track: str) -> list[str]:
        scored: list[ScreenerCandidate] = []
        tech_by_sym: dict[str, TechnicalSnapshot] = {}
        for sym in pool:
            tech = stage1_tech.get(sym)
            if tech is None:
                continue
            tech_by_sym[sym] = tech
            corr_sym, corr_val = corr_cache.get(sym, (None, None))
            r_meta = risk_cache.get(sym, {})
            scored.append(
                score_candidate(
                    symbol=sym,
                    track=track,
                    tech=tech,
                    bundle=None,
                    max_corr=corr_val,
                    corr_holding=corr_sym,
                    is_held=(sym.upper() in held_set),
                    is_watchlist=(sym.upper() in watch_set),
                    strategy=strategy,
                    hist=histories.get(sym),
                    benchmark_hist=_bench_for(sym),
                    weighted_corr=r_meta.get("weighted_corr"),
                    marginal_vol_delta_pct=r_meta.get("marginal_vol_delta_pct"),
                    regime_tag=regime_tag,
                    is_recent_traded=(sym.upper() in recent_traded_set),
                    etf_symbols=etf_set,
                    invest_wallet=portfolio.invest,
                    macro=macro,
                )
            )

        if len(scored) <= preselect_n:
            scored.sort(key=lambda c: c.score, reverse=True)
            return [c.symbol for c in scored]

        picked: list[str] = []
        picked_set: set[str] = set()
        sector_counts: dict[str, int] = {}
        max_per_sector = max(3, preselect_n // 4) if (sector_hint_map and len(scored) > preselect_n * 2) else preselect_n

        def _take(ordered: list[ScreenerCandidate], limit: int, *, apply_sector_cap: bool = True) -> None:
            for c in ordered:
                if len(picked) >= preselect_n or limit <= 0:
                    break
                if c.symbol in picked_set:
                    continue
                sec = sector_hint_map.get(c.symbol.upper(), "")
                if apply_sector_cap and sec and sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                picked.append(c.symbol)
                picked_set.add(c.symbol)
                if sec:
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                limit -= 1

        # Bucket 1 (~40%): Momentum & breakout leaders (Trend+RS + Pattern)
        mom_quota = max(1, (preselect_n * 4) // 10)
        mom_sorted = sorted(
            scored,
            key=lambda c: (
                c.dimension_pct.get("trend_rs", 0.0) + c.dimension_pct.get("pattern_timing", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(mom_sorted, mom_quota)

        # Bucket 2 (~25%): Pullback & low-volatility base candidates (not Phase 4)
        pb_quota = max(1, preselect_n // 4)
        pb_pool = [
            c
            for c in scored
            if "[Phase4弱势]" not in c.cons
            and (
                tech_by_sym[c.symbol].trend_regime == "PULLBACK"
                or (tech_by_sym[c.symbol].rsi14 or 50.0) <= 50.0
                or "VCP收缩" in c.pattern_signals
                or "低ATR收敛" in c.pattern_signals
            )
        ]
        pb_pool.sort(
            key=lambda c: (
                c.dimension_pct.get("pattern_timing", 0.0) + c.dimension_pct.get("portfolio_fit", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(pb_pool, pb_quota)

        # Bucket 3 (~20%): Defensive low-volatility / LSE ETF / portfolio diversification bucket (0 extra IO)
        def_quota = max(1, preselect_n // 5)
        def_pool = [
            c
            for c in scored
            if "[Phase4弱势]" not in c.cons
            and not c.overlap_warning
            and (
                c.asset_type == "ETF"
                or (tech_by_sym[c.symbol].volatility_60d_pct is not None and tech_by_sym[c.symbol].volatility_60d_pct <= 28.0)
                or (c.marginal_vol_delta_pct is not None and c.marginal_vol_delta_pct <= 0.10)
                or (c.max_portfolio_corr is not None and c.max_portfolio_corr <= 0.55)
            )
        ]
        def_pool.sort(
            key=lambda c: (
                c.dimension_pct.get("portfolio_fit", 0.0) + c.dimension_pct.get("value_quality", 0.0),
                c.score,
            ),
            reverse=True,
        )
        _take(def_pool, def_quota)

        # Bucket 4 (~15%, min 1): Watchlist & recent-traded priority protection (exclude severe overlaps)
        prio_quota = max(1, preselect_n // 6)
        prio_pool = sorted(
            [
                c
                for c in scored
                if (c.is_watchlist or c.is_recent_traded) and not c.overlap_warning
            ],
            key=lambda c: c.score,
            reverse=True,
        )
        _take(prio_pool, min(prio_quota, max(1, preselect_n - len(picked))), apply_sector_cap=False)

        # Backfill remaining capacity by overall Stage 1 score (respect sector cap first, relax if still short)
        overall_sorted = sorted(scored, key=lambda c: c.score, reverse=True)
        _take(overall_sorted, preselect_n - len(picked), apply_sector_cap=True)
        if len(picked) < preselect_n:
            _take(overall_sorted, preselect_n - len(picked), apply_sector_cap=False)
        return picked

    isa_shortlist = _multi_bucket_shortlist(isa_pool, "ISA_LONG_TERM")
    invest_shortlist = _multi_bucket_shortlist(invest_pool, "INVEST_TACTICAL")
    top_syms = list(dict.fromkeys([*isa_shortlist, *invest_shortlist]))

    # Stage 2: Enrich only shortlisted candidates with fundamentals/earnings/quote_type/US deep intel
    bundles: dict[str, TickerMarketBundle] = {}
    deep_intel_map: dict[str, dict[str, Any]] = {}
    if top_syms:
        isin_map = {
            p.yahoo_symbol: p.isin
            for p in portfolio.all_positions()
            if p.yahoo_symbol and p.isin
        }
        us_top_syms = [
            s
            for s in top_syms
            if "." not in s and "=" not in s and not s.startswith("^") and s.upper() not in etf_set
        ]

        async def _empty_intel() -> dict[str, dict[str, Any]]:
            return {}

        intel_coro = (
            market_data.get_us_deep_intel_batch(us_top_syms)
            if (us_top_syms and hasattr(market_data, "get_us_deep_intel_batch"))
            else _empty_intel()
        )
        (bundle_hists, bundles), deep_intel_map = await asyncio.gather(
            market_data.get_bundles(
                top_syms,
                include_news=False,
                include_earnings=True,
                include_info=True,
                isin_map=isin_map or None,
            ),
            intel_coro,
        )
        for sym, df in bundle_hists.items():
            if (sym not in histories or histories[sym].empty) and df is not None and not df.empty:
                histories[sym] = df

    stage2_tech: dict[str, TechnicalSnapshot | None] = {
        sym: compute_technical_snapshot(sym, histories.get(sym), bundle=bundles.get(sym))
        for sym in top_syms
    }

    def _build_track(shortlist: list[str], track: str) -> list[ScreenerCandidate]:
        out: list[ScreenerCandidate] = []
        for sym in shortlist:
            tech = stage2_tech.get(sym) or stage1_tech.get(sym)
            if tech is None:
                continue
            corr_sym, corr_val = corr_cache.get(sym, (None, None))
            r_meta = risk_cache.get(sym, {})
            out.append(
                score_candidate(
                    symbol=sym,
                    track=track,
                    tech=tech,
                    bundle=bundles.get(sym),
                    max_corr=corr_val,
                    corr_holding=corr_sym,
                    is_held=(sym.upper() in held_set),
                    is_watchlist=(sym.upper() in watch_set),
                    strategy=strategy,
                    hist=histories.get(sym),
                    benchmark_hist=_bench_for(sym),
                    weighted_corr=r_meta.get("weighted_corr"),
                    marginal_vol_delta_pct=r_meta.get("marginal_vol_delta_pct"),
                    regime_tag=regime_tag,
                    is_recent_traded=(sym.upper() in recent_traded_set),
                    etf_symbols=etf_set,
                    deep_intel=deep_intel_map.get(sym.upper()) or deep_intel_map.get(sym),
                    invest_wallet=portfolio.invest,
                    macro=macro,
                )
            )
        calibrated = _apply_percentile_calibration(out)
        return calibrated[:top_n_per_track]

    return _build_track(isa_shortlist, "ISA_LONG_TERM"), _build_track(invest_shortlist, "INVEST_TACTICAL")
