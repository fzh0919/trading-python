"""150-stock multi-market universe audition and T212 eligibility filtering."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_technical_snapshot,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MarketDataService
from src.engine.screener.patterns import (
    _extract_ohlcv_series,
    _linear_scale,
    compute_relative_strength,
)

logger = logging.getLogger("src.engine.screener")

def _approx_usd_multiplier(symbol: str, currency_hint: str = "") -> float:
    """Approximate FX multiplier to normalize daily turnover into USD for liquidity filtering."""
    curr = (currency_hint or "").strip()
    curr_up = curr.upper()
    if curr in {"GBp", "GBX"} or curr_up == "GBX":
        return 0.0125
    if curr_up == "GBP":
        return 1.25
    if curr_up == "EUR":
        return 1.08
    if curr_up == "DKK":
        return 0.145
    if curr_up == "CHF":
        return 1.12
    if curr_up == "SEK":
        return 0.095
    if curr_up == "USD":
        return 1.0
    sym_up = symbol.upper()
    if sym_up.endswith(".L"):
        return 0.0125
    if sym_up.endswith((".PA", ".DE", ".AS", ".MC", ".MI", ".BR", ".HE")):
        return 1.08
    if sym_up.endswith(".CO"):
        return 0.145
    return 1.0


def _match_t212_eligibility(
    symbol: str,
    elig_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Match Yahoo Finance symbol against SQLite T212 instrument metadata with punctuation variants."""
    if not elig_map:
        return None
    sym_u = symbol.strip().upper()
    if sym_u in elig_map:
        return elig_map[sym_u]
    if "." in sym_u:
        base, suffix = sym_u.rsplit(".", 1)
        alt1 = f"{base.replace('-', '.')}.{suffix}"
        alt2 = f"{base.replace('.', '-')}.{suffix}"
        for alt in (alt1, alt2):
            if alt in elig_map:
                return elig_map[alt]
    else:
        for alt in (sym_u.replace("-", "."), sym_u.replace(".", "-")):
            if alt in elig_map:
                return elig_map[alt]
    return None


def _is_uk_or_eu_symbol(symbol: str) -> bool:
    return symbol.strip().upper().endswith((".L", ".PA", ".DE", ".AS", ".MC", ".MI", ".BR", ".HE", ".CO"))


def _score_audition_equity(
    symbol: str,
    tech: TechnicalSnapshot,
    hist: pd.DataFrame,
    benchmark_hist: pd.DataFrame | None,
    fund: dict[str, Any] | None,
    avg_turnover_usd: float,
) -> float:
    """Score a vetted equity (0..100) for inclusion in the active 150-stock universe pool."""
    # 1) Profitability & Growth (0..35, midpoint 17.5)
    s_prof = 17.5
    if fund:
        roe = fund.get("roe_pct")
        if roe is not None:
            if roe >= 0:
                s_prof += _linear_scale(float(roe), 8.0, 32.0, 0.0, 7.5)
            else:
                s_prof -= _linear_scale(abs(float(roe)), 0.0, 25.0, 2.0, 7.5)
        margin = fund.get("profit_margin_pct")
        if margin is not None:
            if margin >= 0:
                s_prof += _linear_scale(float(margin), 4.0, 25.0, 0.0, 5.5)
            else:
                s_prof -= _linear_scale(abs(float(margin)), 0.0, 20.0, 1.5, 5.5)
        rev_g = fund.get("revenue_growth_pct")
        if rev_g is not None:
            s_prof += _linear_scale(float(rev_g), -5.0, 22.0, -3.5, 4.5)
    s_prof = max(0.0, min(35.0, s_prof))

    # 2) Balance Sheet & Valuation Discipline (0..25, midpoint 13.0)
    s_bal = 13.0
    if fund:
        dte = fund.get("debt_to_equity")
        if dte is not None:
            if dte <= 60.0:
                s_bal += 3.5
            elif dte <= 120.0:
                s_bal += 1.0
            elif dte > 180.0:
                s_bal -= _linear_scale(float(dte), 180.0, 350.0, 1.5, 5.0)
        pe = fund.get("forward_pe") or fund.get("trailing_pe")
        if pe is not None:
            if 8.0 <= pe <= 28.0:
                s_bal += 4.0
            elif 28.0 < pe <= 45.0:
                s_bal += 1.5
            elif pe > 65.0 or pe < 0.0:
                s_bal -= 3.5
        analysts = fund.get("analyst_count")
        if analysts is not None and analysts >= 10:
            s_bal += _linear_scale(float(analysts), 10.0, 35.0, 1.0, 3.0)
    s_bal = max(0.0, min(25.0, s_bal))

    # 3) Scale & Liquidity (0..20)
    s_liq = 8.0
    if avg_turnover_usd > 0:
        s_liq += _linear_scale(float(np.log10(max(avg_turnover_usd, 1e5))), 6.7, 9.0, 0.0, 6.0)
    mcap = fund.get("market_cap") if fund else None
    if mcap is not None and mcap > 0:
        s_liq += _linear_scale(float(np.log10(max(float(mcap), 1e8))), 9.0, 11.8, 0.0, 6.0)
    else:
        s_liq += 2.0
    s_liq = max(0.0, min(20.0, s_liq))

    # 4) Long-Term Trend & Relative Strength Health (0..20, midpoint 10.0)
    s_trend = 10.0
    if tech.trend_regime == "BULL":
        s_trend += 4.0
    elif tech.trend_regime == "PULLBACK":
        s_trend += 3.5
    elif tech.trend_regime == "WEAK":
        s_trend -= 4.0
    rs_info = compute_relative_strength(hist, benchmark_hist)
    rs_60d = rs_info.get("rs_60d_pct")
    if rs_60d is not None:
        s_trend += _linear_scale(float(rs_60d), -15.0, 15.0, -3.5, 4.0)
    if tech.ret_60d_pct is not None and tech.ret_60d_pct > 0:
        s_trend += 1.5
    s_trend = max(0.0, min(20.0, s_trend))

    return round(s_prof + s_bal + s_liq + s_trend, 2)


async def run_universe_audition(
    cfg: Any,
    t212: Any,
    market_data: MarketDataService,
    db: Any,
    *,
    sync_online_indices: bool = False,
    sync_t212_metadata: bool = True,
) -> dict[str, Any]:
    """0-Token multi-market audition: T212 tradability + 1y liquidity + cached fundamentals -> active top-N pool."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from src.engine.universe_seeds import resolve_base_universe_symbols

    universe_cfg = cfg.universe
    target_size = max(20, int(getattr(universe_cfg, "audition_target_size", 150) or 150))
    sector_cap = max(5, int(getattr(universe_cfg, "audition_sector_cap", 28) or 28))
    min_uk_eu_slots = max(0, int(getattr(universe_cfg, "audition_min_uk_eu_slots", 30) or 30))
    min_turnover_usd = max(0.0, float(getattr(universe_cfg, "audition_min_avg_dollar_vol_usd", 5_000_000.0) or 0.0))
    ttl_days = max(1, int(getattr(universe_cfg, "audition_fundamentals_ttl_days", 21) or 21))
    info_conc = max(1, int(getattr(universe_cfg, "audition_info_concurrency", 2) or 2))
    info_delay = max(0.0, float(getattr(universe_cfg, "audition_info_delay_seconds", 0.45) or 0.45))

    previous_rows = await db.get_active_universe_pool() if hasattr(db, "get_active_universe_pool") else []
    prev_equities = {str(r["symbol"]).upper() for r in previous_rows if not r.get("is_etf")}

    # Step 1: Resolve multi-market base pool & T212 account-level eligibility
    logger.info("Audition Step 1/4: resolving multi-market base universe & T212 eligibility...")
    base_data = await resolve_base_universe_symbols(cfg, db=db, sync_online=sync_online_indices)
    equities_map: dict[str, list[str]] = dict(base_data.get("equities") or {})
    etf_syms: list[str] = list(base_data.get("etfs") or [])

    elig_map = await db.get_t212_eligibility_by_yahoo() if hasattr(db, "get_t212_eligibility_by_yahoo") else {}
    latest_meta_ms = max((int(v.get("updated_at_ms") or 0) for v in elig_map.values()), default=0)
    stale_meta = (int(time.time() * 1000) - latest_meta_ms) > (20 * 3600 * 1000)
    if sync_t212_metadata and (not elig_map or stale_meta):
        if hasattr(t212, "sync_all_accounts_instruments_metadata"):
            try:
                await t212.sync_all_accounts_instruments_metadata()
                elig_map = await db.get_t212_eligibility_by_yahoo()
            except Exception as exc:
                logger.warning("Audition T212 metadata sync failed; continuing with existing table: %s", exc)
        elif hasattr(t212, "fetch_instruments_metadata"):
            try:
                await t212.fetch_instruments_metadata()
                elig_map = await db.get_t212_eligibility_by_yahoo()
            except Exception as exc:
                logger.warning("Audition T212 metadata sync failed: %s", exc)

    t212_filtered = bool(elig_map)
    tradable_equities: dict[str, dict[str, Any]] = {}
    for sym, idx_sources in equities_map.items():
        match = _match_t212_eligibility(sym, elig_map) if t212_filtered else None
        if t212_filtered and match is None:
            continue
        if match and str(match.get("inst_type") or "").upper() == "ETF":
            if sym not in etf_syms:
                etf_syms.append(sym)
            continue
        tradable_equities[sym] = {
            "symbol": sym,
            "index_sources": idx_sources,
            "currency_hint": str(match.get("currency_code") or "") if match else "",
            "name_hint": str(match.get("short_name") or "") if match else "",
            "isa_eligible": bool(match.get("isa_eligible", True)) if match else True,
            "invest_eligible": bool(match.get("invest_eligible", True)) if match else True,
        }

    tradable_etfs: list[dict[str, Any]] = []
    for sym in etf_syms:
        match = _match_t212_eligibility(sym, elig_map) if t212_filtered else None
        if t212_filtered and match is None:
            continue
        tradable_etfs.append(
            {
                "symbol": sym,
                "name": str(match.get("short_name") or sym) if match else sym,
                "currency": str(match.get("currency_code") or "GBP") if match else "GBP",
                "isa_eligible": bool(match.get("isa_eligible", True)) if match else True,
                "invest_eligible": bool(match.get("invest_eligible", True)) if match else True,
            }
        )

    # Step 2: Batched 1y OHLCV incremental update & local liquidity/structure pre-filter (0 .info calls)
    benchmarks = ["SPY", "ISF.L", "EXS1.DE"]
    eq_syms = list(tradable_equities.keys())
    logger.info(
        "Audition Step 2/4: fetching/updating 1y OHLCV for %d tradable equities (+%d ETFs)...",
        len(eq_syms),
        len(tradable_etfs),
    )
    histories = await market_data.get_histories([*eq_syms, *benchmarks], period="1y") if eq_syms else {}

    liquid_candidates: list[tuple[str, TechnicalSnapshot, pd.DataFrame, float]] = []
    for sym in eq_syms:
        hist = histories.get(sym)
        if hist is None or hist.empty or len(hist) < 120:
            continue
        ext = _extract_ohlcv_series(hist)
        if ext is None:
            continue
        _, _, _, closes, vols = ext
        if len(closes) < 120:
            continue
        meta = tradable_equities[sym]
        mult = _approx_usd_multiplier(sym, meta["currency_hint"])
        avg_turnover_usd = float((closes.tail(20) * vols.tail(20)).mean() * mult)
        if min_turnover_usd > 0 and avg_turnover_usd < min_turnover_usd:
            continue

        tech = compute_technical_snapshot(sym, hist)
        if tech is None:
            continue
        # Exclude severe long-term structural collapse before spending .info calls
        if (
            tech.ma200
            and tech.last_close < tech.ma200 * 0.55
            and (tech.ret_60d_pct or 0.0) < -25.0
        ):
            continue
        liquid_candidates.append((sym, tech, hist, avg_turnover_usd))

    # Step 3: Load cached fundamentals from SQLite & gently fetch missing/stale with 429 circuit breaker
    liquid_syms = [item[0] for item in liquid_candidates]
    logger.info(
        "Audition Step 3/4: %d/%d equities passed liquidity/structure pre-filter; loading fundamentals (cache TTL %dd)...",
        len(liquid_syms),
        len(eq_syms),
        ttl_days,
    )
    fund_map, fund_stats = await market_data.get_fundamentals_cached(
        liquid_syms,
        max_age_days=ttl_days,
        concurrency=info_conc,
        delay_seconds=info_delay,
        max_consecutive_429=3,
    )

    # Step 4: Score and select sector-balanced Top N (default 150) with UK/EU floor
    scored_rows: list[dict[str, Any]] = []
    for sym, tech, hist, avg_turnover_usd in liquid_candidates:
        meta = tradable_equities[sym]
        fund = fund_map.get(sym)
        if fund and str(fund.get("quote_type") or "").upper() == "ETF":
            continue
        bench_sym = resolve_market_benchmark(sym)[0]
        q_score = _score_audition_equity(
            sym,
            tech,
            hist,
            histories.get(bench_sym),
            fund,
            avg_turnover_usd,
        )
        sector = str((fund or {}).get("sector") or "").strip() or "Unknown"
        scored_rows.append(
            {
                "symbol": sym,
                "name": str((fund or {}).get("short_name") or meta["name_hint"] or sym),
                "exchange": str((fund or {}).get("exchange") or ""),
                "currency": str((fund or {}).get("currency") or meta["currency_hint"] or ""),
                "sector": sector,
                "industry": str((fund or {}).get("industry") or ""),
                "index_sources": meta["index_sources"],
                "quality_score": q_score,
                "avg_daily_turnover_usd": round(avg_turnover_usd, 1),
                "isa_eligible": meta["isa_eligible"],
                "invest_eligible": meta["invest_eligible"],
                "is_etf": False,
            }
        )

    scored_rows.sort(key=lambda r: (r["quality_score"], r["avg_daily_turnover_usd"]), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_set: set[str] = set()
    sector_counts: dict[str, int] = {}

    def _try_pick(row: dict[str, Any], *, enforce_cap: bool = True) -> bool:
        sym = row["symbol"]
        if sym in selected_set or len(selected) >= target_size:
            return False
        sec = row["sector"]
        cap = min(sector_cap, max(12, target_size // 5)) if sec == "Unknown" else sector_cap
        if enforce_cap and sector_counts.get(sec, 0) >= cap:
            return False
        selected.append(row)
        selected_set.add(sym)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        return True

    # Phase A: Reserve UK/EU floor slots first so FTSE + EuroStoxx remain well-represented
    if min_uk_eu_slots > 0:
        uk_eu_rows = [r for r in scored_rows if _is_uk_or_eu_symbol(r["symbol"])]
        for row in uk_eu_rows:
            if len(selected) >= min(min_uk_eu_slots, target_size):
                break
            _try_pick(row, enforce_cap=True)

    # Phase B: Fill remaining slots up to target_size respecting sector cap
    for row in scored_rows:
        if len(selected) >= target_size:
            break
        _try_pick(row, enforce_cap=True)

    # Phase C: Relax sector cap only if still below target_size
    if len(selected) < target_size:
        for row in scored_rows:
            if len(selected) >= target_size:
                break
            _try_pick(row, enforce_cap=False)

    etf_rows = [
        {
            "symbol": e["symbol"],
            "name": e["name"],
            "exchange": "LSE",
            "currency": e["currency"] or "GBP",
            "sector": "ETF",
            "industry": "UCITS ETF",
            "index_sources": ["CURATED_LSE_ETF"],
            "quality_score": 100.0,
            "avg_daily_turnover_usd": 0.0,
            "isa_eligible": e["isa_eligible"],
            "invest_eligible": e["invest_eligible"],
            "is_etf": True,
        }
        for e in tradable_etfs
    ]

    uk_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M")
    all_persisted = [*selected, *etf_rows]
    if all_persisted and hasattr(db, "replace_active_universe_pool"):
        await db.replace_active_universe_pool(all_persisted, updated_at_uk=uk_now)

    new_equities = {r["symbol"] for r in selected}
    added_syms = sorted(new_equities - prev_equities)
    removed_syms = sorted(prev_equities - new_equities) if prev_equities else []

    by_region = {"US": 0, "UK": 0, "EU": 0}
    for r in selected:
        sym = r["symbol"]
        if sym.endswith(".L"):
            by_region["UK"] += 1
        elif _is_uk_or_eu_symbol(sym):
            by_region["EU"] += 1
        else:
            by_region["US"] += 1

    logger.info(
        "Audition Step 4/4 complete: selected %d equities (US=%d, UK=%d, EU=%d) + %d LSE ETFs | fundamentals=%s",
        len(selected),
        by_region["US"],
        by_region["UK"],
        by_region["EU"],
        len(etf_rows),
        fund_stats,
    )

    return {
        "updated_at_uk": uk_now,
        "base_equities_count": len(equities_map),
        "t212_filtered": t212_filtered,
        "t212_tradable_count": len(tradable_equities),
        "liquidity_passed_count": len(liquid_candidates),
        "selected_equities_count": len(selected),
        "selected_etfs_count": len(etf_rows),
        "by_region": by_region,
        "by_sector": dict(sorted(sector_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "added_symbols": added_syms,
        "removed_symbols": removed_syms,
        "fundamentals_stats": fund_stats,
        "top_preview": [
            f"{r['symbol']}({r['quality_score']:.0f}, {r['sector']})"
            for r in selected[:10]
        ],
    }
