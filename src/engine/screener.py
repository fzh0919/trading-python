"""Quantitative multi-market pre-screener for /pick (0 LLM tokens before Agent review)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import (
    TechnicalSnapshot,
    compute_correlation_matrix,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
)
from src.data_providers.market_data import MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData


class ScreenerCandidate(BaseModel):
    symbol: str
    track: str  # ISA_LONG_TERM | INVEST_TACTICAL
    name: str = ""
    asset_type: str = "EQUITY"  # ETF | EQUITY
    currency: str = ""
    score: float
    is_already_held: bool = False
    is_watchlist: bool = False
    max_portfolio_corr: float | None = None
    most_correlated_holding: str | None = None
    overlap_warning: bool = False
    reasons: list[str] = Field(default_factory=list)
    summary_line: str = ""


def score_candidate(
    symbol: str,
    track: str,
    tech: TechnicalSnapshot,
    bundle: TickerMarketBundle | None,
    max_corr: float | None,
    corr_holding: str | None,
    is_held: bool,
    is_watchlist: bool,
    strategy: StrategySection,
) -> ScreenerCandidate:
    """Deterministically score a candidate ticker for ISA or Invest suitability."""
    score = 50.0
    reasons: list[str] = []

    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF") or symbol.endswith(".L") and any(
        k in symbol for k in ("VWRP", "VUAG", "SWDA", "EQGB", "VERG", "VHYL", "IUKD", "CSP1", "CNX1", "IWQU", "IGLT")
    )
    asset_type = "ETF" if is_etf else "EQUITY"

    if track == "ISA_LONG_TERM":
        if tech.trend_regime == "BULL":
            score += 14
            reasons.append("上升趋势(MA50>MA200)")
        elif tech.trend_regime == "PULLBACK":
            score += 18
            reasons.append("多头回踩支撑区")
        elif tech.trend_regime == "WEAK":
            score -= 18
            reasons.append("弱势低于MA200")

        if tech.drawdown_52w_pct is not None and -18.0 <= tech.drawdown_52w_pct <= -4.0:
            score += 10
            reasons.append(f"适度回撤{tech.drawdown_52w_pct:.1f}%")
        if tech.rsi14 is not None and 32.0 <= tech.rsi14 <= 55.0:
            score += 8
            reasons.append(f"RSI={tech.rsi14:.0f}不追高")
        elif tech.rsi14 is not None and tech.rsi14 >= 72.0:
            score -= 12
            reasons.append(f"RSI={tech.rsi14:.0f}偏热")

        if symbol.endswith(".L") and is_etf:
            score += 8
            reasons.append("LSE英镑ETF(免换汇/免印花税)")
        elif symbol.endswith(".L"):
            score += 3
            reasons.append("英镑计价免换汇")

        if tech.volatility_60d_pct is not None and tech.volatility_60d_pct > 45.0:
            score -= 10
            reasons.append(f"高波动({tech.volatility_60d_pct:.0f}%)")
    else:
        # INVEST_TACTICAL: catalyst, momentum pullback, or oversold setup
        if tech.trend_regime == "PULLBACK":
            score += 16
            reasons.append("强趋势回调买点")
        elif tech.trend_regime == "BULL":
            score += 10
            reasons.append("趋势延续")
        elif tech.trend_regime == "WEAK":
            score -= 12

        if tech.rsi14 is not None and tech.rsi14 <= 38.0:
            score += 14
            reasons.append(f"RSI={tech.rsi14:.0f}超卖弹性")
        elif tech.rsi14 is not None and 40.0 <= tech.rsi14 <= 60.0 and tech.ret_20d_pct and tech.ret_20d_pct > 0:
            score += 8
            reasons.append("动量健康")
        elif tech.rsi14 is not None and tech.rsi14 >= 74.0:
            score -= 15
            reasons.append(f"RSI={tech.rsi14:.0f}短期过热")

        if bundle and bundle.days_to_earnings is not None and 0 <= bundle.days_to_earnings <= 5:
            score -= 10
            reasons.append(f"临近财报({bundle.days_to_earnings}天)风险较高")

    overlap_warning = False
    if max_corr is not None and not is_held:
        if max_corr >= strategy.overlap_correlation_threshold:
            overlap_warning = True
            score -= 16
            reasons.append(f"与持仓{corr_holding}相关性高达{max_corr:.2f}(高度重叠)")
        elif max_corr <= 0.55:
            score += 10
            reasons.append(f"组合分散度佳(最高相关{max_corr:.2f})")

    if is_watchlist:
        score += 6
        reasons.append("自选观察池")
    if is_held:
        reasons.append("当前已持有")

    pe_str = ""
    if bundle:
        if bundle.forward_pe:
            pe_str = f" | 远期PE {bundle.forward_pe:.1f}x"
        elif bundle.trailing_pe:
            pe_str = f" | PE {bundle.trailing_pe:.1f}x"
    ed_str = f" | 财报 {bundle.earnings_date}" if bundle and bundle.earnings_date else ""
    corr_str = (
        f" | 最高相关({corr_holding})={max_corr:.2f}"
        if max_corr is not None and corr_holding
        else ""
    )

    summary_line = (
        f"• [{track}][{asset_type}] {symbol}"
        f"{f' ({bundle.short_name})' if bundle and bundle.short_name else ''} "
        f"| 量化初筛分 {score:.0f} | 现价 {tech.last_close:.2f}"
        f"{f' {bundle.currency}' if bundle and bundle.currency else ''} | "
        f"5日 {tech.ret_5d_pct:+.1f}% / 20日 {tech.ret_20d_pct:+.1f}% | "
        f"距MA50 {tech.dist_ma50_pct:+.1f}% / MA200 {tech.dist_ma200_pct:+.1f}% | "
        f"RSI {tech.rsi14:.0f} | 距52w高点 {tech.drawdown_52w_pct:+.1f}%"
        f"{pe_str}{ed_str}{corr_str} | 标签: {', '.join(reasons)}"
        if tech.ret_5d_pct is not None
        and tech.ret_20d_pct is not None
        and tech.dist_ma50_pct is not None
        and tech.dist_ma200_pct is not None
        and tech.rsi14 is not None
        and tech.drawdown_52w_pct is not None
        else f"• [{track}][{asset_type}] {symbol} | 分数 {score:.0f} | 标签: {', '.join(reasons)}"
    )

    return ScreenerCandidate(
        symbol=symbol,
        track=track,
        name=bundle.short_name if bundle else symbol,
        asset_type=asset_type,
        currency=bundle.currency if bundle else "",
        score=score,
        is_already_held=is_held,
        is_watchlist=is_watchlist,
        max_portfolio_corr=max_corr,
        most_correlated_holding=corr_holding,
        overlap_warning=overlap_warning,
        reasons=reasons,
        summary_line=summary_line,
    )


async def run_quant_screener(
    portfolio: PortfolioData,
    market_data: MarketDataService,
    universe: UniverseSection,
    watchlist: list[str],
    strategy: StrategySection,
    existing_histories: dict | None = None,
    top_n_per_track: int = 8,
) -> tuple[list[ScreenerCandidate], list[ScreenerCandidate]]:
    """Pre-screen ISA (stocks + LSE ETFs) & Invest tactical candidates in Python."""
    held_syms = portfolio.unique_yahoo_symbols()
    watch_set = {s.upper() for s in watchlist}

    isa_pool = list(dict.fromkeys([*watchlist, *universe.isa_long_term]))
    invest_pool = list(dict.fromkeys([*watchlist, *universe.invest_tactical]))
    all_needed = list(dict.fromkeys([*held_syms, *isa_pool, *invest_pool]))
    # TODO(IMP-08): [接自 IMP-04 & IMP-05 & IMP-06] 初筛改调 market_data.get_histories(..., period="1y") 并仅对 Top 候选补调 get_bundles(..., isin_map=...)；
    # fetch_needed 须补入 "ISF.L" 以激活分市场去 Beta；compute_technical_snapshot 传入 bundle，并同步更新 _FakeMarketData/_MockMarketData。
    fetch_needed = list(dict.fromkeys([*all_needed, "SPY"]))

    histories, bundles = await market_data.get_bundles(
        fetch_needed, include_news=False, include_earnings=True
    )
    if existing_histories:
        histories.update(existing_histories)

    corr_matrix = compute_correlation_matrix(
        histories,
        all_needed,
        lookback_days=strategy.correlation_window_days,
        benchmark_symbol="SPY",
    )
    tech_cache: dict[str, TechnicalSnapshot | None] = {
        sym: compute_technical_snapshot(sym, histories.get(sym))
        for sym in all_needed
    }

    def _build_track(pool: list[str], track: str) -> list[ScreenerCandidate]:
        out: list[ScreenerCandidate] = []
        for sym in pool:
            tech = tech_cache.get(sym)
            if tech is None:
                continue
            corr_sym, corr_val = max_correlation_against_portfolio(corr_matrix, sym, held_syms)
            out.append(
                score_candidate(
                    symbol=sym,
                    track=track,
                    tech=tech,
                    bundle=bundles.get(sym),
                    max_corr=corr_val,
                    corr_holding=corr_sym,
                    is_held=(sym in held_syms),
                    is_watchlist=(sym.upper() in watch_set),
                    strategy=strategy,
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:top_n_per_track]

    return _build_track(isa_pool, "ISA_LONG_TERM"), _build_track(invest_pool, "INVEST_TACTICAL")
