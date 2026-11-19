"""ETF look-through exposure (regional/tech), distribution policy, and portfolio CVaR attribution."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data_providers.analytics.technical import _extract_close_series
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import PortfolioData


_ISSUER_ALIASES: dict[str, str] = {
    "FB": "META",
    "GOOGL": "GOOG",
    "BRK.A": "BRK-B",
    "BRK-A": "BRK-B",
}

# US mega-cap tech / semiconductor factor bucket used for look-through exposure hints.
_TECH_FACTOR_TICKERS: frozenset[str] = frozenset(
    {
        "MSFT",
        "AVGO",
        "NVDA",
        "AAPL",
        "GOOG",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "NFLX",
        "ORCL",
        "CRM",
        "ADBE",
        "INTC",
        "QCOM",
        "TXN",
        "MU",
        "PLTR",
        "ARM",
        "SMCI",
        "TSM",
        "ASML",
        # v5.5.1: high-beta US software / cyber-security / semi-cap leaders that were
        # silently missing from the factor bucket (PANW/CRWD/GOOGL look-through gaps).
        "PANW",
        "CRWD",
        "NOW",
        "SNOW",
        "DDOG",
        "NET",
        "ANET",
        "MRVL",
        "KLAC",
        "LRCX",
        "AMAT",
        "CDNS",
        "SNPS",
    }
)

# Static APPROXIMATE implied tech weight inside broad UCITS ETFs.
# These are published order-of-magnitude anchors, NOT a live look-through file:
# they only feed a coarse "how much tech do I really own" hint and must never
# be used for a hard risk block.
_ETF_TECH_PENETRATION_PCT: dict[str, float] = {
    # Global all-world / developed
    "VWRP": 25.0,
    "VWRP.L": 25.0,
    "VWRL": 25.0,
    "VWRL.L": 25.0,
    "SWDA": 25.0,
    "SWDA.L": 25.0,
    "IWDA": 24.0,
    "IWDA.L": 24.0,
    "SSAC": 24.0,
    "SSAC.L": 24.0,
    "FWRG": 24.0,
    "FWRG.L": 24.0,
    # S&P 500
    "VUAG": 32.0,
    "VUAG.L": 32.0,
    "CSP1": 32.0,
    "CSP1.L": 32.0,
    "VUSA": 32.0,
    "VUSA.L": 32.0,
    "IUSA": 32.0,
    "IUSA.L": 32.0,
    "SPY": 32.0,
    "VOO": 32.0,
    "IVV": 32.0,
    # NASDAQ 100
    "CNX1": 50.0,
    "CNX1.L": 50.0,
    "EQGB": 50.0,
    "EQGB.L": 50.0,
    "EQQQ": 50.0,
    "EQQQ.L": 50.0,
    "QQQ": 50.0,
    "QQQM": 50.0,
    # FTSE 100 (tech weight is negligible)
    "ISF": 1.0,
    "ISF.L": 1.0,
    "VUKE": 1.0,
    "VUKE.L": 1.0,
    "CUKX": 1.0,
    "CUKX.L": 1.0,
    "VUKG": 1.0,
    "VUKG.L": 1.0,
    # Europe broad (developed / ex-UK)
    "VERG": 8.0,
    "VERG.L": 8.0,
    "VERX": 8.0,
    "VERX.L": 8.0,
    "VEUR": 8.0,
    "VEUR.L": 8.0,
    "VEUA": 8.0,
    "VEUA.L": 8.0,
    "IMEU": 8.0,
    "IMEU.L": 8.0,
    "SMEA": 8.0,
    "SMEA.L": 8.0,
}

# Static APPROXIMATE regional look-through for broad UCITS ETFs (percent of ETF NAV).
# Kept deliberately coarse: it only produces a context line and the deterministic core
# policy gap maths never treats it as a hard risk threshold.
_ETF_REGIONAL_LOOKTHROUGH: dict[str, dict[str, float]] = {
    # Global all-world / developed
    "VWRP": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "VWRL": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "SSAC": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    "FWRG": {"us": 63.0, "uk": 3.5, "eu_ex_uk": 12.5, "row": 21.0},
    # Developed world (MSCI World)
    "SWDA": {"us": 71.0, "uk": 3.8, "eu_ex_uk": 13.5, "row": 11.7},
    "IWDA": {"us": 71.0, "uk": 3.8, "eu_ex_uk": 13.5, "row": 11.7},
    # S&P 500
    "VUAG": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUSA": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "CSP1": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "IUSA": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "SPXP": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "GSPX": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "SPY": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VOO": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    "IVV": {"us": 100.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0},
    # Europe ex-UK (strictly zero UK overlap)
    "VERG": {"us": 0.0, "uk": 0.0, "eu_ex_uk": 100.0, "row": 0.0},
    "VERX": {"us": 0.0, "uk": 0.0, "eu_ex_uk": 100.0, "row": 0.0},
    # Europe including UK (~22% UK exposure)
    "VEUR": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "VEUA": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "IMEU": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    "SMEA": {"us": 0.0, "uk": 22.0, "eu_ex_uk": 78.0, "row": 0.0},
    # FTSE 100
    "ISF": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUKE": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "CUKX": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
    "VUKG": {"us": 0.0, "uk": 100.0, "eu_ex_uk": 0.0, "row": 0.0},
}

# Bond / gold ETFs are excluded from the equity-only regional denominator.
_NON_EQUITY_ETF_BASES: frozenset[str] = frozenset({"IGLT", "VGOV", "IBTA", "SGLN"})

# v5.5.1: distribution policy whitelists (LSE share classes, with/without ``.L``).
_KNOWN_DISTRIBUTING_ETFS: frozenset[str] = frozenset(
    {
        "VWRL", "VUSA", "IUSA", "ISF", "VUKE", "VERX", "VEUR", "IMEU",
        "VHYL", "IUKD", "IGLT", "VGOV", "VFEM", "MIDD",
    }
)
_KNOWN_ACCUMULATING_ETFS: frozenset[str] = frozenset(
    {
        "VWRP", "SWDA", "IWDA", "FWRG", "SSAC", "VUAG", "CSP1", "SPXP", "GSPX",
        "VERG", "VEUA", "SMEA", "CUKX", "VUKG", "CNX1", "EQGB", "IBTA", "IITU",
        "SMGB", "HEAL", "INRG", "EMIM", "WSML", "IWQU", "IWFQ", "IWMO", "VJPN", "SGLN",
    }
)
_LEVERAGED_ETP_TOKENS: tuple[str, ...] = (
    "2x", "3x", "-1x", "-2x", "-3x", "leveraged", "daily short",
    "inverse", "ultrashort", "covered call", "incomeshares",
)

# Underlying index family mapping for duplicate/clone ETF exposure aggregation
_ETF_INDEX_FAMILY_MAP: dict[str, str] = {
    # S&P 500 trackers
    "VUAG": "S&P 500 (美股标普500)",
    "VUAG.L": "S&P 500 (美股标普500)",
    "CSP1": "S&P 500 (美股标普500)",
    "CSP1.L": "S&P 500 (美股标普500)",
    "VUSA": "S&P 500 (美股标普500)",
    "VUSA.L": "S&P 500 (美股标普500)",
    "IUSA": "S&P 500 (美股标普500)",
    "IUSA.L": "S&P 500 (美股标普500)",
    "SPXP": "S&P 500 (美股标普500)",
    "SPXP.L": "S&P 500 (美股标普500)",
    "GSPX": "S&P 500 (美股标普500)",
    "GSPX.L": "S&P 500 (美股标普500)",
    "SPY": "S&P 500 (美股标普500)",
    "VOO": "S&P 500 (美股标普500)",
    "IVV": "S&P 500 (美股标普500)",
    # NASDAQ 100 trackers
    "CNX1": "NASDAQ 100 (美股纳指100)",
    "CNX1.L": "NASDAQ 100 (美股纳指100)",
    "EQGB": "NASDAQ 100 (美股纳指100)",
    "EQGB.L": "NASDAQ 100 (美股纳指100)",
    "EQQQ": "NASDAQ 100 (美股纳指100)",
    "EQQQ.L": "NASDAQ 100 (美股纳指100)",
    "XNAQ": "NASDAQ 100 (美股纳指100)",
    "XNAQ.L": "NASDAQ 100 (美股纳指100)",
    "QQQ": "NASDAQ 100 (美股纳指100)",
    "QQQM": "NASDAQ 100 (美股纳指100)",
    # All-World / Developed World broad trackers
    "VWRP": "Global All-World/Developed (全球宽基)",
    "VWRP.L": "Global All-World/Developed (全球宽基)",
    "VWRL": "Global All-World/Developed (全球宽基)",
    "VWRL.L": "Global All-World/Developed (全球宽基)",
    "SWDA": "Global All-World/Developed (全球宽基)",
    "SWDA.L": "Global All-World/Developed (全球宽基)",
    "IWDA": "Global All-World/Developed (全球宽基)",
    "IWDA.L": "Global All-World/Developed (全球宽基)",
    "FWRG": "Global All-World/Developed (全球宽基)",
    "FWRG.L": "Global All-World/Developed (全球宽基)",
    "SSAC": "Global All-World/Developed (全球宽基)",
    "SSAC.L": "Global All-World/Developed (全球宽基)",
    # FTSE 100 / UK Equity Dividend trackers
    "ISF": "FTSE 100 (英国富时100)",
    "ISF.L": "FTSE 100 (英国富时100)",
    "VUKE": "FTSE 100 (英国富时100)",
    "VUKE.L": "FTSE 100 (英国富时100)",
    "CUKX": "FTSE 100 (英国富时100)",
    "CUKX.L": "FTSE 100 (英国富时100)",
    "VUKG": "FTSE 100 (英国富时100)",
    "VUKG.L": "FTSE 100 (英国富时100)",
    # Europe broad trackers
    "VERG": "Europe Equity (欧洲宽基)",
    "VERG.L": "Europe Equity (欧洲宽基)",
    "VERX": "Europe Equity (欧洲宽基)",
    "VERX.L": "Europe Equity (欧洲宽基)",
    "VEUR": "Europe Equity (欧洲宽基)",
    "VEUR.L": "Europe Equity (欧洲宽基)",
    "VEUA": "Europe Equity (欧洲宽基)",
    "VEUA.L": "Europe Equity (欧洲宽基)",
    "IMEU": "Europe Equity (欧洲宽基)",
    "IMEU.L": "Europe Equity (欧洲宽基)",
    "SMEA": "Europe Equity (欧洲宽基)",
    "SMEA.L": "Europe Equity (欧洲宽基)",
    # UK Gilts
    "IGLT": "UK Gilts (英国国债)",
    "IGLT.L": "UK Gilts (英国国债)",
    "VGOV": "UK Gilts (英国国债)",
    "VGOV.L": "UK Gilts (英国国债)",
}


def normalize_issuer_root(ticker: str) -> str:
    """Normalize dual-class / shared-issuer tickers (e.g. GOOGL -> GOOG) for combined exposure checks."""
    clean = (ticker or "").strip().upper()
    return _ISSUER_ALIASES.get(clean, clean)


def infer_etf_index_family(
    symbol: str,
    bundle: TickerMarketBundle | None = None,
    name_hint: str = "",
) -> str | None:
    """Identify the underlying index family of an ETF to aggregate clone ETF risk exposure."""
    sym_clean = (symbol or "").strip().upper()
    combined_name = " ".join(
        filter(
            None,
            [
                name_hint,
                getattr(bundle, "short_name", "") if bundle else "",
                getattr(bundle, "long_name", "") if bundle else "",
            ],
        )
    ).lower()

    # v5.5.1: leveraged / inverse / covered-call ETPs must never inherit the broad-base
    # core privilege (e.g. "FTSE 100 3x Daily Leveraged" is NOT a core wide-base ETF).
    if any(token in combined_name for token in _LEVERAGED_ETP_TOKENS):
        return None

    if sym_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[sym_clean]
    base_clean = sym_clean.split(".")[0]
    if base_clean in _ETF_INDEX_FAMILY_MAP:
        return _ETF_INDEX_FAMILY_MAP[base_clean]

    is_etf = bool(bundle and bundle.quote_type.upper() == "ETF")
    if not is_etf and not any(k in combined_name for k in ("ucits", "etf", "vanguard", "ishares", "invesco", "spdr")):
        return None

    if "s&p 500" in combined_name or "s&p500" in combined_name or "sp500" in combined_name:
        return "S&P 500 (美股标普500)"
    if "nasdaq-100" in combined_name or "nasdaq 100" in combined_name:
        return "NASDAQ 100 (美股纳指100)"
    if "all-world" in combined_name or "msci world" in combined_name or "ftse all-world" in combined_name:
        return "Global All-World/Developed (全球宽基)"
    if "ftse 100" in combined_name:
        return "FTSE 100 (英国富时100)"
    if (
        "developed europe" in combined_name
        or "msci europe" in combined_name
        or "stoxx europe 600" in combined_name
        or "europe ex uk" in combined_name
    ):
        return "Europe Equity (欧洲宽基)"
    if "gilt" in combined_name:
        return "UK Gilts (英国国债)"
    return None


def classify_etf_distribution_policy(
    symbol: str,
    bundle: TickerMarketBundle | None = None,
) -> Literal["ACC", "DIST", "UNKNOWN"]:
    """Classify an ETF share class as accumulating / distributing / unknown.

    Taxable Invest accounts must only buy distributing classes: accumulating offshore
    funds trigger Excess Reportable Income (ERI) reporting friction in the UK.
    """
    sym_clean = (symbol or "").strip().upper()
    base_clean = sym_clean.split(".")[0]
    if base_clean in _KNOWN_DISTRIBUTING_ETFS or sym_clean in _KNOWN_DISTRIBUTING_ETFS:
        return "DIST"
    if base_clean in _KNOWN_ACCUMULATING_ETFS or sym_clean in _KNOWN_ACCUMULATING_ETFS:
        return "ACC"
    name = " ".join(
        filter(
            None,
            [
                str(getattr(bundle, "short_name", "") or ""),
                str(getattr(bundle, "long_name", "") or ""),
            ],
        )
    ).lower()
    if "(acc)" in name or "accumulating" in name or " acc " in f" {name} ":
        return "ACC"
    if "(dist)" in name or "distributing" in name or " dist " in f" {name} ":
        return "DIST"
    return "UNKNOWN"


def _regional_root(symbol: str) -> str:
    return (symbol or "").strip().upper().split(".")[0]


def _stock_region(symbol: str) -> str | None:
    """Map a single-name listing suffix to a coarse region (ETF-free helper)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    if sym.endswith(".L"):
        return "uk"
    if sym.endswith((".DE", ".PA", ".AS", ".MC", ".MI", ".SW")):
        return "eu_ex_uk"
    if "." not in sym:
        return "us"
    return None


def estimate_portfolio_regional_lookthrough(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None = None,
    *,
    include_pending_core_etfs: bool = True,
) -> dict[str, Any]:
    """Coarse US / UK / Europe-ex-UK split of the equity book (context only)."""
    bmap = bundles or {}
    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), 1.0)
    isa_val = max(float(portfolio.isa.total_account_value or 0.0), 1.0)

    region_values = {"us": 0.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0}
    isa_region_values = {"us": 0.0, "uk": 0.0, "eu_ex_uk": 0.0, "row": 0.0}
    unmapped_value = 0.0
    excluded_value = 0.0

    def _add(acct: str, sym: str, value: float) -> None:
        nonlocal unmapped_value, excluded_value
        base = _regional_root(sym)
        if base in _NON_EQUITY_ETF_BASES:
            excluded_value += value
            return
        b = bmap.get(sym) or bmap.get(sym.strip().upper())
        is_etf = bool((b is not None and str(b.quote_type).upper() == "ETF"))
        row = _ETF_REGIONAL_LOOKTHROUGH.get(base) if (is_etf or base in _ETF_REGIONAL_LOOKTHROUGH) else None
        if row is not None:
            for key in region_values:
                share = float(row.get(key, 0.0)) / 100.0 * value
                region_values[key] += share
                if acct == "ISA":
                    isa_region_values[key] += share
            return
        region = _stock_region(sym)
        if region is None:
            unmapped_value += value
            return
        region_values[region] += value
        if acct == "ISA":
            isa_region_values[region] += value

    for pos in portfolio.all_positions():
        if pos.current_value > 0:
            _add(str(pos.account or "").strip().upper(), pos.yahoo_symbol or pos.ticker, float(pos.current_value))
    if include_pending_core_etfs:
        for order in portfolio.isa.pending_orders:
            if str(order.side or "").strip().upper() != "BUY":
                continue
            base = _regional_root(order.yahoo_symbol or order.ticker)
            if base not in _ETF_REGIONAL_LOOKTHROUGH:
                continue
            ccy = str(order.currency or "").strip()
            raw_value = float(order.value or 0.0)
            if ccy in {"GBp", "GBx", "GBX"}:
                value = raw_value / 100.0
            elif ccy.upper() in {"GBP", ""}:
                value = raw_value
            else:
                # Non-GBP pending value cannot be converted without an FX basis; skip
                # rather than silently mis-scaling the look-through percentages.
                continue
            if value > 0:
                _add("ISA", order.yahoo_symbol or order.ticker, value)

    equity_total = sum(region_values.values())

    def _pct_of(value: float, denom: float) -> float:
        return round(value / denom * 100.0, 2) if denom > 0 else 0.0

    return {
        "us_value": round(region_values["us"], 2),
        "uk_value": round(region_values["uk"], 2),
        "eu_ex_uk_value": round(region_values["eu_ex_uk"], 2),
        "row_value": round(region_values["row"], 2),
        "equity_value_gbp": round(equity_total, 2),
        "excluded_bond_gold_value": round(excluded_value, 2),
        "unmapped_value": round(unmapped_value, 2),
        "us_pct_nav": _pct_of(region_values["us"], total_all),
        "uk_pct_nav": _pct_of(region_values["uk"], total_all),
        "eu_ex_uk_pct_nav": _pct_of(region_values["eu_ex_uk"], total_all),
        "isa_us_pct": _pct_of(isa_region_values["us"], isa_val),
        "isa_uk_pct": _pct_of(isa_region_values["uk"], isa_val),
        "isa_eu_ex_uk_pct": _pct_of(isa_region_values["eu_ex_uk"], isa_val),
        "equity_us_share_pct": _pct_of(region_values["us"], equity_total),
        "equity_uk_share_pct": _pct_of(region_values["uk"], equity_total),
        "equity_eu_ex_uk_share_pct": _pct_of(region_values["eu_ex_uk"], equity_total),
        "include_pending_core_etfs": bool(include_pending_core_etfs),
    }


def estimate_penetration_tech_exposure(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> dict[str, Any]:
    """Estimate combined tech-factor exposure: direct tech holdings + static ETF look-through.

    Values are approximations for context only — never used as a hard blocking threshold.
    """
    bmap = bundles or {}
    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), 1.0)

    direct_by_ticker: dict[str, float] = {}
    direct_val = 0.0
    etf_implied_val = 0.0
    unmapped_etf_val = 0.0
    etf_rows: list[dict[str, Any]] = []

    for p in portfolio.all_positions():
        if p.current_value <= 0:
            continue
        root = normalize_issuer_root((p.ticker or "").strip().upper())
        if root in _TECH_FACTOR_TICKERS:
            direct_val += p.current_value
            direct_by_ticker[root] = direct_by_ticker.get(root, 0.0) + p.current_value
            continue
        sym = (p.yahoo_symbol or "").strip().upper()
        pct = _ETF_TECH_PENETRATION_PCT.get(sym) or _ETF_TECH_PENETRATION_PCT.get(sym.split(".")[0])
        if pct is None:
            b = bmap.get(p.yahoo_symbol)
            if b is not None and b.quote_type.upper() == "ETF":
                unmapped_etf_val += p.current_value
            continue
        implied = p.current_value * pct / 100.0
        etf_implied_val += implied
        etf_rows.append(
            {
                "symbol": p.yahoo_symbol,
                "account": p.account,
                "weight_pct": pct,
                "implied_value": round(implied, 2),
            }
        )

    combined = direct_val + etf_implied_val
    return {
        "direct_value": round(direct_val, 2),
        "direct_positions": {k: round(v, 2) for k, v in direct_by_ticker.items()},
        "etf_implied_value": round(etf_implied_val, 2),
        "etf_breakdown": etf_rows,
        "unmapped_etf_value": round(unmapped_etf_val, 2),
        "combined_value": round(combined, 2),
        "combined_pct": round((combined / total_all) * 100.0, 1),
        "total_value": round(total_all, 2),
    }


def compute_portfolio_cvar_and_risk_attribution(
    portfolio: PortfolioData,
    histories: dict[str, pd.DataFrame],
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Compute PSD-shrunk portfolio annualized volatility, component risk contributions, and 95% historical 1d CVaR."""
    held_values: dict[str, float] = {}
    for pos in portfolio.all_positions():
        sym = (pos.yahoo_symbol or "").strip()
        if sym and pos.current_value > 0.0:
            held_values[sym] = held_values.get(sym, 0.0) + float(pos.current_value)

    if not held_values or not histories:
        return {}

    close_series: dict[str, pd.Series] = {}
    vol_1d: dict[str, float] = {}
    extract_days = max(int(lookback_days), 60) + 10
    for sym in held_values:
        cs = _extract_close_series(histories.get(sym), extract_days)
        if cs is None:
            continue
        # Clean 1-day log-return volatility WITHOUT cross-holiday ffill distortion
        r1 = np.log(cs / cs.shift(1)).tail(lookback_days).dropna()
        if len(r1) < 15:
            continue
        std_1d = float(np.std(r1.to_numpy(dtype=np.float64), ddof=1))
        if std_1d > 0.0:
            close_series[sym] = cs
            vol_1d[sym] = std_1d

    valid_syms = [s for s in held_values if s in close_series]
    if not valid_syms:
        return {}

    raw_w = np.array([held_values[s] for s in valid_syms], dtype=np.float64)
    w_sum = float(np.sum(raw_w))
    if w_sum <= 0.0:
        return {}
    w = raw_w / w_sum

    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(float(raw_total or 0.0), w_sum)

    px_df = pd.DataFrame({s: close_series[s] for s in valid_syms}).sort_index().ffill(limit=1)

    if len(valid_syms) == 1:
        sym0 = valid_syms[0]
        port_vol_pct = float(vol_1d[sym0] * np.sqrt(252.0) * 100.0)
        contributions = [
            {
                "symbol": sym0,
                "risk_pct": 100.0,
                "weight_pct": round((held_values[sym0] / total_all) * 100.0, 1),
                "invested_weight_pct": 100.0,
            }
        ]
    else:
        log3 = np.log(px_df / px_df.shift(3)).tail(lookback_days).dropna()
        n_dim = len(valid_syms)
        if len(log3) >= 20:
            arr3 = log3.to_numpy(dtype=np.float64)
            stds3 = np.std(arr3, axis=0, ddof=1)
            if np.any(stds3 <= 0.0):
                corr3 = np.eye(n_dim, dtype=np.float64)
            else:
                corr3 = np.corrcoef(arr3, rowvar=False)
                if corr3.ndim == 0:
                    corr3 = np.eye(n_dim, dtype=np.float64)
                else:
                    corr3 = np.nan_to_num(corr3, nan=0.0)
                    np.fill_diagonal(corr3, 1.0)
        else:
            corr3 = np.eye(n_dim, dtype=np.float64)

        shrinkage = 0.20
        corr_shrunk = (1.0 - shrinkage) * corr3 + shrinkage * np.eye(n_dim, dtype=np.float64)
        d_vec = np.array([vol_1d[s] for s in valid_syms], dtype=np.float64)
        cov_1d = np.outer(d_vec, d_vec) * corr_shrunk

        port_var = float(w @ cov_1d @ w)
        if port_var <= 0.0:
            return {}
        port_vol_pct = float(np.sqrt(port_var * 252.0) * 100.0)
        mrc_var = cov_1d @ w
        rc_pct_arr = (w * mrc_var) / port_var * 100.0

        contributions = []
        for idx, sym in enumerate(valid_syms):
            contributions.append(
                {
                    "symbol": sym,
                    "risk_pct": round(float(rc_pct_arr[idx]), 1),
                    "weight_pct": round((held_values[sym] / total_all) * 100.0, 1),
                    "invested_weight_pct": round(float(w[idx] * 100.0), 1),
                }
            )
        contributions.sort(key=lambda item: item["risk_pct"], reverse=True)

    # Historical 95% 1-day CVaR using aligned daily simple returns (excluding common no-move days)
    ret1_raw = px_df.pct_change(fill_method=None).tail(lookback_days).dropna(how="all")
    ret1_clean = ret1_raw.fillna(0.0)
    active_mask = (ret1_clean.abs().sum(axis=1) > 1e-12)
    ret1_clean = ret1_clean.loc[active_mask]

    cvar_95_pct: float | None = None
    cvar_95_gbp: float | None = None
    if len(ret1_clean) >= 20:
        port_rets = ret1_clean[valid_syms].to_numpy(dtype=np.float64) @ w
        q05 = float(np.quantile(port_rets, 0.05))
        tail = port_rets[port_rets <= q05]
        if len(tail) == 0:
            k = max(1, int(np.ceil(len(port_rets) * 0.05)))
            tail = np.sort(port_rets)[:k]
        cvar_val = float(np.mean(tail))
        cvar_95_pct = round(cvar_val * 100.0, 2)
        cvar_95_gbp = round(max(0.0, -cvar_val) * w_sum, 2)

    return {
        "portfolio_vol_pct": round(port_vol_pct, 2),
        "cvar_95_pct": cvar_95_pct,
        "cvar_95_gbp": cvar_95_gbp,
        "modeled_value_gbp": round(w_sum, 2),
        "risk_contributions": contributions,
        "top_risk_sources": contributions[:3],
    }


def compute_exposure_summary(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle],
    histories: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Deterministically compute cross-account duplicate issuers, clone ETF underlying index overlap, and sector weights."""
    by_issuer: dict[str, dict[str, float]] = {}
    by_etf_family: dict[str, list[tuple[str, str, float]]] = {}
    sector_vals: dict[str, float] = {}
    invest_sector_vals: dict[str, float] = {}

    raw_total = portfolio.total_combined_value or (
        portfolio.isa.total_account_value + portfolio.invest.total_account_value
    )
    total_all = max(raw_total, 1.0)
    invest_total = max(portfolio.invest.total_account_value, 1.0)

    for p in portfolio.all_positions():
        root = normalize_issuer_root(p.ticker)
        entry = by_issuer.setdefault(root, {"ISA": 0.0, "Invest": 0.0})
        entry[p.account] = entry.get(p.account, 0.0) + p.current_value

        b = bundles.get(p.yahoo_symbol)
        etf_family = infer_etf_index_family(p.yahoo_symbol, b, name_hint=p.name)
        is_etf = bool((b and b.quote_type.upper() == "ETF") or etf_family is not None)

        if etf_family:
            by_etf_family.setdefault(etf_family, []).append((p.account, p.ticker, p.current_value))

        if is_etf:
            sector = "ETF宽基/固收"
        elif b and b.sector:
            sector = b.sector
        else:
            sector = "Other/未分类"

        sector_vals[sector] = sector_vals.get(sector, 0.0) + p.current_value
        if p.account == "Invest":
            invest_sector_vals[sector] = invest_sector_vals.get(sector, 0.0) + p.current_value

    overlap_lines = []
    for root, accts in by_issuer.items():
        if accts.get("ISA", 0.0) > 0 and accts.get("Invest", 0.0) > 0:
            comb = accts["ISA"] + accts["Invest"]
            pct = (comb / total_all) * 100.0
            overlap_lines.append(
                f"• 跨账户重叠 {root}: ISA £{accts['ISA']:,.0f} + Invest £{accts['Invest']:,.0f} = 合计 £{comb:,.0f} (占总资产 {pct:.1f}%)"
            )

    clone_etf_lines = []
    for family, items in by_etf_family.items():
        distinct_tickers = {t for _, t, _ in items}
        if len(items) >= 2 and (len(distinct_tickers) >= 2 or len({a for a, _, _ in items}) >= 2):
            comb = sum(val for _, _, val in items)
            pct = (comb / total_all) * 100.0
            legs = " + ".join(f"{acct} {tkr} £{val:,.0f}" for acct, tkr, val in items)
            clone_etf_lines.append(
                f"• 克隆ETF底层暴露归并 [{family}]: {legs} = 合计 £{comb:,.0f} (占总资产 {pct:.1f}%)"
            )

    top_sectors = sorted(sector_vals.items(), key=lambda x: x[1], reverse=True)[:5]
    sector_str = " | ".join(
        f"{sec} {(val / total_all) * 100.0:.1f}%" for sec, val in top_sectors
    )
    top_inv_sectors = sorted(invest_sector_vals.items(), key=lambda x: x[1], reverse=True)[:3]
    inv_sector_str = " | ".join(
        f"{sec} {(val / invest_total) * 100.0:.1f}%" for sec, val in top_inv_sectors
    )

    lines = [
        f"• 总资产行业分布(前5): {sector_str or 'N/A'}",
        f"• Invest短线账户行业集中度(前3): {inv_sector_str or 'N/A'}",
    ]
    if overlap_lines:
        lines.extend(overlap_lines)
    else:
        lines.append("• 跨账户同名重叠: 无")
    if clone_etf_lines:
        lines.extend(clone_etf_lines)
    penetration = estimate_penetration_tech_exposure(portfolio, bundles)
    if penetration["combined_value"] > 0:
        lines.append(
            f"• 穿透后真实科技暴露(估算): ETF 隐含 £{penetration['etf_implied_value']:,.0f} + "
            f"直接科技个股 £{penetration['direct_value']:,.0f} = £{penetration['combined_value']:,.0f} "
            f"(占总资产 {penetration['combined_pct']:.1f}%；静态权重近似，非实时持仓穿透)"
        )
    if penetration.get("unmapped_etf_value", 0.0) > 0:
        lines.append(
            f"• 另有 £{penetration['unmapped_etf_value']:,.0f} ETF 未映射科技权重（未静默计入）"
        )
    regional = estimate_portfolio_regional_lookthrough(portfolio, bundles)
    if regional["equity_value_gbp"] > 0:
        lines.append(
            f"• 权益区域穿透(估算): 美股 {regional['equity_us_share_pct']:.1f}% | "
            f"英国 {regional['equity_uk_share_pct']:.1f}% | 欧陆(除英) {regional['equity_eu_ex_uk_share_pct']:.1f}%"
        )
    if histories:
        risk_info = compute_portfolio_cvar_and_risk_attribution(portfolio, histories)
        top_sources = risk_info.get("top_risk_sources") or []
        if top_sources:
            top_str = " | ".join(
                f"{item['symbol']} 占总风险 {item['risk_pct']:.0f}% (市值占 {item['weight_pct']:.0f}%)"
                for item in top_sources
            )
            lines.append(f"• 组合真实风险归因(前3波动源): {top_str}")
        cvar_pct = risk_info.get("cvar_95_pct")
        cvar_gbp = risk_info.get("cvar_95_gbp")
        if cvar_pct is not None and cvar_gbp is not None:
            lines.append(f"• 尾部压力测试(95%单日CVaR): 极端日均潜在回撤 {cvar_pct:+.2f}% (约 £{cvar_gbp:,.0f})")
    return "\n".join(lines)
