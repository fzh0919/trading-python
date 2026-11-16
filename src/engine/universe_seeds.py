"""Multi-market index constituent seeds & polite 30-day cached online sync (0 external deps)."""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Curated LSE GBP UCITS ETFs (0% FX fee, 0% UK stamp duty)
CURATED_LSE_ETFS: tuple[str, ...] = (
    "VWRP.L",
    "VUAG.L",
    "SWDA.L",
    "EQGB.L",
    "VERG.L",
    "VHYL.L",
    "IUKD.L",
    "CSP1.L",
    "CNX1.L",
    "IWQU.L",
    "IGLT.L",
    "ISF.L",
    "EMIM.L",
    "WSML.L",
    "SGLN.L",
    "INRG.L",
    "SMGB.L",
    "HEAL.L",
    "IITU.L",
    "VJPN.L",
    "VFEM.L",
    "MIDD.L",
    "VGOV.L",
    "IWFQ.L",
    "IWMO.L",
)

# Verified Yahoo Finance constituent seeds (offline fallback & immediate baseline)
SP500_SEED: tuple[str, ...] = (
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB", "AKAM",
    "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE",
    "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO",
    "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP",
    "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX", "BRK-B", "BBY", "TECH",
    "BIIB", "BLK", "BX", "BK", "BA", "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR",
    "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR",
    "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX",
    "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO",
    "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY",
    "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DVA",
    "DAY", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR", "D",
    "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL",
    "EIX", "EW", "EA", "ELV", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX",
    "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR",
    "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FI",
    "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV",
    "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL",
    "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON",
    "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW",
    "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG", "INTU", "ISRG", "IVZ", "INVH",
    "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "K", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
    "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW",
    "LULU", "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC",
    "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA",
    "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI",
    "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS",
    "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON",
    "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PH", "PAYX", "PAYC", "PYPL",
    "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG",
    "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "RL",
    "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP",
    "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS",
    "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK",
    "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL",
    "TDY", "TER", "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TSCO", "TT", "TDG",
    "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS",
    "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI",
    "V", "VST", "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT",
    "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL",
    "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
)

NASDAQ100_SEED: tuple[str, ...] = (
    "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI", "ANSS", "AAPL",
    "AMAT", "APP", "ARM", "ASML", "AZN", "TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB",
    "BKNG", "AVGO", "CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", "CEG",
    "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DASH", "EA", "EXC",
    "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP",
    "KLAC", "KHC", "LRCX", "LIN", "LULU", "MAR", "MRVL", "MELI", "META", "MCHP", "MU",
    "MSFT", "MSTR", "MDLZ", "MNST", "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR",
    "PLTR", "PANW", "PAYX", "PYPL", "PDD", "QCOM", "REGN", "ROP", "ROST", "SHOP", "SBUX",
    "SNPS", "TTWO", "TMUS", "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
)

FTSE100_SEED: tuple[str, ...] = (
    "AAF.L", "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ALW.L", "ANTO.L", "AUTG.L", "AV.L", "AZN.L",
    "BA.L", "BAB.L", "BARC.L", "BATS.L", "BEZ.L", "BKG.L", "BLND.L", "BME.L", "BNZL.L", "BP.L",
    "BRBY.L", "BT-A.L", "CCH.L", "CNA.L", "CPG.L", "CRH.L", "CTEC.L", "DCC.L", "DGE.L", "DPLM.L",
    "EDV.L", "ENT.L", "EXPN.L", "EZJ.L", "FCIT.L", "FLTR.L", "FRES.L", "GAW.L", "GLEN.L", "GSK.L",
    "HIK.L", "HLMA.L", "HLN.L", "HSBA.L", "HSX.L", "HWDN.L", "IAG.L", "ICG.L", "IHG.L", "III.L",
    "IMB.L", "IMI.L", "INF.L", "ITRK.L", "JD.L", "KGF.L", "LAND.L", "LGEN.L", "LLOY.L", "LMP.L",
    "LSEG.L", "MKS.L", "MNDI.L", "MNG.L", "MRO.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L", "PHNX.L",
    "PRU.L", "PSH.L", "PSN.L", "PSON.L", "REL.L", "RIO.L", "RKT.L", "RMV.L", "RR.L", "RS1.L",
    "RTO.L", "SBRY.L", "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L", "SMIN.L", "SMT.L", "SN.L",
    "SPX.L", "SSE.L", "STAN.L", "STJ.L", "SVT.L", "TSCO.L", "TW.L", "ULVR.L", "UTG.L", "UU.L",
    "VOD.L", "WEIR.L", "WPP.L", "WTB.L",
)

# High-liquidity FTSE 250 mid-cap compounders & UK sector leaders
FTSE250_SEED: tuple[str, ...] = (
    "ABDN.L", "AGK.L", "AJB.L", "AO.L", "APAX.L", "ASHM.L", "ASL.L", "ATST.L", "ATT.L", "AVON.L",
    "BAG.L", "BBOX.L", "BBY.L", "BCG.L", "BCPT.L", "BHMG.L", "BIG.L", "BOY.L", "BREE.L", "BRK.L",
    "BRSC.L", "BYG.L", "BYIT.L", "CAPC.L", "CBG.L", "CCC.L", "CCL.L", "CCR.L", "CHRY.L", "CKN.L",
    "CLDN.L", "CMCX.L", "COA.L", "CRST.L", "CURY.L", "CWK.L", "DARK.L", "DEC.L", "DLN.L", "DOCS.L",
    "DOM.L", "DRX.L", "DSCV.L", "ECM.L", "ELM.L", "EMG.L", "ENOG.L", "ERNS.L", "ESNT.L", "FAN.L",
    "FGP.L", "FORT.L", "FRAS.L", "FUTR.L", "FXPO.L", "GAMA.L", "GEN.L", "GFTU.L", "GNS.L", "GPE.L",
    "GRG.L", "HAS.L", "HBR.L", "HEAD.L", "HFG.L", "HGT.L", "HICL.L", "HILS.L", "HOC.L", "HTG.L",
    "IBST.L", "ICP.L", "IDS.L", "IGG.L", "INCH.L", "INDV.L", "INPP.L", "INVP.L", "IPX.L", "ITV.L",
    "IWG.L", "JAM.L", "JMAT.L", "JMG.L", "JUP.L", "KLR.L", "KNOS.L", "LMP.L", "LRE.L", "LXI.L",
    "MARS.L", "MCRO.L", "MGAM.L", "MGNS.L", "MNKS.L", "MONY.L", "MOON.L", "MPE.L", "MRC.L", "MSLH.L",
    "MTO.L", "MYI.L", "NCC.L", "N91.L", "NET.L", "NEX.L", "OXIG.L", "PAGE.L", "PETS.L", "PIN.L",
    "PLUS.L", "PNN.L", "POLR.L", "PTEC.L", "QLT.L", "QQ.L", "RAT.L", "RCP.L", "RDW.L", "RHIM.L",
    "RNK.L", "ROR.L", "RSW.L", "SAFE.L", "SCT.L", "SDY.L", "SFOR.L", "SHB.L", "SHI.L", "SMDS.L",
    "SMWH.L", "SNR.L", "SPI.L", "SPT.L", "SRP.L", "SSPG.L", "SVS.L", "SYNT.L", "TATE.L", "TBCG.L",
    "TEP.L", "THG.L", "TMPL.L", "TPK.L", "TRIG.L", "TRN.L", "TRY.L", "UKW.L", "VCT.L", "VTY.L",
    "WAG.L", "WG.L", "WHR.L", "WIZZ.L", "WKP.L", "WTAN.L", "XPS.L",
)

# Euro Stoxx 50 / DAX / CAC / AEX blue chips
EUROSTOXX50_SEED: tuple[str, ...] = (
    "ADS.DE", "ADYEN.AS", "AD.AS", "AI.PA", "AIR.PA", "ALV.DE", "AMS.MC", "ABI.BR", "ASML.AS",
    "CS.PA", "BAS.DE", "BAYN.DE", "BBVA.MC", "SAN.MC", "BMW.DE", "BNP.PA", "BN.PA", "DB1.DE",
    "DHL.DE", "DTE.DE", "ENEL.MI", "ENI.MI", "EL.PA", "RACE.MI", "FLTR.L", "RMS.PA", "IBE.MC",
    "ITX.MC", "IFX.DE", "INGA.AS", "ISP.MI", "KER.PA", "OR.PA", "MC.PA", "MBG.DE", "MRK.DE",
    "MUV2.DE", "NDA-FI.HE", "PRX.AS", "SAF.PA", "SGO.PA", "SAN.PA", "SAP.DE", "SU.PA", "SIE.DE",
    "STLAM.MI", "TTE.PA", "DG.PA", "VOW3.DE", "WKL.AS", "NOVO-B.CO",
)

INDEX_SEED_MAP: dict[str, tuple[str, ...]] = {
    "SP500": SP500_SEED,
    "NASDAQ100": NASDAQ100_SEED,
    "FTSE100": FTSE100_SEED,
    "FTSE250": FTSE250_SEED,
    "EUROSTOXX50": EUROSTOXX50_SEED,
}


class _WikiTableParser(HTMLParser):
    """Minimal dependency-free HTML table parser extracting rows of cell text."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._table_classes: list[str] = []
        self._cur_table: list[list[str]] = []
        self._cur_row: list[str] = []
        self._cur_cell: list[str] = []
        self.tables: list[tuple[str, list[list[str]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "table":
            self._in_table = True
            self._table_classes.append(str(attr_map.get("class") or ""))
            self._cur_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._cur_row = []
        elif self._in_row and tag in ("td", "th"):
            self._in_cell = True
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            text = re.sub(r"\s+", " ", "".join(self._cur_cell)).strip()
            self._cur_row.append(text)
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._cur_row:
                self._cur_table.append(self._cur_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            cls_name = self._table_classes.pop() if self._table_classes else ""
            if self._cur_table:
                self.tables.append((cls_name, self._cur_table))

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_cell.append(data)


def _normalize_wiki_ticker(raw: str, suffix: str = "") -> str:
    clean = re.sub(r"\[[^\]]*\]", "", str(raw or "")).strip().upper()
    if not clean:
        return ""
    if ":" in clean:
        clean = clean.split(":")[-1].strip()
    clean = clean.replace(" ", "")
    if not suffix:
        # US dual-class convention on Yahoo Finance: BRK.B -> BRK-B
        clean = clean.replace(".", "-")
    else:
        if clean.endswith("."):
            clean = clean[:-1] + "-"
        if not clean.endswith(suffix.upper()):
            clean = f"{clean}{suffix.upper()}"
    if not re.fullmatch(r"[A-Z0-9-]{1,12}(?:\.[A-Z]{1,4})?", clean):
        return ""
    return clean


def _parse_wikipedia_constituents(html_text: str, index_name: str) -> list[str]:
    """Extract Yahoo-compatible tickers from a Wikipedia index constituents page."""
    parser = _WikiTableParser()
    parser.feed(html_text)
    suffix = ".L" if index_name in ("FTSE100", "FTSE250") else ""

    for cls_name, rows in parser.tables:
        if len(rows) < 25:
            continue
        if "wikitable" not in cls_name.lower() and len(rows) < 40:
            continue
        header = [h.lower() for h in rows[0]]
        ticker_col: int | None = None
        for idx, col in enumerate(header):
            if any(k in col for k in ("symbol", "ticker", "epic")):
                ticker_col = idx
                break
        if ticker_col is None:
            continue

        extracted: list[str] = []
        for row in rows[1:]:
            if len(row) <= ticker_col:
                continue
            sym = _normalize_wiki_ticker(row[ticker_col], suffix=suffix)
            if sym:
                extracted.append(sym)
        if extracted:
            return list(dict.fromkeys(extracted))
    return []


_WIKI_INDEX_SOURCES: dict[str, tuple[str, int]] = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 450),
    "NASDAQ100": ("https://en.wikipedia.org/wiki/Nasdaq-100", 90),
    "FTSE100": ("https://en.wikipedia.org/wiki/FTSE_100_Index", 85),
    "FTSE250": ("https://en.wikipedia.org/wiki/FTSE_250_Index", 180),
}


async def _fetch_online_index_constituents(
    index_name: str,
    *,
    timeout_sec: float = 12.0,
) -> list[str] | None:
    src = _WIKI_INDEX_SOURCES.get(index_name)
    if not src:
        return None
    url, min_expected = src
    headers = {
        "User-Agent": "TradingAdvisorBot/5.2 (+https://github.com/homelab/trading-python; research cache)"
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
        if resp.status_code != 200 or not resp.text:
            logger.debug("Online index sync %s returned HTTP %s", index_name, resp.status_code)
            return None
        parsed = _parse_wikipedia_constituents(resp.text, index_name)
        if len(parsed) >= min_expected:
            return parsed
        logger.debug("Online index sync %s parsed too few tickers (%d < %d)", index_name, len(parsed), min_expected)
    except Exception as exc:
        logger.debug("Online index sync failed for %s: %s", index_name, exc)
    return None


async def resolve_base_universe_symbols(
    cfg: Any,
    db: Any = None,
    *,
    sync_online: bool = False,
) -> dict[str, Any]:
    """Assemble deduplicated multi-market equity base pool + curated LSE ETF list."""
    universe_cfg = getattr(cfg, "universe", cfg)
    enabled_indices = list(getattr(universe_cfg, "audition_indices", None) or INDEX_SEED_MAP.keys())
    do_online = bool(sync_online and getattr(universe_cfg, "audition_sync_online_indices", True))

    equities_sources: dict[str, list[str]] = {}
    counts_by_index: dict[str, int] = {}

    for idx_pos, raw_idx in enumerate(enabled_indices):
        idx_name = str(raw_idx or "").strip().upper()
        if not idx_name:
            continue
        syms: list[str] | None = None
        if db is not None and hasattr(db, "get_index_constituents"):
            try:
                syms = await db.get_index_constituents(idx_name, max_age_days=30)
            except Exception:
                syms = None

        if syms is None and do_online and idx_name in _WIKI_INDEX_SOURCES:
            if idx_pos > 0:
                await asyncio.sleep(1.2)
            online_syms = await _fetch_online_index_constituents(idx_name)
            if online_syms:
                syms = online_syms
                if db is not None and hasattr(db, "save_index_constituents"):
                    try:
                        await db.save_index_constituents(
                            idx_name,
                            online_syms,
                            source_url=_WIKI_INDEX_SOURCES[idx_name][0],
                        )
                    except Exception:
                        pass

        if not syms:
            syms = list(INDEX_SEED_MAP.get(idx_name, ()))

        counts_by_index[idx_name] = len(syms)
        for sym in syms:
            clean = str(sym or "").strip().upper()
            if clean:
                equities_sources.setdefault(clean, [])
                if idx_name not in equities_sources[clean]:
                    equities_sources[clean].append(idx_name)

    # Also keep user-configured config.yaml fallback equities in the audition candidate base
    configured_etfs = set(getattr(universe_cfg, "etf_symbols", []) or ())
    etfs = list(dict.fromkeys([*configured_etfs, *CURATED_LSE_ETFS]))
    etf_set = set(etfs)

    for fallback_sym in [
        *(getattr(universe_cfg, "isa_long_term", []) or ()),
        *(getattr(universe_cfg, "invest_tactical", []) or ()),
        *(getattr(cfg, "watchlist", []) or ()),
    ]:
        clean = str(fallback_sym or "").strip().upper()
        if not clean:
            continue
        if clean in etf_set:
            continue
        if clean not in equities_sources:
            equities_sources[clean] = ["CONFIG_SEED"]

    return {
        "equities": equities_sources,
        "etfs": etfs,
        "counts_by_index": counts_by_index,
    }
