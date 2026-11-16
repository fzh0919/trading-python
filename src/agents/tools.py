"""Deterministic callable tools & Google Search Grounding helpers for Gemini Agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from zoneinfo import ZoneInfo

from google.genai import types

from src.data_providers.analytics import (
    compute_correlation_matrix,
    compute_technical_snapshot,
    format_technical_line,
    max_correlation_against_portfolio,
    resolve_market_benchmark,
)
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import PortfolioData

if TYPE_CHECKING:
    from src.agents.llm_client import LLMBackend
    from src.ops_notifier import OpsNotifier
    from src.storage.db import AdvisorDatabase

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

_EU_SUFFIXES = (".DE", ".AS", ".PA", ".SW", ".MI", ".MC", ".BR", ".VI", ".ST", ".CO", ".HE", ".OL")
# cache_key -> (stored_at_monotonic, expires_at_monotonic, output)
_GROUNDING_CACHE: dict[str, tuple[float, float, str]] = {}
_GROUNDING_CACHE_MAX_ENTRIES = 64
_DEFAULT_GROUNDING_CONCURRENCY = 3
_DEFAULT_PERSIST_TTL_SECONDS = 10800
_INSPECT_TOOL_MAX_CHARS = 3800
_TICKER_TOKEN_RE = re.compile(r"\b([A-Z]{1,6}(?:\.[A-Z]{1,3}|-[A-Z])?)\b")


def _monotonic() -> float:
    return time.monotonic()


def _prune_grounding_cache(now: float) -> None:
    """Evict expired entries (and oldest if over capacity) to prevent memory growth."""
    expired = [k for k, (_, expires_at, _) in _GROUNDING_CACHE.items() if now >= expires_at]
    for k in expired:
        _GROUNDING_CACHE.pop(k, None)
    while len(_GROUNDING_CACHE) > _GROUNDING_CACHE_MAX_ENTRIES:
        oldest_key = min(_GROUNDING_CACHE, key=lambda k: _GROUNDING_CACHE[k][0])
        _GROUNDING_CACHE.pop(oldest_key, None)


def clear_grounding_cache() -> None:
    """Clear in-memory Google Search Grounding cache (useful for tests)."""
    _GROUNDING_CACHE.clear()


def _classify_symbol_region(symbol: str, quote_type: str = "") -> str:
    upper = symbol.strip().upper()
    if upper.endswith(".L"):
        if quote_type.upper() == "ETF":
            return "UK_ETF"
        return "UK"
    if upper.endswith(_EU_SUFFIXES):
        return "EU"
    return "US"


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    if not items:
        return []
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


_GROUNDING_OUTPUT_CONTRACT = (
    "Output strictly in compact Simplified Chinese plain text (no Markdown '#' or '**'): "
    "for tickers with verified material catalysts, write 1-2 factual bullets each "
    "(include exact date, key figures, source/analyst firm, and bull/bear impact). "
    "For tickers with no material breaking news, regulatory filings, or rating changes "
    "(including routine ETF NAV updates), do NOT write individual sentences; "
    "merge them into a single trailing line formatted exactly as: 近期无重大新公告：SYM1, SYM2."
)


async def _single_search_query(
    *,
    llm: LLMBackend,
    model: str,
    prompt: str,
    usage_label: str,
    thinking_level: str = "LOW",
) -> str:
    return await llm.generate(
        model=model,
        prompt=prompt,
        system_instruction=(
            "You are a real-time financial news and regulatory filings researcher covering US, UK (LSE), and European equities/ETFs. "
            "Use Google Search to query primary English financial and regulatory sources (RNS announcements, SEC filings, "
            "European exchange disclosures, earnings dates, guidance revisions, M&A/litigation, buybacks/dividends, and institutional rating changes). "
            "Never fabricate news, pad with known static NAV/closing prices, or write generic commentary. "
            + _GROUNDING_OUTPUT_CONTRACT
        ),
        thinking_level=thinking_level,
        use_google_search=True,
        temperature=0.1,
        usage_label=usage_label,
    )


async def run_forced_market_grounding(
    *,
    llm: LLMBackend | None,
    search_model: str,
    scope: str,
    symbols: list[str],
    quote_types: dict[str, str] | None = None,
    focus_query: str = "",
    ttl_seconds: int = 900,
    ops_notifier: OpsNotifier | None = None,
    max_concurrent_searches: int = _DEFAULT_GROUNDING_CONCURRENCY,
    priority_symbols: list[str] | None = None,
    priority_search_model: str = "",
    priority_thinking_level: str = "LOW",
    db: AdvisorDatabase | None = None,
    persist_ttl_seconds: int = _DEFAULT_PERSIST_TTL_SECONDS,
    bypass_cache: bool = False,
    per_symbol_out: dict[str, str] | None = None,
) -> str:
    """Run grouped concurrent Google Search Grounding across UK/EU/US equities, ETFs, and macro news."""
    if llm is None:
        return ""

    clean_syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    prio_ordered = list(dict.fromkeys(s.strip().upper() for s in (priority_symbols or ()) if s and s.strip()))
    prio_set = set(prio_ordered)
    grouped_syms = [s for s in clean_syms if s not in prio_set]

    cache_key = (
        f"{scope}:{','.join(sorted(clean_syms))}:{','.join(sorted(prio_set))}:{focus_query.strip().lower()}"
    )
    now = _monotonic()
    _prune_grounding_cache(now)
    if not bypass_cache:
        cached = _GROUNDING_CACHE.get(cache_key)
        if cached and ttl_seconds > 0 and (now - cached[0]) < ttl_seconds and now < cached[1]:
            return cached[2]

    qtypes = quote_types or {}
    uk_syms: list[str] = []
    eu_syms: list[str] = []
    us_syms: list[str] = []
    for sym in grouped_syms:
        region = _classify_symbol_region(sym, qtypes.get(sym, ""))
        if region in {"UK", "UK_ETF"}:
            uk_syms.append(sym)
        elif region == "EU":
            eu_syms.append(sym)
        else:
            us_syms.append(sym)

    uk_date = datetime.now(timezone.utc).astimezone(LONDON_TZ).date().isoformat()
    # (title, prompt, db_cache_key, task_symbols, is_priority)
    tasks_meta: list[tuple[str, str, str, list[str], bool]] = []
    usage_label = f"{scope}_grounding"

    if scope == "report":
        tasks_meta.append(
            (
                "【宏观与英/欧/美今日突发主线】",
                "Search Google for key macro catalysts and major policy/rate developments over the past 48 hours affecting the UK (FTSE/BoE/GBP), "
                "Europe (ECB/Eurozone), and the US (Fed/Treasury yields/US pre-market & after-hours movers). "
                "Summarize the verified facts concisely in Simplified Chinese within 240 Chinese characters (plain text, no '#' or '**').",
                f"grounding:{uk_date}:macro",
                [],
                False,
            )
        )
    elif focus_query.strip():
        fq_norm = focus_query.strip().lower()
        tasks_meta.append(
            (
                f"【指定方向实时检索：{focus_query.strip()}】",
                f"Search Google for the latest breaking catalysts, sector demand/supply data, and regulatory risks around '{focus_query.strip()}' "
                "across US equities, UK equities/LSE ETFs, and European equities. "
                "Summarize the verified facts concisely in Simplified Chinese within 260 Chinese characters (plain text, no '#' or '**').",
                f"grounding:{uk_date}:focus:{fq_norm}",
                [],
                False,
            )
        )

    for sym in prio_ordered:
        region = _classify_symbol_region(sym, qtypes.get(sym, ""))
        if region in {"UK", "UK_ETF"}:
            p_prompt = (
                f"Perform a 1-to-1 Google Search deep dive for London Stock Exchange ticker {sym}: "
                "check RNS regulatory announcements, trading updates, earnings/dividend/ex-dividend news, sudden price-move drivers, "
                "and recent analyst rating changes over the past 7 days. "
                "Output verified dates and key numbers concisely in Simplified Chinese within 180 Chinese characters."
            )
        elif region == "EU":
            p_prompt = (
                f"Perform a 1-to-1 Google Search deep dive for European ticker {sym}: "
                "check exchange regulatory filings, earnings guidance, order/contract or regulatory news, sudden move drivers, "
                "and analyst rating changes over the past 7 days. "
                "Output verified dates and key numbers concisely in Simplified Chinese within 180 Chinese characters."
            )
        else:
            p_prompt = (
                f"Perform a 1-to-1 Google Search deep dive for US stock/ETF {sym}: "
                "check overnight breaking news, pre-market/after-hours move drivers, SEC filings, latest earnings guidance/estimates, "
                "dividends/buybacks, and analyst rating changes over the past 72 hours. "
                "Output verified dates and key numbers concisely in Simplified Chinese within 180 Chinese characters."
            )
        tasks_meta.append(
            (f"【重点标的1对1深搜 · {sym}】", p_prompt, f"grounding:{uk_date}:prio:{sym}", [sym], True)
        )

    batch_size = 3
    region_specs = [
        (
            uk_syms,
            "UK",
            "英股与 LSE ETF 联网新闻",
            "Use Google Search to check the following London Stock Exchange equities/ETFs over the past 7 days for RNS regulatory filings, "
            "trading updates, earnings or dividend/ex-dividend news, analyst rating changes, and material sector catalysts: {syms}. "
            + _GROUNDING_OUTPUT_CONTRACT,
        ),
        (
            eu_syms,
            "EU",
            "欧股联网新闻",
            "Use Google Search to check the following European equities over the past 7 days for regulatory filings, "
            "earnings guidance, order/regulatory news, and institutional rating changes: {syms}. "
            + _GROUNDING_OUTPUT_CONTRACT,
        ),
        (
            us_syms,
            "US",
            "美股与全球 ETF 联网新闻",
            "Use Google Search to check the following US equities/ETFs over the past 48-72 hours for overnight breaking news, "
            "pre-market/after-hours move drivers, SEC filings, earnings guidance revisions, litigation/buybacks, and analyst ratings: {syms}. "
            + _GROUNDING_OUTPUT_CONTRACT,
        ),
    ]
    for region_syms, region_code, title_base, prompt_tpl in region_specs:
        for idx, chunk in enumerate(_chunk_list(region_syms, batch_size), 1):
            suffix = f" #{idx}" if len(region_syms) > batch_size else ""
            batch_key = f"grounding:{uk_date}:batch:{region_code}:{','.join(sorted(chunk))}"
            tasks_meta.append(
                (f"【{title_base}{suffix}】", prompt_tpl.format(syms=", ".join(chunk)), batch_key, list(chunk), False)
            )

    if not tasks_meta:
        return ""

    db_cached: dict[str, str] = {}
    if db is not None and not bypass_cache and persist_ttl_seconds > 0 and hasattr(db, "get_grounding_news_cache"):
        try:
            db_cached = await db.get_grounding_news_cache(
                [item[2] for item in tasks_meta],
                max_age_seconds=persist_ttl_seconds,
            )
        except Exception as exc:
            logger.debug("Failed loading grounding_news_cache: %s", exc)
            db_cached = {}

    prio_search_model = (priority_search_model or "").strip() or search_model
    prio_think_lvl = (priority_thinking_level or "LOW").strip() or "LOW"
    sem = asyncio.Semaphore(max(1, max_concurrent_searches))

    async def _bounded_search(prompt: str, *, use_priority_model: bool) -> str:
        chosen_model = prio_search_model if use_priority_model else search_model
        chosen_think = prio_think_lvl if use_priority_model else "LOW"
        async with sem:
            return await _single_search_query(
                llm=llm,
                model=chosen_model,
                prompt=prompt,
                usage_label=usage_label,
                thinking_level=chosen_think,
            )

    missing_indices = [idx for idx, item in enumerate(tasks_meta) if item[2] not in db_cached]
    raw_by_idx: dict[int, Any] = {}
    if missing_indices:
        fetched_results = await asyncio.gather(
            *(
                _bounded_search(
                    tasks_meta[idx][1],
                    use_priority_model=(
                        tasks_meta[idx][4]
                        or tasks_meta[idx][2].endswith(":macro")
                    ),
                )
                for idx in missing_indices
            ),
            return_exceptions=True,
        )
        for idx, res in zip(missing_indices, fetched_results):
            raw_by_idx[idx] = res

    sections: list[str] = []
    errors: list[str] = []
    db_entries_to_save: list[dict[str, Any]] = []
    for idx, (title, _, t_key, t_syms, is_prio) in enumerate(tasks_meta):
        if t_key in db_cached:
            text = db_cached[t_key].strip()
        else:
            res = raw_by_idx.get(idx)
            if isinstance(res, Exception):
                errors.append(f"{title}: {type(res).__name__}")
                continue
            text = str(res or "").strip()
            if not text or text.startswith("⚠️"):
                errors.append(f"{title}: {text[:120] or 'empty'}")
                continue
            if db is not None and persist_ttl_seconds > 0 and hasattr(db, "upsert_grounding_news_cache"):
                db_entries_to_save.append(
                    {
                        "cache_key": t_key,
                        "scope": scope,
                        "symbol": t_syms[0] if len(t_syms) == 1 else "",
                        "summary_text": text,
                    }
                )
                for s in t_syms:
                    sym_key = f"grounding:{uk_date}:sym:{s}"
                    if is_prio or sym_key != t_key:
                        db_entries_to_save.append(
                            {
                                "cache_key": sym_key,
                                "scope": scope,
                                "symbol": s,
                                "summary_text": text,
                            }
                        )
        sections.append(f"{title}\n{text}")
        if per_symbol_out is not None and t_syms:
            for s in t_syms:
                if is_prio or s not in per_symbol_out:
                    per_symbol_out[s] = text

    if db_entries_to_save and db is not None and hasattr(db, "upsert_grounding_news_cache"):
        try:
            await db.upsert_grounding_news_cache(db_entries_to_save)
        except Exception as exc:
            logger.debug("Failed saving grounding_news_cache: %s", exc)

    if errors and ops_notifier is not None:
        sev = "CRITICAL" if not sections else "WARN"
        await ops_notifier.notify(
            severity=sev,
            source=f"grounding:{scope}",
            alert_key=f"grounding_error:{scope}",
            message=f"Google Search Grounding 部分或全部失败（成功 {len(sections)}/{len(tasks_meta)} 批）：{'; '.join(errors[:3])}",
        )

    output = "\n\n".join(sections)
    if output and ttl_seconds > 0:
        write_now = _monotonic()
        _GROUNDING_CACHE[cache_key] = (write_now, write_now + ttl_seconds, output)
        _prune_grounding_cache(write_now)
    return output


def _format_us_deep_intel(intel: dict[str, Any]) -> str:
    """Format Finnhub US deep intel (recommendations, EPS surprises, insider MSPR) into a compact string."""
    if not intel:
        return ""
    parts: list[str] = []
    recs = intel.get("recommendations") or []
    if recs:
        r = recs[0]
        parts.append(
            f"机构评级({r.get('period', '')}): 强买{r.get('strong_buy', 0)}/买{r.get('buy', 0)}/"
            f"持{r.get('hold', 0)}/卖{r.get('sell', 0)}/强卖{r.get('strong_sell', 0)}"
        )
    surprises = intel.get("earnings_surprises") or []
    if surprises:
        s_items = []
        for s in surprises[:2]:
            sp = s.get("surprise_pct")
            sp_str = f"{sp:+.1f}%" if isinstance(sp, (int, float)) else "N/A"
            s_items.append(f"{s.get('period', '')}({sp_str})")
        parts.append("EPS惊喜: " + ", ".join(s_items))
    insider = intel.get("insider_sentiment")
    if isinstance(insider, dict) and insider.get("mspr") is not None:
        parts.append(f"内部人90d MSPR={insider['mspr']:.2f}(净变动{insider.get('change', 0)})")
    return " | ".join(parts)


def _format_past_lessons(rows: list[dict[str, Any]]) -> list[str]:
    """Format settled T+5 reflection lessons compactly for inspect_ticker."""
    lessons: list[str] = []
    for row in rows[:2]:
        date_s = str(row.get("created_at_uk") or row.get("settled_at_uk") or "").strip()[:10]
        action = str(row.get("action") or "").strip()
        alpha = row.get("alpha_pct")
        alpha_str = f" Alpha={alpha:+.1f}%" if isinstance(alpha, (int, float)) else ""
        text = str(row.get("reflection_lesson") or "").strip()[:120]
        if text:
            prefix = f"[{date_s} {action}{alpha_str}]".strip()
            lessons.append(f"{prefix} {text}".strip())
    return lessons


def _compact_inspect_payload(payload: dict[str, Any], max_chars: int = _INSPECT_TOOL_MAX_CHARS) -> dict[str, Any]:
    """Ensure inspect_ticker JSON payload stays strictly under max_tool_result_chars=4000."""
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return payload

    # Step 1: Trim news item titles/summaries if news is bulky
    news = payload.get("news")
    if isinstance(news, list) and news:
        compact_news = []
        for item in news[:3]:
            if isinstance(item, dict):
                compact_news.append({k: (str(v)[:100] if isinstance(v, str) else v) for k, v in item.items()})
            else:
                compact_news.append(str(item)[:100])
        payload["news"] = compact_news
        if len(json.dumps(payload, ensure_ascii=False, default=str)) <= max_chars:
            return payload

    # Step 2: Trim optional prose fields progressively
    for key, limit in (("us_deep_intel", 160), ("identity_summary", 160), ("summary", 900)):
        val = payload.get(key)
        if isinstance(val, str) and len(val) > limit:
            payload[key] = val[: limit - 1] + "…"
    if len(json.dumps(payload, ensure_ascii=False, default=str)) <= max_chars:
        return payload

    # Step 3: Drop news down to 1 item if still over budget
    if isinstance(payload.get("news"), list) and len(payload["news"]) > 1:
        payload["news"] = payload["news"][:1]
    return payload


class MarketToolkit:
    """Provides real-time Yahoo/Finnhub/FRED & Google Search lookups callable by Gemini Agents."""

    def __init__(
        self,
        market_data: MarketDataService,
        portfolio: PortfolioData,
        *,
        llm: LLMBackend | None = None,
        search_model: str = "gemini-4-hydrogen",
        ops_notifier: OpsNotifier | None = None,
        db: AdvisorDatabase | None = None,
        allow_search_tool: bool = True,
        search_cache_ttl_seconds: int = _DEFAULT_PERSIST_TTL_SECONDS,
    ) -> None:
        self.market_data = market_data
        self.portfolio = portfolio
        self.llm = llm
        self.search_model = search_model
        self.ops_notifier = ops_notifier
        self.db = db
        self.allow_search_tool = bool(allow_search_tool)
        self.search_cache_ttl_seconds = max(0, int(search_cache_ttl_seconds))
        self._inspect_cache: dict[str, dict[str, Any]] = {}
        self._corr_cache: dict[str, dict[str, Any]] = {}

    def _find_holding_context(self, sym: str) -> tuple[str | None, str | None]:
        """Return (isin, created_at) from current portfolio positions for a Yahoo symbol."""
        for pos in self.portfolio.all_positions():
            if pos.yahoo_symbol.strip().upper() == sym:
                isin = pos.isin.strip() if pos.isin and pos.isin.strip() else None
                created_at = pos.created_at.strip() if pos.created_at and pos.created_at.strip() else None
                return isin, created_at
        return None, None

    def declarations(self) -> list[types.FunctionDeclaration]:
        decls = [
            types.FunctionDeclaration(
                name="inspect_ticker",
                description=(
                    "Fetch deterministic market data, CYQ chip profit/concentration, weekly trend, MA20/50/200, RSI14, "
                    "52-week drawdown, PE valuation, earnings/ex-dividend dates, past reflection lessons, and recent news for a stock or ETF. "
                    "Pass US tickers directly (e.g. 'NVDA'), append '.L' for UK/LSE ETFs (e.g. 'AZN.L', 'VWRP.L'), "
                    "and append '.DE'/'.AS'/'.PA' for European tickers."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "symbol": {
                            "type": "STRING",
                            "description": "Yahoo Finance ticker symbol, e.g. AAPL, RR.L, VWRP.L, ASML.AS",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.FunctionDeclaration(
                name="check_portfolio_correlation",
                description=(
                    "Compute the maximum 60-trading-day return correlation between a candidate stock/ETF and current holdings "
                    "to check for duplicate exposure or diversification fit."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "symbol": {
                            "type": "STRING",
                            "description": "Candidate Yahoo Finance symbol, e.g. VWRP.L, QQQ, TSM, AZN.L",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.FunctionDeclaration(
                name="get_macro_snapshot",
                description=(
                    "Fetch the latest GBP/USD, EUR/GBP FX rates, US 10Y/2Y Treasury yields, Fed funds rate, VIX, "
                    "and 5-day moves for SPY, QQQ, and ISF.L."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]
        if self.llm is not None and self.allow_search_tool:
            decls.append(
                types.FunctionDeclaration(
                    name="search_market_news",
                    description=(
                        "Run a live Google Search for breaking news, regulatory filings (RNS/SEC/EU), earnings guidance revisions, "
                        "or institutional rating changes for US, UK (.L), European (.DE/.AS/.PA), LSE ETFs, or macro events. "
                        "Always formulate the query in concise English financial keywords."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "English search query with ticker, company name, or event keywords, e.g. 'AZN.L RNS news' or 'NVDA earnings guidance'",
                            }
                        },
                        "required": ["query"],
                    },
                )
            )
        return decls

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_ticker":
            sym = str(args.get("symbol") or "").strip().upper()
            if not sym:
                return {"error": "symbol is required"}
            if sym in self._inspect_cache:
                return self._inspect_cache[sym]
            pos_isin, pos_created_at = self._find_holding_context(sym)
            isin_map = {sym: pos_isin} if pos_isin else None
            histories, bundles = await self.market_data.get_bundles(
                [sym],
                include_news=True,
                include_earnings=True,
                isin_map=isin_map,
            )
            hist = histories.get(sym)
            bundle = bundles.get(sym)
            tech = compute_technical_snapshot(
                sym,
                hist,
                position_created_at=pos_created_at,
                bundle=bundle,
            )

            us_deep_intel_str = ""
            quote_type = bundle.quote_type if bundle else ""
            if _classify_symbol_region(sym, quote_type) == "US" and hasattr(self.market_data, "get_us_deep_intel"):
                try:
                    raw_intel = await self.market_data.get_us_deep_intel(sym)
                    us_deep_intel_str = _format_us_deep_intel(raw_intel or {})
                except Exception as exc:
                    logger.debug("inspect_ticker get_us_deep_intel(%s) failed: %s", sym, exc)

            past_lessons: list[str] = []
            if self.db is not None:
                try:
                    lesson_rows = await self.db.get_symbol_lessons(sym, limit=2)
                    past_lessons = _format_past_lessons(lesson_rows)
                except Exception as exc:
                    logger.debug("inspect_ticker get_symbol_lessons(%s) failed: %s", sym, exc)

            cyq_summary: dict[str, Any] | None = None
            if tech is not None and tech.cyq_benefit_part_pct is not None:
                cyq_summary = {
                    "benefit_part_pct": tech.cyq_benefit_part_pct,
                    "avg_cost": tech.cyq_avg_cost,
                    "concentration_90_pct": tech.cyq_concentration_90_pct,
                    "regime": tech.cyq_regime,
                }

            payload: dict[str, Any] = {
                "symbol": sym,
                "summary": format_technical_line(sym, tech, bundle),
                "name": bundle.short_name if bundle else sym,
                "identity_summary": bundle.identity_summary if bundle else "",
                "currency": bundle.currency if bundle else "",
                "quote_type": quote_type,
                "sector": bundle.sector if bundle else "",
                "dividend_yield_pct": bundle.dividend_yield_pct if bundle else None,
                "target_mean_price": bundle.target_mean_price if bundle else None,
                "ex_dividend_date": bundle.ex_dividend_date if bundle else None,
                "days_to_ex_dividend": (
                    tech.days_to_ex_dividend
                    if tech is not None and tech.days_to_ex_dividend is not None
                    else (bundle.days_to_ex_dividend if bundle else None)
                ),
                "weekly_bullish": tech.weekly_bullish if tech else None,
                "weekly_rsi14": tech.weekly_rsi14 if tech else None,
                "cyq": cyq_summary,
                "us_deep_intel": us_deep_intel_str,
                "past_lessons": past_lessons,
                # [用户授权调整] 保持 inspect_ticker 返回紧凑（新闻截前 3 条）；控制整包 < 4000 字符以免触发裁剪。
                "news": (bundle.news[:3] if bundle else []),
            }
            compacted = _compact_inspect_payload(payload)
            self._inspect_cache[sym] = compacted
            return compacted

        if name == "check_portfolio_correlation":
            sym = str(args.get("symbol") or "").strip().upper()
            held_syms = self.portfolio.unique_yahoo_symbols()
            if not sym:
                return {"error": "symbol is required"}
            if sym in self._corr_cache:
                return self._corr_cache[sym]
            if not held_syms:
                return {"symbol": sym, "note": "当前无持仓可比对"}
            all_syms = list(dict.fromkeys([*held_syms, sym]))
            bench_syms: list[str] = ["SPY", "ISF.L"]
            for s in all_syms:
                for b in resolve_market_benchmark(s):
                    if b not in bench_syms:
                        bench_syms.append(b)
            fetch_syms = list(dict.fromkeys([*all_syms, *bench_syms]))
            histories = await self.market_data.get_histories(fetch_syms, period="1y")
            corr = compute_correlation_matrix(
                histories,
                all_syms,
                lookback_days=60,
                benchmark_symbol="SPY",
                uk_benchmark_symbol="ISF.L",
                eu_benchmark_symbol="EXS1.DE",
            )
            most_sym, corr_val = max_correlation_against_portfolio(corr, sym, held_syms)
            res_corr = {
                "candidate": sym,
                "most_correlated_holding": most_sym,
                "max_correlation_60d": corr_val,
            }
            self._corr_cache[sym] = res_corr
            return res_corr

        if name == "get_macro_snapshot":
            snap = await self.market_data.get_macro_snapshot()
            return snap.model_dump() | {"summary": snap.to_text()}

        if name == "search_market_news":
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}
            if self.llm is None:
                return {"error": "Google Search backend unavailable"}

            norm_q = query.lower()
            mem_key = f"tool_search:{norm_q}"
            now = _monotonic()
            _prune_grounding_cache(now)
            mem_hit = _GROUNDING_CACHE.get(mem_key)
            if mem_hit and now < mem_hit[1]:
                return {"query": query, "summary": mem_hit[2], "cached": True}

            uk_date = datetime.now(timezone.utc).astimezone(LONDON_TZ).date().isoformat()
            tok_match = _TICKER_TOKEN_RE.search(query)
            sym_tok = tok_match.group(1) if tok_match else ""
            db_keys = [f"tool_search:{uk_date}:{norm_q}"]
            if sym_tok and len(query.split()) <= 5:
                db_keys.extend(
                    [
                        f"grounding:{uk_date}:prio:{sym_tok}",
                        f"grounding:{uk_date}:sym:{sym_tok}",
                    ]
                )
            if self.db is not None and self.search_cache_ttl_seconds > 0 and hasattr(self.db, "get_grounding_news_cache"):
                try:
                    cached_map = await self.db.get_grounding_news_cache(
                        db_keys,
                        max_age_seconds=self.search_cache_ttl_seconds,
                    )
                    for k in db_keys:
                        if k in cached_map:
                            cached_val = cached_map[k]
                            _GROUNDING_CACHE[mem_key] = (now, now + 900.0, cached_val)
                            _prune_grounding_cache(now)
                            return {"query": query, "summary": cached_val, "cached": True}
                except Exception as exc:
                    logger.debug("search_market_news cache lookup failed: %s", exc)

            res = await _single_search_query(
                llm=self.llm,
                model=self.search_model,
                prompt=(
                    "Use Google Search to retrieve the latest financial news, regulatory filings (RNS/SEC/EU), and market catalysts "
                    f"for: {query}. Output verified facts with exact dates and key figures concisely in Simplified Chinese within 280 Chinese characters."
                ),
                usage_label="search_tool",
            )
            if res.startswith("⚠️"):
                if self.ops_notifier is not None:
                    await self.ops_notifier.notify(
                        severity="WARN",
                        source="tool:search_market_news",
                        alert_key="search_tool_failed",
                        message=f"search_market_news 联网搜索失败（{query[:40]}）：{res[:160]}",
                    )
                return {"query": query, "summary": res}

            write_now = _monotonic()
            _GROUNDING_CACHE[mem_key] = (write_now, write_now + 900.0, res)
            _prune_grounding_cache(write_now)
            if self.db is not None and self.search_cache_ttl_seconds > 0 and hasattr(self.db, "upsert_grounding_news_cache"):
                try:
                    await self.db.upsert_grounding_news_cache(
                        [
                            {
                                "cache_key": f"tool_search:{uk_date}:{norm_q}",
                                "scope": "search_tool",
                                "symbol": sym_tok,
                                "summary_text": res,
                            }
                        ]
                    )
                except Exception as exc:
                    logger.debug("search_market_news cache write failed: %s", exc)
            return {"query": query, "summary": res}

        return {"error": f"Unknown tool: {name}"}

    def as_executor(self) -> Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
        return self.execute
