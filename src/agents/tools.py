"""Deterministic callable tools & Google Search Grounding helpers for Gemini Agents."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

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

_EU_SUFFIXES = (".DE", ".AS", ".PA", ".SW", ".MI", ".MC", ".BR", ".VI", ".ST", ".CO", ".HE", ".OL")
# cache_key -> (stored_at_monotonic, expires_at_monotonic, output)
_GROUNDING_CACHE: dict[str, tuple[float, float, str]] = {}
_GROUNDING_CACHE_MAX_ENTRIES = 64
_DEFAULT_GROUNDING_CONCURRENCY = 3
_INSPECT_TOOL_MAX_CHARS = 3800


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


async def _single_search_query(
    *,
    llm: LLMBackend,
    model: str,
    prompt: str,
    usage_label: str,
) -> str:
    return await llm.generate(
        model=model,
        prompt=prompt,
        system_instruction=(
            "你是实时财经新闻与监管公告检索员。请必须通过 Google Search 核查最新真实新闻、监管公告（RNS/SEC/欧股公告）、"
            "财报日期、业绩指引修正、诉讼/回购与机构评级变动。严禁编造消息、凑数输出已知静态 NAV/收盘价或展开空泛长篇评论；"
            "仅对有真实异动或重大消息的标的按 1~2 条干货事实输出（注明具体日期/关键数字/评级机构与多空影响）；"
            "无重大突发新闻/监管公告/评级变动的标的严禁逐个单列写长句，必须在段落末尾统一合并为一行略过（如：近期无重大新公告：IUKD.L, IBTA.L, UNH），纯文本禁用 # 与 **。"
        ),
        thinking_level="LOW",
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

    tasks_meta: list[tuple[str, str]] = []
    usage_label = f"{scope}_grounding"

    if scope == "report":
        tasks_meta.append(
            (
                "【宏观与英/欧/美今日突发主线】",
                "请通过 Google Search 检索过去 48 小时内影响英国(FTSE/BoE/英镑)、欧洲(ECB/欧元区)与美国(Fed/美债收益率/美股盘前盘后)的核心宏观突发新闻与重大政策动向，控制在 240 字以内。",
            )
        )
    elif focus_query.strip():
        tasks_meta.append(
            (
                f"【指定方向实时检索：{focus_query.strip()}】",
                f"请通过 Google Search 检索围绕主题「{focus_query.strip()}」在美股、英股/LSE ETF 与欧股中的最新突发催化、行业景气数据与潜在监管风险，控制在 260 字以内。",
            )
        )

    for sym in prio_ordered:
        region = _classify_symbol_region(sym, qtypes.get(sym, ""))
        if region in {"UK", "UK_ETF"}:
            p_prompt = (
                f"请通过 Google Search 针对伦敦交易所标的 {sym} 进行 1 对 1 深度检索：核查近 7 天 RNS 监管公告、"
                "财报/派息动态、突发异动原因及机构最新评级变化，给出日期与核心数字，控制在 180 字以内。"
            )
        elif region == "EU":
            p_prompt = (
                f"请通过 Google Search 针对欧洲标的 {sym} 进行 1 对 1 深度检索：核查近 7 天监管公告、"
                "财报指引、订单/监管消息、突发异动原因及机构评级变化，给出日期与核心数字，控制在 180 字以内。"
            )
        else:
            p_prompt = (
                f"请通过 Google Search 针对美股/ETF {sym} 进行 1 对 1 深度检索：核查近 72 小时隔夜突发新闻、"
                "盘前盘后异动原因、最新财报预期/指引、分红及机构评级调整，给出日期与核心数字，控制在 180 字以内。"
            )
        tasks_meta.append((f"【重点标的1对1深搜 · {sym}】", p_prompt))

    batch_size = 3
    region_specs = [
        (
            uk_syms,
            "英股与 LSE ETF 联网新闻",
            "请通过 Google Search 检索以下伦敦交易所股票/ETF 近 7 天最新 RNS 监管公告、财报或派息动态、分析师评级变化与重大行业新闻：{syms}。"
            "有重大突发消息/公告/评级变动的标的逐一用 1~2 行精华事实输出（含日期与核心数字）；"
            "无重大新公告的标的（含平稳 ETF 的日常 NAV）严禁逐个单列写长句，必须在末尾合并为一行略过（格式：近期无重大新公告：SYM1, SYM2）。",
        ),
        (
            eu_syms,
            "欧股联网新闻",
            "请通过 Google Search 检索以下欧洲股票近 7 天最新监管公告、财报指引、订单/监管消息与机构评级变化：{syms}。"
            "有重大突发消息/公告/评级变动的标的逐一用 1~2 行精华事实输出（含日期与核心数字）；"
            "无重大新公告的标的严禁逐个单列写长句，必须在末尾合并为一行略过（格式：近期无重大新公告：SYM1, SYM2）。",
        ),
        (
            us_syms,
            "美股与全球 ETF 联网新闻",
            "请通过 Google Search 检索以下美股/ETF 近 48~72 小时隔夜突发新闻、盘前盘后异动原因、财报指引修正、诉讼/回购与机构最新评级：{syms}。"
            "有真实异动或重大消息的标的逐一用 1~2 行精华事实输出（含日期与核心数字）；"
            "无重大突发新闻的标的严禁逐个单列写长句，必须在末尾合并为一行略过（格式：近期无重大新公告：SYM1, SYM2）。",
        ),
    ]
    for region_syms, title_base, prompt_tpl in region_specs:
        for idx, chunk in enumerate(_chunk_list(region_syms, batch_size), 1):
            suffix = f" #{idx}" if len(region_syms) > batch_size else ""
            tasks_meta.append((f"【{title_base}{suffix}】", prompt_tpl.format(syms=", ".join(chunk))))

    if not tasks_meta:
        return ""

    sem = asyncio.Semaphore(max(1, max_concurrent_searches))

    async def _bounded_search(prompt: str) -> str:
        async with sem:
            return await _single_search_query(
                llm=llm,
                model=search_model,
                prompt=prompt,
                usage_label=usage_label,
            )

    raw_results = await asyncio.gather(
        *(_bounded_search(p) for _, p in tasks_meta),
        return_exceptions=True,
    )

    sections: list[str] = []
    errors: list[str] = []
    for (title, _), res in zip(tasks_meta, raw_results):
        if isinstance(res, Exception):
            errors.append(f"{title}: {type(res).__name__}")
            continue
        text = str(res or "").strip()
        if not text or text.startswith("⚠️"):
            errors.append(f"{title}: {text[:120] or 'empty'}")
            continue
        sections.append(f"{title}\n{text}")

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
    ) -> None:
        self.market_data = market_data
        self.portfolio = portfolio
        self.llm = llm
        self.search_model = search_model
        self.ops_notifier = ops_notifier
        self.db = db

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
                    "获取任意股票或ETF的实时确定性行情、CYQ筹码获利盘/集中度、周线趋势、MA20/50/200、RSI14、52周回撤、PE估值、财报/除息日、历史复盘教训及最近新闻。"
                    "美股直接传代码如 'NVDA'；英股/LSE ETF 加 '.L' 如 'AZN.L'、'VWRP.L'；欧股加 '.DE'/'.AS'/'.PA'。"
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "symbol": {
                            "type": "STRING",
                            "description": "Yahoo Finance 标的代码，如 AAPL, RR.L, VWRP.L, ASML.AS",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.FunctionDeclaration(
                name="check_portfolio_correlation",
                description="计算候选股票或ETF与用户当前实盘持仓近60个交易日收益率的最高相关系数，用于判断是否重复暴露或适合分散风险。",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "symbol": {
                            "type": "STRING",
                            "description": "候选标的 Yahoo 代码，如 VWRP.L, QQQ, TSM, AZN.L",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            types.FunctionDeclaration(
                name="get_macro_snapshot",
                description="获取最新 GBP/USD、EUR/GBP 汇率、美10年期国债收益率、Fed 利率、VIX 及 SPY/QQQ/ISF.L 近5日涨跌。",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]
        if self.llm is not None:
            decls.append(
                types.FunctionDeclaration(
                    name="search_market_news",
                    description=(
                        "通过 Google Search 实时检索美股、英股(.L)、欧股(.DE/.AS/.PA)、LSE ETF 或宏观突发新闻、"
                        "监管公告(RNS/SEC)、财报指引与机构最新评级。发现疑点或用户询问最新事件时主动调用。"
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "要检索的标的代码、公司名或事件关键词，如 'AZN.L RNS news' 或 'NVDA earnings guidance'",
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
            return _compact_inspect_payload(payload)

        if name == "check_portfolio_correlation":
            sym = str(args.get("symbol") or "").strip().upper()
            held_syms = self.portfolio.unique_yahoo_symbols()
            if not sym:
                return {"error": "symbol is required"}
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
            return {
                "candidate": sym,
                "most_correlated_holding": most_sym,
                "max_correlation_60d": corr_val,
            }

        if name == "get_macro_snapshot":
            snap = await self.market_data.get_macro_snapshot()
            return snap.model_dump() | {"summary": snap.to_text()}

        if name == "search_market_news":
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}
            if self.llm is None:
                return {"error": "Google Search backend unavailable"}
            res = await _single_search_query(
                llm=self.llm,
                model=self.search_model,
                prompt=f"请通过 Google Search 检索以下主题或标的的最新财经新闻、公告与市场催化（含关键日期与数字，控制在 280 字内）：{query}",
                usage_label="search_tool",
            )
            if res.startswith("⚠️") and self.ops_notifier is not None:
                await self.ops_notifier.notify(
                    severity="WARN",
                    source="tool:search_market_news",
                    alert_key="search_tool_failed",
                    message=f"search_market_news 联网搜索失败（{query[:40]}）：{res[:160]}",
                )
            return {"query": query, "summary": res}

        return {"error": f"Unknown tool: {name}"}

    def as_executor(self) -> Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
        return self.execute
