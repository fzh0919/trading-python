"""Deterministic callable tools & Google Search Grounding helpers for Gemini Agents."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from google.genai import types

from src.data_providers.analytics import (
    compute_correlation_matrix,
    compute_technical_snapshot,
    format_technical_line,
    max_correlation_against_portfolio,
)
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import PortfolioData

if TYPE_CHECKING:
    from src.agents.llm_client import LLMBackend
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)

_EU_SUFFIXES = (".DE", ".AS", ".PA", ".SW", ".MI", ".MC", ".BR", ".VI", ".ST", ".CO", ".HE", ".OL")
_GROUNDING_CACHE: dict[str, tuple[float, str]] = {}


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
            "财报日期、业绩指引修正与机构评级变动。严禁编造消息或展开空泛长篇评论；每个标的用 1~2 条干货事实输出（注明具体日期/关键数字/评级机构与多空影响），无重大新闻写“近期无重大新公告”，纯文本禁用 # 与 **。"
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
) -> str:
    """Run grouped concurrent Google Search Grounding across UK/EU/US equities, ETFs, and macro news."""
    if llm is None:
        return ""

    clean_syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    cache_key = f"{scope}:{','.join(sorted(clean_syms))}:{focus_query.strip().lower()}"
    now = time.monotonic()
    cached = _GROUNDING_CACHE.get(cache_key)
    if cached and ttl_seconds > 0 and (now - cached[0]) < ttl_seconds:
        return cached[1]

    qtypes = quote_types or {}
    uk_syms: list[str] = []
    eu_syms: list[str] = []
    us_syms: list[str] = []
    for sym in clean_syms:
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

    batch_size = 3
    region_specs = [
        (
            uk_syms,
            "英股与 LSE ETF 联网新闻",
            "请通过 Google Search 逐一检索以下伦敦交易所股票/ETF 近 7 天最新 RNS 监管公告、财报或派息动态、分析师评级变化与重大行业新闻：{syms}。每个标的 1~2 行精华事实（含日期与核心数字），无重大新闻写“近期无重大新公告”。",
        ),
        (
            eu_syms,
            "欧股联网新闻",
            "请通过 Google Search 逐一检索以下欧洲股票近 7 天最新监管公告、财报指引、订单/监管消息与机构评级变化：{syms}。每个标的 1~2 行精华事实（含日期与核心数字），无重大新闻写“近期无重大新公告”。",
        ),
        (
            us_syms,
            "美股与全球 ETF 联网新闻",
            "请通过 Google Search 逐一检索以下美股/ETF 近 48~72 小时隔夜突发新闻、盘前盘后异动原因、财报指引修正与机构最新评级：{syms}。每个标的 1~2 行精华事实（含日期与核心数字），无重大新闻写“近期无重大新公告”。",
        ),
    ]
    for region_syms, title_base, prompt_tpl in region_specs:
        for idx, chunk in enumerate(_chunk_list(region_syms, batch_size), 1):
            suffix = f" #{idx}" if len(region_syms) > batch_size else ""
            tasks_meta.append((f"【{title_base}{suffix}】", prompt_tpl.format(syms=", ".join(chunk))))

    if not tasks_meta:
        return ""

    raw_results = await asyncio.gather(
        *(
            _single_search_query(llm=llm, model=search_model, prompt=p, usage_label=usage_label)
            for _, p in tasks_meta
        ),
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
        _GROUNDING_CACHE[cache_key] = (now, output)
    return output


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
    ) -> None:
        self.market_data = market_data
        self.portfolio = portfolio
        self.llm = llm
        self.search_model = search_model
        self.ops_notifier = ops_notifier

    def declarations(self) -> list[types.FunctionDeclaration]:
        decls = [
            types.FunctionDeclaration(
                name="inspect_ticker",
                description=(
                    "获取任意股票或ETF的实时确定性行情、MA20/50/200、RSI14、52周回撤、PE估值、财报日期及最近新闻。"
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
            # TODO(IMP-11): [接自 IMP-04 & IMP-05 & IMP-06] inspect_ticker 传入持仓 isin (兼容 None) 与 bundle=bundle，
            # 并按需接入 self.market_data.get_us_deep_intel(sym)、bundle.identity_summary/ex_dividend_date 与 DB 历史反思教训。
            histories, bundles = await self.market_data.get_bundles([sym], include_news=True, include_earnings=True)
            hist = histories.get(sym)
            bundle = bundles.get(sym)
            tech = compute_technical_snapshot(sym, hist)
            return {
                "symbol": sym,
                "summary": format_technical_line(sym, tech, bundle),
                "name": bundle.short_name if bundle else sym,
                "currency": bundle.currency if bundle else "",
                "quote_type": bundle.quote_type if bundle else "",
                "sector": bundle.sector if bundle else "",
                "dividend_yield_pct": bundle.dividend_yield_pct if bundle else None,
                "target_mean_price": bundle.target_mean_price if bundle else None,
                "news": bundle.news if bundle else [],
            }

        if name == "check_portfolio_correlation":
            sym = str(args.get("symbol") or "").strip().upper()
            held_syms = self.portfolio.unique_yahoo_symbols()
            if not sym:
                return {"error": "symbol is required"}
            if not held_syms:
                return {"symbol": sym, "note": "当前无持仓可比对"}
            # TODO(IMP-11): [接自 IMP-05 & IMP-06] 改调 market_data.get_histories() 并补入 "ISF.L" 激活分市场去 Beta 相关性。
            all_syms = list(dict.fromkeys([*held_syms, sym]))
            fetch_syms = list(dict.fromkeys([*all_syms, "SPY"]))
            histories, _ = await self.market_data.get_bundles(fetch_syms, include_news=False, include_earnings=False)
            corr = compute_correlation_matrix(histories, all_syms, lookback_days=60, benchmark_symbol="SPY")
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
