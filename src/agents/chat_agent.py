"""Interactive Telegram daily chat agent (/pos instant snapshot + free-form advisor chat)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from src.agents.llm_client import LLMBackend
from src.agents.prompts import build_agent_system_instruction
from src.agents.schemas import trim_text_at_line_boundary
from src.agents.tools import MarketToolkit
from src.config import AppConfig

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import (
    PortfolioData,
    T212Client,
    build_trade_context,
    format_account_overview,
    format_positions_text,
)
from src.engine.calendar_utils import (
    get_market_calendar_alerts,
    get_uk_weekday_label,
    get_us_market_schedule_uk,
)
from src.storage.db import AdvisorDatabase

_CLAUSE_SPLIT_RE = re.compile(r"[^，。！？!?；;,\n]+")
_DENIED_PREFIX_RE = re.compile(r"(?:不要|别|不想|不必|不用|无需|禁止)[^，。！？!?；;,\n]*$")
_COMPARE_HINT_RE = re.compile(r"对比|比较|哪个好|选谁|还是|vs|VS|PK|pk")
_FOLLOWUP_HINT_RE = re.compile(
    r"它|这只|这支|该股|这个标的|继续|刚才|上面|那只|那支|"
    r"(?:止损|止盈|筹码|成本|估值|目标价|财报|除息|分歧|教训|趋势|风险|支撑|压力|怎么看|怎么样|如何|还能买|能买吗|要卖吗|卖不卖)"
)
_CORP_SUFFIX_RE = re.compile(
    r"(?:\s*,?\s*(?:incorporated|inc\.?|corp\.?|corporation|plc|ltd\.?|limited|"
    r"n\.?v\.?|s\.?e\.?|a\.?g\.?|s\.?a\.?|ucits\s+etf|etf|holdings|group|trust|fund))+$",
    re.IGNORECASE,
)
_GENERIC_NAME_STOPWORDS = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "plc",
        "ltd",
        "limited",
        "etf",
        "ucits",
        "stock",
        "stocks",
        "holdings",
        "group",
        "trust",
        "fund",
        "isa",
        "invest",
        "usd",
        "gbp",
        "gbx",
        "eur",
    }
)
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ChatObjectMention:
    """Single matched code or name mention within NFKC-normalized user text."""

    start: int
    end: int
    raw: str
    kind: str  # "code" | "name"
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class ChatObjectResolution:
    """Deterministic 0-token symbol resolution result for chat prompt slicing."""

    decision_kind: str  # "confirm" | "compare" | "maintain" | "none"
    target_symbols: tuple[str, ...] = ()
    denied_symbols: tuple[str, ...] = ()


def _lookup_ci(mapping: dict[str, Any], key: str) -> Any:
    """Case-insensitive dict lookup by symbol key."""
    if not mapping:
        return None
    if key in mapping:
        return mapping[key]
    key_up = key.strip().upper()
    for k, v in mapping.items():
        if str(k).strip().upper() == key_up:
            return v
    return None


def _has_ascii_token_boundaries(text: str, start: int, end: int) -> bool:
    """Ensure ASCII ticker/name matches do not split alphanumeric tokens or slash commands."""
    if start > 0:
        left = text[start - 1]
        if left == "/" or (left.isascii() and (left.isalnum() or left in "_.")):
            return False
    if end < len(text):
        right = text[end]
        if right.isascii() and (right.isalnum() or right == "_"):
            return False
    return True


def _casefold_with_positions(text: str) -> tuple[str, list[int]]:
    """Build casefolded text alongside a char-offset map back into `text`."""
    folded_parts: list[str] = []
    positions: list[int] = []
    for offset, ch in enumerate(text):
        part = ch.casefold()
        folded_parts.append(part)
        positions.extend([offset] * len(part))
    return "".join(folded_parts), positions


def _build_symbol_alias_index(
    portfolio: PortfolioData,
    context_pack: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Build symbol -> canonical entry map plus code/name alias tables from portfolio & ContextPack."""
    pack_symbols_raw = (context_pack or {}).get("symbols")
    pack_symbols: dict[str, dict[str, Any]] = {}
    if isinstance(pack_symbols_raw, dict):
        for raw_sym, entry in pack_symbols_raw.items():
            sym_up = str(raw_sym or "").strip().upper()
            if sym_up and isinstance(entry, dict):
                pack_symbols[sym_up] = entry

    code_map: dict[str, list[str]] = {}
    name_map: dict[str, list[str]] = {}

    def _add_code(alias: str, sym: str) -> None:
        cleaned = unicodedata.normalize("NFKC", alias).strip().upper()
        if not cleaned or cleaned in _GENERIC_NAME_STOPWORDS:
            return
        bucket = code_map.setdefault(cleaned, [])
        if sym not in bucket:
            bucket.append(sym)

    def _add_name(alias: str, sym: str) -> None:
        raw_norm = unicodedata.normalize("NFKC", alias).strip()
        if not raw_norm:
            return
        candidates = [raw_norm]
        stripped = _CORP_SUFFIX_RE.sub("", raw_norm).strip(" ,.-|")
        if stripped and stripped.casefold() != raw_norm.casefold():
            candidates.append(stripped)
        for cand in candidates:
            cf = cand.casefold()
            if cf in _GENERIC_NAME_STOPWORDS:
                continue
            has_cjk = bool(_CJK_CHAR_RE.search(cand))
            if (has_cjk and len(cand) < 2) or (not has_cjk and len(cand) < 3):
                continue
            bucket = name_map.setdefault(cf, [])
            if sym not in bucket:
                bucket.append(sym)

    def _register_symbol_codes(sym_up: str, ticker: str = "", raw_name: str = "") -> None:
        _add_code(sym_up, sym_up)
        if "." in sym_up:
            base = sym_up.split(".", 1)[0]
            if base:
                _add_code(base, sym_up)
        if "-" in sym_up:
            _add_code(sym_up.replace("-", "."), sym_up)
        if ticker:
            _add_code(ticker, sym_up)
        if raw_name:
            _add_name(raw_name, sym_up)

    for sym_up, entry in pack_symbols.items():
        _register_symbol_codes(sym_up)
        bundle_sum = entry.get("bundle_summary")
        if isinstance(bundle_sum, dict):
            _add_name(str(bundle_sum.get("short_name") or ""), sym_up)
            _add_name(str(bundle_sum.get("long_name") or ""), sym_up)
        identity_sum = str(entry.get("identity_summary") or "")
        if "|" in identity_sum:
            parts = [p.strip() for p in identity_sum.split("|") if p.strip()]
            if len(parts) >= 2 and parts[1].upper() != sym_up:
                _add_name(parts[1], sym_up)
        for held in entry.get("held_accounts") or []:
            if isinstance(held, dict):
                _add_code(str(held.get("ticker") or ""), sym_up)

    for pos in portfolio.all_positions():
        sym_up = str(pos.yahoo_symbol or "").strip().upper()
        if sym_up:
            _register_symbol_codes(sym_up, pos.ticker, pos.name)

    for acc in (portfolio.isa, portfolio.invest):
        for order in acc.pending_orders:
            sym_up = str(order.yahoo_symbol or "").strip().upper()
            if sym_up:
                _register_symbol_codes(sym_up, order.ticker, order.name)

    return (
        pack_symbols,
        {k: tuple(v) for k, v in code_map.items()},
        {k: tuple(v) for k, v in name_map.items()},
    )


def _extract_object_mentions(
    text: str,
    code_aliases: dict[str, tuple[str, ...]],
    name_aliases: dict[str, tuple[str, ...]],
) -> tuple[ChatObjectMention, ...]:
    """Extract non-overlapping code/name mentions on one NFKC view (chat_object_decision pattern)."""
    if not text:
        return ()
    folded, positions = _casefold_with_positions(text)
    if not positions:
        return ()

    mentions: list[ChatObjectMention] = []

    for alias_upper, syms in code_aliases.items():
        alias_cf = alias_upper.casefold()
        for match in re.finditer(re.escape(alias_cf), folded):
            start = positions[match.start()]
            end = positions[match.end() - 1] + 1
            if not _has_ascii_token_boundaries(text, start, end):
                continue
            raw_slice = text[start:end]
            # Short 1-2 char codes (e.g. V, MA) require exact uppercase to avoid English word collisions
            if len(alias_upper) <= 2 and raw_slice != alias_upper:
                continue
            mentions.append(ChatObjectMention(start, end, raw_slice, "code", syms))

    for alias_cf, syms in name_aliases.items():
        for match in re.finditer(re.escape(alias_cf), folded):
            start = positions[match.start()]
            end = positions[match.end() - 1] + 1
            raw_slice = text[start:end]
            if (raw_slice[0].isascii() and raw_slice[0].isalnum()) or (
                raw_slice[-1].isascii() and raw_slice[-1].isalnum()
            ):
                if not _has_ascii_token_boundaries(text, start, end):
                    continue
            mentions.append(ChatObjectMention(start, end, raw_slice, "name", syms))

    # Suppress shorter spans strictly contained in a wider span (e.g. ULVR inside ULVR.L)
    filtered: list[ChatObjectMention] = []
    for m in mentions:
        if any(
            other.start <= m.start
            and m.end <= other.end
            and (other.start, other.end) != (m.start, m.end)
            for other in mentions
        ):
            continue
        filtered.append(m)

    # Deduplicate identical (start, end) preferring "code" over "name"
    by_span: dict[tuple[int, int], ChatObjectMention] = {}
    for m in filtered:
        span = (m.start, m.end)
        prev = by_span.get(span)
        if prev is None or (prev.kind != "code" and m.kind == "code"):
            by_span[span] = m

    return tuple(sorted(by_span.values(), key=lambda item: (item.start, item.end)))


def _resolve_mentions_in_text(
    text: str,
    code_aliases: dict[str, tuple[str, ...]],
    name_aliases: dict[str, tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    """Classify mentioned symbols in `text` into affirmative targets vs negated mentions."""
    mentions = _extract_object_mentions(text, code_aliases, name_aliases)
    if not mentions:
        return [], []

    affirmative: list[str] = []
    denied: list[str] = []
    for clause_match in _CLAUSE_SPLIT_RE.finditer(text):
        c_start, c_end = clause_match.start(), clause_match.end()
        clause_mentions = [m for m in mentions if c_start <= m.start and m.end <= c_end]
        for m in clause_mentions:
            prefix = text[c_start : m.start]
            target_bucket = denied if _DENIED_PREFIX_RE.search(prefix) else affirmative
            for sym in m.symbols:
                if sym not in target_bucket:
                    target_bucket.append(sym)

    affirmative_set = set(affirmative)
    only_denied = [sym for sym in denied if sym not in affirmative_set]
    return affirmative, only_denied


def decide_chat_objects(
    user_query: str,
    portfolio: PortfolioData,
    context_pack: dict[str, Any] | None = None,
    *,
    history_turns: list[tuple[str, str]] | None = None,
    prev_cycle_turns: list[tuple[str, str]] | None = None,
    max_symbols: int = 4,
) -> ChatObjectResolution:
    """Resolve target ticker(s) locally with 0 LLM tokens, supporting negation and follow-up continuity."""
    text = unicodedata.normalize("NFKC", user_query or "").strip()
    if not text:
        return ChatObjectResolution("none")

    _pack_symbols, code_aliases, name_aliases = _build_symbol_alias_index(portfolio, context_pack)
    if not code_aliases and not name_aliases:
        return ChatObjectResolution("none")

    affirmative, denied = _resolve_mentions_in_text(text, code_aliases, name_aliases)
    if affirmative:
        chosen = tuple(affirmative[: max(1, max_symbols)])
        kind = "compare" if (len(chosen) >= 2 or _COMPARE_HINT_RE.search(text)) else "confirm"
        return ChatObjectResolution(kind, chosen, tuple(denied))

    # Follow-up continuity: recover subject from recent chat turns when user uses pronouns/elliptical questions
    if _FOLLOWUP_HINT_RE.search(text):
        denied_set = set(denied)
        all_prior = list(reversed(history_turns or [])) + list(reversed(prev_cycle_turns or []))
        for prev_user, prev_bot in all_prior:
            prev_norm_u = unicodedata.normalize("NFKC", prev_user or "")
            u_aff, _ = _resolve_mentions_in_text(prev_norm_u, code_aliases, name_aliases)
            recovered = [s for s in u_aff if s not in denied_set]
            if not recovered and prev_bot:
                prev_norm_b = unicodedata.normalize("NFKC", prev_bot[:400])
                b_aff, _ = _resolve_mentions_in_text(prev_norm_b, code_aliases, name_aliases)
                recovered = [s for s in b_aff if s not in denied_set]
            if recovered:
                return ChatObjectResolution(
                    "maintain",
                    tuple(recovered[: min(2, max(1, max_symbols))]),
                    tuple(denied),
                )

    return ChatObjectResolution("none", (), tuple(denied))


def _format_context_pack_symbol_slice(sym: str, entry: dict[str, Any]) -> str:
    """Format one symbol's ContextPack slice (identity/technical/bundle/evals/disagreements/lessons)."""
    identity = str(entry.get("identity_summary") or sym).strip()
    lines: list[str] = [f"▸ {identity if sym in identity else f'{sym} | {identity}'}"]

    tech_line = str(entry.get("technical_line") or "").strip()
    tech = entry.get("technical") if isinstance(entry.get("technical"), dict) else {}
    cyq_band = ""
    low90 = tech.get("cyq_cost_90_low")
    high90 = tech.get("cyq_cost_90_high")
    if isinstance(low90, (int, float)) and isinstance(high90, (int, float)):
        cyq_band = f" | 90%筹码带 {float(low90):.2f}-{float(high90):.2f}"
    if tech_line:
        lines.append(f"  技术与筹码: {tech_line}{cyq_band}")

    bundle = entry.get("bundle_summary") if isinstance(entry.get("bundle_summary"), dict) else {}
    if bundle:
        b_parts: list[str] = []
        curr = str(bundle.get("currency") or "").strip()
        cur_p = bundle.get("current_price")
        if isinstance(cur_p, (int, float)) and not tech_line:
            b_parts.append(f"现价 {float(cur_p):.2f} {curr}".strip())
        if isinstance(bundle.get("roe_pct"), (int, float)):
            b_parts.append(f"ROE {float(bundle['roe_pct']):.1f}%")
        if isinstance(bundle.get("profit_margin_pct"), (int, float)):
            b_parts.append(f"净利率 {float(bundle['profit_margin_pct']):.1f}%")
        if isinstance(bundle.get("revenue_growth_pct"), (int, float)):
            b_parts.append(f"营收增速 {float(bundle['revenue_growth_pct']):+.1f}%")
        if isinstance(bundle.get("debt_to_equity"), (int, float)):
            b_parts.append(f"D/E {float(bundle['debt_to_equity']):.1f}")
        if bundle.get("recommendation_key"):
            b_parts.append(f"评级 {bundle['recommendation_key']}")
        if bundle.get("earnings_date") and "财报" not in tech_line:
            d_earn = bundle.get("days_to_earnings")
            b_parts.append(
                f"财报 {bundle['earnings_date']}" + (f"({d_earn}天后)" if d_earn is not None else "")
            )
        if bundle.get("ex_dividend_date") and "除息" not in tech_line:
            d_ex = bundle.get("days_to_ex_dividend")
            b_parts.append(
                f"除息 {bundle['ex_dividend_date']}" + (f"({d_ex}天)" if d_ex is not None else "")
            )
        if b_parts:
            lines.append("  基本面快照: " + " | ".join(b_parts))
        news_items = bundle.get("news") or []
        if isinstance(news_items, list) and news_items:
            titles = " / ".join(
                f"[{n.get('date', '')}] {str(n.get('title', ''))[:60]}"
                for n in news_items[:2]
                if isinstance(n, dict) and n.get("title")
            )
            if titles:
                lines.append(f"  近期新闻: {titles}")

    evals = entry.get("evaluations") or []
    if isinstance(evals, list):
        for ev in evals[:2]:
            if not isinstance(ev, dict):
                continue
            acct = ev.get("account", "")
            bias = ev.get("action_bias", "HOLD")
            wt = ev.get("weight_pct")
            nat_ret = ev.get("native_return_pct")
            eff_stop = ev.get("effective_stop_pct")
            flags: list[str] = []
            if ev.get("has_pending_stop"):
                flags.append("有保护止损单")
            pend_w = ev.get("pending_buy_weight_pct")
            if isinstance(pend_w, (int, float)) and pend_w > 0:
                flags.append(f"在途买单{float(pend_w):.1f}%")
            if ev.get("ex_div_guarded"):
                flags.append("除息防误判豁免")
            ev_parts = [f"[{acct}] {bias}"]
            if isinstance(wt, (int, float)):
                ev_parts.append(f"仓位{float(wt):.1f}%")
            if isinstance(nat_ret, (int, float)):
                ev_parts.append(f"本币收益{float(nat_ret):+.1f}%")
            if isinstance(eff_stop, (int, float)):
                ev_parts.append(f"有效止损{float(eff_stop):.1f}%")
            if flags:
                ev_parts.append("/".join(flags))
            reasons = [str(r) for r in (ev.get("reasons") or [])[:3] if r]
            reason_str = (" -> " + "; ".join(reasons)) if reasons else ""
            lines.append("  风控评估: " + " | ".join(ev_parts) + reason_str)

    disagreements = entry.get("disagreements") or []
    if isinstance(disagreements, list):
        for dis in disagreements[:2]:
            if not isinstance(dis, dict):
                continue
            bull = "; ".join(str(x) for x in (dis.get("bull_points") or [])[:2])
            bear = "; ".join(str(x) for x in (dis.get("bear_points") or [])[:2])
            hint = str(dis.get("decision_hint") or "").strip()
            lines.append(
                f"  多空分歧({dis.get('conflict_type', '')}): 多[{bull}] vs 空[{bear}]"
                + (f" | {hint}" if hint else "")
            )

    lessons = entry.get("lessons") or []
    if isinstance(lessons, list) and lessons:
        for row in lessons[:2]:
            if not isinstance(row, dict):
                continue
            lesson_txt = str(row.get("reflection_lesson") or "").strip()[:120]
            if not lesson_txt:
                continue
            alpha = row.get("alpha_pct")
            alpha_s = f" Alpha {float(alpha):+.1f}%" if isinstance(alpha, (int, float)) else ""
            lines.append(
                f"  历史教训[{row.get('created_at_uk', '')} {row.get('action', '')}{alpha_s}]: {lesson_txt}"
            )

    return "\n".join(lines)


def _format_previous_cycle_summary(prev_turns: list[tuple[str, str]]) -> str:
    """Compact up to 2 turns from the previous trading cycle to prevent 07:30 UK memory cliffs."""
    if not prev_turns:
        return ""
    lines: list[str] = []
    for u, a in prev_turns[-2:]:
        u_one = trim_text_at_line_boundary(" ".join(str(u).splitlines()), 70)
        a_one = trim_text_at_line_boundary(" ".join(str(a).splitlines()), 130)
        lines.append(f"• [上一周期] 问: {u_one} → 答: {a_one}")
    return "\n".join(lines)


def _format_cycle_baseline_section(
    context_pack: dict[str, Any] | None,
    prev_cycle_turns: list[tuple[str, str]],
) -> str:
    """Assemble cycle-invariant baseline block placed in the static prefix for Gemini Implicit Cache."""
    parts: list[str] = []
    if isinstance(context_pack, dict):
        exposure = str(context_pack.get("exposure_summary") or "").strip()
        if exposure:
            parts.append(f"• 组合底层敞口：\n{trim_text_at_line_boundary(exposure, 350)}")
        disagreement_txt = str(context_pack.get("disagreement_text") or "").strip()
        if disagreement_txt and "暂无" not in disagreement_txt:
            parts.append(f"• 组合多空分歧摘要：\n{trim_text_at_line_boundary(disagreement_txt, 550)}")
        recent_lessons = context_pack.get("recent_lessons") or []
        if isinstance(recent_lessons, list) and recent_lessons:
            lesson_bits: list[str] = []
            for row in recent_lessons[:3]:
                if isinstance(row, dict) and row.get("reflection_lesson"):
                    alpha = row.get("alpha_pct")
                    a_str = f" Alpha {float(alpha):+.1f}%" if isinstance(alpha, (int, float)) else ""
                    lesson_str = trim_text_at_line_boundary(str(row["reflection_lesson"]), 100)
                    lesson_bits.append(
                        f"[{row.get('symbol', '')} {row.get('action', '')}{a_str}] "
                        f"{lesson_str}"
                    )
            if lesson_bits:
                parts.append("• 近期选股 T+N 反思：" + "；".join(lesson_bits))

    prev_summary = _format_previous_cycle_summary(prev_cycle_turns)
    if prev_summary:
        parts.append(prev_summary)

    if not parts:
        return ""
    return "【本周期组合基线与跨周期摘要】：\n" + "\n".join(parts) + "\n\n"


async def format_positions_instant(cfg: AppConfig, t212: T212Client) -> str:
    """Instant deterministic /pos portfolio & 30d trade snapshot without waiting for LLM."""
    portfolio = await t212.get_portfolio(use_cache=False)
    if not portfolio.isa.ok and not portfolio.invest.ok:
        return f"⚠️ Trading 212 数据获取失败：\nISA: {portfolio.isa.error}\nInvest: {portfolio.invest.error}"

    trade_ctx = build_trade_context(portfolio, max_recent_lines=5, max_summary_lines=6)
    lines = [
        f"💼 T212 双账户实时快照（{portfolio.fetched_at_uk} UK | 闲置现金利息 {cfg.strategy.cash_interest_rate_pct:.1f}%）",
        f"合计总净值: GBP {portfolio.total_combined_value:,.2f} | 合计现金: GBP {portfolio.total_combined_cash:,.2f}",
        "",
        "【Stocks ISA（长线稳健仓）】",
        format_account_overview(portfolio.isa),
        format_positions_text(portfolio.isa),
        "",
        "【Invest（短线机动仓）】",
        format_account_overview(portfolio.invest),
        format_positions_text(portfolio.invest),
        "",
        "【当前活动挂单】",
        trade_ctx["pending_orders_text"],
        "",
        "【近7天实盘成交】",
        trade_ctx["recent_trades_text"],
    ]
    return "\n".join(lines)


def _format_chat_history_compact(history_turns: list[tuple[str, str]], recent_full_turns: int = 3) -> str:
    """Keep recent turns concise while compacting older turns to control prompt bloat."""
    if not history_turns:
        return "（今日暂无前序对话）"
    split_idx = max(0, len(history_turns) - recent_full_turns)
    lines: list[str] = []
    for u, a in history_turns[:split_idx]:
        u_one_line = trim_text_at_line_boundary(" ".join(u.splitlines()), 60)
        a_one_line = trim_text_at_line_boundary(" ".join(a.splitlines()), 100)
        lines.append(f"• [早期] 问: {u_one_line} → 答: {a_one_line}")
    for u, a in history_turns[split_idx:]:
        lines.append(f"👤 用户: {u}\n🤖 顾问: {trim_text_at_line_boundary(a, 380)}")
    return "\n\n".join(lines)


def _has_complete_embedded_vetoes(morning_block: str, guardrails: list[str]) -> bool:
    """Check whether morning_block already includes the complete daily redlines without truncation."""
    if not morning_block:
        return False
    lines = morning_block.splitlines()
    for idx, ln in enumerate(lines):
        stripped = ln.strip()
        if "今日风控红线" not in stripped:
            continue
        if stripped.endswith(("…", "...")):
            return False
        if not stripped.startswith("【") and ("：" in stripped or ":" in stripped):
            return all(g in morning_block for g in guardrails) if guardrails else True
        if stripped.startswith("【今日风控红线") and idx + 1 < len(lines):
            next_ln = lines[idx + 1].strip()
            if next_ln and not next_ln.endswith(("…", "...")):
                return all(g in morning_block for g in guardrails) if guardrails else True
    return False


async def answer_user_chat(
    user_query: str,
    cfg: AppConfig,
    t212: T212Client,
    market_data: MarketDataService,
    llm: LLMBackend,
    db: AdvisorDatabase,
    ops_notifier: OpsNotifier | None = None,
) -> str:
    """Answer user chat with 0-token ContextPack slicing, Gemini Implicit Cache prefix order, and /deep routing."""
    portfolio = await t212.get_portfolio(use_cache=True)
    trade_ctx = build_trade_context(portfolio, history_days=7, max_recent_lines=3, max_summary_lines=0)

    (
        (morning_full_text, morning_conclusion),
        guardrails,
        (pick_ts, pick_text),
        history_turns,
        prev_cycle_turns,
        context_pack,
    ) = await asyncio.gather(
        db.get_morning_context(include_premarket=True),
        db.get_daily_guardrails(),
        db.get_latest_pick_radar(),
        db.get_chat_history(limit=cfg.strategy.max_daily_chat_turns),
        db.get_previous_cycle_chat_history(limit=2),
        db.get_context_pack(allow_fallback=True),
    )

    history_block = _format_chat_history_compact(history_turns)
    veto_block = "\n".join(f"• {g}" for g in guardrails) if guardrails else "（无）"
    sched = get_us_market_schedule_uk()
    calendar_alerts = get_market_calendar_alerts()
    calendar_inline = ("｜" + "；".join(calendar_alerts)) if calendar_alerts else ""
    morning_block = (
        trim_text_at_line_boundary(morning_full_text, 2050)
        if morning_full_text
        else (morning_conclusion or "（暂无）")
    )
    veto_section = "" if _has_complete_embedded_vetoes(morning_block, guardrails) else f"【今日风控红线】：\n{veto_block}\n\n"
    pick_block = trim_text_at_line_boundary(pick_text, 1050) if pick_text else ""
    reset_str = f"{cfg.schedule.morning_report_hour:02d}:{cfg.schedule.morning_report_minute:02d} UK"

    # [用户授权调整] /deep 单聊 Prompt 软约束放宽至 1000~1200 字，普通单聊保持 500 字以内（均不做代码截断）
    is_deep_query = user_query.strip().lower().startswith("/deep")
    length_rule = "1000~1200 字以内（深度解析可展开多维证据与执行预案）" if is_deep_query else "500 字以内"
    chat_model = cfg.models.chat_deep_model if is_deep_query else cfg.models.chat_model
    chat_thinking = cfg.models.chat_deep_thinking_level if is_deep_query else cfg.models.chat_thinking_level
    max_rounds = 4 if is_deep_query else 3

    resolution = decide_chat_objects(
        user_query,
        portfolio,
        context_pack,
        history_turns=history_turns,
        prev_cycle_turns=prev_cycle_turns,
    )
    pack_symbols = (context_pack or {}).get("symbols") if isinstance(context_pack, dict) else {}
    pack_symbols_dict = pack_symbols if isinstance(pack_symbols, dict) else {}
    target_slice_blocks: list[str] = []
    for sym in resolution.target_symbols:
        entry = _lookup_ci(pack_symbols_dict, sym)
        if isinstance(entry, dict):
            target_slice_blocks.append(_format_context_pack_symbol_slice(sym, entry))
        else:
            target_slice_blocks.append(f"▸ {sym}: （当日 ContextPack 暂无该标的快照，如有需要请调用 inspect_ticker）")

    target_slice_section = ""
    if target_slice_blocks:
        mode_tag = {
            "confirm": "直接提及",
            "compare": "多标的对标",
            "maintain": "上下文沿用",
        }.get(resolution.decision_kind, "本地切片")
        target_slice_section = (
            f"【提问标的本地快照切片（0-Token ContextPack 局部注入 | {mode_tag}）】：\n"
            + "\n\n".join(target_slice_blocks)
            + "\n\n"
        )

    chat_sys = build_agent_system_instruction(
        "你是用户的全天候私人量化投资顾问。\n"
        "交互与成本纪律：\n"
        "• 上来第一句先给明确结论与建议，后面再简要解释。\n"
        "• 上下文已包含今日早报、风控红线、最新持仓占比/浮盈、活动挂单及已匹配提问标的的本地 ContextPack 快照切片（含技术/筹码/分歧/历史教训）；若已有信息足以回答（如仓位诊断、挂单看法、策略解读、已注入切片的标的问询），请直接一轮作答，严禁无故调用工具。\n"
        "• 仅当用户询问持仓与 ContextPack 外的新标的，或确实缺少突发实时公告/组合相关性时，才按需调用 `inspect_ticker`、`check_portfolio_correlation` 或 `search_market_news`；禁止连续重复调用相同工具。\n"
        f"• 直白客观（不迎合已有挂单或前序决定），禁用 Markdown # 和 **，用 • 和【】排版，控制在 {length_rule}。",
        cfg.strategy,
    )

    # Static cycle-invariant prefix FIRST to maximize Gemini Implicit Cache prefix hits across turns
    static_prefix = (
        f"【今日早报及美股哨兵纪要】：\n{morning_block}\n\n"
        + veto_section
        # [用户授权调整] 放宽 /pick 雷达切片至 1050 字符以完整保留精选行动与驳回原因
        + (f"【最近一次选股雷达 ({pick_ts} UK)】：\n{pick_block}\n\n" if pick_block else "")
        + _format_cycle_baseline_section(context_pack, prev_cycle_turns)
    )

    dynamic_tail = (
        f"当前英国时间 {sched.uk_date} ({get_uk_weekday_label()}{calendar_inline})，"
        f"美股状态：{sched.phase}（常规开盘 {sched.open_time_uk} UK，收盘 {sched.close_time_uk} UK）。\n\n"
        f"【ISA 账户】\n{format_account_overview(portfolio.isa)}\n{format_positions_text(portfolio.isa)}\n\n"
        f"【Invest 账户】\n{format_account_overview(portfolio.invest)}\n{format_positions_text(portfolio.invest)}\n\n"
        f"【当前挂单与近7日成交】\n{trade_ctx['pending_orders_text']}\n{trade_ctx['recent_trades_text']}\n\n"
        + target_slice_section
        + f"【今日对话记录（{reset_str} 晨报周期自动重置）】：\n{history_block}\n\n"
        + f"【用户当前问题】：\n{user_query}"
    )

    prompt = static_prefix + dynamic_tail

    toolkit = MarketToolkit(
        market_data,
        portfolio,
        llm=llm,
        search_model=cfg.models.search_model,
        ops_notifier=ops_notifier,
        db=db,
    )
    reply = await llm.generate(
        model=chat_model,
        prompt=prompt,
        system_instruction=chat_sys,
        thinking_level=chat_thinking,
        tools=toolkit.declarations(),
        tool_executor=toolkit.as_executor(),
        temperature=0.25,
        max_tool_rounds=max_rounds,
        usage_label="chat",
    )

    if not reply.startswith("⚠️"):
        await db.append_chat_turn(
            user_query=user_query,
            bot_reply=reply,
            max_turns=cfg.strategy.max_daily_chat_turns,
        )
    return reply
