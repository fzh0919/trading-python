"""Deterministic risk-override guardrail for /report and /pick decisions (VIX, R:R, ISA value, cadence)."""

from __future__ import annotations

import re
from typing import Any

from src.config import StrategySection
from src.data_providers.analytics import (
    _ETF_TECH_PENETRATION_PCT,
    TechnicalSnapshot,
    infer_etf_index_family,
    normalize_issuer_root,
)
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.portfolio_rules.dual_philosophy import (
    _ex_div_yield_hint,
    _pick_guard_mode,
    _resolve_pick_rr,
    evaluate_isa_value_gate,
    resolve_invest_dynamic_rr_floor,
)
from src.engine.portfolio_rules.holding_eval import _resolve_total_portfolio_gbp
from src.engine.portfolio_rules.models_and_friction import (
    DEFAULT_VIX_DANGER_THRESHOLD,
    ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT,
    ISA_VALUE_GUARD_MODE_DEFAULT,
    HoldingEvaluation,
    RiskOverrideRecord,
    RiskOverrideResult,
    _CORE_WIDE_BASE_FAMILIES,
    _estimate_pending_order_gbp,
)
from src.engine.portfolio_rules.sizing import _resolve_pick_sizing

_VETO_BLOCK_KEYWORDS = ("禁止", "严禁", "勿", "不买", "不加", "暂停", "回避", "拦截", "冻结", "VETO", "NO ")
_NON_TICKER_TOKENS = {
    "ISA", "INVEST", "GBP", "GBX", "USD", "EUR", "ETF", "LSE", "NYSE", "VIX",
    "RSI", "ATR", "CYQ", "ADX", "MA20", "MA50", "MA200", "BUY", "ADD", "HOLD",
    "TRIM", "STOP", "KEEP", "MODIFY", "CANCEL", "CASH", "UK", "US", "EU", "PM",
}


def _symbol_matches(a: str, b: str) -> bool:
    sa = a.strip().upper()
    sb = b.strip().upper()
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    return sa.split(".")[0] == sb.split(".")[0]


def _extract_vetoed_symbols(daily_vetoes: list[str] | None) -> dict[str, str]:
    """Extract explicitly vetoed ticker symbols from daily_vetoes text lines."""
    vetoed: dict[str, str] = {}
    for line in daily_vetoes or []:
        text = str(line or "").strip()
        if not text:
            continue
        upper = text.upper()
        if not any(kw in text or kw in upper for kw in _VETO_BLOCK_KEYWORDS):
            continue
        for tok in re.findall(r"(?<![A-Z0-9.])([A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?)(?![A-Z0-9.])", upper):
            base = tok.split(".")[0]
            if tok in _NON_TICKER_TOKENS or base in _NON_TICKER_TOKENS or base.isdigit():
                continue
            vetoed.setdefault(tok, text)
            vetoed.setdefault(base, text)
    return vetoed


def _build_report_action_blocks(
    report_actions: list[Any] | None,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Index same-day /report TRIM/STOP directives by (ACCOUNT, SYMBOL_OR_ROOT) for hard reversal blocking."""
    blocks: dict[tuple[str, str], tuple[str, str]] = {}
    for item in report_actions or []:
        if isinstance(item, dict):
            raw_acct = str(item.get("account") or "").strip().upper()
            raw_sym = str(item.get("symbol") or "").strip().upper()
            raw_act = str(item.get("action") or "").strip().upper()
            raw_reason = str(item.get("reason") or "").strip()
        else:
            raw_acct = str(getattr(item, "account", "") or "").strip().upper()
            raw_sym = str(getattr(item, "symbol", "") or "").strip().upper()
            raw_act = str(getattr(item, "action", "") or "").strip().upper()
            raw_reason = str(getattr(item, "reason", "") or "").strip()

        if raw_act not in {"TRIM", "STOP"} or not raw_sym or raw_sym == "CASH":
            continue

        if raw_acct in {"双账户", "BOTH", "ALL", ""} or ("ISA" in raw_acct and "INVEST" in raw_acct):
            accts = ("ISA", "INVEST")
        elif "INVEST" in raw_acct:
            accts = ("INVEST",)
        elif "ISA" in raw_acct:
            accts = ("ISA",)
        else:
            accts = (raw_acct,)

        base = raw_sym.split(".")[0]
        root = normalize_issuer_root(base)
        keys = {k for k in (raw_sym, base, root) if k}
        for acct in accts:
            for key in keys:
                blocks.setdefault((acct, key), (raw_act, raw_reason))
    return blocks


def _cadence_guard_mode(strategy: StrategySection) -> str:
    """Resolve the cadence guard strength (off / advisory / hard)."""
    mode = str(getattr(strategy, "trade_cadence_guard_mode", "advisory") or "advisory").strip().lower()
    return mode if mode in {"off", "advisory", "hard"} else "advisory"


def _cadence_violation(
    behavior: dict[str, Any],
    sym_up: str,
    base: str,
    strategy: StrategySection,
    is_etf: bool = False,
) -> tuple[str, str] | None:
    """Detect a cadence/cooldown breach: weekly/monthly BUY caps or the re-buy cooldown.

    Returns ``None`` whenever the fetched order history is incomplete (``partial``) so that a
    short history window can never silently freeze all new deployment. This function only
    *detects* the breach — whether it blocks or merely advises is decided by
    ``trade_cadence_guard_mode``.

    v5.4.0: core wide-base ETF buys (VWRP.L etc.) are 100% quota-exempt — buying the
    benchmark or clearing legacy crumbs never consumes any individual-equity quota.
    """
    if is_etf:
        return None
    if not behavior:
        return None
    partial = bool(behavior.get("partial"))
    coverage_days = float(behavior.get("coverage_days") or 0.0)
    # v5.5.7: a hard 15-day re-buy cooldown is *proven* by a real closed position even when the
    # fetched history is shorter than 30 days. Only the frequency caps stay soft under `partial`
    # (a legacy bare {"partial": True} stub with no coverage_days keeps the old no-op behaviour).
    if not partial or coverage_days > 0.0:
        base_root = normalize_issuer_root(base)
        for row in behavior.get("recently_closed") or []:
            if not row.get("in_cooldown"):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            if ticker == sym_up or ticker == base or normalize_issuer_root(ticker) == base_root:
                return (
                    "REBUY_COOLDOWN",
                    f"{ticker} 于 {float(row.get('days_since_sell') or 0):.0f} 天前清仓，处于 "
                    f"{int(behavior.get('cooldown_days') or 0)} 天再买冷却期",
                )
    if partial:
        return None
    weekly_cap = int(getattr(strategy, "max_weekly_new_buys", 0) or 0)
    monthly_cap = int(getattr(strategy, "max_monthly_new_buys", 0) or 0)
    if weekly_cap > 0 and int(behavior.get("buys_7d") or 0) >= weekly_cap:
        return (
            "FREQUENCY_CAP",
            f"近 7 天已新开仓 {int(behavior.get('buys_7d') or 0)} 笔(≥上限{weekly_cap})，本周额度已用完",
        )
    if monthly_cap > 0 and int(behavior.get("buys_30d") or 0) >= monthly_cap:
        return (
            "FREQUENCY_CAP",
            f"近 30 天已新开仓 {int(behavior.get('buys_30d') or 0)} 笔(≥上限{monthly_cap})，本月额度已用完",
        )
    return None


def _build_override_portfolio_context(
    portfolio: PortfolioData | None,
    strat: StrategySection,
    bmap: dict[str, TickerMarketBundle],
    deployment_gate: dict[str, Any] | None,
    trade_behavior: dict[str, Any] | None,
) -> tuple[
    dict[str, float],
    dict[tuple[str, str], float],
    set[str],
    dict[str, float],
    set[str],
    dict[str, TickerMarketBundle],
    dict[str, Any],
    str,
    set[str],
    set[str],
]:
    """Precompute current/pending weights, ETF roots, bundle lookup, and equity-only trade cadence."""
    issuer_weights: dict[str, float] = {}
    acct_weights: dict[tuple[str, str], float] = {}
    etf_roots: set[str] = set()
    account_totals: dict[str, float] = {}
    if portfolio is not None:
        total_all = _resolve_total_portfolio_gbp(portfolio)
        pos_by_sym = {p.yahoo_symbol: p for p in portfolio.all_positions()}
        account_totals = {
            "ISA": portfolio.isa.total_account_value or sum(p.current_value for p in portfolio.isa.positions),
            "INVEST": portfolio.invest.total_account_value or sum(p.current_value for p in portfolio.invest.positions),
        }
        issuer_vals: dict[str, float] = {}
        for p in portfolio.all_positions():
            root = normalize_issuer_root(p.ticker)
            issuer_vals[root] = issuer_vals.get(root, 0.0) + p.current_value
            acct_weights[(p.account.upper(), p.yahoo_symbol.upper())] = p.weight_pct
            acct_weights[(p.account.upper(), p.ticker.upper())] = p.weight_pct
            b = bmap.get(p.yahoo_symbol)
            if b and b.quote_type.upper() == "ETF":
                etf_roots.add(root)
        for o in portfolio.all_pending_orders():
            if o.side.upper() != "BUY":
                continue
            gbp_val = _estimate_pending_order_gbp(o, pos_by_symbol=pos_by_sym)
            if gbp_val <= 0:
                continue
            root = normalize_issuer_root(o.ticker)
            issuer_vals[root] = issuer_vals.get(root, 0.0) + gbp_val
            acct_total = account_totals.get(o.account.upper(), 0.0)
            if acct_total > 0:
                add_w = (gbp_val / acct_total) * 100.0
                for key_sym in {o.yahoo_symbol.upper(), o.ticker.upper()}:
                    if key_sym:
                        acct_weights[(o.account.upper(), key_sym)] = (
                            acct_weights.get((o.account.upper(), key_sym), 0.0) + add_w
                        )
        issuer_weights = {r: (v / total_all) * 100.0 for r, v in issuer_vals.items()}

    # Index bundles case-insensitively and by base symbol; also seed known UCITS ETF roots
    etf_roots.update(normalize_issuer_root(s) for s in _ETF_TECH_PENETRATION_PCT)
    core_etf_syms: set[str] = {str(strat.benchmark_symbol).strip().upper()}
    if isinstance(deployment_gate, dict):
        for etf_sym in deployment_gate.get("etf_set") or []:
            if etf_sym:
                etf_roots.add(normalize_issuer_root(str(etf_sym)))
                core_etf_syms.add(str(etf_sym).strip().upper())
    bundle_lookup: dict[str, TickerMarketBundle] = {}
    for sym, b in bmap.items():
        bundle_lookup[sym.upper()] = b
        bundle_lookup.setdefault(sym.upper().split(".")[0], b)
        if b.quote_type.upper() == "ETF":
            etf_roots.add(normalize_issuer_root(sym))

    # v5.4.0 zero-quota exemption: recompute equity-only BUY counts from raw order history so
    # that core-ETF buys (VWRP.L etc.) and SELL-side cleanups never consume quota. When
    # portfolio is None the caller-supplied trade_behavior is read verbatim (legacy contract).
    effective_behavior = dict(trade_behavior) if trade_behavior else {}
    benchmark_root = normalize_issuer_root(str(strat.benchmark_symbol))
    held_isa_syms: set[str] = set()
    held_isa_bases: set[str] = set()
    if portfolio is not None:
        for p in portfolio.isa.positions:
            held_isa_syms.add(p.yahoo_symbol.upper())
            held_isa_bases.add(p.yahoo_symbol.upper().split(".")[0])
        if effective_behavior:
            recent_all = [*portfolio.isa.recent_orders, *portfolio.invest.recent_orders]
            if recent_all:
                anchor_ms = max(
                    int(getattr(portfolio, "fetched_at_ms", 0) or 0),
                    max(o.filled_at_ms for o in recent_all),
                )

                def _order_is_etf(o: Any) -> bool:
                    root = normalize_issuer_root(str(getattr(o, "ticker", "") or ""))
                    sym = str(getattr(o, "yahoo_symbol", "") or "").upper()
                    if root in etf_roots or root == benchmark_root:
                        return True
                    if sym and (sym in etf_roots or normalize_issuer_root(sym) in etf_roots):
                        return True
                    return False

                equity_buys = [
                    o
                    for o in recent_all
                    if str(getattr(o, "side", "") or "").upper() == "BUY" and not _order_is_etf(o)
                ]
                effective_behavior["buys_7d"] = sum(
                    1 for o in equity_buys if anchor_ms - o.filled_at_ms <= 7 * 86_400_000
                )
                effective_behavior["buys_30d"] = sum(
                    1 for o in equity_buys if anchor_ms - o.filled_at_ms <= 30 * 86_400_000
                )

    return (
        issuer_weights,
        acct_weights,
        etf_roots,
        account_totals,
        core_etf_syms,
        bundle_lookup,
        effective_behavior,
        benchmark_root,
        held_isa_syms,
        held_isa_bases,
    )


def _apply_report_overrides(
    decision: Any,
    check_violation: Any,
    cadence_advisories: list[str],
) -> RiskOverrideResult:
    """Apply deterministic risk overrides to MorningReportDecision.top_actions."""
    overrides: list[RiskOverrideRecord] = []
    notes: list[str] = []
    new_actions = []
    for act in decision.top_actions:
        violation = check_violation(act.account, act.symbol, act.action)
        if violation is not None:
            trig, why = violation
            orig_action = act.action
            new_reason = f"[🛡️风控护栏修正] {why}；原建议({orig_action})已降级HOLD"
            new_actions.append(act.model_copy(update={"action": "HOLD", "reason": new_reason}))
            overrides.append(
                RiskOverrideRecord(
                    account=act.account,
                    symbol=act.symbol,
                    from_action=orig_action,
                    to_action="HOLD",
                    trigger_type=trig,
                    reason=why,
                )
            )
            notes.append(f"[🛡️风控护栏修正] [{act.account}] {act.symbol}: {orig_action}→HOLD（{why}）")
        else:
            new_actions.append(act)
    copied = decision.model_copy(update={"top_actions": new_actions}) if overrides else decision.model_copy()
    notes.extend(cadence_advisories)
    return RiskOverrideResult(
        decision=copied,
        applied=bool(overrides),
        overrides=overrides,
        notes=notes,
    )


def _apply_pick_overrides(
    decision: Any,
    check_violation: Any,
    *,
    portfolio: PortfolioData | None,
    strat: StrategySection,
    bundle_lookup: dict[str, TickerMarketBundle],
    account_totals: dict[str, float],
    tech_lookup: dict[str, TechnicalSnapshot],
    macro: MacroSnapshot | None,
    regime: str,
    risk_budget_scale: float,
    cadence_advisories: list[str],
    core_etf_syms: set[str] | None = None,
) -> RiskOverrideResult:
    """Apply deterministic risk overrides and position sizing notes to PickDecision."""
    overrides: list[RiskOverrideRecord] = []
    notes: list[str] = []
    kept_actions = []
    sizing_map: dict[str, Any] = {}
    new_rejected = list(getattr(decision, "rejected_or_frozen", []) or [])
    sizing_applied = False
    core_etf_syms = {str(s).strip().upper() for s in (core_etf_syms or set()) if str(s).strip()}
    gbp_usd_rate = getattr(portfolio, "gbp_usd_rate", None) if portfolio is not None else None
    for item in decision.selected_actions:
        violation = check_violation(
            item.account, item.symbol, item.action, item, is_pick=True
        )
        if violation is not None:
            trig, why = violation
            overrides.append(
                RiskOverrideRecord(
                    account=item.account,
                    symbol=item.symbol,
                    from_action=item.action,
                    to_action="REJECTED",
                    trigger_type=trig,
                    reason=why,
                )
            )
            note = f"[🛡️风控护栏修正] [{item.account}] {item.symbol}({item.action}→拦截): {why}"
            notes.append(note)
            new_rejected.append(note)
        else:
            keep_item = item
            if str(item.action).strip().upper() in {"BUY", "ADD"}:
                acct_up = item.account.strip().upper()
                sym_up = item.symbol.strip().upper()
                sym_base = sym_up.split(".")[0]
                bundle_item = bundle_lookup.get(sym_up) or bundle_lookup.get(sym_base)
                existing_val = 0.0
                if portfolio is not None:
                    item_root = normalize_issuer_root(sym_up)
                    for held in portfolio.all_positions():
                        if held.account.strip().upper() != acct_up:
                            continue
                        held_root = normalize_issuer_root(
                            str(getattr(held, "ticker", "") or held.yahoo_symbol)
                        )
                        if held_root == item_root:
                            existing_val += float(held.current_value or 0.0)
                sizing_res = _resolve_pick_sizing(
                    item,
                    bundle_item,
                    strat,
                    account=item.account,
                    account_total_value_gbp=account_totals.get(acct_up),
                    existing_position_value_gbp=existing_val,
                    gbp_usd_rate=gbp_usd_rate,
                    wallet=portfolio.invest if (portfolio is not None and acct_up == "INVEST") else None,
                    risk_budget_scale=float(risk_budget_scale or 1.0),
                    tech=tech_lookup.get(sym_up) or tech_lookup.get(sym_base),
                    macro=macro,
                    regime=regime,
                )
                size_note = str(sizing_res.get("note") or "")
                size_dict = sizing_res.get("size")
                if isinstance(size_dict, dict) and size_dict:
                    sizing_map[f"{acct_up}|{sym_up}"] = dict(size_dict)
                if size_note:
                    # risk_note 是 Python 侧保留字段：模型必须返回空串，落地前由这里写入。
                    keep_item = item.model_copy(update={"risk_note": size_note})
                    sizing_applied = True
                # v5.5.7: an unfundable ticket must be hard-blocked, not pushed with a
                # contradictory "建议买 + 勿碎单" line.
                pick_core_etf = False
                if isinstance(size_dict, dict) and size_dict:
                    root = normalize_issuer_root(sym_up)
                    pick_core_etf = bool(
                        sym_up in core_etf_syms
                        or sym_base in core_etf_syms
                        or root in {normalize_issuer_root(s) for s in core_etf_syms}
                    )
                    if not pick_core_etf:
                        family = infer_etf_index_family(sym_up, bundle_item)
                        pick_core_etf = family in _CORE_WIDE_BASE_FAMILIES
                if (
                    _pick_guard_mode(strat) == "hard"
                    and not pick_core_etf
                    and isinstance(size_dict, dict)
                    and bool(size_dict.get("below_min_order"))
                ):
                    min_order = float(size_dict.get("min_order_gbp") or 0.0)
                    budget_sized = float(size_dict.get("budget_sized_gbp") or 0.0)
                    headroom = float(size_dict.get("headroom_gbp") or 0.0)
                    if 0.0 < budget_sized < min_order:
                        why = (
                            f"止损过宽致预算额 £{budget_sized:,.0f} < 最小单笔额 £{min_order:,.0f}，"
                            "无法按以损定仓建仓；请收窄止损或改选标的"
                        )
                    else:
                        why = (
                            f"单仓权重余量 £{headroom:,.0f} < 最小单笔额 £{min_order:,.0f}，"
                            "剩余空间不足以建仓"
                        )
                    overrides.append(
                        RiskOverrideRecord(
                            account=item.account,
                            symbol=item.symbol,
                            from_action=item.action,
                            to_action="REJECTED",
                            trigger_type="BELOW_MIN_ORDER_SIZE",
                            reason=why,
                        )
                    )
                    note = f"[🛡️风控护栏修正] [{item.account}] {item.symbol}({item.action}→拦截): {why}"
                    notes.append(note)
                    new_rejected.append(note)
                    continue
            kept_actions.append(keep_item)
    copied = (
        decision.model_copy(
            update={"selected_actions": kept_actions, "rejected_or_frozen": new_rejected}
        )
        if (overrides or sizing_applied)
        else decision.model_copy()
    )
    notes.extend(cadence_advisories)
    return RiskOverrideResult(
        decision=copied,
        applied=bool(overrides),
        overrides=overrides,
        notes=notes,
        sizing_map=sizing_map,
    )


def apply_risk_override_guardrail(
    decision: Any,
    *,
    portfolio: PortfolioData | None = None,
    macro: MacroSnapshot | None = None,
    strategy: StrategySection | None = None,
    bundle_map: dict[str, TickerMarketBundle] | None = None,
    evals: list[HoldingEvaluation] | None = None,
    high_correlation_symbols: set[str] | None = None,
    daily_vetoes: list[str] | None = None,
    report_actions: list[Any] | None = None,
    trade_behavior: dict[str, Any] | None = None,
    vix_danger_threshold: float = DEFAULT_VIX_DANGER_THRESHOLD,
    deployment_gate: dict[str, Any] | None = None,
    risk_budget_scale: float = 1.0,
    tech_map: dict[str, TechnicalSnapshot] | None = None,
) -> RiskOverrideResult:
    """Apply deterministic post-LLM risk override guardrails to MorningReportDecision or PickDecision."""
    if decision is None:
        return RiskOverrideResult(decision=None, applied=False)

    strat = strategy or StrategySection()
    bmap = bundle_map or {}
    macro_vix = macro.vix if (macro is not None and macro.vix is not None) else None
    macro_high_risk = macro_vix is not None and macro_vix >= vix_danger_threshold
    corr_set = {s.strip().upper() for s in (high_correlation_symbols or set()) if s and s.strip()}
    corr_bases = {s.split(".")[0] for s in corr_set}
    regime = str((deployment_gate or {}).get("regime") or "NEUTRAL").strip().upper()
    tech_lookup: dict[str, TechnicalSnapshot] = {}
    for t_sym, t_snap in (tech_map or {}).items():
        if not t_sym or t_snap is None:
            continue
        tech_lookup[t_sym.strip().upper()] = t_snap
        tech_lookup.setdefault(t_sym.strip().upper().split(".")[0], t_snap)

    merged_vetoes = list(daily_vetoes or [])
    if hasattr(decision, "daily_vetoes") and getattr(decision, "daily_vetoes"):
        merged_vetoes.extend(getattr(decision, "daily_vetoes"))
    vetoed_map = _extract_vetoed_symbols(merged_vetoes)
    report_action_blocks = _build_report_action_blocks(report_actions)

    (
        issuer_weights,
        acct_weights,
        etf_roots,
        account_totals,
        core_etf_syms,
        bundle_lookup,
        effective_behavior,
        benchmark_root,
        held_isa_syms,
        held_isa_bases,
    ) = _build_override_portfolio_context(
        portfolio, strat, bmap, deployment_gate, trade_behavior
    )

    stop_alerts: set[tuple[str, str]] = {
        (e.account.upper(), e.yahoo_symbol.upper())
        for e in (evals or [])
        if e.action_bias == "REVIEW_STOP"
    }
    cadence_advisories: list[str] = []

    def _core_dca_guard(
        sym_up: str,
        base: str,
        acct_up: str,
        bundle: TickerMarketBundle | None,
        guard_mode: str,
    ) -> tuple[str, str] | None:
        """v5.5.1 deterministic DCA guard replacing the old blank R:R exemption."""
        if guard_mode != "hard":
            return None
        isa_gate = (
            deployment_gate.get("isa")
            if isinstance(deployment_gate, dict) and isinstance(deployment_gate.get("isa"), dict)
            else {}
        )
        policy = isa_gate.get("core_policy") if isinstance(isa_gate.get("core_policy"), dict) else {}
        bucket_map = policy.get("bucket_by_symbol") if isinstance(policy.get("bucket_by_symbol"), dict) else {}
        bucket = bucket_map.get(sym_up) or bucket_map.get(base)
        if not bucket:
            family = infer_etf_index_family(sym_up, bundle)
            bucket = {
                "Global All-World/Developed (全球宽基)": "GLOBAL_ANCHOR",
                "S&P 500 (美股标普500)": "US_TILT",
                "Europe Equity (欧洲宽基)": "EU_DIVERSIFIER",
                "FTSE 100 (英国富时100)": "UK_DIVERSIFIER",
            }.get(family or "")
        committed_core_pct = float(
            policy.get("committed_core_pct")
            or isa_gate.get("committed_core_etf_pct")
            or 0.0
        )
        if committed_core_pct >= 65.0:
            return "ACCOUNT_OVERWEIGHT", "ISA 核心宽基已达65%硬上限，暂停新增宽基定投"
        buckets = policy.get("buckets") if isinstance(policy.get("buckets"), dict) else {}
        sleeve = buckets.get(bucket) if bucket else None
        if bucket == "US_TILT" and policy.get("us_tilt_buy_frozen"):
            return (
                "ACCOUNT_OVERWEIGHT",
                "美股与标普500敞口已超上限，禁止继续加码标普500宽基",
            )
        if isinstance(sleeve, dict):
            cap = float(sleeve.get("cap_pct") or 0.0)
            pct_now = float(sleeve.get("committed_pct") or 0.0)
            if cap > 0 and pct_now >= cap:
                return (
                    "ACCOUNT_OVERWEIGHT",
                    f"该核心子桶已达 {cap:.0f}% 硬上限，暂停继续加仓",
                )
        if acct_up == "ISA" and portfolio is not None and portfolio.isa.total_account_value > 0:
            isa_val = float(portfolio.isa.total_account_value)
            floor_gbp = (
                isa_val * float(getattr(strat, "isa_cash_band_low_pct", 5.0) or 0.0) / 100.0
                + float(getattr(strat, "min_isa_etf_order_gbp", 400.0) or 0.0)
            )
            if float(portfolio.isa.available_to_trade) < floor_gbp:
                return (
                    "ACCOUNT_OVERWEIGHT",
                    "ISA 可用现金逼近 5% 结构地板，暂停宽基定投",
                )
        if bucket in {"EU_DIVERSIFIER", "UK_DIVERSIFIER", "US_TILT"}:
            plan = (
                isa_gate.get("core_allocation_plan")
                if isinstance(isa_gate.get("core_allocation_plan"), dict)
                else {}
            )
            metrics_map = plan.get("symbol_metrics") if isinstance(plan.get("symbol_metrics"), dict) else {}
            metrics = metrics_map.get(sym_up) or metrics_map.get(base) or {}
            price = metrics.get("price")
            ma200 = metrics.get("ma200")
            dd52 = metrics.get("drawdown_52w_pct")
            rsi = metrics.get("rsi14")
            if (
                price is not None
                and ma200 is not None
                and float(price) < float(ma200) * 0.95
                and (dd52 is None or float(dd52) > -15.0)
            ):
                return (
                    "ACCOUNT_OVERWEIGHT",
                    "跌破MA200逾5%且未达-15%深跌阶梯，暂缓左侧接刀",
                )
            if rsi is not None and float(rsi) >= 72.0:
                return "ACCOUNT_OVERWEIGHT", "RSI14≥72 追高，暂停宽基买入"
        return None

    def _check_symbol_violation(
        account: str,
        symbol: str,
        action: str,
        item: Any | None = None,
        *,
        is_pick: bool = False,
    ) -> tuple[str, str] | None:
        sym_up = symbol.strip().upper()
        base = sym_up.split(".")[0]
        acct_up = account.strip().upper()
        act_up = action.strip().upper()
        if act_up not in {"BUY", "ADD", "SWAP"} or sym_up == "CASH":
            return None

        if sym_up in vetoed_map or base in vetoed_map:
            return "DAILY_VETO", f"违背今日早报红线({vetoed_map.get(sym_up) or vetoed_map.get(base)})"

        if report_action_blocks:
            root = normalize_issuer_root(base)
            hit = (
                report_action_blocks.get((acct_up, sym_up))
                or report_action_blocks.get((acct_up, base))
                or report_action_blocks.get((acct_up, root))
            )
            if hit is not None:
                rep_act, rep_reason = hit
                reason_tail = f"：{rep_reason}" if rep_reason else ""
                return "REPORT_ACTION_CONFLICT", f"今日早报已对该账户持仓裁决 {rep_act}{reason_tail}，禁止逆势买入/加仓"

        if sym_up in corr_set or base in corr_bases:
            return "CORRELATION_OVERLAP", f"与现有持仓相关性触发≥{strat.overlap_correlation_threshold:.2f}重叠拦截"

        if act_up in {"BUY", "ADD"}:
            root = normalize_issuer_root(base)
            b = bundle_lookup.get(sym_up) or bundle_lookup.get(base)
            benchmark_up = str(strat.benchmark_symbol).strip().upper()
            is_etf = bool(
                (b is not None and b.quote_type.upper() == "ETF")
                or (root in etf_roots)
                or (root == benchmark_root)
                or sym_up == benchmark_up
                or base == benchmark_up
            )
            # v5.4.0: core wide-base ETF buys are 100% quota-exempt (zero Quota).
            cadence_breach = _cadence_violation(effective_behavior, sym_up, base, strat, is_etf=is_etf)
            if cadence_breach is not None:
                guard_mode = _cadence_guard_mode(strat)
                if guard_mode == "hard":
                    return cadence_breach
                if guard_mode == "advisory":
                    _trig, _why = cadence_breach
                    note = (
                        f"[⚠️纪律提醒] [{account}] {symbol}: {_why}；"
                        "advisory 模式不拦截，若仍执行请在理由中写明催化/避险依据"
                    )
                    if note not in cadence_advisories:
                        cadence_advisories.append(note)

            if not is_etf:
                issuer_w = issuer_weights.get(root, 0.0)
                # v5.4.0 hard single-name cap across both accounts (12% of the whole portfolio).
                if issuer_w >= strat.max_single_name_portfolio_pct:
                    return (
                        "ISSUER_OVERWEIGHT",
                        f"同发行人跨账户占全组合{issuer_w:.1f}%(≥单票硬上限{strat.max_single_name_portfolio_pct:.0f}%)",
                    )
                if issuer_w >= strat.max_total_portfolio_weight_pct:
                    return (
                        "ISSUER_OVERWEIGHT",
                        f"同发行人跨账户占全组合{issuer_w:.1f}%(≥上限{strat.max_total_portfolio_weight_pct:.0f}%)",
                    )
                acct_w = acct_weights.get((acct_up, sym_up)) or acct_weights.get((acct_up, base), 0.0)
                acct_cap = strat.max_invest_single_weight_pct if acct_up == "INVEST" else strat.max_isa_single_weight_pct
                if acct_w >= acct_cap:
                    return (
                        "ACCOUNT_OVERWEIGHT",
                        f"占{account}单仓达{acct_w:.1f}%(≥上限{acct_cap:.0f}%)",
                    )
                # v5.4.0: ISA satellite layer saturated (holdings + pending equity buys over
                # 30% / 8 names) — block NEW ISA equity buys, wide-base ETFs stay allowed.
                if (
                    is_pick
                    and acct_up == "ISA"
                    and sym_up not in held_isa_syms
                    and base not in held_isa_bases
                    and isinstance(deployment_gate, dict)
                    and isinstance(deployment_gate.get("isa"), dict)
                    and deployment_gate["isa"].get("isa_satellite_saturated")
                ):
                    return (
                        "ISA_SATELLITE_SATURATED",
                        "ISA 个股卫星层已饱和，暂停新开个股，请归拢核心宽基",
                    )

            if (acct_up, sym_up) in stop_alerts or any(
                k[0] == acct_up and _symbol_matches(k[1], sym_up) for k in stop_alerts
            ):
                return "ACCOUNT_OVERWEIGHT", "标的已触发止损复核红线，严禁逆势加仓"

            if b and b.days_to_earnings is not None and 0 <= b.days_to_earnings <= 5:
                return "EARNINGS_WINDOW", f"距财报仅{b.days_to_earnings}天，禁止财报前主动加仓"

            # v5.3.4: BUY-side ex-dividend window + Python-recomputed R:R floor (pick only).
            guard_mode = _pick_guard_mode(strat)
            if is_pick and guard_mode != "off":
                wallet = (
                    portfolio.invest
                    if (portfolio is not None and acct_up == "INVEST")
                    else None
                )
                pick_core_etf = bool(
                    sym_up in core_etf_syms
                    or base in core_etf_syms
                    or normalize_issuer_root(sym_up)
                    in {normalize_issuer_root(s) for s in core_etf_syms}
                )
                if not pick_core_etf:
                    family = infer_etf_index_family(sym_up, b)
                    pick_core_etf = family in _CORE_WIDE_BASE_FAMILIES
                if pick_core_etf:
                    dca_breach = _core_dca_guard(sym_up, base, acct_up, b, guard_mode)
                    if dca_breach is not None:
                        return dca_breach
                ex_window = int(getattr(strat, "ex_div_entry_guard_days", 3) or 0)
                ex_days = int(b.days_to_ex_dividend) if (b and b.days_to_ex_dividend is not None) else None
                if ex_days is not None and ex_window > 0 and 0 < ex_days <= ex_window:
                    ex_msg = (
                        f"距除息仅{ex_days}天({b.ex_dividend_date or '日期待核'})，"
                        f"除权低开≈{_ex_div_yield_hint(b)}；"
                        "ISA 股息免税可抢息，Invest 应税须按股息税后净额核 R:R"
                    )
                    if guard_mode == "hard" and not pick_core_etf:
                        return "EX_DIV_WINDOW", ex_msg
                    advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {ex_msg}"
                    if advisory not in cadence_advisories:
                        cadence_advisories.append(advisory)

                if item is not None:
                    # Core wide-base / diversifier DCA is a structural allocation, not a
                    # tactical R:R trade: the 2:1 floor and numeric stop/target do not apply.
                    t_snap = tech_lookup.get(sym_up) or tech_lookup.get(base)
                    if pick_core_etf:
                        pass
                    elif acct_up == "ISA" and not is_etf:
                        # v5.5.1 ISA branch: quality gate + margin of safety REPLACE the 2:1 R:R
                        # hurdle entirely (a Buffett value name may legitimately carry stop_price=0).
                        isa_value_mode = str(
                            getattr(strat, "isa_value_guard_mode", ISA_VALUE_GUARD_MODE_DEFAULT) or ""
                        ).strip().lower()
                        if isa_value_mode != "off":
                            is_crisis = bool(macro_high_risk or regime == "RISK_OFF")
                            vg = evaluate_isa_value_gate(
                                b,
                                t_snap,
                                macro,
                                strat,
                                item=item,
                                symbol=sym_up,
                                crisis_mode=is_crisis,
                            )
                            if not vg["quality_pass"]:
                                vg_msg = f"ISA 质量门未通过：{'；'.join(vg['quality_fails'])}"
                                if isa_value_mode == "hard":
                                    return "ISA_QUALITY_FAIL", vg_msg
                                _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {vg_msg}"
                                if _adv not in cadence_advisories:
                                    cadence_advisories.append(_adv)
                            elif not vg["mos_pass"]:
                                mode_lbl = "危机深折价模式" if is_crisis else "常规模式"
                                vg_msg = (
                                    f"ISA 安全边际不足（{mode_lbl}）："
                                    f"{'；'.join(vg['mos_reasons']) or '估值证据不足'}"
                                )
                                if isa_value_mode == "hard":
                                    return "ISA_MOS_INSUFFICIENT", vg_msg
                                _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {vg_msg}"
                                if _adv not in cadence_advisories:
                                    cadence_advisories.append(_adv)
                    else:
                        # v5.5.1 Invest branch: regime × structure dynamic R:R + ATR noise floor.
                        rr_info = _resolve_pick_rr(
                            item,
                            b,
                            strat,
                            symbol=sym_up,
                            account=acct_up,
                            wallet=wallet,
                            tech=t_snap,
                            regime=regime,
                        )
                        if rr_info.get("stop_too_tight"):
                            st_msg = (
                                f"止损幅仅{float(rr_info.get('risk_pct') or 0.0):.1f}%，"
                                f"低于ATR噪声地板{float(rr_info.get('min_stop_pct') or 0.0):.1f}%，"
                                "禁止借窄止损虚增R:R"
                            )
                            if guard_mode == "hard":
                                return "STOP_TOO_TIGHT", st_msg
                            _adv = f"[⚠️纪律提醒] [{account}] {symbol}: {st_msg}"
                            if _adv not in cadence_advisories:
                                cadence_advisories.append(_adv)
                        rr = rr_info.get("rr")
                        if rr is None:
                            rr_msg = (
                                "Python 无法复核 R:R（stop_price / target_price 未给出有效数字）；"
                                "hard 模式要求先给出数值止损与目标价"
                            )
                            if guard_mode == "hard":
                                return "RR_UNVERIFIED", rr_msg
                            advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {rr_msg}"
                            if advisory not in cadence_advisories:
                                cadence_advisories.append(advisory)
                        else:
                            min_rr, tier_label = resolve_invest_dynamic_rr_floor(
                                regime,
                                t_snap,
                                strat,
                                vix=macro_vix,
                            )
                            if float(rr) < min_rr:
                                rr_msg = (
                                    f"Python 复算 R:R 仅{float(rr):.2f}:1(<{min_rr:.1f}:1 {tier_label}；"
                                    f"目标空间{float(rr_info['upside_pct'] or 0.0):+.1f}% "
                                    f"扣摩擦{float(rr_info['friction_pct'] or 0.0):.2f}% "
                                    f"vs 止损{float(rr_info['risk_pct'] or 0.0):.1f}%)"
                                )
                                if guard_mode == "hard":
                                    return "RR_BELOW_FLOOR", rr_msg
                                advisory = f"[⚠️纪律提醒] [{account}] {symbol}: {rr_msg}"
                                if advisory not in cadence_advisories:
                                    cadence_advisories.append(advisory)

            if macro_high_risk and macro_vix is not None:
                core_etf_buy = bool(
                    sym_up in core_etf_syms
                    or base in core_etf_syms
                    or normalize_issuer_root(sym_up)
                    in {normalize_issuer_root(s) for s in core_etf_syms}
                )
                if not core_etf_buy:
                    family = infer_etf_index_family(sym_up, b)
                    core_etf_buy = family in _CORE_WIDE_BASE_FAMILIES
                # v5.5.1: ISA deep-margin-of-safety quality value names keep a staged
                # contrarian entry lane during a crisis; Invest stays 100% frozen.
                isa_crisis_value_exempt = False
                if (
                    acct_up == "ISA"
                    and not is_etf
                    and b is not None
                    and bool(
                        getattr(
                            strat,
                            "isa_value_crisis_buy_enabled",
                            ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT,
                        )
                    )
                ):
                    vg_crisis = evaluate_isa_value_gate(
                        b,
                        tech_lookup.get(sym_up) or tech_lookup.get(base),
                        macro,
                        strat,
                        item=item,
                        symbol=sym_up,
                        crisis_mode=True,
                    )
                    isa_crisis_value_exempt = bool(vg_crisis["passed"])
                if not core_etf_buy and not isa_crisis_value_exempt:
                    return (
                        "MACRO_VIX",
                        f"VIX={macro_vix:.1f}(≥{vix_danger_threshold:.0f}高危阈值)，暂停非深折价优质个股买入/加仓",
                    )

        return None

    # Branch 1: MorningReportDecision (has top_actions)
    if hasattr(decision, "top_actions"):
        return _apply_report_overrides(decision, _check_symbol_violation, cadence_advisories)

    # Branch 2: PickDecision (has selected_actions & rejected_or_frozen)
    if hasattr(decision, "selected_actions"):
        return _apply_pick_overrides(
            decision,
            _check_symbol_violation,
            portfolio=portfolio,
            strat=strat,
            bundle_lookup=bundle_lookup,
            account_totals=account_totals,
            tech_lookup=tech_lookup,
            macro=macro,
            regime=regime,
            risk_budget_scale=risk_budget_scale,
            cadence_advisories=cadence_advisories,
            core_etf_syms=core_etf_syms,
        )

    return RiskOverrideResult(decision=decision, applied=False)
