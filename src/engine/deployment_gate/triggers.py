"""A/B/C/D deployment triggers, account decisions, and one-way PM veto clamp."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot, normalize_issuer_root
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import PortfolioData
from src.engine.portfolio_rules import ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT
from src.engine.deployment_gate.common import (
    ISA_EXTREME_LOW_CASH_PCT,
    LONDON_TZ,
    TRIGGER_EARNINGS_EXCLUDE_DAYS,
    TRIGGER_EXDIV_EXCLUDE_DAYS,
    TRIGGER_RSI_EXCLUDE_ABOVE,
    _account_value_gbp,
    _close_series,
    _round2,
    _sym_up,
)
from src.engine.deployment_gate.core_policy import _is_etf_symbol

# ---------------------------------------------------------------------------
# 4. Triggers and account-level decisions
# ---------------------------------------------------------------------------
def _core_etf_drawdown_step(
    histories: dict[str, pd.DataFrame] | None,
    strategy: StrategySection,
) -> dict[str, Any]:
    """A-trigger: benchmark 52w-high drawdown ladder (first crossing of -5/-10/-15/-20%).

    Considers both the GBP-quoted core benchmark (VWRP.L) and the USD-native regime
    benchmark (SPY) and triggers on the deeper side. In a sterling crisis the GBP
    quote can mask a genuine -15% USD drawdown, leaving the ladder permanently idle.
    """
    primary_sym = _sym_up(strategy.benchmark_symbol)
    out: dict[str, Any] = {
        "symbol": primary_sym,
        "hit": None,
        "drawdown_52w_pct": None,
        "drawdown_52w_pct_by_symbol": {},
        "first_cross": False,
    }
    samples: list[tuple[str, pd.Series, float, float]] = []
    for raw_sym in (
        primary_sym,
        _sym_up(strategy.regime_benchmark_symbol),
    ):
        if not raw_sym:
            continue
        closes = _close_series((histories or {}).get(raw_sym))
        if len(closes) < 2:
            continue
        price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        high52 = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
        if high52 <= 0:
            continue
        dd_now = (price / high52 - 1.0) * 100.0
        dd_prev = (prev_price / high52 - 1.0) * 100.0
        out["drawdown_52w_pct_by_symbol"][raw_sym] = _round2(dd_now)
        samples.append((raw_sym, closes, dd_now, dd_prev))
    if not samples:
        return out
    chosen_sym, _closes, dd_now, dd_prev = min(samples, key=lambda row: row[2])
    out["symbol"] = chosen_sym
    out["drawdown_52w_pct"] = _round2(dd_now)
    for step in sorted((float(s) for s in strategy.core_etf_drawdown_steps_pct), reverse=True):
        if dd_now <= -step and dd_prev > -step:
            out["hit"] = step
            out["first_cross"] = True
            break
    return out


def _scan_a_equity_candidates(
    candidates: list[dict[str, Any]] | None,
    portfolio: PortfolioData,
    strategy: StrategySection,
    histories: dict[str, pd.DataFrame] | None,
    *,
    etf_set: set[str] | None = None,
    bundles: dict[str, TickerMarketBundle] | None = None,
) -> list[dict[str, Any]]:
    """A-trigger: right-side uptrend equities pulling back to MA50, cross-account deduped."""
    held_or_pending = {
        _sym_up(p.yahoo_symbol) for p in portfolio.all_positions()
    } | {_sym_up(o.yahoo_symbol) for o in portfolio.all_pending_orders()}
    held_or_pending_roots = {normalize_issuer_root(s) for s in held_or_pending if s}
    known_etfs = etf_set or set()
    hits: list[dict[str, Any]] = []
    tol = float(strategy.trigger_ma_tolerance_pct)
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        sym = _sym_up(str(cand.get("symbol") or ""))
        if not sym or sym in held_or_pending or normalize_issuer_root(sym) in held_or_pending_roots:
            continue
        tech = cand.get("tech")
        bundle = cand.get("bundle") or (bundles or {}).get(sym)
        if _is_etf_symbol(sym, known_etfs, bundles) or (
            bundle is not None and str(getattr(bundle, "quote_type", "") or "").upper() == "ETF"
        ):
            continue
        if not isinstance(tech, TechnicalSnapshot):
            continue
        if bundle is not None:
            days_earn = getattr(bundle, "days_to_earnings", None)
            if days_earn is not None and 0 <= int(days_earn) <= TRIGGER_EARNINGS_EXCLUDE_DAYS:
                continue
        ex_days = None
        if bundle is not None and getattr(bundle, "days_to_ex_dividend", None) is not None:
            ex_days = int(getattr(bundle, "days_to_ex_dividend"))
        if ex_days is None and getattr(tech, "days_to_ex_dividend", None) is not None:
            ex_days = int(getattr(tech, "days_to_ex_dividend"))
        if ex_days is None and bool(getattr(tech, "is_ex_div_window", False)):
            ex_days = 0
        if ex_days is not None and 0 <= ex_days <= TRIGGER_EXDIV_EXCLUDE_DAYS:
            continue
        rsi = getattr(tech, "rsi14", None)
        if rsi is not None and float(rsi) >= TRIGGER_RSI_EXCLUDE_ABOVE:
            continue
        dist_ma50 = getattr(tech, "dist_ma50_pct", None)
        pullback = dist_ma50 is not None and abs(float(dist_ma50)) <= tol
        if not pullback:
            continue
        ma50 = getattr(tech, "ma50", None)
        ma200 = getattr(tech, "ma200", None)
        ma200_rising = None
        hist = (histories or {}).get(cand.get("symbol"))
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = _close_series(hist)
            if len(closes) >= 220:
                ma200_now = float(closes.rolling(200).mean().iloc[-1])
                ma200_prev = float(closes.rolling(200).mean().iloc[-21])
                ma200_rising = ma200_now > ma200_prev
        uptrend = bool(
            (ma50 is not None and ma200 is not None and float(ma50) > float(ma200))
            or (ma200_rising is True and getattr(tech, "trend_regime", "") in {"BULL", "PULLBACK"})
        )
        if ma200_rising is not None:
            uptrend = bool(
                ma200_rising
                and ((ma50 is not None and ma200 is not None and float(ma50) > float(ma200))
                     or getattr(tech, "trend_regime", "") in {"BULL", "PULLBACK"})
            )
        if not uptrend:
            continue
        hits.append({"symbol": sym, "dist_ma50_pct": _round2(dist_ma50), "account": str(cand.get("account") or "Invest")})
        if len(hits) >= int(strategy.trigger_max_symbols):
            break
    return hits


def _today_external_inflow(
    portfolio: PortfolioData,
    strategy: StrategySection,
    now_uk: datetime | None,
) -> dict[str, Any]:
    now = now_uk or datetime.now(LONDON_TZ)
    today_start_ms = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    per_acct: dict[str, float] = {}
    for acct in (portfolio.isa, portfolio.invest):
        net = 0.0
        for tx in acct.recent_transactions:
            if tx.date_time_ms < today_start_ms:
                continue
            if tx.tx_type == "DEPOSIT":
                net += float(tx.amount)
            elif tx.tx_type == "WITHDRAW":
                net -= float(tx.amount)
        per_acct[acct.account_label.upper()] = _round2(net)
    triggered: list[str] = []
    for label, net in per_acct.items():
        acct = portfolio.isa if label == "ISA" else portfolio.invest
        acct_val = _account_value_gbp(acct)
        if net >= float(strategy.inflow_trigger_min_gbp) and acct_val > 0 and net >= acct_val * float(strategy.inflow_trigger_pct) / 100.0:
            triggered.append(label)
    return {"per_account_net_gbp": per_acct, "triggered_accounts": triggered}


def scan_triggers_and_decide(
    portfolio: PortfolioData,
    bundles: dict[str, TickerMarketBundle] | None,
    strategy: StrategySection,
    ledgers: dict[str, Any],
    regime_info: dict[str, Any],
    histories: dict[str, pd.DataFrame] | None,
    classified: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    gbp_usd: float | None = None,
    now_uk: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate A/B/C/D triggers and produce per-account + global gate decisions."""
    isa = ledgers["isa"]
    invest = ledgers["invest"]
    regime = regime_info.get("regime", "UNKNOWN")

    a_core = _core_etf_drawdown_step(histories, strategy)
    a_equity = _scan_a_equity_candidates(
        candidates,
        portfolio,
        strategy,
        histories,
        etf_set=set(ledgers.get("etf_set") or []),
        bundles=bundles,
    )

    b_dca = bool(
        isa["ok"]
        and isa["deployable_pct"] > float(strategy.isa_cash_band_high_pct) + float(strategy.min_deploy_excess_pct)
        and isa["committed_core_etf_pct"] < float(strategy.isa_core_etf_target_pct)
    )
    inflow = _today_external_inflow(portfolio, strategy, now_uk)
    c_inflow = bool(inflow["triggered_accounts"])

    d_defend = bool(
        regime == "RISK_OFF"
        and (
            invest["pending_equity_buys_all"]
            or isa["pending_new_equity_buys"]
            or invest["slots_used"] > 0
        )
    )
    isa_extreme_low_cash = bool(
        isa["ok"] and isa["total_cash_pct"] < ISA_EXTREME_LOW_CASH_PCT
    )
    # v5.5.1: ISA keeps a staged contrarian value lane in a crisis (quality + >=20% discount),
    # while Invest stays 100% frozen (slots_cap = 0, DEFEND).
    crisis_value_enabled = bool(
        getattr(strategy, "isa_value_crisis_buy_enabled", ISA_VALUE_CRISIS_BUY_ENABLED_DEFAULT)
    )
    # Persistent deep drawdown (>=-10%) during RISK_OFF must keep the core-ETF channel open
    # even after the first-cross day has passed, otherwise a whole sideways bottom is missed.
    persistent_deep_drawdown = bool(
        isa["ok"]
        and a_core.get("drawdown_52w_pct") is not None
        and float(a_core["drawdown_52w_pct"]) <= -10.0
        and isa["deployable_pct"] > float(strategy.isa_cash_band_high_pct)
        and isa["committed_core_etf_pct"] < float(strategy.isa_core_etf_target_pct)
        and len(isa["pending_core_etf_buys"]) == 0
    )

    # ---------- ISA decision ----------
    if not isa["ok"]:
        isa_status, isa_mode = "UNKNOWN", "none"
        isa_reason = "ISA 账户快照降级/不可用，不催买不乱喊减仓"
    else:
        ladder_hit = a_core.get("hit")
        if regime == "RISK_OFF":
            if ladder_hit is not None and float(ladder_hit) >= 10.0:
                isa_status, isa_mode = "RUN", "ladder"
                isa_reason = f"退潮期 ISA 个股通道关闭，宽基深跌{-float(ladder_hit):.0f}%阶梯逆向播种"
            elif persistent_deep_drawdown:
                isa_status, isa_mode = "RUN", "ladder"
                isa_reason = (
                    f"退潮期宽基持续深回撤{float(a_core['drawdown_52w_pct']):.1f}%"
                    "且尚未挂出接针单，允许补挂核心宽基分批买单"
                )
            else:
                isa_status, isa_mode = "WAIT", "none"
                isa_reason = "退潮期冻结新开个股，宽基未达深跌阶梯暂缓"
        elif c_inflow and "ISA" in inflow["triggered_accounts"]:
            isa_status, isa_mode = "RUN", "inflow"
            isa_reason = f"外部净入金 £{inflow['per_account_net_gbp']['ISA']:,.0f} 达标，引导分批配置核心宽基"
        elif b_dca:
            isa_status, isa_mode = "RUN", "dca"
            if isa.get("prefer_core_diversifier"):
                policy = isa.get("core_policy") or {}
                gap_txt = float(policy.get("eu_gap_pct") or 0.0) + float(policy.get("uk_gap_pct") or 0.0)
                isa_reason = (
                    f"ISA 可用现金 {isa['deployable_pct']:.1f}%>上沿且宽基仅 {isa['committed_core_etf_pct']:.0f}%（含挂单），"
                    f"优先补 FTSE100/欧洲宽基缺口 {gap_txt:.1f}%，分散美股科技，勿只加 VWRP"
                )
            else:
                isa_reason = f"ISA 可用现金 {isa['deployable_pct']:.1f}%>上沿且宽基仅 {isa['committed_core_etf_pct']:.0f}%（含挂单），分批归拢 VWRP"
        elif ladder_hit is not None:
            isa_status, isa_mode = "RUN", "ladder"
            isa_reason = f"核心宽基距52周高点回撤首次触及 -{float(ladder_hit):.0f}% 阶梯"
        else:
            isa_status, isa_mode = "WAIT", "none"
            isa_reason = "现金带与宽基占比均在区间内"
        if isa["isa_satellite_saturated"]:
            isa_reason = f"ISA 个股卫星层饱和(持仓+挂单 {isa['committed_satellite_count']}只/{isa['committed_satellite_pct']:.1f}%)，仅配宽基：" + isa_reason
    isa["status"] = isa_status
    isa["mode"] = isa_mode
    isa["reason"] = isa_reason

    # ---------- Invest decision (never dca on idle cash) ----------
    slots_cap = {
        "RISK_ON": int(strategy.invest_slots_risk_on),
        "NEUTRAL": int(strategy.invest_slots_neutral),
        "RISK_OFF": int(strategy.invest_slots_risk_off),
        "UNKNOWN": int(strategy.invest_slots_neutral),
    }.get(regime, int(strategy.invest_slots_neutral))
    invest["slots_cap"] = slots_cap

    if not invest["ok"]:
        invest_status, invest_mode = "UNKNOWN", "none"
        invest_reason = "Invest 账户快照降级/不可用"
    elif regime == "RISK_OFF":
        invest_status, invest_mode = "DEFEND", "defend"
        invest_reason = "市场退潮：0 槽位冻结新开个股，启动四步渐进防守"
    elif invest["in_loss_cooldown"]:
        invest_status, invest_mode = "WAIT", "cooldown"
        invest_reason = (
            f"gate_epoch_date 后近{strategy.consecutive_loss_lookback_days}天连损"
            f"{invest['tactical_losses_recent']}笔，冷静{strategy.consecutive_loss_cooldown_days}个交易日"
        )
    elif invest["daily_cap_reached"]:
        invest_status, invest_mode = "WAIT", "daily_cap"
        invest_reason = (
            f"今日已新开/新挂 {invest['today_new_equity_entries']} 笔个股"
            f"(上限{strategy.max_new_equity_entries_per_day})，Best Idea Only 单日止步"
        )
    elif invest["slots_used"] >= slots_cap and slots_cap > 0:
        if a_equity:
            invest_status, invest_mode = "ROTATE", "rotate"
            funding_from = ""
            if invest["pending_new_equity_buys"]:
                funding_from = invest["pending_new_equity_buys"][0]
            elif invest.get("pending_add_equity_buys"):
                funding_from = invest["pending_add_equity_buys"][0]
            elif invest["unfinanced_positions"]:
                funding_from = invest["unfinanced_positions"][0]
            invest["funding_from"] = funding_from
            invest_reason = (
                f"在险槽位 {invest['slots_used']}/{slots_cap} 已满但右侧强信号命中，"
                f"须先腾挪 {funding_from} 等风险置换"
            )
        else:
            invest_status, invest_mode = "WAIT", "slots_full"
            invest_reason = (
                f"在险槽位 {invest['slots_used']}/{slots_cap} 已满"
                f"({','.join(invest['unfinanced_positions']) or '无'} 未脱险)，静待推保本解锁"
            )
    elif a_equity:
        invest_status, invest_mode = "RUN", "targeted"
        invest_reason = f"右侧猎物触发({','.join(h['symbol'] for h in a_equity)})，在险槽位 {invest['slots_used']}/{slots_cap}"
    else:
        invest_status, invest_mode = "WAIT", "none"
        invest_reason = "无右侧猎物，现金吃息等待(绝不因现金多催买)"
    invest["status"] = invest_status
    invest["mode"] = invest_mode
    invest["reason"] = invest_reason

    if d_defend or isa_extreme_low_cash:
        invest["defend_note"] = (
            "①优先撤除 Invest 未成交个股买单；ISA 低位价值买单须核查基本面与≥20%安全边际，"
            "未破逻辑不盲撤 ②保护保本仓与已挂止损逆势仓，只收紧止盈 ③清退 Invest 无保护浮亏弱股"
        )
        if isa_extreme_low_cash:
            invest["defend_note"] += " ④ISA 现金跌破2%结构垫，停止一切非必要支出/买入"

    triggers = {
        "A": {
            "core_etf_step": a_core,
            "equity_candidates": a_equity,
        },
        "B": {"isa_core_dca": b_dca},
        "C": inflow,
        "D": {
            "risk_off_defend": d_defend,
            "isa_extreme_low_cash": isa_extreme_low_cash,
        },
    }

    # ---------- Global merge: DEFEND > RUN > ROTATE > WAIT > UNKNOWN ----------
    statuses = {isa_status, invest_status}
    if "DEFEND" in statuses:
        global_status = "DEFEND"
    elif "RUN" in statuses:
        global_status = "RUN"
    elif "ROTATE" in statuses:
        global_status = "ROTATE"
    elif "WAIT" in statuses:
        global_status = "WAIT"
    else:
        global_status = "UNKNOWN"

    # ---------- Default pick gate (deterministic baseline, PM may only tighten) ----------
    isa["value_crisis_mode"] = bool(regime == "RISK_OFF" and crisis_value_enabled)
    isa_crisis_value_open = bool(
        regime == "RISK_OFF"
        and crisis_value_enabled
        and not isa_extreme_low_cash
    )
    isa_equity_open = bool(
        isa["ok"]
        and not isa["isa_satellite_saturated"]
        and (regime != "RISK_OFF" or isa_crisis_value_open)
    )
    # In RISK_OFF the ISA wide-base channel only opens on a valid drawdown ladder step;
    # otherwise the gate WAIT text and the actual permission would contradict each other.
    isa_etf_open = bool(
        isa["ok"]
        and (regime != "RISK_OFF" or a_core.get("hit") is not None or persistent_deep_drawdown)
    )
    if regime == "RISK_OFF" and isa_equity_open:
        crisis_note = "ISA 保留深度安全边际价值分批通道（≥20%折价+高质量，严禁追高/接低质飞刀）"
        if crisis_note not in str(isa.get("reason") or ""):
            isa["reason"] = f"{isa.get('reason') or ''}；{crisis_note}".strip("；")
    isa["equity_open"] = isa_equity_open
    isa["etf_open"] = isa_etf_open
    invest_open = invest_status == "RUN"
    invest_rotate = invest_status == "ROTATE"
    allow = isa_equity_open or isa_etf_open or invest_open or invest_rotate
    if invest_rotate:
        pm_mode, pm_funding = "rotate", invest.get("funding_from", "")
        pm_reason = invest_reason
    elif allow:
        pm_mode, pm_funding = "run", ""
        pm_reason = (
            "Invest 已解锁右侧槽位" if invest_open else
            ("ISA 个股饱和，仅限宽基 ETF" if isa["isa_satellite_saturated"] else "存在可用开仓通道")
        )
    else:
        pm_mode, pm_funding = "none", ""
        pm_reason = "当前无任何可用开仓通道"

    return {
        "isa": isa,
        "invest": invest,
        "regime": regime,
        "triggers": triggers,
        "global_status": global_status,
        "pm": {
            "allow": allow,
            "mode": pm_mode,
            "funding_from": pm_funding,
            "reason": pm_reason,
            "pm_overridden": False,
        },
        "gbp_usd": _round2(gbp_usd) if gbp_usd else None,
        "fx_missing": gbp_usd is None,
    }


# ---------------------------------------------------------------------------
# 5. One-way PM veto clamp
# ---------------------------------------------------------------------------
def reconcile_pm_pick_gate(
    gate_dict: dict[str, Any],
    pm_pick_gate: dict[str, Any] | None,
    whitelist_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """Enforce the one-way PM contract on the final pick gate.

    PM may only tighten (allow=false); Python can never be upgraded from WAIT / DEFEND /
    UNKNOWN into RUN. A ROTATE without a legal funding source degrades to WAIT, and any
    funding symbol outside the whitelist is stripped.
    """
    pm = dict(gate_dict.get("pm") or {})
    whitelist = {_sym_up(s) for s in (whitelist_symbols or set()) if s}
    pm["pm_overridden"] = False

    if not isinstance(pm_pick_gate, dict) or not pm_pick_gate:
        return pm

    wants_allow = bool(pm_pick_gate.get("allow", True))
    if not wants_allow:
        pm["allow"] = False
        pm["mode"] = "none"
        pm["funding_from"] = ""
        pm["reason"] = str(pm_pick_gate.get("reason") or "PM 主动收紧否决")
        return pm

    # PM wants allow=True — deterministic bottom line decides.
    base_mode = pm.get("mode", "none")
    if base_mode == "none":
        pm["allow"] = False
        pm["mode"] = "none"
        pm["funding_from"] = ""
        pm["reason"] = (
            f"组合门控底线 {gate_dict.get('global_status', 'WAIT')} 态，PM 无权升级为 RUN"
        )
        pm["pm_overridden"] = True
        return pm
    if base_mode == "rotate":
        funding = str(pm_pick_gate.get("funding_from") or pm.get("funding_from") or "").strip()
        funding_up = _sym_up(funding)
        if not funding_up or (whitelist and funding_up not in whitelist):
            pm["allow"] = False
            pm["mode"] = "none"
            pm["funding_from"] = ""
            pm["reason"] = "ROTATE 缺合法置换源(未平弱仓/未撤旧买单)，自动降级 WAIT"
            pm["pm_overridden"] = True
            return pm
        pm["allow"] = True
        pm["mode"] = "rotate"
        pm["funding_from"] = funding_up
        pm["reason"] = str(pm_pick_gate.get("reason") or pm.get("reason") or "等风险置换")
        return pm
    # base_mode == "run"
    pm["allow"] = True
    pm["mode"] = "run"
    pm["funding_from"] = ""
    pm["reason"] = str(pm_pick_gate.get("reason") or pm.get("reason") or "允许运行 /pick")
    return pm
