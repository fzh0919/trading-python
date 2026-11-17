"""Shared prompt building blocks (concise, unified rules across all agents)."""

from __future__ import annotations

from src.config import StrategySection
from src.data_providers.t212_client import PortfolioData


def build_wallet_context_text(
    portfolio: PortfolioData,
    strategy: StrategySection,
    *,
    compact: bool = False,
) -> str:
    """Render the ISA GBP / Invest GBP+USD wallet ledger for dynamic prompt context only.

    Never place this text inside ``build_investor_profile`` — amounts change daily and
    would break the Gemini implicit-cache static prefix.
    """
    isa = portfolio.isa
    inv = portfolio.invest
    gbp_rate = float(strategy.cash_interest_rate_pct)
    usd_rate = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    fx = float(strategy.fx_fee_pct)
    stamp = float(strategy.uk_stamp_duty_pct)
    rate = getattr(portfolio, "gbp_usd_rate", None)
    rate_str = f"GBP/USD {rate:.4f}" if rate else "GBP/USD 暂缺"

    multi = bool(getattr(inv, "multi_currency_enabled", False))
    gbp_total = getattr(inv, "gbp_wallet_total", None) if multi else None
    if compact:
        parts = [f"ISA £{isa.total_cash:,.2f}(可用 £{isa.available_to_trade:,.2f}, {gbp_rate:.1f}%)"]
        if multi:
            gbp_part = f"GBP≈£{gbp_total:,.2f}" if gbp_total is not None else "GBP待折算"
            parts.append(
                f"Invest £{inv.total_cash:,.2f}({gbp_part} {gbp_rate:.1f}% + "
                f"USD ${inv.usd_cash_total:,.2f} {usd_rate:.1f}%)"
            )
        else:
            parts.append(f"Invest £{inv.total_cash:,.2f}(可用 £{inv.available_to_trade:,.2f}, {gbp_rate:.1f}%)")
        line = f"现金({rate_str}): " + " | ".join(parts)
        warning = str(getattr(inv, "wallet_reconcile_warning", "") or "")
        if warning:
            line += f" | ⚠️ {warning[:70]}"
        return line

    lines = [f"【现金与双钱包（GBP 口径，实时 {rate_str}）】"]
    lines.append(
        f"• ISA: GBP 现金 £{isa.total_cash:,.2f}（可用 £{isa.available_to_trade:,.2f} / 挂单 £{isa.reserved_cash:,.2f}），"
        f"利息 {gbp_rate:.1f}%；英股/LSE英镑ETF 0% 换汇（英股买入另收 {stamp:.1f}% 印花税），美/欧股每笔换汇 {fx:.2f}%"
    )
    if multi:
        if gbp_total is None:
            gbp_text = "GBP 钱包 待汇率折算"
        else:
            gbp_text = (
                f"GBP 钱包 ≈ £{gbp_total:,.2f}（可用 £{inv.gbp_wallet_available or 0.0:,.2f} / "
                f"挂单 £{inv.gbp_wallet_reserved or 0.0:,.2f}，{gbp_rate:.1f}%）"
            )
        usd_equiv = f" ≈ £{inv.usd_cash_total / rate:,.2f}" if rate else ""
        lines.append(
            f"• Invest: T212 总现金 £{inv.total_cash:,.2f} = {gbp_text} + "
            f"USD 钱包 ${inv.usd_cash_total:,.2f}（可用 ${inv.usd_available_to_trade:,.2f} / "
            f"挂单 ${inv.usd_reserved_cash:,.2f}{usd_equiv}，{usd_rate:.1f}%）"
        )
        lines.append(
            f"• Invest 资金路径: 复用同币种钱包 0% 换汇；跨币种买入（GBP→USD 买美股，"
            f"或 GBP 不足时 USD→GBP 买英股/LSE ETF）计单边 {fx:.2f}% 换汇费；卖出所得默认留存原币种钱包"
        )
        warning = str(getattr(inv, "wallet_reconcile_warning", "") or "")
        if warning:
            lines.append(f"⚠️ 钱包校准: {warning}")
    else:
        lines.append(
            f"• Invest: GBP 现金 £{inv.total_cash:,.2f}（可用 £{inv.available_to_trade:,.2f} / "
            f"挂单 £{inv.reserved_cash:,.2f}），利息 {gbp_rate:.1f}%"
        )
    return "\n".join(lines)


def build_investor_profile(
    strategy: StrategySection,
    *,
    include_order_audit_rule: bool = True,
) -> str:
    benchmark = getattr(strategy, "benchmark_symbol", "VWRP.L")
    gbp_rate = float(strategy.cash_interest_rate_pct)
    usd_rate = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    min_invest = float(getattr(strategy, "min_invest_order_gbp", 700.0))
    min_isa_eq = float(getattr(strategy, "min_isa_equity_order_gbp", 1200.0))
    min_isa_etf = float(getattr(strategy, "min_isa_etf_order_gbp", 400.0))
    risk_budget = float(getattr(strategy, "invest_risk_budget_gbp", 85.0))
    isa_trigger = float(getattr(strategy, "isa_cash_deploy_trigger_pct", 15.0))
    invest_idle = float(getattr(strategy, "invest_cash_idle_warn_pct", 40.0))
    weekly_cap = int(getattr(strategy, "max_weekly_new_buys", 2))
    monthly_cap = int(getattr(strategy, "max_monthly_new_buys", 6))
    cooldown_days = int(getattr(strategy, "rebuy_cooldown_days", 15))
    spread_us = float(getattr(strategy, "spread_estimate_us_pct", 0.05))
    spread_uk = float(getattr(strategy, "spread_estimate_uk_eu_pct", 0.20))
    cadence_mode = str(getattr(strategy, "trade_cadence_guard_mode", "advisory") or "advisory").strip().lower()
    if cadence_mode == "hard":
        cadence_rule = (
            f"频率硬约束（Python 强制拦截）：近 7 天新开仓 ≤{weekly_cap} 笔、近 30 天 ≤{monthly_cap} 笔，"
            f"清仓后 {cooldown_days} 天内不得同标的回补。"
        )
    elif cadence_mode == "off":
        cadence_rule = (
            f"频率仅作参考（历史节奏约 7 天 {weekly_cap} 笔 / 30 天 {monthly_cap} 笔，清仓 {cooldown_days} 天后再回补更稳妥），完全由你判断。"
        )
    else:
        cadence_rule = (
            f"频率参考区间（非硬性禁止，由你判断）：近 7 天 ≤{weekly_cap} 笔、近 30 天 ≤{monthly_cap} 笔，"
            f"清仓后 {cooldown_days} 天内同标的回补需谨慎；若确有必要超额开仓或抢回补，须写明催化/避险依据。"
        )
    audit_rule = (
        "\n• 客观审计原则（反迎合，一律按挂单限价核算）：对任何已有挂单或前序 AI 结论，以该挂单的限价（而非现价）为成本基线估算目标空间与盈亏比；"
        "距现价 -3%~-12% 且基本面未坏的回踩埋伏单属于有效等待，严禁以“现价空间不足”或“释放现金吃息”为由建议撤单；"
        "只有当按限价成交后扣摩擦的盈亏比仍 <2:1、或基本面/趋势已证伪时，才给出 CANCEL 或带具体数字的 MODIFY。"
        if include_order_audit_rule
        else ""
    )
    return (
        "【投资者画像与硬性规则】：\n"
        "• 英国个人投资者，持有 Trading 212 Stocks ISA 与 Invest 双账户，不使用杠杆/期权/做空。\n"
        f"• 核心宗旨：双账户合计净值长期跑赢全球宽基基准 {benchmark}（大盘参照）；"
        f"{gbp_rate:.1f}%/{usd_rate:.1f}% 现金利息只是等待期保底收益与 Invest 空窗底线，不是投资终点，严禁把“守住现金吃息”当默认结论。\n"
        "• Stocks ISA（长线稳健仓）：配置宽基/主题 ETF 与优质护城河个股；重确定性与复利，容忍正常波动，非基本面恶化或极端泡沫不轻易卖出。\n"
        "• Invest（短线机动仓）：聚焦事件催化、趋势回踩或超跌高赔率机会；逻辑证伪或触及止损线须果断减仓/止损，不死扛摊平。\n"
        f"• 资金分桶与部署纪律：① ISA 现金占比 >{isa_trigger:.0f}% 时须给出具体价位与金额的宽基/优质低估值长线分批配置提案；"
        f"② Invest 现金长期 >{invest_idle:.0f}% 须说明为何没有达标的 R:R ≥ 2:1 机会（资金效率提醒）；"
        "③ 但严禁为“把钱花掉或凑满仓位”而降低标准，宁可不交易也不做 <2:1 的低赔率单。\n"
        f"• 卖出三分类（其余情况不下卖出结论）：① 逻辑证伪或触及硬止损；② 长期弱势机会成本置换（连续跑输 {benchmark} 且跌破长期趋势如 MA200）；"
        "③ 穿透后集中度超限的再平衡。\n"
        f"• 开仓硬纪律：单笔金额 Invest ≥ £{min_invest:,.0f}、ISA 个股 ≥ £{min_isa_eq:,.0f}、ISA ETF ≥ £{min_isa_etf:,.0f}，严禁碎单；"
        f"以损定仓——单笔最大可承受亏损约 £{risk_budget:,.0f}（仓位 ≈ £{risk_budget:,.0f} ÷ 止损百分比）；"
        f"{cadence_rule}\n"
        f"• 闲置现金与挂单计息：ISA 仅有 GBP 钱包（约 {gbp_rate:.1f}% 年化）；"
        f"Invest 为 GBP 钱包（{gbp_rate:.1f}%）+ USD 钱包（{usd_rate:.1f}%）双钱包，按各自币种计息，"
        "且买入限价挂单所占用的资金在未成交前照常产生同等日息（不存在“挂单期间损失利息”的机会成本）。"
        "因此严禁以“释放挂单资金吃息”为由阻止低位限价挂单或建议撤单。\n"
        f"• 交易摩擦与资金路径：① ISA 仅 GBP，英股/LSE英镑ETF 0% 换汇（英股买入另收 {strategy.uk_stamp_duty_pct:.1f}% 印花税），"
        f"美股/欧股每笔均需换汇 {strategy.fx_fee_pct:.2f}%（往返 {strategy.fx_fee_pct * 2:.2f}%）；买卖价差按保守估算计入"
        f"（美股约 {spread_us:.2f}%，英股/LSE ETF 约 {spread_uk:.2f}%）；"
        "② Invest 为 GBP+USD 双钱包，复用同币种钱包 0% 换汇；跨币种买入（GBP→USD 买美股，"
        f"或 GBP 不足时 USD→GBP 买英股/LSE ETF）计单边 {strategy.fx_fee_pct:.2f}% 换汇费，卖出所得默认留存原币种钱包；"
        "③ 理由中须写明资金路径（如「用 USD 钱包免换汇」或「GBP 不足需 USD 换回含损耗」）；"
        "④ Invest 为应税账户，低赔率频繁换仓会侵蚀净收益。"
        "\n"
        f"• ISA 核心-卫星结构（v5.4.0）：ISA 核心宽基目标占比 ≥{strategy.isa_core_etf_target_pct:.0f}%（统一归拢 {benchmark}），"
        f"个股卫星硬上限 ≤{strategy.isa_satellite_max_total_pct:.0f}% 且 ≤{strategy.isa_satellite_max_count} 只"
        "（含未成交个股买单）；一旦超限即锁死 ISA 新开个股通道，只准审撤个股买单并把闲钱分批归拢核心宽基；"
        f"存量 <£{min_isa_eq:,.0f} 的历史小额散仓享老仓豁免，只在反弹接近回本/阻力窗顺势归拢，严禁低位急割。\n"
        f"• 零 Quota 豁免（v5.4.0）：卖出任何存量仓位/清理边角料，以及买入核心宽基 ETF（{benchmark} 等），"
        "永远不消耗周/月开仓次数配额，也不占用在险槽位。\n"
        f"• Invest 在险槽位与以盈解锁（v5.4.0）：Invest 现金再多也绝不催买，只有右侧突破猎物才触发开仓；"
        "已挂保本止损（止损价≥成本+摩擦）的仓位占用 0 槽位并立即释放新开仓额度，未脱险仓位与未成交个股买单各占 1 槽位；"
        f"同一交易日最多新开 1 笔个股；近 {strategy.consecutive_loss_lookback_days} 天连损 {strategy.consecutive_loss_limit} 笔"
        f"自动冷静 {strategy.consecutive_loss_cooldown_days} 个交易日，恢复首单按半仓试探（约 £{risk_budget * strategy.recovery_pilot_risk_scale:,.0f}），首单盈利推保本后恢复全额火力。"
        f"{audit_rule}"
    )


def build_agent_system_instruction(
    role_instruction: str,
    strategy: StrategySection,
    *,
    include_order_audit_rule: bool = True,
) -> str:
    """Combine stable role instructions and investor profile for Gemini `system_instruction`."""
    return (
        f"{role_instruction.strip()}\n\n"
        f"{build_investor_profile(strategy, include_order_audit_rule=include_order_audit_rule)}"
    )


def format_preference_notes_block(
    notes: list[dict[str, Any]] | None,
    *,
    max_notes: int = 6,
) -> str:
    """Render TTL-bound user preferences with verbatim quote anchors (never paraphrase-only)."""
    if not notes:
        return ""
    lines: list[str] = []
    for note in notes[: max(1, max_notes)]:
        if not isinstance(note, dict):
            continue
        quote = str(note.get("quote_text") or "").strip()
        rule = str(note.get("rule_summary") or "").strip()
        if not quote or not rule:
            continue
        date = str(note.get("source_cycle_date") or "").strip()[:10]
        expires = str(note.get("expires_on_uk") or "").strip()[:10]
        head = f'[{date} · 原话: "{quote[:80]}"]' if date else f'[原话: "{quote[:80]}"]'
        tail = f"（有效至 {expires}）" if expires else ""
        lines.append(f"• {head} {rule}{tail}")
    if not lines:
        return ""
    return "【用户长期偏好与资金计划（含原话锚点，优先遵守）】：\n" + "\n".join(lines)
