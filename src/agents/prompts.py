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
    audit_rule = (
        "\n• 客观审计原则（反迎合）：对任何已有挂单或前序 AI 结论，只要最新现价、均线结构、目标价空间扣除摩擦后无法显著跑赢"
        f"{strategy.cash_interest_rate_pct:.1f}%现金利息，必须直白建议撤单（CANCEL）或改价（MODIFY），严禁盲目维持或附和。"
        if include_order_audit_rule
        else ""
    )
    return (
        "【投资者画像与硬性规则】：\n"
        "• 英国个人投资者，持有 Trading 212 Stocks ISA 与 Invest 双账户，不使用杠杆/期权/做空。\n"
        "• Stocks ISA（长线稳健仓）：配置宽基/主题 ETF 与优质护城河个股；重确定性与复利，容忍正常波动，非基本面恶化或极端泡沫不轻易卖出。\n"
        "• Invest（短线机动仓）：聚焦事件催化、趋势回踩或超跌高赔率机会；逻辑证伪或触及止损线须果断减仓/止损，不死扛摊平。\n"
        f"• 闲置现金与挂单计息：ISA 仅有 GBP 钱包（约 {strategy.cash_interest_rate_pct:.1f}% 年化）；"
        f"Invest 为 GBP 钱包（{strategy.cash_interest_rate_pct:.1f}%）+ USD 钱包"
        f"（{getattr(strategy, 'usd_cash_interest_rate_pct', 3.4):.1f}%）双钱包，按各自币种计息，"
        "且买入限价挂单所占用的资金在未成交前照常产生同等日息（不存在“挂单期间损失利息”的机会成本）。"
        "因此严禁以“释放挂单资金吃息”为由阻止低位限价挂单或建议撤单；只有“实际成交持有后”的预期风险收益比明显跑赢"
        "对应币种无风险利息+交易摩擦时才建议出手。\n"
        f"• 交易摩擦与资金路径：① ISA 仅 GBP，英股/LSE英镑ETF 0% 换汇（英股买入另收 {strategy.uk_stamp_duty_pct:.1f}% 印花税），"
        f"美股/欧股每笔均需换汇 {strategy.fx_fee_pct:.2f}%（往返 {strategy.fx_fee_pct * 2:.2f}%）；"
        "② Invest 为 GBP+USD 双钱包，复用同币种钱包 0% 换汇；跨币种买入（GBP→USD 买美股，"
        f"或 GBP 不足时 USD→GBP 买英股/LSE ETF）计单边 {strategy.fx_fee_pct:.2f}% 换汇费，卖出所得默认留存原币种钱包；"
        f"③ 必须结合当前 GBP/USD 汇率与扣费后空间，对比对应币种现金利息（Invest 用 USD 钱包买美股比 "
        f"{getattr(strategy, 'usd_cash_interest_rate_pct', 3.4):.1f}%，用 GBP 比 {strategy.cash_interest_rate_pct:.1f}%），"
        "并在理由中写明资金路径（如「用 USD 钱包免换汇」或「GBP 不足需 USD 换回含 0.15% 损耗」）；"
        "④ Invest 为应税账户，低赔率频繁换仓会侵蚀净收益。"
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
