"""Shared prompt building blocks (concise, unified rules across all agents)."""

from __future__ import annotations

from src.config import StrategySection


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
        f"• 闲置现金基准：ISA 与 Invest 未投资现金均享有约 {strategy.cash_interest_rate_pct:.1f}% 年化利息。"
        "因此严禁把现金余额当作“必须花掉的钱”，只有预期风险收益比明显跑赢"
        f"{strategy.cash_interest_rate_pct:.1f}%无风险利息+交易摩擦时才建议出手。\n"
        f"• 交易摩擦门槛：① 非英镑资产买卖各约 {strategy.fx_fee_pct:.2f}% 换汇费（往返约 {strategy.fx_fee_pct * 2:.2f}%），"
        f"英镑资产（英股/LSE英镑ETF）免换汇费；② 买入英国个股另收 {strategy.uk_stamp_duty_pct:.1f}% 印花税（LSE ETF 免税，适合长线）；"
        "③ Invest 为应税账户，低赔率频繁换仓会侵蚀净收益。"
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
