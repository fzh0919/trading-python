"""Shared prompt building blocks (concise, unified rules across all agents)."""

from __future__ import annotations

import re
from typing import Any

from src.config import StrategySection
from src.data_providers.t212_client import PortfolioData


# v5.5.8: plain-text helpers shared by chat_agent and report_agent/sentinels.
# They live here (the common upstream of both) so sentinels.py never has to
# reverse-import chat_agent.py.
_MD_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")


def _normalize_redline_text(text: str) -> str:
    """v5.5.8: strip trailing sentence punctuation (。；.;) per line for redline comparison.

    The morning report joins vetoes with `；` and drops trailing `。`, while DB guardrails
    keep theirs — a strict substring compare therefore always failed and the redlines were
    injected twice (~180 wasted tokens).
    """
    normalized: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        while line and line[-1] in "。；.;":
            line = line[:-1].rstrip()
        normalized.append(line)
    return "\n".join(normalized)


def _all_guardrails_embedded(morning_block: str, guardrails: list[str]) -> bool:
    if not guardrails:
        return True
    morning_norm = _normalize_redline_text(morning_block)
    return all(
        bool(_normalize_redline_text(g)) and _normalize_redline_text(g) in morning_norm
        for g in guardrails
    )


def has_complete_embedded_vetoes(morning_block: str, guardrails: list[str]) -> bool:
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
            return _all_guardrails_embedded(morning_block, guardrails)
        if stripped.startswith("【今日风控红线") and idx + 1 < len(lines):
            next_ln = lines[idx + 1].strip()
            if next_ln and not next_ln.endswith(("…", "...")):
                return _all_guardrails_embedded(morning_block, guardrails)
    return False


def clean_telegram_plain_text(text: str) -> str:
    """v5.5.8: strip Telegram-hostile Markdown (** pairs, leading # headings) only.

    Never touches single `*` (protects 3*ATR math), preserves newlines / • lists / 【】
    sections, and never truncates.
    """
    if not text:
        return text
    cleaned = text.replace("**", "")
    cleaned = _MD_HEADING_RE.sub("", cleaned)
    return cleaned


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
        lines.append(
            f"• 跨钱包补足: 若按风险算出的美股建议金额大于当前 USD 钱包可用余额，"
            f"差额正常由 GBP 钱包换汇补足（计单边 {fx:.2f}% 换汇费）；"
            "严禁为省换汇费而人为把仓位压缩到 USD 余额以内。"
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
    compact_for_pick_critic: bool = False,
    compact_for_chat: bool = False,
    compact_for_us_sentinel: bool = False,
) -> str:
    if compact_for_us_sentinel:
        # v5.5.8: /us only does incremental verification against the morning report;
        # it must not re-adjudicate /report /pick machinery (cuts ~1,850 -> ~450 tokens
        # and stops inviting order re-audits the morning report already settled).
        return _build_compact_us_sentinel_profile(strategy)
    if compact_for_chat:
        # v5.5.8: daily chat only needs the Q&A-critical discipline, not the full
        # /report & /pick machinery (cuts systemInstruction from ~1,900 to ~1,000 tokens).
        return _build_compact_chat_profile(strategy, include_order_audit_rule=include_order_audit_rule)
    if compact_for_pick_critic:
        return _build_compact_pick_critic_profile(strategy)
    benchmark = getattr(strategy, "benchmark_symbol", "VWRP.L")
    gbp_rate = float(strategy.cash_interest_rate_pct)
    usd_rate = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    min_invest = float(getattr(strategy, "min_invest_order_gbp", 700.0))
    min_pilot = float(getattr(strategy, "min_invest_pilot_order_gbp", 500.0))
    min_isa_eq = float(getattr(strategy, "min_isa_equity_order_gbp", 1200.0))
    min_isa_etf = float(getattr(strategy, "min_isa_etf_order_gbp", 400.0))
    risk_budget = float(getattr(strategy, "invest_risk_budget_gbp", 85.0))
    risk_pct = float(getattr(strategy, "invest_risk_budget_pct", 0.75) or 0.75)
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
        "• Stocks ISA（长线稳健仓，巴菲特价值流派）：核心宽基 + 优质护城河个股；"
        "不把短期股价波动当风险——优质股下跌若基本面与护城河未证伪、且估值安全边际扩大，反而是分批左侧加仓机会；"
        "严禁仅因浮盈 25%、RSI≥75 或短期回撤 12% 卖飞好公司；"
        "ISA 卖出/减仓仅限 4 情形：基本面/护城河证伪、极端估值泡沫、显著更优机会成本置换、集中度超限/边角料归核。\n"
        "• Invest（短线机动仓，右侧动量战术流派）：严守硬止损与以损定仓、绝不摊平；"
        "按当日四态市场状态（RISK_ON 强攻 / NEUTRAL 震荡 / RECOVERY 早春轻兵试探 / RISK_OFF 危机防御）"
        "动态适配 R:R（RISK_ON 右侧领涨 ≥1.6:1 / NEUTRAL ≥2.0:1 / 弱势 ≥2.5:1；RECOVERY 仅 1 槽位/1.0%热力/0.6R试探，VIX<22 为 ≥2.0:1、VIX≥22 收紧至 ≥2.5:1）"
        "与止损 ATR 噪声地板（禁止窄止损作弊）；"
        "非 RISK_OFF 下确认强领涨股（MA200上方+BULL/PULLBACK+周线不空+RS20>0且RS60>0）目标上限解锁双锚"
        "max(卖方均值×1.15, 卖方高位×0.90, 入场+4×ATR)，非领涨/弱势/RISK_OFF 标的仍严锁卖方均值 ±15%；"
        "牛市站稳 MA20 不因单一 RSI 急卖、而是上移移动止损；"
        "Invest 浮盈+15%推保本（止损上移至成本+往返摩擦即转0热力/0槽位解锁新火力），"
        "浮盈+25%站稳MA20无放量派发不催卖、改将止损抬升至max(成本+10%锁盈地板,沿MA20跟踪)让利润奔跑；"
        "震荡/早春/退潮收紧锁利并结合前瞻 R:R（<0.8:1）做置换。\n"
        f"• 资金分桶与部署纪律：① ISA 现金占比 >{isa_trigger:.0f}% 时须给出具体价位与金额的宽基/优质低估值长线分批配置提案；"
        f"② Invest 现金长期 >{invest_idle:.0f}% 须说明为何没有达到当日动态 R:R 门槛（RISK_ON≥1.6 / NEUTRAL≥2.0 / 弱势≥2.5）的机会（资金效率提醒）；"
        "③ 但严禁为“把钱花掉或凑满仓位”而降低标准，宁可不交易也不做低于当日门槛的低赔率单；"
        f"④ Invest 溢出现金仅在 Python 签发派息宽基蓄水池票据时停泊（仅 DIST 白名单 {'/'.join(getattr(strategy, 'invest_reservoir_etf_whitelist', ['VWRL.L', 'ISF.L', 'VERX.L', 'VEUR.L']))}，"
        "零槽位零热力零Quota，应税账户严禁自选代码或买累积型 ETF）；"
        "⑤ 若右侧个股建仓金额>Invest 可用现金且账户持有蓄水池，由 Python 确定性生成“先卖蓄水池宽基释放现金→再建仓”配对指令（含价差/汇率缓冲），"
        "严禁自行口算股数或绕过该指令另筹资金。\n"
        f"• 卖出分类（其余情况不下卖出结论）：① 逻辑证伪或触及硬止损；② 长期弱势机会成本置换（连续跑输 {benchmark} 且跌破长期趋势如 MA200）；"
        "③ 穿透后集中度超限的再平衡；④ ISA 另含极端估值泡沫与历史边角料归核。\n"
        f"• 开仓硬纪律：单笔金额 Invest ≥ £{min_invest:,.0f}（试探缩放期地板 £{min_pilot:,.0f}）、ISA 个股 ≥ £{min_isa_eq:,.0f}、ISA ETF ≥ £{min_isa_etf:,.0f}，严禁碎单；"
        f"以损定仓——Invest 单笔最大可承受亏损 = max(£{risk_budget:,.0f} 地板, Invest 净值×{risk_pct:.2f}%)（仓位 ≈ 该风险预算 ÷ 止损百分比）；"
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
        "若按风险算出的美股建议金额大于当前 USD 钱包可用余额，超额部分正常由 GBP 钱包换汇补足（计单边换汇费），"
        "严禁为省换汇费而人为把仓位压缩到 USD 余额以内；"
        "③ 理由中须写明资金路径（如「用 USD 钱包免换汇」或「GBP 不足需 USD 换回含损耗」）；"
        "④ Invest 为应税账户，低赔率频繁换仓会侵蚀净收益。"
        "\n"
        f"• ISA 核心-卫星结构（v5.5.6）：ISA 核心宽基目标占比 ≥{strategy.isa_core_etf_target_pct:.0f}%（按全球/美股/欧洲/英国四桶宽基政策配置，严禁偏离 Python 政策票据），"
        f"个股卫星硬上限 ≤{strategy.isa_satellite_max_total_pct:.0f}% 且 ≤{strategy.isa_satellite_max_count} 只"
        "（含未成交个股买单）；一旦超限即锁死 ISA 新开个股通道，只准审撤个股买单并把闲钱分批归拢核心宽基；"
        f"存量 <£{min_isa_eq:,.0f} 的历史小额散仓享老仓豁免，只在反弹接近回本/阻力窗顺势归拢，严禁低位急割；"
        f"已挂全额卖出限价单的边角料不计入 {strategy.isa_satellite_max_count} 只锁定席位（撤销卖单即自动回锁）。\n"
        f"• 零 Quota 豁免（v5.4.0）：卖出任何存量仓位/清理边角料，以及买入核心宽基 ETF（{benchmark} 等），"
        "永远不消耗周/月开仓次数配额，也不占用在险槽位。\n"
        f"• Invest 在险槽位与以盈解锁（v5.4.0）：Invest 现金再多也绝不催买个股，只有右侧突破猎物才触发开仓"
        "（溢出现金仅按 Python 蓄水池票据停泊派息宽基，属现金停泊而非催买）；"
        "已挂保本止损（止损价≥成本+摩擦）的仓位占用 0 槽位并立即释放新开仓额度，未脱险仓位与未成交个股买单各占 1 槽位；"
        f"同一交易日最多新开 1 笔个股；近 {strategy.consecutive_loss_lookback_days} 天连损 {strategy.consecutive_loss_limit} 笔"
        f"自动冷静 {strategy.consecutive_loss_cooldown_days} 个交易日，恢复首单按半仓试探（风险预算×{float(strategy.recovery_pilot_risk_scale):.1f}），首单盈利推保本后恢复全额火力。"
        f"{audit_rule}"
    )


def _build_compact_pick_critic_profile(strategy: StrategySection) -> str:
    """v5.5.7: /pick Critic-only condensed profile — drops /report-only sell/audit and crumb rules."""
    benchmark = getattr(strategy, "benchmark_symbol", "VWRP.L")
    fx = float(strategy.fx_fee_pct)
    stamp = float(strategy.uk_stamp_duty_pct)
    min_invest = float(getattr(strategy, "min_invest_order_gbp", 700.0))
    min_pilot = float(getattr(strategy, "min_invest_pilot_order_gbp", 500.0))
    min_isa_eq = float(getattr(strategy, "min_isa_equity_order_gbp", 1200.0))
    min_isa_etf = float(getattr(strategy, "min_isa_etf_order_gbp", 400.0))
    cadence_mode = str(getattr(strategy, "trade_cadence_guard_mode", "advisory") or "advisory").strip().lower()
    return (
        "【投资者画像与硬性规则（/pick 风控终审精简版）】：\n"
        "• 英国个人投资者，Trading 212 Stocks ISA + Invest 双账户，不用杠杆/期权/做空。\n"
        f"• 目标：双账户合计净值长期跑赢全球宽基 {benchmark}；现金利息只是等待期保底，不是投资终点。\n"
        "• ISA 长线（巴菲特价值）：核心宽基 + 优质护城河个股；开仓只审「质量门 + 安全边际（常规≥12% / 危机≥20%）」，"
        "不以短期波动或浮盈卖飞；仅基本面/护城河证伪、极端估值泡沫、更优机会成本置换、集中度超限才卖出。\n"
        "• Invest 短线（右侧动量）：严守当日 Python 动态 R:R 门槛与止损 ATR 噪声地板，按以损定仓、绝不摊平；"
        "非 RISK_OFF 确认强领涨股（MA200上方+BULL/PULLBACK+周线不空+RS20>0且RS60>0）解锁双锚目标上限，非领涨/弱势/RISK_OFF 仍死锁卖方均值 ±15%；"
        f"单笔 ≥£{min_invest:,.0f}（试探缩放期地板 £{min_pilot:,.0f}），ISA 个股 ≥£{min_isa_eq:,.0f}、ISA ETF ≥£{min_isa_etf:,.0f}，严禁碎单。\n"
        f"• 摩擦口径：换汇 {fx:.2f}%/边、英股印花税 {stamp:.1f}%、买卖价差按保守估算；"
        "理由须写明资金路径；应税 Invest 低赔率频繁换仓会侵蚀净收益。\n"
        f"• 频率与冷却（模式 {cadence_mode}）：由 Python 统计近 7/30 天笔数与清仓冷却，禁止重推已被拦截的标的。\n"
        "• ISA 核心宽基只按 Python 政策票据执行，严禁自行换代码；Invest 槽位/热力/单日额度触顶即须等待或置换。\n"
        "• Invest 派息宽基蓄水池（DIST 白名单）零槽位零热力零Quota：溢出现金停泊与“清宽基换股”配对指令均以 Python 输出为准，"
        "严禁自行选码/口算股数/买入累积型 ETF。\n"
        "• 零 Quota：卖出存量仓位与买入核心宽基 ETF 永不消耗周/月开仓配额，也不占在险槽位。"
    )


def _build_compact_us_sentinel_profile(strategy: StrategySection) -> str:
    """v5.5.8: /us Delta-Sentinel condensed profile (~450 tokens vs ~1,850 full profile).

    Keeps only what a pre-open incremental verification needs: dual-account identity,
    the ISA-value vs Invest-momentum philosophies, wallet/pending-order interest
    semantics, and the "morning report is the sole baseline" discipline. Drops the
    ISA four-bucket policy ticket, cash-deployment proposals, order sizing quotas,
    loss-streak recovery rules, and the anti-pandering order re-audit paragraph
    (the latter would invite /us to re-adjudicate orders /report already settled).
    """
    gbp_rate = float(strategy.cash_interest_rate_pct)
    usd_rate = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    fx = float(strategy.fx_fee_pct)
    return (
        "【投资者画像与硬性规则（美股增量哨兵精简版）】：\n"
        "• 英国个人投资者，持有 Trading 212 Stocks ISA 与 Invest 双账户，不使用杠杆/期权/做空。\n"
        "• ISA（长线稳健仓，巴菲特价值流派）：核心宽基 + 优质护城河个股；"
        "不把盘前短期波动当风险，严禁仅因跳空、RSI 超买或短期回撤卖飞好公司；"
        "ISA 卖出/减仓仅限 4 情形：基本面/护城河证伪、极端估值泡沫、显著更优机会成本置换、集中度超限/边角料归核。\n"
        "• Invest（短线机动仓，右侧动量战术流派）：严守硬止损与当日 Regime 动态 R:R 门槛，按以损定仓、绝不摊平；"
        "趋势未破坏时上移移动止损而非恐慌卖出，破位则不犹豫。\n"
        f"• 双钱包与挂单计息：ISA 仅 GBP 钱包（{gbp_rate:.1f}%）；Invest 为 GBP（{gbp_rate:.1f}%）+ USD（{usd_rate:.1f}%）双钱包，"
        f"跨币种买入计单边 {fx:.2f}% 换汇费；买入限价挂单占用资金在未成交前照常计息，"
        "严禁以「释放挂单资金吃息」为由建议撤单。\n"
        "• 唯一职责边界：以今日 /report 全文为绝对基线做增量核验与修正，"
        "严禁另起炉灶、新增开仓/配置提案，或重审早报已审计过的挂单。\n"
    )


def _build_compact_chat_profile(strategy: StrategySection, *, include_order_audit_rule: bool = True) -> str:
    """v5.5.8: daily-chat condensed profile — drops /report /pick-only rules, keeps Q&A discipline.

    Each previously duplicated clause (cash interest, dynamic R:R, zero quota, never
    push buys) is stated exactly once; the loss-streak recovery and four-bucket ticket
    wording lives in /report & /pick prompts only.
    """
    benchmark = getattr(strategy, "benchmark_symbol", "VWRP.L")
    gbp_rate = float(strategy.cash_interest_rate_pct)
    usd_rate = float(getattr(strategy, "usd_cash_interest_rate_pct", 3.4))
    fx = float(strategy.fx_fee_pct)
    stamp = float(strategy.uk_stamp_duty_pct)
    min_invest = float(getattr(strategy, "min_invest_order_gbp", 700.0))
    min_pilot = float(getattr(strategy, "min_invest_pilot_order_gbp", 500.0))
    min_isa_eq = float(getattr(strategy, "min_isa_equity_order_gbp", 1200.0))
    min_isa_etf = float(getattr(strategy, "min_isa_etf_order_gbp", 400.0))
    risk_budget = float(getattr(strategy, "invest_risk_budget_gbp", 85.0))
    risk_pct = float(getattr(strategy, "invest_risk_budget_pct", 0.75) or 0.75)
    audit_rule = (
        "• 反迎合审计：对任何已有挂单或前序 AI 结论，一律按挂单限价核算成本、目标空间与盈亏比；"
        "距现价 -3%~-12% 且基本面未坏的回踩埋伏单属于有效等待，严禁以“现价空间不足”或“释放现金吃息”为由建议撤单；"
        "只有当按限价成交后扣摩擦的盈亏比仍 <2:1、或基本面/趋势已证伪时，才给出 CANCEL 或带具体数字的 MODIFY。\n"
        if include_order_audit_rule
        else ""
    )
    return (
        "【投资者画像与硬性规则（日常聊天精简版）】：\n"
        "• 英国个人投资者，持有 Trading 212 Stocks ISA 与 Invest 双账户，不使用杠杆/期权/做空。\n"
        f"• 核心宗旨：双账户合计净值长期跑赢全球宽基基准 {benchmark}；"
        f"{gbp_rate:.1f}%/{usd_rate:.1f}% 现金利息只是等待期保底收益，不是投资终点，严禁把“守住现金吃息”当默认结论。\n"
        f"• ISA（长线稳健仓，巴菲特价值流派，核心-卫星结构）：核心宽基目标占比 ≥{strategy.isa_core_etf_target_pct:.0f}%"
        f"（按全球/美股/欧洲/英国四桶宽基政策配置，以 Python 政策票据为准），"
        f"个股卫星硬上限 ≤{strategy.isa_satellite_max_total_pct:.0f}% 且 ≤{strategy.isa_satellite_max_count} 只（含未成交个股买单），"
        "一旦超限即锁死 ISA 新开个股通道，只准审撤个股买单并把闲钱分批归拢核心宽基；"
        "优质股下跌若基本面与护城河未证伪、且估值安全边际扩大，反而是分批左侧加仓机会，严禁仅因浮盈、RSI 超买或短期回撤卖飞好公司；"
        "ISA 卖出/减仓仅限 4 情形：基本面/护城河证伪、极端估值泡沫、显著更优机会成本置换、集中度超限/边角料归核。\n"
        "• Invest（短线机动仓，右侧动量战术流派）：Invest 现金再多也绝不催买，只有右侧突破猎物才触发开仓；"
        "严守当日四态市场状态（RISK_ON / NEUTRAL / RECOVERY 早春试探 / RISK_OFF）动态适配 R:R（RISK_ON 右侧领涨 ≥1.6:1 / NEUTRAL ≥2.0:1 / 弱势 ≥2.5:1，以当日注入门槛为准）"
        "与止损 ATR 噪声地板，按以损定仓、绝不摊平；"
        "已挂保本止损（止损价≥成本+摩擦）的仓位占用 0 槽位并立即释放新开仓额度，未脱险仓位与未成交个股买单各占 1 槽位；"
        "持仓中的派息宽基蓄水池（VWRL.L/ISF.L/VERX.L/VEUR.L）是 Beta 停车场——旁路短线止损与移动止盈、零槽位零热力零Quota，"
        "仅退潮期复查防御处置，勿按个股短线纪律点评它。\n"
        f"• 开仓硬纪律：单笔金额 Invest ≥ £{min_invest:,.0f}（试探缩放期地板 £{min_pilot:,.0f}）、ISA 个股 ≥ £{min_isa_eq:,.0f}、ISA ETF ≥ £{min_isa_etf:,.0f}，严禁碎单；"
        f"Invest 单笔最大可承受亏损 = max(£{risk_budget:,.0f} 地板, Invest 净值×{risk_pct:.2f}%)（仓位 ≈ 该风险预算 ÷ 止损百分比）。\n"
        f"• 钱包与摩擦：ISA 仅 GBP 钱包（{gbp_rate:.1f}%）；Invest 为 GBP（{gbp_rate:.1f}%）+ USD（{usd_rate:.1f}%）双钱包，"
        f"复用同币种钱包 0% 换汇，跨币种买入计单边 {fx:.2f}% 换汇费（英股买入另收 {stamp:.1f}% 印花税），卖出所得默认留存原币种钱包；"
        "买入限价挂单所占用的资金在未成交前照常产生同等日息，严禁以“释放挂单资金吃息”为由阻止低位限价挂单；"
        "若按风险算出的美股金额大于 USD 钱包可用余额，正常由 GBP 换汇补足，严禁为省换汇费人为缩仓。\n"
        "• 零 Quota 豁免：卖出任何存量仓位/清理边角料，以及买入核心宽基 ETF，永远不消耗周/月开仓次数配额，也不占用在险槽位。\n"
        f"{audit_rule}"
        "• Invest 为应税账户，低赔率频繁换仓会侵蚀净收益；回答须写明资金路径与摩擦影响。"
    )


def build_agent_system_instruction(
    role_instruction: str,
    strategy: StrategySection,
    *,
    include_order_audit_rule: bool = True,
    compact_for_pick_critic: bool = False,
    compact_for_chat: bool = False,
    compact_for_us_sentinel: bool = False,
) -> str:
    """Combine stable role instructions and investor profile for Gemini `system_instruction`."""
    return (
        f"{role_instruction.strip()}\n\n"
        f"{build_investor_profile(strategy, include_order_audit_rule=include_order_audit_rule, compact_for_pick_critic=compact_for_pick_critic, compact_for_chat=compact_for_chat, compact_for_us_sentinel=compact_for_us_sentinel)}"
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
