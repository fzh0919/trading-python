"""v5.5.1 dual-philosophy regression matrix (ISA Buffett value × Invest right-side momentum).

Fully offline: every provider call is mocked or avoided; no real network request is made.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.agents.schemas import (
    CoreDeploymentItem,
    MorningReportDecision,
    PendingOrderAuditItem,
    PickActionItem,
    PickDecision,
    ReportActionItem,
    render_morning_report,
)
from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import (
    MacroSnapshot,
    TickerMarketBundle,
    _compute_safe_fcf_yield_pct,
    apply_basic_financials,
)
from src.data_providers.t212_client import AccountSnapshot, PendingOrderItem, PortfolioData, PositionItem
from src.engine.portfolio_rules import (
    _resolve_pick_rr,
    apply_risk_override_guardrail,
    compute_invest_forward_rr,
    evaluate_holding,
    evaluate_portfolio,
    is_strong_right_side_leader,
    resolve_invest_dynamic_rr_floor,
    resolve_invest_target_anchors,
)


def _macro(vix: float = 15.0) -> MacroSnapshot:
    return MacroSnapshot(
        fetched_at_utc="2026-11-19T07:25:00Z",
        gbp_usd=1.25,
        us_10y_yield=4.2,
        vix=vix,
        hy_oas_spread_pct=3.0,
    )


def _pos(
    account: str,
    ticker: str,
    *,
    value: float = 2400.0,
    qty: float = 24.0,
    avg: float = 100.0,
    cur: float = 100.0,
    return_pct: float = 0.0,
    weight_pct: float = 10.0,
    ccy: str = "USD",
) -> PositionItem:
    return PositionItem(
        account=account,
        raw_ticker=f"{ticker}_{ccy}_EQ",
        ticker=ticker,
        yahoo_symbol=ticker,
        name=f"{ticker} Inc.",
        inst_currency=ccy,
        quantity=qty,
        avg_price=avg,
        cur_price=cur,
        current_value=value,
        total_cost=value * (1.0 - return_pct / 100.0),
        pnl=value * return_pct / 100.0,
        fx_impact=0.0,
        return_pct=return_pct,
        weight_pct=weight_pct,
    )


def _portfolio(positions: list[PositionItem], total: float = 24000.0) -> PortfolioData:
    isa = [p for p in positions if p.account.upper() == "ISA"]
    inv = [p for p in positions if p.account.upper() == "INVEST"]
    return PortfolioData(
        fetched_at_uk="2026-11-19 07:30",
        fetched_at_ms=1763537400000,
        isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=total, positions=isa),
        invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=total, positions=inv),
        total_combined_value=total * 2,
    )


def _pick(
    account: str,
    symbol: str = "MSFT",
    *,
    action: str = "BUY",
    entry: float = 100.0,
    stop: float = 0.0,
    target: float = 0.0,
) -> PickDecision:
    return PickDecision(
        headline="测试",
        selected_actions=[
            PickActionItem(
                account=account,
                symbol=symbol,
                action=action,  # type: ignore[arg-type]
                thesis="测试逻辑",
                entry_zone="100 附近",
                stop_condition="基本面/护城河证伪" if stop <= 0 else "跌破硬止损",
                target_window="3个月",
                entry_price=entry,
                stop_price=stop,
                target_price=target,
            )
        ],
        consistency_and_cash_note="测试",
    )


def _quality_bundle(symbol: str = "MSFT", **overrides) -> TickerMarketBundle:
    data = dict(
        symbol=symbol,
        quote_type="EQUITY",
        currency="USD",
        current_price=100.0,
        forward_pe=16.0,
        roe_pct=22.0,
        profit_margin_pct=18.0,
        revenue_growth_pct=9.0,
        debt_to_equity=40.0,
        current_ratio=1.6,
        fcf_yield_pct=5.0,
        ev_ebitda=9.0,
        analyst_count=8,
        target_mean_price=117.0,
    )
    data.update(overrides)
    return TickerMarketBundle(**data)


class TestV552MarketDataPhase1(unittest.TestCase):
    """1. FCF yield / target range extraction with currency-scale guard + Finnhub fallback."""

    def test_market_data_fcf_yield_and_target_range_with_currency_guard(self) -> None:
        # Normal ratio -> percentage with 2dp
        self.assertEqual(_compute_safe_fcf_yield_pct(5_000_000.0, 100_000_000.0), 5.0)
        # Cross-currency ADR garbage (150x market cap) must be rejected, not stored
        self.assertIsNone(_compute_safe_fcf_yield_pct(15_000_000_000.0, 100_000_000.0))
        self.assertIsNone(_compute_safe_fcf_yield_pct(-9_000_000_000.0, 100_000_000.0))
        # Missing inputs
        self.assertIsNone(_compute_safe_fcf_yield_pct(None, 100.0))
        self.assertIsNone(_compute_safe_fcf_yield_pct(10.0, 0.0))

        bundle = TickerMarketBundle(
            symbol="MSFT",
            quote_type="EQUITY",
            currency="USD",
            current_price=100.0,
            target_high_price=135.0,
            target_low_price=88.0,
            peg_ratio=1.4,
            fcf_yield_pct=4.2,
        )
        self.assertEqual(bundle.target_high_price, 135.0)
        self.assertEqual(bundle.target_low_price, 88.0)
        self.assertEqual(bundle.peg_ratio, 1.4)

        # Finnhub P/FCF = 20 -> 5% yield, only fills a missing value
        empty_bundle = TickerMarketBundle(symbol="KO", quote_type="EQUITY", current_price=60.0)
        changed = apply_basic_financials(empty_bundle, {"pfcfShareTTM": 20.0})
        self.assertTrue(changed)
        self.assertEqual(empty_bundle.fcf_yield_pct, 5.0)
        kept = TickerMarketBundle(symbol="KO", quote_type="EQUITY", fcf_yield_pct=7.5)
        apply_basic_financials(kept, {"pfcfShareTTM": 20.0})
        self.assertEqual(kept.fcf_yield_pct, 7.5)

        # Old cached JSON (pre-v5.5.1, no new keys) still deserializes with None defaults
        legacy = TickerMarketBundle.model_validate(
            {
                "symbol": "AAPL",
                "quote_type": "EQUITY",
                "currency": "USD",
                "current_price": 190.0,
                "trailing_pe": 28.0,
            }
        )
        self.assertIsNone(legacy.fcf_yield_pct)
        self.assertIsNone(legacy.target_high_price)
        self.assertIsNone(legacy.target_low_price)
        self.assertIsNone(legacy.peg_ratio)


class TestV552IsaValueGate(unittest.TestCase):
    """2 & 3. ISA quality gate + margin-of-safety gate (stop_price=0 is legal for value names)."""

    def test_isa_quality_gate_rejects_loss_making_and_high_leverage_exempts_financials(self) -> None:
        payload = dict(
            macro=_macro(),
            strategy=StrategySection(pick_rr_guard_mode="hard"),
            trade_behavior={"partial": True},
        )
        losers = _quality_bundle(
            "LOSS", profit_margin_pct=-4.0, roe_pct=-8.0, analyst_count=0, target_mean_price=None
        )
        res = apply_risk_override_guardrail(
            _pick("ISA", "LOSS", stop=0.0, target=130.0),
            bundle_map={"LOSS": losers},
            **payload,
        )
        self.assertTrue(res.applied)
        self.assertEqual(res.overrides[0].trigger_type, "ISA_QUALITY_FAIL")

        shrinking = _quality_bundle(
            "SHRINK", revenue_growth_pct=-18.0, analyst_count=0, target_mean_price=None
        )
        res2 = apply_risk_override_guardrail(
            _pick("ISA", "SHRINK", stop=0.0, target=130.0),
            bundle_map={"SHRINK": shrinking},
            **payload,
        )
        self.assertTrue(res2.applied)
        self.assertEqual(res2.overrides[0].trigger_type, "ISA_QUALITY_FAIL")

        over_levered = _quality_bundle(
            "LEV",
            debt_to_equity=260.0,
            current_ratio=0.7,
            analyst_count=0,
            target_mean_price=None,
        )
        res3 = apply_risk_override_guardrail(
            _pick("ISA", "LEV", stop=0.0, target=130.0),
            bundle_map={"LEV": over_levered},
            **payload,
        )
        self.assertTrue(res3.applied)
        self.assertEqual(res3.overrides[0].trigger_type, "ISA_QUALITY_FAIL")
        self.assertIn("高杠杆", res3.overrides[0].reason)

        # Banks legitimately run D/E > 180 with a low current ratio -> leverage exemption
        bank = _quality_bundle(
            "BANK",
            sector="Financial Services",
            industry="Banks",
            debt_to_equity=520.0,
            current_ratio=0.55,
            analyst_count=0,
            target_mean_price=None,
        )
        bank_res = apply_risk_override_guardrail(
            _pick("ISA", "BANK", stop=0.0, target=130.0),
            bundle_map={"BANK": bank},
            **payload,
        )
        self.assertFalse(bank_res.applied)
        self.assertEqual(len(bank_res.decision.selected_actions), 1)

    def test_isa_mos_gate_allows_value_without_stop_price_and_blocks_bubble(self) -> None:
        payload = dict(
            macro=_macro(),
            strategy=StrategySection(pick_rr_guard_mode="hard"),
            trade_behavior={"partial": True},
        )
        value_bundle = _quality_bundle(
            "VALUE", analyst_count=0, target_mean_price=None, forward_pe=14.0
        )
        portfolio = _portfolio([])
        res = apply_risk_override_guardrail(
            _pick("ISA", "VALUE", stop=0.0, target=117.0),  # +17% - 0.35% friction
            portfolio=portfolio,
            bundle_map={"VALUE": value_bundle},
            **payload,
        )
        self.assertFalse(res.applied)
        kept = res.decision.selected_actions[0]
        self.assertEqual(kept.symbol, "VALUE")
        self.assertIn("ISA价值分批建议", kept.risk_note)
        self.assertIn("安全边际", kept.risk_note)

        bubble = _quality_bundle(
            "BUBBLE",
            forward_pe=52.0,
            peg_ratio=3.2,
            revenue_growth_pct=24.0,
            analyst_count=0,
            target_mean_price=None,
        )
        bubble_res = apply_risk_override_guardrail(
            _pick("ISA", "BUBBLE", stop=0.0, target=140.0),
            bundle_map={"BUBBLE": bubble},
            **payload,
        )
        self.assertTrue(bubble_res.applied)
        self.assertEqual(bubble_res.overrides[0].trigger_type, "ISA_MOS_INSUFFICIENT")
        self.assertIn("泡沫", bubble_res.overrides[0].reason)


class TestV552CrisisLane(unittest.TestCase):
    """4. Crisis: ISA deep-margin value lane opens, Invest stays 100% frozen."""

    def test_crisis_vix_allows_deep_mos_isa_value_but_freezes_invest_and_shallow_isa(self) -> None:
        payload = dict(
            macro=_macro(vix=32.0),
            strategy=StrategySection(pick_rr_guard_mode="hard"),
            deployment_gate={"regime": "RISK_OFF"},
            trade_behavior={"partial": True},
        )
        deep = _quality_bundle(
            "DEEP", analyst_count=0, target_mean_price=None, forward_pe=15.0, current_price=100.0
        )
        deep_res = apply_risk_override_guardrail(
            _pick("ISA", "DEEP", stop=0.0, target=130.0),  # capped +25% -> deep >=20%
            bundle_map={"DEEP": deep},
            **payload,
        )
        self.assertFalse(deep_res.applied)
        self.assertEqual(len(deep_res.decision.selected_actions), 1)

        shallow = _quality_bundle(
            "SHAL", analyst_count=0, target_mean_price=None, forward_pe=15.0, current_price=100.0
        )
        shallow_res = apply_risk_override_guardrail(
            _pick("ISA", "SHAL", stop=0.0, target=115.0),  # only ~14.6% discount
            bundle_map={"SHAL": shallow},
            **payload,
        )
        self.assertTrue(shallow_res.applied)
        self.assertIn(
            shallow_res.overrides[0].trigger_type,
            {"ISA_MOS_INSUFFICIENT", "MACRO_VIX"},
        )

        weak = _quality_bundle(
            "WEAK", profit_margin_pct=-6.0, roe_pct=-9.0, analyst_count=0, target_mean_price=None
        )
        weak_res = apply_risk_override_guardrail(
            _pick("ISA", "WEAK", stop=0.0, target=130.0),
            bundle_map={"WEAK": weak},
            **payload,
        )
        self.assertTrue(weak_res.applied)
        self.assertEqual(weak_res.overrides[0].trigger_type, "ISA_QUALITY_FAIL")

        # Invest: a textbook strong R:R setup is STILL frozen at VIX 32 / RISK_OFF
        invest_bundle = TickerMarketBundle(
            symbol="NVDA", quote_type="EQUITY", currency="USD", current_price=100.0
        )
        invest_res = apply_risk_override_guardrail(
            _pick("Invest", "NVDA", stop=93.0, target=125.0),
            bundle_map={"NVDA": invest_bundle},
            **payload,
        )
        self.assertTrue(invest_res.applied)
        self.assertEqual(invest_res.overrides[0].trigger_type, "MACRO_VIX")


class TestV552IsaHoldingBehaviour(unittest.TestCase):
    """5 & 6. ISA left-side value add + no short-term volatility shake-out."""

    def test_isa_left_side_value_add_four_locks(self) -> None:
        tech = TechnicalSnapshot(
            symbol="MOAT",
            last_close=85.0,
            dist_ma200_pct=-8.0,
            dist_ma50_pct=-3.0,
            trend_regime="PULLBACK",
            drawdown_52w_pct=-15.0,
            rsi14=41.0,
        )
        bundle = _quality_bundle("MOAT", current_price=85.0, target_mean_price=112.0)
        pos = _pos("ISA", "MOAT", cur=85.0, avg=100.0, return_pct=-15.0, weight_pct=10.0)
        ev = evaluate_holding(pos, tech, bundle, StrategySection(), total_issuer_weight_pct=10.0)
        self.assertEqual(ev.action_bias, "ADD_CANDIDATE")
        joined = " ".join(ev.reasons)
        self.assertIn("ISA价值左侧加仓候补", joined)
        self.assertIn("须联网确认", joined)
        self.assertEqual(ev.philosophy_tag, "ISA_VALUE")
        self.assertIsNotNone(ev.value_mos_pct)

        # Lock 2 broken: quality gate fails -> never add, and it flags a quality review
        bad_bundle = bundle.model_copy(update={"profit_margin_pct": -3.0, "roe_pct": -5.0})
        ev_bad = evaluate_holding(pos, tech, bad_bundle, StrategySection(), total_issuer_weight_pct=10.0)
        self.assertNotEqual(ev_bad.action_bias, "ADD_CANDIDATE")
        self.assertIn("质量门预警", " ".join(ev_bad.reasons))

        # Lock 1 broken: overweight -> never add
        ev_heavy = evaluate_holding(
            pos.model_copy(update={"weight_pct": 32.0}), tech, bundle, StrategySection(),
            total_issuer_weight_pct=10.0,
        )
        self.assertNotEqual(ev_heavy.action_bias, "ADD_CANDIDATE")

    def test_isa_not_shaken_out_by_short_term_volatility_but_flags_bubble_and_overweight(self) -> None:
        tech = TechnicalSnapshot(
            symbol="KO",
            last_close=130.0,
            dist_ma200_pct=14.0,
            dist_ma50_pct=6.0,
            drawdown_60d_pct=-13.0,
            rsi14=78.0,
            trend_regime="BULL",
        )
        healthy = _quality_bundle("KO", current_price=130.0, forward_pe=18.0, peg_ratio=1.2)
        pos = _pos("ISA", "KO", cur=130.0, avg=100.0, return_pct=30.0, weight_pct=10.0)
        ev = evaluate_holding(pos, tech, healthy, StrategySection())
        self.assertNotIn("移动止盈", " ".join(ev.reasons))
        self.assertNotIn("超买区", " ".join(ev.reasons))
        self.assertEqual(ev.action_bias, "HOLD")

        bubble = healthy.model_copy(
            update={"forward_pe": 52.0, "peg_ratio": 3.4, "revenue_growth_pct": 22.0}
        )
        ev_bubble = evaluate_holding(pos, tech, bubble, StrategySection())
        self.assertEqual(ev_bubble.action_bias, "TRIM_ALERT")
        self.assertIn("泡沫", " ".join(ev_bubble.reasons))

        ev_heavy = evaluate_holding(
            pos.model_copy(update={"weight_pct": 33.0}), tech, healthy, StrategySection()
        )
        self.assertEqual(ev_heavy.action_bias, "TRIM_ALERT")


class TestV552InvestDynamicRR(unittest.TestCase):
    """7, 8, 9. Regime × structure R:R floor, ATR noise floor, RISK_ON target ceiling."""

    @staticmethod
    def _leader(**overrides) -> TechnicalSnapshot:
        data = dict(
            symbol="NVDA",
            last_close=100.0,
            dist_ma200_pct=6.0,
            dist_ma50_pct=1.0,
            dist_ma20_pct=1.5,
            trend_regime="BULL",
            weekly_bullish=True,
            atr_pct=4.0,  # 4x ATR target ceiling = +16%; noise floor = 6%
            rs_20d_pct=4.5,
            rs_60d_pct=9.0,
        )
        data.update(overrides)
        return TechnicalSnapshot(**data)

    def test_invest_dynamic_rr_floor_by_regime_and_structure(self) -> None:
        bundle = TickerMarketBundle(
            symbol="NVDA", quote_type="EQUITY", currency="USD", current_price=100.0
        )
        base = dict(
            macro=_macro(),
            strategy=StrategySection(pick_rr_guard_mode="hard"),
            bundle_map={"NVDA": bundle},
            tech_map={"NVDA": self._leader()},
            trade_behavior={"partial": True},
        )
        # entry 100 / stop 94 (6%, >= ATR floor) / target 112 -> 11.65 / 6 = 1.94:1
        risk_on = apply_risk_override_guardrail(
            _pick("Invest", "NVDA", stop=94.0, target=112.0),
            deployment_gate={"regime": "RISK_ON"},
            **base,
        )
        self.assertFalse(risk_on.applied)

        neutral = apply_risk_override_guardrail(
            _pick("Invest", "NVDA", stop=94.0, target=112.0),
            deployment_gate={"regime": "NEUTRAL"},
            **base,
        )
        self.assertTrue(neutral.applied)
        self.assertEqual(neutral.overrides[0].trigger_type, "RR_BELOW_FLOOR")

        # Weak structure (below MA200) needs >= 2.5:1 -> 2.28:1 is rejected
        weak_base = dict(base)
        weak_base["tech_map"] = {"NVDA": self._leader(dist_ma200_pct=-6.0, trend_regime="WEAK")}
        weak = apply_risk_override_guardrail(
            _pick("Invest", "NVDA", stop=94.0, target=114.0),
            deployment_gate={"regime": "NEUTRAL"},
            **weak_base,
        )
        self.assertTrue(weak.applied)
        self.assertEqual(weak.overrides[0].trigger_type, "RR_BELOW_FLOOR")
        self.assertIn("弱势", weak.overrides[0].reason)

    def test_invest_stop_too_tight_atr_noise_floor_blocks_rr_gaming(self) -> None:
        bundle = TickerMarketBundle(
            symbol="TSLA", quote_type="EQUITY", currency="USD", current_price=100.0
        )
        # ATR 3.5% -> floor 5.25%; a -2.5% stop would fake a 3:1 R:R
        res = apply_risk_override_guardrail(
            _pick("Invest", "TSLA", stop=97.5, target=108.0),
            macro=_macro(),
            strategy=StrategySection(pick_rr_guard_mode="hard"),
            bundle_map={"TSLA": bundle},
            tech_map={"TSLA": self._leader(symbol="TSLA", atr_pct=3.5)},
            deployment_gate={"regime": "NEUTRAL"},
            trade_behavior={"partial": True},
        )
        self.assertTrue(res.applied)
        self.assertEqual(res.overrides[0].trigger_type, "STOP_TOO_TIGHT")
        self.assertIn("ATR噪声地板", res.overrides[0].reason)

    def test_invest_risk_on_leader_target_ceiling_unblocks_lagging_analyst_mean(self) -> None:
        bundle = TickerMarketBundle(
            symbol="NVDA",
            quote_type="EQUITY",
            currency="USD",
            current_price=100.0,
            target_mean_price=105.0,
            target_high_price=135.0,
            analyst_count=12,
        )
        leader = self._leader()
        confident = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=94.0, target=120.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=leader,
            regime="RISK_ON",
        )
        # Not dragged back to the stale 105 analyst mean
        self.assertGreater(confident["target"], 110.0)
        self.assertLessEqual(confident["target"], 125.0)

        hallucinated = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=94.0, target=160.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=leader,
            regime="RISK_ON",
        )
        # Capped at the sane upper ceiling (analyst mean +15% / high target * haircut)
        self.assertLessEqual(hallucinated["target"], 122.0)
        self.assertGreaterEqual(hallucinated["target"], 120.0)

        # v5.5.9 M2: confirmed RS leader also unlocks dual-anchor ceiling in NEUTRAL & RECOVERY
        for non_crisis_regime in ("NEUTRAL", "RECOVERY"):
            unlocked = _resolve_pick_rr(
                _pick("Invest", "NVDA", stop=94.0, target=120.0).selected_actions[0],
                bundle,
                StrategySection(),
                symbol="NVDA",
                account="Invest",
                tech=leader,
                regime=non_crisis_regime,
            )
            self.assertAlmostEqual(unlocked["target"], 120.0, places=2)
            self.assertFalse(unlocked["target_replaced"])

        # RISK_OFF and weak/non-RS names remain deadlocked to the legacy ±15% analyst mean
        risk_off = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=94.0, target=130.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=leader,
            regime="RISK_OFF",
        )
        self.assertAlmostEqual(risk_off["target"], 105.0, places=2)

        laggard = self._leader(rs_20d_pct=-1.5, rs_60d_pct=3.0)
        neutral_laggard = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=94.0, target=130.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=laggard,
            regime="NEUTRAL",
        )
        self.assertAlmostEqual(neutral_laggard["target"], 105.0, places=2)


class TestV552InvestHoldingBehaviour(unittest.TestCase):
    """10 & 11. Bull run lets winners run; neutral tightens trailing; forward R:R decay."""

    def test_invest_bull_holds_above_ma20_and_neutral_tightens_trailing_stop(self) -> None:
        bull_tech = TechnicalSnapshot(
            symbol="NVDA",
            last_close=126.0,
            dist_ma20_pct=2.5,
            dist_ma50_pct=8.0,
            dist_ma200_pct=25.0,
            drawdown_60d_pct=-2.0,
            rsi14=77.0,
            trend_regime="BULL",
            weekly_bullish=True,
            atr_pct=2.5,
        )
        pos = _pos("Invest", "NVDA", cur=126.0, avg=100.0, return_pct=26.0, weight_pct=10.0)
        ev_bull = evaluate_holding(pos, bull_tech, None, StrategySection(), regime="RISK_ON")
        self.assertNotEqual(ev_bull.action_bias, "TRIM_ALERT")
        self.assertIn("RISK_ON", " ".join(ev_bull.reasons))
        self.assertEqual(ev_bull.philosophy_tag, "INVEST_RIGHT_SIDE")

        # NEUTRAL tightens the trailing drawdown from -12% to -8.5%
        neutral_tech = bull_tech.model_copy(update={"drawdown_60d_pct": -9.0, "dist_ma20_pct": -1.0})
        ev_neutral = evaluate_holding(pos, neutral_tech, None, StrategySection(), regime="NEUTRAL")
        self.assertEqual(ev_neutral.action_bias, "TRIM_ALERT")
        self.assertIn("移动止盈", " ".join(ev_neutral.reasons))

    def test_invest_forward_rr_deterioration_warning(self) -> None:
        tech = TechnicalSnapshot(
            symbol="AMD",
            last_close=100.0,
            dist_ma20_pct=-1.5,
            dist_ma50_pct=-3.0,
            dist_ma200_pct=4.0,
            ret_20d_pct=1.0,
            trend_regime="PULLBACK",
            weekly_bullish=True,
            atr_pct=3.0,  # adaptive stop -8.4%
        )
        bundle = TickerMarketBundle(
            symbol="AMD",
            quote_type="EQUITY",
            currency="USD",
            current_price=100.0,
            target_mean_price=103.0,
            analyst_count=9,
        )
        pos = _pos("Invest", "AMD", cur=100.0, avg=99.0, return_pct=1.0, weight_pct=10.0)
        ev = evaluate_holding(pos, tech, bundle, StrategySection(), regime="NEUTRAL")
        joined = " ".join(ev.reasons)
        self.assertIn("前瞻R:R恶化", joined)
        self.assertIn("ROTATE", joined)
        self.assertIsNotNone(ev.forward_rr)
        self.assertLess(float(ev.forward_rr), 0.8)


class TestV552ScreenerValueTrack(unittest.TestCase):
    """12. ISA screen rewards discounted quality over expensive momentum; loss-makers sink."""

    def test_screener_isa_rewards_discounted_quality_over_expensive_momentum(self) -> None:
        from src.engine.screener import score_candidate

        strategy = StrategySection()
        history = pd.DataFrame(
            {
                "Open": [100.0] * 80,
                "High": [101.0] * 80,
                "Low": [99.0] * 80,
                "Close": [100.0] * 80,
                "Volume": [1_000_000.0] * 80,
            }
        )
        quality_tech = TechnicalSnapshot(
            symbol="QUAL",
            last_close=80.0,
            dist_ma200_pct=-8.0,
            dist_ma50_pct=-6.0,
            trend_regime="WEAK",
            drawdown_52w_pct=-25.0,
            rsi14=38.0,
            weekly_bullish=False,
        )
        quality_bundle = _quality_bundle(
            "QUAL", current_price=80.0, forward_pe=11.0, fcf_yield_pct=6.5, roe_pct=24.0
        )
        momentum_tech = TechnicalSnapshot(
            symbol="HYPE",
            last_close=200.0,
            dist_ma200_pct=40.0,
            dist_ma50_pct=10.0,
            trend_regime="BULL",
            drawdown_52w_pct=-1.0,
            rsi14=79.0,
            weekly_bullish=True,
            atr_pct=2.0,
        )
        momentum_bundle = _quality_bundle(
            "HYPE",
            current_price=200.0,
            forward_pe=68.0,
            peg_ratio=4.0,
            revenue_growth_pct=12.0,
            fcf_yield_pct=0.5,
        )
        trap_tech = TechnicalSnapshot(
            symbol="TRAP",
            last_close=12.0,
            dist_ma200_pct=-45.0,
            trend_regime="WEAK",
            drawdown_52w_pct=-60.0,
            rsi14=40.0,
            weekly_bullish=False,
        )
        trap_bundle = _quality_bundle(
            "TRAP",
            current_price=12.0,
            profit_margin_pct=-12.0,
            roe_pct=-15.0,
            revenue_growth_pct=-22.0,
            forward_pe=9.0,
            fcf_yield_pct=-6.0,
        )

        def _score(sym: str, tech: TechnicalSnapshot, bundle: TickerMarketBundle):
            return score_candidate(
                sym,
                "ISA_LONG_TERM",
                tech,
                bundle,
                0.30,
                "NVDA",
                False,
                False,
                strategy,
                hist=history,
            )

        quality = _score("QUAL", quality_tech, quality_bundle)
        momentum = _score("HYPE", momentum_tech, momentum_bundle)
        trap = _score("TRAP", trap_tech, trap_bundle)

        self.assertGreater(quality.score, momentum.score)
        self.assertGreater(momentum.score, trap.score)
        self.assertIn("优质股安全边际回撤", " ".join(quality.pros))
        self.assertIn("盈利为负(不符ISA质量门)", " ".join(trap.cons))
        self.assertNotIn("[Phase4弱势]", " ".join(quality.cons))


class TestV552ReportGroupedRendering(unittest.TestCase):
    """13. Single-message dual-account layout + 2800-char ceiling + plain text."""

    @staticmethod
    def _decision() -> MorningReportDecision:
        return MorningReportDecision(
            headline="ISA 逢低加仓，Invest 让利润奔跑",
            top_actions=[
                ReportActionItem(
                    account="ISA", symbol="MSFT", action="ADD", price_zone="410 USD",
                    size_hint="1/3", reason="护城河稳固且安全边际扩大",
                ),
                ReportActionItem(
                    account="Invest", symbol="NVDA", action="TRIM", price_zone="126 USD",
                    size_hint="25%", reason="上移移动止损并锁部分利润",
                ),
            ],
            pending_order_audits=[
                PendingOrderAuditItem(
                    account="ISA", symbol="ULVR.L", verdict="CANCEL",
                    order_summary="BUY 4900 GBp", reason="目标空间不足且含印花税",
                ),
                PendingOrderAuditItem(
                    account="Invest", symbol="AMD", verdict="KEEP",
                    order_summary="BUY 118 USD", reason="回踩埋伏单有效等待",
                ),
            ],
            core_deployment=[
                CoreDeploymentItem(
                    account="ISA", symbol="ISF.L", price_zone="812-816 GBp",
                    size_hint="£1,600", reason="分散美股科技",
                )
            ],
            daily_vetoes=["Invest 半导体占比超限，冻结新开芯片仓"],
            macro_and_cash="10Y美债偏高，保留GBP现金吃4%年化利息。" + "宏观" * 150,
            account_diagnosis="ISA 科技集中，NVDA 仅在 ISA 持有。" + "诊断" * 150,
            flagged_positions=[],
            hold_summary="其余长线底仓持有不动",
            trade_review="近30日交易克制。" + "复盘" * 120,
        )

    def test_report_dual_account_grouped_rendering_respects_explicit_limit(self) -> None:
        gate_banner = "组合门控：ISA 可分批宽基 / Invest WAIT"
        text = render_morning_report(self._decision(), gate_banner=gate_banner, max_chars=2200)

        # Grouped deterministic layout
        self.assertIn("🏛️ ISA 价值账户", text)
        self.assertIn("⚡ Invest 右侧账户", text)
        self.assertIn("1. [ISA] ADD MSFT", text)
        self.assertIn("1. [Invest] TRIM NVDA", text)
        self.assertIn("【核心宽基部署】", text)
        self.assertIn("BUY ISF.L", text)
        self.assertIn("• [ISA] ULVR.L (BUY 4900 GBp) →【CANCEL】", text)
        self.assertIn("• [Invest] AMD (BUY 118 USD) →【KEEP】", text)

        # Three top-level headings preserved; gate banner pinned as sec2 first line
        self.assertTrue(text.startswith("【今日核心行动与挂单裁决】"))
        self.assertIn("【宏观、账户与今日风控红线】\n" + gate_banner, text)
        self.assertIn("【财报预警、复盘与重点异动】", text)

        # Explicit ``max_chars`` callers still win over the v5.5.2 default of 2800.
        self.assertLessEqual(len(text), 2200)
        self.assertNotIn("#", text)
        self.assertNotIn("**", text)

        # Grouping order: ISA block before Invest block
        self.assertLess(text.index("🏛️ ISA 价值账户"), text.index("⚡ Invest 右侧账户"))

    def test_report_empty_account_collapses_to_one_line_and_default_ceiling(self) -> None:
        decision = self._decision()
        decision.top_actions = [decision.top_actions[0]]  # ISA only
        decision.pending_order_audits = [decision.pending_order_audits[0]]
        text = render_morning_report(decision)
        self.assertIn("⚡ Invest 右侧账户：今日无右侧新动作，现金吃息待命", text)
        self.assertNotIn("1. [Invest]", text)

        invest_only = self._decision()
        invest_only.top_actions = [invest_only.top_actions[1]]
        invest_only.pending_order_audits = [invest_only.pending_order_audits[1]]
        invest_only.core_deployment = []
        text2 = render_morning_report(invest_only)
        self.assertIn("🏛️ ISA 价值账户：今日持仓观望，核心定投/安全边际待命", text2)

        # Default ceiling is 2800 (v5.5.2); a >2200-char report is preserved intact
        long_decision = self._decision()
        long_decision.macro_and_cash = "宏观" * 500
        long_decision.account_diagnosis = "诊断" * 500
        rendered = render_morning_report(long_decision)
        self.assertGreater(len(rendered), 2200)
        self.assertLessEqual(len(rendered), 2800)
        self.assertIn("其余长线底仓持有不动", rendered)


class TestV557VixDynamicRRFloor(unittest.TestCase):
    """v5.5.7 Wave 1: one VIX-inclusive Invest R:R floor shared by every layer."""

    def _leader(self) -> TechnicalSnapshot:
        return TechnicalSnapshot(
            symbol="NVDA",
            last_close=100.0,
            dist_ma20_pct=2.0,
            dist_ma50_pct=6.0,
            dist_ma200_pct=20.0,
            trend_regime="BULL",
            weekly_bullish=True,
            atr_pct=2.5,
        )

    def test_vix_tightens_neutral_to_defensive_floor(self) -> None:
        rr, label = resolve_invest_dynamic_rr_floor(
            "NEUTRAL", self._leader(), StrategySection(), vix=25.3
        )
        self.assertEqual(rr, 2.5)
        self.assertIn("VIX", label)

        calm, _ = resolve_invest_dynamic_rr_floor(
            "NEUTRAL", self._leader(), StrategySection(), vix=16.0
        )
        self.assertEqual(calm, 2.0)

        leader, _ = resolve_invest_dynamic_rr_floor(
            "RISK_ON", self._leader(), StrategySection(), vix=15.0
        )
        self.assertEqual(leader, 1.6)

        tightened, label2 = resolve_invest_dynamic_rr_floor(
            "RISK_ON", self._leader(), StrategySection(), vix=24.0
        )
        self.assertEqual(tightened, 2.5)
        self.assertIn("VIX", label2)

        # No vix passed => byte-identical legacy behaviour.
        legacy, _ = resolve_invest_dynamic_rr_floor("NEUTRAL", self._leader(), StrategySection())
        self.assertEqual(legacy, 2.0)

    def test_risk_override_applies_vix_floor_to_recomputed_rr(self) -> None:
        bundle = TickerMarketBundle(
            symbol="NVDA",
            quote_type="EQUITY",
            currency="USD",
            current_price=100.0,
        )
        tech = self._leader()
        decision = _pick("Invest", "NVDA", stop=95.0, target=111.0)
        hard = StrategySection(pick_rr_guard_mode="hard", invest_min_rr_neutral=2.0)
        # Upside 11% vs 5% risk => ~2.2:1, fine in calm NEUTRAL but below the 2.5 VIX floor.
        res = apply_risk_override_guardrail(
            decision,
            portfolio=_portfolio([]),
            macro=_macro(vix=25.3),
            strategy=hard,
            bundle_map={"NVDA": bundle},
            trade_behavior={"partial": True},
            deployment_gate={"regime": "NEUTRAL"},
            tech_map={"NVDA": tech},
        )
        self.assertTrue(res.applied)
        self.assertEqual(res.overrides[0].trigger_type, "RR_BELOW_FLOOR")
        self.assertIn("VIX", res.overrides[0].reason)


class TestV557Wave5UnifiedMath(unittest.TestCase):
    """v5.5.7 Wave 5: planned price clamp, friction-aware net R:R, nullable price schema."""

    def test_planned_entry_is_clamped_and_falls_back(self) -> None:
        bundle = TickerMarketBundle(
            symbol="NVDA", quote_type="EQUITY", currency="USD", current_price=100.0
        )
        base = _pick("Invest", "NVDA", stop=94.0, target=120.0).selected_actions[0]

        in_band = _resolve_pick_rr(
            base.model_copy(update={"planned_entry_price": 98.0}),
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
        )
        self.assertAlmostEqual(in_band["entry"], 98.0, places=2)
        self.assertTrue(in_band["entry_planned"])

        too_low = _resolve_pick_rr(
            base.model_copy(update={"planned_entry_price": 80.0}),
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
        )
        self.assertAlmostEqual(too_low["entry"], 100.0, places=2)
        self.assertFalse(too_low["entry_planned"])

        too_high = _resolve_pick_rr(
            base.model_copy(update={"planned_entry_price": 110.0}),
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
        )
        self.assertAlmostEqual(too_high["entry"], 100.0, places=2)

        below_stop = _resolve_pick_rr(
            base.model_copy(update={"planned_entry_price": 93.0}),
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
        )
        self.assertAlmostEqual(below_stop["entry"], 100.0, places=2)

    def test_net_rr_uses_friction_denominator_but_atr_floor_uses_pure_stop(self) -> None:
        bundle = TickerMarketBundle(
            symbol="NVDA", quote_type="EQUITY", currency="USD", current_price=100.0
        )
        tech = TechnicalSnapshot(
            symbol="NVDA",
            last_close=100.0,
            dist_ma200_pct=20.0,
            trend_regime="BULL",
            weekly_bullish=True,
            atr_pct=3.0,
        )
        info = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=97.0, target=120.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=tech,
            regime="NEUTRAL",
        )
        self.assertAlmostEqual(info["risk_pct"], 3.0, places=2)
        self.assertAlmostEqual(
            info["effective_risk_pct"], round(3.0 + info["friction_pct"], 2), places=2
        )
        self.assertGreater(info["effective_risk_pct"], info["risk_pct"])
        expected_rr = round(
            (info["upside_pct"] - info["friction_pct"]) / info["effective_risk_pct"], 2
        )
        self.assertAlmostEqual(info["rr"], expected_rr, places=2)
        # ATR floor = max(1.5*3.0, 4.0) = 4.5 — the 3% pure stop is still too tight.
        self.assertTrue(info["stop_too_tight"])
        self.assertAlmostEqual(info["min_stop_pct"], 4.5, places=2)

    def test_pick_schema_accepts_null_stop_target_and_new_optional_fields(self) -> None:
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA",
                    symbol="ULVR.L",
                    action="BUY",
                    thesis="护城河稳健",
                    entry_zone="3850-3900 GBp",
                    stop_condition="基本面证伪",
                    target_window="3年",
                    entry_price=3880.0,
                    stop_price=None,
                    target_price=None,
                    planned_entry_price=3860.0,
                    planned_amount_gbp=1500.0,
                )
            ],
            rejected_or_frozen=[],
            consistency_and_cash_note="继承早报红线",
        )
        self.assertIsNone(decision.selected_actions[0].stop_price)
        self.assertEqual(decision.selected_actions[0].planned_amount_gbp, 1500.0)
        info = _resolve_pick_rr(
            decision.selected_actions[0],
            TickerMarketBundle(symbol="ULVR.L", quote_type="EQUITY", currency="GBp", current_price=3880.0),
            StrategySection(),
            symbol="ULVR.L",
            account="ISA",
        )
        self.assertIsNone(info["risk_pct"])
        self.assertIsNone(info["rr"])
        self.assertAlmostEqual(info["entry"], 3860.0, places=2)


class TestV559M2DualAnchorDecoupling(unittest.TestCase):
    """v5.5.9 M2: shared right-side leader dual-anchor helper across scoring, /pick & holding_eval."""

    @staticmethod
    def _strong_tech(**overrides) -> TechnicalSnapshot:
        data = dict(
            symbol="NVDA",
            last_close=100.0,
            ret_5d_pct=2.0,
            ret_20d_pct=8.0,
            dist_ma20_pct=2.0,
            dist_ma50_pct=4.5,
            dist_ma200_pct=15.0,
            rsi14=58.0,
            drawdown_52w_pct=-3.0,
            trend_regime="BULL",
            weekly_bullish=True,
            atr14=3.0,
            atr_pct=3.0,
            rs_20d_pct=4.0,
            rs_60d_pct=11.0,
        )
        data.update(overrides)
        return TechnicalSnapshot(**data)

    @staticmethod
    def _bundle(**overrides) -> TickerMarketBundle:
        data = dict(
            symbol="NVDA",
            quote_type="EQUITY",
            currency="USD",
            current_price=100.0,
            target_mean_price=103.0,
            target_high_price=135.0,  # 135 * 0.90 = 121.5
            analyst_count=12,
        )
        data.update(overrides)
        return TickerMarketBundle(**data)

    def test_shared_helper_unlocks_neutral_and_recovery_leaders_and_locks_weak_or_risk_off(self) -> None:
        from src.engine.screener import score_candidate
        from src.engine.screener.patterns import _normalize_regime_tag

        self.assertEqual(_normalize_regime_tag("RECOVERY"), "RECOVERY")
        strat = StrategySection(pick_rr_guard_mode="hard")
        bundle = self._bundle()
        leader = self._strong_tech()
        pos = _pos("Invest", "NVDA", cur=100.0, avg=95.0, return_pct=5.26, weight_pct=8.0)

        # Across NEUTRAL and RECOVERY, all three call sites unlock the dual-anchor ceiling/target
        for reg in ("NEUTRAL", "RECOVERY"):
            self.assertTrue(is_strong_right_side_leader(leader, regime=reg))
            anchors = resolve_invest_target_anchors(
                ref_price=100.0, bundle=bundle, tech=leader, strategy=strat, regime=reg, symbol="NVDA"
            )
            self.assertTrue(anchors["is_leader"])
            self.assertAlmostEqual(float(anchors["llm_ceiling"]), 121.5, places=2)
            self.assertAlmostEqual(float(anchors["deterministic_target"]), 121.5, places=2)

            # 1) scoring.py: est_rr computed via llm_ceiling (+21.5% vs ~4.5% stop) -> NO penalty
            cand = score_candidate(
                "NVDA",
                "INVEST_TACTICAL",
                leader,
                bundle,
                0.25,
                "MSFT",
                False,
                False,
                strat,
                regime_tag=reg,
            )
            self.assertNotIn("预估R:R", " ".join(cand.cons))
            self.assertIsNotNone(cand.filter_metrics.get("est_rr"))
            self.assertGreater(float(cand.filter_metrics["est_rr"]), 2.5)

            # 2) dual_philosophy._resolve_pick_rr (/pick guardrail): 120 target accepted (not pulled to 103)
            rr_info = _resolve_pick_rr(
                _pick("Invest", "NVDA", stop=95.0, target=120.0).selected_actions[0],
                bundle,
                strat,
                symbol="NVDA",
                account="Invest",
                tech=leader,
                regime=reg,
            )
            self.assertTrue(rr_info["is_right_side_leader"])
            self.assertAlmostEqual(rr_info["target"], 120.0, places=2)
            self.assertGreater(float(rr_info["rr"]), 2.5)

            # 3) holding_eval / compute_invest_forward_rr: uses target_high*0.90 (121.5) instead of 103
            fwd = compute_invest_forward_rr(
                pos, leader, bundle, strat, effective_stop_pct=-8.4, regime=reg
            )
            self.assertIsNotNone(fwd)
            self.assertGreater(float(fwd), 2.0)

            ev = evaluate_holding(pos, leader, bundle, strat, regime=reg)
            self.assertIsNotNone(ev.forward_rr)
            self.assertGreater(float(ev.forward_rr), 2.0)
            self.assertIsNotNone(ev.rr_ratio)
            self.assertGreater(float(ev.rr_ratio), 2.0)

        # RISK_OFF locks even a strong RS leader back to analyst_target (103) in all 3 places
        self.assertFalse(is_strong_right_side_leader(leader, regime="RISK_OFF"))
        cand_off = score_candidate(
            "NVDA", "INVEST_TACTICAL", leader, bundle, 0.25, "MSFT", False, False, strat, regime_tag="RISK_OFF"
        )
        self.assertIn("预估R:R", " ".join(cand_off.cons))
        rr_off = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=95.0, target=120.0).selected_actions[0],
            bundle,
            strat,
            symbol="NVDA",
            account="Invest",
            tech=leader,
            regime="RISK_OFF",
        )
        self.assertAlmostEqual(rr_off["target"], 103.0, places=2)
        fwd_off = compute_invest_forward_rr(
            pos, leader, bundle, strat, effective_stop_pct=-8.4, regime="RISK_OFF"
        )
        self.assertLess(float(fwd_off), 0.5)

    def test_weak_or_unconfirmed_rs_stocks_stay_deadlocked_to_analyst_target(self) -> None:
        from src.engine.screener import score_candidate

        strat = StrategySection(pick_rr_guard_mode="hard")
        bundle = self._bundle()
        pos = _pos("Invest", "NVDA", cur=100.0, avg=95.0, return_pct=5.26, weight_pct=8.0)

        non_leaders = [
            ("neg_rs20", self._strong_tech(rs_20d_pct=-0.5)),
            ("neg_rs60", self._strong_tech(rs_60d_pct=-1.0)),
            ("missing_rs", self._strong_tech(rs_20d_pct=None, rs_60d_pct=None)),
            ("below_ma200", self._strong_tech(dist_ma200_pct=-2.0)),
            ("weak_regime", self._strong_tech(trend_regime="WEAK")),
            ("bearish_weekly", self._strong_tech(weekly_bullish=False)),
        ]
        for label, bad_tech in non_leaders:
            for reg in ("RISK_ON", "NEUTRAL", "RECOVERY"):
                with self.subTest(case=label, regime=reg):
                    self.assertFalse(is_strong_right_side_leader(bad_tech, regime=reg))
                    rr_info = _resolve_pick_rr(
                        _pick("Invest", "NVDA", stop=95.0, target=120.0).selected_actions[0],
                        bundle,
                        strat,
                        symbol="NVDA",
                        account="Invest",
                        tech=bad_tech,
                        regime=reg,
                    )
                    self.assertAlmostEqual(rr_info["target"], 103.0, places=2)
                    self.assertTrue(rr_info["target_replaced"])

                    cand = score_candidate(
                        "NVDA",
                        "INVEST_TACTICAL",
                        bad_tech,
                        bundle,
                        0.25,
                        "MSFT",
                        False,
                        False,
                        strat,
                        regime_tag=reg,
                    )
                    self.assertIn("预估R:R", " ".join(cand.cons))
                    self.assertLess(float(cand.filter_metrics["est_rr"]), 1.0)

                    fwd = compute_invest_forward_rr(
                        pos, bad_tech, bundle, strat, effective_stop_pct=-8.4, regime=reg
                    )
                    self.assertLess(float(fwd), 0.5)

    def test_forward_rr_does_not_fabricate_atr_upside_when_high_target_exhausted(self) -> None:
        strat = StrategySection()
        leader = self._strong_tech(
            dist_ma20_pct=-1.0,
            ret_20d_pct=1.0,
            trend_regime="PULLBACK",
            atr_pct=3.5,
        )
        # Even for a confirmed RS leader, if price is near target_high*0.90 (105*0.90=94.5 < 103 mean),
        # deterministic_target must NOT invent +4*ATR (+14%) upside — forward R:R warning still fires.
        exhausted = self._bundle(target_mean_price=103.0, target_high_price=105.0)
        pos = _pos("Invest", "NVDA", cur=100.0, avg=98.0, return_pct=2.0, weight_pct=8.0)
        ev = evaluate_holding(pos, leader, exhausted, strat, regime="NEUTRAL")
        self.assertIsNotNone(ev.forward_rr)
        self.assertLess(float(ev.forward_rr), 0.8)
        self.assertIn("前瞻R:R恶化", " ".join(ev.reasons))

    def test_uk_dot_l_gbp_100x_calibration_and_min_analyst_count(self) -> None:
        from src.engine.screener import score_candidate

        strat = StrategySection()
        uk_tech = self._strong_tech(symbol="HSBA.L", last_close=700.0, atr14=21.0, atr_pct=3.0)
        # ratio = 15.0 / 700.0 = 0.0214 (> 0.02 generic cutoff, caught by .L 0.005..0.05 guard!)
        uk_bundle = TickerMarketBundle(
            symbol="HSBA.L",
            quote_type="EQUITY",
            currency="GBp",
            current_price=700.0,
            target_mean_price=7.20,   # 720 GBp (+2.9%)
            target_high_price=15.00,  # 1500 GBp -> *0.90 = 1350 GBp (+92.9%)
            analyst_count=6,
        )
        anchors = resolve_invest_target_anchors(
            ref_price=700.0,
            bundle=uk_bundle,
            tech=uk_tech,
            strategy=strat,
            regime="NEUTRAL",
            symbol="HSBA.L",
        )
        self.assertAlmostEqual(float(anchors["analyst_mean"]), 720.0, places=2)
        self.assertAlmostEqual(float(anchors["target_high_adj"]), 1350.0, places=2)
        self.assertAlmostEqual(float(anchors["llm_ceiling"]), 1350.0, places=2)

        rr_uk = _resolve_pick_rr(
            _pick("Invest", "HSBA.L", entry=700.0, stop=660.0, target=920.0).selected_actions[0],
            uk_bundle,
            strat,
            symbol="HSBA.L",
            account="Invest",
            tech=uk_tech,
            regime="NEUTRAL",
        )
        self.assertAlmostEqual(rr_uk["target"], 920.0, places=2)
        self.assertFalse(rr_uk["target_replaced"])

        cand_uk = score_candidate(
            "HSBA.L", "INVEST_TACTICAL", uk_tech, uk_bundle, 0.1, "MSFT", False, False, strat, regime_tag="NEUTRAL"
        )
        self.assertNotIn("预估R:R", " ".join(cand_uk.cons))

        pos_uk = _pos("Invest", "HSBA.L", cur=700.0, avg=680.0, return_pct=2.9, weight_pct=8.0, ccy="GBp")
        fwd_uk = compute_invest_forward_rr(
            pos_uk, uk_tech, uk_bundle, strat, effective_stop_pct=-8.4, regime="NEUTRAL"
        )
        self.assertIsNotNone(fwd_uk)
        self.assertGreater(float(fwd_uk), 5.0)

        # Single-analyst coverage (<3) must not let an outlier target_high_price inflate anchors
        thin_bundle = uk_bundle.model_copy(update={"analyst_count": 1})
        thin_anchors = resolve_invest_target_anchors(
            ref_price=700.0,
            bundle=thin_bundle,
            tech=uk_tech,
            strategy=strat,
            regime="NEUTRAL",
            symbol="HSBA.L",
        )
        self.assertIsNone(thin_anchors["analyst_mean"])
        self.assertIsNone(thin_anchors["target_high_adj"])
        self.assertIsNone(thin_anchors["deterministic_target"])

    def test_evaluate_portfolio_derives_benchmark_rs_from_histories_for_holding_eval(self) -> None:
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        spy_close = [100.0 + 0.05 * i for i in range(100)]
        nvda_close = [100.0 + 0.40 * i for i in range(100)]

        def _df(vals: list[float]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": vals,
                    "High": [v * 1.01 for v in vals],
                    "Low": [v * 0.99 for v in vals],
                    "Close": vals,
                    "Volume": [1_000_000.0] * len(vals),
                },
                index=dates,
            )

        # TechnicalSnapshot intentionally lacks pre-filled RS -> evaluate_portfolio derives it from SPY
        tech_no_rs = self._strong_tech(rs_20d_pct=None, rs_60d_pct=None)
        pos = _pos("Invest", "NVDA", cur=100.0, avg=95.0, return_pct=5.26, weight_pct=8.0)
        port = _portfolio([pos])
        evals = evaluate_portfolio(
            port,
            {"NVDA": tech_no_rs},
            {"NVDA": self._bundle()},
            StrategySection(),
            histories={"NVDA": _df(nvda_close), "SPY": _df(spy_close)},
            regime="RECOVERY",
        )
        self.assertEqual(len(evals), 1)
        self.assertIsNotNone(evals[0].forward_rr)
        self.assertGreater(float(evals[0].forward_rr), 2.0)

    def test_pick_post_screen_uses_isolated_bench_histories_for_adhoc_and_held_add(self) -> None:
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.agents.pick_agent import _post_screen_validate_decision

        import math

        dates = pd.date_range("2025-06-01", periods=250, freq="B")
        spy_vals = [100.0 + 0.05 * i for i in range(250)]
        stock_vals = [80.0 + 0.40 * i + 2.0 * math.sin(i * 0.35) for i in range(250)]
        adhoc_vals = [80.0 + 0.40 * i + 2.0 * math.cos(i * 0.57) for i in range(250)]
        cur_p = round(stock_vals[-1], 2)
        adhoc_p = round(adhoc_vals[-1], 2)

        def _df(vals: list[float]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": [v * 0.995 for v in vals],
                    "High": [v * 1.015 for v in vals],
                    "Low": [v * 0.985 for v in vals],
                    "Close": vals,
                    "Volume": [2_000_000.0] * len(vals),
                },
                index=dates,
            )

        spy_df = _df(spy_vals)
        amzn_df = _df(stock_vals)
        adhoc_df = _df(adhoc_vals)

        amzn_bundle = TickerMarketBundle(
            symbol="AMZN",
            quote_type="EQUITY",
            currency="USD",
            current_price=cur_p,
            target_mean_price=round(cur_p * 1.03, 2),
            target_high_price=round(cur_p * 1.35, 2),
            analyst_count=12,
        )
        adhoc_bundle = TickerMarketBundle(
            symbol="PLTR",
            quote_type="EQUITY",
            currency="USD",
            current_price=adhoc_p,
            target_mean_price=round(adhoc_p * 1.03, 2),
            target_high_price=round(adhoc_p * 1.35, 2),
            analyst_count=12,
        )

        class _FakeMD:
            async def get_bundles(self, syms, **_kw):
                out_h, out_b = {}, {}
                for s in syms:
                    if s.upper() == "PLTR":
                        out_h["PLTR"] = adhoc_df
                        out_b["PLTR"] = adhoc_bundle
                return out_h, out_b

            async def get_histories(self, syms, **_kw):
                return {"SPY": spy_df} if "SPY" in syms else {}

        cfg = SimpleNamespace(
            strategy=StrategySection(
                pick_rr_guard_mode="hard",
                trade_cadence_guard_mode="off",
                deployment_gate_mode="off",
                overlap_correlation_threshold=1.0,
            ),
            models=SimpleNamespace(ds_semantic_audit_enabled=False),
        )
        held_pos = _pos("Invest", "AMZN", value=1200.0, cur=cur_p, avg=cur_p * 0.95, return_pct=5.26, weight_pct=7.5)
        port = _portfolio([held_pos])
        decision = PickDecision(
            headline="右侧领涨精选",
            selected_actions=[
                PickActionItem(
                    account="Invest",
                    symbol="PLTR",
                    action="BUY",
                    thesis="非池临时强领涨标的",
                    entry_zone=f"{adhoc_p:.2f}",
                    stop_condition="跌破止损",
                    target_window="4周",
                    entry_price=adhoc_p,
                    stop_price=round(adhoc_p * 0.94, 2),
                    target_price=round(adhoc_p * 1.20, 2),
                ),
                PickActionItem(
                    account="Invest",
                    symbol="AMZN",
                    action="ADD",
                    thesis="已有持仓强领涨加码",
                    entry_zone=f"{cur_p:.2f}",
                    stop_condition="跌破止损",
                    target_window="4周",
                    entry_price=cur_p,
                    stop_price=round(cur_p * 0.94, 2),
                    target_price=round(cur_p * 1.20, 2),
                ),
            ],
            rejected_or_frozen=[],
            consistency_and_cash_note="遵循红线",
        )

        captured_tech_keys: list[set[str]] = []
        orig_guardrail = apply_risk_override_guardrail

        def _spy_guardrail(*args, **kwargs):
            captured_tech_keys.append(set((kwargs.get("tech_map") or {}).keys()))
            return orig_guardrail(*args, **kwargs)

        async def _run(pass_bench: bool):
            return await _post_screen_validate_decision(
                decision,
                portfolio=port,
                macro=_macro(vix=17.0),
                cfg=cfg,
                market_data=_FakeMD(),
                held_bundles={"AMZN": amzn_bundle},
                held_histories={"AMZN": amzn_df},  # Intentionally NO SPY in held_histories
                evals=[],
                isa_cands=[],
                invest_cands=[],
                guardrails=[],
                deployment_gate={"regime": "NEUTRAL"},
                stage2_tech={},
                bench_histories=({"SPY": spy_df} if pass_bench else None),
            )

        with patch("src.agents.pick_agent.apply_risk_override_guardrail", side_effect=_spy_guardrail):
            # Case A: explicit bench_histories passed from screener
            _, picks_a, _shadow_a = asyncio.run(_run(pass_bench=True))
            # Case B: bench_histories omitted -> lazy cache fetch via market_data.get_histories
            _, picks_b, _shadow_b = asyncio.run(_run(pass_bench=False))

        for picks in (picks_a, picks_b):
            syms = {p["symbol"] for p in picks}
            self.assertEqual(syms, {"PLTR", "AMZN"})
        # SPY must never pollute all_tech_map
        for keys in captured_tech_keys:
            self.assertNotIn("SPY", keys)

    def test_screener_runner_falls_back_benchmark_chain_and_exports_bench_histories(self) -> None:
        import asyncio

        from src.config import UniverseSection
        from src.engine.screener import run_quant_screener

        dates = pd.date_range("2025-06-01", periods=250, freq="B")
        bench_vals = [100.0 + 0.05 * i for i in range(250)]
        eu_vals = [100.0 + 0.35 * i for i in range(250)]

        def _df(vals: list[float]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": [v * 0.995 for v in vals],
                    "High": [v * 1.015 for v in vals],
                    "Low": [v * 0.985 for v in vals],
                    "Close": vals,
                    "Volume": [1_500_000.0] * len(vals),
                },
                index=dates,
            )

        spy_df = _df(bench_vals)
        isf_df = _df(bench_vals)
        asml_df = _df(eu_vals)

        class _FakeScreenerMD:
            async def get_histories(self, syms, **_kw):
                # EXS1.DE intentionally fails/missing -> must fall back to ISF.L / SPY
                out = {}
                for s in syms:
                    if s == "ASML.AS":
                        out[s] = asml_df
                    elif s == "ISF.L":
                        out[s] = isf_df
                    elif s == "SPY":
                        out[s] = spy_df
                return out

            async def get_bundles(self, syms, **_kw):
                out_h, out_b = {}, {}
                for s in syms:
                    if s == "ASML.AS":
                        out_h[s] = asml_df
                        out_b[s] = TickerMarketBundle(
                            symbol="ASML.AS",
                            quote_type="EQUITY",
                            currency="EUR",
                            current_price=float(eu_vals[-1]),
                            target_mean_price=float(eu_vals[-1]) * 1.03,
                            target_high_price=float(eu_vals[-1]) * 1.35,
                            analyst_count=10,
                        )
                return out_h, out_b

        out_tech: dict[str, TechnicalSnapshot] = {}
        out_bench: dict[str, pd.DataFrame] = {}
        _, inv_cands = asyncio.run(
            run_quant_screener(
                portfolio=_portfolio([]),
                market_data=_FakeScreenerMD(),
                universe=UniverseSection(isa_long_term=[], invest_tactical=["ASML.AS"]),
                watchlist=[],
                strategy=StrategySection(),
                regime_tag="NEUTRAL",
                enable_isa_equities=False,
                enable_isa_etfs=False,
                out_tech=out_tech,
                out_bench_histories=out_bench,
            )
        )
        self.assertIn("SPY", out_bench)
        self.assertIn("ISF.L", out_bench)
        self.assertIn("ASML.AS", out_tech)
        self.assertIsNotNone(out_tech["ASML.AS"].rs_20d_pct)
        self.assertIsNotNone(out_tech["ASML.AS"].rs_60d_pct)
        self.assertGreater(float(out_tech["ASML.AS"].rs_20d_pct), 0.0)
        self.assertGreater(float(out_tech["ASML.AS"].rs_60d_pct), 0.0)
        self.assertEqual(len(inv_cands), 1)
        self.assertNotIn("预估R:R", " ".join(inv_cands[0].cons))


class TestV559M3ScreenerRsiBias(unittest.TestCase):
    """v5.5.9 M3: Invest momentum track stops paying oversold bonuses / punishing Stage-2 leaders.

    The ISA value track's own RSI rules (32-55 quiet entry +3 / >=72 bubble penalty) must
    stay byte-identical — full dual-track isolation.
    """

    @staticmethod
    def _leader_hist() -> pd.DataFrame:
        # 220-bar Stage-2 uptrend with progressively tighter VCP contractions (passes
        # detect_minervini_stage2 + gives positive 20d/60d RS + rising RS slope vs the bench).
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        base = np.linspace(80.0, 130.0, 100)
        seg1 = 130.0 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 12.0
        seg2 = 132.0 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 5.5
        seg3 = 133.5 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 2.0
        closes = np.concatenate([base, seg1, seg2, seg3]).copy()
        closes[-5:] = np.linspace(134.0, 136.5, 5)
        vols = np.full(220, 2_000_000.0)
        vols[-12:] = 850_000.0
        return pd.DataFrame(
            {"Open": closes * 0.997, "High": closes * 1.012, "Low": closes * 0.988,
             "Close": closes, "Volume": vols},
            index=dates,
        )

    @staticmethod
    def _bench_hist() -> pd.DataFrame:
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        bench = np.linspace(100.0, 118.0, 160).tolist() + np.linspace(118.0, 114.0, 60).tolist()
        return pd.DataFrame(
            {"Open": np.array(bench) * 0.998, "High": np.array(bench) * 1.005,
             "Low": np.array(bench) * 0.995, "Close": bench, "Volume": np.full(220, 5_000_000.0)},
            index=dates,
        )

    @staticmethod
    def _tech(**over) -> TechnicalSnapshot:
        data = dict(
            symbol="AVGO", last_close=136.5, ma50=128.8, ma200=112.0,
            dist_ma50_pct=6.0, dist_ma200_pct=22.0, ret_60d_pct=3.4,
            trend_regime="BULL", weekly_bullish=None,
        )
        data.update(over)
        return TechnicalSnapshot(**data)

    def _score(self, tech: TechnicalSnapshot, track: str = "INVEST_TACTICAL", **kwargs):
        from src.engine.screener import score_candidate

        return score_candidate(
            "AVGO", track, tech, None, 0.30, "NVDA", False, False, StrategySection(), **kwargs
        )

    def test_invest_leader_rsi70_81_exempt_and_82plus_light_caution(self) -> None:
        # Same Stage-2 leader (minervini passes + strong RS), only RSI varies.
        hist, bench = self._leader_hist(), self._bench_hist()
        kwargs = {"hist": hist, "benchmark_hist": bench}
        c50 = self._score(self._tech(rsi14=50.0, ret_20d_pct=-1.0), **kwargs)
        c76 = self._score(self._tech(rsi14=76.0), **kwargs)
        c84 = self._score(self._tech(rsi14=84.0), **kwargs)
        c92 = self._score(self._tech(rsi14=92.0), **kwargs)
        # RSI 76 leader: zero penalty — identical dim1 to the RSI 50 no-bonus baseline.
        self.assertAlmostEqual(
            c76.dimension_scores["trend_rs"], c50.dimension_scores["trend_rs"], places=2
        )
        self.assertNotIn("极端超买", " ".join(c76.cons))
        # RSI 84 leader: light caution only (linear 1.0->3.0 over 82->92 -> -1.4 at 84).
        self.assertAlmostEqual(
            c84.dimension_scores["trend_rs"], c76.dimension_scores["trend_rs"] - 1.4, places=1
        )
        self.assertIn("极端超买(≥82)轻提示", " ".join(c84.cons))
        # RSI 92 leader: the light caution caps at -3.0.
        self.assertAlmostEqual(
            c92.dimension_scores["trend_rs"], c76.dimension_scores["trend_rs"] - 3.0, places=1
        )

    def test_invest_non_leader_keeps_the_old_70_84_penalty(self) -> None:
        # Without hist/benchmark neither Stage-2 nor strong RS can be confirmed, so the
        # classic overbought penalty stays fully intact (zero loosening for weak names).
        c50 = self._score(self._tech(rsi14=50.0, ret_20d_pct=5.0))
        c76 = self._score(self._tech(rsi14=76.0))
        c34 = self._score(self._tech(rsi14=34.0))
        # 50(+3.0 momentum-healthy) vs 76(-4.14 linear over 70->84): delta 7.14.
        self.assertAlmostEqual(
            c50.dimension_scores["trend_rs"] - c76.dimension_scores["trend_rs"], 7.14, places=1
        )
        # Oversold bonus cut 4.5 -> 1.5: momentum-healthy now BEATS deep oversold by 1.5
        # (under the old code the oversold name scored 1.5 points HIGHER).
        self.assertAlmostEqual(
            c50.dimension_scores["trend_rs"] - c34.dimension_scores["trend_rs"], 1.5, places=1
        )

    def test_isa_value_track_rsi_rules_stay_byte_identical(self) -> None:
        kw = {}
        isa55 = self._score(self._tech(rsi14=55.0, ret_20d_pct=-1.0), track="ISA_LONG_TERM", **kw)
        isa76 = self._score(self._tech(rsi14=76.0), track="ISA_LONG_TERM", **kw)
        isa38 = self._score(self._tech(rsi14=38.0), track="ISA_LONG_TERM", **kw)
        # ISA 32-55 quiet-entry bonus (+3.0) vs >=72 bubble penalty (-3.23 at 76): delta 6.23.
        self.assertAlmostEqual(
            isa55.dimension_scores["trend_rs"] - isa76.dimension_scores["trend_rs"], 6.23, places=1
        )
        # The oversold 38 sits inside the same 32-55 band as 55 — no +4.5 bonus ever existed
        # on the ISA track and the M3 Invest-side change must not leak in.
        self.assertAlmostEqual(
            isa38.dimension_scores["trend_rs"], isa55.dimension_scores["trend_rs"], places=2
        )


class TestV559M4RightTail(unittest.TestCase):
    """v5.5.9 M4: breakeven/slot-unlock ladder advisory, +25% structure-gated stop lift,
    and the NEUTRAL leader trailing tier (-11 vs -8.5). Hard stops stay byte-identical."""

    @staticmethod
    def _leader_tech(**overrides) -> TechnicalSnapshot:
        data = dict(
            symbol="NVDA",
            last_close=126.0,
            dist_ma20_pct=2.5,
            dist_ma50_pct=10.0,
            dist_ma200_pct=30.0,
            drawdown_60d_pct=-3.0,
            rsi14=60.0,
            trend_regime="BULL",
            weekly_bullish=True,
            atr_pct=3.0,
            volume_ratio_20d=1.2,
            ret_1d_pct=0.8,
            ret_20d_pct=8.0,
            rs_20d_pct=4.0,
            rs_60d_pct=9.0,
        )
        data.update(overrides)
        return TechnicalSnapshot(**data)

    def _eval(self, pos: PositionItem, tech: TechnicalSnapshot, regime: str = "NEUTRAL", **kw):
        return evaluate_holding(pos, tech, None, StrategySection(), regime=regime, **kw)

    def test_breakeven_advisory_fires_in_plus15_band_and_keeps_hold_bias(self) -> None:
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        ev = self._eval(pos, self._leader_tech())
        joined = " ".join(ev.reasons)
        self.assertIn("保本解锁线", joined)
        self.assertIn("止损上移至成本×(1+往返摩擦", joined)
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertNotIn("分批止盈", joined)

    def test_breakeven_advisory_suppressed_when_breakeven_stop_already_placed(self) -> None:
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        stop_order = PendingOrderItem(
            id="ord-1",
            account="Invest",
            raw_ticker="NVDA_USD_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA Inc.",
            side="SELL",
            order_type="STOP",
            quantity=24.0,
            value=2832.0,
            stop_price=105.0,
            currency="USD",
        )
        ev = self._eval(pos, self._leader_tech(), pending_orders=[stop_order])
        self.assertNotIn("保本解锁线", " ".join(ev.reasons))

    def test_high_take_profit_limit_order_does_not_silence_breakeven_advisory(self) -> None:
        # P2-1: a SELL LIMIT @ +40% is profit-taking, not downside protection.
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        tp_limit = PendingOrderItem(
            id="ord-2",
            account="Invest",
            raw_ticker="NVDA_USD_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA Inc.",
            side="SELL",
            order_type="LIMIT",
            quantity=24.0,
            value=3360.0,
            limit_price=140.0,
            currency="USD",
        )
        ev = self._eval(pos, self._leader_tech(), pending_orders=[tp_limit])
        self.assertIn("保本解锁线", " ".join(ev.reasons))

    def test_partially_covered_stop_does_not_silence_breakeven_advisory(self) -> None:
        # P2-1: a 12/24-share stop leaves 50% naked downside — not financed, not silenced.
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        half_stop = PendingOrderItem(
            id="ord-3",
            account="Invest",
            raw_ticker="NVDA_USD_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVDA Inc.",
            side="SELL",
            order_type="STOP",
            quantity=12.0,
            value=1416.0,
            stop_price=105.0,
            currency="USD",
        )
        ev = self._eval(pos, self._leader_tech(), pending_orders=[half_stop])
        self.assertIn("保本解锁线", " ".join(ev.reasons))

    def test_risk_off_keeps_old_defensive_take_profit_nudge_at_plus25(self) -> None:
        # P2-2: RISK_OFF freezes the offense ladder but must keep the v5.5.8
        # defensive take-profit review — defence never gets looser in a crisis.
        pos = _pos("Invest", "NVDA", cur=130.0, avg=100.0, return_pct=30.0)
        ev = self._eval(pos, self._leader_tech(), regime="RISK_OFF")
        joined = " ".join(ev.reasons)
        self.assertIn("需评估催化剂", joined)
        self.assertNotIn("锁盈地板", joined)
        self.assertNotIn("让利润奔跑", joined)

    def test_risk_off_never_emits_offense_ladder_advisories(self) -> None:
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        ev = self._eval(pos, self._leader_tech(), regime="RISK_OFF")
        joined = " ".join(ev.reasons)
        self.assertNotIn("保本解锁线", joined)
        self.assertNotIn("锁盈地板", joined)

    def test_ex_div_window_appends_caution_note_to_breakeven_advisory(self) -> None:
        pos = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        bundle = TickerMarketBundle(
            symbol="NVDA", quote_type="EQUITY", currency="USD", days_to_ex_dividend=0
        )
        ev = evaluate_holding(pos, self._leader_tech(), bundle, StrategySection(), regime="NEUTRAL")
        self.assertIn("防除权低开击穿保本止损", " ".join(ev.reasons))

    def test_plus25_leader_on_ma20_gets_stop_lift_instead_of_trim_nudge(self) -> None:
        pos = _pos("Invest", "NVDA", cur=126.0, avg=100.0, return_pct=26.0)
        ev = self._eval(pos, self._leader_tech())
        joined = " ".join(ev.reasons)
        self.assertIn("锁盈地板", joined)
        self.assertIn("沿MA20跟踪", joined)
        self.assertNotIn("需评估催化剂", joined)
        self.assertEqual(ev.action_bias, "HOLD")

    def test_plus25_below_ma20_keeps_old_take_profit_nudge(self) -> None:
        pos = _pos("Invest", "NVDA", cur=126.0, avg=100.0, return_pct=26.0)
        ev = self._eval(pos, self._leader_tech(dist_ma20_pct=-1.5))
        joined = " ".join(ev.reasons)
        self.assertIn("需评估催化剂", joined)
        self.assertNotIn("锁盈地板", joined)

    def test_plus25_heavy_down_volume_re_arms_old_nudge(self) -> None:
        pos = _pos("Invest", "NVDA", cur=126.0, avg=100.0, return_pct=26.0)
        ev = self._eval(pos, self._leader_tech(volume_ratio_20d=2.2, ret_1d_pct=-2.6))
        joined = " ".join(ev.reasons)
        self.assertIn("需评估催化剂", joined)
        self.assertNotIn("锁盈地板", joined)
        self.assertEqual(ev.action_bias, "TRIM_ALERT")

    def test_neutral_leader_trailing_tier_minus11_bounds(self) -> None:
        # dd -10.0 from peak: wider -11 leader buffer suppresses the old -8.5 warning...
        pos = _pos("Invest", "NVDA", cur=121.0, avg=100.0, return_pct=21.0)
        ev_ok = self._eval(pos, self._leader_tech(drawdown_60d_pct=-10.0))
        self.assertNotIn("移动止盈", " ".join(ev_ok.reasons))
        self.assertEqual(ev_ok.action_bias, "HOLD")
        # ...but -11.5 breaches the leader tier and re-arms the protection warning.
        ev_warn = self._eval(pos, self._leader_tech(drawdown_60d_pct=-11.5))
        joined = " ".join(ev_warn.reasons)
        self.assertIn("移动止盈", joined)
        self.assertEqual(ev_warn.action_bias, "TRIM_ALERT")

    def test_neutral_laggard_and_recovery_keep_minus85_tier(self) -> None:
        pos = _pos("Invest", "NVDA", cur=121.0, avg=100.0, return_pct=21.0)
        # RS fail-closed (missing benchmark) -> not a leader -> old -8.5 fires at -10.
        laggard = self._leader_tech(drawdown_60d_pct=-10.0, rs_20d_pct=None, rs_60d_pct=None)
        ev_lag = self._eval(pos, laggard)
        self.assertIn("移动止盈", " ".join(ev_lag.reasons))
        # RECOVERY is scoped out of the leader tier by the M4 plan: -8.5 stays.
        ev_rec = self._eval(pos, self._leader_tech(drawdown_60d_pct=-10.0), regime="RECOVERY")
        self.assertIn("移动止盈", " ".join(ev_rec.reasons))

    def test_risk_on_trailing_tier_stays_minus12(self) -> None:
        pos = _pos("Invest", "NVDA", cur=121.0, avg=100.0, return_pct=21.0)
        ev_ok = self._eval(pos, self._leader_tech(drawdown_60d_pct=-10.5), regime="RISK_ON")
        self.assertNotIn("移动止盈", " ".join(ev_ok.reasons))
        ev_warn = self._eval(pos, self._leader_tech(drawdown_60d_pct=-12.5), regime="RISK_ON")
        self.assertIn("移动止盈", " ".join(ev_warn.reasons))

    def test_advisories_are_pct_only_no_absolute_prices(self) -> None:
        pos = _pos("Invest", "NVDA", cur=126.0, avg=100.0, return_pct=26.0)
        ev = self._eval(pos, self._leader_tech())
        pos15 = _pos("Invest", "NVDA", cur=118.0, avg=100.0, return_pct=18.0)
        ev15 = self._eval(pos15, self._leader_tech())
        for joined in (" ".join(ev.reasons), " ".join(ev15.reasons)):
            self.assertNotIn("£", joined)
            self.assertNotIn("$", joined)
        self.assertIn("成本+10%锁盈地板", " ".join(ev.reasons))

    def test_investor_profile_prompts_carry_ladder_rules(self) -> None:
        from src.agents.prompts import build_investor_profile

        profile = build_investor_profile(StrategySection())
        self.assertIn("浮盈+15%推保本", profile)
        self.assertIn("max(成本+10%锁盈地板,沿MA20跟踪)", profile)


if __name__ == "__main__":
    unittest.main()
