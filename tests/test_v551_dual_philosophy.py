"""v5.5.1 dual-philosophy regression matrix (ISA Buffett value × Invest right-side momentum).

Fully offline: every provider call is mocked or avoided; no real network request is made.
"""

from __future__ import annotations

import unittest

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
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, PositionItem
from src.engine.portfolio_rules import (
    _resolve_pick_rr,
    apply_risk_override_guardrail,
    evaluate_holding,
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

        # NEUTRAL keeps the legacy ±15% analyst-mean clamp untouched
        neutral = _resolve_pick_rr(
            _pick("Invest", "NVDA", stop=94.0, target=130.0).selected_actions[0],
            bundle,
            StrategySection(),
            symbol="NVDA",
            account="Invest",
            tech=leader,
            regime="NEUTRAL",
        )
        self.assertAlmostEqual(neutral["target"], 105.0, places=2)


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
    """13. Single-message dual-account layout + 2200-char ceiling + plain text."""

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

    def test_report_dual_account_grouped_rendering_and_2200_ceiling(self) -> None:
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

        # Ceiling + plain text rules
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

        # Default ceiling is 2200; a ~1950-char report is preserved intact
        long_decision = self._decision()
        long_decision.macro_and_cash = "宏观" * 300
        long_decision.account_diagnosis = "诊断" * 300
        rendered = render_morning_report(long_decision)
        self.assertGreater(len(rendered), 1800)
        self.assertLessEqual(len(rendered), 2200)
        self.assertIn("其余长线底仓持有不动", rendered)


if __name__ == "__main__":
    unittest.main()
