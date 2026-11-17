"""Tests for pandas technical indicators, correlation matrix, Add/Trim rules, and screener."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import StrategySection, UniverseSection
from src.data_providers.analytics import (
    compute_correlation_matrix,
    compute_rsi_wilder,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
)
from src.data_providers.market_data import MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, HistoricalOrderItem, PositionItem
from src.engine.portfolio_rules import (
    _round_trip_friction_pct,
    _wallet_cash_basis_pct,
    evaluate_holding,
    format_evaluations_text,
)
from src.engine.screener import (
    compute_market_breadth,
    compute_portfolio_risk_metrics,
    compute_relative_strength,
    compute_vpvr_profile,
    detect_breakthrough_platform,
    detect_low_atr_contraction,
    detect_minervini_stage2,
    detect_vcp_contraction,
    detect_volume_ma_breakout,
    run_offline_universe_and_threshold_audit,
    run_quant_screener,
    score_candidate,
)


def _make_hist(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=dates)


def _wallet_snap(*, usd: float, gbp: float) -> AccountSnapshot:
    return AccountSnapshot(
        ok=True,
        account_label="Invest",
        total_cash=2000.0,
        available_to_trade=1500.0,
        reserved_cash=500.0,
        multi_currency_enabled=True,
        usd_cash_total=usd,
        usd_available_to_trade=usd,
        gbp_wallet_total=gbp,
        gbp_wallet_available=gbp,
        gbp_wallet_reserved=0.0,
    )


class TestDualWalletFrictionMatrix(unittest.TestCase):
    """ISA vs Invest asymmetric friction and cash-interest basis (Phase 4 matrix)."""

    def setUp(self) -> None:
        self.strategy = StrategySection()

    def _pos(self, account: str, symbol: str, ccy: str) -> PositionItem:
        return PositionItem(
            account=account,
            raw_ticker=symbol.replace(".", "_"),
            ticker=symbol.split(".")[0],
            yahoo_symbol=symbol,
            name=symbol,
            inst_currency=ccy,
            quantity=10,
            avg_price=100.0,
            cur_price=101.0,
            current_value=1010.0,
            total_cost=1000.0,
            pnl=10.0,
            fx_impact=0.0,
            return_pct=1.0,
            weight_pct=10.0,
        )

    def test_isa_us_equity_keeps_round_trip_fx_and_gbp_basis(self) -> None:
        pos = self._pos("ISA", "MSFT", "USD")
        wallet = _wallet_snap(usd=600.0, gbp=1520.0)
        # 0.30% round-trip FX + 0.05% conservative US spread estimate (v5.3.2)
        self.assertAlmostEqual(_round_trip_friction_pct(pos, None, self.strategy, wallet), 0.35)
        self.assertAlmostEqual(_wallet_cash_basis_pct(pos, self.strategy, wallet), 4.0)

    def test_invest_us_equity_with_usd_wallet_is_zero_fx(self) -> None:
        pos = self._pos("Invest", "NVDA", "USD")
        self.assertAlmostEqual(
            _round_trip_friction_pct(pos, None, self.strategy, _wallet_snap(usd=600.0, gbp=0.0)), 0.05
        )
        self.assertAlmostEqual(_wallet_cash_basis_pct(pos, self.strategy, _wallet_snap(usd=600.0, gbp=0.0)), 3.4)
        self.assertAlmostEqual(
            _round_trip_friction_pct(pos, None, self.strategy, _wallet_snap(usd=0.0, gbp=900.0)), 0.20
        )
        self.assertAlmostEqual(_wallet_cash_basis_pct(pos, self.strategy, _wallet_snap(usd=0.0, gbp=900.0)), 4.0)

    def test_invest_uk_equity_and_lse_etf_follow_gbp_wallet(self) -> None:
        uk = self._pos("Invest", "AZN.L", "GBP")
        etf = self._pos("Invest", "VUAG.L", "GBP")
        etf_bundle = TickerMarketBundle(symbol="VUAG.L", quote_type="ETF")

        rich = _wallet_snap(usd=0.0, gbp=900.0)
        self.assertAlmostEqual(_round_trip_friction_pct(uk, None, self.strategy, rich), 0.70)
        self.assertAlmostEqual(_round_trip_friction_pct(etf, etf_bundle, self.strategy, rich), 0.20)
        self.assertAlmostEqual(_wallet_cash_basis_pct(uk, self.strategy, rich), 4.0)

        poor = _wallet_snap(usd=600.0, gbp=0.0)
        self.assertAlmostEqual(_round_trip_friction_pct(uk, None, self.strategy, poor), 0.85)
        self.assertAlmostEqual(_round_trip_friction_pct(etf, etf_bundle, self.strategy, poor), 0.35)
        self.assertAlmostEqual(_wallet_cash_basis_pct(uk, self.strategy, poor), 3.4)

    def test_invest_eur_equity_keeps_round_trip_and_gbp_basis(self) -> None:
        pos = self._pos("Invest", "ASML.AS", "EUR")
        wallet = _wallet_snap(usd=600.0, gbp=0.0)
        self.assertAlmostEqual(_round_trip_friction_pct(pos, None, self.strategy, wallet), 0.35)
        self.assertAlmostEqual(_wallet_cash_basis_pct(pos, self.strategy, wallet), 4.0)

    def test_evaluate_holding_friction_note_reflects_funding_source(self) -> None:
        us = self._pos("Invest", "NVDA", "USD")
        with_usd = evaluate_holding(us, None, None, self.strategy, wallet=_wallet_snap(usd=600.0, gbp=0.0))
        self.assertIn("可用USD免换汇", with_usd.friction_note)
        no_usd = evaluate_holding(us, None, None, self.strategy, wallet=_wallet_snap(usd=0.0, gbp=900.0))
        self.assertIn("USD不足需GBP换汇", no_usd.friction_note)


class TestAnalyticsAndRules(unittest.TestCase):
    def test_technicals_uptrend_and_rsi(self) -> None:
        prices = [100.0 + i * 0.4 for i in range(260)]
        df = _make_hist(prices)
        snap = compute_technical_snapshot("NVDA", df)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.trend_regime, "BULL")
        self.assertIsNotNone(snap.ma50)
        self.assertIsNotNone(snap.ma200)
        self.assertGreater(snap.dist_ma50_pct or 0, 0)
        self.assertGreater(snap.rsi14 or 0, 90)  # pure monotonic rise -> high RSI

    def test_correlation_overlap_detection(self) -> None:
        rng = np.random.default_rng(42)
        base_rets = rng.normal(0.001, 0.015, 100)
        base_prices = 100 * np.cumprod(1 + base_rets)
        clone_prices = 200 * np.cumprod(1 + base_rets * 1.05)  # ~1.0 corr
        hedge_prices = 100 * np.cumprod(1 - base_rets * 0.8 + rng.normal(0, 0.002, 100))

        histories = {
            "VUAG.L": _make_hist(base_prices.tolist()),
            "CSP1.L": _make_hist(clone_prices.tolist()),
            "IGLT.L": _make_hist(hedge_prices.tolist()),
        }
        corr = compute_correlation_matrix(histories, lookback_days=60)
        sym_dup, val_dup = max_correlation_against_portfolio(corr, "CSP1.L", ["VUAG.L"])
        self.assertEqual(sym_dup, "VUAG.L")
        self.assertGreater(val_dup or 0, 0.99)

        sym_hdg, val_hdg = max_correlation_against_portfolio(corr, "IGLT.L", ["VUAG.L"])
        self.assertEqual(sym_hdg, "VUAG.L")
        self.assertLess(val_hdg or 1, 0.0)

    def test_add_trim_stop_rules(self) -> None:
        strategy = StrategySection()

        # Case 1: Pullback near MA50 + healthy drawdown from high -> ADD_CANDIDATE
        # Build uptrend then pull back slightly onto MA50
        prices = [100.0 + i * 0.3 for i in range(220)] + [160.0 - i * 1.2 for i in range(12)]
        tech_pb = compute_technical_snapshot("MSFT", _make_hist(prices))
        assert tech_pb is not None
        pos_isa = PositionItem(
            account="ISA",
            raw_ticker="MSFT_US_EQ",
            ticker="MSFT",
            yahoo_symbol="MSFT",
            name="Microsoft",
            inst_currency="USD",
            quantity=10,
            avg_price=400,
            cur_price=410,
            current_value=3200,
            total_cost=3100,
            pnl=100,
            fx_impact=0,
            return_pct=3.2,
            weight_pct=12.0,
        )
        ev_add = evaluate_holding(pos_isa, tech_pb, TickerMarketBundle(symbol="MSFT"), strategy)
        self.assertEqual(ev_add.action_bias, "ADD_CANDIDATE")

        # Earnings in 2 days dampens ADD_CANDIDATE
        ev_earn = evaluate_holding(
            pos_isa,
            tech_pb,
            TickerMarketBundle(symbol="MSFT", earnings_date="2026-11-16", days_to_earnings=2),
            strategy,
        )
        self.assertNotEqual(ev_earn.action_bias, "ADD_CANDIDATE")

        # Case 2: Invest position at -15% triggers REVIEW_STOP
        pos_stop = pos_isa.model_copy(update={"account": "Invest", "return_pct": -15.5, "weight_pct": 10.0})
        ev_stop = evaluate_holding(pos_stop, tech_pb, None, strategy)
        self.assertEqual(ev_stop.action_bias, "REVIEW_STOP")

        # Case 3: Overweight Invest position + overbought RSI triggers TRIM_ALERT
        hot_prices = [100.0 + i * 0.2 for i in range(210)] + [142.0 + i * 2.5 for i in range(25)]
        tech_hot = compute_technical_snapshot("NVDA", _make_hist(hot_prices))
        assert tech_hot is not None
        pos_hot = pos_isa.model_copy(update={"account": "Invest", "weight_pct": 28.0, "return_pct": 30.0})
        ev_trim = evaluate_holding(pos_hot, tech_hot, None, strategy)
        self.assertEqual(ev_trim.action_bias, "TRIM_ALERT")

    def test_screener_penalizes_overlap_and_rewards_lse_etf(self) -> None:
        strategy = StrategySection()
        prices = [100.0 + i * 0.2 for i in range(210)] + [142.0 - i * 0.5 for i in range(12)]
        tech = compute_technical_snapshot("VWRP.L", _make_hist(prices))
        assert tech is not None
        etf_bundle = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")

        c_diverse = score_candidate("VWRP.L", "ISA_LONG_TERM", tech, etf_bundle, 0.40, "RR.L", False, False, strategy)
        c_overlap = score_candidate("CSP1.L", "ISA_LONG_TERM", tech, etf_bundle, 0.96, "VUAG.L", False, False, strategy)
        self.assertFalse(c_diverse.overlap_warning)
        self.assertTrue(c_overlap.overlap_warning)
        self.assertGreater(c_diverse.score, c_overlap.score)

    def test_format_evaluations_compacts_routine_holds(self) -> None:
        from src.engine.portfolio_rules import HoldingEvaluation

        evals = [
            HoldingEvaluation(
                account="ISA",
                ticker="VWRP",
                yahoo_symbol="VWRP.L",
                weight_pct=20.0,
                return_pct=5.0,
                action_bias="HOLD",
                reasons=["技术面与仓位均在常规区间,默认持有观望"],
                friction_note="LSE英镑ETF(0%换汇/0%印花税)",
            ),
            HoldingEvaluation(
                account="ISA",
                ticker="AZN",
                yahoo_symbol="AZN.L",
                weight_pct=10.0,
                return_pct=2.0,
                action_bias="HOLD",
                reasons=["技术面与仓位均在常规区间,默认持有观望"],
                friction_note="英股(0%换汇, 买入0.5%印花税)",
            ),
            HoldingEvaluation(
                account="ISA",
                ticker="INTC",
                yahoo_symbol="INTC",
                weight_pct=5.0,
                return_pct=-4.0,
                action_bias="HOLD",
                reasons=[
                    "长线ISA处于MA200下方弱势区间(未达右侧趋势门槛),暂不触发加仓信号",
                    "CYQ上方套牢盘较重(仅22%获利),反弹留意抛压",
                ],
                friction_note="美股(0.30%往返换汇)",
            ),
            HoldingEvaluation(
                account="ISA",
                ticker="MSFT",
                yahoo_symbol="MSFT",
                weight_pct=12.0,
                return_pct=8.0,
                action_bias="HOLD",
                reasons=["⚠️ 距财报仅3天(2026-11-19),财报前谨慎重仓加码"],
                friction_note="美股(0.30%往返换汇)",
            ),
            HoldingEvaluation(
                account="Invest",
                ticker="NVDA",
                yahoo_symbol="NVDA",
                weight_pct=26.0,
                return_pct=-13.0,
                action_bias="REVIEW_STOP",
                reasons=["Invest短线仓浮亏-13.0%(已触发止损复核线-12%)"],
                friction_note="美股(0.30%往返换汇)",
            ),
        ]
        txt = format_evaluations_text(evals, compact_hold=True)
        self.assertIn("常规持有 x3 只均无异常", txt)
        self.assertIn("VWRP[ISA 20.0%/+5.0%]", txt)
        self.assertIn("INTC[ISA 5.0%/-4.0%]", txt)
        self.assertIn("[ISA] MSFT ⚪ [常规持有]", txt)
        self.assertIn("[Invest] NVDA 🔴 [止损/证伪复核]", txt)

    def test_v505_enhanced_risk_rules_and_debeta_correlation(self) -> None:
        from src.data_providers.analytics import TechnicalSnapshot
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData
        from src.engine.portfolio_rules import evaluate_portfolio

        strategy = StrategySection()

        # 1. Combined portfolio weight 22% triggers TRIM_ALERT for stocks (including GOOG/GOOGL) but EXEMPTS ETFs
        pos_goog_isa = PositionItem(
            account="ISA",
            raw_ticker="GOOGL_US_EQ",
            ticker="GOOGL",
            yahoo_symbol="GOOGL",
            name="Alphabet A",
            inst_currency="USD",
            quantity=10,
            avg_price=150,
            cur_price=160,
            current_value=3200,
            total_cost=3000,
            pnl=200,
            fx_impact=0,
            return_pct=6.7,
            weight_pct=16.0,
        )
        pos_goog_inv = pos_goog_isa.model_copy(
            update={"account": "Invest", "raw_ticker": "GOOG_US_EQ", "ticker": "GOOG", "yahoo_symbol": "GOOG", "current_value": 1600, "weight_pct": 16.0}
        )
        pos_etf_isa = pos_goog_isa.model_copy(
            update={
                "raw_ticker": "VWRPl_EQ",
                "ticker": "VWRP",
                "yahoo_symbol": "VWRP.L",
                "name": "Vanguard All-World",
                "inst_currency": "GBP",
                "current_value": 7000,
                "weight_pct": 35.0,
            }
        )
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", account_currency="GBP", total_account_value=20000, available_to_trade=9800, positions=[pos_goog_isa, pos_etf_isa]),
            invest=AccountSnapshot(ok=True, account_label="Invest", account_currency="GBP", total_account_value=6400, available_to_trade=3400, positions=[pos_goog_inv.model_copy(update={"current_value": 3000, "weight_pct": 20.0})]),
        )

        bundle_map = {
            "GOOGL": TickerMarketBundle(symbol="GOOGL", quote_type="EQUITY"),
            "GOOG": TickerMarketBundle(symbol="GOOG", quote_type="EQUITY"),
            "VWRP.L": TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP"),
        }
        evals = evaluate_portfolio(portfolio, {}, bundle_map, strategy)
        by_sym = {e.yahoo_symbol: e for e in evals}
        self.assertEqual(by_sym["GOOGL"].action_bias, "TRIM_ALERT")
        self.assertIn("同发行人跨账户合并", " ".join(by_sym["GOOGL"].reasons))
        self.assertEqual(by_sym["VWRP.L"].action_bias, "HOLD")  # ETF completely exempt from weight caps

        # 2. Trailing drawdown (-12% from 60d peak after +20% profit) triggers TRIM_ALERT, while <20% profit does not
        tech_trail = TechnicalSnapshot(
            symbol="NVDA",
            last_close=120.0,
            dist_ma50_pct=-3.0,
            dist_ma200_pct=10.0,
            drawdown_60d_pct=-13.5,
            drawdown_52w_pct=-15.0,
            trend_regime="PULLBACK",
        )
        pos_win = pos_goog_inv.model_copy(update={"ticker": "NVDA", "yahoo_symbol": "NVDA", "weight_pct": 10.0, "return_pct": 21.5})
        ev_trail = evaluate_holding(pos_win, tech_trail, None, strategy)
        self.assertEqual(ev_trail.action_bias, "TRIM_ALERT")
        self.assertIn("移动止盈", " ".join(ev_trail.reasons))

        # 3. require_uptrend_for_add blocks ADD_CANDIDATE when below MA200 (even in PULLBACK band -1.5%)
        tech_below_ma200 = TechnicalSnapshot(
            symbol="AMD",
            last_close=98.5,
            dist_ma50_pct=-2.0,
            dist_ma200_pct=-1.5,
            rsi14=34.0,
            drawdown_52w_pct=-15.0,
            trend_regime="PULLBACK",
        )
        pos_isa_amd = pos_goog_isa.model_copy(update={"ticker": "AMD", "yahoo_symbol": "AMD", "weight_pct": 8.0, "return_pct": -5.0})
        ev_no_add = evaluate_holding(pos_isa_amd, tech_below_ma200, None, strategy)
        self.assertEqual(ev_no_add.action_bias, "HOLD")
        self.assertNotIn("可考虑分批低吸", " ".join(ev_no_add.reasons))

        # 4. ISA deep drawdown (-25%) triggers review warning and prevents ADD_CANDIDATE
        pos_isa_deep = pos_isa_amd.model_copy(update={"return_pct": -28.0})
        tech_above_ma200 = tech_below_ma200.model_copy(update={"dist_ma200_pct": 2.0})
        ev_deep = evaluate_holding(pos_isa_deep, tech_above_ma200, None, strategy)
        self.assertNotEqual(ev_deep.action_bias, "ADD_CANDIDATE")
        self.assertIn("深套复核线", " ".join(ev_deep.reasons))

        # 5. De-beta'd correlation strips systemic SPY crash while preserving index ETF clone detection
        rng = np.random.default_rng(7)
        spy_rets = rng.normal(-0.004, 0.025, 90)  # systemic volatile selloff
        bank_rets = 1.0 * spy_rets + rng.normal(0.0, 0.006, 90)
        semi_rets = 1.1 * spy_rets + rng.normal(0.0, 0.007, 90)
        clone_rets = spy_rets + rng.normal(0.0, 0.0005, 90)
        clone2_rets = spy_rets + rng.normal(0.0, 0.0005, 90)
        h_crash = {
            "SPY": _make_hist((100 * np.cumprod(1 + spy_rets)).tolist()),
            "JPM": _make_hist((100 * np.cumprod(1 + bank_rets)).tolist()),
            "NVDA": _make_hist((100 * np.cumprod(1 + semi_rets)).tolist()),
            "VUAG.L": _make_hist((100 * np.cumprod(1 + clone_rets)).tolist()),
            "CSP1.L": _make_hist((100 * np.cumprod(1 + clone2_rets)).tolist()),
        }
        raw_only = compute_correlation_matrix(h_crash, ["JPM", "NVDA"], lookback_days=60, benchmark_symbol=None)
        debeta = compute_correlation_matrix(h_crash, ["JPM", "NVDA", "VUAG.L", "CSP1.L"], lookback_days=60, benchmark_symbol="SPY")
        self.assertGreater(float(raw_only.loc["JPM", "NVDA"]), 0.85)  # raw corr spikes during systemic crash
        self.assertLess(float(debeta.loc["JPM", "NVDA"]), 0.40)     # residual corr exposes true independence
        self.assertGreater(float(debeta.loc["VUAG.L", "CSP1.L"]), 0.95)  # index ETF clones still flagged!

    def test_cyq_metrics_bottom_single_peak_overhead_supply_and_low_price_precision(self) -> None:
        from src.data_providers.analytics import compute_cyq_metrics

        dates = pd.date_range("2025-10-01", periods=210, freq="B")

        # 1. Bottom single-peak accumulation: long tight base around 100, then slight lift to 104
        base_closes = [100.0 + (i % 5 - 2) * 0.6 for i in range(195)] + [102.0 + i * 0.15 for i in range(15)]
        df_base = pd.DataFrame(
            {
                "Open": [c - 0.3 for c in base_closes],
                "High": [c + 0.8 for c in base_closes],
                "Low": [c - 0.8 for c in base_closes],
                "Close": base_closes,
                "Volume": [1_000_000.0 for _ in range(195)] + [1_400_000.0 for _ in range(15)],
            },
            index=dates,
        )
        cyq_bottom = compute_cyq_metrics(df_base)
        self.assertIsNotNone(cyq_bottom)
        assert cyq_bottom is not None
        self.assertGreaterEqual(cyq_bottom.benefit_part_pct, 80.0)
        self.assertLessEqual(cyq_bottom.concentration_90_pct, 8.0)
        self.assertEqual(cyq_bottom.regime, "底部单峰密集")

        # 2. Overhead trapped supply: heavy volume distribution around 150, then drop to 105 on light volume
        trap_closes = [150.0 + (i % 4 - 1.5) * 1.0 for i in range(170)] + [145.0 - i * 1.0 for i in range(40)]
        df_trap = pd.DataFrame(
            {
                "Open": [c + 0.4 for c in trap_closes],
                "High": [c + 1.5 for c in trap_closes],
                "Low": [c - 1.5 for c in trap_closes],
                "Close": trap_closes,
                "Volume": [5_000_000.0 for _ in range(170)] + [400_000.0 for _ in range(40)],
            },
            index=dates,
        )
        cyq_trap = compute_cyq_metrics(df_trap)
        self.assertIsNotNone(cyq_trap)
        assert cyq_trap is not None
        self.assertLessEqual(cyq_trap.benefit_part_pct, 20.0)
        self.assertGreater(cyq_trap.avg_cost, 120.0)
        self.assertEqual(cyq_trap.regime, "上方套牢峰")

        # 3. Low-priced GBP stock (0.72 ~ 0.82 GBP) + zero-volume suspended rows + close-only fallback
        low_closes = [0.75 + (i % 5) * 0.005 for i in range(200)] + [0.81 for _ in range(10)]
        vols_with_zero = [250_000.0 if i != 150 else 0.0 for i in range(210)]
        df_low_gbp = pd.DataFrame(
            {
                "Open": [c - 0.002 for c in low_closes],
                "High": [c + 0.006 for c in low_closes],
                "Low": [c - 0.006 for c in low_closes],
                "Close": low_closes,
                "Volume": vols_with_zero,
            },
            index=dates,
        )
        cyq_low = compute_cyq_metrics(df_low_gbp)
        self.assertIsNotNone(cyq_low)
        assert cyq_low is not None
        self.assertAlmostEqual(cyq_low.avg_cost, 0.76, delta=0.04)
        self.assertGreater(cyq_low.benefit_part_pct, 80.0)

        # Close-only DataFrame gracefully returns None for CYQ without throwing
        self.assertIsNone(compute_cyq_metrics(_make_hist(base_closes)))

    def test_multi_timeframe_atr_adx_bb_squeeze_volume_and_post_entry_drawdown(self) -> None:
        from src.data_providers.analytics import compute_drawdown_since_date, compute_technical_snapshot

        dates = pd.date_range("2025-10-01", periods=260, freq="B")
        # Construct steady uptrend for 230 days with wide swings, then tight volatility compression for last 30 days
        closes: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        vols: list[float] = []
        for i in range(260):
            if i < 230:
                c = 100.0 + i * 0.4 + math.sin(i * 0.5) * 3.5 if "math" in globals() else 100.0 + i * 0.4 + ((i % 7) - 3) * 1.8
                h = c + 2.8
                l = c - 2.8
                v = 1_000_000.0
            else:
                # Tight Bollinger squeeze around 192 with final day volume surge
                c = 192.0 + ((i % 3) - 1) * 0.15
                h = c + 0.35
                l = c - 0.35
                v = 2_400_000.0 if i == 259 else 900_000.0
            closes.append(c)
            highs.append(h)
            lows.append(l)
            vols.append(v)

        df = pd.DataFrame(
            {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=dates,
        )
        snap = compute_technical_snapshot("AZN.L", df)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertIsNotNone(snap.atr14)
        self.assertIsNotNone(snap.atr_pct)
        self.assertGreater(snap.atr14 or 0.0, 0.2)
        self.assertIsNotNone(snap.volume_ratio_20d)
        self.assertGreater(snap.volume_ratio_20d or 0.0, 2.0)  # final day 2.4M vs ~950k 20d avg
        self.assertTrue(snap.weekly_bullish)
        self.assertIsNotNone(snap.weekly_ma10)
        self.assertIsNotNone(snap.weekly_ma30)
        self.assertIsNotNone(snap.weekly_rsi14)
        self.assertTrue(snap.bb_squeeze)
        self.assertIsNotNone(snap.adx14)
        self.assertIsNotNone(snap.cyq_benefit_part_pct)
        self.assertIsNotNone(snap.cyq_avg_cost)

        # Post-entry drawdown: spike to 240 occurred BEFORE position entry date, after entry price stayed around 192
        spike_closes = closes.copy()
        spike_closes[210] = 240.0  # 50 trading days ago (within 60d window, before entry at index 240)
        df_spike = df.copy()
        df_spike["Close"] = spike_closes
        entry_date_iso = dates[240].strftime("%Y-%m-%dT09:00:00Z")

        raw_dd_60d = compute_drawdown_since_date(df_spike, created_at=None, max_lookback_days=60)
        post_entry_dd = compute_drawdown_since_date(df_spike, created_at=entry_date_iso, max_lookback_days=60)
        self.assertLess(raw_dd_60d or 0.0, -18.0)       # -20% from pre-entry 240 spike
        self.assertGreater(post_entry_dd or -10.0, -2.0)  # < 2% drawdown since actual entry date!

    def test_regional_benchmark_debeta_and_clone_etf_exposure_aggregation(self) -> None:
        from src.data_providers.analytics import (
            compute_exposure_summary,
            format_technical_line,
            infer_etf_index_family,
            resolve_market_benchmark,
        )
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        # 1. Benchmark resolution separates LSE US-tracker ETFs (VUAG.L -> SPY) from domestic UK stocks (AZN.L -> ISF.L)
        self.assertEqual(resolve_market_benchmark("VUAG.L")[0], "SPY")
        self.assertEqual(resolve_market_benchmark("CSP1.L")[0], "SPY")
        self.assertEqual(resolve_market_benchmark("AZN.L")[0], "ISF.L")
        self.assertEqual(resolve_market_benchmark("SAP.DE")[0], "EXS1.DE")

        # 2. Regional de-Beta strips ISF.L from UK stocks while keeping VUAG.L/CSP1.L anchored to SPY
        rng = np.random.default_rng(19)
        uk_mkt_rets = rng.normal(-0.003, 0.022, 90)
        us_mkt_rets = rng.normal(0.001, 0.015, 90)
        azn_rets = 1.05 * uk_mkt_rets + rng.normal(0.0, 0.005, 90)
        hsba_rets = 0.95 * uk_mkt_rets + rng.normal(0.0, 0.005, 90)
        vuag_rets = 1.00 * us_mkt_rets + rng.normal(0.0, 0.0004, 90)
        csp1_rets = 1.00 * us_mkt_rets + rng.normal(0.0, 0.0004, 90)

        h_multi = {
            "SPY": _make_hist((100 * np.cumprod(1 + us_mkt_rets)).tolist()),
            "ISF.L": _make_hist((100 * np.cumprod(1 + uk_mkt_rets)).tolist()),
            "AZN.L": _make_hist((100 * np.cumprod(1 + azn_rets)).tolist()),
            "HSBA.L": _make_hist((100 * np.cumprod(1 + hsba_rets)).tolist()),
            "VUAG.L": _make_hist((100 * np.cumprod(1 + vuag_rets)).tolist()),
            "CSP1.L": _make_hist((100 * np.cumprod(1 + csp1_rets)).tolist()),
        }
        raw_uk = compute_correlation_matrix(h_multi, ["AZN.L", "HSBA.L"], lookback_days=60, benchmark_symbol=None)
        debeta_uk = compute_correlation_matrix(
            h_multi,
            ["AZN.L", "HSBA.L", "VUAG.L", "CSP1.L"],
            lookback_days=60,
            benchmark_symbol="SPY",
            uk_benchmark_symbol="ISF.L",
        )
        self.assertGreater(float(raw_uk.loc["AZN.L", "HSBA.L"]), 0.85)
        self.assertLess(float(debeta_uk.loc["AZN.L", "HSBA.L"]), 0.35)
        self.assertGreater(float(debeta_uk.loc["VUAG.L", "CSP1.L"]), 0.95)

        # 3. Clone ETF underlying exposure aggregation in compute_exposure_summary
        self.assertEqual(infer_etf_index_family("VUAG.L"), "S&P 500 (美股标普500)")
        self.assertEqual(infer_etf_index_family("CSP1.L"), "S&P 500 (美股标普500)")
        pos_vuag = PositionItem(
            account="ISA",
            raw_ticker="VUAGl_EQ",
            ticker="VUAG",
            yahoo_symbol="VUAG.L",
            name="Vanguard S&P 500",
            inst_currency="GBP",
            quantity=30,
            avg_price=80,
            cur_price=90,
            current_value=2700,
            total_cost=2400,
            pnl=300,
            fx_impact=0,
            return_pct=12.5,
            weight_pct=27.0,
        )
        pos_csp1 = pos_vuag.model_copy(
            update={
                "account": "Invest",
                "raw_ticker": "CSP1l_EQ",
                "ticker": "CSP1",
                "yahoo_symbol": "CSP1.L",
                "name": "iShares Core S&P 500",
                "current_value": 1800,
                "weight_pct": 18.0,
            }
        )
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", account_currency="GBP", total_account_value=10000, available_to_trade=2000, positions=[pos_vuag]),
            invest=AccountSnapshot(ok=True, account_label="Invest", account_currency="GBP", total_account_value=10000, available_to_trade=2000, positions=[pos_csp1]),
        )
        bundles = {
            "VUAG.L": TickerMarketBundle(symbol="VUAG.L", quote_type="ETF", currency="GBP"),
            "CSP1.L": TickerMarketBundle(symbol="CSP1.L", quote_type="ETF", currency="GBP"),
        }
        exp_txt = compute_exposure_summary(portfolio, bundles)
        self.assertIn("克隆ETF底层暴露归并 [S&P 500 (美股标普500)]", exp_txt)
        self.assertIn("ISA VUAG £2,700 + Invest CSP1 £1,800 = 合计 £4,500 (占总资产 22.5%)", exp_txt)

        # 4. Compact format_technical_line includes new fields and omits N/A noise
        from src.data_providers.analytics import TechnicalSnapshot

        tech_rich = TechnicalSnapshot(
            symbol="HSBA.L",
            last_close=720.5,
            ret_5d_pct=2.4,
            ret_20d_pct=6.1,
            dist_ma50_pct=3.2,
            dist_ma200_pct=11.5,
            rsi14=58.2,
            trend_regime="BULL",
            atr14=11.4,
            atr_pct=1.58,
            volume_ratio_20d=1.65,
            weekly_bullish=True,
            weekly_rsi14=61.0,
            bb_width_pct=5.2,
            bb_squeeze=True,
            adx14=28.4,
            cyq_benefit_part_pct=84.0,
            cyq_avg_cost=690.0,
            cyq_concentration_90_pct=9.5,
            cyq_regime="底部单峰密集",
        )
        bundle_rich = TickerMarketBundle(
            symbol="HSBA.L",
            currency="GBp",
            forward_pe=7.8,
            dividend_yield_pct=6.45,
            ex_dividend_date="2026-11-26",
            days_to_ex_dividend=12,
        )
        line = format_technical_line("HSBA.L", tech_rich, bundle_rich)
        self.assertIn("周多头(周RSI 61)", line)
        self.assertIn("ATR14 11.40(1.6%)", line)
        self.assertIn("量比 1.65x(放量)", line)
        self.assertIn("ADX 28", line)
        self.assertIn("BB挤压(5.2%)", line)
        self.assertIn("筹码[底部单峰密集] 获利84%/均成本690.00/集中9.5%", line)
        self.assertIn("股息率 6.45%", line)
        self.assertIn("除息 2026-11-26(12天后)", line)
        self.assertNotIn("N/A", line)

    def test_split_adjustment_cross_timezone_lag_corr_and_ex_div_guard(self) -> None:
        from src.data_providers.analytics import (
            compute_correlation_matrix,
            compute_cyq_metrics,
            compute_technical_snapshot,
            format_technical_line,
        )

        dates = pd.date_range("2025-10-01", periods=210, freq="B")

        # 1. Unadjusted 10-for-1 stock split + 1-bar 100x GBp bad tick:
        # First 180 bars trade around 1000 (pre-split), last 30 bars around 103 (post 10:1 split),
        # with bar 195 corrupted by a single-day 100x GBp quote (10300.0).
        raw_closes = [1000.0 + (i % 5 - 2) * 5.0 for i in range(180)] + [102.0 + (i % 3) * 0.5 for i in range(30)]
        raw_opens = [c - 2.0 if i < 180 else c - 0.2 for i, c in enumerate(raw_closes)]
        raw_highs = [c + 6.0 if i < 180 else c + 0.6 for i, c in enumerate(raw_closes)]
        raw_lows = [c - 6.0 if i < 180 else c - 0.6 for i, c in enumerate(raw_closes)]
        # Inject 1-bar 100x GBp bad tick at index 195
        raw_closes[195] = 10300.0
        raw_opens[195] = 10250.0
        raw_highs[195] = 10350.0
        raw_lows[195] = 10200.0

        df_unadj_split = pd.DataFrame(
            {
                "Open": raw_opens,
                "High": raw_highs,
                "Low": raw_lows,
                "Close": raw_closes,
                "Volume": [1_000_000.0 for _ in range(210)],
            },
            index=dates,
        )
        cyq_split = compute_cyq_metrics(df_unadj_split)
        self.assertIsNotNone(cyq_split)
        assert cyq_split is not None
        self.assertGreaterEqual(cyq_split.benefit_part_pct, 80.0)
        self.assertAlmostEqual(cyq_split.avg_cost, 100.5, delta=3.0)
        self.assertNotEqual(cyq_split.regime, "上方套牢峰")

        # 2. Cross-timezone closing lag (LSE 16:30 vs NYSE 21:00) + asynchronous 1-day holidays
        rng = np.random.default_rng(42)
        n_days = 80
        spy_r = rng.normal(0.0005, 0.014, n_days)
        # VUAG.L only captures 35% of SPY's same-day move before 16:30 UK, and 65% next morning at LSE open
        vuag_r = np.zeros(n_days, dtype=float)
        vuag_r[0] = spy_r[0]
        for t in range(1, n_days):
            vuag_r[t] = 0.35 * spy_r[t] + 0.65 * spy_r[t - 1]

        spy_df = _make_hist((100.0 * np.cumprod(1.0 + spy_r)).tolist())
        vuag_df = _make_hist((100.0 * np.cumprod(1.0 + vuag_r)).tolist())
        # Drop asynchronous single-day holidays (e.g. UK Bank Holiday vs US holiday)
        spy_df = spy_df.drop(index=[spy_df.index[25], spy_df.index[55]])
        vuag_df = vuag_df.drop(index=[vuag_df.index[35], vuag_df.index[65]])

        corr_mat = compute_correlation_matrix(
            {"SPY": spy_df, "VUAG.L": vuag_df},
            ["SPY", "VUAG.L"],
            lookback_days=60,
            benchmark_symbol="SPY",
        )
        self.assertGreaterEqual(float(corr_mat.loc["SPY", "VUAG.L"]), 0.85)

        # 3. Ex-dividend window & drop guard linkage in TechnicalSnapshot and format_technical_line
        bundle_ex_today = TickerMarketBundle(
            symbol="RIO.L",
            currency="GBp",
            dividend_yield_pct=5.8,
            ex_dividend_date="2026-11-14",
            days_to_ex_dividend=0,
        )
        snap_ex = compute_technical_snapshot("RIO.L", df_unadj_split, bundle=bundle_ex_today)
        self.assertIsNotNone(snap_ex)
        assert snap_ex is not None
        self.assertEqual(snap_ex.days_to_ex_dividend, 0)
        self.assertTrue(snap_ex.is_ex_div_window)
        self.assertTrue(snap_ex.is_ex_div_drop_guard)
        line_ex_today = format_technical_line("RIO.L", snap_ex, bundle_ex_today)
        self.assertIn("除息 2026-11-14(今日⚠️刚除息自然扣息勿误判破位)", line_ex_today)

        # Also verify format_technical_line syncs flags onto a snapshot created without bundle
        snap_no_b = compute_technical_snapshot("RIO.L", df_unadj_split)
        assert snap_no_b is not None
        self.assertFalse(snap_no_b.is_ex_div_drop_guard)
        bundle_ex_yesterday = bundle_ex_today.model_copy(update={"days_to_ex_dividend": -1})
        line_ex_yest = format_technical_line("RIO.L", snap_no_b, bundle_ex_yesterday)
        self.assertTrue(snap_no_b.is_ex_div_window)
        self.assertTrue(snap_no_b.is_ex_div_drop_guard)
        self.assertIn("1天前⚠️刚除息自然扣息勿误判破位", line_ex_yest)

    def test_imp07_post_entry_trailing_atr_stop_ex_div_and_fx_distortion(self) -> None:
        from src.data_providers.analytics import TechnicalSnapshot, compute_technical_snapshot
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData
        from src.engine.portfolio_rules import evaluate_holding, evaluate_portfolio

        strategy = StrategySection()

        # 1. Post-entry trailing drawdown: peak of 150 occurs at bar 175 (within 60d window);
        # ISA entered before the peak (bar 172 -> triggers trailing stop), Invest entered after (bar 185 -> exempt).
        dates = pd.date_range("2026-01-01", periods=220, freq="B")
        closes = (
            [100.0 + i * 0.08 for i in range(170)]
            + [120.0, 128.0, 136.0, 144.0, 148.0, 150.0, 140.0, 126.0, 115.0, 108.0]
            + [105.0 + (i % 3) * 0.3 for i in range(40)]
        )
        hist = pd.DataFrame(
            {
                "Open": [c - 0.5 for c in closes],
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
                "Close": closes,
                "Volume": [1_000_000.0] * 220,
            },
            index=dates,
        )
        entry_date_recent = dates[185].strftime("%Y-%m-%dT10:00:00Z")
        entry_date_old = dates[172].strftime("%Y-%m-%dT10:00:00Z")
        tech_raw = compute_technical_snapshot("NVDA", hist)  # without created_at -> drawdown_60d_pct ~ -30% from 150!
        assert tech_raw is not None

        pos_recent = PositionItem(
            account="Invest",
            raw_ticker="NVDA_US_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVIDIA",
            inst_currency="USD",
            quantity=10,
            avg_price=85.0,
            cur_price=105.0,
            current_value=820,
            total_cost=660,
            pnl=160,
            fx_impact=0,
            return_pct=24.2,
            weight_pct=10.0,
            created_at=entry_date_recent,
        )
        pos_old = pos_recent.model_copy(update={"account": "ISA", "created_at": entry_date_old})
        portfolio_dual = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=8200, positions=[pos_old]),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=8200, positions=[pos_recent]),
        )
        evals_dual = evaluate_portfolio(portfolio_dual, {"NVDA": tech_raw}, {}, strategy, histories={"NVDA": hist})
        by_acct = {e.account: e for e in evals_dual}
        self.assertNotIn("移动止盈", " ".join(by_acct["Invest"].reasons))
        self.assertNotEqual(by_acct["Invest"].action_bias, "TRIM_ALERT")
        self.assertIn("移动止盈", " ".join(by_acct["ISA"].reasons))
        self.assertEqual(by_acct["ISA"].action_bias, "TRIM_ALERT")

        # 2. ATR adaptive stop: high volatility (atr_pct=5.5% -> stop=-15.4%) avoids premature -12% stop;
        # low volatility (atr_pct=2.0% -> stop=-8.0%) triggers REVIEW_STOP at -9.5%.
        tech_hi_vol = TechnicalSnapshot(symbol="TSLA", last_close=200.0, atr_pct=5.5, trend_regime="PULLBACK")
        pos_tsla = pos_recent.model_copy(
            update={"ticker": "TSLA", "yahoo_symbol": "TSLA", "avg_price": 230.0, "cur_price": 200.0, "return_pct": -13.0, "created_at": None}
        )
        ev_hi_vol = evaluate_holding(pos_tsla, tech_hi_vol, None, strategy)
        self.assertEqual(ev_hi_vol.effective_stop_pct, -15.4)
        self.assertNotEqual(ev_hi_vol.action_bias, "REVIEW_STOP")

        tech_lo_vol = TechnicalSnapshot(symbol="KO", last_close=60.0, atr_pct=2.0, trend_regime="PULLBACK")
        pos_ko = pos_recent.model_copy(
            update={"ticker": "KO", "yahoo_symbol": "KO", "avg_price": 66.3, "cur_price": 60.0, "return_pct": -9.5, "created_at": None}
        )
        ev_lo_vol = evaluate_holding(pos_ko, tech_lo_vol, None, strategy)
        self.assertEqual(ev_lo_vol.effective_stop_pct, -8.0)
        self.assertEqual(ev_lo_vol.action_bias, "REVIEW_STOP")

        # 3. FX distortion: native stock dropped -13.5% (breaches -12% stop), while GBP return shows only -5.0%
        pos_fx_masked = pos_recent.model_copy(
            update={"ticker": "AMD", "yahoo_symbol": "AMD", "avg_price": 100.0, "cur_price": 86.5, "return_pct": -5.0, "created_at": None}
        )
        ev_fx = evaluate_holding(pos_fx_masked, None, None, strategy)
        self.assertEqual(ev_fx.action_bias, "REVIEW_STOP")
        self.assertIn("受汇率掩盖", " ".join(ev_fx.reasons))

        # 4. Ex-dividend drop guard exempts mild stop/trailing/MA200 breakdown, but NOT deep -22% crash
        tech_ex = TechnicalSnapshot(
            symbol="RIO.L",
            last_close=4800.0,
            dist_ma200_pct=-1.8,
            drawdown_60d_pct=-13.5,
            atr_pct=3.0,  # adaptive stop -8.4%
            trend_regime="WEAK",
            is_ex_div_window=True,
            is_ex_div_drop_guard=True,
            days_to_ex_dividend=0,
        )
        bundle_rio = TickerMarketBundle(symbol="RIO.L", currency="GBp", dividend_yield_pct=5.5, days_to_ex_dividend=0)
        pos_rio_mild = pos_recent.model_copy(
            update={"ticker": "RIO", "yahoo_symbol": "RIO.L", "inst_currency": "GBp", "avg_price": 5200.0, "cur_price": 4700.0, "return_pct": -9.6, "created_at": None}
        )
        ev_rio_mild = evaluate_holding(pos_rio_mild, tech_ex, bundle_rio, strategy)
        self.assertTrue(ev_rio_mild.ex_div_guarded)
        self.assertNotEqual(ev_rio_mild.action_bias, "REVIEW_STOP")
        self.assertIn("除息窗口", " ".join(ev_rio_mild.reasons))

        pos_rio_crash = pos_rio_mild.model_copy(update={"cur_price": 4000.0, "return_pct": -22.0})
        ev_rio_crash = evaluate_holding(pos_rio_crash, tech_ex, bundle_rio, strategy)
        self.assertEqual(ev_rio_crash.action_bias, "REVIEW_STOP")

    def test_imp07_pending_orders_cash_hurdle_and_cyq_rules(self) -> None:
        from src.data_providers.analytics import TechnicalSnapshot
        from src.data_providers.t212_client import AccountSnapshot, PendingOrderItem, PortfolioData
        from src.engine.portfolio_rules import evaluate_holding, evaluate_portfolio

        strategy = StrategySection()
        tech_pb = TechnicalSnapshot(
            symbol="MSFT",
            last_close=400.0,
            dist_ma50_pct=-1.5,
            dist_ma200_pct=6.0,
            drawdown_52w_pct=-12.0,
            rsi14=42.0,
            weekly_bullish=True,
            adx14=28.0,
            volume_ratio_20d=0.75,
            cyq_regime="底部单峰密集",
            cyq_benefit_part_pct=72.0,
            trend_regime="PULLBACK",
        )
        pos_msft = PositionItem(
            account="ISA",
            raw_ticker="MSFT_US_EQ",
            ticker="MSFT",
            yahoo_symbol="MSFT",
            name="Microsoft",
            inst_currency="USD",
            quantity=10,
            avg_price=390.0,
            cur_price=400.0,
            current_value=1800.0,
            total_cost=1750.0,
            pnl=50.0,
            fx_impact=0.0,
            return_pct=2.9,
            weight_pct=18.0,
        )
        # Pending BUY order pushes projected cross-account weight over 22% -> blocks ADD_CANDIDATE
        buy_order = PendingOrderItem(
            id="ord-1",
            account="ISA",
            raw_ticker="MSFT_US_EQ",
            ticker="MSFT",
            yahoo_symbol="MSFT",
            name="Microsoft",
            side="BUY",
            order_type="LIMIT",
            quantity=5.0,  # +900 GBP -> 2700 / 10000 = 27% >= 22%
            value=0.0,
            limit_price=390.0,
            currency="USD",
        )
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=10000.0, positions=[pos_msft], pending_orders=[buy_order]),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=10000.0,
        )
        ev_pending_block = evaluate_portfolio(portfolio, {"MSFT": tech_pb}, {}, strategy)[0]
        self.assertNotEqual(ev_pending_block.action_bias, "ADD_CANDIDATE")
        self.assertGreater(ev_pending_block.pending_buy_weight_pct, 5.0)
        self.assertIn("在途BUY挂单", " ".join(ev_pending_block.reasons))

        # Pending STOP order recognized during REVIEW_STOP
        pos_inv_stop = pos_msft.model_copy(update={"account": "Invest", "return_pct": -16.0, "weight_pct": 8.0})
        stop_order = PendingOrderItem(
            id="ord-2",
            account="Invest",
            raw_ticker="MSFT_US_EQ",
            ticker="MSFT",
            yahoo_symbol="MSFT",
            name="Microsoft",
            side="SELL",
            order_type="STOP",
            quantity=10.0,
            value=0.0,
            stop_price=335.0,
            currency="USD",
        )
        ev_with_stop = evaluate_holding(pos_inv_stop, tech_pb, None, strategy, pending_orders=[stop_order])
        self.assertEqual(ev_with_stop.action_bias, "REVIEW_STOP")
        self.assertTrue(ev_with_stop.has_pending_stop)
        self.assertIn("已有在途卖出/止损挂单保护@335.00", " ".join(ev_with_stop.reasons))

        # v5.3.2: ISA long-term holdings are no longer vetoed by the 4.0% cash hurdle;
        # the same thin-upside setup on Invest must be blocked by the R:R < 2:1 gate instead.
        tech_borderline = TechnicalSnapshot(
            symbol="MSFT",
            last_close=400.0,
            dist_ma50_pct=-1.0,
            dist_ma200_pct=4.0,
            trend_regime="PULLBACK",
        )
        bundle_low_upside = TickerMarketBundle(
            symbol="MSFT",
            current_price=400.0,
            target_mean_price=410.0,  # +2.5% far below a 2:1 reward vs a 12% stop
            analyst_count=12,
        )
        ev_low_upside_isa = evaluate_holding(
            pos_msft.model_copy(update={"weight_pct": 10.0}), tech_borderline, bundle_low_upside, strategy
        )
        self.assertNotIn("现金+摩擦门槛", " ".join(ev_low_upside_isa.reasons))
        self.assertNotIn("风险回报比", " ".join(ev_low_upside_isa.reasons))

        pos_invest_thin = pos_msft.model_copy(update={"account": "Invest", "weight_pct": 10.0, "return_pct": 1.0})
        ev_low_upside_inv = evaluate_holding(pos_invest_thin, tech_borderline, bundle_low_upside, strategy)
        self.assertNotEqual(ev_low_upside_inv.action_bias, "ADD_CANDIDATE")
        self.assertIn("风险回报比仅", " ".join(ev_low_upside_inv.reasons))

        # Crowded CYQ profit + RSI>=70 + volume spike triggers TRIM_ALERT
        tech_crowded = TechnicalSnapshot(
            symbol="NVDA",
            last_close=150.0,
            rsi14=76.0,
            cyq_benefit_part_pct=95.0,
            cyq_regime="获利盘高企",
            volume_ratio_20d=2.1,
            ret_1d_pct=-2.5,
            trend_regime="BULL",
        )
        ev_crowded = evaluate_holding(pos_msft.model_copy(update={"weight_pct": 10.0}), tech_crowded, None, strategy)
        self.assertEqual(ev_crowded.action_bias, "TRIM_ALERT")
        self.assertIn("CYQ获利盘达95%", " ".join(ev_crowded.reasons))

    def test_imp07_disagreement_matrix_and_risk_override_guardrail(self) -> None:
        from src.agents.schemas import (
            MorningReportDecision,
            PickActionItem,
            PickDecision,
            ReportActionItem,
        )
        from src.data_providers.analytics import TechnicalSnapshot, normalize_issuer_root
        from src.data_providers.market_data import MacroSnapshot
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData
        from src.engine.portfolio_rules import (
            apply_risk_override_guardrail,
            build_disagreement_matrix,
            format_disagreement_matrix_text,
        )

        strategy = StrategySection()
        pos_nvda = PositionItem(
            account="ISA",
            raw_ticker="NVDA_US_EQ",
            ticker="NVDA",
            yahoo_symbol="NVDA",
            name="NVIDIA",
            inst_currency="USD",
            quantity=10,
            avg_price=100.0,
            cur_price=140.0,
            current_value=5000.0,
            total_cost=3500.0,
            pnl=1500.0,
            fx_impact=0.0,
            return_pct=42.8,
            weight_pct=25.0,
        )
        pos_calm = pos_nvda.model_copy(
            update={"ticker": "VWRP", "yahoo_symbol": "VWRP.L", "inst_currency": "GBP", "current_value": 4000.0, "weight_pct": 20.0}
        )
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-14 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(ok=True, account_label="ISA", total_account_value=20000.0, positions=[pos_nvda, pos_calm]),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=20000.0,
        )
        tech_map = {
            "NVDA": TechnicalSnapshot(
                symbol="NVDA",
                last_close=140.0,
                rsi14=78.0,
                dist_ma50_pct=20.0,
                cyq_benefit_part_pct=94.0,
                weekly_bullish=True,
                trend_regime="BULL",
            ),
            "VWRP.L": TechnicalSnapshot(symbol="VWRP.L", last_close=100.0, rsi14=55.0, trend_regime="BULL"),
        }
        bundle_map = {
            "NVDA": TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", current_price=140.0, target_mean_price=165.0, analyst_count=20),
            "VWRP.L": TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP"),
        }

        disagreements = build_disagreement_matrix(portfolio, tech_map, bundle_map, strategy)
        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0].yahoo_symbol, "NVDA")
        self.assertEqual(disagreements[0].conflict_type, "BULL_VS_OVERWEIGHT")
        txt = format_disagreement_matrix_text(disagreements)
        self.assertIn("NVDA", txt)
        self.assertIn("⚔️", txt)
        self.assertNotIn("**", txt)

        # Collapsing low-signal SUPPORT_VS_WEAK_TREND rows while keeping high-priority rows expanded
        from src.engine.portfolio_rules import DisagreementItem

        mixed_txt = format_disagreement_matrix_text(
            [
                *disagreements,
                DisagreementItem(
                    account="ISA",
                    ticker="UNH",
                    yahoo_symbol="UNH",
                    conflict_type="SUPPORT_VS_WEAK_TREND",
                    bull_points=["一致目标空间+18%"],
                    bear_points=["日/周线趋势偏弱"],
                    decision_hint="估值/超卖有修复空间但右侧尚未转强，等待均线企稳",
                ),
                DisagreementItem(
                    account="Invest",
                    ticker="UBER",
                    yahoo_symbol="UBER",
                    conflict_type="SUPPORT_VS_WEAK_TREND",
                    bull_points=["RSI14=28.0超卖"],
                    bear_points=["日/周线趋势偏弱"],
                    decision_hint="估值/超卖有修复空间但右侧尚未转强，等待均线企稳",
                ),
            ]
        )
        self.assertIn("NVDA (BULL_VS_OVERWEIGHT)", mixed_txt)
        self.assertIn("[SUPPORT_VS_WEAK_TREND 合并 x2] UNH[ISA], UBER[Invest]", mixed_txt)
        self.assertEqual(mixed_txt.count("SUPPORT_VS_WEAK_TREND"), 1)

        # MorningReportDecision guardrail: NVDA ADD downgraded by overweight, VWRP.L ADD downgraded by VIX>=28,
        # while TRIM/HOLD remain untouched and caller's original object is not mutated.
        report_dec = MorningReportDecision(
            headline="测试早报",
            top_actions=[
                ReportActionItem(account="ISA", symbol="NVDA", action="ADD", price_zone="138", reason="趋势强"),
                ReportActionItem(account="ISA", symbol="VWRP.L", action="ADD", price_zone="100", reason="定投"),
                ReportActionItem(account="Invest", symbol="AMD", action="TRIM", price_zone="150", reason="兑现"),
            ],
            daily_vetoes=["严禁追高超重股"],
            macro_and_cash="VIX高企",
            account_diagnosis="集中度偏高",
            hold_summary="其余持有",
            trade_review="无",
        )
        macro_vix = MacroSnapshot(fetched_at_utc="2026-11-14T07:25:00Z", vix=30.5)
        res_report = apply_risk_override_guardrail(
            report_dec,
            portfolio=portfolio,
            macro=macro_vix,
            strategy=strategy,
            bundle_map=bundle_map,
        )
        self.assertTrue(res_report.applied)
        self.assertEqual(len(res_report.overrides), 2)
        self.assertEqual(report_dec.top_actions[0].action, "ADD")  # input preserved
        self.assertEqual(res_report.decision.top_actions[0].action, "HOLD")
        self.assertIn("[🛡️风控护栏修正]", res_report.decision.top_actions[0].reason)
        self.assertEqual(res_report.decision.top_actions[1].action, "HOLD")
        self.assertEqual(res_report.decision.top_actions[2].action, "TRIM")

        # PickDecision guardrail: blocks daily_vetoes, high_correlation, and VIX>=28
        pick_dec = PickDecision(
            headline="精选测试",
            selected_actions=[
                PickActionItem(account="ISA", symbol="TSM", action="BUY", thesis="AI晶圆", entry_zone="180", stop_condition="165", target_window="210", entry_price=180.0),
                PickActionItem(account="Invest", symbol="META", action="ADD", thesis="广告", entry_zone="550", stop_condition="510", target_window="620", entry_price=550.0),
                PickActionItem(account="Invest", symbol="INTC", action="TRIM", thesis="减仓", entry_zone="22", stop_condition="N/A", target_window="N/A", entry_price=22.0),
            ],
            rejected_or_frozen=[],
            consistency_and_cash_note="现金充足",
        )
        res_pick = apply_risk_override_guardrail(
            pick_dec,
            portfolio=portfolio,
            macro=MacroSnapshot(fetched_at_utc="2026-11-14T07:25:00Z", vix=18.0),
            strategy=strategy,
            bundle_map=bundle_map,
            high_correlation_symbols={"META"},
            daily_vetoes=["禁止买入TSM等高波动半导体"],
        )
        self.assertTrue(res_pick.applied)
        self.assertEqual([a.symbol for a in res_pick.decision.selected_actions], ["INTC"])
        self.assertEqual(len(res_pick.decision.rejected_or_frozen), 2)
        self.assertIn("[🛡️风控护栏修正]", res_pick.decision.rejected_or_frozen[0])

        # Legacy FB -> META issuer alias & same-day /report TRIM/STOP hard reversal blocking
        self.assertEqual(normalize_issuer_root("FB"), "META")
        rev_pick = PickDecision(
            headline="逆势加仓测试",
            selected_actions=[
                PickActionItem(account="ISA", symbol="NVDA", action="BUY", thesis="抄底", entry_zone="165", stop_condition="155", target_window="185", entry_price=165.0),
                PickActionItem(account="Invest", symbol="INTC", action="ADD", thesis="博弈", entry_zone="56", stop_condition="52", target_window="65", entry_price=56.0),
            ],
            rejected_or_frozen=[],
            consistency_and_cash_note="现金充足",
        )
        res_rev = apply_risk_override_guardrail(
            rev_pick,
            portfolio=portfolio,
            macro=MacroSnapshot(fetched_at_utc="2026-11-14T07:25:00Z", vix=18.0),
            strategy=strategy,
            bundle_map=bundle_map,
            report_actions=[
                {"account": "ISA", "symbol": "NVDA", "action": "TRIM", "reason": "财报前减仓"},
                {"account": "双账户", "symbol": "INTC", "action": "STOP", "reason": "跌破年线止损"},
            ],
        )
        self.assertTrue(res_rev.applied)
        self.assertEqual(res_rev.decision.selected_actions, [])
        self.assertEqual([ov.trigger_type for ov in res_rev.overrides], ["REPORT_ACTION_CONFLICT", "REPORT_ACTION_CONFLICT"])

    def test_imp08_ref_breakout_patterns_and_volume_enter(self) -> None:
        # 1) Platform breakthrough: 90 bars flat around 100 (so MA60 ~ 100), then volume breakout bar
        n_plat = 100
        dates_plat = pd.date_range("2025-10-01", periods=n_plat, freq="B")
        opens_p = np.full(n_plat, 100.0)
        closes_p = np.array([99.5 + (i % 3) * 0.4 for i in range(n_plat)], dtype=float)
        highs_p = closes_p + 0.8
        lows_p = closes_p - 0.8
        vols_p = np.full(n_plat, 1_000_000.0)
        # Breakthrough at index 88: open below MA60, close >= MA60 with +3.5% gain and 2.8x volume
        opens_p[88] = 99.0
        closes_p[88] = 103.5
        highs_p[88] = 104.0
        lows_p[88] = 98.8
        vols_p[88] = 2_800_000.0
        for j in range(89, n_plat):
            opens_p[j] = 103.2
            closes_p[j] = 103.6 + (j - 89) * 0.1
            highs_p[j] = closes_p[j] + 0.6
            lows_p[j] = closes_p[j] - 0.6
        df_plat = pd.DataFrame(
            {"Open": opens_p, "High": highs_p, "Low": lows_p, "Close": closes_p, "Volume": vols_p},
            index=dates_plat,
        )
        self.assertTrue(detect_breakthrough_platform(df_plat))

        # No-volume breakout should fail platform check
        df_plat_novol = df_plat.copy()
        df_plat_novol.iloc[88, df_plat_novol.columns.get_loc("Volume")] = 1_000_000.0
        self.assertFalse(detect_breakthrough_platform(df_plat_novol))

        # 2) Low-ATR volatility contraction: volatile uptrend then tight 12-day coil above MA50
        n_atr = 140
        dates_atr = pd.date_range("2025-09-01", periods=n_atr, freq="B")
        rng = np.random.default_rng(7)
        base_c = 100.0 + np.linspace(0, 22, n_atr) + np.sin(np.arange(n_atr) / 3.0) * 1.5
        # Make last 14 bars very tight coiling near 123
        base_c[-14:] = 123.0 + np.linspace(-0.2, 0.25, 14)
        spread = np.full(n_atr, 3.2)
        spread[-14:] = 0.35
        df_atr = pd.DataFrame(
            {
                "Open": base_c - spread * 0.2,
                "High": base_c + spread,
                "Low": base_c - spread,
                "Close": base_c,
                "Volume": np.full(n_atr, 1_200_000.0),
            },
            index=dates_atr,
        )
        tech_atr = compute_technical_snapshot("LOWATR", df_atr)
        assert tech_atr is not None
        self.assertTrue(detect_low_atr_contraction(df_atr, tech_atr))

        # 3) enter.py volume MA breakout: dips below MA20 then surges +3.5% on 2.5x volume above MA20
        n_ent = 65
        dates_ent = pd.date_range("2025-11-01", periods=n_ent, freq="B")
        c_ent = np.linspace(100.0, 112.0, n_ent)
        c_ent[-6:-1] = [109.5, 108.8, 108.2, 107.8, 107.6]
        c_ent[-1] = 111.8
        o_ent = c_ent.copy() - 0.3
        o_ent[-1] = 108.0
        v_ent = np.full(n_ent, 1_000_000.0)
        v_ent[-1] = 2_600_000.0
        df_ent = pd.DataFrame(
            {
                "Open": o_ent,
                "High": np.maximum(o_ent, c_ent) + 0.4,
                "Low": np.minimum(o_ent, c_ent) - 0.4,
                "Close": c_ent,
                "Volume": v_ent,
            },
            index=dates_ent,
        )
        tech_ent = compute_technical_snapshot("ENTER", df_ent)
        assert tech_ent is not None
        self.assertTrue(detect_volume_ma_breakout(df_ent, tech_ent))

    def test_imp08_four_dimensional_scoring_cyq_fundamentals_and_etf_config(self) -> None:
        strategy = StrategySection()
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        closes = np.linspace(100.0, 130.0, 220)
        df = pd.DataFrame(
            {
                "Open": closes * 0.996,
                "High": closes * 1.01,
                "Low": closes * 0.99,
                "Close": closes,
                "Volume": np.full(220, 1_500_000.0),
            },
            index=dates,
        )
        tech_base = compute_technical_snapshot("AZN.L", df)
        assert tech_base is not None

        # Hardcoded ETF list removed: without quote_type="ETF" or configured etf_symbols, VWRP.L is EQUITY
        cand_no_etf = score_candidate(
            "VWRP.L", "ISA_LONG_TERM", tech_base, None, 0.35, "NVDA", False, False, strategy
        )
        self.assertEqual(cand_no_etf.asset_type, "EQUITY")
        cand_cfg_etf = score_candidate(
            "CUSTOM.L",
            "ISA_LONG_TERM",
            tech_base,
            None,
            0.35,
            "NVDA",
            False,
            False,
            strategy,
            etf_symbols={"CUSTOM.L"},
        )
        self.assertEqual(cand_cfg_etf.asset_type, "ETF")
        self.assertGreater(cand_cfg_etf.score, cand_no_etf.score)

        # Fundamental valuation + GBp/GBP 100x target guard + CYQ bottom single peak + weekly resonance
        tech_rich = tech_base.model_copy(
            update={
                "cyq_regime": "底部单峰密集",
                "cyq_benefit_part_pct": 62.0,
                "cyq_concentration_90_pct": 9.5,
                "weekly_bullish": True,
                "bb_squeeze": True,
                "volume_ratio_20d": 1.8,
                "ret_1d_pct": 1.4,
            }
        )
        # UK stock priced in GBp (10000p) while analyst target is given in GBP (125) -> 100x guard normalizes to +25%
        good_bundle = TickerMarketBundle(
            symbol="AZN.L",
            quote_type="EQUITY",
            currency="GBp",
            short_name="AstraZeneca",
            current_price=10000.0,
            forward_pe=17.5,
            dividend_yield_pct=3.2,
            target_mean_price=125.0,
            analyst_count=12,
        )
        weak_bundle = TickerMarketBundle(
            symbol="EXPENSIVE.L",
            quote_type="EQUITY",
            currency="GBp",
            short_name="Expensive",
            current_price=100.0,
            forward_pe=68.0,
            dividend_yield_pct=0.0,
            target_mean_price=80.0,
            analyst_count=10,
        )
        tech_weak = tech_base.model_copy(
            update={"cyq_regime": "上方套牢峰", "weekly_bullish": False, "bb_squeeze": False}
        )
        c_good = score_candidate(
            "AZN.L", "ISA_LONG_TERM", tech_rich, good_bundle, 0.30, "NVDA", False, False, strategy
        )
        c_bad = score_candidate(
            "EXPENSIVE.L", "ISA_LONG_TERM", tech_weak, weak_bundle, 0.30, "NVDA", False, False, strategy
        )
        self.assertIn("目标空间 +25%", c_good.summary_line)
        self.assertNotIn("目标价空间+25%", c_good.summary_line)
        self.assertNotIn("估值合理(PE=", c_good.summary_line)
        self.assertNotIn("股息率", c_good.summary_line)
        self.assertIn("CYQ底部单峰密集", c_good.reasons)
        self.assertIn("周/日线多头共振", c_good.reasons)
        self.assertGreater(c_good.score - c_bad.score, 35.0)

    def test_imp08_two_stage_run_quant_screener_benchmarks_and_recent_trades(self) -> None:
        import asyncio
        import time
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        dates = pd.date_range("2025-09-01", periods=220, freq="B")

        def _make_ohlcv(seed: int, slope: float = 0.15) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            rets = rng.normal(0.0005, 0.012, 220)
            c = 100.0 * np.cumprod(1.0 + rets) + np.linspace(0.0, slope * 220, 220)
            return pd.DataFrame(
                {
                    "Open": c * 0.996,
                    "High": c * 1.01,
                    "Low": c * 0.99,
                    "Close": c,
                    "Volume": np.full(220, 1_500_000.0),
                },
                index=dates,
            )

        class _SpyMarketData(MarketDataService):
            def __init__(self) -> None:
                super().__init__()
                self.hist_calls: list[list[str]] = []
                self.bundle_calls: list[dict] = []

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                self.hist_calls.append(list(symbols))
                return {s: _make_ohlcv(abs(hash(s)) % 1000) for s in symbols}

            async def get_bundles(
                self,
                symbols: list[str],
                include_news: bool = True,
                include_earnings: bool = True,
                include_info: bool = True,
                isin_map: dict[str, str] | None = None,
            ):
                self.bundle_calls.append(
                    {
                        "symbols": list(symbols),
                        "include_news": include_news,
                        "include_earnings": include_earnings,
                        "include_info": include_info,
                        "isin_map": dict(isin_map or {}),
                    }
                )
                hists = {s: _make_ohlcv(abs(hash(s)) % 1000) for s in symbols}
                bundles = {
                    s: TickerMarketBundle(
                        symbol=s,
                        quote_type="ETF" if s == "CUSTOMETF.L" else "EQUITY",
                        currency="GBP" if s.endswith(".L") else "USD",
                        short_name=s,
                        isin=(isin_map or {}).get(s, ""),
                        forward_pe=18.0,
                        dividend_yield_pct=3.0 if s.endswith(".L") else 1.0,
                        target_mean_price=float(hists[s]["Close"].iloc[-1]) * 1.25,
                        analyst_count=8,
                        days_to_ex_dividend=-1 if s == "AZN.L" else None,
                        has_history=True,
                    )
                    for s in symbols
                }
                return hists, bundles

        now_ms = int(time.time() * 1000)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 10:00",
            fetched_at_ms=now_ms,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        inst_currency="USD",
                        quantity=10,
                        avg_price=120,
                        cur_price=140,
                        current_value=1100,
                        total_cost=950,
                        pnl=150,
                        fx_impact=0,
                        return_pct=15.8,
                        weight_pct=50.0,
                        isin="US67066G1040",
                    ),
                    PositionItem(
                        account="ISA",
                        raw_ticker="AZN_EQ",
                        ticker="AZN",
                        yahoo_symbol="AZN.L",
                        name="AstraZeneca",
                        inst_currency="GBp",
                        quantity=10,
                        avg_price=10000,
                        cur_price=10500,
                        current_value=1050,
                        total_cost=1000,
                        pnl=50,
                        fx_impact=0,
                        return_pct=5.0,
                        weight_pct=50.0,
                        isin=None,
                    ),
                ],
                recent_orders=[
                    HistoricalOrderItem(
                        id="recent-1",
                        account="ISA",
                        raw_ticker="SNOW_US_EQ",
                        ticker="SNOW",
                        yahoo_symbol="SNOW",
                        name="Snowflake",
                        inst_currency="USD",
                        side="BUY",
                        order_type="MARKET",
                        filled_at="2026-11-01T15:00:00Z",
                        filled_at_ms=now_ms - 14 * 86400 * 1000,
                        uk_time_str="2026-11-01 15:00",
                        quantity=2,
                        price=160.0,
                        net_value=250.0,
                        wallet_currency="GBP",
                        realised_pnl=0.0,
                        tax_total=0.0,
                    ),
                    HistoricalOrderItem(
                        id="stale-95d",
                        account="Invest",
                        raw_ticker="OLD_EQ",
                        ticker="OLD",
                        yahoo_symbol="OLD.L",
                        name="Old Stock",
                        inst_currency="GBp",
                        side="SELL",
                        order_type="MARKET",
                        filled_at="2026-07-01T15:00:00Z",
                        filled_at_ms=now_ms - 120 * 86400 * 1000,
                        uk_time_str="2026-07-01 15:00",
                        quantity=2,
                        price=100.0,
                        net_value=200.0,
                        wallet_currency="GBP",
                        realised_pnl=10.0,
                        tax_total=0.0,
                    ),
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )
        universe = UniverseSection(
            isa_long_term=[
                "CUSTOMETF.L", "SAP.DE", "REL.L", "ULVR.L", "GSK.L", "HSBA.L",
                "MSFT", "AAPL", "GOOGL", "AMZN", "V", "COST",
            ],
            invest_tactical=["AMD", "META", "CRM", "UBER", "MU", "PANW", "NFLX", "LLY", "JPM"],
            etf_symbols=["CUSTOMETF.L"],
        )
        spy_md = _SpyMarketData()
        existing = {"NVDA": _make_ohlcv(101)}

        isa_res, inv_res = asyncio.run(
            run_quant_screener(
                portfolio=portfolio,
                market_data=spy_md,
                universe=universe,
                watchlist=["TSM"],
                strategy=StrategySection(),
                existing_histories=existing,
                top_n_per_track=2,
            )
        )
        self.assertEqual(len(isa_res), 2)
        self.assertEqual(len(inv_res), 2)
        # Stage 1 fetched histories in a single batch, skipping preloaded NVDA and adding regional benchmarks
        self.assertEqual(len(spy_md.hist_calls), 1)
        fetched_syms = set(spy_md.hist_calls[0])
        self.assertNotIn("NVDA", fetched_syms)
        self.assertIn("SPY", fetched_syms)
        self.assertIn("ISF.L", fetched_syms)
        self.assertIn("EXS1.DE", fetched_syms)
        self.assertIn("SNOW", fetched_syms)
        self.assertNotIn("OLD.L", fetched_syms)

        # Stage 2 called get_bundles ONLY for the shortlisted candidates (< total universe)
        self.assertEqual(len(spy_md.bundle_calls), 1)
        b_call = spy_md.bundle_calls[0]
        self.assertFalse(b_call["include_news"])
        self.assertTrue(b_call["include_earnings"])
        self.assertTrue(b_call["include_info"])
        self.assertEqual(b_call["isin_map"], {"NVDA": "US67066G1040"})
        self.assertLess(len(b_call["symbols"]), len(fetched_syms))

    def test_imp08_v2_minervini_vcp_rs_and_phase4(self) -> None:
        strategy = StrategySection()
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        # 220-bar Stage 2 uptrend whose last 120 bars form 3 progressively tighter VCP contractions with drying volume
        base = np.linspace(80.0, 130.0, 100)
        seg1 = 130.0 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 12.0   # ~17% swing
        seg2 = 132.0 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 5.5    # ~8% swing
        seg3 = 133.5 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 2.0    # ~3% swing
        closes_vcp = np.concatenate([base, seg1, seg2, seg3])
        closes_vcp[-5:] = np.linspace(134.0, 136.5, 5)
        vols_vcp = np.full(220, 2_000_000.0)
        vols_vcp[-12:] = 850_000.0  # clear volume dry-up in final contraction

        df_vcp = pd.DataFrame(
            {
                "Open": closes_vcp * 0.997,
                "High": closes_vcp * 1.012,
                "Low": closes_vcp * 0.988,
                "Close": closes_vcp,
                "Volume": vols_vcp,
            },
            index=dates,
        )
        # Benchmark lagging noticeably in recent 60 days
        bench_closes = np.linspace(100.0, 118.0, 160).tolist() + np.linspace(118.0, 114.0, 60).tolist()
        df_bench = pd.DataFrame(
            {
                "Open": np.array(bench_closes) * 0.998,
                "High": np.array(bench_closes) * 1.005,
                "Low": np.array(bench_closes) * 0.995,
                "Close": bench_closes,
                "Volume": np.full(220, 5_000_000.0),
            },
            index=dates,
        )
        tech_vcp = compute_technical_snapshot("AVGO", df_vcp)
        assert tech_vcp is not None
        rs = compute_relative_strength(df_vcp, df_bench)
        self.assertIsNotNone(rs["rs_20d_pct"])
        self.assertGreater(rs["rs_20d_pct"], 2.0)
        self.assertIsNotNone(rs["rs_slope_20d"])
        self.assertGreater(rs["rs_slope_20d"], 0.0)

        m_res = detect_minervini_stage2(df_vcp, tech_vcp, rs_60d_pct=rs["rs_60d_pct"])
        self.assertEqual(m_res["phase"], 2)
        self.assertTrue(m_res["passes"])
        self.assertTrue(detect_vcp_contraction(df_vcp, tech_vcp))

        cand_vcp = score_candidate(
            "AVGO",
            "INVEST_TACTICAL",
            tech_vcp,
            None,
            0.35,
            "NVDA",
            False,
            False,
            strategy,
            hist=df_vcp,
            benchmark_hist=df_bench,
        )
        self.assertIn("VCP收缩", cand_vcp.pattern_signals)
        self.assertIn("Minervini阶段2", cand_vcp.pattern_signals)
        self.assertLessEqual(cand_vcp.dimension_scores["trend_rs"], 30.0)
        self.assertLessEqual(cand_vcp.dimension_scores["pattern_timing"], 25.0)
        self.assertLessEqual(cand_vcp.dimension_scores["value_quality"], 25.0)
        self.assertLessEqual(cand_vcp.dimension_scores["portfolio_fit"], 20.0)

        # Phase 4 downtrend retains visibility with penalty + explicit warning tag (Option 2A)
        down_closes = np.linspace(150.0, 90.0, 220)
        df_down = pd.DataFrame(
            {
                "Open": down_closes * 1.002,
                "High": down_closes * 1.01,
                "Low": down_closes * 0.99,
                "Close": down_closes,
                "Volume": np.full(220, 1_000_000.0),
            },
            index=dates,
        )
        tech_down = compute_technical_snapshot("FALL.L", df_down)
        assert tech_down is not None
        m_down = detect_minervini_stage2(df_down, tech_down)
        self.assertEqual(m_down["phase"], 4)
        self.assertFalse(m_down["passes"])
        cand_down = score_candidate(
            "FALL.L",
            "ISA_LONG_TERM",
            tech_down,
            None,
            0.30,
            "NVDA",
            False,
            False,
            strategy,
            hist=df_down,
        )
        self.assertGreater(cand_down.score, 0.0)
        self.assertIn("[Phase4弱势]", cand_down.reasons)
        self.assertLess(cand_down.score, cand_vcp.score)

    def test_imp08_v2_linear_quality_and_portfolio_marginal_risk(self) -> None:
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        strategy = StrategySection()
        dates = pd.date_range("2025-10-01", periods=100, freq="B")
        rng = np.random.default_rng(42)
        rets_a = rng.normal(0.001, 0.02, 100)
        rets_hedge = -rets_a * 0.8 + rng.normal(0.0005, 0.005, 100)  # negatively correlated
        rets_clone = rets_a * 0.95 + rng.normal(0.0, 0.004, 100)       # highly redundant

        def _df_from_rets(r: np.ndarray) -> pd.DataFrame:
            c = 100.0 * np.cumprod(1.0 + r)
            return pd.DataFrame(
                {"Open": c * 0.998, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(100, 1e6)},
                index=dates,
            )

        hist_a = _df_from_rets(rets_a)
        hist_hedge = _df_from_rets(rets_hedge)
        hist_clone = _df_from_rets(rets_clone)
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 10:00",
            fetched_at_ms=1794736800000,
            isa=AccountSnapshot(
                ok=True,
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="HELD_US_EQ",
                        ticker="HELD",
                        yahoo_symbol="HELD",
                        name="Held",
                        inst_currency="USD",
                        quantity=10,
                        avg_price=100,
                        cur_price=110,
                        current_value=5000.0,
                        total_cost=4500.0,
                        pnl=500.0,
                        fx_impact=0.0,
                        return_pct=11.1,
                        weight_pct=100.0,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True),
        )
        risk = compute_portfolio_risk_metrics(
            {"HELD": hist_a, "HEDGE": hist_hedge, "CLONE": hist_clone},
            portfolio,
            ["HEDGE", "CLONE"],
            lookback_days=60,
        )
        self.assertIn("HEDGE", risk)
        self.assertIn("CLONE", risk)
        self.assertLess(risk["HEDGE"]["weighted_corr"], 0.0)
        self.assertGreater(risk["CLONE"]["weighted_corr"], 0.7)
        self.assertLess(risk["HEDGE"]["marginal_vol_delta_pct"], risk["CLONE"]["marginal_vol_delta_pct"])

        # Quality factor linear differentiation (ROE, margins, revenue growth, debt/equity)
        tech = compute_technical_snapshot("HEDGE", hist_hedge)
        assert tech is not None
        high_q = TickerMarketBundle(
            symbol="HEDGE",
            quote_type="EQUITY",
            currency="USD",
            forward_pe=19.0,
            roe_pct=28.0,
            profit_margin_pct=24.0,
            revenue_growth_pct=22.0,
            debt_to_equity=25.0,
        )
        low_q = TickerMarketBundle(
            symbol="CLONE",
            quote_type="EQUITY",
            currency="USD",
            forward_pe=48.0,
            roe_pct=4.0,
            profit_margin_pct=1.0,
            revenue_growth_pct=-2.0,
            debt_to_equity=250.0,
        )
        c_hq = score_candidate(
            "HEDGE",
            "ISA_LONG_TERM",
            tech,
            high_q,
            0.10,
            "HELD",
            False,
            False,
            strategy,
            weighted_corr=risk["HEDGE"]["weighted_corr"],
            marginal_vol_delta_pct=risk["HEDGE"]["marginal_vol_delta_pct"],
        )
        c_lq = score_candidate(
            "CLONE",
            "ISA_LONG_TERM",
            tech,
            low_q,
            0.85,
            "HELD",
            False,
            False,
            strategy,
            weighted_corr=risk["CLONE"]["weighted_corr"],
            marginal_vol_delta_pct=risk["CLONE"]["marginal_vol_delta_pct"],
        )
        self.assertGreater(c_hq.dimension_scores["value_quality"], c_lq.dimension_scores["value_quality"])
        self.assertGreater(c_hq.dimension_scores["portfolio_fit"], c_lq.dimension_scores["portfolio_fit"])

    def test_imp08_v2_market_breadth_and_multibucket_shortlist(self) -> None:
        import asyncio
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        dates = pd.date_range("2025-10-01", periods=210, freq="B")

        def _trend_df(step: float, pullback: bool = False) -> pd.DataFrame:
            c = np.linspace(100.0, 100.0 + step * 210, 210)
            if pullback:
                c[-8:] = np.linspace(c[-9], c[-9] * 0.94, 8)
            return pd.DataFrame(
                {"Open": c * 0.998, "High": c * 1.008, "Low": c * 0.992, "Close": c, "Volume": np.full(210, 1e6)},
                index=dates,
            )

        snaps = {f"B{i}": compute_technical_snapshot(f"B{i}", _trend_df(0.25)) for i in range(6)}
        breadth_on = compute_market_breadth(snaps, snaps.keys())
        self.assertEqual(breadth_on["regime_tag"], "RISK-ON")

        weak_snaps = {f"W{i}": compute_technical_snapshot(f"W{i}", _trend_df(-0.2)) for i in range(6)}
        breadth_off = compute_market_breadth(weak_snaps, weak_snaps.keys())
        self.assertEqual(breadth_off["regime_tag"], "RISK-OFF")

        # Build a large ISA pool (> preselect_n=8 for top_n_per_track=2) where WATCH_PB sits in Bucket 2/3
        mom_syms = [f"MOM{i}" for i in range(14)]
        hist_map = {s: _trend_df(0.28 + i * 0.01) for i, s in enumerate(mom_syms)}
        hist_map["WATCH_PB"] = _trend_df(0.18, pullback=True)
        hist_map["SPY"] = _trend_df(0.10)
        hist_map["ISF.L"] = _trend_df(0.08)

        class _BucketMarketData(MarketDataService):
            def __init__(self) -> None:
                super().__init__()
                self.bundle_syms: list[str] = []

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                return {s: hist_map[s] for s in symbols if s in hist_map}

            async def get_bundles(self, symbols: list[str], **_kwargs):
                self.bundle_syms = list(symbols)
                b = {
                    s: TickerMarketBundle(
                        symbol=s,
                        quote_type="EQUITY",
                        currency="USD",
                        short_name=s,
                        forward_pe=18.0,
                        roe_pct=22.0,
                    )
                    for s in symbols
                }
                return {s: hist_map[s] for s in symbols if s in hist_map}, b

        md = _BucketMarketData()
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 10:00",
            fetched_at_ms=1794736800000,
            isa=AccountSnapshot(ok=True),
            invest=AccountSnapshot(ok=True),
        )
        universe = UniverseSection(isa_long_term=mom_syms, invest_tactical=["MOM0", "MOM1"])
        isa_res, _ = asyncio.run(
            run_quant_screener(
                portfolio=portfolio,
                market_data=md,
                universe=universe,
                watchlist=["WATCH_PB"],
                strategy=StrategySection(),
                top_n_per_track=2,
            )
        )
        self.assertIn("WATCH_PB", md.bundle_syms)
        self.assertLess(len(md.bundle_syms), len(mom_syms) + 3)
        self.assertEqual(isa_res[0].regime_tag, "RISK-ON")
        self.assertIsNotNone(isa_res[0].rs_20d_pct)
        self.assertIn("trend_rs", isa_res[0].dimension_scores)

    def test_imp08_v3_vpvr_cross_tz_asymmetric_weights_and_defensive_bucket(self) -> None:
        import asyncio
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        strategy = StrategySection()
        dates = pd.date_range("2025-09-01", periods=220, freq="B")

        # 1) ETF VPVR POC support + exemption from CYQ >=92% profit penalty & Stage 2 exemption
        c_etf = np.concatenate([np.full(170, 100.0), np.linspace(100.0, 104.5, 50)])
        v_etf = np.concatenate([np.full(195, 3_000_000.0), np.full(25, 400_000.0)])  # dense POC around ~100
        df_etf = pd.DataFrame(
            {"Open": c_etf * 0.998, "High": c_etf * 1.005, "Low": c_etf * 0.995, "Close": c_etf, "Volume": v_etf},
            index=dates,
        )
        vpvr = compute_vpvr_profile(df_etf, window=60)
        assert vpvr is not None
        self.assertGreaterEqual(vpvr["poc_dist_pct"], 0.0)
        self.assertLessEqual(vpvr["poc_dist_pct"], 8.0)

        tech_base = compute_technical_snapshot("VWRP.L", df_etf)
        assert tech_base is not None
        tech_96pct = tech_base.model_copy(
            update={"cyq_regime": "获利盘高企", "cyq_benefit_part_pct": 96.0}
        )
        etf_bundle = TickerMarketBundle(
            symbol="VWRP.L", quote_type="ETF", currency="GBP", dividend_yield_pct=2.8
        )
        cand_etf = score_candidate(
            "VWRP.L",
            "ISA_LONG_TERM",
            tech_96pct,
            etf_bundle,
            0.25,
            "NVDA",
            False,
            False,
            strategy,
            hist=df_etf,
        )
        self.assertTrue(any("VPVR支撑单峰" in p for p in cand_etf.pros))
        self.assertFalse(any("获利盘高企" in c for c in cand_etf.cons))

        # Stage 2 equity with upward RS slope also exempt from >=92% CYQ penalty
        c_leader = np.linspace(80.0, 150.0, 220)
        df_leader = pd.DataFrame(
            {
                "Open": c_leader * 0.997,
                "High": c_leader * 1.01,
                "Low": c_leader * 0.99,
                "Close": c_leader,
                "Volume": np.full(220, 2_000_000.0),
            },
            index=dates,
        )
        df_bench_lag = pd.DataFrame(
            {
                "Open": np.full(220, 100.0),
                "High": np.full(220, 101.0),
                "Low": np.full(220, 99.0),
                "Close": np.linspace(100.0, 105.0, 220),
                "Volume": np.full(220, 4_000_000.0),
            },
            index=dates,
        )
        tech_leader = compute_technical_snapshot("NVDA", df_leader)
        assert tech_leader is not None
        tech_leader_96 = tech_leader.model_copy(
            update={"cyq_regime": "获利盘高企", "cyq_benefit_part_pct": 96.0}
        )
        cand_leader = score_candidate(
            "NVDA",
            "INVEST_TACTICAL",
            tech_leader_96,
            TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", currency="USD", forward_pe=35.0),
            0.35,
            "TSM",
            False,
            False,
            strategy,
            hist=df_leader,
            benchmark_hist=df_bench_lag,
        )
        self.assertFalse(any("获利盘高企" in c for c in cand_leader.cons))

        # 2) Cross-timezone 3-day overlapping covariance catches 1-day lagged UK/EU vs US co-movement
        rng = np.random.default_rng(7)
        us_rets = rng.normal(0.001, 0.02, 100)
        uk_lag_rets = np.roll(us_rets, 1) * 0.95 + rng.normal(0.0, 0.002, 100)
        dates_tz = pd.date_range("2025-10-01", periods=100, freq="B")
        c_us = 100.0 * np.cumprod(1.0 + us_rets)
        c_uk = 100.0 * np.cumprod(1.0 + uk_lag_rets)
        h_us = pd.DataFrame(
            {"Open": c_us, "High": c_us * 1.01, "Low": c_us * 0.99, "Close": c_us, "Volume": np.full(100, 1e6)},
            index=dates_tz,
        )
        h_uk = pd.DataFrame(
            {"Open": c_uk, "High": c_uk * 1.01, "Low": c_uk * 0.99, "Close": c_uk, "Volume": np.full(100, 1e6)},
            index=dates_tz,
        )
        port_us = PortfolioData(
            fetched_at_uk="2026-11-15 10:00",
            fetched_at_ms=1794736800000,
            isa=AccountSnapshot(
                ok=True,
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        inst_currency="USD",
                        quantity=10,
                        avg_price=100,
                        cur_price=110,
                        current_value=5000.0,
                        total_cost=4500.0,
                        pnl=500.0,
                        fx_impact=0.0,
                        return_pct=11.1,
                        weight_pct=100.0,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True),
        )
        tz_risk = compute_portfolio_risk_metrics(
            {"NVDA": h_us, "UKLAG.L": h_uk},
            port_us,
            ["UKLAG.L"],
            lookback_days=60,
        )
        self.assertGreater(tz_risk["UKLAG.L"]["weighted_corr"], 0.45)

        # 3) Asymmetric dual-track weights + structured attribution pros/cons/dimension_pct
        cand_isa_etf = score_candidate(
            "VWRP.L", "ISA_LONG_TERM", tech_base, etf_bundle, 0.25, "NVDA", False, False, strategy, hist=df_etf
        )
        cand_inv_etf = score_candidate(
            "VWRP.L", "INVEST_TACTICAL", tech_base, etf_bundle, 0.25, "NVDA", False, False, strategy, hist=df_etf
        )
        self.assertGreater(cand_isa_etf.score, cand_inv_etf.score)
        self.assertEqual(set(cand_isa_etf.dimension_pct), {"trend_rs", "pattern_timing", "value_quality", "portfolio_fit"})
        self.assertIn("优势:", cand_isa_etf.summary_line)

        # 4) Dual-anchored breadth pulls neutral pool down when benchmarks are in downtrend
        down_b = np.linspace(150.0, 95.0, 220)
        df_bear_bench = pd.DataFrame(
            {"Open": down_b, "High": down_b * 1.01, "Low": down_b * 0.99, "Close": down_b, "Volume": np.full(220, 1e6)},
            index=dates,
        )
        breadth_bear = compute_market_breadth(
            {"M1": tech_base, "M2": tech_base},
            ["M1", "M2"],
            histories={"SPY": df_bear_bench, "ISF.L": df_bear_bench},
        )
        self.assertEqual(breadth_bear["regime_tag"], "TRANSITIONAL")
        self.assertLess(breadth_bear["combined_score"], 60.0)

        # 5) Defensive bucket rescues a low-vol LSE ETF into Stage 2 enrichment despite 14 momentum equities
        mom_syms = [f"FAST{i}" for i in range(14)]
        hist_pool = {s: df_leader for s in mom_syms}
        hist_pool["SAFEETF.L"] = df_etf
        hist_pool["SPY"] = df_bench_lag
        hist_pool["ISF.L"] = df_bench_lag

        class _DefMarketData(MarketDataService):
            def __init__(self) -> None:
                super().__init__()
                self.stage2_syms: list[str] = []

            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                return {s: hist_pool[s] for s in symbols if s in hist_pool}

            async def get_bundles(self, symbols: list[str], **_kwargs):
                self.stage2_syms = list(symbols)
                b = {
                    s: TickerMarketBundle(
                        symbol=s,
                        quote_type="ETF" if s == "SAFEETF.L" else "EQUITY",
                        currency="GBP" if s.endswith(".L") else "USD",
                        short_name=s,
                    )
                    for s in symbols
                }
                return {s: hist_pool[s] for s in symbols if s in hist_pool}, b

        d_md = _DefMarketData()
        asyncio.run(
            run_quant_screener(
                portfolio=port_us,
                market_data=d_md,
                universe=UniverseSection(
                    isa_long_term=[*mom_syms, "SAFEETF.L"],
                    invest_tactical=["FAST0"],
                    etf_symbols=["SAFEETF.L"],
                ),
                watchlist=[],
                strategy=strategy,
                top_n_per_track=2,
            )
        )
        self.assertIn("SAFEETF.L", d_md.stage2_syms)

    def test_screener_mixed_tz_aware_and_naive_indices(self) -> None:
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        # Span UK BST -> GMT transition (Oct -> Nov) to verify local exchange date preservation
        dates_naive = pd.date_range("2026-07-01", periods=100, freq="B")
        dates_us = dates_naive.tz_localize("America/New_York")
        dates_uk = dates_naive.tz_localize("Europe/London")
        dates_eu = dates_naive.tz_localize("Europe/Berlin")

        rng = np.random.default_rng(19)
        base_rets = rng.normal(0.001, 0.015, 100)
        p_us = 100.0 * np.cumprod(1.0 + base_rets)
        p_uk = 200.0 * np.cumprod(1.0 + base_rets * 0.8 + rng.normal(0.0, 0.004, 100))
        p_vwrl = 90.0 * np.cumprod(1.0 + base_rets * 0.7 + rng.normal(0.0, 0.004, 100))
        p_msft = 300.0 * np.cumprod(1.0 + base_rets * 0.9 + rng.normal(0.0, 0.003, 100))
        p_sap = 180.0 * np.cumprod(1.0 + base_rets * 0.6 + rng.normal(0.0, 0.005, 100))

        def _ohlcv(arr: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
            return pd.DataFrame(
                {"Open": arr, "High": arr * 1.01, "Low": arr * 0.99, "Close": arr, "Volume": np.full(len(arr), 1e6)},
                index=idx,
            )

        histories = {
            "NVDA": _ohlcv(p_us, dates_us),
            "AZN.L": _ohlcv(p_uk, dates_uk),
            "VWRL.L": _ohlcv(p_vwrl, dates_naive),
            "MSFT": _ohlcv(p_msft, dates_naive),
            "SAP.DE": _ohlcv(p_sap, dates_eu),
            "SPY": _ohlcv(p_vwrl, dates_naive),
        }
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-15 21:00",
            fetched_at_ms=1794776400000,
            isa=AccountSnapshot(
                ok=True,
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        inst_currency="USD",
                        quantity=10,
                        avg_price=100,
                        cur_price=110,
                        current_value=3000.0,
                        total_cost=2700.0,
                        pnl=300.0,
                        fx_impact=0.0,
                        return_pct=11.1,
                        weight_pct=50.0,
                    ),
                    PositionItem(
                        account="ISA",
                        raw_ticker="AZN_EQ",
                        ticker="AZN",
                        yahoo_symbol="AZN.L",
                        name="AstraZeneca",
                        inst_currency="GBp",
                        quantity=20,
                        avg_price=10000,
                        cur_price=10500,
                        current_value=2000.0,
                        total_cost=1900.0,
                        pnl=100.0,
                        fx_impact=0.0,
                        return_pct=5.2,
                        weight_pct=33.3,
                    ),
                ],
            ),
            invest=AccountSnapshot(
                ok=True,
                positions=[
                    PositionItem(
                        account="INVEST",
                        raw_ticker="VWRL_EQ",
                        ticker="VWRL",
                        yahoo_symbol="VWRL.L",
                        name="Vanguard FTSE All-World",
                        inst_currency="GBP",
                        quantity=10,
                        avg_price=90,
                        cur_price=95,
                        current_value=1000.0,
                        total_cost=950.0,
                        pnl=50.0,
                        fx_impact=0.0,
                        return_pct=5.2,
                        weight_pct=100.0,
                    ),
                ],
            ),
        )

        risk = compute_portfolio_risk_metrics(histories, portfolio, ["MSFT", "SAP.DE"], lookback_days=60)
        self.assertIn("MSFT", risk)
        self.assertIn("SAP.DE", risk)
        self.assertIsNotNone(risk["MSFT"]["weighted_corr"])
        self.assertGreater(risk["MSFT"]["weighted_corr"], 0.7)
        self.assertIsNotNone(risk["SAP.DE"]["marginal_vol_delta_pct"])

        rs = compute_relative_strength(histories["NVDA"], histories["SPY"])
        self.assertIsNotNone(rs["rs_20d_pct"])
        self.assertIsNotNone(rs["rs_slope_20d"])

    def test_imp21_portfolio_cvar_risk_attribution_and_screener_deep_intel(self) -> None:
        import asyncio
        from src.data_providers.analytics import (
            compute_exposure_summary,
            compute_portfolio_cvar_and_risk_attribution,
        )
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        dates = pd.date_range("2026-04-01", periods=130, freq="B")
        rng = np.random.default_rng(21)
        # NVDA high vol (~3.5% daily), AZN.L medium vol (~1.5%), VWRL.L low vol (~0.8%)
        r_nvda = rng.normal(0.0005, 0.035, 130)
        r_nvda[15] = -0.11  # realistic left-tail shock
        r_nvda[42] = -0.09
        r_azn = rng.normal(0.0003, 0.015, 130)
        r_vwrl = rng.normal(0.0002, 0.008, 130)

        def _mk(rets: np.ndarray, start: float) -> pd.DataFrame:
            c = start * np.cumprod(1.0 + rets)
            return pd.DataFrame(
                {
                    "Open": c * 0.998,
                    "High": c * 1.01,
                    "Low": c * 0.99,
                    "Close": c,
                    "Volume": np.full(len(c), 1_200_000.0),
                },
                index=dates,
            )

        histories = {
            "NVDA": _mk(r_nvda, 120.0),
            "AZN.L": _mk(r_azn, 10000.0),
            "VWRL.L": _mk(r_vwrl, 95.0),
            "MSFT": _mk(rng.normal(0.0006, 0.018, 130), 400.0),
            "SPY": _mk(rng.normal(0.0004, 0.010, 130), 520.0),
            "ISF.L": _mk(rng.normal(0.0002, 0.009, 130), 800.0),
        }
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-16 10:00",
            fetched_at_ms=1794823200000,
            total_combined_value=10000.0,
            isa=AccountSnapshot(
                ok=True,
                total_account_value=6000.0,
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="NVDA_US_EQ",
                        ticker="NVDA",
                        yahoo_symbol="NVDA",
                        name="NVIDIA",
                        inst_currency="USD",
                        quantity=20,
                        avg_price=100,
                        cur_price=120,
                        current_value=3500.0,
                        total_cost=3000.0,
                        pnl=500.0,
                        fx_impact=0.0,
                        return_pct=16.7,
                        weight_pct=58.3,
                    ),
                    PositionItem(
                        account="ISA",
                        raw_ticker="AZNl_EQ",
                        ticker="AZN",
                        yahoo_symbol="AZN.L",
                        name="AstraZeneca",
                        inst_currency="GBp",
                        quantity=25,
                        avg_price=9800,
                        cur_price=10200,
                        current_value=2500.0,
                        total_cost=2400.0,
                        pnl=100.0,
                        fx_impact=0.0,
                        return_pct=4.2,
                        weight_pct=41.7,
                    ),
                ],
            ),
            invest=AccountSnapshot(
                ok=True,
                total_account_value=4000.0,
                positions=[
                    PositionItem(
                        account="Invest",
                        raw_ticker="VWRLl_EQ",
                        ticker="VWRL",
                        yahoo_symbol="VWRL.L",
                        name="Vanguard All-World",
                        inst_currency="GBP",
                        quantity=30,
                        avg_price=90,
                        cur_price=95,
                        current_value=3000.0,
                        total_cost=2850.0,
                        pnl=150.0,
                        fx_impact=0.0,
                        return_pct=5.3,
                        weight_pct=75.0,
                    ),
                ],
            ),
        )
        bundles = {
            "NVDA": TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", currency="USD", sector="Technology"),
            "AZN.L": TickerMarketBundle(symbol="AZN.L", quote_type="EQUITY", currency="GBp", sector="Healthcare"),
            "VWRL.L": TickerMarketBundle(symbol="VWRL.L", quote_type="ETF", currency="GBP"),
        }

        risk_info = compute_portfolio_cvar_and_risk_attribution(portfolio, histories, lookback_days=120)
        self.assertGreater(risk_info["portfolio_vol_pct"], 0.0)
        self.assertLess(risk_info["cvar_95_pct"], 0.0)
        self.assertGreater(risk_info["cvar_95_gbp"], 0.0)
        rc_sum = sum(r["risk_pct"] for r in risk_info["risk_contributions"])
        self.assertAlmostEqual(rc_sum, 100.0, delta=0.5)
        self.assertEqual(risk_info["top_risk_sources"][0]["symbol"], "NVDA")
        self.assertGreater(risk_info["top_risk_sources"][0]["risk_pct"], risk_info["top_risk_sources"][0]["weight_pct"])

        exp_compat = compute_exposure_summary(portfolio, bundles)
        self.assertNotIn("95%单日CVaR", exp_compat)
        exp_full = compute_exposure_summary(portfolio, bundles, histories=histories)
        self.assertIn("组合真实风险归因(前3波动源)", exp_full)
        self.assertIn("尾部压力测试(95%单日CVaR)", exp_full)

        # Verify score_candidate and run_quant_screener Stage 2 US deep intel integration
        tech = compute_technical_snapshot("MSFT", histories["MSFT"])
        assert tech is not None
        strategy = StrategySection()
        msft_bundle = TickerMarketBundle(symbol="MSFT", quote_type="EQUITY", currency="USD", forward_pe=24.0)
        intel_bull = {
            "recommendations": [{"period": "2026-11-01", "strong_buy": 20, "buy": 8, "hold": 2, "sell": 0, "strong_sell": 0}],
            "earnings_surprises": [{"surprise_pct": 5.0}, {"surprise_pct": 4.2}, {"surprise_pct": 6.1}, {"surprise_pct": 3.0}],
            "insider_sentiment": {"mspr": 18.5, "change": 900},
        }
        intel_bear = {
            "recommendations": [{"period": "2026-11-01", "strong_buy": 1, "buy": 2, "hold": 10, "sell": 5, "strong_sell": 2}],
            "earnings_surprises": [{"surprise_pct": -2.0}],
            "insider_sentiment": {"mspr": -14.0, "change": -800},
        }
        c_bull = score_candidate("MSFT", "ISA_LONG_TERM", tech, msft_bundle, 0.3, "AZN.L", False, False, strategy, hist=histories["MSFT"], deep_intel=intel_bull)
        c_bear = score_candidate("MSFT", "ISA_LONG_TERM", tech, msft_bundle, 0.3, "AZN.L", False, False, strategy, hist=histories["MSFT"], deep_intel=intel_bear)
        self.assertGreater(c_bull.dimension_scores["value_quality"], c_bear.dimension_scores["value_quality"])
        self.assertIn("连续财报Beat", c_bull.pros)
        self.assertTrue(any("内部人净增持" in p for p in c_bull.pros))
        self.assertTrue(any("内部人减持" in c for c in c_bear.cons))

        batch_calls: list[list[str]] = []

        class _ScreenerStubMD(MarketDataService):
            async def get_histories(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
                return {s: histories[s] for s in symbols if s in histories}

            async def get_bundles(self, symbols: list[str], **kwargs):
                return {s: histories[s] for s in symbols if s in histories}, {
                    s: TickerMarketBundle(symbol=s, quote_type="EQUITY", currency="USD", current_price=400.0)
                    for s in symbols
                }

            async def get_us_deep_intel_batch(self, symbols: list[str], max_concurrency: int = 4) -> dict[str, dict]:
                batch_calls.append(list(symbols))
                return {s: intel_bull for s in symbols}

        universe = UniverseSection(
            isa_long_term=["MSFT"],
            invest_tactical=["MSFT"],
            uk_lse_etfs=[],
        )
        isa_res, _ = asyncio.run(
            run_quant_screener(
                portfolio=portfolio,
                market_data=_ScreenerStubMD(),
                universe=universe,
                watchlist=[],
                strategy=strategy,
                existing_histories=histories,
            )
        )
        self.assertTrue(batch_calls)
        self.assertIn("MSFT", batch_calls[0])
        self.assertTrue(any("连续财报Beat" in c.pros for c in isa_res))

    def test_imp22_offline_universe_and_threshold_audit(self) -> None:
        from src.data_providers.t212_client import AccountSnapshot, PortfolioData

        dates = pd.date_range("2025-01-01", periods=360, freq="B")
        rng = np.random.default_rng(42)

        def _ohlcv_from_close(closes: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": closes * 0.996,
                    "High": closes * 1.012,
                    "Low": closes * 0.988,
                    "Close": closes,
                    "Volume": np.full(len(closes), 1_200_000.0),
                },
                index=dates,
            )

        # Held position: HELD1 with realistic noisy upward drift
        held_rets = 0.0007 + rng.normal(0.0, 0.015, size=len(dates))
        held_closes = 100.0 * np.cumprod(1.0 + held_rets)

        # High-overlap candidate: tracks HELD1 closely over >=120d (corr ~ 0.97)
        dup_rets = held_rets + rng.normal(0.0, 0.002, size=len(dates))
        dup_closes = 80.0 * np.cumprod(1.0 + dup_rets)

        # Zombie candidate: rises early, then persistent downtrend for last 140 bars (Close < MA200 & MA50 < MA200 >= 60d)
        zombie_closes = np.concatenate(
            [
                np.linspace(100.0, 140.0, 120),
                np.linspace(140.0, 60.0, 240),
            ]
        )

        # Breakout winner + loser candidate: steady uptrend with periodic breakout spikes triggering trailing TP & ATR stop
        brk_closes = np.linspace(50.0, 80.0, 220)
        seg_up = np.linspace(80.0, 108.0, 35)  # +35% surge -> hits +20% activation
        seg_pb = np.linspace(108.0, 92.0, 15)  # -14.8% pullback from peak -> triggers -12% trailing TP
        seg_base2 = np.linspace(92.0, 96.0, 30)
        seg_drop = np.linspace(96.0, 78.0, 30)  # triggers ATR stop
        seg_tail = np.linspace(78.0, 85.0, 30)
        brk_full = np.concatenate([brk_closes, seg_up, seg_pb, seg_base2, seg_drop, seg_tail])
        brk_df = _ohlcv_from_close(brk_full)
        # Ensure breakout day cleanly clears prior 20d high and drop triggers Low <= stop
        brk_df.iloc[220, brk_df.columns.get_loc("Close")] = 84.0
        brk_df.iloc[220, brk_df.columns.get_loc("High")] = 84.5

        histories = {
            "HELD1": _ohlcv_from_close(held_closes),
            "DUP1": _ohlcv_from_close(dup_closes),
            "ZOMB1": _ohlcv_from_close(zombie_closes),
            "BRK1": brk_df,
        }
        portfolio = PortfolioData(
            fetched_at_uk="2026-11-14 10:00",
            fetched_at_ms=1794650400000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                positions=[
                    PositionItem(
                        account="ISA",
                        raw_ticker="HELD1_US_EQ",
                        ticker="HELD1",
                        yahoo_symbol="HELD1",
                        name="Held One",
                        inst_currency="USD",
                        quantity=10.0,
                        avg_price=100.0,
                        cur_price=float(held_closes[-1]),
                        current_value=5000.0,
                        total_cost=4000.0,
                        pnl=1000.0,
                        fx_impact=0.0,
                        return_pct=25.0,
                        weight_pct=100.0,
                    )
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest"),
        )
        universe = UniverseSection(
            isa_long_term=["DUP1", "ZOMB1"],
            invest_tactical=["BRK1"],
        )
        strategy = StrategySection()

        audit = run_offline_universe_and_threshold_audit(histories, portfolio, universe, strategy)
        self.assertEqual(audit["universe_count"], 3)
        self.assertEqual(audit["held_count"], 1)

        overlap_syms = [x["symbol"] for x in audit["overlap_candidates"]]
        self.assertIn("DUP1", overlap_syms)
        self.assertNotIn("ZOMB1", overlap_syms)

        zombie_syms = [x["symbol"] for x in audit["zombie_candidates"]]
        self.assertIn("ZOMB1", zombie_syms)
        self.assertNotIn("DUP1", zombie_syms)

        bt = audit["threshold_backtest"]
        self.assertGreaterEqual(bt["sample_count"], 1)
        self.assertGreaterEqual(bt["trailing_tp_triggers"] + bt["atr_stop_triggers"] + bt["horizon_exits"], 1)

        summary = audit["summary_text"]
        self.assertNotIn("#", summary)
        self.assertNotIn("**", summary)
        self.assertIn("DUP1", summary)
        self.assertIn("ZOMB1", summary)
        self.assertIn("阈值回测", summary)


class TestStage2PenetrationAndSizing(unittest.TestCase):
    """v5.3.2 Stage 2: ETF look-through tech exposure + risk-based ticket sizing."""

    def _pos(self, account: str, symbol: str, ccy: str, value: float) -> PositionItem:
        return PositionItem(
            account=account,
            raw_ticker=symbol.replace(".", "_"),
            ticker=symbol.split(".")[0],
            yahoo_symbol=symbol,
            name=symbol,
            inst_currency=ccy,
            quantity=10,
            avg_price=100.0,
            cur_price=value / 10.0,
            current_value=value,
            total_cost=value,
            pnl=0.0,
            fx_impact=0.0,
            return_pct=0.0,
            weight_pct=value / 20.0,
        )

    def test_penetration_merges_etf_implied_and_direct_tech(self) -> None:
        from src.data_providers.analytics import estimate_penetration_tech_exposure
        from src.data_providers.t212_client import PortfolioData

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-17 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                total_account_value=2000.0,
                positions=[
                    self._pos("ISA", "VWRP.L", "GBp", 1000.0),
                    self._pos("ISA", "NVDA", "USD", 500.0),
                ],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=2000.0,
        )
        res = estimate_penetration_tech_exposure(portfolio, {})
        self.assertAlmostEqual(res["etf_implied_value"], 250.0, places=2)  # 25% of £1,000
        self.assertAlmostEqual(res["direct_value"], 500.0, places=2)
        self.assertAlmostEqual(res["combined_value"], 750.0, places=2)
        self.assertAlmostEqual(res["combined_pct"], 37.5, places=1)
        self.assertEqual(res["etf_breakdown"][0]["symbol"], "VWRP.L")

    def test_penetration_skips_unmapped_etf_and_marks_it(self) -> None:
        from src.data_providers.analytics import estimate_penetration_tech_exposure
        from src.data_providers.t212_client import PortfolioData

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-17 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                total_account_value=1000.0,
                positions=[self._pos("ISA", "ZZZZ.L", "GBP", 300.0)],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=1000.0,
        )
        bundles = {"ZZZZ.L": TickerMarketBundle(symbol="ZZZZ.L", quote_type="ETF")}
        res = estimate_penetration_tech_exposure(portfolio, bundles)
        self.assertEqual(res["combined_value"], 0.0)
        self.assertAlmostEqual(res["unmapped_etf_value"], 300.0, places=2)

    def test_exposure_summary_reports_penetration_line(self) -> None:
        from src.data_providers.analytics import compute_exposure_summary
        from src.data_providers.t212_client import PortfolioData

        portfolio = PortfolioData(
            fetched_at_uk="2026-11-17 07:30",
            fetched_at_ms=1763105400000,
            isa=AccountSnapshot(
                ok=True,
                account_label="ISA",
                total_account_value=2000.0,
                positions=[self._pos("ISA", "VUAG.L", "GBp", 1000.0)],
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=2000.0,
        )
        text = compute_exposure_summary(portfolio, {})
        self.assertIn("穿透后真实科技暴露(估算)", text)
        self.assertIn("静态权重近似", text)

    def test_risk_based_size_uses_budget_and_flags_fractional_orders(self) -> None:
        from src.engine.portfolio_rules import compute_risk_based_size_gbp

        strategy = StrategySection()
        wide = compute_risk_based_size_gbp(
            account="Invest",
            strategy=strategy,
            effective_stop_pct=-12.0,
            account_total_value_gbp=10000.0,
            existing_position_value_gbp=0.0,
        )
        self.assertAlmostEqual(wide["budget_sized_gbp"], 708.33, places=2)  # £85 / 12%
        self.assertAlmostEqual(wide["suggested_gbp"], 708.33, places=2)
        self.assertFalse(wide["below_min_order"])

        tight = compute_risk_based_size_gbp(
            account="Invest",
            strategy=strategy,
            effective_stop_pct=-8.0,
            account_total_value_gbp=10000.0,
        )
        self.assertAlmostEqual(tight["budget_sized_gbp"], 1062.5, places=2)

        too_wide = compute_risk_based_size_gbp(
            account="Invest",
            strategy=strategy,
            effective_stop_pct=-50.0,
            account_total_value_gbp=10000.0,
        )
        self.assertAlmostEqual(too_wide["suggested_gbp"], 170.0, places=2)
        self.assertTrue(too_wide["below_min_order"])

    def test_risk_based_size_caps_by_headroom_and_isa_min_tickets(self) -> None:
        from src.engine.portfolio_rules import compute_risk_based_size_gbp

        strategy = StrategySection()
        capped = compute_risk_based_size_gbp(
            account="Invest",
            strategy=strategy,
            effective_stop_pct=-8.0,
            account_total_value_gbp=3000.0,
            existing_position_value_gbp=700.0,
        )
        # 25% single-name cap = £750, minus £700 held => only £50 headroom left
        self.assertAlmostEqual(capped["headroom_gbp"], 50.0, places=2)
        self.assertAlmostEqual(capped["suggested_gbp"], 50.0, places=2)
        self.assertTrue(capped["below_min_order"])

        isa_etf = compute_risk_based_size_gbp(account="ISA", strategy=strategy, is_etf=True)
        isa_eq = compute_risk_based_size_gbp(account="ISA", strategy=strategy, is_etf=False)
        self.assertAlmostEqual(isa_etf["min_order_gbp"], 400.0)
        self.assertAlmostEqual(isa_eq["min_order_gbp"], 1200.0)

class TestStage2CadenceAndOpportunityCost(unittest.TestCase):
    """v5.3.2 Stage 2: frequency caps, re-buy cooldown and ISA opportunity-cost replacement."""

    def _pick(self, action: str, symbol: str = "MSFT", account: str = "ISA"):
        from src.agents.schemas import PickActionItem, PickDecision

        return PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account=account,
                    symbol=symbol,
                    action=action,
                    thesis="测试逻辑",
                    entry_zone="£700 于 400 下方",
                    stop_condition="跌破 380",
                    target_window="3个月",
                )
            ],
            consistency_and_cash_note="测试",
        )

    def test_frequency_cap_blocks_new_buy_but_not_trim(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {
            "buys_7d": 2,
            "buys_30d": 3,
            "partial": False,
            "recently_closed": [],
            "cooldown_days": 15,
        }
        res = apply_risk_override_guardrail(
            self._pick("BUY"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertTrue(res.applied)
        self.assertEqual(res.overrides[0].trigger_type, "FREQUENCY_CAP")
        self.assertIn("近 7 天已新开仓", res.overrides[0].reason)

        trim_res = apply_risk_override_guardrail(
            self._pick("TRIM"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertFalse(trim_res.applied)

    def test_monthly_cap_blocks_when_weekly_still_open(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {"buys_7d": 1, "buys_30d": 6, "partial": False, "recently_closed": [], "cooldown_days": 15}
        res = apply_risk_override_guardrail(
            self._pick("ADD"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertTrue(res.applied)
        self.assertEqual(res.overrides[0].trigger_type, "FREQUENCY_CAP")
        self.assertIn("近 30 天已新开仓", res.overrides[0].reason)

    def test_rebuy_cooldown_blocks_reentry_but_not_other_names(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {
            "buys_7d": 0,
            "buys_30d": 1,
            "partial": False,
            "recently_closed": [
                {"ticker": "NVDA", "account": "ISA", "days_since_sell": 3.0, "in_cooldown": True}
            ],
            "cooldown_days": 15,
        }
        blocked = apply_risk_override_guardrail(
            self._pick("BUY", symbol="NVDA"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertTrue(blocked.applied)
        self.assertEqual(blocked.overrides[0].trigger_type, "REBUY_COOLDOWN")
        self.assertIn("再买冷却期", blocked.overrides[0].reason)

        allowed = apply_risk_override_guardrail(
            self._pick("BUY", symbol="MSFT"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertFalse(allowed.applied)

    def test_partial_history_never_hard_blocks(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {
            "buys_7d": 9,
            "buys_30d": 20,
            "partial": True,
            "recently_closed": [
                {"ticker": "MSFT", "account": "ISA", "days_since_sell": 1.0, "in_cooldown": True}
            ],
            "cooldown_days": 15,
        }
        res = apply_risk_override_guardrail(
            self._pick("BUY"),
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            trade_behavior=behavior,
        )
        self.assertFalse(res.applied)

    def test_advisory_mode_keeps_action_and_only_advises(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {
            "buys_7d": 2,
            "buys_30d": 6,
            "partial": False,
            "recently_closed": [],
            "cooldown_days": 15,
        }
        res = apply_risk_override_guardrail(
            self._pick("BUY"), strategy=StrategySection(), trade_behavior=behavior
        )
        self.assertFalse(res.applied)
        self.assertEqual(res.overrides, [])
        self.assertEqual(len(res.decision.selected_actions), 1)
        notes = " ".join(res.notes)
        self.assertIn("[⚠️纪律提醒]", notes)
        self.assertIn("近 7 天已新开仓", notes)
        self.assertIn("advisory 模式不拦截", notes)
        # Tone consistency: advisory must not read as a prohibition
        self.assertIn("本周额度已用完", notes)
        self.assertNotIn("禁止继续买入", notes)

    def test_off_mode_is_silent_and_never_blocks(self) -> None:
        from src.engine.portfolio_rules import apply_risk_override_guardrail

        behavior = {
            "buys_7d": 9,
            "buys_30d": 20,
            "partial": False,
            "recently_closed": [
                {"ticker": "MSFT", "account": "ISA", "days_since_sell": 1.0, "in_cooldown": True}
            ],
            "cooldown_days": 15,
        }
        res = apply_risk_override_guardrail(
            self._pick("BUY"),
            strategy=StrategySection(trade_cadence_guard_mode="off"),
            trade_behavior=behavior,
        )
        self.assertFalse(res.applied)
        self.assertEqual(res.notes, [])

    def test_cadence_guard_mode_normalizes_unknown_values(self) -> None:
        self.assertEqual(
            StrategySection(trade_cadence_guard_mode="HARD").trade_cadence_guard_mode, "hard"
        )
        self.assertEqual(
            StrategySection(trade_cadence_guard_mode="nonsense").trade_cadence_guard_mode, "advisory"
        )
        self.assertEqual(StrategySection().trade_cadence_guard_mode, "advisory")

    def test_isa_weak_trend_triggers_opportunity_cost_not_cash_veto(self) -> None:
        from src.data_providers.analytics import TechnicalSnapshot

        strategy = StrategySection()
        pos = PositionItem(
            account="ISA",
            raw_ticker="MSFT_US_EQ",
            ticker="MSFT",
            yahoo_symbol="MSFT",
            name="Microsoft",
            inst_currency="USD",
            quantity=10,
            avg_price=500.0,
            cur_price=400.0,
            current_value=4000.0,
            total_cost=5000.0,
            pnl=-1000.0,
            fx_impact=0.0,
            return_pct=-20.0,
            weight_pct=20.0,
        )
        tech_weak = TechnicalSnapshot(
            symbol="MSFT",
            last_close=400.0,
            dist_ma200_pct=-5.0,
            trend_regime="WEAK",
        )
        bundle = TickerMarketBundle(symbol="MSFT", current_price=400.0, target_mean_price=401.0, analyst_count=8)
        ev = evaluate_holding(pos, tech_weak, bundle, strategy)
        joined = " ".join(ev.reasons)
        self.assertIn("机会成本置换复核", joined)
        self.assertIn("不以现金利息否决", joined)
        self.assertNotIn("现金+摩擦门槛", joined)
        self.assertEqual(ev.action_bias, "HOLD")


if __name__ == "__main__":
    unittest.main()
