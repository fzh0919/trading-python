"""Tests for pandas technical indicators, correlation matrix, Add/Trim rules, and screener."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import StrategySection
from src.data_providers.analytics import (
    compute_correlation_matrix,
    compute_rsi_wilder,
    compute_technical_snapshot,
    max_correlation_against_portfolio,
)
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import PositionItem
from src.engine.portfolio_rules import evaluate_holding, format_evaluations_text
from src.engine.screener import score_candidate


def _make_hist(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=dates)


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
        self.assertIn("常规持有 x2 只均无异常", txt)
        self.assertIn("VWRP[ISA 20.0%/+5.0%]", txt)
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

        # 4% cash hurdle + friction gate dampens marginal upside ADD_CANDIDATE
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
            target_mean_price=410.0,  # +2.5% < 4.3% hurdle
            analyst_count=12,
        )
        ev_low_upside = evaluate_holding(pos_msft.model_copy(update={"weight_pct": 10.0}), tech_borderline, bundle_low_upside, strategy)
        self.assertNotEqual(ev_low_upside.action_bias, "ADD_CANDIDATE")
        self.assertIn("现金+摩擦门槛", " ".join(ev_low_upside.reasons))

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
        from src.data_providers.analytics import TechnicalSnapshot
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


if __name__ == "__main__":
    unittest.main()
