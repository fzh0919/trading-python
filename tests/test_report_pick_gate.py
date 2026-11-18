"""v5.4.0 deployment-gate offline test matrix (all offline mocks, no network, no live DB).

Covers the 8 core groups from docs/V540_IMPLEMENTATION_BLUEPRINT.md Task 7:
1. cash-ledger identity & pending-order triage (2026-11-17 real snapshot regression)
2. Invest risk slots & financed unlock
3. gate_epoch_date anchor & zero-quota exemptions
4. ISA legacy crumb gentle pruning (never force-liquidate)
5. FX-distortion-free regime + anti-deadlock 2y replay
6. master DEFEND protocol & dual-account inverse coupling
7. one-way PM clamp + <=1800-char report + off-mode byte-identical passthrough
8. /pick track filtering wiring
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.agents.pick_agent import _apply_gate_channel_constraints, _format_gate_constraint_block
from src.agents.report_agent import _build_report_recommendations, _enrich_gate_candidate_bundles
from src.agents.schemas import (
    MorningReportDecision,
    PickActionItem,
    PickDecision,
    ReportActionItem,
    render_morning_report,
)
from src.config import AppConfig, EnvSettings, StrategySection, UniverseSection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import MacroSnapshot, TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
)
from src.engine.deployment_gate import (
    PENDING_CORE_ETF_BUY,
    PENDING_NEW_EQUITY_BUY,
    PENDING_PROTECTIVE_SELL_STOP,
    _scan_a_equity_candidates,
    classify_pending_orders,
    evaluate_deployment_gate,
    evaluate_market_regime,
    format_deployment_gate_banner,
    reconcile_pm_pick_gate,
    replay_deployment_gate_history,
)
from src.engine.portfolio_rules import (
    apply_risk_override_guardrail,
    build_report_size_notes,
    derive_action_size_note,
    evaluate_portfolio,
)
from src.storage.db import AdvisorDatabase

LONDON = ZoneInfo("Europe/London")


def _make_hist(closes: list[float], start_date: str = "2025-06-02") -> pd.DataFrame:
    idx = pd.bdate_range(start_date, periods=len(closes))
    closes_arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes_arr * 0.998,
            "High": closes_arr * 1.005,
            "Low": closes_arr * 0.995,
            "Close": closes_arr,
            "Volume": np.full(len(closes_arr), 1_500_000.0),
        },
        index=idx,
    )


def _spy_neutral_hist(n: int = 260) -> pd.DataFrame:
    """SPY above MA200 with a -10% 52w-high drawdown -> NEUTRAL raw regime."""
    closes = [round(100.0 + i * 1.6, 2) for i in range(n - 10)]  # 100 -> ~498
    peak = closes[-1]
    for k in range(10):  # last 10 bars pull back to -10% off peak
        closes.append(round(peak * (1.0 - 0.10 * (k + 1) / 10.0), 2))
    return _make_hist(closes)


def _spy_risk_on_hist(n: int = 260) -> pd.DataFrame:
    closes = [round(100.0 + i * 1.2, 2) for i in range(n)]  # monotonic rise
    return _make_hist(closes)


def _vwrp_ladder_hist() -> pd.DataFrame:
    """VWRP.L: steady rise, then the last two bars first-cross the -15% 52w-high step."""
    closes = [round(100.0 + i * 0.45, 2) for i in range(258)]  # -> ~215.65
    high = max(closes)
    closes.append(round(high * 0.855, 2))  # -14.5%
    closes.append(round(high * 0.8475, 2))  # -15.25% first cross
    return _make_hist(closes)


def _neutral_macro(vix: float = 16.0) -> MacroSnapshot:
    return MacroSnapshot(
        fetched_at_utc="2026-11-17T07:25:00Z",
        gbp_usd=1.25,
        vix=vix,
        hy_oas_spread_pct=3.0,
    )


def _pos(
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    value: float,
    qty: float,
    avg: float,
    cur: float,
    return_pct: float,
    weight_pct: float,
    ccy: str = "USD",
) -> PositionItem:
    return PositionItem(
        account=account,
        raw_ticker=f"{ticker}_{ccy}_EQ",
        ticker=ticker,
        yahoo_symbol=yahoo_symbol,
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


def _real_snapshot_portfolio() -> PortfolioData:
    """Precise 2026-11-17 live-state regression fixture (design doc §1.1)."""
    isa_equities = [
        _pos("ISA", "MSFT", "MSFT", value=6000.0, qty=15.0, avg=300.0, cur=400.0, return_pct=25.0, weight_pct=10.5),
        _pos("ISA", "NVDA", "NVDA", value=3600.0, qty=20.0, avg=120.0, cur=180.0, return_pct=40.0, weight_pct=6.3),
        _pos("ISA", "JPM", "JPM", value=2000.0, qty=8.0, avg=200.0, cur=250.0, return_pct=12.0, weight_pct=3.5),
        _pos("ISA", "TSLA", "TSLA", value=1800.0, qty=6.0, avg=280.0, cur=300.0, return_pct=4.0, weight_pct=3.1),
        _pos("ISA", "UBER", "UBER", value=2168.0, qty=20.0, avg=95.0, cur=108.4, return_pct=8.0, weight_pct=3.8),
        # 8 legacy crumbs (< £1,200)
        _pos("ISA", "AAL", "AAL", value=550.0, qty=40.0, avg=16.0, cur=13.75, return_pct=-14.0, weight_pct=1.0),
        _pos("ISA", "PLUG", "PLUG", value=480.0, qty=60.0, avg=9.5, cur=8.0, return_pct=-16.0, weight_pct=0.8),
        _pos("ISA", "SNAP", "SNAP", value=520.0, qty=40.0, avg=15.0, cur=13.0, return_pct=-13.0, weight_pct=0.9),
        _pos("ISA", "LCID", "LCID", value=610.0, qty=100.0, avg=7.2, cur=6.1, return_pct=-15.0, weight_pct=1.1),
        _pos("ISA", "OPEN", "OPEN", value=340.0, qty=20.0, avg=20.0, cur=17.0, return_pct=-15.0, weight_pct=0.6),
        _pos("ISA", "SOFI", "SOFI", value=700.0, qty=70.0, avg=11.0, cur=10.0, return_pct=-9.0, weight_pct=1.2),
        _pos("ISA", "DKNG", "DKNG", value=420.0, qty=10.0, avg=48.0, cur=42.0, return_pct=-12.5, weight_pct=0.7),
        _pos("ISA", "RIVN", "RIVN", value=380.0, qty=25.0, avg=18.0, cur=15.2, return_pct=-15.6, weight_pct=0.7),
    ]
    assert abs(sum(p.current_value for p in isa_equities) - 19568.0) < 0.01
    isa_positions = [
        _pos("ISA", "VUAG", "VUAG.L", value=8408.0, qty=84.08, avg=100.0, cur=100.0, return_pct=6.0, weight_pct=14.7, ccy="GBP"),
        _pos("ISA", "VWRP", "VWRP.L", value=8200.0, qty=71.4, avg=114.84, cur=114.84, return_pct=8.0, weight_pct=14.3, ccy="GBP"),
        _pos("ISA", "IBTA", "IBTA.L", value=305.0, qty=55.45, avg=5.5, cur=5.5, return_pct=1.0, weight_pct=0.5, ccy="GBP"),
        *isa_equities,
    ]
    isa_pending = [
        PendingOrderItem(
            id="p-shel", account="ISA", raw_ticker="SHELl_EQ", ticker="SHEL", yahoo_symbol="SHEL.L",
            name="Shell", side="BUY", order_type="LIMIT", quantity=70.0, value=244800.0,
            limit_price=3480.0, currency="GBp", time_in_force="GTC",
        ),
        PendingOrderItem(
            id="p-rkt", account="ISA", raw_ticker="RKTl_EQ", ticker="RKT", yahoo_symbol="RKT.L",
            name="Reckitt", side="BUY", order_type="LIMIT", quantity=38.2, value=200500.0,
            limit_price=5250.0, currency="GBp", time_in_force="GTC",
        ),
        PendingOrderItem(
            id="p-vwrp", account="ISA", raw_ticker="VWRP_LSE_ETF", ticker="VWRP", yahoo_symbol="VWRP.L",
            name="VWRP", side="BUY", order_type="LIMIT", quantity=0.0, value=2168.0,
            limit_price=114.0, currency="GBP", time_in_force="GTC",
        ),
        PendingOrderItem(
            id="p-vuag", account="ISA", raw_ticker="VUAG_LSE_ETF", ticker="VUAG", yahoo_symbol="VUAG.L",
            name="VUAG", side="BUY", order_type="LIMIT", quantity=0.0, value=2120.0,
            limit_price=99.8, currency="GBP", time_in_force="GTC",
        ),
    ]
    invest_positions = [
        _pos("Invest", "CF", "CF", value=1631.0, qty=16.0, avg=121.41, cur=119.47, return_pct=-1.6, weight_pct=11.5),
        _pos("Invest", "AMZN", "AMZN", value=1248.0, qty=5.0, avg=238.42, cur=259.0, return_pct=8.6, weight_pct=8.8),
    ]
    invest_pending = [
        PendingOrderItem(
            id="p-amzn-stop", account="Invest", raw_ticker="AMZN_US_EQ", ticker="AMZN", yahoo_symbol="AMZN",
            name="Amazon", side="SELL", order_type="STOP", quantity=5.0, value=0.0,
            limit_price=None, stop_price=243.50, currency="USD", time_in_force="GTC",
        ),
        PendingOrderItem(
            id="p-cf-stop", account="Invest", raw_ticker="CF_US_EQ", ticker="CF", yahoo_symbol="CF",
            name="CF Industries", side="SELL", order_type="STOP", quantity=16.0, value=0.0,
            limit_price=None, stop_price=113.50, currency="USD", time_in_force="GTC",
        ),
    ]
    return PortfolioData(
        fetched_at_uk="2026-11-17 07:30",
        fetched_at_ms=1763364600000,
        total_combined_value=71546.24,
        total_combined_cash=32286.60,
        isa=AccountSnapshot(
            ok=True, account_label="ISA", total_account_value=57343.73,
            investments_value=36481.23, investments_cost=34000.0, unrealized_pnl=2481.23,
            total_cash=20862.50, available_to_trade=12121.34, reserved_cash=8741.16,
            cash_ratio_pct=36.4, positions=isa_positions, pending_orders=isa_pending,
        ),
        invest=AccountSnapshot(
            ok=True, account_label="Invest", total_account_value=14202.51,
            investments_value=2879.0, investments_cost=2806.0, unrealized_pnl=73.0,
            total_cash=11324.10, available_to_trade=11324.10, reserved_cash=0.0,
            cash_ratio_pct=79.73, positions=invest_positions, pending_orders=invest_pending,
        ),
        gbp_usd_rate=1.25,
    )


def _run_snapshot_gate(
    portfolio: PortfolioData | None = None,
    *,
    strategy: StrategySection | None = None,
    vix: float = 16.0,
    spy_hist: pd.DataFrame | None = None,
    candidates: list[dict] | None = None,
    now_uk: datetime | None = None,
    prev_gate_states: list[dict] | None = None,
) -> dict:
    strat = strategy or StrategySection()
    spy = spy_hist if spy_hist is not None else _spy_neutral_hist()
    return evaluate_deployment_gate(
        portfolio=portfolio if portfolio is not None else _real_snapshot_portfolio(),
        strategy=strat,
        bundles={"IBTA.L": TickerMarketBundle(symbol="IBTA.L", quote_type="ETF", currency="GBP")},
        histories={"SPY": spy},
        macro=_neutral_macro(vix=vix),
        candidates=candidates,
        universe_etf_symbols=["VWRP.L", "VUAG.L"],
        now_uk=now_uk,
        prev_gate_states=prev_gate_states,
    )


def _up_candidate(symbol: str = "PANW") -> dict:
    return {
        "symbol": symbol,
        "account": "Invest",
        "tech": TechnicalSnapshot(
            symbol=symbol, last_close=300.0, ma50=295.0, ma200=270.0,
            dist_ma50_pct=0.8, rsi14=60.0, trend_regime="BULL",
        ),
        "bundle": None,
    }


def _sell_order(
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    at_uk: datetime,
    pnl: float,
) -> HistoricalOrderItem:
    ms = int(at_uk.timestamp() * 1000)
    return HistoricalOrderItem(
        id=f"s-{ticker}-{ms}", account=account, raw_ticker=f"{ticker}_US_EQ", ticker=ticker,
        yahoo_symbol=yahoo_symbol, name=ticker, inst_currency="USD", side="SELL",
        order_type="MARKET", filled_at=at_uk.isoformat(), filled_at_ms=ms,
        uk_time_str=at_uk.strftime("%Y-%m-%d %H:%M"), quantity=1.0, price=100.0,
        net_value=100.0, wallet_currency="USD", realised_pnl=pnl, tax_total=0.0,
    )


def _buy_order(
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    at_uk: datetime,
) -> HistoricalOrderItem:
    ms = int(at_uk.timestamp() * 1000)
    return HistoricalOrderItem(
        id=f"b-{ticker}-{ms}", account=account, raw_ticker=f"{ticker}_US_EQ", ticker=ticker,
        yahoo_symbol=yahoo_symbol, name=ticker, inst_currency="USD", side="BUY",
        order_type="MARKET", filled_at=at_uk.isoformat(), filled_at_ms=ms,
        uk_time_str=at_uk.strftime("%Y-%m-%d %H:%M"), quantity=1.0, price=100.0,
        net_value=100.0, wallet_currency="USD", realised_pnl=0.0, tax_total=0.0,
    )


# ===========================================================================
# Group 1: cash-ledger identity & pending-order triage (real snapshot)
# ===========================================================================
class TestGateLedgerIdentityAndOrderTriage(unittest.TestCase):
    def test_ledger_identity_deployable_pct_is_net_not_double_subtracted(self) -> None:
        gate = _run_snapshot_gate()
        isa = gate["isa"]
        # available_to_trade is ALREADY net of reserved BUY cash: 12121.34 / 57343.73.
        # A v2-style second subtraction would wrongly yield ~5.9% — we must see 21.14%.
        self.assertAlmostEqual(isa["deployable_pct"], 21.14, places=2)
        self.assertNotAlmostEqual(isa["deployable_pct"], 5.9, places=1)
        self.assertAlmostEqual(isa["total_cash_pct"], 36.38, places=1)
        self.assertAlmostEqual(isa["available_to_trade_gbp"], 12121.34, places=2)
        self.assertAlmostEqual(isa["reserved_cash_gbp"], 8741.16, places=2)

    def test_committed_satellite_counts_include_pending_equity_buys(self) -> None:
        gate = _run_snapshot_gate()
        isa = gate["isa"]
        self.assertEqual(isa["committed_satellite_count"], 15)  # 13 held + SHEL + RKT
        self.assertAlmostEqual(isa["committed_satellite_pct"], 41.9, places=1)
        self.assertAlmostEqual(isa["satellite_equity_pct"], 34.1, places=1)
        self.assertAlmostEqual(isa["core_etf_pct"], 29.0, places=1)
        self.assertTrue(isa["isa_satellite_saturated"])
        self.assertEqual(isa["legacy_crumb_count"], 8)
        self.assertEqual(isa["pending_new_equity_buys"], ["SHEL.L", "RKT.L"])
        self.assertEqual(sorted(isa["pending_core_etf_buys"]), ["VUAG.L", "VWRP.L"])

    def test_pending_order_triage_three_classes(self) -> None:
        portfolio = _real_snapshot_portfolio()
        etf_set = {"VWRP.L", "VUAG.L"}
        classified = classify_pending_orders(portfolio, StrategySection(), etf_set)
        by_class = classified["by_class"]
        self.assertEqual(
            {o.yahoo_symbol for o in by_class[PENDING_PROTECTIVE_SELL_STOP]},
            {"AMZN", "CF"},
        )
        self.assertEqual(
            {o.yahoo_symbol for o in by_class[PENDING_CORE_ETF_BUY]},
            {"VWRP.L", "VUAG.L"},
        )
        self.assertEqual(
            {o.yahoo_symbol for o in by_class[PENDING_NEW_EQUITY_BUY]},
            {"SHEL.L", "RKT.L"},
        )
        # STOP sell orders are protection shields, never "pending buys"
        self.assertFalse(
            any(
                o.side.upper() == "SELL"
                for o in by_class[PENDING_NEW_EQUITY_BUY] + by_class[PENDING_CORE_ETF_BUY]
            )
        )

    def test_protective_stops_never_block_the_gate(self) -> None:
        gate = _run_snapshot_gate()
        invest = gate["invest"]
        # Both AMZN & CF carry STOP sells — this must NOT read as "pending orders pause entries".
        self.assertEqual(sorted(invest["pending_protective_stops"]), ["AMZN", "CF"])
        self.assertEqual(invest["pending_new_equity_buys"], [])
        self.assertEqual(invest["slots_used"], 1)  # only CF (unfinanced) occupies a slot

    def test_amzn_financed_zero_heat_and_cf_one_slot(self) -> None:
        gate = _run_snapshot_gate()
        invest = gate["invest"]
        self.assertEqual(invest["financed_positions"], ["AMZN"])  # STOP 243.50 >= avg + friction
        self.assertEqual(invest["unfinanced_positions"], ["CF"])
        self.assertEqual(invest["heat_breakdown"]["AMZN"], 0.0)
        cf_heat = invest["heat_breakdown"]["CF"]
        self.assertGreaterEqual(cf_heat, 80.0)
        self.assertLessEqual(cf_heat, 110.0)  # design: ~£82-£108 principal at risk
        self.assertGreater(invest["heat_gbp_total"], 0.0)

    def test_gate_dict_is_json_serializable(self) -> None:
        gate = _run_snapshot_gate()
        dumped = json.dumps(gate, ensure_ascii=False)
        self.assertIn("deployable_pct", dumped)
        self.assertIn("financed_positions", dumped)

    def test_gbp_pence_pending_buys_estimated_in_gbp(self) -> None:
        portfolio = _real_snapshot_portfolio()
        # SHEL.L 244800 GBp -> £2,448 ; RKT.L 200500 GBp -> £2,005 (100x scale safety)
        from src.engine.portfolio_rules import _estimate_pending_order_gbp

        by_sym = {o.id: o for o in portfolio.isa.pending_orders}
        pos_by_sym = {p.yahoo_symbol: p for p in portfolio.all_positions()}
        self.assertAlmostEqual(_estimate_pending_order_gbp(by_sym["p-shel"], pos_by_symbol=pos_by_sym), 2448.0, places=1)
        self.assertAlmostEqual(_estimate_pending_order_gbp(by_sym["p-rkt"], pos_by_symbol=pos_by_sym), 2005.0, places=1)


# ===========================================================================
# Group 2: Invest risk slots & financed unlock
# ===========================================================================
class TestInvestSlotsAndFinancedUnlock(unittest.TestCase):
    def test_neutral_full_slot_waits_despite_80pct_cash(self) -> None:
        gate = _run_snapshot_gate()
        invest = gate["invest"]
        self.assertAlmostEqual(invest["cash_pct"], 79.7, places=1)
        self.assertEqual(invest["status"], "WAIT")
        self.assertEqual(invest["mode"], "slots_full")
        self.assertIn("CF", invest["reason"])
        # Never a dca-style "spend the idle cash" RUN for Invest
        self.assertNotEqual(invest["mode"], "dca")

    def test_raising_cf_stop_to_breakeven_releases_slot(self) -> None:
        portfolio = _real_snapshot_portfolio()
        for order in portfolio.invest.pending_orders:
            if order.yahoo_symbol == "CF":
                order.stop_price = 122.50  # above avg+friction -> financed
        gate = _run_snapshot_gate(portfolio)
        invest = gate["invest"]
        self.assertEqual(sorted(invest["financed_positions"]), ["AMZN", "CF"])
        self.assertEqual(invest["unfinanced_positions"], [])
        self.assertEqual(invest["slots_used"], 0)
        self.assertEqual(invest["heat_gbp_total"], 0.0)

    def test_freed_slot_with_a_candidate_unlocks_run_targeted(self) -> None:
        portfolio = _real_snapshot_portfolio()
        for order in portfolio.invest.pending_orders:
            if order.yahoo_symbol == "CF":
                order.stop_price = 122.50
        gate = _run_snapshot_gate(portfolio, candidates=[_up_candidate()])
        invest = gate["invest"]
        self.assertEqual(invest["status"], "RUN")
        self.assertEqual(invest["mode"], "targeted")
        self.assertIn("PANW", invest["reason"])
        self.assertEqual(gate["pm"]["mode"], "run")

    def test_full_slot_with_a_candidate_rotates_not_adds(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        invest = gate["invest"]
        self.assertEqual(invest["status"], "ROTATE")
        self.assertEqual(invest["mode"], "rotate")
        self.assertEqual(invest["funding_from"], "CF")
        # ISA dca (RUN) outranks Invest ROTATE in the global merge (RUN > ROTATE)
        self.assertEqual(gate["global_status"], "RUN")
        self.assertEqual(gate["pm"]["mode"], "rotate")

    def test_best_idea_only_daily_cap_blocks_second_entry(self) -> None:
        portfolio = _real_snapshot_portfolio()
        today_09 = datetime(2026, 11, 17, 9, 30, tzinfo=LONDON)
        portfolio.invest.recent_orders = [_buy_order("Invest", "PANW", "PANW", at_uk=today_09)]
        gate = _run_snapshot_gate(portfolio, now_uk=datetime(2026, 11, 17, 10, 0, tzinfo=LONDON))
        invest = gate["invest"]
        self.assertEqual(invest["today_new_equity_entries"], 1)
        self.assertTrue(invest["daily_cap_reached"])
        self.assertEqual(invest["status"], "WAIT")
        self.assertEqual(invest["mode"], "daily_cap")

    def test_add_to_existing_position_does_not_consume_new_entry_cap(self) -> None:
        portfolio = _real_snapshot_portfolio()
        today_09 = datetime(2026, 11, 17, 9, 30, tzinfo=LONDON)
        portfolio.invest.recent_orders = [_buy_order("Invest", "CF", "CF", at_uk=today_09)]
        gate = _run_snapshot_gate(portfolio, now_uk=datetime(2026, 11, 17, 10, 0, tzinfo=LONDON))
        invest = gate["invest"]
        self.assertEqual(invest["today_new_equity_entries"], 0)
        self.assertFalse(invest["daily_cap_reached"])

    def test_risk_on_regime_grants_two_slots(self) -> None:
        prev = {"regime": "NEUTRAL", "raw_regime": "RISK_ON", "raw_streak_days": 2, "ease_streak_days": 0}
        gate = _run_snapshot_gate(
            spy_hist=_spy_risk_on_hist(),
            prev_gate_states=[prev],
            candidates=[_up_candidate()],
        )
        self.assertEqual(gate["regime"], "RISK_ON")
        invest = gate["invest"]
        self.assertEqual(invest["slots_cap"], 2)
        self.assertEqual(invest["status"], "RUN")  # 1/2 slots used, candidate present
        self.assertEqual(invest["mode"], "targeted")


# ===========================================================================
# Group 3: gate_epoch_date anchor & zero-quota exemptions
# ===========================================================================
class TestGateEpochAndZeroQuota(unittest.TestCase):
    def _strat(self, **kwargs) -> StrategySection:
        return StrategySection(gate_epoch_date="2026-11-18", **kwargs)

    def test_pre_epoch_losses_are_fully_exempt(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 10, 1, 9, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 10, 3, 9, 0, tzinfo=LONDON), pnl=-100.0),
            _buy_order("Invest", "X", "PANW", at_uk=datetime(2026, 10, 4, 9, 0, tzinfo=LONDON)),
            _sell_order("Invest", "X", "PANW", at_uk=datetime(2026, 10, 6, 9, 0, tzinfo=LONDON), pnl=-90.0),
        ]
        gate = _run_snapshot_gate(
            portfolio,
            strategy=self._strat(),
            now_uk=datetime(2026, 11, 17, 10, 0, tzinfo=LONDON),
        )
        invest = gate["invest"]
        self.assertEqual(invest["tactical_losses_recent"], 0)
        self.assertFalse(invest["in_loss_cooldown"])
        self.assertEqual(invest["risk_budget_scale"], 1.0)

    def test_post_epoch_legacy_cleanup_sells_without_buy_fills_are_exempt(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.recent_orders = [
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 20, 9, 0, tzinfo=LONDON), pnl=-80.0),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 21, 9, 0, tzinfo=LONDON), pnl=-70.0),
        ]
        gate = _run_snapshot_gate(
            portfolio,
            strategy=self._strat(),
            now_uk=datetime(2026, 11, 22, 10, 0, tzinfo=LONDON),
        )
        invest = gate["invest"]
        self.assertEqual(invest["tactical_losses_recent"], 0)
        self.assertFalse(invest["in_loss_cooldown"])

    def test_core_etf_buys_consume_zero_equity_quota(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "VWRP", "VWRP.L", at_uk=datetime(2026, 11, 10, 9, 0, tzinfo=LONDON)),
            _buy_order("Invest", "VWRP", "VWRP.L", at_uk=datetime(2026, 11, 11, 9, 0, tzinfo=LONDON)),
            _buy_order("Invest", "VWRP", "VWRP.L", at_uk=datetime(2026, 11, 12, 9, 0, tzinfo=LONDON)),
        ]
        msft_bundle = TickerMarketBundle(symbol="MSFT", quote_type="EQUITY", currency="USD", current_price=400.0)
        vwrp_bundle = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP", current_price=114.0)
        pick = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="MSFT", action="BUY", thesis="右侧突破",
                    entry_zone="400", stop_condition="380", target_window="460",
                    entry_price=400.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        res = apply_risk_override_guardrail(
            pick,
            portfolio=portfolio,
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            bundle_map={"MSFT": msft_bundle, "VWRP.L": vwrp_bundle},
            macro=_neutral_macro(),
            trade_behavior={"buys_7d": 3, "buys_30d": 3, "partial": False, "recently_closed": [], "cooldown_days": 15},
        )
        # 3 "buys" were all VWRP.L ETF -> recomputed equity quota is 0 -> no FREQUENCY_CAP
        self.assertFalse(res.applied)

    def test_etf_buy_target_is_cadence_exempt(self) -> None:
        portfolio = _real_snapshot_portfolio()
        vwrp_bundle = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP", current_price=114.0)
        pick = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="VWRP.L", action="BUY", thesis="归拢宽基",
                    entry_zone="114", stop_condition="跌破MA200", target_window="长线",
                    entry_price=114.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        res = apply_risk_override_guardrail(
            pick,
            portfolio=portfolio,
            strategy=StrategySection(trade_cadence_guard_mode="hard"),
            bundle_map={"VWRP.L": vwrp_bundle},
            macro=_neutral_macro(),
            trade_behavior={"buys_7d": 9, "buys_30d": 20, "partial": False, "recently_closed": [], "cooldown_days": 15},
        )
        self.assertFalse(res.applied)  # ETF buys never trigger FREQUENCY_CAP / cooldown

    def test_post_epoch_two_consecutive_losses_trigger_cooldown_wait(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions = []  # closed everything
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 19, 9, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 21, 9, 0, tzinfo=LONDON), pnl=-30.0),
            _buy_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 21, 15, 0, tzinfo=LONDON)),
            _sell_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 23, 9, 0, tzinfo=LONDON), pnl=-40.0),
        ]
        gate = _run_snapshot_gate(
            portfolio,
            strategy=self._strat(),
            now_uk=datetime(2026, 11, 24, 10, 0, tzinfo=LONDON),
        )
        invest = gate["invest"]
        self.assertEqual(invest["tactical_losses_recent"], 2)
        self.assertTrue(invest["in_loss_cooldown"])
        self.assertEqual(invest["status"], "WAIT")
        self.assertEqual(invest["mode"], "cooldown")

    def test_partial_sells_of_same_loser_count_once(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions = []
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 19, 9, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 21, 9, 0, tzinfo=LONDON), pnl=-30.0),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 21, 14, 0, tzinfo=LONDON), pnl=-20.0),
        ]
        gate = _run_snapshot_gate(
            portfolio,
            strategy=self._strat(),
            now_uk=datetime(2026, 11, 24, 10, 0, tzinfo=LONDON),
        )
        invest = gate["invest"]
        self.assertEqual(invest["tactical_losses_recent"], 1)
        self.assertFalse(invest["in_loss_cooldown"])

    def test_cooldown_expired_recovery_pilot_runs_at_half_risk(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions = []
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 19, 9, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 21, 9, 0, tzinfo=LONDON), pnl=-30.0),
            _buy_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 21, 15, 0, tzinfo=LONDON)),
            _sell_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 23, 9, 0, tzinfo=LONDON), pnl=-40.0),
        ]
        gate = _run_snapshot_gate(
            portfolio,
            strategy=self._strat(),
            now_uk=datetime(2026, 12, 4, 10, 0, tzinfo=LONDON),
        )
        invest = gate["invest"]
        self.assertFalse(invest["in_loss_cooldown"])
        self.assertEqual(invest["risk_budget_scale"], 0.5)


# ===========================================================================
# Group 4: ISA legacy crumb gentle pruning
# ===========================================================================
class TestIsaCrumbPruning(unittest.TestCase):
    def _crumb_portfolio(self, crumb_specs: list[tuple[str, float, float]]) -> PortfolioData:
        positions = [
            _pos("ISA", "VWRP", "VWRP.L", value=20000.0, qty=100.0, avg=200.0, cur=200.0, return_pct=5.0, weight_pct=60.0, ccy="GBP")
        ]
        for sym, value, ret in crumb_specs:
            positions.append(
                _pos("ISA", sym, sym, value=value, qty=10.0, avg=value / 10.0, cur=value / 10.0, return_pct=ret, weight_pct=1.0)
            )
        return PortfolioData(
            fetched_at_uk="2026-11-17 07:30",
            fetched_at_ms=1763364600000,
            isa=AccountSnapshot(
                ok=True, account_label="ISA", total_account_value=25000.0,
                total_cash=4000.0, available_to_trade=4000.0, positions=positions,
            ),
            invest=AccountSnapshot(ok=True, account_label="Invest", total_account_value=0.0),
            total_combined_value=25000.0,
        )

    def _tech(self, rsi: float, dd20: float | None) -> TechnicalSnapshot:
        return TechnicalSnapshot(symbol="X", last_close=100.0, rsi14=rsi, drawdown_20d_pct=dd20, trend_regime="PULLBACK")

    def test_drawdown_crumbs_are_never_forced_to_trim(self) -> None:
        portfolio = self._crumb_portfolio(
            [(f"CR{i}", 500.0, -8.0 - i) for i in range(8)]
        )
        tech_map = {f"CR{i}": self._tech(42.0, -9.0) for i in range(8)}
        bundle_map = {f"CR{i}": TickerMarketBundle(symbol=f"CR{i}", quote_type="EQUITY", currency="USD") for i in range(8)}
        bundle_map["VWRP.L"] = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")
        evals = evaluate_portfolio(portfolio, tech_map, bundle_map, StrategySection())
        crumb_evals = [e for e in evals if e.ticker.startswith("CR")]
        self.assertTrue(all(e.action_bias == "HOLD" for e in crumb_evals))
        self.assertTrue(all(not e.is_crumb_prune_candidate for e in crumb_evals))

    def test_rebound_crumbs_are_capped_at_two_per_report(self) -> None:
        portfolio = self._crumb_portfolio(
            [(f"CR{i}", 500.0, -2.0) for i in range(4)]
        )
        tech_map = {f"CR{i}": self._tech(65.0, -1.0) for i in range(4)}
        bundle_map = {f"CR{i}": TickerMarketBundle(symbol=f"CR{i}", quote_type="EQUITY", currency="USD") for i in range(4)}
        bundle_map["VWRP.L"] = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")
        evals = evaluate_portfolio(portfolio, tech_map, bundle_map, StrategySection())
        crumb_evals = [e for e in evals if e.ticker.startswith("CR")]
        trims = [e for e in crumb_evals if e.action_bias == "TRIM_ALERT"]
        self.assertEqual(len(trims), 2)  # isa_crumb_pruning_max_per_report == 2
        for e in trims:
            self.assertTrue(e.is_crumb_prune_candidate)
            self.assertTrue(any("归拢至核心宽基" in r for r in e.reasons))
        holds = [e for e in crumb_evals if e.action_bias == "HOLD"]
        self.assertEqual(len(holds), 2)
        for e in holds:
            self.assertFalse(e.is_crumb_prune_candidate)
            self.assertTrue(any("不低位急割" in r for r in e.reasons))

    def test_crumb_below_breakeven_floor_is_not_surfaced(self) -> None:
        portfolio = self._crumb_portfolio([("CR0", 500.0, -4.0)])
        tech_map = {"CR0": self._tech(68.0, -1.0)}
        bundle_map = {"CR0": TickerMarketBundle(symbol="CR0", quote_type="EQUITY", currency="USD")}
        bundle_map["VWRP.L"] = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")
        evals = evaluate_portfolio(portfolio, tech_map, bundle_map, StrategySection())
        crumb = next(e for e in evals if e.ticker == "CR0")
        self.assertEqual(crumb.action_bias, "HOLD")
        self.assertFalse(crumb.is_crumb_prune_candidate)

    def test_crumb_near_earnings_is_not_surfaced(self) -> None:
        portfolio = self._crumb_portfolio([("CR0", 500.0, -2.0)])
        tech_map = {"CR0": self._tech(66.0, -1.0)}
        bundle_map = {
            "CR0": TickerMarketBundle(
                symbol="CR0", quote_type="EQUITY", currency="USD",
                days_to_earnings=3, earnings_date="2026-11-20",
            )
        }
        bundle_map["VWRP.L"] = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")
        evals = evaluate_portfolio(portfolio, tech_map, bundle_map, StrategySection())
        crumb = next(e for e in evals if e.ticker == "CR0")
        self.assertFalse(crumb.is_crumb_prune_candidate)


# ===========================================================================
# Group 5: FX-distortion-free regime & anti-deadlock replay
# ===========================================================================
class TestRegimeAndReplay(unittest.TestCase):
    def test_regime_anchors_spy_not_gbp_distorted_vwrp(self) -> None:
        # GBP surges -> VWRP.L 虚跌破位; SPY (USD) healthy -> regime must follow SPY.
        vwrp_closes = [round(220.0 * (1.0 - 0.004 * i), 2) for i in range(260)]
        spy = _spy_risk_on_hist()
        macro = _neutral_macro(vix=16.0)
        info = evaluate_market_regime(
            {"SPY": spy, "VWRP.L": _make_hist(vwrp_closes)},
            macro,
            StrategySection(),
        )
        self.assertEqual(info["benchmark_symbol"], "SPY")
        self.assertEqual(info["raw_regime"], "RISK_ON")
        self.assertNotEqual(info["raw_regime"], "RISK_OFF")

    def test_risk_off_enters_fast_and_exits_stepwise(self) -> None:
        strat = StrategySection()
        # Day 1: VIX 30 -> raw RISK_OFF, enters immediately (1-day confirm).
        d1 = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=30.0), strat)
        self.assertEqual(d1["regime"], "RISK_OFF")
        # Days 2-3: crisis eased (VIX 21, price above MA200) but needs 3-day confirmation.
        prev = {"regime": d1["regime"], "raw_regime": d1["raw_regime"], "raw_streak_days": d1["raw_streak_days"], "ease_streak_days": d1["ease_streak_days"]}
        d2 = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev)
        self.assertEqual(d2["regime"], "RISK_OFF")
        prev = {"regime": d2["regime"], "raw_regime": d2["raw_regime"], "raw_streak_days": d2["raw_streak_days"], "ease_streak_days": d2["ease_streak_days"]}
        d3 = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev)
        self.assertEqual(d3["regime"], "RISK_OFF")
        prev = {"regime": d3["regime"], "raw_regime": d3["raw_regime"], "raw_streak_days": d3["raw_streak_days"], "ease_streak_days": d3["ease_streak_days"]}
        d4 = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev)
        self.assertEqual(d4["regime"], "NEUTRAL")
        self.assertEqual(d4["ease_streak_days"], 3)

    def test_exit_never_requires_vix_under_18(self) -> None:
        # VIX stuck at 21 (a normal bull-regime level) must NOT deadlock RISK_OFF.
        strat = StrategySection()
        d1 = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=30.0), strat)
        self.assertEqual(d1["regime"], "RISK_OFF")
        prev = {"regime": d1["regime"], "raw_regime": d1["raw_regime"], "raw_streak_days": d1["raw_streak_days"], "ease_streak_days": d1["ease_streak_days"]}
        for _ in range(3):
            day = evaluate_market_regime({"SPY": _spy_neutral_hist()}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev)
            prev = {"regime": day["regime"], "raw_regime": day["raw_regime"], "raw_streak_days": day["raw_streak_days"], "ease_streak_days": day["ease_streak_days"]}
        self.assertEqual(prev["regime"], "NEUTRAL")

    def test_risk_on_needs_three_day_confirmation(self) -> None:
        strat = StrategySection()
        day1 = evaluate_market_regime({"SPY": _spy_risk_on_hist()}, _neutral_macro(vix=16.0), strat)
        self.assertEqual(day1["regime"], "NEUTRAL")  # streak 1 < 3
        prev = {"regime": day1["regime"], "raw_regime": day1["raw_regime"], "raw_streak_days": day1["raw_streak_days"], "ease_streak_days": day1["ease_streak_days"]}
        day2 = evaluate_market_regime({"SPY": _spy_risk_on_hist()}, _neutral_macro(vix=16.0), strat, prev_gate_state=prev)
        self.assertEqual(day2["regime"], "NEUTRAL")
        prev = {"regime": day2["regime"], "raw_regime": day2["raw_regime"], "raw_streak_days": day2["raw_streak_days"], "ease_streak_days": day2["ease_streak_days"]}
        day3 = evaluate_market_regime({"SPY": _spy_risk_on_hist()}, _neutral_macro(vix=16.0), strat, prev_gate_state=prev)
        self.assertEqual(day3["regime"], "RISK_ON")

    def test_stale_macro_caps_regime_at_neutral(self) -> None:
        macro = _neutral_macro(vix=15.0)
        macro.is_stale = True
        info = evaluate_market_regime({"SPY": _spy_risk_on_hist()}, macro, StrategySection())
        self.assertEqual(info["raw_regime"], "NEUTRAL")
        self.assertTrue(any("过期" in n for n in info["degraded_notes"]))

    def test_missing_benchmark_degrades_to_unknown(self) -> None:
        info = evaluate_market_regime({}, _neutral_macro(), StrategySection())
        self.assertEqual(info["regime"], "UNKNOWN")

    def test_two_year_replay_no_deadlock_with_vix_stuck_at_21(self) -> None:
        # 500 trading days: rise -> -35% crash (RISK_OFF) -> recovery above MA200 with VIX 21.
        # With the v2 "VIX<18 required to exit" bug this would stay locked in RISK_OFF forever;
        # v5.4.0 must exit step-wise once the benchmark stabilizes above MA200.
        closes: list[float] = []
        closes += [round(100.0 * (1.0015**i), 2) for i in range(250)]  # 100 -> ~145
        peak = closes[-1]
        closes += [round(peak * (1.0 - 0.012 * (k + 1)), 2) for k in range(30)]  # -36% crash
        trough = closes[-1]
        closes += [round(trough * (1.008**i), 2) for i in range(220)]  # sharp V-shaped recovery
        bench = _make_hist(closes)
        vix_closes = [16.0] * 250 + [35.0] * 30 + [21.0] * 220  # VIX never falls below 18 again
        vix_df = _make_hist(vix_closes)
        hy_oas = pd.Series([3.0] * len(closes), index=bench.index)
        summary = replay_deployment_gate_history(bench, vix_df, StrategySection(), hy_oas_series=hy_oas)
        self.assertFalse(summary["deadlock_detected"], summary)
        self.assertGreater(summary["state_days"]["RISK_OFF"], 0)
        self.assertGreater(summary["state_days"]["NEUTRAL"], 0)
        self.assertGreater(summary["state_days"]["RISK_ON"], 0)  # healthy early phase
        self.assertLess(summary["max_risk_off_run_days"], 200)
        self.assertEqual(summary["days_evaluated"], 300)

    def test_replay_flags_true_pathological_lock_above_ma200(self) -> None:
        # Crash, then the benchmark sits far above MA200 for months while the gate (hypothetically)
        # stays locked in RISK_OFF: the replay must flag this as a deadlock, not swallow it.
        closes: list[float] = []
        closes += [round(100.0 * (1.0015**i), 2) for i in range(200)]
        peak = closes[-1]
        closes += [round(peak * (1.0 - 0.015 * (k + 1)), 2) for k in range(20)]
        trough = closes[-1]
        closes += [round(trough * (1.01**i), 2) for i in range(280)]  # +1%/day far above MA200
        bench = _make_hist(closes)
        # VIX stays at 30 forever -> raw regime is RISK_OFF even with a fully recovered index,
        # and the index sits far above MA200: a genuine pathological lock.
        vix_df = _make_hist([30.0] * len(closes))
        summary = replay_deployment_gate_history(bench, vix_df, StrategySection())
        self.assertTrue(summary["deadlock_detected"])
        self.assertGreater(summary["max_risk_off_run_days"], 60)


# ===========================================================================
# Group 6: master DEFEND protocol & dual-account inverse coupling
# ===========================================================================
class TestDefendAndInverseCoupling(unittest.TestCase):
    def test_vix_30_risk_off_triggers_invest_defend(self) -> None:
        gate = _run_snapshot_gate(vix=30.0)
        self.assertEqual(gate["regime"], "RISK_OFF")
        invest = gate["invest"]
        self.assertEqual(invest["status"], "DEFEND")
        self.assertEqual(invest["mode"], "defend")
        self.assertTrue(gate["triggers"]["D"]["risk_off_defend"])
        self.assertIn("撤除未成交个股买单", invest.get("defend_note", ""))

    def test_isa_wide_base_ladder_still_sowed_during_risk_off(self) -> None:
        # VIX 30 (RISK_OFF) must NOT kill ISA core-ETF ladder seeding at the -15% step.
        portfolio = _real_snapshot_portfolio()
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": _spy_neutral_hist(), "VWRP.L": _vwrp_ladder_hist()},
            macro=_neutral_macro(vix=30.0),
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        isa = gate["isa"]
        self.assertEqual(isa["status"], "RUN")
        self.assertEqual(isa["mode"], "ladder")
        self.assertTrue(isa["etf_open"])
        self.assertEqual(gate["invest"]["status"], "DEFEND")
        self.assertEqual(gate["triggers"]["A"]["core_etf_step"]["hit"], 15.0)

    def test_isa_equity_channel_closed_during_risk_off(self) -> None:
        portfolio = _real_snapshot_portfolio()
        for order in list(portfolio.isa.pending_orders):
            if order.yahoo_symbol in {"SHEL.L", "RKT.L"}:
                portfolio.isa.pending_orders.remove(order)
        portfolio.isa.positions = portfolio.isa.positions[:5]  # only big 5, no saturation
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(vix=30.0),
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        isa = gate["isa"]
        self.assertFalse(isa["isa_satellite_saturated"])
        self.assertNotEqual(isa["status"], "RUN")  # equity channel shut in RISK_OFF (no ladder hit)
        self.assertEqual(isa["mode"], "none")
        self.assertFalse(isa["equity_open"])
        self.assertFalse(isa["etf_open"])
        self.assertEqual(gate["invest"]["status"], "DEFEND")

    def test_isa_extreme_low_cash_triggers_defence_note(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.isa.total_cash = 500.0
        gate = _run_snapshot_gate(portfolio, vix=16.0)
        isa = gate["isa"]
        self.assertLess(isa["total_cash_pct"], 2.0)
        self.assertTrue(gate["triggers"]["D"]["isa_extreme_low_cash"])
        self.assertIn("2%结构垫", gate["invest"].get("defend_note", ""))


# ===========================================================================
# Group 7: one-way PM clamp, banner, 1800-char cap, off-mode passthrough
# ===========================================================================
class TestPmClampBannerAndRendering(unittest.TestCase):
    def test_pm_cannot_upgrade_a_closed_gate(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.is_fallback = True
        portfolio.isa.is_fallback = True
        portfolio.invest.is_fallback = True
        gate = _run_snapshot_gate(portfolio)
        self.assertEqual(gate["global_status"], "UNKNOWN")
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "reason": "PM 想抄底"})
        self.assertFalse(pm["allow"])
        self.assertTrue(pm["pm_overridden"])
        self.assertIn("无权升级", pm["reason"])

    def test_pm_tightening_to_false_is_respected(self) -> None:
        gate = _run_snapshot_gate()
        pm = reconcile_pm_pick_gate(gate, {"allow": False, "reason": "美联储议息夜"})
        self.assertFalse(pm["allow"])
        self.assertEqual(pm["mode"], "none")
        self.assertIn("美联储议息夜", pm["reason"])

    def test_rotate_with_empty_pm_funding_keeps_deterministic_source(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        self.assertEqual(gate["invest"]["status"], "ROTATE")
        # The deterministic engine already computed a legal funding source (CF); an
        # untouched PM funding_from falls back to it instead of falsely degrading to WAIT.
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "funding_from": ""}, whitelist_symbols={"CF"})
        self.assertTrue(pm["allow"])
        self.assertEqual(pm["mode"], "rotate")
        self.assertEqual(pm["funding_from"], "CF")

    def test_rotate_with_illegal_funding_is_rejected(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "funding_from": "TSLA"}, whitelist_symbols={"CF"})
        self.assertFalse(pm["allow"])
        self.assertIn("置换源", pm["reason"])

    def test_rotate_with_valid_funding_is_kept(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "funding_from": "CF"}, whitelist_symbols={"CF"})
        self.assertTrue(pm["allow"])
        self.assertEqual(pm["mode"], "rotate")
        self.assertEqual(pm["funding_from"], "CF")

    def test_snapshot_banner_content_and_limits(self) -> None:
        gate = _run_snapshot_gate()
        banner = gate["banner"]
        self.assertLessEqual(len(banner), 110)
        self.assertNotIn("#", banner)
        self.assertNotIn("**", banner)
        self.assertIn("ISA 仅配宽基", banner)
        self.assertIn("Invest WAIT", banner)
        self.assertIn("21.1%", banner)
        self.assertIn("15只", banner)
        self.assertIn("42%", banner)

    def test_all_state_banners_under_110_no_markdown(self) -> None:
        # RUN targeted
        portfolio = _real_snapshot_portfolio()
        for order in portfolio.invest.pending_orders:
            if order.yahoo_symbol == "CF":
                order.stop_price = 122.50
        run_gate = _run_snapshot_gate(portfolio, candidates=[_up_candidate()])
        # ROTATE
        rotate_gate = _run_snapshot_gate(candidates=[_up_candidate()])
        # DEFEND
        defend_gate = _run_snapshot_gate(vix=30.0)
        # UNKNOWN
        fallback = _real_snapshot_portfolio()
        fallback.is_fallback = True
        fallback.isa.is_fallback = True
        fallback.invest.is_fallback = True
        unknown_gate = _run_snapshot_gate(fallback)
        for gate in (run_gate, rotate_gate, defend_gate, unknown_gate):
            banner = format_deployment_gate_banner(gate)
            self.assertLessEqual(len(banner), 110, banner)
            self.assertNotIn("#", banner)
            self.assertNotIn("**", banner)
            self.assertNotEqual(banner.strip(), "")

    def test_gate_banner_pinned_under_sec2_and_report_under_1800(self) -> None:
        gate = _run_snapshot_gate()
        banner = gate["banner"]
        decision = MorningReportDecision(
            headline="核心动作置顶测试",
            top_actions=[ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="N/A", reason="观望")],
            daily_vetoes=["冻结半导体"],
            macro_and_cash="宏观" * 60,
            account_diagnosis="诊断" * 60,
            flagged_positions=[f"H{i}: " + ("细节" * 30) for i in range(6)],
            hold_summary="常规持有",
            trade_review="复盘" * 30,
        )
        rendered = render_morning_report(decision, max_chars=1800, gate_banner=banner)
        self.assertLessEqual(len(rendered), 1800)
        header = "【宏观、账户与今日风控红线】"
        self.assertIn(header, rendered)
        self.assertIn(banner.splitlines()[0], rendered)
        # The banner is the FIRST line of section 2 — nothing may separate it from the header.
        self.assertIn(f"{header}\n{banner.splitlines()[0]}", rendered)

    def test_render_strips_markdown_markers(self) -> None:
        decision = MorningReportDecision(
            headline="**加仓** #核心",
            top_actions=[ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="N/A", reason="**观望**")],
            daily_vetoes=["# 禁止追高"],
            macro_and_cash="宏观",
            account_diagnosis="诊断",
            hold_summary="持有",
            trade_review="复盘",
        )
        rendered = render_morning_report(decision, max_chars=1800)
        self.assertNotIn("**", rendered)
        self.assertNotIn("#", rendered)
        self.assertIn("加仓", rendered)

    def test_report_actions_enter_the_tn_ledger(self) -> None:
        portfolio = _real_snapshot_portfolio()
        bundles = {
            "CF": TickerMarketBundle(symbol="CF", quote_type="EQUITY", currency="USD", current_price=119.47),
            "AMZN": TickerMarketBundle(symbol="AMZN", quote_type="EQUITY", currency="USD", current_price=259.0),
            "NVDA": TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", currency="USD", current_price=135.0),
        }
        decision = MorningReportDecision(
            headline="测试",
            top_actions=[
                ReportActionItem(account="Invest", symbol="CF", action="ADD", price_zone="119", reason="加仓"),
                ReportActionItem(account="Invest", symbol="AMZN", action="TRIM", price_zone="260", reason="减仓"),
                ReportActionItem(account="ISA", symbol="NVDA", action="STOP", price_zone="130", reason="止损"),
                ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="N/A", reason="观望"),
            ],
            macro_and_cash="宏观",
            account_diagnosis="诊断",
            hold_summary="持有",
            trade_review="复盘",
        )
        cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999"))
        recs = _build_report_recommendations(decision, portfolio, bundles, cfg)
        self.assertEqual([r["symbol"] for r in recs], ["CF", "AMZN", "NVDA"])
        self.assertTrue(all(r["source"] == "report" for r in recs))
        self.assertTrue(all(r["fill_status"] == "ASSUMED" for r in recs))
        self.assertTrue(all(r["entry_price"] > 0 for r in recs))


# ===========================================================================
# Group 7b: /report integration — off-mode passthrough & advise-mode injection
# ===========================================================================
class TestReportGateIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_degraded_pm_parse_does_not_persist_a_sot(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio
        from src.agents.report_agent import generate_morning_report

        class _BadPMLLM:
            async def generate(self, **kwargs):
                return "not valid json"

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "degraded_pm.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                strategy=StrategySection(deployment_gate_mode="off"),
            )
            md = _FakeMarketData()
            try:
                out = await generate_morning_report(
                    cfg, _FakeT212(_sample_portfolio()), md, _BadPMLLM(), db
                )
            finally:
                await md.close()
            self.assertTrue(out.startswith("⚠️"))
            text, _concl, stale = await db.get_morning_context_with_status()
            self.assertEqual(text, "")
            self.assertFalse(stale)
            await db.close()

    async def test_off_mode_is_byte_stable_with_zero_extra_calls(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _RecordingLLM, _sample_portfolio
        from src.agents.report_agent import generate_morning_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "gate_off.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                strategy=StrategySection(deployment_gate_mode="off"),
            )
            md = _FakeMarketData()
            llm = _RecordingLLM()
            try:
                out = await generate_morning_report(cfg, _FakeT212(_sample_portfolio()), md, llm, db)
            finally:
                await md.close()
                await db.close()
            self.assertNotIn("组合门控", out)
            labels = [c.get("usage_label") for c in llm.calls]
            self.assertEqual(sorted(labels), ["report_analyst", "report_auditor", "report_pm"])
            self.assertLessEqual(len(out), 1800)

    async def test_advise_mode_injects_banner_and_pm_bottom_line(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _RecordingLLM, _sample_portfolio
        from src.agents.report_agent import generate_morning_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "gate_on.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                strategy=StrategySection(deployment_gate_mode="advise"),
            )
            md = _FakeMarketData()
            llm = _RecordingLLM()
            try:
                out = await generate_morning_report(cfg, _FakeT212(_sample_portfolio()), md, llm, db)
            finally:
                await md.close()
                await db.close()
            self.assertIn("组合门控", out)
            self.assertLessEqual(len(out), 1800)
            pm_call = next(c for c in llm.calls if c.get("usage_label") == "report_pm")
            self.assertIn("今日组合门控", pm_call["prompt"])
            self.assertIn("严禁把 WAIT/DEFEND 升为 RUN", pm_call["prompt"])
            self.assertEqual(len(llm.calls), 3)  # zero extra LLM calls
            pack = await db.get_context_pack(allow_fallback=False)
            self.assertIsNotNone(pack)
            self.assertIn("deployment_gate", pack)


# ===========================================================================
# Group 8: /pick gate wiring
# ===========================================================================
class TestPickGateWiring(unittest.TestCase):
    def test_gate_constraint_block_renders_saturation_and_rotate(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        block = _format_gate_constraint_block(gate)
        self.assertIn("今日组合门控约束", block)
        self.assertIn("ISA 个股卫星层已饱和", block)
        self.assertIn("ROTATE 等风险置换", block)
        self.assertIn("CF", block)

    def test_cross_account_duplicate_buy_is_intercepted(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {"isa": {"equity_open": True}, "invest": {"status": "RUN", "mode": "targeted"}}
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="AMZN", action="BUY", thesis="想重复买AMZN",
                    entry_zone="259", stop_condition="240", target_window="300", entry_price=259.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("跨账户同票去重", notes[0])
        self.assertIn("INVEST", notes[0])

    def test_cross_account_core_etf_buy_is_allowed(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions.append(
            _pos(
                "Invest", "VWRP", "VWRP.L", value=1000.0, qty=8.0, avg=120.0,
                cur=125.0, return_pct=4.0, weight_pct=7.0, ccy="GBP",
            )
        )
        gate = {
            "isa": {"equity_open": True, "etf_open": True, "isa_satellite_saturated": True},
            "invest": {"status": "WAIT", "mode": "none"},
            "etf_set": ["VWRP.L"],
        }
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="VWRP.L", action="BUY", thesis="ISA 归拢核心宽基",
                    entry_zone="118 GBP", stop_condition="100", target_window="130",
                    entry_price=118.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual([a.symbol for a in hardened.selected_actions], ["VWRP.L"])
        self.assertEqual(notes, [])

    def test_invest_defend_blocks_new_equity(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {"invest": {"status": "DEFEND", "mode": "defend"}}
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="AZN.L", action="BUY", thesis="测试",
                    entry_zone="100", stop_condition="90", target_window="120",
                    entry_price=100.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertIn("DEFEND", notes[0])

    def test_invest_wait_slots_full_blocks_new_equity(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {"invest": {"status": "WAIT", "mode": "slots_full", "reason": "槽位已满"}}
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="AZN.L", action="BUY", thesis="测试",
                    entry_zone="100", stop_condition="90", target_window="120",
                    entry_price=100.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertIn("slots_full", notes[0])

    def test_isa_channels_closed_block_respective_assets(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {
            "isa": {"equity_open": False, "etf_open": False},
            "invest": {"status": "WAIT", "mode": "none"},
            "etf_set": ["VWRP.L"],
        }
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="AZN.L", action="BUY", thesis="个股",
                    entry_zone="100", stop_condition="90", target_window="120",
                    entry_price=100.0,
                ),
                PickActionItem(
                    account="ISA", symbol="VWRP.L", action="BUY", thesis="宽基",
                    entry_zone="118", stop_condition="100", target_window="130",
                    entry_price=118.0,
                ),
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertTrue(any("个股通道关闭" in n for n in notes))
        self.assertTrue(any("宽基通道关闭" in n for n in notes))

    def test_rotate_invest_buy_without_funding_in_thesis_is_intercepted(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = _run_snapshot_gate(portfolio, candidates=[_up_candidate()])
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="PANW", action="BUY", thesis="右侧突破直接买",
                    entry_zone="300", stop_condition="285", target_window="340", entry_price=300.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("ROTATE 等风险置换", notes[0])

    def test_rotate_invest_buy_with_funding_in_thesis_is_kept(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = _run_snapshot_gate(portfolio, candidates=[_up_candidate()])
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="PANW", action="BUY", thesis="先平掉 CF 换入 PANW 右侧突破",
                    entry_zone="300", stop_condition="285", target_window="340", entry_price=300.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(len(hardened.selected_actions), 1)
        self.assertEqual(notes, [])


class TestPickGateEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_pick_satellite_saturated_blocks_isa_equity_keeps_etf(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio
        from src.agents.pick_agent import generate_pick_radar
        from src.agents.tools import clear_grounding_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "gate_pick.db")
            await db.init()

            portfolio = _sample_portfolio()  # ISA NVDA 70% + ULVR.L pending BUY -> saturated
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                universe=UniverseSection(isa_long_term=["VWRP.L", "AZN.L"], invest_tactical=["SHEL.L"]),
                strategy=StrategySection(settlement_holding_days=7, overlap_correlation_threshold=1.0),
            )
            gate = evaluate_deployment_gate(
                portfolio=portfolio,
                strategy=cfg.strategy,
                bundles={},
                histories={"SPY": _spy_neutral_hist()},
                macro=_neutral_macro(),
                universe_etf_symbols=["VWRP.L"],
            )
            self.assertTrue(gate["isa"]["isa_satellite_saturated"])
            seeded_pack = {
                "pack_version": "1.0",
                "cycle_date": "2026-11-17",
                "exposure_summary": "半导体敞口偏高",
                "recent_lessons": [],
                "symbols": {},
                "deployment_gate": gate,
            }
            await db.save_morning_report(
                "【今日核心行动与挂单裁决】\n• [ISA] VWRP.L →【KEEP】",
                "ISA 仅配宽基",
                context_pack=seeded_pack,
            )

            class _GatePickLLM:
                def __init__(self) -> None:
                    self.calls: list[dict] = []

                async def generate(self, **kwargs) -> str:
                    self.calls.append(kwargs)
                    if getattr(kwargs.get("response_schema"), "__name__", "") == "PickDecision":
                        return PickDecision(
                            headline="精选测试",
                            selected_actions=[
                                PickActionItem(
                                    account="ISA", symbol="AZN.L", action="BUY", thesis="英股医药防御配置",
                                    entry_zone="13400 GBp", stop_condition="12800 GBp", target_window="15000 GBp",
                                    entry_price=13400.0, stop_price=12800.0, target_price=15000.0,
                                ),
                                PickActionItem(
                                    account="ISA", symbol="VWRP.L", action="BUY", thesis="全球宽基零换汇零印花税",
                                    entry_zone="118 GBP", stop_condition="跌破MA200", target_window="长线",
                                    entry_price=118.0,
                                ),
                                PickActionItem(
                                    account="Invest", symbol="META", action="BUY", thesis="广告现金流稳健",
                                    entry_zone="510-525 USD", stop_condition="480", target_window="580",
                                    entry_price=520.0, stop_price=480.0, target_price=580.0,
                                ),
                            ],
                            rejected_or_frozen=[],
                            consistency_and_cash_note="ISA 仅配宽基。",
                        ).model_dump_json()
                    return "投研员提案：ISA 归拢 VWRP.L。"

            llm = _GatePickLLM()
            clear_grounding_cache()
            md = _FakeMarketData()
            try:
                out = await generate_pick_radar(cfg, _FakeT212(portfolio), md, llm, db)
            finally:
                await md.close()
                await db.close()

            self.assertLessEqual(len(out), 1000)
            self.assertIn("AZN.L(BUY→拦截)", out)
            self.assertIn("ISA 个股卫星层已饱和", out)
            self.assertIn("VWRP.L", out)
            self.assertIn("META", out)
            research_call = next(c for c in llm.calls if c.get("usage_label") == "pick_researcher")
            critic_call = next(c for c in llm.calls if c.get("usage_label") == "pick_critic")
            self.assertIn("今日组合门控约束", research_call["prompt"])
            self.assertIn("ISA 个股卫星层已饱和", critic_call["prompt"])
            self.assertIn("Invest 状态 WAIT", research_call["prompt"])

    async def test_pick_is_closed_by_pm_veto_and_stale_baseline(self) -> None:
        from tests.test_db_and_end_to_end import _FakeT212, _sample_portfolio
        from src.agents.pick_agent import generate_pick_radar

        class _NoCallLLM:
            async def generate(self, **kwargs):
                raise AssertionError("LLM must not be called when the gate/baseline is closed")

        cfg = AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            models={"enable_forced_search_grounding": False},
        )
        portfolio = _sample_portfolio()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "pm_veto.db")
            await db.init()
            await db.save_morning_report(
                "REPORT",
                "conclusion",
                context_pack={
                    "symbols": {},
                    "deployment_gate": {
                        "isa": {"equity_open": True, "etf_open": True},
                        "invest": {"status": "WAIT", "mode": "none"},
                        "pm": {"allow": False, "reason": "美联储议息夜"},
                    },
                },
            )
            out = await generate_pick_radar(
                cfg, _FakeT212(portfolio), SimpleNamespace(), _NoCallLLM(), db
            )
            await db.close()
        self.assertIn("⛔", out)
        self.assertIn("美联储议息夜", out)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "stale_baseline.db")
            await db.init()
            await db.save_morning_report(
                "OLD REPORT", "old conclusion", context_pack={"symbols": {}}, cycle_date="2020-01-01"
            )
            out = await generate_pick_radar(
                cfg, _FakeT212(portfolio), SimpleNamespace(), _NoCallLLM(), db
            )
            await db.close()
        self.assertIn("⛔", out)
        self.assertIn("上一交易日", out)

    async def test_pick_half_risk_scale_shrinks_sizing_note(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _sample_portfolio
        from src.agents.pick_agent import _post_screen_validate_decision

        portfolio = _sample_portfolio()
        cfg = AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            strategy=StrategySection(overlap_correlation_threshold=1.0),
        )
        decision = PickDecision(
            headline="半仓试探",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="META", action="BUY", thesis="右侧突破试探",
                    entry_zone="550", stop_condition="522.5", target_window="620",
                    entry_price=550.0, stop_price=522.5, target_price=620.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        gate = {
            "isa": {"isa_satellite_saturated": True},
            "invest": {"risk_budget_scale": 0.5, "status": "WAIT", "funding_from": "", "slots_used": 0, "slots_cap": 1},
            "regime": "NEUTRAL",
            "global_status": "WAIT",
        }
        md = _FakeMarketData()
        try:
            (held_histories, held_bundles) = await md.get_bundles(
                portfolio.unique_yahoo_symbols(include_pending=True),
                include_news=False,
                include_earnings=True,
                include_info=True,
            )
            text, picks = await _post_screen_validate_decision(
                decision,
                portfolio=portfolio,
                macro=_neutral_macro(),
                cfg=cfg,
                market_data=md,
                held_bundles=held_bundles,
                held_histories=held_histories,
                evals=[],
                isa_cands=[],
                invest_cands=[],
                guardrails=[],
                llm=None,
                deployment_gate=gate,
            )
        finally:
            await md.close()
        # 0.5R = £42.5 budget -> 42.5 / 5% stop = £850 cap (full 1R would be £1,700)
        self.assertIn("预算£42", text)
        self.assertIn("£850", text)


# ===========================================================================
# Group 9: Carbon-HIGH audit regression suite
# ===========================================================================
class TestPostAuditRegressions(unittest.IsolatedAsyncioTestCase):
    def test_persisted_gate_dict_carries_streak_and_skips_same_day_rerun(self) -> None:
        portfolio = _real_snapshot_portfolio()
        strat = StrategySection()
        histories = {"SPY": _spy_neutral_hist()}

        # Day 1 (2026-11-18): VIX=30 -> RISK_OFF
        d1 = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            histories=histories,
            macro=_neutral_macro(vix=30.0),
            now_uk=datetime(2026, 11, 18, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(d1["regime"], "RISK_OFF")

        # Day 2 morning (2026-11-19 07:30): VIX=21 -> ease_streak=1, stays RISK_OFF
        d2 = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            histories=histories,
            macro=_neutral_macro(vix=21.0),
            prev_gate_states=[d1],
            now_uk=datetime(2026, 11, 19, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(d2["regime"], "RISK_OFF")
        self.assertEqual(d2["ease_streak_days"], 1)

        # Day 2 afternoon manual /report re-run (2026-11-19 14:00): d2 is newest in DB,
        # must skip same-day d2 and anchor on d1 -> ease_streak remains 1 (NOT 2!)
        d2_rerun = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            histories=histories,
            macro=_neutral_macro(vix=21.0),
            prev_gate_states=[d2, d1],
            now_uk=datetime(2026, 11, 19, 14, 0, tzinfo=LONDON),
        )
        self.assertEqual(d2_rerun["ease_streak_days"], 1)

        # Day 3 & Day 4: persisted gate dicts increment ease_streak 2 -> 3 and exit RISK_OFF!
        d3 = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            histories=histories,
            macro=_neutral_macro(vix=21.0),
            prev_gate_states=[d2_rerun, d1],
            now_uk=datetime(2026, 11, 20, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(d3["regime"], "RISK_OFF")
        self.assertEqual(d3["ease_streak_days"], 2)

        d4 = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            histories=histories,
            macro=_neutral_macro(vix=21.0),
            prev_gate_states=[d3, d2_rerun, d1],
            now_uk=datetime(2026, 11, 21, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(d4["ease_streak_days"], 3)
        self.assertEqual(d4["regime"], "NEUTRAL")

    def test_loss_cooldown_anchors_most_recent_loss_and_keeps_half_risk_with_old_amzn(self) -> None:
        portfolio = _real_snapshot_portfolio()
        # Keep pre-existing financed AMZN (opened before the loss streak), remove CF
        portfolio.invest.positions = [p for p in portfolio.invest.positions if p.yahoo_symbol == "AMZN"]
        portfolio.invest.recent_orders = [
            # Loss #1 closed 7 days before Nov 26 (Nov 19)
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 18, 10, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 19, 15, 0, tzinfo=LONDON), pnl=-35.0),
            # Loss #2 closed 1 day before Nov 26 (Nov 25) -> within 5-day cooldown!
            _buy_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 24, 15, 0, tzinfo=LONDON)),
            _sell_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 25, 15, 0, tzinfo=LONDON), pnl=-40.0),
        ]
        gate_during = _run_snapshot_gate(
            portfolio,
            strategy=StrategySection(gate_epoch_date="2026-11-18"),
            now_uk=datetime(2026, 11, 26, 9, 0, tzinfo=LONDON),
        )
        self.assertTrue(gate_during["invest"]["in_loss_cooldown"])
        self.assertEqual(gate_during["invest"]["status"], "WAIT")

        # 6 days after Loss #2 (Dec 1): cooldown expired, pre-existing AMZN must NOT cancel 0.5R pilot
        gate_after = _run_snapshot_gate(
            portfolio,
            strategy=StrategySection(gate_epoch_date="2026-11-18"),
            now_uk=datetime(2026, 12, 1, 16, 0, tzinfo=LONDON),
        )
        self.assertFalse(gate_after["invest"]["in_loss_cooldown"])
        self.assertEqual(gate_after["invest"]["risk_budget_scale"], 0.5)

    def test_stale_equity_orders_20d_old_do_not_freeze_7d_cadence_quota(self) -> None:
        portfolio = _real_snapshot_portfolio()
        now_dt = datetime(2026, 12, 10, 9, 0, tzinfo=LONDON)
        portfolio.fetched_at_ms = int(now_dt.timestamp() * 1000)
        # Two equity buys 20 days ago, zero trades in the last 7 days
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "AAPL", "AAPL", at_uk=datetime(2026, 11, 20, 15, 0, tzinfo=LONDON)),
            _buy_order("Invest", "GOOG", "GOOG", at_uk=datetime(2026, 11, 20, 16, 0, tzinfo=LONDON)),
        ]
        msft_bundle = TickerMarketBundle(symbol="MSFT", quote_type="EQUITY", currency="USD", current_price=400.0)
        pick = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="MSFT", action="BUY", thesis="右侧突破",
                    entry_zone="400", stop_condition="380", target_window="460",
                    entry_price=400.0, stop_price=380.0, target_price=460.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        res = apply_risk_override_guardrail(
            pick,
            portfolio=portfolio,
            strategy=StrategySection(trade_cadence_guard_mode="hard", max_weekly_new_buys=2),
            bundle_map={"MSFT": msft_bundle},
            macro=_neutral_macro(),
            trade_behavior={"buys_7d": 0, "buys_30d": 2, "partial": False, "recently_closed": [], "cooldown_days": 15},
        )
        self.assertFalse(res.applied)

    def test_etf_candidates_excluded_from_invest_a_trigger_scan(self) -> None:
        portfolio = _real_snapshot_portfolio()
        # Free CF slot so Invest has 0/1 slots open
        portfolio.invest.positions = [p for p in portfolio.invest.positions if p.yahoo_symbol == "AMZN"]
        swda_cand = _up_candidate("SWDA.L")
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={"SWDA.L": TickerMarketBundle(symbol="SWDA.L", quote_type="ETF", currency="GBP", current_price=88.0)},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(),
            candidates=[swda_cand],
            universe_etf_symbols=["VWRP.L", "SWDA.L"],
        )
        self.assertEqual(gate["invest"]["status"], "WAIT")
        self.assertEqual(gate["triggers"]["A"]["equity_candidates"], [])

    async def test_pick_log_mode_bypasses_gate_interception(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _sample_portfolio
        from src.agents.pick_agent import _post_screen_validate_decision

        portfolio = _sample_portfolio()
        cfg = AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            strategy=StrategySection(deployment_gate_mode="log", overlap_correlation_threshold=1.0),
        )
        decision = PickDecision(
            headline="影子模式放行",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="AZN.L", action="BUY", thesis="护城河医药",
                    entry_zone="13400 GBp", stop_condition="12800 GBp", target_window="15000 GBp",
                    entry_price=13400.0, stop_price=12800.0, target_price=15000.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        gate = {"isa": {"isa_satellite_saturated": True}, "invest": {"status": "WAIT"}, "regime": "NEUTRAL"}
        md = _FakeMarketData()
        try:
            (held_histories, held_bundles) = await md.get_bundles(
                portfolio.unique_yahoo_symbols(include_pending=True),
                include_news=False,
                include_earnings=True,
                include_info=True,
            )
            text, picks = await _post_screen_validate_decision(
                decision,
                portfolio=portfolio,
                macro=_neutral_macro(),
                cfg=cfg,
                market_data=md,
                held_bundles=held_bundles,
                held_histories=held_histories,
                evals=[],
                isa_cands=[],
                invest_cands=[],
                guardrails=[],
                llm=None,
                deployment_gate=gate,
            )
        finally:
            await md.close()
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["symbol"], "AZN.L")


class TestReportSizeHint(unittest.TestCase):
    """v5.4.1: /report must state action size; share counts are derived in Python, never by the LLM."""

    def _position(self) -> PositionItem:
        return _pos(
            "ISA",
            "GOOGL",
            "GOOGL",
            value=3000.0,
            qty=10.0,
            avg=250.0,
            cur=300.0,
            return_pct=20.0,
            weight_pct=5.0,
        )

    def test_trim_percent_and_fraction_convert_to_shares(self) -> None:
        pos = self._position()
        self.assertEqual(derive_action_size_note("TRIM", "30%", pos), "≈3股")
        self.assertEqual(derive_action_size_note("TRIM", "1/4", pos), "≈2.50股")

    def test_trim_gbp_amount_uses_position_gbp_per_share(self) -> None:
        pos = self._position()  # 3000 GBP / 10 shares = 300 GBP per share
        self.assertEqual(derive_action_size_note("TRIM", "£600", pos), "≈2股")

    def test_full_exit_and_stop_semantics(self) -> None:
        pos = self._position()
        self.assertEqual(derive_action_size_note("TRIM", "清仓", pos), "")
        self.assertEqual(derive_action_size_note("STOP", "", pos), "触发即全出")
        self.assertEqual(derive_action_size_note("STOP", "全出", pos), "")

    def test_add_percentage_is_never_guessed(self) -> None:
        self.assertEqual(derive_action_size_note("ADD", "30%", self._position()), "")
        self.assertEqual(derive_action_size_note("HOLD", "30%", self._position()), "")

    def test_render_shows_hint_and_derived_shares(self) -> None:
        dec = MorningReportDecision(
            headline="数量口径测试",
            top_actions=[
                ReportActionItem(
                    account="ISA",
                    symbol="GOOGL",
                    action="TRIM",
                    price_zone="285-295 USD",
                    size_hint="30%",
                    reason="降科技集中度",
                )
            ],
            daily_vetoes=["冻结科技"],
            macro_and_cash="现金待命。",
            account_diagnosis="ISA 个股超限。",
            hold_summary="其余持有",
            trade_review="近30天平稳",
        )
        notes = build_report_size_notes(dec.top_actions, SimpleNamespace(all_positions=lambda: [self._position()]))
        self.assertEqual(notes, {("ISA", "GOOGL"): "≈3股"})
        rendered = render_morning_report(dec, max_chars=1800, size_notes=notes)
        self.assertIn("TRIM GOOGL @ 285-295 USD · 30% (≈3股)", rendered)

    def test_render_without_notes_keeps_legacy_line(self) -> None:
        dec = MorningReportDecision(
            headline="无数量",
            top_actions=[ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="", reason="观望")],
            macro_and_cash="现金。",
            account_diagnosis="无。",
            hold_summary="持有",
            trade_review="平稳",
        )
        rendered = render_morning_report(dec, max_chars=1800)
        self.assertIn("HOLD CASH — 观望", rendered)


class TestV550PostAuditHardening(unittest.TestCase):
    """v5.5.0 Phase 1: financed coverage, Invest WAIT none, pseudo-core ETF demotion."""

    def test_financed_unlock_requires_full_share_coverage(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions = [
            _pos(
                "Invest", "AMZN", "AMZN", value=2500.0, qty=10.0, avg=238.42,
                cur=250.0, return_pct=4.9, weight_pct=17.6,
            )
        ]
        portfolio.invest.pending_orders = [
            PendingOrderItem(
                id="p-amzn-stop-partial", account="Invest", raw_ticker="AMZN_US_EQ",
                ticker="AMZN", yahoo_symbol="AMZN", name="Amazon", side="SELL",
                order_type="STOP", quantity=6.0, value=0.0, limit_price=None,
                stop_price=250.0, currency="USD", time_in_force="GTC",
            )
        ]
        gate = _run_snapshot_gate(portfolio)
        invest = gate["invest"]
        self.assertNotIn("AMZN", invest["financed_positions"])
        self.assertIn("AMZN", invest["unfinanced_positions"])
        self.assertEqual(invest["slots_used"], 1)
        self.assertGreater(float(invest["heat_gbp_total"]), 0.0)

    def test_invest_wait_mode_none_blocks_pick_equity_buy(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {
            "isa": {"etf_open": True, "equity_open": True},
            "invest": {"status": "WAIT", "mode": "none", "reason": "无右侧猎物"},
            "etf_set": ["VWRP.L"],
        }
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="AXON", action="BUY", thesis="想买新个股",
                    entry_zone="300", stop_condition="285", target_window="340",
                    entry_price=300.0,
                ),
                PickActionItem(
                    account="Invest", symbol="IUKD.L", action="BUY", thesis="想买高息主题ETF",
                    entry_zone="7", stop_condition="6.5", target_window="8",
                    entry_price=7.0,
                ),
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertEqual(len(notes), 2)
        self.assertTrue(all("Invest WAIT/none" in n for n in notes))

    def test_one_pick_run_cannot_outrun_invest_slot_and_daily_caps(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {
            "isa": {"etf_open": True, "equity_open": True},
            "invest": {
                "status": "RUN", "mode": "targeted", "slots_cap": 1, "slots_used": 0,
                "daily_cap": 1, "today_new_equity_entries": 0,
            },
            "etf_set": ["VWRP.L"],
        }
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="PANW", action="BUY", thesis="第一笔",
                    entry_zone="300", stop_condition="285", target_window="340",
                    entry_price=300.0,
                ),
                PickActionItem(
                    account="Invest", symbol="AXON", action="BUY", thesis="第二笔",
                    entry_zone="500", stop_condition="470", target_window="580",
                    entry_price=500.0,
                ),
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual([a.symbol for a in hardened.selected_actions], ["PANW"])
        self.assertTrue(any("额度已用尽" in n for n in notes))

    def test_themed_and_dividend_etfs_lose_core_privilege(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(),
            universe_etf_symbols=[
                "VWRP.L", "VWRL.L", "VUAG.L", "SWDA.L",
                "IUKD.L", "VHYL.L", "SMGB.L", "INRG.L", "IITU.L", "IGLT.L", "SGLN.L", "CNX1.L",
            ],
        )
        self.assertEqual(
            gate["etf_set"], ["SWDA.L", "VUAG.L", "VWRL.L", "VWRP.L"]
        )

    def test_vix_30_does_not_override_core_etf_buy_in_guardrail(self) -> None:
        strat = StrategySection(pick_rr_guard_mode="off", trade_cadence_guard_mode="off")
        payload = dict(
            strategy=strat,
            bundle_map={
                "VWRP.L": TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP", current_price=114.0),
                "IUKD.L": TickerMarketBundle(symbol="IUKD.L", quote_type="ETF", currency="GBP", current_price=7.0),
                "MSFT": TickerMarketBundle(symbol="MSFT", quote_type="EQUITY", currency="USD", current_price=400.0),
            },
            macro=MacroSnapshot(fetched_at_utc="2026-11-18T07:25:00Z", vix=30.0),
            deployment_gate={"etf_set": ["VWRP.L"]},
            trade_behavior={"partial": True},
        )
        decision = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="VWRP.L", action="ADD", thesis="危机期核心宽基定投",
                    entry_zone="114", stop_condition="105", target_window="140", entry_price=114.0,
                ),
                PickActionItem(
                    account="ISA", symbol="IUKD.L", action="BUY", thesis="高息主题ETF",
                    entry_zone="7", stop_condition="6.5", target_window="8", entry_price=7.0,
                ),
                PickActionItem(
                    account="ISA", symbol="MSFT", action="BUY", thesis="个股",
                    entry_zone="400", stop_condition="380", target_window="460", entry_price=400.0,
                ),
            ],
            consistency_and_cash_note="测试",
        )
        res = apply_risk_override_guardrail(decision, **payload)
        kept = [a.symbol for a in res.decision.selected_actions]
        self.assertEqual(kept, ["VWRP.L"])
        blocked = {o.symbol: o.trigger_type for o in res.overrides}
        self.assertEqual(blocked.get("IUKD.L"), "MACRO_VIX")
        self.assertEqual(blocked.get("MSFT"), "MACRO_VIX")

    def test_bear_bounce_below_ma200_stays_risk_off_and_recovery_above_ma200_exits(self) -> None:
        strat = StrategySection()
        # Long 150 plateau -> -27% crash -> 20-day dead-cat bounce: mom20>0 but still below MA200.
        bounce_closes = [150.0] * 300
        bounce_closes += [round(150.0 * (1.0 - 0.015 * (k + 1)), 2) for k in range(20)]
        trough = bounce_closes[-1]
        bounce_closes += [round(trough * (1.0 + 0.0065 * (k + 1)), 2) for k in range(20)]
        bounce_hist = _make_hist(bounce_closes)
        prev = {"regime": "RISK_OFF", "raw_regime": "RISK_OFF", "raw_streak_days": 5, "ease_streak_days": 2}
        bounce = evaluate_market_regime({"SPY": bounce_hist}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev)
        self.assertEqual(bounce["regime"], "RISK_OFF")
        self.assertGreater(float(bounce["mom20_pct"] or 0.0), 0.0)
        self.assertLess(float(bounce["price"]), float(bounce["ma200"]))
        self.assertEqual(bounce["ease_streak_days"], 0)

        # -25% bottom then reclaim MA200 for 3 days while the stale 52w drawdown flag remains.
        recovery_closes = [100.0] * 300
        recovery_closes += [round(100.0 + 2.5 * (k + 1), 2) for k in range(20)]  # spike to 150
        recovery_closes += [round(150.0 - 1.5 * (k + 1), 2) for k in range(20)]  # crash to 120
        recovery_closes += [120.0] * 60  # sit at -20% off the old high, above MA200
        recovery_hist = _make_hist(recovery_closes)
        prev = {"regime": "RISK_OFF", "raw_regime": "RISK_OFF", "raw_streak_days": 5, "ease_streak_days": 2}
        recovery = evaluate_market_regime(
            {"SPY": recovery_hist}, _neutral_macro(vix=21.0), strat, prev_gate_state=prev
        )
        self.assertEqual(recovery["regime"], "NEUTRAL")
        self.assertEqual(recovery["ease_streak_days"], 3)
        self.assertTrue(all("回撤" in f for f in recovery["risk_off_flags"]))

    def test_persistent_drawdown_keeps_etf_ladder_open_after_first_cross_day(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.isa.pending_orders = [
            o for o in portfolio.isa.pending_orders if o.yahoo_symbol not in {"VWRP.L", "VUAG.L"}
        ]
        # SPY sits at a persistent -12% off its 52w high with no first-cross today.
        closes = [round(100.0 + i * 1.0, 2) for i in range(250)]
        peak = max(closes)
        closes += [round(peak * 0.88, 2)] * 6
        spy = _make_hist(closes)
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": spy},
            macro=_neutral_macro(vix=30.0),
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        self.assertEqual(gate["regime"], "RISK_OFF")
        self.assertIsNone(gate["triggers"]["A"]["core_etf_step"]["hit"])
        self.assertEqual(gate["isa"]["status"], "RUN")
        self.assertEqual(gate["isa"]["mode"], "ladder")
        self.assertTrue(gate["isa"]["etf_open"])


class TestV550GateCandidateBundles(unittest.IsolatedAsyncioTestCase):
    """v5.5.0 Phase 3: un-held gate candidates must carry earnings/ex-div bundles."""

    async def test_unheld_pullback_candidate_gets_earnings_bundle_and_is_excluded(self) -> None:
        class _RecordingMarketData:
            def __init__(self) -> None:
                self.calls: list[tuple[list[str], bool, bool]] = []

            async def get_bundles(self, symbols, include_news=True, include_earnings=True, **kwargs):
                self.calls.append((list(symbols), include_news, include_earnings))
                return {}, {
                    s: TickerMarketBundle(
                        symbol=s,
                        quote_type="EQUITY",
                        currency="USD",
                        current_price=300.0,
                        days_to_earnings=2,
                    )
                    for s in symbols
                }

        md = _RecordingMarketData()
        snaps = {
            "PANW": TechnicalSnapshot(
                symbol="PANW", last_close=300.0, ma50=297.0, ma200=270.0,
                dist_ma50_pct=1.0, rsi14=55.0, trend_regime="BULL",
            ),
            "MSFT": TechnicalSnapshot(
                symbol="MSFT", last_close=400.0, ma50=380.0, ma200=350.0,
                dist_ma50_pct=5.0, rsi14=60.0, trend_regime="BULL",
            ),
            "ZZZ": TechnicalSnapshot(symbol="ZZZ", last_close=10.0),
        }
        held_bundles = {"MSFT": TickerMarketBundle(symbol="MSFT", quote_type="EQUITY", currency="USD")}
        out = await _enrich_gate_candidate_bundles(
            md, ["PANW", "MSFT", "ZZZ"], snaps, held_bundles
        )
        # Only the un-held MA50 pullback is fetched; held names are never re-fetched.
        self.assertEqual(md.calls, [(["PANW"], False, True)])
        self.assertEqual(out["PANW"].days_to_earnings, 2)
        self.assertIs(out["MSFT"], held_bundles["MSFT"])

        hits = _scan_a_equity_candidates(
            [{"symbol": "PANW", "tech": snaps["PANW"], "bundle": out["PANW"], "account": "Invest"}],
            _real_snapshot_portfolio(),
            StrategySection(),
            None,
        )
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
