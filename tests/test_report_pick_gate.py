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
from src.agents.report_agent import _build_report_recommendations, _enrich_gate_candidate_bundles, _validate_core_deployment
from src.agents.report_agent.core_deployment import _format_core_policy_ticket_block
from src.agents.report_agent.morning_report import _build_regime_anchor_text
from src.agents.schemas import (
    CoreDeploymentItem,
    MorningReportDecision,
    PendingOrderAuditItem,
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


def _gentle_core_hist(n: int = 260, base: float = 100.0) -> pd.DataFrame:
    """Oscillating history: price hugs MA200, mild 52w drawdown, RSI ~50 (no trend guard)."""
    closes = [round(base + 3.0 * np.sin(i / 9.0) + i * 0.01, 4) for i in range(n)]
    return _make_hist(closes)


def _revised_plan_snapshot_portfolio() -> PortfolioData:
    """Exact 2026-11-18 18:39 UK live snapshot from the revised v5.5.1 blueprint (§2.1)."""
    satellite_vals = {
        "GOOGL": 2067.0, "PANW": 1052.0, "CRWD": 789.0, "VST": 1200.0,
        "TSLA": 1500.0, "AMZN": 453.0, "UBER": 400.0,
    }
    remaining = 19674.23 - sum(satellite_vals.values())
    isa_equities = [
        _pos("ISA", sym, sym, value=val, qty=10.0, avg=val / 10.0, cur=val / 10.0,
             return_pct=0.0, weight_pct=0.0)
        for sym, val in satellite_vals.items()
    ]
    isa_equities.append(
        _pos("ISA", "MSFT", "MSFT", value=remaining, qty=10.0, avg=remaining / 10.0,
             cur=remaining / 10.0, return_pct=0.0, weight_pct=0.0)
    )
    isa_positions = [
        _pos("ISA", "VWRP", "VWRP.L", value=11654.06, qty=100.0, avg=116.5406,
             cur=116.5406, return_pct=0.0, weight_pct=20.33, ccy="GBP"),
        _pos("ISA", "VUAG", "VUAG.L", value=8277.57, qty=80.0, avg=85.0,
             cur=103.4696, return_pct=21.72, weight_pct=14.44, ccy="GBP"),
        _pos("ISA", "IBTA", "IBTA.L", value=305.94, qty=55.0, avg=5.56,
             cur=5.5625, return_pct=2.13, weight_pct=0.53, ccy="GBP"),
        *isa_equities,
    ]
    isa_pending = [
        PendingOrderItem(
            id="p-vwrp-1", account="ISA", raw_ticker="VWRP_LSE_ETF", ticker="VWRP",
            yahoo_symbol="VWRP.L", name="VWRP", side="BUY", order_type="LIMIT",
            quantity=25.0, value=3337.50, limit_price=133.50, currency="GBP",
            time_in_force="GTC",
        ),
        PendingOrderItem(
            id="p-vwrp-2", account="ISA", raw_ticker="VWRP_LSE_ETF", ticker="VWRP",
            yahoo_symbol="VWRP.L", name="VWRP", side="BUY", order_type="LIMIT",
            quantity=25.0, value=3405.00, limit_price=136.20, currency="GBP",
            time_in_force="GTC",
        ),
    ]
    return PortfolioData(
        fetched_at_uk="2026-11-18 18:39",
        fetched_at_ms=1763487540000,
        total_combined_value=71526.78,
        total_combined_cash=30318.57,
        isa=AccountSnapshot(
            ok=True, account_label="ISA", total_account_value=57330.94,
            investments_value=39911.20, investments_cost=34000.0, unrealized_pnl=2481.23,
            total_cash=17419.74, available_to_trade=10677.24, reserved_cash=6742.50,
            cash_ratio_pct=30.38, positions=isa_positions, pending_orders=isa_pending,
        ),
        invest=AccountSnapshot(
            ok=True, account_label="Invest", total_account_value=14195.84,
            investments_value=1297.01, investments_cost=1192.10, unrealized_pnl=104.91,
            total_cash=12898.83, available_to_trade=12898.83, reserved_cash=0.0,
            cash_ratio_pct=90.86,
            positions=[
                _pos("Invest", "AMZN", "AMZN", value=1297.01, qty=5.0, avg=238.42,
                     cur=259.402, return_pct=8.8, weight_pct=9.14, ccy="USD"),
            ],
            pending_orders=[
                PendingOrderItem(
                    id="p-amzn-stop", account="Invest", raw_ticker="AMZN_US_EQ", ticker="AMZN",
                    yahoo_symbol="AMZN", name="Amazon", side="SELL", order_type="STOP",
                    quantity=5.0, value=0.0, stop_price=243.50, currency="USD",
                    time_in_force="GTC",
                ),
            ],
        ),
        gbp_usd_rate=1.25,
    )


def _run_revised_snapshot_gate() -> dict:
    histories = {
        "SPY": _spy_neutral_hist(),
        "VWRP.L": _gentle_core_hist(base=116.0),
        "VUAG.L": _gentle_core_hist(base=103.0),
        "VERG.L": _gentle_core_hist(base=35.0),
        "ISF.L": _gentle_core_hist(base=8.2),
    }
    bundles = {
        sym: TickerMarketBundle(symbol=sym, quote_type="ETF", currency="GBP", current_price=price)
        for sym, price in {
            "VWRP.L": 116.54, "VUAG.L": 103.47, "VERG.L": 35.20,
            "ISF.L": 8.20, "IBTA.L": 5.56,
        }.items()
    }
    return evaluate_deployment_gate(
        portfolio=_revised_plan_snapshot_portfolio(),
        strategy=StrategySection(),
        bundles=bundles,
        histories=histories,
        macro=_neutral_macro(),
        universe_etf_symbols=["VWRP.L", "VUAG.L", "SWDA.L", "VERG.L", "ISF.L", "CUKX.L"],
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
            dist_ma50_pct=0.8, dist_ma200_pct=11.1, rsi14=60.0, trend_regime="BULL",
            weekly_bullish=True,
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
    def test_neutral_two_slots_allow_second_entry_when_heat_is_cool(self) -> None:
        # v5.5.5: NEUTRAL grants 2 risk slots; CF heat (~0.8%) stays under the 2.0% heat cap.
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        invest = gate["invest"]
        self.assertAlmostEqual(invest["cash_pct"], 79.7, places=1)
        self.assertEqual(invest["slots_cap"], 2)
        self.assertEqual(invest["slots_used"], 1)
        self.assertLess(invest["committed_heat_pct"], 2.0)
        self.assertEqual(invest["status"], "RUN")
        self.assertEqual(invest["mode"], "targeted")
        # Never a dca-style "spend the idle cash" RUN for Invest
        self.assertNotEqual(invest["mode"], "dca")

    def test_neutral_heat_full_blocks_despite_free_slot(self) -> None:
        portfolio = _real_snapshot_portfolio()
        for pos in portfolio.invest.positions:
            if pos.yahoo_symbol == "CF":
                pos.quantity = 50.0
                pos.current_value = pos.quantity * pos.cur_price
        portfolio.invest.pending_orders = [
            o for o in portfolio.invest.pending_orders if o.yahoo_symbol != "CF"
        ]
        gate = _run_snapshot_gate(portfolio)
        invest = gate["invest"]
        self.assertEqual(invest["slots_used"], 1)
        self.assertEqual(invest["slots_cap"], 2)
        self.assertGreaterEqual(invest["committed_heat_pct"], 2.0)
        self.assertEqual(invest["heat_cap_pct"], 2.0)
        self.assertEqual(invest["status"], "WAIT")
        self.assertEqual(invest["mode"], "heat_full")
        self.assertIn("承诺热力", invest["reason"])

        rotated = _run_snapshot_gate(portfolio, candidates=[_up_candidate()])
        self.assertEqual(rotated["invest"]["status"], "ROTATE")
        self.assertEqual(rotated["invest"]["mode"], "rotate")
        self.assertIn("承诺热力", rotated["invest"]["reason"])

    def test_pending_new_equity_buy_commits_one_r_of_forward_heat(self) -> None:
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.pending_orders.append(
            PendingOrderItem(
                id="p-de", account="Invest", raw_ticker="DE_US_EQ", ticker="DE", yahoo_symbol="DE",
                name="Deere", side="BUY", order_type="LIMIT", quantity=2.0, value=800.0,
                limit_price=400.0, currency="USD", time_in_force="GTC",
            )
        )
        gate = _run_snapshot_gate(portfolio)
        invest = gate["invest"]
        self.assertEqual(invest["pending_new_equity_buys"], ["DE"])
        self.assertAlmostEqual(invest["pending_heat_gbp"], invest["unit_risk_gbp"], places=2)
        self.assertAlmostEqual(
            invest["committed_heat_gbp"],
            round(invest["heat_gbp_total"] + invest["pending_heat_gbp"], 2),
            places=2,
        )
        self.assertGreater(invest["committed_heat_pct"], 0.8)
        # The pending buy also consumes a slot, so 1 held + 1 pending = 2/2 -> full.
        self.assertEqual(invest["slots_used"], 2)
        self.assertEqual(invest["status"], "WAIT")
        self.assertIn(invest["mode"], {"heat_full", "slots_full"})

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
        gate = _run_snapshot_gate(
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
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

    def test_risk_on_regime_grants_three_slots(self) -> None:
        prev = {"regime": "NEUTRAL", "raw_regime": "RISK_ON", "raw_streak_days": 2, "ease_streak_days": 0}
        gate = _run_snapshot_gate(
            spy_hist=_spy_risk_on_hist(),
            prev_gate_states=[prev],
            candidates=[_up_candidate()],
        )
        self.assertEqual(gate["regime"], "RISK_ON")
        invest = gate["invest"]
        self.assertEqual(invest["slots_cap"], 3)
        self.assertEqual(invest["status"], "RUN")  # 1/3 slots used, candidate present
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
        self.assertEqual(invest["heat_cap_pct"], 0.0)
        self.assertEqual(invest["heat_headroom_pct"], 0.0)
        self.assertTrue(gate["triggers"]["D"]["risk_off_defend"])
        self.assertIn("撤除 Invest 未成交个股买单", invest.get("defend_note", ""))
        self.assertIn("ISA 低位价值买单须核查基本面", invest.get("defend_note", ""))

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

    def test_isa_crisis_value_channel_and_invest_freeze_during_risk_off(self) -> None:
        def _trimmed_portfolio():
            p = _real_snapshot_portfolio()
            for order in list(p.isa.pending_orders):
                if order.yahoo_symbol in {"SHEL.L", "RKT.L"}:
                    p.isa.pending_orders.remove(order)
            p.isa.positions = p.isa.positions[:5]  # only big 5, no saturation
            return p

        portfolio = _trimmed_portfolio()
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
        # v5.5.1: ISA keeps a quality + >=20% deep-discount contrarian lane open in RISK_OFF,
        # while the wide-base ETF ladder itself still needs a valid drawdown step.
        self.assertTrue(isa["value_crisis_mode"])
        self.assertTrue(isa["equity_open"])
        self.assertFalse(isa["etf_open"])
        self.assertIn("深度安全边际价值分批通道", isa["reason"])
        self.assertEqual(gate["invest"]["status"], "DEFEND")
        self.assertEqual(gate["invest"]["slots_cap"], 0)

        # Branch 2: explicitly disabling the crisis value lane re-closes the ISA equity channel.
        off_strategy = StrategySection().model_copy(update={"isa_value_crisis_buy_enabled": False})
        gate_off = evaluate_deployment_gate(
            portfolio=_trimmed_portfolio(),
            strategy=off_strategy,
            bundles={},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(vix=30.0),
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        self.assertFalse(gate_off["isa"]["value_crisis_mode"])
        self.assertFalse(gate_off["isa"]["equity_open"])
        self.assertEqual(gate_off["invest"]["status"], "DEFEND")

        # Branch 3: extreme low cash (<2%) also shuts the lane.
        portfolio_low = _trimmed_portfolio()
        portfolio_low.isa.total_cash = 500.0
        gate_low = evaluate_deployment_gate(
            portfolio=portfolio_low,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(vix=30.0),
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        self.assertFalse(gate_low["isa"]["equity_open"])
        self.assertEqual(gate_low["invest"]["status"], "DEFEND")

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
        gate = _run_snapshot_gate(
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
        self.assertEqual(gate["invest"]["status"], "ROTATE")
        # The deterministic engine already computed a legal funding source (CF); an
        # untouched PM funding_from falls back to it instead of falsely degrading to WAIT.
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "funding_from": ""}, whitelist_symbols={"CF"})
        self.assertTrue(pm["allow"])
        self.assertEqual(pm["mode"], "rotate")
        self.assertEqual(pm["funding_from"], "CF")

    def test_rotate_with_illegal_funding_is_rejected(self) -> None:
        gate = _run_snapshot_gate(
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
        pm = reconcile_pm_pick_gate(gate, {"allow": True, "funding_from": "TSLA"}, whitelist_symbols={"CF"})
        self.assertFalse(pm["allow"])
        self.assertIn("置换源", pm["reason"])

    def test_rotate_with_valid_funding_is_kept(self) -> None:
        gate = _run_snapshot_gate(
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
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

    def test_run_banner_shows_dynamic_unit_risk_and_heat(self) -> None:
        gate = _run_snapshot_gate(candidates=[_up_candidate()])
        invest = gate["invest"]
        self.assertEqual(invest["status"], "RUN")
        banner = gate["banner"]
        self.assertLessEqual(len(banner), 110)
        self.assertNotIn("风险预算£85", banner)
        self.assertIn(f"单笔风险£{invest['unit_risk_gbp']:.0f}", banner)
        self.assertIn("热力", banner)
        self.assertIn(f"{float(invest['committed_heat_pct']):.1f}%", banner)

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

    def test_default_report_ceiling_is_2800(self) -> None:
        moderate = MorningReportDecision(
            headline="2800字默认天花板验证",
            top_actions=[],
            daily_vetoes=["控制集中度"],
            macro_and_cash="宏" * 2300,
            account_diagnosis="",
            flagged_positions=[],
            hold_summary="",
            trade_review="",
        )
        rendered = render_morning_report(moderate)
        self.assertGreater(len(rendered), 2200)
        self.assertLessEqual(len(rendered), 2800)

        oversized = moderate.model_copy(update={"macro_and_cash": "宏" * 5000})
        capped = render_morning_report(oversized)
        self.assertLessEqual(len(capped), 2800)
        self.assertTrue(capped.startswith("【今日核心行动与挂单裁决】"))

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
    def test_guardrail_notes_render_in_dedicated_section_under_1000(self) -> None:
        from src.agents.schemas import render_pick_report
        from src.engine.portfolio_rules import RiskOverrideResult

        dec = PickDecision(
            headline="半导体回踩分批",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="TSM", action="BUY", thesis="晶圆龙头",
                    entry_zone="180", stop_condition="165", target_window="210", entry_price=180.0,
                )
            ],
            consistency_and_cash_note="ISA 现金 22%",
        )
        res = RiskOverrideResult(
            decision=dec,
            applied=True,
            overrides=[],
            notes=["[🛡️风控护栏修正] [ISA] ISF.L(BUY→拦截): 现金门槛不足"],
        )
        rendered = render_pick_report(dec, max_chars=1000, risk_override=res)
        self.assertLessEqual(len(rendered), 1000)
        self.assertTrue(rendered.startswith("【本轮精选行动（四要素直给）】"))
        self.assertIn("【风控护栏记录】", rendered)
        guard_block = rendered.split("【风控护栏记录】", 1)[1]
        self.assertIn("[🛡️风控护栏修正]", guard_block)
        # Guardrail rows must no longer sit inside the conclusion block before this header.
        conclusion_block = rendered.split("【风控护栏记录】", 1)[0]
        self.assertNotIn("[🛡️风控护栏修正]", conclusion_block)
        self.assertNotIn("**", rendered)
        self.assertNotIn("#", rendered)

    def test_gate_constraint_block_renders_saturation_and_rotate(self) -> None:
        gate = _run_snapshot_gate(
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
        block = _format_gate_constraint_block(gate)
        self.assertIn("今日组合门控约束", block)
        self.assertIn("ISA 个股卫星层已饱和", block)
        self.assertIn("ROTATE 等风险置换", block)
        self.assertIn("CF", block)
        self.assertIn("承诺热力", block)
        self.assertIn("单笔风险预算", block)

    def test_run_gate_with_heat_exhausted_blocks_new_invest_equity(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {
            "isa": {"equity_open": True},
            "invest": {
                "status": "RUN",
                "mode": "targeted",
                "slots_used": 1,
                "slots_cap": 2,
                "committed_heat_pct": 2.2,
                "heat_cap_pct": 2.0,
                "reason": "测试热力触顶",
            },
        }
        decision = PickDecision(
            headline="热力兜底测试",
            selected_actions=[
                PickActionItem(
                    account="Invest", symbol="PANW", action="BUY", thesis="右侧突破",
                    entry_zone="300", stop_condition="285", target_window="340", entry_price=300.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate)
        self.assertEqual(hardened.selected_actions, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("承诺热力", notes[0])
        self.assertIn("上移止损", notes[0])

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
        gate = _run_snapshot_gate(
            portfolio,
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
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
        gate = _run_snapshot_gate(
            portfolio,
            strategy=StrategySection(invest_slots_neutral=1),
            candidates=[_up_candidate()],
        )
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
            "invest": {"risk_budget_scale": 0.5, "status": "RUN", "funding_from": "", "slots_used": 0, "slots_cap": 1},
            "regime": "NEUTRAL",
            "global_status": "RUN",
        }
        md = _FakeMarketData()
        try:
            (held_histories, held_bundles) = await md.get_bundles(
                portfolio.unique_yahoo_symbols(include_pending=True),
                include_news=False,
                include_earnings=True,
                include_info=True,
            )
            text, picks, _shadow = await _post_screen_validate_decision(
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
        # 0.5R = £42.5 budget ÷ 含摩擦止损 5.2% (5% price + 0.20% spread) ≈ £817 cap
        # (full 1R with the pure 5% stop would be £1,700).
        self.assertIn("预算£42", text)
        self.assertIn("£817", text)
        self.assertIn("含摩擦止损5.2%", text)
        self.assertIn("≈1股", text)


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
            strategy=StrategySection(
                deployment_gate_mode="log", overlap_correlation_threshold=1.0
            ).model_copy(update={"isa_value_guard_mode": "off"}),  # v5.5.1: keep this case about gate mode only
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
            text, picks, _shadow = await _post_screen_validate_decision(
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
                "VWRP.L", "VWRL.L", "VUAG.L", "SWDA.L", "ISF.L", "VERG.L",
                "IUKD.L", "VHYL.L", "SMGB.L", "INRG.L", "IITU.L", "IGLT.L", "SGLN.L", "CNX1.L",
            ],
        )
        self.assertEqual(
            gate["etf_set"], ["ISF.L", "SWDA.L", "VERG.L", "VUAG.L", "VWRL.L", "VWRP.L"]
        )

    def test_core_diversifiers_count_toward_target_and_soft_preference(self) -> None:
        # Default snapshot: VWRP+VUAG only -> core <60% and 0% diversifier => soft preference.
        gate = _run_snapshot_gate()
        self.assertTrue(gate["isa"]["prefer_core_diversifier"])
        self.assertIn("FTSE100/欧洲", gate["isa"]["reason"])

        # FTSE100 + Europe broad-base now count toward the 60% core target.
        portfolio = _real_snapshot_portfolio()
        portfolio.isa.positions = [
            _pos(
                "ISA", "ISF", "ISF.L", value=20000.0, qty=200.0, avg=100.0, cur=100.0,
                return_pct=0.0, weight_pct=34.9, ccy="GBP",
            ),
            _pos(
                "ISA", "VERG", "VERG.L", value=15000.0, qty=150.0, avg=100.0, cur=100.0,
                return_pct=0.0, weight_pct=26.2, ccy="GBP",
            ),
        ]
        portfolio.isa.pending_orders = []
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=StrategySection(),
            bundles={},
            histories={"SPY": _spy_neutral_hist()},
            macro=_neutral_macro(),
            universe_etf_symbols=["VWRP.L", "VUAG.L", "ISF.L", "VERG.L"],
        )
        self.assertGreaterEqual(gate["isa"]["core_etf_pct"], 60.0)
        self.assertGreaterEqual(gate["isa"]["core_diversifier_pct"], 60.0)
        self.assertFalse(gate["isa"]["prefer_core_diversifier"])
        self.assertFalse(gate["triggers"]["B"]["isa_core_dca"])

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


class TestV551CoreDeployment(unittest.TestCase):
    """v5.5.1: /report can originate un-held core/diversifier ETF BUYs, guarded."""

    @staticmethod
    def _decision_with_core(symbol: str = "VERG.L", account: str = "ISA", amount: str = "£2,000"):
        return MorningReportDecision(
            headline="核心宽基分散部署",
            top_actions=[
                ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="", reason="保留现金")
            ],
            core_deployment=[
                CoreDeploymentItem(
                    account=account,
                    symbol=symbol,
                    price_zone="35.0-35.4 GBP",
                    size_hint=amount,
                    reason="分散美股科技",
                )
            ],
            macro_and_cash="现金充足。",
            account_diagnosis="ISA 核心宽基不足。",
            hold_summary="其余持有",
            trade_review="无",
        )

    def _cfg(self) -> AppConfig:
        return AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            strategy=StrategySection(deployment_gate_mode="advise"),
        )

    def test_core_deployment_renders_and_enters_ledger(self) -> None:
        portfolio = _real_snapshot_portfolio()
        bundles = {
            "VERG.L": TickerMarketBundle(
                symbol="VERG.L", quote_type="ETF", currency="GBP", current_price=35.2
            )
        }
        decision = self._decision_with_core()
        filtered, notes = _validate_core_deployment(
            decision,
            portfolio,
            self._cfg(),
            {"etf_set": ["VWRP.L", "VUAG.L", "ISF.L", "VERG.L"], "isa": {"etf_open": True}},
            bundles,
        )
        self.assertEqual(notes, [])
        self.assertEqual([d.symbol for d in filtered.core_deployment], ["VERG.L"])
        text = render_morning_report(filtered, max_chars=1800)
        self.assertIn("【核心宽基部署】", text)
        self.assertIn("BUY VERG.L", text)
        self.assertLessEqual(len(text), 1800)
        recs = _build_report_recommendations(filtered, portfolio, bundles, self._cfg())
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["symbol"], "VERG.L")
        self.assertEqual(recs[0]["action"], "BUY")
        self.assertEqual(recs[0]["source"], "core_dca")
        self.assertEqual(recs[0]["horizon_days"], 20)
        self.assertEqual(recs[0]["fill_status"], "ASSUMED")

    def test_core_deployment_rejects_non_core_invest_held_and_tiny_tickets(self) -> None:
        portfolio = _real_snapshot_portfolio()  # holds VWRP.L
        bundles = {
            "VERG.L": TickerMarketBundle(symbol="VERG.L", quote_type="ETF", currency="GBP", current_price=35.2),
            "IUKD.L": TickerMarketBundle(symbol="IUKD.L", quote_type="ETF", currency="GBP", current_price=7.0),
            "VWRP.L": TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP", current_price=114.0),
        }
        gate = {"etf_set": ["VWRP.L", "VUAG.L", "ISF.L", "VERG.L"], "isa": {"etf_open": True}}
        cfg = self._cfg()

        non_core, notes = _validate_core_deployment(
            self._decision_with_core(symbol="IUKD.L"), portfolio, cfg, gate, bundles
        )
        self.assertEqual(non_core.core_deployment, [])
        self.assertTrue(any("不属于核心/分散宽基集合" in n for n in notes))

        held, notes = _validate_core_deployment(
            self._decision_with_core(symbol="VWRP.L"), portfolio, cfg, gate, bundles
        )
        self.assertEqual(held.core_deployment, [])
        self.assertTrue(any("已持仓请用 top_actions ADD" in n for n in notes))

        invest, notes = _validate_core_deployment(
            self._decision_with_core(symbol="VERG.L", account="Invest"), portfolio, cfg, gate, bundles
        )
        self.assertEqual(invest.core_deployment, [])
        self.assertTrue(any("仅限 ISA" in n for n in notes))

        tiny, notes = _validate_core_deployment(
            self._decision_with_core(symbol="VERG.L", amount="£100"), portfolio, cfg, gate, bundles
        )
        self.assertEqual(tiny.core_deployment, [])
        self.assertTrue(any("低于 ISA ETF 最小建仓额" in n for n in notes))

        closed, notes = _validate_core_deployment(
            self._decision_with_core(symbol="VERG.L"),
            portfolio,
            cfg,
            {"etf_set": ["VERG.L"], "isa": {"etf_open": False}},
            bundles,
        )
        self.assertEqual(closed.core_deployment, [])
        self.assertTrue(any("宽基通道关闭" in n for n in notes))

    def test_invest_accumulating_core_blocked_but_distributing_and_isa_allowed(self) -> None:
        portfolio = _real_snapshot_portfolio()
        gate = {
            "isa": {
                "etf_open": True, "equity_open": True,
                "deployable_pct": 5.0, "committed_core_etf_pct": 50.0,
            },
            "invest": {
                "status": "RUN", "mode": "targeted", "slots_cap": 10, "slots_used": 0,
                "daily_cap": 10, "today_new_equity_entries": 0,
            },
            "etf_set": ["VWRP.L", "VERG.L"],
        }
        bundles = {
            sym: TickerMarketBundle(symbol=sym, quote_type="ETF", currency="GBP", current_price=10.0)
            for sym in ("VWRP.L", "VERG.L", "VERX.L", "CUKX.L", "ISF.L", "VWRL.L")
        }

        def _buy(account: str, symbol: str) -> PickActionItem:
            return PickActionItem(
                account=account, symbol=symbol, action="BUY", thesis="测试",
                entry_zone="10", stop_condition="9", target_window="12", entry_price=10.0,
            )

        decision = PickDecision(
            headline="测试",
            selected_actions=[
                _buy("Invest", "VWRP.L"),   # accumulating -> hard block
                _buy("Invest", "VERG.L"),   # accumulating -> hard block
                _buy("Invest", "CUKX.L"),   # accumulating FTSE100 -> hard block
                _buy("Invest", "VERX.L"),   # distributing Europe ex-UK -> allowed
                _buy("Invest", "ISF.L"),    # distributing FTSE100 -> allowed
                _buy("Invest", "VWRL.L"),   # distributing all-world -> allowed
                _buy("ISA", "VERG.L"),      # ISA accumulating is tax-sheltered -> allowed
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate, bundles=bundles)
        kept = [a.symbol for a in hardened.selected_actions]
        self.assertEqual(kept, ["VERX.L", "ISF.L", "VWRL.L", "VERG.L"])
        self.assertTrue(any("累积型" in n for n in notes) or any("派息型" in n for n in notes))

        # The same un-held core ETF can be originated by /report for ISA.
        cfg = self._cfg()
        filtered, core_notes = _validate_core_deployment(
            self._decision_with_core(),
            portfolio,
            cfg,
            gate,
            {"VERG.L": TickerMarketBundle(symbol="VERG.L", quote_type="ETF", currency="GBP", current_price=35.0)},
        )
        self.assertEqual(core_notes, [])
        self.assertEqual(len(filtered.core_deployment), 1)


class TestV551RevisedCorePolicy(unittest.TestCase):
    """Revised v5.5.1: deterministic four-bucket policy ticket + PM one-way verdicts."""

    def _cfg(self) -> AppConfig:
        return AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            strategy=StrategySection(deployment_gate_mode="advise"),
        )

    def _decision_with_ticket(
        self,
        *,
        symbol: str | None = None,
        verdict: str = "APPROVE",
        approved_gbp: float | None = None,
        amount: str = "£1,600",
        account: str = "ISA",
    ) -> MorningReportDecision:
        return MorningReportDecision(
            headline="核心宽基政策票据审批",
            top_actions=[
                ReportActionItem(account="ISA", symbol="CASH", action="HOLD", price_zone="", reason="保留现金")
            ],
            core_deployment=[
                CoreDeploymentItem(
                    account=account,
                    symbol=symbol or "VERG.L",
                    verdict=verdict,
                    approved_gbp=approved_gbp,
                    price_zone="34.8-35.4 GBP",
                    size_hint=amount,
                    reason="欧洲宽基补缺口",
                )
            ],
            macro_and_cash="现金充足。",
            account_diagnosis="ISA 核心宽基缺口。",
            hold_summary="其余持有",
            trade_review="无",
        )

    def test_user_2026_11_18_snapshot_freezes_vuag_respects_vwrp_orders_and_tickets_verg_or_isf(self) -> None:
        gate = _run_revised_snapshot_gate()
        isa = gate["isa"]
        policy = isa["core_policy"]

        # 1) VUAG stays inside the 60% denominator; no false liquidation/breach.
        self.assertAlmostEqual(float(isa["committed_core_etf_pct"]), 46.53, delta=0.15)
        # 2) US tilt frozen and the two pending VWRP orders already satisfy the 32% anchor floor.
        self.assertTrue(policy["us_tilt_buy_frozen"])
        self.assertEqual(float(policy["anchor_gap_pct"]), 0.0)
        self.assertGreaterEqual(float(policy["excess_us_tilt_pct"]), 10.0)
        # 3) Only the Europe/UK gaps remain, and exactly one policy ticket is issued.
        self.assertEqual(policy["priority"][0], "EU_DIVERSIFIER")
        self.assertGreaterEqual(float(policy["core_total_gap_gbp"]), 7500.0)
        self.assertLessEqual(float(policy["core_total_gap_gbp"]), 7900.0)
        plan = isa["core_allocation_plan"]
        self.assertEqual(len(plan["tickets"]), 1)
        ticket = plan["tickets"][0]
        self.assertIn(ticket["symbol"], {"VERG.L", "ISF.L"})
        self.assertGreaterEqual(float(ticket["recommended_gbp"]), 1000.0)
        self.assertLessEqual(float(ticket["recommended_gbp"]), 1600.0)
        # 3b) Post-trade cash stays far above the 5% (£2,867) structural floor.
        cash_after = float(_revised_plan_snapshot_portfolio().isa.available_to_trade) - float(
            ticket["recommended_gbp"]
        )
        self.assertGreater(cash_after, 57330.94 * 0.05)
        # 3c) The banner names the deterministic ticket instead of a vague menu.
        self.assertLessEqual(len(gate["banner"]), 110)
        self.assertIn(ticket["symbol"], gate["banner"])

    def test_pm_swap_or_inflate_is_blocked_or_clamped_and_defer_is_not_a_buy(self) -> None:
        portfolio = _revised_plan_snapshot_portfolio()
        cfg = self._cfg()
        bundles = {
            "VERG.L": TickerMarketBundle(symbol="VERG.L", quote_type="ETF", currency="GBP", current_price=35.20),
            "VUAG.L": TickerMarketBundle(symbol="VUAG.L", quote_type="ETF", currency="GBP", current_price=103.47),
            "IUKD.L": TickerMarketBundle(symbol="IUKD.L", quote_type="ETF", currency="GBP", current_price=7.0),
        }

        # PM tries to swap the Python ticket for VUAG / IUKD -> hard intercept.
        for bad in ("VUAG.L", "IUKD.L"):
            gate = _run_revised_snapshot_gate()
            filtered, notes = _validate_core_deployment(
                self._decision_with_ticket(symbol=bad), portfolio, cfg, gate, bundles
            )
            self.assertEqual(filtered.core_deployment, [])
            self.assertTrue(any("严禁换代码" in n for n in notes), notes)

        # PM tries to inflate £8,000 -> clamped down to ticket.recommended_gbp.
        gate = _run_revised_snapshot_gate()
        ticket = gate["isa"]["core_allocation_plan"]["tickets"][0]
        filtered, notes = _validate_core_deployment(
            self._decision_with_ticket(
                symbol=ticket["symbol"], approved_gbp=8000.0, amount="£8,000"
            ),
            portfolio,
            cfg,
            gate,
            bundles,
        )
        self.assertEqual(len(filtered.core_deployment), 1)
        self.assertAlmostEqual(float(filtered.core_deployment[0].approved_gbp), float(ticket["recommended_gbp"]))
        self.assertEqual(filtered.core_deployment[0].size_hint, f"£{float(ticket['recommended_gbp']):,.0f}")
        self.assertTrue(any("钳位" in n for n in notes))
        self.assertAlmostEqual(
            float(gate["isa"]["approved_report_core_gbp"]), float(ticket["recommended_gbp"])
        )

        # PM DEFER -> no BUY, no T+N ledger row.
        gate = _run_revised_snapshot_gate()
        deferred, notes = _validate_core_deployment(
            self._decision_with_ticket(symbol=ticket["symbol"], verdict="DEFER", approved_gbp=0.0),
            portfolio,
            cfg,
            gate,
            bundles,
        )
        self.assertEqual(deferred.core_deployment, [])
        self.assertTrue(any("DEFER" in n for n in notes))
        self.assertEqual(_build_report_recommendations(deferred, portfolio, bundles, cfg), [])
        text = render_morning_report(deferred, max_chars=1800)
        self.assertNotIn("BUY " + ticket["symbol"], text)

    def test_approved_ticket_enters_core_dca_ledger_and_blocks_same_day_pick(self) -> None:
        portfolio = _revised_plan_snapshot_portfolio()
        cfg = self._cfg()
        bundles = {
            "VERG.L": TickerMarketBundle(symbol="VERG.L", quote_type="ETF", currency="GBP", current_price=35.20),
            "ISF.L": TickerMarketBundle(symbol="ISF.L", quote_type="ETF", currency="GBP", current_price=8.20),
        }
        gate = _run_revised_snapshot_gate()
        ticket = gate["isa"]["core_allocation_plan"]["tickets"][0]
        filtered, _notes = _validate_core_deployment(
            self._decision_with_ticket(symbol=ticket["symbol"]), portfolio, cfg, gate, bundles
        )
        recs = _build_report_recommendations(filtered, portfolio, bundles, cfg)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "core_dca")
        self.assertEqual(recs[0]["horizon_days"], 20)
        self.assertEqual(recs[0]["fill_status"], "ASSUMED")

        # Same-day /pick cannot spend the same ISA cash a second time.
        pick = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol=ticket["symbol"], action="BUY", thesis="同日宽基",
                    entry_zone="35", stop_condition="32", target_window="42", entry_price=35.2,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(pick, portfolio, gate, bundles=bundles)
        self.assertEqual(hardened.selected_actions, [])
        self.assertTrue(any("/report 已安排" in n for n in notes), notes)

        # A fresh gate with no /report approval still freezes VUAG (US tilt) and caps ISA core at 1.
        fresh = _run_revised_snapshot_gate()
        fresh["isa"]["approved_report_core_gbp"] = 0.0
        vuag_pick = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="VUAG.L", action="BUY", thesis="标普加码",
                    entry_zone="103", stop_condition="95", target_window="120", entry_price=103.47,
                )
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(vuag_pick, portfolio, fresh, bundles=bundles)
        self.assertEqual(hardened.selected_actions, [])
        self.assertTrue(any("标普500" in n for n in notes), notes)

        two_core = PickDecision(
            headline="测试",
            selected_actions=[
                PickActionItem(
                    account="ISA", symbol="VERG.L", action="BUY", thesis="欧洲",
                    entry_zone="35", stop_condition="32", target_window="42", entry_price=35.2,
                ),
                PickActionItem(
                    account="ISA", symbol="ISF.L", action="BUY", thesis="英国",
                    entry_zone="8", stop_condition="7.5", target_window="10", entry_price=8.2,
                ),
            ],
            consistency_and_cash_note="测试",
        )
        hardened, notes = _apply_gate_channel_constraints(two_core, portfolio, fresh, bundles=bundles)
        self.assertEqual([a.symbol for a in hardened.selected_actions], ["VERG.L"])
        self.assertTrue(any("最多保留 1 条" in n for n in notes), notes)



class TestV556Wave4ReportPolish(unittest.TestCase):
    """v5.5.6 Wave 4: Regime truth source, GBp currency labelling, ISA section split, veto punctuation."""

    def _cfg(self) -> AppConfig:
        return AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            strategy=StrategySection(deployment_gate_mode="advise"),
        )

    def test_regime_anchor_text_is_deterministic_and_forbids_rejudging(self) -> None:
        cfg = self._cfg()
        gate_preview = {
            "regime": "NEUTRAL",
            "regime_reason": "常规回调/震荡分化",
            "isa": {"status": "RUN", "reason": "现金充足"},
            "invest": {"status": "WAIT", "reason": "槽位已满"},
        }
        anchor = _build_regime_anchor_text(cfg, gate_preview, "NEUTRAL")
        self.assertIn("当日确定性市场状态 = NEUTRAL（常规回调/震荡分化）", anchor)
        self.assertIn("Invest 基础净 R:R 门槛 = ≥2.0:1", anchor)
        self.assertIn("ISA 门控 = RUN（现金充足）", anchor)
        self.assertIn("Invest 门控 = WAIT（槽位已满）", anchor)
        self.assertIn("不得自行改判 Regime", anchor)
        self.assertIn("在 NEUTRAL 基础上主动收紧", anchor)

        risk_on = _build_regime_anchor_text(cfg, None, "RISK_ON")
        self.assertIn("RISK_ON 右侧领涨 ≥1.6:1", risk_on)
        self.assertIn("未启用门控", risk_on)

    def test_core_policy_ticket_block_labels_gbp_pence_with_pound_reference(self) -> None:
        gate_preview = {
            "isa": {
                "core_allocation_plan": {
                    "tickets": [
                        {
                            "symbol": "ISF.L",
                            "sleeve_label": "英国宽基",
                            "current_pct": 3.0,
                            "target_pct": 10.0,
                            "cap_pct": 20.0,
                            "recommended_gbp": 1600.0,
                            "currency": "GBp",
                            "price": 1025.4,
                            "ma50": 1041.1,
                            "ma200": 1031.97,
                            "execution_style": "MARKET",
                        }
                    ],
                    "held_actions": [],
                    "deferred_notes": [],
                }
            }
        }
        block = _format_core_policy_ticket_block(gate_preview, {}, {})
        self.assertIn("现价 1025.4 GBp (≈£10.25)", block)
        self.assertIn("MA50 1041.1 GBp", block)
        self.assertIn("MA200 1031.97 GBp", block)

    def test_isa_pending_audits_get_own_subheading_and_vetoes_lose_double_punctuation(self) -> None:
        decision = MorningReportDecision(
            headline="核心部署与挂单裁决",
            top_actions=[],
            pending_order_audits=[
                PendingOrderAuditItem(
                    account="ISA",
                    symbol="ULVR.L",
                    verdict="KEEP",
                    order_summary="BUY 4900 GBp",
                    reason="回踩埋伏有效等待",
                )
            ],
            core_deployment=[
                CoreDeploymentItem(
                    account="ISA",
                    symbol="VERG.L",
                    verdict="APPROVE",
                    price_zone="34.8-35.4 GBP",
                    size_hint="£1,600",
                    reason="欧洲宽基补缺口",
                )
            ],
            daily_vetoes=[
                "Invest 半导体占比超限，冻结新开芯片仓。",
                "ULVR.L 挂单维持不撤；",
            ],
            macro_and_cash="现金充足。",
            account_diagnosis="ISA 核心宽基缺口。",
            hold_summary="其余持有",
            trade_review="无",
        )
        text = render_morning_report(decision)
        core_idx = text.index("【核心宽基部署】")
        audit_idx = text.index("【活动挂单裁决】")
        self.assertLess(core_idx, audit_idx)
        self.assertLess(audit_idx, text.index("ULVR.L (BUY 4900 GBp)"))
        self.assertEqual(text.count("【活动挂单裁决】"), 1)
        self.assertNotIn("。；", text)
        self.assertNotIn("；；", text)
        self.assertIn("Invest 半导体占比超限，冻结新开芯片仓；ULVR.L 挂单维持不撤", text)


class TestV556Wave4RegimeInjection(unittest.IsolatedAsyncioTestCase):
    async def test_bull_and_bear_prompts_carry_python_regime_and_dynamic_rr_floor(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _RecordingLLM, _sample_portfolio
        from src.agents.report_agent import generate_morning_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "wave4_regime.db")
            await db.init()
            cfg = AppConfig(
                env=EnvSettings(_env_file=None, telegram_chat_id="999"),
                models={"enable_forced_search_grounding": False},
                strategy=StrategySection(deployment_gate_mode="advise"),
            )
            md = _FakeMarketData()
            llm = _RecordingLLM()
            try:
                await generate_morning_report(cfg, _FakeT212(_sample_portfolio()), md, llm, db)
            finally:
                await md.close()
                await db.close()

        by_label = {c.get("usage_label"): c for c in llm.calls}
        for label in ("report_analyst", "report_auditor"):
            call = by_label[label]
            prompt = call["prompt"]
            self.assertIn("Python 确定性市场状态", prompt)
            self.assertIn("当日确定性市场状态 = ", prompt)
            self.assertIn("Invest 基础净 R:R 门槛 = ≥", prompt)
            self.assertIn("不得自行改判 Regime", prompt)
            self.assertIn("ISA 门控 = ", prompt)
        self.assertIn("无达到当日动态 R:R 门槛的强信号时明确维持持币待命", by_label["report_analyst"]["system_instruction"])


class TestV557Wave1DeterministicTruth(unittest.TestCase):
    """v5.5.7 Wave 1: VIX floor sync, fake right-side rejection, gate ticket dedupe."""

    @staticmethod
    def _portfolio_with_pending(symbol: str = "ISF.L") -> PortfolioData:
        pending = PendingOrderItem(
            id="p1",
            account="ISA",
            raw_ticker=f"{symbol}_EQ",
            ticker=symbol.split(".")[0],
            yahoo_symbol=symbol,
            name=symbol,
            side="BUY",
            order_type="LIMIT",
            quantity=100.0,
            value=1020.0,
            currency="GBP",
        )
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=100000.0,
            positions=[],
            pending_orders=[pending],
        )
        invest = AccountSnapshot(
            ok=True, account_label="Invest", total_account_value=10000.0, positions=[]
        )
        return PortfolioData(
            fetched_at_uk="2026-11-19 09:30",
            fetched_at_ms=1763544600000,
            isa=isa,
            invest=invest,
            total_combined_value=110000.0,
        )

    def test_scan_a_rejects_below_ma200_and_weekly_bearish(self) -> None:
        healthy = TechnicalSnapshot(
            symbol="LSEG.L", last_close=9000.0, ma50=8950.0, ma200=8000.0,
            dist_ma50_pct=0.5, dist_ma200_pct=12.0, rsi14=55.0,
            trend_regime="PULLBACK", weekly_bullish=True,
        )
        below_ma200 = TechnicalSnapshot(
            symbol="RR.L", last_close=500.0, ma50=498.0, ma200=560.0,
            dist_ma50_pct=0.4, dist_ma200_pct=-8.0, rsi14=52.0,
            trend_regime="PULLBACK", weekly_bullish=True,
        )
        weekly_bearish = TechnicalSnapshot(
            symbol="SIE.DE", last_close=180.0, ma50=179.0, ma200=150.0,
            dist_ma50_pct=0.3, dist_ma200_pct=18.0, rsi14=50.0,
            trend_regime="PULLBACK", weekly_bullish=False,
        )
        candidates = [
            {"symbol": t.symbol, "tech": t, "account": "Invest"}
            for t in (healthy, below_ma200, weekly_bearish)
        ]
        hits = _scan_a_equity_candidates(
            candidates, self._portfolio_with_pending("ZZZ.L"), StrategySection(), None
        )
        self.assertEqual([h["symbol"] for h in hits], ["LSEG.L"])

    def test_gate_block_skips_core_ticket_after_report_approval(self) -> None:
        gate = {
            "regime": "NEUTRAL",
            "global_status": "RUN",
            "etf_set": ["ISF.L", "VWRL.L"],
            "isa": {
                "approved_report_core_gbp": 1000.0,
                "prefer_core_diversifier": True,
                "core_allocation_plan": {
                    "tickets": [
                        {
                            "symbol": "ISF.L",
                            "sleeve_label": "UK_DIVERSIFIER",
                            "current_pct": 10.0,
                            "target_pct": 15.0,
                            "recommended_gbp": 1000.0,
                        }
                    ]
                },
            },
            "invest": {
                "status": "WAIT",
                "slots_used": 0,
                "slots_cap": 2,
                "invest_min_rr_floor": 2.5,
                "invest_rr_tier_label": "高波防御门槛(VIX≥22)",
            },
        }
        text = _format_gate_constraint_block(gate, self._portfolio_with_pending())
        self.assertNotIn("今日核心宽基政策票据", text)
        self.assertIn("不再重复签发核心宽基票据", text)
        self.assertIn("≥2.5:1", text)
        self.assertIn("高波防御门槛", text)

    def test_gate_block_marks_ticket_already_pending(self) -> None:
        gate = {
            "regime": "NEUTRAL",
            "global_status": "RUN",
            "etf_set": ["ISF.L"],
            "isa": {
                "approved_report_core_gbp": 0.0,
                "prefer_core_diversifier": True,
                "core_allocation_plan": {
                    "tickets": [
                        {
                            "symbol": "ISF.L",
                            "sleeve_label": "UK_DIVERSIFIER",
                            "current_pct": 10.0,
                            "target_pct": 15.0,
                            "recommended_gbp": 1000.0,
                        }
                    ]
                },
            },
            "invest": {"status": "WAIT", "slots_used": 0, "slots_cap": 2},
        }
        text = _format_gate_constraint_block(gate, self._portfolio_with_pending("ISF.L"))
        self.assertIn("今日核心宽基政策票据", text)
        self.assertIn("已挂单执行中，勿重复建仓", text)


class TestV557PickFxRefresh(unittest.IsolatedAsyncioTestCase):
    """v5.5.7 Wave 1: /pick re-derives the in-memory portfolio FX before any cash text."""

    async def test_pick_refreshes_portfolio_fx_before_cash_base(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio
        from src.agents.pick_agent import generate_pick_radar
        from src.agents.tools import clear_grounding_cache

        portfolio = _sample_portfolio().model_copy(update={"gbp_usd_rate": None})
        self.assertIsNone(portfolio.gbp_usd_rate)

        class _FxT212(_FakeT212):
            def __init__(self, data: PortfolioData) -> None:
                super().__init__(data)
                self.refresh_rates: list[float] = []

            def refresh_portfolio_fx(self, pf: PortfolioData, rate: float | None) -> PortfolioData:
                self.refresh_rates.append(rate)  # type: ignore[arg-type]
                return pf.model_copy(update={"gbp_usd_rate": rate})

        class _IdleLLM:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def generate(self, **kwargs) -> str:
                self.calls.append(kwargs)
                if getattr(kwargs.get("response_schema"), "__name__", "") == "PickDecision":
                    return PickDecision(
                        headline="本轮无达标机会（ISA 安全边际 / Invest 动态 R:R 未达标），现金待命",
                        selected_actions=[],
                        rejected_or_frozen=[],
                        consistency_and_cash_note="现金待命",
                    ).model_dump_json()
                return "投研员提案：本轮维持现金待命。"

        cfg = AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            models={"enable_forced_search_grounding": False},
            universe=UniverseSection(isa_long_term=["VWRP.L"], invest_tactical=["SHEL.L"]),
            strategy=StrategySection(overlap_correlation_threshold=1.0),
        )
        llm = _IdleLLM()
        t212 = _FxT212(portfolio)
        clear_grounding_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "pick_fx.db")
            await db.init()
            await db.save_morning_report(
                "REPORT",
                "conclusion",
                context_pack={
                    "symbols": {},
                    "deployment_gate": {
                        "isa": {"equity_open": True, "etf_open": True},
                        "invest": {"status": "WAIT", "mode": "none", "slots_used": 0, "slots_cap": 2},
                        "pm": {"allow": True},
                        "regime": "NEUTRAL",
                        "global_status": "RUN",
                        "etf_set": [],
                    },
                },
            )
            md = _FakeMarketData()
            try:
                await generate_pick_radar(cfg, t212, md, llm, db)
            finally:
                await md.close()
                await db.close()

        self.assertEqual(t212.refresh_rates, [1.275])
        researcher = next(c for c in llm.calls if c.get("usage_label") == "pick_researcher")
        self.assertIn("GBP/USD 1.2750", researcher["prompt"])
        self.assertNotIn("GBP/USD 暂缺", researcher["prompt"])

    async def test_closed_isa_equity_channel_short_circuits_screener(self) -> None:
        from tests.test_db_and_end_to_end import _FakeMarketData, _FakeT212, _sample_portfolio
        from src.agents.pick_agent import generate_pick_radar
        from src.agents.tools import clear_grounding_cache
        from unittest.mock import patch

        captured: dict = {}

        async def _fake_screener(**kwargs):
            captured.update(kwargs)
            return [], []

        class _IdleLLM:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def generate(self, **kwargs) -> str:
                self.calls.append(kwargs)
                if getattr(kwargs.get("response_schema"), "__name__", "") == "PickDecision":
                    return PickDecision(
                        headline="ISA 个股通道关闭，现金待命",
                        selected_actions=[],
                        rejected_or_frozen=[],
                        consistency_and_cash_note="现金待命",
                    ).model_dump_json()
                return "投研员提案：ISA 个股通道关闭。"

        cfg = AppConfig(
            env=EnvSettings(_env_file=None, telegram_chat_id="999"),
            models={"enable_forced_search_grounding": False},
            universe=UniverseSection(isa_long_term=["VWRP.L", "ULVR.L"], invest_tactical=["SHEL.L"]),
            strategy=StrategySection(
                overlap_correlation_threshold=1.0,
                user_preference_memory_enabled=False,
            ),
        )
        portfolio = _sample_portfolio()
        clear_grounding_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "pick_closed_isa.db")
            await db.init()
            await db.upsert_preference_notes(
                [
                    {
                        "source_cycle_date": "2026-11-18",
                        "quote_text": "我更喜欢英镑计价的宽基",
                        "rule_summary": "优先 LSE 英镑 ETF",
                        "expires_on_uk": "2030-01-01",
                    }
                ]
            )
            await db.save_morning_report(
                "REPORT",
                "conclusion",
                context_pack={
                    "symbols": {},
                    "deployment_gate": {
                        "isa": {"equity_open": False, "etf_open": True},
                        "invest": {"status": "WAIT", "mode": "none", "slots_used": 0, "slots_cap": 2},
                        "pm": {"allow": True},
                        "regime": "NEUTRAL",
                        "global_status": "RUN",
                        "etf_set": ["VWRP.L"],
                    },
                },
            )
            md = _FakeMarketData()
            llm = _IdleLLM()
            try:
                with patch("src.agents.pick_agent.run_quant_screener", _fake_screener):
                    await generate_pick_radar(cfg, _FakeT212(portfolio), md, llm, db)
            finally:
                await md.close()
                await db.close()

        self.assertIs(captured.get("enable_isa_equities"), False)
        self.assertIs(captured.get("enable_isa_etfs"), True)
        researcher = next(c for c in llm.calls if c.get("usage_label") == "pick_researcher")
        self.assertNotIn("优先 LSE 英镑 ETF", researcher["prompt"])
        self.assertNotIn("用户长期偏好", researcher["prompt"])


class TestV557Wave4CriticFacts(unittest.TestCase):
    """v5.5.7 Wave 4: strict whitelist fact card + audit trimming."""

    def test_fact_card_uses_whitelist_intersection_only(self) -> None:
        from types import SimpleNamespace
        from src.agents.pick_agent import _build_proposal_fact_card

        bundle = TickerMarketBundle(
            symbol="NVDA",
            quote_type="EQUITY",
            currency="USD",
            sector="Technology",
            current_price=120.0,
            target_mean_price=140.0,
            analyst_count=10,
            days_to_earnings=9,
        )
        tech = TechnicalSnapshot(
            symbol="NVDA", last_close=120.0, rsi14=58.0, atr14=3.2, atr_pct=2.7,
            dist_ma50_pct=4.0, dist_ma200_pct=22.0, trend_regime="BULL", weekly_bullish=True,
        )
        toolkit = SimpleNamespace(_inspect_cache={"TSM": {}}, _corr_cache={})
        text, syms = _build_proposal_fact_card(
            "建议 NVDA 突破，另关注 TSM；ATR 与 RSI 仅作指标，ISA 通道关闭。",
            isa_cands=[],
            invest_cands=[],
            merged_bundles={"NVDA": bundle, "TSM": bundle.model_copy(update={"symbol": "TSM"})},
            tech_map={},
            stage2_tech={"NVDA": tech, "TSM": tech.model_copy(update={"symbol": "TSM"})},
            toolkit=toolkit,
            portfolio=TestV557Wave1DeterministicTruth._portfolio_with_pending("ZZZ.L"),
            cfg=AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999")),
            macro=MacroSnapshot(fetched_at_utc="2026-11-19T07:25:00Z", gbp_usd=1.25, vix=15.0),
            regime="NEUTRAL",
        )
        self.assertIn("【提案标的一手事实卡", text)
        self.assertIn("• NVDA", text)
        self.assertIn("• TSM", text)
        self.assertEqual(syms, {"NVDA", "TSM"})
        self.assertNotIn("• ATR", text)
        self.assertNotIn("• RSI", text)
        self.assertNotIn("• ISA", text)

    def test_fact_card_returns_empty_without_whitelist_hit(self) -> None:
        from types import SimpleNamespace
        from src.agents.pick_agent import _build_proposal_fact_card

        text, syms = _build_proposal_fact_card(
            "只讨论 ATR/RSI/VIX 与 ISA 结构，没有具体标的。",
            isa_cands=[],
            invest_cands=[],
            merged_bundles={"NVDA": TickerMarketBundle(symbol="NVDA", quote_type="EQUITY", currency="USD")},
            tech_map={},
            stage2_tech={},
            toolkit=SimpleNamespace(_inspect_cache={}, _corr_cache={}),
            portfolio=TestV557Wave1DeterministicTruth._portfolio_with_pending("ZZZ.L"),
            cfg=AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="999")),
            macro=None,
            regime="NEUTRAL",
        )
        self.assertEqual(text, "")
        self.assertEqual(syms, set())

    def test_critic_audit_filter_keeps_pending_and_related_holdings(self) -> None:
        from src.agents.pick_agent import _filter_audit_text_for_critic

        text = "\n".join(
            [
                "• [持仓裁决] [ISA] TRIM NVDA @ 168-170 USD — 财报前减仓",
                "• [挂单裁决] [Invest] SHEL.L (BUY 3675 GBp) →【CANCEL】印花税",
                "• [持仓裁决] [ISA] HOLD ULVR.L — 长线持有",
            ]
        )
        filtered = _filter_audit_text_for_critic(text, {"NVDA"})
        self.assertIn("TRIM NVDA", filtered)
        self.assertIn("SHEL.L", filtered)
        self.assertNotIn("ULVR.L", filtered)


class TestV557Wave5Guards(unittest.TestCase):
    """v5.5.7 Wave 5: duplicate Invest pending BUY + Python-owned post-execution cash."""

    def _invest_pending_portfolio(self) -> PortfolioData:
        pending = PendingOrderItem(
            id="inv-panw",
            account="Invest",
            raw_ticker="PANW_EQ",
            ticker="PANW",
            yahoo_symbol="PANW",
            name="Palo Alto",
            side="BUY",
            order_type="LIMIT",
            quantity=2.0,
            value=600.0,
            currency="USD",
        )
        return PortfolioData(
            fetched_at_uk="2026-11-19 09:30",
            fetched_at_ms=1763544600000,
            isa=AccountSnapshot(
                ok=True, account_label="ISA", total_account_value=50000.0, available_to_trade=10000.0
            ),
            invest=AccountSnapshot(
                ok=True,
                account_label="Invest",
                total_account_value=5000.0,
                available_to_trade=2000.0,
                positions=[],
                pending_orders=[pending],
            ),
            total_combined_value=55000.0,
        )

    def test_invest_duplicate_pending_buy_is_hard_intercepted(self) -> None:
        portfolio = self._invest_pending_portfolio()
        decision = PickDecision(
            headline="再次买入 PANW",
            selected_actions=[
                PickActionItem(
                    account="Invest",
                    symbol="PANW",
                    action="BUY",
                    thesis="右侧突破",
                    entry_zone="300",
                    stop_condition="285",
                    target_window="340",
                    entry_price=300.0,
                    stop_price=285.0,
                    target_price=340.0,
                )
            ],
            consistency_and_cash_note="测试",
        )
        gate = {
            "regime": "NEUTRAL",
            "global_status": "RUN",
            "etf_set": [],
            "isa": {},
            "pm": {"allow": True},
            "invest": {
                "status": "RUN",
                "slots_used": 0,
                "slots_cap": 3,
                "daily_cap": 1,
                "today_new_equity_entries": 0,
                "committed_heat_pct": 0.0,
                "heat_cap_pct": 3.0,
            },
        }
        hardened, notes = _apply_gate_channel_constraints(decision, portfolio, gate, bundles={})
        self.assertEqual(hardened.selected_actions, [])
        self.assertTrue(any("未成交买入挂单" in n and "叠加热力" in n for n in notes))

    def test_deterministic_cash_note_clamps_and_never_double_deducts(self) -> None:
        from src.agents.pick_agent import _build_deterministic_cash_note

        portfolio = self._invest_pending_portfolio()
        portfolio.isa.pending_orders = [
            PendingOrderItem(
                id="isa-isf",
                account="ISA",
                raw_ticker="ISF_EQ",
                ticker="ISF",
                yahoo_symbol="ISF.L",
                name="FTSE 100",
                side="BUY",
                order_type="LIMIT",
                quantity=100.0,
                value=1020.0,
                currency="GBP",
            )
        ]
        decision = PickDecision(
            headline="ISA 定投",
            selected_actions=[
                PickActionItem(
                    account="ISA",
                    symbol="ISF.L",
                    action="BUY",
                    thesis="核心宽基定投",
                    entry_zone="£1,600",
                    stop_condition="跌破MA200",
                    target_window="长线",
                    entry_price=10.2,
                    planned_amount_gbp=1600.0,
                )
            ],
            consistency_and_cash_note="继承早报红线",
        )
        note = _build_deterministic_cash_note(
            decision,
            portfolio,
            {"ISA|ISF.L": {"suggested_gbp": 1800.0}},
            {"isa": {"core_allocation_plan": {"tickets": [{"symbol": "ISF.L", "recommended_gbp": 1000.0}]}}},
        )
        # planned £1,600 → ticket cap £1,000; already £1,020 pending → no extra deduction.
        self.assertIn("ISA £10,000", note)
        self.assertIn("Invest £2,000", note)
        self.assertIn("不重复扣挂单", note)
        self.assertIn("继承早报红线", note)

        empty = PickDecision(
            headline="现金待命",
            selected_actions=[],
            consistency_and_cash_note="现金待命",
        )
        note_empty = _build_deterministic_cash_note(empty, portfolio, {}, None)
        self.assertIn("ISA £10,000", note_empty)
        self.assertIn("Invest £2,000", note_empty)


class TestV557Wave6RenderAndAudit(unittest.TestCase):
    """v5.5.7 Wave 6: headline leak (incl. Chinese aliases), empty-state copy, fact anchors."""

    def test_reconcile_headline_catches_chinese_broad_base_alias(self) -> None:
        from src.agents.pick_agent import _reconcile_pick_headline

        original = [
            PickActionItem(
                account="ISA", symbol="ISF.L", action="BUY", thesis="x",
                entry_zone="1020 GBp", stop_condition="跌破MA200", target_window="长线",
                entry_price=1020.0,
            ),
            PickActionItem(
                account="Invest", symbol="META", action="BUY", thesis="y",
                entry_zone="520 USD", stop_condition="480", target_window="580",
                entry_price=520.0, stop_price=480.0, target_price=580.0,
            ),
        ]
        decision = PickDecision(
            headline="加仓富时100并买入 META",
            selected_actions=[original[1]],
            rejected_or_frozen=[],
            consistency_and_cash_note="测试",
        )
        rebuilt = _reconcile_pick_headline(decision, original_actions=original)
        self.assertIn("风控筛后执行", rebuilt.headline)
        self.assertIn("META", rebuilt.headline)
        self.assertNotIn("富时100", rebuilt.headline)
        self.assertNotIn("ISF", rebuilt.headline)

    def test_empty_state_copy_has_no_hardcoded_rr(self) -> None:
        from src.agents.schemas import render_pick_report

        decision = PickDecision(
            headline="本轮无达标机会（ISA 安全边际 / Invest 动态 R:R 未达标），现金待命",
            selected_actions=[],
            rejected_or_frozen=[],
            consistency_and_cash_note="Python精确余现：ISA £10,000 / Invest £2,000",
        )
        text = render_pick_report(decision, max_chars=1000)
        self.assertIn("本轮无达标机会", text)
        self.assertIn("ISA 安全边际 / Invest 动态 R:R 未达标", text)
        self.assertNotIn("R:R≥2:1", text)

    def test_fact_baseline_anchors_new_pick_price_and_currency(self) -> None:
        from src.agents.semantic_auditor import build_fact_baseline

        baseline = build_fact_baseline(
            scope="pick",
            portfolio=TestV557Wave1DeterministicTruth._portfolio_with_pending("ZZZ.L"),
            candidate_bundles={
                "ISF.L": TickerMarketBundle(
                    symbol="ISF.L", quote_type="ETF", currency="GBp", current_price=1020.0
                )
            },
        )
        self.assertIn("本次入选新标的现价锚点", baseline)
        self.assertIn("ISF.L | 现价 1,020.00 GBp", baseline)


class TestV559M1FourStateRegimeAndBreadth(unittest.TestCase):
    """v5.5.9 M1: four-state Regime 2.0 (RISK_ON/NEUTRAL/RECOVERY/RISK_OFF), breadth & pilot sizing."""

    @staticmethod
    def _spy_below_ma200_above_ma50() -> pd.DataFrame:
        # 240 flat at 200 -> 30-day crash to 130 (-35%) -> 35-day rally to 158:
        # price (~158) > MA50 (~144), still well below MA200 (~185).
        closes = [200.0] * 240
        closes += [round(200.0 - 70.0 * ((k + 1) / 30.0), 2) for k in range(30)]
        closes += [round(130.0 + 28.0 * ((k + 1) / 35.0), 2) for k in range(35)]
        return _make_hist(closes)

    @staticmethod
    def _spy_below_ma50_and_ma200() -> pd.DataFrame:
        closes = [200.0] * 240
        closes += [round(200.0 - 70.0 * ((k + 1) / 40.0), 2) for k in range(40)]
        return _make_hist(closes)

    @staticmethod
    def _prev_from(info: dict) -> dict:
        return {
            "regime": info["regime"],
            "raw_regime": info["raw_regime"],
            "raw_streak_days": info["raw_streak_days"],
            "ease_streak_days": info["ease_streak_days"],
            "risk_on_exit_streak_days": info.get("risk_on_exit_streak_days", 0),
            "recovery_streak_days": info.get("recovery_streak_days", 0),
        }

    def test_vix_18_to_20_enters_risk_on_and_two_day_exit_hysteresis(self) -> None:
        strat = StrategySection()
        spy = _spy_risk_on_hist()

        # VIX 19.2 (<20.0) for 3 consecutive days enters RISK_ON.
        prev = None
        for expected in ("NEUTRAL", "NEUTRAL", "RISK_ON"):
            info = evaluate_market_regime({"SPY": spy}, _neutral_macro(vix=19.2), strat, prev_gate_state=prev)
            self.assertEqual(info["regime"], expected)
            prev = self._prev_from(info)

        # Single-day VIX=20.8 jitter stays RISK_ON (exit hysteresis day 1/2).
        d_jit = evaluate_market_regime({"SPY": spy}, _neutral_macro(vix=20.8), strat, prev_gate_state=prev)
        self.assertEqual(d_jit["regime"], "RISK_ON")
        self.assertEqual(d_jit["risk_on_exit_streak_days"], 1)

        # Returning to VIX=19.0 resets exit hysteresis to 0.
        d_back = evaluate_market_regime(
            {"SPY": spy}, _neutral_macro(vix=19.0), strat, prev_gate_state=self._prev_from(d_jit)
        )
        self.assertEqual(d_back["regime"], "RISK_ON")
        self.assertEqual(d_back["risk_on_exit_streak_days"], 0)

        # Two consecutive days of VIX=20.8 step down RISK_ON -> NEUTRAL.
        d_e1 = evaluate_market_regime(
            {"SPY": spy}, _neutral_macro(vix=20.8), strat, prev_gate_state=self._prev_from(d_back)
        )
        self.assertEqual(d_e1["regime"], "RISK_ON")
        d_e2 = evaluate_market_regime(
            {"SPY": spy}, _neutral_macro(vix=20.8), strat, prev_gate_state=self._prev_from(d_e1)
        )
        self.assertEqual(d_e2["regime"], "NEUTRAL")

        # VIX >= 22.0 triggers immediate 1-day downgrade from RISK_ON -> NEUTRAL.
        d_fast = evaluate_market_regime({"SPY": spy}, _neutral_macro(vix=22.5), strat, prev_gate_state=prev)
        self.assertEqual(d_fast["regime"], "NEUTRAL")

        # Crisis VIX >= 28 triggers immediate 1-day switch from RISK_ON -> RISK_OFF.
        d_crash = evaluate_market_regime({"SPY": spy}, _neutral_macro(vix=28.5), strat, prev_gate_state=prev)
        self.assertEqual(d_crash["regime"], "RISK_OFF")

    def test_recovery_three_day_entry_filters_two_day_bounce_and_fast_relapse(self) -> None:
        strat = StrategySection()
        rebound_spy = self._spy_below_ma200_above_ma50()
        weak_spy = self._spy_below_ma50_and_ma200()

        # Start in RISK_OFF.
        d0 = evaluate_market_regime({"SPY": weak_spy}, _neutral_macro(vix=32.0), strat)
        self.assertEqual(d0["regime"], "RISK_OFF")
        prev = self._prev_from(d0)

        # Stale macro must block RISK_OFF -> RECOVERY.
        stale_macro = _neutral_macro(vix=21.5)
        stale_macro.is_stale = True
        d_stale = evaluate_market_regime({"SPY": rebound_spy}, stale_macro, strat, prev_gate_state=prev)
        self.assertEqual(d_stale["regime"], "RISK_OFF")
        self.assertEqual(d_stale["recovery_streak_days"], 0)

        # Days 1-2 above MA50 stay RISK_OFF (filters 2-day dead-cat bounces); Day 3 enters RECOVERY.
        d1 = evaluate_market_regime({"SPY": rebound_spy}, _neutral_macro(vix=21.5), strat, prev_gate_state=prev)
        self.assertEqual(d1["regime"], "RISK_OFF")
        self.assertEqual(d1["recovery_streak_days"], 1)
        d2 = evaluate_market_regime(
            {"SPY": rebound_spy}, _neutral_macro(vix=21.5), strat, prev_gate_state=self._prev_from(d1)
        )
        self.assertEqual(d2["regime"], "RISK_OFF")
        self.assertEqual(d2["recovery_streak_days"], 2)
        d3 = evaluate_market_regime(
            {"SPY": rebound_spy}, _neutral_macro(vix=21.5), strat, prev_gate_state=self._prev_from(d2)
        )
        self.assertEqual(d3["regime"], "RECOVERY")
        rec_prev = self._prev_from(d3)

        # 1-day fast relapse on VIX >= 26.0.
        r_vix = evaluate_market_regime({"SPY": rebound_spy}, _neutral_macro(vix=26.2), strat, prev_gate_state=rec_prev)
        self.assertEqual(r_vix["regime"], "RISK_OFF")

        # 1-day fast relapse on losing MA50.
        r_ma50 = evaluate_market_regime({"SPY": weak_spy}, _neutral_macro(vix=21.0), strat, prev_gate_state=rec_prev)
        self.assertEqual(r_ma50["regime"], "RISK_OFF")

        # Reclaiming MA200 for 3 consecutive days promotes RECOVERY -> NEUTRAL (no deadlock).
        p = rec_prev
        for expected in ("RECOVERY", "RECOVERY", "NEUTRAL"):
            step = evaluate_market_regime(
                {"SPY": _spy_neutral_hist()}, _neutral_macro(vix=21.0), strat, prev_gate_state=p
            )
            self.assertEqual(step["regime"], expected)
            p = self._prev_from(step)
        self.assertEqual(p["ease_streak_days"], 3)

    def test_breadth_fusion_blocks_narrow_risk_on_and_degrades_neutrally(self) -> None:
        from src.engine.deployment_gate import compute_regime_breadth

        strat = StrategySection()
        spy = _spy_risk_on_hist()

        # 1) Pure SPY only -> breadth degrades neutrally (None), never synthesizes 50.0.
        b_none = compute_regime_breadth({"SPY": spy})
        self.assertIsNone(b_none["score"])
        self.assertEqual(b_none["source"], "unavailable")
        info_none = evaluate_market_regime({"SPY": spy}, _neutral_macro(vix=16.0), strat)
        self.assertEqual(info_none["raw_regime"], "RISK_ON")

        # 2) Narrow local cache breadth (20 equities, only 4 bullish) suppresses RISK_ON -> NEUTRAL.
        up_df = _spy_risk_on_hist()
        down_df = self._spy_below_ma50_and_ma200()
        narrow_cache = {f"EQ{i}": (up_df if i < 4 else down_df) for i in range(20)}
        info_narrow = evaluate_market_regime(
            {"SPY": spy}, _neutral_macro(vix=16.0), strat, cached_breadth_histories=narrow_cache
        )
        self.assertEqual(info_narrow["raw_regime"], "NEUTRAL")
        self.assertTrue(info_narrow["breadth_blocked_risk_on"])
        self.assertIn("窄宽度背离", info_narrow["reason"])

        # 3) Healthy local cache + multi-region benchmarks allows RISK_ON.
        healthy_cache = {f"EQ{i}": (up_df if i < 16 else down_df) for i in range(20)}
        b_healthy = compute_regime_breadth(
            {"SPY": spy, "ISF.L": up_df, "VERG.L": up_df},
            cached_histories=healthy_cache,
        )
        self.assertEqual(b_healthy["source"], "ohlcv_cache+benchmarks")
        self.assertGreaterEqual(float(b_healthy["score"]), 55.0)
        self.assertEqual(b_healthy["pool_count"], 20)
        info_healthy = evaluate_market_regime(
            {"SPY": spy, "ISF.L": up_df, "VERG.L": up_df},
            _neutral_macro(vix=16.0),
            strat,
            cached_breadth_histories=healthy_cache,
        )
        self.assertEqual(info_healthy["raw_regime"], "RISK_ON")

        # 4) Fallback to pool_snaps when ohlcv_cache is empty; uses unified MA200 > 0 parity.
        snaps = {
            f"S{i}": SimpleNamespace(
                dist_ma50_pct=3.0 if i < 12 else -4.0,
                dist_ma200_pct=5.0 if i < 12 else -6.0,
                trend_regime="NEUTRAL" if i < 12 else "WEAK",
            )
            for i in range(16)
        }
        b_snaps = compute_regime_breadth({"SPY": spy}, pool_snaps=snaps)
        self.assertEqual(b_snaps["source"], "pool_snaps")
        self.assertAlmostEqual(float(b_snaps["score"]), 75.0, places=1)

    def test_recovery_gate_slots_heat_scale_min_not_multiplied_and_pilot_min_order(self) -> None:
        from src.agents.pick_agent import _format_gate_constraint_block
        from src.agents.report_agent.morning_report import _build_regime_anchor_text
        from src.engine.portfolio_rules import compute_risk_based_size_gbp, resolve_invest_dynamic_rr_floor

        strat = StrategySection()
        portfolio = _real_snapshot_portfolio()
        portfolio.invest.positions = [p for p in portfolio.invest.positions if p.yahoo_symbol == "AMZN"]
        portfolio.invest.pending_orders = [
            o for o in portfolio.invest.pending_orders if o.yahoo_symbol == "AMZN"
        ]
        prev_rec = {
            "cycle_date": "2026-11-19",
            "regime": "RISK_OFF",
            "raw_regime": "RISK_OFF",
            "raw_streak_days": 5,
            "ease_streak_days": 0,
            "recovery_streak_days": 2,
        }
        gate = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            bundles={},
            histories={"SPY": self._spy_below_ma200_above_ma50()},
            macro=_neutral_macro(vix=21.5),
            candidates=[_up_candidate()],
            prev_gate_states=[prev_rec],
            now_uk=datetime(2026, 11, 20, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(gate["regime"], "RECOVERY")
        inv = gate["invest"]
        self.assertEqual(inv["slots_cap"], 1)
        self.assertEqual(inv["heat_cap_pct"], 1.0)
        self.assertEqual(inv["risk_budget_scale"], 0.6)
        self.assertEqual(inv["risk_scale_reason"], "RECOVERY早春试探")
        self.assertLessEqual(len(gate["banner"]), 110)
        self.assertNotIn("#", gate["banner"])
        self.assertNotIn("**", gate["banner"])

        # Pick gate prompt distinguishes RECOVERY pilot from loss-streak recovery.
        ctx_txt = _format_gate_constraint_block(gate)
        self.assertIn("RECOVERY早春试探", ctx_txt)
        self.assertNotIn("连损恢复首单", ctx_txt)

        # Morning report regime anchor renders RECOVERY and breadth line.
        anchor = _build_regime_anchor_text(AppConfig(env=EnvSettings(_env_file=None)), gate, "RECOVERY", vix=21.5)
        self.assertIn("RECOVERY", anchor)
        self.assertIn("市场宽度", anchor)

        # Loss-streak (0.5R) + RECOVERY (0.6R) takes min = 0.5R (never 0.3R).
        portfolio.invest.recent_orders = [
            _buy_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 18, 10, 0, tzinfo=LONDON)),
            _sell_order("Invest", "CF", "CF", at_uk=datetime(2026, 11, 18, 15, 0, tzinfo=LONDON), pnl=-35.0),
            _buy_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 19, 10, 0, tzinfo=LONDON)),
            _sell_order("Invest", "PANW", "PANW", at_uk=datetime(2026, 11, 19, 15, 0, tzinfo=LONDON), pnl=-40.0),
        ]
        gate_both = evaluate_deployment_gate(
            portfolio=portfolio,
            strategy=strat,
            bundles={},
            histories={"SPY": self._spy_below_ma200_above_ma50()},
            macro=_neutral_macro(vix=21.5),
            prev_gate_states=[prev_rec],
            now_uk=datetime(2026, 11, 27, 7, 30, tzinfo=LONDON),
        )
        self.assertEqual(gate_both["regime"], "RECOVERY")
        self.assertEqual(gate_both["invest"]["risk_budget_scale"], 0.5)
        self.assertIn("连损恢复+RECOVERY", gate_both["invest"]["risk_scale_reason"])

        # Pilot min order floor (£500) prevents 0.6R deadlock while keeping 1.0R at £700.
        pilot_ok = compute_risk_based_size_gbp(
            account="Invest", strategy=strat, effective_stop_pct=-10.0,
            account_total_value_gbp=10000.0, risk_budget_scale=0.6,
        )
        self.assertAlmostEqual(pilot_ok["suggested_gbp"], 510.0, places=2)
        self.assertAlmostEqual(pilot_ok["min_order_gbp"], 500.0, places=2)
        self.assertFalse(pilot_ok["below_min_order"])

        pilot_too_wide = compute_risk_based_size_gbp(
            account="Invest", strategy=strat, effective_stop_pct=-15.0,
            account_total_value_gbp=10000.0, risk_budget_scale=0.6,
        )
        self.assertTrue(pilot_too_wide["below_min_order"])

        normal_blocked = compute_risk_based_size_gbp(
            account="Invest", strategy=strat, effective_stop_pct=-14.0,
            account_total_value_gbp=10000.0, risk_budget_scale=1.0,
        )
        self.assertAlmostEqual(normal_blocked["min_order_gbp"], 700.0, places=2)
        self.assertTrue(normal_blocked["below_min_order"])

        # Dynamic R:R in RECOVERY: 2.0 when VIX < 22, still tightens to 2.5 when VIX >= 22.
        rr_low, lbl_low = resolve_invest_dynamic_rr_floor("RECOVERY", None, strat, vix=20.0)
        self.assertEqual(rr_low, 2.0)
        self.assertIn("RECOVERY", lbl_low)
        rr_hi, lbl_hi = resolve_invest_dynamic_rr_floor("RECOVERY", None, strat, vix=22.5)
        self.assertEqual(rr_hi, 2.5)
        self.assertIn("VIX≥22", lbl_hi)

    def test_legacy_prev_gate_state_without_new_streak_keys_is_backward_compatible(self) -> None:
        strat = StrategySection()
        legacy_prev = {
            "regime": "RISK_ON",
            "raw_regime": "RISK_ON",
            "raw_streak_days": 5,
            "ease_streak_days": 0,
        }
        info = evaluate_market_regime(
            {"SPY": _spy_risk_on_hist()}, _neutral_macro(vix=20.5), strat, prev_gate_state=legacy_prev
        )
        self.assertEqual(info["regime"], "RISK_ON")
        self.assertEqual(info["risk_on_exit_streak_days"], 1)

    def test_replay_two_year_v_bottom_visits_all_four_states_without_deadlock(self) -> None:
        closes: list[float] = []
        closes += [round(100.0 * (1.0015**i), 2) for i in range(250)]
        peak = closes[-1]
        closes += [round(peak * (1.0 - 0.012 * (k + 1)), 2) for k in range(30)]
        trough = closes[-1]
        closes += [round(trough * (1.008**i), 2) for i in range(220)]
        bench = _make_hist(closes)
        vix_df = _make_hist([16.0] * 250 + [35.0] * 30 + [21.0] * 220)
        hy_oas = pd.Series([3.0] * len(closes), index=bench.index)
        summary = replay_deployment_gate_history(bench, vix_df, StrategySection(), hy_oas_series=hy_oas)
        self.assertFalse(summary["deadlock_detected"], summary)
        for state in ("RISK_ON", "RISK_OFF", "RECOVERY", "NEUTRAL"):
            self.assertGreater(summary["state_days"][state], 0, summary)

    def test_cached_breadth_freshness_rejects_stale_friday_cache_after_monday_bar(self) -> None:
        from src.agents.report_agent.morning_report import _filter_fresh_cached_breadth_histories
        from src.data_providers.market_data.models_and_parsing import _serialize_ohlcv_df

        # Ref SPY printed Monday 2026-11-23; stale Sunday-night audition cache holds Friday 2026-11-20 (lag=3d > 2d).
        spy_mon = pd.DataFrame(
            {"Close": [500.0, 495.0]},
            index=pd.to_datetime(["2026-11-20", "2026-11-23"]),
        )
        fri_df = pd.DataFrame(
            {"Open": [100.0] * 5, "High": [101.0] * 5, "Low": [99.0] * 5, "Close": [100.0] * 5, "Volume": [1000] * 5},
            index=pd.bdate_range("2026-11-16", periods=5),
        )
        mon_df = pd.DataFrame(
            {"Open": [100.0] * 5, "High": [101.0] * 5, "Low": [99.0] * 5, "Close": [100.0] * 5, "Volume": [1000] * 5},
            index=pd.bdate_range("2026-11-17", periods=5),
        )
        fri_row = _serialize_ohlcv_df("STALE", "2y", fri_df)
        mon_row = _serialize_ohlcv_df("FRESH", "2y", mon_df)
        self.assertIsNotNone(fri_row)
        self.assertIsNotNone(mon_row)

        filtered = _filter_fresh_cached_breadth_histories(
            {"STALE": fri_row, "FRESH": mon_row},
            spy_mon,
            max_bar_lag_days=2,
        )
        self.assertNotIn("STALE", filtered)
        self.assertIn("FRESH", filtered)

        # On Monday morning (SPY latest bar = Friday 2026-11-20), Friday audition cache has lag=0 -> kept.
        spy_fri = pd.DataFrame({"Close": [500.0]}, index=pd.to_datetime(["2026-11-20"]))
        filtered_mon_am = _filter_fresh_cached_breadth_histories(
            {"STALE": fri_row},
            spy_fri,
            max_bar_lag_days=2,
        )
        self.assertIn("STALE", filtered_mon_am)

    def test_pilot_min_order_floor_reads_strategy_config_across_pick_and_prompts(self) -> None:
        from src.agents.pick_agent import _format_gate_constraint_block
        from src.agents.prompts import build_investor_profile

        custom_strat = StrategySection(min_invest_pilot_order_gbp=550.0)
        full_profile = build_investor_profile(custom_strat)
        critic_profile = build_investor_profile(custom_strat, compact_for_pick_critic=True)
        chat_profile = build_investor_profile(custom_strat, compact_for_chat=True)
        for text in (full_profile, critic_profile, chat_profile):
            self.assertIn("试探缩放期地板 £550", text)

        gate = {
            "regime": "RECOVERY",
            "global_status": "RUN",
            "isa": {},
            "invest": {
                "status": "RUN",
                "slots_used": 0,
                "slots_cap": 1,
                "risk_budget_scale": 0.6,
                "risk_scale_reason": "RECOVERY早春试探",
                "min_invest_pilot_order_gbp": 550.0,
            },
        }
        block = _format_gate_constraint_block(gate, strategy=custom_strat)
        self.assertIn("试探最小单笔地板 £550", block)
        self.assertNotIn("£500", block)

    def test_recovery_hold_reason_reflects_vix_relapse_line_and_custom_params(self) -> None:
        strat = StrategySection(
            regime_benchmark_symbol="QQQ",
            invest_slots_recovery=2,
            invest_recovery_risk_scale=0.55,
        )
        rebound_qqq = self._spy_below_ma200_above_ma50()
        rec_prev = {
            "regime": "RECOVERY",
            "raw_regime": "RISK_OFF",
            "raw_streak_days": 5,
            "ease_streak_days": 0,
            "recovery_streak_days": 0,
        }
        # Day 4 in RECOVERY with VIX=24.8 (between 24 entry and 26 relapse): stays RECOVERY and must not print VIX<24.
        held = evaluate_market_regime(
            {"QQQ": rebound_qqq}, _neutral_macro(vix=24.8), strat, prev_gate_state=rec_prev
        )
        self.assertEqual(held["regime"], "RECOVERY")
        self.assertIn("QQQ", held["reason"])
        self.assertIn("2槽位/0.6R", held["reason"])
        self.assertIn("VIX=24.8<急退线26", held["reason"])
        self.assertNotIn("VIX<24", held["reason"])


class TestV559M3DualChannel(unittest.TestCase):
    """v5.5.9 M3: A1 pullback + A2 breakout dual channels with regime-linked policy gates."""

    @staticmethod
    def _vcp_hist() -> pd.DataFrame:
        # 220-bar Stage-2 uptrend whose final contractions tighten with drying volume.
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
        # Benchmark lagging the leader over the recent 60 days -> positive RS.
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        bench = np.linspace(100.0, 118.0, 160).tolist() + np.linspace(118.0, 114.0, 60).tolist()
        return pd.DataFrame(
            {"Open": np.array(bench) * 0.998, "High": np.array(bench) * 1.005,
             "Low": np.array(bench) * 0.995, "Close": bench, "Volume": np.full(220, 5_000_000.0)},
            index=dates,
        )

    @staticmethod
    def _platform_hist() -> pd.DataFrame:
        # 100-bar rise, 120-bar tight coil riding MA60, then a 2.5x-volume platform breakout.
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        closes = np.concatenate([
            np.linspace(80.0, 125.0, 100),
            125.0 + np.sin(np.linspace(0, 2 * np.pi * 1.7, 120)) * 0.6,
        ]).copy()
        closes[205] = 129.0
        for j in range(206, 220):
            closes[j] = 129.0 + (j - 206) * 0.15
        vols = np.full(220, 2_000_000.0)
        vols[205] = 5_000_000.0
        opens = closes * 0.997
        opens[205] = 124.0  # open below MA60, close reclaims it with volume
        return pd.DataFrame(
            {"Open": opens, "High": closes * 1.012, "Low": closes * 0.988,
             "Close": closes, "Volume": vols},
            index=dates,
        )

    @staticmethod
    def _a2_tech(symbol: str = "AVGO", *, dist_ma50: float = 6.0, rsi: float = 75.0, **over) -> TechnicalSnapshot:
        data = dict(
            symbol=symbol, last_close=136.5, ma50=128.8, ma200=112.0,
            dist_ma50_pct=dist_ma50, dist_ma200_pct=22.0, rsi14=rsi,
            trend_regime="BULL", weekly_bullish=True,
        )
        data.update(over)
        return TechnicalSnapshot(**data)

    @staticmethod
    def _a1_tech(symbol: str = "LLY", **over) -> TechnicalSnapshot:
        data = dict(
            symbol=symbol, last_close=300.0, ma50=295.0, ma200=270.0,
            dist_ma50_pct=0.8, dist_ma200_pct=11.1, rsi14=60.0,
            trend_regime="BULL", weekly_bullish=True,
        )
        data.update(over)
        return TechnicalSnapshot(**data)

    @staticmethod
    def _clean_bundle(symbol: str = "AVGO", **over) -> TickerMarketBundle:
        data = dict(symbol=symbol, quote_type="EQUITY", currency="USD")
        data.update(over)
        return TickerMarketBundle(**data)

    def _scan(
        self,
        candidates: list[dict],
        *,
        regime: str | None = None,
        benchmark: pd.DataFrame | None = None,
        histories: dict[str, pd.DataFrame] | None = None,
    ) -> list[dict]:
        return _scan_a_equity_candidates(
            candidates,
            _real_snapshot_portfolio(),
            StrategySection(),
            histories or {},
            regime=regime,
            benchmark_hist=benchmark,
        )

    def test_regime_none_keeps_legacy_a1_only(self) -> None:
        cands = [
            {"symbol": "LLY", "tech": self._a1_tech("LLY"), "bundle": None, "account": "Invest"},
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        hits = self._scan(cands, histories={"AVGO": self._vcp_hist()})
        self.assertEqual([h["symbol"] for h in hits], ["LLY"])
        self.assertEqual(hits[0]["channel"], "A1")

    def test_risk_on_surfaces_a2_vcp_breakout(self) -> None:
        cands = [
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["symbol"], "AVGO")
        self.assertEqual(hits[0]["channel"], "A2")
        self.assertEqual(hits[0]["pattern"], "VCP")

    def test_risk_on_surfaces_a2_platform_breakout(self) -> None:
        cands = [
            {"symbol": "PLTR", "tech": self._a2_tech("PLTR", dist_ma50=5.5, rsi=74.0),
             "bundle": self._clean_bundle("PLTR"), "account": "Invest"},
        ]
        hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories={"PLTR": self._platform_hist()})
        self.assertEqual([h["pattern"] for h in hits], ["PLATFORM"])
        self.assertEqual(hits[0]["channel"], "A2")

    def test_a2_without_bundle_fails_closed(self) -> None:
        cands = [
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": None, "account": "Invest"},
        ]
        hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
        self.assertEqual(hits, [])

    def test_a2_calendar_windows_match_a1_exclusions(self) -> None:
        for bundle in (
            self._clean_bundle("AVGO", days_to_earnings=2),
            self._clean_bundle("AVGO", days_to_ex_dividend=1),
        ):
            cands = [{"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": bundle, "account": "Invest"}]
            hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
            self.assertEqual(hits, [])

    def test_a2_rsi_and_extension_bounds(self) -> None:
        cases = [
            (6.0, 81.9, True),    # below the 82 extreme line -> admitted
            (6.0, 82.0, False),   # >=82 extreme blow-off -> rejected
            (14.0, 75.0, True),   # extension cap inclusive
            (14.01, 75.0, False), # beyond +14% -> chase guard
            (1.5, 75.0, False),   # band edge belongs to A1, whose 72 cap rejects
            (1.6, 75.0, True),    # first A2-band step admits RSI 75
        ]
        for dist, rsi, expected in cases:
            cands = [
                {"symbol": "AVGO", "tech": self._a2_tech("AVGO", dist_ma50=dist, rsi=rsi),
                 "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
            ]
            hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
            self.assertEqual([h["symbol"] for h in hits], ["AVGO"] if expected else [], msg=f"dist={dist} rsi={rsi}")

    def test_neutral_admits_vcp_strong_rs_but_rejects_platform(self) -> None:
        vcp_cands = [
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        hits = self._scan(vcp_cands, regime="NEUTRAL", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
        self.assertEqual([h["channel"] for h in hits], ["A2"])
        self.assertEqual(hits[0]["pattern"], "VCP")
        plat_cands = [
            {"symbol": "PLTR", "tech": self._a2_tech("PLTR", dist_ma50=5.5, rsi=74.0),
             "bundle": self._clean_bundle("PLTR"), "account": "Invest"},
        ]
        hits = self._scan(plat_cands, regime="NEUTRAL", benchmark=self._bench_hist(), histories={"PLTR": self._platform_hist()})
        self.assertEqual(hits, [])

    def test_recovery_a2_requires_strong_rs(self) -> None:
        cands = [
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        hits = self._scan(cands, regime="RECOVERY", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
        self.assertEqual([h["channel"] for h in hits], ["A2"])
        # Missing benchmark -> RS unconfirmed -> fail-closed in the pilot state.
        hits = self._scan(cands, regime="RECOVERY", histories={"AVGO": self._vcp_hist()})
        self.assertEqual(hits, [])

    def test_risk_off_never_surfaces_a2(self) -> None:
        cands = [
            {"symbol": "LLY", "tech": self._a1_tech("LLY"), "bundle": None, "account": "Invest"},
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        hits = self._scan(cands, regime="RISK_OFF", benchmark=self._bench_hist(), histories={"AVGO": self._vcp_hist()})
        self.assertEqual([(h["symbol"], h["channel"]) for h in hits], [("LLY", "A1")])

    def test_regime_priority_ordering_and_cap(self) -> None:
        cands = [
            {"symbol": "LLY", "tech": self._a1_tech("LLY"), "bundle": None, "account": "Invest"},
            {"symbol": "NFLX", "tech": self._a1_tech("NFLX"), "bundle": None, "account": "Invest"},
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
            {"symbol": "PANW", "tech": self._a2_tech("PANW"), "bundle": self._clean_bundle("PANW"), "account": "Invest"},
        ]
        hist_map = {"AVGO": self._vcp_hist(), "PANW": self._vcp_hist()}
        # RISK_ON: A2 breakouts lead the 3-symbol board
        hits = self._scan(cands, regime="RISK_ON", benchmark=self._bench_hist(), histories=hist_map)
        self.assertEqual([h["symbol"] for h in hits], ["AVGO", "PANW", "LLY"])
        self.assertEqual([h["channel"] for h in hits], ["A2", "A2", "A1"])
        # NEUTRAL: A1 pullbacks lead; the 2nd A2 drops off the 3-symbol cap
        hits = self._scan(cands, regime="NEUTRAL", benchmark=self._bench_hist(), histories=hist_map)
        self.assertEqual([h["symbol"] for h in hits], ["LLY", "NFLX", "AVGO"])
        self.assertEqual([h["channel"] for h in hits], ["A1", "A1", "A2"])

    def test_gate_level_a2_unlocks_run_targeted_with_channel_tag(self) -> None:
        candidates = [
            {"symbol": "AVGO", "tech": self._a2_tech("AVGO"), "bundle": self._clean_bundle("AVGO"), "account": "Invest"},
        ]
        # Deterministic non-RISK_OFF tape: SPY above MA200, VIX 16, HY 3.0. Both the
        # day-1 NEUTRAL (hysteresis) and a confirmed RISK_ON admit this strong-RS VCP hit.
        spy_closes = np.concatenate([np.linspace(100.0, 130.0, 160), np.linspace(130.0, 126.0, 100)])
        spy = pd.DataFrame(
            {"Open": spy_closes * 0.998, "High": spy_closes * 1.005, "Low": spy_closes * 0.995,
             "Close": spy_closes, "Volume": np.full(260, 5_000_000.0)},
            index=pd.date_range("2025-09-01", periods=260, freq="B"),
        )
        gate = evaluate_deployment_gate(
            portfolio=_real_snapshot_portfolio(),
            strategy=StrategySection(),
            bundles={"IBTA.L": TickerMarketBundle(symbol="IBTA.L", quote_type="ETF", currency="GBP")},
            histories={"SPY": spy, "AVGO": self._vcp_hist()},
            macro=_neutral_macro(vix=16.0),
            candidates=candidates,
            universe_etf_symbols=["VWRP.L", "VUAG.L"],
        )
        self.assertIn(gate["regime"], {"NEUTRAL", "RISK_ON"})
        invest = gate["invest"]
        self.assertEqual(invest["status"], "RUN")
        self.assertEqual(invest["mode"], "targeted")
        self.assertIn("AVGO·A2突破", invest["reason"])


class TestV559M3P1BundlePrefilter(unittest.IsolatedAsyncioTestCase):
    """v5.5.9 M3 P1: ordinary bull-tape risers must never eat the bundle batch cap.

    Shape-only pre-filtering (dist_ma50 in (tol, +14%], BULL, RSI<82) matches dozens of
    plain uptrends in RISK_ON; the batch cap then starves the genuine breakout names at
    the back of pool_syms, and the scan's fail-closed silently drops them. The pre-filter
    must pattern-confirm locally (0 network) before allocating a bundle slot.
    """

    @staticmethod
    def _ordinary_hist() -> pd.DataFrame:
        # Smooth linear rise with CONSTANT volume: passes the cheap A2 shape (and even
        # the Minervini template) but never a VCP dry-up nor a platform volume cross.
        dates = pd.date_range("2025-09-01", periods=220, freq="B")
        closes = np.linspace(80.0, 130.0, 220)
        return pd.DataFrame(
            {"Open": closes * 0.997, "High": closes * 1.012, "Low": closes * 0.988,
             "Close": closes, "Volume": np.full(220, 2_000_000.0)},
            index=dates,
        )

    @staticmethod
    def _ordinary_tech(symbol: str) -> TechnicalSnapshot:
        return TechnicalSnapshot(
            symbol=symbol, last_close=130.0, ma50=122.6, ma200=105.0,
            dist_ma50_pct=6.0, dist_ma200_pct=22.0, rsi14=70.0,
            trend_regime="BULL", weekly_bullish=True,
        )

    @staticmethod
    def _pullback_tech(symbol: str, dist: float) -> TechnicalSnapshot:
        return TechnicalSnapshot(
            symbol=symbol, last_close=300.0, ma50=295.0, ma200=270.0,
            dist_ma50_pct=dist, dist_ma200_pct=11.1, rsi14=55.0,
            trend_regime="BULL", weekly_bullish=True,
        )

    class _RecordingMD:
        def __init__(self, earnings: dict[str, int] | None = None) -> None:
            self.calls: list[tuple[list[str], bool, bool]] = []
            self._earnings = earnings or {}

        async def get_bundles(self, symbols, include_news=True, include_earnings=True, **kwargs):
            self.calls.append((list(symbols), include_news, include_earnings))
            return {}, {
                s: TickerMarketBundle(
                    symbol=s, quote_type="EQUITY", currency="USD",
                    days_to_earnings=self._earnings.get(s),
                )
                for s in symbols
            }

    async def test_bull_pool_bundles_go_only_to_pattern_confirmed_breakouts(self) -> None:
        from src.agents.report_agent import _enrich_gate_candidate_bundles

        vcp_hist = TestV559M3DualChannel._vcp_hist()
        pool_syms: list[str] = [f"ORD{i}" for i in range(10)] + ["BRK1", "BRK2"]
        snaps = {f"ORD{i}": self._ordinary_tech(f"ORD{i}") for i in range(10)}
        snaps["BRK1"] = TestV559M3DualChannel._a2_tech("BRK1")
        snaps["BRK2"] = TestV559M3DualChannel._a2_tech("BRK2")
        hist_map = {sym: self._ordinary_hist() for sym in pool_syms[:10]}
        hist_map["BRK1"] = vcp_hist
        hist_map["BRK2"] = vcp_hist

        md = self._RecordingMD()
        out = await _enrich_gate_candidate_bundles(
            md, pool_syms, snaps, {}, tolerance_pct=1.5, histories=hist_map
        )
        # Only the two pattern-confirmed breakouts at the BACK of the pool get the
        # batch — the 10 front-row ordinary risers must not consume any slot.
        self.assertEqual(md.calls, [(["BRK1", "BRK2"], False, True)])
        self.assertIn("BRK1", out)
        self.assertIn("BRK2", out)

        # End-to-end: with these bundles the scan surfaces both breakouts — no starvation.
        cands = [
            {"symbol": s, "tech": snaps[s], "bundle": out.get(s), "account": "Invest"}
            for s in pool_syms
        ]
        hits = _scan_a_equity_candidates(
            cands, _real_snapshot_portfolio(), StrategySection(), hist_map,
            regime="RISK_ON", benchmark_hist=TestV559M3DualChannel._bench_hist(),
        )
        self.assertEqual([h["symbol"] for h in hits], ["BRK1", "BRK2"])
        self.assertTrue(all(h["channel"] == "A2" and h["pattern"] == "VCP" for h in hits))

    async def test_a1_band_alignment_covers_the_minus_two_sub_band(self) -> None:
        from src.agents.report_agent import _enrich_gate_candidate_bundles

        snaps = {
            "M18": self._pullback_tech("M18", -1.8),   # inside [-2.0, -1.5) sub-band
            "M21": self._pullback_tech("M21", -2.1),   # below the A1 band
        }
        md = self._RecordingMD(earnings={"M18": 2})
        out = await _enrich_gate_candidate_bundles(
            md, ["M21", "M18"], snaps, {}, tolerance_pct=1.5, histories={}
        )
        self.assertEqual(md.calls, [(["M18"], False, True)])
        self.assertEqual(out["M18"].days_to_earnings, 2)

        # The previously-blind sub-band pullback now hits the earnings mine too.
        cands = [{"symbol": "M18", "tech": snaps["M18"], "bundle": out["M18"], "account": "Invest"}]
        hits = _scan_a_equity_candidates(
            cands, _real_snapshot_portfolio(), StrategySection(), {}, regime="RISK_ON"
        )
        self.assertEqual(hits, [])

    async def test_a2_shape_without_history_is_skipped_not_fetched(self) -> None:
        from src.agents.report_agent import _enrich_gate_candidate_bundles

        snaps = {"AVGO": TestV559M3DualChannel._a2_tech("AVGO")}
        md = self._RecordingMD()
        out = await _enrich_gate_candidate_bundles(
            md, ["AVGO"], snaps, {}, tolerance_pct=1.5, histories=None
        )
        self.assertEqual(md.calls, [])
        self.assertNotIn("AVGO", out)


if __name__ == "__main__":
    unittest.main()
