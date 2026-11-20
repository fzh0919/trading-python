"""v5.5.9 M5 capital-efficiency tests (offline, fully mocked).

Covers the three M5 deliverables:
  A) Invest distributing-ETF beta reservoir admission (four locks + 30% anti-churn band);
  B) holding_eval stop/trailing/ladder bypass for reservoir members (whitelist only);
  C) deterministic ETF-to-equity funding notes (清宽基换股) for /pick and /report;
  D) ISA legacy-crumb satellite-seat release via full SELL LIMIT orders;
  E) banner / constraint-block / prompts render-and-parity checks.

Defensive red lines pinned throughout: isa_satellite_max_total_pct untouched, RISK_OFF
never advertises new reservoir builds, non-whitelist Invest ETFs keep the full tactical
treatment, and ETF sells never feed the loss-streak cooldown.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.pick_agent import _format_gate_constraint_block
from src.agents.prompts import (
    _build_compact_chat_profile,
    _build_compact_pick_critic_profile,
    build_investor_profile,
)
from src.agents.schemas import PickActionItem, PickDecision, render_pick_report
from src.config import StrategySection
from src.data_providers.analytics import TechnicalSnapshot
from src.data_providers.market_data import TickerMarketBundle
from src.data_providers.t212_client import (
    AccountSnapshot,
    HistoricalOrderItem,
    PendingOrderItem,
    PortfolioData,
    PositionItem,
)
from src.engine.deployment_gate.banner import format_deployment_gate_banner
from src.engine.deployment_gate.common import _BANNER_MAX_CHARS
from src.engine.deployment_gate.ledgers import (
    classify_pending_orders,
    compute_account_ledgers,
    evaluate_tactical_quota_guards,
)
from src.engine.deployment_gate.triggers import scan_triggers_and_decide
from src.engine.portfolio_rules import (
    build_etf_to_equity_funding_note,
    build_report_size_notes,
    evaluate_holding,
    is_invest_core_beta_symbol,
    is_invest_reservoir_symbol,
)

LONDON_TZ = ZoneInfo("Europe/London")
_ETF_SET = ("VWRP.L", "VUAG.L", "VWRL.L", "ISF.L", "VERX.L", "VEUR.L")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
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


def _pending_buy(
    order_id: str,
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    quantity: float,
    value: float,
    limit_price: float | None,
    currency: str = "GBP",
) -> PendingOrderItem:
    return PendingOrderItem(
        id=order_id,
        account=account,
        raw_ticker=f"{ticker}_LSE_ETF",
        ticker=ticker,
        yahoo_symbol=yahoo_symbol,
        name=ticker,
        side="BUY",
        order_type="LIMIT",
        quantity=quantity,
        value=value,
        limit_price=limit_price,
        currency=currency,
        time_in_force="GTC",
    )


def _pending_sell_limit(
    order_id: str,
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    quantity: float,
    limit_price: float | None,
    currency: str = "USD",
    order_type: str = "LIMIT",
) -> PendingOrderItem:
    return PendingOrderItem(
        id=order_id,
        account=account,
        raw_ticker=f"{ticker}_US_EQ",
        ticker=ticker,
        yahoo_symbol=yahoo_symbol,
        name=ticker,
        side="SELL",
        order_type=order_type,
        quantity=quantity,
        value=0.0,
        limit_price=limit_price,
        currency=currency,
        time_in_force="GTC",
    )


def _hist_sell(
    account: str,
    ticker: str,
    yahoo_symbol: str,
    *,
    at_uk: datetime,
    pnl: float,
    inst_currency: str = "GBP",
) -> HistoricalOrderItem:
    ms = int(at_uk.timestamp() * 1000)
    return HistoricalOrderItem(
        id=f"s-{ticker}-{ms}",
        account=account,
        raw_ticker=f"{ticker}_LSE_ETF",
        ticker=ticker,
        yahoo_symbol=yahoo_symbol,
        name=ticker,
        inst_currency=inst_currency,
        side="SELL",
        order_type="MARKET",
        filled_at=at_uk.isoformat(),
        filled_at_ms=ms,
        uk_time_str=at_uk.strftime("%Y-%m-%d %H:%M"),
        quantity=10.0,
        price=100.0,
        net_value=1000.0,
        wallet_currency=inst_currency,
        realised_pnl=pnl,
        tax_total=0.0,
    )


def _acct(
    positions: list[PositionItem],
    *,
    total: float,
    cash: float,
    avail: float,
    reserved: float = 0.0,
    pending: list[PendingOrderItem] | None = None,
    orders: list[HistoricalOrderItem] | None = None,
    label: str = "X",
) -> AccountSnapshot:
    return AccountSnapshot(
        ok=True,
        account_label=label,
        total_account_value=total,
        investments_value=max(0.0, total - cash),
        investments_cost=max(0.0, total - cash),
        unrealized_pnl=0.0,
        total_cash=cash,
        available_to_trade=avail,
        reserved_cash=reserved,
        cash_ratio_pct=(cash / total * 100.0 if total > 0 else 0.0),
        positions=positions,
        pending_orders=pending or [],
        recent_orders=orders or [],
    )


def _portfolio(isa: AccountSnapshot, invest: AccountSnapshot) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-20 07:30",
        fetched_at_ms=1764000000000,
        total_combined_value=isa.total_account_value + invest.total_account_value,
        total_combined_cash=isa.total_cash + invest.total_cash,
        isa=isa,
        invest=invest,
        gbp_usd_rate=1.25,
    )


def _isa_core_funded() -> AccountSnapshot:
    """ISA with 62% committed core (lock L2 passes via the funded-core arm)."""
    positions = [
        _pos("ISA", "VWRP", "VWRP.L", value=20000.0, qty=171.4, avg=114.84, cur=114.84, return_pct=8.0, weight_pct=40.0, ccy="GBP"),
        _pos("ISA", "VUAG", "VUAG.L", value=11000.0, qty=94.5, avg=100.0, cur=100.0, return_pct=6.0, weight_pct=22.0, ccy="GBP"),
        _pos("ISA", "MSFT", "MSFT", value=6000.0, qty=15.0, avg=300.0, cur=400.0, return_pct=25.0, weight_pct=12.0),
    ]
    return _acct(positions, total=50000.0, cash=13000.0, avail=13000.0, label="ISA")


def _isa_gap_like_today() -> AccountSnapshot:
    """Today-snapshot shape: 36% core, 21.2% deployable — lock L2 must fail (ISA first)."""
    positions = [
        _pos("ISA", "VWRP", "VWRP.L", value=12000.0, qty=103.0, avg=114.84, cur=114.84, return_pct=8.0, weight_pct=24.0, ccy="GBP"),
        _pos("ISA", "VUAG", "VUAG.L", value=6000.0, qty=51.5, avg=100.0, cur=100.0, return_pct=6.0, weight_pct=12.0, ccy="GBP"),
        _pos("ISA", "MSFT", "MSFT", value=5000.0, qty=12.5, avg=300.0, cur=400.0, return_pct=25.0, weight_pct=10.0),
        _pos("ISA", "NVDA", "NVDA", value=4000.0, qty=22.2, avg=120.0, cur=180.0, return_pct=40.0, weight_pct=8.0),
    ]
    return _acct(positions, total=50000.0, cash=23000.0, avail=10600.0, label="ISA")


def _invest_idle(cash: float, total: float = 16000.0, **kw) -> AccountSnapshot:
    return _acct(
        kw.pop("positions", []),
        total=total,
        cash=cash,
        avail=kw.pop("avail", cash),
        pending=kw.pop("pending", None),
        orders=kw.pop("orders", None),
        label="Invest",
    )


def _scan(
    isa: AccountSnapshot,
    invest: AccountSnapshot,
    *,
    strategy: StrategySection | None = None,
    regime: str = "NEUTRAL",
    histories: dict[str, pd.DataFrame] | None = None,
) -> dict:
    strat = strategy or StrategySection()
    portfolio = _portfolio(isa, invest)
    etf_set = set(_ETF_SET)
    classified = classify_pending_orders(portfolio, strat, etf_set)
    ledgers = compute_account_ledgers(portfolio, classified, None, strat, etf_set)
    return scan_triggers_and_decide(
        portfolio,
        None,
        strat,
        ledgers,
        {"regime": regime, "benchmark_symbol": ""},
        histories,
        classified,
    )


def _reservoir_position(
    sym: str,
    *,
    value: float,
    qty: float,
    price: float,
    ccy: str = "GBP",
    return_pct: float = 5.2,
) -> PositionItem:
    base = sym.split(".")[0]
    return _pos(
        "Invest",
        base,
        sym,
        value=value,
        qty=qty,
        avg=round(price * 0.952, 4),
        cur=price,
        return_pct=return_pct,
        weight_pct=50.0,
        ccy=ccy,
    )


def _ladder_hist() -> pd.DataFrame:
    """VWRP.L closing -5.1% below the 52w high (first crossing of the -5% ladder step)."""
    dates = pd.bdate_range("2026-01-05", periods=120)
    closes = [110.0] * 119 + [104.4]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=dates,
    )


# ===========================================================================
# Group A — reservoir admission (four locks + 30% anti-churn band)
# ===========================================================================
class TestV559M5ReservoirAdmission(unittest.TestCase):
    def test_core_funded_idle_cash_opens_reservoir_with_overflow_ticket(self) -> None:
        gate = _scan(_isa_core_funded(), _invest_idle(9000.0))
        res = gate["invest"]["reservoir_advice"]
        self.assertTrue(res["eligible"])
        self.assertEqual(res["ticket_symbol"], "VWRL.L")  # no holding yet -> whitelist head
        # ticket = cash - 30% NAV = 9000 - 4800 = 4200
        self.assertEqual(res["ticket_gbp"], 4200.0)
        self.assertEqual(res["reserve_gbp"], 4800.0)
        self.assertIn("蓄水池", gate["invest"]["reason"])

    def test_today_snapshot_shape_keeps_reservoir_closed_isa_first(self) -> None:
        gate = _scan(_isa_gap_like_today(), _invest_idle(14906.11, total=16132.71))
        res = gate["invest"]["reservoir_advice"]
        self.assertFalse(res["eligible"])
        self.assertIn("isa_core_gap_priority", res["locks_failed"])

    def test_isa_cash_dry_arm_alone_opens_lock_l2(self) -> None:
        # ISA core below target BUT deployable <= 15% -> cash-dry arm passes lock L2.
        isa = _acct(
            [
                _pos("ISA", "VWRP", "VWRP.L", value=20000.0, qty=171.4, avg=114.84, cur=114.84, return_pct=8.0, weight_pct=57.1, ccy="GBP"),
            ],
            total=35000.0,
            cash=3500.0,
            avail=3500.0,
            label="ISA",
        )
        gate = _scan(isa, _invest_idle(9000.0, total=16000.0))
        res = gate["invest"]["reservoir_advice"]
        self.assertTrue(res["eligible"])
        self.assertNotIn("isa_core_gap_priority", res["locks_failed"])

    def test_regime_lock_blocks_risk_off_and_recovery(self) -> None:
        for regime in ("RISK_OFF", "RECOVERY"):
            gate = _scan(_isa_core_funded(), _invest_idle(9000.0), regime=regime)
            res = gate["invest"]["reservoir_advice"]
            self.assertFalse(res["eligible"], regime)
            self.assertIn("regime_not_open", res["locks_failed"])
            # RISK_OFF defence branch never advertises the reservoir either.
            if regime == "RISK_OFF":
                self.assertNotIn("蓄水池", gate["invest"]["reason"])

    def test_idle_cash_boundary_is_strictly_greater_than_floor(self) -> None:
        strat = StrategySection(invest_reservoir_min_idle_cash_pct=50.0)
        gate_at = _scan(_isa_core_funded(), _invest_idle(8000.0), strategy=strat)  # exactly 50%
        self.assertIn("idle_cash_below_floor", gate_at["invest"]["reservoir_advice"]["locks_failed"])
        gate_above = _scan(_isa_core_funded(), _invest_idle(8016.0), strategy=strat)  # 50.1%
        self.assertNotIn("idle_cash_below_floor", gate_above["invest"]["reservoir_advice"]["locks_failed"])
        self.assertTrue(gate_above["invest"]["reservoir_advice"]["eligible"])

    def test_small_account_ticket_below_min_never_churns(self) -> None:
        # Invest NAV 4,000, cash 2,050 (51.25% idle): ticket = 2050 - 1200 = 850 < 1000.
        gate = _scan(_isa_core_funded(), _invest_idle(2050.0, total=4000.0))
        res = gate["invest"]["reservoir_advice"]
        self.assertFalse(res["eligible"])
        self.assertIn("ticket_below_min", res["locks_failed"])

    def test_overflow_ticket_uses_available_cash_never_double_spends_resting_orders(self) -> None:
        """v5.5.9 M5 P2 fix: £5k of the £14k cash is already reserved by resting BUY
        orders — the ticket must be computed from AVAILABLE cash only, otherwise buying
        the ticket punches the free-cash band straight through the 30% reserve floor."""
        # 55% available -> eligible, ticket = 11,000 - 6,000 = 5,000 (NOT 14,000-6,000=8,000).
        gate = _scan(_isa_core_funded(), _invest_idle(14000.0, total=20000.0, avail=11000.0))
        res = gate["invest"]["reservoir_advice"]
        self.assertTrue(res["eligible"])
        self.assertEqual(res["ticket_gbp"], 5000.0)
        # 45% available (resting orders eat past the idle floor) -> lock L3 fails.
        gate_tight = _scan(_isa_core_funded(), _invest_idle(14000.0, total=20000.0, avail=9000.0))
        self.assertIn("idle_cash_below_floor", gate_tight["invest"]["reservoir_advice"]["locks_failed"])
        self.assertFalse(gate_tight["invest"]["reservoir_advice"]["eligible"])

    def test_pending_reservoir_buy_dedupes_new_ticket(self) -> None:
        invest = _invest_idle(
            9000.0,
            pending=[_pending_buy("p-vwrl", "Invest", "VWRL", "VWRL.L", quantity=0.0, value=4200.0, limit_price=130.0)],
        )
        gate = _scan(_isa_core_funded(), invest)
        res = gate["invest"]["reservoir_advice"]
        self.assertFalse(res["eligible"])
        self.assertIn("reservoir_buy_pending", res["locks_failed"])
        self.assertEqual(gate["invest"]["pending_reservoir_buys"], ["VWRL.L"])

    def test_accumulating_class_never_becomes_reservoir(self) -> None:
        invest = _invest_idle(
            9000.0,
            positions=[_reservoir_position("VERG.L", value=6000.0, qty=46.0, price=130.0)],
        )
        gate = _scan(_isa_core_funded(), invest)
        res = gate["invest"]["reservoir_advice"]
        self.assertEqual(res["held_symbols"], [])  # ACC class is invisible to the reservoir
        self.assertEqual(gate["invest"]["reservoir_positions"], [])

    def test_top_up_targets_largest_held_reservoir_symbol(self) -> None:
        invest = _invest_idle(
            9000.0,
            positions=[
                _reservoir_position("VWRL.L", value=3000.0, qty=23.0, price=130.0),
                _reservoir_position("ISF.L", value=8000.0, qty=790.0, price=1010.80, ccy="GBp"),
            ],
        )
        gate = _scan(_isa_core_funded(), invest)
        res = gate["invest"]["reservoir_advice"]
        self.assertTrue(res["eligible"])
        self.assertEqual(res["ticket_symbol"], "ISF.L")  # largest holding wins
        self.assertEqual(res["held_symbols"], ["ISF.L", "VWRL.L"])

    def test_master_switch_disables_reservoir_wholesale(self) -> None:
        strat = StrategySection(invest_reservoir_enabled=False)
        gate = _scan(_isa_core_funded(), _invest_idle(9000.0), strategy=strat)
        res = gate["invest"]["reservoir_advice"]
        self.assertFalse(res["eligible"])
        self.assertIn("reservoir_disabled", res["locks_failed"])
        self.assertNotIn("蓄水池", gate["invest"]["reason"])
        self.assertFalse(is_invest_reservoir_symbol("VWRL.L", strat))
        # The curated core-family tier is a classification truth and survives the
        # switch: a core broad-base ETF held in Invest is never ATR-nagged either way,
        # while sector ETFs (IITU.L) stay tactical in both states.
        self.assertTrue(is_invest_core_beta_symbol("VWRL.L", strat))
        self.assertTrue(is_invest_core_beta_symbol("VUSA.L", strat))
        self.assertFalse(is_invest_core_beta_symbol("IITU.L", strat))


# ===========================================================================
# Group B — holding_eval stop/trailing/ladder bypass (whitelist only)
# ===========================================================================
class TestV559M5HoldingBypass(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = StrategySection()

    def _reservoir_pos(self, return_pct: float) -> PositionItem:
        return _pos(
            "Invest",
            "VWRL",
            "VWRL.L",
            value=8000.0,
            qty=80.0,
            avg=100.0,
            cur=100.0 * (1 + return_pct / 100.0),
            return_pct=return_pct,
            weight_pct=50.0,
            ccy="GBP",
        )

    def _bundle(self) -> TickerMarketBundle:
        return TickerMarketBundle(symbol="VWRL.L", quote_type="ETF", currency="GBP")

    def test_reservoir_drawdown_bypasses_atr_stop_review(self) -> None:
        tech = TechnicalSnapshot(
            symbol="VWRL.L",
            last_close=87.0,
            ma50=95.0,
            ma200=100.0,
            dist_ma50_pct=-8.4,
            dist_ma200_pct=-13.0,
            rsi14=28.0,
            trend_regime="BEAR",
            atr_pct=4.0,
        )
        ev = evaluate_holding(self._reservoir_pos(-13.0), tech, self._bundle(), self.strategy, regime="NEUTRAL")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertTrue(any("蓄水池" in r for r in ev.reasons))
        self.assertFalse(any("止损复核" in r for r in ev.reasons))

    def test_reservoir_bypasses_trailing_take_profit(self) -> None:
        tech = TechnicalSnapshot(
            symbol="VWRL.L",
            last_close=130.0,
            ma50=120.0,
            ma200=110.0,
            dist_ma50_pct=8.3,
            dist_ma200_pct=18.2,
            rsi14=61.0,
            trend_regime="BULL",
            drawdown_60d_pct=-10.0,
        )
        ev = evaluate_holding(self._reservoir_pos(30.0), tech, self._bundle(), self.strategy, regime="NEUTRAL")
        self.assertEqual(ev.action_bias, "HOLD")
        # The bypass note itself mentions 移动止盈 (it documents what is bypassed);
        # the real trailing warnings must never fire.
        self.assertFalse(any("触发移动止盈保护警戒" in r for r in ev.reasons))
        self.assertFalse(any("暂免移动止盈误触" in r for r in ev.reasons))

    def test_reservoir_bypasses_rsi_and_stretch_trims(self) -> None:
        tech = TechnicalSnapshot(
            symbol="VWRL.L",
            last_close=105.0,
            ma50=88.0,
            ma200=80.0,
            dist_ma50_pct=19.3,
            dist_ma200_pct=31.2,
            rsi14=80.0,
            trend_regime="BULL",
        )
        ev = evaluate_holding(self._reservoir_pos(5.0), tech, self._bundle(), self.strategy, regime="RISK_ON")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertFalse(any("超买" in r for r in ev.reasons))
        self.assertFalse(any("乖离" in r for r in ev.reasons))

    def test_reservoir_bypasses_m4_ladder_advisories(self) -> None:
        tech = TechnicalSnapshot(
            symbol="VWRL.L",
            last_close=125.0,
            ma50=118.0,
            ma200=100.0,
            dist_ma50_pct=5.9,
            dist_ma200_pct=25.0,
            dist_ma20_pct=4.0,
            rsi14=62.0,
            trend_regime="BULL",
        )
        ev = evaluate_holding(self._reservoir_pos(25.0), tech, self._bundle(), self.strategy, regime="RISK_ON")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertFalse(any("保本解锁" in r for r in ev.reasons))
        self.assertFalse(any("让利润奔跑" in r for r in ev.reasons))
        self.assertFalse(any("分批止盈" in r for r in ev.reasons))

    def test_risk_off_reservoir_gets_defensive_note_only(self) -> None:
        tech = TechnicalSnapshot(
            symbol="VWRL.L",
            last_close=87.0,
            ma50=95.0,
            ma200=100.0,
            dist_ma50_pct=-8.4,
            dist_ma200_pct=-13.0,
            rsi14=28.0,
            trend_regime="BEAR",
            atr_pct=4.0,
        )
        ev = evaluate_holding(self._reservoir_pos(-13.0), tech, self._bundle(), self.strategy, regime="RISK_OFF")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertTrue(any("退潮期派息宽基蓄水池" in r for r in ev.reasons))

    def test_non_whitelist_invest_etf_keeps_full_tactical_treatment(self) -> None:
        pos = _pos(
            "Invest",
            "IITU",
            "IITU.L",
            value=4000.0,
            qty=40.0,
            avg=100.0,
            cur=87.0,
            return_pct=-13.0,
            weight_pct=25.0,
            ccy="GBP",
        )
        tech = TechnicalSnapshot(
            symbol="IITU.L",
            last_close=87.0,
            ma50=95.0,
            ma200=100.0,
            dist_ma50_pct=-8.4,
            dist_ma200_pct=-13.0,
            rsi14=28.0,
            trend_regime="BEAR",
            atr_pct=4.0,
        )
        bundle = TickerMarketBundle(symbol="IITU.L", quote_type="ETF", currency="GBP")
        ev = evaluate_holding(pos, tech, bundle, self.strategy, regime="NEUTRAL")
        self.assertEqual(ev.action_bias, "REVIEW_STOP")  # defence never gets looser
        self.assertFalse(any("蓄水池" in r for r in ev.reasons))
        self.assertFalse(any("核心宽基" in r for r in ev.reasons))
        self.assertFalse(is_invest_core_beta_symbol("IITU.L", self.strategy))

    def test_legacy_core_etf_dist_channel_holding_bypasses_atr_stop(self) -> None:
        """v5.5.9 M5 connection fix: any curated core-family ETF held in Invest (e.g. a
        legacy VUSA.L bought through the ERI-guarded DIST channel) is a beta vehicle and
        must never be nagged by the single-name ATR stop review."""
        pos = _pos(
            "Invest",
            "VUSA",
            "VUSA.L",
            value=6000.0,
            qty=60.0,
            avg=100.0,
            cur=87.0,
            return_pct=-13.0,
            weight_pct=37.5,
            ccy="GBP",
        )
        tech = TechnicalSnapshot(
            symbol="VUSA.L",
            last_close=87.0,
            ma50=95.0,
            ma200=100.0,
            dist_ma50_pct=-8.4,
            dist_ma200_pct=-13.0,
            rsi14=28.0,
            trend_regime="BEAR",
            atr_pct=4.0,
        )
        bundle = TickerMarketBundle(symbol="VUSA.L", quote_type="ETF", currency="GBP")
        ev = evaluate_holding(pos, tech, bundle, self.strategy, regime="NEUTRAL")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertTrue(any("核心宽基ETF" in r for r in ev.reasons))
        self.assertFalse(any("止损复核" in r for r in ev.reasons))
        self.assertTrue(is_invest_core_beta_symbol("VUSA.L", self.strategy))
        self.assertFalse(is_invest_reservoir_symbol("VUSA.L", self.strategy))  # admission/funding stay whitelist-only

    def test_legacy_core_etf_bypasses_m4_ladder(self) -> None:
        pos = _pos(
            "Invest",
            "VUSA",
            "VUSA.L",
            value=6000.0,
            qty=60.0,
            avg=100.0,
            cur=125.0,
            return_pct=25.0,
            weight_pct=37.5,
            ccy="GBP",
        )
        tech = TechnicalSnapshot(
            symbol="VUSA.L",
            last_close=125.0,
            ma50=118.0,
            ma200=100.0,
            dist_ma50_pct=5.9,
            dist_ma200_pct=25.0,
            dist_ma20_pct=4.0,
            rsi14=62.0,
            trend_regime="BULL",
        )
        bundle = TickerMarketBundle(symbol="VUSA.L", quote_type="ETF", currency="GBP")
        ev = evaluate_holding(pos, tech, bundle, self.strategy, regime="RISK_ON")
        self.assertEqual(ev.action_bias, "HOLD")
        self.assertFalse(any("保本解锁" in r for r in ev.reasons))
        self.assertFalse(any("让利润奔跑" in r for r in ev.reasons))
        self.assertFalse(any("分批止盈" in r for r in ev.reasons))

    def test_invest_equity_stop_untouched_regression(self) -> None:
        pos = _pos(
            "Invest",
            "NVDA",
            "NVDA",
            value=8462.0,
            qty=62.0,
            avg=136.48,
            cur=119.0,
            return_pct=-12.8,
            weight_pct=52.0,
        )
        tech = TechnicalSnapshot(
            symbol="NVDA",
            last_close=119.0,
            ma50=128.0,
            ma200=150.0,
            dist_ma50_pct=-7.0,
            dist_ma200_pct=-20.7,
            rsi14=33.0,
            trend_regime="BEAR",
            atr_pct=3.0,
        )
        ev = evaluate_holding(pos, tech, None, self.strategy, regime="NEUTRAL")
        self.assertEqual(ev.action_bias, "REVIEW_STOP")

    def test_isa_accumulating_core_etf_unaffected(self) -> None:
        pos = _pos(
            "ISA",
            "VWRP",
            "VWRP.L",
            value=11282.0,
            qty=84.5,
            avg=120.51,
            cur=133.52,
            return_pct=10.8,
            weight_pct=20.2,
            ccy="GBP",
        )
        bundle = TickerMarketBundle(symbol="VWRP.L", quote_type="ETF", currency="GBP")
        ev = evaluate_holding(pos, None, bundle, self.strategy, regime="NEUTRAL")
        self.assertFalse(any("蓄水池" in r for r in ev.reasons))

    def test_reservoir_etf_sell_loss_never_feeds_loss_cooldown(self) -> None:
        now = datetime(2026, 11, 20, 12, 0, tzinfo=LONDON_TZ)
        sells = [
            _hist_sell("Invest", "VWRL", "VWRL.L", at_uk=datetime(2026, 11, 19, 15, 0, tzinfo=LONDON_TZ), pnl=-120.0),
            _hist_sell("Invest", "ISF", "ISF.L", at_uk=datetime(2026, 11, 20, 15, 0, tzinfo=LONDON_TZ), pnl=-80.0),
        ]
        invest = _invest_idle(9000.0, orders=sells)
        portfolio = _portfolio(_isa_core_funded(), invest)
        guards = evaluate_tactical_quota_guards(
            portfolio, self.strategy, set(_ETF_SET), now_uk=now
        )
        self.assertEqual(guards["tactical_losses_recent"], 0)
        self.assertFalse(guards["in_loss_cooldown"])


# ===========================================================================
# Group C — deterministic ETF-to-equity funding notes (清宽基换股)
# ===========================================================================
class TestV559M5FundingNote(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = StrategySection()

    def _portfolio_with_reservoir(
        self,
        *,
        free: float = 1000.0,
        reservoir: list[PositionItem] | None = None,
    ) -> PortfolioData:
        if reservoir is None:
            reservoir = [_reservoir_position("VWRL.L", value=6000.0, qty=60.0, price=100.0)]
        invest = _acct(
            reservoir,
            total=16000.0,
            cash=free,
            avail=free,
            label="Invest",
        )
        return _portfolio(_isa_core_funded(), invest)

    def _action(self, **kw) -> SimpleNamespace:
        defaults = {
            "account": "INVEST",
            "symbol": "NVDA",
            "action": "BUY",
            "planned_amount_gbp": 1500.0,
            "entry_zone": "",
        }
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_gap_computes_need_with_buffer_and_whole_share_ceil(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        note = build_etf_to_equity_funding_note(
            [self._action()],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertIn("零Quota", note)
        self.assertIn("VWRL.L", note)
        self.assertIn("NVDA", note)
        self.assertIn("£510", note)  # ceil((1500-1000) * 1.02) = 510
        self.assertIn("(≈6股)", note)  # ceil(510 / £100.00) = 6 shares
        self.assertIn("2%", note)
        self.assertIn("CGT", note)
        self.assertIn("换汇", note)  # USD target -> single-leg FX path note

    def test_free_cash_covers_ticket_never_touches_reservoir(self) -> None:
        pf = self._portfolio_with_reservoir(free=2000.0)
        note = build_etf_to_equity_funding_note(
            [self._action(planned_amount_gbp=1500.0)],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertEqual(note, "")

    def test_no_reservoir_held_no_note(self) -> None:
        pf = self._portfolio_with_reservoir(reservoir=[])
        note = build_etf_to_equity_funding_note(
            [self._action()],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertEqual(note, "")

    def test_sub_floor_gap_defers_to_plain_fx_path(self) -> None:
        pf = self._portfolio_with_reservoir(free=1350.0)
        note = build_etf_to_equity_funding_note(
            [self._action(planned_amount_gbp=1500.0)],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertEqual(note, "")  # gap 150 < £200 noise floor

    def test_swap_and_rotate_actions_excluded(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        for act in ("SWAP", "TRIM", "HOLD"):
            note = build_etf_to_equity_funding_note(
                [self._action(action=act)],
                pf,
                strategy=self.strategy,
                sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
            )
            self.assertEqual(note, "", act)

    def test_largest_reservoir_picked_and_gbp_shares_gbx_safe(self) -> None:
        reservoir = [
            _reservoir_position("VWRL.L", value=3000.0, qty=23.0, price=130.0),
            _reservoir_position("ISF.L", value=8000.0, qty=790.0, price=1010.80, ccy="GBp"),
        ]
        pf = self._portfolio_with_reservoir(free=1000.0, reservoir=reservoir)
        note = build_etf_to_equity_funding_note(
            [self._action()],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertIn("ISF.L", note)
        self.assertIn("(≈51股)", note)  # ceil(510 / £10.108 GBp-rescaled) = 51

    def test_reservoir_too_small_to_cover_gap_skips_note(self) -> None:
        reservoir = [_reservoir_position("VWRL.L", value=400.0, qty=4.0, price=100.0)]
        pf = self._portfolio_with_reservoir(free=1000.0, reservoir=reservoir)
        note = build_etf_to_equity_funding_note(
            [self._action()],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertEqual(note, "")

    def test_resting_buy_offset_mirrors_cash_note_deduction(self) -> None:
        pending = [
            _pending_buy("p-nvda", "Invest", "NVDA", "NVDA", quantity=0.0, value=800.0, limit_price=136.0, currency="USD"),
        ]
        invest = _acct(
            [_reservoir_position("VWRL.L", value=6000.0, qty=60.0, price=100.0)],
            total=16000.0,
            cash=600.0,
            avail=600.0,
            pending=pending,
            label="Invest",
        )
        pf = _portfolio(_isa_core_funded(), invest)
        # planned 2000, resting 800 -> effective ticket 1200 vs free 600 -> gap 600 -> need 612.
        note = build_etf_to_equity_funding_note(
            [self._action(planned_amount_gbp=2000.0)],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 2000.0, "is_etf": False}},
        )
        self.assertIn("£612", note)

    def test_reservoir_topup_action_never_funds_itself(self) -> None:
        pf = self._portfolio_with_reservoir(free=100.0)
        note = build_etf_to_equity_funding_note(
            [self._action(symbol="VWRL.L", planned_amount_gbp=4200.0)],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|VWRL.L": {"suggested_gbp": 4200.0, "is_etf": True}},
        )
        self.assertEqual(note, "")

    def test_master_switch_disables_funding_notes(self) -> None:
        strat = StrategySection(invest_reservoir_enabled=False)
        pf = self._portfolio_with_reservoir(free=1000.0)
        note = build_etf_to_equity_funding_note(
            [self._action()],
            pf,
            strategy=strat,
            sizing_map={"INVEST|NVDA": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertEqual(note, "")

    def test_report_size_note_parses_gbp_hint_and_appends_funding(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        act = SimpleNamespace(
            account="Invest",
            symbol="NVDA",
            action="BUY",
            size_hint="£1,600",
            price_zone="130-136",
        )
        notes = build_report_size_notes([act], pf, self.strategy)
        self.assertIn("先卖Invest VWRL.L", notes[("INVEST", "NVDA")])
        self.assertIn("零Quota", notes[("INVEST", "NVDA")])

    def test_report_size_note_unparseable_hint_defers_to_pick(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        act = SimpleNamespace(
            account="Invest",
            symbol="NVDA",
            action="BUY",
            size_hint="分批建仓",
            price_zone="130-136",
        )
        notes = build_report_size_notes([act], pf, self.strategy)
        self.assertNotIn(("INVEST", "NVDA"), notes)

    def test_report_size_note_without_strategy_is_byte_identical_v558(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        act = SimpleNamespace(
            account="Invest",
            symbol="NVDA",
            action="BUY",
            size_hint="£1,600",
            price_zone="130-136",
        )
        notes = build_report_size_notes([act], pf, None)
        self.assertNotIn(("INVEST", "NVDA"), notes)

    def test_report_isa_action_never_gets_invest_funding_suffix(self) -> None:
        """v5.5.9 M5 P1-2 fix: an ISA ticket must never be told to sell the taxable
        Invest reservoir — cross-account, cross-tax-scheme funding is impossible."""
        pf = self._portfolio_with_reservoir(free=1000.0)
        acts = [
            SimpleNamespace(account="ISA", symbol="MSFT", action="ADD", size_hint="£2,000", price_zone=""),
            SimpleNamespace(account="ISA", symbol="VWRP.L", action="BUY", size_hint="£1,600", price_zone=""),
        ]
        notes = build_report_size_notes(acts, pf, self.strategy)
        self.assertNotIn(("ISA", "MSFT"), notes)  # no funding suffix (no base note either)
        self.assertNotIn(("ISA", "VWRP.L"), notes)  # core-ETF tickets are the parking machinery itself

    def test_funding_note_eu_equity_uses_eur_fx_path(self) -> None:
        pf = self._portfolio_with_reservoir(free=1000.0)
        note = build_etf_to_equity_funding_note(
            [self._action(symbol="SIE.DE", planned_amount_gbp=1500.0)],
            pf,
            strategy=self.strategy,
            sizing_map={"INVEST|SIE.DE": {"suggested_gbp": 1500.0, "is_etf": False}},
        )
        self.assertIn("GBP→EUR单边0.15%换汇", note)
        self.assertNotIn("GBP→USD", note)


# ===========================================================================
# Group D — ISA legacy-crumb satellite-seat release (30% weight cap untouched)
# ===========================================================================
class TestV559M5IsaCrumbSeatRelease(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = StrategySection()

    def _isa_with_crumb(
        self,
        *,
        crumb_value: float = 550.0,
        sell_order: PendingOrderItem | None,
        extra_equity_value: float = 2000.0,
    ) -> AccountSnapshot:
        equities = [
            _pos("ISA", "MSFT", "MSFT", value=extra_equity_value, qty=5.0, avg=300.0, cur=400.0, return_pct=25.0, weight_pct=4.0),
            _pos("ISA", "NVDA", "NVDA", value=2000.0, qty=11.1, avg=120.0, cur=180.0, return_pct=40.0, weight_pct=4.0),
            _pos("ISA", "JPM", "JPM", value=2000.0, qty=8.0, avg=200.0, cur=250.0, return_pct=12.0, weight_pct=4.0),
            _pos("ISA", "TSLA", "TSLA", value=2000.0, qty=6.6, avg=280.0, cur=300.0, return_pct=4.0, weight_pct=4.0),
            _pos("ISA", "UBER", "UBER", value=2000.0, qty=18.5, avg=95.0, cur=108.4, return_pct=8.0, weight_pct=4.0),
            _pos("ISA", "META", "META", value=2000.0, qty=3.6, avg=550.0, cur=612.0, return_pct=8.0, weight_pct=4.0),
            _pos("ISA", "AMZN", "AMZN", value=2000.0, qty=3.4, avg=238.0, cur=252.4, return_pct=6.8, weight_pct=4.0),
            # legacy crumb below £1,200 with a full sell limit resting
            _pos("ISA", "AAL", "AAL", value=crumb_value, qty=40.0, avg=16.0, cur=13.75, return_pct=-14.0, weight_pct=1.1),
        ]
        pending = [sell_order] if sell_order is not None else []
        return _acct(equities, total=50000.0, cash=13000.0, avail=13000.0, pending=pending, label="ISA")

    def _ledgers(self, isa: AccountSnapshot) -> dict:
        portfolio = _portfolio(isa, _invest_idle(9000.0))
        etf_set = set(_ETF_SET)
        classified = classify_pending_orders(portfolio, self.strategy, etf_set)
        return compute_account_ledgers(portfolio, classified, None, self.strategy, etf_set)

    def test_full_sell_limit_releases_the_crumb_seat(self) -> None:
        isa = self._isa_with_crumb(
            sell_order=_pending_sell_limit("p-aal", "ISA", "AAL", "AAL", quantity=40.0, limit_price=14.0, currency="USD"),
        )
        ledger = self._ledgers(isa)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 7)  # 8 held - 1 released
        self.assertEqual(ledger["crumb_seat_released_count"], 1)
        self.assertEqual(ledger["crumb_seat_released_symbols"], ["AAL"])
        self.assertEqual(ledger["legacy_crumb_count"], 1)  # display counter untouched
        self.assertFalse(ledger["isa_satellite_saturated"])

    def test_no_sell_order_keeps_seat_locked(self) -> None:
        ledger = self._ledgers(self._isa_with_crumb(sell_order=None))["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 8)
        self.assertTrue(ledger["isa_satellite_saturated"])

    def test_partial_sell_limit_keeps_seat_locked(self) -> None:
        isa = self._isa_with_crumb(
            sell_order=_pending_sell_limit("p-aal", "ISA", "AAL", "AAL", quantity=20.0, limit_price=14.0, currency="USD"),
        )
        ledger = self._ledgers(isa)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 8)

    def test_protective_stop_sell_keeps_seat_locked(self) -> None:
        stop = _pending_sell_limit("p-aal-stop", "ISA", "AAL", "AAL", quantity=40.0, limit_price=None, currency="USD")
        stop = stop.model_copy(update={"order_type": "STOP", "stop_price": 13.5, "limit_price": None})
        isa = self._isa_with_crumb(sell_order=stop)
        ledger = self._ledgers(isa)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 8)

    def test_full_size_holding_never_releases_a_seat(self) -> None:
        isa = self._isa_with_crumb(
            crumb_value=1500.0,  # >= £1,200 -> not a crumb
            sell_order=_pending_sell_limit("p-msft", "ISA", "MSFT", "MSFT", quantity=5.0, limit_price=400.0, currency="USD"),
        )
        ledger = self._ledgers(isa)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 8)
        self.assertEqual(ledger["crumb_seat_released_count"], 0)

    def test_weight_cap_still_saturates_even_after_seat_release(self) -> None:
        isa = self._isa_with_crumb(
            extra_equity_value=2600.0,
            sell_order=_pending_sell_limit("p-aal", "ISA", "AAL", "AAL", quantity=40.0, limit_price=14.0, currency="USD"),
        )
        # satellite pct = (7*2000 + 2600 + 550)/50000 = 33.3% >= 30% -> still saturated.
        ledger = self._ledgers(isa)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 7)
        self.assertTrue(ledger["isa_satellite_saturated"])
        self.assertGreaterEqual(ledger["committed_satellite_pct"], 30.0)

    def test_seat_release_reopens_isa_equity_channel_in_scan(self) -> None:
        isa = self._isa_with_crumb(
            sell_order=_pending_sell_limit("p-aal", "ISA", "AAL", "AAL", quantity=40.0, limit_price=14.0, currency="USD"),
        )
        gate = _scan(isa, _invest_idle(9000.0))
        self.assertTrue(gate["isa"]["equity_open"])
        locked_gate = _scan(self._isa_with_crumb(sell_order=None), _invest_idle(9000.0))
        self.assertFalse(locked_gate["isa"]["equity_open"])

    def test_cross_account_duplicate_name_unlocks_isa_crumb_seat(self) -> None:
        """v5.5.9 M5 P1-1 fix: with a same-name Invest holding (live case: ISA AMZN crumb
        2.2269 shares + Invest AMZN 6.3696 shares), the covering position must come from
        the ISA-only map — the all-portfolio map silently resolves to the Invest row and
        the seat never unlocks."""
        isa = AccountSnapshot(
            ok=True,
            account_label="ISA",
            total_account_value=50000.0,
            investments_value=2400.0,
            investments_cost=2400.0,
            unrealized_pnl=0.0,
            total_cash=47600.0,
            available_to_trade=47600.0,
            reserved_cash=0.0,
            cash_ratio_pct=95.2,
            positions=[
                # ISA crumb: AMZN 2.2269 shares (≈£600) with a FULL sell limit resting
                _pos("ISA", "AMZN", "AMZN", value=600.0, qty=2.2269, avg=245.0, cur=275.0, return_pct=10.0, weight_pct=1.2),
                # filler satellites to make the count matter
                _pos("ISA", "MSFT", "MSFT", value=2000.0, qty=5.0, avg=300.0, cur=400.0, return_pct=25.0, weight_pct=4.0),
                _pos("ISA", "NVDA", "NVDA", value=2000.0, qty=11.1, avg=120.0, cur=180.0, return_pct=40.0, weight_pct=4.0),
                _pos("ISA", "JPM", "JPM", value=2000.0, qty=8.0, avg=200.0, cur=250.0, return_pct=12.0, weight_pct=4.0),
                _pos("ISA", "TSLA", "TSLA", value=2000.0, qty=6.6, avg=280.0, cur=300.0, return_pct=4.0, weight_pct=4.0),
                _pos("ISA", "UBER", "UBER", value=2000.0, qty=18.5, avg=95.0, cur=108.4, return_pct=8.0, weight_pct=4.0),
                _pos("ISA", "META", "META", value=2000.0, qty=3.6, avg=550.0, cur=612.0, return_pct=8.0, weight_pct=4.0),
                _pos("ISA", "COST", "COST", value=2000.0, qty=1.6, avg=980.0, cur=1250.0, return_pct=12.0, weight_pct=4.0),
            ],
            pending_orders=[
                _pending_sell_limit("p-amzn", "ISA", "AMZN", "AMZN", quantity=2.2269, limit_price=275.0, currency="USD"),
            ],
            recent_orders=[],
        )
        # Invest holds the SAME name with a much bigger size — the old all-portfolio
        # lookup would resolve ISA's AMZN to this 6.3696-share row and never unlock.
        invest = _acct(
            [
                _reservoir_position("VWRL.L", value=8000.0, qty=61.5, price=130.0),
                _pos("Invest", "AMZN", "AMZN", value=1600.0, qty=6.3696, avg=238.42, cur=251.2, return_pct=5.3, weight_pct=10.0),
            ],
            total=16000.0,
            cash=6400.0,
            avail=6400.0,
            label="Invest",
        )
        portfolio = _portfolio(isa, invest)
        etf_set = set(_ETF_SET)
        classified = classify_pending_orders(portfolio, self.strategy, etf_set)
        ledger = compute_account_ledgers(portfolio, classified, None, self.strategy, etf_set)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 7)  # 8 held - AMZN seat released
        self.assertEqual(ledger["crumb_seat_released_symbols"], ["AMZN"])
        self.assertFalse(ledger["isa_satellite_saturated"])
        # The reservoir side is unaffected by the same-name Invest holding.
        invest_ledger = compute_account_ledgers(portfolio, classified, None, self.strategy, etf_set)["invest"]
        self.assertEqual([r["symbol"] for r in invest_ledger["reservoir_positions"]], ["VWRL.L"])

    def test_split_sell_limit_tickets_aggregate_to_full_cover(self) -> None:
        """A full exit executed as two partial SELL LIMITs (25 + 15 = 40 shares) must
        still release the seat — quantities aggregate per symbol before the 0.99 cover
        check."""
        isa = self._isa_with_crumb(
            sell_order=_pending_sell_limit("p-aal-1", "ISA", "AAL", "AAL", quantity=25.0, limit_price=14.0, currency="USD"),
        )
        isa.pending_orders.append(
            _pending_sell_limit("p-aal-2", "ISA", "AAL", "AAL", quantity=15.0, limit_price=14.2, currency="USD")
        )
        portfolio = _portfolio(isa, _invest_idle(9000.0))
        etf_set = set(_ETF_SET)
        classified = classify_pending_orders(portfolio, self.strategy, etf_set)
        ledger = compute_account_ledgers(portfolio, classified, None, self.strategy, etf_set)["isa"]
        self.assertEqual(ledger["committed_satellite_count"], 7)
        self.assertEqual(ledger["crumb_seat_released_symbols"], ["AAL"])


# ===========================================================================
# Group E — rendering & cross-module parity
# ===========================================================================
class TestV559M5RenderAndParity(unittest.TestCase):
    def test_banner_carries_reservoir_tag_within_char_budget(self) -> None:
        gate = _scan(_isa_core_funded(), _invest_idle(9000.0))
        banner = format_deployment_gate_banner(gate)
        self.assertIn("蓄水池", banner)
        self.assertIn("VWRL.L", banner)
        self.assertLessEqual(len(banner), _BANNER_MAX_CHARS)

    def test_isa_run_banner_branch_carries_reservoir_tag(self) -> None:
        gate = _scan(
            _isa_core_funded(),
            _invest_idle(9000.0),
            histories={"VWRP.L": _ladder_hist()},  # ladder hit -> ISA RUN, Invest WAIT
        )
        self.assertEqual(gate["isa"]["status"], "RUN")
        banner = format_deployment_gate_banner(gate)
        self.assertIn("蓄水池", banner)
        self.assertLessEqual(len(banner), _BANNER_MAX_CHARS)

    def test_defend_banner_never_advertises_reservoir(self) -> None:
        gate_dict = {
            "isa": {"status": "WAIT"},
            "invest": {
                "status": "DEFEND",
                "heat_gbp_total": 0.0,
                "financed_positions": [],
                "reservoir_advice": {"eligible": True, "ticket_symbol": "VWRL.L", "ticket_gbp": 4200.0},
            },
            "regime": "RISK_OFF",
            "global_status": "DEFEND",
            "fx_missing": False,
        }
        banner = format_deployment_gate_banner(gate_dict)
        self.assertNotIn("蓄水池", banner)

    def test_pick_constraint_block_emits_ticket_and_funding_source_lines(self) -> None:
        gate_ctx = {
            "isa": {
                "status": "WAIT",
                "deployable_pct": 26.0,
                "committed_core_etf_pct": 62.0,
                "equity_open": True,
                "etf_open": True,
            },
            "invest": {
                "status": "WAIT",
                "slots_used": 0,
                "slots_cap": 2,
                "reason": "无右侧猎物",
                "reservoir_advice": {
                    "eligible": True,
                    "ticket_symbol": "VWRL.L",
                    "ticket_gbp": 4200.0,
                    "held_symbols": [],
                },
            },
            "regime": "NEUTRAL",
            "global_status": "WAIT",
        }
        block = _format_gate_constraint_block(gate_ctx)
        self.assertIn("派息宽基蓄水池票据：VWRL.L £4,200", block)
        self.assertIn("严禁自行更换宽基代码", block)

        held_ctx = {
            "isa": gate_ctx["isa"],
            "invest": {
                "status": "WAIT",
                "slots_used": 1,
                "slots_cap": 2,
                "reason": "无右侧猎物",
                "reservoir_total_gbp": 6000.0,
                "reservoir_advice": {
                    "eligible": False,
                    "locks_failed": ["regime_not_open"],
                    "held_symbols": ["VWRL.L"],
                },
            },
            "regime": "NEUTRAL",
            "global_status": "WAIT",
        }
        held_block = _format_gate_constraint_block(held_ctx)
        self.assertIn("零Quota资金源", held_block)
        self.assertIn("严禁自行口算股数", held_block)

    def test_render_pick_report_appends_funding_note_in_cash_section(self) -> None:
        decision = PickDecision(
            headline="NVDA 回踩买点",
            selected_actions=[
                PickActionItem(
                    account="Invest",
                    symbol="NVDA",
                    action="BUY",
                    thesis="回踩 MA50 缩量企稳",
                    entry_zone="130-136",
                    stop_condition="跌破 118",
                    target_window="160+",
                    planned_amount_gbp=1500.0,
                )
            ],
            rejected_or_frozen=[],
            consistency_and_cash_note="已按护栏复算余现",
        )
        funding = (
            "[资金腾挪·零Quota] 先卖出 Invest VWRL.L £510(≈6股) 释放现金 → 再建仓 NVDA £1,500"
            "（含2%价差/汇率缓冲；GBP→USD单边0.15%换汇；应税部分处置涉CGT自行核查）"
        )
        text = render_pick_report(decision, max_chars=1000, funding_note=funding)
        self.assertIn("资金腾挪", text)
        self.assertIn("£510", text)
        self.assertLessEqual(len(text), 1000)
        # Default empty funding note keeps the legacy rendering byte-compatible.
        legacy = render_pick_report(decision, max_chars=1000)
        self.assertNotIn("资金腾挪", legacy)

    def test_prompts_profiles_sync_reservoir_wording(self) -> None:
        strat = StrategySection()
        profile = build_investor_profile(strat)
        self.assertIn("派息宽基蓄水池票据", profile)
        self.assertIn("绝不催买个股", profile)  # rewritten: never-push-buy now scopes to equities
        self.assertIn("配对指令", profile)
        self.assertIn("不计入 8 只锁定席位", profile)
        critic = _build_compact_pick_critic_profile(strat)
        self.assertIn("蓄水池", critic)
        chat = _build_compact_chat_profile(strat)
        self.assertIn("Beta 停车场", chat)


if __name__ == "__main__":
    unittest.main()
