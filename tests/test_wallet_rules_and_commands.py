"""Offline tests for the dual-wallet ledger commands (/wallet, /usd) and wallet-aware friction rules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.bot.handlers import AdvisorBot
from src.config import AppConfig, EnvSettings, StrategySection
from src.data_providers.market_data import MacroSnapshot, MarketDataService, TickerMarketBundle
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, PositionItem, T212Client
from src.engine.portfolio_rules import (
    _round_trip_friction_pct,
    _wallet_cash_basis_pct,
    evaluate_holding,
    wallet_friction_context,
)
from src.storage.db import AdvisorDatabase


def _wallet(*, usd: float = 0.0, gbp: float = 0.0, enabled: bool = True) -> AccountSnapshot:
    return AccountSnapshot(
        ok=True,
        account_label="Invest",
        account_currency="GBP",
        total_account_value=5000.0,
        total_cash=2000.0,
        available_to_trade=1500.0,
        reserved_cash=500.0,
        multi_currency_enabled=enabled,
        usd_cash_total=usd,
        usd_available_to_trade=usd,
        gbp_wallet_total=gbp,
        gbp_wallet_available=gbp,
        gbp_wallet_reserved=0.0,
    )


def _pos(account: str, symbol: str, inst_currency: str) -> PositionItem:
    return PositionItem(
        account=account,
        raw_ticker=symbol.replace(".", "_"),
        ticker=symbol.split(".")[0],
        yahoo_symbol=symbol,
        name=symbol,
        inst_currency=inst_currency,
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


def _etf_bundle(symbol: str) -> TickerMarketBundle:
    return TickerMarketBundle(symbol=symbol, quote_type="ETF")


class TestWalletFrictionRules(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = StrategySection()

    def test_friction_is_asymmetric_for_invest_dual_wallet(self) -> None:
        us = _pos("Invest", "NVDA", "USD")
        uk_stock = _pos("Invest", "AZN.L", "GBP")
        uk_etf = _pos("Invest", "VUAG.L", "GBP")
        eur = _pos("Invest", "ASML.AS", "EUR")

        rich = _wallet(usd=600.0, gbp=1520.0)
        # v5.3.2: conservative spread estimates are part of round-trip friction
        self.assertAlmostEqual(_round_trip_friction_pct(us, None, self.strategy, rich), 0.05)
        self.assertAlmostEqual(_round_trip_friction_pct(uk_stock, None, self.strategy, rich), 0.70)
        self.assertAlmostEqual(
            _round_trip_friction_pct(uk_etf, _etf_bundle("VUAG.L"), self.strategy, rich), 0.20
        )
        self.assertAlmostEqual(_round_trip_friction_pct(eur, None, self.strategy, rich), 0.35)

        usd_only = _wallet(usd=600.0, gbp=0.0)
        self.assertAlmostEqual(_round_trip_friction_pct(uk_stock, None, self.strategy, usd_only), 0.85)
        self.assertAlmostEqual(
            _round_trip_friction_pct(uk_etf, _etf_bundle("VUAG.L"), self.strategy, usd_only), 0.35
        )

        no_usd = _wallet(usd=0.0, gbp=1520.0)
        self.assertAlmostEqual(_round_trip_friction_pct(us, None, self.strategy, no_usd), 0.20)

    def test_isa_friction_is_unchanged_by_wallet(self) -> None:
        isa_us = _pos("ISA", "MSFT", "USD")
        isa_uk = _pos("ISA", "AZN.L", "GBP")
        isa_etf = _pos("ISA", "VUAG.L", "GBP")
        rich = _wallet(usd=600.0, gbp=1520.0)

        self.assertAlmostEqual(_round_trip_friction_pct(isa_us, None, self.strategy, rich), 0.35)
        self.assertAlmostEqual(_round_trip_friction_pct(isa_uk, None, self.strategy, rich), 0.70)
        self.assertAlmostEqual(
            _round_trip_friction_pct(isa_etf, _etf_bundle("VUAG.L"), self.strategy, rich), 0.20
        )
        self.assertAlmostEqual(_wallet_cash_basis_pct(isa_us, self.strategy, rich), 4.0)

    def test_cash_basis_switches_to_usd_rate_when_spending_usd(self) -> None:
        us = _pos("Invest", "NVDA", "USD")
        uk = _pos("Invest", "AZN.L", "GBP")
        eur = _pos("Invest", "ASML.AS", "EUR")

        self.assertAlmostEqual(_wallet_cash_basis_pct(us, self.strategy, _wallet(usd=600.0, gbp=0.0)), 3.4)
        self.assertAlmostEqual(_wallet_cash_basis_pct(us, self.strategy, _wallet(usd=0.0, gbp=900.0)), 4.0)
        self.assertAlmostEqual(_wallet_cash_basis_pct(uk, self.strategy, _wallet(usd=600.0, gbp=0.0)), 3.4)
        self.assertAlmostEqual(_wallet_cash_basis_pct(uk, self.strategy, _wallet(usd=0.0, gbp=900.0)), 4.0)
        self.assertAlmostEqual(_wallet_cash_basis_pct(eur, self.strategy, _wallet(usd=600.0, gbp=0.0)), 4.0)
        # Wallet disabled (or missing) always falls back to the GBP 4.0% anchor
        self.assertAlmostEqual(_wallet_cash_basis_pct(us, self.strategy, _wallet(usd=600.0, enabled=False)), 4.0)
        self.assertAlmostEqual(_wallet_cash_basis_pct(us, self.strategy, None), 4.0)

    def test_friction_hint_labels_spell_out_wallet_funding(self) -> None:
        us = _pos("Invest", "NVDA", "USD")
        uk = _pos("Invest", "AZN.L", "GBP")
        etf = _pos("Invest", "VUAG.L", "GBP")

        with_usd = evaluate_holding(us, None, None, self.strategy, wallet=_wallet(usd=600.0, gbp=0.0))
        self.assertIn("可用USD免换汇", with_usd.friction_note)
        self.assertIn("Invest美股", with_usd.friction_note)

        no_usd = evaluate_holding(us, None, None, self.strategy, wallet=_wallet(usd=0.0, gbp=900.0))
        self.assertIn("USD不足需GBP换汇", no_usd.friction_note)

        uk_short = evaluate_holding(uk, None, None, self.strategy, wallet=_wallet(usd=600.0, gbp=0.0))
        self.assertIn("GBP不足需USD换回+0.15%换汇", uk_short.friction_note)
        self.assertIn("0.5%印花税", uk_short.friction_note)

        uk_rich = evaluate_holding(uk, None, None, self.strategy, wallet=_wallet(usd=0.0, gbp=900.0))
        self.assertIn("GBP充足", uk_rich.friction_note)

        etf_short = evaluate_holding(
            etf, None, _etf_bundle("VUAG.L"), self.strategy, wallet=_wallet(usd=600.0, gbp=0.0)
        )
        self.assertIn("GBP不足需USD换回", etf_short.friction_note)
        self.assertNotIn("印花税", etf_short.friction_note)

        isa_us = _pos("ISA", "MSFT", "USD")
        legacy = evaluate_holding(isa_us, None, None, self.strategy, wallet=_wallet(usd=600.0, gbp=900.0))
        self.assertIn("0.30%往返换汇", legacy.friction_note)

    def test_prompt_uses_dynamic_invest_budget_and_forbids_fx_shrinking(self) -> None:
        from src.agents.prompts import build_investor_profile, build_wallet_context_text

        strategy = StrategySection()
        profile = build_investor_profile(strategy)
        self.assertIn("Invest 净值×0.75%", profile)
        self.assertNotIn("单笔最大可承受亏损约 £85", profile)
        self.assertIn("严禁为省换汇费而人为把仓位压缩到 USD 余额以内", profile)

        portfolio = _portfolio()
        portfolio.invest.multi_currency_enabled = True
        wallet_text = build_wallet_context_text(portfolio, strategy)
        self.assertIn("严禁为省换汇费而人为把仓位压缩到 USD 余额以内", wallet_text)
        self.assertIn("差额正常由 GBP 钱包换汇补足", wallet_text)

    def test_wallet_context_falls_back_to_api_available_cash(self) -> None:
        partial = AccountSnapshot(
            ok=True,
            multi_currency_enabled=True,
            usd_available_to_trade=250.0,
            available_to_trade=1500.0,
            gbp_wallet_available=None,
        )
        enabled, usd_avail, gbp_avail = wallet_friction_context(partial)
        self.assertTrue(enabled)
        self.assertEqual(usd_avail, 250.0)
        self.assertEqual(gbp_avail, 1500.0)
        self.assertEqual(wallet_friction_context(None), (False, 0.0, 0.0))



class _StubT212(T212Client):
    def __init__(self, portfolio: PortfolioData, rate: float | None = None) -> None:
        super().__init__()
        self._stub_portfolio = portfolio
        if rate is not None:
            self.set_gbp_usd_rate(rate)

    async def get_portfolio(self, use_cache: bool = True, gbp_usd_rate: float | None = None) -> PortfolioData:
        if gbp_usd_rate is not None:
            self.set_gbp_usd_rate(gbp_usd_rate)
        return self._stub_portfolio


class _StubMarket(MarketDataService):
    def __init__(self, rate: float | None) -> None:
        self._rate = rate

    async def get_macro_snapshot(self) -> MacroSnapshot:
        return MacroSnapshot(fetched_at_utc="2026-11-17 09:00 UTC", gbp_usd=self._rate)


def _portfolio() -> PortfolioData:
    return PortfolioData(
        fetched_at_uk="2026-11-17 09:00",
        fetched_at_ms=1_763_000_000_000,
        isa=AccountSnapshot(ok=True, account_label="ISA", total_cash=1000.0),
        invest=AccountSnapshot(
            ok=True,
            account_label="Invest",
            account_currency="GBP",
            total_account_value=5000.0,
            total_cash=2000.0,
            available_to_trade=1500.0,
            reserved_cash=500.0,
        ),
        total_combined_value=15000.0,
        total_combined_cash=3000.0,
    )


class TestWalletCommands(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = AdvisorDatabase(Path(self._tmp.name) / "wallet_cmd.db")
        await self.db.init()
        self.cfg = AppConfig(env=EnvSettings(_env_file=None, telegram_chat_id="12345"))
        self.t212 = _StubT212(_portfolio(), rate=1.25)
        self.market = _StubMarket(1.25)
        self.bot = AdvisorBot(self.cfg, self.t212, self.market, None, self.db, None)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_wallet_args(self) -> None:
        parse = AdvisorBot._parse_wallet_args
        self.assertEqual(parse("/usd", []), ("", {}))
        self.assertEqual(parse("/usd", ["600"]), ("", {"USD": (600.0, 0.0)}))
        self.assertEqual(parse("/usd", ["600", "25"]), ("", {"USD": (600.0, 25.0)}))
        self.assertEqual(parse("/wallet", []), ("", {}))
        self.assertEqual(parse("/wallet", ["gbp", "1500"]), ("", {"GBP": (1500.0, 0.0)}))
        self.assertEqual(
            parse("/wallet", ["gbp", "1500", "usd", "600"]),
            ("", {"GBP": (1500.0, 0.0), "USD": (600.0, 0.0)}),
        )
        err, updates = parse("/wallet", ["gbp", "abc"])
        self.assertIn("格式不正确", err)
        self.assertEqual(updates, {})
        self.assertIn("负数", parse("/usd", ["-5"])[0])
        self.assertIn("缺少金额", parse("/wallet", ["gbp"])[0])

    async def test_usd_report_writes_anchor_and_clears_gbp_override(self) -> None:
        await self.db.upsert_wallet_balance("Invest", "GBP", 1300.0)
        reply = await self.bot._handle_wallet_command("/usd 600 25")

        self.assertIn("USD 钱包: $625.00", reply)
        self.assertIn("核对差额", reply)
        records = await self.db.get_wallet_balances("Invest")
        self.assertEqual(records["USD"]["available_amount"], 600.0)
        self.assertEqual(records["USD"]["reserved_amount"], 25.0)
        self.assertNotIn("GBP", records)

    async def test_usd_report_over_cash_is_blocked(self) -> None:
        reply = await self.bot._handle_wallet_command("/usd 3000")

        self.assertIn("⛔ 已阻断", reply)
        self.assertIn("2,400.00", reply)
        records = await self.db.get_wallet_balances("Invest")
        self.assertEqual(records["USD"]["available_amount"], 0.0)  # untouched initial anchor

    async def test_gbp_only_report_back_solves_usd(self) -> None:
        reply = await self.bot._handle_wallet_command("/wallet gbp 1520")

        self.assertIn("反推 USD ≈ $600.00", reply)
        records = await self.db.get_wallet_balances("Invest")
        self.assertEqual(records["GBP"]["available_amount"], 1520.0)
        self.assertEqual(records["USD"]["available_amount"], 600.0)

    async def test_dual_report_shows_mismatch(self) -> None:
        reply = await self.bot._handle_wallet_command("/wallet gbp 1200 usd 600")

        self.assertIn("手动上报", reply)
        self.assertIn("£-320.00", reply)
        self.assertIn("校准提示", reply)

    async def test_wallet_view_is_read_only(self) -> None:
        before = await self.db.get_wallet_balances("Invest")
        reply = await self.bot._handle_wallet_command("/wallet")

        self.assertIn("Invest 双钱包对账", reply)
        self.assertIn("T212 总现金(GBP): £2,000.00", reply)
        self.assertIn("实时 GBP/USD: 1.2500", reply)
        self.assertEqual(await self.db.get_wallet_balances("Invest"), before)


if __name__ == "__main__":
    unittest.main()

