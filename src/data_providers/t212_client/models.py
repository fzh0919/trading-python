"""Pydantic data models for Trading 212 positions, orders, dividends, accounts, and portfolio snapshots."""

from __future__ import annotations

from pydantic import BaseModel, Field

class PositionItem(BaseModel):
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str
    quantity: float
    avg_price: float
    cur_price: float
    current_value: float
    total_cost: float
    pnl: float
    fx_impact: float
    return_pct: float
    weight_pct: float = 0.0
    created_at: str | None = None
    isin: str | None = None
    cumulative_dividends_gbp: float = 0.0
    total_return_with_div_pct: float | None = None


class PendingOrderItem(BaseModel):
    id: str
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float = 0.0
    status: str = ""
    value: float
    limit_price: float | None = None
    stop_price: float | None = None
    currency: str
    time_in_force: str = ""
    created_at: str = ""


class HistoricalOrderItem(BaseModel):
    id: str
    order_id: str = ""
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str
    side: str
    order_type: str
    filled_at: str
    filled_at_ms: int
    uk_time_str: str
    quantity: float
    price: float
    net_value: float
    wallet_currency: str
    realised_pnl: float
    tax_total: float


class HistoricalTransactionItem(BaseModel):
    id: str
    account: str
    tx_type: str  # DEPOSIT | WITHDRAW | TRANSFER | FEE | INTEREST_ON_FREE_CASH | LENDING_INTEREST
    amount: float
    currency: str = "GBP"
    date_time: str
    date_time_ms: int
    uk_time_str: str


class HistoricalDividendItem(BaseModel):
    id: str
    account: str
    raw_ticker: str
    ticker: str
    yahoo_symbol: str
    name: str
    inst_currency: str = ""
    dividend_type: str = ""
    amount: float
    currency: str = "GBP"
    gross_per_share: float = 0.0
    quantity: float = 0.0
    paid_on: str
    paid_on_ms: int
    uk_time_str: str


class AccountSnapshot(BaseModel):
    ok: bool = False
    error: str = ""
    account_label: str = ""
    account_currency: str = "GBP"
    total_account_value: float = 0.0
    investments_value: float = 0.0
    investments_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float | None = None
    total_cash: float = 0.0
    available_to_trade: float = 0.0
    reserved_cash: float = 0.0
    in_pies_cash: float = 0.0
    cash_ratio_pct: float = 0.0
    total_fx_impact: float = 0.0
    positions: list[PositionItem] = Field(default_factory=list)
    pending_orders: list[PendingOrderItem] = Field(default_factory=list)
    recent_orders: list[HistoricalOrderItem] = Field(default_factory=list)
    recent_transactions: list[HistoricalTransactionItem] = Field(default_factory=list)
    recent_dividends: list[HistoricalDividendItem] = Field(default_factory=list)
    pending_orders_supported: bool = False
    pending_orders_error: str = ""
    order_history_supported: bool = False
    transactions_supported: bool = False
    dividends_supported: bool = False
    cash_status_note: str = ""
    is_fallback: bool = False
    fallback_source: str = ""
    fallback_reason: str = ""
    snapshot_fetched_at_uk: str = ""
    # Invest GBP+USD dual-wallet split (T212 API v0 only returns GBP-converted total cash)
    multi_currency_enabled: bool = False
    usd_cash_total: float = 0.0
    usd_available_to_trade: float = 0.0
    usd_reserved_cash: float = 0.0
    gbp_wallet_total: float | None = None
    gbp_wallet_available: float | None = None
    gbp_wallet_reserved: float | None = None
    wallet_source: str = ""
    wallet_updated_at_uk: str = ""
    wallet_reconcile_warning: str = ""
    gbp_usd_rate: float | None = None
    cash_interest_rate_pct: float = 4.0
    usd_cash_interest_rate_pct: float = 3.4


class PortfolioData(BaseModel):
    fetched_at_uk: str
    fetched_at_ms: int
    isa: AccountSnapshot
    invest: AccountSnapshot
    total_combined_value: float = 0.0
    total_combined_cash: float = 0.0
    cash_interest_rate_pct: float = 4.0
    usd_cash_interest_rate_pct: float = 3.4
    gbp_usd_rate: float | None = None
    is_fallback: bool = False
    fallback_source: str = ""
    fallback_note: str = ""

    def all_positions(self) -> list[PositionItem]:
        return [*self.isa.positions, *self.invest.positions]

    def all_pending_orders(self) -> list[PendingOrderItem]:
        return [*self.isa.pending_orders, *self.invest.pending_orders]

    def unique_yahoo_symbols(self, include_pending: bool = True) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.all_positions():
            seen.setdefault(p.yahoo_symbol, None)
        if include_pending:
            for o in self.all_pending_orders():
                if o.yahoo_symbol:
                    seen.setdefault(o.yahoo_symbol, None)
        return list(seen.keys())
