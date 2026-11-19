"""Idempotent SQLite schema initialization and column migrations for AdvisorDatabase."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.storage.db.core import LONDON_TZ, _ensure_columns


class SchemaMixin:
    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect(write=True) as db:
            await db.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_date TEXT NOT NULL,
                    fetched_at_uk TEXT NOT NULL,
                    fetched_at_ms INTEGER NOT NULL,
                    total_combined_value REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_fetched_ms ON portfolio_snapshots(fetched_at_ms);

                -- IMP-17: lightweight daily NAV anchor for TWR performance (never pruned; GBP-only scalars)
                CREATE TABLE IF NOT EXISTS nav_history (
                    uk_date TEXT PRIMARY KEY,
                    total_value_gbp REAL NOT NULL,
                    cash_gbp REAL NOT NULL DEFAULT 0,
                    invested_gbp REAL NOT NULL DEFAULT 0,
                    net_flow_gbp REAL NOT NULL DEFAULT 0,
                    isa_value_gbp REAL NOT NULL DEFAULT 0,
                    invest_value_gbp REAL NOT NULL DEFAULT 0,
                    isa_net_flow_gbp REAL NOT NULL DEFAULT 0,
                    invest_net_flow_gbp REAL NOT NULL DEFAULT 0,
                    benchmark_close REAL,
                    cash_rate_pct REAL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nav_history_created ON nav_history(created_at_ms);

                CREATE TABLE IF NOT EXISTS morning_reports (
                    cycle_date TEXT PRIMARY KEY,
                    report_text TEXT NOT NULL,
                    conclusion_text TEXT NOT NULL,
                    premarket_note TEXT DEFAULT '',
                    daily_guardrails TEXT DEFAULT '[]',
                    context_pack_json TEXT DEFAULT '',
                    updated_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pick_radars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    focus_query TEXT DEFAULT '',
                    research_text TEXT DEFAULT '',
                    final_text TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pick_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    account TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    thesis TEXT DEFAULT '',
                    horizon_days INTEGER DEFAULT 5,
                    benchmark_symbol TEXT DEFAULT '',
                    settled_at_uk TEXT DEFAULT '',
                    raw_return_pct REAL,
                    alpha_pct REAL,
                    outcome_status TEXT DEFAULT 'PENDING',
                    reflection_lesson TEXT DEFAULT '',
                    t10_return_pct REAL,
                    t10_alpha_pct REAL,
                    t20_return_pct REAL,
                    t20_alpha_pct REAL,
                    mfe_pct REAL,
                    mae_pct REAL,
                    source TEXT DEFAULT 'pick',
                    fill_status TEXT DEFAULT '',
                    actual_filled INTEGER DEFAULT NULL,
                    actual_fill_price REAL DEFAULT NULL,
                    execution_slippage_pct REAL DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_date TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    bot_reply TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    cycle_date TEXT NOT NULL,
                    call_label TEXT NOT NULL,
                    model TEXT NOT NULL,
                    thinking_level TEXT NOT NULL,
                    api_rounds INTEGER NOT NULL DEFAULT 1,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    thought_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 1,
                    search_queries INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_llm_usage_created_ms ON llm_usage_logs(created_at_ms);

                CREATE TABLE IF NOT EXISTS ops_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    alert_key TEXT NOT NULL,
                    message TEXT NOT NULL,
                    notified INTEGER NOT NULL DEFAULT 0,
                    suppressed_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_ops_events_created_ms ON ops_events(created_at_ms);
                CREATE INDEX IF NOT EXISTS idx_ops_events_notified_ms ON ops_events(notified, created_at_ms);

                CREATE TABLE IF NOT EXISTS scheduler_runs (
                    job_name TEXT NOT NULL,
                    uk_date TEXT NOT NULL,
                    fired_at_uk TEXT NOT NULL,
                    PRIMARY KEY (job_name, uk_date)
                );

                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    last_bar_date TEXT NOT NULL,
                    last_close REAL NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (symbol, period)
                );
                CREATE INDEX IF NOT EXISTS idx_ohlcv_cache_updated_ms ON ohlcv_cache(updated_at_ms);

                CREATE TABLE IF NOT EXISTS t212_instruments_metadata (
                    ticker TEXT PRIMARY KEY,
                    yahoo_symbol TEXT NOT NULL DEFAULT '',
                    isin TEXT NOT NULL DEFAULT '',
                    currency_code TEXT NOT NULL DEFAULT '',
                    short_name TEXT NOT NULL DEFAULT '',
                    inst_type TEXT NOT NULL DEFAULT '',
                    extended_hours INTEGER NOT NULL DEFAULT 0,
                    working_schedule_id INTEGER,
                    updated_at_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_t212_inst_yahoo ON t212_instruments_metadata(yahoo_symbol);

                CREATE TABLE IF NOT EXISTS t212_dividend_archive (
                    id TEXT PRIMARY KEY,
                    account TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    yahoo_symbol TEXT NOT NULL,
                    amount_gbp REAL NOT NULL,
                    gross_per_share REAL NOT NULL DEFAULT 0.0,
                    paid_on TEXT NOT NULL DEFAULT '',
                    paid_on_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_t212_div_acc_sym ON t212_dividend_archive(account, yahoo_symbol);

                CREATE TABLE IF NOT EXISTS quant_health_audits (
                    audit_date TEXT PRIMARY KEY,
                    created_at_uk TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}',
                    ai_diagnosis TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS user_preference_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_cycle_date TEXT NOT NULL DEFAULT '',
                    quote_text TEXT NOT NULL DEFAULT '',
                    rule_summary TEXT NOT NULL DEFAULT '',
                    expires_on_uk TEXT NOT NULL DEFAULT '',
                    created_at_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_pref_expires ON user_preference_notes(expires_on_uk);

                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    symbol TEXT PRIMARY KEY,
                    currency TEXT NOT NULL DEFAULT '',
                    sector TEXT NOT NULL DEFAULT '',
                    industry TEXT NOT NULL DEFAULT '',
                    quote_type TEXT NOT NULL DEFAULT '',
                    market_cap REAL,
                    trailing_pe REAL,
                    forward_pe REAL,
                    roe_pct REAL,
                    profit_margin_pct REAL,
                    revenue_growth_pct REAL,
                    debt_to_equity REAL,
                    dividend_yield_pct REAL,
                    analyst_count INTEGER,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    updated_at_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_fundamentals_cache_updated ON fundamentals_cache(updated_at_ms);

                CREATE TABLE IF NOT EXISTS active_universe_pool (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    exchange TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT '',
                    sector TEXT NOT NULL DEFAULT '',
                    industry TEXT NOT NULL DEFAULT '',
                    index_sources TEXT NOT NULL DEFAULT '',
                    quality_score REAL NOT NULL DEFAULT 0.0,
                    avg_daily_turnover_usd REAL NOT NULL DEFAULT 0.0,
                    isa_eligible INTEGER NOT NULL DEFAULT 1,
                    invest_eligible INTEGER NOT NULL DEFAULT 1,
                    is_etf INTEGER NOT NULL DEFAULT 0,
                    updated_at_uk TEXT NOT NULL DEFAULT '',
                    updated_at_ms INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS index_constituents_cache (
                    index_name TEXT PRIMARY KEY,
                    symbols_json TEXT NOT NULL DEFAULT '[]',
                    source_url TEXT NOT NULL DEFAULT '',
                    updated_at_ms INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS grounding_news_cache (
                    cache_key TEXT PRIMARY KEY,
                    scope TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL DEFAULT '',
                    summary_text TEXT NOT NULL DEFAULT '',
                    fetched_at_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_grounding_news_fetched ON grounding_news_cache(fetched_at_ms);

                CREATE TABLE IF NOT EXISTS wallet_balances (
                    account_label TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    available_amount REAL NOT NULL DEFAULT 0.0,
                    reserved_amount REAL NOT NULL DEFAULT 0.0,
                    updated_at_uk TEXT NOT NULL DEFAULT '',
                    updated_at_ms INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'manual',
                    note TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (account_label, currency)
                );

                CREATE TABLE IF NOT EXISTS pending_order_snapshots (
                    account_label TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    yahoo_symbol TEXT NOT NULL DEFAULT '',
                    side TEXT NOT NULL DEFAULT '',
                    quantity REAL NOT NULL DEFAULT 0.0,
                    filled_quantity REAL NOT NULL DEFAULT 0.0,
                    limit_price REAL,
                    stop_price REAL,
                    order_status TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT '',
                    first_seen_uk TEXT NOT NULL DEFAULT '',
                    first_seen_ms INTEGER NOT NULL DEFAULT 0,
                    last_seen_uk TEXT NOT NULL DEFAULT '',
                    last_seen_ms INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_label, order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pending_order_snapshots_seen
                    ON pending_order_snapshots(last_seen_ms);
                """
            )
            await _ensure_columns(
                db,
                "t212_instruments_metadata",
                {
                    "isa_eligible": "INTEGER NOT NULL DEFAULT 1",
                    "invest_eligible": "INTEGER NOT NULL DEFAULT 1",
                },
            )
            await _ensure_columns(
                db,
                "morning_reports",
                {
                    "daily_guardrails": "TEXT DEFAULT '[]'",
                    "context_pack_json": "TEXT DEFAULT ''",
                },
            )
            await _ensure_columns(
                db,
                "nav_history",
                {
                    "isa_net_flow_gbp": "REAL NOT NULL DEFAULT 0",
                    "invest_net_flow_gbp": "REAL NOT NULL DEFAULT 0",
                },
            )
            await _ensure_columns(
                db,
                "pick_recommendations",
                {
                    "horizon_days": "INTEGER DEFAULT 5",
                    "benchmark_symbol": "TEXT DEFAULT ''",
                    "settled_at_uk": "TEXT DEFAULT ''",
                    "raw_return_pct": "REAL",
                    "alpha_pct": "REAL",
                    "outcome_status": "TEXT DEFAULT 'PENDING'",
                    "reflection_lesson": "TEXT DEFAULT ''",
                    "t10_return_pct": "REAL",
                    "t10_alpha_pct": "REAL",
                    "t20_return_pct": "REAL",
                    "t20_alpha_pct": "REAL",
                    "mfe_pct": "REAL",
                    "mae_pct": "REAL",
                    "source": "TEXT DEFAULT 'pick'",
                    "fill_status": "TEXT DEFAULT ''",
                    "actual_filled": "INTEGER DEFAULT NULL",
                    "actual_fill_price": "REAL DEFAULT NULL",
                    "execution_slippage_pct": "REAL DEFAULT NULL",
                },
            )
            await _ensure_columns(
                db,
                "llm_usage_logs",
                {"search_queries": "INTEGER NOT NULL DEFAULT 0"},
            )
            await _ensure_columns(
                db,
                "quant_health_audits",
                {"ai_diagnosis": "TEXT DEFAULT ''"},
            )
            await db.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_pick_rec_status_symbol ON pick_recommendations(outcome_status, symbol);
                CREATE INDEX IF NOT EXISTS idx_pick_rec_symbol_created ON pick_recommendations(symbol, created_at_ms);
                """
            )
            # Seed the Invest USD wallet anchor once (idempotent); API v0 cannot report per-currency cash.
            await db.execute(
                """
                INSERT INTO wallet_balances (
                    account_label, currency, available_amount, reserved_amount,
                    updated_at_uk, updated_at_ms, source, note
                )
                SELECT 'Invest', 'USD', 0.0, 0.0, ?, ?, 'init', '默认 0 USD 基准，若实际持有 USD 请校准'
                WHERE NOT EXISTS (
                    SELECT 1 FROM wallet_balances WHERE account_label = 'Invest' AND currency = 'USD'
                )
                """,
                (
                    datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                    int(time.time() * 1000),
                ),
            )
            await db.commit()
