"""Async SQLite repository for portfolio snapshots, reports, picks, settlements, and daily chat history."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite

from src.data_providers.t212_client import PortfolioData

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

_RECOMMENDATION_SELECT_COLS = (
    "id, created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis, "
    "horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, "
    "outcome_status, reflection_lesson, "
    "t10_return_pct, t10_alpha_pct, t20_return_pct, t20_alpha_pct, mfe_pct, mae_pct, "
    "source, fill_status"
)


def get_trading_cycle_id(
    now_utc: datetime | None = None,
    reset_hour: int = 7,
    reset_minute: int = 30,
) -> str:
    """Return the active trading cycle key cut over at configured morning report time (default 07:30 UK)."""
    dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
    if (dt_uk.hour, dt_uk.minute) < (reset_hour, reset_minute):
        dt_uk = dt_uk - timedelta(days=1)
    return dt_uk.date().isoformat()


def get_9am_cycle_id(now_utc: datetime | None = None) -> str:
    """Backward-compatible alias defaulting to the active trading cycle reset."""
    return get_trading_cycle_id(now_utc)


def _row_to_recommendation(row: tuple[Any, ...]) -> dict[str, Any]:
    """Map a full pick_recommendations row tuple to a backward-compatible dictionary."""
    return {
        "id": int(row[0]),
        "created_at_uk": str(row[1]),
        "created_at_ms": int(row[2]),
        "account": str(row[3]),
        "symbol": str(row[4]),
        "action": str(row[5]),
        "entry_price": row[6],
        "thesis": str(row[7] or ""),
        "horizon_days": int(row[8] if row[8] is not None else 5),
        "benchmark_symbol": str(row[9] or ""),
        "settled_at_uk": str(row[10] or ""),
        "raw_return_pct": row[11],
        "alpha_pct": row[12],
        "outcome_status": str(row[13] or "PENDING"),
        "reflection_lesson": str(row[14] or ""),
        "t10_return_pct": row[15],
        "t10_alpha_pct": row[16],
        "t20_return_pct": row[17],
        "t20_alpha_pct": row[18],
        "mfe_pct": row[19],
        "mae_pct": row[20],
        "source": str(row[21] or "pick"),
        "fill_status": str(row[22] or ""),
    }


async def _ensure_columns(
    db: aiosqlite.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    """Idempotently add missing columns to an existing SQLite table without dropping data."""
    cur = await db.execute(f"PRAGMA table_info({table_name})")
    existing = {str(row[1]) for row in await cur.fetchall()}
    for col_name, col_def in columns.items():
        if col_name not in existing:
            await db.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")


class AdvisorDatabase:
    def __init__(
        self,
        db_path: str | Path = "data/advisor.db",
        cycle_reset_hour: int = 7,
        cycle_reset_minute: int = 30,
    ) -> None:
        self.db_path = Path(db_path)
        self.cycle_reset_hour = int(cycle_reset_hour)
        self.cycle_reset_minute = int(cycle_reset_minute)
        self._write_lock = asyncio.Lock()

    def _cycle_id(self, now_utc: datetime | None = None) -> str:
        return get_trading_cycle_id(
            now_utc=now_utc,
            reset_hour=self.cycle_reset_hour,
            reset_minute=self.cycle_reset_minute,
        )

    @asynccontextmanager
    async def _connect(self, *, write: bool = False) -> AsyncIterator[aiosqlite.Connection]:
        """Open a tuned SQLite connection with busy_timeout and optional in-process write serialization."""
        if write:
            async with self._write_lock:
                async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                    await self._apply_connection_pragmas(db)
                    yield db
        else:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await self._apply_connection_pragmas(db)
                yield db

    @staticmethod
    async def _apply_connection_pragmas(db: aiosqlite.Connection) -> None:
        await db.executescript(
            "PRAGMA busy_timeout=5000;\nPRAGMA synchronous=NORMAL;\nPRAGMA temp_store=MEMORY;"
        )

    async def close(self) -> None:
        """Reserved lifecycle hook for graceful shutdown (connections are scoped per operation)."""
        return None

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
                    fill_status TEXT DEFAULT ''
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

    async def save_snapshot(self, portfolio: PortfolioData) -> None:
        cycle_date = self._cycle_id()
        payload = portfolio.model_dump_json()
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO portfolio_snapshots (cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cycle_date,
                    portfolio.fetched_at_uk,
                    portfolio.fetched_at_ms,
                    portfolio.total_combined_value,
                    payload,
                ),
            )
            await db.commit()

    async def get_previous_snapshot(self, min_age_hours: float = 4.0) -> PortfolioData | None:
        cutoff_ms = int((time.time() - min_age_hours * 3600) * 1000)
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT payload_json FROM portfolio_snapshots
                WHERE fetched_at_ms <= ?
                ORDER BY fetched_at_ms DESC, id DESC LIMIT 1
                """,
                (cutoff_ms,),
            )
            row = await cur.fetchone()
            if row is None:
                cur = await db.execute(
                    "SELECT payload_json FROM portfolio_snapshots ORDER BY fetched_at_ms DESC, id DESC LIMIT 1"
                )
                row = await cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            return PortfolioData.model_validate_json(row[0])
        except Exception as exc:
            logger.warning("Failed to parse portfolio snapshot JSON: %s", exc)
            return None

    async def get_latest_snapshot(self) -> PortfolioData | None:
        """Return the most recent persisted PortfolioData snapshot for offline/5xx fallback."""
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT payload_json FROM portfolio_snapshots ORDER BY fetched_at_ms DESC, id DESC LIMIT 1"
            )
            row = await cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            return PortfolioData.model_validate_json(row[0])
        except Exception as exc:
            logger.warning("Failed to parse latest portfolio snapshot JSON: %s", exc)
            return None

    async def upsert_nav_history(self, row: dict[str, Any]) -> None:
        """Idempotently persist one day's NAV scalar row (GBP-only) for TWR performance.

        Never pruned by ``prune_old_records``: this table is the long-horizon equity
        curve anchor, while ``portfolio_snapshots`` keeps the heavy JSON payloads.
        """
        uk_date = str(row.get("uk_date") or "").strip()[:10]
        if not uk_date:
            return
        now_ms = int(time.time() * 1000)
        payload = (
            uk_date,
            float(row.get("total_value_gbp") or 0.0),
            float(row.get("cash_gbp") or 0.0),
            float(row.get("invested_gbp") or 0.0),
            float(row.get("net_flow_gbp") or 0.0),
            float(row.get("isa_value_gbp") or 0.0),
            float(row.get("invest_value_gbp") or 0.0),
            (None if row.get("benchmark_close") is None else float(row["benchmark_close"])),
            (None if row.get("cash_rate_pct") is None else float(row["cash_rate_pct"])),
            now_ms,
        )
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO nav_history (
                    uk_date, total_value_gbp, cash_gbp, invested_gbp, net_flow_gbp,
                    isa_value_gbp, invest_value_gbp, benchmark_close, cash_rate_pct, created_at_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uk_date) DO UPDATE SET
                    total_value_gbp = excluded.total_value_gbp,
                    cash_gbp = excluded.cash_gbp,
                    invested_gbp = excluded.invested_gbp,
                    net_flow_gbp = excluded.net_flow_gbp,
                    isa_value_gbp = excluded.isa_value_gbp,
                    invest_value_gbp = excluded.invest_value_gbp,
                    benchmark_close = COALESCE(excluded.benchmark_close, nav_history.benchmark_close),
                    cash_rate_pct = COALESCE(excluded.cash_rate_pct, nav_history.cash_rate_pct),
                    created_at_ms = excluded.created_at_ms
                """,
                payload,
            )
            await db.commit()

    async def get_nav_history(self, limit: int = 420) -> list[dict[str, Any]]:
        """Return persisted daily NAV rows ordered by ``uk_date`` ascending (oldest first)."""
        max_rows = max(2, int(limit))
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT uk_date, total_value_gbp, cash_gbp, invested_gbp, net_flow_gbp,
                       isa_value_gbp, invest_value_gbp, benchmark_close, cash_rate_pct
                FROM nav_history
                ORDER BY uk_date DESC
                LIMIT ?
                """,
                (max_rows,),
            )
            rows = await cur.fetchall()
        cols = (
            "uk_date",
            "total_value_gbp",
            "cash_gbp",
            "invested_gbp",
            "net_flow_gbp",
            "isa_value_gbp",
            "invest_value_gbp",
            "benchmark_close",
            "cash_rate_pct",
        )
        out = [dict(zip(cols, row)) for row in rows]
        out.reverse()
        return out

    async def save_morning_report(
        self,
        report_text: str,
        conclusion_text: str,
        guardrails: list[str] | None = None,
        cycle_date: str | None = None,
        context_pack: dict[str, Any] | str | None = None,
    ) -> None:
        cdate = cycle_date or self._cycle_id()
        now_ms = int(time.time() * 1000)
        guardrails_json = json.dumps(guardrails or [], ensure_ascii=False)
        pack_json: str | None
        if context_pack is None:
            pack_json = None
        elif isinstance(context_pack, str):
            pack_json = context_pack
        else:
            pack_json = json.dumps(context_pack, ensure_ascii=False)

        async with self._connect(write=True) as db:
            if pack_json is None:
                await db.execute(
                    """
                    INSERT INTO morning_reports (
                        cycle_date, report_text, conclusion_text, premarket_note,
                        daily_guardrails, context_pack_json, updated_at_ms
                    )
                    VALUES (?, ?, ?, '', ?, '', ?)
                    ON CONFLICT(cycle_date) DO UPDATE SET
                        report_text = excluded.report_text,
                        conclusion_text = excluded.conclusion_text,
                        daily_guardrails = excluded.daily_guardrails,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (cdate, report_text, conclusion_text, guardrails_json, now_ms),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO morning_reports (
                        cycle_date, report_text, conclusion_text, premarket_note,
                        daily_guardrails, context_pack_json, updated_at_ms
                    )
                    VALUES (?, ?, ?, '', ?, ?, ?)
                    ON CONFLICT(cycle_date) DO UPDATE SET
                        report_text = excluded.report_text,
                        conclusion_text = excluded.conclusion_text,
                        daily_guardrails = excluded.daily_guardrails,
                        context_pack_json = excluded.context_pack_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (cdate, report_text, conclusion_text, guardrails_json, pack_json, now_ms),
                )
            await db.commit()

    async def get_context_pack(
        self,
        cycle_date: str | None = None,
        *,
        allow_fallback: bool = True,
    ) -> dict[str, Any] | None:
        """Return the precomputed AnalysisContextPack snapshot for the cycle (or latest if allowed)."""
        cdate = cycle_date or self._cycle_id()
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT context_pack_json FROM morning_reports WHERE cycle_date = ?",
                (cdate,),
            )
            row = await cur.fetchone()
            raw_json = str(row[0] or "").strip() if row else ""
            if not raw_json and allow_fallback:
                cur = await db.execute(
                    """
                    SELECT context_pack_json FROM morning_reports
                    WHERE context_pack_json IS NOT NULL AND TRIM(context_pack_json) != ''
                    ORDER BY updated_at_ms DESC LIMIT 1
                    """
                )
                row = await cur.fetchone()
                raw_json = str(row[0] or "").strip() if row else ""
        if not raw_json:
            return None
        try:
            parsed = json.loads(raw_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.warning("Failed to parse morning context_pack_json: %s", exc)
            return None

    async def get_recent_context_packs(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent valid morning context packs (most recent first).

        Read-only helper for the v5.4.0 deployment gate's 1d-enter / 3d-exit regime
        confirmation hysteresis — the gate engine reads prior raw_regime streaks from
        these packs. Never writes, never alters any table.
        """
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT context_pack_json FROM morning_reports
                WHERE context_pack_json IS NOT NULL AND TRIM(context_pack_json) != ''
                ORDER BY updated_at_ms DESC LIMIT ?
                """,
                (max(0, int(limit)),),
            )
            rows = await cur.fetchall()
        packs: list[dict[str, Any]] = []
        for row in rows:
            raw_json = str(row[0] or "").strip()
            if not raw_json:
                continue
            try:
                parsed = json.loads(raw_json)
            except Exception as exc:
                logger.warning("Failed to parse morning context_pack_json: %s", exc)
                continue
            if isinstance(parsed, dict):
                packs.append(parsed)
        return packs

    async def append_premarket_note(self, memo_text: str, cycle_date: str | None = None) -> None:
        cdate = cycle_date or self._cycle_id()
        now_ms = int(time.time() * 1000)
        async with self._connect(write=True) as db:
            cur = await db.execute(
                "SELECT report_text FROM morning_reports WHERE cycle_date = ?", (cdate,)
            )
            row = await cur.fetchone()
            if row:
                await db.execute(
                    "UPDATE morning_reports SET premarket_note = ?, updated_at_ms = ? WHERE cycle_date = ?",
                    (memo_text, now_ms, cdate),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO morning_reports (cycle_date, report_text, conclusion_text, premarket_note, updated_at_ms)
                    VALUES (?, '', '', ?, ?)
                    """,
                    (cdate, memo_text, now_ms),
                )
            await db.commit()

    async def get_morning_context(
        self, cycle_date: str | None = None, include_premarket: bool = True
    ) -> tuple[str, str]:
        """Return (full_report_with_optional_premarket, short_conclusion) for active cycle or latest available."""
        report_text, conclusion, _stale = await self.get_morning_context_with_status(
            cycle_date, include_premarket=include_premarket
        )
        return report_text, conclusion

    async def get_morning_context_with_status(
        self, cycle_date: str | None = None, include_premarket: bool = True
    ) -> tuple[str, str, bool]:
        """Return (report, conclusion, stale) where stale means the row is from another cycle.

        ``stale=False`` when the requested cycle exists; when it is missing we fall back to
        the latest available report and return ``stale=True`` so callers can label or reject it.
        """
        cdate = cycle_date or self._cycle_id()
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT report_text, conclusion_text, premarket_note FROM morning_reports WHERE cycle_date = ?",
                (cdate,),
            )
            row = await cur.fetchone()
            stale = False
            if not row:
                cur = await db.execute(
                    "SELECT report_text, conclusion_text, premarket_note FROM morning_reports ORDER BY updated_at_ms DESC LIMIT 1"
                )
                row = await cur.fetchone()
                stale = row is not None
        if not row:
            return "", "", False
        report_text, conclusion, pre_note = row
        combined = report_text or ""
        if include_premarket and pre_note:
            combined = f"{combined}\n\n【今日美股开盘前哨补充】\n{pre_note}".strip()
        return combined, (conclusion or ""), stale

    async def get_daily_guardrails(self, cycle_date: str | None = None) -> list[str]:
        """Return audited daily risk vetoes / guardrails from /report for current cycle or latest."""
        guardrails, _stale = await self.get_daily_guardrails_with_status(cycle_date)
        return guardrails

    async def get_daily_guardrails_with_status(
        self, cycle_date: str | None = None
    ) -> tuple[list[str], bool]:
        """Return (guardrails, stale) for the requested cycle, falling back to the latest row."""
        cdate = cycle_date or self._cycle_id()
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT daily_guardrails FROM morning_reports WHERE cycle_date = ?", (cdate,)
            )
            row = await cur.fetchone()
            stale = False
            if not row:
                cur = await db.execute(
                    "SELECT daily_guardrails FROM morning_reports ORDER BY updated_at_ms DESC LIMIT 1"
                )
                row = await cur.fetchone()
                stale = row is not None
        if not row or not row[0]:
            return [], False
        try:
            items = json.loads(row[0])
            return [str(x) for x in items if str(x).strip()] if isinstance(items, list) else [], stale
        except Exception:
            return [], stale

    async def get_past_decision_context(self, limit: int = 5) -> str:
        """Brief prior decision log for Portfolio Manager continuity (TradingAgents Memory Log pattern)."""
        lines: list[str] = []
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT cycle_date, conclusion_text FROM morning_reports ORDER BY updated_at_ms DESC LIMIT 2"
            )
            for cdate, concl in await cur.fetchall():
                if concl:
                    lines.append(f"• [晨报 {cdate}] {str(concl).splitlines()[0][:140]}")
            cur = await db.execute(
                "SELECT created_at_uk, account, symbol, action, entry_price, thesis "
                "FROM pick_recommendations ORDER BY created_at_ms DESC, id DESC LIMIT ?",
                (limit,),
            )
            for dt, acct, sym, act, price, thesis in await cur.fetchall():
                p_str = f" @ {price}" if price is not None else ""
                lines.append(f"• [选股/调仓 {dt}] [{acct}] {act} {sym}{p_str}: {str(thesis)[:80]}")
        return "\n".join(lines) if lines else "（首次运行，暂无历史决策档案）"

    async def save_pick_radar(
        self, focus_query: str, research_text: str, final_text: str, created_at_uk: str
    ) -> None:
        now_ms = int(time.time() * 1000)
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO pick_radars (created_at_uk, created_at_ms, focus_query, research_text, final_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_at_uk, now_ms, focus_query, research_text, final_text),
            )
            await db.commit()

    async def get_latest_pick_radar(self) -> tuple[str, str]:
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT created_at_uk, final_text FROM pick_radars ORDER BY created_at_ms DESC, id DESC LIMIT 1"
            )
            row = await cur.fetchone()
        if not row:
            return "", ""
        return str(row[0]), str(row[1])

    async def log_recommendations(
        self,
        items: list[dict[str, Any]],
        *,
        default_horizon_days: int = 5,
        source: str = "pick",
        dedupe_same_day: bool = False,
    ) -> None:
        if not items:
            return
        now_ms = int(time.time() * 1000)
        now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        fallback_horizon = max(1, int(default_horizon_days))
        default_source = str(source or "pick").strip().lower() or "pick"
        rows_to_insert = [
            (
                str(item.get("created_at_uk") or now_uk),
                int(item.get("created_at_ms") or now_ms),
                str(item.get("account") or "ISA"),
                str(item.get("symbol") or "").strip().upper(),
                str(item.get("action") or "WATCH").strip().upper(),
                item.get("entry_price"),
                str(item.get("thesis") or "")[:300],
                max(1, int(item["horizon_days"])) if item.get("horizon_days") is not None else fallback_horizon,
                str(item.get("benchmark_symbol") or "").strip().upper(),
                str(item.get("source") or default_source).strip().lower() or "pick",
                str(
                    item.get("fill_status")
                    or ("ASSUMED" if default_source == "report" else "")
                ).strip().upper(),
            )
            for item in items
        ]
        async with self._connect(write=True) as db:
            if dedupe_same_day:
                cur = await db.execute(
                    """
                    SELECT account, symbol, action
                    FROM pick_recommendations
                    WHERE source = ? AND substr(created_at_uk, 1, 10) = ?
                    """,
                    (default_source, now_uk[:10]),
                )
                existing = {
                    (str(r[0]).strip().upper(), str(r[1]).strip().upper(), str(r[2]).strip().upper())
                    for r in await cur.fetchall()
                }
                rows_to_insert = [
                    row
                    for row in rows_to_insert
                    if (str(row[2]).strip().upper(), str(row[3]).strip().upper(), str(row[4]).strip().upper())
                    not in existing
                ]
                if not rows_to_insert:
                    return
            await db.executemany(
                """
                INSERT INTO pick_recommendations (
                    created_at_uk, created_at_ms, account, symbol, action, entry_price,
                    thesis, horizon_days, benchmark_symbol, source, fill_status, outcome_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                rows_to_insert,
            )
            await db.commit()

    async def get_recent_recommendations(self, limit: int = 10) -> list[dict[str, Any]]:
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                "ORDER BY created_at_ms DESC, id DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def get_pending_settlements(
        self,
        limit: int = 50,
        *,
        only_with_entry_price: bool = False,
    ) -> list[dict[str, Any]]:
        """Return unsettled recommendation records in chronological order for T+N settlement."""
        price_clause = " AND entry_price IS NOT NULL AND entry_price > 0" if only_with_entry_price else ""
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                f"WHERE (outcome_status IS NULL OR outcome_status = 'PENDING'){price_clause} "
                "ORDER BY created_at_ms ASC, id ASC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def settle_recommendation(
        self,
        rec_id: int,
        *,
        benchmark_symbol: str,
        settled_at_uk: str,
        raw_return_pct: float | None,
        alpha_pct: float | None,
        outcome_status: str = "SETTLED",
        reflection_lesson: str = "",
        mfe_pct: float | None = None,
        mae_pct: float | None = None,
        fill_status: str | None = None,
    ) -> bool:
        """Idempotently settle a PENDING recommendation; returns False if already settled."""
        status = str(outcome_status or "SETTLED").strip().upper()
        bench = str(benchmark_symbol or "").strip().upper()
        lesson = str(reflection_lesson or "").strip()[:600]
        fill = str(fill_status or "").strip().upper()
        async with self._connect(write=True) as db:
            cur = await db.execute(
                """
                UPDATE pick_recommendations
                SET benchmark_symbol = ?,
                    settled_at_uk = ?,
                    raw_return_pct = ?,
                    alpha_pct = ?,
                    outcome_status = ?,
                    reflection_lesson = ?,
                    mfe_pct = COALESCE(?, mfe_pct),
                    mae_pct = COALESCE(?, mae_pct),
                    fill_status = CASE WHEN ? = '' THEN fill_status ELSE ? END
                WHERE id = ? AND (outcome_status IS NULL OR outcome_status = 'PENDING')
                """,
                (
                    bench,
                    str(settled_at_uk),
                    raw_return_pct,
                    alpha_pct,
                    status,
                    lesson,
                    mfe_pct,
                    mae_pct,
                    fill,
                    fill,
                    int(rec_id),
                ),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def get_settled_recommendations_for_followup(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return SETTLED recommendations that still lack T+10 or T+20 follow-up metrics."""
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                "WHERE outcome_status = 'SETTLED' "
                "AND entry_price IS NOT NULL AND entry_price > 0 "
                "AND (t10_alpha_pct IS NULL OR t20_alpha_pct IS NULL) "
                "ORDER BY created_at_ms ASC, id ASC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def update_recommendation_followup(
        self,
        rec_id: int,
        *,
        t10_return_pct: float | None = None,
        t10_alpha_pct: float | None = None,
        t20_return_pct: float | None = None,
        t20_alpha_pct: float | None = None,
        mfe_pct: float | None = None,
        mae_pct: float | None = None,
    ) -> bool:
        """Update non-null T+10/T+20 follow-up returns and MFE/MAE metrics on a SETTLED recommendation."""
        updates: list[str] = []
        params: list[Any] = []
        field_values = (
            ("t10_return_pct", t10_return_pct),
            ("t10_alpha_pct", t10_alpha_pct),
            ("t20_return_pct", t20_return_pct),
            ("t20_alpha_pct", t20_alpha_pct),
            ("mfe_pct", mfe_pct),
            ("mae_pct", mae_pct),
        )
        for col, val in field_values:
            if val is not None:
                updates.append(f"{col} = ?")
                params.append(float(val))
        if not updates:
            return False

        params.append(int(rec_id))
        async with self._connect(write=True) as db:
            cur = await db.execute(
                f"UPDATE pick_recommendations SET {', '.join(updates)} "
                "WHERE id = ? AND outcome_status = 'SETTLED'",
                params,
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def get_symbol_lessons(self, symbol: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return the most recent settled reflection lessons for a specific ticker symbol."""
        sym = str(symbol or "").strip().upper()
        if not sym:
            return []
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                "WHERE symbol = ? AND outcome_status = 'SETTLED' "
                "AND reflection_lesson IS NOT NULL AND TRIM(reflection_lesson) != '' "
                "ORDER BY created_at_ms DESC, id DESC LIMIT ?",
                (sym, limit),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def get_recent_lessons(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent settled reflection lessons across all tickers."""
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                "WHERE outcome_status = 'SETTLED' "
                "AND reflection_lesson IS NOT NULL AND TRIM(reflection_lesson) != '' "
                "ORDER BY created_at_ms DESC, id DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def get_recommendation_stats(self) -> dict[str, Any]:
        """Aggregate settled recommendation win rates, multi-window Alpha, MFE/MAE, and average returns for /stats."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN outcome_status IS NULL OR outcome_status = 'PENDING' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SKIPPED' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'NOT_FILLED' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SETTLED' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SETTLED' AND raw_return_pct > 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SETTLED' AND alpha_pct > 0 THEN 1 ELSE 0 END), 0),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN raw_return_pct END),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN alpha_pct END),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN t10_alpha_pct END),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN t20_alpha_pct END),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN mfe_pct END),
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN mae_pct END)
                FROM pick_recommendations
                """
            )
            row = await cur.fetchone()
        if not row:
            return {
                "total_count": 0,
                "pending_count": 0,
                "skipped_count": 0,
                "not_filled_count": 0,
                "settled_count": 0,
                "win_count": 0,
                "alpha_win_count": 0,
                "win_rate_pct": 0.0,
                "alpha_win_rate_pct": 0.0,
                "avg_raw_return_pct": None,
                "avg_alpha_pct": None,
                "avg_t10_alpha_pct": None,
                "avg_t20_alpha_pct": None,
                "avg_mfe_pct": None,
                "avg_mae_pct": None,
            }
        settled = int(row[4] or 0)
        win_cnt = int(row[5] or 0)
        alpha_win_cnt = int(row[6] or 0)
        return {
            "total_count": int(row[0] or 0),
            "pending_count": int(row[1] or 0),
            "skipped_count": int(row[2] or 0),
            "not_filled_count": int(row[3] or 0),
            "settled_count": settled,
            "win_count": win_cnt,
            "alpha_win_count": alpha_win_cnt,
            "win_rate_pct": round((win_cnt / settled) * 100.0, 1) if settled > 0 else 0.0,
            "alpha_win_rate_pct": round((alpha_win_cnt / settled) * 100.0, 1) if settled > 0 else 0.0,
            "avg_raw_return_pct": round(float(row[7]), 2) if row[7] is not None else None,
            "avg_alpha_pct": round(float(row[8]), 2) if row[8] is not None else None,
            "avg_t10_alpha_pct": round(float(row[9]), 2) if row[9] is not None else None,
            "avg_t20_alpha_pct": round(float(row[10]), 2) if row[10] is not None else None,
            "avg_mfe_pct": round(float(row[11]), 2) if row[11] is not None else None,
            "avg_mae_pct": round(float(row[12]), 2) if row[12] is not None else None,
        }

    async def get_settled_outcome_series(self, limit: int = 400) -> list[dict[str, Any]]:
        """Return raw settled-recommendation Alpha rows (not averaged) for t-stat / decay math."""
        max_rows = max(1, int(limit))
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT symbol, created_at_uk, alpha_pct, t10_alpha_pct, t20_alpha_pct
                FROM pick_recommendations
                WHERE outcome_status = 'SETTLED'
                ORDER BY created_at_ms DESC, id DESC
                LIMIT ?
                """,
                (max_rows,),
            )
            rows = await cur.fetchall()
        return [
            {
                "symbol": row[0],
                "created_at_uk": row[1],
                "alpha_pct": row[2],
                "t10_alpha_pct": row[3],
                "t20_alpha_pct": row[4],
            }
            for row in rows
        ]

    async def append_chat_turn(
        self, user_query: str, bot_reply: str, max_turns: int = 12, cycle_date: str | None = None
    ) -> None:
        cdate = cycle_date or self._cycle_id()
        now_ms = int(time.time() * 1000)
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO chat_history (cycle_date, user_query, bot_reply, created_at_ms)
                VALUES (?, ?, ?, ?)
                """,
                (cdate, user_query, bot_reply[:1600], now_ms),
            )
            await db.execute(
                """
                DELETE FROM chat_history
                WHERE cycle_date = ? AND id NOT IN (
                    SELECT id FROM chat_history WHERE cycle_date = ? ORDER BY created_at_ms DESC, id DESC LIMIT ?
                )
                """,
                (cdate, cdate, max_turns),
            )
            await db.commit()

    async def get_chat_history(self, cycle_date: str | None = None, limit: int = 12) -> list[tuple[str, str]]:
        cdate = cycle_date or self._cycle_id()
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT user_query, bot_reply FROM (
                    SELECT id, user_query, bot_reply, created_at_ms
                    FROM chat_history
                    WHERE cycle_date = ?
                    ORDER BY created_at_ms DESC, id DESC
                    LIMIT ?
                )
                ORDER BY created_at_ms ASC, id ASC
                """,
                (cdate, limit),
            )
            rows = await cur.fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    async def get_previous_cycle_chat_history(
        self, cycle_date: str | None = None, limit: int = 2
    ) -> list[tuple[str, str]]:
        """Return recent chat turns from cycles prior to `cycle_date` for 07:30 UK crossover continuity."""
        cdate = cycle_date or self._cycle_id()
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT user_query, bot_reply FROM (
                    SELECT id, user_query, bot_reply, created_at_ms
                    FROM chat_history
                    WHERE cycle_date < ?
                    ORDER BY created_at_ms DESC, id DESC
                    LIMIT ?
                )
                ORDER BY created_at_ms ASC, id ASC
                """,
                (cdate, limit),
            )
            rows = await cur.fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    async def clear_chat_history(self, cycle_date: str | None = None) -> None:
        cdate = cycle_date or self._cycle_id()
        async with self._connect(write=True) as db:
            await db.execute("DELETE FROM chat_history WHERE cycle_date = ?", (cdate,))
            await db.commit()

    async def log_llm_usage(self, record: dict[str, Any]) -> None:
        now_utc = datetime.now(timezone.utc)
        now_ms = int(now_utc.timestamp() * 1000)
        now_uk = now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        cdate = str(record.get("cycle_date") or self._cycle_id(now_utc))
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO llm_usage_logs (
                    created_at_uk, created_at_ms, cycle_date, call_label, model, thinking_level,
                    api_rounds, prompt_tokens, cached_tokens, output_tokens, thought_tokens,
                    total_tokens, estimated_cost_usd, latency_ms, success, search_queries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_uk,
                    now_ms,
                    cdate,
                    str(record.get("call_label") or "unknown"),
                    str(record.get("model") or ""),
                    str(record.get("thinking_level") or "DEFAULT"),
                    int(record.get("api_rounds") or 1),
                    int(record.get("prompt_tokens") or 0),
                    int(record.get("cached_tokens") or 0),
                    int(record.get("output_tokens") or 0),
                    int(record.get("thought_tokens") or 0),
                    int(record.get("total_tokens") or 0),
                    float(record.get("estimated_cost_usd") or 0.0),
                    int(record.get("latency_ms") or 0),
                    1 if record.get("success", True) else 0,
                    int(record.get("search_queries") or 0),
                ),
            )
            await db.commit()

    async def get_usage_summary(self, days: int = 30, now_utc: datetime | None = None) -> dict[str, Any]:
        ref = now_utc or datetime.now(timezone.utc)
        cdate = self._cycle_id(ref)
        now_ms = int(ref.timestamp() * 1000)
        cutoff_7d = now_ms - 7 * 86400 * 1000
        cutoff_nd = now_ms - max(days, 1) * 86400 * 1000
        month_start_uk = ref.astimezone(LONDON_TZ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        cutoff_month_ms = int(month_start_uk.timestamp() * 1000)

        agg_cols = (
            "COUNT(*), COALESCE(SUM(api_rounds), 0), COALESCE(SUM(prompt_tokens), 0), "
            "COALESCE(SUM(cached_tokens), 0), COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(thought_tokens), 0), COALESCE(SUM(estimated_cost_usd), 0.0), "
            "COALESCE(SUM(search_queries), 0)"
        )

        def _row_to_dict(row: tuple | None) -> dict[str, Any]:
            if not row:
                return {
                    "calls": 0,
                    "rounds": 0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "thought_tokens": 0,
                    "cost_usd": 0.0,
                    "search_queries": 0,
                }
            return {
                "calls": int(row[0] or 0),
                "rounds": int(row[1] or 0),
                "prompt_tokens": int(row[2] or 0),
                "cached_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "thought_tokens": int(row[5] or 0),
                "cost_usd": float(row[6] or 0.0),
                "search_queries": int(row[7] or 0),
            }

        async with self._connect() as db:
            cur = await db.execute(f"SELECT {agg_cols} FROM llm_usage_logs WHERE cycle_date = ?", (cdate,))
            today = _row_to_dict(await cur.fetchone())

            cur = await db.execute(f"SELECT {agg_cols} FROM llm_usage_logs WHERE created_at_ms >= ?", (cutoff_7d,))
            last_7d = _row_to_dict(await cur.fetchone())

            cur = await db.execute(f"SELECT {agg_cols} FROM llm_usage_logs WHERE created_at_ms >= ?", (cutoff_nd,))
            last_nd = _row_to_dict(await cur.fetchone())

            cur = await db.execute(
                f"SELECT {agg_cols} FROM llm_usage_logs WHERE created_at_ms >= ?",
                (cutoff_month_ms,),
            )
            this_month = _row_to_dict(await cur.fetchone())

            cur = await db.execute(
                f"SELECT call_label, {agg_cols} FROM llm_usage_logs "
                "WHERE created_at_ms >= ? GROUP BY call_label "
                "ORDER BY SUM(prompt_tokens + output_tokens + thought_tokens) DESC",
                (cutoff_7d,),
            )
            by_label_7d = [
                {"label": str(r[0]), **_row_to_dict(r[1:])}
                for r in await cur.fetchall()
            ]

            cur = await db.execute(
                f"SELECT model, {agg_cols} FROM llm_usage_logs "
                "WHERE created_at_ms >= ? GROUP BY model "
                "ORDER BY SUM(prompt_tokens + output_tokens + thought_tokens) DESC",
                (cutoff_nd,),
            )
            by_model_nd = [
                {"model": str(r[0]), **_row_to_dict(r[1:])}
                for r in await cur.fetchall()
            ]

        return {
            "cycle_date": cdate,
            "days": days,
            "today": today,
            "last_7d": last_7d,
            "last_nd": last_nd,
            "this_month": this_month,
            "by_label_7d": by_label_7d,
            "by_model_nd": by_model_nd,
        }

    async def log_ops_event(
        self,
        *,
        severity: str,
        source: str,
        alert_key: str,
        message: str,
        notified: bool,
        suppressed_count: int = 0,
    ) -> None:
        now_utc = datetime.now(timezone.utc)
        now_ms = int(now_utc.timestamp() * 1000)
        now_uk = now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO ops_events (
                    created_at_uk, created_at_ms, severity, source, alert_key,
                    message, notified, suppressed_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_uk,
                    now_ms,
                    severity.upper(),
                    source,
                    alert_key,
                    message[:1000],
                    1 if notified else 0,
                    int(suppressed_count),
                ),
            )
            await db.commit()

    async def get_recent_ops_events(self, limit: int = 10) -> list[dict[str, Any]]:
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT created_at_uk, severity, source, alert_key, message, notified, suppressed_count
                FROM ops_events
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        return [
            {
                "created_at_uk": str(r[0]),
                "severity": str(r[1]),
                "source": str(r[2]),
                "alert_key": str(r[3]),
                "message": str(r[4]),
                "notified": bool(r[5]),
                "suppressed_count": int(r[6] or 0),
            }
            for r in rows
        ]

    async def get_unnotified_ops_events(
        self,
        *,
        hours: int = 24,
        severities: tuple[str, ...] | None = ("CRITICAL", "WARN"),
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch recent unnotified ops events for post-restart alert recovery (IMP-17)."""
        cutoff_ms = int((time.time() - max(hours, 1) * 3600) * 1000)
        params: list[Any] = [cutoff_ms]
        sev_clause = ""
        if severities:
            clean_sevs = [str(s).strip().upper() for s in severities if str(s).strip()]
            if clean_sevs:
                placeholders = ", ".join("?" for _ in clean_sevs)
                sev_clause = f" AND severity IN ({placeholders})"
                params.extend(clean_sevs)
        params.append(limit)

        async with self._connect() as db:
            cur = await db.execute(
                f"""
                SELECT id, created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count
                FROM ops_events
                WHERE notified = 0 AND created_at_ms >= ?{sev_clause}
                ORDER BY created_at_ms ASC, id ASC LIMIT ?
                """,
                tuple(params),
            )
            rows = await cur.fetchall()
        return [
            {
                "id": int(r[0]),
                "created_at_uk": str(r[1]),
                "created_at_ms": int(r[2]),
                "severity": str(r[3]),
                "source": str(r[4]),
                "alert_key": str(r[5]),
                "message": str(r[6]),
                "notified": bool(r[7]),
                "suppressed_count": int(r[8] or 0),
            }
            for r in rows
        ]

    async def mark_ops_events_notified(self, event_ids: list[int]) -> int:
        """Mark recovered ops events as notified=1; returns the number of updated rows."""
        clean_ids = sorted({int(eid) for eid in event_ids if int(eid) > 0})
        if not clean_ids:
            return 0
        placeholders = ", ".join("?" for _ in clean_ids)
        async with self._connect(write=True) as db:
            cur = await db.execute(
                f"UPDATE ops_events SET notified = 1 WHERE notified = 0 AND id IN ({placeholders})",
                tuple(clean_ids),
            )
            await db.commit()
            return int(cur.rowcount or 0)

    async def prune_old_records(
        self,
        retention_days: int = 90,
        *,
        now_utc: datetime | None = None,
    ) -> dict[str, int]:
        """Delete expired rows from portfolio_snapshots, llm_usage_logs, ops_events, and ohlcv_cache while preserving archives and latest snapshot."""
        ref_sec = (now_utc or datetime.now(timezone.utc)).timestamp()
        cutoff_ms = int((ref_sec - max(retention_days, 1) * 86400) * 1000)
        async with self._connect(write=True) as db:
            cur_snap = await db.execute(
                """
                DELETE FROM portfolio_snapshots
                WHERE fetched_at_ms < ?
                  AND id != COALESCE(
                      (SELECT id FROM portfolio_snapshots ORDER BY fetched_at_ms DESC, id DESC LIMIT 1),
                      -1
                  )
                """,
                (cutoff_ms,),
            )
            cur_usage = await db.execute(
                "DELETE FROM llm_usage_logs WHERE created_at_ms < ?",
                (cutoff_ms,),
            )
            cur_ops = await db.execute(
                "DELETE FROM ops_events WHERE created_at_ms < ?",
                (cutoff_ms,),
            )
            cur_ohlcv = await db.execute(
                "DELETE FROM ohlcv_cache WHERE updated_at_ms < ?",
                (cutoff_ms,),
            )
            cur_grounding = await db.execute(
                "DELETE FROM grounding_news_cache WHERE fetched_at_ms < ?",
                (cutoff_ms,),
            )
            # Pending-order snapshots are a short-lived diff anchor: keep 3 days only.
            cur_pending = await db.execute(
                "DELETE FROM pending_order_snapshots WHERE last_seen_ms < ?",
                (int((ref_sec - 3 * 86400) * 1000),),
            )
            await db.commit()
            return {
                "portfolio_snapshots": int(cur_snap.rowcount or 0),
                "llm_usage_logs": int(cur_usage.rowcount or 0),
                "ops_events": int(cur_ops.rowcount or 0),
                "ohlcv_cache": int(cur_ohlcv.rowcount or 0),
                "grounding_news_cache": int(cur_grounding.rowcount or 0),
                "pending_order_snapshots": int(cur_pending.rowcount or 0),
            }

    async def has_scheduler_job_fired(self, job_name: str, uk_date: str) -> bool:
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT 1 FROM scheduler_runs WHERE job_name = ? AND uk_date = ? LIMIT 1",
                (job_name, uk_date),
            )
            return (await cur.fetchone()) is not None

    async def mark_scheduler_job_fired(self, job_name: str, uk_date: str, fired_at_uk: str) -> None:
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO scheduler_runs (job_name, uk_date, fired_at_uk)
                VALUES (?, ?, ?)
                """,
                (job_name, uk_date, fired_at_uk),
            )
            await db.commit()

    async def clear_scheduler_job_fired(self, job_name: str, uk_date: str) -> None:
        """Release a claimed one-shot run so a transient failure can be retried."""
        async with self._connect(write=True) as db:
            await db.execute(
                "DELETE FROM scheduler_runs WHERE job_name = ? AND uk_date = ?",
                (job_name, uk_date),
            )
            await db.commit()

    async def upsert_pending_order_snapshots(
        self,
        rows: list[dict[str, Any]],
        *,
        seen_at_ms: int = 0,
        seen_at_uk: str = "",
    ) -> int:
        """Insert-or-refresh live pending-order rows, preserving each row's original first_seen anchor."""
        now_ms = int(seen_at_ms or time.time() * 1000)
        uk_str = str(seen_at_uk or "").strip() or datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime(
            "%Y-%m-%d %H:%M"
        )
        payload: list[tuple[Any, ...]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            account = str(row.get("account_label") or "").strip()
            order_id = str(row.get("order_id") or "").strip()
            if not account or not order_id:
                continue

            def _num(value: Any) -> float | None:
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    return None
                return num if math.isfinite(num) else None

            payload.append(
                (
                    account,
                    order_id,
                    str(row.get("yahoo_symbol") or "").strip().upper(),
                    str(row.get("side") or "").strip().upper(),
                    _num(row.get("quantity")) or 0.0,
                    _num(row.get("filled_quantity")) or 0.0,
                    _num(row.get("limit_price")),
                    _num(row.get("stop_price")),
                    str(row.get("order_status") or ""),
                    str(row.get("currency") or ""),
                    uk_str,
                    now_ms,
                    uk_str,
                    now_ms,
                )
            )
        if not payload:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                """
                INSERT INTO pending_order_snapshots (
                    account_label, order_id, yahoo_symbol, side, quantity,
                    filled_quantity, limit_price, stop_price, order_status, currency,
                    first_seen_uk, first_seen_ms, last_seen_uk, last_seen_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_label, order_id) DO UPDATE SET
                    yahoo_symbol = excluded.yahoo_symbol,
                    side = excluded.side,
                    quantity = excluded.quantity,
                    filled_quantity = excluded.filled_quantity,
                    limit_price = excluded.limit_price,
                    stop_price = excluded.stop_price,
                    order_status = excluded.order_status,
                    currency = excluded.currency,
                    last_seen_uk = excluded.last_seen_uk,
                    last_seen_ms = excluded.last_seen_ms
                """,
                payload,
            )
            await db.commit()
        return len(payload)

    async def get_pending_order_snapshots(self) -> list[dict[str, Any]]:
        """Return the last-seen live pending-order snapshot rows."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT account_label, order_id, yahoo_symbol, side, quantity,
                       filled_quantity, limit_price, stop_price, order_status, currency,
                       first_seen_uk, first_seen_ms, last_seen_uk, last_seen_ms
                FROM pending_order_snapshots
                ORDER BY account_label ASC, order_id ASC
                """
            )
            rows = await cur.fetchall()
        return [
            {
                "account_label": str(r[0] or ""),
                "order_id": str(r[1] or ""),
                "yahoo_symbol": str(r[2] or ""),
                "side": str(r[3] or ""),
                "quantity": float(r[4] or 0.0),
                "filled_quantity": float(r[5] or 0.0),
                "limit_price": None if r[6] is None else float(r[6]),
                "stop_price": None if r[7] is None else float(r[7]),
                "order_status": str(r[8] or ""),
                "currency": str(r[9] or ""),
                "first_seen_uk": str(r[10] or ""),
                "first_seen_ms": int(r[11] or 0),
                "last_seen_uk": str(r[12] or ""),
                "last_seen_ms": int(r[13] or 0),
            }
            for r in rows
        ]

    async def delete_pending_order_snapshots(self, keys: list[tuple[str, str]]) -> int:
        """Delete snapshot rows for orders that already vanished (filled or cancelled)."""
        clean = [
            (str(account or "").strip(), str(order_id or "").strip())
            for account, order_id in (keys or [])
            if str(account or "").strip() and str(order_id or "").strip()
        ]
        if not clean:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                "DELETE FROM pending_order_snapshots WHERE account_label = ? AND order_id = ?",
                clean,
            )
            await db.commit()
        return len(clean)

    async def get_cached_ohlcv(self, symbols: list[str], period: str = "2y") -> dict[str, dict[str, Any]]:
        """Load persisted OHLCV JSON cache rows for the given symbols and period."""
        clean_syms = list(dict.fromkeys(str(s or "").strip() for s in symbols if str(s or "").strip()))
        clean_period = str(period or "2y").strip()
        if not clean_syms or not clean_period:
            return {}
        out: dict[str, dict[str, Any]] = {}
        async with self._connect() as db:
            for idx in range(0, len(clean_syms), 500):
                chunk = clean_syms[idx : idx + 500]
                placeholders = ", ".join("?" for _ in chunk)
                cur = await db.execute(
                    f"""
                    SELECT symbol, period, last_bar_date, last_close, updated_at_ms, payload_json
                    FROM ohlcv_cache
                    WHERE period = ? AND symbol IN ({placeholders})
                    """,
                    (clean_period, *chunk),
                )
                for row in await cur.fetchall():
                    payload_json = str(row[5] or "").strip()
                    last_close = float(row[3] or 0.0)
                    if not payload_json or payload_json in ("{}", "[]") or not math.isfinite(last_close) or last_close <= 0:
                        continue
                    sym_key = str(row[0])
                    out[sym_key] = {
                        "symbol": sym_key,
                        "period": str(row[1]),
                        "last_bar_date": str(row[2]),
                        "last_close": last_close,
                        "updated_at_ms": int(row[4] or 0),
                        "payload_json": payload_json,
                    }
        return out

    async def save_cached_ohlcv(self, entries: list[dict[str, Any]]) -> int:
        """Persist non-empty OHLCV history cache rows under the single write lock."""
        if not entries:
            return 0
        now_ms = int(time.time() * 1000)
        rows_to_write: list[tuple[str, str, str, float, int, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol") or "").strip()
            period = str(entry.get("period") or "").strip()
            last_bar_date = str(entry.get("last_bar_date") or "").strip()
            try:
                last_close = float(entry.get("last_close") or 0.0)
            except (TypeError, ValueError):
                continue
            raw_payload = entry.get("payload_json")
            if isinstance(raw_payload, (dict, list)):
                if not raw_payload:
                    continue
                payload_json = json.dumps(raw_payload)
            else:
                payload_json = str(raw_payload or "").strip()
            if (
                not sym
                or not period
                or not last_bar_date
                or not math.isfinite(last_close)
                or last_close <= 0
                or not payload_json
                or payload_json in ("{}", "[]")
            ):
                continue
            raw_ts = entry.get("updated_at_ms")
            try:
                updated_at_ms = int(raw_ts) if raw_ts is not None and int(raw_ts) > 0 else now_ms
            except (TypeError, ValueError):
                updated_at_ms = now_ms
            rows_to_write.append((sym, period, last_bar_date, last_close, updated_at_ms, payload_json))

        if not rows_to_write:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO ohlcv_cache (
                    symbol, period, last_bar_date, last_close, updated_at_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_write,
            )
            await db.commit()
        return len(rows_to_write)

    async def upsert_t212_instruments(
        self,
        items: list[dict[str, Any]],
        *,
        account_scope: str | None = None,
        full_snapshot: bool = False,
    ) -> int:
        """Upsert Trading 212 instrument metadata into the long-term archive table."""
        if not items:
            return 0
        now_ms = int(time.time() * 1000)
        scope = str(account_scope or "").strip().upper()
        rows_to_write: list[tuple[str, str, str, str, str, str, int, int | None, int, int, int]] = []
        seen_tickers: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip()
            if not ticker:
                continue
            yahoo_symbol = str(item.get("yahoo_symbol") or "").strip()
            isin = str(item.get("isin") or "").strip().upper()
            currency_code = str(item.get("currency_code") or item.get("currencyCode") or "").strip()
            short_name = str(item.get("short_name") or item.get("shortName") or item.get("name") or "").strip()
            inst_type = str(item.get("inst_type") or item.get("type") or "").strip().upper()
            raw_ext = item.get("extended_hours") if "extended_hours" in item else item.get("extendedHours", False)
            extended_hours = 1 if bool(raw_ext) else 0
            raw_sched = item.get("working_schedule_id") if "working_schedule_id" in item else item.get("workingScheduleId")
            working_schedule_id: int | None = None
            if raw_sched is not None and str(raw_sched).strip() != "":
                try:
                    working_schedule_id = int(raw_sched)
                except (TypeError, ValueError):
                    working_schedule_id = None
            raw_ts = item.get("updated_at_ms")
            try:
                updated_at_ms = int(raw_ts) if raw_ts is not None and int(raw_ts) > 0 else now_ms
            except (TypeError, ValueError):
                updated_at_ms = now_ms

            if scope == "ISA":
                isa_elig = 1
                inv_elig = 1 if bool(item.get("invest_eligible", False)) else 0
            elif scope == "INVEST":
                isa_elig = 1 if bool(item.get("isa_eligible", False)) else 0
                inv_elig = 1
            else:
                isa_elig = 1 if bool(item.get("isa_eligible", True)) else 0
                inv_elig = 1 if bool(item.get("invest_eligible", True)) else 0

            seen_tickers.append(ticker)
            rows_to_write.append(
                (
                    ticker,
                    yahoo_symbol,
                    isin,
                    currency_code,
                    short_name,
                    inst_type,
                    extended_hours,
                    working_schedule_id,
                    isa_elig,
                    inv_elig,
                    updated_at_ms,
                )
            )

        if not rows_to_write:
            return 0
        async with self._connect(write=True) as db:
            if full_snapshot and scope == "ISA":
                await db.execute("UPDATE t212_instruments_metadata SET isa_eligible = 0")
            elif full_snapshot and scope == "INVEST":
                await db.execute("UPDATE t212_instruments_metadata SET invest_eligible = 0")

            if scope == "ISA":
                sql = """
                INSERT INTO t212_instruments_metadata (
                    ticker, yahoo_symbol, isin, currency_code, short_name,
                    inst_type, extended_hours, working_schedule_id,
                    isa_eligible, invest_eligible, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    yahoo_symbol = excluded.yahoo_symbol,
                    isin = excluded.isin,
                    currency_code = excluded.currency_code,
                    short_name = excluded.short_name,
                    inst_type = excluded.inst_type,
                    extended_hours = excluded.extended_hours,
                    working_schedule_id = excluded.working_schedule_id,
                    isa_eligible = 1,
                    updated_at_ms = excluded.updated_at_ms
                """
            elif scope == "INVEST":
                sql = """
                INSERT INTO t212_instruments_metadata (
                    ticker, yahoo_symbol, isin, currency_code, short_name,
                    inst_type, extended_hours, working_schedule_id,
                    isa_eligible, invest_eligible, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    yahoo_symbol = excluded.yahoo_symbol,
                    isin = excluded.isin,
                    currency_code = excluded.currency_code,
                    short_name = excluded.short_name,
                    inst_type = excluded.inst_type,
                    extended_hours = excluded.extended_hours,
                    working_schedule_id = excluded.working_schedule_id,
                    invest_eligible = 1,
                    updated_at_ms = excluded.updated_at_ms
                """
            else:
                sql = """
                INSERT OR REPLACE INTO t212_instruments_metadata (
                    ticker, yahoo_symbol, isin, currency_code, short_name,
                    inst_type, extended_hours, working_schedule_id,
                    isa_eligible, invest_eligible, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            await db.executemany(sql, rows_to_write)
            await db.commit()
        return len(rows_to_write)

    async def get_t212_instrument_map(self) -> dict[str, dict[str, Any]]:
        """Return {raw_t212_ticker: metadata_dict} from t212_instruments_metadata."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT ticker, yahoo_symbol, isin, currency_code, short_name,
                       inst_type, extended_hours, working_schedule_id,
                       isa_eligible, invest_eligible, updated_at_ms
                FROM t212_instruments_metadata
                """
            )
            rows = await cur.fetchall()
        return {
            str(r[0]): {
                "ticker": str(r[0]),
                "yahoo_symbol": str(r[1] or ""),
                "isin": str(r[2] or ""),
                "currency_code": str(r[3] or ""),
                "short_name": str(r[4] or ""),
                "inst_type": str(r[5] or ""),
                "extended_hours": bool(r[6]),
                "working_schedule_id": int(r[7]) if r[7] is not None else None,
                "isa_eligible": bool(r[8]) if r[8] is not None else True,
                "invest_eligible": bool(r[9]) if r[9] is not None else True,
                "updated_at_ms": int(r[10] or 0),
            }
            for r in rows
            if r[0]
        }

    async def get_t212_eligibility_by_yahoo(self) -> dict[str, dict[str, Any]]:
        """Return {YAHOO_SYMBOL_UPPER: aggregated eligibility & metadata} from SQLite."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT yahoo_symbol, inst_type, currency_code, isin,
                       MAX(isa_eligible) AS isa_ok, MAX(invest_eligible) AS inv_ok,
                       MAX(updated_at_ms) AS updated_ms
                FROM t212_instruments_metadata
                WHERE yahoo_symbol != ''
                GROUP BY UPPER(yahoo_symbol)
                """
            )
            rows = await cur.fetchall()
        return {
            str(r[0]).strip().upper(): {
                "yahoo_symbol": str(r[0]).strip().upper(),
                "inst_type": str(r[1] or "").strip().upper(),
                "currency_code": str(r[2] or "").strip().upper(),
                "isin": str(r[3] or "").strip().upper(),
                "isa_eligible": bool(r[4]),
                "invest_eligible": bool(r[5]),
                "updated_at_ms": int(r[6] or 0),
            }
            for r in rows
            if r[0] and str(r[0]).strip()
        }

    async def get_fundamentals_cache(
        self,
        symbols: list[str],
        *,
        max_age_days: int = 21,
    ) -> dict[str, dict[str, Any]]:
        """Load non-expired cached fundamental snapshots for symbols."""
        clean_syms = list(dict.fromkeys(str(s or "").strip().upper() for s in symbols if str(s or "").strip()))
        if not clean_syms:
            return {}
        cutoff_ms = int((time.time() - max(1, int(max_age_days)) * 86400) * 1000)
        out: dict[str, dict[str, Any]] = {}
        async with self._connect() as db:
            for idx in range(0, len(clean_syms), 400):
                chunk = clean_syms[idx : idx + 400]
                placeholders = ",".join("?" for _ in chunk)
                cur = await db.execute(
                    f"""
                    SELECT symbol, currency, sector, industry, quote_type,
                           market_cap, trailing_pe, forward_pe, roe_pct,
                           profit_margin_pct, revenue_growth_pct, debt_to_equity,
                           dividend_yield_pct, analyst_count, payload_json, updated_at_ms
                    FROM fundamentals_cache
                    WHERE UPPER(symbol) IN ({placeholders}) AND updated_at_ms >= ?
                    """,
                    (*chunk, cutoff_ms),
                )
                rows = await cur.fetchall()
                for r in rows:
                    sym = str(r[0]).strip().upper()
                    extra: dict[str, Any] = {}
                    if r[14]:
                        try:
                            loaded = json.loads(str(r[14]))
                            if isinstance(loaded, dict):
                                extra = loaded
                        except Exception:
                            extra = {}
                    out[sym] = {
                        **extra,
                        "symbol": sym,
                        "currency": str(r[1] or ""),
                        "sector": str(r[2] or ""),
                        "industry": str(r[3] or ""),
                        "quote_type": str(r[4] or ""),
                        "market_cap": float(r[5]) if r[5] is not None else None,
                        "trailing_pe": float(r[6]) if r[6] is not None else None,
                        "forward_pe": float(r[7]) if r[7] is not None else None,
                        "roe_pct": float(r[8]) if r[8] is not None else None,
                        "profit_margin_pct": float(r[9]) if r[9] is not None else None,
                        "revenue_growth_pct": float(r[10]) if r[10] is not None else None,
                        "debt_to_equity": float(r[11]) if r[11] is not None else None,
                        "dividend_yield_pct": float(r[12]) if r[12] is not None else None,
                        "analyst_count": int(r[13]) if r[13] is not None else None,
                        "updated_at_ms": int(r[15] or 0),
                    }
        return out

    async def upsert_fundamentals_cache(self, items: list[dict[str, Any]]) -> int:
        """Upsert cached Yahoo fundamental snapshots."""
        if not items:
            return 0
        now_ms = int(time.time() * 1000)
        rows: list[tuple[Any, ...]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or "").strip().upper()
            if not sym:
                continue
            payload = {
                k: item.get(k)
                for k in ("short_name", "exchange", "recommendation_key", "beta", "target_mean_price")
                if item.get(k) is not None
            }
            rows.append(
                (
                    sym,
                    str(item.get("currency") or ""),
                    str(item.get("sector") or ""),
                    str(item.get("industry") or ""),
                    str(item.get("quote_type") or ""),
                    item.get("market_cap"),
                    item.get("trailing_pe"),
                    item.get("forward_pe"),
                    item.get("roe_pct"),
                    item.get("profit_margin_pct"),
                    item.get("revenue_growth_pct"),
                    item.get("debt_to_equity"),
                    item.get("dividend_yield_pct"),
                    item.get("analyst_count"),
                    json.dumps(payload, ensure_ascii=False),
                    int(item.get("updated_at_ms") or now_ms),
                )
            )
        if not rows:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO fundamentals_cache (
                    symbol, currency, sector, industry, quote_type,
                    market_cap, trailing_pe, forward_pe, roe_pct,
                    profit_margin_pct, revenue_growth_pct, debt_to_equity,
                    dividend_yield_pct, analyst_count, payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            await db.commit()
        return len(rows)

    async def replace_active_universe_pool(
        self,
        items: list[dict[str, Any]],
        *,
        updated_at_uk: str = "",
    ) -> int:
        """Atomically replace the curated active universe pool in SQLite."""
        now_ms = int(time.time() * 1000)
        uk_str = updated_at_uk or self._now_uk_str()
        rows: list[tuple[Any, ...]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or "").strip().upper()
            if not sym:
                continue
            idx_src = item.get("index_sources") or ""
            if isinstance(idx_src, (list, tuple, set)):
                idx_src_str = ",".join(str(x).strip() for x in idx_src if str(x).strip())
            else:
                idx_src_str = str(idx_src).strip()
            rows.append(
                (
                    sym,
                    str(item.get("name") or sym),
                    str(item.get("exchange") or ""),
                    str(item.get("currency") or ""),
                    str(item.get("sector") or ""),
                    str(item.get("industry") or ""),
                    idx_src_str,
                    float(item.get("quality_score") or 0.0),
                    float(item.get("avg_daily_turnover_usd") or 0.0),
                    1 if bool(item.get("isa_eligible", True)) else 0,
                    1 if bool(item.get("invest_eligible", True)) else 0,
                    1 if bool(item.get("is_etf", False)) else 0,
                    uk_str,
                    now_ms,
                )
            )
        async with self._connect(write=True) as db:
            await db.execute("DELETE FROM active_universe_pool")
            if rows:
                await db.executemany(
                    """
                    INSERT INTO active_universe_pool (
                        symbol, name, exchange, currency, sector, industry,
                        index_sources, quality_score, avg_daily_turnover_usd,
                        isa_eligible, invest_eligible, is_etf, updated_at_uk, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            await db.commit()
        return len(rows)

    async def get_active_universe_pool(self) -> list[dict[str, Any]]:
        """Return the active auditioned universe pool sorted by ETF flag & quality score."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT symbol, name, exchange, currency, sector, industry,
                       index_sources, quality_score, avg_daily_turnover_usd,
                       isa_eligible, invest_eligible, is_etf, updated_at_uk, updated_at_ms
                FROM active_universe_pool
                ORDER BY is_etf ASC, quality_score DESC, symbol ASC
                """
            )
            rows = await cur.fetchall()
        return [
            {
                "symbol": str(r[0]).strip().upper(),
                "name": str(r[1] or ""),
                "exchange": str(r[2] or ""),
                "currency": str(r[3] or ""),
                "sector": str(r[4] or ""),
                "industry": str(r[5] or ""),
                "index_sources": [s for s in str(r[6] or "").split(",") if s],
                "quality_score": float(r[7] or 0.0),
                "avg_daily_turnover_usd": float(r[8] or 0.0),
                "isa_eligible": bool(r[9]),
                "invest_eligible": bool(r[10]),
                "is_etf": bool(r[11]),
                "updated_at_uk": str(r[12] or ""),
                "updated_at_ms": int(r[13] or 0),
            }
            for r in rows
            if r[0]
        ]

    async def save_index_constituents(
        self,
        index_name: str,
        symbols: list[str],
        source_url: str = "",
    ) -> None:
        """Persist refreshed index constituent ticker list."""
        clean_name = str(index_name or "").strip().upper()
        clean_syms = list(dict.fromkeys(str(s or "").strip().upper() for s in symbols if str(s or "").strip()))
        if not clean_name or not clean_syms:
            return
        now_ms = int(time.time() * 1000)
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO index_constituents_cache (
                    index_name, symbols_json, source_url, updated_at_ms
                ) VALUES (?, ?, ?, ?)
                """,
                (clean_name, json.dumps(clean_syms, ensure_ascii=False), str(source_url or ""), now_ms),
            )
            await db.commit()

    async def get_index_constituents(
        self,
        index_name: str,
        *,
        max_age_days: int = 30,
    ) -> list[str] | None:
        """Load cached index constituent tickers if present and within max_age_days."""
        clean_name = str(index_name or "").strip().upper()
        if not clean_name:
            return None
        cutoff_ms = int((time.time() - max(1, int(max_age_days)) * 86400) * 1000)
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT symbols_json FROM index_constituents_cache
                WHERE index_name = ? AND updated_at_ms >= ?
                """,
                (clean_name, cutoff_ms),
            )
            row = await cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            data = json.loads(str(row[0]))
            if isinstance(data, list):
                return [str(s).strip().upper() for s in data if str(s).strip()]
        except Exception:
            return None
        return None

    async def upsert_dividends(self, items: list[dict[str, Any]]) -> int:
        """Archive historical GBP dividend payouts without duplicates."""
        if not items:
            return 0
        rows_to_write: list[tuple[str, str, str, str, float, float, str, int]] = []
        for raw in items:
            d = raw.model_dump() if hasattr(raw, "model_dump") else raw
            if not isinstance(d, dict):
                continue
            div_id = str(d.get("id") or "").strip()
            account = str(d.get("account") or "").strip()
            ticker = str(d.get("ticker") or d.get("raw_ticker") or "").strip()
            yahoo_symbol = str(d.get("yahoo_symbol") or "").strip()
            raw_amt = d.get("amount_gbp") if d.get("amount_gbp") is not None else d.get("amount")
            try:
                amount_gbp = float(raw_amt or 0.0)
                gross_per_share = float(d.get("gross_per_share") or 0.0)
            except (TypeError, ValueError):
                continue
            if not div_id or not account or not yahoo_symbol or not math.isfinite(amount_gbp) or amount_gbp <= 0:
                continue
            paid_on = str(d.get("paid_on") or "").strip()
            try:
                paid_on_ms = int(d.get("paid_on_ms") or 0)
            except (TypeError, ValueError):
                paid_on_ms = 0
            rows_to_write.append(
                (
                    div_id,
                    account,
                    ticker,
                    yahoo_symbol,
                    amount_gbp,
                    gross_per_share if math.isfinite(gross_per_share) else 0.0,
                    paid_on,
                    paid_on_ms,
                )
            )

        if not rows_to_write:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO t212_dividend_archive (
                    id, account, ticker, yahoo_symbol, amount_gbp,
                    gross_per_share, paid_on, paid_on_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_write,
            )
            await db.commit()
        return len(rows_to_write)

    async def get_cumulative_dividends_by_symbol(self) -> dict[tuple[str, str], float]:
        """Aggregate archived GBP dividends grouped by (account, yahoo_symbol)."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT account, yahoo_symbol, COALESCE(SUM(amount_gbp), 0.0)
                FROM t212_dividend_archive
                WHERE account != '' AND yahoo_symbol != ''
                GROUP BY account, yahoo_symbol
                """
            )
            rows = await cur.fetchall()
        return {
            (str(r[0]), str(r[1])): round(float(r[2] or 0.0), 4)
            for r in rows
            if r[0] and r[1] and float(r[2] or 0.0) > 0
        }

    async def save_quant_audit(
        self,
        audit_date: str,
        summary_text: str,
        payload: dict[str, Any] | None = None,
        *,
        created_at_uk: str | None = None,
        ai_diagnosis: str = "",
    ) -> None:
        """Persist weekly offline universe and strategy threshold health audit idempotently by audit_date."""
        uk_ts = created_at_uk or datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        payload_str = json.dumps(payload or {}, ensure_ascii=False)
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO quant_health_audits (
                    audit_date, created_at_uk, summary_text, payload_json, ai_diagnosis
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (str(audit_date), uk_ts, str(summary_text or ""), payload_str, str(ai_diagnosis or "")),
            )
            await db.commit()

    async def get_latest_quant_audit(self) -> dict[str, Any] | None:
        """Return the most recent offline universe & threshold health audit row, if any."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT audit_date, created_at_uk, summary_text, payload_json, ai_diagnosis
                FROM quant_health_audits
                ORDER BY audit_date DESC, created_at_uk DESC
                LIMIT 1
                """
            )
            row = await cur.fetchone()
        if not row:
            return None
        raw_payload = str(row[3] or "{}")
        try:
            parsed = json.loads(raw_payload)
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            parsed = {}
        return {
            "audit_date": str(row[0]),
            "created_at_uk": str(row[1]),
            "summary_text": str(row[2]),
            "payload_json": raw_payload,
            "payload": parsed,
            "ai_diagnosis": str(row[4] or ""),
        }

    async def get_user_queries_since(self, start_ms: int, limit: int = 60) -> list[str]:
        """Return raw user utterances since an epoch-ms cutoff (oldest first) for verbatim quote anchoring."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT user_query FROM chat_history
                WHERE created_at_ms >= ?
                ORDER BY created_at_ms ASC, id ASC
                LIMIT ?
                """,
                (int(start_ms), int(limit)),
            )
            rows = await cur.fetchall()
        return [str(r[0]) for r in rows if r and str(r[0]).strip()]

    async def upsert_preference_notes(self, notes: list[dict[str, Any]]) -> int:
        """Insert validated preference notes, skipping duplicates of the same rule within the last 14 days."""
        if not notes:
            return 0
        now_ms = int(time.time() * 1000)
        dedupe_window_ms = 14 * 86_400_000
        saved = 0
        async with self._connect(write=True) as db:
            for note in notes:
                if not isinstance(note, dict):
                    continue
                quote = str(note.get("quote_text") or "").strip()
                rule = str(note.get("rule_summary") or "").strip()
                expires = str(note.get("expires_on_uk") or "").strip()
                if not quote or not rule or not expires:
                    continue
                cur = await db.execute(
                    """
                    SELECT 1 FROM user_preference_notes
                    WHERE rule_summary = ? AND created_at_ms >= ?
                    LIMIT 1
                    """,
                    (rule, now_ms - dedupe_window_ms),
                )
                if await cur.fetchone():
                    continue
                await db.execute(
                    """
                    INSERT INTO user_preference_notes (
                        source_cycle_date, quote_text, rule_summary, expires_on_uk, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(note.get("source_cycle_date") or ""), quote, rule, expires, now_ms),
                )
                saved += 1
            if saved:
                await db.commit()
        return saved

    async def get_active_preference_notes(
        self,
        now_uk: str | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Return non-expired preference notes (oldest first) for prompt injection."""
        today = (
            now_uk or datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d")
        )[:10]
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT source_cycle_date, quote_text, rule_summary, expires_on_uk
                FROM user_preference_notes
                WHERE expires_on_uk >= ?
                ORDER BY created_at_ms DESC, id DESC
                LIMIT ?
                """,
                (today, int(limit)),
            )
            rows = await cur.fetchall()
        notes = [
            {
                "source_cycle_date": str(r[0]),
                "quote_text": str(r[1]),
                "rule_summary": str(r[2]),
                "expires_on_uk": str(r[3]),
            }
            for r in rows
        ]
        return list(reversed(notes))

    async def prune_expired_preference_notes(self, now_uk: str | None = None) -> int:
        """Delete preference notes whose TTL has elapsed."""
        today = (
            now_uk or datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d")
        )[:10]
        async with self._connect(write=True) as db:
            cur = await db.execute(
                "DELETE FROM user_preference_notes WHERE expires_on_uk < ?",
                (today,),
            )
            await db.commit()
        return max(0, int(cur.rowcount or 0))

    async def get_grounding_news_cache(
        self,
        cache_keys: list[str],
        *,
        max_age_seconds: int = 10800,
    ) -> dict[str, str]:
        """Load non-expired persisted Google Search summaries keyed by cache_key."""
        clean_keys = list(dict.fromkeys(str(k or "").strip() for k in cache_keys if str(k or "").strip()))
        if not clean_keys or max_age_seconds <= 0:
            return {}
        cutoff_ms = int(time.time() * 1000) - int(max_age_seconds) * 1000
        placeholders = ",".join("?" for _ in clean_keys)
        async with self._connect() as db:
            cur = await db.execute(
                f"""
                SELECT cache_key, summary_text
                FROM grounding_news_cache
                WHERE cache_key IN ({placeholders})
                  AND fetched_at_ms >= ?
                  AND TRIM(summary_text) != ''
                """,
                (*clean_keys, cutoff_ms),
            )
            rows = await cur.fetchall()
        return {str(r[0]): str(r[1]) for r in rows if r and r[0] and r[1]}

    async def upsert_grounding_news_cache(self, entries: list[dict[str, Any]]) -> int:
        """Persist Google Search grounding summaries idempotently by cache_key."""
        if not entries:
            return 0
        now_ms = int(time.time() * 1000)
        rows: list[tuple[str, str, str, str, int]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            ckey = str(item.get("cache_key") or "").strip()
            summary = str(item.get("summary_text") or "").strip()
            if not ckey or not summary or summary.startswith("⚠️"):
                continue
            rows.append(
                (
                    ckey,
                    str(item.get("scope") or "").strip(),
                    str(item.get("symbol") or "").strip().upper(),
                    summary,
                    int(item.get("fetched_at_ms") or now_ms),
                )
            )
        if not rows:
            return 0
        async with self._connect(write=True) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO grounding_news_cache (
                    cache_key, scope, symbol, summary_text, fetched_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            await db.commit()
        return len(rows)

    async def get_wallet_balances(self, account_label: str = "Invest") -> dict[str, dict[str, Any]]:
        """Return local per-currency wallet anchors for one account keyed by upper-case currency."""
        label = str(account_label or "Invest").strip()
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT currency, available_amount, reserved_amount, updated_at_uk, updated_at_ms, source, note
                FROM wallet_balances
                WHERE account_label = ?
                """,
                (label,),
            )
            rows = await cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not row or not row[0]:
                continue
            curr = str(row[0]).strip().upper()
            if not curr:
                continue
            out[curr] = {
                "currency": curr,
                "available_amount": float(row[1] or 0.0),
                "reserved_amount": float(row[2] or 0.0),
                "updated_at_uk": str(row[3] or ""),
                "updated_at_ms": int(row[4] or 0),
                "source": str(row[5] or ""),
                "note": str(row[6] or ""),
            }
        return out

    async def upsert_wallet_balance(
        self,
        account_label: str,
        currency: str,
        available_amount: float,
        reserved_amount: float = 0.0,
        *,
        source: str = "manual",
        note: str = "",
    ) -> None:
        """Persist one wallet anchor; negative or non-finite amounts are clamped to 0.0."""
        label = str(account_label or "").strip()
        curr = str(currency or "").strip().upper()
        if not label or not curr:
            raise ValueError("account_label and currency are required for wallet balances")
        avail = float(available_amount or 0.0)
        reserved = float(reserved_amount or 0.0)
        avail = avail if math.isfinite(avail) and avail > 0.0 else 0.0
        reserved = reserved if math.isfinite(reserved) and reserved > 0.0 else 0.0
        now_ms = int(time.time() * 1000)
        now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO wallet_balances (
                    account_label, currency, available_amount, reserved_amount,
                    updated_at_uk, updated_at_ms, source, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_label, currency) DO UPDATE SET
                    available_amount = excluded.available_amount,
                    reserved_amount = excluded.reserved_amount,
                    updated_at_uk = excluded.updated_at_uk,
                    updated_at_ms = excluded.updated_at_ms,
                    source = excluded.source,
                    note = excluded.note
                """,
                (label, curr, avail, reserved, now_uk, now_ms, str(source or "manual"), str(note or "")),
            )
            await db.commit()

    async def clear_manual_gbp_override(self, account_label: str = "Invest") -> int:
        """Drop a user-reported GBP anchor so GBP reverts to TOTAL - USD auto-derivation."""
        label = str(account_label or "Invest").strip()
        async with self._connect(write=True) as db:
            cur = await db.execute(
                """
                DELETE FROM wallet_balances
                WHERE account_label = ? AND currency = 'GBP' AND source LIKE 'manual%'
                """,
                (label,),
            )
            deleted = int(cur.rowcount or 0)
            await db.commit()
        return deleted
