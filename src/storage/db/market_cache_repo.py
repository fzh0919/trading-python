"""OHLCV cache, T212 instrument metadata, fundamentals cache, active universe pool, constituents, and dividends."""

from __future__ import annotations

import json
import math
import time
from typing import Any


class MarketCacheRepoMixin:
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
