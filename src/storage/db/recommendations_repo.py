"""Recommendation ledger, T+5/10/20 settlement & follow-up, reflection lessons, and attribution stats."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.storage.db.core import LONDON_TZ

logger = logging.getLogger(__name__)

_RECOMMENDATION_SELECT_COLS = (
    "id, created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis, "
    "horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, "
    "outcome_status, reflection_lesson, "
    "t10_return_pct, t10_alpha_pct, t20_return_pct, t20_alpha_pct, mfe_pct, mae_pct, "
    "source, fill_status, actual_filled, actual_fill_price, execution_slippage_pct, "
    "regime, shadow_trigger, shadow_reason, process_quadrant, holding_catalyst"
)


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
        "actual_filled": (None if row[23] is None else int(row[23])),
        "actual_fill_price": row[24],
        "execution_slippage_pct": row[25],
        "regime": str(row[26] or ""),
        "shadow_trigger": str(row[27] or ""),
        "shadow_reason": str(row[28] or ""),
        "process_quadrant": str(row[29] or ""),
        "holding_catalyst": str(row[30] or ""),
    }


class RecommendationsRepoMixin:
    async def log_recommendations(
        self,
        items: list[dict[str, Any]],
        *,
        default_horizon_days: int = 5,
        source: str = "pick",
        dedupe_same_day: bool = False,
        dedupe_window_days: int = 0,
        dedupe_cross_source: bool = False,
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
                str(item.get("regime") or "").strip().upper()[:16],
            )
            for item in items
        ]
        window_days = max(0, int(dedupe_window_days or 0))
        async with self._connect(write=True) as db:
            if window_days > 0:
                window_ms = window_days * 86_400_000
                min_ts = min(int(r[1]) for r in rows_to_insert)
                cutoff_ms = min_ts - window_ms
                if dedupe_cross_source:
                    cur = await db.execute(
                        """
                        SELECT account, symbol, action, created_at_ms
                        FROM pick_recommendations
                        WHERE COALESCE(source, 'pick') != 'core_dca'
                          AND COALESCE(outcome_status, 'PENDING') != 'SKIPPED'
                          AND COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                          AND created_at_ms >= ?
                        """,
                        (cutoff_ms,),
                    )
                else:
                    cur = await db.execute(
                        """
                        SELECT account, symbol, action, created_at_ms
                        FROM pick_recommendations
                        WHERE COALESCE(source, 'pick') = ?
                          AND COALESCE(outcome_status, 'PENDING') != 'SKIPPED'
                          AND COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                          AND created_at_ms >= ?
                        """,
                        (default_source, cutoff_ms),
                    )
                existing_ts: dict[tuple[str, str, str], list[int]] = {}
                for r in await cur.fetchall():
                    key = (str(r[0]).strip().upper(), str(r[1]).strip().upper(), str(r[2]).strip().upper())
                    existing_ts.setdefault(key, []).append(int(r[3] or 0))
                filtered_rows = []
                for row in rows_to_insert:
                    key = (str(row[2]).strip().upper(), str(row[3]).strip().upper(), str(row[4]).strip().upper())
                    ts = int(row[1])
                    if any(abs(ts - prev_ts) < window_ms for prev_ts in existing_ts.get(key, ())):
                        continue
                    filtered_rows.append(row)
                    existing_ts.setdefault(key, []).append(ts)
                rows_to_insert = filtered_rows
                if not rows_to_insert:
                    return
            elif dedupe_same_day:
                dates = sorted({str(r[0])[:10] for r in rows_to_insert if str(r[0])[:10]})
                placeholders = ",".join("?" for _ in dates) or "?"
                date_params = dates or [now_uk[:10]]
                if dedupe_cross_source:
                    cur = await db.execute(
                        f"""
                        SELECT substr(created_at_uk, 1, 10), account, symbol, action
                        FROM pick_recommendations
                        WHERE COALESCE(source, 'pick') != 'core_dca'
                          AND COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                          AND substr(created_at_uk, 1, 10) IN ({placeholders})
                        """,
                        tuple(date_params),
                    )
                else:
                    cur = await db.execute(
                        f"""
                        SELECT substr(created_at_uk, 1, 10), account, symbol, action
                        FROM pick_recommendations
                        WHERE COALESCE(source, 'pick') = ?
                          AND COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                          AND substr(created_at_uk, 1, 10) IN ({placeholders})
                        """,
                        (default_source, *date_params),
                    )
                existing = {
                    (str(r[0]), str(r[1]).strip().upper(), str(r[2]).strip().upper(), str(r[3]).strip().upper())
                    for r in await cur.fetchall()
                }
                filtered_rows = []
                for row in rows_to_insert:
                    key = (
                        str(row[0])[:10],
                        str(row[2]).strip().upper(),
                        str(row[3]).strip().upper(),
                        str(row[4]).strip().upper(),
                    )
                    if key in existing:
                        continue
                    filtered_rows.append(row)
                    existing.add(key)
                rows_to_insert = filtered_rows
                if not rows_to_insert:
                    return
            await db.executemany(
                """
                INSERT INTO pick_recommendations (
                    created_at_uk, created_at_ms, account, symbol, action, entry_price,
                    thesis, horizon_days, benchmark_symbol, source, fill_status, regime, outcome_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                rows_to_insert,
            )
            await db.commit()

    async def dedupe_existing_pending_recommendations(self, window_days: int = 5) -> int:
        """Idempotently remove duplicate unsettled tactical copies within `window_days`, keeping the earliest row.

        Settled/NOT_FILLED/SKIPPED history and `core_dca` rows are never touched.
        """
        window_ms = max(1, int(window_days)) * 86_400_000
        async with self._connect(write=True) as db:
            cur = await db.execute(
                """
                SELECT id, created_at_ms, account, symbol, action
                FROM pick_recommendations
                WHERE COALESCE(source, 'pick') != 'core_dca'
                  AND (outcome_status IS NULL OR outcome_status = 'PENDING')
                ORDER BY created_at_ms ASC, id ASC
                """
            )
            rows = await cur.fetchall()
            last_kept_ms: dict[tuple[str, str, str], int] = {}
            dup_ids: list[int] = []
            for row_id, created_at_ms, account, symbol, action in rows:
                key = (
                    str(account or "").strip().upper(),
                    str(symbol or "").strip().upper(),
                    str(action or "").strip().upper(),
                )
                ts = int(created_at_ms or 0)
                if key in last_kept_ms and (ts - last_kept_ms[key]) < window_ms:
                    dup_ids.append(int(row_id))
                else:
                    last_kept_ms[key] = ts
            if dup_ids:
                await db.executemany(
                    "DELETE FROM pick_recommendations WHERE id = ?",
                    [(rid,) for rid in dup_ids],
                )
                await db.commit()
                logger.info(
                    "Deduplicated %d duplicate pending tactical recommendation(s) (%dd window)",
                    len(dup_ids),
                    window_days,
                )
            return len(dup_ids)

    async def get_recent_recommendations(
        self,
        limit: int = 10,
        *,
        exclude_core_dca: bool = False,
        only_core_dca: bool = False,
    ) -> list[dict[str, Any]]:
        if only_core_dca:
            where_clause = "WHERE COALESCE(source, 'pick') = 'core_dca' "
        elif exclude_core_dca:
            where_clause = (
                "WHERE COALESCE(source, 'pick') != 'core_dca' "
                "AND COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED') "
            )
        else:
            where_clause = (
                "WHERE COALESCE(outcome_status, 'PENDING') NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED') "
            )
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                f"{where_clause}"
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
        actual_filled: int | None = None,
        actual_fill_price: float | None = None,
        execution_slippage_pct: float | None = None,
        process_quadrant: str | None = None,
        holding_catalyst: str | None = None,
    ) -> bool:
        """Idempotently settle a PENDING recommendation; returns False if already settled."""
        status = str(outcome_status or "SETTLED").strip().upper()
        bench = str(benchmark_symbol or "").strip().upper()
        lesson = str(reflection_lesson or "").strip()[:600]
        fill = str(fill_status or "").strip().upper()
        quadrant = None if process_quadrant is None else str(process_quadrant or "").strip().upper()[:24]
        catalyst = None if holding_catalyst is None else str(holding_catalyst or "").strip()[:200]
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
                    fill_status = CASE WHEN ? = '' THEN fill_status ELSE ? END,
                    actual_filled = COALESCE(?, actual_filled),
                    actual_fill_price = COALESCE(?, actual_fill_price),
                    execution_slippage_pct = COALESCE(?, execution_slippage_pct),
                    process_quadrant = COALESCE(?, process_quadrant),
                    holding_catalyst = COALESCE(?, holding_catalyst)
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
                    (None if actual_filled is None else int(actual_filled)),
                    actual_fill_price,
                    execution_slippage_pct,
                    quadrant,
                    catalyst,
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
        reflection_lesson: str | None = None,
        process_quadrant: str | None = None,
        holding_catalyst: str | None = None,
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
        lesson_txt = str(reflection_lesson or "").strip()[:600]
        if lesson_txt:
            updates.append("reflection_lesson = ?")
            params.append(lesson_txt)
        if process_quadrant is not None:
            updates.append("process_quadrant = ?")
            params.append(str(process_quadrant or "").strip().upper()[:24])
        if holding_catalyst is not None:
            updates.append("holding_catalyst = ?")
            params.append(str(holding_catalyst or "").strip()[:200])
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
                "AND COALESCE(source, 'pick') != 'core_dca' "
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
                "AND COALESCE(source, 'pick') != 'core_dca' "
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
                    COALESCE(SUM(CASE WHEN COALESCE(outcome_status, 'PENDING')
                                   NOT IN ('SHADOW_REJECTED', 'SHADOW_SETTLED') THEN 1 ELSE 0 END), 0),
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
                    AVG(CASE WHEN outcome_status = 'SETTLED' THEN mae_pct END),
                    COALESCE(SUM(CASE WHEN COALESCE(outcome_status, 'PENDING')
                                   IN ('SHADOW_REJECTED', 'SHADOW_SETTLED') THEN 1 ELSE 0 END), 0)
                FROM pick_recommendations
                WHERE COALESCE(source, 'pick') != 'core_dca'
                """
            )
            row = await cur.fetchone()
            core_cur = await db.execute(
                "SELECT COUNT(*) FROM pick_recommendations "
                "WHERE COALESCE(source, 'pick') = 'core_dca'"
            )
            core_row = await core_cur.fetchone()
            core_dca_count = int(core_row[0] or 0) if core_row else 0
            split_cur = await db.execute(
                """
                SELECT account,
                       COALESCE(SUM(CASE WHEN outcome_status = 'SETTLED' THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN outcome_status = 'SETTLED' AND raw_return_pct > 0 THEN 1 ELSE 0 END), 0),
                       AVG(CASE WHEN outcome_status = 'SETTLED' THEN alpha_pct END),
                       AVG(CASE WHEN outcome_status = 'SETTLED' THEN t10_alpha_pct END),
                       AVG(CASE WHEN outcome_status = 'SETTLED' THEN t20_alpha_pct END)
                FROM pick_recommendations
                WHERE COALESCE(source, 'pick') != 'core_dca'
                GROUP BY account
                """
            )
            split_rows = await split_cur.fetchall()
            adopt_cur = await db.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN actual_filled IS NOT NULL THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN actual_filled = 1 THEN 1 ELSE 0 END), 0),
                    AVG(CASE WHEN actual_filled = 1 THEN execution_slippage_pct END)
                FROM pick_recommendations
                WHERE COALESCE(source, 'pick') != 'core_dca'
                """
            )
            adopt_row = await adopt_cur.fetchone()
            quadrant_cur = await db.execute(
                """
                SELECT UPPER(COALESCE(NULLIF(TRIM(process_quadrant), ''), '')), COUNT(*)
                FROM pick_recommendations
                WHERE outcome_status = 'SETTLED'
                  AND COALESCE(source, 'pick') != 'core_dca'
                GROUP BY 1
                """
            )
            quadrant_rows = await quadrant_cur.fetchall()

        quadrant_counts: dict[str, int] = {
            "GOOD_WIN": 0,
            "GOOD_LOSS": 0,
            "BAD_WIN": 0,
            "BAD_LOSS": 0,
        }
        for q_row in quadrant_rows or []:
            label = str(q_row[0] or "").strip().upper()
            if label in quadrant_counts:
                quadrant_counts[label] = int(q_row[1] or 0)

        by_account: dict[str, dict[str, Any]] = {
            "ISA": {
                "settled_count": 0,
                "win_rate_pct": 0.0,
                "avg_alpha_pct": None,
                "avg_t10_alpha_pct": None,
                "avg_t20_alpha_pct": None,
            },
            "Invest": {
                "settled_count": 0,
                "win_rate_pct": 0.0,
                "avg_alpha_pct": None,
                "avg_t10_alpha_pct": None,
                "avg_t20_alpha_pct": None,
            },
        }
        for split_row in split_rows or []:
            label = str(split_row[0] or "").strip()
            key = "ISA" if label.upper() == "ISA" else ("Invest" if label else "")
            if key not in by_account:
                continue
            settled_n = int(split_row[1] or 0)
            win_n = int(split_row[2] or 0)
            by_account[key] = {
                "settled_count": settled_n,
                "win_rate_pct": round((win_n / settled_n) * 100.0, 1) if settled_n > 0 else 0.0,
                "avg_alpha_pct": round(float(split_row[3]), 2) if split_row[3] is not None else None,
                "avg_t10_alpha_pct": round(float(split_row[4]), 2) if split_row[4] is not None else None,
                "avg_t20_alpha_pct": round(float(split_row[5]), 2) if split_row[5] is not None else None,
            }
        checked_count = int(adopt_row[0] or 0) if adopt_row else 0
        filled_count = int(adopt_row[1] or 0) if adopt_row else 0
        adoption_stats = {
            "checked_count": checked_count,
            "filled_count": filled_count,
            "adoption_rate_pct": round((filled_count / checked_count) * 100.0, 1) if checked_count > 0 else 0.0,
            "avg_slippage_pct": (
                round(float(adopt_row[2]), 3) if adopt_row and adopt_row[2] is not None else None
            ),
        }
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
                "core_dca_count": core_dca_count,
                "shadow_count": 0,
                "by_account": by_account,
                "adoption_stats": adoption_stats,
                "quadrant_counts": quadrant_counts,
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
            "core_dca_count": core_dca_count,
            # v5.5.9 M6: shadow counterfactual rows live in their own lane — never
            # inside total_count / win-rate / adoption (isolation contract).
            "shadow_count": int(row[13] or 0),
            "by_account": by_account,
            "adoption_stats": adoption_stats,
            "quadrant_counts": quadrant_counts,
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
                  AND COALESCE(source, 'pick') != 'core_dca'
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

    # ------------------------------------------------------------------
    # v5.5.9 M6: shadow counterfactual lane (SHADOW_REJECTED -> SHADOW_SETTLED).
    # Isolation contract: these rows carry source='shadow' + a dedicated status
    # family, are excluded from every live dedupe / stats / display query above,
    # and never enter the real win-rate ledger.
    # ------------------------------------------------------------------

    _SHADOW_MISSED_WINNER_PCT: float = 10.0
    _SHADOW_STATUS_FAMILY: tuple[str, ...] = ("SHADOW_REJECTED", "SHADOW_SETTLED")

    async def log_shadow_rejections(
        self,
        items: list[dict[str, Any]],
        *,
        regime: str = "",
        default_horizon_days: int = 20,
    ) -> int:
        """Persist /pick hard-rejected Top candidates as shadow counterfactual rows.

        Same-day same-(account, symbol) duplicates are skipped so repeated /pick
        runs cannot inflate the counterfactual sample. Real recommendation rows
        are never touched by this lane.
        """
        if not items:
            return 0
        now_ms = int(time.time() * 1000)
        now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        fallback_regime = str(regime or "").strip().upper()[:16]
        rows_to_insert = [
            (
                str(item.get("created_at_uk") or now_uk),
                int(item.get("created_at_ms") or now_ms),
                str(item.get("account") or "Invest"),
                str(item.get("symbol") or "").strip().upper(),
                str(item.get("action") or "BUY").strip().upper(),
                item.get("entry_price"),
                str(item.get("thesis") or "")[:300],
                max(1, int(item["horizon_days"])) if item.get("horizon_days") is not None else default_horizon_days,
                str(item.get("benchmark_symbol") or "").strip().upper(),
                # Per-row regime wins (each /pick run knows its own gate label);
                # the function arg is only a lane-wide fallback.
                str(item.get("regime") or fallback_regime or "").strip().upper()[:16],
                str(item.get("shadow_trigger") or "").strip().upper()[:24],
                str(item.get("shadow_reason") or "").strip()[:200],
            )
            for item in items
        ]
        async with self._connect(write=True) as db:
            cur = await db.execute(
                """
                SELECT substr(created_at_uk, 1, 10), account, symbol
                FROM pick_recommendations
                WHERE COALESCE(outcome_status, 'PENDING') IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                """
            )
            existing = {
                (str(r[0]), str(r[1]).strip().upper(), str(r[2]).strip().upper())
                for r in await cur.fetchall()
            }
            filtered_rows = []
            for row in rows_to_insert:
                key = (str(row[0])[:10], str(row[2]).strip().upper(), str(row[3]).strip().upper())
                if key in existing:
                    continue
                filtered_rows.append(row)
                existing.add(key)
            if not filtered_rows:
                return 0
            await db.executemany(
                """
                INSERT INTO pick_recommendations (
                    created_at_uk, created_at_ms, account, symbol, action, entry_price,
                    thesis, horizon_days, benchmark_symbol, source, fill_status,
                    regime, shadow_trigger, shadow_reason, outcome_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'shadow', '', ?, ?, ?, 'SHADOW_REJECTED')
                """,
                filtered_rows,
            )
            await db.commit()
            return len(filtered_rows)

    async def get_pending_shadow_settlements(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return unsettled SHADOW_REJECTED rows (T+20 counterfactual settlement queue)."""
        async with self._connect() as db:
            cur = await db.execute(
                f"SELECT {_RECOMMENDATION_SELECT_COLS} FROM pick_recommendations "
                "WHERE outcome_status = 'SHADOW_REJECTED' "
                "AND entry_price IS NOT NULL AND entry_price > 0 "
                "AND t20_return_pct IS NULL "
                "ORDER BY created_at_ms ASC, id ASC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_recommendation(r) for r in rows]

    async def settle_shadow_rejection(
        self,
        rec_id: int,
        *,
        benchmark_symbol: str,
        settled_at_uk: str,
        raw_return_pct: float | None,
        t20_return_pct: float | None,
        t20_alpha_pct: float | None,
        mfe_pct: float | None = None,
        mae_pct: float | None = None,
    ) -> bool:
        """Idempotently settle one shadow row to SHADOW_SETTLED; False if not pending-shadow."""
        bench = str(benchmark_symbol or "").strip().upper()
        async with self._connect(write=True) as db:
            cur = await db.execute(
                """
                UPDATE pick_recommendations
                SET benchmark_symbol = ?,
                    settled_at_uk = ?,
                    raw_return_pct = ?,
                    t20_return_pct = ?,
                    t20_alpha_pct = ?,
                    mfe_pct = COALESCE(?, mfe_pct),
                    mae_pct = COALESCE(?, mae_pct),
                    outcome_status = 'SHADOW_SETTLED'
                WHERE id = ? AND outcome_status = 'SHADOW_REJECTED'
                """,
                (
                    bench,
                    str(settled_at_uk),
                    raw_return_pct,
                    t20_return_pct,
                    t20_alpha_pct,
                    mfe_pct,
                    mae_pct,
                    int(rec_id),
                ),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def get_shadow_rejection_stats(self) -> dict[str, Any]:
        """Counterfactual scoreboard: guardrail avoided-loser vs missed-winner rates.

        A settled shadow row counts as an avoided loser when its T+20 raw return is
        <= 0, and as a missed winner (错杀大牛股) when >= +10% in 20 trading days;
        the (0, +10) band stays neutral. Rates are honest about sample size — the
        display layer must keep the n<30 caveat visible.
        """
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN COALESCE(outcome_status, 'PENDING')
                                   IN ('SHADOW_REJECTED', 'SHADOW_SETTLED') THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_REJECTED' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED'
                                   AND t20_return_pct <= 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED'
                                   AND t20_return_pct >= ? THEN 1 ELSE 0 END), 0),
                    AVG(CASE WHEN outcome_status = 'SHADOW_SETTLED' THEN t20_return_pct END),
                    COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED'
                                   AND t20_alpha_pct > 0 THEN 1 ELSE 0 END), 0)
                FROM pick_recommendations
                """,
                (self._SHADOW_MISSED_WINNER_PCT,),
            )
            row = await cur.fetchone()
            regime_cur = await db.execute(
                """
                SELECT COALESCE(NULLIF(TRIM(regime), ''), 'UNKNOWN'),
                       COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED'
                                      AND t20_return_pct <= 0 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED'
                                      AND t20_return_pct >= ? THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN outcome_status = 'SHADOW_SETTLED' THEN 1 ELSE 0 END), 0)
                FROM pick_recommendations
                WHERE COALESCE(outcome_status, 'PENDING') IN ('SHADOW_REJECTED', 'SHADOW_SETTLED')
                GROUP BY 1
                ORDER BY 4 DESC
                """,
                (self._SHADOW_MISSED_WINNER_PCT,),
            )
            regime_rows = await regime_cur.fetchall()

        total_n = int(row[0] or 0)
        settled_n = int(row[2] or 0)
        avoided_n = int(row[3] or 0)
        missed_n = int(row[4] or 0)
        by_regime: dict[str, dict[str, int]] = {}
        for r in regime_rows or []:
            label = str(r[0] or "UNKNOWN")
            by_regime[label] = {
                "settled": int(r[3] or 0),
                "avoided_loser": int(r[1] or 0),
                "missed_winner": int(r[2] or 0),
            }
        return {
            "total": total_n,
            "pending": int(row[1] or 0),
            "settled": settled_n,
            "avoided_loser": avoided_n,
            "missed_winner": missed_n,
            "neutral": max(settled_n - avoided_n - missed_n, 0),
            "avoided_loser_rate_pct": round((avoided_n / settled_n) * 100.0, 1) if settled_n > 0 else None,
            "missed_winner_rate_pct": round((missed_n / settled_n) * 100.0, 1) if settled_n > 0 else None,
            "avg_t20_return_pct": round(float(row[5]), 2) if row[5] is not None else None,
            "alpha_positive": int(row[6] or 0),
            "missed_winner_threshold_pct": self._SHADOW_MISSED_WINNER_PCT,
            "small_sample": settled_n < 30,
            "by_regime": by_regime,
        }
