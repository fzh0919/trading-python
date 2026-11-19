"""Recommendation ledger, T+5/10/20 settlement & follow-up, reflection lessons, and attribution stats."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from src.storage.db.core import LONDON_TZ

_RECOMMENDATION_SELECT_COLS = (
    "id, created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis, "
    "horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, "
    "outcome_status, reflection_lesson, "
    "t10_return_pct, t10_alpha_pct, t20_return_pct, t20_alpha_pct, mfe_pct, mae_pct, "
    "source, fill_status, actual_filled, actual_fill_price, execution_slippage_pct"
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
    }


class RecommendationsRepoMixin:
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
        actual_filled: int | None = None,
        actual_fill_price: float | None = None,
        execution_slippage_pct: float | None = None,
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
                    fill_status = CASE WHEN ? = '' THEN fill_status ELSE ? END,
                    actual_filled = COALESCE(?, actual_filled),
                    actual_fill_price = COALESCE(?, actual_fill_price),
                    execution_slippage_pct = COALESCE(?, execution_slippage_pct)
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
                "by_account": by_account,
                "adoption_stats": adoption_stats,
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
            "by_account": by_account,
            "adoption_stats": adoption_stats,
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
