"""Chat history, LLM usage/reliability & ops quality rollups, ops events, scheduler locks, and pending orders."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

from src.storage.db.core import LONDON_TZ


class OpsAndChatRepoMixin:
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

    async def get_llm_reliability_summary(
        self,
        days: int = 7,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """0-schema-change LLM reliability rollup: success rate + P50/P95 latency by model/label.

        Percentiles are computed in Python over the sorted raw latency list so no SQLite
        extension (percentile/median) is required on the deployed build.
        """
        ref = now_utc or datetime.now(timezone.utc)
        cutoff_ms = int(ref.timestamp() * 1000) - max(days, 1) * 86400 * 1000
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT model, call_label, latency_ms, success FROM llm_usage_logs "
                "WHERE created_at_ms >= ?",
                (cutoff_ms,),
            )
            rows = await cur.fetchall()

        def _percentile(values: list[int], pct: float) -> int | None:
            if not values:
                return None
            ordered = sorted(values)
            idx = int(round((len(ordered) - 1) * pct))
            return int(ordered[max(0, min(idx, len(ordered) - 1))])

        def _bucket() -> dict[str, Any]:
            return {"calls": 0, "failed_calls": 0, "latencies": []}

        total = _bucket()
        by_model: dict[str, dict[str, Any]] = {}
        by_label: dict[str, dict[str, Any]] = {}
        for row in rows:
            model = str(row[0] or "unknown")
            label = str(row[1] or "unknown")
            latency = int(row[2] or 0)
            ok = bool(row[3])
            for bucket in (
                total,
                by_model.setdefault(model, _bucket()),
                by_label.setdefault(label, _bucket()),
            ):
                bucket["calls"] += 1
                if not ok:
                    bucket["failed_calls"] += 1
                if latency > 0:
                    bucket["latencies"].append(latency)

        def _finalise(bucket: dict[str, Any]) -> dict[str, Any]:
            calls = int(bucket["calls"])
            failed = int(bucket["failed_calls"])
            latencies = bucket["latencies"]
            return {
                "calls": calls,
                "failed_calls": failed,
                "success_rate_pct": round(((calls - failed) / calls) * 100.0, 1) if calls > 0 else None,
                "p50_latency_ms": _percentile(latencies, 0.5),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            }

        models = {
            name: _finalise(bucket)
            for name, bucket in sorted(by_model.items(), key=lambda kv: kv[1]["calls"], reverse=True)
        }
        labels = {
            name: _finalise(bucket)
            for name, bucket in sorted(by_label.items(), key=lambda kv: kv[1]["calls"], reverse=True)
        }
        slowest = sorted(
            (name for name, item in labels.items() if item["p95_latency_ms"] is not None),
            key=lambda name: int(labels[name]["p95_latency_ms"] or 0),
            reverse=True,
        )[:3]
        most_failed = sorted(
            (name for name, item in labels.items() if item["failed_calls"] > 0),
            key=lambda name: labels[name]["failed_calls"],
            reverse=True,
        )[:3]
        return {
            "days": max(days, 1),
            "total": _finalise(total),
            "by_model": models,
            "by_label": labels,
            "slowest_labels": slowest,
            "most_failed_labels": most_failed,
        }

    async def get_ops_quality_summary(
        self,
        days: int = 7,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Semantic-audit + degradation alert rollup from ops_events (read-only, no schema change)."""
        ref = now_utc or datetime.now(timezone.utc)
        cutoff_ms = int(ref.timestamp() * 1000) - max(days, 1) * 86400 * 1000
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT source, alert_key, message FROM ops_events WHERE created_at_ms >= ?",
                (cutoff_ms,),
            )
            rows = await cur.fetchall()

        semantic = {"total": 0, "report": 0, "pick": 0}
        tag_counts = {
            "事实价不符": 0,
            "建议价方向异常": 0,
            "内部驳回": 0,
        }
        degraded = 0
        for row in rows:
            source = str(row[0] or "")
            alert_key = str(row[1] or "")
            message = str(row[2] or "")
            lowered_key = alert_key.lower()
            if "degraded" in lowered_key or "fallback" in lowered_key:
                degraded += 1
            if source.startswith("semantic_audit:"):
                semantic["total"] += 1
                scope = source.split(":", 1)[1].strip().lower()
                if scope == "report":
                    semantic["report"] += 1
                elif scope == "pick":
                    semantic["pick"] += 1
                for tag in tag_counts:
                    if tag in message or tag in alert_key:
                        tag_counts[tag] += 1
        return {
            "days": max(days, 1),
            "semantic_audit": semantic,
            "violation_tags": {k: v for k, v in tag_counts.items() if v > 0},
            "degraded_alerts": degraded,
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
            # Post-close ledgers follow the same 90-day window but always keep the latest row.
            cur_postmarket = await db.execute(
                """
                DELETE FROM daily_postmarket_reviews
                WHERE created_at_ms < ?
                  AND uk_date != COALESCE(
                      (SELECT uk_date FROM daily_postmarket_reviews
                       ORDER BY created_at_ms DESC, uk_date DESC LIMIT 1),
                      ''
                  )
                """,
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
                "daily_postmarket_reviews": int(cur_postmarket.rowcount or 0),
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
