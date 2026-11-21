"""Weekly quant audits, user preference memory notes, grounding news cache, and Invest dual-wallet balances."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from src.storage.db.core import LONDON_TZ


class MemoryAndWalletRepoMixin:
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

    @staticmethod
    def _postmarket_row_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        raw_payload = str(row[9] or "{}")
        try:
            parsed = json.loads(raw_payload)
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            parsed = {}
        return {
            "uk_date": str(row[0] or ""),
            "created_at_uk": str(row[1] or ""),
            "created_at_ms": int(row[2] or 0),
            "regime": str(row[3] or ""),
            "local_pnl_gbp": float(row[4] or 0.0),
            "fx_pnl_gbp": float(row[5] or 0.0),
            "total_mark_pnl_gbp": float(row[6] or 0.0),
            "discipline_alert_count": int(row[7] or 0),
            "summary_text": str(row[8] or ""),
            "payload_json": raw_payload,
            "payload": parsed,
        }

    async def upsert_daily_postmarket_review(self, row: dict[str, Any]) -> None:
        """Idempotently persist one UK-date post-close execution ledger (0-Token, GBP scalars)."""
        uk_date = str(row.get("uk_date") or "").strip()[:10]
        if not uk_date:
            return
        now_ms = int(row.get("created_at_ms") or 0) or int(time.time() * 1000)
        created_at_uk = str(row.get("created_at_uk") or "").strip() or datetime.now(
            timezone.utc
        ).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, str) and payload_raw.strip():
            payload_json = payload_raw
        else:
            payload_json = json.dumps(row.get("payload") or {}, ensure_ascii=False, default=str)
        summary_text = str(row.get("summary_text") or "").strip()
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO daily_postmarket_reviews (
                    uk_date, created_at_uk, created_at_ms, regime,
                    local_pnl_gbp, fx_pnl_gbp, total_mark_pnl_gbp,
                    discipline_alert_count, summary_text, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uk_date,
                    created_at_uk,
                    now_ms,
                    str(row.get("regime") or "").strip().upper()[:16],
                    float(row.get("local_pnl_gbp") or 0.0),
                    float(row.get("fx_pnl_gbp") or 0.0),
                    float(row.get("total_mark_pnl_gbp") or 0.0),
                    int(row.get("discipline_alert_count") or 0),
                    summary_text,
                    payload_json,
                ),
            )
            await db.commit()

    async def get_latest_postmarket_review(self) -> dict[str, Any] | None:
        """Return the most recent post-close execution ledger, if any."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT uk_date, created_at_uk, created_at_ms, regime,
                       local_pnl_gbp, fx_pnl_gbp, total_mark_pnl_gbp,
                       discipline_alert_count, summary_text, payload_json
                FROM daily_postmarket_reviews
                ORDER BY uk_date DESC, created_at_ms DESC
                LIMIT 1
                """
            )
            row = await cur.fetchone()
        return self._postmarket_row_dict(row) if row else None

    async def get_recent_postmarket_reviews(self, limit: int = 7) -> list[dict[str, Any]]:
        """Return the most recent N post-close execution ledgers, newest first."""
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT uk_date, created_at_uk, created_at_ms, regime,
                       local_pnl_gbp, fx_pnl_gbp, total_mark_pnl_gbp,
                       discipline_alert_count, summary_text, payload_json
                FROM daily_postmarket_reviews
                ORDER BY uk_date DESC, created_at_ms DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = await cur.fetchall()
        return [self._postmarket_row_dict(r) for r in rows]

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
