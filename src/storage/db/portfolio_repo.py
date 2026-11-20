"""Portfolio snapshots, NAV history, morning reports, context packs, guardrails, and pick radar."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.data_providers.t212_client import PortfolioData

logger = logging.getLogger("src.storage.db")


def _truncate_decision_text(text: str, max_chars: int = 140) -> str:
    """Trim a decision line at a clause boundary so no half-cut bracket/sentence reaches the PM."""
    first = (text.splitlines()[0] if text else "").strip()
    if len(first) <= max_chars:
        return first
    head = first[:max_chars]
    cut = max(head.rfind(sep) for sep in (";", "；", "|"))
    if cut >= max_chars // 3:
        head = head[:cut]
    else:
        for opener, closer in (("(", ")"), ("（", "）"), ("[", "]")):
            if head.count(opener) > head.count(closer):
                idx = head.rfind(opener)
                if idx > 0:
                    head = head[:idx]
    return head.rstrip("。；;，,、|·-— ") + "…"


class PortfolioRepoMixin:
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
            float(row.get("isa_net_flow_gbp") or 0.0),
            float(row.get("invest_net_flow_gbp") or 0.0),
            (None if row.get("benchmark_close") is None else float(row["benchmark_close"])),
            (None if row.get("cash_rate_pct") is None else float(row["cash_rate_pct"])),
            now_ms,
        )
        async with self._connect(write=True) as db:
            await db.execute(
                """
                INSERT INTO nav_history (
                    uk_date, total_value_gbp, cash_gbp, invested_gbp, net_flow_gbp,
                    isa_value_gbp, invest_value_gbp, isa_net_flow_gbp, invest_net_flow_gbp,
                    benchmark_close, cash_rate_pct, created_at_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uk_date) DO UPDATE SET
                    total_value_gbp = excluded.total_value_gbp,
                    cash_gbp = excluded.cash_gbp,
                    invested_gbp = excluded.invested_gbp,
                    net_flow_gbp = excluded.net_flow_gbp,
                    isa_value_gbp = excluded.isa_value_gbp,
                    invest_value_gbp = excluded.invest_value_gbp,
                    isa_net_flow_gbp = excluded.isa_net_flow_gbp,
                    invest_net_flow_gbp = excluded.invest_net_flow_gbp,
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
                       isa_value_gbp, invest_value_gbp, isa_net_flow_gbp, invest_net_flow_gbp,
                       benchmark_close, cash_rate_pct
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
            "isa_net_flow_gbp",
            "invest_net_flow_gbp",
            "benchmark_close",
            "cash_rate_pct",
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(zip(cols, row))
            # Old rows migrated by `_ensure_columns` carry 0 defaults; normalise defensively.
            item["isa_net_flow_gbp"] = float(item.get("isa_net_flow_gbp") or 0.0)
            item["invest_net_flow_gbp"] = float(item.get("invest_net_flow_gbp") or 0.0)
            out.append(item)
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

    async def get_regime_history(self, limit: int = 420) -> list[dict[str, Any]]:
        """v5.5.9 M6: per-day regime labels from morning context packs (read-only).

        The morning report is the only regime source of truth — the evening NAV job
        never re-derives it (that would need extra VIX/HY fetches). Days without a
        report simply carry no label and show up as unattributed in /perf.
        Malformed packs are skipped, never guessed.
        """
        async with self._connect() as db:
            cur = await db.execute(
                """
                SELECT cycle_date, context_pack_json FROM morning_reports
                WHERE context_pack_json IS NOT NULL AND TRIM(context_pack_json) != ''
                ORDER BY cycle_date DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = await cur.fetchall()
        history: list[dict[str, Any]] = []
        for row in rows or []:
            day = str(row[0] or "").strip()[:10]
            if len(day) != 10:
                continue
            try:
                parsed = json.loads(str(row[1] or "").strip())
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            gate_node = parsed.get("deployment_gate")
            gate_regime = gate_node.get("regime") if isinstance(gate_node, dict) else ""
            label = str(parsed.get("regime") or gate_regime or "").strip().upper()
            if not label:
                continue
            history.append({"uk_date": day, "regime": label})
        history.reverse()  # ascending by date for attribution joins
        return history

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

    async def get_morning_context_meta(
        self, cycle_date: str | None = None
    ) -> tuple[str, str, str, bool, str]:
        """v5.5.8: single-query morning context -> (static_report, premarket_note, conclusion, stale, cycle_date).

        ``stale=True`` means the requested cycle has no row and the latest available report
        (previous cycle) was returned so callers can label it as outdated.
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
            return "", "", "", False, cdate
        return str(row[0] or ""), str(row[2] or ""), str(row[1] or ""), stale, cdate

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

    async def get_past_decision_context(
        self, limit: int = 5, *, exclude_cycle_date: str | None = None
    ) -> str:
        """Brief prior decision log for Portfolio Manager continuity (TradingAgents Memory Log pattern).

        ``exclude_cycle_date`` skips the morning report of the current cycle so re-running
        ``/report`` on the same day never anchors the PM on the stale report it is about to replace.
        """
        cycle_filter = str(exclude_cycle_date).strip() if exclude_cycle_date else ""
        lines: list[str] = []
        async with self._connect() as db:
            report_sql = (
                "SELECT cycle_date, conclusion_text FROM morning_reports "
                + ("WHERE cycle_date != ? " if cycle_filter else "")
                + "ORDER BY updated_at_ms DESC LIMIT 2"
            )
            cur = await db.execute(report_sql, (cycle_filter,) if cycle_filter else ())
            for cdate, concl in await cur.fetchall():
                if concl:
                    lines.append(f"• [晨报 {cdate}] {_truncate_decision_text(str(concl))}")
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

    async def get_latest_pick_radar_with_ts(self) -> tuple[str, str, int]:
        """v5.5.8: latest pick radar plus ``created_at_ms`` so callers can gate by trading cycle."""
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT created_at_uk, final_text, created_at_ms FROM pick_radars ORDER BY created_at_ms DESC, id DESC LIMIT 1"
            )
            row = await cur.fetchone()
        if not row:
            return "", "", 0
        return str(row[0]), str(row[1]), int(row[2] or 0)
