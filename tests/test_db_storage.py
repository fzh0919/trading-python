"""Offline unit tests for IMP-03 SQLite repository upgrades (migrations, context pack, settlements, pruning)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data_providers.t212_client import AccountSnapshot, PortfolioData
from src.storage.db import AdvisorDatabase


def _make_portfolio(fetched_at_uk: str, fetched_at_ms: int, total_value: float) -> PortfolioData:
    return PortfolioData(
        fetched_at_uk=fetched_at_uk,
        fetched_at_ms=fetched_at_ms,
        isa=AccountSnapshot(
            account_name="ISA",
            ok=True,
            free_cash=total_value * 0.5,
            invested_value=total_value * 0.5,
            total_account_value=total_value * 0.5,
        ),
        invest=AccountSnapshot(
            account_name="Invest",
            ok=True,
            free_cash=total_value * 0.5,
            invested_value=0.0,
            total_account_value=total_value * 0.5,
        ),
        total_combined_value=total_value,
        total_combined_cash=total_value * 0.75,
    )


class TestDbStorageImp03(unittest.IsolatedAsyncioTestCase):
    async def test_idempotent_migration_from_legacy_schema_and_pragmas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            # Seed a legacy pre-IMP-03 SQLite file with existing rows
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE morning_reports (
                    cycle_date TEXT PRIMARY KEY,
                    report_text TEXT NOT NULL,
                    conclusion_text TEXT NOT NULL,
                    premarket_note TEXT DEFAULT '',
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO morning_reports (cycle_date, report_text, conclusion_text, premarket_note, updated_at_ms)
                VALUES ('2026-11-10', 'legacy report', 'legacy conclusion', '', 1700000000000)
                """
            )
            conn.execute(
                """
                CREATE TABLE pick_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_uk TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    account TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    thesis TEXT DEFAULT '',
                    horizon_days INTEGER DEFAULT 30
                )
                """
            )
            conn.execute(
                """
                INSERT INTO pick_recommendations (created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis)
                VALUES ('2026-11-10 10:00', 1700000000000, 'ISA', 'NVDA', 'BUY', 135.5, 'legacy thesis')
                """
            )
            conn.execute(
                """
                CREATE TABLE llm_usage_logs (
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
                    success INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()
            conn.close()

            db = AdvisorDatabase(db_path)
            await db.init()
            # Re-running init() must remain strictly idempotent
            await db.init()

            async with db._connect() as conn_async:
                cur_j = await conn_async.execute("PRAGMA journal_mode;")
                journal_mode = (await cur_j.fetchone())[0]
                cur_b = await conn_async.execute("PRAGMA busy_timeout;")
                busy_timeout = (await cur_b.fetchone())[0]
                self.assertEqual(str(journal_mode).lower(), "wal")
                self.assertGreaterEqual(int(busy_timeout), 5000)

                cur_mr = await conn_async.execute("PRAGMA table_info(morning_reports)")
                mr_cols = {row[1] for row in await cur_mr.fetchall()}
                self.assertIn("daily_guardrails", mr_cols)
                self.assertIn("context_pack_json", mr_cols)

                cur_pr = await conn_async.execute("PRAGMA table_info(pick_recommendations)")
                pr_cols = {row[1] for row in await cur_pr.fetchall()}
                for expected_col in (
                    "benchmark_symbol",
                    "settled_at_uk",
                    "raw_return_pct",
                    "alpha_pct",
                    "outcome_status",
                    "reflection_lesson",
                ):
                    self.assertIn(expected_col, pr_cols)

            report_text, conclusion = await db.get_morning_context(cycle_date="2026-11-10")
            self.assertEqual(report_text, "legacy report")
            self.assertEqual(conclusion, "legacy conclusion")

            pending = await db.get_pending_settlements()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["symbol"], "NVDA")
            self.assertEqual(pending[0]["outcome_status"], "PENDING")

    async def test_morning_context_pack_persistence_and_fallback_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "pack.db")
            await db.init()

            sample_pack = {
                "cycle_date": "2026-11-14",
                "tickers": {"NVDA": {"cyq_benefit_pct": 74.2, "weekly_uptrend": True}},
                "disagreements": ["NVDA: 基本面偏多 vs RSI 接近超买"],
            }
            await db.save_morning_report(
                "report body",
                "short headline",
                guardrails=["冻结半导体"],
                cycle_date="2026-11-14",
                context_pack=sample_pack,
            )

            # Updating report without passing context_pack must preserve existing snapshot
            await db.save_morning_report(
                "updated report body",
                "updated headline",
                guardrails=["冻结半导体"],
                cycle_date="2026-11-14",
            )
            loaded = await db.get_context_pack(cycle_date="2026-11-14", allow_fallback=False)
            self.assertEqual(loaded, sample_pack)

            # Strict cycle lookup returns None for another cycle; fallback returns latest
            self.assertIsNone(await db.get_context_pack(cycle_date="2026-11-15", allow_fallback=False))
            self.assertEqual(await db.get_context_pack(cycle_date="2026-11-15", allow_fallback=True), sample_pack)

            # Corrupted JSON must fail gracefully without crashing
            await db.save_morning_report(
                "bad json report",
                "headline",
                cycle_date="2026-11-16",
                context_pack="{not-valid-json",
            )
            self.assertIsNone(await db.get_context_pack(cycle_date="2026-11-16", allow_fallback=False))

    async def test_recommendation_settlement_idempotency_lessons_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "settlement.db")
            await db.init()

            base_ms = 1_760_000_000_000
            await db.log_recommendations(
                [
                    {
                        "created_at_uk": "2026-11-01 16:00",
                        "created_at_ms": base_ms + 1000,
                        "account": "ISA",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 120.0,
                        "thesis": "AI算力需求景气",
                    },
                    {
                        "created_at_uk": "2026-11-02 16:00",
                        "created_at_ms": base_ms + 2000,
                        "account": "ISA",
                        "symbol": "NVDA",
                        "action": "ADD",
                        "entry_price": 125.0,
                        "thesis": "突破前高加仓",
                        "horizon_days": 10,
                    },
                    {
                        "created_at_uk": "2026-11-03 16:00",
                        "created_at_ms": base_ms + 3000,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 130.0,
                        "thesis": "财报前博弈",
                    },
                    {
                        "created_at_uk": "2026-11-04 16:00",
                        "created_at_ms": base_ms + 4000,
                        "account": "Invest",
                        "symbol": "NVDA",
                        "action": "BUY",
                        "entry_price": 132.0,
                        "thesis": "回调低吸",
                    },
                    {
                        "created_at_uk": "2026-11-05 16:00",
                        "created_at_ms": base_ms + 5000,
                        "account": "ISA",
                        "symbol": "VWRP.L",
                        "action": "WATCH",
                        "entry_price": None,
                        "thesis": "观察宽基回调",
                    },
                ]
            )

            priced_pending = await db.get_pending_settlements(only_with_entry_price=True)
            self.assertEqual(len(priced_pending), 4)
            all_pending = await db.get_pending_settlements()
            self.assertEqual(len(all_pending), 5)
            self.assertEqual(all_pending[0]["horizon_days"], 5)
            self.assertEqual(all_pending[1]["horizon_days"], 10)

            outcomes = [
                (4.0, 2.5, "教训1：景气主升段顺势买入有效。"),
                (-3.0, -4.0, "教训2：追高突破易遇短期获利回吐。"),
                (6.0, 4.5, "教训3：财报前动量强，仓位控制得当。"),
                (2.0, 1.0, "教训4：缩量回踩均线低吸胜率更高。"),
            ]
            for rec, (raw_ret, alpha_ret, lesson) in zip(priced_pending, outcomes):
                ok = await db.settle_recommendation(
                    rec["id"],
                    benchmark_symbol="SPY",
                    settled_at_uk="2026-11-14 21:15",
                    raw_return_pct=raw_ret,
                    alpha_pct=alpha_ret,
                    reflection_lesson=lesson,
                )
                self.assertTrue(ok)

            # Second settle attempt on an already-settled record must be rejected idempotently
            re_settled = await db.settle_recommendation(
                priced_pending[0]["id"],
                benchmark_symbol="QQQ",
                settled_at_uk="2026-11-15 21:15",
                raw_return_pct=99.0,
                alpha_pct=99.0,
                reflection_lesson="should not overwrite",
            )
            self.assertFalse(re_settled)

            # Mark no-price WATCH item as SKIPPED
            skipped_ok = await db.settle_recommendation(
                all_pending[4]["id"],
                benchmark_symbol="ISF.L",
                settled_at_uk="2026-11-14 21:15",
                raw_return_pct=None,
                alpha_pct=None,
                outcome_status="SKIPPED",
                reflection_lesson="",
            )
            self.assertTrue(skipped_ok)
            self.assertEqual(await db.get_pending_settlements(), [])

            # Top 3 lessons for NVDA should return the 3 newest settled lessons
            nvda_lessons = await db.get_symbol_lessons("nvda", limit=3)
            self.assertEqual(len(nvda_lessons), 3)
            self.assertTrue(nvda_lessons[0]["reflection_lesson"].startswith("教训4"))
            self.assertTrue(nvda_lessons[1]["reflection_lesson"].startswith("教训3"))
            self.assertTrue(nvda_lessons[2]["reflection_lesson"].startswith("教训2"))

            recent_lessons = await db.get_recent_lessons(limit=2)
            self.assertEqual(len(recent_lessons), 2)

            # Backward-compatible recent recommendations keeps all legacy keys + adds settlement fields
            recent = await db.get_recent_recommendations(limit=5)
            self.assertEqual(len(recent), 5)
            self.assertEqual(recent[0]["symbol"], "VWRP.L")
            self.assertEqual(recent[0]["outcome_status"], "SKIPPED")
            self.assertIn("thesis", recent[1])
            self.assertEqual(recent[1]["outcome_status"], "SETTLED")

            stats = await db.get_recommendation_stats()
            self.assertEqual(stats["total_count"], 5)
            self.assertEqual(stats["pending_count"], 0)
            self.assertEqual(stats["skipped_count"], 1)
            self.assertEqual(stats["settled_count"], 4)
            self.assertEqual(stats["win_count"], 3)
            self.assertEqual(stats["alpha_win_count"], 3)
            self.assertEqual(stats["win_rate_pct"], 75.0)
            self.assertEqual(stats["alpha_win_rate_pct"], 75.0)
            self.assertEqual(stats["avg_raw_return_pct"], 2.25)
            self.assertEqual(stats["avg_alpha_pct"], 1.0)

    async def test_prune_old_records_preserves_latest_snapshot_and_recent_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "prune.db")
            await db.init()

            now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
            old_ms = int((now - timedelta(days=120)).timestamp() * 1000)
            older_ms = int((now - timedelta(days=150)).timestamp() * 1000)

            # Case 1: Even when ALL snapshots are older than retention_days, the newest snapshot is kept
            await db.save_snapshot(_make_portfolio("2026-06-17 12:00", older_ms, 10000.0))
            await db.save_snapshot(_make_portfolio("2026-07-17 12:00", old_ms, 11000.0))

            async with db._connect(write=True) as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_usage_logs (
                        created_at_uk, created_at_ms, cycle_date, call_label, model, thinking_level, total_tokens
                    ) VALUES (?, ?, '2026-07-17', 'chat', 'gemini-4-hydrogen', 'LOW', 100)
                    """,
                    ("2026-07-17 12:00", old_ms),
                )
                await conn.execute(
                    """
                    INSERT INTO ops_events (
                        created_at_uk, created_at_ms, severity, source, alert_key, message, notified
                    ) VALUES (?, ?, 'WARN', 'test', 'k_old', 'old event', 1)
                    """,
                    ("2026-07-17 12:00", old_ms),
                )
                await conn.commit()

            await db.log_llm_usage(
                {
                    "call_label": "chat",
                    "model": "gemini-4-hydrogen",
                    "thinking_level": "LOW",
                    "prompt_tokens": 50,
                    "output_tokens": 20,
                    "total_tokens": 70,
                }
            )
            await db.log_ops_event(
                severity="WARN",
                source="test",
                alert_key="k_new",
                message="fresh event",
                notified=False,
            )

            deleted = await db.prune_old_records(retention_days=90, now_utc=now)
            self.assertEqual(deleted["portfolio_snapshots"], 1)
            self.assertEqual(deleted["llm_usage_logs"], 1)
            self.assertEqual(deleted["ops_events"], 1)

            latest = await db.get_latest_snapshot()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.total_combined_value, 11000.0)
            self.assertEqual(len(await db.get_recent_ops_events(limit=10)), 1)

    async def test_chat_ordering_crossover_unnotified_ops_and_monthly_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = AdvisorDatabase(Path(tmpdir) / "misc.db")
            await db.init()

            # Insert 4 turns into prior cycle; verify limit=2 returns LATEST 2 in chronological order
            for i in range(1, 5):
                await db.append_chat_turn(f"q{i}", f"a{i}", max_turns=12, cycle_date="2026-11-13")

            latest_two_same = await db.get_chat_history(cycle_date="2026-11-13", limit=2)
            self.assertEqual(latest_two_same, [("q3", "a3"), ("q4", "a4")])

            prev_two = await db.get_previous_cycle_chat_history(cycle_date="2026-11-14", limit=2)
            self.assertEqual(prev_two, [("q3", "a3"), ("q4", "a4")])

            # Unnotified ops event recovery helpers
            await db.log_ops_event(
                severity="INFO",
                source="lifecycle",
                alert_key="startup",
                message="started",
                notified=False,
            )
            await db.log_ops_event(
                severity="CRITICAL",
                source="gemini",
                alert_key="outage",
                message="all down",
                notified=False,
            )
            await db.log_ops_event(
                severity="WARN",
                source="t212",
                alert_key="timeout",
                message="rate limit",
                notified=True,
            )

            unnotified = await db.get_unnotified_ops_events(hours=24)
            self.assertEqual(len(unnotified), 1)
            self.assertEqual(unnotified[0]["alert_key"], "outage")
            updated = await db.mark_ops_events_notified([unnotified[0]["id"]])
            self.assertEqual(updated, 1)
            self.assertEqual(await db.get_unnotified_ops_events(hours=24), [])

            # Usage summary includes calendar-month bucket `this_month`
            await db.log_llm_usage(
                {
                    "call_label": "report_grounding",
                    "model": "gemini-4-hydrogen",
                    "thinking_level": "LOW",
                    "prompt_tokens": 200,
                    "output_tokens": 80,
                    "total_tokens": 280,
                    "search_queries": 4,
                }
            )
            usage = await db.get_usage_summary(days=30)
            self.assertIn("this_month", usage)
            self.assertEqual(usage["this_month"]["search_queries"], 4)


if __name__ == "__main__":
    unittest.main()
