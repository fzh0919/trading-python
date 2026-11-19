"""SQLite connection core, single write-lock entry (_connect), cycle helpers, and column migration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite

LONDON_TZ = ZoneInfo("Europe/London")


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


class DatabaseCore:
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
