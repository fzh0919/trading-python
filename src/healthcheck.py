"""Pure-Python container healthcheck verifying event-loop heartbeat freshness and SQLite readiness."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_HEARTBEAT_PATH = Path(os.environ.get("ADVISOR_HEARTBEAT_PATH", "/tmp/advisor.heartbeat"))
DEFAULT_DB_PATH = Path(os.environ.get("ADVISOR_DB_PATH", "/app/data/advisor.db"))
DEFAULT_MAX_STALE_SECONDS = 180.0


def write_heartbeat(path: str | Path = DEFAULT_HEARTBEAT_PATH, now_ts: float | None = None) -> Path:
    """Atomically touch/update the process heartbeat file for Docker healthcheck probing."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time() if now_ts is None else float(now_ts)
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(f"{ts:.3f}\n", encoding="utf-8")
    os.replace(tmp_path, target)
    return target


def check_health(
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    max_stale_seconds: float = DEFAULT_MAX_STALE_SECONDS,
    now_ts: float | None = None,
) -> tuple[bool, str]:
    """Check heartbeat file freshness and read-only SQLite database status."""
    hb = Path(heartbeat_path)
    if not hb.exists():
        return False, f"heartbeat missing: {hb}"

    try:
        mtime = hb.stat().st_mtime
    except OSError as exc:
        return False, f"heartbeat stat failed: {exc}"

    current = time.time() if now_ts is None else float(now_ts)
    age = current - mtime
    if age > max_stale_seconds:
        return False, f"heartbeat stale: age={age:.1f}s > {max_stale_seconds:.1f}s"

    db = Path(db_path)
    if not db.exists():
        return False, f"database missing: {db}"

    conn: sqlite3.Connection | None = None
    try:
        db_uri = f"file:{db.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=2.0)
        row = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()
        table_count = int(row[0]) if row else 0
        if table_count < 1:
            return False, "database has no tables"
    except sqlite3.Error as exc:
        return False, f"database check failed: {exc}"
    finally:
        if conn is not None:
            conn.close()

    return True, "ok"


def main() -> int:
    ok, reason = check_health()
    if not ok:
        print(f"UNHEALTHY: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
