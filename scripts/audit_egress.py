"""Read-only egress & secret-handling audit for this repository (stdlib only).

It never imports project code and never touches the network: it only reads ``*.py``
files and ``requirements.txt``, then prints what a human must glance at after any
change made with a third-party model in the loop.

Checks
------
1. Outbound URL literals          -> unknown host = FAIL (a new host is a red light)
2. Credential-looking log calls   -> FAIL
3. Secret embedded in a URL       -> WARN (httpx logs are muted in main.py, keep it that way)
4. Environment-variable reads     -> INFO (verify no secret is echoed)
5. Process spawn / dynamic exec   -> INFO
6. Runtime dependency drift       -> FAIL (any new dependency needs human review)

Usage
-----
    python scripts/audit_egress.py
    python scripts/audit_egress.py --tests

Exit code: 0 = clean, 1 = review required.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "api.anthropic.com",
        "api.deepseek.com",
        "api.stlouisfed.org",
        "api.telegram.org",
        "demo.trading212.com",
        "en.wikipedia.org",
        "finnhub.io",
        "generativelanguage.googleapis.com",
        "github.com",
        "live.trading212.com",
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    }
)

EXPECTED_DEPENDENCIES: frozenset[str] = frozenset(
    {
        "aiosqlite",
        "google-genai",
        "httpx",
        "numpy",
        "pandas",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "pyyaml",
        "tenacity",
        "tzdata",
        "yfinance",
    }
)

SKIP_DIRS = {"__pycache__", ".venv", "_ref", ".git", "node_modules"}
URL_RE = re.compile(r"https?://[^\s\"')<>,;`]+")
_SECRET_WORD = r"(?:api[_-]?key|apikey|token|secret|password|passwd)"
# Only flag a secret that is actually *evaluated* inside the call: an f-string
# interpolation ({...}), a self.<attr>, or cfg.env. Plain prose such as
# "TELEGRAM_BOT_TOKEN is empty" must not raise a finding.
SECRET_LOG_RE = re.compile(
    r"\b(?:logger|logging)\.(?:debug|info|warning|error|exception|critical)\("
    r"[^\n]*?(?:\{[^}\n]*" + _SECRET_WORD + r"|\bself\.[A-Za-z0-9_.]*" + _SECRET_WORD + r"|\bcfg\.env\b)",
    re.IGNORECASE,
)
PRINT_SECRET_RE = re.compile(
    r"\bprint\([^\n]*?(?:\{[^}\n]*" + _SECRET_WORD + r"|\bself\.[A-Za-z0-9_.]*" + _SECRET_WORD + r"|\bcfg\.env\b)",
    re.IGNORECASE,
)
ENV_RE = re.compile(r"os\.environ|os\.getenv|\bgetenv\(")
EXEC_RE = re.compile(r"subprocess\.|os\.system\(|\beval\(|\bexec\(")
SECRET_IN_URL_RE = re.compile(r"(?:token|api[_-]?key|secret|bot)\{", re.IGNORECASE)


def _iter_python_files(roots: list[str], include_tests: bool) -> list[Path]:
    bases = [ROOT / name for name in roots]
    if include_tests:
        bases.append(ROOT / "tests")
    files: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if SKIP_DIRS.intersection(path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def _host_ok(url: str) -> tuple[str, bool]:
    """Return (host, is_template). Template URLs such as ``https://{host}/...`` are skipped."""
    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0].split("@")[-1].split(":")[0].lower()
    return host, "{" in host or "}" in host


def _parse_requirements(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = re.split(r"[<>=!~;\[ ]", stripped, 1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Static egress & secret-handling audit.")
    parser.add_argument("--tests", action="store_true", help="also scan the tests/ tree")
    parser.add_argument("--roots", nargs="*", default=["src"], help="source roots to scan")
    args = parser.parse_args()

    files = _iter_python_files(args.roots, args.tests)
    unknown_hosts: dict[str, list[str]] = {}
    known_hosts: dict[str, int] = {}
    secret_in_url: list[str] = []
    secret_logs: list[str] = []
    env_reads: list[str] = []
    exec_sites: list[str] = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, start=1):
            for url in URL_RE.findall(line):
                host, is_template = _host_ok(url)
                if is_template:
                    continue
                if host in ALLOWED_HOSTS:
                    known_hosts[host] = known_hosts.get(host, 0) + 1
                else:
                    unknown_hosts.setdefault(host, []).append(f"{rel}:{lineno}")
                if SECRET_IN_URL_RE.search(url):
                    secret_in_url.append(f"{rel}:{lineno}  {url[:96]}")
            if SECRET_LOG_RE.search(line) or PRINT_SECRET_RE.search(line):
                secret_logs.append(f"{rel}:{lineno}  {line.strip()[:110]}")
            if ENV_RE.search(line):
                env_reads.append(f"{rel}:{lineno}")
            if EXEC_RE.search(line):
                exec_sites.append(f"{rel}:{lineno}  {line.strip()[:96]}")

    print(f"scanned files : {len(files)}")
    known = ", ".join(f"{h}({n})" for h, n in sorted(known_hosts.items())) or "(none)"
    print(f"known hosts   : {known}")

    print("\n[FAIL] unknown outbound hosts")
    for host, where in sorted(unknown_hosts.items()):
        print(f"  - {host}: {', '.join(where[:4])}")
    if not unknown_hosts:
        print("  (none)")

    print("\n[FAIL] credential-looking log/print calls")
    for item in secret_logs or ["  (none)"]:
        print(f"  {item}")

    print("\n[WARN] secret embedded in a URL literal")
    for item in secret_in_url or ["  (none)"]:
        print(f"  {item}")

    print("\n[INFO] environment-variable reads")
    for item in env_reads or ["  (none)"]:
        print(f"  {item}")

    print("\n[INFO] process spawn / dynamic exec")
    for item in exec_sites or ["  (none)"]:
        print(f"  {item}")

    actual_deps = _parse_requirements(ROOT / "requirements.txt")
    added = sorted(actual_deps - EXPECTED_DEPENDENCIES)
    removed = sorted(EXPECTED_DEPENDENCIES - actual_deps)
    print("\n[FAIL] runtime dependency drift")
    print(f"  added  : {', '.join(added) or '(none)'}")
    print(f"  removed: {', '.join(removed) or '(none)'}")

    failed = bool(unknown_hosts or secret_logs or added or removed)
    print("\nRESULT: " + ("REVIEW REQUIRED" if failed else "CLEAN"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
