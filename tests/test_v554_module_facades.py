"""Architectural guardrail tests for v5.5.4 core module slimdown."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

REFACTORED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("src.engine.deployment_gate", "src/engine/deployment_gate"),
    ("src.data_providers.analytics", "src/data_providers/analytics"),
    ("src.engine.screener", "src/engine/screener"),
    ("src.storage.db", "src/storage/db"),
    ("src.engine.portfolio_rules", "src/engine/portfolio_rules"),
    ("src.data_providers.t212_client", "src/data_providers/t212_client"),
    ("src.data_providers.market_data", "src/data_providers/market_data"),
    ("src.agents.report_agent", "src/agents/report_agent"),
    ("src.bot.scheduler", "src/bot/scheduler"),
)

MAX_SUBMODULE_LINES = 1000


class TestV554ModuleFacades(unittest.TestCase):
    def test_all_nine_packages_import_cleanly_and_expose_declared_all(self) -> None:
        for mod_name, rel_dir in REFACTORED_PACKAGES:
            pkg_dir = ROOT / rel_dir
            self.assertTrue(pkg_dir.is_dir(), f"Expected subpackage directory: {rel_dir}")
            self.assertFalse(
                pkg_dir.with_suffix(".py").exists(),
                f"Legacy monolithic file should be replaced by package: {rel_dir}.py",
            )
            mod = importlib.import_module(mod_name)
            exported = getattr(mod, "__all__", None)
            self.assertIsInstance(exported, list, f"{mod_name} must define __all__")
            self.assertTrue(len(exported) > 0, f"{mod_name}.__all__ must be non-empty")
            for sym in exported:
                self.assertTrue(
                    hasattr(mod, sym),
                    f"{mod_name}.__all__ declares missing symbol {sym!r}",
                )

    def test_refactored_submodules_stay_below_line_ceiling(self) -> None:
        oversized: list[str] = []
        for _, rel_dir in REFACTORED_PACKAGES:
            for py_file in sorted((ROOT / rel_dir).glob("*.py")):
                line_count = len(py_file.read_text(encoding="utf-8").splitlines())
                if line_count >= MAX_SUBMODULE_LINES:
                    oversized.append(f"{rel_dir}/{py_file.name} ({line_count}L)")
        self.assertEqual(
            oversized,
            [],
            f"Refactored submodules exceeded {MAX_SUBMODULE_LINES}L ceiling: {oversized}",
        )

    def test_db_single_write_lock_entry_iron_law(self) -> None:
        """Only src/storage/db/core.py may import or call aiosqlite directly."""
        db_dir = ROOT / "src/storage/db"
        for py_file in sorted(db_dir.glob("*.py")):
            if py_file.name == "core.py":
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(
                            alias.name,
                            "aiosqlite",
                            f"{py_file.name} must not import aiosqlite directly",
                        )
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(
                        node.module,
                        "aiosqlite",
                        f"{py_file.name} must not import from aiosqlite directly",
                    )

    def test_mock_patch_facade_seams_remain_bound(self) -> None:
        market_data = importlib.import_module("src.data_providers.market_data")
        self.assertTrue(hasattr(market_data, "httpx"))
        self.assertTrue(hasattr(market_data, "yf"))
        self.assertTrue(hasattr(market_data, "time"))

        report_agent = importlib.import_module("src.agents.report_agent")
        for attr in (
            "get_market_calendar_alerts",
            "get_us_market_schedule_uk",
            "is_uk_bank_holiday",
            "run_forced_market_grounding",
        ):
            self.assertTrue(
                hasattr(report_agent, attr),
                f"report_agent facade missing patch seam attribute {attr!r}",
            )

        scheduler = importlib.import_module("src.bot.scheduler")
        for attr in ("generate_morning_report", "run_universe_audition"):
            self.assertTrue(
                hasattr(scheduler, attr),
                f"scheduler facade missing patch seam attribute {attr!r}",
            )


if __name__ == "__main__":
    unittest.main()
