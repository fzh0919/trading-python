"""Test runner with hard faulthandler watchdog timeout (never hangs silently)."""

from __future__ import annotations

import faulthandler
import os
import sys
import unittest

DEFAULT_WATCHDOG_SECONDS = max(1, int(os.environ.get("TEST_WATCHDOG_SECONDS", "60")))


def main() -> int:
    faulthandler.enable()
    # Hard watchdog across the entire mocked test suite (overridable via TEST_WATCHDOG_SECONDS)
    faulthandler.dump_traceback_later(DEFAULT_WATCHDOG_SECONDS, exit=True)
    try:
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
