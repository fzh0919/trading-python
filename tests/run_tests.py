"""Test runner with hard faulthandler watchdog timeout (never hangs silently)."""

from __future__ import annotations

import faulthandler
import sys
import unittest


def main() -> int:
    faulthandler.enable()
    # Hard 10-second watchdog across the entire mocked test suite
    faulthandler.dump_traceback_later(10, exit=True)
    try:
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
