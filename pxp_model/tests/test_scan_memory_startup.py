"""Memory startup checks without importing numerical dependencies."""

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .. import pxp_autocorrelation_scan as scan


class MemoryStartupTest(unittest.TestCase):
    def test_run_starts_with_unknown_or_insufficient_memory_budget(self):
        for budget in ("", "memory_gib = 0.001\n"):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "scan.toml"
                config_path.write_text(
                    'sizes = [30]\nms = [0.0]\ntemperatures = [1.0]\n'
                    'real_times = [0.0]\nimaginary_times = [0.0]\n' + budget,
                    encoding="utf-8",
                )
                completed = Future()
                completed.set_result(Path("result.npz"))
                with patch.dict(os.environ, {}, clear=True), \
                     patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                     patch.object(scan, "check_memory") as check, \
                     patch.object(scan, "ProcessPoolExecutor") as executor, \
                     redirect_stdout(io.StringIO()):
                    executor.return_value.__enter__.return_value.submit.return_value = completed
                    scan.main()
                check.assert_not_called()
                executor.return_value.__enter__.return_value.submit.assert_called_once()

    def test_manual_check_still_rejects_insufficient_memory(self):
        config = dict(temperatures=[1.0], real_times=[0.0], imaginary_times=[0.0],
                      memory_gib=0.001)
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "exceeds budget"):
                scan.check_memory(config, [(4, 0, 0.0)], 1)


if __name__ == "__main__":
    unittest.main()
