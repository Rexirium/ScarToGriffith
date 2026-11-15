"""Exercise local scheduling without running expensive parameter points."""

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


class ScanSchedulingTest(unittest.TestCase):
    def run_schedule(self, sizes, workers, *, large_first=False, fail=False):
        submitted, batches, active = [], [], {}

        def submit(function, L, index, m, *args):
            self.assertIs(function, scan.run_point)
            future = Future()
            point = (L, index, m)
            submitted.append(point)
            active[future] = point
            self.assertLessEqual(len(active), workers)
            self.assertLessEqual(sum(p[0] == 28 for p in active.values()), 1)
            return future

        def finish(futures, *, return_when):
            self.assertEqual(return_when, scan.FIRST_COMPLETED)
            self.assertEqual(set(futures), set(active))
            self.assertTrue(active)
            batches.append(list(active.values()))
            # Exercise either a slow L=28 or L=28 finishing before its peers.
            future = next((f for f, p in active.items() if (p[0] == 28) == large_first),
                          next(iter(active)))
            active.pop(future)
            if fail:
                future.set_exception(ValueError("worker failed"))
            else:
                future.set_result(Path("result.npz"))
            return {future}, set(active)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "scan.toml"
            config.write_text(
                f'sizes = {sizes}\nworkers = {workers}\nthreads = 1\n'
                'ms = [0.0, 0.1, 0.2]\ntemperatures = [1.0]\n'
                'real_times = [0.0]\nimaginary_times = [0.0]\n', encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(sys, "argv", ["scan", "--config", str(config)]), \
                 patch.object(scan, "ProcessPoolExecutor") as executor, \
                 patch.object(scan, "wait", side_effect=finish), redirect_stdout(io.StringIO()):
                executor.return_value.__enter__.return_value.submit.side_effect = submit
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "Autocorrelation task failed") as error:
                        scan.main()
                    self.assertIsInstance(error.exception.__cause__, ValueError)
                else:
                    scan.main()
        return submitted, batches, active

    def test_limit_and_coverage_under_different_completion_orders(self):
        for sizes, workers in (([28, 26, 24], 3), ([28], 4), ([26, 24], 3),
                               ([28, 26], 1), ([30, 28, 26], 4)):
            for large_first in (False, True):
                with self.subTest(sizes=sizes, workers=workers, large_first=large_first):
                    submitted, batches, active = self.run_schedule(
                        sizes, workers, large_first=large_first)
                    expected = [(L, i, m) for L in sizes for i, m in enumerate([0., .1, .2])]
                    self.assertCountEqual(submitted, expected)
                    self.assertFalse(active)
                    if sizes == [28, 26, 24]:
                        self.assertEqual([p[0] for p in batches[0]], [28, 26, 26])
                        if not large_first:
                            # Small jobs keep advancing while the first L=28 is running.
                            self.assertTrue(any(any(p[0] == 24 for p in batch)
                                                and (28, 0, 0.) in batch for batch in batches))

    def test_failure_cancels_active_jobs_without_submitting_more(self):
        submitted, batches, active = self.run_schedule([28, 26], 3, fail=True)
        self.assertEqual(len(submitted), 3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(active), 2)
        self.assertTrue(all(future.cancelled() for future in active))

    def test_manual_memory_estimate_accounts_for_l28_limit(self):
        config = {"memory_gib": 8}
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(scan, "estimate_point_memory", return_value=3 * 2**30), \
             redirect_stdout(io.StringIO()):
            scan.check_memory(config, [(28, i, float(i)) for i in range(4)], 4)
            with self.assertRaisesRegex(ValueError, "exceeds budget"):
                scan.check_memory(config, [(26, i, float(i)) for i in range(4)], 4)


if __name__ == "__main__":
    unittest.main()
