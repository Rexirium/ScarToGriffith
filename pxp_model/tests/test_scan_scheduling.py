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
    def run_schedule(self, sizes, workers, memory_gib, *, large_first=False, fail=False):
        submitted, batches, active = [], [], {}
        Lmax = max(sizes)
        estimates = {L: (i + 1) * 2**30 for i, L in enumerate(sorted(sizes))}

        def submit(function, L, index, m, *args):
            self.assertIs(function, scan.run_point)
            future = Future()
            point = (L, index, m)
            submitted.append(point)
            active[future] = point
            self.assertLessEqual(len(active), workers)
            self.assertLessEqual(512 * 2**20 + sum(estimates[p[0]] for p in active.values()),
                                 memory_gib * 2**30)
            return future

        def finish(futures, *, return_when):
            self.assertEqual(return_when, scan.FIRST_COMPLETED)
            self.assertEqual(set(futures), set(active))
            self.assertTrue(active)
            batches.append(list(active.values()))
            # Exercise either a slow largest-size task or one finishing first.
            future = next((f for f, p in active.items() if (p[0] == Lmax) == large_first),
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
                f'memory_gib = {memory_gib}\n'
                'ms = [0.0, 0.1, 0.2]\ntemperatures = [1.0]\n'
                'real_times = [0.0]\nimaginary_times = [0.0]\n', encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(sys, "argv", ["scan", "--config", str(config)]), \
                 patch.object(scan, "estimate_point_memory", side_effect=lambda L, _: estimates[L]), \
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

    def test_budget_and_coverage_under_different_completion_orders(self):
        for sizes, workers, memory_gib in (
            ([28, 26, 24], 3, 7.5), ([28], 4, 3.5), ([26], 4, 1.5),
            ([26, 24], 3, 4.5), ([28, 26], 1, 8),
            ([30, 28, 26], 4, 4.5), ([26, 30, 28], 4, 16),
        ):
            for large_first in (False, True):
                with self.subTest(sizes=sizes, workers=workers, budget=memory_gib, large_first=large_first):
                    submitted, _, active = self.run_schedule(
                        sizes, workers, memory_gib, large_first=large_first)
                    expected = [(L, i, m) for L in sizes for i, m in enumerate([0., .1, .2])]
                    self.assertCountEqual(submitted, expected)
                    self.assertFalse(active)

    def test_multiple_largest_points_fit_exact_budget(self):
        _, batches, _ = self.run_schedule([26, 24], 3, 6.5)
        self.assertEqual([p[0] for p in batches[0]], [26, 26, 26])

    def test_smaller_points_fill_remaining_memory(self):
        _, batches, _ = self.run_schedule([28, 26, 24], 4, 4.5)
        self.assertEqual([p[0] for p in batches[0]], [28, 24])
        self.assertIn((28, 0, 0.), batches[1])
        self.assertIn((24, 1, .1), batches[1])

    def test_completed_point_releases_budget(self):
        _, batches, _ = self.run_schedule([26, 24], 3, 4.5, large_first=True)
        self.assertEqual([p[0] for p in batches[0]], [26, 26])
        self.assertEqual([p[0] for p in batches[1]], [26, 26])
        self.assertIn((26, 2, .2), batches[1])

    def test_failure_cancels_active_jobs_without_submitting_more(self):
        submitted, batches, active = self.run_schedule([28, 26], 3, 6.5, fail=True)
        self.assertEqual(len(submitted), 3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(active), 2)
        self.assertTrue(all(future.cancelled() for future in active))

    def test_rank_only_budgets_its_assigned_sizes(self):
        points = [(26, i, m) for i, m in enumerate([0., .1, .2])]
        with patch.object(scan, "slurm_tasks", return_value=(0, 1, points)):
            submitted, batches, active = self.run_schedule([26, 30], 3, 3.5)
        self.assertEqual(batches[0], points)
        self.assertCountEqual(submitted, points)
        self.assertFalse(active)

    def test_preflight_accepts_points_that_need_memory_throttling(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(scan, "estimate_point_memory", return_value=3 * 2**30), \
             redirect_stdout(io.StringIO()):
            for Lmax in (26, 28, 30):
                with self.subTest(Lmax=Lmax):
                    config = {"memory_gib": 8, "sizes": [Lmax - 2, Lmax]}
                    budget, estimates = scan.check_memory(
                        config, [(L, i, float(i)) for L in config["sizes"] for i in range(4)], 4)
                    self.assertEqual(budget, 7.5 * 2**30)
                    self.assertEqual(estimates, {L: 3 * 2**30 for L in config["sizes"]})


if __name__ == "__main__":
    unittest.main()
