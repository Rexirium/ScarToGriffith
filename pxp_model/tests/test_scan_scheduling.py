"""Memory startup and local scheduling checks without numerical dependencies."""

from concurrent.futures import Future
from contextlib import redirect_stderr, redirect_stdout
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
            pool_size = executor.call_args.kwargs["max_workers"]
            self.assertLessEqual(len(active), pool_size)
            self.assertLessEqual((1 + pool_size - len(active)) * scan._PROCESS_MEMORY
                                 + sum(estimates[p[0]] for p in active.values()),
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
        _, batches, _ = self.run_schedule([28, 26, 24], 3, 5.0)
        self.assertEqual([p[0] for p in batches[0]], [28, 24])
        self.assertIn((28, 0, 0.), batches[1])
        self.assertIn((24, 1, .1), batches[1])

    def test_completed_point_releases_budget(self):
        _, batches, _ = self.run_schedule([26, 24], 3, 5.0, large_first=True)
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
                    budget, estimates, workers = scan.check_memory(
                        config, [(L, i, float(i)) for L in config["sizes"] for i in range(4)], 4)
                    self.assertEqual(budget, 7.5 * 2**30)
                    self.assertEqual(estimates, {L: 3 * 2**30 for L in config["sizes"]})
                    self.assertEqual(workers, 4)

    def test_idle_workers_are_reserved_when_only_one_point_runs(self):
        _, batches, _ = self.run_schedule([28, 26, 24], 4, 4.5)
        self.assertEqual([p[0] for p in batches[0]], [28])

    def test_unschedulable_point_fails_instead_of_waiting_on_empty_futures(self):
        with patch.object(scan, "ProcessPoolExecutor"), patch.object(scan, "wait") as wait:
            with self.assertRaisesRegex(RuntimeError, "No queued point fits"):
                scan._run_points({}, [(28, 0, .655)], Path("unused"), 1, 0,
                                 .5 * 2**30, {28: 2**30})
            wait.assert_not_called()


class MemoryStartupTest(unittest.TestCase):
    def test_invalid_budget_fails_before_output_or_pool(self):
        for L, budget, environment, message in (
            (4, "", {}, "Memory budget is unknown"),
            (4, "", {"SLURM_MEM_PER_NODE": "0"}, "Memory budget is unknown"),
            (4, "memory_gib = 0.5\n", {}, "exceeds budget"),
            (30, "memory_gib = 128\n", {}, "exceeds budget"),
            (4, "memory_gib = 8\n", {"SLURM_MEM_PER_NODE": "512"}, "exceeds budget"),
            (4, "memory_gib = 8\n", {"SLURM_MEM_PER_NODE": "invalid"}, "invalid"),
            (4, "memory_gib = 8\n", {"SLURM_NTASKS_PER_NODE": "2"}, "one Slurm task per node"),
            (4, "memory_gib = 8\n", {"SLURM_TASKS_PER_NODE": "1,2(x3)"}, "one Slurm task per node"),
            (4, "memory_gib = 1e308\n", {}, "finite positive byte budget"),
        ):
            with self.subTest(L=L, budget=budget, environment=environment), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "scan.toml"
                config_path.write_text(
                    f'sizes = [{L}]\nms = [0.0]\ntemperatures = [1.0]\n'
                    'real_times = [0.0]\nimaginary_times = [0.0]\n' + budget,
                    encoding="utf-8",
                )
                stderr = io.StringIO()
                with patch.dict(os.environ, environment, clear=True), \
                     patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                     patch.object(scan, "ProcessPoolExecutor") as executor, \
                     redirect_stdout(io.StringIO()), redirect_stderr(stderr), \
                     self.assertRaises(SystemExit) as error:
                    scan.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                executor.assert_not_called()
                self.assertFalse((Path(directory) / "data").exists())

    def test_run_uses_slurm_budget_without_explicit_limit(self):
        for environment in ({"SLURM_MEM_PER_NODE": "2048"},
                            {"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_PER_TASK": "2"}):
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "scan.toml"
                config_path.write_text(
                    'sizes = [4]\nms = [0.0]\ntemperatures = [1.0]\n'
                    'real_times = [0.0]\nimaginary_times = [0.0]\n', encoding="utf-8",
                )
                completed = Future()
                completed.set_result(Path("result.npz"))
                with patch.dict(os.environ, environment, clear=True), \
                     patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                     patch.object(scan, "ProcessPoolExecutor") as executor, redirect_stdout(io.StringIO()):
                    executor.return_value.__enter__.return_value.submit.return_value = completed
                    scan.main()
                executor.return_value.__enter__.return_value.submit.assert_called_once()
                output_dir = executor.return_value.__enter__.return_value.submit.call_args.args[-1]
                self.assertRegex(output_dir.name, r"^\d{8}_\d{6}$")

    def test_empty_rank_needs_no_memory_budget_or_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scan.toml"
            config_path.write_text(
                'sizes = [4]\nms = [0.0]\ntemperatures = [1.0]\n'
                'real_times = [0.0]\nimaginary_times = [0.0]\n', encoding="utf-8",
            )
            run = Path(directory) / "data" / "autocorrelation" / "empty"
            with patch.dict(os.environ, {}, clear=True):
                scan.prepare_output(run, config_path.read_text(encoding="utf-8"), 0, 2)
            with patch.dict(os.environ, {"SLURM_NTASKS": "2", "SLURM_PROCID": "1"}, clear=True), \
                 patch.object(sys, "argv", ["scan", "--config", str(config_path), "--run-id", "empty"]), \
                 patch.object(scan, "check_memory") as check, \
                 patch.object(scan, "ProcessPoolExecutor") as executor, redirect_stdout(io.StringIO()):
                scan.main()
            check.assert_not_called()
            executor.assert_not_called()

    def test_shipped_config_keeps_eight_workers_and_threads(self):
        config, _ = scan.load_config(Path(scan.__file__).with_suffix(".toml"))
        self.assertEqual((config["workers"], config["threads"]), (8, 8))

    def test_preflight_reports_each_size_and_reduces_idle_pool_overhead(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(scan, "estimate_point_memory", return_value=2 * 2**30), \
             redirect_stdout(io.StringIO()) as output:
            budget, estimates, workers = scan.check_memory(
                {"memory_gib": 3}, [(28, 0, .655), (26, 0, .655), (26, 1, .7)], 3)
        self.assertEqual(workers, 2)
        self.assertEqual(budget, 2.5 * 2**30)
        for L in estimates:
            self.assertIn(f"L={L}: largest block=", output.getvalue())
        self.assertIn("up to 1 concurrent points", output.getvalue())

    def test_l28_config_fits_the_node_memory_and_cpu_budget(self):
        config, _ = scan.load_config(Path(scan.__file__).with_name("pxp_autocorrelation_scan_L28.toml"))
        self.assertEqual(config["sizes"], [28])
        self.assertEqual((config["workers"], config["threads"]), (2, 32))
        self.assertEqual(config["workers"] * config["threads"], 64)
        peak = .5 * 2**30 + config["workers"] * scan.estimate_point_memory(28, config)
        self.assertLess(peak, config["memory_gib"] * 2**30)
        self.assertLessEqual(config["memory_gib"], 1024)

    def test_shipped_configs_have_log_temperatures_and_the_exact_field(self):
        for name in ("pxp_autocorrelation_scan.toml", "pxp_autocorrelation_scan_L28.toml"):
            with self.subTest(config=name):
                config, _ = scan.load_config(Path(scan.__file__).with_name(name))
                self.assertEqual(len(config["temperatures"]), 102)
                self.assertEqual(config["temperatures"][0], 0.)
                temperatures = config["temperatures"][1:]
                self.assertEqual((temperatures[0], temperatures[-1]), (.01, 1.))
                for left, right in zip(temperatures, temperatures[1:]):
                    self.assertAlmostEqual(right / left, 10**.02)
                self.assertEqual(len(config["ms"]), 52)
                expected_fields = sorted([i / 50 for i in range(51)] + [.655])
                for actual, expected in zip(config["ms"], expected_fields):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(config["ms"][32:35], [.64, .655, .66])
                self.assertEqual(config["ms"].count(.655), 1)


if __name__ == "__main__":
    unittest.main()
