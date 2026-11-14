"""Checks: python -m unittest pxp_model.tests.test_pxp_autocorrelation_scan"""

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .. import pxp_autocorrelation as correlation
from .. import pxp_autocorrelation_scan as scan


class BothTimeDomainsTest(unittest.TestCase):
    def test_normalized_times_share_ed_and_match_individual_temperature_calls(self):
        temperatures = [0., .1, .5, 10., np.inf]
        times = np.array([0., .1, 2000., 5000., 8000., 1e4])
        with patch.object(correlation, "build_hamiltonian", wraps=correlation.build_hamiltonian) as build:
            real, imaginary = correlation.connected_autocorrelation_both_times(
                6, .2, temperatures, [0., .3], times)
        self.assertEqual(build.call_count, 4)
        baseline = correlation.connected_autocorrelation(6, .2, temperatures, [0., .3])
        for actual, expected in zip(real, baseline):
            np.testing.assert_allclose(actual, expected, atol=3e-12)
        for column, temperature in enumerate(temperatures):
            baseline = correlation.connected_autocorrelation(6, .2, temperature, times, True)
            for actual, expected in zip(imaginary, baseline):
                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_allclose(actual[:, column], expected[:, 0], atol=3e-12)
                if 0 < temperature < np.inf:
                    np.testing.assert_allclose(actual[[0, 2], column], actual[[-1, -2], column], atol=3e-12)
                if temperature == 0:
                    self.assertLess(abs(actual[-1, column]), 1e-12)

    def test_time_grid_normalization_and_edge_cases(self):
        temperatures = [0., .5, 2., np.inf]
        times = np.array([4., 0., 2., 4.])
        _, actual = correlation.prepare_time_grid(temperatures, times, True)
        np.testing.assert_array_equal(actual, [[4., 2., .5, 0.], [0., 0., 0., 0.],
                                               [2., 1., .25, 0.], [4., 2., .5, 0.]])
        np.testing.assert_array_equal(times, [4., 0., 2., 4.])
        _, rescaled = correlation.prepare_time_grid(temperatures, times * 100, True)
        np.testing.assert_array_equal(actual[:, 1:], rescaled[:, 1:])
        np.testing.assert_array_equal(rescaled[:, 0], times * 100)
        _, real = correlation.prepare_time_grid(temperatures, times, False)
        np.testing.assert_array_equal(real, times)
        _, zeros = correlation.prepare_time_grid(temperatures, [0., 0.], True)
        np.testing.assert_array_equal(zeros, np.zeros((2, 4)))
        _, singleton = correlation.prepare_time_grid(temperatures, [4.], True)
        np.testing.assert_array_equal(singleton, [[4., 2., .5, 0.]])
        _, ground = correlation.prepare_time_grid([0.], [-1., 0.], True)
        np.testing.assert_array_equal(ground[:, 0], [-1., 0.])
        with patch.object(correlation, "_momentum_eigensystems") as diagonalize:
            for Ts, grid in (([0., 1.], [-1., 0.]), ([np.nextafter(0., 1.)], [0., 1.])):
                with self.subTest(Ts=Ts, grid=grid), self.assertRaises(ValueError):
                    correlation.connected_autocorrelation_both_times(
                        4, .2, Ts, [0.], grid)
            real, imaginary = correlation.connected_autocorrelation_both_times(
                4, .2, [0., 1.], [], [])
            self.assertEqual(real[0].shape, (0, 2))
            self.assertEqual(imaginary[0].shape, (0, 2))
            diagonalize.assert_not_called()

    def test_shared_ed_and_weights_match_independent_calls(self):
        temperatures = [0., .5, np.inf]
        times, taus = [1., 0., -.2, 1.], [0., .1, .3]
        with patch.object(correlation, "build_hamiltonian",
                          wraps=correlation.build_hamiltonian) as build, \
             patch.object(correlation, "_thermal_log_weights",
                          wraps=correlation._thermal_log_weights) as weights:
            real, imaginary = correlation.connected_autocorrelation_both_times(
                6, .2, temperatures, times, taus)
        self.assertEqual(build.call_count, 4)
        weights.assert_called_once()
        for actual, grid, is_imaginary in ((real, times, False), (imaginary, taus, True)):
            expected = correlation.connected_autocorrelation(6, .2, temperatures, grid, is_imaginary)
            for result, baseline in zip(actual, expected):
                np.testing.assert_allclose(result, baseline, atol=2e-12)
                self.assertEqual(result.shape, (len(grid), 3))
                self.assertEqual(np.iscomplexobj(result), not is_imaginary)
            self.assertFalse(np.shares_memory(*actual))

    def test_empty_grids_and_invalid_imaginary_grid(self):
        with patch.object(correlation, "_momentum_eigensystems") as diagonalize:
            real, imaginary = correlation.connected_autocorrelation_both_times(4, .2, .5, [], [])
            self.assertEqual(real[0].shape, (0, 1))
            self.assertEqual(imaginary[0].shape, (0, 1))
            for invalid in ([np.inf], [[0.]], [1j], 1):
                with self.assertRaises(ValueError):
                    correlation.connected_autocorrelation_both_times(4, .2, .5, [0.], invalid)
            diagonalize.assert_not_called()
        for times, taus in (([], [0., .1]), ([0., .1], [])):
            with patch.object(correlation, "_uniform_connected", wraps=correlation._uniform_connected) as uniform, \
                 patch.object(correlation, "_local_from_average", wraps=correlation._local_from_average) as local:
                actual = correlation.connected_autocorrelation_both_times(4, .2, [.5, 0.], times, taus)
                uniform.assert_called_once()
                local.assert_called_once()
                self.assertEqual(uniform.call_args.args[2].shape[0], 2)
                self.assertEqual(local.call_args.args[2].shape[0], 2)
            for result, grid, imaginary in zip(actual, (times, taus), (False, True)):
                expected = correlation.connected_autocorrelation(4, .2, [.5, 0.], grid, imaginary)
                self.assertIsNot(result[0], result[1])
                self.assertFalse(np.shares_memory(*result))
                for value, baseline in zip(result, expected):
                    np.testing.assert_allclose(value, baseline, atol=2e-12)
                    self.assertEqual(value.shape, (len(grid), 2))
                    self.assertEqual(value.dtype, baseline.dtype)

    def test_low_temperature_kms_and_ground_state_tail(self):
        taus = [0., .2, 500., 999.8, 1000.]
        _, imaginary = correlation.connected_autocorrelation_both_times(
            6, .2, [0., .001], [0., 1.], taus)
        for result in imaginary:
            self.assertTrue(np.all(np.isfinite(result)))
            np.testing.assert_allclose(result[:, 1], result[::-1, 1], atol=3e-12)
            self.assertLess(abs(result[-1, 0]), 1e-12)


class AutocorrelationScanTest(unittest.TestCase):
    def setUp(self):
        # Keep numerical tests small regardless of the production scan grids.
        temperatures = ", ".join(f"{i / 100:.2f}" for i in range(1, 101))
        self.template = f'''sizes = [4, 6]
ms = [0.0, 0.2]
temperatures = [{temperatures}]
real_times = [0.0, 0.1, 1.0, 10.0]
imaginary_times = [0.0, 0.1, 0.5, 1.0]
workers = 2
threads = 1
memory_gib = 8.0
output_dir = "data/autocorrelation"
'''

    def write_config(self, directory, source=None):
        path = Path(directory) / "scan.toml"
        path.write_text(self.template if source is None else source, encoding="utf-8")
        return path

    def test_config_validation_and_relative_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory)
            config, _ = scan.load_config(path)
            self.assertEqual(len(config["temperatures"]), 100)
            self.assertEqual(config["output_dir"], Path(directory).resolve() / "data/autocorrelation")
            invalid = [
                ("sizes = [4, 6]", "sizes = [4, 4]"),
                ("sizes = [4, 6]", "sizes = [4.0]"),
                ("sizes = [4, 6]", "sizes = [33]"),
                ("ms = [0.0, 0.2]", "ms = [0.0, -0.0]"),
                ("ms = [0.0, 0.2]", "ms = [inf]"),
                ("ms = [0.0, 0.2]", "ms = [true]"),
                ("0.01, 0.02", "-0.01, 0.02"),
                ("0.01, 0.02", "nan, 0.02"),
                ("real_times = [0.0, 0.1, 1.0, 10.0]", "real_times = []"),
                ("imaginary_times = [0.0, 0.1, 0.5, 1.0]", "imaginary_times = [inf]"),
                ("workers = 2", "workers = 0"),
                ("threads = 1", "threads = true"),
                ("memory_gib = 8.0", "memory_gib = 0"),
                ("memory_gib = 8.0", "memory_gib = true"),
                ("memory_gib = 8.0", "memory_gib = inf"),
                ('output_dir = "data/autocorrelation"', 'output_dir = ""'),
                ("sizes =", "size ="),
            ]
            for old, new in invalid:
                with self.subTest(new=new), self.assertRaises(ValueError):
                    scan.load_config(self.write_config(directory, self.template.replace(old, new)))
            valid = self.template.replace("0.01, 0.02", "0.0, inf")
            config, _ = scan.load_config(self.write_config(directory, valid))
            self.assertEqual(config["temperatures"][:2], [0., np.inf])

    def test_task_granularity_and_thread_limits(self):
        def completed(*args):
            future = Future()
            future.set_result(Path("result.npz"))
            return future

        with tempfile.TemporaryDirectory() as directory:
            config_path = self.write_config(directory)
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), \
                 patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                 patch.object(scan, "ProcessPoolExecutor") as executor, redirect_stdout(io.StringIO()):
                executor.return_value.__enter__.return_value.submit.side_effect = completed
                scan.main()
                calls = executor.return_value.__enter__.return_value.submit.call_args_list
                self.assertEqual([(call.args[1], call.args[3]) for call in calls],
                                 [(6, 0.), (6, .2), (4, 0.), (4, .2)])
                self.assertTrue(all(len(call.args[4]) == 100 for call in calls))
                self.assertEqual(executor.call_args.kwargs["max_workers"], 2)
                self.assertEqual(executor.call_args.kwargs["mp_context"].get_start_method(), "spawn")
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
                    self.assertEqual(os.environ[name], "1")
            config_path = self.write_config(directory, self.template.replace("threads = 1", "threads = 2"))
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), \
                 patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                 patch.object(scan, "ProcessPoolExecutor") as executor, redirect_stdout(io.StringIO()), \
                 patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                scan.main()
            self.assertEqual(error.exception.code, 2)
            executor.assert_not_called()

    def test_invalid_slurm_cpu_count_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self.write_config(directory)
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "invalid"}, clear=True), \
                 patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                 patch.object(scan, "ProcessPoolExecutor") as executor, \
                 patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                 self.assertRaises(SystemExit) as error:
                scan.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid", stderr.getvalue())
            executor.assert_not_called()
            self.assertFalse((Path(directory) / "data").exists())

    def test_config_defaults_bom_and_missing_keys(self):
        source = "\n".join(line for line in self.template.splitlines()
                           if not line.startswith(("workers =", "threads =", "output_dir =")))
        with tempfile.TemporaryDirectory() as directory:
            config, saved_source = scan.load_config(self.write_config(directory, "\ufeff" + source))
            self.assertEqual(saved_source, source)
            self.assertEqual((config["workers"], config["threads"]), (1, 1))
            self.assertEqual(config["output_dir"], Path(directory).resolve() / "data/autocorrelation")
            missing = "\n".join(line for line in source.splitlines()
                                if not line.startswith(("ms =", "real_times =")))
            with self.assertRaisesRegex(ValueError, "Missing config keys: ms, real_times"):
                scan.load_config(self.write_config(directory, missing))

    def test_range_tables_expand_with_inclusive_endpoints(self):
        source = '''sizes = [4, 6]
ms = { start = 0.0, stop = 1.0, length = 51 }
temperatures = { start = 0.0, stop = 10.0, length = 101, scale = "linear" }
real_times = { start = -1, stop = 4, length = 201, scale = "log10" }
imaginary_times = { start = -1, stop = 4, length = 201, scale = "log10" }
'''
        with tempfile.TemporaryDirectory() as directory:
            config, saved_source = scan.load_config(self.write_config(directory, source))
        self.assertEqual(saved_source, source)
        for key, expected in (
            ("ms", np.linspace(0, 1, 51)),
            ("temperatures", np.linspace(0, 10, 101)),
            ("real_times", np.logspace(-1, 4, 201)),
            ("imaginary_times", np.logspace(-1, 4, 201)),
        ):
            self.assertEqual(len(config[key]), len(expected))
            self.assertEqual((config[key][0], config[key][-1]), (expected[0], expected[-1]))
            np.testing.assert_allclose(config[key], expected, rtol=3e-15, atol=1e-15)
        self.assertEqual(scan.config_grid(dict(start=2, stop=2, length=1), "ms"), [2])
        self.assertEqual(scan.config_grid(dict(start=1, stop=0, length=3), "ms"), [1., .5, 0.])
        self.assertEqual(scan.config_grid(dict(start=-1e308, stop=1e308, length=3), "ms"),
                         [-1e308, 0., 1e308])

    def test_invalid_range_tables_are_rejected(self):
        invalid = [
            '{ start = 0, stop = 1 }',
            '{ start = 0, stop = 1, length = 3, step = 0.5 }',
            '{ start = 0, stop = 1, length = 0 }',
            '{ start = 0, stop = 1, length = true }',
            '{ start = 0, stop = 1, length = 3.0 }',
            '{ start = true, stop = 1, length = 3 }',
            '{ start = 0, stop = inf, length = 3 }',
            '{ start = nan, stop = 1, length = 3 }',
            '{ start = 0, stop = 1, length = 3, scale = "log" }',
            '{ start = 0, stop = 400, length = 3, scale = "log10" }',
            '{ start = -400, stop = 0, length = 3, scale = "log10" }',
            '{ start = -1, stop = 1, length = 3 }',
        ]
        old = next(line for line in self.template.splitlines() if line.startswith("temperatures ="))
        with tempfile.TemporaryDirectory() as directory:
            for table in invalid:
                with self.subTest(table=table), self.assertRaises(ValueError):
                    source = self.template.replace(old, "temperatures = " + table)
                    scan.load_config(self.write_config(directory, source))

    def test_six_rank_partition_covers_every_point_once_and_balances_sizes(self):
        sizes, ms = list(range(4, 24, 2)), [i / 100 for i in range(100)]
        assigned = []
        counts = []
        for rank in range(6):
            with patch.dict(os.environ, {"SLURM_NTASKS": "6", "SLURM_PROCID": str(rank)}, clear=True):
                actual_rank, ranks, points = scan.slurm_tasks(sizes, ms)
            self.assertEqual((actual_rank, ranks), (rank, 6))
            self.assertEqual([p[0] for p in points], sorted([p[0] for p in points], reverse=True))
            counts.append({L: sum(point[0] == L for point in points) for L in sizes})
            assigned.extend(points)
        self.assertEqual(len(assigned), 1000)
        self.assertEqual(set(assigned), {(L, index, m) for L in sizes for index, m in enumerate(ms)})
        for L in sizes:
            self.assertLessEqual(max(c[L] for c in counts) - min(c[L] for c in counts), 1)

    def test_memory_sector_counts_match_quspin(self):
        for L in range(2, 11):
            actual = scan.momentum_dimensions(L)
            expected = [correlation.pxp_basis_1d(L, kblock=k).Ns for k in range(L)]
            self.assertEqual(actual, expected)
        self.assertEqual(sum(scan.momentum_dimensions(30)), 1860498)

    def test_memory_budget_respects_slurm_and_explicit_limit(self):
        cases = [
            ({}, {"memory_gib": 8}, 8),
            ({"SLURM_MEM_PER_NODE": "4096"}, {"memory_gib": 8}, 4),
            ({"SLURM_MEM_PER_NODE": "16384"}, {"memory_gib": 8}, 8),
            ({"SLURM_MEM_PER_CPU": "2048", "SLURM_CPUS_PER_TASK": "4"}, {}, 8),
            ({"SLURM_MEM_PER_NODE": "0"}, {"memory_gib": 8}, 8),
        ]
        for environment, config, expected in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(scan.memory_budget(config), expected * 2**30)
        for environment in ({}, {"SLURM_MEM_PER_NODE": "0"}, {"SLURM_MEM_PER_NODE": "-1"},
                            {"SLURM_MEM_PER_CPU": "1000"}, {"SLURM_NTASKS_PER_NODE": "2"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    scan.memory_budget({})

    def test_memory_check_counts_concurrent_points_and_rejects_before_pool(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            config_path = self.write_config(directory)
            config, _ = scan.load_config(config_path)
            config["memory_gib"] = 1.25
            points = [(4, 0, 0.), (4, 1, .2)]
            with redirect_stdout(io.StringIO()):
                scan.check_memory(config, points, 1)
                with self.assertRaisesRegex(ValueError, "exceeds budget"):
                    scan.check_memory(config, points, 2)
            self.assertGreater(scan.estimate_point_memory(30, config), 1024 * 2**30)
            config_path = self.write_config(directory, self.template.replace("sizes = [4, 6]", "sizes = [30]"))
            with patch.object(sys, "argv", ["scan", "--config", str(config_path)]), \
                 patch.object(scan, "ProcessPoolExecutor") as executor, redirect_stdout(io.StringIO()), \
                 patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit):
                scan.main()
            self.assertIn("exceeds budget", stderr.getvalue())
            executor.assert_not_called()
            self.assertFalse(config["output_dir"].exists())
        with patch.dict(os.environ, {}, clear=True):
            scan.check_memory({}, [], 0)

    def test_saved_times_include_zero_and_infinite_temperatures(self):
        temperatures, times = [0., 1., np.inf], [.1, 1e4]
        with tempfile.TemporaryDirectory() as directory:
            path = scan.run_point(4, 0, .2, temperatures, [0., .1], times, directory)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["imaginary_times_input"], times)
                np.testing.assert_allclose(saved["imaginary_times"], [[.1, 1e-5, 0.], [1e4, 1., 0.]], atol=1e-16)
                for column, T in enumerate(temperatures):
                    expected = correlation.connected_autocorrelation(4, .2, T, times, True)
                    for name, baseline in zip(("average", "local"), expected):
                        np.testing.assert_allclose(saved[f"{name}_imaginary_time"][:, column], baseline[:, 0], atol=3e-12)

    def test_bad_slurm_environment_and_run_id_fail_before_pool(self):
        for environment in (
            {"SLURM_NTASKS": "6"},
            {"SLURM_NTASKS": "0"},
            {"SLURM_NTASKS": "6", "SLURM_PROCID": "6"},
            {"SLURM_NTASKS": "6", "SLURM_PROCID": "-1"},
            {"SLURM_NTASKS": "invalid"},
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    scan.slurm_tasks([4], [0.])
        with tempfile.TemporaryDirectory() as directory:
            config_path = self.write_config(directory)
            for extra in ([], ["--run-id", "../escape"], ["--run-id", ".."]):
                with self.subTest(extra=extra), \
                     patch.dict(os.environ, {"SLURM_NTASKS": "6", "SLURM_PROCID": "0"}, clear=True), \
                     patch.object(sys, "argv", ["scan", "--config", str(config_path), *extra]), \
                     patch.object(scan, "ProcessPoolExecutor") as executor, \
                     patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                    scan.main()
                self.assertEqual(error.exception.code, 2)
                executor.assert_not_called()
            self.assertFalse((Path(directory) / "data").exists())

    def test_six_simulated_ranks_share_run_without_duplicate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.template.replace("workers = 2", "workers = 1")
            config_path = self.write_config(directory, source)
            command = [sys.executable, scan.__file__, "--config", str(config_path), "--run-id", "six-ranks"]
            base_env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
            processes = []
            try:
                for rank in range(6):
                    processes.append(subprocess.Popen(
                        command, env=dict(base_env, SLURM_NTASKS="6", SLURM_PROCID=str(rank),
                                          SLURM_CPUS_PER_TASK="1"),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    ))
                for rank, process in enumerate(processes):
                    stdout, stderr = process.communicate(timeout=180)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
                    if rank >= 4:
                        self.assertIn("No assigned points", stdout)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            config, _ = scan.load_config(config_path)
            run = config["output_dir"] / "six-ranks"
            paths = list(run.glob("rank_*/*.npz"))
            self.assertEqual(len(paths), 4)
            self.assertEqual(len({path.name for path in paths}), 4)
            self.assertEqual(list(run.rglob("*.tmp")), [])
            for rank in range(6):
                self.assertEqual((run / f"rank_{rank:04d}" / "config.toml").read_text(encoding="utf-8"), source)
            for path in paths:
                with np.load(path, allow_pickle=False) as saved:
                    L, m = int(saved["L"]), float(saved["m"])
                    real, imaginary = correlation.connected_autocorrelation_both_times(
                        L, m, config["temperatures"], config["real_times"], config["imaginary_times"])
                    for name, real_values, imaginary_values in zip(("average", "local"), real, imaginary):
                        np.testing.assert_allclose(saved[f"{name}_real_time_real"], real_values.real, atol=3e-12)
                        np.testing.assert_allclose(saved[f"{name}_real_time_imag"], real_values.imag, atol=3e-12)
                        np.testing.assert_allclose(saved[f"{name}_imaginary_time"], imaginary_values, atol=3e-12)
            before = {path: path.read_bytes() for path in paths}
            retry = subprocess.run(command, env=dict(base_env, SLURM_NTASKS="6", SLURM_PROCID="0"),
                                   capture_output=True, text=True, timeout=30)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("FileExistsError", retry.stderr)
            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_spawned_results_match_serial_for_all_temperatures(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.template.replace("threads = 1", "threads = 2")
            source = source.replace("ms = [0.0, 0.2]",
                                    'ms = { start = 0.0, stop = 0.2, length = 2, scale = "linear" }')
            config_path = self.write_config(directory, source)
            process = subprocess.run(
                [sys.executable, scan.__file__, "--config", str(config_path)],
                env=dict(os.environ, SLURM_CPUS_PER_TASK="4"),
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            config, _ = scan.load_config(config_path)
            runs = list(config["output_dir"].iterdir())
            self.assertEqual(len(runs), 1)
            self.assertEqual((runs[0] / "config.toml").read_text(encoding="utf-8"), source)
            paths = list(runs[0].glob("*.npz"))
            self.assertEqual(len(paths), 4)
            self.assertEqual(list(runs[0].glob("*.tmp")), [])
            seen = set()
            for path in paths:
                with np.load(path, allow_pickle=False) as saved:
                    L, m = int(saved["L"]), float(saved["m"])
                    seen.add((L, m))
                    self.assertEqual(m, config["ms"][int(saved["m_index"])])
                    for name in ("temperatures", "real_times"):
                        np.testing.assert_array_equal(saved[name], config[name])
                    np.testing.assert_array_equal(saved["imaginary_times_input"], config["imaginary_times"])
                    fractions = np.array(config["imaginary_times"]) / max(config["imaginary_times"])
                    np.testing.assert_array_equal(saved["imaginary_times"],
                                                  fractions[:, None] / np.array(config["temperatures"]))
                    for imaginary, grid in ((False, "real_times"), (True, "imaginary_times")):
                        if imaginary:
                            columns = [correlation.connected_autocorrelation(
                                L, m, T, config[grid], True) for T in config["temperatures"]]
                            expected = tuple(np.concatenate([column[i] for column in columns], axis=1)
                                             for i in (0, 1))
                        else:
                            expected = correlation.connected_autocorrelation(
                                L, m, config["temperatures"], config[grid], False)
                        for observable, baseline in zip(("average", "local"), expected):
                            if imaginary:
                                actual = saved[f"{observable}_imaginary_time"]
                            else:
                                actual = (saved[f"{observable}_real_time_real"]
                                          + 1j * saved[f"{observable}_real_time_imag"])
                            np.testing.assert_allclose(actual, baseline, atol=3e-12)
                            self.assertEqual(actual.shape, (len(config[grid]), 100))
                    for name in saved.files:
                        self.assertFalse(np.iscomplexobj(saved[name]))
            self.assertEqual(seen, {(L, m) for L in config["sizes"] for m in config["ms"]})


if __name__ == "__main__":
    unittest.main()
