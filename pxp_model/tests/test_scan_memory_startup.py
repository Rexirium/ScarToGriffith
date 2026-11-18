"""Memory startup checks without importing numerical dependencies."""

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


if __name__ == "__main__":
    unittest.main()
