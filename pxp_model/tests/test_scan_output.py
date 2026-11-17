"""Shared-run initialization checks without numerical dependencies."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from .. import pxp_autocorrelation_scan as scan


class ScanOutputTest(unittest.TestCase):
    def test_ranks_share_one_config_and_no_rank_directories(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            run = Path(directory) / "run"
            source = "sizes = [4, 6]\n"
            scan.prepare_output(run, source, 0, 4)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(scan.prepare_output, run, source, rank, 4) for rank in (1, 2, 3)]
                for future in futures:
                    future.result(timeout=5)
            self.assertEqual(list(run.rglob("config.toml")), [run / "config.toml"])
            self.assertEqual((run / "config.toml").read_text(encoding="utf-8"), source)
            self.assertFalse(any(p.is_dir() for p in run.iterdir()))
            self.assertEqual(len(list(run.glob(".rank_*"))), 4)
            before = {p: p.read_bytes() for p in run.iterdir()}
            for rank in range(4):
                with self.subTest(rank=rank), self.assertRaises(FileExistsError):
                    scan.prepare_output(run, source, rank, 4)
                self.assertEqual(before, {p: p.read_bytes() for p in run.iterdir()})

    def test_nonzero_rank_waits_for_rank_zero(self):
        waiting, ready = Event(), Event()

        def wait_for_owner(_):
            waiting.set()
            if not ready.wait(timeout=5):
                raise TimeoutError("Test owner did not initialize")

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch.object(scan, "sleep", side_effect=wait_for_owner), ThreadPoolExecutor(max_workers=1) as pool:
            run = Path(directory) / "run"
            source = "sizes = [4]\n"
            future = pool.submit(scan.prepare_output, run, source, 1, 2)
            try:
                self.assertTrue(waiting.wait(timeout=5))
                self.assertFalse(run.exists())
                scan.prepare_output(run, source, 0, 2)
            finally:
                ready.set()
            future.result(timeout=5)
            self.assertTrue((run / ".rank_0001").is_file())
            self.assertEqual((run / "config.toml").read_text(encoding="utf-8"), source)

    def test_mismatched_config_or_job_is_rejected_without_changes(self):
        environment = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            run = Path(directory) / "run"
            source = "sizes = [4]\n"
            scan.prepare_output(run, source, 0, 2)
            before = {p: p.read_bytes() for p in run.iterdir()}
            for candidate, ranks, overrides in (
                ("sizes = [6]\n", 2, {}), (source, 3, {}),
                (source, 2, {"SLURM_JOB_ID": "456"}),
                (source, 2, {"SLURM_STEP_ID": "1"}),
            ):
                with self.subTest(source=candidate, ranks=ranks, overrides=overrides), \
                     patch.dict(os.environ, overrides), self.assertRaises(ValueError):
                    scan.prepare_output(run, candidate, 1, ranks)
                self.assertEqual(before, {p: p.read_bytes() for p in run.iterdir()})

    def test_duplicate_rank_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            run = Path(directory) / "run"
            scan.prepare_output(run, "", 0, 2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(scan.prepare_output, run, "", 1, 2) for _ in range(2)]
                errors = [future.exception(timeout=5) for future in futures]
            self.assertEqual(sum(error is None for error in errors), 1)
            self.assertEqual(sum(isinstance(error, FileExistsError) for error in errors), 1)

    def test_missing_owner_times_out_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(scan, "_OUTPUT_WAIT_SECONDS", 0):
            run = Path(directory) / "run"
            with self.assertRaisesRegex(TimeoutError, "rank 0"):
                scan.prepare_output(run, "", 1, 2)
            self.assertFalse(run.exists())

    def test_failed_config_write_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with patch.object(Path, "write_text", side_effect=OSError("write failed")), \
                 self.assertRaisesRegex(OSError, "write failed"):
                scan.prepare_output(run, "sizes = [4]\n", 0, 2)
            self.assertFalse((run / "config.toml").exists())
            self.assertEqual(list(run.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
