"""Small-chain checks: python -m unittest pxp_model.tests.test_pxp_scan"""

import tempfile
import unittest
from concurrent.futures import Future
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from .. import pxp_basis, pxp_scan
from ..pxp_basis import pxp_basis_1d, resolve_degeneracies
from ..pxp_spectrum import build_hamiltonian, solve
from .test_pxp_spectrum import explicit_pxp
import numpy as np
from scipy.linalg import expm


class PXPScanTest(unittest.TestCase):
    def test_default_parallel_sizes(self):
        def completed(*args):
            future = Future()
            future.set_result("unused.npz")
            return future
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "56"}), \
             patch.object(sys, "argv", ["pxp_scan.py"]), \
             patch.object(pxp_scan, "ProcessPoolExecutor") as executor:
            pool = executor.return_value.__enter__.return_value
            pool.submit.side_effect = completed
            pxp_scan.main()
            self.assertEqual([call.args[1] for call in pool.submit.call_args_list], list(range(16, 29, 2)))
            self.assertEqual(executor.call_args.kwargs["max_workers"], 7)
            self.assertEqual(executor.call_args.kwargs["mp_context"].get_start_method(), "spawn")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
                self.assertEqual(os.environ[name], "8")

    def test_spawned_scan_matches_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, pxp_scan.__file__, "--L", "4", "6", "--workers", "2",
                 "--threads", "2", "--output-dir", directory],
                env=dict(os.environ, SLURM_CPUS_PER_TASK="4"),
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 2)
            for L in (4, 6):
                with np.load(Path(directory) / f"pxp_scan_L={L}.npz", allow_pickle=False) as saved:
                    for i, result in enumerate(pxp_scan.scan(L)):
                        for key in ("energies", "matched_scar_indices", "tracked_scar_indices",
                                    "fsa_energies", "fsa_overlaps", "z1_expectations", "overlaps", "entropies", "z1_t"):
                            np.testing.assert_allclose(saved[key][i], result[key], atol=1e-8, rtol=1e-10)

    def test_tracking_and_reuse(self):
        L = 12
        times = np.array([0., 0.1])
        baseline = solve(L, 0, times)
        basis = pxp_basis_1d(L, kblock=0, pblock=1)
        construct = type(basis).construct_fsa_eigensystem
        with patch.object(pxp_scan, "pxp_basis_1d", wraps=pxp_basis_1d) as make_basis, \
             patch.object(pxp_basis, "pxp_basis_1d", wraps=pxp_basis_1d) as extra_basis, \
             patch.object(pxp_basis, "build_forward_operator", wraps=pxp_basis.build_forward_operator) as forward, \
             patch.object(pxp_scan, "resolve_degeneracies", wraps=resolve_degeneracies) as rotate, \
             patch.object(type(basis), "construct_fsa_eigensystem", autospec=True, side_effect=construct) as fsa:
            results = list(pxp_scan.scan(L, times))
        self.assertEqual(make_basis.call_count, 2)
        extra_basis.assert_not_called()
        forward.assert_called_once()
        self.assertEqual(rotate.call_count, 11)
        self.assertEqual(fsa.call_count, 11)
        np.testing.assert_array_equal([call.kwargs["m"] for call in fsa.call_args_list], np.linspace(0, 1, 11))
        for key in ("energies", "overlaps", "entropies", "fsa_energies", "fsa_overlaps"):
            np.testing.assert_allclose(results[0][key], baseline[key], atol=1e-12)
        z2 = np.zeros(basis.Ns)
        z2[basis.index("10" * (L // 2))] = 1
        for call in rotate.call_args_list:
            np.testing.assert_array_equal(call.args[2], z2)
        previous = None
        for result in results:
            E, U = build_hamiltonian(basis, result["m"]).eigh()
            resolve_degeneracies(E, U, z2)
            if previous is None:
                indices = baseline["scar_indices"]
            else:
                indices = np.argmax(abs(U.T @ previous)**2, axis=0)
            np.testing.assert_array_equal(result["tracked_scar_indices"], indices)
            np.testing.assert_allclose(result["energies"], E, atol=1e-12)
            np.testing.assert_allclose(result["overlaps"], abs(z2 @ U)**2, atol=1e-12)
            single = solve(L, result["m"], times)
            np.testing.assert_array_equal(np.unique(result["matched_scar_indices"]), single["scar_indices"])
            for key in ("fsa_energies", "fsa_overlaps"):
                np.testing.assert_allclose(result[key], single[key], atol=1e-12)
            previous = U[:, indices]

    def test_fsa_matching_preserves_order_and_duplicates(self):
        basis = pxp_basis_1d(6, kblock=0, pblock=1)
        E, U = build_hamiltonian(basis, 0).eigh()
        z2_index = basis.index("10" * 3)
        z2 = np.eye(basis.Ns)[:, z2_index]
        resolve_degeneracies(E, U, z2)
        # Exclude low-weight projections; keep repeated matches in FSA order.
        psi = np.column_stack([
            0.9 * U[:, 2], U[:, 0], 0.1 * U[:, 1],
            0.8 * U[:, 2], U[:, 1], 0.7 * U[:, 1], np.zeros(basis.Ns),
        ])
        with patch.object(type(basis), "construct_fsa_eigensystem",
                          return_value=dict(psi=psi, energies=np.arange(7.))):
            result = next(pxp_scan.scan(6, [0.]))
        np.testing.assert_array_equal(result["matched_scar_indices"], [2, 0, 2, 1])
        np.testing.assert_array_equal(result["tracked_scar_indices"], [0, 1, 2])
        np.testing.assert_array_equal(result["fsa_energies"], [0, 1, 3, 4])
        np.testing.assert_allclose(result["fsa_overlaps"], abs(U[z2_index, [2, 0, 2, 1]])**2, atol=1e-12)

    def test_local_observables_against_full_spin_space(self):
        L = 8
        times = np.array([0., 0.1, 0.7])
        full = pxp_basis_1d(L)
        basis = pxp_basis_1d(L, kblock=0, pblock=1)
        P = basis.pxp_get_proj(full).toarray()
        z1 = 2 * ((np.arange(2**L) >> (L - 1)) & 1) - 1
        initial = np.zeros(2**L)
        initial[int("10" * (L // 2), 2)] = 1 / np.sqrt(2)
        initial[int("01" * (L // 2), 2)] = 1 / np.sqrt(2)
        for result in pxp_scan.scan(L, times):
            H = explicit_pxp(L, result["m"])
            E, U = build_hamiltonian(basis, result["m"]).eigh()
            resolve_degeneracies(E, U, P.T @ initial[full.states])
            states = np.zeros((2**L, basis.Ns))
            states[full.states] = P @ U
            np.testing.assert_allclose(H @ states, states * E, atol=1e-12)
            np.testing.assert_allclose(result["z1_expectations"], z1 @ abs(states)**2, atol=1e-12)
            entropy = []
            for state in states.T:
                probabilities = np.linalg.svd(state.reshape(2**(L // 2), -1), compute_uv=False)**2
                probabilities = probabilities[probabilities > 1e-30]
                entropy.append(-np.sum(probabilities * np.log(probabilities)))
            np.testing.assert_allclose(result["entropies"], entropy, atol=1e-12)
            expected = [z1 @ abs(expm(-1j * H * t) @ initial)**2 for t in times]
            np.testing.assert_allclose(result["z1_t"], expected, atol=1e-12)

    def test_default_grid_and_npz(self):
        results = list(pxp_scan.scan(4))
        result = results[0]
        np.testing.assert_array_equal(result["times"], np.r_[0., np.geomspace(0.1, 1e6, 1001)])
        self.assertAlmostEqual(result["z1_t"][0], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = pxp_scan.save_results(iter(results), directory)
            self.assertEqual(path.name, "pxp_scan_L=4.npz")
            self.assertEqual(list(path.parent.glob("*.npz")), [path])
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["L"], 4)
                np.testing.assert_array_equal(saved["times"], result["times"])
                np.testing.assert_array_equal(saved["ms"], np.linspace(0, 1, 11))
                for i, point in enumerate(results):
                    for key in ("energies", "matched_scar_indices", "tracked_scar_indices",
                                "fsa_energies", "fsa_overlaps", "z1_expectations", "overlaps", "entropies", "z1_t"):
                        np.testing.assert_array_equal(saved[key][i], point[key])
                self.assertEqual((saved["kblock"], saved["parity"]), (0, 1))
                self.assertEqual(saved["fsa_weight_cutoff"], 0.5)
            with self.assertRaises(ValueError):
                pxp_scan.save_results([], directory)
            with self.assertRaises(ValueError):
                pxp_scan.save_results([result, dict(result, times=np.array([0.]))], directory)
        for L in (3, 5, 34, 6.5):
            with self.assertRaises(ValueError):
                next(pxp_scan.scan(L))
        for times in ([], [-1], [0, 0], [1, 0], [np.nan], [[0]]):
            with self.assertRaises(ValueError):
                next(pxp_scan.scan(4, times))


if __name__ == "__main__":
    unittest.main()
