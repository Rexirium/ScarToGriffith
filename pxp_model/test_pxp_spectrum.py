"""Small-chain checks against an explicit 2**L Pauli-space Hamiltonian.

    python -m unittest pxp_model.test_pxp_spectrum
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .pxp_spectrum import build_hamiltonian, spectrum_spacings, save_results, solve
from .pxp_basis import pxp_basis_1d
import numpy as np
from scipy.linalg import eigh


def explicit_pxp(L, m):
    """Independent full spin space, including frozen blockade-violating states."""
    H = np.zeros((2**L, 2**L))
    for state in range(2**L):
        H[state, state] = -m * (2 * state.bit_count() - L)
        for site in range(L):
            left, right = (site - 1) % L, (site + 1) % L
            if not ((state >> left) & 1 or (state >> right) & 1):
                H[state ^ (1 << site), state] = 1
    return H


class PXPSpectrumTest(unittest.TestCase):
    def test_block_and_symmetric_state_dynamics(self):
        times = np.linspace(0, 4, 17)
        for L, m in ((6, 0.0), (8, 0.2), (8, -0.37)):
            with self.subTest(L=L, m=m):
                full = pxp_basis_1d(L)
                dense_spin = explicit_pxp(L, m)
                H = dense_spin[np.ix_(full.states, full.states)]
                # No coupling from the blockade subspace to forbidden states.
                forbidden = np.setdiff1d(np.arange(2**L), full.states)
                np.testing.assert_allclose(dense_spin[np.ix_(forbidden, full.states)], 0)
                np.testing.assert_allclose(build_hamiltonian(full, m).toarray(), H)
                row_by_state = {int(s): i for i, s in enumerate(full.states)}
                translation_rows = [
                    row_by_state[((int(s) << 1) & (2**L - 1)) | (int(s) >> (L - 1))]
                    for s in full.states
                ]
                reflection_rows = [
                    row_by_state[int(f"{s:0{L}b}"[::-1], 2)] for s in full.states
                ]
                basis = pxp_basis_1d(L, a=1, kblock=0, pblock=1)
                P = basis.pxp_project_from(np.eye(basis.Ns), full, enforce_pure=True)
                reduced = build_hamiltonian(basis, m).toarray()
                np.testing.assert_allclose(P.T @ P, np.eye(basis.Ns), atol=1e-12)
                np.testing.assert_allclose(H @ P, P @ reduced, atol=1e-12)
                np.testing.assert_allclose(P[translation_rows], P, atol=1e-12)
                np.testing.assert_allclose(P[reflection_rows], P, atol=1e-12)

                E, U = eigh(H)
                initial = np.zeros(full.Ns)
                initial[full.index("10" * (L // 2))] = 1 / np.sqrt(2)
                initial[full.index("01" * (L // 2))] = 1 / np.sqrt(2)
                states = U @ ((U.conj().T @ initial)[:, None] * np.exp(-1j * E[:, None] * times))
                z = np.array([[2 * ((int(s) >> i) & 1) - 1 for i in range(L)]
                              for s in full.states])
                zz = np.mean(z * np.roll(z, 1, axis=1), axis=1)
                result = solve(L, m, times)
                np.testing.assert_allclose(result["fidelity"], np.abs(initial @ states)**2, atol=1e-11)
                np.testing.assert_allclose(result["zz"], zz @ np.abs(states)**2, atol=1e-11)
                self.assertAlmostEqual(result["overlaps"].sum(), 1)
                self.assertAlmostEqual(result["fidelity"][0], 1)
                self.assertAlmostEqual(result["zz"][0], -1)
                self.assertAlmostEqual(result["energies"] @ result["overlaps"], 0)

                # Compare matching in the reduced basis with full-space overlaps.
                _, reduced_U = build_hamiltonian(basis, m).eigh()
                full_fsa = full.construct_fsa_eigensystem(m=m)
                raw_fsa = full_fsa["psi"]
                norms = np.linalg.norm(P.T @ raw_fsa, axis=0)
                valid = norms**2 > 0.5
                self.assertEqual(np.count_nonzero(valid), L // 2 + 1)
                np.testing.assert_allclose(result["fsa_projection_weights"], norms**2, atol=1e-12)
                np.testing.assert_allclose(result["fsa_energies"], full_fsa["energies"][valid])
                projected_fsa = P @ (P.T @ raw_fsa[:, valid]) / norms[valid]
                np.testing.assert_allclose(
                    result["fsa_overlaps"], np.abs(initial.conj() @ projected_fsa)**2,
                    atol=1e-12,
                )
                expected = np.abs(
                    (P @ reduced_U).T @ raw_fsa[:, valid] / norms[valid]
                )**2
                np.testing.assert_array_equal(
                    result["scar_indices"], np.unique(np.argmax(expected, axis=0)),
                )

    def test_filter_before_normalization_and_repeated_matches(self):
        basis = pxp_basis_1d(6, kblock=0, pblock=1)
        _, U = build_hamiltonian(basis, 0.2).eigh()
        psi = np.column_stack([
            0.9 * U[:, 0], U[:, 0], 0.1 * U[:, 2],
            0.8 * U[:, 1], U[:, 1], 0.7 * U[:, 2], np.zeros(basis.Ns),
        ])
        with patch.object(type(basis), "construct_fsa_eigensystem",
                          return_value=dict(psi=psi, energies=np.arange(7.))) as construct:
            result = solve(6, 0.2, np.array([0.]))
        construct.assert_called_once_with(normalized=False, m=0.2)
        np.testing.assert_allclose(result["fsa_projection_weights"], [.81, 1, .01, .64, 1, .49, 0])
        np.testing.assert_array_equal(result["scar_indices"], [0, 1])
        np.testing.assert_array_equal(result["fsa_energies"], [0, 1, 3, 4])

    def test_half_chain_entropy_against_full_tensor_product(self):
        L = 8
        full = pxp_basis_1d(L)
        basis = pxp_basis_1d(L, a=1, kblock=0, pblock=1)
        _, U = build_hamiltonian(basis, 0.2).eigh()
        for v in U.T:
            state = basis.pxp_project_from(v, full)
            spin_state = np.zeros(2**L, dtype=complex)
            spin_state[full.states] = state
            probabilities = np.linalg.svd(spin_state.reshape(2**(L // 2), -1),
                                         compute_uv=False)**2
            probabilities = probabilities[probabilities > 1e-15]
            self.assertAlmostEqual(full.my_ent_entropy(state),
                                   -probabilities @ np.log(probabilities))

    def test_unfolded_spacings(self):
        np.testing.assert_allclose(spectrum_spacings(np.arange(60.)), np.ones(40))
        energies = np.arange(60.)**3
        energies[30] = energies[29]
        indices = np.arange(10, 50)
        gaps = np.diff(energies)[indices]
        local_means = (energies[indices + 5] - energies[indices - 5]) / 10
        expected = (gaps / local_means)[gaps > 1e-8]
        expected /= expected.mean()
        np.testing.assert_allclose(spectrum_spacings(energies), expected)
        self.assertEqual(spectrum_spacings(np.zeros(60)).size, 0)
        self.assertEqual(spectrum_spacings(np.arange(8.)).size, 0)

    def test_saved_outputs(self):
        result = solve(6, 0.2, np.linspace(0, 1, 3))
        with tempfile.TemporaryDirectory() as directory:
            save_results(result, directory)
            png = Path(directory) / "pxp_spectrum_L=6_m=0.2.png"
            data = png.with_suffix(".npz")
            self.assertGreater(png.stat().st_size, 1000)
            with np.load(data, allow_pickle=False) as saved:
                self.assertNotIn("fsa_matches", saved.files)
                for key, value in result.items():
                    np.testing.assert_array_equal(saved[key], value)
                self.assertEqual(saved["fsa_weight_cutoff"], 0.5)
                self.assertEqual((saved["kblock"], saved["parity"]), (0, 1))
                self.assertEqual(saved["initial_state"], "Z2_plus")
                self.assertEqual(saved["unfolding_window"], 5)
                self.assertEqual(saved["spacing_etol"], 1e-8)


if __name__ == "__main__":
    unittest.main()
