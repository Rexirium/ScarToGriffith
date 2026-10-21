"""Independent small-chain checks: python -m unittest pxp_model.test_pxp_fsa."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .pxp_fsa import build_forward_operator, construct_fsa_eigensystem
from .pxp_basis import pxp_basis_1d, fibonacci
import numpy as np


class PXPFSATest(unittest.TestCase):
    def test_against_explicit_spin_flips(self):
        for L in (4, 6, 8, 10):
            with self.subTest(L=L):
                basis = pxp_basis_1d(L)
                with patch.object(basis, "pxp_get_proj", side_effect=AssertionError("Unneeded projection")):
                    result = basis.construct_fsa_eigensystem()
                states = basis.states
                V, beta = result["fsa_basis"], result["beta"]
                dim = V.shape[1]
                z2 = int("10" * (L // 2), 2)
                distance = np.array([(int(s) ^ z2).bit_count() for s in states])
                rows = {int(s): i for i, s in enumerate(states)}
                H = np.zeros((len(states), len(states)))
                Hp = np.zeros_like(H)
                for col, s in enumerate(states):
                    s = int(s)
                    for bit in range(L):
                        if (s >> ((bit - 1) % L) & 1 or
                                s >> ((bit + 1) % L) & 1):
                            continue
                        row = rows[s ^ (1 << bit)]
                        H[row, col] = 1
                        if distance[row] == distance[col] + 1:
                            Hp[row, col] = 1
                np.testing.assert_allclose(
                    build_forward_operator(basis).toarray(), Hp)
                self.assertEqual(len(states), fibonacci(L - 1) + fibonacci(L + 1))
                for n in range(dim):
                    np.testing.assert_allclose(V[distance != n, n], 0)
                np.testing.assert_allclose(V.T @ V, np.eye(dim), atol=1e-12)
                np.testing.assert_allclose(Hp @ V[:, :-1], V[:, 1:] * beta, atol=1e-12)
                np.testing.assert_allclose(Hp @ V[:, -1], 0)
                self.assertAlmostEqual(V[rows[z2], 0], 1)
                self.assertEqual(dim, L + 1)
                self.assertAlmostEqual(V[rows[z2 ^ ((1 << L) - 1)], -1], 1)
                self.assertAlmostEqual(beta[0], np.sqrt(L / 2))
                np.testing.assert_allclose(V.T @ H @ V, result["H_fsa"], atol=1e-12)
                U, E = result["eigenvectors"], result["energies"]
                np.testing.assert_allclose(result["H_fsa"] @ U, U * E, atol=1e-12)
                np.testing.assert_allclose(E, -E[::-1], atol=1e-12)
                psi = V @ U
                np.testing.assert_allclose(result["psi"], psi)
                np.testing.assert_allclose(psi.T @ psi, np.eye(dim), atol=1e-12)
                np.testing.assert_allclose(result["initial_overlaps"], np.abs(psi[rows[z2]])**2)
                self.assertAlmostEqual(result["initial_overlaps"].sum(), 1)
                self.assertEqual(set(result), {"fsa_basis", "beta", "H_fsa", "energies",
                                               "eigenvectors", "initial_overlaps", "psi"})

    def test_invalid_size(self):
        for L in (0, 2, 5, 34, 6.0):
            with self.subTest(L=L), self.assertRaises(ValueError):
                construct_fsa_eigensystem(SimpleNamespace(N=L))

    def test_symmetry_projection(self):
        full = pxp_basis_1d(8)
        original = full.construct_fsa_eigensystem(normalized=False)
        for blocks in (dict(kblock=0, pblock=1), dict(kblock=4, pblock=-1),
                       dict(kblock=1), dict(pblock=-1), dict(a=2, kblock=0)):
            with self.subTest(blocks=blocks):
                basis = pxp_basis_1d(8, **blocks)
                with patch.object(basis, "pxp_get_proj", wraps=basis.pxp_get_proj) as get_proj:
                    result = basis.construct_fsa_eigensystem(normalized=False)
                    get_proj.assert_called_once()
                P = basis.pxp_get_proj(full)
                np.testing.assert_allclose(result["fsa_basis"], P.conj().T @ original["fsa_basis"], atol=1e-12)
                np.testing.assert_allclose(result["psi"], P.conj().T @ original["psi"], atol=1e-12)
                np.testing.assert_allclose(result["psi"], result["fsa_basis"] @ result["eigenvectors"])
                for key in ("beta", "H_fsa", "energies", "eigenvectors", "initial_overlaps"):
                    np.testing.assert_allclose(result[key], original[key])
                if blocks == dict(kblock=1):
                    np.testing.assert_allclose(result["psi"], 0, atol=1e-12)
                auxiliary = basis._pxp_fsa_full_basis
                cached_P = basis.pxp_get_proj(auxiliary)
                with patch("pxp_model.pxp_fsa.pxp_basis_1d", side_effect=AssertionError("Rebuilt basis")):
                    basis.construct_fsa_eigensystem()
                self.assertIs(basis._pxp_fsa_full_basis, auxiliary)
                self.assertIs(basis.pxp_get_proj(auxiliary), cached_P)

    def test_normalized_states(self):
        for blocks in ({}, dict(kblock=0, pblock=1), dict(kblock=4, pblock=-1),
                       dict(kblock=1)):
            with self.subTest(blocks=blocks):
                basis = pxp_basis_1d(8, **blocks)
                raw = basis.construct_fsa_eigensystem(normalized=False)
                normalized = basis.construct_fsa_eigensystem()
                norms = np.linalg.norm(raw["psi"], axis=0)
                nonzero = norms > 1e-12
                psi = normalized["psi"]
                self.assertEqual(psi.shape, raw["psi"].shape)
                self.assertTrue(np.all(np.isfinite(psi)))
                np.testing.assert_allclose(np.linalg.norm(psi[:, nonzero], axis=0), 1, atol=1e-12)
                np.testing.assert_allclose(psi[:, nonzero] * norms[nonzero],
                                           raw["psi"][:, nonzero], atol=1e-12)
                np.testing.assert_array_equal(psi[:, ~nonzero], 0)
                for key in raw.keys() - {"psi"}:
                    np.testing.assert_array_equal(normalized[key], raw[key])
                if blocks == dict(kblock=1):
                    self.assertFalse(nonzero.any())


if __name__ == "__main__":
    unittest.main()
