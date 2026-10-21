"""Independent small-chain checks: python -m unittest pxp_model.test_pxp_fsa."""

import unittest

from .pxp_fsa import build_forward_operator, construct_fsa_eigensystem
from .pxp_basis import pxp_basis_1d, fibonacci
import numpy as np


class PXPFSATest(unittest.TestCase):
    def test_against_explicit_spin_flips(self):
        cases = [(L, None) for L in (4, 6, 8, 10)]
        cases += [(8, "01010101"), (8, "00000000"), (8, "10001000"), (8, 1)]
        for L, initial_state in cases:
            with self.subTest(L=L, initial_state=initial_state):
                result = construct_fsa_eigensystem(L, initial_state)
                basis = pxp_basis_1d(L)
                states = basis.states
                V, beta = result["fsa_basis"], result["beta"]
                dim = V.shape[1]
                z2 = (int("10" * (L // 2), 2) if initial_state is None else
                      int(initial_state, 2) if isinstance(initial_state, str) else initial_state)
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
                    build_forward_operator(basis, initial_state).toarray(), Hp)
                self.assertEqual(len(states), fibonacci(L - 1) + fibonacci(L + 1))
                for n in range(dim):
                    np.testing.assert_allclose(V[distance != n, n], 0)
                np.testing.assert_allclose(V.T @ V, np.eye(dim), atol=1e-12)
                np.testing.assert_allclose(Hp @ V[:, :-1], V[:, 1:] * beta, atol=1e-12)
                np.testing.assert_allclose(Hp @ V[:, -1], 0)
                self.assertAlmostEqual(V[rows[z2], 0], 1)
                if initial_state is None:
                    self.assertEqual(dim, L + 1)
                    self.assertAlmostEqual(V[rows[z2 ^ ((1 << L) - 1)], -1], 1)
                    self.assertAlmostEqual(beta[0], np.sqrt(L / 2))
                if initial_state == "00000000":
                    self.assertEqual(dim, L // 2 + 1)
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
                construct_fsa_eigensystem(L)

    def test_invalid_initial_state(self):
        for state in ("1010", "0000000x", "11000000", "10000001", -1, 256, 1.5):
            with self.subTest(state=state), self.assertRaises(ValueError):
                construct_fsa_eigensystem(8, state)


if __name__ == "__main__":
    unittest.main()
