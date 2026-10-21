"""Projection checks: python -m unittest pxp_model.test_pxp_projection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .pxp_basis import pxp_basis_1d
import numpy as np
from scipy.sparse import isspmatrix_csc


class PXPProjectionTest(unittest.TestCase):
    def test_symmetry_free_identity(self):
        basis = pxp_basis_1d(6)
        for full in (basis, pxp_basis_1d(6)):
            with patch.object(basis, "get_amp", side_effect=AssertionError("Unneeded expansion")):
                P = basis.pxp_get_proj(full)
                self.assertTrue(isspmatrix_csc(P))
                np.testing.assert_array_equal(P.toarray(), np.eye(basis.Ns))
                # Cached calls must not repeat the state comparison.
                with patch("numpy.array_equal", side_effect=AssertionError("Repeated comparison")):
                    self.assertIs(P, basis.pxp_get_proj(full))
                for shape in ((basis.Ns,), (basis.Ns, 2), (basis.Ns, basis.Ns)):
                    v = np.arange(np.prod(shape)).reshape(shape).astype(complex)
                    for pure in (False, True):
                        out = basis.pxp_project_from(v, full, enforce_pure=pure)
                        np.testing.assert_array_equal(out, v)
                        self.assertFalse(np.shares_memory(out, v))

    def test_symmetry_free_mismatch(self):
        basis = pxp_basis_1d(6)
        for states in (basis.states[::-1], basis.states[:-1]):
            full = SimpleNamespace(N=6, blocks={}, states=states, Ns=len(states))
            with self.assertRaisesRegex(ValueError, "identical states and ordering"):
                basis.pxp_get_proj(full)
        for full in (pxp_basis_1d(8), pxp_basis_1d(6, kblock=0)):
            with self.assertRaises(ValueError):
                basis.pxp_get_proj(full)

    def test_projector_and_state_shapes(self):
        full = pxp_basis_1d(8)
        rng = np.random.default_rng(42)
        for k, p in ((0, 1), (4, -1), (1, None)):
            with self.subTest(k=k, p=p):
                basis = pxp_basis_1d(8, kblock=k, pblock=p)
                P = basis.pxp_get_proj(full)
                self.assertTrue(isspmatrix_csc(P))
                self.assertIs(P, basis.pxp_get_proj(full))
                self.assertEqual(P.shape, (full.Ns, basis.Ns))
                np.testing.assert_allclose((P.conj().T @ P).toarray(),
                                           np.eye(basis.Ns), atol=1e-12)
                vectors = rng.normal(size=(basis.Ns, 2)) + 1j * rng.normal(size=(basis.Ns, 2))
                vectors /= np.linalg.norm(vectors, axis=0)
                for v in (vectors[:, 0], vectors):
                    np.testing.assert_allclose(basis.pxp_project_from(v, full), P @ v)
                # A pure density matrix must lift to the outer product of its lifted ket.
                v = vectors[:, 0]
                rho = np.outer(v, v.conj())
                lifted = P @ v
                np.testing.assert_allclose(basis.pxp_project_from(rho, full),
                                           np.outer(lifted, lifted.conj()), atol=1e-12)
                np.testing.assert_allclose(
                    basis.pxp_project_from(np.eye(basis.Ns), full, enforce_pure=True),
                    P.toarray())

    def test_invalid_inputs(self):
        full = pxp_basis_1d(6)
        basis = pxp_basis_1d(6, kblock=0)
        for target in (basis, pxp_basis_1d(8)):
            with self.assertRaises(ValueError):
                basis.pxp_get_proj(target)
        for v in (0, np.zeros(basis.Ns + 1), np.zeros((basis.Ns, 1, 1))):
            with self.assertRaises(ValueError):
                basis.pxp_project_from(v, full)


if __name__ == "__main__":
    unittest.main()
