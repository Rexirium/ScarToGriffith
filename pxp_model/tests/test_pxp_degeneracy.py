"""Reference alignment checks: python -m unittest pxp_model.tests.test_pxp_degeneracy."""

import unittest

import numpy as np

from ..pxp_basis import pxp_basis_1d, resolve_degeneracies
from ..pxp_spectrum import build_hamiltonian


class PXPDegeneracyTest(unittest.TestCase):
    def test_real_and_complex_eigenvectors(self):
        basis = pxp_basis_1d(12, kblock=0, pblock=1)
        H = build_hamiltonian(basis, 0)
        energies, original = H.eigh()
        zero = np.flatnonzero(abs(energies) < 1e-10)
        self.assertGreater(zero.size, 1)
        z2 = basis.index("10" * 6)
        reference = np.eye(basis.Ns)[z2]
        rng = np.random.default_rng(42)
        for complex_vectors in (False, True):
            with self.subTest(complex_vectors=complex_vectors):
                vectors = original.astype(complex if complex_vectors else float)
                if complex_vectors:
                    rotation, _ = np.linalg.qr(
                        rng.normal(size=(zero.size, zero.size))
                        + 1j * rng.normal(size=(zero.size, zero.size)))
                    vectors[:, zero] = vectors[:, zero] @ rotation
                before = vectors.copy()
                expected = before[:, zero] @ before[z2, zero].conj()
                weight = np.linalg.norm(expected)**2
                blocks = resolve_degeneracies(energies, vectors, reference)
                self.assertIn(slice(zero[0], zero[-1] + 1), blocks)
                np.testing.assert_allclose(vectors[:, zero[0]], expected / np.sqrt(weight),
                                           atol=1e-12)
                np.testing.assert_allclose(vectors[z2, zero[1:]], 0, atol=1e-12)
                self.assertAlmostEqual(abs(vectors[z2, zero[0]])**2, weight)
                np.testing.assert_allclose(vectors.conj().T @ vectors,
                                           np.eye(basis.Ns), atol=1e-12)
                np.testing.assert_allclose(H.dot(vectors), vectors * energies, atol=1e-12)
                for block in blocks:
                    np.testing.assert_allclose(
                        vectors[:, block] @ vectors[:, block].conj().T,
                        before[:, block] @ before[:, block].conj().T, atol=1e-12)
                untouched = np.ones(basis.Ns, dtype=bool)
                for block in blocks:
                    untouched[block] = False
                np.testing.assert_array_equal(vectors[:, untouched], before[:, untouched])

    def test_nondegenerate_spectrum_is_unchanged(self):
        basis = pxp_basis_1d(12, kblock=0, pblock=1)
        energies, vectors = build_hamiltonian(basis, 0.2).eigh()
        before = vectors.copy()
        reference = np.eye(basis.Ns)[basis.index("10" * 6)]
        self.assertEqual(resolve_degeneracies(energies, vectors, reference), [])
        np.testing.assert_array_equal(vectors, before)

    def test_zero_overlap_and_full_block_width(self):
        basis = pxp_basis_1d(6)
        z2 = basis.index("10" * 3)
        # Degenerate vectors orthogonal to Z2 must not be changed.
        rows = [i for i in range(basis.Ns) if i != z2][:3]
        vectors = np.eye(basis.Ns)[:, rows]
        before = vectors.copy()
        reference = np.eye(basis.Ns)[z2]
        blocks = resolve_degeneracies([0., 0.75e-6, 1.5e-6], vectors, reference, atol=1e-6)
        self.assertEqual(blocks, [slice(0, 2)])
        np.testing.assert_array_equal(vectors, before)

    def test_arbitrary_complex_reference_and_multiple_blocks(self):
        rng = np.random.default_rng(7)
        original, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
        reference = rng.normal(size=5) + 1j * rng.normal(size=5)
        reference_before = reference.copy()
        energies = np.array([-1., -1., 0., 2., 2.])
        vectors = original.copy()
        blocks = resolve_degeneracies(energies, vectors, reference)
        self.assertEqual(blocks, [slice(0, 2), slice(3, 5)])
        for block in blocks:
            projection = original[:, block] @ (original[:, block].conj().T @ reference)
            np.testing.assert_allclose(vectors[:, block.start],
                                       projection / np.linalg.norm(projection), atol=1e-12)
            np.testing.assert_allclose(reference.conj() @ vectors[:, block.start+1:block.stop],
                                       0, atol=1e-12)
        np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(5), atol=1e-12)
        H = (original * energies) @ original.conj().T
        np.testing.assert_allclose(H @ vectors, vectors * energies, atol=1e-12)
        np.testing.assert_array_equal(vectors[:, 2], original[:, 2])
        np.testing.assert_array_equal(reference, reference_before)
        scaled = original.copy()
        resolve_degeneracies(energies, scaled, 5 * reference)
        np.testing.assert_allclose(scaled, vectors, atol=1e-12)

    def test_complex_reference_requires_complex_storage(self):
        vectors = np.eye(2)
        before = vectors.copy()
        with self.assertRaisesRegex(ValueError, "requires complex eigenvectors"):
            resolve_degeneracies([0., 0.], vectors, [1., 1j])
        np.testing.assert_array_equal(vectors, before)


if __name__ == "__main__":
    unittest.main()
