"""Basis checks: python -m unittest pxp_model.tests.test_pxp_basis."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.sparse import isspmatrix_csc

from .. import pxp_basis
from ..pxp_basis import (
    build_forward_operator, construct_fsa_eigensystem, pxp_basis_1d, resolve_degeneracies,
)
from ..pxp_spectrum import build_hamiltonian
from ..utils import fibonacci


class PXPProjectionTest(unittest.TestCase):
    def test_symmetry_free_identity(self):
        basis = pxp_basis_1d(6)
        for full in (basis, pxp_basis_1d(6)):
            P = basis.pxp_get_proj(full)
            self.assertTrue(isspmatrix_csc(P))
            np.testing.assert_array_equal(P.toarray(), np.eye(basis.Ns))
            for shape in ((basis.Ns,), (basis.Ns, 2), (basis.Ns, basis.Ns)):
                v = np.arange(np.prod(shape)).reshape(shape).astype(complex)
                out = basis.pxp_project_from(v, full, enforce_pure=True)
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
        for k, p in ((0, 1), (4, -1), (1, None), (2, None), (3, None)):
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


class PXPFSATest(unittest.TestCase):
    def test_reuse_across_fields_and_normalization(self):
        for blocks in ({}, dict(kblock=0, pblock=1), dict(kblock=1)):
            with self.subTest(blocks=blocks):
                basis = pxp_basis_1d(8, **blocks)
                first = basis.construct_fsa_eigensystem(m=0.3)
                with patch.object(pxp_basis, "build_forward_operator") as forward, \
                     patch.object(pxp_basis, "pxp_basis_1d") as make_basis:
                    raw = basis.construct_fsa_eigensystem(m=-0.2)
                    normalized = basis.construct_fsa_eigensystem(m=-0.2, normalized=True)
                    zero = basis.construct_fsa_eigensystem(m=0)
                forward.assert_not_called()
                make_basis.assert_not_called()
                self.assertIs(raw["fsa_basis"], first["fsa_basis"])
                self.assertIs(raw["beta"], first["beta"])
                self.assertFalse(raw["fsa_basis"].flags.writeable)
                self.assertFalse(raw["beta"].flags.writeable)
                fresh = pxp_basis_1d(8, **blocks).construct_fsa_eigensystem(m=-0.2)
                for key in raw:
                    np.testing.assert_allclose(raw[key], fresh[key], atol=1e-12)
                norms = np.linalg.norm(raw["psi"], axis=0)
                expected = raw["psi"].copy()
                nonzero = norms > 1e-12
                expected[:, nonzero] /= norms[nonzero]
                expected[:, ~nonzero] = 0
                np.testing.assert_allclose(normalized["psi"], expected, atol=1e-12)
                np.testing.assert_array_equal(np.diag(zero["H_fsa"]), 0)
                # Mutating per-call outputs must not poison later cached solves.
                raw["H_fsa"][:] = 0
                raw["psi"][:] = 0
                again = basis.construct_fsa_eigensystem(m=-0.2)
                np.testing.assert_allclose(again["H_fsa"], fresh["H_fsa"], atol=1e-12)
                np.testing.assert_allclose(again["psi"], fresh["psi"], atol=1e-12)

    def test_against_explicit_spin_flips(self):
        for L in (4, 6, 8, 10):
            with self.subTest(L=L):
                basis = pxp_basis_1d(L)
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
                magnetization = np.array([
                    sum(1 if int(s) & (1 << bit) else -1 for bit in range(L))
                    for s in states
                ])
                for m in (0, -0.2, 0.2):
                    detuned = basis.construct_fsa_eigensystem(m=m)
                    np.testing.assert_array_equal(detuned["fsa_basis"], V)
                    np.testing.assert_allclose(
                        detuned["H_fsa"], V.T @ (H - m * np.diag(magnetization)) @ V,
                        atol=1e-12,
                    )
                    U, E = detuned["eigenvectors"], detuned["energies"]
                    np.testing.assert_allclose(detuned["H_fsa"] @ U, U * E, atol=1e-12)
                    self.assertAlmostEqual(detuned["H_fsa"][1, 1], 2 * m)

    def test_invalid_field(self):
        for m in (np.nan, np.inf, -np.inf):
            with self.subTest(m=m), self.assertRaises(ValueError):
                construct_fsa_eigensystem(SimpleNamespace(N=8), m=m)

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
                result = basis.construct_fsa_eigensystem(normalized=False)
                P = basis.pxp_get_proj(full)
                np.testing.assert_allclose(result["fsa_basis"], P.conj().T @ original["fsa_basis"], atol=1e-12)
                np.testing.assert_allclose(result["psi"], P.conj().T @ original["psi"], atol=1e-12)
                np.testing.assert_allclose(result["psi"], result["fsa_basis"] @ result["eigenvectors"])
                for key in ("beta", "H_fsa", "energies", "eigenvectors", "initial_overlaps"):
                    np.testing.assert_allclose(result[key], original[key])
                if blocks == dict(kblock=1):
                    np.testing.assert_allclose(result["psi"], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
