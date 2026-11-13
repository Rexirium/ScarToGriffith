"""Small-system trace checks for projected connected autocorrelations.

    python -m unittest pxp_model.tests.test_pxp_autocorrelation
"""

import unittest
from unittest.mock import patch

from ..pxp_autocorrelation import (
    connected_autocorrelation, local_connected_autocorrelation,
    _lift_to_translation_basis,
)
from ..pxp_basis import pxp_basis_1d
from ..pxp_spectrum import build_hamiltonian
from .test_pxp_spectrum import explicit_pxp
import numpy as np
from scipy.linalg import eigh, expm


def projected_matrices(L, m):
    """Project explicit spin-space H and the single-site Pauli Z separately."""
    full = pxp_basis_1d(L)
    sector = pxp_basis_1d(L, kblock=0, pblock=1)
    P = sector.pxp_get_proj(full).toarray()
    H = explicit_pxp(L, m)[np.ix_(full.states, full.states)]
    z1 = 2 * ((full.states >> (L - 1)) & 1).astype(float) - 1
    return P.T @ H @ P, P.T @ (z1[:, None] * P)


class PXPAutocorrelationTest(unittest.TestCase):
    def test_against_matrix_exponentials(self):
        temperatures = [0, 0.3, 1.2, np.inf]
        times = np.array([0.4, 0, -0.2, 0.4])
        for L, m in ((2, 0.0), (5, -0.3), (6, 0.2)):
            H, A = projected_matrices(L, m)
            E, U = eigh(H)
            for imaginary in (False, True):
                with self.subTest(L=L, m=m, imaginary=imaginary):
                    actual = connected_autocorrelation(L, m, temperatures, times, imaginary)
                    expected = np.empty(actual.shape, dtype=complex)
                    for column, T in enumerate(temperatures):
                        if T == 0:
                            rho = np.outer(U[:, 0], U[:, 0].conj())
                        else:
                            rho = expm(-(H - E[0] * np.eye(len(E))) / T)
                            rho /= np.trace(rho)
                        for row, t in enumerate(times):
                            s = 1 if imaginary else 1j
                            evolved = expm(s * t * H) @ A @ expm(-s * t * H)
                            expected[row, column] = (
                                np.trace(rho @ evolved @ A)
                                - np.trace(rho @ evolved) * np.trace(rho @ A)
                            )
                    np.testing.assert_allclose(actual, expected, atol=2e-12)
                    self.assertEqual(np.iscomplexobj(actual), not imaginary)
                    # The projected A has A^2 != I, even at t=0.
                    self.assertGreater(abs(np.trace(A @ A) / len(E) - 1), 0.1)

    def test_low_temperature_imaginary_time_and_kms(self):
        beta = 1000.0
        times = np.array([0, 0.2, beta / 2, beta - 0.2, beta])
        actual = connected_autocorrelation(6, 0.2, 1 / beta, times, imaginary=True)
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, actual[::-1], atol=2e-12)
        ground = connected_autocorrelation(6, 0.2, 0, [0, 0.2, 1e4], imaginary=True)
        np.testing.assert_allclose(actual[:2], ground[:2], atol=2e-12)
        self.assertAlmostEqual(ground[-1, 0], 0, places=12)
        tiny = connected_autocorrelation(6, 0.2, np.nextafter(0., 1.),
                                         [0, 0.2], imaginary=True)
        np.testing.assert_allclose(tiny, ground[:2], atol=2e-12)

    def test_scalar_temperature_and_real_time_conjugation(self):
        times = [-1, 0, 1]
        actual = connected_autocorrelation(6, 0.2, 0.5, times)
        self.assertEqual(actual.shape, (3, 1))
        np.testing.assert_allclose(actual[0], actual[2].conj(), atol=1e-12)
        self.assertGreater(abs(actual[2, 0].imag), 1e-4)
        multiple = connected_autocorrelation(6, 0.2, [0.5, 0, 0.5], times)
        np.testing.assert_allclose(multiple[:, [0, 2]], np.repeat(actual, 2, axis=1))
        self.assertEqual(connected_autocorrelation(6, 0.2, [0, 1], []).shape, (0, 2))

    def test_invalid_inputs_fail_before_building_basis(self):
        invalid = [dict(L=0), dict(L=3.5), dict(L=True), dict(L=33),
                   dict(m=np.nan), dict(m=[0.2]), dict(m=1j),
                   dict(Ts=-1), dict(Ts=[]), dict(Ts=[[1]]),
                   dict(Ts=np.nan), dict(Ts=1j), dict(ts=1),
                   dict(ts=[np.inf]), dict(ts=[[0]]), dict(ts=[1j]),
                   dict(imaginary="False")]
        with patch("pxp_model.pxp_autocorrelation.pxp_basis_1d") as construct:
            for function in (connected_autocorrelation, local_connected_autocorrelation):
                for overrides in invalid:
                    args = dict(L=6, m=0.2, Ts=1, ts=[0])
                    args.update(overrides)
                    with self.subTest(function=function.__name__, overrides=overrides):
                        with self.assertRaises(ValueError):
                            function(**args)
            construct.assert_not_called()


class PXPLocalAutocorrelationTest(unittest.TestCase):
    def test_local_against_full_space_matrix_exponentials(self):
        temperatures = [0, 0.3, 1.2, np.inf]
        times = np.array([0.4, 0, -0.2, 0.4])
        for L, m in ((2, 0.0), (5, -0.3), (6, 0.2)):
            full = pxp_basis_1d(L)
            sector = pxp_basis_1d(L, kblock=0, pblock=1)
            P = sector.pxp_get_proj(full).toarray()
            H = explicit_pxp(L, m)[np.ix_(full.states, full.states)]
            H0 = P.T @ H @ P
            E, U = eigh(H0)
            z = np.array([[2 * ((int(s) >> j) & 1) - 1 for s in full.states]
                          for j in range(L)])
            for imaginary in (False, True):
                with self.subTest(L=L, m=m, imaginary=imaginary):
                    actual = local_connected_autocorrelation(L, m, temperatures, times,
                                                            imaginary=imaginary)
                    expected = np.zeros(actual.shape, complex)
                    for column, T in enumerate(temperatures):
                        if T == 0:
                            rho0 = np.outer(U[:, 0], U[:, 0].conj())
                        else:
                            rho0 = expm(-(H0 - E[0] * np.eye(len(E))) / T)
                            rho0 /= np.trace(rho0)
                        rho = P @ rho0 @ P.T
                        for row, t in enumerate(times):
                            s = 1 if imaginary else 1j
                            left, right = expm(s * t * H), expm(-s * t * H)
                            for diagonal in z:
                                A = np.diag(diagonal)
                                evolved = left @ A @ right
                                expected[row, column] += (
                                    np.trace(rho @ evolved @ A)
                                    - np.trace(rho @ evolved) * np.trace(rho @ A)
                                ) / L
                        mean = np.trace(rho @ np.diag(z[0]))
                        self.assertAlmostEqual(actual[1, column], 1 - mean**2, places=12)
                    np.testing.assert_allclose(actual, expected, atol=3e-12)
                    self.assertEqual(np.iscomplexobj(actual), not imaginary)
                    projected = connected_autocorrelation(L, m, temperatures, [0], imaginary)
                    self.assertTrue(np.all(actual[1].real > projected[0].real + 0.1))

    def test_sector_shift_and_parity_embedding_against_full_projectors(self):
        # L=10 has k=0 states of both parities, so the embedding is nontrivial.
        L = 10
        full = pxp_basis_1d(L)
        sector = pxp_basis_1d(L, kblock=0, pblock=1)
        source = pxp_basis_1d(L, kblock=0)
        self.assertGreater(source.Ns, sector.Ns)
        vectors = np.eye(sector.Ns)
        lifted = _lift_to_translation_basis(sector, source, vectors)
        P0 = source.pxp_get_proj(full).toarray()
        Psector = sector.pxp_get_proj(full).toarray()
        np.testing.assert_allclose(P0 @ lifted, Psector, atol=1e-12)
        np.testing.assert_allclose(lifted.conj().T @ lifted, vectors, atol=1e-12)
        z = np.array([[2 * ((int(state) >> (L - j - 1)) & 1) - 1
                       for j in range(L)] for state in full.states])
        H = explicit_pxp(L, 0.2)[np.ix_(full.states, full.states)]
        for q in (1, 2, L // 2, L - 1):
            with self.subTest(q=q):
                target = pxp_basis_1d(L, kblock=q)
                # QuSpin's full-spin projector is an independent complex-sector oracle.
                Pq = target.get_proj(np.complex128).toarray()[2**L - 1 - full.states]
                phases = np.exp(2j * np.pi * q * np.arange(L) / L) / L
                ops = [["z", [j], phases[j]] for j in range(L)]
                shifted = target.Op_shift_sector(source, ops, lifted, dtype=np.complex128)
                explicit = (z @ phases)[:, None] * Psector
                np.testing.assert_allclose(Pq @ shifted, explicit, atol=1e-12)
                Hq = build_hamiltonian(target, 0.2, dtype=np.complex128).toarray()
                np.testing.assert_allclose(Hq, Pq.conj().T @ H @ Pq, atol=1e-12)
                np.testing.assert_allclose(Hq, Hq.conj().T, atol=1e-12)

    def test_local_shapes_conjugation_and_zero_temperature(self):
        real = local_connected_autocorrelation(6, 0.2, 0.5, [-1, 0, 1])
        self.assertEqual(real.shape, (3, 1))
        np.testing.assert_allclose(real[0], real[2].conj(), atol=1e-12)
        multiple = local_connected_autocorrelation(6, 0.2, [0.5, 0, 0.5], [-1, 0, 1])
        np.testing.assert_allclose(multiple[:, [0, 2]], np.repeat(real, 2, axis=1))
        self.assertEqual(local_connected_autocorrelation(6, 0.2, [0, 1], []).shape, (0, 2))
        ground = local_connected_autocorrelation(6, 0.2, 0, [0, 0.2, 1e4], imaginary=True)
        self.assertAlmostEqual(ground[-1, 0], 0, places=12)
        tiny = local_connected_autocorrelation(6, 0.2, np.nextafter(0., 1.),
                                               [0, 0.2], imaginary=True)
        np.testing.assert_allclose(tiny, ground[:2], atol=1e-12)
        # A sector-restricted Gibbs state has no local-operator KMS symmetry.
        beta = 1000.
        thermal = local_connected_autocorrelation(6, 0.2, 1 / beta, [0, beta], imaginary=True)
        self.assertTrue(np.all(np.isfinite(thermal)))
        self.assertGreater(abs(thermal[0, 0] - thermal[1, 0]), 0.1)


if __name__ == "__main__":
    unittest.main()
