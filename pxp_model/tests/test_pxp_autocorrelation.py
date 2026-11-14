"""Small-system trace checks for full-ensemble connected autocorrelations.

    python -m unittest pxp_model.tests.test_pxp_autocorrelation
"""

import unittest
from unittest.mock import patch

from ..pxp_autocorrelation import (
    connected_autocorrelation, average_connected_autocorrelation, local_connected_autocorrelation,
    _momentum_eigensystems, _thermal_log_weights, _uniform_connected,
)
from ..pxp_basis import pxp_basis_1d
from ..pxp_spectrum import build_hamiltonian
from .test_pxp_spectrum import explicit_pxp
import numpy as np
from scipy.linalg import eigh, expm


def full_matrices(L, m):
    """Restrict an explicit spin-space H only by the blockade constraint."""
    full = pxp_basis_1d(L)
    H = explicit_pxp(L, m)[np.ix_(full.states, full.states)]
    z = np.array([[2 * ((int(s) >> j) & 1) - 1 for s in full.states]
                  for j in range(L)])
    return full, H, z


class PXPAutocorrelationTest(unittest.TestCase):
    def test_against_matrix_exponentials(self):
        temperatures = [0, 0.3, 1.2, np.inf]
        times = np.array([0.4, 0, -0.2, 0.4])
        for L, m in ((2, 0.0), (3, 0.2), (5, -0.3), (6, 0.2)):
            _, H, z = full_matrices(L, m)
            A = np.diag(z.mean(axis=0))
            E, U = eigh(H)
            for imaginary in (False, True):
                with self.subTest(L=L, m=m, imaginary=imaginary):
                    actual = average_connected_autocorrelation(L, m, temperatures, times, imaginary)
                    expected = np.empty(actual.shape, dtype=complex)
                    for column, T in enumerate(temperatures):
                        if T == 0:
                            rho = np.outer(U[:, 0], U[:, 0].conj())
                        else:
                            rho = expm(-(H - E[0] * np.eye(len(E))) / T)
                            rho /= np.trace(rho)
                        for row, t in enumerate(times):
                            if imaginary and T > 0:
                                t = t / times.max() / T
                            s = 1 if imaginary else 1j
                            evolved = expm(s * t * H) @ A @ expm(-s * t * H)
                            expected[row, column] = (
                                np.trace(rho @ evolved @ A)
                                - np.trace(rho @ evolved) * np.trace(rho @ A)
                            )
                    np.testing.assert_allclose(actual, expected, atol=2e-12)
                    self.assertEqual(np.iscomplexobj(actual), not imaginary)
                    # Average magnetization has A^2 != I, even at t=0.
                    self.assertGreater(abs(np.trace(A @ A) / len(E) - 1), 0.1)

    def test_low_temperature_imaginary_time_and_kms(self):
        beta = 1000.0
        times = np.array([0, 0.2, beta / 2, beta - 0.2, beta])
        actual = average_connected_autocorrelation(6, 0.2, 1 / beta, times, imaginary=True)
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, actual[::-1], atol=2e-12)
        ground = average_connected_autocorrelation(6, 0.2, 0, [0, 0.2, 1e4], imaginary=True)
        np.testing.assert_allclose(actual[:2], ground[:2], atol=2e-12)
        self.assertAlmostEqual(ground[-1, 0], 0, places=12)
        with self.assertRaisesRegex(ValueError, "normalized imaginary times must be finite"):
            average_connected_autocorrelation(6, 0.2, np.nextafter(0., 1.),
                                              [0, 0.2], imaginary=True)

    def test_scalar_temperature_and_real_time_conjugation(self):
        times = [-1, 0, 1]
        actual = average_connected_autocorrelation(6, 0.2, 0.5, times)
        self.assertEqual(actual.shape, (3, 1))
        np.testing.assert_allclose(actual[0], actual[2].conj(), atol=1e-12)
        self.assertGreater(abs(actual[2, 0].imag), 1e-4)
        multiple = average_connected_autocorrelation(6, 0.2, [0.5, 0, 0.5], times)
        np.testing.assert_allclose(multiple[:, [0, 2]], np.repeat(actual, 2, axis=1))
        self.assertEqual(average_connected_autocorrelation(6, 0.2, [0, 1], []).shape, (0, 2))

    def test_global_log_weights_unsorted_and_degenerate(self):
        energies = np.array([2., -3., 0., -3., 1.])
        temperatures = np.array([0, np.inf, .5])
        logs = _thermal_log_weights(energies, temperatures)
        weights = np.exp(logs)
        np.testing.assert_allclose(weights[:, 0], [0, .5, 0, .5, 0])
        np.testing.assert_allclose(weights[:, 1], np.full(5, .2))
        expected = np.exp(-(energies + 3) / .5)
        np.testing.assert_allclose(weights[:, 2], expected / expected.sum())
        np.testing.assert_allclose(weights.sum(axis=0), 1)
        np.testing.assert_allclose(_thermal_log_weights(energies + 7, temperatures), logs)

    def test_full_infinite_temperature_trace_and_global_subtraction(self):
        # L=6 full trace: <M>=-4/9, <M^2>=7/27; a single-sector trace differs.
        uniform = average_connected_autocorrelation(6, .2, np.inf, [0])
        local = local_connected_autocorrelation(6, .2, np.inf, [0])
        self.assertAlmostEqual(uniform[0, 0], 5 / 81, places=12)
        self.assertAlmostEqual(local[0, 0], 65 / 81, places=12)

    def test_ed_reuse_and_empty_times(self):
        for function in (average_connected_autocorrelation, local_connected_autocorrelation):
            with patch("pxp_model.pxp_autocorrelation.build_hamiltonian",
                       wraps=build_hamiltonian) as build:
                function(6, .2, [0, .5, 1., np.inf], [0, .1, .3])
                # Only k=0,1,2,3 need ED, independently of the number of T/t pairs.
                self.assertEqual(build.call_count, 4)
            with patch("pxp_model.pxp_autocorrelation.pxp_basis_1d") as construct:
                self.assertEqual(function(6, .2, [0, 1], []).shape, (0, 2))
                construct.assert_not_called()

    def test_centered_diagonal_variance_preserves_tiny_intersector_fluctuations(self):
        # Two one-state sectors: variance is tiny compared with <M>^2 ~ 1.
        delta = 2.**-28
        temperatures = np.array([0., .5, 1., np.inf])
        logs = _thermal_log_weights(np.array([0., 1.]), temperatures)
        sectors = [(None, np.array([energy]), np.ones((1, 1))) for energy in (0., 1.)]
        p = np.exp(logs[1])
        expected = p * (1 - p) * (2 * delta)**2
        for imaginary in (False, True):
            times = np.array([0., 2.])
            if imaginary:
                times = np.broadcast_to(times[:, None], (times.size, temperatures.size))
            with patch("pxp_model.pxp_autocorrelation._magnetization_in_eigenbasis",
                       side_effect=[np.array([[1 - delta]]), np.array([[1 + delta]])]):
                actual = _uniform_connected(sectors, [logs[:1], logs[1:]],
                                            times, imaginary)
            np.testing.assert_allclose(actual, np.tile(expected, (2, 1)), rtol=2e-15, atol=0)

    def test_zero_temperature_imaginary_tail_below_machine_epsilon(self):
        # Exact L=2,m=0 gaps: sqrt(2) and 2*sqrt(2). No trace subtraction.
        times = np.array([10., 20., 30.])
        uniform_expected = .25 * np.exp(-2 * np.sqrt(2) * times)
        local_expected = uniform_expected + .5 * np.exp(-np.sqrt(2) * times)
        for function, expected in ((average_connected_autocorrelation, uniform_expected),
                                   (local_connected_autocorrelation, local_expected)):
            actual = function(2, 0., 0., times, imaginary=True)[:, 0]
            self.assertTrue(np.all(actual > 0))
            self.assertLess(expected[-1], np.finfo(float).eps)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)

    def test_combined_results_share_computation_but_not_output_storage(self):
        times = [0., .2, 1.]
        for imaginary in (False, True):
            for temperatures in (.5, [0., .5, np.inf]):
                with self.subTest(imaginary=imaginary, temperatures=temperatures):
                    with patch("pxp_model.pxp_autocorrelation.build_hamiltonian",
                               wraps=build_hamiltonian) as build, patch(
                            "pxp_model.pxp_autocorrelation._uniform_connected",
                            wraps=_uniform_connected) as uniform:
                        result = connected_autocorrelation(6, .2, temperatures, times, imaginary)
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 2)
                        average, local = result
                        self.assertEqual(build.call_count, 4)
                        uniform.assert_called_once()
                    expected_average = average_connected_autocorrelation(
                        6, .2, temperatures, times, imaginary)
                    expected_local = local_connected_autocorrelation(
                        6, .2, temperatures, times, imaginary)
                    np.testing.assert_allclose(average, expected_average, atol=1e-12)
                    np.testing.assert_allclose(local, expected_local, atol=1e-12)
                    self.assertEqual(average.shape, (3, np.size(temperatures)))
                    self.assertEqual(local.shape, average.shape)
                    self.assertEqual(np.iscomplexobj(average), not imaginary)
                    self.assertEqual(local.dtype, average.dtype)
                    self.assertFalse(np.shares_memory(average, local))
                    local[:] = 0
                    np.testing.assert_array_equal(average, expected_average)

    def test_combined_empty_times_skip_ed(self):
        with patch("pxp_model.pxp_autocorrelation.pxp_basis_1d") as construct:
            for imaginary in (False, True):
                average, local = connected_autocorrelation(6, .2, [0, 1], [], imaginary)
                self.assertEqual(average.shape, (0, 2))
                self.assertEqual(local.shape, (0, 2))
                self.assertIsNot(average, local)
                self.assertEqual(np.iscomplexobj(average), not imaginary)
                self.assertEqual(local.dtype, average.dtype)
            construct.assert_not_called()

    def test_invalid_inputs_fail_before_building_basis(self):
        invalid = [dict(L=0), dict(L=3.5), dict(L=True), dict(L=33),
                   dict(m=np.nan), dict(m=[0.2]), dict(m=1j),
                   dict(Ts=-1), dict(Ts=[]), dict(Ts=[[1]]),
                   dict(Ts=np.nan), dict(Ts=1j), dict(ts=1),
                   dict(ts=[np.inf]), dict(ts=[[0]]), dict(ts=[1j]),
                   dict(imaginary="False")]
        with patch("pxp_model.pxp_autocorrelation.pxp_basis_1d") as construct:
            for function in (connected_autocorrelation, average_connected_autocorrelation,
                             local_connected_autocorrelation):
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
        for L, m in ((2, 0.0), (3, 0.2), (5, -0.3), (6, 0.2)):
            _, H, z = full_matrices(L, m)
            E, U = eigh(H)
            for imaginary in (False, True):
                with self.subTest(L=L, m=m, imaginary=imaginary):
                    actual = local_connected_autocorrelation(L, m, temperatures, times,
                                                            imaginary=imaginary)
                    expected = np.zeros(actual.shape, complex)
                    for column, T in enumerate(temperatures):
                        if T == 0:
                            rho = np.outer(U[:, 0], U[:, 0].conj())
                        else:
                            rho = expm(-(H - E[0] * np.eye(len(E))) / T)
                            rho /= np.trace(rho)
                        for row, t in enumerate(times):
                            if imaginary and T > 0:
                                t = t / times.max() / T
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
                    uniform = average_connected_autocorrelation(L, m, temperatures, [0], imaginary)
                    self.assertTrue(np.all(actual[1].real > uniform[0].real + 0.1))

    def test_momentum_spectra_and_nonzero_source_sector_shifts(self):
        for L in (5, 10):
            full, H, z = full_matrices(L, .2)
            sectors = _momentum_eigensystems(L, .2)
            self.assertEqual(sum(basis.Ns for basis, _, _ in sectors), full.Ns)
            all_E = np.concatenate([E for _, E, _ in sectors])
            np.testing.assert_allclose(np.sort(all_E), eigh(H, eigvals_only=True), atol=2e-12)
            projectors = []
            for basis, E, U in sectors:
                P = basis.get_proj(np.complex128).toarray()[2**L - 1 - full.states]
                projectors.append(P)
                Hk = build_hamiltonian(basis, .2, dtype=np.complex128).toarray()
                np.testing.assert_allclose(Hk, P.conj().T @ H @ P, atol=1e-12)
                np.testing.assert_allclose(Hk @ U, U * E, atol=2e-12)
                np.testing.assert_allclose(U.conj().T @ U, np.eye(basis.Ns), atol=1e-12)
            # Check general initial k and wraparound, not just k=0 -> q.
            for k, target_k in ((0, 1), (1, 3), (L - 1, 1)):
                source, _, U = sectors[k]
                target, _, V = sectors[target_k]
                q = (target_k - k) % L
                phases = np.exp(2j * np.pi * q * np.arange(L) / L) / L
                ops = [["z", [j], phases[j]] for j in range(L)]
                forward = target.Op_shift_sector(source, ops, U, dtype=np.complex128)
                forward = forward.reshape(target.Ns, source.Ns)
                # Callback site 0 is the MSB, opposite to z's bit order.
                explicit = (z[::-1].T @ phases)[:, None] * (projectors[k] @ U)
                np.testing.assert_allclose(projectors[target_k] @ forward, explicit, atol=1e-12)
                reverse_ops = [["z", [j], phases[j].conjugate()] for j in range(L)]
                reverse = source.Op_shift_sector(target, reverse_ops, V, dtype=np.complex128)
                reverse = reverse.reshape(source.Ns, target.Ns)
                np.testing.assert_allclose(V.conj().T @ forward,
                                           (U.conj().T @ reverse).conj().T, atol=1e-12)

    def test_local_shapes_conjugation_and_zero_temperature(self):
        real = local_connected_autocorrelation(6, 0.2, 0.5, [-1, 0, 1])
        self.assertEqual(real.shape, (3, 1))
        np.testing.assert_allclose(real[0], real[2].conj(), atol=1e-12)
        multiple = local_connected_autocorrelation(6, 0.2, [0.5, 0, 0.5], [-1, 0, 1])
        np.testing.assert_allclose(multiple[:, [0, 2]], np.repeat(real, 2, axis=1))
        self.assertEqual(local_connected_autocorrelation(6, 0.2, [0, 1], []).shape, (0, 2))
        ground = local_connected_autocorrelation(6, 0.2, 0, [0, 0.2, 1e4], imaginary=True)
        self.assertAlmostEqual(ground[-1, 0], 0, places=12)
        with self.assertRaisesRegex(ValueError, "normalized imaginary times must be finite"):
            local_connected_autocorrelation(6, 0.2, np.nextafter(0., 1.),
                                             [0, 0.2], imaginary=True)
        # Full Gibbs weights restore local-operator KMS symmetry.
        for beta in (.5, 2., 1000.):
            times = [0, .1, beta / 2, beta - .1, beta]
            thermal = local_connected_autocorrelation(6, .2, 1 / beta, times, imaginary=True)
            self.assertTrue(np.all(np.isfinite(thermal)))
            np.testing.assert_allclose(thermal, thermal[::-1], atol=3e-12)
        mixed = local_connected_autocorrelation(6, .2, [0, 1e-3], [0, .2], imaginary=True)
        np.testing.assert_allclose(mixed[:, 0], ground[:2, 0], atol=1e-12)
        self.assertAlmostEqual(mixed[0, 1], mixed[1, 1], places=12)  # tau=0 and beta


if __name__ == "__main__":
    unittest.main()
