"""Small-system trace checks for full-ensemble connected autocorrelations.

    python -m unittest pxp_model.tests.test_pxp_autocorrelation
"""

import unittest
from collections import Counter
from itertools import combinations
from unittest.mock import patch

from ..pxp_autocorrelation import (
    connected_autocorrelation, average_connected_autocorrelation, local_connected_autocorrelation,
    _momentum_eigensystems, _thermal_log_weights, _uniform_connected, _spectral_correlation,
    _eigenbasis_product, _magnetization_in_eigenbasis, _full_ensemble, _local_from_average,
    _momentum_z_shift, prepare_time_grid,
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


class SpectralBatchTest(unittest.TestCase):
    def test_batches_match_dense_sum_without_modifying_inputs(self):
        rng = np.random.default_rng(1729)
        energies = np.linspace(-1., 1., 517)
        intermediate = np.linspace(-.8, 1.2, 13)
        # A transposed, read-only input also covers the reverse-transition path.
        strengths = rng.random((13, 517)).T
        logs = _thermal_log_weights(
            np.concatenate((energies, intermediate)), np.array([0., .2, np.inf]),
        )[:energies.size]
        for array in (energies, intermediate, strengths, logs):
            array.setflags(write=False)
        gaps = energies[:, None] - intermediate[None, :]
        for imaginary in (False, True):
            times = (np.array([[0., 0., 0.], [.2, 2.5, 0.], [1e4, 5., 0.]])
                     if imaginary else np.array([.3, 0., -.3, .3]))
            expected = np.zeros((len(times), 3), dtype=float if imaginary else complex)
            for row, time in enumerate(times):
                for column in range(3):
                    exponent = logs[:, column, None] + (time[column] if imaginary else 1j * time) * gaps
                    expected[row, column] = np.sum(strengths * np.exp(exponent))
            for batch_rows in (7, 512):
                for batch_queries in (1, 7, 128):
                    with self.subTest(imaginary=imaginary, batch_rows=batch_rows,
                                      batch_queries=batch_queries), \
                         patch('pxp_model.pxp_autocorrelation._SPECTRAL_BLOCK_ROWS', batch_rows), \
                         patch('pxp_model.pxp_autocorrelation._SPECTRAL_BLOCK_QUERIES', batch_queries):
                        actual = _spectral_correlation(energies, intermediate, strengths, logs, times, imaginary)
                    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_empty_times_and_zero_weight_sector(self):
        energies = np.array([0., 1.])
        strengths = np.ones((2, 2))
        logs = np.full((2, 1), -np.inf)
        for imaginary in (False, True):
            times = np.zeros((2, 1)) if imaginary else np.zeros(2)
            actual = _spectral_correlation(energies, energies, strengths, logs, times, imaginary)
            np.testing.assert_array_equal(actual, np.zeros((2, 1)))
            self.assertEqual(_spectral_correlation(
                energies, energies, strengths, logs, times[:0], imaginary).shape, (0, 1))
            np.testing.assert_array_equal(_spectral_correlation(
                energies, np.empty(0), np.empty((2, 0)), np.zeros((2, 1)),
                times, imaginary), np.zeros((2, 1)))

    def test_long_real_times_and_common_energy_shift(self):
        energies, intermediate = np.arange(9) / 8, np.arange(7) / 16
        strengths = np.arange(63, dtype=float).reshape(9, 7) / 63
        logs = _thermal_log_weights(np.r_[energies, intermediate], np.array([0., .2, np.inf]))[:9]
        times = np.r_[np.linspace(-1e4, 1e4, 131), 0., 1e4]
        expected = np.array([
            (np.sum(strengths * np.exp(1j * t * (energies[:, None] - intermediate)), axis=1)
             @ np.exp(logs)) for t in times
        ])
        for offset in (0., 1e8):
            actual = _spectral_correlation(energies + offset, intermediate + offset,
                                           strengths, logs, times, False)
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)

    def test_imaginary_low_temperature_endpoint_and_tiny_tail(self):
        # exp(log_weight) underflows, but its tau=beta contribution is one.
        actual = _spectral_correlation(np.array([1.]), np.array([0.]), np.ones((1, 1)),
                                      np.array([[-1000.]]), np.array([[0.], [500.], [1000.]]), True)
        np.testing.assert_allclose(actual[:, 0], [0., np.exp(-500.), 1.], rtol=1e-14, atol=0.)
        tail = _spectral_correlation(np.array([0.]), np.array([1.]), np.ones((1, 1)),
                                    np.zeros((1, 1)), np.array([[700.]]), True)
        np.testing.assert_allclose(tail, [[np.exp(-700.)]], rtol=1e-14, atol=0.)

    def test_imaginary_growing_factors_preserve_underflowed_intermediates(self):
        # Separate exponentials would lose finite products outside [0, beta].
        for energy, intermediate, strengths, time, exponent in (
            (100., [0., 400.], [[0., 1.]], 2., -700.),
            (0., [0., 800.], [[1., 0.]], -1., -100.),
        ):
            actual = _spectral_correlation(np.array([energy]), np.array(intermediate),
                                           np.array(strengths), np.array([[-100.]]),
                                           np.array([[time]]), True)
            np.testing.assert_allclose(actual, [[np.exp(exponent)]], rtol=1e-14, atol=0.)

    def test_imaginary_overflow_still_raises(self):
        with self.assertRaises(FloatingPointError):
            _spectral_correlation(np.array([1000.]), np.array([0.]), np.ones((1, 1)),
                                  np.zeros((1, 1)), np.ones((1, 1)), True)


class SharedEigenvectorTest(unittest.TestCase):
    def test_blas_product_matches_explicit_adjoint_without_changing_inputs(self):
        rng = np.random.default_rng(2901)
        for complex_vectors in (False, True):
            for complex_shifted in (False, True):
                vectors = rng.normal(size=(7, 5))
                shifted = rng.normal(size=(7, 3))
                if complex_vectors:
                    vectors = vectors + 1j * rng.normal(size=vectors.shape)
                if complex_shifted:
                    shifted = shifted + 1j * rng.normal(size=shifted.shape)
                before_vectors, before_shifted = vectors.copy(), shifted.copy()
                for conjugated in (False, True):
                    with self.subTest(complex_vectors=complex_vectors,
                                      complex_shifted=complex_shifted, conjugated=conjugated):
                        actual_vectors = vectors.conj() if conjugated else vectors
                        expected = actual_vectors.conj().T @ shifted
                        actual = _eigenbasis_product(vectors, shifted, conjugated)
                        np.testing.assert_allclose(actual, expected, atol=2e-14)
                        self.assertTrue(actual.flags.c_contiguous)
                        np.testing.assert_array_equal(vectors, before_vectors)
                        np.testing.assert_array_equal(shifted, before_shifted)

    def test_shared_vectors_survive_both_time_domains(self):
        for L in (5, 6):
            sectors, logs = _full_ensemble(L, .2, np.array([0., .5, np.inf]))
            before = [vectors.copy() for _, _, vectors, _ in sectors]
            for imaginary in (False, True):
                times = (np.array([[0., 0., 0.], [.2, 2., 0.]])
                         if imaginary else np.array([0., .2]))
                average = _uniform_connected(sectors, logs, times, imaginary)
                _local_from_average(sectors, logs, times, imaginary, average)
                for (basis, _, vectors, conjugated), original in zip(sectors, before):
                    np.testing.assert_array_equal(vectors, original)
                    if conjugated and basis.Ns:
                        actual = _magnetization_in_eigenbasis(basis, vectors.conj())
                        canonical = _magnetization_in_eigenbasis(basis, vectors)
                        np.testing.assert_allclose(actual, canonical.conj(), atol=1e-14)


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
        sectors = [(None, np.array([energy]), np.ones((1, 1)), False) for energy in (0., 1.)]
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

    def test_uniform_conjugate_weights_preserve_tiny_variance_and_ground_mixture(self):
        delta = 2.**-28
        temperatures = np.array([0., .5, 1., np.inf])
        for L in (3, 4):
            energies = np.ones(L)
            energies[[1, L - 1]] = 0.
            if L % 2 == 0:
                energies[L // 2] = 0.
            logs = _thermal_log_weights(energies, temperatures)
            sectors = [(None, np.array([energy]), np.ones((1, 1)), k > L // 2)
                       for k, energy in enumerate(energies)]
            # The conjugate pair has M=1+delta; self-conjugate sectors have 1-delta.
            probability = np.exp(logs[1]) + np.exp(logs[-1])
            expected = probability * (1 - probability) * (2 * delta)**2
            for imaginary in (False, True):
                times = np.zeros((2, len(temperatures))) if imaginary else np.array([0., .2])
                observables = [np.array([[1 + delta if k == 1 else 1 - delta]])
                               for k in range(L // 2 + 1)]
                with self.subTest(L=L, imaginary=imaginary), patch(
                        'pxp_model.pxp_autocorrelation._magnetization_in_eigenbasis',
                        side_effect=observables) as magnetization:
                    actual = _uniform_connected(
                        sectors, [logs[k:k + 1] for k in range(L)], times, imaginary,
                    )
                self.assertEqual(magnetization.call_count, L // 2 + 1)
                np.testing.assert_allclose(actual, np.tile(expected, (2, 1)), rtol=2e-15, atol=0.)

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
    def test_orbit_z_matrices_match_quspin_in_every_momentum_pair(self):
        for L in range(2, 9):
            bases = [pxp_basis_1d(L, kblock=k) for k in range(L)]
            for k, source in enumerate(bases):
                vectors = np.eye(source.Ns, dtype=complex)
                for target_k, target in enumerate(bases):
                    q = target_k - k
                    ops = [["z", [j], np.exp(2j * np.pi * q * j / L) / L]
                           for j in range(L)]
                    expected = target.Op_shift_sector(source, ops, vectors,
                                                      dtype=np.complex128)
                    expected = expected.reshape(target.Ns, source.Ns)
                    with self.subTest(L=L, k=k, target_k=target_k):
                        actual = _momentum_z_shift(source, target, vectors, q)
                        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
                np.testing.assert_array_equal(vectors, np.eye(source.Ns))

    def test_conjugate_pair_coverage_and_infinite_temperature_sum_rule(self):
        for L in range(2, 13):
            sectors, logs = _full_ensemble(L, .2, np.array([np.inf]))
            indices = {id(basis): k for k, (basis, _, _, _) in enumerate(sectors)}
            visited = []

            def shift(source, target, *args, **kwargs):
                visited.append((indices[id(source)], indices[id(target)]))
                return _momentum_z_shift(source, target, *args, **kwargs)

            times = np.array([0.])
            average = _uniform_connected(sectors, logs, times, False)
            before = average.copy()
            with self.subTest(L=L), patch('pxp_model.pxp_autocorrelation._momentum_z_shift', shift):
                actual = _local_from_average(sectors, logs, times, False, average)
            covered = Counter()
            for pair in visited:
                covered.update({pair, tuple(sorted(((-pair[0]) % L, (-pair[1]) % L)))})
            self.assertEqual(covered, Counter(combinations(range(L), 2)))
            # Fixed pairs: k+k'=L, plus {0,L/2} for even L.
            fixed = (L - 1) // 2 + int(L % 2 == 0)
            self.assertEqual(len(visited), (L * (L - 1) // 2 + fixed) // 2)
            full = pxp_basis_1d(L)
            mean = np.mean([2 * (int(state) & 1) - 1 for state in full.states])
            np.testing.assert_allclose(actual, [[1 - mean**2]], atol=2e-12)
            np.testing.assert_array_equal(average, before)

    def test_conjugate_reuse_matches_full_space_thermal_spectrum(self):
        for L, m in ((4, .2), (7, 0.), (8, -.3)):
            _, H, z = full_matrices(L, m)
            energies, vectors = eigh(H)
            observable = vectors.conj().T @ (z[0, :, None] * vectors)
            strengths = np.abs(observable.T)**2
            for temperatures in (np.array([0.]), np.array([0., .001, .5, np.inf])):
                sectors, logs = _full_ensemble(L, m, temperatures)
                full_logs = _thermal_log_weights(energies, temperatures)
                mean = observable.diagonal().real @ np.exp(full_logs)
                for imaginary in (False, True):
                    grid = [0., .5, 1.] if imaginary else [.4, 0., -.2, .4]
                    _, times = prepare_time_grid(temperatures, grid, imaginary)
                    average = _uniform_connected(sectors, logs, times, imaginary)
                    actual = _local_from_average(sectors, logs, times, imaginary, average)
                    expected = _spectral_correlation(
                        energies, energies, strengths, full_logs, times, imaginary,
                    ) - mean**2
                    with self.subTest(L=L, imaginary=imaginary, temperatures=temperatures):
                        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)

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
            self.assertEqual(sum(basis.Ns for basis, _, _, _ in sectors), full.Ns)
            all_E = np.concatenate([E for _, E, _, _ in sectors])
            np.testing.assert_allclose(np.sort(all_E), eigh(H, eigvals_only=True), atol=2e-12)
            projectors = []
            for k, (basis, E, stored, conjugated) in enumerate(sectors):
                self.assertEqual(conjugated, k > L // 2)
                self.assertTrue(stored.flags.c_contiguous)
                if conjugated:
                    self.assertIs(stored, sectors[L - k][2])
                    self.assertIs(E, sectors[L - k][1])
                U = stored.conj() if conjugated else stored
                P = basis.get_proj(np.complex128).toarray()[2**L - 1 - full.states]
                projectors.append(P)
                Hk = build_hamiltonian(basis, .2, dtype=np.complex128).toarray()
                np.testing.assert_allclose(Hk, P.conj().T @ H @ P, atol=1e-12)
                np.testing.assert_allclose(Hk @ U, U * E, atol=2e-12)
                np.testing.assert_allclose(U.conj().T @ U, np.eye(basis.Ns), atol=1e-12)
            # Check general initial k and wraparound, not just k=0 -> q.
            for k, target_k in ((0, 1), (1, 3), (L - 1, 1)):
                source, _, U, source_conjugated = sectors[k]
                target, _, V, target_conjugated = sectors[target_k]
                U = U.conj() if source_conjugated else U
                V = V.conj() if target_conjugated else V
                q = (target_k - k) % L
                phases = np.exp(2j * np.pi * q * np.arange(L) / L) / L
                ops = [["z", [j], phases[j]] for j in range(L)]
                forward = target.Op_shift_sector(source, ops, U, dtype=np.complex128)
                forward = forward.reshape(target.Ns, source.Ns)
                np.testing.assert_allclose(_momentum_z_shift(source, target, U, q),
                                           forward, atol=1e-12)
                # Callback site 0 is the MSB, opposite to z's bit order.
                explicit = (z[::-1].T @ phases)[:, None] * (projectors[k] @ U)
                np.testing.assert_allclose(projectors[target_k] @ forward, explicit, atol=1e-12)
                reverse_ops = [["z", [j], phases[j].conjugate()] for j in range(L)]
                reverse = source.Op_shift_sector(target, reverse_ops, V, dtype=np.complex128)
                reverse = reverse.reshape(source.Ns, target.Ns)
                np.testing.assert_allclose(_momentum_z_shift(target, source, V, -q),
                                           reverse, atol=1e-12)
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
