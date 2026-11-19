"""Small-chain checks against an explicit 2**L Pauli-space Hamiltonian.

    python -m unittest pxp_model.tests.test_pxp_spectrum
"""

import io
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from ..pxp_spectrum import (
    _evolve_observables, _save_figure, build_hamiltonian, main, spectrum_spacings,
    save_results, solve,
)
from ..pxp_basis import pxp_basis_1d, resolve_degeneracies
import numpy as np
from scipy.linalg import eigh, expm


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
    def test_batched_evolution_preserves_sector_weights_and_complex_amplitudes(self):
        rng = np.random.default_rng(42)
        energies = np.sort(rng.normal(size=7))
        times = np.linspace(0, 4, 131)
        zz_diagonal = rng.uniform(-1, 1, energies.size)
        phases = np.exp(-1j * energies[:, None] * times)
        for complex_system in (False, True):
            matrix = rng.normal(size=(energies.size, energies.size))
            coefficients = rng.normal(size=(energies.size, 2))
            if complex_system:
                matrix = matrix + 1j * rng.normal(size=matrix.shape)
                coefficients = coefficients + 1j * rng.normal(size=coefficients.shape)
            vectors, _ = np.linalg.qr(matrix)
            coefficients *= np.array([0.6, 0.3]) / np.linalg.norm(coefficients, axis=0)
            for active in ((True, True), (True, False), (False, True), (False, False)):
                with self.subTest(complex_system=complex_system, active=active):
                    selected = coefficients * active
                    expected_zz, expected_amplitude = [], []
                    for column in selected.T:
                        initial = vectors @ column
                        states = vectors @ (column[:, None] * phases)
                        expected_zz.append(zz_diagonal @ abs(states)**2)
                        expected_amplitude.append(initial.conj() @ states)
                    zz, amplitude = _evolve_observables(
                        energies, vectors, selected, zz_diagonal, times,
                    )
                    np.testing.assert_allclose(zz, expected_zz, atol=1e-13)
                    np.testing.assert_allclose(amplitude, expected_amplitude, atol=1e-13)

    def test_block_eigenvectors_are_released_before_next_diagonalization(self):
        vector_refs = []

        def track_hamiltonian(*args, **kwargs):
            block = build_hamiltonian(*args, **kwargs)
            block_ref = weakref.ref(block)
            diagonalize = type(block).eigh

            def track_eigh(*args, **kwargs):
                self.assertTrue(all(reference() is None for reference in vector_refs))
                energies, vectors = diagonalize(block_ref(), *args, **kwargs)
                vector_refs.append(weakref.ref(vectors))
                return energies, vectors

            block.eigh = track_eigh
            return block

        with patch("pxp_model.pxp_spectrum.build_hamiltonian", side_effect=track_hamiltonian):
            result = solve(8, 0.2, [0., 0.4])
        self.assertEqual(len(vector_refs), len(result["sector_k"]))
        self.assertGreater(len(vector_refs), 1)
        self.assertTrue(all(reference() is None for reference in vector_refs))

    def test_momentum_blocks_preserve_selected_space_observables(self):
        L, m = 8, 0.217
        full = pxp_basis_1d(L)
        H = explicit_pxp(L, m)[np.ix_(full.states, full.states)]
        rows = {int(state): row for row, state in enumerate(full.states)}
        reflected = [rows[int(f"{int(state):0{L}b}"[::-1], 2)] for state in full.states]
        zz = np.array([1 - 4 * int(state).bit_count() / L for state in full.states])
        neel = np.zeros(full.Ns)
        neel[full.index("10" * (L // 2))] = 1
        fsa = full.construct_fsa_eigensystem(m=m)
        for p in (None, 1, -1):
            with self.subTest(p=p):
                records = []

                def record_hamiltonian(basis, field, **kwargs):
                    block = build_hamiltonian(basis, field, **kwargs)
                    diagonalize = block.eigh

                    def record_eigh(*args, **options):
                        energies, vectors = diagonalize(*args, **options)
                        records.append((basis, energies, vectors))
                        return energies, vectors

                    block.eigh = record_eigh
                    return block

                with patch("pxp_model.pxp_spectrum.build_hamiltonian", side_effect=record_hamiltonian):
                    result = solve(L, m, [0., 0.4, 1.], p=p)
                target = pxp_basis_1d(L, pblock=p)
                target_P = target.pxp_get_proj(full).toarray()
                momenta = [basis.blocks["T_block"] for basis, _, _ in records]
                dimensions = [basis.Ns for basis, _, _ in records]
                self.assertEqual(sum(dimensions), target.Ns)
                self.assertLess(max(dimensions), target.Ns)
                expected_momenta = [
                    k for k in range(L if p is None else L // 2 + 1)
                    if pxp_basis_1d(L, kblock=k,
                                    pblock=p if k in (0, L // 2) else None).Ns
                ]
                self.assertEqual(momenta, expected_momenta)
                np.testing.assert_array_equal(result["sector_k"], momenta)
                np.testing.assert_array_equal(result["sector_dimensions"], dimensions)
                self.assertTrue(np.all(np.diff(result["energies"]) >= 0))
                vectors_full = np.empty((full.Ns, target.Ns), dtype=complex)
                for basis, energies, vectors in records:
                    momentum = basis.blocks["T_block"]
                    Q = basis.pxp_get_proj(full).toarray()
                    if p is not None and momentum not in (0, L // 2):
                        Q = (Q + p * Q[reflected]) / np.sqrt(2)
                    states = Q @ vectors
                    np.testing.assert_allclose(H @ states, states * energies, atol=1e-11)
                    if p is not None:
                        np.testing.assert_allclose(states[reflected], p * states, atol=1e-12)
                    selected = result["eigenstate_k"] == momentum
                    self.assertEqual(np.count_nonzero(selected), basis.Ns)
                    np.testing.assert_allclose(result["energies"][selected], energies, atol=1e-12)
                    expected_entropy = full.my_ent_entropy(states, enforce_pure=True)
                    np.testing.assert_allclose(result["entropies"][selected], expected_entropy, atol=1e-11)
                    vectors_full[:, selected] = states
                np.testing.assert_allclose(vectors_full.conj().T @ vectors_full,
                                           np.eye(target.Ns), atol=1e-11)
                np.testing.assert_allclose(vectors_full @ vectors_full.conj().T,
                                           target_P @ target_P.conj().T, atol=1e-11)

                initial = target_P @ (target_P.conj().T @ neel)
                initial /= np.linalg.norm(initial)
                np.testing.assert_allclose(result["overlaps"], abs(initial.conj() @ vectors_full)**2,
                                           atol=1e-12)
                raw_fsa = target_P @ (target_P.conj().T @ fsa["psi"])
                weights = np.sum(abs(raw_fsa)**2, axis=0)
                valid = weights > 0.5
                normalized_fsa = raw_fsa[:, valid] / np.sqrt(weights[valid])
                np.testing.assert_allclose(result["fsa_projection_weights"], weights, atol=1e-12)
                np.testing.assert_allclose(result["fsa_energies"], fsa["energies"][valid], atol=1e-12)
                np.testing.assert_allclose(result["fsa_overlaps"], abs(initial.conj() @ normalized_fsa)**2,
                                           atol=1e-12)
                matches = abs(vectors_full.conj().T @ normalized_fsa)**2
                np.testing.assert_allclose(matches[result["scar_indices"]].max(axis=0),
                                           matches.max(axis=0), atol=1e-12)
                rho = expm(-result["beta_th"] * (target_P.conj().T @ H @ target_P))
                rho /= np.trace(rho)
                self.assertAlmostEqual(np.trace(rho @ (target_P.conj().T @ H @ target_P)), 0)
                self.assertAlmostEqual(result["zz_th"],
                                       np.trace(rho @ (target_P.conj().T @ (zz[:, None] * target_P))))

    def test_unrestricted_fidelity_combines_sector_amplitudes(self):
        times = np.array([0., 0.3, 1.])
        result = solve(8, 0.2, times)
        amplitudes = []
        for k in (0, 4):
            sector = solve(8, 0.2, times, k=k)
            amplitudes.append(0.5 * (sector["overlaps"] @
                                    np.exp(-1j * sector["energies"][:, None] * times)))
        np.testing.assert_allclose(result["fidelity"], abs(sum(amplitudes))**2, atol=1e-12)
        incoherent = sum(abs(amplitude)**2 for amplitude in amplitudes)
        self.assertGreater(np.max(abs(result["fidelity"] - incoherent)), 0.1)

    def test_spacings_are_unfolded_within_each_momentum_block(self):
        samples = []

        def block_spacings(energies):
            index = len(samples)
            sample = np.array([0.25 + 0.01 * index, 1.75 - 0.01 * index])
            samples.append(sample)
            return sample

        with patch("pxp_model.pxp_spectrum.spectrum_spacings", side_effect=block_spacings) as unfold:
            result = solve(8, 0.2, [0.])
        self.assertEqual(unfold.call_count, 8)
        self.assertEqual([call.args[0].size for call in unfold.call_args_list],
                         result["sector_dimensions"].tolist())
        np.testing.assert_allclose(result["spacings"], np.concatenate(samples))

    def test_optional_sectors_against_full_spin_space(self):
        L, m = 8, 0.2
        times = np.array([0., 0.3, 1.])
        full = pxp_basis_1d(L)
        H = explicit_pxp(L, m)[np.ix_(full.states, full.states)]
        initial = np.zeros((full.Ns, 2))
        initial[full.index("10" * (L // 2)), 0] = 1
        initial[full.index("0" * L), 1] = 1
        zz = np.array([1 - 4 * int(s).bit_count() / L for s in full.states])
        for k, p in ((None, None), (0, None), (None, 1), (None, -1),
                     (0, 1), (L // 2, None), (L // 2, -1), (1, None)):
            with self.subTest(k=k, p=p):
                basis = pxp_basis_1d(L, kblock=k, pblock=p)
                P = basis.pxp_get_proj(full).toarray()
                reduced = P.conj().T @ H @ P
                result = solve(L, m, times, k=k, p=p)
                np.testing.assert_allclose(result["energies"], eigh(reduced)[0], atol=1e-12)
                self.assertEqual(result["kblock"], -1 if k is None else k)
                self.assertEqual(result["parity"], 0 if p is None else p)
                projected = P.conj().T @ initial
                weights = np.sum(abs(projected)**2, axis=0)
                np.testing.assert_allclose(result["initial_state_weights"], weights, atol=1e-12)
                for column, suffix in ((0, ""), (1, "_zero")):
                    if weights[column] < 1e-12:
                        self.assertTrue(np.isnan(result["zz" + suffix]).all())
                        self.assertTrue(np.isnan(result["fidelity" + suffix]).all())
                        continue
                    state = projected[:, column] / np.sqrt(weights[column])
                    states = np.column_stack([expm(-1j * reduced * t) @ state for t in times])
                    np.testing.assert_allclose(
                        result["zz" + suffix], zz @ abs(P @ states)**2, atol=1e-11,
                    )
                    np.testing.assert_allclose(
                        result["fidelity" + suffix], abs(state.conj() @ states)**2, atol=1e-11,
                    )
                if weights[0] < 1e-12:
                    np.testing.assert_array_equal(result["overlaps"], 0)
                    self.assertTrue(np.isnan(result["beta_th"]))
                    self.assertTrue(np.isnan(result["zz_th"]))
                else:
                    self.assertAlmostEqual(result["overlaps"].sum(), 1)
                    expected_state = ("Z2" if k is None and p is None else
                                      "Z2_minus" if k == L // 2 or p == -1 else "Z2_plus")
                    self.assertEqual(result["initial_state"], expected_state)
                    self.assertAlmostEqual(result["energies"] @ result["overlaps"], 0)

    def test_invalid_and_small_sectors(self):
        for blocks in (dict(k=-1), dict(k=8), dict(k=1.5), dict(p=0),
                       dict(p=1.0), dict(k=1, p=1)):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                solve(8, times=[0.], **blocks)
        with self.assertRaisesRegex(ValueError, "[Ee]mpty"):
            solve(4, times=[0.], k=0, p=-1)
        for L, k, p in ((4, 1, None), (10, 0, -1)):
            with self.subTest(L=L, k=k, p=p):
                result = solve(L, times=[0.], k=k, p=p)
                self.assertEqual(result["energies"].size, 1)
                self.assertEqual(result["spacings"].size, 0)
                np.testing.assert_array_equal(result["initial_state_weights"], [0, 0])
                self.assertTrue(np.isnan(result["zz"]).all())

    def test_cli_sector_arguments(self):
        for arguments, k, p in (([], None, None), (["--k", "1"], 1, None),
                                (["--p", "-1"], None, -1),
                                (["--k", "0", "--p", "+1"], 0, 1)):
            with self.subTest(arguments=arguments), \
                    patch("sys.argv", ["pxp_spectrum.py", "--L", "8", *arguments]), \
                    patch("pxp_model.pxp_spectrum.solve") as solver, \
                    patch("pxp_model.pxp_spectrum.save_results") as save:
                main()
                self.assertEqual(solver.call_args.kwargs, dict(k=k, p=p))
                self.assertEqual(solver.call_args.args[0], 8)
                save.assert_called_once()
                self.assertEqual(save.call_args.args[1],
                                 Path(__file__).resolve().parents[1] / "figures" / "spectrum")
        for arguments in (["--k", "1.5"], ["--p", "0"], ["--p", "1.0"]):
            with self.subTest(arguments=arguments), \
                    patch("sys.argv", ["pxp_spectrum.py", *arguments]), \
                    patch("sys.stderr", new_callable=io.StringIO), \
                    self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_excluded_initial_state_rendered_at_floor(self):
        result = solve(8, times=[0., 1.], k=1)
        with patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            _save_figure(result, "unused.png")
        spectrum = save.call_args.args[0].axes[0]
        points = spectrum.collections[0].get_offsets()
        self.assertEqual(len(points), result["energies"].size)
        np.testing.assert_allclose(points[:, 1], 1e-15, atol=0)

    def test_zero_modes_aligned_before_observables(self):
        L = 12
        basis = pxp_basis_1d(L, kblock=0, pblock=1)
        E, U = build_hamiltonian(basis, 0).eigh()
        z2 = basis.index("10" * (L // 2))
        zero = np.flatnonzero(abs(E) < 1e-10)
        self.assertGreater(zero.size, 1)
        projected = U[:, zero] @ U[z2, zero].conj()
        weight = np.linalg.norm(projected)**2
        times = np.linspace(0, 4, 17)
        result = solve(L, 0, times, k=0, p=1)
        np.testing.assert_allclose(result["energies"], E, atol=1e-12)
        self.assertAlmostEqual(result["overlaps"][zero[0]], weight)
        np.testing.assert_allclose(result["overlaps"][zero[1:]], 0, atol=1e-24)

        full = pxp_basis_1d(L)
        maximizer = basis.pxp_project_from(projected / np.sqrt(weight), full)
        self.assertAlmostEqual(result["entropies"][zero[0]], full.my_ent_entropy(maximizer))

        # Physical dynamics must agree with the original, unrotated ED basis.
        states = U @ (U[z2, :].conj()[:, None] * np.exp(-1j * E[:, None] * times))
        bits = ((basis.states[:, None] >> np.arange(L)) & 1).astype(float)
        z = 2 * bits - 1
        zz = np.mean(z * np.roll(z, 1, axis=1), axis=1)
        np.testing.assert_allclose(result["fidelity"], abs(states[z2])**2, atol=1e-12)
        np.testing.assert_allclose(result["zz"], zz @ abs(states)**2, atol=1e-12)

    def test_block_and_symmetric_state_dynamics(self):
        # Cross both batch boundaries and exercise a partial final batch.
        times = np.linspace(0, 4, 131)
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
                result = solve(L, m, times, k=0, p=1)
                np.testing.assert_allclose(result["fidelity"], np.abs(initial @ states)**2, atol=1e-11)
                np.testing.assert_allclose(result["zz"], zz @ np.abs(states)**2, atol=1e-11)
                zero_initial = np.zeros(full.Ns)
                zero_initial[full.index("0" * L)] = 1
                zero_states = U @ (
                    (U.conj().T @ zero_initial)[:, None] * np.exp(-1j * E[:, None] * times)
                )
                np.testing.assert_allclose(
                    result["zz_zero"], zz @ np.abs(zero_states)**2, atol=1e-11,
                )
                self.assertAlmostEqual(result["zz_zero"][0], 1)
                np.testing.assert_allclose(
                    result["fidelity_zero"], np.abs(zero_initial @ zero_states)**2, atol=1e-11,
                )
                self.assertAlmostEqual(result["fidelity_zero"][0], 1)
                self.assertAlmostEqual(result["overlaps"].sum(), 1)
                self.assertAlmostEqual(result["fidelity"][0], 1)
                self.assertAlmostEqual(result["zz"][0], -1)
                self.assertAlmostEqual(result["energies"] @ result["overlaps"], 0)

                # Check the thermal trace independently via a matrix exponential.
                H_sector = P.T @ H @ P
                rho = expm(-result["beta_th"] * H_sector)
                rho /= np.trace(rho)
                zz_sector = P.T @ (zz[:, None] * P)
                self.assertAlmostEqual(np.trace(rho @ H_sector), 0, places=10)
                self.assertAlmostEqual(result["zz_th"], np.trace(rho @ zz_sector), places=11)
                if m == 0:
                    self.assertEqual(result["beta_th"], 0.)
                    self.assertAlmostEqual(result["zz_th"], np.trace(zz_sector) / basis.Ns)

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
            result = solve(6, 0.2, np.array([0.]), k=0, p=1)
        construct.assert_called_once_with(normalized=False, m=0.2)
        np.testing.assert_allclose(result["fsa_projection_weights"], [.81, 1, .01, .64, 1, .49, 0])
        np.testing.assert_array_equal(result["scar_indices"], [0, 1])
        np.testing.assert_array_equal(result["fsa_energies"], [0, 1, 3, 4])

    def test_batched_spectrum_entropies_match_individual_states(self):
        # This sector has 49 states, including a final one-column batch.
        full = pxp_basis_1d(14)
        basis = pxp_basis_1d(14, kblock=0, pblock=1)
        energies, vectors = build_hamiltonian(basis, 0.2).eigh()
        z2 = np.zeros(basis.Ns)
        z2[basis.index("10" * 7)] = 1
        resolve_degeneracies(energies, vectors, z2)
        projector = basis.pxp_get_proj(full)
        expected = [full.my_ent_entropy(projector @ v) for v in vectors.T]
        result = solve(14, 0.2, np.array([0.]), k=0, p=1)
        np.testing.assert_allclose(result["entropies"], expected, atol=1e-12, rtol=0)

    def test_unfolded_spacings(self):
        np.testing.assert_allclose(spectrum_spacings(np.arange(60.)), np.ones(48))
        energies = np.arange(60.)**3
        energies[30] = energies[29]
        indices = np.arange(6, 54)
        gaps = np.diff(energies)[indices]
        local_means = (energies[indices + 5] - energies[indices - 5]) / 10
        expected = (gaps / local_means)[gaps > 1e-8]
        expected /= expected.mean()
        np.testing.assert_allclose(spectrum_spacings(energies), expected)
        self.assertEqual(spectrum_spacings(np.zeros(60)).size, 0)
        self.assertEqual(spectrum_spacings(np.arange(8.)).size, 0)

    def test_residual_check_reaches_last_eigenvector_batch(self):
        def corrupt_last_vector(energies, vectors, reference):
            resolve_degeneracies(energies, vectors, reference)
            vectors[:, -1] += 0.01 * vectors[:, 0]

        with patch("pxp_model.pxp_spectrum.resolve_degeneracies", side_effect=corrupt_last_vector):
            with self.assertRaisesRegex(RuntimeError, "Eigenpair residual too large"):
                solve(14, 0.2, np.array([0.]), k=0, p=1)

    def test_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for k, p, suffix in ((0, 1, "k=0_p=+1"), (None, None, "k=all_p=all"),
                                 (1, None, "k=1_p=all")):
                with self.subTest(k=k, p=p):
                    result = solve(6, 0.2, np.linspace(0, 1, 3), k=k, p=p)
                    result["L"] = np.asarray(result["L"])  # Accept NPZ scalars.
                    save_results(result, directory)
                    png = Path(directory) / f"pxp_spectrum_L=6_m=0.2_{suffix}.png"
                    self.assertGreater(png.stat().st_size, 1000)
                    with np.load(png.with_suffix(".npz"), allow_pickle=False) as saved:
                        for key, value in result.items():
                            np.testing.assert_array_equal(saved[key], value)
                        self.assertEqual(saved["fsa_weight_cutoff"], 0.5)
                        self.assertEqual(saved["zz_zero_initial_state"], "0" * int(result["L"]))
                        self.assertEqual(saved["unfolding_window"], 5)
                        self.assertEqual(saved["spacing_etol"], 1e-8)


if __name__ == "__main__":
    unittest.main()
