"""PBC PXP chain: H = sum_i P_{i-1} X_i P_{i+1} - m sum_i Z_i.

Run from the repository root (in an environment with requirements.txt installed):
    python pxp_model/pxp_spectrum.py --L 20 --m 0.2 --k 0 --p +1
    python -m pxp_model.pxp_spectrum --L 12 --tmax 30 --nt 301

X, Z are Pauli matrices; bit 1 has Z=+1, bit 0 has Z=-1.
P=(1-Z)/2, the drive coefficient and hbar are 1. The Hilbert space
contains only configurations without adjacent 1s, including the boundary.
Thus the projected "x" operator is PXP, with no extra factors of 1/2.
The model-card detuning is Delta=2*m, with additive energy shift m*L.

Translation is by ONE site (a=1), with reflection i -> L-1-i. Optional
--k (0 <= k < L) and --p (+1 or -1) select momentum and reflection sectors;
omitting either leaves that symmetry unrestricted. Joint k,p sectors require
k=0 or L/2. Without --k, diagonalize momentum blocks sequentially and merge
their spectra, keeping only one block's dense eigensystem in memory. If --p
is supplied alone, generic k and L-k are paired into the requested parity;
k=0 and L/2 use their joint k,p blocks. The selected Hilbert space is unchanged.
The initial state is |Z2> = |1010...>, projected into the selected sector
and normalized: no symmetry gives |Z2>, k=0 or p=+1 gives |Z2,+>, and
k=L/2 or p=-1 gives |Z2,->. The latter states are the normalized sum or
difference of the two Neel configurations. A zero projection gives zero
overlaps and unavailable (NaN) dynamics. The ZZ correlation and return
probability panels also evolve the projected |0> = |00...0> when present.
The ZZ thermal reference uses the canonical ensemble in this sector at the
initial state's mean energy (zero); it is NaN if that state is absent.
After ED, degenerate eigenspaces within each block are rotated to concentrate the initial-state
overlap in the first state (energy tolerance 1e-10). All subsequent
calculations use these rotated eigenvectors; different blocks are never mixed.
The return amplitude is summed over blocks before taking its squared modulus.
Thermal weights are normalized once over the entire selected spectrum.
Entropy is the half-chain von Neumann entropy, using natural logarithms.
Of the L+1 FSA eigenstates constructed at the same m, retain only
those with projection weight > 1/2 in this sector, then normalize them.
Large circles mark the ED state of maximum squared overlap with each
retained FSA state; repeated matches mark the same ED state once.
Red crosses show the retained FSA states at their effective-Hamiltonian
eigenenergies and initial-state overlaps, after projection and normalization.
Saved fsa_projection_weights are measured before normalization and retain
all L+1 FSA indices; scar_indices contains unique matched ED indices.
Overlaps below 1e-15 are displayed at the log-axis floor; saved
overlaps retain their actual values.
Level statistics use locally unfolded spacings in the central 4/5 of the
spectrum of each block (window=5, etol=1e-8), normalized to mean spacing 1
before pooling. Gaps <= etol are excluded, including zero-mode degeneracies.

Outputs: a four-panel PNG and an NPZ containing the plotted numerical data.
NPZ data keys match the flat dictionary returned by solve(), without k0_ prefixes.
Unrestricted symmetry metadata uses kblock=-1 and parity=0 (no pickled None).
sector_k and sector_dimensions describe the nonempty diagonalized blocks;
eigenstate_k follows the sorted energies. For parity-paired blocks the k label
denotes the pair (k,L-k), not a simultaneous momentum and parity eigenstate.
Filenames include k and p, with 'all' for an unrestricted symmetry.
Default outputs go to figures/spectrum/ beside this script, independent of cwd.
Conventions: .knowledge/models/rydberg-pxp/MODEL.md
Projected-drive construction: https://quspin.github.io/QuSpin/examples/example14.html
"""

import argparse
import os
from pathlib import Path

# Small symmetry blocks benefit from avoiding OpenMP/BLAS oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from quspin.operators import hamiltonian

if __package__:
    from .pxp_basis import pxp_basis_1d, resolve_degeneracies
    from .utils import canonical_weights, fibonacci, unfolded_spacings
else:
    from pxp_basis import pxp_basis_1d, resolve_degeneracies
    from utils import canonical_weights, fibonacci, unfolded_spacings


FSA_WEIGHT_CUTOFF = 0.5


def build_hamiltonian(basis, m, *, dtype=np.float64):
    """Build PXP; use complex dtype for general momentum sectors."""
    static = [
        ["x", [[1.0, i] for i in range(basis.N)]],
        ["z", [[-m, i] for i in range(basis.N)]],
    ]
    # user_basis does not implement QuSpin's automatic symmetry checks.
    # Independent small-chain tests check the projected matrix and symmetries.
    return hamiltonian(
        static, [], basis=basis, dtype=dtype,
        check_symm=False, check_pcon=False, check_herm=False,
    )


def spectrum_spacings(energies):
    """Match the reference unfolding; return no samples for tiny/flat spectra."""
    energies = np.sort(np.asarray(energies, dtype=float))
    trim = len(energies) // 10
    start, stop = trim, len(energies) - trim
    first, last = max(start, 5), min(stop, len(energies) - 6)
    if last <= first or not np.any(np.diff(energies)[first:last] > 1e-8):
        return np.empty(0)
    return unfolded_spacings(energies, start=start, stop=stop, window=5, etol=1e-8)


def _diagonalization_blocks(L, k, p, basis, full_basis):
    """Yield one momentum block at a time, pairing k and -k for fixed parity."""
    if k is not None:
        yield k, basis, basis.pxp_get_proj(full_basis)
        return
    if p is not None:
        row_by_state = {int(s): i for i, s in enumerate(full_basis.states)}
        reflection_rows = [row_by_state[int(f"{s:0{L}b}"[::-1], 2)]
                           for s in full_basis.states]
    for momentum in range(L if p is None else L // 2 + 1):
        special = momentum in (0, L // 2)
        sector_basis = pxp_basis_1d(L, kblock=momentum, pblock=p if special else None)
        if not sector_basis.Ns:
            continue
        projector = sector_basis.pxp_get_proj(full_basis)
        if p is not None and not special:
            # Reflection exchanges k and -k; their parity combination is isometric.
            projector = (projector + p * projector[reflection_rows]) / np.sqrt(2)
        yield momentum, sector_basis, projector


def _eigenstate_observables(H, energies, vectors, projector, full_basis, zz_diagonal):
    """Check eigenpairs and compute entropy and ZZ in bounded column batches."""
    entropies = np.empty(energies.size)
    zz_expectations = np.empty(energies.size)
    # At most 16 MiB of projected states, unless one vector exceeds the budget.
    batch_size = max(1, min(16, (16 * 2**20) // (full_basis.Ns * vectors.dtype.itemsize)))
    residual = 0.0
    tolerance = 1e-9 * max(1.0, np.max(np.abs(energies)))
    for start in range(0, energies.size, batch_size):
        block = slice(start, start + batch_size)
        states = vectors[:, block]
        residual = max(residual, np.max(np.linalg.norm(H.dot(states) - states * energies[block], axis=0)))
        if residual > tolerance:
            raise RuntimeError(f"Eigenpair residual too large: {residual:g}")
        zz_expectations[block] = zz_diagonal @ np.abs(states)**2
        entropies[block] = full_basis.my_ent_entropy(projector @ states, enforce_pure=True)
    return entropies, zz_expectations, residual


def _evolve_observables(energies, vectors, coefficients, zz_diagonal, times):
    """Return this block's ZZ contribution and complex return amplitude."""
    zz = np.zeros((2, times.size))
    amplitude = np.zeros_like(zz, dtype=complex)
    weights = np.abs(coefficients)**2
    active = np.flatnonzero(weights.sum(axis=0) > 1e-24)
    if not active.size:
        return zz, amplitude
    coefficients, weights = coefficients[:, active], weights[:, active]
    real_eigensystem = np.isrealobj(vectors) and np.isrealobj(coefficients)
    # Two states x 64 times bounds each propagated block to 128 columns.
    for start in range(0, times.size, 64):
        block = slice(start, start + 64)
        phases = energies[:, None] * times[block]
        if real_eigensystem:
            # Real products avoid converting the full eigenvector matrix to complex.
            cos, sin = np.cos(phases), np.sin(phases)
            real = vectors @ (coefficients[:, :, None] * cos[:, None, :]).reshape(energies.size, -1)
            imag = vectors @ (coefficients[:, :, None] * sin[:, None, :]).reshape(energies.size, -1)
            np.square(real, out=real)
            np.square(imag, out=imag)
            real += imag
            probabilities = real
            amplitude[active, block] = weights.T @ cos - 1j * (weights.T @ sin)
        else:
            phase_factors = np.exp(-1j * phases)
            states = vectors @ (coefficients[:, :, None] * phase_factors[:, None, :]).reshape(energies.size, -1)
            probabilities = np.abs(states)**2
            amplitude[active, block] = weights.T @ phase_factors
        zz[active, block] = (zz_diagonal @ probabilities).reshape(active.size, -1)
    return zz, amplitude


def _prepare_states(basis, full_basis, m, k, p):
    """Prepare normalized references and metadata in target or full coordinates.

    A specified k keeps target-block coordinates; otherwise expand into the
    full constrained basis for projection into each diagonalization block.
    """
    L = full_basis.N
    target_projector = basis.pxp_get_proj(full_basis)
    initial_rows = [full_basis.index("10" * (L // 2)), full_basis.index("0" * L)]
    initial = np.real_if_close(target_projector[initial_rows].conj().T.toarray())
    initial_state_weights = np.sum(np.abs(initial)**2, axis=0)
    present = initial_state_weights > 1e-24
    initial[:, present] /= np.sqrt(initial_state_weights[present])
    initial[:, ~present] = 0
    initial_state = ("Z2" if k is None and p is None else
                     "Z2_plus" if present[0] and (k == 0 or p == 1) else
                     "Z2_minus" if present[0] else "Z2_projected")

    basis._pxp_fsa_full_basis = full_basis
    fsa = basis.construct_fsa_eigensystem(normalized=False, m=m)
    fsa_projection_weights = np.sum(np.abs(fsa["psi"])**2, axis=0)
    valid_fsa = fsa_projection_weights > FSA_WEIGHT_CUTOFF
    psi_fsa = fsa["psi"][:, valid_fsa] / np.sqrt(fsa_projection_weights[valid_fsa])
    metadata = dict(
        initial_state=initial_state, initial_state_weights=initial_state_weights,
        fsa_energies=fsa["energies"][valid_fsa],
        fsa_overlaps=np.abs(initial[:, 0].conj() @ psi_fsa)**2,
        fsa_projection_weights=fsa_projection_weights,
    )
    if k is None:
        psi_fsa = target_projector @ psi_fsa
        initial = target_projector @ initial
    return initial, psi_fsa, metadata


def _solve_block(basis, full_basis, projector, m, times, initial, psi_fsa, *, dtype):
    """Compute one block's observables; no dense eigensystem escapes this call."""
    H = build_hamiltonian(basis, m, dtype=dtype)
    E, U = H.eigh()
    present = np.sum(np.abs(initial)**2, axis=0) > 1e-24
    initial[:, ~present] = 0
    if present[0]:
        resolve_degeneracies(E, U, initial[:, 0])
    # Conjugate the small state arrays instead of the dense eigenvector matrix.
    coefficients = (initial.conj().T @ U).conj().T

    fsa_scores = np.abs(psi_fsa.conj().T @ U)**2
    matches = np.argmax(fsa_scores, axis=1)
    match_weights = fsa_scores[np.arange(matches.size), matches]

    # Blockade gives n_i n_{i+1}=0, hence the bond average ZZ = 1 - 4*N_1/L.
    zz_diagonal = np.fromiter((1 - 4 * int(s).bit_count() / basis.N for s in basis.states),
                             dtype=float, count=basis.Ns)
    entropies, zz_expectations, residual = _eigenstate_observables(
        H, E, U, projector, full_basis, zz_diagonal,
    )
    zz, amplitude = _evolve_observables(E, U, coefficients, zz_diagonal, times)
    return dict(
        energies=E, overlaps=np.abs(coefficients[:, 0])**2, entropies=entropies,
        zz_expectations=zz_expectations, spacings=spectrum_spacings(E),
        zz=zz, return_amplitude=amplitude, residual=residual,
        fsa_indices=matches, fsa_match_weights=match_weights,
    )


def solve(L=20, m=0.2, times=None, *, k=None, p=None):
    """Return a flat dictionary of plot data and FSA matching diagnostics.

    Parameters: L, m, times, optional k and p. Spectrum: energies, overlaps, entropies,
    spacings, scar_indices. FSA spectrum: fsa_energies, fsa_overlaps (retained
    states only). Dynamics: zz and fidelity for the projected Neel state, zz_zero and
    fidelity_zero for |00...0>. Thermal: beta_th and zz_th at mean energy 0.
    FSA diagnostics:
    fsa_projection_weights (before normalization).
    Missing k/p means no corresponding restriction. initial_state_weights
    records raw projection weights for |1010...> and |00...0>; supported
    initial states are normalized before computing observables.
    With k=None, compute and release each momentum eigensystem in turn.
    Entropies refer to block-resolved eigenstates even at degenerate energies.
    """
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
    if not np.isfinite(m):
        raise ValueError("m must be finite")
    if k is not None and (not isinstance(k, (int, np.integer)) or not 0 <= k < L):
        raise ValueError("k must be an integer with 0 <= k < L")
    if p is not None and (not isinstance(p, (int, np.integer)) or p not in (-1, 1)):
        raise ValueError("p must be +1 or -1")
    if k is not None and p is not None and k not in (0, L // 2):
        raise ValueError("joint k,p sectors require k=0 or k=L/2")

    if times is None:
        times = np.linspace(0.0, 100.0, 1001)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be a nonempty, finite, increasing array with t >= 0")

    # Build the constrained bases once for ED, FSA and entropy calculations.
    full_basis = pxp_basis_1d(L)
    assert full_basis.Ns == fibonacci(L - 1) + fibonacci(L + 1)
    basis = full_basis if k is None and p is None else pxp_basis_1d(L, a=1, kblock=k, pblock=p)
    dim = basis.Ns
    if not dim:
        raise ValueError(f"empty symmetry sector for L={L}, k={k}, p={p}")
    print(
        f"L={L}, m={m:g}, PBC, k={k if k is not None else 'all'}, "
        f"p={p if p is not None else 'all'}; dimension={dim}/{full_basis.Ns}",
        flush=True,
    )

    initial, psi_fsa, metadata = _prepare_states(basis, full_basis, m, k, p)
    present = metadata["initial_state_weights"] > 1e-24
    fsa_count = metadata["fsa_energies"].size
    spectrum = {key: [] for key in ("energies", "overlaps", "entropies", "zz_expectations")}
    spacings = []
    sector_k, sector_dimensions = [], []
    best_fsa_weights = np.full(fsa_count, -1.)
    best_fsa_indices = np.zeros(fsa_count, dtype=int)
    zz_t = np.zeros((2, times.size))
    return_amplitude = np.zeros_like(zz_t, dtype=complex)
    offset, residual = 0, 0.0
    for momentum, sector_basis, projector in _diagonalization_blocks(L, k, p, basis, full_basis):
        dtype = np.float64 if momentum in (0, L // 2) else np.complex128
        block_label = (f"({momentum},{L - momentum})" if k is None and p is not None
                       and momentum not in (0, L // 2) else str(momentum))
        print(f"  k={block_label}; dimension={sector_basis.Ns}; dense H + eigenvectors >= "
              f"{2 * np.dtype(dtype).itemsize * sector_basis.Ns**2 / 2**20:.1f} MiB", flush=True)
        if k is None:
            project_back = projector.conj().T
            block_initial = np.real_if_close(project_back @ initial)
            block_fsa = project_back @ psi_fsa
        else:
            block_initial, block_fsa = initial, psi_fsa
        block = _solve_block(
            sector_basis, full_basis, projector, m, times, block_initial, block_fsa, dtype=dtype,
        )
        improved = block["fsa_match_weights"] > best_fsa_weights
        best_fsa_indices[improved] = offset + block["fsa_indices"][improved]
        best_fsa_weights[improved] = block["fsa_match_weights"][improved]
        zz_t += block["zz"]
        return_amplitude += block["return_amplitude"]
        for key in spectrum:
            spectrum[key].append(block[key])
        spacings.append(block["spacings"])
        sector_k.append(momentum)
        sector_dimensions.append(sector_basis.Ns)
        offset += sector_basis.Ns
        residual = max(residual, block["residual"])

    assert offset == dim
    spectrum = {key: np.concatenate(values) for key, values in spectrum.items()}
    order = np.argsort(spectrum["energies"], kind="stable")
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(dim)
    scar_indices = np.unique(inverse_order[best_fsa_indices])
    spectrum = {key: values[order] for key, values in spectrum.items()}
    E = spectrum["energies"]
    zz_expectations = spectrum.pop("zz_expectations")
    eigenstate_k = np.repeat(sector_k, sector_dimensions)[order]
    spacings = np.concatenate(spacings)
    fidelity_t = np.abs(return_amplitude)**2
    zz_t[~present] = np.nan
    fidelity_t[~present] = np.nan

    beta_th, zz_th = np.nan, np.nan
    if present[0]:
        if np.allclose(E, 0, rtol=0, atol=1e-12):
            beta_th, zz_th = 0.0, float(np.mean(zz_expectations))
        else:
            beta_th, thermal_weights = canonical_weights(E, 0.0)
            zz_th = float(thermal_weights @ zz_expectations)
    print(
        f"Done: FSA {fsa_count}/{L + 1}, scars={scar_indices.size}, "
        f"spacings={spacings.size}, max residual={residual:.2e}",
        flush=True,
    )
    return dict(
        L=L, m=m, times=times, kblock=-1 if k is None else k, parity=0 if p is None else p,
        **metadata, **spectrum,
        eigenstate_k=eigenstate_k, sector_k=np.asarray(sector_k),
        sector_dimensions=np.asarray(sector_dimensions),
        spacings=spacings, scar_indices=scar_indices,
        zz=zz_t[0], zz_zero=zz_t[1], fidelity=fidelity_t[0], fidelity_zero=fidelity_t[1],
        beta_th=beta_th, zz_th=zz_th,
    )


def _save_figure(result, path):
    """Render the four panels from computed data without running the solver."""
    import matplotlib.pyplot as plt

    L, m = result["L"], result["m"]
    k, p = int(result["kblock"]), int(result["parity"])
    sector = f"k={k if k >= 0 else 'all'}, p={f'{p:+d}' if p else 'all'}"
    initial_label = {
        "Z2": r"$|\mathbb{Z}_2\rangle$",
        "Z2_plus": r"$|\mathbb{Z}_2^{+}\rangle$",
        "Z2_minus": r"$|\mathbb{Z}_2^{-}\rangle$",
        "Z2_projected": r"$P_{\rm sector}|\mathbb{Z}_2\rangle=0$",
    }[str(result["initial_state"])]
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.edgecolor": "none",
    }):
        fig = plt.figure(figsize=(11, 9), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=(2, 1, 1))
        spectrum = fig.add_subplot(grid[0, 0])
        statistics = fig.add_subplot(grid[0, 1])
        correlation = fig.add_subplot(grid[1, :])
        fidelity = fig.add_subplot(grid[2, :], sharex=correlation)

        weights = result["overlaps"]
        scars = result["scar_indices"]
        background = np.ones(weights.size, dtype=bool)
        background[scars] = False
        colors = dict(cmap="plasma", vmin=0, vmax=result["entropies"].max())
        points = spectrum.scatter(
            result["energies"][background], np.maximum(weights[background], 1e-15),
            c=result["entropies"][background], **colors,
            marker="o", s=6, label=sector,
        )
        # Keep selected states visible even below the logarithmic plotting floor.
        spectrum.scatter(
            result["energies"][scars], np.maximum(weights[scars], 1e-15),
            c=result["entropies"][scars], **colors,
            marker="o", s=36, edgecolors="black", linewidths=0.4,
            label="FSA-matched scars", zorder=3,
        )
        spectrum.scatter(
            result["fsa_energies"], np.maximum(result["fsa_overlaps"], 1e-15),
            color="red", marker="x", s=40, linewidths=1.2,
            label="FSA eigenstates", zorder=4,
        )
        spectrum.axvline(0, color="0.5", linestyle="--", lw=1)
        spectrum.set(
            xlabel=r"$E_n$", ylabel=r"$|\langle \psi_0|E_n\rangle|^2$",
            yscale="log", ylim=(1e-15, 1),
        )
        spectrum.legend()
        fig.colorbar(points, ax=spectrum, pad=0.02, label=r"$S_{L/2}$ (nats)")

        spacings = result["spacings"]
        if spacings.size:
            statistics.hist(
                spacings, bins=40, density=True,
                histtype="bar", alpha=0.3,
            )
        else:
            statistics.text(0.5, 0.5, "Too few nondegenerate levels", ha="center",
                            transform=statistics.transAxes)

        s = np.linspace(0, max(4.0, spacings.max() if spacings.size else 0.0), 500)
        statistics.plot(s, np.exp(-s), "-", label="Poisson")
        statistics.plot(
            s, (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4),
            label="WD (GOE)",
        )
        statistics.plot(s, 4 * s * np.exp(-2 * s), "--", label="Semi-Poisson")
        statistics.legend(loc="upper right", bbox_to_anchor=(1, 0.88))

        statistics.set(xlabel=r"$s$", ylabel=r"$P(s)$", xlim=(0, None))
        statistics.text(
            0.97, 0.97, f"Central 4/5 per k block\n{sector}",
            ha="right", va="top", transform=statistics.transAxes,
        )
        for ax, key in ((correlation, "zz"), (fidelity, "fidelity")):
            if np.any(np.isfinite(result[key])):
                ax.plot(result["times"], result[key], label=initial_label)
            if np.any(np.isfinite(result[key + "_zero"])):
                ax.plot(result["times"], result[key + "_zero"], color="red", label=r"$|0\rangle$")
            if not ax.lines:
                ax.text(0.5, 0.5, "Initial states have zero projection in this sector",
                        ha="center", transform=ax.transAxes)
        if np.isfinite(result["zz_th"]):
            correlation.axhline(
                result["zz_th"], color="black", linestyle="--",
                label=r"$\langle Z_i Z_{i+1}\rangle_{\mathrm{th}}$",
            )
        correlation.set(ylabel=r"$\langle Z_i Z_{i+1}\rangle_t$")
        if correlation.lines:
            correlation.legend(loc="upper right", ncols=3)
        correlation.tick_params(labelbottom=False)
        fidelity.set(xlabel=r"$t$", ylabel=r"$|\langle \psi_0|\psi(t)\rangle|^2$", xlim=(0, 100), ylim=(-0.02, 1.02))
        if fidelity.lines:
            fidelity.legend()

        fig.suptitle(
            r"$H=\sum_i P_{i-1}X_iP_{i+1}-m\sum_i Z_i$"
            f"\nPBC, L={L}, m={m:g}, a=1; {sector}; {initial_label}"
        )
        # Finish automatic label layout, then join the two shared-time panels.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        upper, lower = correlation.get_position(), fidelity.get_position()
        shared_y = (upper.y0 + lower.y1) / 2
        correlation.set_position([upper.x0, shared_y, upper.width, upper.y1 - shared_y])
        fidelity.set_position([lower.x0, lower.y0, lower.width, shared_y - lower.y0])
        # Keep the tick labels beside the shared border from touching.
        for ax, index, alignment in ((correlation, 0, "bottom"), (fidelity, -1, "top")):
            visible_ticks = [tick for tick in ax.yaxis.get_major_ticks()
                             if ax.get_ylim()[0] <= tick.get_loc() <= ax.get_ylim()[1]]
            visible_ticks[index].label1.set_verticalalignment(alignment)
        try:
            fig.savefig(path, dpi=180)
        finally:
            plt.close(fig)


def save_results(result, output_dir):
    """Save the figure and self-contained arrays (no pickled objects)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    L, m = int(result["L"]), result["m"]
    k, p = int(result["kblock"]), int(result["parity"])
    stem = output_dir / (f"pxp_spectrum_L={L}_m={m:g}"
                         f"_k={k if k >= 0 else 'all'}_p={f'{p:+d}' if p else 'all'}")
    np.savez_compressed(
        str(stem) + ".npz", **result, unfolding_window=5, spacing_etol=1e-8,
        zz_zero_initial_state="0" * L,
        fsa_weight_cutoff=FSA_WEIGHT_CUTOFF,
    )
    _save_figure(result, str(stem) + ".png")
    print(f"Saved {stem}.png and {stem}.npz", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=20, help="even chain length (default: 20)")
    parser.add_argument("--m", type=float, default=0.0, help="uniform Z-field parameter (default: 0.0)")
    parser.add_argument("--k", type=int, help="momentum sector, 0 <= k < L (default: diagonalize all k blocks)")
    parser.add_argument("--p", type=int, choices=(1, -1), help="reflection parity (default: unrestricted)")
    parser.add_argument("--tmax", type=float, default=100.0)
    parser.add_argument("--nt", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures" / "spectrum")
    args = parser.parse_args()
    if not np.isfinite(args.tmax) or args.tmax <= 0 or args.nt < 2:
        parser.error("--tmax must be finite and positive; --nt must be at least 2")
    try:
        result = solve(args.L, args.m, np.linspace(0, args.tmax, args.nt), k=args.k, p=args.p)
    except ValueError as error:
        parser.error(str(error))
    save_results(result, args.output_dir)


if __name__ == "__main__":
    main()
