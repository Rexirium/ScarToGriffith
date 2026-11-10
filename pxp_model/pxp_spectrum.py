"""PBC PXP chain: H = sum_i P_{i-1} X_i P_{i+1} - m sum_i Z_i.

Run from the repository root (in an environment with requirements.txt installed):
    python pxp_model/pxp_spectrum.py --L 20 --m 0.2
    python -m pxp_model.pxp_spectrum --L 12 --tmax 30 --nt 301

X, Z are Pauli matrices; bit 1 has Z=+1, bit 0 has Z=-1.
P=(1-Z)/2, the drive coefficient and hbar are 1. The Hilbert space
contains only configurations without adjacent 1s, including the boundary.
Thus the projected "x" operator is PXP, with no extra factors of 1/2.
The model-card detuning is Delta=2*m, with additive energy shift m*L.

Translation is by ONE site (a=1). Only the (k=0, reflection=+1) sector
is diagonalized, with reflection i -> L-1-i. The normalized initial state
is |Z2,+> = (|1010...> + |0101...>)/sqrt(2). Spectrum overlaps refer to
this symmetric state within this sector. The ZZ correlation and return
probability panels also include evolution from |0> = |00...0>, which lies
in the same sector. Both initial states share one eigensystem.
The ZZ thermal reference uses the canonical ensemble in this sector, with
beta chosen so Tr(H exp(-beta H))/Z = 0 (the energy of |Z2,+>).
After ED, degenerate eigenspaces are rotated to concentrate their |Z2,+>
overlap in the first state (energy tolerance 1e-10). All subsequent
calculations use these rotated eigenvectors.
Entropy is the half-chain von Neumann entropy, using natural logarithms.
Of the L+1 FSA eigenstates constructed at the same m, retain only
those with projection weight > 1/2 in this sector, then normalize them.
Large circles mark the ED state of maximum squared overlap with each
retained FSA state; repeated matches mark the same ED state once.
Red crosses show the retained FSA states at their effective-Hamiltonian
eigenenergies and overlaps with |Z2,+>, after projection and normalization.
Saved fsa_projection_weights are measured before normalization and retain
all L+1 FSA indices; scar_indices contains unique matched ED indices.
Selected overlaps below 1e-15 are displayed at the log-axis floor; saved
overlaps retain their actual values.
Level statistics use locally unfolded spacings in the central 4/5 of the
spectrum (window=5, etol=1e-8), normalized to mean spacing 1, as in the
reference script. Gaps <= etol are excluded, including zero-mode degeneracies.

Outputs: a four-panel PNG and an NPZ containing the plotted numerical data.
NPZ data keys match the flat dictionary returned by solve(), without k0_ prefixes.
Default output paths are relative to this script, independent of the cwd.
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

import matplotlib.pyplot as plt
import numpy as np
from quspin.operators import hamiltonian

if __package__:
    from .pxp_basis import pxp_basis_1d, resolve_degeneracies
    from .utils import canonical_weights, fibonacci, max_overlap_indices, unfolded_spacings
else:
    from pxp_basis import pxp_basis_1d, resolve_degeneracies
    from utils import canonical_weights, fibonacci, max_overlap_indices, unfolded_spacings


FSA_WEIGHT_CUTOFF = 0.5


def build_hamiltonian(basis, m):
    """The existing constrained user_basis implements the neighbor projectors."""
    static = [
        ["x", [[1.0, i] for i in range(basis.N)]],
        ["z", [[-m, i] for i in range(basis.N)]],
    ]
    # user_basis does not implement QuSpin's automatic symmetry checks.
    # Independent small-chain tests check the projected matrix and symmetries.
    return hamiltonian(
        static, [], basis=basis, dtype=np.float64,
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


def _eigenstate_observables(H, energies, vectors, basis, full_basis, zz_diagonal):
    """Check eigenpairs and compute entropy and ZZ in bounded column batches."""
    projector = basis.pxp_get_proj(full_basis)
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


def _evolve_observables(energies, vectors, coefficients, initial_indices, zz_diagonal, times):
    """Evolve both basis-state initial conditions using the real eigensystem."""
    zz = np.empty((2, times.size))
    fidelity = np.empty_like(zz)
    # Two states x 64 times bounds each propagated block to 128 columns.
    for start in range(0, times.size, 64):
        block = slice(start, start + 64)
        phases = energies[:, None] * times[block]
        # Real products avoid converting the full eigenvector matrix to complex.
        real = vectors @ (coefficients[:, :, None] * np.cos(phases)[:, None, :]).reshape(energies.size, -1)
        imag = vectors @ (coefficients[:, :, None] * np.sin(phases)[:, None, :]).reshape(energies.size, -1)
        np.square(real, out=real)
        np.square(imag, out=imag)
        real += imag
        zz[:, block] = (zz_diagonal @ real).reshape(2, -1)
        probabilities = real.reshape(energies.size, 2, -1)
        fidelity[:, block] = probabilities[initial_indices, [0, 1]]
    return zz, fidelity


def solve(L=20, m=0.2, times=None):
    """Return a flat dictionary of plot data and FSA matching diagnostics.

    Parameters: L, m, times. Spectrum: energies, overlaps, entropies,
    spacings, scar_indices. FSA spectrum: fsa_energies, fsa_overlaps (retained
    states only). Dynamics: zz and fidelity for |Z2,+>, zz_zero and
    fidelity_zero for |00...0>. Thermal: beta_th and zz_th at mean energy 0.
    FSA diagnostics:
    fsa_projection_weights (before normalization).
    The sector is always k=0, p=+ and overlaps refer to normalized |Z2,+>.
    """
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
    if not np.isfinite(m):
        raise ValueError("m must be finite")

    if times is None:
        times = np.linspace(0.0, 100.0, 1001)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be a nonempty, finite, increasing array with t >= 0")

    # Build the constrained bases once for ED, FSA and entropy calculations.
    full_basis = pxp_basis_1d(L)
    assert full_basis.Ns == fibonacci(L - 1) + fibonacci(L + 1)
    basis = pxp_basis_1d(L, a=1, kblock=0, pblock=1)
    dim = basis.Ns
    print(
        f"L={L}, m={m:g}, PBC, k=0, p=+; dimension={dim}/{full_basis.Ns}; "
        f"dense H + eigenvectors >= {16 * dim**2 / 2**20:.1f} MiB",
        flush=True,
    )

    H = build_hamiltonian(basis, m)
    E, U = H.eigh()

    # This representative denotes the normalized |Z2,+> in this sector.
    z2_index = basis.index("10" * (L // 2))
    z2_plus = np.zeros(dim)
    z2_plus[z2_index] = 1.0
    resolve_degeneracies(E, U, z2_plus)
    initial_indices = [z2_index, basis.index("0" * L)]
    coefficients = U[initial_indices, :].T

    # Project and normalize the retained FSA states before matching them to ED.
    basis._pxp_fsa_full_basis = full_basis
    fsa = basis.construct_fsa_eigensystem(normalized=False, m=m)
    fsa_projection_weights = np.sum(np.abs(fsa["psi"])**2, axis=0)
    valid_fsa = fsa_projection_weights > FSA_WEIGHT_CUTOFF
    psi_fsa = fsa["psi"][:, valid_fsa] / np.sqrt(fsa_projection_weights[valid_fsa])
    scar_indices = np.unique(max_overlap_indices(psi_fsa, U))
    fsa_energies = fsa["energies"][valid_fsa]
    fsa_overlaps = np.abs(psi_fsa[z2_index, :])**2
    del fsa, psi_fsa

    # Blockade gives n_i n_{i+1}=0, hence the bond average ZZ = 1 - 4*N_1/L.
    zz_diagonal = np.fromiter((1 - 4 * int(s).bit_count() / L for s in basis.states),
                              dtype=float, count=dim)
    entropies, zz_expectations, residual = _eigenstate_observables(
        H, E, U, basis, full_basis, zz_diagonal,
    )
    del H, basis, full_basis

    beta_th, thermal_weights = canonical_weights(E, 0.0)
    zz_th = float(thermal_weights @ zz_expectations)
    zz_t, fidelity_t = _evolve_observables(E, U, coefficients, initial_indices, zz_diagonal, times)

    spacings = spectrum_spacings(E)
    print(
        f"Done: FSA {valid_fsa.sum()}/{L + 1}, scars={scar_indices.size}, "
        f"spacings={spacings.size}, max residual={residual:.2e}",
        flush=True,
    )
    return dict(
        L=L, m=m, times=times,
        energies=E, overlaps=coefficients[:, 0]**2, entropies=entropies,
        spacings=spacings, scar_indices=scar_indices,
        fsa_energies=fsa_energies, fsa_overlaps=fsa_overlaps,
        zz=zz_t[0], zz_zero=zz_t[1], fidelity=fidelity_t[0], fidelity_zero=fidelity_t[1],
        beta_th=beta_th, zz_th=zz_th,
        fsa_projection_weights=fsa_projection_weights,
    )


def _save_figure(result, path):
    """Render the four panels from computed data without running the solver."""
    L, m = result["L"], result["m"]
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
        background = weights > 1e-15
        background[scars] = False
        colors = dict(cmap="plasma", vmin=0, vmax=result["entropies"].max())
        points = spectrum.scatter(
            result["energies"][background], weights[background],
            c=result["entropies"][background], **colors,
            marker="o", s=6, label=r"$k=0,\ p=+$",
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
        spectrum.axvline(0, color="0.5", linestyle="--", lw=1, label=r"$E_{\mathbb{Z}_2^{+}}=0$")
        spectrum.set(
            xlabel=r"$E_n$", ylabel=r"$|\langle \mathbb{Z}_2^{+}|E_n\rangle|^2$",
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
            0.97, 0.97, r"Central 4/5, $k=0,\ p=+$",
            ha="right", va="top", transform=statistics.transAxes,
        )
        # statistics.set_ylim(top=statistics.get_ylim()[1] * 1.15)

        for ax, key in ((correlation, "zz"), (fidelity, "fidelity")):
            ax.plot(result["times"], result[key], label=r"$|\mathbb{Z}_2^{+}\rangle$")
            ax.plot(result["times"], result[key + "_zero"], color="red", label=r"$|0\rangle$")
        correlation.axhline(
            result["zz_th"], color="black", linestyle="--",
            label=r"$\langle Z_i Z_{i+1}\rangle_{\mathrm{th}}$",
        )
        correlation.set(ylabel=r"$\langle Z_i Z_{i+1}\rangle_t$")
        correlation.legend(loc="upper right", ncols=3)
        correlation.tick_params(labelbottom=False)
        fidelity.set(xlabel=r"$t$", ylabel=r"$|\langle \psi_0|\psi(t)\rangle|^2$", xlim=(0, 100), ylim=(-0.02, 1.02))
        fidelity.legend()

        fig.suptitle(
            r"$H=\sum_i P_{i-1}X_iP_{i+1}-m\sum_i Z_i$"
            f"\nPBC, L={L}, m={m:g}, a=1; "
            r"$k=0,\ p=+;\quad |\mathbb{Z}_2^{+}\rangle=(|1010\cdots\rangle+|0101\cdots\rangle)/\sqrt{2}$"
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
    stem = output_dir / f"pxp_spectrum_L={L}_m={m:g}"
    np.savez_compressed(
        str(stem) + ".npz", **result, kblock=0, parity=1,
        initial_state="Z2_plus", unfolding_window=5, spacing_etol=1e-8,
        zz_zero_initial_state="0" * L,
        fsa_weight_cutoff=FSA_WEIGHT_CUTOFF,
    )
    _save_figure(result, str(stem) + ".png")
    print(f"Saved {stem}.png and {stem}.npz", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=20, help="even chain length (default: 20)")
    parser.add_argument("--m", type=float, default=0.2, help="uniform Z-field parameter (default: 0.2)")
    parser.add_argument("--tmax", type=float, default=100.0)
    parser.add_argument("--nt", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    if not np.isfinite(args.tmax) or args.tmax <= 0 or args.nt < 2:
        parser.error("--tmax must be finite and positive; --nt must be at least 2")
    result = solve(args.L, args.m, np.linspace(0, args.tmax, args.nt))
    save_results(result, args.output_dir)


if __name__ == "__main__":
    main()
