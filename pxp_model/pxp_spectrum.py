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
is |Z2,+> = (|1010...> + |0101...>)/sqrt(2). All overlaps, correlations
and return probabilities refer to this symmetric state within this sector.
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
Level statistics use locally unfolded spacings in the central 2/3 of the
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
    from .pxp_basis import fibonacci, pxp_basis_1d, resolve_degeneracies
    from .utils import max_overlap_indices, unfolded_spacings
else:
    from pxp_basis import fibonacci, pxp_basis_1d, resolve_degeneracies
    from utils import max_overlap_indices, unfolded_spacings


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
    trim = len(energies) // 6
    start, stop = trim, len(energies) - trim
    first, last = max(start, 5), min(stop, len(energies) - 6)
    if last <= first or not np.any(np.diff(energies)[first:last] > 1e-8):
        return np.empty(0)
    return unfolded_spacings(energies, start=start, stop=stop, window=5, etol=1e-8)


def solve(L=20, m=0.2, times=None):
    """Return a flat dictionary of plot data and FSA matching diagnostics.

    Parameters: L, m, times. Spectrum: energies, overlaps, entropies,
    spacings, scar_indices. FSA spectrum: fsa_energies, fsa_overlaps (retained
    states only). Dynamics: zz, fidelity. FSA diagnostics:
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

    residual = np.max(np.linalg.norm(H.dot(U) - U * E, axis=0))
    if residual > 1e-9 * max(1.0, np.max(np.abs(E))):
        raise RuntimeError(f"Eigenpair residual too large: {residual:g}")

    # Project and normalize the retained FSA states before matching them to ED.
    basis._pxp_fsa_full_basis = full_basis
    fsa = basis.construct_fsa_eigensystem(normalized=False, m=m)
    fsa_projection_weights = np.sum(np.abs(fsa["psi"])**2, axis=0)
    valid_fsa = fsa_projection_weights > FSA_WEIGHT_CUTOFF
    psi_fsa = fsa["psi"][:, valid_fsa] / np.sqrt(fsa_projection_weights[valid_fsa])
    scar_indices = np.unique(max_overlap_indices(psi_fsa, U))

    fsa_overlaps = np.abs(psi_fsa[z2_index, :])**2
    coefficients = U[z2_index, :].conj()
    cat_overlaps = np.abs(coefficients)**2

    # Reuse the entropy buffer within small batches. Limit projected states
    # to 16 MiB per batch, unless a single vector already exceeds that size.
    projector = basis.pxp_get_proj(full_basis)
    entropies = np.empty(dim)
    batch_size = max(1, min(16, (16 * 2**20) // (full_basis.Ns * U.dtype.itemsize)))
    for start in range(0, dim, batch_size):
        block = slice(start, start + batch_size)
        states = projector @ U[:, block]
        entropies[block] = full_basis.my_ent_entropy(states, enforce_pure=True)
        del states

    # ZZ averaged over all bonds is diagonal and invariant under both maps.
    bits = ((basis.states[:, None] >> np.arange(L)) & 1).astype(float)
    z = 2 * bits - 1
    zz_diagonal = np.mean(z * np.roll(z, 1, axis=1), axis=1)
    fidelity_t = np.empty(times.size)
    zz_t = np.empty(times.size)

    # Keep memory bounded when many time points are requested.
    for start in range(0, times.size, 128):
        block = slice(start, start + 128)
        phases = np.exp(-1j * E[:, None] * times[block])
        states_t = U @ (coefficients[:, None] * phases)
        probabilities = np.abs(states_t)**2
        zz_t[block] = zz_diagonal @ probabilities
        # |Z2,+> is one basis vector, so its population is the return probability.
        fidelity_t[block] = probabilities[z2_index]

    spacings = spectrum_spacings(E)
    print(
        f"Done: FSA {valid_fsa.sum()}/{L + 1}, scars={scar_indices.size}, "
        f"spacings={spacings.size}, max residual={residual:.2e}",
        flush=True,
    )
    return dict(
        L=L, m=m, times=times,
        energies=E, overlaps=cat_overlaps, entropies=entropies,
        spacings=spacings, scar_indices=scar_indices,
        fsa_energies=fsa["energies"][valid_fsa], fsa_overlaps=fsa_overlaps,
        zz=zz_t, fidelity=fidelity_t,
        fsa_projection_weights=fsa_projection_weights,
    )


def save_results(result, output_dir):
    """Save the figure and self-contained arrays (no pickled objects)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    L, m = result["L"], result["m"]
    stem = output_dir / f"pxp_spectrum_L={L}_m={m:g}"
    np.savez_compressed(
        str(stem) + ".npz", **result, kblock=0, parity=1,
        initial_state="Z2_plus", unfolding_window=5, spacing_etol=1e-8,
        fsa_weight_cutoff=FSA_WEIGHT_CUTOFF,
    )

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.edgecolor": "none",
    }):
        fig = plt.figure(figsize=(11, 9), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=(1.4, 1, 1))
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
            marker="o", s=9, label=r"$k=0,\ p=+$",
        )
        # Keep selected states visible even below the logarithmic plotting floor.
        spectrum.scatter(
            result["energies"][scars], np.maximum(weights[scars], 1e-15),
            c=result["entropies"][scars], **colors,
            marker="o", s=40, edgecolors="black", linewidths=0.4,
            label="FSA-matched scars", zorder=3,
        )
        spectrum.scatter(
            result["fsa_energies"], np.maximum(result["fsa_overlaps"], 1e-15),
            color="red", marker="x", s=40, linewidths=1.2,
            label="FSA eigenstates", zorder=4,
        )

        spacings = result["spacings"]
        if spacings.size:
            statistics.hist(
                spacings, bins=30, density=True,
                histtype="stepfilled", alpha=0.7,
            )
        else:
            statistics.text(0.5, 0.5, "Too few nondegenerate levels", ha="center",
                            transform=statistics.transAxes)

        spectrum.axvline(0, color="0.5", linestyle="--", lw=1, label=r"$E_{\mathbb{Z}_2^{+}}=0$")
        spectrum.set(
            xlabel=r"$E_n$", ylabel=r"$|\langle \mathbb{Z}_2^{+}|E_n\rangle|^2$",
            yscale="log", ylim=(1e-15, 1),
        )
        spectrum.legend()
        fig.colorbar(points, ax=spectrum, pad=0.02, label=r"$S_{L/2}$ (nats)")

        statistics.set(
            xlabel=r"$s$", ylabel=r"$P(s)$",
        )
        statistics.set_xlim(left=0)
        statistics.text(
            0.97, 0.97, r"Central 2/3, $k=0,\ p=+$",
            ha="right", va="top", transform=statistics.transAxes,
        )
        statistics.set_ylim(top=statistics.get_ylim()[1] * 1.15)

        correlation.plot(result["times"], result["zz"])
        correlation.set(ylabel=r"$\langle Z_i Z_{i+1}\rangle_t$", ylim=(-1.05, 1.05))
        correlation.tick_params(labelbottom=False)
        fidelity.plot(result["times"], result["fidelity"])
        fidelity.set(xlabel=r"$t$", ylabel=r"$|\langle \mathbb{Z}_2^{+}|\psi(t)\rangle|^2$", ylim=(-0.02, 1.02))

        fig.suptitle(
            rf"$H=\sum_i P_{{i-1}}X_iP_{{i+1}}-m\sum_i Z_i$"
            f"\nPBC, L={L}, m={m:g}, a=1; "
            r"$k=0,\ p=+;\quad |\mathbb{Z}_2^{+}\rangle=(|1010\cdots\rangle+|0101\cdots\rangle)/\sqrt{2}$"
        )
        fig.savefig(str(stem) + ".png", dpi=180)
        plt.close(fig)
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
