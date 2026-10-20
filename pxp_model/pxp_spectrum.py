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
Entropy is the half-chain von Neumann entropy, using natural logarithms.
Level statistics use locally unfolded spacings in the central 2/3 of the
spectrum (window=5, etol=1e-8), normalized to mean spacing 1, as in the
reference script. Gaps <= etol are excluded, including zero-mode degeneracies.

Outputs: a four-panel PNG and an NPZ containing the plotted numerical data.
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
    from .pxp_basis import fibonacci, pxp_basis_1d
    from .utils import unfolded_spacings
else:
    from pxp_basis import fibonacci, pxp_basis_1d
    from utils import unfolded_spacings


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
    """Return the k=0, p=+ spectrum and normalized |Z2,+> dynamics."""
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

    full_basis = pxp_basis_1d(L)
    assert full_basis.Ns == fibonacci(L - 1) + fibonacci(L + 1)
    print(f"L={L}, m={m:g}, PBC, a=1; constrained dimension={full_basis.Ns}", flush=True)
    return_amplitude = np.zeros(times.size, dtype=complex)
    zz_t = np.zeros(times.size)

    basis = pxp_basis_1d(L, a=1, kblock=0, pblock=1)
    dim = basis.Ns
    print(
        f"kblock=0, parity=+1, dimension={dim}; "
        f"dense H + eigenvectors >= {16 * dim**2 / 2**20:.1f} MiB",
        flush=True,
    )
    H = build_hamiltonian(basis, m)
    E, U = H.eigh()
    residual = np.max(np.linalg.norm(H.dot(U) - U * E, axis=0))
    if residual > 1e-9 * max(1.0, np.max(np.abs(E))):
        raise RuntimeError(f"Eigenpair residual too large: {residual:g}")

    # This representative denotes the normalized symmetric cat in this block.
    z2_index = basis.index("10" * (L // 2))
    coefficients = U[z2_index, :].conj()
    cat_overlaps = np.abs(coefficients)**2
    entropies = np.empty(dim)
    # Lift one eigenvector at a time, avoiding a full_dim x dim allocation.
    for n in range(dim):
        state = basis.pxp_project_from(U[:, n], full_basis, enforce_pure=True)
        entropies[n] = full_basis.my_ent_entropy(state)

    # ZZ averaged over all bonds is diagonal and invariant under both maps.
    bits = ((basis.states[:, None] >> np.arange(L)) & 1).astype(float)
    z = 2 * bits - 1
    zz_diagonal = np.mean(z * np.roll(z, 1, axis=1), axis=1)
    # Keep memory bounded when many time points are requested.
    for start in range(0, times.size, 128):
        block = slice(start, start + 128)
        phases = np.exp(-1j * E[:, None] * times[block])
        states_t = U @ (coefficients[:, None] * phases)
        zz_t[block] = zz_diagonal @ np.abs(states_t)**2
        return_amplitude[block] = cat_overlaps @ phases

    spacings = spectrum_spacings(E)
    print(
        f"  max residual={residual:.2e}; unfolded spacings={spacings.size}",
        flush=True,
    )
    sectors = [dict(
        kblock=0, parity=1, energies=E, overlaps=cat_overlaps,
        entropies=entropies, spacings=spacings, residual=residual,
    )]

    return dict(
        L=L, m=m, times=times, sectors=sectors, zz=zz_t,
        fidelity=np.abs(return_amplitude)**2,
    )


def save_results(result, output_dir):
    """Save the figure and self-contained arrays (no pickled objects)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    L, m = result["L"], result["m"]
    stem = output_dir / f"pxp_spectrum_L={L}_m={m:g}"
    arrays = {key: result[key] for key in ("L", "m", "times", "zz", "fidelity")}
    arrays["initial_state"] = "Z2_plus"
    arrays["unfolding_window"] = 5
    arrays["spacing_etol"] = 1e-8
    for sector in result["sectors"]:
        for key, value in sector.items():
            arrays[f"k{sector['kblock']}_{key}"] = value
    np.savez_compressed(str(stem) + ".npz", **arrays)

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
        max_entropy = max(s["entropies"].max() for s in result["sectors"])
        for sector in result["sectors"]:
            label = r"$k=0,\ p=+$"
            weights = sector["overlaps"]
            visible = weights > 1e-15
            points = spectrum.scatter(
                sector["energies"][visible], weights[visible],
                c=sector["entropies"][visible], cmap="plasma", vmin=0, vmax=max_entropy,
                marker="o", s=np.where(weights[visible] > 0.01, 25, 9), label=label,
            )
            spacings = sector["spacings"]
            if spacings.size:
                statistics.hist(
                    spacings, bins=30, density=True,
                    histtype="stepfilled", alpha=0.7,
                )
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
        if not any(s["spacings"].size for s in result["sectors"]):
            statistics.text(0.5, 0.5, "Too few nondegenerate levels", ha="center",
                            transform=statistics.transAxes)
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
