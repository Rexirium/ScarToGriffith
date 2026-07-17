from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from quspin.operators import quantum_operator

from pxp_basis import pxp_basis_1d


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

# Generic momentum blocks are complex even though the real-space H is real.
MATRIX_DTYPE = np.complex128
NO_CHECKS = dict(check_symm=False, check_pcon=False, check_herm=False)


def independent_momentum_sectors(L):
    """Independent two-site momentum blocks and k <-> -k multiplicities."""
    num_sectors = L // 2
    for k in range(num_sectors // 2 + 1):
        multiplicity = 1 if k == 0 or 2 * k == num_sectors else 2
        yield k, multiplicity


def build_pxp_terms(L, r):
    """Terms of H(g) = H0 + g V, constructed once for all momentum blocks."""
    x_terms = [[1.0, site] for site in range(L)]
    staggered_z_terms = [[(-1) ** site, site] for site in range(L)]
    staggered_zzz_terms = [
        [
            r * (-1) ** site,
            (site - 1) % L,
            site,
            (site + 1) % L,
        ]
        for site in range(L)
    ]
    return {
        "pxp": [["x", x_terms]],
        "deformation": [
            ["z", staggered_z_terms],
            ["zzz", staggered_zzz_terms],
        ],
    }


def update_thermal_accumulators(
    spectrum,
    multiplicity,
    betas,
    energy_shift,
    partition,
    first_moment,
    second_moment,
):
    """Add one sector spectrum to stable online thermal sums."""
    new_shift = min(energy_shift, spectrum[0])

    if np.isfinite(energy_shift) and new_shift < energy_shift:
        shift_change = energy_shift - new_shift
        rescale = np.exp(-betas * shift_change)
        second_moment[:] = rescale * (
            second_moment
            + 2.0 * shift_change * first_moment
            + np.square(shift_change) * partition
        )
        first_moment[:] = rescale * (
            first_moment + shift_change * partition
        )
        partition *= rescale

    shifted_energy = spectrum - new_shift
    boltzmann = np.exp(-np.outer(betas, shifted_energy))
    partition += multiplicity * boltzmann.sum(axis=1)
    first_moment += multiplicity * (boltzmann @ shifted_energy)
    second_moment += multiplicity * (
        boltzmann @ np.square(shifted_energy)
    )
    return new_shift


def thermodynamics_vs_g(L, g_values, r, Ts):
    """Compute U(g) and C_V(g) with online thermal accumulation."""
    g_values = np.asarray(g_values, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    betas = 1.0 / Ts
    operator_terms = build_pxp_terms(L, r)
    shape = (g_values.size, Ts.size)
    energy_shifts = np.full(g_values.size, np.inf)
    partitions = np.zeros(shape)
    first_moments = np.zeros(shape)
    second_moments = np.zeros(shape)

    for k, multiplicity in independent_momentum_sectors(L):
        basis = pxp_basis_1d(L, a=2, kblock=k)
        parameterized_hamiltonian = quantum_operator(
            operator_terms,
            basis=basis,
            dtype=MATRIX_DTYPE,
            **NO_CHECKS,
        )
        for g_index, g in enumerate(g_values):
            spectrum = parameterized_hamiltonian.eigvalsh(
                pars={"pxp": 1.0, "deformation": g}
            )
            energy_shifts[g_index] = update_thermal_accumulators(
                spectrum,
                multiplicity,
                betas,
                energy_shifts[g_index],
                partitions[g_index],
                first_moments[g_index],
                second_moments[g_index],
            )

    mean_shifted_energy = first_moments / partitions
    energy_variances = np.maximum(
        second_moments / partitions - np.square(mean_shifted_energy),
        0.0,
    )
    energies = energy_shifts[:, None] + mean_shifted_energy
    capacities = np.square(betas)[None, :] * energy_variances
    return energies, capacities


def plot_thermodynamics_vs_g(
    g_values,
    Ts,
    energies,
    capacities,
    L,
    r,
    output_path,
):
    """Plot U(g) and C_V(g) at several fixed temperatures."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for T_index, T in enumerate(Ts):
        axes[0].plot(
            g_values,
            energies[:, T_index],
            label=rf"$T={T:g}$",
        )
        axes[1].plot(
            g_values,
            capacities[:, T_index],
            label=rf"$T={T:g}$",
        )

    axes[0].set(xlabel=r"$g$", ylabel=r"$\langle H\rangle$")
    axes[1].set(xlabel=r"$g$", ylabel=r"$C_V$")
    for axis in axes:
        axis.legend()
    fig.suptitle(rf"$L={L},\ r={r:g}$")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    return fig


def main():
    L, r = 20, 0.2
    g_values = np.linspace(-0.5, 0.0, 51)
    Ts = np.array([0.01, 0.1, 0.2, 0.5, 0.8, 1.0])

    energies, capacities = thermodynamics_vs_g(
        L, g_values, r, Ts
    )
    output_path = (
        Path(__file__).resolve().parent
        / "figures"
        / f"pxp_capacity_g_L={L}_r={r:.1f}.png"
    )
    fig = plot_thermodynamics_vs_g(
        g_values,
        Ts,
        energies,
        capacities,
        L,
        r,
        output_path,
    )
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
