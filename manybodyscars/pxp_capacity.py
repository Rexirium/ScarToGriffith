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


def canonical_observables(sector_spectra, multiplicities, Ts):
    """Total energy and heat capacity at fixed temperatures, with k_B = 1."""
    energy_shift = min(spectrum[0] for spectrum in sector_spectra)
    betas = 1.0 / Ts
    partition = np.zeros_like(Ts)
    first_moment = np.zeros_like(Ts)
    second_moment = np.zeros_like(Ts)

    for spectrum, multiplicity in zip(sector_spectra, multiplicities):
        shifted_energy = spectrum - energy_shift
        boltzmann = np.exp(-np.outer(betas, shifted_energy))
        partition += multiplicity * boltzmann.sum(axis=1)
        first_moment += multiplicity * (boltzmann @ shifted_energy)
        second_moment += multiplicity * (
            boltzmann @ np.square(shifted_energy)
        )

    mean_shifted_energy = first_moment / partition
    energy_variance = np.maximum(
        second_moment / partition - np.square(mean_shifted_energy),
        0.0,
    )
    energy = energy_shift + mean_shifted_energy
    capacity = np.square(betas) * energy_variance
    return energy, capacity


def thermodynamics_vs_temperature(L, g_values, r, Ts):
    """Compute energy and C_V versus T for a small set of g values."""
    g_values = np.asarray(g_values, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    operator_terms = build_pxp_terms(L, r)
    spectra_by_g = [[] for _ in g_values]
    multiplicities = []

    for k, multiplicity in independent_momentum_sectors(L):
        basis = pxp_basis_1d(L, a=2, kblock=k)
        parameterized_hamiltonian = quantum_operator(
            operator_terms,
            basis=basis,
            dtype=MATRIX_DTYPE,
            **NO_CHECKS,
        )
        multiplicities.append(multiplicity)
        for g_index, g in enumerate(g_values):
            spectrum = parameterized_hamiltonian.eigvalsh(
                pars={"pxp": 1.0, "deformation": g}
            )
            spectra_by_g[g_index].append(spectrum)

    energies = np.empty((g_values.size, Ts.size))
    capacities = np.empty_like(energies)
    for g_index, sector_spectra in enumerate(spectra_by_g):
        energies[g_index], capacities[g_index] = canonical_observables(
            sector_spectra, multiplicities, Ts
        )
    return energies, capacities


def plot_thermodynamics_vs_temperature(
    Ts,
    g_values,
    energies,
    capacities,
    L,
    r,
    output_path,
):
    """Plot energy and C_V versus temperature for each g."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for g_index, g in enumerate(g_values):
        label = rf"$g={g:g}$"
        axes[0].plot(Ts, energies[g_index], label=label)
        axes[1].plot(Ts, capacities[g_index], label=label)

    axes[0].set(xlabel=r"$T$", ylabel=r"$\langle H\rangle$")
    axes[1].set(xlabel=r"$T$", ylabel=r"$C_V$")
    for axis in axes:
        axis.legend()
    fig.suptitle(rf"$L={L},\ r={r:g}$")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    return fig


def main():
    L, r = 20, 0.2
    g_values = np.array([0.0, -0.1, -0.2, -0.3, -0.4, -0.5])
    Ts = np.linspace(0.02, 2.0, 200)

    energies, capacities = thermodynamics_vs_temperature(
        L, g_values, r, Ts
    )
    output_path = (
        Path(__file__).resolve().parent
        / "figures"
        / f"pxp_capacity_T_L={L}_r={r:.1f}.png"
    )
    fig = plot_thermodynamics_vs_temperature(
        Ts,
        g_values,
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
