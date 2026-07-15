import numpy as np
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt
from pxp_basis import *

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "font.size": 14, 
    "xtick.direction": "in",
    "ytick.direction": "in", 
    "legend.frameon": False,
    "legend.edgecolor": "none"
})


# A generic momentum block is complex even though the real-space Hamiltonian
eltype = np.complex128
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)


def energy_vs_temperature(L:int, g, r, Ts:np.ndarray):
    
    # Hamiltonian
    x_list = [[1.0, i] for i in range(L)]
    z_list = [[g * (-1) ** i, i] for i in range(L)]
    zzz_list = [[g * r * (-1) ** i, (i - 1) % L, i, (i + 1) % L]for i in range(L)]

    static = [
        ["x", x_list], 
        ["z", z_list], 
        ["zzz", zzz_list]
    ]

    sector_spectra = []
    Emin = np.inf

    for k in range(0, L//2):
        basis = pxp_basis_1d(L, a=2, kblock=k)

        H = hamiltonian(static, [], dtype=eltype, basis=basis, **no_checks)

        E = H.eigvalsh()
        sector_spectra.append(E)
        Emin = min(Emin, E.min())

    # Accumulate the partition function and energy numerator sector by sector:
    #     Z_k = sum_n exp[-beta (E_nk - Emin)],
    #     N_k = sum_n E_nk exp[-beta (E_nk - Emin)],
    #     <E> = sum_k N_k / sum_k Z_k.
    # The common energy shift prevents overflow and cancels in the ratio.
    partition = np.zeros(nT)
    energy_numerator = np.zeros(nT)

    betas = 1 / Ts

    for E in sector_spectra:
        boltzmann = np.exp(-np.outer(betas, E - Emin))
        partition += boltzmann.sum(axis=1)
        energy_numerator += boltzmann @ E

    energies = energy_numerator / partition
    
    return energies

####################### Calculation ###############################

L = 20
r = 0.2
gs = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5]
nT = 200
Ts, dT = np.linspace(0.05, 10, nT, retstep=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

for i, g in enumerate(gs):
    energies = energy_vs_temperature(L, g, r, Ts)

    energies_expand = np.pad(energies, pad_width=1, mode="edge")
    capacities = (energies_expand[2 :] - energies_expand[: -2]) / (2 * dT)

    axes[0].plot(Ts, energies, label=rf"$g={g}$")
    axes[1].plot(Ts, capacities, label=rf"$g={g}$")
    
axes[0].set(
    xlabel=r"$T$",
    ylabel=r"$\langle E \rangle$",
    title=f"L={L}, r={r} Energy"
)
axes[0].legend()

axes[1].set(
    xlabel=r"$T$",
    ylabel=r"$C_V$",
    title=f"L={L}, r={r} Capacity"
)
axes[1].legend()

plt.savefig(f"manybodyscars/figures/pxp_capacity_L={L}_r={r:.1f}.png")
