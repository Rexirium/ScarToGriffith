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

L = 16
g, r = -0.4, 0.2
nT = 200
Ts, dT = np.linspace(0.05, 10, nT, retstep=True)

# Hamiltonian parameters
eltype = np.float64
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

betas = 1 / Ts

# Hamiltonian
x_list = [[1.0, i] for i in range(L)]
z_list = [[g * (-1) ** i, i] for i in range(L)]
zzz_list = [[g * r * (-1) ** i, (i - 1) % L, i, (i + 1) % L]for i in range(L)]

static = [
    ["x", x_list], 
    ["z", z_list], 
    ["zzz", zzz_list]
]

energies = np.zeros(nT)

for k in range(0, L//2):
    basis = pxp_basis_1d(L, a=2, kblock=k)

    H = hamiltonian(static, [], dtype=eltype, basis=basis_full, **no_checks)

    E, U = H.eigh()
    Emin = E.min()


    for i, beta in enumerate(betas):
        boltzmann = np.exp(-beta * (E - Emin))
        rho = (U * boltzmann) @ U.conj().T / boltzmann.sum()
        energy = H.expt_value(rho, enforce_pure=False)
        
        energies[i] += energy


energies_expand = np.pad(energies, pad_width=1, mode="edge")

capacities = (energies_expand[2 :] - energies_expand[: -2]) / (2 * dT)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

axes[0].plot(Ts, energies)
axes[0].set(
    xlabel=r"$T$",
    ylabel=r"$\langle E \rangle$",
)

axes[1].plot(Ts, capacities)
axes[1].set(
    xlabel=r"$T$",
    ylabel=r"$C_V$",
)

plt.show()
