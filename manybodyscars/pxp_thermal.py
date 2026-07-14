import numpy as np
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt
from pxp_basis import pxp_basis_1d

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
nT = 101
Ts = np.geomspace(0.01, 1e2, nT)

# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)

# Hamiltonian parameters
eltype = np.float64
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

betas = 1 / Ts

# Hamiltonian
x_list = [[1.0, i] for i in range(L)]
z_list = [[g * (-1) ** i, i] for i in range(L)]
zzz_list = [
    [g * r * (-1) ** i, (i - 1) % L, i, (i + 1) % L]
    for i in range(L)
]
static = [
    ["x", x_list], 
    ["z", z_list], 
    ["zzz", zzz_list]
]

H = hamiltonian(static, [], dtype=eltype, basis=basis_full, **no_checks)

E, U = H.eigh()

entropies = np.zeros(nT)
for i, beta in enumerate(betas):
    boltzmann = np.exp(-beta * (E - E.min()))
    rho = (U * boltzmann) @ U.conj().T / boltzmann.sum()
    # Lift rho to the symmetry-free PXP basis, then trace out half of the chain.
    # rho_full = basis.pxp_project_from(rho, basis_full)
    entropies[i] = basis_full.ops_ent_entropy(rho)

fig, ax = plt.subplots(layout="constrained")
ax.semilogx(Ts, entropies, marker="o", markersize=3, linewidth=1.5)
ax.set(
    xlabel=r"$T$",
    ylabel="OSEE",
    title=rf"$L={L},\ g={g},\ r={r}$",
)
ax.grid(True, which="both", alpha=0.25)
# plt.savefig(f"manybodyscars/figures/pxp_osee_vs_T_L={L}_g={g:.1f}_ratio={r:.1f}.png", dpi=300)
plt.show()
