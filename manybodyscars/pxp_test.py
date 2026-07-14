import numpy as np
from quspin.operators import quantum_LinearOperator, hamiltonian
from quspin.tools.evolution import ED_state_vs_time
import matplotlib.pyplot as plt
from pxp_basis import *
from utils import unfolded_spacings

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

L = 20
g, r = -0.4, 0.2
h = -0.05
# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)
dim = basis.Ns

# Hamiltonian parameters
eltype = np.float64
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

x_list = [[1.0, i] for i in range(L)]
z_list = [[g * (-1)**i, i] for i in range(L)]
xz_list = [[h, i, (i + 2) % L] for i in range(L)]
zzz_list = [[g * r * (-1)**i, (i - 1) % L, i, (i + 1) % L] for i in range(L)]

# Make Hamiltonian
my_static = [
    ["x", x_list], 
    ["z", z_list], 
    ["zzz", zzz_list]
]
ex_static = [
    ["x", x_list], 
    ["xz", xz_list], 
    ["zx", xz_list]
]
H = hamiltonian(my_static, [], basis=basis, dtype=eltype, **no_checks)

# Define initial state
Z2state = np.zeros(dim, dtype=eltype)
Z2idx = basis.index("10" * (L // 2))
Z2state[Z2idx] = 1.0
# time steps
ts = np.linspace(0.0, 40.0, 2001)

# local correlation to evaluate
zz_local = quantum_LinearOperator(
    [["zz", [[1.0, b, (b + 1) % L] for b in range(L)]]],
    basis=basis, dtype=eltype, **no_checks,
)

E, U = H.eigh()
print(f"g={g:.1f} r={r:.1f} PXP spectrum solved.")
start, stop = int(np.floor(dim / 6)), dim - int(np.floor(dim / 6))
spacings = unfolded_spacings(E, start=start, stop=stop, window=5, etol=1e-8)

# compute overlaps
overlaps = np.abs(U[Z2idx, :]) ** 2
marksizes = np.where(overlaps > 0.01, 15, 5)
# label the initial state energy
E0 = np.dot(E, overlaps)

# Compute the entanglement entropy of every eigen states
full_states = basis.pxp_project_from(U, basis_full, enforce_pure=True)
entropies = np.empty(dim)
for n in range(dim):
    entropies[n] = basis_full.my_ent_entropy(full_states[:, n])
    
# time evolution from initial state
states_t = ED_state_vs_time(Z2state, E, U, ts, iterate=False)
# local correlation and fidelity vs time
zz_correlation_t = np.real(zz_local.expt_value(states_t)) / L
z2_overlap_t = np.abs(states_t[Z2idx]) ** 2

# Plotting
fig = plt.figure(figsize=(11, 8), layout="constrained")
grid = fig.add_gridspec(3, 2, height_ratios=(1.3, 1, 1))
time_grid = grid[1:, :].subgridspec(2, 1, hspace=0)
time_axes = time_grid.subplots(sharex=True)
axes = np.concatenate((
    [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
    time_axes,
))

spec = axes[0].scatter(E, overlaps, c=entropies, cmap='plasma', s=marksizes)
axes[0].axvline(E0, color="black", linestyle="--", lw=1)

colorbar = fig.colorbar(spec, ax=axes[0], location="right", pad=0.02)
colorbar.ax.tick_params(length=0, labelsize=10)

axes[0].set(
    xlabel=r"$E_n$", ylabel=r"$|\langle \mathbb{Z}_2 |\psi \rangle|^2$", 
)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-15, 1.0)

axes[1].hist(spacings, bins=30, density=True, histtype="stepfilled", alpha=0.7)
axes[1].set(
    xlabel=r"$s$",
    ylabel=r"$P(s)$",
)

axes[2].plot(ts, zz_correlation_t)
axes[2].set(
    ylabel=r"$\langle Z_{i}Z_{i+1}\rangle$",
)
axes[2].tick_params(axis="x", labelbottom=False, bottom=False)

axes[3].plot(ts, z2_overlap_t)
axes[3].set(
    xlabel=r"$t$",
    ylabel=r"$|\langle \mathbb{Z}_2|\psi(t)\rangle|^2$",
)
# plt.show()
# plt.savefig(f"manybodyscars/figures/pxp_zandzzz_L={L}_g={g:.1f}_ratio={r:.1f}.png")
plt.show()
