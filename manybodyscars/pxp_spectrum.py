import numpy as np
from quspin.operators import quantum_LinearOperator, quantum_operator
from quspin.tools.evolution import ED_state_vs_time
from pxp_basis import pxp_basis_1d
import matplotlib.pyplot as plt

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
# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)

# Hamiltonian parameters
eltype = np.float64
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

# Define initial state
Z2state = np.zeros(basis.Ns, dtype=eltype)
Z2idx = basis.index("10" * (L // 2))
Z2state[Z2idx] = 1.0
# time steps
ts = np.linspace(0.0, 40.0, 2001)

# local correlation to evaluate
zz_local = quantum_LinearOperator(
    [["zz", [[1.0, b, (b + 1) % L] for b in range(L)]]],
    basis=basis, dtype=eltype, **no_checks,
)

# parameter-dependent Hamiltonian
x_list = [[1.0, i] for i in range(L)]
z_list = [[(-1)**i, i] for i in range(L)]
zzz_list = [[(-1)**i, (i - 1) % L, i, (i + 1) % L] for i in range(L)]
H_pxp_operator = quantum_operator(
    {
        "x": [["x", x_list]],
        "z": [["z", z_list]],
        "zzz": [["zzz", zzz_list]],
    },
    basis=basis, dtype=eltype, **no_checks,
)


def main(g, r): 
    H_pxp = H_pxp_operator.tohamiltonian(
        {"x": 1.0, "z": g, "zzz": g * r}
    )

    # Diagonalizing
    E, U = H_pxp.eigh()
    print(f"g={g:.1f} r={r:.1f} PXP spectrum solved.")

    # compute overlaps
    overlaps = np.abs(U[Z2idx, :]) ** 2
    # specify the markersize for scattering
    marksizes = np.where(overlaps > 0.01, 15, 5)
    # label the initial state energy
    energy_expected = np.dot(E, overlaps)

    # Compute the entanglement entropy of every eigen states
    full_states = basis.pxp_project_from(U, basis_full, enforce_pure=True)
    entropies = np.empty(basis.Ns)
    for n in range(basis.Ns):
        entropies[n] = basis_full.my_ent_entropy(full_states[:, n])

    # time evolution from initial state
    states_t = ED_state_vs_time(Z2state, E, U, ts, iterate=False)
    # local correlation and fidelity vs time
    zz_correlation_t = np.real(zz_local.expt_value(states_t)) / L
    z2_overlap_t = np.abs(states_t[Z2idx]) ** 2

    # Plotting
    fig = plt.figure(figsize=(6, 8),layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.3, 2))
    time_grid = grid[1].subgridspec(2, 1, hspace=0)
    time_axes = time_grid.subplots(sharex=True)
    axes = np.concatenate(([fig.add_subplot(grid[0])], time_axes))

    spec = axes[0].scatter(E, overlaps, c=entropies, cmap='plasma', s=marksizes)
    axes[0].axvline(energy_expected, color="black", linestyle="--")

    colorbar = fig.colorbar(spec, ax=axes[0], location="right", pad=0.02)
    colorbar.ax.tick_params(length=0, labelsize=10)

    axes[0].set(
        xlabel=r"$E_n$", ylabel=r"$|\langle \mathbb{Z}_2 |\psi \rangle|^2$", 
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-12, 1.0)

    axes[1].plot(ts, zz_correlation_t)
    axes[1].set(
        ylabel=r"$\langle Z_{i}Z_{i+1}\rangle$",
    )
    axes[1].tick_params(axis="x", labelbottom=False, bottom=False)

    axes[2].plot(ts, z2_overlap_t)
    axes[2].set(
        xlabel=r"$t$",
        ylabel=r"$|\langle \mathbb{Z}_2|\psi(t)\rangle|^2$",
    )
    # plt.show()
    plt.savefig(f"manybodyscars/figures/pxp_zandzzz_L={L}_g={g:.1f}_ratio={r:.1f}.png")
    plt.close(fig)


gs = np.linspace(-0.5, -0.0, 6)
rs = np.linspace(0.0, 0.5, 6)

for g in gs:
    for r in rs:
        main(g, r)
