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

L = 20
g, r = -0.4, 0.2
Temp = 1.0
beta = 1 / Temp

# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
#basis_full = pxp_basis_1d(L)

# Hamiltonian parameters
eltype = np.float64
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

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

H1 = hamiltonian(static, [], dtype=eltype, basis=basis, **no_checks)
H0 = hamiltonian([["x", x_list]], [], dtype=eltype, basis=basis, **no_checks)

E0, U0 = H0.eigh()
E1, U1 = H1.eigh()

boltzmann = np.exp(-beta * (E0 - E0.min()))
weights = boltzmann / boltzmann.sum()

# Z_odd = sum_{i=1,3,...} Z_i.  It is invariant under the two-site
# translation used to define ``basis``, so it can be evaluated in this block.
z_odd_list = [[2/L, i] for i in range(1, L, 2)]
Z_odd = hamiltonian(
    [["z", z_odd_list]], [], dtype=eltype, basis=basis, **no_checks
).toarray()

# Express the initial thermal density matrix and observable in the H1
# eigenbasis.  This describes a quench from H0 to H1 at temperature Temp.
S = U1.conj().T @ U0
rho_E = (S * weights[None, :]) @ S.conj().T
Z_odd_E = U1.conj().T @ Z_odd @ U1

# Tr[rho Z_odd(t) Z_odd(0)] and
# <Z_odd(t)><Z_odd(0)> written as phase contractions in the H1 eigenbasis.
W = Z_odd_E * (Z_odd_E @ rho_E).T
W_t = np.trace(Z_odd_E @ rho_E) * (Z_odd_E * rho_E.T)

ts = np.linspace(0.0, 100.0, 501)
phases = np.exp(1j * np.outer(ts, E1))
corrs_raw = np.einsum("tb,tb->t", phases @ W, phases.conj(), optimize=True)
corrs_t = np.einsum("tb,tb->t", phases @ W_t, phases.conj(), optimize=True)
corrs = (corrs_raw - corrs_t).real

fig, ax = plt.subplots()
ax.plot(ts, corrs)
ax.set(
    xlabel=r"$t$",
    ylabel=r"$\langle Z_{\mathrm{odd}}(t) Z_{\mathrm{odd}}(0) \rangle_c$",
    title=rf"$L={L},\ g={g},\ r={r},\ T={Temp}$",
)
# plt.savefig(
#     f"manybodyscars/figures/pxp_odd_autocorr_L={L}_g={g:.1f}_r={r:.1f}_T={Temp:.2f}.png",
#     dpi=300,
# )
plt.show()
