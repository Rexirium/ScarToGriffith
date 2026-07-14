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
g, r = -0.3, 0.0
Temp = 0.1
beta = 1 / Temp

# Basis construction
basis = pxp_basis_1d(L, a=2, kblock=0)
basis_full = pxp_basis_1d(L)

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

H1 = hamiltonian(static, [], dtype=eltype, basis=basis_full, **no_checks)
H0 = hamiltonian([["x", x_list]], [], dtype=eltype, basis=basis_full, **no_checks)

E0, U0 = H0.eigh()
E1, U1 = H1.eigh()

boltzmann = np.exp(-beta * (E0 - E0.min()))
weights = boltzmann / boltzmann.sum()

S = U1.conj().T @ U0
rho_E = (S * weights[None, :]) @ S.conj().T
# rho0_E = (U0 * weights[None, :]) @ U0.conj().T
# rho = basis.pxp_project_from(rho0_E, basis_full)
# rho_E = U1 @ rho @ U1.conj().T

W = np.zeros((basis_full.Ns, basis_full.Ns), dtype=np.complex128)
W_t = np.zeros((basis_full.Ns, basis_full.Ns), dtype=np.complex128)
for site in range(L):
    bitpos = L - site - 1
    z_diag = 2.0 * ((basis_full.states >> bitpos) & 1) - 1.0
    Z_E = U1.conj().T @ (z_diag[:, None] * U1)
    B = Z_E @ rho_E
    W += Z_E * B.T
    W_one = Z_E * rho_E.T
    z_0 = np.trace(Z_E @ rho_E)
    W_t += z_0 * W_one

def correlation(t):
    phase = np.exp(1j * E1 * t)
    return phase @ (W - W_t) @ phase.conj() / L

ts = np.linspace(0.0, 100.0, 501)
phases = np.exp(1j * np.outer(ts, E1))
corrs_raw = np.einsum("tb,tb->t", phases @ W, phases.conj(), optimize=True)
corrs_t = np.einsum("tb,tb->t", phases @ W_t, phases.conj(), optimize=True)

corrs = (corrs_raw - corrs_t).real / L

fig, ax = plt.subplots()
ax.plot(ts, corrs)
ax.set(
    xlabel=r"$t$", 
    ylabel=r"$C_{\mathrm{conn}}(t) / L$",
    title=rf"$L={L},\ g={g},\ r={r},\ T={Temp}$"
)
# plt.savefig(f"manybodyscars/figures/pxp_autocorr_L={L}_g={g:.1f}_r={r:.1f}_T={Temp:.1f}.png", dpi=300)
plt.show()
