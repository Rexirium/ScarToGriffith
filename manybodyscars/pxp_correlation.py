import numpy as np
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pxp_basis import *

def averaging(arr, window=10):
    """Return the local moving average of a one-dimensional array."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be one-dimensional")
    if not isinstance(window, (int, np.integer)) or window < 0:
        raise ValueError("window must be a non-negative integer")
    if arr.size == 0:
        return arr.astype(np.result_type(arr.dtype, np.float64))

    indices = np.arange(arr.size)
    starts = np.maximum(indices - window, 0)
    stops = np.minimum(indices + window + 1, arr.size)

    dtype = np.result_type(arr.dtype, np.float64)
    cumulative_sum = np.concatenate(
        (np.zeros(1, dtype=dtype), np.cumsum(arr, dtype=dtype))
    )
    return (cumulative_sum[stops] - cumulative_sum[starts]) / (stops - starts)


def exponential_decay(t, amplitude, decay_rate, plateau):
    """Exponential decay with a finite-size long-time plateau."""
    return amplitude * np.exp(-decay_rate * t) + plateau


def power_law_decay(t, amplitude, exponent, plateau):
    """Power-law decay regularized at t = 0 with a finite-size plateau."""
    return amplitude * (1.0 + t) ** (-exponent) + plateau


def upper_envelope_indices(corrs):
    """Return indices of local maxima, including endpoint maxima."""
    peak_indices, _ = find_peaks(corrs)
    endpoint_indices = []
    if corrs.size == 1 or corrs[0] >= corrs[1]:
        endpoint_indices.append(0)
    if corrs.size > 1 and corrs[-1] >= corrs[-2]:
        endpoint_indices.append(corrs.size - 1)
    return np.unique(np.concatenate((endpoint_indices, peak_indices))).astype(int)


def fit_power_law_decay(ts, corrs):
    """Fit C_env(t) = A (1 + t)^(-alpha) + C_inf to local maxima."""
    peak_indices = upper_envelope_indices(corrs)
    peak_ts = ts[peak_indices]
    peak_corrs = corrs[peak_indices]
    plateau_guess = np.median(peak_corrs[-max(1, peak_corrs.size // 5):])
    parameters, covariance = curve_fit(
        power_law_decay,
        peak_ts,
        peak_corrs,
        p0=(peak_corrs[0] - plateau_guess, 1.0, plateau_guess),
        bounds=(0.0, np.inf),
    )
    return parameters, np.sqrt(np.diag(covariance)), peak_indices

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
rho_quench_E = (S * weights[None, :]) @ S.conj().T
# Dephase the initial state in the H1 eigenbasis while retaining its populations.
populations = np.real(np.diag(rho_quench_E)).copy()
populations /= populations.sum()
rho_E = np.diag(populations)

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

corrs = abs(corrs_raw - corrs_t) / L
fit_parameters, fit_errors, envelope_indices = fit_power_law_decay(ts, corrs)
amplitude, exponent, plateau = fit_parameters
amplitude_error, exponent_error, plateau_error = fit_errors
corrs_fit = power_law_decay(ts, *fit_parameters)

print(
    "Upper-envelope fit C_env(t) = A (1 + t)^(-alpha) + C_inf:\n"
    f"  envelope points = {envelope_indices.size}\n"
    f"  A       = {amplitude:.6g} +/- {amplitude_error:.2g}\n"
    f"  alpha   = {exponent:.6g} +/- {exponent_error:.2g}\n"
    f"  C_inf   = {plateau:.6g} +/- {plateau_error:.2g}"
)

fig, ax = plt.subplots()
ax.plot(ts, corrs, label="correlation")
ax.plot(
    ts[envelope_indices],
    corrs[envelope_indices],
    ".",
    markersize=4,
    label="upper-envelope peaks",
)
ax.plot(
    ts,
    corrs_fit,
    "--",
    linewidth=2,
    label=rf"power-law envelope fit: $\alpha={exponent:.3g}$",
)
ax.set(
    xlabel=r"$t$", 
    ylabel=r"$C_{\mathrm{conn}}(t) / L$",
    title=rf"$L={L},\ g={g},\ r={r},\ T={Temp}$"
)
ax.legend()
# plt.savefig(f"manybodyscars/figures/pxp_autocorr_avg_L={L}_g={g:.1f}_r={r:.1f}_T={Temp:.1f}.png", dpi=300)
plt.show()
