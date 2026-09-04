"""Sachdev-Ye-Kitaev (SYK_4) oracle: finite-N Majorana ED level statistics and
the large-N Schwinger-Dyson entropy trend.

Model (Maldacena-Stanford 2016, PRD 94, 106002): N Majorana fermions
chi_i with {chi_i,chi_j}=2 delta_ij, Hamiltonian
    H = sum_{i<j<k<l} J_{ijkl} chi_i chi_j chi_k chi_l,
J_{ijkl} independent zero-mean Gaussians with variance 3! J^2 / N^3 =
(q-1)! J^2 / N^{q-1} at q=4 (MS Eq. 2.3).  This is the D-tier "exact in
the limit" card: the model is *solvable in the N->infinity melonic limit*
(closed G-Sigma equations, conformal IR); finite N is chaotic and studied
by ED.  Two scriptable pieces:

(i) small-N ED (2^{N/2}-dim Clifford construction, fixed seed): the
    disorder-averaged adjacent-gap ratio <r-tilde> lands in the random-
    matrix class fixed by the Bott periodicity of the Clifford algebra,
    N mod 8 = 0 -> GOE, 2,6 -> GUE, 4 -> GSE (You-Ludwig-Xu 2017;
    Garcia-Garcia-Verbaarschot 2016).  N=16 (GOE) and N=12 (GSE, Kramers-
    doubled) are checked against the CROSS-LOADED random-matrix-stats card
    constants -- a genuine cross-card oracle use.

(ii) large-N Schwinger-Dyson in imaginary time (G-Sigma self-consistency,
     beta-ramped to beta*J ~ 40): the thermodynamic-integration entropy
     S(T)/N falls monotonically from the free (1/2)ln2 toward the web-
     verified zero-temperature value S_0/N ~ 0.2324 (MS).  Reported
     observed-as-observed: the finite-beta value is NOT claimed to equal
     S_0 (a subleading -(3/2)ln(beta J)/N tail and the c/(2 beta J)
     specific-heat term keep it above S_0 at any finite beta).

Conformal exponents and the chaos bound are TABULATED with citations
below.  Links models/sachdev-ye-kitaev.
"""
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _lib.cli import oracle_main  # noqa: E402


def _load_rms():
    """Cross-load the random-matrix-stats card's verified <r-tilde> constants
    (genuine cross-card oracle use: the SYK class targets ARE those numbers)."""
    path = ROOT / "random-matrix-stats" / "oracle.py"
    spec = importlib.util.spec_from_file_location("rms_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RMS = _load_rms()
# targets for the SYK level-statistics anchors, taken from the sibling card
CLASS_TARGET = {
    0: ("GOE", _RMS.RATIO_RTILDE_NUMERIC["GOE"]),
    2: ("GUE", _RMS.RATIO_RTILDE_NUMERIC["GUE"]),
    4: ("GSE", _RMS.RATIO_RTILDE_NUMERIC["GSE"]),
    6: ("GUE", _RMS.RATIO_RTILDE_NUMERIC["GUE"]),
}
# bands = >= 4 sigma of the measured run-to-run spread across 5 dev seeds PLUS
# the finite-size centering offset; measured sigma_12 = 0.0038 (GSE, small
# 2^5-dim sector after Kramers dedup), sigma_16 = 0.0020 (GOE, 2^7 sector).
LEVEL_BANDS = {12: 0.018, 16: 0.013}

# --- tabulated large-N facts (citations in ORACLE.md) ---------------------
CONFORMAL_DELTA = 0.25                       # fermion dimension Delta = 1/q, q=4
S0_PER_MAJORANA = 0.2324                     # MS zero-T entropy density S_0/N
CHAOS_BOUND = "lambda_L = 2 pi k_B T / hbar"  # maximal chaos (saturates bound)


# ==========================================================================
# (i) finite-N Majorana ED
# ==========================================================================

def _popcount(a):
    return np.array([bin(int(x)).count("1") for x in a])


def _majoranas(N):
    """N Majorana operators as Pauli strings (x-mask, z-mask, phase) on N/2
    qubits, standard Jordan-Wigner: chi_{2k}=X_k prod_{j<k}Z_j,
    chi_{2k+1}=Y_k prod_{j<k}Z_j = i X_k Z_k prod_{j<k}Z_j.  {chi_a,chi_b}=2delta."""
    nq = N // 2
    ops = []
    for k in range(nq):
        zpre = (1 << k) - 1                 # Z on all qubits j < k
        xk = 1 << k
        ops.append((xk, zpre, 1.0 + 0j))    # chi_{2k}
        ops.append((xk, zpre | (1 << k), 1j))  # chi_{2k+1}
    return ops[:N], nq, 1 << nq


def _pmul(a, b):
    """Product of two Pauli strings in X^x Z^z form."""
    x1, z1, c1 = a
    x2, z2, c2 = b
    sign = (-1.0) ** bin(z1 & x2).count("1")   # from Z^z1 X^x2 = (-1)^<z1,x2> X Z
    return (x1 ^ x2, z1 ^ z2, c1 * c2 * sign)


def _quartic_terms(N):
    ops, nq, dim = _majoranas(N)
    terms = [_pmul(_pmul(ops[i], ops[j]), _pmul(ops[k], ops[l]))
             for i, j, k, l in itertools.combinations(range(N), 4)]
    return terms, nq, dim


def _rtilde(ev, drop=0.15):
    ev = np.sort(ev)
    lo = int(len(ev) * drop)
    ev = ev[lo:len(ev) - lo]
    s = np.diff(ev)
    s = s[s > 0]
    r = s[1:] / s[:-1]
    return np.minimum(r, 1.0 / r).mean()


def syk_level_rtilde(N=16, n_real=None, seed=1, drop=0.15):
    """Disorder-averaged <r-tilde> of the SYK_4 spectrum in one fermion-parity
    sector.  For the GSE class (N mod 8 = 4) the Kramers doubling is asserted
    then deduplicated.  Returns (ensemble, <r-tilde>, herm_residual, pairgap)."""
    if n_real is None:
        n_real = 200 if N <= 12 else 30
    terms, nq, dim = _quartic_terms(N)
    b = np.arange(dim)
    pre = [(b ^ x, c * ((-1.0) ** _popcount(z & b))) for (x, z, c) in terms]
    even = np.where(_popcount(b) % 2 == 0)[0]        # even-parity sector
    ensemble, _ = CLASS_TARGET[N % 8]
    rng = np.random.default_rng(seed)
    herm = 0.0
    pairgap = 0.0
    acc = []
    for _ in range(n_real):
        H = np.zeros((dim, dim), dtype=complex)
        for J, (rows, ph) in zip(rng.standard_normal(len(terms)), pre):
            H[rows, b] += J * ph
        herm = max(herm, float(np.max(np.abs(H - H.conj().T))))
        ev = np.linalg.eigvalsh(H[np.ix_(even, even)])
        if ensemble == "GSE":
            evs = np.sort(ev)
            pairgap = max(pairgap, float(np.max(evs[1::2] - evs[0::2])))
            ev = evs[0::2]                            # deduplicate Kramers pairs
        acc.append(_rtilde(ev, drop))
    return ensemble, float(np.mean(acc)), herm, pairgap


# ==========================================================================
# (ii) large-N Schwinger-Dyson (imaginary time)
# ==========================================================================

def _sd_grids(beta, Nt):
    k = np.arange(Nt)
    tw = np.exp(1j * np.pi * k / Nt)                 # antiperiodic twist
    m = np.where(k < Nt // 2, k, k - Nt)
    wn = (2 * m + 1) * np.pi / beta                  # signed Matsubara freqs
    return k, tw, wn


def _sd_step(beta, J, q, Nt, Gt, x, iters, tol):
    """Iterate 1/G(iw) = -iw - Sigma, Sigma(tau)=J^2 G(tau)^{q-1} to convergence."""
    k, tw, wn = _sd_grids(beta, Nt)
    to_w = lambda g: beta * np.fft.ifft(g * tw)
    to_tau = lambda g: (1.0 / beta) * np.conj(tw) * np.fft.fft(g)
    diff = np.inf
    it = 0
    for it in range(iters):
        Sw = to_w(J ** 2 * Gt ** (q - 1))
        Gtn = to_tau(1.0 / (-1j * wn - Sw)).real
        diff = float(np.max(np.abs(Gtn - Gt)))
        Gt = (1.0 - x) * Gt + x * Gtn
        if diff < tol:
            break
    Sw = to_w(J ** 2 * Gt ** (q - 1))
    Gw = to_w(Gt)
    energy = (1.0 / q) * (1.0 / beta) * float(np.sum(Sw * Gw).real)   # E/N
    return Gt, energy, it, diff, wn


def sd_solve(beta, J=1.0, q=4, Nt=2 ** 14, x=0.3, iters=30000, tol=1e-10, Gt=None):
    """Solve the SD equations at inverse temperature `beta` (warm-start `Gt`)."""
    if Gt is None:
        k, tw, wn = _sd_grids(beta, Nt)
        Gt = ((1.0 / beta) * np.conj(tw) * np.fft.fft(1.0 / (-1j * wn))).real
    return _sd_step(beta, J, q, Nt, Gt, x, iters, tol)


def entropy_trend(beta_max=40.0, J=1.0, q=4, Nt=2 ** 14, x=0.3, npts=32):
    """S(T)/N via thermodynamic integration of the (robust) energy density:
        log Z/N = (1/2)ln 2 - int_0^beta (E/N) dbeta',   S/N = beta E/N + log Z/N.
    Solves on a beta-ramp warm-started from the free solution.  Returns the
    grid, S/N(beta), E/N(beta), the max residual, and the conformal amplitude
    b at tau=beta/2 (should trend to (4 pi)^{-1/4})."""
    betas = np.concatenate([np.linspace(0.5, 4.0, npts // 4),
                            np.linspace(5.0, beta_max, npts - npts // 4)])
    Es = []
    Gt = None
    max_diff = 0.0
    for b in betas:
        Gt, E, it, diff, wn = sd_solve(b, J, q, Nt, x, Gt=Gt)
        Es.append(E)
        max_diff = max(max_diff, diff)
    Es = np.array(Es)
    bg = np.concatenate([[0.0], betas])
    Eg = np.concatenate([[0.0], Es])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (Eg[1:] + Eg[:-1]) * np.diff(bg))])
    logZ = 0.5 * math.log(2.0) - cum[1:]
    S = betas * Es + logZ
    b_half = float(Gt[Nt // 2] * (beta_max / math.pi) ** 0.5)
    return betas, S, Es, max_diff, b_half


# ==========================================================================
# CLI entry
# ==========================================================================

def compute(task="levels", N=16, seed=1, beta_max=40.0):
    """SYK_4 oracle.  task='levels' -> finite-N ED <r-tilde> vs its random-
    matrix class; task='entropy' -> large-N SD entropy trend toward S_0."""
    if task == "levels":
        ensemble, rt, herm, pairgap = syk_level_rtilde(N=N, seed=seed)
        _, target = CLASS_TARGET[N % 8]
        return {
            "task": "levels", "N": N, "N_mod_8": N % 8, "class": ensemble,
            "rtilde": rt, "rtilde_target": target,
            "in_band": abs(rt - target) < LEVEL_BANDS.get(N, 0.02),
            "kramers_pairgap": pairgap, "herm_residual": herm,
        }
    elif task == "entropy":
        betas, S, Es, max_diff, b_half = entropy_trend(beta_max=beta_max)
        return {
            "task": "entropy", "beta_max": beta_max,
            "S_over_N_highT": float(S[0]), "S_over_N_lowT": float(S[-1]),
            "S0_tabulated": S0_PER_MAJORANA, "E_over_N": float(Es[-1]),
            "conformal_b_half": b_half, "b_conformal_exact": (4 * math.pi) ** -0.25,
            "monotone_decreasing": bool(np.all(np.diff(S) < 1e-6)),
            "max_sd_residual": max_diff,
        }
    else:
        raise ValueError(task)


def self_test():
    # (i) finite-N ED: the two identity-proof classes land on the cross-loaded
    #     random-matrix-stats targets, Kramers doubling is exact for GSE, and H
    #     is Hermitian.
    ens16, rt16, herm16, pg16 = syk_level_rtilde(N=16, seed=1)
    assert ens16 == "GOE"
    assert herm16 < 1e-10
    assert abs(rt16 - _RMS.RATIO_RTILDE_NUMERIC["GOE"]) < LEVEL_BANDS[16], rt16

    ens12, rt12, herm12, pg12 = syk_level_rtilde(N=12, seed=1)
    assert ens12 == "GSE"
    assert herm12 < 1e-10
    assert pg12 < 1e-9, ("Kramers doubling broken", pg12)   # identity-proof
    assert abs(rt12 - _RMS.RATIO_RTILDE_NUMERIC["GSE"]) < LEVEL_BANDS[12], rt12

    # reproducibility: a fixed seed is bit-identical.
    assert syk_level_rtilde(N=16, seed=1)[1] == rt16

    # (ii) large-N SD: converges, entropy falls monotonically from (1/2)ln2
    #      toward the tabulated S_0 (observed-as-observed, NOT claimed exact),
    #      the conformal amplitude approaches (4 pi)^{-1/4}, E/N ~ ground energy.
    betas, S, Es, max_diff, b_half = entropy_trend(beta_max=40.0)
    assert max_diff < 1e-8, max_diff                        # convergence assert
    assert abs(S[0] - 0.5 * math.log(2.0)) < 0.01, S[0]     # high-T -> (1/2)ln2
    assert np.all(np.diff(S) < 1e-6)                        # monotone decreasing
    assert S0_PER_MAJORANA < S[-1] < 0.28, S[-1]            # above S_0, trending down
    assert -0.05 < Es[-1] < -0.03, Es[-1]                   # ground energy per Majorana
    assert abs(b_half - (4 * math.pi) ** -0.25) < 0.03, b_half  # -> conformal b


if __name__ == "__main__":
    oracle_main(compute, {
        "task": (str, "levels"),
        "N": (int, 16),
        "seed": (int, 1),
        "beta_max": (float, 40.0),
    })
