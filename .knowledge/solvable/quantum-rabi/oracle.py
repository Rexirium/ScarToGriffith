"""Quantum Rabi oracle: Braak's G-function for the exact (transcendental) spectrum.

HAMILTONIAN (Braak 2011, PRL 107, 100401, Eq. 1; units hbar = 1) [@Braak2011]:
    H_R = omega a†a + g sigma^x (a + a†) + Delta sigma^z
a (a†) annihilate/create a photon of the single mode (frequency omega); sigma^{x,z} are
Pauli matrices for a two-level system with LEVEL SPLITTING 2*Delta (Braak's convention,
stated verbatim); g is the coupling.  This is the non-RWA parent of `jaynes-cummings`:
dropping the counter-rotating piece a sigma^- + a† sigma^+ of g sigma^x (a+a†) = g(a+a†)
(sigma^+ + sigma^-) gives H_JC with (omega0/2) = Delta.

EXACT SOLUTION — regular spectrum via the G-function (Braak Eqs. 3-5, transcribed
verbatim from the paper, NOT from memory).  The Z2 parity Pi = sigma^z exp(i pi a†a)
commutes with H_R and splits the space into H_+ and H_-.  In each parity sector the
regular eigenvalues are the zeros of a transcendental function of a variable x:
    G_pm(x) = sum_{n=0}^inf  K_n(x) [ 1 -/+ Delta/(x - n omega) ] (g/omega)^n        (3)
    n K_n = f_{n-1}(x) K_{n-1} - K_{n-2},   K_0 = 1,  K_1(x) = f_0(x)                 (4)
    f_n(x) = 2g/omega + (1/(2g)) ( n omega - x + Delta^2 / (x - n omega) )            (5)
G_pm has simple poles at x = 0, omega, 2omega, ... (the uncoupled-mode levels); its zeros
x^pm_n give the parity-(+/-) energies
    E^pm_n = x^pm_n - g^2 / omega .
This is Tier B: the solution is EXACT but TRANSCENDENTAL — there is no closed algebraic
form, only the roots of G_pm(x).  The exceptional (doubly degenerate, baseline) spectrum
E = n omega - g^2/omega occurs only for special (g, Delta) tuned so K_n(n) = 0 and is not
scanned here.

THE ED MATCH IS THE GATE.  The truncated-boson ED (Fock cutoff n_max) is diagonalized,
its eigenstates split by the parity operator, and the lowest ~8 energies in each parity
are compared to the G-function roots to 1e-8 (CONVERGENCE-CHECKed n_max = 120 vs 200).
If the G-roots disagreed with converged ED, the transcription would be wrong — so this
comparison certifies the transcription.  Braak's own Fig. 1 point (g = 0.7, Delta = 0.4,
omega = 1) is one benchmark, and the paper's stated ROOT COUNT there — six parity-(+)
zeros and five parity-(-) zeros in x in [-1, 5] — is pinned as a literature check.

Cross-refs: `jaynes-cummings` is the RWA of this model (small g/omega, near resonance);
`dicke-tavis-cummings` is its N-atom generalization; same T6 family as `lmg`.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402


def G(x, g, Delta, omega=1.0, parity=+1, nterms=160):
    """Braak G_parity(x) (Eq. 3), parity in {+1,-1}; series truncated at nterms."""
    r = g / omega
    K_prev2 = 0.0                 # K_{-1} placeholder (unused)
    K_prev1 = 1.0                 # K_0 = 1
    # f_0(x)
    f_prev = 2.0 * g / omega + (1.0 / (2.0 * g)) * (0.0 - x + Delta * Delta / (x - 0.0))
    total = K_prev1 * (1.0 - parity * Delta / (x - 0.0)) * 1.0        # n = 0 term
    K_curr = f_prev                                                   # K_1 = f_0
    rn = r
    for n in range(1, nterms):
        total += K_curr * (1.0 - parity * Delta / (x - n * omega)) * rn
        # advance recurrence to K_{n+1}: (n+1) K_{n+1} = f_n K_n - K_{n-1}
        f_n = 2.0 * g / omega + (1.0 / (2.0 * g)) * (
            n * omega - x + Delta * Delta / (x - n * omega))
        K_next = (f_n * K_curr - K_prev1) / (n + 1)
        K_prev1, K_curr = K_curr, K_next
        rn *= r
    return total


def g_roots(g, Delta, omega=1.0, parity=+1, x_max=8.0, nterms=160, ngrid=4000):
    """Roots x of G_parity between/around the poles x = 0, omega, 2omega, ...; sorted."""
    poles = [k * omega for k in range(int(np.ceil(x_max / omega)) + 1)]
    x_lo = -1.5 * abs(omega) - 2.0 * g * g / omega          # comfortably below GS
    edges = sorted(set([x_lo] + poles + [x_max]))
    roots = []
    eps = 1e-7
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = a + eps, b - eps
        if b2 <= a2:
            continue
        xs = np.linspace(a2, b2, ngrid)
        gs = G(xs, g, Delta, omega, parity, nterms)        # vectorized over the grid
        for i in range(len(xs) - 1):
            if np.isfinite(gs[i]) and np.isfinite(gs[i + 1]) and gs[i] * gs[i + 1] < 0:
                try:
                    r = brentq(G, xs[i], xs[i + 1],
                               args=(g, Delta, omega, parity, nterms), xtol=1e-13)
                    roots.append(r)
                except ValueError:
                    pass
    return np.sort(np.array(roots))


def rabi_energies(g, Delta, omega=1.0, parity=+1, x_max=8.0, nterms=160):
    """Regular parity-`parity` energies E = x - g^2/omega from the G-function roots."""
    return g_roots(g, Delta, omega, parity, x_max, nterms) - g * g / omega


# ---- truncated-boson ED (the gate) ---------------------------------------------------

def boson_ops(n_max):
    """Truncated annihilation a and number a†a on Fock space {|0>,...,|n_max>}."""
    d = n_max + 1
    a = np.zeros((d, d))
    for k in range(1, d):
        a[k - 1, k] = np.sqrt(k)
    return a, a.T @ a


def _rabi_H_and_parity(n_max, g, Delta, omega=1.0):
    a, num = boson_ops(n_max)
    d = n_max + 1
    Ib = np.eye(d)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    H = (omega * np.kron(num, np.eye(2))
         + g * np.kron(a + a.T, sx)
         + Delta * np.kron(Ib, sz))
    Pb = np.diag((-1.0) ** np.arange(d))            # exp(i pi a†a)
    Pi = np.kron(Pb, sz)                             # parity Pi = sigma^z exp(i pi N)
    return H, Pi


def ed_energies_by_parity(n_max, g, Delta, omega=1.0):
    """Truncated-boson ED energies split into (parity_+1, parity_-1) sorted arrays."""
    H, Pi = _rabi_H_and_parity(n_max, g, Delta, omega)
    w, v = np.linalg.eigh(H)
    p = np.einsum("ij,jk,ik->i", v.T, Pi, v.T)      # <psi_i|Pi|psi_i>, real +-1
    ep = np.sort(w[p > 0])
    em = np.sort(w[p < 0])
    return ep, em


def compute(g=0.7, Delta=0.4, omega=1.0):
    """Quantum Rabi exact spectrum via Braak's G-function (regular roots)."""
    ep = rabi_energies(g, Delta, omega, parity=+1, x_max=6.0)
    em = rabi_energies(g, Delta, omega, parity=-1, x_max=6.0)
    e_all = np.sort(np.concatenate([ep, em]))
    return {
        "e_ground": float(e_all[0]),
        "e_first_excited": float(e_all[1]),
        "gap": float(e_all[1] - e_all[0]),
        "n_parity_plus_below_6": int(len(ep)),
        "n_parity_minus_below_6": int(len(em)),
    }


def self_test():
    # anchor 1 (THE GATE — G-ROOTS == TRUNCATED ED, per parity, CONVERGED): at two
    #   parameter points (Braak's own Fig-1 point g=0.7,Delta=0.4 and a stronger-coupling
    #   detuned point) the lowest 8 G-function roots in EACH parity match the
    #   parity-resolved truncated-boson ED energies to 1e-8, with the ED Fock-converged
    #   (n_max = 120 vs 200 agree to 1e-10 on those levels).  This certifies the Braak
    #   transcription: a wrong recurrence would miss the ED.
    for g, Delta in ((0.7, 0.4), (1.0, 0.7)):
        ed120p, ed120m = ed_energies_by_parity(120, g, Delta, 1.0)
        ed200p, ed200m = ed_energies_by_parity(200, g, Delta, 1.0)
        for parity in (+1, -1):
            gfun = rabi_energies(g, Delta, 1.0, parity=parity, x_max=10.0)[:8]
            ed120 = (ed120p if parity > 0 else ed120m)[:8]
            ed200 = (ed200p if parity > 0 else ed200m)[:8]
            assert np.max(np.abs(ed120 - ed200)) < 1e-10, (g, Delta, parity)  # CONVERGED
            assert np.max(np.abs(gfun - ed120)) < 1e-8, (
                g, Delta, parity, np.max(np.abs(gfun - ed120)))

    # anchor 2 (PARITY SYMMETRY, operator level): [H, Pi] = 0 with Pi = sigma^z e^{i pi N}
    #   on the truncated space (exact up to the cutoff), so the parity split is legitimate.
    H, Pi = _rabi_H_and_parity(40, 0.9, 0.5, 1.0)
    assert np.max(np.abs(H @ Pi - Pi @ H)) < 1e-12
    assert np.max(np.abs(Pi @ Pi - np.eye(Pi.shape[0]))) < 1e-12   # Pi^2 = 1

    # anchor 3 (BRAAK FIG-1 ROOT COUNT, literature-pinned): at Braak's plotted point
    #   g=0.7, Delta=0.4, omega=1 the paper states G_+ has SIX zeros and G_- has FIVE
    #   (including the ground state) in x in [-1, 5].  Reproduce those counts exactly.
    rp = g_roots(0.7, 0.4, 1.0, parity=+1, x_max=5.0)
    rm = g_roots(0.7, 0.4, 1.0, parity=-1, x_max=5.0)
    rp = rp[rp >= -1.0]
    rm = rm[rm >= -1.0]
    assert len(rp) == 6, (len(rp), rp)
    assert len(rm) == 5, (len(rm), rm)

    # anchor 4 (JC LIMIT, band assert, documented): at small g/omega near resonance the
    #   Rabi ED ground/first-excited approach the Jaynes–Cummings closed form.  The
    #   counter-rotating terms shift energies by O(g^2/omega) (Bloch–Siegert), so we
    #   assert the Rabi levels sit within a documented O(g^2) band of the JC prediction
    #   and that the band SHRINKS as g decreases (2x smaller g -> >~3x closer).
    import importlib.util
    jc_path = Path(__file__).resolve().parents[1] / "jaynes-cummings" / "oracle.py"
    spec = importlib.util.spec_from_file_location("jc_oracle", jc_path)
    jc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jc)

    def rabi_gap_to_jc(g):
        omega0 = 2.0 * 0.5                                   # Delta = omega0/2, resonant
        Delta = 0.5
        ep, em = ed_energies_by_parity(120, g, Delta, 1.0)
        e_all = np.sort(np.concatenate([ep, em]))
        # JC: ground -omega0/2, first excited = lower of E_-(0) and next
        jc_e0 = -0.5 * omega0
        jc_e1 = jc.dressed_levels(0, g, omega0 - 1.0, 1.0)[0]
        return abs(e_all[0] - jc_e0), abs(e_all[1] - jc_e1)

    d_ground_big, d_exc_big = rabi_gap_to_jc(0.2)
    d_ground_sm, d_exc_sm = rabi_gap_to_jc(0.1)
    assert d_ground_big < 0.03 and d_exc_big < 0.03, (d_ground_big, d_exc_big)  # O(g^2) band
    assert d_ground_sm < d_ground_big / 3.0, (d_ground_sm, d_ground_big)        # shrinks ~g^2


if __name__ == "__main__":
    oracle_main(compute, {"g": (float, 0.7), "Delta": (float, 0.4), "omega": (float, 1.0)})
