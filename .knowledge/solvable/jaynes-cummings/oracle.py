"""Jaynes–Cummings oracle: H = omega a†a + (omega0/2) sigma^z + g (a sigma^+ + a† sigma^-).

The rotating-wave (RWA) single-atom cavity-QED model [@JaynesCummings1963].  a (a†)
annihilate (create) a photon of the single cavity mode (frequency omega); sigma^{x,y,z}
are Pauli matrices for a two-level atom with transition frequency omega0 (level splitting
omega0, so the atomic term is (omega0/2) sigma^z); g is the atom–field coupling.  Detuning
delta = omega0 - omega.

EXACT SOLUTION (dressed states).  The excitation number
    C = a†a + (sigma^z + 1)/2                          (photons + atomic excitation)
commutes with H, so H is block-diagonal: C = 0 is the one-dimensional space spanned by
|0, down> (photon vacuum, atom in ground state), and for each C = m >= 1 the block is the
TWO-dimensional space {|n, up>, |n+1, down>} with n = m - 1 = 0, 1, 2, ...  In that basis
    H_block(n) = [[ omega n + omega0/2 ,      g sqrt(n+1)     ],
                  [    g sqrt(n+1)     , omega(n+1) - omega0/2 ]].
The mean of the two diagonal entries is omega(n + 1/2) and their difference is
(omega0 - omega) = delta, so diagonalizing the 2x2 gives the DRESSED-STATE energies
    E_pm(n) = omega(n + 1/2) +/- sqrt( g^2 (n+1) + delta^2/4 ),      n = 0, 1, 2, ...
with generalized Rabi splitting  Omega_n = 2 sqrt( g^2 (n+1) + delta^2/4 )  (vacuum Rabi
splitting 2g at resonance, n = 0).  The C = 0 ground state |0, down> is UNCOUPLED and has
energy exactly -omega0/2 for every g.  This is Tier A: the entire spectrum is closed-form.

Truncated bosons: the full-space cross-check builds a (a†) as (n_max+1)-dim matrices
(Fock cutoff n_max); the block spectrum is reproduced by the low-lying levels well below
the cutoff, and doubling n_max leaves them stable (CONVERGENCE-CHECK in self_test).

Cross-refs: this is the RWA of `quantum-rabi` (drop the counter-rotating a sigma^- + a†
sigma^+); it is the N = 1 case of `dicke-tavis-cummings` (Tavis–Cummings).  Same T6 family
as `lmg`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402


def dressed_levels(n, g, delta, omega=1.0):
    """Closed-form dressed pair (E_minus, E_plus) of excitation manifold C = n+1."""
    mid = omega * (n + 0.5)
    half = np.sqrt(g * g * (n + 1) + 0.25 * delta * delta)
    return mid - half, mid + half


def rabi_splitting(n, g, delta):
    """Generalized Rabi splitting 2 sqrt(g^2 (n+1) + delta^2/4) of manifold C = n+1."""
    return 2.0 * np.sqrt(g * g * (n + 1) + 0.25 * delta * delta)


def block_2x2(n, g, omega0, omega=1.0):
    """The independent 2x2 block on {|n,up>, |n+1,down>}; eigenvalues by eigvalsh."""
    H = np.array([[omega * n + 0.5 * omega0, g * np.sqrt(n + 1)],
                  [g * np.sqrt(n + 1), omega * (n + 1) - 0.5 * omega0]])
    return np.linalg.eigvalsh(H)


def spectrum(n_manifolds, g, omega0, omega=1.0):
    """Sorted low-lying spectrum: ground -omega0/2 plus dressed pairs up to given manifold."""
    delta = omega0 - omega
    levels = [-0.5 * omega0]
    for n in range(n_manifolds):
        lo, hi = dressed_levels(n, g, delta, omega)
        levels += [lo, hi]
    return np.sort(np.array(levels))


# ---- truncated-boson full-space ED (cross-check) -------------------------------------

def boson_ops(n_max):
    """Truncated annihilation a and number a†a on Fock space {|0>,...,|n_max>}."""
    d = n_max + 1
    a = np.zeros((d, d))
    for k in range(1, d):
        a[k - 1, k] = np.sqrt(k)          # a|k> = sqrt(k)|k-1>
    return a, a.T @ a


def full_ed(n_max, g, omega0, omega=1.0):
    """Sorted spectrum of the truncated full boson (x) 2-level Hamiltonian."""
    a, num = boson_ops(n_max)
    d = n_max + 1
    I_b = np.eye(d)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])   # |up><down|
    sm = sp.T
    H = (omega * np.kron(num, np.eye(2))
         + 0.5 * omega0 * np.kron(I_b, sz)
         + g * (np.kron(a, sp) + np.kron(a.T, sm)))
    return np.linalg.eigvalsh(H)


def compute(g=0.5, omega0=1.0, omega=1.0):
    """Jaynes–Cummings dressed-state quantities (exact, closed form)."""
    delta = omega0 - omega
    lo0, hi0 = dressed_levels(0, g, delta, omega)
    return {
        "e_ground": -0.5 * omega0,
        "detuning": delta,
        "vacuum_rabi_splitting": rabi_splitting(0, g, delta),
        "E_lower_n0": lo0,
        "E_upper_n0": hi0,
        "E_lower_n1": dressed_levels(1, g, delta, omega)[0],
    }


def self_test():
    # anchor 1 (CLOSED FORM == BLOCK DIAGONALIZATION): the analytic dressed pair equals
    #   an independently constructed 2x2 block, for every manifold n <= 20 at three
    #   (g, delta) points (resonant and detuned), to machine precision (1e-14).
    for g, delta in ((0.5, 0.0), (0.3, 0.7), (1.2, -0.9)):
        omega0 = 1.0 + delta                       # omega = 1, delta = omega0 - omega
        for n in range(21):
            lo, hi = dressed_levels(n, g, delta, 1.0)
            w = block_2x2(n, g, omega0, 1.0)
            assert abs(w[0] - lo) < 1e-14 and abs(w[1] - hi) < 1e-14, (g, delta, n, w, lo, hi)

    # anchor 2 (EXACT |0,down> EIGENSTATE): the uncoupled C = 0 state |0, down> is an
    #   exact eigenstate with energy -omega0/2 for ANY coupling (present in the spectrum
    #   to 1e-12).  It is the GROUND state precisely when it lies below E_-(0), i.e. in
    #   the weak-coupling / positive-detuning regime; at strong g a dressed level dips
    #   lower and becomes the ground state (checked both ways).
    for g, omega0 in ((0.5, 1.0), (1.5, 0.6), (0.2, 1.4)):
        w = full_ed(60, g, omega0, 1.0)
        assert np.min(np.abs(w - (-0.5 * omega0))) < 1e-12, (g, omega0)
    assert abs(full_ed(60, 0.2, 1.4, 1.0)[0] - (-0.7)) < 1e-12    # weak: vacuum is GS
    assert full_ed(60, 1.5, 0.6, 1.0)[0] < -0.5 * 0.6 - 1e-6     # strong: dressed GS lower

    # anchor 3 (TRUNCATED ED REPRODUCES THE BLOCK SPECTRUM + CONVERGENCE): the lowest
    #   levels of the truncated full-space Hamiltonian match the closed-form spectrum to
    #   1e-12, and are stable under doubling the Fock cutoff n_max = 40 -> 80 (the low
    #   levels sit far below the cutoff, so truncation does not touch them).
    for g, omega0 in ((0.5, 1.0), (0.8, 1.3)):
        K = 20                                     # compare lowest 21 levels (<< cutoff)
        ref = spectrum(30, g, omega0, 1.0)[:K + 1]
        ed40 = full_ed(40, g, omega0, 1.0)[:K + 1]
        ed80 = full_ed(80, g, omega0, 1.0)[:K + 1]
        assert np.max(np.abs(ed40 - ref)) < 1e-12, (g, omega0, np.max(np.abs(ed40 - ref)))
        assert np.max(np.abs(ed40 - ed80)) < 1e-12, (g, omega0)   # CONVERGENCE-CHECK

    # anchor 4 (VACUUM RABI SPLITTING): at resonance (delta = 0) the n = 0 dressed gap is
    #   exactly 2g; detuning widens it to 2 sqrt(g^2 + delta^2/4).
    assert abs(rabi_splitting(0, 0.5, 0.0) - 1.0) < 1e-14
    assert abs(rabi_splitting(0, 0.3, 0.8) - 2.0 * np.sqrt(0.09 + 0.16)) < 1e-14


if __name__ == "__main__":
    oracle_main(compute, {"g": (float, 0.5), "omega0": (float, 1.0), "omega": (float, 1.0)})
