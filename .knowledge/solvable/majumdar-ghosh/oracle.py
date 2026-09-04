"""Majumdar-Ghosh chain oracle: H = J sum_i [ S_i.S_{i+1} + (1/2) S_i.S_{i+2} ]
   (S=sigma/2, J=1, PBC, L a multiple of 4).

T5 (frustration-free / exact eigenstates), Tier C.  At the MG point J2/J1 = 1/2
the two nearest-neighbour dimer-covering states -- singlets on (0,1)(2,3)... and
on (1,2)(3,4)...(L-1,0) -- are *exact* ground states with energy E0 = -3JL/8
(each singlet contributes -3/4; the frustrating NNN term annihilates a dimer
product because a spin coupled to both members of a neighbouring singlet sees
S_a + S_b = 0).  The ground space is exactly 2-fold; only these two states are
exact -- the rest of the spectrum and the (positive) gap are NOT closed form and
are quoted as numerical ED values.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402


def mg_H(L, J=1.0):
    """H = J sum_i [ S_i.S_{i+1} + (1/2) S_i.S_{i+2} ], PBC."""
    sx, sy, sz = ed.spin_ops(L)

    def dot(i, j):
        return sx[i] @ sx[j] + sy[i] @ sy[j] + sz[i] @ sz[j]

    H = 0
    for i in range(L):
        H = H + dot(i, (i + 1) % L) + 0.5 * dot(i, (i + 2) % L)
    return J * H


def singlet_product_state(pairs, L):
    """Vector (2^L) for a product of two-site singlets (|up dn> - |dn up>)/sqrt2.

    Bit convention matches _lib.ed (kron of site 0..L-1; site 0 is the MSB).
    """
    dim = 1 << L
    psi = np.zeros(dim, complex)
    for state in range(dim):
        amp = 1.0
        for (i, j) in pairs:
            bi = (state >> (L - 1 - i)) & 1
            bj = (state >> (L - 1 - j)) & 1
            if bi == 0 and bj == 1:        # up_i dn_j
                amp *= 1.0 / np.sqrt(2)
            elif bi == 1 and bj == 0:      # dn_i up_j
                amp *= -1.0 / np.sqrt(2)
            else:
                amp = 0.0
                break
        psi[state] = amp
    return psi


def dimer_coverings(L):
    """The two MG dimer coverings as lists of NN singlet pairs."""
    cover_a = [(i, i + 1) for i in range(0, L, 2)]                 # (0,1)(2,3)...
    cover_b = [(i, i + 1) for i in range(1, L - 1, 2)] + [(L - 1, 0)]  # (1,2)...(L-1,0)
    return cover_a, cover_b


def compute(L=12, J=1.0):
    """Majumdar-Ghosh chain: exact dimer ground state + numerical gap."""
    H = mg_H(L, J)
    return {
        "e0_per_site": -3.0 * J / 8.0,            # exact at the MG point
        "e0_total": -3.0 * J * L / 8.0,           # exact
        "ground_degeneracy": ed.ground_states(H),  # exactly 2
        "gap_ed": ed.gap(H),                       # numerical (not closed form)
    }


def self_test():
    # anchor 1 (GROUND TRUTH): E0/L == -3/8 exactly and 2-fold GS, L in {8,12}.
    for L in (8, 12):
        H = mg_H(L)
        assert abs(ed.ground_energy(H) / L - (-3.0 / 8.0)) < 1e-12, L
        assert ed.ground_states(H) == 2, L

    # anchor 2: BOTH dimer coverings are exact eigenstates, E = -3L/8, residual ~0.
    for L in (8, 12):
        H = mg_H(L)
        states = []
        for pairs in dimer_coverings(L):
            psi = singlet_product_state(pairs, L)
            psi /= np.linalg.norm(psi)
            Hpsi = H @ psi
            E = np.vdot(psi, Hpsi).real
            assert abs(E - (-3.0 * L / 8.0)) < 1e-12, (L, pairs)
            assert np.linalg.norm(Hpsi - E * psi) < 1e-12, (L, pairs)  # operator-level
            states.append(psi)
        # the two coverings span the ED 2-fold ground space (projector check).
        from scipy.sparse.linalg import eigsh
        _, v = eigsh(H, k=2, which="SA")
        # orthonormalise the two (non-orthogonal) dimer states
        B = np.column_stack(states)
        Q, _r = np.linalg.qr(B)
        # every ED ground vector must lie in span{Q}: |P_Q v| == |v|
        P = Q @ Q.conj().T
        for c in range(2):
            gv = v[:, c]
            assert abs(np.linalg.norm(P @ gv) - 1.0) < 1e-10, (L, c)

    # anchor 3: gap is positive (observed), L=12.
    assert compute(L=12)["gap_ed"] > 1e-6


if __name__ == "__main__":
    oracle_main(compute, {"L": (int, 12), "J": (float, 1.0)})
