"""Shastry-Sutherland orthogonal-dimer model oracle.

    H = J sum_{square-lattice NN bonds} S_i.S_j  +  J' sum_{dimer bonds} S_i.S_j
    (S = sigma/2).  Convention: J is the square-lattice nearest-neighbour
    coupling, J' the dimer (diagonal) coupling -- matching the physics card
    .knowledge/models/shastry-sutherland (and Corboz-Mila, PRB 87, 115144).
    Note: the original Shastry-Sutherland and Miyahara-Ueda papers swap the
    letters (their J is the dimer coupling; their transition reads (J'/J)_c).

T5 (frustration-free / exact eigenstates), Tier C.  The orthogonal-dimer
geometry makes the dimer-singlet product state |Psi> = prod_dimers |singlet> an
*exact* eigenstate of the full H for ALL J/J', with energy E = -(3/4) J' N_dimer
(the J bonds annihilate it: each external spin couples to both members of a
neighbouring singlet, and S_a + S_b = 0 on a singlet).  It is the ground state
only for small J/J' (dimer phase); above a level crossing the ground state is a
different, non-exact state.  ONLY the dimer state is exact -- the generic
spectrum and the phase-transition point are numerical.

Cluster: the smallest SS cluster that faithfully realises the orthogonal-dimer
connectivity under PBC is the 4x4 = 16-site torus (8 dimers).  Smaller tori
(e.g. 4x2) break the dimer-orthogonality across the wrap and the dimer state
is NOT an eigenstate there -- so 16 sites is the minimal faithful choice.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402

LX = LY = 4  # 16-site torus


def _idx(x, y):
    return (x % LX) * LY + (y % LY)


def ss_bonds():
    """Return (dimer_bonds, square_bonds) for the 16-site SS torus.

    Sites sit on a 4x4 square lattice (all NN edges are J bonds).  Dimer (J')
    bonds are diagonals on the checkerboard of plaquettes (x+y even), oriented
    NE for x even and NW for x odd -- an orthogonal-dimer perfect matching.
    """
    square = set()
    for x in range(LX):
        for y in range(LY):
            for dx, dy in ((1, 0), (0, 1)):
                a, b = _idx(x, y), _idx(x + dx, y + dy)
                square.add((min(a, b), max(a, b)))
    dimer = set()
    for x in range(LX):
        for y in range(LY):
            if (x + y) % 2 == 0:
                if x % 2 == 0:                       # NE diagonal
                    a, b = _idx(x, y), _idx(x + 1, y + 1)
                else:                                # NW diagonal
                    a, b = _idx(x + 1, y), _idx(x, y + 1)
                dimer.add((min(a, b), max(a, b)))
    return sorted(dimer), sorted(square - dimer)


def ss_H(dimer, square, J=0.5, Jp=1.0):
    """H = J sum_square S.S + J' sum_dimer S.S on the 16-site torus."""
    sx, sy, sz = ed.spin_ops(LX * LY)

    def dot(i, j):
        return sx[i] @ sx[j] + sy[i] @ sy[j] + sz[i] @ sz[j]

    H = 0
    for (i, j) in dimer:
        H = H + Jp * dot(i, j)
    for (i, j) in square:
        H = H + J * dot(i, j)
    return H


def singlet_product_state(pairs, L):
    """Vector (2^L) for a product of two-site singlets (|up dn> - |dn up>)/sqrt2."""
    dim = 1 << L
    psi = np.zeros(dim, complex)
    for state in range(dim):
        amp = 1.0
        for (i, j) in pairs:
            bi = (state >> (L - 1 - i)) & 1
            bj = (state >> (L - 1 - j)) & 1
            if bi == 0 and bj == 1:
                amp *= 1.0 / np.sqrt(2)
            elif bi == 1 and bj == 0:
                amp *= -1.0 / np.sqrt(2)
            else:
                amp = 0.0
                break
        psi[state] = amp
    return psi


def dimer_energy(Jp=1.0):
    """Exact dimer-singlet energy E = -(3/4) J' N_dimer on the 16-site cluster."""
    dimer, _ = ss_bonds()
    return -0.75 * Jp * len(dimer)


def compute(J=0.5, Jp=1.0):
    """Shastry-Sutherland 16-site torus: exact dimer energy vs ED ground energy."""
    dimer, square = ss_bonds()
    e_dimer = -0.75 * Jp * len(dimer)
    e0 = ed.ground_energy(ss_H(dimer, square, J, Jp))
    return {
        "e_dimer_exact": e_dimer,                       # -(3/4) J' N_dimer, all J/J'
        "e_dimer_per_spin": e_dimer / (LX * LY),        # -3J'/8
        "e0_ed": e0,
        "dimer_is_ground_state": bool(abs(e0 - e_dimer) < 1e-8),
    }


def self_test():
    dimer, square = ss_bonds()
    L = LX * LY
    # sanity: the dimer bonds form a perfect matching of all 16 sites.
    covered = [0] * L
    for (i, j) in dimer:
        covered[i] += 1
        covered[j] += 1
    assert covered == [1] * L and len(dimer) == 8

    psi = singlet_product_state(dimer, L)
    psi /= np.linalg.norm(psi)
    e_dimer = -0.75 * len(dimer)   # J' = 1

    # anchor 1: dimer-singlet product is an EXACT eigenstate for ALL J/J'
    # (operator-level residual), with energy -(3/4) N_dimer independent of J.
    for J in (0.3, 0.5, 0.9):
        H = ss_H(dimer, square, J, 1.0)
        Hpsi = H @ psi
        E = np.vdot(psi, Hpsi).real
        assert abs(E - e_dimer) < 1e-12, J
        assert np.linalg.norm(Hpsi - E * psi) < 1e-12, J

    # anchor 2 (GROUND TRUTH): the J=0 limit is isolated dimers -- the dimer
    # product is the UNIQUE ground state, E0 = -(3/4) N_dimer.
    H0 = ss_H(dimer, square, 0.0, 1.0)
    assert abs(ed.ground_energy(H0) - e_dimer) < 1e-10
    assert ed.ground_states(H0) == 1

    # anchor 3: the dimer state IS the ED ground state at J/J' = 0.5 (dimer
    # phase) but NOT at J/J' = 0.9 (the cluster level crossing is near 0.667,
    # close to the thermodynamic dimer->plaquette boundary J/J' = 0.675).
    assert abs(ed.ground_energy(ss_H(dimer, square, 0.5, 1.0)) - e_dimer) < 1e-8
    assert ed.ground_energy(ss_H(dimer, square, 0.9, 1.0)) < e_dimer - 1e-6


if __name__ == "__main__":
    oracle_main(compute, {"J": (float, 0.5), "Jp": (float, 1.0)})
