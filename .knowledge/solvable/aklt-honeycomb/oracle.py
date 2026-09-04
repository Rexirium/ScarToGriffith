"""Honeycomb (spin-3/2) AKLT oracle: H = Σ_bonds P^{(S=3)}_bond  (P scope).

T5 (frustration-free / exact ground state), Tier C.  Each site of the honeycomb
carries a spin-3/2 (d=4, coordination z=3).  On every bond the parent Hamiltonian
is the projector onto the pair's total-spin-3 channel, built from the pair Casimir:
    S_i·S_j has eigenvalues  x_{S'} = [S'(S'+1) − 2·(15/4)]/2  →
        S'=0:−15/4,  S'=1:−11/4,  S'=2:−3/4,  S'=3:+9/4,
    P^{(3)} = (S_i·S_j + 15/4)(S_i·S_j + 11/4)(S_i·S_j + 3/4) / 90 .
Since every P^{(3)} ≥ 0, H ≥ 0 (frustration-free floor).  The honeycomb valence-bond
solid (each site = symmetric combination of three virtual spin-½'s, one per bond;
each bond a singlet) has zero total-spin-3 weight on every bond, so H annihilates it
→ it is an exact E=0 ground state.

SCRIPTABLE OBJECT (this card): the smallest honeycomb torus feasible for spin-3/2 ED
— the 2×2-unit-cell cluster = 8 sites, dim 4^8 = 65536, 12 bonds, every site degree 3
(no self-loops or multi-edges).  We (1) verify the frustration-free floor E₀ = 0
(E ≥ 0) by sparse ED and record the observed ground-state degeneracy and spectral
gap on this cluster, and (2) CONSTRUCT the explicit VBS/PEPS state and show H
annihilates it at operator level (‖H|ψ⟩‖ < 1e-10) and that it is the ED ground
state.  TABULATED (not scripted), with citations: the thermodynamic-limit
correlations and the PROVEN nonzero spectral gap [@PomataWei2020; @LemmSandvikWang2020].
"""
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402

# ---- spin-3/2 local operators (d=4 basis S^z = +3/2,+1/2,−1/2,−3/2) ----
_S = 1.5
_MVALS = [1.5, 0.5, -0.5, -1.5]
SZ = np.diag(_MVALS).astype(complex)
SP = np.zeros((4, 4), complex)
for _a in range(3):
    _m = _MVALS[_a + 1]
    SP[_a, _a + 1] = sqrt(_S * (_S + 1) - _m * (_m + 1))
SM = SP.conj().T
SX = 0.5 * (SP + SM)
SY = 0.5j * (SM - SP)
_ID4 = sp.identity(4, format="csr", dtype=complex)


def _site_op(op, i, N):
    mats = [_ID4] * N
    mats[i] = sp.csr_matrix(op)
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


def spin32_ops(N):
    return ([_site_op(SX, i, N) for i in range(N)],
            [_site_op(SY, i, N) for i in range(N)],
            [_site_op(SZ, i, N) for i in range(N)])


def pair_projector_S3():
    """16×16 projector onto total-spin-3 for two spin-3/2's (from the Casimir)."""
    sx, sy, sz = SX, SY, SZ
    SS = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    I16 = np.eye(16)
    return (SS + 3.75 * I16) @ (SS + 2.75 * I16) @ (SS + 0.75 * I16) / 90.0


# ---- 2×2 honeycomb torus (8 sites, 12 bonds) ----
def honeycomb_2x2_bonds():
    """A(x,y)=2(2x+y), B(x,y)=2(2x+y)+1; each A bonds to B in-cell and two shifts."""
    def A(x, y):
        return 2 * (2 * x + y)

    def B(x, y):
        return 2 * (2 * x + y) + 1

    bonds = []
    for x in range(2):
        for y in range(2):
            bonds.append((A(x, y), B(x, y)))
            bonds.append((A(x, y), B((x - 1) % 2, y)))
            bonds.append((A(x, y), B(x, (y - 1) % 2)))
    return bonds


N_SITES = 8


def honeycomb_H(bonds=None):
    """H = Σ_bonds P^{(3)}_bond on the 8-site torus (sparse 65536×65536)."""
    bonds = bonds or honeycomb_2x2_bonds()
    Sx, Sy, Sz = spin32_ops(N_SITES)
    Idn = sp.identity(4 ** N_SITES, format="csr", dtype=complex)
    H = 0
    for i, j in bonds:
        SS = Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j]
        H = H + (SS + 3.75 * Idn) @ (SS + 2.75 * Idn) @ (SS + 0.75 * Idn) / 90.0
    return H.tocsr()


def vbs_state(bonds=None):
    """Explicit honeycomb VBS/PEPS state on the 8-site torus (dense 4^8 vector).

    Each site = symmetric projection of three virtual spin-½ legs (one per bond);
    each bond = singlet (↑↓−↓↑)/√2.  Contracted by summing the 2^{#bonds} singlet
    branches; local symmetrizer weight 1/√C(3, n_up).
    """
    bonds = bonds or honeycomb_2x2_bonds()
    inc = {i: [] for i in range(N_SITES)}
    for bidx, (i, j) in enumerate(bonds):
        inc[i].append(bidx)
        inc[j].append(bidx)
    psi = np.zeros(4 ** N_SITES)
    for combo in range(2 ** len(bonds)):
        legval = {}
        w = 1.0
        for bidx, (i, j) in enumerate(bonds):
            c = (combo >> bidx) & 1
            si, sj = inc[i].index(bidx), inc[j].index(bidx)
            if c == 0:                                   # ↑_i ↓_j, +1/√2
                legval[(i, si)], legval[(j, sj)] = 1, 0
                w *= 1.0 / sqrt(2.0)
            else:                                        # ↓_i ↑_j, −1/√2
                legval[(i, si)], legval[(j, sj)] = 0, 1
                w *= -1.0 / sqrt(2.0)
        idx = 0
        for i in range(N_SITES):
            nup = sum(legval[(i, sl)] for sl in range(3))
            w *= 1.0 / sqrt(comb(3, nup))
            idx += (3 - nup) * (4 ** (N_SITES - 1 - i))
        psi[idx] += w
    return psi / np.linalg.norm(psi)


def compute(L_arg=2):
    """Honeycomb spin-3/2 AKLT on the 2×2 torus: frustration-free floor + VBS check."""
    if L_arg != 2:
        raise ValueError("only the 2×2-unit-cell honeycomb torus (8 sites) is scripted")
    bonds = honeycomb_2x2_bonds()
    H = honeycomb_H(bonds)
    w = np.sort(eigsh(H, k=6, which="SA", return_eigenvectors=False))
    gsd = int(np.sum(np.abs(w - w[0]) < 1e-8))
    psi = vbs_state(bonds)
    Hpsi = H @ psi
    return {
        "n_sites": N_SITES,
        "n_bonds": len(bonds),
        "e0": float(w[0]),                               # frustration-free floor ≈ 0
        "spectrum_floor_nonneg": bool(w[0] > -1e-8),
        "gsd_observed": gsd,                             # 1 on the torus
        "gap_observed": float(w[gsd] - w[0]),            # numerical, this cluster
        "vbs_energy": float(np.vdot(psi, Hpsi).real),    # ≈ 0
        "vbs_residual": float(np.linalg.norm(Hpsi)),     # ‖H|ψ⟩‖ ≈ 0
    }


def self_test():
    bonds = honeycomb_2x2_bonds()

    # anchor 1 (geometry + projector): 12 unique bonds, every site degree 3, no
    #   multi-edges; and P^{(3)} is a genuine rank-7 projector (S=3 sector of two
    #   spin-3/2's has 2·3+1=7 states) built from the pair Casimir.
    assert len(bonds) == 12
    assert len(set(tuple(sorted(b)) for b in bonds)) == 12    # no multi-edges
    deg = {i: 0 for i in range(N_SITES)}
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1
    assert all(d == 3 for d in deg.values())
    P3 = pair_projector_S3()
    assert np.linalg.norm(P3 @ P3 - P3) < 1e-10               # idempotent
    assert abs(np.trace(P3) - 7.0) < 1e-10                    # rank 7 = dim(S=3)
    # S_i·S_j spectrum: the four Casimir eigenvalues with the right multiplicities
    SS = (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ))
    ev = np.round(np.sort(np.linalg.eigvalsh(SS)), 6)
    from collections import Counter
    cnt = Counter(ev.tolist())
    assert cnt[-3.75] == 1 and cnt[-2.75] == 3 and cnt[-0.75] == 5 and cnt[2.25] == 7

    # anchor 2 (FRUSTRATION-FREE FLOOR, ground truth): E₀ = 0 to 1e-10, spectrum
    #   floor E ≥ 0, and the observed GS is unique with a positive gap on this torus.
    H = honeycomb_H(bonds)
    wv, vv = eigsh(H, k=6, which="SA")
    order = np.argsort(wv)
    w, vv = wv[order], vv[:, order]
    assert w[0] > -1e-8                                       # frustration-free ≥ 0
    assert abs(w[0]) < 1e-10                                  # zero-energy GS exists
    gsd = int(np.sum(np.abs(w - w[0]) < 1e-8))
    assert gsd == 1                                           # unique on the torus
    assert w[1] - w[0] > 1e-3                                 # positive gap (observed)

    # anchor 3 (EXPLICIT VBS, operator-level): the constructed honeycomb PEPS state
    #   is annihilated by H (‖H|ψ⟩‖ < 1e-10) and equals the ED ground state.
    psi = vbs_state(bonds)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    Hpsi = H @ psi
    assert abs(np.vdot(psi, Hpsi).real) < 1e-10
    assert np.linalg.norm(Hpsi) < 1e-10                       # operator residual
    assert abs(abs(np.vdot(vv[:, 0], psi)) - 1.0) < 1e-8      # = ED ground state


if __name__ == "__main__":
    oracle_main(compute, {"L_arg": (int, 2)})
