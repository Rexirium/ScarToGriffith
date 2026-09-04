"""Colorless spin-1 Motzkin chain (Bravyi-Caha-Movassagh-Nagaj-Shor) oracle.

Each site is a spin-1 with basis {u, 0, d} = step {+1, 0, -1}.  A length-n state
maps to a lattice walk (partial sums of the steps); a MOTZKIN WALK stays >= 0 and
returns to height 0.  The frustration-free Hamiltonian is a sum of local
projectors that make neighbouring walks resonate:

    H = sum_{j=1}^{n-1} Pi_{j,j+1}  +  |d><d|_1  +  |u><u|_n ,

    Pi_{j,j+1} = |D_1><D_1| + |D_2><D_2| + |D_3><D_3|,
    |D_1> = (|u0> - |0u>)/sqrt2,  |D_2> = (|0d> - |d0>)/sqrt2,
    |D_3> = (|ud> - |00>)/sqrt2.

The three bulk projectors annihilate the symmetric combinations |u0>+|0u>,
|0d>+|d0>, |ud>+|00>: they enforce that any two walks differing by one of the
local moves (u0<->0u, 0d<->d0, ud<->00) have EQUAL amplitude.  The boundary terms
|d><d|_1 and |u><u|_n forbid starting with a down-step or ending with an up-step,
pinning the walk to height 0 at both ends and >= 0 throughout.  Together these
make the UNIFORM superposition of all Motzkin walks the unique zero-energy ground
state; the number of Motzkin walks of length n is the Motzkin number M_n.

T5 (frustration-free / exact eigenstates), Tier C.  What is EXACT: the ground
state (uniform Motzkin superposition), E0 = 0, uniqueness, and the half-chain
entanglement entropy computed exactly from path-counting Schmidt weights.  The
excited spectrum is NOT closed-form -- the gap is only known to close
polynomially in 1/n [@BravyiEtAl2012] and the ~(1/2) ln n entanglement growth is
reported observed-as-observed, not as a fitted exponent.

Fredkin (spin-1/2) variant: the analogous frustration-free chain whose ground
state is the uniform superposition of DYCK paths (up/down steps only, Catalan
numbers).  It shares the ~(1/2) ln n half-chain entropy.  Here it is backed by
cheap combinatorics (Catalan counts + exact Schmidt entropy from ballot numbers),
no ED; see `fredkin_*` helpers and the card.  [@SalbergerKorepin2017]
"""
import sys
from collections import defaultdict
from math import comb, log
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402

U, ZERO, D = 0, 1, 2   # base-3 site states; step values +1, 0, -1
_STEP = {U: +1, ZERO: 0, D: -1}


# ------------------------------- Motzkin combinatorics -----------------------
def motzkin_number(n):
    """M_n via the standard recursion M_{m+1} = M_m + sum_{k=0}^{m-1} M_k M_{m-1-k}."""
    M = [1, 1]
    while len(M) <= n:
        m = len(M) - 1
        M.append(M[m] + sum(M[k] * M[m - 1 - k] for k in range(m)))
    return M[n]


def motzkin_walks(n):
    """All length-n Motzkin walks as digit tuples (prefix sums >= 0, end at 0)."""
    walks = []

    def rec(seq, h):
        if len(seq) == n:
            if h == 0:
                walks.append(tuple(seq))
            return
        for s in (U, ZERO, D):
            nh = h + _STEP[s]
            if nh >= 0:
                rec(seq + [s], nh)

    rec([], 0)
    return walks


def _height_counts(length):
    """{h: number of length-`length` walks 0->h staying >= 0} (steps +1,0,-1)."""
    cur = {0: 1}
    for _ in range(length):
        nxt = defaultdict(int)
        for h, c in cur.items():
            for ds in (+1, 0, -1):
                if h + ds >= 0:
                    nxt[h + ds] += c
        cur = dict(nxt)
    return cur


def half_chain_entropy(n):
    """Exact half-chain (cut at n/2) von Neumann entropy of the Motzkin GS.

    The Schmidt decomposition is diagonal in the height h at the cut: the weight
    of block h is p_h = N(n/2, h)^2 / M_n, where N(L, h) counts length-L walks
    0->h staying >= 0 (left half) and, by reflection, h->0 (right half).  These
    weights are exact combinatorics -- no ED -- so the entropy is available for
    any n.  (Uses sum_h N(n/2,h)^2 = M_n.)
    """
    assert n % 2 == 0
    counts = _height_counts(n // 2)
    Mn = motzkin_number(n)
    S = 0.0
    for h, c in counts.items():
        p = c * c / Mn
        if p > 0:
            S -= p * log(p)
    return S


# ------------------------------- Hamiltonian (ED) ----------------------------
def _bulk_projector():
    """9x9 sum of the three bulk rank-1 projectors, indexed (a*3+b)."""
    def ket(a, b):
        v = np.zeros(9)
        v[a * 3 + b] = 1.0
        return v
    P = np.zeros((9, 9))
    for t in (ket(U, ZERO) - ket(ZERO, U),
              ket(ZERO, D) - ket(D, ZERO),
              ket(U, D) - ket(ZERO, ZERO)):
        t = t / np.linalg.norm(t)
        P += np.outer(t, t)
    return P


def motzkin_H(n):
    """Sparse Hamiltonian H = sum_j Pi_{j,j+1} + |d><d|_1 + |u><u|_n on 3^n states."""
    d = 3
    P = _bulk_projector().reshape(3, 3, 3, 3)
    dim = d ** n
    H = sp.lil_matrix((dim, dim))
    pw = [d ** (n - 1 - k) for k in range(n)]
    for idx in range(dim):
        digits = [(idx // pw[k]) % d for k in range(n)]
        for j in range(n - 1):
            a, b = digits[j], digits[j + 1]
            for a2 in range(3):
                for b2 in range(3):
                    val = P[a2, b2, a, b]
                    if val != 0.0:
                        idx2 = idx + (a2 - a) * pw[j] + (b2 - b) * pw[j + 1]
                        H[idx2, idx] += val
        if digits[0] == D:
            H[idx, idx] += 1.0
        if digits[-1] == U:
            H[idx, idx] += 1.0
    return H.tocsr()


def motzkin_gs(n):
    """Uniform-superposition Motzkin ground state as a 3^n vector."""
    d = 3
    pw = [d ** (n - 1 - k) for k in range(n)]
    psi = np.zeros(d ** n)
    for w in motzkin_walks(n):
        psi[sum(w[k] * pw[k] for k in range(n))] = 1.0
    return psi / np.linalg.norm(psi)


# ------------------------------- Fredkin (Dyck) combinatorics -----------------
def catalan(m):
    """m-th Catalan number C_m = comb(2m, m)/(m+1)."""
    return comb(2 * m, m) // (m + 1)


def _dyck_height_counts(length):
    """{h: # length-`length` up/down walks 0->h staying >= 0} (steps +1,-1)."""
    cur = {0: 1}
    for _ in range(length):
        nxt = defaultdict(int)
        for h, c in cur.items():
            for ds in (+1, -1):
                if h + ds >= 0:
                    nxt[h + ds] += c
        cur = dict(nxt)
    return cur


def fredkin_half_entropy(n):
    """Exact half-chain entropy of the spin-1/2 Fredkin (Dyck) ground state.

    n even (length-n Dyck paths, C_{n/2} of them).  Schmidt weights by cut height
    h: p_h = D(n/2, h)^2 / C_{n/2}, D(L,h) = # length-L up/down walks 0->h >= 0.
    Pure combinatorics, no ED -- backs the tabulated Fredkin card row.
    """
    assert n % 2 == 0
    counts = _dyck_height_counts(n // 2)
    Cn = catalan(n // 2)
    S = 0.0
    for h, c in counts.items():
        p = c * c / Cn
        if p > 0:
            S -= p * log(p)
    return S


# ------------------------------- driver --------------------------------------
def compute(n=8):
    """Motzkin spin-1 chain at length n: GS residual, gap, entropy, M_n."""
    H = motzkin_H(n)
    gs = motzkin_gs(n)
    ev = np.sort(eigsh(H, k=2, which="SA", return_eigenvectors=False))
    return {
        "motzkin_number": motzkin_number(n),
        "gs_residual": float(np.linalg.norm(H @ gs)),
        "e0": float(ev[0]),
        "gap": float(ev[1] - ev[0]),                 # > 0  => unique GS
        "half_chain_entropy": half_chain_entropy(n),
    }


def self_test():
    # anchor 1: Motzkin numbers -- recursion values AND that the explicit walk
    # enumerator produces exactly M_n walks for n <= 10.
    assert [motzkin_number(k) for k in range(11)] == \
        [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188]
    for n in range(11):
        assert len(motzkin_walks(n)) == motzkin_number(n), n

    # anchor 2 (GROUND TRUTH): the uniform Motzkin superposition is an
    # operator-level exact E=0 ground state, and it is UNIQUE (second-lowest ED
    # eigenvalue > 0), for n in {4, 6, 8}.
    for n in (4, 6, 8):
        H = motzkin_H(n)
        gs = motzkin_gs(n)
        assert np.linalg.norm(H @ gs) < 1e-12, n            # H|GS> = 0, residual
        assert abs(np.vdot(gs, H @ gs)) < 1e-12, n
        ev = np.sort(eigsh(H, k=2, which="SA", return_eigenvectors=False))
        assert abs(ev[0]) < 1e-10, (n, ev[0])               # E0 = 0
        assert ev[1] > 1e-6, (n, ev[1])                     # unique (gap > 0)
        # the ED ground vector equals the combinatorial Motzkin state (overlap 1)
        _, vecs = eigsh(H, k=1, which="SA")
        assert abs(abs(np.vdot(vecs[:, 0], gs)) - 1.0) < 1e-8, n

    # anchor 3: half-chain Schmidt structure.  sum_h N(n/2,h)^2 == M_n exactly,
    # and the exact combinatorial entropy matches the entropy of the ED ground
    # state's reduced density matrix (cross-check of the Schmidt formula) at n=8.
    for n in (4, 6, 8, 10, 12):
        counts = _height_counts(n // 2)
        assert sum(c * c for c in counts.values()) == motzkin_number(n), n
    n = 8
    gs = motzkin_gs(n)
    half = n // 2
    psi = gs.reshape(3 ** half, 3 ** half)                  # bipartition at the cut
    sv = np.linalg.svd(psi, compute_uv=False)
    probs = sv ** 2
    probs = probs[probs > 1e-15]
    S_ed = float(-np.sum(probs * np.log(probs)))
    assert abs(S_ed - half_chain_entropy(n)) < 1e-10, (S_ed, half_chain_entropy(n))

    # anchor 4: entanglement entropy grows monotonically with n across
    # {4,6,8,10,12} (pure combinatorics -- no ED needed for the larger n).  The
    # growth tracks the ~(1/2) ln n law [@BravyiEtAl2012] (observed-as-observed;
    # not asserted as an exact coefficient).
    S = [half_chain_entropy(n) for n in (4, 6, 8, 10, 12)]
    assert all(S[i + 1] > S[i] for i in range(len(S) - 1)), S

    # anchor 5: Fredkin (spin-1/2 Dyck) combinatorics back the tabulated card.
    # Catalan counts, and monotone ~(1/2) ln n half-chain entropy.  Dyck paths
    # (no flat step) pin the midpoint height to the parity of n/2, so the
    # finite-size entropy oscillates BETWEEN parity classes but is monotone
    # WITHIN one -- assert monotone growth along n = 4,8,12,16,20 (n/2 even).
    assert [catalan(m) for m in range(8)] == [1, 1, 2, 5, 14, 42, 132, 429]
    for n in (4, 6, 8, 10, 12):
        counts = _dyck_height_counts(n // 2)
        assert sum(c * c for c in counts.values()) == catalan(n // 2), n
    Sf = [fredkin_half_entropy(n) for n in (4, 8, 12, 16, 20)]
    assert all(Sf[i + 1] > Sf[i] for i in range(len(Sf) - 1)), Sf


if __name__ == "__main__":
    oracle_main(compute, {"n": (int, 8)})
