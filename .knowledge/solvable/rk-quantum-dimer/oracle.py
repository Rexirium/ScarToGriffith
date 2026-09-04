"""Rokhsar-Kivelson quantum dimer model on the L x L square-lattice torus.

    H_RK = sum_plaquettes [ -t (|=><‖| + |‖><=|) + v (|=><=| + |‖><‖|) ]

acting on the Hilbert space spanned by the hard-core dimer coverings (perfect
matchings) of the lattice.  For an elementary plaquette p, |=> is the state with
its two HORIZONTAL edges both occupied by dimers and |‖> the state with its two
VERTICAL edges occupied; a plaquette is "flippable" in a covering iff it is in
one of these two configurations, and the off-diagonal term flips =<->‖.  At the
RK point t = v = 1 each plaquette term factorises as a projector

    h_p = (|=> - |‖>)(<=| - <‖|)  =  v(|=><=|+|‖><‖|) - t(|=><‖|+|‖><=|)   (t=v)

which is manifestly positive-semidefinite (rank-1, PSD), so H_RK = sum_p h_p >= 0
is FRUSTRATION-FREE.  The equal-weight superposition of the coverings within any
flip-connected set is annihilated term-by-term (each flip pairs a =-covering with
its ‖-partner at equal amplitude, so <=|-<‖| gives 1-1=0) and is therefore an
exact E=0 ground state.

T5 (frustration-free / exact eigenstates), Tier C.  ONLY the RK point t=v is
solved here: the equal-weight RVB ground states and the RK<->classical-dimer
mapping are exact.  Away from t=v NOTHING in this card is exact -- the QDM phase
diagram (columnar / plaquette / staggered order on the square lattice; the
deconfined U(1) liquid on the triangular lattice) is quoted from the literature,
not derived.

Cluster: the 4x4 torus (16 sites, 32 edges) has exactly 272 dimer coverings
(verified three independent ways: two backtracking enumerations with different
site orders, and the permanent of the 8x8 biadjacency matrix via Ryser).  This
is a PERIODIC lattice -- no direct count transfer from the wave-2 `dimer-kasteleyn`
card, which counts OPEN-boundary tilings (a different graph); the two share only
the perfect-matching methodology.
"""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402

L = 4  # 4x4 torus


def _sid(x, y):
    return (x % L) * L + (y % L)


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _neighbors(sid):
    x, y = divmod(sid, L)
    return [_sid(x + 1, y), _sid(x - 1, y), _sid(x, y + 1), _sid(x, y - 1)]


def dimer_coverings(order=None):
    """All dimer coverings (perfect matchings) of the L x L torus.

    Each covering is a frozenset of edges (sorted site-id pairs).  `order` is the
    site-visitation order for the backtracking search (defaults to 0..L^2-1); the
    resulting SET of coverings is independent of the order -- exploited in
    self_test for an independent cross-check.
    """
    sites = list(range(L * L)) if order is None else list(order)
    covers = []

    def rec(matched, chosen):
        u = None
        for s in sites:
            if s not in matched:
                u = s
                break
        if u is None:
            covers.append(frozenset(chosen))
            return
        for v in _neighbors(u):
            if v in matched:
                continue
            rec(matched | {u, v}, chosen | {_edge(u, v)})

    rec(set(), set())
    return covers


def count_coverings_permanent():
    """Independent covering count: permanent of the 8x8 biadjacency matrix (Ryser).

    The torus is bipartite (A: (x+y) even, B: odd).  The number of perfect
    matchings equals per(M), M[a,b]=1 iff A-site a adjacent to B-site b.  On the
    4x4 torus there are no double edges, so M is 0/1 and per(M) is the exact count.
    """
    A = [s for s in range(L * L) if (s // L + s % L) % 2 == 0]
    B = [s for s in range(L * L) if (s // L + s % L) % 2 == 1]
    bi = {s: i for i, s in enumerate(B)}
    n = len(A)
    M = np.zeros((n, n))
    for i, a in enumerate(A):
        for nb in _neighbors(a):
            M[i, bi[nb]] += 1.0
    # Ryser's formula for the permanent
    total = 0.0
    for k in range(1, n + 1):
        for cols in combinations(range(n), k):
            prod = 1.0
            for i in range(n):
                prod *= M[i, list(cols)].sum()
            total += ((-1) ** k) * prod
    return int(round(((-1) ** n) * total))


def _plaquettes():
    """Elementary plaquettes as (horizontal-pair, vertical-pair) edge frozensets.

    Plaquette with lower-left corner (x,y) has corners (x,y),(x+1,y),(x,y+1),
    (x+1,y+1); |=> = the two horizontal edges, |‖> = the two vertical edges.
    """
    out = []
    for x in range(L):
        for y in range(L):
            a, b = _sid(x, y), _sid(x + 1, y)
            c, d = _sid(x, y + 1), _sid(x + 1, y + 1)
            hor = frozenset({_edge(a, b), _edge(c, d)})
            ver = frozenset({_edge(a, c), _edge(b, d)})
            out.append((hor, ver))
    return out


def winding(cov):
    """Topological winding numbers (W_x, W_y) of a covering.

    W_x counts horizontal dimers crossing the vertical cut between columns 0 and 1
    with sign (-1)^(x_left + y); W_y counts vertical dimers crossing the horizontal
    cut between rows 0 and 1 with sign (-1)^(x + y_low).  Both are independent of
    which cut column/row is chosen (verified in self_test) -- the defining property
    of a topological invariant -- and are preserved by every plaquette flip, so
    H_RK is block-diagonal by sector.
    """
    wx = wy = 0
    for (a, b) in cov:
        ax, ay = divmod(a, L)
        bx, by = divmod(b, L)
        if ay == by:  # horizontal edge (differs in x)
            if {ax, bx} == {0, 1}:  # crosses the 0|1 vertical cut (not the wrap)
                wx += 1 if (0 + ay) % 2 == 0 else -1
        else:         # vertical edge (differs in y)
            if {ay, by} == {0, 1}:
                wy += 1 if (ax + 0) % 2 == 0 else -1
    return (wx, wy)


def _flip(cov, hor, ver):
    """If plaquette (hor,ver) is flippable in cov, return the flipped covering."""
    if hor <= cov:
        return (cov - hor) | ver
    if ver <= cov:
        return (cov - ver) | hor
    return None


def rk_hamiltonian(covs, t=1.0, v=1.0):
    """Dense H_RK in the covering basis; returns (H, index) with index: cov->row."""
    plaqs = _plaquettes()
    idx = {c: i for i, c in enumerate(covs)}
    n = len(covs)
    H = np.zeros((n, n))
    for c in covs:
        i = idx[c]
        for (hor, ver) in plaqs:
            if hor <= c or ver <= c:      # flippable
                H[i, i] += v
                cf = _flip(c, hor, ver)
                H[idx[cf], i] += -t
    return H, idx


def equal_weight_state(covs, members, idx):
    """Normalised equal-weight superposition of `members` in the covering basis."""
    psi = np.zeros(len(covs))
    for c in members:
        psi[idx[c]] = 1.0
    return psi / np.linalg.norm(psi)


def sectors(covs):
    """Group coverings by (W_x, W_y); returns {sector: [coverings]}."""
    out = {}
    for c in covs:
        out.setdefault(winding(c), []).append(c)
    return out


def _flip_components(members):
    """Number of plaquette-flip-connected components within a covering set."""
    plaqs = _plaquettes()
    mem = set(members)
    seen = set()
    comps = 0
    for start in members:
        if start in seen:
            continue
        comps += 1
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for (hor, ver) in plaqs:
                nb = _flip(cur, hor, ver)
                if nb is not None and nb in mem and nb not in seen:
                    stack.append(nb)
    return comps


def ground_state_degeneracy(covs):
    """Exact zero-energy count = number of flip-connected components (summed over
    sectors).  Each frozen (non-flippable) covering is its own component and an
    exact E=0 eigenstate; each resonating cluster contributes one equal-weight GS.
    """
    return sum(_flip_components(m) for m in sectors(covs).values())


def compute(L_arg=4):
    """RK quantum dimer model on the 4x4 torus: covering/sector/GS structure."""
    if L_arg != 4:
        raise ValueError(
            f"only L=4 implemented (got --L_arg {L_arg}): the covering "
            "enumeration, winding classification, and all card anchors are "
            "built for the 4x4 torus")
    covs = dimer_coverings()
    H, idx = rk_hamiltonian(covs)
    evals = np.linalg.eigvalsh(H)
    sec = sectors(covs)
    return {
        "n_coverings": len(covs),
        "n_winding_sectors": len(sec),
        "gsd_zero_modes": int(np.sum(np.abs(evals) < 1e-9)),
        "gsd_flip_components": ground_state_degeneracy(covs),
        "spectrum_floor": float(evals[0]),
    }


def self_test():
    covs = dimer_coverings()
    # anchor 1 (GROUND TRUTH): the 4x4 torus has exactly 272 coverings, agreed by
    # three independent methods -- two backtracking orders + the Ryser permanent.
    assert len(covs) == 272, len(covs)
    covs_b = dimer_coverings(order=sorted(range(L * L),
                                          key=lambda s: (s % L, s // L)))
    assert set(covs) == set(covs_b)               # covering SET is order-independent
    assert count_coverings_permanent() == 272     # permanent of biadjacency

    # every covering is a genuine perfect matching (each site covered once)
    for c in covs:
        deg = [0] * (L * L)
        for (a, b) in c:
            deg[a] += 1
            deg[b] += 1
        assert deg == [1] * (L * L)

    # anchor 2: winding numbers are cut-independent (topological).  Recompute
    # W_x, W_y at EVERY cut column/row and require a single value per covering.
    for c in covs:
        wxs, wys = set(), set()
        for cut in range(L):
            wx = wy = 0
            for (a, b) in c:
                ax, ay = divmod(a, L); bx, by = divmod(b, L)
                if ay == by and {ax, bx} == {cut, (cut + 1) % L}:
                    wx += 1 if (cut + ay) % 2 == 0 else -1
                if ax == bx and {ay, by} == {cut, (cut + 1) % L}:
                    wy += 1 if (ax + cut) % 2 == 0 else -1
            wxs.add(wx); wys.add(wy)
        assert len(wxs) == 1 and len(wys) == 1, c      # cut-independent
        assert (wxs.pop(), wys.pop()) == winding(c)

    H, idx = rk_hamiltonian(covs)
    assert np.allclose(H, H.T)

    # anchor 3: PROJECTOR FORM / frustration-free.  On each FLIPPABLE plaquette,
    # restricted to the 2-dim space {|c>, |flip(c)>} of a flippable covering and
    # its partner, the term is the rank-1 PSD operator (|=>-|‖>)(<=|-<‖|).  H is
    # the direct sum of these rank-1 blocks over all flippable pairs, so H >= 0
    # (frustration-free).  Check the spectrum floor >= 0 (operator-level: min
    # eigenvalue) AND rebuild H as the explicit sum of per-pair rank-1 projectors
    # (each flippable pair keyed once, by its =-config member).
    evals = np.linalg.eigvalsh(H)
    assert evals[0] > -1e-10, evals[0]                  # E >= 0, frustration-free
    Hp = np.zeros_like(H)
    for c in covs:
        for (hor, ver) in _plaquettes():
            if hor <= c:                                # c is the =-config; partner ver
                cf = (c - hor) | ver
                phi = np.zeros(len(covs))
                phi[idx[c]] = 1.0                       # |=> - |‖>
                phi[idx[cf]] = -1.0
                Hp += np.outer(phi, phi)                # rank-1 PSD block
    assert np.allclose(H, Hp, atol=1e-12)               # H == sum over pairs (rank-1)

    # anchor 4: block structure + equal-weight sector states are E=0.  In EACH of
    # the winding sectors the equal-weight superposition is annihilated by H
    # (operator-level residual), i.e. an exact zero-energy ground state.
    sec = sectors(covs)
    assert len(sec) == 13, len(sec)
    assert sum(len(m) for m in sec.values()) == 272
    for key, members in sec.items():
        psi = equal_weight_state(covs, members, idx)
        assert np.linalg.norm(H @ psi) < 1e-12, key      # E=0, residual
        assert abs(np.vdot(psi, H @ psi)) < 1e-12, key
    # winding is preserved by every flip -> H is block-diagonal by sector
    for c in covs:
        for (hor, ver) in _plaquettes():
            cf = _flip(c, hor, ver)
            if cf is not None:
                assert winding(cf) == winding(c)

    # anchor 5: GSD.  The exact zero-mode count from ED equals the number of
    # flip-connected components (= 17), NOT the number of winding sectors (= 13).
    # Precisely: 12 fully-FROZEN (no flippable plaquette) coverings in total --
    # 8 in the four (+-1,+-1) corner sectors (2 each, causing those sectors to
    # split into two components) and 4 staggered singletons (+-2,0)/(0,+-2) --
    # plus the 5 resonating sectors ((0,0) and (+-1,0),(0,+-1)), each one
    # flip-connected cluster: 12 + 5 = 17 zero modes.
    nz = int(np.sum(np.abs(evals) < 1e-9))
    assert nz == 17, nz
    assert ground_state_degeneracy(covs) == 17
    plaqs = _plaquettes()
    frozen = [c for c in covs
              if not any(hor <= c or ver <= c for (hor, ver) in plaqs)]
    assert len(frozen) == 12, len(frozen)          # 12 fully-frozen coverings
    frozen_by_sector = Counter(winding(c) for c in frozen)
    assert frozen_by_sector == {(1, 1): 2, (1, -1): 2, (-1, 1): 2, (-1, -1): 2,
                                (2, 0): 1, (-2, 0): 1, (0, 2): 1, (0, -2): 1}
    split = {k: _flip_components(m) for k, m in sec.items() if _flip_components(m) > 1}
    assert set(split) == {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    assert all(v == 2 for v in split.values())
    # the 5 resonating sectors are each a single flip-connected cluster
    for key in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        assert _flip_components(sec[key]) == 1, key

    # anchor 6: single-plaquette hand check.  An isolated plaquette (4 sites, one
    # square) has exactly two coverings {|=>, |‖>}; H = [[1,-1],[-1,1]] with the
    # flip matrix element <=|H|‖> = -t = -1, eigenvalues {0, 2}, GS = (|=>+|‖>)/√2.
    Hsq = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert Hsq[0, 1] == -1.0                                       # flip element -t
    w = np.linalg.eigvalsh(Hsq)
    assert np.allclose(w, [0.0, 2.0])
    gs = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.linalg.norm(Hsq @ gs) < 1e-12

    # anchor 7: RK <-> classical dimer ensemble.  The LOAD-BEARING check here is
    # that the full equal-weight state (over ALL 272 coverings, the sum of the
    # per-sector zero modes) is itself an E=0 zero mode -- that is nontrivial
    # physics.  Given that state, the correlator equality below is an ALGEBRAIC
    # identity, not an independent numeric cross-validation: dimer occupations
    # are diagonal in the covering basis, so for a uniform state over the same
    # covering set <psi|n_b n_b'|psi> == (1/Z) sum_cov n_b n_b' by construction.
    # It is asserted anyway as an implementation check (basis bookkeeping,
    # normalisation), i.e. the RK<->classical statement made mechanically visible.
    full = equal_weight_state(covs, covs, idx)
    assert np.linalg.norm(H @ full) < 1e-12                       # full RVB is E=0
    bonds = sorted({e for c in covs for e in c})
    occ = np.array([[1.0 if b in c else 0.0 for b in bonds] for c in covs])  # 272xNb
    w2 = full ** 2                                                # |amplitude|^2
    quantum = occ.T @ (w2[:, None] * occ)                        # <n_b n_b'>
    classical = occ.T @ occ / len(covs)                          # ensemble average
    assert np.max(np.abs(quantum - classical)) < 1e-12
    # and the single-bond dimer density is uniform 1/4 (each site has 4 bonds,
    # exactly one occupied: average bond occupation = (#sites/2)/(#bonds) = 8/32).
    assert abs(np.mean(np.diag(classical)) - 8.0 / 32.0) < 1e-12


if __name__ == "__main__":
    oracle_main(compute, {"L_arg": (int, 4)})
