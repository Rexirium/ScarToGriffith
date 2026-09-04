"""eta-pairing Hubbard oracle: H = -t Σ_{iσ}(c†_{iσ}c_{i+1σ}+h.c.) + U Σ_i n_{i↑}n_{i↓}.

T5 (frustration-free / exact eigenstates), Tier C — but the exact states here are
exact EXCITED eigenstates, *not* the ground state.  On a bipartite lattice the
staggered pair operator
    η† = Σ_j (−1)^j c†_{j↑} c†_{j↓}
obeys the exact OPERATOR identity  [H, η†] = U η†  (derived below; NO chemical
potential and NO −U/2 particle-hole shift are included in H, which is what makes
the constant clean).  Consequences, all exact:

  * η†^m|vac⟩ is an eigenstate with energy E_m = m·U  (H|vac⟩ = 0, so
    H η†^m|vac⟩ = mU η†^m|vac⟩ by induction from the commutator).  These sit at
    positive energy mU while the true ground state of the same (N↑,N↓)=(m,m)
    sector is far below — hence exact *excited* states (identity-proof anchor).
  * norm:  ‖η†^m|vac⟩‖² = m! · L!/(L−m)!   (η is an su(2) pseudospin raising
    operator on the spin-L/2 lowest-weight state |vac⟩; Σ_{k<m}(k+1)(L−k)).
  * ODLRO (Yang):  in the normalised m-pair state, with η†_i=(−1)^i c†_{i↑}c†_{i↓},
        ⟨η†_i η_i⟩ = m/L                    (diagonal: pair density),
        ⟨η†_i η_j⟩ = m(L−m)/(L(L−1))  (i≠j)  (off-diagonal, r-independent → ODLRO).

Convention note: with the particle-hole-symmetric  U Σ(n↑−½)(n↓−½)  form the shift
becomes  [H_phs, η†] = 0  (η-states join the SO(4) degenerate multiplet); with a
−μN term the shift is (U−2μ)η† because [N,η†]=2η†.  This card uses the plain
H above → clean shift U.  Bipartite chain: PBC ring, L even.
"""
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402


def _hubbard_ops(L, t=1.0, U=4.0):
    """Spinful Hubbard on an L-site PBC ring as a 2L-site Jordan–Wigner chain.

    Site ordering 0↑ … (L-1)↑, 0↓ … (L-1)↓ (the validated hubbard-1d-lieb-wu
    species-block layout).  Returns H (csr), and the η†/η pair operators plus
    per-site η†_i, η_i and number operators for correlators.
    """
    c, cd = ed.fermion_ops(2 * L)
    n = [cd[k] @ c[k] for k in range(2 * L)]
    up = lambda i: i
    dn = lambda i: L + i

    H = 0
    for i in range(L):
        j = (i + 1) % L
        for site in (up, dn):
            a, b = site(i), site(j)
            H = H - t * (cd[a] @ c[b] + cd[b] @ c[a])
    for i in range(L):
        H = H + U * (n[up(i)] @ n[dn(i)])

    # local staggered pair operators η†_i = (−1)^i c†_{i↑}c†_{i↓}
    eta_i_dag = [((-1.0) ** i) * (cd[up(i)] @ cd[dn(i)]) for i in range(L)]
    eta_i = [op.conj().T.tocsr() for op in eta_i_dag]
    eta_dag = sum(eta_i_dag)                       # η† = Σ_i η†_i
    eta = eta_dag.conj().T.tocsr()
    return H.tocsr(), eta_dag.tocsr(), eta.tocsr(), eta_i_dag, eta_i, n, up, dn


def eta_state(L, m, t=1.0, U=4.0):
    """Normalised η†^m|vac⟩ as a dense 4^L vector (vac = empty lattice)."""
    _, eta_dag, *_ = _hubbard_ops(L, t, U)
    dim = 4 ** L
    psi = np.zeros(dim, complex)
    psi[0] = 1.0                                    # |vac⟩ = all sites empty
    for _ in range(m):
        psi = eta_dag @ psi
    nrm = np.linalg.norm(psi)
    return psi / nrm, nrm


def eta_norm_sq(L, m):
    """Closed form ‖η†^m|vac⟩‖² = m! · L!/(L−m)!."""
    return math.factorial(m) * math.factorial(L) // math.factorial(L - m)


def odlro_offdiag(L, m):
    """Yang closed form ⟨η†_i η_j⟩ (i≠j) = m(L−m)/(L(L−1)) in the m-pair state."""
    return m * (L - m) / (L * (L - 1))


def sector_ground_energy(L, nup, ndn, t=1.0, U=4.0):
    """ED ground energy in the fixed (N↑,N↓) sector (dense sub-block)."""
    H, _, _, _, _, n, up, dn = _hubbard_ops(L, t, U)
    Nup = np.array(sum(n[up(i)] for i in range(L)).diagonal()).real
    Ndn = np.array(sum(n[dn(i)] for i in range(L)).diagonal()).real
    idx = np.where((np.round(Nup) == nup) & (np.round(Ndn) == ndn))[0]
    sub = H[np.ix_(idx, idx)].toarray()
    return float(np.linalg.eigvalsh(sub)[0])


def compute(L=6, U=4.0, m=2):
    """eta-pairing Hubbard: exact excited η-pairing eigenstate quantities (t=1)."""
    _, nrm = eta_state(L, m, U=U)
    return {
        "eta_shift": U,                                   # [H,η†] = U η†
        "energy_m_pair": m * U,                           # E_m = mU (exact, excited)
        "norm_sq_closed": eta_norm_sq(L, m),              # m! L!/(L-m)!
        "norm_sq_direct": nrm ** 2,
        "odlro_offdiag_closed": odlro_offdiag(L, m),      # m(L-m)/(L(L-1))
        "pair_density_diag": m / L,                       # ⟨η†_i η_i⟩
        "sector_gs_energy": sector_ground_energy(L, m, m, U=U),  # << mU (excited)
    }


def self_test():
    L, U = 6, 4.0

    # anchor 1 (GROUND TRUTH, operator identity): [H, η†] = U η† to 1e-12, and the
    #   kinetic part commutes with η† by itself (bipartite staggering) — the U-term
    #   alone carries the shift.  Checked at L=4 and L=6.
    for Lc in (4, 6):
        H, eta_dag, _, _, _, nops, up, dn = _hubbard_ops(Lc, U=U)
        comm = H @ eta_dag - eta_dag @ H
        assert sp.linalg.norm(comm - U * eta_dag) < 1e-12, Lc
        # kinetic-only commutator vanishes on the bipartite ring (L even)
        c, cd = ed.fermion_ops(2 * Lc)
        T = 0
        for i in range(Lc):
            j = (i + 1) % Lc
            for s in (lambda x: x, lambda x: Lc + x):
                T = T - (cd[s(i)] @ c[s(j)] + cd[s(j)] @ c[s(i)])
        assert sp.linalg.norm((T @ eta_dag - eta_dag @ T)) < 1e-12, Lc
        # U-term commutator == U η†
        Uterm = U * sum(nops[up(i)] @ nops[dn(i)] for i in range(Lc))
        assert sp.linalg.norm((Uterm @ eta_dag - eta_dag @ Uterm) - U * eta_dag) < 1e-12

    # anchor 2: η†^m|vac⟩ is an OPERATOR-level exact eigenstate, E_m = mU, m=1,2,3.
    H, _, _, _, _, _, _, _ = _hubbard_ops(L, U=U)
    for m in (1, 2, 3):
        psi, _ = eta_state(L, m, U=U)
        Hpsi = H @ psi
        E = np.vdot(psi, Hpsi).real
        assert abs(E - m * U) < 1e-12, m
        assert np.linalg.norm(Hpsi - m * U * psi) < 1e-12, m       # operator residual

    # anchor 3: norm formula ‖η†^m|vac⟩‖² = m! L!/(L−m)!  vs direct, m=1..L.
    for m in range(1, L + 1):
        _, nrm = eta_state(L, m, U=U)
        assert abs(nrm ** 2 - eta_norm_sq(L, m)) < 1e-9, m
    assert eta_norm_sq(L, 1) == L                     # η†|vac⟩: L doublon configs
    assert eta_norm_sq(L, 2) == 2 * L * (L - 1)

    # anchor 4: ODLRO — diagonal m/L and off-diagonal m(L−m)/(L(L−1)) closed forms
    #   vs direct expectation, and off-diagonal is r-INDEPENDENT (long-range).
    _, eta_dag, eta, eta_i_dag, eta_i, _, _, _ = _hubbard_ops(L, U=U)
    for m in (2, 3):
        psi, _ = eta_state(L, m, U=U)
        diag = np.vdot(psi, eta_i_dag[0] @ (eta_i[0] @ psi)).real
        assert abs(diag - m / L) < 1e-10, m
        vals = [np.vdot(psi, eta_i_dag[0] @ (eta_i[j] @ psi)).real
                for j in range(1, L)]
        assert max(vals) - min(vals) < 1e-10          # r-independent
        assert abs(vals[0] - odlro_offdiag(L, m)) < 1e-10, m
        # consistency: Σ_{i≠j}⟨η†_iη_j⟩ = ⟨η†η⟩ − m
        full = np.vdot(psi, eta_dag @ (eta @ psi)).real
        assert abs((full - m) - L * (L - 1) * odlro_offdiag(L, m)) < 1e-9, m

    # anchor 5 (IDENTITY-PROOF: these are EXCITED, not ground, states).  In the
    #   (N↑,N↓)=(m,m) sector at U=4 the ED ground energy is strictly, and by a wide
    #   margin, below the η-state energy mU (which is positive; the GS is negative).
    for m in (1, 2, 3):
        e_gs = sector_ground_energy(L, m, m, U=U)
        assert e_gs < m * U - 1.0, (m, e_gs)          # excited by a wide margin
        assert e_gs < 0.0                              # GS is bound (negative)


if __name__ == "__main__":
    oracle_main(compute, {"L": (int, 6), "U": (float, 4.0), "m": (int, 2)})
