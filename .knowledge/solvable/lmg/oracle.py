"""Lipkin–Meshkov–Glick oracle: H = -(J/N) S_x^2 - h S_z  (collective spin, J,h >= 0).

S_a = (1/2) sum_i sigma^a_i is the total (collective) spin; this is the isotropic
(gamma = 0) member of the LMG family H = -(J/N)(S_x^2 + gamma S_y^2) - h S_z.  Because
H is built only from the components of the total spin, [S^2, H] = 0 and H is block-
diagonal in the total-spin j sectors.  The maximal-spin sector j = N/2 (dimension N+1)
contains the ground state (ferromagnetic ordering: the two energy-lowering terms
-(J/N)S_x^2 and -h S_z are both minimized by the largest available spin length), so the
entire GS problem reduces to diagonalizing one (N+1)-dim block — exact at ANY finite N.

Diagonal trick: quantize along x.  In the |j, m_x> basis S_x is diagonal (= m_x) and S_z
acts as a ladder in m_x, so H is TRIDIAGONAL:
    diag_k     = -(J/N) m_x^2                       (m_x = -j .. j)
    offdiag_k  = -(h/2) sqrt(j(j+1) - m_x(m_x+1))   (m_x = -j .. j-1)
solved to machine precision by scipy.linalg.eigh_tridiagonal for N up to millions.

Thermodynamic per-spin energy by spin-coherent minimization (S = (N/2) n_hat):
    e_cl(theta) = -(J/4) sin^2(theta) - (h/2) cos(theta)   (phi = 0 optimal)
  broken     (h <= J):  e0 = -J/4 - h^2/(4J)   at cos(theta) = h/J
  symmetric  (h >= J):  e0 = -h/2               at theta = 0 (fully z-polarized)
continuous, first-derivative-matching at h = J with a second-derivative kink -> a
second-order QPT.  Finite-size gap at criticality closes as ~ N^{-1/3} (observed).
[@LipkinMeshkovGlick1965], [@DusuelVidal2005]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402


def collective_block(N, J, h, j=None):
    """Tridiagonal (diag, offdiag) of H = -(J/N)S_x^2 - h S_z in the spin-j x-basis.

    j defaults to the maximal-spin sector N/2 (dim N+1); pass a smaller j to build a
    lower total-spin sector at the SAME coupling J/N (used to check sector ordering).
    """
    if j is None:
        j = N / 2.0
    m = np.arange(-j, j + 1)                       # m_x eigenvalues, length 2j+1
    diag = -(J / N) * m ** 2
    mm = m[:-1]
    off = -(h / 2.0) * np.sqrt(j * (j + 1) - mm * (mm + 1))
    return diag, off


def block_lowest(N, J, h, k=2, j=None):
    """Lowest k eigenvalues of the collective spin-j block (default j = N/2)."""
    d, e = collective_block(N, J, h, j=j)
    k = min(k, len(d))
    return eigh_tridiagonal(d, e, select="i", select_range=(0, k - 1))[0]


def e0_collective(N, J, h):
    """Exact ground energy per spin from the (N+1)-dim j = N/2 block."""
    return block_lowest(N, J, h, k=1)[0] / N


def gap_collective(N, J, h):
    """Exact first excitation gap in the j = N/2 sector."""
    w = block_lowest(N, J, h, k=2)
    return float(w[1] - w[0])


def e0_thermo(J, h):
    """Thermodynamic-limit ground energy per spin (spin-coherent minimization)."""
    if h >= J:
        return -h / 2.0
    return -J / 4.0 - h * h / (4.0 * J)


def sx_order_thermo(J, h):
    """Symmetry-breaking order parameter <S_x>/N in the thermodynamic limit."""
    if h >= J:
        return 0.0
    return 0.5 * np.sqrt(1.0 - (h / J) ** 2)     # (1/2) sin(theta), cos(theta)=h/J


def _full_ed(N, J, h):
    """Brute-force 2^N ED of H = -(J/N)S_x^2 - h S_z; returns (E0, <S^2>_GS)."""
    sx, sy, sz = ed.spin_ops(N)
    Sx, Sy, Sz = sum(sx), sum(sy), sum(sz)
    H = -(J / N) * (Sx @ Sx) - h * Sz
    Hd = H.toarray()
    w, v = np.linalg.eigh(Hd)
    g = v[:, 0]
    S2 = (Sx @ Sx + Sy @ Sy + Sz @ Sz).toarray()
    return float(w[0]), float(np.real(g.conj() @ S2 @ g))


def compute(N=100, h=0.5, J=1.0):
    """LMG collective-spin exact quantities (j = N/2 block + thermodynamic limit)."""
    return {
        "e0_per_spin": e0_collective(N, J, h),
        "e0_thermodynamic": e0_thermo(J, h),
        "gap": gap_collective(N, J, h),
        "sx_order_thermo": sx_order_thermo(J, h),
        "phase": "symmetric" if h >= J else "broken",
    }


def self_test():
    # anchor 1 (IDENTITY PROOF): the j = N/2 collective block reproduces the full 2^N
    #   ED ground energy to machine precision, and the GS carries <S^2> = (N/2)(N/2+1),
    #   proving the ground state lies in the maximal-spin sector (not assumed).
    for N in (8, 10):
        for h in (0.5, 1.5):
            e_full, s2 = _full_ed(N, 1.0, h)
            e_blk = block_lowest(N, 1.0, h, k=1)[0]
            assert abs(e_full - e_blk) < 1e-12, (N, h, e_full, e_blk)
            jmax = (N / 2) * (N / 2 + 1)
            assert abs(s2 - jmax) < 1e-8, (N, h, s2, jmax)

    # anchor 2 (SECTOR ORDERING): at N = 8 the block minimum strictly rises as the
    #   total spin drops (j = 4 > 3 > 2) — the ferromagnetic reason the GS sits at
    #   j = N/2.  Same coupling J/N = 1/8 in every sector.
    for h in (0.0, 0.5, 1.5):
        e = [block_lowest(8, 1.0, h, k=1, j=j)[0] for j in (4, 3, 2)]
        assert e[0] < e[1] < e[2], (h, e)

    # anchor 3 (THERMODYNAMIC LIMIT): collective diag at large N converges to the
    #   spin-coherent closed form as O(1/N).  At N = 4000 both phases agree < 1e-4;
    #   the symmetric-phase deviation halves from N = 2000 -> 4000 (O(1/N) rate).
    for h in (0.5, 1.5):
        assert abs(e0_collective(4000, 1.0, h) - e0_thermo(1.0, h)) < 1e-4, h
    d2000 = abs(e0_collective(2000, 1.0, 1.5) - e0_thermo(1.0, 1.5))
    d4000 = abs(e0_collective(4000, 1.0, 1.5) - e0_thermo(1.0, 1.5))
    assert d4000 < d2000 < 3e-4 and d2000 / d4000 > 1.7, (d2000, d4000)

    # anchor 4 (QPT): closed form is continuous and C^1 but not C^2 at h = J (second-
    #   order transition).  First derivative de0/dh matches from both sides (= -1/2);
    #   the second derivative jumps (-1/(2J) below, 0 above).
    J = 1.0
    dh = 1e-6
    left = (e0_thermo(J, J - dh) - e0_thermo(J, J - 3 * dh)) / (2 * dh)
    right = (e0_thermo(J, J + 3 * dh) - e0_thermo(J, J + dh)) / (2 * dh)
    assert abs(left - right) < 1e-4 and abs(left + 0.5) < 1e-4, (left, right)

    # anchor 5 (CRITICAL GAP): at h = J the finite-size gap closes as ~ N^{-1/3}
    #   (observed exponent, Dusuel–Vidal).  Log-fit over N in {200,400,800,1600}
    #   lands within +/-0.05 of -1/3.
    Ns = np.array([200, 400, 800, 1600])
    gaps = np.array([gap_collective(int(N), 1.0, 1.0) for N in Ns])
    slope = np.polyfit(np.log(Ns), np.log(gaps), 1)[0]
    assert -1.0 / 3.0 - 0.05 < slope < -1.0 / 3.0 + 0.05, slope


if __name__ == "__main__":
    oracle_main(compute, {"N": (int, 100), "h": (float, 0.5), "J": (float, 1.0)})
