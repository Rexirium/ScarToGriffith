"""Curie–Weiss (fully connected) transverse-field Ising oracle:
    H = -(J/N) sum_{i<j} sigma^z_i sigma^z_j - h sum_i sigma^x_i   (Pauli, J,h >= 0).

Every spin couples to every other with strength J/N (all-to-all) — the infinite-range
mean-field limit of the TFIM.  With the collective spin S_a = (1/2) sum_i sigma^a_i,
    sum_{i<j} sigma^z_i sigma^z_j = 2 S_z^2 - N/2,     sum_i sigma^x_i = 2 S_x,
so exactly
    H = -(2J/N) S_z^2 - 2h S_x + J/2 .
This is a collective (LMG-type) Hamiltonian: [S^2, H] = 0 and the ground state sits in
the maximal-spin sector j = N/2 (dim N+1), so it is EXACTLY diagonalizable at ANY finite
N.  Quantizing along z makes S_z^2 diagonal and S_x a ladder -> H is TRIDIAGONAL:
    diag_k    = -(2J/N) m_z^2 + J/2                 (m_z = -j .. j)
    offdiag_k = -h sqrt(j(j+1) - m_z(m_z+1))        (m_z = -j .. j-1)

TIER D: the *thermodynamic* statements below are exact only as N -> inf (mean field
becomes exact for all-to-all coupling); at any finite N the collective diagonalization
is itself exact.  Spin-coherent minimization (S = (N/2) n_hat), dropping the O(1/N)
constant J/(2N):
    e_cl(theta) = -(J/2) cos^2(theta) - h sin(theta)      (phi = 0 optimal)
  ferromagnetic (h <= J):  e0 = -J/2 - h^2/(2J)   at sin(theta) = h/J
  paramagnetic  (h >= J):  e0 = -h                 at theta = pi/2 (x-polarized)
continuous C^1 with a second-derivative kink -> second-order QPT at h = J.  This is the
fully-connected mean-field cousin of `tfim-chain`: same critical field h = J, but here
mean-field exponents are EXACT (infinite range), versus the 1D chain's nu = 1, beta = 1/8.
[@BotetJullien1983], [@DusuelVidal2005]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402


def collective_block(N, J, h):
    """Tridiagonal (diag, offdiag) of H = -(2J/N)S_z^2 - 2h S_x + J/2, j = N/2, z-basis."""
    j = N / 2.0
    m = np.arange(-j, j + 1)                       # m_z eigenvalues, length N+1
    diag = -(2.0 * J / N) * m ** 2 + J / 2.0
    mm = m[:-1]
    off = -h * np.sqrt(j * (j + 1) - mm * (mm + 1))
    return diag, off


def block_lowest(N, J, h, k=2):
    """Lowest k eigenvalues of the (N+1)-dim collective block."""
    d, e = collective_block(N, J, h)
    k = min(k, len(d))
    return eigh_tridiagonal(d, e, select="i", select_range=(0, k - 1))[0]


def e0_collective(N, J, h):
    """Exact ground energy per spin (full E0/N, constant J/2 included)."""
    return block_lowest(N, J, h, k=1)[0] / N


def gap_collective(N, J, h):
    """Exact first excitation gap in the collective sector."""
    w = block_lowest(N, J, h, k=2)
    return float(w[1] - w[0])


def e0_thermo(J, h):
    """Thermodynamic-limit ground energy per spin (spin-coherent minimization)."""
    if h >= J:
        return -h
    return -J / 2.0 - h * h / (2.0 * J)


def mz_order_thermo(J, h):
    """Ferromagnetic order parameter <sigma^z>/N = cos(theta) in the N -> inf limit."""
    if h >= J:
        return 0.0
    return np.sqrt(1.0 - (h / J) ** 2)


def _full_ed(N, J, h):
    """Brute-force 2^N ED of the Pauli Hamiltonian; returns ground energy."""
    px, _, pz = ed.pauli_ops(N)
    H = -(J / N) * sum(pz[i] @ pz[k] for i in range(N) for k in range(i + 1, N)) \
        - h * sum(px)
    return ed.ground_energy(H)


def compute(N=100, h=0.5, J=1.0):
    """Curie–Weiss TFIM exact quantities (collective block + thermodynamic limit)."""
    return {
        "e0_per_spin": e0_collective(N, J, h),
        "e0_thermodynamic": e0_thermo(J, h),
        "gap": gap_collective(N, J, h),
        "mz_order_thermo": mz_order_thermo(J, h),
        "phase": "paramagnetic" if h >= J else "ferromagnetic",
    }


def self_test():
    # anchor 1 (IDENTITY PROOF): the collective (N+1)-dim block reproduces the full
    #   2^N Pauli ED ground energy to machine precision at N = 8 — the collective-sector
    #   restriction is exact for the GS, including the constant sum_{i<j} = 2S_z^2 - N/2.
    for h in (0.5, 1.5):
        e_full = _full_ed(8, 1.0, h)
        e_blk = block_lowest(8, 1.0, h, k=1)[0]
        assert abs(e_full - e_blk) < 1e-12, (h, e_full, e_blk)

    # anchor 2 (COLLECTIVE ALGEBRA): the reduction constant is exact.  With sigma^z
    #   collective, sum_{i<j} sigma^z_i sigma^z_j == 2 S_z^2 - N/2 as operators.
    for N in (4, 6):
        _, _, pz = ed.pauli_ops(N)
        Sz = 0.5 * sum(pz)
        lhs = sum(pz[i] @ pz[k] for i in range(N) for k in range(i + 1, N)).toarray()
        rhs = (2.0 * (Sz @ Sz) - 0.5 * N * ed.sp.identity(2 ** N)).toarray()
        assert np.allclose(lhs, rhs, atol=1e-12), N

    # anchor 3 (THERMODYNAMIC LIMIT, D-tier): collective diag at N = 4000 converges to
    #   the mean-field closed form within 1e-4 in BOTH phases (exact only as N -> inf).
    for h in (0.5, 1.5):
        assert abs(e0_collective(4000, 1.0, h) - e0_thermo(1.0, h)) < 1e-4, h
    # convergence is O(1/N): the deviation shrinks from N = 2000 to N = 4000.
    for h in (0.5, 1.5):
        d2 = abs(e0_collective(2000, 1.0, h) - e0_thermo(1.0, h))
        d4 = abs(e0_collective(4000, 1.0, h) - e0_thermo(1.0, h))
        assert d4 < d2, (h, d2, d4)

    # anchor 4 (QPT at h = J): closed form continuous and C^1 (de0/dh = -1 both sides)
    #   but not C^2 — a second-order transition at the standard critical field h = J.
    J = 1.0
    dh = 1e-6
    left = (e0_thermo(J, J - dh) - e0_thermo(J, J - 3 * dh)) / (2 * dh)
    right = (e0_thermo(J, J + 3 * dh) - e0_thermo(J, J + dh)) / (2 * dh)
    assert abs(left - right) < 1e-4 and abs(left + 1.0) < 1e-4, (left, right)

    # anchor 5 (ORDER PARAMETER): ferromagnetic moment onsets as sqrt(1-(h/J)^2) below
    #   h = J and is exactly 0 above — mean-field beta = 1/2.
    assert abs(mz_order_thermo(1.0, 0.0) - 1.0) < 1e-12
    assert mz_order_thermo(1.0, 1.5) == 0.0
    assert abs(mz_order_thermo(1.0, 0.6) - np.sqrt(1 - 0.36)) < 1e-12


if __name__ == "__main__":
    oracle_main(compute, {"N": (int, 100), "h": (float, 0.5), "J": (float, 1.0)})
