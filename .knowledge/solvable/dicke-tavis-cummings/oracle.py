"""Dicke / Tavis–Cummings oracle: N two-level atoms (collective spin J = N/2) + one mode.

TAVIS–CUMMINGS (RWA) [@TavisCummings1968]:
    H_TC = omega a†a + omega0 J_z + (g/sqrt(N)) (a J_+ + a† J_-)
DICKE (full, counter-rotating terms kept) [@Dicke1954]:
    H_D  = omega a†a + omega0 J_z + (lambda/sqrt(N)) (a + a†)(J_+ + J_-)
J_{x,y,z}, J_pm are collective spin operators of the maximal spin j = N/2 (all atoms
symmetric): J_z has eigenvalues -j..j, J_+ raises J_z.  a (a†) annihilate/create a photon.
The 1/sqrt(N) is the standard Kac scaling that keeps H/N finite as N -> inf (equivalently
lambda/sqrt(2j)).  omega0 is the atomic frequency, delta = omega0 - omega.

TAVIS–CUMMINGS is EXACTLY BLOCK-DIAGONAL.  The excitation number
    C = a†a + J_z + N/2   (photons + excited atoms)
commutes with H_TC.  In sector C = c the states are |n = c-k photons, J_z = -j+k>,
k = 0 .. min(c, N) (need n >= 0 and k <= N), so the block has dimension
    dim(c) = min(c, N) + 1 = min(c + 1, N + 1).
Within a block the RWA coupling is tridiagonal in k, and diagonalizing the (small) block
gives the exact spectrum — no Fock truncation error (Tier B: the block structure is the
exact solution; TC is Bethe-ansatz / Richardson-Gaudin integrable, but for collective j
the blocks are already finite and are diagonalized directly).  N = 1 reduces H_TC exactly
to `jaynes-cummings` (J_z = sigma^z/2, J_pm = sigma^pm, g/sqrt(1) = g).

DICKE has NO such conservation (the counter-rotating a J_- + a† J_+ break C down to the
Z2 parity Pi = exp(i pi (a†a + J_z + N/2))), so there is no exact finite block: it needs a
truncated-boson full-space ED (Fock cutoff n_max, CONVERGENCE-CHECKed).  In the
THERMODYNAMIC LIMIT (N -> inf) the Holstein–Primakoff / mean-field treatment is exact and
gives a SUPERRADIANT quantum phase transition at
    lambda_c = sqrt(omega omega0) / 2
[@EmaryBrandes2003].  (HP: J_z = b†b - j, J_+ ~ sqrt(2j) b† in the normal phase; the two
coupled bosonic modes have energies eps_pm^2 = (1/2)[omega^2 + omega0^2 +/- sqrt((omega0^2
- omega^2)^2 + 16 lambda^2 omega omega0)], and the soft mode eps_- -> 0 exactly when
4 omega^2 omega0^2 = 16 lambda^2 omega omega0, i.e. lambda = sqrt(omega omega0)/2.)  At
finite N the ground-state photon occupation per atom <a†a>/N is a smooth precursor that
rises across lambda_c (observed-as-observed, not a sharp transition at finite N).

Cross-refs: TC is the RWA of the Dicke model; TC at N = 1 is `jaynes-cummings`; same T6
collective family as `lmg`.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402


# ---- collective-spin matrices --------------------------------------------------------

def spin_ops(N):
    """Collective spin j = N/2 matrices (Jz, Jp, Jm) on the (N+1)-dim symmetric multiplet."""
    j = N / 2.0
    m = np.arange(-j, j + 1)                 # J_z eigenvalues, length N+1
    Jz = np.diag(m)
    # J_+ |j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1> RAISES J_z -> the coefficient sits on the
    # SUBdiagonal (row of m+1, column of m), so J_+ is lower-triangular.
    cp = np.sqrt(j * (j + 1) - m[:-1] * (m[:-1] + 1))
    Jp = np.diag(cp, -1)
    Jm = Jp.T.copy()
    return Jz, Jp, Jm


def boson_ops(n_max):
    """Truncated annihilation a and number a†a on Fock space {|0>,...,|n_max>}."""
    d = n_max + 1
    a = np.zeros((d, d))
    for k in range(1, d):
        a[k - 1, k] = np.sqrt(k)
    return a, a.T @ a


# ---- Tavis–Cummings: exact finite blocks ---------------------------------------------

def tc_block_dim(c, N):
    """Dimension of the excitation-C block: min(c, N) + 1 = min(c+1, N+1)."""
    return min(c, N) + 1


def tc_block(c, N, g, omega0, omega=1.0):
    """Tridiagonal C = c block of H_TC in the k = (# excited atoms) basis; eigenvalues."""
    j = N / 2.0
    ks = np.arange(0, min(c, N) + 1)                      # excited-atom count
    n = c - ks                                            # photon number
    m = -j + ks                                           # J_z
    diag = omega * n + omega0 * m
    # coupling <k+1| (g/sqrt N)(a J_+ + a† J_-) |k>: a J_+ sends (n,m)->(n-1,m+1)
    kk = ks[:-1]
    nn = c - kk
    mm = -j + kk
    off = (g / np.sqrt(N)) * np.sqrt(nn) * np.sqrt(j * (j + 1) - mm * (mm + 1))
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(H)


def tc_spectrum(N, g, omega0, omega=1.0, c_max=12):
    """Sorted low-lying TC spectrum from the exact finite blocks up to excitation c_max."""
    levels = []
    for c in range(c_max + 1):
        levels.extend(tc_block(c, N, g, omega0, omega))
    return np.sort(np.array(levels))


# ---- truncated-boson full-space ED (TC cross-check and Dicke) ------------------------

def _build_H(N, n_max, omega0, omega, rwa, coupling):
    """Sparse H on (Fock cutoff n_max) x (collective spin j = N/2).

    rwa=True  -> Tavis–Cummings coupling (g/sqrt N)(a J_+ + a† J_-);
    rwa=False -> Dicke coupling (lambda/sqrt N)(a + a†)(J_+ + J_-).  `coupling` is g/lambda.
    """
    a, num = boson_ops(n_max)
    Jz, Jp, Jm = spin_ops(N)
    Ib = sp.identity(n_max + 1)
    Is = sp.identity(N + 1)
    a, num = sp.csr_matrix(a), sp.csr_matrix(num)
    Jz, Jp, Jm = sp.csr_matrix(Jz), sp.csr_matrix(Jp), sp.csr_matrix(Jm)
    H = omega * sp.kron(num, Is) + omega0 * sp.kron(Ib, Jz)
    c = coupling / np.sqrt(N)
    if rwa:
        H = H + c * (sp.kron(a, Jp) + sp.kron(a.T, Jm))
    else:
        H = H + c * sp.kron(a + a.T, Jp + Jm)
    return H.tocsr()


def tc_full_ed(N, n_max, g, omega0, omega=1.0, k=16):
    """Lowest k eigenvalues of the truncated full-space TC Hamiltonian."""
    H = _build_H(N, n_max, omega0, omega, rwa=True, coupling=g)
    return np.sort(eigsh(H, k=min(k, H.shape[0] - 1), which="SA",
                         return_eigenvectors=False))


def dicke_photon_per_atom(N, n_max, lam, omega0, omega=1.0):
    """Ground-state <a†a>/N of the Dicke model (truncated-boson ED)."""
    H = _build_H(N, n_max, omega0, omega, rwa=False, coupling=lam)
    w, v = eigsh(H, k=1, which="SA")
    g = v[:, 0]
    a, num = boson_ops(n_max)
    Nph = sp.kron(sp.csr_matrix(num), sp.identity(N + 1)).tocsr()
    return float(g.conj() @ (Nph @ g)) / N


def lambda_c(omega0, omega=1.0):
    """Superradiant critical coupling lambda_c = sqrt(omega omega0)/2 (N -> inf)."""
    return 0.5 * np.sqrt(omega * omega0)


def compute(N=4, g=0.5, omega0=1.0, omega=1.0):
    """Tavis–Cummings exact low levels + Dicke superradiant critical coupling."""
    w = tc_spectrum(N, g, omega0, omega)
    return {
        "N": N,
        "tc_e_ground": float(w[0]),
        "tc_gap": float(w[1] - w[0]),
        "tc_block_dim_c3": tc_block_dim(3, N),
        "dicke_lambda_c": lambda_c(omega0, omega),
    }


def self_test():
    # anchor 1 (BLOCK DIM FORMULA): the exact-block dimension is min(c+1, N+1); verify the
    #   tridiagonal builder returns exactly that many eigenvalues for a range of c, N.
    for N in (1, 2, 4, 7):
        for c in range(0, 10):
            assert len(tc_block(c, N, 0.4, 1.0)) == min(c + 1, N + 1), (N, c)

    # anchor 2 (EXACT BLOCKS == TRUNCATED FULL ED, converged): the union of TC finite
    #   blocks reproduces the lowest truncated-boson ED levels to 1e-10 at N = 4, two
    #   (g, delta) points, and the ED is Fock-converged (n_max = 30 vs 60 agree).
    for g, omega0 in ((0.5, 1.0), (0.9, 1.4)):
        ref = tc_spectrum(4, g, omega0, 1.0)[:12]
        ed30 = tc_full_ed(4, 30, g, omega0, 1.0, k=12)
        ed60 = tc_full_ed(4, 60, g, omega0, 1.0, k=12)
        assert np.max(np.abs(ed30 - ref)) < 1e-10, (g, omega0, np.max(np.abs(ed30 - ref)))
        assert np.max(np.abs(ed30 - ed60)) < 1e-10, (g, omega0)   # CONVERGENCE-CHECK

    # anchor 3 (N = 1 REDUCES TO JAYNES–CUMMINGS): load the JC oracle and compare its
    #   closed-form spectrum to the TC N = 1 block spectrum to 1e-12.
    import importlib.util
    jc_path = Path(__file__).resolve().parents[1] / "jaynes-cummings" / "oracle.py"
    spec = importlib.util.spec_from_file_location("jc_oracle", jc_path)
    jc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jc)
    for g, omega0 in ((0.5, 1.0), (0.7, 1.3)):
        tc = tc_spectrum(1, g, omega0, 1.0, c_max=12)[:15]
        jcs = jc.spectrum(14, g, omega0, 1.0)[:15]
        assert np.max(np.abs(tc - jcs)) < 1e-12, (g, omega0, np.max(np.abs(tc - jcs)))

    # anchor 4 (SUPERRADIANT lambda_c CLOSED FORM): lambda_c = sqrt(omega omega0)/2 for a
    #   few (omega, omega0); resonant omega = omega0 = 1 gives exactly 1/2.
    assert abs(lambda_c(1.0, 1.0) - 0.5) < 1e-14
    assert abs(lambda_c(4.0, 1.0) - 0.5 * 2.0) < 1e-14
    assert abs(lambda_c(2.0, 0.5) - 0.5) < 1e-14

    # anchor 5 (FINITE-N SUPERRADIANT PRECURSOR, observed): at N = 24 (resonant,
    #   omega = omega0 = 1, lambda_c = 0.5) the ground-state <a†a>/N stays small in the
    #   normal phase and rises across lambda_c into the superradiant phase.  This is an
    #   OBSERVED finite-N precursor (smooth, not a sharp transition), Fock-converged
    #   (n_max = 60 vs 90 at the largest coupling agree to 1e-6).
    N = 24
    below = dicke_photon_per_atom(N, 60, 0.25, 1.0, 1.0)   # deep normal phase
    near = dicke_photon_per_atom(N, 60, 0.50, 1.0, 1.0)    # at lambda_c
    above = dicke_photon_per_atom(N, 60, 0.90, 1.0, 1.0)   # superradiant
    assert below < near < above, (below, near, above)
    assert below < 0.05 and above > 0.3, (below, above)
    conv = dicke_photon_per_atom(N, 90, 0.90, 1.0, 1.0)    # CONVERGENCE-CHECK
    assert abs(above - conv) < 1e-6, (above, conv)


if __name__ == "__main__":
    oracle_main(compute, {"N": (int, 4), "g": (float, 0.5),
                          "omega0": (float, 1.0), "omega": (float, 1.0)})
