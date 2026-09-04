"""PXP quantum-many-body-scar oracle: H = Σ_j P_{j−1} X_j P_{j+1} (OBC).

T5 (exact eigenstates embedded in a thermal spectrum), Tier C.  P = |0⟩⟨0| is the
ground(g)-state projector, X = σ^x, |0⟩=g, |1⟩=r (Rydberg); OBC boundary terms are
X_1 P_2 and P_{N−1} X_N (matches .knowledge/models/rydberg-pxp).  The blockade
forbids adjacent excitations, so the constrained Hilbert space is Fibonacci-
dimensional (OBC: F_{N+2}; PBC: Lucas L_N).

Exact scars (Lin–Motrunich, PRL 122, 173401): block two sites (2b,2b+1) into one
"block-site" with allowed states O=(00), and the two single-excitation states, and
use the bond-dimension-2 MPS  |Γ_{a,b}⟩ = Σ v_a^T A^{s_1}…A^{s_{N/2}} v_b |s⟩ with
    A^O = [[0,−1],[1,0]],  and the two excitation matrices
    [[√2,0],[0,0]], [[0,0],[0,−√2]]  (assignment fixed by the blockade A A = 0),
    v_1 = (1,1)^T,  v_2 = (1,−1)^T.
Then |Γ_{1,1}⟩,|Γ_{2,2}⟩ are E=0 eigenstates and |Γ_{1,2}⟩,|Γ_{2,1}⟩ are E=±√2
eigenstates (this H convention, X=σ^x).  Only these few states are exact; the bulk
spectrum is chaotic (ETH).  Identity-proof of weak ergodicity breaking: the scar's
half-chain entanglement entropy sits far below the mean of eigenstates in the same
narrow energy window — it is an ETH outlier, not a generic eigenstate.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402

_P = sp.csr_matrix(np.array([[1, 0], [0, 0]], complex))   # |0><0| ground projector
_X = sp.csr_matrix(np.array([[0, 1], [1, 0]], complex))   # sigma^x


def _op(mat, i, N):
    return ed._site_op(sp.csr_matrix(mat), i, N)


def pxp_H(N, pbc=False):
    """H = Σ_j P_{j−1} X_j P_{j+1}; OBC uses boundary terms X_0 P_1 and P_{N−2} X_{N−1}."""
    H = 0
    lo, hi = (0, N) if pbc else (0, N)
    for j in range(lo, hi):
        left = _op(_P.toarray(), (j - 1) % N, N) if (pbc or j > 0) else None
        right = _op(_P.toarray(), (j + 1) % N, N) if (pbc or j < N - 1) else None
        term = _op(_X.toarray(), j, N)
        if left is not None:
            term = left @ term
        if right is not None:
            term = term @ right
        H = H + term
    return H.tocsr()


# ---- constrained (Fibonacci) basis -------------------------------------------------

def allowed_bitstrings(N, pbc=False):
    """All length-N bitstrings with no two adjacent 1s (PBC also forbids b_{N-1},b_0)."""
    out = []
    for x in range(2 ** N):
        bits = [(x >> (N - 1 - i)) & 1 for i in range(N)]   # bits[0]=site0 = high bit
        if any(bits[i] and bits[i + 1] for i in range(N - 1)):
            continue
        if pbc and bits[-1] and bits[0]:
            continue
        out.append((x, tuple(bits)))
    return out


def fibonacci(n):
    a, b = 1, 1                                              # F_1=F_2=1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def lucas(n):
    a, b = 2, 1                                              # L_0=2, L_1=1
    for _ in range(n):
        a, b = b, a + b
    return a


def constrained_dim(N, pbc=False):
    return len(allowed_bitstrings(N, pbc))


# ---- Lin–Motrunich exact scar MPS --------------------------------------------------

_A_O = np.array([[0, -1], [1, 0]], complex)
_A_EXC = (np.array([[np.sqrt(2), 0], [0, 0]], complex),      # goes with block (1,0)
          np.array([[0, 0], [0, -np.sqrt(2)]], complex))     # goes with block (0,1)
_V = {1: np.array([1, 1], complex), 2: np.array([1, -1], complex)}


def scar_state(N, a, b):
    """Full 2^N vector of the Lin–Motrunich MPS |Γ_{a,b}⟩ (N even, OBC)."""
    assert N % 2 == 0
    Lb = N // 2
    va, vb = _V[a], _V[b]
    psi = np.zeros(2 ** N, complex)
    # block alphabet: 0->O=(0,0), 1->(1,0), 2->(0,1); matrices A[k]
    A = [_A_O, _A_EXC[0], _A_EXC[1]]
    bitpair = [(0, 0), (1, 0), (0, 1)]
    for code in range(3 ** Lb):
        ks, t = [], code
        for _ in range(Lb):
            ks.append(t % 3)
            t //= 3
        M = va.reshape(1, 2)
        for k in ks:
            M = M @ A[k]
        amp = complex((M @ vb)[0])
        if amp == 0:
            continue
        idx = 0
        for blk, k in enumerate(ks):
            hb, lb = bitpair[k]
            idx += hb << (N - 1 - 2 * blk)
            idx += lb << (N - 1 - 2 * blk - 1)
        psi[idx] = amp
    nrm = np.linalg.norm(psi)
    return psi / nrm


# ---- entanglement / ETH-outlier diagnostics ----------------------------------------

def half_chain_entropy(psi, N):
    """Von-Neumann entropy of the left N/2 sites (natural log)."""
    half = N // 2
    mat = psi.reshape(2 ** half, 2 ** (N - half))
    s = np.linalg.svd(mat, compute_uv=False)
    p = (s ** 2)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def _constrained_spectrum(N, pbc=False):
    """Dense (E, eigenvectors-as-full-2^N-columns) of PXP in the constrained sector."""
    H = pxp_H(N, pbc)
    idx = [x for x, _ in allowed_bitstrings(N, pbc)]
    Hc = H[np.ix_(idx, idx)].toarray()
    w, v = np.linalg.eigh(Hc)
    full = np.zeros((2 ** N, len(idx)), complex)
    full[idx, :] = v
    return w, full


def eth_outlier(N=12):
    """Scar half-chain entropy vs mean entropy of eigenstates in |E−√2|<0.5 (OBC)."""
    w, full = _constrained_spectrum(N, pbc=False)
    ent = np.array([half_chain_entropy(full[:, k], N) for k in range(full.shape[1])])
    scar = scar_state(N, 1, 2)                              # E=+√2 scar
    s_scar = half_chain_entropy(scar, N)
    win = np.abs(w - np.sqrt(2.0)) < 0.5
    return {
        "scar_entropy": s_scar,
        "window_mean_entropy": float(ent[win].mean()),
        "window_count": int(win.sum()),
        "n_zero_modes": int(np.sum(np.abs(w) < 1e-8)),
    }


def compute(N=12):
    """PXP scars: constrained dimensions, exact-scar energies, ETH-outlier entropies."""
    out = {
        "dim_constrained_obc": constrained_dim(N, pbc=False),
        "fibonacci_F_Nplus2": fibonacci(N + 2),
        "dim_constrained_pbc": constrained_dim(N, pbc=True),
        "lucas_L_N": lucas(N),
    }
    H = pxp_H(N, pbc=False)
    for label, (a, b) in {"scar_plus": (1, 2), "scar_minus": (2, 1),
                          "scar_zero": (1, 1)}.items():
        psi = scar_state(N, a, b)
        E = np.vdot(psi, H @ psi).real
        out[f"E_{label}"] = E
    out.update({f"eth_{k}": v for k, v in eth_outlier(N).items()})
    return out


def self_test():
    # anchor 1 (COMBINATORIAL GROUND TRUTH): constrained dim == Fibonacci (OBC,
    #   F_{N+2}) and Lucas (PBC, L_N) for N ≤ 14.
    for N in range(2, 15):
        assert constrained_dim(N, pbc=False) == fibonacci(N + 2), N
    for N in range(3, 15):
        assert constrained_dim(N, pbc=True) == lucas(N), N

    # anchor 2 (EXACT SCARS, operator-level): the Lin–Motrunich MPS states are exact
    #   eigenstates.  Γ_{1,2}→+√2, Γ_{2,1}→−√2, Γ_{1,1}=Γ_{2,2}→0, at N=8 and N=12.
    for N in (8, 12):
        H = pxp_H(N, pbc=False)
        for (a, b), Eref in {(1, 2): np.sqrt(2.0), (2, 1): -np.sqrt(2.0),
                             (1, 1): 0.0, (2, 2): 0.0}.items():
            psi = scar_state(N, a, b)
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-12, (N, a, b)
            Hpsi = H @ psi
            E = np.vdot(psi, Hpsi).real
            assert abs(E - Eref) < 1e-10, (N, a, b, E)
            assert np.linalg.norm(Hpsi - Eref * psi) < 1e-10, (N, a, b)  # operator res.
        # the ±√2 scars live in the constrained sector and are genuine eigenvectors
        w, _ = _constrained_spectrum(N, pbc=False)
        assert np.min(np.abs(w - np.sqrt(2.0))) < 1e-10, N
        assert np.min(np.abs(w + np.sqrt(2.0))) < 1e-10, N

    # anchor 3: the ±√2 scars are ORTHOGONAL to the E=0 scars and to each other
    #   (distinct exact eigenvalues) — a sanity check on the construction.
    N = 8
    sp_, sm_, s0 = scar_state(N, 1, 2), scar_state(N, 2, 1), scar_state(N, 1, 1)
    assert abs(np.vdot(sp_, sm_)) < 1e-10
    assert abs(np.vdot(sp_, s0)) < 1e-10

    # anchor 4 (IDENTITY-PROOF, ETH outlier): at N=12 the +√2 scar's half-chain
    #   entropy is far below the mean of eigenstates in |E−√2|<0.5 (weak ergodicity
    #   breaking).  Also record the exponential E=0 zero-mode degeneracy.
    d = eth_outlier(12)
    assert d["window_count"] >= 5                          # a real thermal window
    assert d["scar_entropy"] < 0.5 * d["window_mean_entropy"]  # outlier by ≥2×
    assert d["n_zero_modes"] >= 10                         # exponentially many (chiral)


if __name__ == "__main__":
    oracle_main(compute, {"N": (int, 12)})
