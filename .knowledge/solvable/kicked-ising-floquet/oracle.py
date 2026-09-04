"""Self-dual kicked Ising Floquet model: U_F = e^{-iH_K} e^{-iH_I}.

    H_I = J Σ_j σ^z_j σ^z_{j+1} + Σ_j h_j σ^z_j   (Ising + longitudinal disorder),
    H_K = b Σ_j σ^x_j                              (transverse kick),   PBC.

Self-dual point |J| = |b| = π/4. This is the "kicked" gauge in which
Bertini–Kos–Prosen proved an exact spectral form factor [@BertiniKosProsen2018];
it is a space-time-similarity relative of the brickwork dual-unitary gate of the
`dual-unitary-circuits` card (same physics, different gauge — the two operators
are NOT identical and are not isospectral).

Anchors:
 (1) U_F assembled two ways agrees to 1e-12 (L=8, PBC): the monolithic operator
     exponential e^{-iH_K} e^{-iH_I} versus a brickwork-ordered product of local
     factors — the 2-site Ising gate e^{-iJ σ^z σ^z} is IMPORTED (importlib) from
     the dual-unitary card, multiplied over even then odd bonds, dressed with the
     single-site field phases and transverse kicks. (They coincide because the
     H_I terms mutually commute; the check validates the bond wiring and PBC.)
 (2) SPECTRAL FORM FACTOR at self-duality. Define the Floquet SFF (t>0)

         K(t) = ⟨ |tr U_F^t[h]|^2 ⟩_h ,

     averaged over i.i.d. longitudinal fields h_j. BKP prove that in the
     thermodynamic limit the DISORDER-AVERAGED SFF equals the circular-orthogonal
     RMT ramp: for ODD t,  lim_{L→∞} K(t) = 2t−1 (t ≤ 5) and 2t (t ≥ 7)
     [@BertiniKosProsen2018, Eq. (24)]; the RMT comparison curve is
     K_COE(t) = 2t − t ln(1+2t/N), N = 2^L. Averaging over the h_j is what opens
     a gap in the transfer matrix and selects the 2t "universal" eigenvalues; the
     TL result is independent of the disorder distribution and variance. Here at
     finite L=8,10 with a fixed-seed disorder average the odd-t values 2t−1 are
     already reproduced within a documented band, while the NEGATIVE CONTROL
     b=π/5 (away from self-duality) leaves the band.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def _load_dual_unitary():
    """Cross-load the sibling dual-unitary card for its 2-site Ising gate."""
    p = Path(__file__).resolve().parents[1] / "dual-unitary-circuits" / "oracle.py"
    spec = importlib.util.spec_from_file_location("dual_unitary_oracle", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def op_at(op, site, L):
    mats = [I2] * L
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# ---- two independent assemblies of the Floquet operator --------------------

def uf_expm(L, J, b, h):
    """Monolithic U_F = e^{-iH_K} e^{-iH_I} via full operator exponentials."""
    sz = [op_at(Z, s, L) for s in range(L)]
    sx = [op_at(X, s, L) for s in range(L)]
    HI = sum(J * sz[s] @ sz[(s + 1) % L] for s in range(L)) \
        + sum(h[s] * sz[s] for s in range(L))
    HK = b * sum(sx)
    return expm(-1j * HK) @ expm(-1j * HI)


def uf_brickwork(L, J, b, h, du=None):
    """U_F from a brickwork-ordered product of local factors.

    Uses the 2-site Ising gate e^{-iJ σ^z σ^z} imported from the dual-unitary
    card, applied on even then odd bonds (they commute), then the single-site
    field phases e^{-i h_j σ^z} and transverse kicks e^{-i b σ^x_j}.
    """
    du = du or _load_dual_unitary()
    d = 2 ** L
    zz = du.zz_gate(J)                                # e^{-iJ σ^z σ^z}
    U = np.eye(d, dtype=complex)
    for i in range(0, L, 2):                          # even Ising bonds
        U = du._bond(zz, i, L) @ U
    for i in range(1, L - 1, 2):                      # odd Ising bonds
        U = du._bond(zz, i, L) @ U
    U = du._wrap_bond(zz, L) @ U                      # periodic bond (L-1,0)
    kick = expm(-1j * b * X)                          # single-qubit factors
    for s in range(L):                               # longitudinal fields
        U = op_at(expm(-1j * h[s] * Z), s, L) @ U
    for s in range(L):                               # transverse kick
        U = op_at(kick, s, L) @ U
    return U


# ---- spectral form factor --------------------------------------------------

def _kick_operator(L, b):
    """e^{-i b Σ σ^x} = ⊗_j e^{-i b σ^x_j} (a tensor product — cheap)."""
    k = expm(-1j * b * X)
    out = k
    for _ in range(L - 1):
        out = np.kron(out, k)
    return out


def spectral_form_factor(L, b, nreal, tmax, seed, J=np.pi / 4):
    """K(t) = ⟨|tr U_F^t|^2⟩_h over `nreal` i.i.d. fields h_j ~ U(-π,π).

    U_F = kick · diag(e^{-iE_z}) with E_z = J Σ s_j s_{j+1} + Σ h_j s_j diagonal
    in the σ^z basis (site 0 = most significant bit). Returns (mean, band) where
    band is the standard deviation of the mean across 4 disorder sub-batches.
    """
    rng = np.random.default_rng(seed)
    d = 2 ** L
    bits = (np.arange(d)[:, None] >> np.arange(L)[None, :]) & 1
    s = 1 - 2 * bits                                  # s[:,j] = ±1 spin at site j
    zz = np.sum(s * np.roll(s, -1, axis=1), axis=1)   # Σ_j s_j s_{j+1}, PBC
    K = _kick_operator(L, b)
    per = np.zeros((nreal, tmax + 1))
    for r in range(nreal):
        h = rng.uniform(-np.pi, np.pi, L)
        Ez = J * zz + s @ h
        UF = K * np.exp(-1j * Ez)[None, :]            # kick · diag(phases)
        E = np.exp(1j * np.angle(np.linalg.eigvals(UF)))
        p = np.ones_like(E)
        for t in range(1, tmax + 1):
            p = p * E
            per[r, t] = abs(p.sum()) ** 2
    mean = per.mean(0)
    subs = np.array_split(per, 4)
    band = np.array([sub.mean(0) for sub in subs]).std(0)
    return mean, band


def compute(L=8, nreal=48, tmax=5, seed=7):
    """Kicked-Ising diagnostics: two-way U_F residual + self-dual/control SFF."""
    rng = np.random.default_rng(1)
    h = rng.uniform(-1.0, 1.0, L)
    resid = float(np.max(np.abs(
        uf_expm(L, np.pi / 4, np.pi / 4, h) - uf_brickwork(L, np.pi / 4, np.pi / 4, h))))
    Ksd, _ = spectral_form_factor(L, np.pi / 4, nreal, tmax, seed)
    Kct, _ = spectral_form_factor(L, np.pi / 5, nreal, tmax, seed)
    out = {"twoway_UF_residual": resid}
    for t in range(1, tmax + 1):
        out[f"sff_selfdual_t{t}"] = Ksd[t]
        out[f"sff_control_t{t}"] = Kct[t]
    return out


def self_test():
    du = _load_dual_unitary()
    # anchor 1: the two assemblies of U_F agree to machine precision (L=8, PBC),
    # with disorder switched on so the field phases are exercised.
    rng = np.random.default_rng(3)
    for _ in range(2):
        h = rng.uniform(-np.pi, np.pi, 8)
        A = uf_expm(8, np.pi / 4, np.pi / 4, h)
        B = uf_brickwork(8, np.pi / 4, np.pi / 4, h, du=du)
        assert np.max(np.abs(A - B)) < 1e-12, np.max(np.abs(A - B))
        assert np.max(np.abs(A.conj().T @ A - np.eye(256))) < 1e-12   # unitary

    # anchor 2: self-dual SFF reproduces the BKP odd-t ramp K(t)=2t−1 (t≤5)
    # within a finite-size / finite-sample band; the b=π/5 control leaves it.
    # Fixed seed → deterministic. Measured (L=8, 24 realizations, seed 7):
    #   self-dual  K(1)≈0.76, K(3)≈5.4    (2t−1 = 1, 5)
    #   control    K(1)≈6.6,  K(3)≈9.9
    Ksd, band = spectral_form_factor(8, np.pi / 4, 24, 3, seed=7)
    Kct, _ = spectral_form_factor(8, np.pi / 5, 24, 3, seed=7)
    assert Ksd[1] < 1.8, Ksd[1]                       # band around 2t−1 = 1
    assert 3.0 < Ksd[3] < 7.0, Ksd[3]                 # band around 2t−1 = 5
    # negative control: at least the sharp t=1 discriminator leaves the band.
    assert Kct[1] > 3.0, Kct[1]
    assert Kct[1] > 3 * Ksd[1], (Ksd[1], Kct[1])
    # the disorder average is genuine (non-zero spread reported).
    assert band[1] > 0.0


if __name__ == "__main__":
    oracle_main(compute, {"L": (int, 8), "nreal": (int, 48),
                          "tmax": (int, 5), "seed": (int, 7)})
