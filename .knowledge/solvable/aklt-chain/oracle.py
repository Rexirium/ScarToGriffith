"""AKLT spin-1 chain oracle: H = sum_i [ S_i.S_{i+1} + (1/3)(S_i.S_{i+1})^2 ]  (PBC).

T5 (frustration-free / exact eigenstates), Tier C. On each bond the local term
equals 2 P2 - 2/3, where P2 projects the bond onto total spin 2; since P2 >= 0
the valence-bond-solid (VBS) state with zero weight in every spin-2 channel is
the *exact* zero-of-P2 ground state, with energy E0 = -2N/3 (N = #bonds = L, PBC).
Only the ground state is exact: the VBS is a bond-dimension-2 matrix-product
state built from A^{+}=sqrt(2/3) sigma^+, A^{0}=-sqrt(1/3) sigma^z,
A^{-}=-sqrt(2/3) sigma^-.  The generic spectrum and the Haldane gap are NOT
closed-form and are quoted as numerical ED values.

Exact closed forms (from the 2-site MPS transfer matrix, eigenvalues 1, -1/3):
  <S^z_0 S^z_r> = (4/3)(-1/3)^r ,   string order O^z = -4/9 .
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.cli import oracle_main  # noqa: E402
from _lib import ed  # noqa: E402

# ---- spin-1 local operators (d=3 basis ordered S^z = +1, 0, -1) ----
_S2 = np.sqrt(2.0)
SP = np.array([[0, _S2, 0], [0, 0, _S2], [0, 0, 0]], complex)   # S^+
SM = SP.conj().T                                                # S^-
SX = 0.5 * (SP + SM)
SY = 0.5j * (SM - SP)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
_ID3 = sp.identity(3, format="csr", dtype=complex)

# ---- exact AKLT MPS matrices (physical index +1, 0, -1) ----
A_PLUS = np.sqrt(2.0 / 3.0) * np.array([[0, 1], [0, 0]], complex)
A_ZERO = -np.sqrt(1.0 / 3.0) * np.array([[1, 0], [0, -1]], complex)
A_MINUS = -np.sqrt(2.0 / 3.0) * np.array([[0, 0], [1, 0]], complex)
A_MATS = [A_PLUS, A_ZERO, A_MINUS]
_SZVAL = [1.0, 0.0, -1.0]


def _site_op(op, i, L):
    mats = [_ID3] * L
    mats[i] = sp.csr_matrix(op)
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


def spin1_ops(L):
    """Lists of sparse spin-1 S^x, S^y, S^z operators on an L-site chain."""
    return ([_site_op(SX, i, L) for i in range(L)],
            [_site_op(SY, i, L) for i in range(L)],
            [_site_op(SZ, i, L) for i in range(L)])


def aklt_H(L, pbc=True):
    """H = sum_bonds [ S_i.S_j + (1/3)(S_i.S_j)^2 ]; PBC -> L bonds, OBC -> L-1."""
    Sx, Sy, Sz = spin1_ops(L)
    H = 0
    for b in range(L if pbc else L - 1):
        i, j = b, (b + 1) % L
        SS = Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j]
        H = H + SS + (1.0 / 3.0) * (SS @ SS)
    return H


def aklt_mps_state(L):
    """Exact periodic AKLT VBS state vector (3^L): psi(m) = Tr(A^{m_0}...A^{m_{L-1}})."""
    dim = 3 ** L
    psi = np.zeros(dim, complex)
    for idx in range(dim):
        ms, t = [], idx
        for _ in range(L):
            ms.append(t % 3)
            t //= 3
        M = np.eye(2, dtype=complex)
        for m in reversed(ms):            # site 0 is most significant (kron order)
            M = M @ A_MATS[m]
        psi[idx] = np.trace(M)
    return psi / np.linalg.norm(psi)


# ---- transfer-matrix closed forms (infinite chain) ----
def _transfer():
    T = sum(np.kron(A, A.conj()) for A in A_MATS)
    Tsz = sum(_SZVAL[m] * np.kron(A_MATS[m], A_MATS[m].conj()) for m in range(3))
    Tu = sum(np.exp(1j * np.pi * _SZVAL[m]) * np.kron(A_MATS[m], A_MATS[m].conj())
             for m in range(3))
    return T, Tsz, Tu


def szsz_correlator(r):
    """Exact infinite-chain <S^z_0 S^z_r> = (4/3)(-1/3)^r (r >= 1)."""
    return (4.0 / 3.0) * (-1.0 / 3.0) ** r


def string_order():
    """Exact AKLT string order parameter O^z = -4/9."""
    return -4.0 / 9.0


def compute(L=8):
    """AKLT spin-1 chain: exact VBS ground-state quantities + numerical Haldane gap."""
    return {
        "e0_per_site": -2.0 / 3.0,                # exact, VBS
        "e0_total_pbc": -2.0 * L / 3.0,           # exact, N = L bonds
        "szsz_r1": szsz_correlator(1),            # -4/9
        "szsz_r2": szsz_correlator(2),            # 4/27
        "string_order": string_order(),          # -4/9
        "gap_pbc_ed": ed.gap(aklt_H(L, pbc=True)),  # numerical (not closed form)
    }


def self_test():
    # anchor 1 (GROUND TRUTH): PBC E0 == -2L/3 exactly and GS unique, L in {6,8}.
    for L in (6, 8):
        H = aklt_H(L, pbc=True)
        assert abs(ed.ground_energy(H) - (-2.0 * L / 3.0)) < 1e-10, L
        assert ed.ground_states(H) == 1, L

    # anchor 2: OBC L=6 -> E0 = -2(L-1)/3 and 4-fold GS (the two spin-1/2 edges).
    H6 = aklt_H(6, pbc=False)
    assert abs(ed.ground_energy(H6) - (-2.0 * 5 / 3.0)) < 1e-10
    assert ed.ground_states(H6) == 4

    # anchor 3: explicit MPS state is an exact eigenstate; H|psi> = -2L/3 |psi>.
    L = 8
    psi = aklt_mps_state(L)
    H = aklt_H(L, pbc=True)
    Hpsi = H @ psi
    E = np.vdot(psi, Hpsi).real
    assert abs(E - (-2.0 * L / 3.0)) < 1e-10
    assert np.linalg.norm(Hpsi - E * psi) < 1e-10           # operator-level residual
    # ...and it equals the ED ground state (overlap 1 up to phase).
    from scipy.sparse.linalg import eigsh
    _, v = eigsh(H, k=1, which="SA")
    assert abs(abs(np.vdot(v[:, 0], psi)) - 1.0) < 1e-10

    # anchor 4: transfer-matrix correlator == closed form; and finite-L=8 PBC
    # transfer-matrix trace == direct MPS-state expectation value.
    T, Tsz, Tu = _transfer()
    w, vr = np.linalg.eig(T); ir = int(np.argmax(w.real))
    wl, vl = np.linalg.eig(T.T); il = int(np.argmax(wl.real))
    lv, rv = vl[:, il], vr[:, ir]; nrm = lv @ rv
    # transfer eigenvalues are exactly {1, -1/3, -1/3, -1/3}
    assert abs(np.max(w.real) - 1.0) < 1e-12
    Sz = spin1_ops(L)[2]
    for r in range(1, 5):
        inf_val = (lv @ Tsz @ np.linalg.matrix_power(T, r - 1) @ Tsz @ rv / nrm).real
        assert abs(inf_val - szsz_correlator(r)) < 1e-12, r
        # finite-L trace formula vs explicit MPS state
        num = np.trace(Tsz @ np.linalg.matrix_power(T, r - 1)
                       @ Tsz @ np.linalg.matrix_power(T, L - r - 1))
        fin = (num / np.trace(np.linalg.matrix_power(T, L))).real
        direct = np.vdot(psi, Sz[0] @ Sz[r] @ psi).real
        assert abs(fin - direct) < 1e-10, r

    # anchor 5: string order parameter == -4/9 exactly (infinite chain).
    d = 32
    o = (lv @ Tsz @ np.linalg.matrix_power(Tu, d - 1) @ Tsz @ rv / nrm).real
    assert abs(o - (-4.0 / 9.0)) < 1e-12


if __name__ == "__main__":
    oracle_main(compute, {"L": (int, 8)})
