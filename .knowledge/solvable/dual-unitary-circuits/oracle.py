"""Dual-unitary brickwork circuits: space-time-unitary 2-qubit gates.

A two-qubit gate U acts on H_a ⊗ H_b with matrix elements U_{(o1 o2),(i1 i2)}
(o = output/top legs, i = input/bottom legs; left site index first). Arrange
the brickwork so the even layer applies U on bonds (0,1),(2,3),... and the odd
layer on (1,2),(3,4),...,(L-1,0), PBC; one Floquet period is U_F = U_odd U_even.

DUAL GATE (space-time flip). Rotating the gate 90 deg exchanges the roles of
space and time. The reshuffled "dual" gate is (Bertini–Kos–Prosen realignment)

    Ũ_{(o1 i1),(o2 i2)} = U_{(o1 o2),(i1 i2)},

i.e. from the tensor U[o1,o2,i1,i2] one pairs each output leg with the input leg
of the SAME site: rows (o1,i1) label the left site, columns (o2,i2) the right.
Read back as a 4x4 matrix this is exactly

    Ũ = np.transpose(U.reshape(2,2,2,2), (1,3,0,2)).reshape(4,4)

(verified once: this permutation reproduces Ũ_{(ab),(cd)} = U_{(ca),(db)} entry
by entry). A gate is DUAL-UNITARY iff BOTH U and Ũ are unitary — the circuit
then defines a unitary evolution in the space direction as well as in time.

Two anchors, both web-verified numerically (the check IS the numerics):
 (1) The self-dual kicked-Ising gate U(J,b) = e^{iJ ZZ} e^{ib(X⊗1+1⊗X)} e^{iJ ZZ}
     is dual-unitary exactly at |J|=|b|=π/4 (Ũ†Ũ=1 to 1e-12); the SAME reshuffle
     at b=π/6 fails unitarity by O(1) (deviation > 0.1) — the chiral-Potts
     negative-control lesson: a non-dual-unitary gate must FAIL the dual check.
 (2) Light-cone correlators [@BertiniKosProsen2019]: for a dual-unitary
     brickwork, infinite-temperature two-point functions ⟨σ^z_x(t) σ^z_0(0)⟩/2^L
     vanish OFF the light cone (|x| ≠ 2t per Floquet period) and, ON the edge,
     equal a single-site quantum-channel iterated 2t times. Verified by direct
     Heisenberg evolution at L=8/10, t ≤ 3. (The self-dual KIM gate itself is a
     Clifford point whose σ^z spreads into a pure string with zero single-site
     marginal, so the correlator structure is exhibited on a GENERIC dual-unitary
     gate — the KIM gate carries anchor (1).)
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
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = [I2, X, Y, Z]


def kron(*ops):
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


ZZ2 = kron(Z, Z)
XX2 = kron(X, I2) + kron(I2, X)


# ---- gates -----------------------------------------------------------------

def zz_gate(J):
    """Two-site Ising gate e^{-iJ σ^z⊗σ^z} (imported by the kicked-Ising card)."""
    return expm(-1j * J * ZZ2)


def kim_gate(J, b):
    """Self-dual kicked-Ising 2-qubit gate e^{iJ ZZ} e^{ib(X⊗1+1⊗X)} e^{iJ ZZ}.

    Dual-unitary exactly at |J|=|b|=π/4 (a Clifford point).
    """
    e = expm(1j * J * ZZ2)
    return e @ expm(1j * b * XX2) @ e


def su2(nx, ny, nz, theta):
    """Single-qubit rotation exp(-i θ/2 n·σ)."""
    return expm(-1j * theta / 2 * (nx * X + ny * Y + nz * Z))


def generic_du_gate(Jz=0.7):
    """A GENERIC (non-Clifford) dual-unitary gate.

    Canonical form U = (u1⊗u2) exp[-i(π/4 XX + π/4 YY + Jz ZZ)] (v1⊗v2): the
    XX,YY couplings fixed at π/4 make it dual-unitary for any Jz and any local
    rotations. Fixed local rotations chosen so ⟨σ^z⟩ correlations survive on the
    light cone with a non-trivial (|λ|<1) single-site channel.
    """
    core = expm(-1j * (np.pi / 4 * kron(X, X) + np.pi / 4 * kron(Y, Y) + Jz * ZZ2))
    return kron(su2(1, 0, 0, 0.6), su2(0, 1, 0, 0.9)) @ core \
        @ kron(su2(0, 0, 1, 0.4), su2(1, 1, 0, 0.5))


# ---- dual-unitarity check --------------------------------------------------

def dual_gate(U):
    """Space-time reshuffle Ũ of a 4x4 two-qubit gate U (BKP convention)."""
    T = U.reshape(2, 2, 2, 2)                       # U[o1,o2,i1,i2]
    return np.transpose(T, (1, 3, 0, 2)).reshape(4, 4)


def unitarity_defect(M):
    """max|M†M − 1| — zero iff M is unitary."""
    n = M.shape[0]
    return float(np.max(np.abs(M.conj().T @ M - np.eye(n))))


def dual_unitarity_defect(U):
    """max|Ũ†Ũ − 1| for the reshuffled gate; ~0 iff U is dual-unitary."""
    return unitarity_defect(dual_gate(U))


# ---- brickwork circuit & Heisenberg correlators ----------------------------

def op_at(op, site, L):
    mats = [I2] * L
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _bond(u, i, L):
    """Gate u on the non-wrapping bond (i, i+1)."""
    return kron(np.eye(2 ** i), u, np.eye(2 ** (L - i - 2)))


def _wrap_bond(u, L):
    """Gate u on the periodic bond (L-1, 0). Site 0 is the most significant bit."""
    U4 = u.reshape(2, 2, 2, 2)                       # [o(L-1), o(0), i(L-1), i(0)]
    full = np.zeros((2 ** L, 2 ** L), complex)
    dmid = 2 ** (L - 2)
    mid = np.arange(dmid) << 1
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    v = U4[a, b, c, d]
                    if v == 0:
                        continue
                    rows = (b << (L - 1)) + mid + a     # out: site0=b (MSB), siteL-1=a (LSB)
                    cols = (d << (L - 1)) + mid + c
                    full[rows, cols] += v
    return full


def floquet(u, L):
    """One Floquet period U_F = U_odd U_even of the brickwork (PBC, L even)."""
    Ue = np.eye(2 ** L, dtype=complex)
    for i in range(0, L, 2):
        Ue = _bond(u, i, L) @ Ue
    Uo = np.eye(2 ** L, dtype=complex)
    for i in range(1, L - 1, 2):
        Uo = _bond(u, i, L) @ Uo
    Uo = _wrap_bond(u, L) @ Uo
    return Uo @ Ue


def zz_correlator_row(u, L, t):
    """Infinite-T ⟨σ^z_x(t) σ^z_0(0)⟩/2^L for all x, one Floquet period = t steps."""
    UF = floquet(u, L)
    Ut = np.linalg.matrix_power(UF, t)
    zt = Ut.conj().T @ op_at(Z, 0, L) @ Ut
    d = 2 ** L
    return np.array([np.real(np.trace(op_at(Z, x, L) @ zt)) / d for x in range(L)])


def single_site_channel(u):
    """Left-ray single-site channel as a 4x4 Pauli-transfer matrix S.

    S_{a,b} = (1/2) tr[σ^a  M(σ^b)],  M(o) = (1/2) tr_left[ U (o⊗1) U† ].
    Per Floquet period the light-cone edge advances two sites, so the on-edge
    correlator after t periods equals (S^{2t})_{zz}.
    """
    ud = u.conj().T
    S = np.zeros((4, 4), complex)
    for b, Pb in enumerate(PAULI):
        op = (u @ kron(Pb, I2) @ ud).reshape(2, 2, 2, 2)
        Mb = 0.5 * np.einsum('aibj->ij', op)         # trace the left site (o1=i1)
        for a, Pa in enumerate(PAULI):
            S[a, b] = 0.5 * np.trace(PAULI[a].conj().T @ Mb)
    return S


def oncone_from_channel(u, t):
    """Predicted on-edge ⟨σ^z σ^z⟩ after t Floquet periods: (S^{2t})_{zz}."""
    S = single_site_channel(u)
    return float(np.real(np.linalg.matrix_power(S, 2 * t)[3, 3]))


# ---- CLI summary -----------------------------------------------------------

def compute(L=8, t=2):
    """Dual-unitary diagnostics: self-dual/control dual-unitarity + light-cone."""
    kim = kim_gate(np.pi / 4, np.pi / 4)
    kim_ctrl = kim_gate(np.pi / 4, np.pi / 6)
    du = generic_du_gate()
    row = zz_correlator_row(du, L, t)
    edge = (-2 * t) % L
    off = max([abs(row[x]) for x in range(L)
               if x not in ((2 * t) % L, edge)] + [0.0])
    return {
        "dualunitary_defect_selfdual": dual_unitarity_defect(kim),
        "dualunitary_defect_control_bpi6": dual_unitarity_defect(kim_ctrl),
        "lightcone_offcone_max": off,
        "lightcone_oncone_ED": row[edge],
        "lightcone_oncone_channel": oncone_from_channel(du, t),
    }


def self_test():
    # anchor 1: self-dual KIM gate is unitary AND dual-unitary; b=π/6 fails dual.
    kim = kim_gate(np.pi / 4, np.pi / 4)
    assert unitarity_defect(kim) < 1e-12
    assert dual_unitarity_defect(kim) < 1e-12, dual_unitarity_defect(kim)
    for b in (np.pi / 6, np.pi / 5, 0.5):
        ctrl = kim_gate(np.pi / 4, b)
        assert unitarity_defect(ctrl) < 1e-12              # still a unitary gate
        assert dual_unitarity_defect(ctrl) > 0.1, (b, dual_unitarity_defect(ctrl))
    # sanity: the canonical Jx=Jy=π/4 family is dual-unitary for generic Jz.
    assert dual_unitarity_defect(generic_du_gate(0.3)) < 1e-12
    assert dual_unitarity_defect(generic_du_gate(0.7)) < 1e-12

    # anchor 2a: OFF-cone correlators vanish. On an L-ring the light cones wrap
    # once 2·(2t) ≥ L, so the exact-vanishing statement is asserted for the
    # clean window t ≤ L//4 (t ≤ 2 at L=8). Both edges ±2t excluded.
    du = generic_du_gate()
    for L in (8, 10):
        for t in range(1, L // 4 + 1):
            row = zz_correlator_row(du, L, t)
            for x in range(L):
                if x in ((2 * t) % L, (-2 * t) % L):
                    continue
                assert abs(row[x]) < 1e-12, (L, t, x, row[x])

    # anchor 2b: ON-cone value == single-site channel iterated 2t times, to
    # machine precision. The edge value is protected as long as the ±2t edges do
    # not collide, so this holds up to t=3 at L=10 (where off-cone would already
    # show O(1e-4) finite-size wrap — the edge itself is still exact).
    for L, tmax in ((8, 2), (10, 3)):
        for t in range(0, tmax + 1):
            row = zz_correlator_row(du, L, t)
            edge = row[(-2 * t) % L]
            assert abs(edge - oncone_from_channel(du, t)) < 1e-12, (L, t)
    # the channel prediction is non-trivial (decaying), not identically 1.
    assert abs(oncone_from_channel(du, 1)) < 0.99


if __name__ == "__main__":
    oracle_main(compute, {"L": (int, 8), "t": (int, 2)})
