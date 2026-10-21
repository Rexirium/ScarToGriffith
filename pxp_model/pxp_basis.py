"""Periodic blockade-constrained PXP basis, symmetry projections and FSA.

Bit 1 has Pauli Z=+1; site 0 is the most significant bit. Importing this
module registers projection, entanglement and FSA methods on user_basis.

    basis = pxp_basis_1d(16, kblock=0, pblock=1)
    result = basis.construct_fsa_eigensystem(m=0.2)
"""

from functools import lru_cache
from math import gcd

import numpy as np
from numba import carray, cfunc, int32, uint32
from quspin.basis.user import (
    user_basis,
    pre_check_state_sig_32,
    op_sig_32,
    map_sig_32,
)
from quspin.operators import hamiltonian
from scipy.sparse import csc_matrix, eye

if __package__:
    from .entanglement import my_ent_entropy, my_negativity, ops_ent_entropy
else:
    from entanglement import my_ent_entropy, my_negativity, ops_ent_entropy

# QuSpin callbacks: local Pauli operators, blockade and spatial symmetries.


@cfunc(op_sig_32, locals=dict(s=int32, b=uint32))
def op(op_struct_ptr, op_str, ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    ind = N - ind - 1  # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state >> ind) & 1) << 1) - 1
    b = 1 << ind
    if op_str == 120:  # ord("x")
        op_struct.state ^= b
    elif op_str == 121:  # ord("y")
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j * s
    elif op_str == 122:  # ord("z")
        op_struct.matrix_ele *= s
    else:
        op_struct.matrix_ele = 0
        err = -1
    return err


@cfunc(
    pre_check_state_sig_32,
    locals=dict(s_shift_left=uint32, s_shift_right=uint32),
)
def pre_check_state(s, N, args):
    """Require empty neighbors for each excitation; supports up to 32 sites."""
    mask = 0xFFFFFFFF >> (32 - N)  # works for lattices of up to 32 sites
    # cycle bits left by 1 periodically
    s_shift_left = ((s << 1) & mask) | ((s >> (N - 1)) & mask)
    # cycle bits right by 1 periodically
    s_shift_right = ((s >> 1) & mask) | ((s << (N - 1)) & mask)
    return (((s_shift_right | s_shift_left) & s)) == 0


@cfunc(
    map_sig_32,
    locals=dict(
        shift=uint32,
        xmax=uint32,
        x1=uint32,
        x2=uint32,
        period=int32,
        l=int32,
    ),
)
def translation(x, N, sign_ptr, args):
    """works for all system sizes N."""
    shift = args[0]  # translate state by shift sites
    period = N  # periodicity/cyclicity of translation
    xmax = args[1]
    l = (shift + period) % period
    x1 = x >> (period - l)
    x2 = (x << l) & xmax
    return x2 | x1


@cfunc(
    map_sig_32,
    locals=dict(
        out=uint32,
        s=int32,
    ),
)
def parity(x, N, sign_ptr, args):
    """works for all system sizes N."""
    out = 0
    s = args[0]  # N-1
    out ^= x & 1
    x >>= 1
    while x:
        out <<= 1
        out ^= x & 1
        x >>= 1
        s -= 1
    out <<= s
    return out


# Basis construction.


@lru_cache(maxsize=None)
def fibonacci(n: int):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)


def pxp_basis_1d(N: int, a: int = 1, kblock=None, pblock=None):
    """Build a periodic PXP basis with optional translation/reflection sectors."""
    if not isinstance(a, (int, np.integer)) or a <= 0:
        raise ValueError("a must be a positive integer")

    op_args = np.array([], dtype=np.uint32)

    maps = dict()
    arrays_to_keep = [op_args]
    est_size = int(1.618 ** N)
    # 1. 挂载平移对称性 (kblock)
    if kblock is not None:
        translation_period = N // gcd(N, a)
        if not (0 <= kblock < translation_period):
            raise ValueError(
                "kblock must be an integer between 0 and "
                f"{translation_period - 1} for N={N}, a={a}"
            )

        T_args = np.array([a, (1 << N) - 1], dtype=np.uint32)

        # 平移 a 个格点的本征值为
        # exp(-i * 2pi * kblock / translation_period)。
        maps["T_block"] = (
            translation,
            translation_period,
            kblock,
            T_args,
        )
        arrays_to_keep.append(T_args)
        est_size = est_size // translation_period + translation_period

    # 2. 挂载宇称对称性 (pblock)
    if pblock is not None:
        if pblock not in [1, -1]:
            raise ValueError("pblock must be either 1 (even parity) or -1 (odd parity)")

        P_args = np.array([N - 1], dtype=np.uint32)
        # 将用户友好的 1 和 -1 转换为 QuSpin user_basis 所需的 q
        # 本征值为 exp(-i * 2pi * q / 2) -> q=0 得 +1, q=1 得 -1
        q_parity = 0 if pblock == 1 else 1

        maps["P_block"] = (parity, 2, q_parity, P_args)
        arrays_to_keep.append(P_args)
        est_size = est_size // 2 + 2

    op_dict = dict(op=op, op_args=op_args)
    check_state = (pre_check_state, None)
    est_size = max(128, int(1.2 * est_size))
    # 构建 user_basis。
    basis = user_basis(
        np.uint32,
        N,
        op_dict,
        allowed_ops=set("xyz"),
        sps=2,
        pre_check_state=check_state,
        Ns_block_est=est_size,  # 动态估算内存分配量
        **maps,
    )
    # Keep callback arrays alive while Numba accesses their pointers.
    basis._kept_arrays = arrays_to_keep

    return basis


# Projection between symmetry sectors and the full constrained basis.


def pxp_get_proj(self, full_pxp_basis):
    """Return the cached expansion matrix P into the symmetry-free PXP basis.

    Parameters
    ----------
    full_pxp_basis : user_basis
        A symmetry-free PXP basis constructed beforehand with
        ``pxp_basis_1d(self.N)``. Its state ordering determines the rows of
        P; columns follow ``self.states``.

    Returns
    -------
    scipy.sparse.csc_matrix
        Shape ``(full_pxp_basis.Ns, self.Ns)``. ``P @ v`` expands a state;
        ``P.conj().T @ w`` projects back into this symmetry sector.
        The returned matrix is cached; copy it before modifying it.
        Neither basis may be modified after caching. For two symmetry-free
        bases, matching states and ordering are required and P is identity.
    """
    cache = getattr(self, "_pxp_projection_cache", {})
    cached = cache.get(id(full_pxp_basis))
    if cached is not None and cached[0] is full_pxp_basis:
        return cached[1]

    if full_pxp_basis.N != self.N:
        raise ValueError(
            "full_pxp_basis and the symmetry-reduced basis must have "
            "the same system size"
        )
    if getattr(full_pxp_basis, "blocks", {}):
        raise ValueError(
            "full_pxp_basis must not contain spatial symmetry blocks; "
            "construct it with pxp_basis_1d(N)"
        )

    if not getattr(self, "blocks", {}):
        if self is not full_pxp_basis and not np.array_equal(
                self.states, full_pxp_basis.states):
            raise ValueError("Symmetry-free bases must have identical states and ordering")
        projector = eye(self.Ns, format="csc", dtype=np.float64)
    else:
        representatives = full_pxp_basis.states.copy()
        amplitudes = np.real_if_close(
            self.get_amp(representatives, mode="full_basis")
        )
        rows = np.flatnonzero(amplitudes)
        columns = self.Ns - 1 - np.searchsorted(
            self.states[::-1], representatives[rows]
        )
        projector = csc_matrix(
            (amplitudes[rows], (rows, columns)),
            shape=(full_pxp_basis.Ns, self.Ns),
        )
    # Validate and construct once per pair of unchanged bases.
    cache[id(full_pxp_basis)] = (full_pxp_basis, projector)
    self._pxp_projection_cache = cache
    return projector


def pxp_project_from(self, v0, full_pxp_basis, enforce_pure=False):
    """Lift states or a density matrix to the symmetry-free PXP basis.

    This is the PXP analogue of QuSpin's ``basis.project_from()``. The
    output rows contain only blockade-compatible configurations, in the
    order of ``full_pxp_basis.states``, instead of all ``2**N`` spin states.

    Parameters
    ----------
    v0 : array_like
        A state vector of shape ``(self.Ns,)``, several state vectors stored
        column-wise with shape ``(self.Ns, nvec)``, or a square density matrix.
    full_pxp_basis : user_basis
        A symmetry-free PXP basis constructed with ``pxp_basis_1d(self.N)``.
    enforce_pure : bool, optional
        If false, a square ``v0`` is treated as a density matrix and
        transformed using ``P @ v0 @ P.conj().T``. If true, a square ``v0``
        is treated as column-wise pure states. The default is false.

    Returns
    -------
    numpy.ndarray
        State(s) expressed in the symmetry-free PXP-constrained basis.
    """
    v0 = np.asarray(v0)
    if v0.ndim not in (1, 2) or v0.shape[0] != self.Ns:
        raise ValueError(
            f"v0 must have shape ({self.Ns},) or ({self.Ns}, nvec)"
        )
    projector = self.pxp_get_proj(full_pxp_basis)
    if not getattr(self, "blocks", {}):
        return v0.copy()

    if not enforce_pure and v0.shape == (self.Ns, self.Ns):
        return np.asarray(projector @ v0 @ projector.conj().T)
    return np.asarray(projector @ v0)


# Forward scattering approximation from |Z2> = |1010...>.


def build_forward_operator(basis):
    """Return QuSpin H+; constrained user_basis supplies the projectors."""
    if getattr(basis, "blocks", {}):
        raise ValueError("Use pxp_basis_1d(L) without spatial symmetry blocks")
    # Existing user_basis supports x,y,z, so sigma+/- = (X +/- iY)/2.
    return hamiltonian(
        [["x", [[0.5, i] for i in range(basis.N)]],
         ["y", [[-0.5j if i % 2 == 0 else 0.5j, i]
                for i in range(basis.N)]]],
        [], basis=basis, dtype=np.complex128,
        check_symm=False, check_pcon=False, check_herm=False,
    )


def construct_fsa_eigensystem(self, m=0, normalized=False):
    """Return FSA basis, tridiagonal Hamiltonian, eigenpairs and physical states.

    self must be a basis created by pxp_basis_1d, with even N >= 4.
    m is the finite longitudinal field in H = sum_i PXP - m sum_i Z_i.
    It changes the effective diagonal, leaving the forward basis unchanged.
    The initial state is |1010...>. fsa_basis and psi have shape (self.Ns, N+1).
    For spatial symmetry blocks, both are projected into self. If normalized
    is True, psi columns with norm > 1e-12 are normalized; smaller
    columns are set to zero and retained to preserve eigenstate indexing.
    If False (default), psi retains the raw projection amplitudes.
    fsa_basis is never rescaled.
    The auxiliary symmetry-free basis and its projector are reused on later
    calls; as with pxp_get_proj, the bases must remain unchanged.

    V[:, n] = |n> follows self.states after projection; beta[n] = ||H+|n>||.
    H+ lowers occupied even sites and raises empty odd sites. H_fsa has
    hopping beta and diagonal -m <n|sum_i Z_i|n>, with Pauli X coefficient 1.
    eigenvectors[:, a] contains FSA coefficients in ascending energy order.
    With normalized=False, psi = fsa_basis @ eigenvectors. Projected
    fsa_basis need not be orthonormal. beta, H_fsa, energies, eigenvectors
    and initial_overlaps retain their full-space meaning; initial_overlaps
    refers to the unprojected Z2 state.

    Memory is O(D*(N+1)) plus sparse operators, D=F_(N-1)+F_(N+1).
    This explicit-basis implementation is intended for modest sizes, not
    N=32 paper-scale runs; no full D-by-D dense Hamiltonian is constructed.
    References: papers/Turner_2018_Weak_ergodicity_breaking.md, Eqs. (3)-(5)
    (the hopping sum ends at N-1); .knowledge/models/rydberg-pxp/MODEL.md.
    """
    L = self.N
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
    if not np.isfinite(m):
        raise ValueError("m must be finite")
    if getattr(self, "blocks", {}):
        full_basis = getattr(self, "_pxp_fsa_full_basis", None)
        if full_basis is None:
            full_basis = pxp_basis_1d(L)
            self._pxp_fsa_full_basis = full_basis
    else:
        full_basis = self
    forward = build_forward_operator(full_basis)
    # X +/- iY is real in this product basis; keep the large arrays real.
    forward = hamiltonian([forward.tocsr().real], [], dtype=np.float64)
    V = np.zeros((full_basis.Ns, L + 1))
    V[full_basis.index("10" * (L // 2)), 0] = 1.0
    beta = []
    for n in range(L):
        next_vector = forward.dot(V[:, n])
        norm = np.linalg.norm(next_vector)
        if not np.isfinite(norm) or norm == 0:
            raise RuntimeError(f"FSA recurrence broke down at n={n}")
        beta.append(norm)
        V[:, n + 1] = next_vector / norm

    beta = np.asarray(beta)
    H_fsa = np.diag(beta, 1) + np.diag(beta, -1)
    if m != 0:
        magnetization = np.fromiter(
            (2 * int(s).bit_count() - L for s in full_basis.states),
            dtype=np.float64, count=full_basis.Ns,
        )
        H_fsa += np.diag(-m * np.einsum("ij,i,ij->j", V, magnetization, V))

    effective = hamiltonian([H_fsa], [], dtype=np.float64)
    energies, eigenvectors = effective.eigh()
    if full_basis is not self:
        V = self.pxp_get_proj(full_basis).conj().T @ V
    psi = V @ eigenvectors
    if normalized:
        norms = np.linalg.norm(psi, axis=0)
        nonzero = norms > 1e-12
        psi[:, nonzero] /= norms[nonzero]
        psi[:, ~nonzero] = 0
    return dict(fsa_basis=V, beta=beta,
                H_fsa=H_fsa, energies=energies, eigenvectors=eigenvectors,
                initial_overlaps=np.abs(eigenvectors[0])**2,
                psi=psi)


# Register all PXP convenience methods together.
user_basis.my_ent_entropy = my_ent_entropy
user_basis.my_negativity = my_negativity
user_basis.ops_ent_entropy = ops_ent_entropy
user_basis.pxp_get_proj = pxp_get_proj
user_basis.pxp_project_from = pxp_project_from
user_basis.construct_fsa_eigensystem = construct_fsa_eigensystem


if __name__ == "__main__":
    N = 10
    basis = pxp_basis_1d(N, kblock=0)
    print(basis.Ns)

    psi = np.zeros(basis.Ns)
    Z2idx = basis.index("10" * (N // 2))
    psi[Z2idx] = 1.0

    full_basis = pxp_basis_1d(N)
    psi_full = basis.pxp_project_from(psi, full_basis)

    print(len(psi_full) == full_basis.Ns)
