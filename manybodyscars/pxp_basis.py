from functools import lru_cache
from math import gcd
from quspin.basis.user import user_basis  # Hilbert space user basis
from quspin.basis.user import (
    pre_check_state_sig_32,
    op_sig_32,
    map_sig_32,
)  # user_basis dtypes
from numba import carray, cfunc  # numba helper functions
from numba import uint32, int32  # numba data types
import numpy as np
from scipy.sparse import csc_matrix
from utils import *

#
######  function to call when applying operators
@cfunc(op_sig_32, locals=dict(s=int32, b=uint32))
def op(op_struct_ptr, op_str, ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    ind = N - ind - 1  # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state >> ind) & 1) << 1) - 1
    b = 1 << ind
    #
    if op_str == 120:  # "x" is integer value 120 (check with ord("x"))
        op_struct.state ^= b
    elif op_str == 121:  # "y" is integer value 120 (check with ord("y"))
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j * s
    elif op_str == 122:  # "z" is integer value 120 (check with ord("z"))
        op_struct.matrix_ele *= s
    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err

#
######  function to filter states/project states out of the basis
#
@cfunc(
    pre_check_state_sig_32,
    locals=dict(s_shift_left=uint32, s_shift_right=uint32),
)
def pre_check_state(s, N, args):
    """imposes that that a bit with 1 must be preceded and followed by 0,
    i.e. a particle on a given site must have empty neighboring sites.
    #
    Works only for lattices of up to N=32 sites (otherwise, change mask)
    #
    """
    mask = 0xFFFFFFFF >> (32 - N)  # works for lattices of up to 32 sites
    # cycle bits left by 1 periodically
    s_shift_left = ((s << 1) & mask) | ((s >> (N - 1)) & mask)
    #
    # cycle bits right by 1 periodically
    s_shift_right = ((s >> 1) & mask) | ((s << (N - 1)) & mask)
    #
    return (((s_shift_right | s_shift_left) & s)) == 0

#
######  define symmetry maps
#
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
    #
    l = (shift + period) % period
    x1 = x >> (period - l)
    x2 = (x << l) & xmax
    #
    return x2 | x1

#
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
    #
    out ^= x & 1
    x >>= 1
    while x:
        out <<= 1
        out ^= x & 1
        x >>= 1
        s -= 1
    #
    out <<= s
    return out

def pxp_basis_1d(N:int, a:int=1, kblock=None, pblock=None):
    if not isinstance(a, (int, np.integer)) or a <= 0:
        raise ValueError("a must be a positive integer")

    op_args = np.array([], dtype=np.uint32)
    T_args = np.array([a, (1 << N) - 1], dtype=np.uint32)
    P_args = np.array([N - 1], dtype=np.uint32)

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
    # 4. 构建 user_basis
    basis = user_basis(
        np.uint32,
        N,
        op_dict,
        allowed_ops=set("xyz"),
        sps=2,
        pre_check_state=check_state,
        Ns_block_est=est_size, # 动态估算内存分配量
        **maps,
    )
     # 因为 Numba 底层使用指针访问这些数组，如果对象在函数结束时被释放，会导致 Segment Fault (内存越界)。
    basis._kept_arrays = arrays_to_keep
    
    return basis


def pxp_project_from(self, v0, full_pxp_basis, enforce_pure=False):
    """Lift states or a density matrix to the symmetry-free PXP basis.

    This is the PXP analogue of QuSpin's ``basis.project_from()``.  The
    difference is that the output rows contain only blockade-compatible
    configurations, in the same (descending-integer) order as
    ``pxp_basis_1d(self.N).states``, instead of all ``2**N`` spin states.

    Parameters
    ----------
    v0 : array_like
        A state vector of shape ``(self.Ns,)``, several state vectors stored
        column-wise with shape ``(self.Ns, nvec)``, or a square density matrix.
    full_pxp_basis : user_basis
        A symmetry-free PXP basis constructed beforehand with
        ``pxp_basis_1d(self.N)``. Its state ordering determines the rows of
        the returned vector or matrix.
    enforce_pure : bool, optional
        If false, a square ``v0`` is treated as a density matrix and
        transformed using ``P @ v0 @ P.conj().T``. If true, a square ``v0``
        is treated as column-wise pure states. The default is false.

    Returns
    -------
    numpy.ndarray
        State(s) expressed in the symmetry-free PXP-constrained basis.
    """
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

    v0 = np.asarray(v0)
    if v0.ndim not in (1, 2) or v0.shape[0] != self.Ns:
        raise ValueError(
            f"v0 must have shape ({self.Ns},) or ({self.Ns}, nvec)"
        )
    is_density = not enforce_pure and v0.shape == (self.Ns, self.Ns)

    # Cache the symmetry expansion because orbit representatives and their
    # amplitudes depend only on the two bases, not on the projected vectors.
    cache = getattr(self, "_pxp_projection_cache", {})
    cached = cache.get(id(full_pxp_basis))
    if cached is None or cached[0] is not full_pxp_basis:
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
        cached = (full_pxp_basis, projector)
        cache[id(full_pxp_basis)] = cached
        self._pxp_projection_cache = cache

    _, projector = cached

    if is_density:
        return np.asarray(projector @ v0 @ projector.conj().T)
    return np.asarray(projector @ v0)

@lru_cache(maxsize=None)
def fibonacci(n:int):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)

user_basis.my_ent_entropy = my_ent_entropy
user_basis.my_negativity = my_negativity
user_basis.ops_ent_entropy = ops_ent_entropy
user_basis.pxp_project_from = pxp_project_from
##############################################################################
##############################################################################

if __name__ == '__main__':
    N = 10
    basis = pxp_basis_1d(N, kblock=0)
    print(basis.Ns)
    
    psi = np.zeros(basis.Ns)
    Z2idx = basis.index("10" * (N // 2))
    psi[Z2idx] = 1.0
    
    full_basis = pxp_basis_1d(N)
    psi_full = basis.pxp_project_from(psi, full_basis)
    
    print(len(psi_full) == full_basis.Ns)
