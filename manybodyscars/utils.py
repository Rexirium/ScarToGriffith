import warnings

import numpy as np
import scipy.linalg as sla
from quspin.basis import spin_basis_1d

def merge_basis_index(bases:list[spin_basis_1d]):
    
    merged_states = np.concatenate([basis.states for basis in bases])
    sort_idx = np.argsort(merged_states)[::-1]
    # 4. 计算逆排列 (求出原数组每个元素在新数组中的指标)
    new_indices = np.empty_like(sort_idx)
    new_indices[sort_idx] = np.arange(len(sort_idx))
    
    # 5. 按照原数组的长度，将 new_indices 切分回原来的结构
    lengths = [basis.Ns for basis in bases]
    split_points = np.cumsum(lengths)[:-1]  # 计算切分点
    basis_indices = np.split(new_indices, split_points)
    
    return basis_indices


def _get_bipartition(self, cut=None):
    """Return cached basis indices for the bipartition at ``cut``."""
    if cut is None:
        cut = self.N // 2

    cache = getattr(self, "_ent_entropy_cache", {})
    partition = cache.get(cut)
    if partition is None:
        # state = left * sps^(N-cut) + right in QuSpin's integer convention.
        divisor = self.sps ** (self.N - cut)
        left_parts, right_parts = divmod(self.states, divisor)
        left_states, l_idx = np.unique(left_parts, return_inverse=True)
        right_states, r_idx = np.unique(right_parts, return_inverse=True)
        partition = (l_idx, r_idx, left_states.size, right_states.size)
        cache[cut] = partition
        self._ent_entropy_cache = cache

    return partition


def my_ent_entropy(self, psi, cut=None, enforce_pure=False):
    """Return the von Neumann entropy of the first ``cut`` sites.

    ``psi`` may be a dense state vector, column-wise state vectors, or a
    density matrix. A square array is treated as a density matrix unless
    ``enforce_pure=True``, in which case its columns are treated as pure
    states. A single state returns a scalar; multiple states return an array.
    """
    psi = np.asarray(psi)

    # Symmetry-reduced states generally need projection to a symmetry-free
    # basis before their tensor-product structure can be identified.
    if hasattr(self, "blocks") and len(self.blocks) > 0:
        warnings.warn(
            "Use a symmetry-free basis for reliable entanglement entropy.",
            UserWarning,
            stacklevel=2,
        )
    
    l_idx, r_idx, n_left, n_right = _get_bipartition(self, cut)
    
    if not enforce_pure and psi.shape == (self.Ns, self.Ns):
        # Partial trace over the right subsystem:
        # rho_left[l, l'] = sum_r rho[(l, r), (l', r)].
        reduced = np.zeros((n_left, n_left), dtype=psi.dtype)
        for right in range(n_right):
            indices = np.flatnonzero(r_idx == right)
            left_indices = l_idx[indices]
            reduced[np.ix_(left_indices, left_indices)] += psi[
                np.ix_(indices, indices)
            ]
        ps = sla.eigvalsh(reduced, overwrite_a=True, check_finite=False)
        ps = ps[(ps > 1e-300) & (ps < 1.0)]
        return -np.sum(ps * np.log(ps))

    if psi.shape == (self.Ns,):
        states = psi[:, None]
    elif psi.shape == (1, self.Ns):
        states = psi.T
    elif psi.ndim == 2 and psi.shape[0] == self.Ns:
        states = psi
    else:
        raise ValueError(
            f"psi must have shape ({self.Ns},), ({self.Ns}, nvec), "
            f"or ({self.Ns}, {self.Ns}); got {psi.shape}"
        )

    entropies = np.empty(states.shape[1], dtype=np.float64)
    mat = np.zeros((n_left, n_right), dtype=psi.dtype, order="F")
    for j, state in enumerate(states.T):
        # Arrange each state's amplitudes as a bipartite coefficient matrix.
        # Its squared singular values are the eigenvalues of the reduced
        # density matrix.
        mat.fill(0)
        mat[l_idx, r_idx] = state
        sv = sla.svdvals(mat, overwrite_a=True, check_finite=False)
        ps = sv * sv
        ps = ps[(ps > 1e-300) & (ps < 1.0)]
        entropies[j] = -np.sum(ps * np.log(ps))

    return entropies[0] if entropies.size == 1 else entropies

spin_basis_1d.my_ent_entropy = my_ent_entropy


def my_negativity(self, psi, cut=None, log=False, enforce_pure=False):
    """Return the negativity across the bipartition at ``cut``.

    Set ``logarithmic=True`` for the logarithmic negativity. Square arrays are
    treated as density matrices unless ``enforce_pure=True``, in which case
    their columns are treated as pure states.
    """
    psi = np.asarray(psi)

    if hasattr(self, "blocks") and len(self.blocks) > 0:
        warnings.warn(
            "Use a symmetry-free basis for reliable negativity.",
            UserWarning,
            stacklevel=2,
        )

    l_idx, r_idx, n_left, n_right = _get_bipartition(self, cut)

    if not enforce_pure and psi.shape == (self.Ns, self.Ns):
        size = n_left * n_right
        rho_pt = np.zeros((size, size), dtype=psi.dtype, order="F")

        # rho[i, j] -> rho_pt[(l_j, r_i), (l_i, r_j)]. Construct the
        # partially-transposed matrix directly, without an intermediate copy.
        row_offsets = r_idx + size * l_idx * n_right
        column_offsets = l_idx * n_right + size * r_idx
        destination = row_offsets[:, None] + column_offsets[None, :]
        rho_pt.ravel(order="F")[destination] = psi

        # Negativity is minus the sum of the negative eigenvalues, so positive
        # eigenvalues need not be computed.
        neg_ps = sla.eigvalsh(
            rho_pt,
            overwrite_a=True,
            check_finite=False,
            subset_by_value=(-np.inf, 0.0),
        )
        negativity = -neg_ps.sum()
        if log:
            return np.log(np.trace(psi).real + 2.0 * negativity)
        return negativity

    if psi.shape == (self.Ns,):
        states = psi[:, None]
    elif psi.shape == (1, self.Ns):
        states = psi.T
    elif psi.ndim == 2 and psi.shape[0] == self.Ns:
        states = psi
    else:
        raise ValueError(
            f"psi must have shape ({self.Ns},), ({self.Ns}, nvec), "
            f"or ({self.Ns}, {self.Ns}); got {psi.shape}"
        )

    negativities = np.empty(states.shape[1], dtype=np.float64)
    mat = np.zeros((n_left, n_right), dtype=psi.dtype, order="F")
    for j, state in enumerate(states.T):
        mat.fill(0)
        mat[l_idx, r_idx] = state
        sv = sla.svdvals(mat, overwrite_a=True, check_finite=False)
        trace_norm = sv.sum() ** 2
        if log:
            negativities[j] = np.log(trace_norm)
        else:
            negativities[j] = 0.5 * (trace_norm - np.vdot(state, state).real)

    return negativities[0] if negativities.size == 1 else negativities


spin_basis_1d.my_negativity = my_negativity


def ops_ent_entropy(self, rho, cut=None):
    """Return the operator-space entanglement entropy of a density matrix."""
    rho = np.asarray(rho)
    if rho.shape != (self.Ns, self.Ns):
        raise ValueError(
            f"rho must have shape ({self.Ns}, {self.Ns}); got {rho.shape}"
        )

    if hasattr(self, "blocks") and len(self.blocks) > 0:
        warnings.warn(
            "Use a symmetry-free basis for reliable operator-space entropy.",
            UserWarning,
            stacklevel=2,
        )

    l_idx, r_idx, n_left, n_right = _get_bipartition(self, cut)
    n_rows = n_left * n_left
    n_cols = n_right * n_right

    # Write rho[i, j] directly at ((l_i, l_j), (r_i, r_j)), avoiding an
    # intermediate full product-basis density matrix and a transposed copy.
    operator_state = np.zeros((n_rows, n_cols), dtype=rho.dtype, order="F")
    row_offsets = l_idx * n_left + n_rows * r_idx * n_right
    column_offsets = l_idx + n_rows * r_idx
    destination = row_offsets[:, None] + column_offsets[None, :]
    operator_state.ravel(order="F")[destination] = rho

    if n_rows <= n_cols:
        gram = operator_state @ operator_state.conj().T
    else:
        gram = operator_state.conj().T @ operator_state

    ps = sla.eigvalsh(gram, overwrite_a=True, check_finite=False)
    ps /= ps.sum()
    ps = ps[ps > 1e-300]
    return -np.sum(ps * np.log(ps))


spin_basis_1d.ops_ent_entropy = ops_ent_entropy


if __name__ == "__main__":
    rng = np.random.default_rng()
    from numpy.linalg import norm
    
    basis = spin_basis_1d(L=10, Nup=5)
    basis_full = spin_basis_1d(L=10)
    b = 2
    
    psi = rng.normal(size = basis.Ns)
    psi /= norm(psi)
    psi_full = basis.project_from(psi, sparse=False)
    

    subA = tuple(range(b))
    myent = basis.my_ent_entropy(psi, b)
    myent_full = basis_full.my_ent_entropy(psi_full, b)
    qsent = basis.ent_entropy(psi, subA, density=False)
    qsent_full = basis_full.ent_entropy(psi_full, subA, density=False)
    
    print(f"my entropy is {myent}")
    print(f"my entropy full is {myent_full}")
    print(f"quspin entropy is {qsent["Sent_A"]}")
    print(f"quspin entropy full is {qsent_full["Sent_A"]}")
