"""Turner (2018), Eqs. (3)-(5): forward scattering approximation.

    from pxp_model.pxp_basis import pxp_basis_1d
    import pxp_model.pxp_fsa

    result = pxp_basis_1d(16).construct_fsa_eigensystem()

Pure PXP, Pauli X coefficient 1, periodic boundaries, even L. Bit 1 is
excited; site 0 is the most significant bit. Starting from |Z2>=|1010...>,
H+ lowers occupied even sites and raises empty odd sites.
V[:, n] = |n>, beta[n] = ||H+ |n>||, giving L+1 FSA vectors.
H_FSA is constructed directly from beta, with zero diagonal.

Import pxp_fsa to register the method, then call
basis.construct_fsa_eigensystem() on a pxp_basis_1d instance.
Rows of fsa_basis V follow basis.states; eigenvectors[:,a]
contains the FSA coefficients of eigenstate a, ordered by ascending energies.
psi contains the approximate physical states, normalized column-wise by default.
For a symmetry-reduced basis, V and psi are projected with P.conj().T,
without column removal. Projected V is not normalized and need not be orthonormal.
With normalized=False, psi = fsa_basis @ eigenvectors retains projection weights.
beta, H_fsa, energies, eigenvectors and initial_overlaps retain their original
full-space FSA meaning; initial_overlaps refers to the unprojected Z2 state.

References: papers/Turner_2018_Weak_ergodicity_breaking.md;
.knowledge/models/rydberg-pxp/MODEL.md. Eq. (4)'s hopping sum ends at L-1.
Memory grows as O(D*(L+1)) plus sparse operators, D=F_(L-1)+F_(L+1).
This explicit-basis implementation is intended for modest sizes, not L=32
paper-scale runs. No full D-by-D dense Hamiltonian is constructed.
"""

import numpy as np
from quspin.basis.user import user_basis
from quspin.operators import hamiltonian

if __package__:
    from .pxp_basis import pxp_basis_1d
else:
    from pxp_basis import pxp_basis_1d


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


def construct_fsa_eigensystem(self, normalized=True):
    """Return FSA basis, tridiagonal Hamiltonian, eigenpairs and physical states.

    self must be a basis created by pxp_basis_1d, with even N >= 4.
    The initial state is |1010...>. fsa_basis and psi have shape (self.Ns, N+1).
    For spatial symmetry blocks, both are projected into self. If normalized
    is True (default), psi columns with norm > 1e-12 are normalized; smaller
    columns are set to zero and retained to preserve eigenstate indexing.
    If False, psi retains the raw projection amplitudes. fsa_basis is never
    rescaled. The effective eigensystem and
    initial_overlaps describe the original FSA before symmetry projection.
    The auxiliary symmetry-free basis and its projector are reused on later
    calls; as with pxp_get_proj, the bases must remain unchanged.
    """
    L = self.N
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
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


user_basis.construct_fsa_eigensystem = construct_fsa_eigensystem
