"""Turner (2018), Eqs. (3)-(5): forward scattering approximation.

    python -m pxp_model.pxp_fsa --L 16

Pure PXP, Pauli X coefficient 1, periodic boundaries, even L. Bit 1 is
excited; site 0 is the most significant bit. We reuse pxp_basis_1d without
spatial symmetry reduction: |Z2>=|1010...> is not a one-site momentum state.
H+ flips spins away from the chosen initial product state. With the default
Z2 state it lowers occupied even sites and raises empty odd sites.
V[:, n] = |n>, beta[n] = ||H+ |n>||. Stop when H+ annihilates the vector;
the default Z2 state gives L+1 vectors, other states may give fewer.
H_FSA is constructed directly from beta, with zero diagonal.

NPZ arrays: rows of fsa_basis V follow pxp_basis_1d(L).states; eigenvectors[:,a]
contains the FSA coefficients of eigenstate a, ordered by ascending energies.
psi = fsa_basis @ eigenvectors contains the approximate physical states.
initial_overlaps gives their squared overlaps with the initial product state.

References: papers/Turner_2018_Weak_ergodicity_breaking.md;
.knowledge/models/rydberg-pxp/MODEL.md. Eq. (4)'s hopping sum ends at L-1.
Memory grows as O(D*(L+1)) plus sparse operators, D=F_(L-1)+F_(L+1).
This explicit-basis implementation is intended for modest sizes, not L=32
paper-scale runs. No full D-by-D dense Hamiltonian is constructed.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from quspin.operators import hamiltonian

if __package__:
    from .pxp_basis import pxp_basis_1d
else:
    from pxp_basis import pxp_basis_1d


def _initial_bitstring(L, initial_state):
    if initial_state is None:
        initial_state = "10" * (L // 2)
    if isinstance(initial_state, (int, np.integer)) and 0 <= initial_state < 2**L:
        initial_state = format(initial_state, f"0{L}b")
    if (not isinstance(initial_state, str) or len(initial_state) != L
            or set(initial_state) - {"0", "1"}):
        raise ValueError("initial_state must be an L-bit string or integer in [0, 2**L)")
    if "11" in initial_state + initial_state[0]:
        raise ValueError("initial_state violates the periodic nearest-neighbor blockade")
    return initial_state


def build_forward_operator(basis, initial_state=None):
    """Return QuSpin H+; constrained user_basis supplies the projectors."""
    if getattr(basis, "blocks", {}):
        raise ValueError("Use pxp_basis_1d(L) without spatial symmetry blocks")
    initial_state = _initial_bitstring(basis.N, initial_state)
    # Existing user_basis supports x,y,z, so sigma+/- = (X +/- iY)/2.
    return hamiltonian(
        [["x", [[0.5, i] for i in range(basis.N)]],
         ["y", [[-0.5j if initial_state[i] == "1" else 0.5j, i]
                for i in range(basis.N)]]],
        [], basis=basis, dtype=np.complex128,
        check_symm=False, check_pcon=False, check_herm=False,
    )


def construct_fsa_eigensystem(L, initial_state=None):
    """Return FSA basis, tridiagonal Hamiltonian, eigenpairs and physical states.

    initial_state is an L-bit product-state string (site 0 first) or its
    integer encoding; None selects |1010...>. It must obey the PBC blockade.
    Columns of fsa_basis and psi use the row order of pxp_basis_1d(L).states.
    eigenvectors contains FSA coefficients; psi contains physical coefficients.
    """
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
    initial_state = _initial_bitstring(L, initial_state)
    basis = pxp_basis_1d(L)
    forward = build_forward_operator(basis, initial_state)
    # X +/- iY is real in this product basis; keep the large arrays real.
    forward = hamiltonian([forward.tocsr().real], [], dtype=np.float64)
    V = np.zeros((basis.Ns, L + 1))
    V[basis.index(initial_state), 0] = 1.0
    beta = []
    for n in range(L):
        next_vector = forward.dot(V[:, n])
        norm = np.linalg.norm(next_vector)
        if not np.isfinite(norm):
            raise RuntimeError(f"FSA recurrence broke down at n={n}")
        if norm == 0:
            break
        beta.append(norm)
        V[:, n + 1] = next_vector / norm

    V = V[:, :len(beta) + 1].copy()
    beta = np.asarray(beta)
    H_fsa = np.diag(beta, 1) + np.diag(beta, -1)
    effective = hamiltonian([H_fsa], [], dtype=np.float64)
    energies, eigenvectors = effective.eigh()
    return dict(fsa_basis=V, beta=beta,
                H_fsa=H_fsa, energies=energies, eigenvectors=eigenvectors,
                initial_overlaps=np.abs(eigenvectors[0])**2,
                psi=V @ eigenvectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--initial-state", help="Initial L-bit product state; default: 1010...")
    parser.add_argument("--output", type=Path, help="Output NPZ path")
    args = parser.parse_args()
    result = construct_fsa_eigensystem(args.L, args.initial_state)
    output = args.output or Path(__file__).parent / "figures" / f"pxp_fsa_L={args.L}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **result)
    print(f"Constrained dimension={result['fsa_basis'].shape[0]}, "
          f"FSA dimension={result['fsa_basis'].shape[1]}")
    print("Energies:", result["energies"])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
