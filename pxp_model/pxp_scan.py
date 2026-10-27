"""Scan H = H_PXP - m sum_i Z_i in the PBC k=0, p=+ sector.

    python pxp_model/pxp_scan.py --workers 7 --threads 8
    python pxp_model/pxp_scan.py --L 20 --workers 1 --threads 8

The default sizes are 16, 18, ..., 28. Each spawned process scans one L
and writes its own NPZ; m points remain sequential for scar tracking.
Each invocation writes to <output-dir>/YYYYMMDD_HHMMSS (UTC+08:00),
with one shared timestamp for all L workers.
Each L reuses its two Hilbert-space bases, projector, operators, entropy
partition and m-independent FSA recurrence throughout the scan.
--threads sets OpenMP/BLAS thread limits before the child imports NumPy.
Allocate at least workers * threads CPUs on a single node.

Uses the Pauli conventions of pxp_spectrum.py and
.knowledge/models/rydberg-pxp/MODEL.md. At every m, resolve degeneracies
against |Z2,+> and match normalized FSA projections of weight > 1/2.
matched_scar_indices follows the retained FSA states in ascending FSA energy.
tracked_scar_indices starts with the unique m=0 matches and follows independent
maximum-overlap matches to the previous scars at subsequent points.
Both contain zero-based indices into ascending ED energies; repeated matches
are retained (no one-to-one assignment).

One pxp_scan_L=<L>.npz contains L, ms, times, energies,
matched_scar_indices, tracked_scar_indices, fsa_energies,
fsa_overlaps = |<FSA|Z2,+>|**2 (after projection and normalization),
fsa_z1_expectations = <FSA|Z_1|FSA> for the same retained, normalized states,
z1_expectations, overlaps = |<E_n|Z2,+>|**2, entropies (half-chain,
natural log), z1_t, beta_th, and z1_th. The thermal values use the canonical
ensemble within k=0, p=+, matching the energy of |Z2,+> at each m.
All observables have m as their first axis:
spectrum arrays are (21, dim), matched_scar_indices and FSA arrays are
(21, n_fsa), tracked_scar_indices is (21, n_initial_scars), and z1_t is
(21, len(times)); beta_th and z1_th have shape (21,).
L and times are stored only once.
Default times are [0] followed by np.geomspace(0.1, 1e6, 1001).
Evolution is exp(-i H t) from t=0, with hbar=1.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
import os
from pathlib import Path

if __package__:
    from .pxp_spectrum import build_hamiltonian, FSA_WEIGHT_CUTOFF
    from .pxp_basis import fibonacci, pxp_basis_1d, resolve_degeneracies
    from .utils import max_overlap_indices
else:
    from pxp_spectrum import build_hamiltonian, FSA_WEIGHT_CUTOFF
    from pxp_basis import fibonacci, pxp_basis_1d, resolve_degeneracies
    from utils import max_overlap_indices
import numpy as np
from quspin.operators import hamiltonian
from scipy.optimize import brentq
from scipy.sparse import diags


def canonical_weights(energies, target_energy):
    """Return finite beta and Gibbs weights matching a sector's mean energy.

    The target must lie strictly inside a nonconstant spectrum. Spectral-edge
    targets require infinite beta and are not part of this scan.
    """
    energies = np.asarray(energies, dtype=float)
    if (energies.ndim != 1 or energies.size < 2
            or not np.all(np.isfinite(energies)) or not np.isfinite(target_energy)):
        raise ValueError("energies must be a finite 1D spectrum and target_energy finite")
    low, high = energies.min(), energies.max()
    if not low < target_energy < high:
        raise ValueError("target_energy must lie strictly inside the spectrum")
    width = high - low
    scaled = (energies - target_energy) / width

    def weights(x):
        # Anchor at the favored spectral edge so all exponents are nonpositive.
        anchor = scaled.min() if x >= 0 else scaled.max()
        values = np.exp(-x * (scaled - anchor))
        return values / values.sum()

    def residual(x):
        return float(weights(x) @ scaled)

    at_zero = residual(0.0)
    if abs(at_zero) <= 1e-12:
        return 0.0, weights(0.0)
    bound = float(np.sign(at_zero))
    for _ in range(64):
        if residual(bound) * at_zero <= 0:
            break
        bound *= 2
    else:
        raise RuntimeError("could not bracket the energy-matching inverse temperature")
    x = brentq(residual, min(0.0, bound), max(0.0, bound), xtol=1e-12, rtol=1e-12)
    return x / width, weights(x)


def scan(L, times=None):
    """Scan m=0:0.05:1, replacing 0.65 with m_c=0.655, reusing bases/operators."""
    if not isinstance(L, (int, np.integer)) or L < 4 or L > 32 or L % 2:
        raise ValueError("L must be an even integer between 4 and 32")
    times = np.r_[0.0, np.geomspace(0.1, 1e6, 1001)] if times is None else np.asarray(times, dtype=float)
    if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be a nonempty, finite, increasing array with t >= 0")
    full_basis = pxp_basis_1d(L)
    assert full_basis.Ns == fibonacci(L - 1) + fibonacci(L + 1)
    basis = pxp_basis_1d(L, a=1, kblock=0, pblock=1)
    basis._pxp_fsa_full_basis = full_basis
    projector = basis.pxp_get_proj(full_basis)
    dim = basis.Ns
    print(f"L={L}, PBC, k=0, p=+; dimension={dim}/{full_basis.Ns}; "
          f"dense H + eigenvectors >= {16 * dim**2 / 2**20:.1f} MiB", flush=True)
    H0 = build_hamiltonian(basis, 0)
    magnetization = np.fromiter((2 * int(s).bit_count() - L for s in basis.states),
                               dtype=float, count=dim)
    field = hamiltonian([diags(magnetization)], [], dtype=np.float64)
    # Within k=0, P^dagger Z_1 P = P^dagger (sum_i Z_i / L) P.
    z1_diagonal = magnetization / L
    z2_index = basis.index("10" * (L // 2))
    z2_plus = np.zeros(dim)
    z2_plus[z2_index] = 1
    entropy_batch = max(1, min(16, (16 * 2**20) // (full_basis.Ns * 8)))
    previous_scars = None
    ms = np.linspace(0.0, 1.0, 21)
    ms[13] = 0.655  # Replace m=0.65 with m_c.
    for m in ms:
        H = H0 - m * field
        E, U = H.eigh()
        resolve_degeneracies(E, U, z2_plus)
        fsa = basis.construct_fsa_eigensystem(normalized=False, m=m)
        weights = np.sum(np.abs(fsa["psi"])**2, axis=0)
        valid = weights > FSA_WEIGHT_CUTOFF
        psi_fsa = fsa["psi"][:, valid] / np.sqrt(weights[valid])
        matched_scar_indices = max_overlap_indices(psi_fsa, U)
        fsa_energies = fsa["energies"][valid]
        fsa_overlaps = np.abs(psi_fsa[z2_index, :])**2
        fsa_z1_expectations = z1_diagonal @ np.abs(psi_fsa)**2
        del fsa, psi_fsa
        if m == 0:
            tracked_scar_indices = np.unique(matched_scar_indices)
        else:
            tracked_scar_indices = max_overlap_indices(previous_scars, U)
        previous_scars = U[:, tracked_scar_indices]
        entropies = np.empty(dim)
        z1_expectations = np.empty(dim)
        residual = 0.0
        for start in range(0, dim, entropy_batch):
            block = slice(start, start + entropy_batch)
            vectors = U[:, block]
            residual = max(residual, np.max(np.linalg.norm(H.dot(vectors) - vectors * E[block], axis=0)))
            z1_expectations[block] = z1_diagonal @ np.abs(vectors)**2
            entropies[block] = full_basis.my_ent_entropy(projector @ vectors, enforce_pure=True)
        if residual > 1e-9 * max(1.0, np.max(np.abs(E))):
            raise RuntimeError(f"m={m:g}: eigenpair residual too large: {residual:g}")
        coefficients = U[z2_index, :].conj()
        overlaps = np.abs(coefficients)**2
        beta_th, thermal_weights = canonical_weights(E, overlaps @ E)
        z1_th = float(thermal_weights @ z1_expectations)
        z1_t = np.empty(times.size)
        for start in range(0, times.size, 128):
            block = slice(start, start + 128)
            states_t = U @ (coefficients[:, None] * np.exp(-1j * E[:, None] * times[block]))
            z1_t[block] = z1_diagonal @ np.abs(states_t)**2
        print(f"L={L}, m={m:g}: matched={matched_scar_indices.tolist()}, "
              f"tracked={tracked_scar_indices.tolist()}, max residual={residual:.2e}", flush=True)
        yield dict(L=L, m=m, times=times, energies=E,
                   matched_scar_indices=matched_scar_indices, tracked_scar_indices=tracked_scar_indices,
                   fsa_energies=fsa_energies, fsa_overlaps=fsa_overlaps,
                   fsa_z1_expectations=fsa_z1_expectations,
                   z1_expectations=z1_expectations, overlaps=overlaps,
                   entropies=entropies, z1_t=z1_t, beta_th=beta_th, z1_th=z1_th)
        del U, vectors, states_t, coefficients, H


def save_results(results, output_dir):
    """Consume scan results and save all parameter points in one numeric NPZ."""
    results = list(results)
    if not results:
        raise ValueError("results must contain at least one parameter point")
    L, times = results[0]["L"], results[0]["times"]
    if any(r["L"] != L or not np.array_equal(r["times"], times) for r in results):
        raise ValueError("all results must have the same L and times")
    arrays = {key: np.stack([r[key] for r in results]) for key in
              ("energies", "matched_scar_indices", "tracked_scar_indices", "fsa_energies",
               "fsa_overlaps", "fsa_z1_expectations", "z1_expectations", "overlaps", "entropies", "z1_t",
               "beta_th", "z1_th")}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pxp_scan_L={L}.npz"
    np.savez_compressed(path, L=L, ms=np.array([r["m"] for r in results]), times=times,
                        **arrays, kblock=0, parity=1, initial_state="Z2_plus",
                        fsa_weight_cutoff=FSA_WEIGHT_CUTOFF,
                        entropy_cut=L // 2, entropy_log_base="e")
    return path


def run_size(L, output_dir):
    """Keep bases and eigenvectors inside the worker; return only the saved path."""
    return save_results(scan(L), output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, nargs="+", default=list(range(16, 29, 2)),
                        help="even chain lengths (default: 16 18 20 22 24 26 28)")
    parser.add_argument("--workers", type=int, default=7, help="maximum concurrent sizes (default: 7)")
    parser.add_argument("--threads", type=int, default=8, help="OpenMP/BLAS threads per worker (default: 8)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="parent directory for a YYYYMMDD_HHMMSS run subdirectory (UTC+08:00)")
    args = parser.parse_args()
    if any(L < 4 or L > 32 or L % 2 for L in args.L) or len(set(args.L)) != len(args.L):
        parser.error("--L must contain distinct even integers between 4 and 32")
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be positive")
    workers = min(args.workers, len(args.L))
    if "SLURM_CPUS_PER_TASK" in os.environ and workers * args.threads > int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("workers * threads exceeds SLURM_CPUS_PER_TASK")
    # spawn inherits these settings before importing numerical libraries.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    output_dir = args.output_dir / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    print(f"Sizes={args.L}; workers={workers}, threads/worker={args.threads}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_size, L, output_dir) for L in args.L]
        for future in as_completed(futures):
            print(f"Saved {future.result()}", flush=True)


if __name__ == "__main__":
    main()
