"""Full-ensemble connected Z correlations for the periodic PXP chain.

All functions use rho = exp(-H/T)/Z over the FULL PXP-constrained space.
average_connected_autocorrelation measures M = (sum_j Z_j)/L;
local_connected_autocorrelation measures the site-averaged local Z_j trace.
connected_autocorrelation returns both, sharing a single set of eigensystems.
connected_autocorrelation_both_times shares eigensystems across real/imaginary time.
All momentum sectors, including both parities, enter the partition function.
Pauli and Hamiltonian conventions follow pxp_spectrum.py.
Calculation functions are silent; the scan script reports task progress.
"""

if __package__:
    from .pxp_basis import pxp_basis_1d
    from .pxp_spectrum import build_hamiltonian
else:
    from pxp_basis import pxp_basis_1d
    from pxp_spectrum import build_hamiltonian

import numpy as np
from scipy.linalg.blas import get_blas_funcs
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


_SPECTRAL_BLOCK_ROWS = 512
_SPECTRAL_BLOCK_QUERIES = 128


def prepare_time_grid(Ts, ts, imaginary=False):
    """Return temperatures and the actual time grid, preserving input order.

    Real time stays 1D. Imaginary time has axes (time, temperature): for T>0
    use ts/max(ts)/T; at T=0 retain ts. Empty and all-zero grids stay empty
    and zero, respectively. A nonzero grid with max(ts)=0 cannot be normalized.
    """
    if not isinstance(imaginary, (bool, np.bool_)):
        raise ValueError("imaginary must be a boolean")
    if not np.isrealobj(Ts) or not np.isrealobj(ts):
        raise ValueError("Ts and ts must be real")
    temperatures = np.atleast_1d(np.asarray(Ts, dtype=float))
    times = np.asarray(ts, dtype=float)
    if (temperatures.ndim != 1 or not temperatures.size
            or np.any(np.isnan(temperatures)) or np.any(temperatures < 0)):
        raise ValueError("Ts must be a nonempty scalar or 1D array with T >= 0")
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("ts must be a finite 1D array")
    if imaginary:
        actual = np.broadcast_to(times[:, None], (times.size, temperatures.size)).copy()
        positive = temperatures > 0
        if times.size and np.any(times) and np.any(positive):
            maximum = times.max()
            if maximum == 0:
                raise ValueError("max(ts) must be nonzero to normalize a nonzero imaginary-time grid")
            with np.errstate(over="ignore", invalid="ignore"):
                actual[:, positive] = (times / maximum)[:, None] / temperatures[positive]
            if not np.all(np.isfinite(actual)):
                raise ValueError("normalized imaginary times must be finite")
        times = actual
    return temperatures, times


def _validate_inputs(L, m, Ts, ts, imaginary):
    """Shared public argument validation and temperature/time conversion."""
    if (not isinstance(L, (int, np.integer)) or isinstance(L, (bool, np.bool_))
            or not 2 <= L <= 32):
        raise ValueError("L must be an integer between 2 and 32")
    if np.ndim(m) != 0 or not np.isrealobj(m) or not np.isfinite(m):
        raise ValueError("m must be a finite real scalar")
    return prepare_time_grid(Ts, ts, imaginary)


def _thermal_log_weights(energies, temperatures):
    """Globally normalized weights for an unsorted concatenation of spectra.

    At T=0, mix ground states degenerate within eigensolver roundoff equally.
    """
    shifted = energies - energies.min()
    tolerance = 32 * np.finfo(float).eps * max(1., np.max(np.abs(energies)))
    ground = shifted <= tolerance
    logs = np.full((energies.size, temperatures.size), -np.inf)
    for column, temperature in enumerate(temperatures):
        if temperature == 0:
            logs[ground, column] = -np.log(np.count_nonzero(ground))
        else:
            # Extremely small T can send excited-state log weights to -inf.
            with np.errstate(over="ignore"):
                values = -shifted / temperature
            logs[:, column] = values - logsumexp(values)
    return logs


def _momentum_eigensystems(L, m):
    """All translation-only blocks; reuse H_{L-k}=H_k.conj() for real PXP.

    Conjugate blocks have identical representatives and normalization in
    pxp_basis_1d. Entries are (basis, energies, stored_vectors, conjugated):
    the actual eigenvectors are stored_vectors.conj() when conjugated is true.
    Conjugate partners share arrays instead of storing a second copy; callers
    must not modify them. QuSpin requires writable input buffers.
    """
    sectors = []
    for k in range(L):
        basis = pxp_basis_1d(L, kblock=k)
        conjugated = k > L // 2
        if conjugated:
            _, energies, vectors, _ = sectors[L - k]
        elif basis.Ns:
            dtype = np.float64 if (2 * k) % L == 0 else np.complex128
            energies, vectors = build_hamiltonian(basis, m, dtype=dtype).eigh()
        else:
            energies, vectors = np.empty(0), np.empty((0, 0))
        # QuSpin consumes C-order inputs; BLAS can use their transpose views.
        vectors = np.ascontiguousarray(vectors)
        sectors.append((basis, energies, vectors, conjugated))
    return sectors


def _full_ensemble(L, m, temperatures):
    """Diagonalize once, normalize Gibbs weights globally, then split by k."""
    sectors = _momentum_eigensystems(L, m)
    sizes = [len(energies) for _, energies, _, _ in sectors]
    energies = np.concatenate([energies for _, energies, _, _ in sectors])
    logs = _thermal_log_weights(energies, temperatures)
    return sectors, np.split(logs, np.cumsum(sizes)[:-1])


def _eigenbasis_product(vectors, shifted, conjugated=False):
    """Return V_actual.dagger @ shifted; V_actual is V or V.conj().

    C-contiguous inputs have F-contiguous transpose views. Compute the
    transposed product so BLAS needs no layout or explicit conjugate copy
    when both inputs already have the same dtype.
    """
    gemm = get_blas_funcs("gemm", (vectors, shifted))
    return gemm(1., shifted.T, vectors.T, trans_b=1 if conjugated else 2).T


def _magnetization_in_eigenbasis(basis, vectors):
    """Average magnetization is diagonal on translation-orbit representatives."""
    diagonal = np.fromiter(
        (2 * int(state).bit_count() / basis.N - 1 for state in basis.states),
        dtype=float, count=basis.Ns,
    )
    return _eigenbasis_product(vectors, diagonal[:, None] * vectors)


def _momentum_z_shift(source, target, vectors, q):
    """Apply B_q to columns in a k basis, returning columns in k+q (mod L).

    Both bases must use one-site translation only and the same representatives.
    B_q preserves each orbit; shared orbits have equal normalization in both
    sectors, including short orbits, so no orbit-length factor is needed.
    """
    L = source.N
    states, source_rows, target_rows = np.intersect1d(
        source.states, target.states, return_indices=True,
    )
    diagonal = np.zeros(states.size, dtype=complex)
    for j in range(L):
        z = 2 * ((states >> (L - 1 - j)) & 1).astype(float) - 1
        diagonal += np.exp(2j * np.pi * q * j / L) / L * z
    operator = csr_matrix((diagonal, (target_rows, source_rows)),
                          shape=(target.Ns, source.Ns))
    return operator @ vectors


def _spectral_correlation(energies, intermediate_energies, strengths,
                          log_weights, times, imaginary):
    """Sum w_n |A_rn|^2 exp(s t (E_n-E_r)) with query-batched real GEMM.

    strengths has axes (n, r). Factor phases in real time; in imaginary
    time combine log weights with the source exponent before exponentiating.
    Queries are times (real) or time/temperature pairs (imaginary), with
    O((N_source + N_target) * block_queries) factor workspace. Growing
    imaginary-time factors use the original row-batched sum instead.
    """
    result = np.zeros((times.shape[0], log_weights.shape[1]),
                      dtype=np.float64 if imaginary else np.complex128)
    if (not times.shape[0] or not intermediate_energies.size
            or not np.any(np.isfinite(log_weights))):
        return result
    if not imaginary:
        origin = min(energies.min(), intermediate_energies.min())
        source = energies - origin
        target = intermediate_energies - origin
        weights = np.exp(log_weights)
        for start in range(0, times.size, _SPECTRAL_BLOCK_QUERIES):
            stop = min(start + _SPECTRAL_BLOCK_QUERIES, times.size)
            grid = times[None, start:stop]
            angles = target[:, None] * grid
            real = strengths @ np.cos(angles)
            imag = strengths @ np.sin(angles)
            angles = source[:, None] * grid
            cosine, sine = np.cos(angles), np.sin(angles)
            result.real[start:stop] = (real * cosine + imag * sine).T @ weights
            result.imag[start:stop] = (real * sine - imag * cosine).T @ weights
        return result

    minimum, maximum = intermediate_energies.min(), intermediate_energies.max()
    for start in range(0, times.size, _SPECTRAL_BLOCK_QUERIES):
        stop = min(start + _SPECTRAL_BLOCK_QUERIES, times.size)
        rows, columns = np.divmod(np.arange(start, stop), log_weights.shape[1])
        tau = times[rows, columns]
        reference = np.where(tau >= 0, minimum, maximum)
        left = log_weights[:, columns] + (energies[:, None] - reference) * tau
        # Nonpositive factors avoid recovering underflowed values by amplification.
        stable = np.all(left <= 0, axis=0)
        if np.any(stable):
            with np.errstate(over="raise", invalid="raise"):
                source = np.exp(left[:, stable])
                target = np.exp((reference[stable] - intermediate_energies[:, None])
                                * tau[stable])
                result[rows[stable], columns[stable]] = np.sum(
                    source * (strengths @ target), axis=0,
                )
        for query in np.flatnonzero(~stable):
            row, column = rows[query], columns[query]
            for first in range(0, energies.size, _SPECTRAL_BLOCK_ROWS):
                last = min(first + _SPECTRAL_BLOCK_ROWS, energies.size)
                work = energies[first:last, None] - intermediate_energies[None, :]
                work *= tau[query]
                work += log_weights[first:last, column, None]
                with np.errstate(over="raise", invalid="raise"):
                    np.exp(work, out=work)
                    work *= strengths[first:last]
                    result[row, column] += np.sum(work)
    return result


def _uniform_connected(sectors, log_weights, times, imaginary):
    """Correlate delta M: off-diagonal transitions plus global diagonal variance.

    Only the diagonal depends on the temperature-dependent global mean.
    Keeping its variance separate avoids subtracting nearly equal traces.
    Off-diagonal elements within degenerate energy subspaces are retained.
    Evaluate each conjugate momentum pair once, including its multiplicity
    in both spectral contributions and globally normalized diagonal weights.
    """
    result = np.zeros((times.shape[0], log_weights[0].shape[1]),
                      dtype=np.float64 if imaginary else np.complex128)
    diagonals, weights = [], []
    L = len(sectors)
    for k, ((basis, energies, vectors, _), logs) in enumerate(
            zip(sectors[:L // 2 + 1], log_weights[:L // 2 + 1])):
        if not energies.size or not np.any(np.isfinite(logs)):
            continue
        multiplicity = 1 if (2 * k) % L == 0 else 2
        # Real diagonal M gives identical real diagonals and |M_nr|^2 for V*.
        observable = _magnetization_in_eigenbasis(basis, vectors)
        diagonals.append(observable.diagonal().real.copy())
        weights.append(multiplicity * np.exp(logs))
        strengths = np.abs(observable)
        del observable
        np.square(strengths, out=strengths)
        np.fill_diagonal(strengths, 0.)
        result += multiplicity * _spectral_correlation(
            energies, energies, strengths, logs, times, imaginary,
        )
        del strengths
    diagonal = np.concatenate(diagonals)
    weights = np.concatenate(weights, axis=0)
    # Remove a common offset before forming the mean, preserving small fluctuations.
    shifted = diagonal - diagonal[0]
    mean_shift = shifted @ weights
    variance = np.sum(weights * (shifted[:, None] - mean_shift)**2, axis=0)
    return result + variance


def _local_from_average(sectors, logs, times, imaginary, average):
    """Add q!=0 pair orbits with global Gibbs weights; leave average unchanged."""
    L = len(sectors)
    result = average.copy()
    # Canonical pairs start in the nonconjugated half of the momentum sectors.
    for k, (source, energies, vectors, _) in enumerate(sectors[:(L + 1) // 2]):
        if not energies.size:
            continue
        for target_k in range(k + 1, L):
            pair = (k, target_k)
            conjugate_pair = tuple(sorted(((-k) % L, (-target_k) % L)))
            if pair > conjugate_pair:
                continue
            multiplicity = 1 if pair == conjugate_pair else 2
            target, target_energies, target_vectors, target_conjugated = sectors[target_k]
            if (not target_energies.size or not
                    (np.any(np.isfinite(logs[k])) or np.any(np.isfinite(logs[target_k])))):
                continue
            q = target_k - k
            shifted = _momentum_z_shift(source, target, vectors, q)
            strengths = np.abs(_eigenbasis_product(
                target_vectors, shifted, target_conjugated)).T
            del shifted
            np.square(strengths, out=strengths)
            result += multiplicity * _spectral_correlation(
                energies, target_energies, strengths, logs[k], times, imaginary,
            )
            result += multiplicity * _spectral_correlation(
                target_energies, energies, strengths.T, logs[target_k], times, imaginary,
            )
            del strengths
    return result


def _connected_for_grids(L, m, temperatures, grids):
    """Evaluate validated time grids, preparing the full Gibbs ensemble once."""
    sectors = logs = None
    results = []
    # shortcut: Recompute operators per domain; share them if profiling warrants it.
    for times, imaginary in grids:
        if not times.size:
            empty = np.empty((0, temperatures.size), dtype=float if imaginary else complex)
            results.append((empty, empty.copy()))
            continue
        if sectors is None:
            sectors, logs = _full_ensemble(L, m, temperatures)
        average = _uniform_connected(sectors, logs, times, imaginary)
        local = _local_from_average(sectors, logs, times, imaginary, average)
        results.append((average, local))
    return tuple(results)


def average_connected_autocorrelation(L, m, Ts, ts, imaginary=False):
    """Return <M(t) M(0)> - <M>^2 in the full constrained Gibbs ensemble.

    M = sum_j Z_j/L is the average Pauli magnetization. The thermal trace
    includes ALL momentum sectors and both parities, rather than only k=0,p=+.

    Parameters
    ----------
    L : int
        Number of sites, 2 <= L <= 32, with periodic boundary conditions.
        Dense full ED of the momentum blocks is practical only at modest L.
    m : float
        Finite field in H = sum_i P_{i-1} X_i P_{i+1} - m sum_i Z_i.
    Ts : float or one-dimensional array_like
        Nonnegative temperatures, normalized by ONE partition function over
        the full constrained space. T=0 selects the global ground state
        (equal mixture if degenerate within numerical precision); +inf gives
        equal weight to every constrained state, not to every sector.
    ts : one-dimensional array_like
        Finite times, retained in the supplied order. Empty arrays are valid.
        For imaginary=True and T>0, actual tau = ts/max(ts)/T; at T=0,
        actual tau = ts unchanged. All-zero grids stay zero. For nonnegative
        grids the finite-temperature times lie in [0,1/T].
    imaginary : bool, optional
        False: M(t) = exp(i H t) M exp(-i H t).
        True: M(tau) = exp(H tau) M exp(-H tau), in units hbar=k_B=1.
        Negative input times can leave the thermal interval and overflow.

    Returns
    -------
    numpy.ndarray
        Shape (len(ts), number of temperatures), including for scalar Ts.
        Complex for real time (unsymmetrized), real for imaginary time.

    Notes
    -----
    M preserves momentum, so only within-block matrix elements are needed.
    Each independent momentum block is diagonalized once per function call;
    conjugate k and L-k blocks share eigenvalues and eigenvector storage.
    Their magnetization contributions are also evaluated once with multiplicity.
    Spectra are reused for every temperature and time, with no persistent
    cache. Eigensystem storage is O(sum_k D_k**2), plus thermal weights and
    the output. No full-space dense Hamiltonian or time-by-D-by-D array is
    constructed. Evaluate delta M = M - <M> directly: sum off-diagonal
    transitions and the global weighted variance of M_nn. This retains
    fluctuations between sectors without a final subtraction of <M>^2,
    improving accuracy when the connected correlator is very small.

    Example
    -------
    >>> C = average_connected_autocorrelation(8, 0.2, [0, 0.5, 1.0], [0, 0.1, 1])
    >>> C.shape
    (3, 3)
    """
    temperatures, times = _validate_inputs(L, m, Ts, ts, imaginary)
    if not times.size:
        return np.empty((0, temperatures.size), dtype=float if imaginary else complex)
    sectors, logs = _full_ensemble(L, m, temperatures)
    return _uniform_connected(sectors, logs, times, imaginary)


def local_connected_autocorrelation(L, m, Ts, ts, imaginary=False):
    """Return L^-1 sum_j [<Z_j(t) Z_j(0)> - <Z_j>^2] in the full Gibbs state.

    Parameters, units, validation and output layout match
    average_connected_autocorrelation. ALL initial and intermediate momentum sectors
    are included. In particular C(0)=1-<Z_1>^2, and at finite positive T the
    imaginary-time result obeys C(tau)=C(1/T-tau).

    Define B_q = L^-1 sum_j exp(+2 pi i q j/L) Z_j. For this project's
    translation convention B_q maps k to (k+q)%L. Its orbit-diagonal action
    gives those matrix elements between translation-only bases. The raw
    local correlator sums w_kn |<k+q,r|B_q|k,n>|^2 exp(s*t*(E_kn-E_{k+q,r})),
    s=i or 1, over k,q,n,r with globally normalized weights.

    The q=0 contribution uses the centered uniform operator M-<M> to avoid
    subtracting the disconnected term numerically. Other q components are
    unchanged by centering Z_j-<Z_j>. For every unordered pair of
    different momentum blocks, compute the matrix elements only once:
    B_{-q}=B_q.dagger gives the reverse transition strengths by transposition.
    Complex-conjugate unordered pairs {k,k'} and {-k,-k'} have identical
    bidirectional contributions. Evaluate one representative with multiplicity
    two, or one when the pair maps to itself; Gibbs weights remain global.
    Reuse each block's eigensystem for all source/target pairs, temperatures
    and times. Only one pair's transition matrix is stored at a time; total
    eigensystem storage is O(sum_k D_k**2). No parity lifting is necessary.

    Method: https://quspin.github.io/QuSpin/examples/example19.html
    """
    return connected_autocorrelation(L, m, Ts, ts, imaginary)[1]


def connected_autocorrelation(L, m, Ts, ts, imaginary=False):
    """Return (average, local) connected correlations with shared diagonalization.

    Parameters and full-PXP Gibbs reference match average_connected_autocorrelation.
    The first matrix is <M(t) M(0)> - <M>^2 for M=sum_j Z_j/L; the second
    is L^-1 sum_j [<Z_j(t) Z_j(0)> - <Z_j>^2]. Both have shape
    (len(ts), number of temperatures), including when Ts is a scalar.
    Both are complex for real time and real for imaginary time.

    Compute all independent momentum eigensystems, global thermal weights
    and the centered q=0 correlation once. The local result adds the q!=0
    contributions to a copy of the average result. The returned matrices
    do not share storage. Empty ts returns two empty matrices without ED.

    Example
    -------
    >>> average, local = connected_autocorrelation(8, 0.2, [0, 0.5], [0, 0.1, 1])
    >>> average.shape, local.shape
    ((3, 2), (3, 2))
    """
    temperatures, times = _validate_inputs(L, m, Ts, ts, imaginary)
    return _connected_for_grids(L, m, temperatures, ((times, imaginary),))[0]


def connected_autocorrelation_both_times(L, m, Ts, ts, taus):
    """Return ((average_real, local_real), (average_imaginary, local_imaginary)).

    Conventions and validation match connected_autocorrelation. ts and taus
    are independent 1D real- and imaginary-time grids, respectively. Each
    output has axes (time, temperature); real-time outputs are complex,
    imaginary-time outputs are real. Here "real" names the time domain,
    not the real part of the correlation.

    Actual imaginary time is taus/max(taus)/T for T>0, and taus unchanged
    for T=0, exactly as in connected_autocorrelation(..., imaginary=True).

    Reuse one set of eigensystems and global Gibbs weights for both domains.
    Evaluate each domain with the same helpers as connected_autocorrelation;
    operator matrix elements and diagonal variance are recomputed per domain.
    Both grids may be empty; if both are empty, no diagonalization is performed.
    """
    temperatures, times = _validate_inputs(L, m, Ts, ts, False)
    _, imaginary_times = prepare_time_grid(temperatures, taus, True)
    return _connected_for_grids(
        L, m, temperatures, ((times, False), (imaginary_times, True)),
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L, m = 20, 0.2
    Ts = np.linspace(0.0, 1.0, 6)
    ts = np.linspace(0.0, 100.0, 101)

    corrs = local_connected_autocorrelation(L, m, Ts, ts, imaginary=False)

    fig, ax = plt.subplots()
    ax.plot(ts, corrs.real)
    ax.set(xlabel="$\\tau$", ylabel="$C(\\tau)$")
    ax.legend()
    plt.show()
