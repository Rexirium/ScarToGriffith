"""Full-ensemble connected Z correlations for the periodic PXP chain.

All functions use rho = exp(-H/T)/Z over the FULL PXP-constrained space.
average_connected_autocorrelation measures M = (sum_j Z_j)/L;
local_connected_autocorrelation measures the site-averaged local Z_j trace.
connected_autocorrelation returns both, sharing a single set of eigensystems.
connected_autocorrelation_both_times shares eigensystems across real/imaginary time.
All momentum sectors, including both parities, enter the partition function.
Pauli and Hamiltonian conventions follow pxp_spectrum.py.
"""

if __package__:
    from .pxp_basis import pxp_basis_1d
    from .pxp_spectrum import build_hamiltonian
else:
    from pxp_basis import pxp_basis_1d
    from pxp_spectrum import build_hamiltonian

import numpy as np
from scipy.special import logsumexp


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


def _spectral_correlation(energies, intermediate_energies, strengths,
                          log_weights, times, imaginary):
    """Sum w_n |A_rn|^2 exp(s t (E_n-E_r)); strengths has axes (n, r)."""
    gaps = energies[:, None] - intermediate_energies[None, :]
    result = np.zeros((times.shape[0], log_weights.shape[1]),
                      dtype=np.float64 if imaginary else np.complex128)
    if not np.any(np.isfinite(log_weights)):
        return result
    if not imaginary:
        weights = np.exp(log_weights)
    for row, time in enumerate(times):
        if imaginary:
            for column in range(log_weights.shape[1]):
                # Combine Gibbs and evolution exponents before exponentiating:
                # separate factors would produce 0 * inf at low temperature.
                exponent = log_weights[:, column, None] + time[column] * gaps
                with np.errstate(over="raise", invalid="raise"):
                    result[row, column] = np.sum(strengths * np.exp(exponent))
        else:
            correlations = np.sum(strengths * np.exp(1j * time * gaps), axis=1)
            result[row] = correlations @ weights
    return result


def _magnetization_in_eigenbasis(basis, vectors):
    """Average magnetization is diagonal on translation-orbit representatives."""
    diagonal = np.fromiter(
        (2 * int(state).bit_count() / basis.N - 1 for state in basis.states),
        dtype=float, count=basis.Ns,
    )
    return vectors.conj().T @ (diagonal[:, None] * vectors)


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


def _momentum_eigensystems(L, m):
    """All translation-only blocks; reuse H_{L-k}=H_k.conj() for real PXP.

    Conjugate blocks have identical representatives and normalization in
    pxp_basis_1d. Their eigenvectors are conjugated without another ED.
    """
    sectors = []
    for k in range(L):
        basis = pxp_basis_1d(L, kblock=k)
        if k > L // 2:
            _, energies, vectors = sectors[L - k]
            vectors = vectors.conj()
        elif basis.Ns:
            dtype = np.float64 if (2 * k) % L == 0 else np.complex128
            energies, vectors = build_hamiltonian(basis, m, dtype=dtype).eigh()
        else:
            energies, vectors = np.empty(0), np.empty((0, 0))
        sectors.append((basis, energies, vectors))
    return sectors


def _full_ensemble(L, m, temperatures):
    """Diagonalize once, normalize Gibbs weights globally, then split by k."""
    sectors = _momentum_eigensystems(L, m)
    sizes = [len(energies) for _, energies, _ in sectors]
    energies = np.concatenate([energies for _, energies, _ in sectors])
    logs = _thermal_log_weights(energies, temperatures)
    return sectors, np.split(logs, np.cumsum(sizes)[:-1])


def _uniform_connected(sectors, log_weights, times, imaginary):
    """Correlate delta M: off-diagonal transitions plus global diagonal variance.

    Only the diagonal depends on the temperature-dependent global mean.
    Keeping its variance separate avoids subtracting nearly equal traces.
    Off-diagonal elements within degenerate energy subspaces are retained.
    """
    result = np.zeros((times.shape[0], log_weights[0].shape[1]),
                      dtype=np.float64 if imaginary else np.complex128)
    diagonals, weights = [], []
    for (basis, energies, vectors), logs in zip(sectors, log_weights):
        if not energies.size or not np.any(np.isfinite(logs)):
            continue
        observable = _magnetization_in_eigenbasis(basis, vectors)
        diagonals.append(observable.diagonal().real.copy())
        weights.append(np.exp(logs))
        strengths = np.abs(observable)**2
        np.fill_diagonal(strengths, 0.)
        result += _spectral_correlation(
            energies, energies, strengths, logs, times, imaginary,
        )
    diagonal = np.concatenate(diagonals)
    weights = np.concatenate(weights, axis=0)
    # Remove a common offset before forming the mean, preserving small fluctuations.
    shifted = diagonal - diagonal[0]
    mean_shift = shifted @ weights
    variance = np.sum(weights * (shifted[:, None] - mean_shift)**2, axis=0)
    return result + variance


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
    conjugate k and L-k blocks share eigenvalues and conjugated eigenvectors.
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
    translation convention B_q maps k to (k+q)%L. Op_shift_sector computes
    those matrix elements directly between translation-only bases. The raw
    local correlator sums w_kn |<k+q,r|B_q|k,n>|^2 exp(s*t*(E_kn-E_{k+q,r})),
    s=i or 1, over k,q,n,r with globally normalized weights.

    The q=0 contribution uses the centered uniform operator M-<M> to avoid
    subtracting the disconnected term numerically. Other q components are
    unchanged by centering Z_j-<Z_j>. For every unordered pair of
    different momentum blocks, compute the matrix elements only once:
    B_{-q}=B_q.dagger gives the reverse transition strengths by transposition.
    Reuse each block's eigensystem for all source/target pairs, temperatures
    and times. Only one pair's transition matrix is stored at a time; total
    eigensystem storage is O(sum_k D_k**2). No parity lifting is necessary.

    Method: https://quspin.github.io/QuSpin/examples/example19.html
    """
    return connected_autocorrelation(L, m, Ts, ts, imaginary)[1]


def _local_from_average(sectors, logs, times, imaginary, average):
    """Add q!=0 contributions without modifying the supplied average matrix."""
    L = len(sectors)
    result = average.copy()
    for k, (source, energies, vectors) in enumerate(sectors):
        if not energies.size:
            continue
        for target_k in range(k + 1, L):
            target, target_energies, target_vectors = sectors[target_k]
            if (not target_energies.size or not
                    (np.any(np.isfinite(logs[k])) or np.any(np.isfinite(logs[target_k])))):
                continue
            q = target_k - k
            op_list = [["z", [j], np.exp(2j * np.pi * q * j / L) / L]
                       for j in range(L)]
            shifted = target.Op_shift_sector(source, op_list, vectors,
                                             dtype=np.complex128)
            shifted = shifted.reshape(target.Ns, source.Ns)
            strengths = np.abs(target_vectors.conj().T @ shifted).T**2
            result += _spectral_correlation(
                energies, target_energies, strengths, logs[k], times, imaginary,
            )
            result += _spectral_correlation(
                target_energies, energies, strengths.T, logs[target_k], times, imaginary,
            )
    return result


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
    if not times.size:
        empty = np.empty((0, temperatures.size), dtype=float if imaginary else complex)
        return empty, empty.copy()
    sectors, logs = _full_ensemble(L, m, temperatures)
    average = _uniform_connected(sectors, logs, times, imaginary)
    local = _local_from_average(sectors, logs, times, imaginary, average)
    return average, local


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
    if not times.size and not imaginary_times.size:
        real = np.empty((0, temperatures.size), dtype=complex)
        imaginary = np.empty((0, temperatures.size), dtype=float)
        return (real, real.copy()), (imaginary, imaginary.copy())
    sectors, logs = _full_ensemble(L, m, temperatures)
    # shortcut: Recompute operators per domain; share them if profiling warrants it.
    results = []
    for grid, imaginary in ((times, False), (imaginary_times, True)):
        if not grid.size:
            empty = np.empty((0, temperatures.size), dtype=float if imaginary else complex)
            results.append((empty, empty.copy()))
            continue
        average = _uniform_connected(sectors, logs, grid, imaginary)
        local = _local_from_average(sectors, logs, grid, imaginary, average)
        results.append((average, local))
    return tuple(results)
