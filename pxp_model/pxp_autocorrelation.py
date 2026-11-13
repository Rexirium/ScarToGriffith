"""Projected and local connected Z correlations for the periodic PXP chain.

Both functions use a thermal reference in the k=0, p=+ sector.
connected_autocorrelation uses A = P_sector Z_1 P_sector = (sum_i Z_i)/L.
local_connected_autocorrelation retains all intermediate momentum sectors
and computes the site-averaged local correlator of the unprojected Z_j.
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
    """Normalized log Gibbs weights, with the first eigenstate at T=0."""
    shifted = energies - energies[0]
    logs = np.full((energies.size, temperatures.size), -np.inf)
    for column, temperature in enumerate(temperatures):
        if temperature == 0:
            logs[0, column] = 0.0
        else:
            # Extremely small T can send excited-state log weights to -inf.
            with np.errstate(over="ignore"):
                values = -shifted / temperature
            logs[:, column] = values - logsumexp(values)
    return logs


def _spectral_correlation(energies, intermediate_energies, strengths,
                          log_weights, times, imaginary):
    """Sum w_n |A_rn|^2 exp(s t (E_n-E_r)); strengths has axes (n, r)."""
    weights = np.exp(log_weights)
    gaps = energies[:, None] - intermediate_energies[None, :]
    result = np.empty((times.size, log_weights.shape[1]),
                      dtype=np.float64 if imaginary else np.complex128)
    for row, time in enumerate(times):
        if imaginary:
            for column in range(log_weights.shape[1]):
                # Combine Gibbs and evolution exponents before exponentiating:
                # separate factors would produce 0 * inf at low temperature.
                exponent = log_weights[:, column, None] + time * gaps
                with np.errstate(over="raise", invalid="raise"):
                    result[row, column] = np.sum(strengths * np.exp(exponent))
        else:
            correlations = np.sum(strengths * np.exp(1j * time * gaps), axis=1)
            result[row] = correlations @ weights
    return result


def _connected_from_spectrum(energies, observable, log_weights, times, imaginary):
    """Evaluate the connected correlator with both states in the same sector."""
    mean = observable.diagonal().real @ np.exp(log_weights)
    return _spectral_correlation(
        energies, energies, np.abs(observable)**2, log_weights, times, imaginary,
    ) - mean**2


def _magnetization_in_eigenbasis(basis, vectors):
    """Projected Z_1 equals average magnetization in the k=0 sector."""
    diagonal = np.fromiter(
        (2 * int(state).bit_count() / basis.N - 1 for state in basis.states),
        dtype=float, count=basis.Ns,
    )
    return vectors.conj().T @ (diagonal[:, None] * vectors)


def _validate_inputs(L, m, Ts, ts, imaginary):
    """Shared public argument validation and temperature/time array conversion."""
    if (not isinstance(L, (int, np.integer)) or isinstance(L, (bool, np.bool_))
            or not 2 <= L <= 32):
        raise ValueError("L must be an integer between 2 and 32")
    if np.ndim(m) != 0 or not np.isrealobj(m) or not np.isfinite(m):
        raise ValueError("m must be a finite real scalar")
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
    return temperatures, times


def _lift_to_translation_basis(source, target, vectors):
    """Embed k=0,p=+ states into k=0 without expanding the full PXP space.

    target must have only translation symmetry. On each translation orbit,
    the coefficient ratio is the ratio of source/target product-state
    amplitudes. get_amp also maps each representative to the source orbit.
    """
    representatives = target.states.copy()
    amplitudes = source.get_amp(representatives, mode="full_basis")
    target_amplitudes = target.get_amp(target.states.copy(), mode="full_basis")
    rows = np.flatnonzero(amplitudes)
    columns = source.Ns - 1 - np.searchsorted(source.states[::-1], representatives[rows])
    factors = np.real_if_close(amplitudes[rows] / target_amplitudes[rows])
    lifted = np.zeros((target.Ns, vectors.shape[1]), dtype=np.result_type(vectors, factors))
    lifted[rows] = factors[:, None] * vectors[columns]
    return lifted


def connected_autocorrelation(L, m, Ts, ts, imaginary=False):
    """Return <A(t) A(0)> - <A(t)><A(0)> in the k=0, p=+ sector.

    Parameters
    ----------
    L : int
        Number of sites, 2 <= L <= 32, with periodic boundary conditions.
        Dense full ED is practical only for modest sector dimensions.
    m : float
        Finite field in H = sum_i P_{i-1} X_i P_{i+1} - m sum_i Z_i.
    Ts : float or one-dimensional array_like
        Nonnegative temperatures. T=0 selects the sector ground state;
        positive infinity gives equal weights to all sector eigenstates.
    ts : one-dimensional array_like
        Finite times, retained in the supplied order.
    imaginary : bool, optional
        False: A(t) = exp(i H t) A exp(-i H t).
        True: A(tau) = exp(H tau) A exp(-H tau). The usual thermal
        interval is 0 <= tau <= 1/T; outside it values may overflow,
        in which case FloatingPointError is raised.

    Returns
    -------
    numpy.ndarray
        Shape (len(ts), number of temperatures), including for scalar Ts.
        Complex for real time (the unsymmetrized correlator), real for
        imaginary time. A is the projected Z_1, equivalent to average
        magnetization in this sector; in particular A**2 is not identity.

    Notes
    -----
    One full eigensystem is reused for every time and temperature. Since
    rho commutes with H, <A(t)> = <A(0)>. Working memory is O(D**2 + D*nT
    + nt*nT), with D the sector dimension; no time-by-D-by-D array is built.

    Example
    -------
    >>> C = connected_autocorrelation(8, 0.2, [0, 0.5, 1.0], [0, 0.1, 1])
    >>> C.shape
    (3, 3)
    """
    temperatures, times = _validate_inputs(L, m, Ts, ts, imaginary)
    basis = pxp_basis_1d(L, kblock=0, pblock=1)
    energies, vectors = build_hamiltonian(basis, m).eigh()
    observable = _magnetization_in_eigenbasis(basis, vectors)
    log_weights = _thermal_log_weights(energies, temperatures)
    return _connected_from_spectrum(energies, observable, log_weights, times, imaginary)


def local_connected_autocorrelation(L, m, Ts, ts, imaginary=False):
    """Return L^-1 sum_j [<Z_j(t)Z_j(0)> - <Z_j(t)><Z_j(0)>].

    Arguments, units, PBC, and output layout match connected_autocorrelation:
    Ts is a nonnegative scalar or 1D array (0 selects the ground state,
    +inf the uniform sector mixture); ts is a finite 1D array. Output has
    shape (len(ts), number of temperatures), complex for real time and real
    for imaginary time, with the same exp(+iHt) / exp(+H tau) convention.

    The reference rho = exp(-H_sector/T)/Z remains restricted to k=0,p=+.
    Intermediate states span every momentum sector of the constrained
    Hilbert space. Thus this is the full LOCAL correlator for that reference,
    not a full-Hilbert-space Gibbs ensemble. In particular the local operator
    leaves the reference sector, so finite-T KMS symmetry C(tau)=C(beta-tau)
    need not hold. C(0)=1-<Z_1>^2 provides a normalization check.

    Define B_q = L^-1 sum_j exp(+2 pi i q j/L) Z_j. The sign matches the
    translation callback in pxp_basis.py, and B_q maps k=0 to k=q. The raw
    local correlator is sum_q sum_{n in sector,r in q} w_n |<r|B_q|n>|^2
    exp(s*t*(E_n-E_r)), s=i or 1. The q=0 term is the existing projected
    connected correlator, which subtracts <Z_1>^2 exactly once.

    Op_shift_sector requires matching symmetry maps, so the reference
    eigenvectors are embedded into a translation-only k=0 basis first.
    Each q!=0 Hamiltonian is diagonalized separately, with no parity block
    and no full-space dense Hamiltonian or projector. All temperatures reuse
    these spectra. Dense ED cost is O(D0^3 + sum_{q!=0} Dq^3); only the
    reference and one target sector's eigenvectors are held at a time.
    Imaginary-time values outside the thermal interval may overflow.

    Method: https://quspin.github.io/QuSpin/examples/example19.html
    """
    temperatures, times = _validate_inputs(L, m, Ts, ts, imaginary)
    basis = pxp_basis_1d(L, kblock=0, pblock=1)
    energies, vectors = build_hamiltonian(basis, m).eigh()
    log_weights = _thermal_log_weights(energies, temperatures)
    observable = _magnetization_in_eigenbasis(basis, vectors)
    result = _connected_from_spectrum(energies, observable, log_weights, times, imaginary)
    if not times.size:
        return result

    source = pxp_basis_1d(L, kblock=0)
    source_vectors = _lift_to_translation_basis(basis, source, vectors)
    del vectors, observable
    for q in range(1, L):
        target = pxp_basis_1d(L, kblock=q)
        if not target.Ns:
            continue
        target_energies, target_vectors = build_hamiltonian(
            target, m, dtype=np.complex128,
        ).eigh()
        op_list = [["z", [j], np.exp(2j * np.pi * q * j / L) / L] for j in range(L)]
        shifted = target.Op_shift_sector(source, op_list, source_vectors,
                                         dtype=np.complex128)
        shifted = shifted.reshape(target.Ns, energies.size)
        strengths = np.abs(target_vectors.conj().T @ shifted).T**2
        result += _spectral_correlation(
            energies, target_energies, strengths, log_weights, times, imaginary,
        )
        del target_vectors, shifted, strengths
    return result
