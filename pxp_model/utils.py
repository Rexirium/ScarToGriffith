from functools import lru_cache

import numpy as np
from quspin.basis import spin_basis_1d
from scipy.optimize import brentq


@lru_cache(maxsize=None)
def fibonacci(n: int):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)


def canonical_weights(energies, target_energy):
    """Return finite beta and Gibbs weights matching a sector's mean energy.

    The target must lie strictly inside a nonconstant spectrum. Spectral-edge
    targets require infinite beta and are not supported here.
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


def max_overlap_indices(states, candidates):
    """Match each column of states to a column of candidates by squared overlap.

    Inputs have shapes (D, N) and (D, M), in the same basis and row order.
    Returns an integer array of shape (N,), with entry i equal to
    argmax_j |<candidates[:, j] | states[:, i]>|**2. Matches may repeat;
    ties select the first candidate. Columns are not normalized internally;
    supply normalized states to compare fidelities.
    """
    states, candidates = np.asarray(states), np.asarray(candidates)
    if states.ndim != 2 or candidates.ndim != 2:
        raise ValueError("states and candidates must be two-dimensional")
    if states.shape[0] != candidates.shape[0]:
        raise ValueError("states and candidates must have the same number of rows")
    if candidates.shape[1] == 0:
        raise ValueError("candidates must contain at least one state")
    overlaps = np.abs(candidates.conj().T @ states)**2
    return np.argmax(overlaps, axis=0)


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


def unfolded_spacings(energies, start=0, stop=None, window=20, etol=1e-6):
    """
    Unfolding using local mean density of state in a window with width 2 * window.
    Levels can be specified in a range betweeen start and stop.
    """
    E = np.sort(np.asarray(energies, dtype=float))
    
    if stop is None:
        stop = len(E) - 1
    
    idx_start = max(start, window)
    idx_stop = min(stop, len(E) - window - 1)

    if idx_stop <= idx_start:
        raise ValueError("level number less than the width of window")

    spacings = []

    for n in range(idx_start, idx_stop):
        local_mean_gap = (
            E[n + window] - E[n - window]
        ) / (2 * window)

        delta = E[n+1] - E[n]
        if local_mean_gap > 0 and delta > etol:
            spacings.append(delta / local_mean_gap)

    spacings = np.asarray(spacings)
    spacings /= np.mean(spacings)

    return spacings

def averaging(arr, window=10):
    """Return the local moving average of a one-dimensional array."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be one-dimensional")
    if not isinstance(window, (int, np.integer)) or window < 0:
        raise ValueError("window must be a non-negative integer")
    if arr.size == 0:
        return arr.astype(np.result_type(arr.dtype, np.float64))

    indices = np.arange(arr.size)
    starts = np.maximum(indices - window, 0)
    stops = np.minimum(indices + window + 1, arr.size)

    dtype = np.result_type(arr.dtype, np.float64)
    cumulative_sum = np.concatenate(
        (np.zeros(1, dtype=dtype), np.cumsum(arr, dtype=dtype))
    )
    return (cumulative_sum[stops] - cumulative_sum[starts]) / (stops - starts)
