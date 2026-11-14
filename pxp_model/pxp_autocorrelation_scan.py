"""Scan full-ensemble PXP correlations with one spawned task per (L, m).

    python pxp_model/pxp_autocorrelation_scan.py --config path/to/scan.toml

Requires Python >= 3.11 and pxp_model/requirements.txt. The default config
is pxp_autocorrelation_scan.toml beside this script. sizes is an integer
array; other grids accept arrays or {start, stop, length, scale} tables.
scale defaults to "linear"; "log10" interprets start/stop as base-10 exponents.
workers and threads control LOCAL process count and BLAS/OpenMP threads per
process. Thread limits are inherited before workers import NumPy/QuSpin.
No additional pools are created for temperatures, times or momentum blocks.
imaginary_times specifies the input grid: the correlation functions use
ts/max(ts)/T for T>0 and retain ts at T=0. No scaling is done by the scanner.
Memory is checked before starting workers using Slurm's allocation and/or
memory_gib, a per-node budget in GiB. One Slurm task per node is required
for this memory accounting. Without an identifiable allocation, set memory_gib.

For Slurm, use submit_autocorrelation.sh: srun launches one copy per node.
SLURM_PROCID and SLURM_NTASKS partition (L,m) points without MPI dependencies.
Larger sizes are queued first, with each size spread across ranks. Each rank
has its own local pool; there is no work stealing between nodes.

output_dir is relative to the TOML file, unless absolute. Each invocation
creates a new YYYYMMDD_HHMMSS_microseconds directory (UTC+08:00), copies the
config there, and saves one NPZ per (L, m). Completed files survive a later
task failure; failed tasks cause a nonzero exit. Existing runs are preserved.
Multi-rank runs require the SAME --run-id on every rank and shared storage
for the code, config and output. Results and config copies go under
output_dir/<run-id>/rank_<rank>/; an existing rank directory is rejected.
Single-rank runs keep the flat timestamped layout unless --run-id is given.

Each NPZ contains L, m, m_index (zero-based index into ms), temperatures,
real_times, imaginary_times_input, imaginary_times,
elapsed_seconds, and six float64 correlation matrices:
average_real_time_real, average_real_time_imag,
local_real_time_real, local_real_time_imag,
average_imaginary_time, local_imaginary_time.
Every matrix has axes (time, temperature), in the configured array order.
imaginary_times now stores the actual 2D tau grid with the same axes;
imaginary_times_input stores the original 1D TOML grid.
Both observables and both time domains share eigensystems and Gibbs weights.
Real and imaginary time reuse the same calculation functions, recomputing
operator matrix elements. Eigenvectors remain inside the worker and are not saved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
from time import perf_counter
import tomllib


def _range_grid(value, name):
    """Expand a range table without importing NumPy before thread limits apply."""
    required = {"start", "stop", "length"}
    if not required <= value.keys() or value.keys() - required - {"scale"}:
        raise ValueError(f"{name} range requires start, stop, length and optional scale")
    start, stop, length = value["start"], value["stop"], value["length"]
    scale = value.get("scale", "linear")
    if scale not in ("linear", "log10"):
        raise ValueError(f"{name} scale must be linear or log10")
    if type(length) is not int or length < 1:
        raise ValueError(f"{name} length must be a positive integer")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, stop)):
        raise ValueError(f"{name} start and stop must be finite numbers")
    fractions = (i / (length - 1) for i in range(length)) if length > 1 else (0.,)
    grid = ((1 - fraction) * start + fraction * stop for fraction in fractions)
    if scale == "log10":
        try:
            grid = [10.0 ** x for x in grid]
        except OverflowError as error:
            raise ValueError(f"{name} log10 range overflows floating-point values") from error
        if any(x == 0 for x in grid):
            raise ValueError(f"{name} log10 range underflows to zero")
    else:
        grid = list(grid)
    if any(not math.isfinite(x) for x in grid):
        raise ValueError(f"{name} range must produce finite values")
    return grid


def config_grid(value, name):
    """Normalize and validate one grid, retaining input order and numeric types."""
    if isinstance(value, dict) and name != "sizes":
        values = _range_grid(value, name)
    else:
        if (not isinstance(value, list) or not value
                or any(type(x) not in (int, float) for x in value)):
            raise ValueError(f"{name} must be a nonempty array of numbers")
        values = value
    if name == "sizes":
        if any(type(x) is not int or not 2 <= x <= 32 for x in values):
            raise ValueError("sizes must contain integers between 2 and 32")
    elif name == "temperatures":
        if any(math.isnan(x) or x < 0 for x in values):
            raise ValueError("temperatures must be >= 0 (positive inf is allowed)")
    elif any(not math.isfinite(x) for x in values):
        raise ValueError(f"{name} must contain only finite numbers")
    if name in ("sizes", "ms") and len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def load_config(path):
    """Validate grids and resource limits without importing numerical libraries."""
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8-sig")
    config = tomllib.loads(source)
    grids = ("sizes", "ms", "temperatures", "real_times", "imaginary_times")
    unknown = config.keys() - set(grids) - {"workers", "threads", "output_dir", "memory_gib"}
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    missing = set(grids) - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(sorted(missing))}")
    config.update({key: config_grid(config[key], key) for key in grids})
    for key in ("workers", "threads"):
        value = config.setdefault(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if "memory_gib" in config:
        value = config["memory_gib"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("memory_gib must be a finite positive number")
    output = config.setdefault("output_dir", "data/autocorrelation")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("output_dir must be a nonempty path string")
    config["output_dir"] = (path.parent / output).resolve()
    return config, source


def momentum_dimensions(L):
    """Count translation sectors from primitive blockade orbits, without ED."""
    fibonacci = [0, 1]
    for _ in range(L):
        fibonacci.append(sum(fibonacci[-2:]))
    primitive = {}
    for period in range(1, L + 1):
        if L % period == 0:
            # F[p-1]+F[p+1] includes only the empty state for period one.
            primitive[period] = (fibonacci[period - 1] + fibonacci[period + 1]
                                 - sum(n for d, n in primitive.items() if period % d == 0))
    return [sum(n // period for period, n in primitive.items() if k * period % L == 0)
            for k in range(L)]


def estimate_point_memory(L, config):
    """Heuristic peak bytes, not a guaranteed upper bound or measured RSS.

    Count stored real/complex eigenvectors exactly, then budget eight complex
    largest-block buffers, thermal arrays and returned matrices. Double these
    arrays for solver/allocator uncertainty and add 512 MiB for the runtime.
    """
    dimensions = momentum_dimensions(L)
    eigenvectors = sum((8 if (2 * k) % L == 0 else 16) * d**2
                       for k, d in enumerate(dimensions))
    buffers = 128 * max(dimensions)**2
    thermal = 64 * sum(dimensions) * len(config["temperatures"])
    outputs = (32 * len(config["real_times"]) + 24 * len(config["imaginary_times"])) * len(config["temperatures"])
    return 2 * (eigenvectors + buffers + thermal + outputs) + 512 * 2**20


def memory_budget(config):
    """Use the smaller of an explicit budget and a known Slurm allocation."""
    if int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")) != 1:
        raise ValueError("Memory preflight requires one Slurm task per node (--ntasks-per-node=1)")
    budgets = []
    if "memory_gib" in config:
        budgets.append(config["memory_gib"] * 2**30)
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        if name in os.environ:
            megabytes = int(os.environ[name])
            if megabytes < 0:
                raise ValueError(f"{name} must be nonnegative")
            if megabytes == 0:  # --mem=0 requests all memory, not a numeric budget.
                continue
            if name == "SLURM_MEM_PER_CPU":
                cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
                if cpus < 1:
                    raise ValueError("SLURM_MEM_PER_CPU requires SLURM_CPUS_PER_TASK")
                megabytes *= cpus
            budgets.append(megabytes * 2**20)
    if not budgets:
        raise ValueError("Memory budget is unknown: set memory_gib in TOML or request Slurm --mem/--mem-per-cpu")
    return min(budgets)


def check_memory(config, points, workers):
    """Reject the heaviest possible local combination before allocating matrices."""
    if not points:
        return
    budget = memory_budget(config)
    estimates = {L: estimate_point_memory(L, config) for L in {point[0] for point in points}}
    peak = 512 * 2**20 + sum(sorted((estimates[L] for L, _, _ in points), reverse=True)[:workers])
    print(f"Memory estimate: {peak / 2**30:.1f} GiB/node with {workers} workers; "
          f"budget={budget / 2**30:.1f} GiB/node (heuristic, not an RSS guarantee)", flush=True)
    if peak > budget:
        largest = max(estimates, key=estimates.get)
        raise ValueError(f"Estimated memory {peak / 2**30:.1f} GiB exceeds budget {budget / 2**30:.1f} GiB; "
                         f"L={largest} needs about {estimates[largest] / 2**30:.1f} GiB per worker. "
                         "Reduce sizes/workers or request sufficient memory; a single oversized point cannot be split across nodes.")


def slurm_tasks(sizes, ms):
    """Return rank, rank count and disjoint points, balanced per size."""
    ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ranks > 1 and "SLURM_PROCID" not in os.environ:
        raise ValueError("Multiple Slurm tasks require srun (SLURM_PROCID is missing)")
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if ranks < 1 or not 0 <= rank < ranks:
        raise ValueError("Require SLURM_NTASKS >= 1 and 0 <= SLURM_PROCID < SLURM_NTASKS")
    points = [(L, index, m) for L in sorted(sizes, reverse=True)
              for index, m in enumerate(ms)]
    return rank, ranks, points[rank::ranks]


def run_point(L, m_index, m, temperatures, real_times, imaginary_times, output_dir):
    """Compute and atomically save one parameter point; return only its path."""
    # Keep these imports inside the worker, after inherited thread limits apply.
    import numpy as np
    if __package__:
        from .pxp_autocorrelation import connected_autocorrelation_both_times, prepare_time_grid
    else:
        from pxp_autocorrelation import connected_autocorrelation_both_times, prepare_time_grid

    started = perf_counter()
    _, actual_imaginary_times = prepare_time_grid(temperatures, imaginary_times, imaginary=True)
    (average, local), (average_imaginary, local_imaginary) = (
        connected_autocorrelation_both_times(L, m, temperatures, real_times, imaginary_times)
    )
    arrays = dict(
        average_real_time_real=average.real,
        average_real_time_imag=average.imag,
        local_real_time_real=local.real,
        local_real_time_imag=local.imag,
        average_imaginary_time=average_imaginary,
        local_imaginary_time=local_imaginary,
    )
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise FloatingPointError(f"Nonfinite correlation at L={L}, m={m!r}")
    path = Path(output_dir) / f"autocorrelation_L={L}_m_index={m_index:04d}.npz"
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle, L=L, m=m, m_index=m_index, temperatures=temperatures,
                real_times=real_times, imaginary_times=actual_imaginary_times,
                imaginary_times_input=imaginary_times,
                elapsed_seconds=perf_counter() - started, **arrays,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_suffix(".toml"),
                        help="TOML parameter grids and process/thread limits")
    parser.add_argument("--run-id", help="shared run directory name; required for multiple Slurm ranks")
    args = parser.parse_args()
    try:
        config, source = load_config(args.config)
        rank, ranks, points = slurm_tasks(config["sizes"], config["ms"])
        if ranks > 1 and args.run_id is None:
            raise ValueError("Multiple Slurm ranks require the same --run-id on every rank")
        if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("--run-id must start with a letter/digit and contain only letters, digits, _, . or -")
        workers = min(config["workers"], len(points))
        threads = config["threads"]
        if "SLURM_CPUS_PER_TASK" in os.environ:
            if workers * threads > int(os.environ["SLURM_CPUS_PER_TASK"]):
                raise ValueError("workers * threads exceeds SLURM_CPUS_PER_TASK")
        check_memory(config, points, workers)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    total = len(config["sizes"]) * len(config["ms"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = config["output_dir"] / (args.run_id or stamp)
    if ranks > 1:
        output_dir = output_dir / f"rank_{rank:04d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.toml").write_text(source, encoding="utf-8")
    print(f"Rank={rank}/{ranks}, local tasks={len(points)}/{total}, "
          f"temperatures={len(config['temperatures'])}, "
          f"workers={workers}, threads/worker={threads}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    if not points:
        print("No assigned points; exiting without a worker pool.", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {
            pool.submit(run_point, L, index, m, config["temperatures"],
                        config["real_times"], config["imaginary_times"], output_dir): (L, m)
            for L, index, m in points
        }
        for completed, future in enumerate(as_completed(futures), 1):
            L, m = futures[future]
            try:
                path = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Autocorrelation task failed: L={L}, m={m!r}") from error
            print(f"[rank {rank}: {completed}/{len(points)}] L={L}, m={m!r}: {path.name}", flush=True)


if __name__ == "__main__":
    main()
