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
Memory-limited scheduling uses the smaller of Slurm's allocation and
memory_gib (GiB per node), requiring at least one known budget and one Slurm
task per node. Unknown budgets or individually oversized points fail before
creating output or workers. Estimates are heuristic, not an RSS guarantee.

For Slurm, use submit_autocorrelation.sh: srun launches one copy per node.
The supplied Slurm script uses 8 nodes, 1 task/node and 64 CPUs/task.
The TOML controls worker processes/task and threads/process.
SLURM_PROCID and SLURM_NTASKS partition (L,m) points without MPI dependencies.
Larger sizes are queued first, with each size spread across ranks. Each rank
has its own local pool; there is no work stealing between nodes.
Each rank reserves 512 MiB for the parent and admits tasks while their summed
memory estimates fit the remaining budget, up to workers concurrent points.
Smaller sizes can fill slots when a larger queued point does not fit.

output_dir is relative to the TOML file, unless absolute. Each run uses a new
YYYYMMDD_HHMMSS_microseconds directory (UTC+08:00) or the supplied --run-id.
Rank zero creates this directory and atomically publishes its only config.toml.
Results go under <run>/L=<L>/, with one NPZ per (L, m). Completed files survive
a later task failure; failed tasks cause a nonzero exit. Existing runs and
results are never overwritten.
Multi-rank runs require the SAME unique --run-id and config on every rank,
and shared storage for code, config and output. Other ranks wait up to 60 s
for config.toml, then verify the config, rank count and Slurm job/step IDs.
Hidden .rank_<rank> files reject duplicate rank launches; do not remove them.

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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
from time import perf_counter, sleep
import tomllib


_OUTPUT_WAIT_SECONDS = 60


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

    Count independent real/complex eigenvectors once, then budget eight complex
    largest-block buffers for operator products and workspace, thermal arrays
    and returned matrices. Add 25% for solver/allocator uncertainty and
    512 MiB for the runtime.
    """
    dimensions = momentum_dimensions(L)
    eigenvectors = sum((8 if (2 * k) % L == 0 else 16) * d**2
                       for k, d in enumerate(dimensions[:L // 2 + 1]))
    buffers = 128 * max(dimensions)**2
    thermal = 64 * sum(dimensions) * len(config["temperatures"])
    outputs = ((32 * len(config["real_times"]) + 24 * len(config["imaginary_times"]))
               * len(config["temperatures"]))
    return 1.25 * (eigenvectors + buffers + thermal + outputs) + 512 * 2**20


def memory_budget(config):
    """Use the smaller of an explicit budget and a known Slurm allocation."""
    if int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")) != 1:
        raise ValueError("Memory scheduling requires one Slurm task per node (--ntasks-per-node=1)")
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
    """Reject oversized points; return the worker budget and per-size estimates."""
    if not points:
        return
    budget = memory_budget(config)
    estimates = {L: estimate_point_memory(L, config) for L in {point[0] for point in points}}
    largest = max(estimates, key=estimates.get)
    peak = 512 * 2**20 + estimates[largest]
    if peak > budget:
        raise ValueError(f"Estimated memory {peak / 2**30:.1f} GiB exceeds budget {budget / 2**30:.1f} GiB; "
                         f"L={largest} needs about {estimates[largest] / 2**30:.1f} GiB per worker. "
                         "Reduce sizes or request sufficient memory; a single oversized point cannot be split across nodes.")
    print(f"Memory budget={budget / 2**30:.1f} GiB/node, reserving 0.5 GiB for the parent; "
          f"largest point L={largest}: {estimates[largest] / 2**30:.1f} GiB; "
          f"up to {workers} workers, limited by memory (heuristic, not an RSS guarantee)", flush=True)
    return budget - 512 * 2**20, estimates


def slurm_tasks(sizes, ms):
    """Return rank, rank count and disjoint points, balanced per size."""
    ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ranks > 1 and "SLURM_PROCID" not in os.environ:
        raise ValueError("Multiple Slurm tasks require srun (SLURM_PROCID is missing)")
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if ranks < 1 or not 0 <= rank < ranks:
        raise ValueError("Require SLURM_NTASKS >= 1 and 0 <= SLURM_PROCID < SLURM_NTASKS")
    points = [(L, m_index, m) for L in sorted(sizes, reverse=True)
              for m_index, m in enumerate(ms)]
    return rank, ranks, points[rank::ranks]


def prepare_output(output_dir, source, rank, ranks):
    """Publish one config, then admit each matching rank exactly once."""
    config_path = output_dir / "config.toml"
    owner_path = output_dir / ".rank_0000"
    identity = {"ranks": ranks, "job_id": os.environ.get("SLURM_JOB_ID"),
                "step_id": os.environ.get("SLURM_STEP_ID")}
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        with owner_path.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle)
        temporary = config_path.with_suffix(".toml.tmp")
        try:
            temporary.write_text(source, encoding="utf-8")
            temporary.replace(config_path)
        finally:
            temporary.unlink(missing_ok=True)
        return
    deadline = perf_counter() + _OUTPUT_WAIT_SECONDS
    while True:
        try:
            saved_source = config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if perf_counter() >= deadline:
                raise TimeoutError(f"Timed out waiting for rank 0 to initialize {output_dir}")
            sleep(.1)
            continue
        if json.loads(owner_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run directory belongs to a different Slurm job/step or rank count")
        if saved_source != source:
            raise ValueError("Run directory config.toml differs from this rank's config")
        break
    (output_dir / f".rank_{rank:04d}").touch(exist_ok=False)


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
    (average_real, local_real), (average_imaginary, local_imaginary) = (
        connected_autocorrelation_both_times(L, m, temperatures, real_times, imaginary_times)
    )
    arrays = dict(
        average_real_time_real=average_real.real,
        average_real_time_imag=average_real.imag,
        local_real_time_real=local_real.real,
        local_real_time_imag=local_real.imag,
        average_imaginary_time=average_imaginary,
        local_imaginary_time=local_imaginary,
    )
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise FloatingPointError(f"Nonfinite correlation at L={L}, m={m!r}")
    size_dir = Path(output_dir) / f"L={L}"
    size_dir.mkdir(parents=True, exist_ok=True)
    path = size_dir / f"autocorrelation_L={L}_m_index={m_index:04d}.npz"
    temporary = path.with_suffix(".npz.tmp")
    # Acquire outside try so a competing writer's temporary file is never removed.
    handle = temporary.open("xb")
    try:
        with handle:
            if path.exists():
                raise FileExistsError(f"Result already exists: {path}")
            np.savez_compressed(
                handle, L=L, m=m, m_index=m_index, temperatures=temperatures,
                real_times=real_times, imaginary_times=actual_imaginary_times,
                imaginary_times_input=imaginary_times,
                elapsed_seconds=perf_counter() - started, **arrays,
            )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _run_points(config, points, output_dir, workers, rank, budget, estimates):
    """Run the first queued point that fits the remaining worker memory budget."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        queued = list(points)
        futures = {}
        completed = 0
        while queued or futures:
            while queued and len(futures) < workers:
                used = sum(estimates[L] for L, _ in futures.values())
                eligible = next((i for i, (L, _, _) in enumerate(queued)
                                 if used + estimates[L] <= budget), None)
                if eligible is None:
                    break
                L, m_index, m = queued.pop(eligible)
                future = pool.submit(run_point, L, m_index, m, config["temperatures"],
                                     config["real_times"], config["imaginary_times"], output_dir)
                futures[future] = (L, m)
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                L, m = futures.pop(future)
                try:
                    path = future.result()
                except Exception as error:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"Autocorrelation task failed: L={L}, m={m!r}") from error
                completed += 1
                print(f"[rank {rank}: {completed}/{len(points)}] L={L}, m={m!r}: {path.name}", flush=True)


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
        if ("SLURM_CPUS_PER_TASK" in os.environ
                and workers * threads > int(os.environ["SLURM_CPUS_PER_TASK"])):
            raise ValueError("workers * threads exceeds SLURM_CPUS_PER_TASK")
        if points:
            budget, estimates = check_memory(config, points, workers)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    total = len(config["sizes"]) * len(config["ms"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = config["output_dir"] / (args.run_id or stamp)
    prepare_output(output_dir, source, rank, ranks)
    print(f"Rank={rank}/{ranks}, local tasks={len(points)}/{total}, "
          f"temperatures={len(config['temperatures'])}, "
          f"workers={workers}, threads/worker={threads}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    if not points:
        print("No assigned points; exiting without a worker pool.", flush=True)
        return
    _run_points(config, points, output_dir, workers, rank, budget, estimates)


if __name__ == "__main__":
    main()
