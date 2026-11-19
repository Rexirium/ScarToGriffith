"""Plot local and average-magnetization correlations side by side.

Run from the repository root:
    python pxp_model/plot_pxp_autocorrelation.py --L 26 --T_index 0

Requires only NumPy and Matplotlib. Default paths are relative to this script.
Defaults: L=26 and T_index=0 (zero temperature, which must exist in the data).
T_index=1, 2, ... select positive temperatures in their saved order.
Read ten m points from the saved scan, with spacing 0.02 near m=0.655.
Both axes are logarithmic; correlation axes show 1e-12 <= C <= 1.
The local panel spans all positive times; the average panel shows 0.1 <= tau <= 1e3.
Nonpositive times/correlations are masked, including underflowed zeros.
For each trace, compare A*exp(-gamma*tau) and A*tau**(-alpha) on all saved
finite, positive time/correlation pairs, independently of the display limits,
minimizing equally weighted squared residuals in ln(C).
Dashed curves show the model with lower log-space RMSE, only over its fit range.
Save both models' parameters, scores and selection in a CSV beside the figure.
Default output: figures/autocorrelation/<data-timestamp>/ beside this script.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .plot_pxp_common import HAMILTONIAN, ROOT, STYLE
else:
    from plot_pxp_common import HAMILTONIAN, ROOT, STYLE

DEFAULT_DATA_DIR = ROOT / "data" / "autocorrelation" / "20261118_201651_17391427_Lle26"
M_INDICES = (0, 20, 30, 31, 32, 33, 34, 35, 40, 50)
Y_LIMITS = (1e-12, 1)
OBSERVABLES = {
    "local": r"$C_{\mathrm{loc}}(\tau)$",
    "average": r"$C_M(\tau)$",
}


def fit_decay_models(tau, trace):
    """Fit all unmasked finite positive samples, independent of display limits."""
    tau = np.asarray(tau, dtype=float)
    trace = np.ma.asarray(trace, dtype=float)
    if tau.ndim != 1 or trace.shape != tau.shape:
        raise ValueError("Expected matching one-dimensional times and correlations")
    values = trace.data
    valid = (~np.ma.getmaskarray(trace) & np.isfinite(tau) & np.isfinite(values)
             & (tau > 0) & (values > 0))
    times, logs = tau[valid], np.log(values[valid])
    if np.unique(times).size < 3:
        raise ValueError("Need at least three distinct valid times for model comparison")
    fits = []
    for model, x in (("exponential", times), ("power_law", np.log(times))):
        slope, intercept = np.polyfit(x, logs, 1)
        rmse = float(np.sqrt(np.mean((logs - (intercept + slope * x))**2)))
        fits.append(dict(model=model, amplitude=float(np.exp(intercept)),
                         gamma=float(-slope) if model == "exponential" else None,
                         alpha=float(-slope) if model == "power_law" else None,
                         rmse_log=rmse, n_points=int(times.size),
                         tau_min=float(times.min()), tau_max=float(times.max())))
    best = min(fits, key=lambda fit: fit["rmse_log"])
    for fit in fits:
        fit["selected"] = fit is best
    grid = np.geomspace(times.min(), times.max(), 400)
    if best["model"] == "exponential":
        predicted = best["amplitude"] * np.exp(-best["gamma"] * grid)
    else:
        predicted = best["amplitude"] * grid**(-best["alpha"])
    return grid, predicted, fits


def plot_autocorrelation(data_dir=DEFAULT_DATA_DIR, L=26, T_index=0):
    """Return the two-panel figure and fits; index 0 is T=0, 1+ are saved T>0."""
    if T_index < 0:
        raise ValueError("T_index must be >= 0 (0 selects T=0)")
    curves = []
    temperature = None
    for index in M_INDICES:
        path = Path(data_dir) / f"L={L}" / f"autocorrelation_L={L}_m_index={index:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if data["L"].item() != L:
                raise ValueError(f"Expected L={L}: {path}")
            temperatures = data["temperatures"]
            if (temperatures.ndim != 1 or np.any(np.isnan(temperatures))
                    or np.any(temperatures < 0)):
                raise ValueError(f"Expected one-dimensional nonnegative temperatures: {path}")
            if T_index == 0:
                columns = np.flatnonzero(temperatures == 0)
                if columns.size != 1:
                    raise ValueError(f"T_index=0 requires exactly one T=0 column: {path}")
                column = columns.item()
            else:
                columns = np.flatnonzero(temperatures > 0)
                if T_index > columns.size:
                    raise ValueError(f"T_index={T_index} out of range; "
                                     f"found {columns.size} positive temperatures: {path}")
                column = columns[T_index - 1]
            selected_temperature = temperatures[column].item()
            if temperature is not None and selected_temperature != temperature:
                raise ValueError(f"Inconsistent temperature for T_index={T_index}: {path}")
            temperature = selected_temperature
            times = data["imaginary_times"]
            if times.ndim != 2 or times.shape[1] != temperatures.size:
                raise ValueError(f"Expected imaginary_times axes (time, temperature): {path}")
            tau = times[:, column]
            m = data["m"].item()
            if not np.isfinite(m) or not np.all(np.isfinite(tau)):
                raise ValueError(f"Nonfinite m or imaginary times: {path}")
            traces = {}
            for observable in OBSERVABLES:
                values = data[f"{observable}_imaginary_time"]
                if values.shape != times.shape:
                    raise ValueError(f"{observable} shape differs from imaginary_times: {path}")
                trace = values[:, column]
                if not np.all(np.isfinite(trace)):
                    raise ValueError(f"Nonfinite {observable} correlation at T={temperature:g}: {path}")
                valid = (tau > 0) & (trace > 0)
                if np.count_nonzero(valid) < 2:
                    raise ValueError(f"Need two positive {observable} samples: {path}")
                traces[observable] = np.ma.masked_where(~valid, trace)
            curves.append((m, tau, traces))

    positive_times = np.concatenate([tau[tau > 0] for _, tau, _ in curves])
    fit_records = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
        for ax, (observable, label) in zip(axes, OBSERVABLES.items()):
            xlim = ((0.1, 1e3) if observable == "average"
                    else (positive_times.min(), positive_times.max()))
            for color, (m, tau, traces) in enumerate(curves):
                ax.loglog(tau, traces[observable], color=f"C{color}", lw=1.8,
                          label=rf"$m={m:g}$")
            for color, (m, tau, traces) in enumerate(curves):
                grid, predicted, fits = fit_decay_models(tau, traces[observable])
                fit_records.extend(dict(L=L, T_index=T_index, T=temperature,
                                        observable=observable, m=m, **fit) for fit in fits)
                # White edges keep coincident dashed fits visible over the data.
                ax.plot(grid, predicted, color=f"C{color}", ls="--", lw=2.2,
                        path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
            ax.set(xlabel=r"$\tau$", ylabel=label, ylim=Y_LIMITS, xlim=xlim)
            ax.grid(which="major", color="0.9", linewidth=0.6)
            ax.text(0.04, 0.06, "Dashed: best log-space fit", transform=ax.transAxes, fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, fontsize=11, loc="outside lower center")
        fig.suptitle(f"{HAMILTONIAN}\nPBC, L={L}, T={temperature:g}")
    return fig, fit_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Autocorrelation run directory containing L=<L>/")
    parser.add_argument("--L", type=int, default=26, help="Chain length (default: 26)")
    parser.add_argument("--T_index", type=int, default=0,
                        help="0: T=0 (default); 1, 2, ...: positive temperatures in saved order")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "figures" / "autocorrelation" / args.data_dir.resolve().name
    try:
        fig, fit_records = plot_autocorrelation(args.data_dir, L=args.L, T_index=args.T_index)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    temperature = fit_records[0]["T"]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"pxp_autocorrelation_T={temperature:g}_L={args.L}.png"
        fig.savefig(output, dpi=180)
        print(f"Saved {output}")
        fit_output = output_dir / f"pxp_autocorrelation_fits_T={temperature:g}_L={args.L}.csv"
        with fit_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fit_records[0].keys())
            writer.writeheader()
            writer.writerows(fit_records)
        print(f"Saved {fit_output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
