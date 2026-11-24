"""Plot real-time local and average-magnetization connected correlations.

Run from the repository root:
    python pxp_model/plot_pxp_autocorrelation_real_time.py --L 26 --T 0.1

Requires only NumPy and Matplotlib. Defaults match plot_pxp_autocorrelation.py:
the same data directory, ten m points, L=26 and the first saved temperature.
--T_index is a zero-based saved column index and takes precedence over --T.
--T matches exactly one saved temperature with rtol=1e-10, atol=0.
Columns show local and average-magnetization correlations, respectively;
the first row shows real parts and the second row shows imaginary parts.
Use saved real_times without rescaling. --xscale accepts log10 (default) or
linear; only nonpositive times on a log axis are masked. All vertical axes
are linear to retain signed correlations. Lines join the saved samples;
a sparse time grid need not resolve the intervening oscillations.
No fitting is performed. Save one PNG under
figures/autocorrelation/<data-timestamp>/ beside this script by default.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .plot_pxp_autocorrelation import DEFAULT_DATA_DIR, M_INDICES
    from .plot_pxp_common import HAMILTONIAN, ROOT, STYLE
else:
    from plot_pxp_autocorrelation import DEFAULT_DATA_DIR, M_INDICES
    from plot_pxp_common import HAMILTONIAN, ROOT, STYLE

OBSERVABLES = {"local": r"C_{\mathrm{loc}}(t)", "average": r"C_M(t)"}
COMPONENTS = {"real": r"\mathrm{Re}", "imag": r"\mathrm{Im}"}


def plot_autocorrelation_real_time(data_dir=DEFAULT_DATA_DIR, L=26, T=None,
                                  T_index=None, xscale="log10"):
    """Return the four-panel figure and the selected saved temperature."""
    if xscale not in ("log10", "linear"):
        raise ValueError("xscale must be 'log10' or 'linear'")
    if T_index is None:
        T_index = 0
    else:
        T = None
    if T_index < 0:
        raise ValueError("T_index must be >= 0")
    if T is not None and (np.isnan(T) or T < 0):
        raise ValueError("T must be >= 0 (positive inf is allowed)")
    curves = []
    temperature = None
    for index in M_INDICES:
        path = Path(data_dir) / f"L={L}" / f"autocorrelation_L={L}_m_index={index:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if data["L"].item() != L:
                raise ValueError(f"Expected L={L}: {path}")
            temperatures = data["temperatures"]
            if (temperatures.ndim != 1 or not temperatures.size or np.any(np.isnan(temperatures))
                    or np.any(temperatures < 0)):
                raise ValueError(f"Expected nonempty one-dimensional nonnegative temperatures: {path}")
            column = T_index
            if T is not None:
                matches = np.flatnonzero(np.isclose(temperatures, T, rtol=1e-10, atol=0))
                if not matches.size:
                    raise ValueError(f"No saved temperature matches T={T:g}: {path}")
                if matches.size != 1:
                    raise ValueError(f"T={T:g} matches multiple temperature columns: {path}")
                column = matches.item()
            if column >= temperatures.size:
                raise ValueError(f"T_index={column} out of range; "
                                 f"found {temperatures.size} temperatures: {path}")
            selected_temperature = temperatures[column].item()
            if temperature is not None and selected_temperature != temperature:
                raise ValueError(f"Inconsistent selected temperature: {path}")
            temperature = selected_temperature
            times = data["real_times"]
            m = data["m"].item()
            if times.ndim != 1:
                raise ValueError(f"Expected one-dimensional real_times: {path}")
            if not np.isfinite(m) or not np.all(np.isfinite(times)):
                raise ValueError(f"Nonfinite m or real times: {path}")
            valid = (times > 0) | (xscale == "linear")
            if np.unique(times[valid]).size < 2:
                raise ValueError(f"Need two distinct displayed real times: {path}")
            traces = {}
            for observable in OBSERVABLES:
                for component in COMPONENTS:
                    key = f"{observable}_real_time_{component}"
                    values = data[key]
                    if values.shape != (times.size, temperatures.size):
                        raise ValueError(f"Expected {key} axes (time, temperature): {path}")
                    trace = values[:, column]
                    if not np.all(np.isfinite(trace)):
                        raise ValueError(f"Nonfinite {key} at T={temperature:g}: {path}")
                    traces[observable, component] = np.ma.masked_where(~valid, trace)
            curves.append((m, times, traces))

    displayed_times = np.concatenate([times[(times > 0) | (xscale == "linear")]
                                      for _, times, _ in curves])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, layout="constrained")
        for row, (component, part_label) in enumerate(COMPONENTS.items()):
            for column, (observable, label) in enumerate(OBSERVABLES.items()):
                ax = axes[row, column]
                ax.set_xscale("log" if xscale == "log10" else "linear")
                for color, (m, times, traces) in enumerate(curves):
                    ax.plot(times, traces[observable, component], color=f"C{color}", lw=1.8,
                            label=rf"$m={m:g}$")
                ax.set(ylabel=rf"${part_label}\,{label}$",
                       xlim=(displayed_times.min(), displayed_times.max()))
                if row == 1:
                    ax.set_xlabel(r"$t$")
                ax.grid(which="major", color="0.9", linewidth=0.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, fontsize=11, loc="outside lower center")
        fig.suptitle(f"{HAMILTONIAN}\nPBC, L={L}, T={temperature:g}")
    return fig, temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Autocorrelation run directory containing L=<L>/")
    parser.add_argument("--L", type=int, default=26, help="Chain length (default: 26)")
    parser.add_argument("--T", type=float,
                        help="Match a saved temperature value; ignored when --T_index is given")
    parser.add_argument("--T_index", type=int,
                        help="Zero-based saved column index; takes precedence over --T. "
                             "If neither is given, use the first saved temperature")
    parser.add_argument("--xscale", choices=("log10", "linear"), default="log10",
                        help="Horizontal axis scale (default: log10); vertical axes stay linear")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "figures" / "autocorrelation" / args.data_dir.resolve().name
    try:
        fig, temperature = plot_autocorrelation_real_time(
            args.data_dir, L=args.L, T=args.T, T_index=args.T_index, xscale=args.xscale)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"pxp_autocorrelation_real_time_T={temperature:g}_L={args.L}.png"
        fig.savefig(output, dpi=180)
        print(f"Saved {output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
