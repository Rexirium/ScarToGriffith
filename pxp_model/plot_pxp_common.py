"""Shared data loading, CLI and rendering for the PXP scan plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "20261027_222310"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "legend.edgecolor": "none",
}
HAMILTONIAN = r"$H=\sum_i P_{i-1}X_iP_{i+1}-m\sum_i Z_i$"


def resolve_scan(data_dir, L=None):
    """Resolve an explicit size or the numerically largest scan in a directory."""
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    if L is not None:
        path = data_dir / f"pxp_scan_L={L}.npz"
        if not path.is_file():
            raise ValueError(f"Scan file does not exist: {path}")
        return path, L
    scans = []
    for path in data_dir.glob("pxp_scan_L=*.npz"):
        size = path.stem.removeprefix("pxp_scan_L=")
        if path.is_file() and size.isdecimal():
            scans.append((int(size), path))
    if not scans:
        raise ValueError(f"No pxp_scan_L=<L>.npz files found in {data_dir}")
    L, path = max(scans, key=lambda scan: scan[0])
    return path, L


def load_scan(data_path, keys, L=None):
    """Load requested observables and find the eight rows in ms_plot order."""
    if data_path is None:
        data_path, L = resolve_scan(DEFAULT_DATA_DIR, L)
    with np.load(data_path, allow_pickle=False) as archive:
        saved_L = archive["L"].item()
        if L is not None and saved_L != L:
            raise ValueError(f"Expected L={L} scan data, found L={saved_L}")
        if archive["kblock"].item() != 0 or archive["parity"].item() != 1:
            raise ValueError("Expected k=0, p=+ scan data")
        data = {key: archive[key] for key in ("ms", "ms_plot", *keys)}
    data["L"] = saved_L
    if data["ms_plot"].shape != (8,):
        raise ValueError("Expected ms_plot with shape (8,) for eight panels")
    rows = []
    for m in data["ms_plot"]:
        matches = np.flatnonzero(np.isclose(data["ms"], m, rtol=0, atol=1e-10))
        if matches.size != 1:
            raise ValueError(f"Expected exactly one saved parameter point for m={m:g}")
        rows.append(matches.item())
    return data, rows


def run_plot(plot, name, description):
    """Run a plot with the shared directory, size and output arguments."""
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing pxp_scan_L=<L>.npz")
    parser.add_argument("--L", type=int, help="Chain length (default: largest in --data-dir)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        data_path, L = resolve_scan(args.data_dir, args.L)
    except ValueError as error:
        parser.error(str(error))
    output = args.output
    if output is None:
        output = ROOT / "figures" / args.data_dir.resolve().name / f"pxp_{name}_L={L}.png"
    fig = plot(data_path, L=L)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)
    print(f"Saved {output}")


def plot_spectrum(data_path, L, *, value_key, fsa_key, ylabel, title_suffix,
                  logarithmic=False):
    """Draw the shared ED/FSA energy panels with entropy colors and scar markers."""
    data, rows = load_scan(data_path, (
        "energies", "entropies", "matched_scar_indices", "tracked_scar_indices",
        "fsa_energies", value_key, fsa_key,
    ), L)
    if logarithmic:
        ylim = (1e-15, 1)
    else:
        plotted_values = np.concatenate((data[value_key][rows], data[fsa_key][rows]), axis=1)
        ymin, ymax = plotted_values.min(), plotted_values.max()
        padding = max(0.05 * (ymax - ymin), 0.01)
        ylim = (ymin - padding, ymax + padding)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(19.2, 10), sharey="row")
        fig.subplots_adjust(left=0.055, right=0.95, bottom=0.12, top=0.85,
                            wspace=0, hspace=0.28)
        # All ED markers and the colorbar share the same entropy scale per row.
        row_norms = [Normalize(vmin=0, vmax=data["entropies"][rows[start:start + 4]].max())
                     for start in (0, 4)]
        for panel, (ax, m, row) in enumerate(zip(axes.flat, data["ms_plot"], rows)):
            energies = data["energies"][row]
            values = data[value_key][row]
            fsa_values = data[fsa_key][row]
            entropies = data["entropies"][row]
            matched = np.unique(data["matched_scar_indices"][row])
            tracked = np.unique(data["tracked_scar_indices"][row])
            background = values > 1e-15 if logarithmic else np.ones(energies.size, dtype=bool)
            if logarithmic:
                values = np.maximum(values, 1e-15)
                fsa_values = np.maximum(fsa_values, 1e-15)
            background[np.union1d(matched, tracked)] = False
            colors = dict(cmap="plasma", norm=row_norms[panel // 4])
            points = ax.scatter(
                energies[background], values[background],
                c=entropies[background], **colors,
                marker="o", s=9, label=r"$k=0,\ p=+$",
            )
            ax.scatter(
                energies[matched], values[matched], c=entropies[matched], **colors,
                marker="o", s=90, edgecolors="black", linewidths=0.4,
                label="FSA-matched scars", zorder=3,
            )
            ax.scatter(
                energies[tracked], values[tracked], c=entropies[tracked], **colors,
                marker="^", s=75, edgecolors="black", linewidths=0.4,
                label="Tracked scars", zorder=3.5,
            )
            ax.scatter(
                data["fsa_energies"][row], fsa_values,
                color="red", marker="x", s=40, linewidths=1.2,
                label="FSA eigenstates", zorder=4,
            )
            ax.axvline(0, color="0.5", linestyle="--", lw=1,
                       label=r"$E_{\mathbb{Z}_2^{+}}=0$")
            ax.set(
                title=rf"$m={str(m)}$", xlabel=r"$E_n$",
                ylabel=ylabel,
                yscale="log" if logarithmic else "linear", ylim=ylim,
            )
            # Keep tick labels clear of the seams between adjoining panels.
            xmin, xmax = ax.get_xlim()
            ticks = MaxNLocator(nbins=6).tick_values(xmin, xmax)
            margin = 0.04 * (xmax - xmin)
            ax.set_xticks(ticks[(ticks > xmin + margin) & (ticks < xmax - margin)])
            if panel % 4:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if panel % 4 == 3:
                fig.colorbar(points, ax=axes[panel // 4, :], pad=0.02, fraction=0.012,
                             label=r"$S_{L/2}$ (nats)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
        fig.suptitle(f"{HAMILTONIAN}\nPBC, L={data['L']}, a=1; " + title_suffix)
        # Make the plotting areas square without opening gaps between columns.
        panel_box = axes[0, 0].get_position()
        fig.set_figheight(fig.get_figwidth() * panel_box.width / panel_box.height)
    return fig
