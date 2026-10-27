"""Plot <sigma_1^z> versus energy for L=28, m=0, 0.2, ..., 1.

    python pxp_model/plot_pxp_z1.py

Requires only NumPy and Matplotlib. Expectations come from z1_expectations
in the saved scan. Circles and triangles mark matched and tracked ED states;
FSA eigenstate expectations are not plotted. Default paths are script-relative.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

if __package__:
    from .plot_pxp_overlaps import ROOT, MS
else:
    from plot_pxp_overlaps import ROOT, MS


def plot_z1(data_path=ROOT / "data" / "pxp_scan_L=28.npz"):
    """Return six panels with common data-based y limits and row colorbars."""
    keys = ("ms", "energies", "z1_expectations", "entropies",
            "matched_scar_indices", "tracked_scar_indices")
    with np.load(data_path, allow_pickle=False) as archive:
        if (archive["L"].item() != 28 or archive["kblock"].item() != 0
                or archive["parity"].item() != 1):
            raise ValueError("Expected L=28, k=0, p=+ scan data")
        data = {key: archive[key] for key in keys}
    rows = []
    for m in MS:
        matches = np.flatnonzero(np.isclose(data["ms"], m, rtol=0, atol=1e-10))
        if matches.size != 1:
            raise ValueError(f"Expected exactly one saved parameter point for m={m:g}")
        rows.append(matches.item())

    # Use one scale for comparison, with 5% padding around all plotted states.
    plotted_values = data["z1_expectations"][rows]
    ymin, ymax = plotted_values.min(), plotted_values.max()
    padding = max(0.05 * (ymax - ymin), 0.01)
    ylim = (ymin - padding, ymax + padding)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.edgecolor": "none",
    }):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
        # All ED markers and the colorbar share the same entropy scale per row.
        row_norms = [Normalize(vmin=0, vmax=data["entropies"][rows[start:start + 3]].max())
                     for start in (0, 3)]
        for panel, (ax, m, row) in enumerate(zip(axes.flat, MS, rows)):
            energies = data["energies"][row]
            values = data["z1_expectations"][row]
            entropies = data["entropies"][row]
            matched = np.unique(data["matched_scar_indices"][row])
            tracked = np.unique(data["tracked_scar_indices"][row])
            background = np.ones(energies.size, dtype=bool)
            background[np.union1d(matched, tracked)] = False
            colors = dict(cmap="plasma", norm=row_norms[panel // 3])
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
            ax.axvline(0, color="0.5", linestyle="--", lw=1,
                       label=r"$E_{\mathbb{Z}_2^{+}}=0$")
            ax.set(
                title=rf"$m={m:.1f}$", xlabel=r"$E_n$",
                ylabel=r"$\langle E_n|Z_1|E_n\rangle$",
                yscale="linear", ylim=ylim,
            )
            if panel % 3 == 2:
                fig.colorbar(points, ax=axes[panel // 3, :], pad=0.02,
                             label=r"$S_{L/2}$ (nats)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
        fig.suptitle(
            r"$H=\sum_i P_{i-1}X_iP_{i+1}-m\sum_i Z_i$"
            "\nPBC, L=28, a=1; "
            r"$k=0,\ p=+;\quad \sigma_1^z=Z_1$"
        )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pxp_scan_L=28.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "pxp_z1_L=28.png")
    args = parser.parse_args()
    fig = plot_z1(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
