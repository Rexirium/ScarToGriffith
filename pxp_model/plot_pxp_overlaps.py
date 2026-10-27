"""Plot six L=28 Z2-overlap panels from the saved scan (no ED required).

Run from the repository root:
    python pxp_model/plot_pxp_overlaps.py

Requires only NumPy and Matplotlib. Default paths are relative to this script.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parent
MS = np.linspace(0.0, 1.0, 6)


def plot_overlaps(data_path=ROOT / "data" / "pxp_scan_L=28.npz"):
    """Return the figure; indices and FSA overlaps use the saved scan conventions."""
    keys = ("ms", "energies", "overlaps", "entropies", "matched_scar_indices",
            "tracked_scar_indices", "fsa_energies", "fsa_overlaps")
    with np.load(data_path, allow_pickle=False) as archive:
        if archive["L"].item() != 28:
            raise ValueError("Expected L=28 scan data")
        data = {key: archive[key] for key in keys}
    rows = []
    for m in MS:
        matches = np.flatnonzero(np.isclose(data["ms"], m, rtol=0, atol=1e-10))
        if matches.size != 1:
            raise ValueError(f"Expected exactly one saved parameter point for m={m:g}")
        rows.append(matches.item())

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
            weights = data["overlaps"][row]
            entropies = data["entropies"][row]
            matched = np.unique(data["matched_scar_indices"][row])
            tracked = np.unique(data["tracked_scar_indices"][row])
            background = weights > 1e-15
            background[np.union1d(matched, tracked)] = False
            colors = dict(cmap="plasma", norm=row_norms[panel // 3])
            points = ax.scatter(
                energies[background], weights[background],
                c=entropies[background], **colors,
                marker="o", s=9, label=r"$k=0,\ p=+$",
            )
            # The larger circle remains visible around a coincident triangle.
            ax.scatter(
                energies[matched], np.maximum(weights[matched], 1e-15),
                c=entropies[matched], **colors,
                marker="o", s=90, edgecolors="black", linewidths=0.4,
                label="FSA-matched scars", zorder=3,
            )
            ax.scatter(
                energies[tracked], np.maximum(weights[tracked], 1e-15),
                c=entropies[tracked], **colors,
                marker="^", s=75, edgecolors="black", linewidths=0.4,
                label="Tracked scars", zorder=3.5,
            )
            ax.scatter(
                data["fsa_energies"][row],
                np.maximum(data["fsa_overlaps"][row], 1e-15),
                color="red", marker="x", s=40, linewidths=1.2,
                label="FSA eigenstates", zorder=4,
            )
            ax.axvline(0, color="0.5", linestyle="--", lw=1,
                       label=r"$E_{\mathbb{Z}_2^{+}}=0$")
            ax.set(
                title=rf"$m={m:.1f}$", xlabel=r"$E_n$",
                ylabel=r"$|\langle \mathbb{Z}_2^{+}|E_n\rangle|^2$",
                yscale="log", ylim=(1e-15, 1),
            )
            if panel % 3 == 2:
                fig.colorbar(points, ax=axes[panel // 3, :], pad=0.02,
                             label=r"$S_{L/2}$ (nats)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
        fig.suptitle(
            r"$H=\sum_i P_{i-1}X_iP_{i+1}-m\sum_i Z_i$"
            "\nPBC, L=28, a=1; "
            r"$k=0,\ p=+;\quad |\mathbb{Z}_2^{+}\rangle="
            r"(|1010\cdots\rangle+|0101\cdots\rangle)/\sqrt{2}$"
        )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pxp_scan_L=28.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "pxp_overlaps_L=28.png")
    args = parser.parse_args()
    fig = plot_overlaps(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
