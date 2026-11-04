"""Plot eight Z1 time traces and their saved thermal expectations.

    python pxp_model/plot_pxp_z1_time.py --data-dir pxp_model/data/20261027_222310 --L 28

Requires only NumPy and Matplotlib. The t=0 sample is omitted on the log axis.
Default output is figures/<data-directory-name>/pxp_z1_time_L=<L>.png.
The eight plotted m values and their order come from the saved ms_plot field.
Omit --L to use the largest saved chain length in --data-dir.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __package__:
    from .plot_pxp_common import HAMILTONIAN, STYLE, load_scan, run_plot
else:
    from plot_pxp_common import HAMILTONIAN, STYLE, load_scan, run_plot


def plot_z1_time(data_path=None, L=None):
    """Return four rows of two panels, sharing logarithmic time axes by column."""
    data, rows = load_scan(data_path, ("times", "z1_t", "z1_th"), L)

    # A logarithmic axis cannot display the initial t=0 sample.
    positive = data["times"] > 0
    times = data["times"][positive]
    if times.size < 2:
        raise ValueError("Expected at least two positive saved times")
    traces = data["z1_t"][rows][:, positive]
    thermal = data["z1_th"][rows]
    ymin, ymax = min(traces.min(), thermal.min()), max(traces.max(), thermal.max())
    padding = max(0.05 * (ymax - ymin), 0.01)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex="col")
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.89,
                            wspace=0.28, hspace=0)
        for panel, (ax, m, trace, z1_th) in enumerate(zip(axes.flat, data["ms_plot"], traces, thermal)):
            ax.plot(times, trace, color="tab:blue", lw=1,
                    label=r"$\langle Z_1(t)\rangle$")
            ax.axhline(z1_th, color="red", linestyle="--", lw=1.4,
                       label=r"$\langle Z_1\rangle_{\mathrm{th}}$", zorder=3)
            ax.set(
                ylabel=r"$\langle Z_1(t)\rangle$",
                xscale="log", xlim=(times[0], times[-1]),
                ylim=(ymin - padding, ymax + padding),
            )
            ax.text(0.96, 0.92, rf"$m={str(m)}$", transform=ax.transAxes,
                    ha="right", va="top")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10],
                                                  prune="both"))
            if panel // 2 == 3:
                ax.set_xlabel(r"$t$")
            else:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        fig.suptitle(
            f"{HAMILTONIAN}\nPBC, L={data['L']}, a=1; "
            r"$k=0,\ p=+;\quad |\psi(0)\rangle=|\mathbb{Z}_2^{+}\rangle$"
        )
        # Keep a 2:1 plotting area while adjoining panels touch vertically.
        panel_box = axes[0, 0].get_position()
        fig.set_figheight(fig.get_figwidth() * panel_box.width / (2 * panel_box.height))
    return fig


def main():
    run_plot(plot_z1_time, "z1_time", __doc__)


if __name__ == "__main__":
    main()
