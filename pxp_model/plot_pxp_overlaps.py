"""Plot eight Z2-overlap panels from the saved scan (no ED required).

Run from the repository root:
    python pxp_model/plot_pxp_overlaps.py --data-dir pxp_model/data/20261027_222310 --L 28

Requires only NumPy and Matplotlib. Default paths are relative to this script.
The eight plotted m values and their order come from the saved ms_plot field.
Omit --L to use the largest saved chain length in --data-dir.
"""

if __package__:
    from .plot_pxp_common import plot_spectrum, run_plot
else:
    from plot_pxp_common import plot_spectrum, run_plot


def plot_overlaps(data_path=None, L=None):
    """Return eight overlap panels using the saved scan conventions."""
    return plot_spectrum(
        data_path, L, value_key="overlaps", fsa_key="fsa_overlaps",
        ylabel=r"$|\langle \mathbb{Z}_2^{+}|E_n\rangle|^2$",
        title_suffix=(r"$k=0,\ p=+;\quad |\mathbb{Z}_2^{+}\rangle="
                      r"(|1010\cdots\rangle+|0101\cdots\rangle)/\sqrt{2}$"),
        logarithmic=True,
    )


def main():
    run_plot(plot_overlaps, "overlaps", __doc__)


if __name__ == "__main__":
    main()
