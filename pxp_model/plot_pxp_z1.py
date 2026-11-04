"""Plot eight <sigma_1^z> versus energy panels.

    python pxp_model/plot_pxp_z1.py --data-dir pxp_model/data/20261027_222310 --L 28

Requires only NumPy and Matplotlib. Expectations come from z1_expectations
and fsa_z1_expectations in the saved scan. Circles and triangles mark matched
and tracked ED states; red crosses mark FSA eigenstates. Paths are script-relative.
The eight plotted m values and their order come from the saved ms_plot field.
Omit --L to use the largest saved chain length in --data-dir.
"""

if __package__:
    from .plot_pxp_common import plot_spectrum, run_plot
else:
    from plot_pxp_common import plot_spectrum, run_plot


def plot_z1(data_path=None, L=None):
    """Return eight Z1 panels with common y limits and row colorbars."""
    return plot_spectrum(
        data_path, L, value_key="z1_expectations", fsa_key="fsa_z1_expectations",
        ylabel=r"$\langle E_n|Z_1|E_n\rangle$",
        title_suffix=r"$k=0,\ p=+;\quad \sigma_1^z=Z_1$",
    )


def main():
    run_plot(plot_z1, "z1", __doc__)


if __name__ == "__main__":
    main()
