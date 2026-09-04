from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from pxp_capacity import thermodynamics_vs_temperature


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})


def fit_power_law_capacity(Ts, capacities):
    """Fit C_V(T) = A T^lambda up to half the peak temperature."""
    peak_temperature = Ts[np.argmax(capacities)]
    fit_indices = np.flatnonzero(Ts <= peak_temperature / 2.0)
    if fit_indices.size < 2:
        raise ValueError("the fit range must contain at least two points")

    fit_Ts = Ts[fit_indices]
    fit_capacities = capacities[fit_indices]
    if np.any(fit_Ts <= 0.0) or np.any(fit_capacities <= 0.0):
        raise ValueError("fit temperatures and capacities must be positive")

    regression = linregress(np.log(fit_Ts), np.log(fit_capacities))
    amplitude = np.exp(regression.intercept)
    parameters = np.array([amplitude, regression.slope])
    errors = np.array([
        amplitude * regression.intercept_stderr,
        regression.stderr,
    ])
    return parameters, errors, fit_indices, regression.rvalue**2


def main():
    L, r = 20, 0.2
    g_values = np.array([0.0, -0.1, -0.2, -0.3, -0.4, -0.5])
    Ts = np.geomspace(0.02, 2.0, 201)

    _, capacities = thermodynamics_vs_temperature(L, g_values, r, Ts)
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")

    for g, capacity in zip(g_values, capacities):
        parameters, errors, fit_indices, r_squared = fit_power_law_capacity(
            Ts, capacity
        )
        amplitude, exponent = parameters
        amplitude_error, exponent_error = errors
        line, = ax.plot(Ts, capacity)
        ax.plot(
            Ts[fit_indices],
            amplitude * Ts[fit_indices] ** exponent,
            "--",
            color=line.get_color(),
            label=rf"$g={g:g},\ \lambda={exponent:.3f}$",
        )
        print(
            f"g={g:g}, T <= T_fit_max={Ts[fit_indices[-1]]:.6g}: "
            f"A={amplitude:.6g} +/- {amplitude_error:.2g}, "
            f"lambda={exponent:.6g} +/- {exponent_error:.2g}, "
            f"R^2={r_squared:.6g}, points={fit_indices.size}"
        )

    ax.set(
        xlabel=r"$T$",
        ylabel=r"$C_V$",
        xscale="log",
        yscale="log",
        title=rf"$L={L},\ r={r:g}$",
    )
    ax.legend(ncols=2, fontsize=9)

    output_path = (
        Path(__file__).resolve().parent
        / "figures"
        / f"pxp_capacity_fit_L={L}_r={r:.1f}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
