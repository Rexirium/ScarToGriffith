"""Check: python -m unittest pxp_model.tests.test_plot_pxp_autocorrelation"""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..plot_pxp_autocorrelation import M_INDICES, fit_decay_models, plot_autocorrelation


class AutocorrelationPlotTest(unittest.TestCase):
    def test_selects_known_exponential_and_power_law(self):
        tau = np.geomspace(.1, 100., 80)
        for model, values, amplitude, parameter in (
                ("exponential", .7 * np.exp(-.06 * tau), .7, .06),
                ("power_law", .2 * tau**(-.6), .2, .6)):
            with self.subTest(model=model):
                grid, predicted, fits = fit_decay_models(tau, values)
                best, = [fit for fit in fits if fit["selected"]]
                self.assertEqual(best["model"], model)
                self.assertAlmostEqual(best["amplitude"], amplitude)
                key = "gamma" if model == "exponential" else "alpha"
                self.assertAlmostEqual(best[key], parameter)
                expected = (amplitude * np.exp(-parameter * grid) if model == "exponential"
                            else amplitude * grid**(-parameter))
                np.testing.assert_allclose(predicted, expected, rtol=1e-12)
                self.assertLess(best["rmse_log"], 1e-12)
                self.assertEqual([fit["n_points"] for fit in fits], [80, 80])

    def test_uses_valid_samples_beyond_display_limits(self):
        tau = np.array([0., .01, .1, 1., 10., 1e3, 1e5, 2e5, 3e5])
        values = np.ma.array([0., 2e5, 200., .2, 2e-4, 2e-10, 2e-16, -1., .9],
                             mask=[False]*8 + [True])
        grid, _, fits = fit_decay_models(tau, values)
        best, = [fit for fit in fits if fit["selected"]]
        self.assertEqual(best["model"], "power_law")
        self.assertAlmostEqual(best["amplitude"], .2)
        self.assertAlmostEqual(best["alpha"], 3.)
        self.assertEqual([fit["n_points"] for fit in fits], [6, 6])
        np.testing.assert_allclose(grid[[0, -1]], [.01, 1e5])
        with self.assertRaisesRegex(ValueError, "three distinct"):
            fit_decay_models([1., 2., 3., 4.], [0., -1., .2, np.nan])

    def test_both_observables_use_zero_temperature_and_log_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            size_dir = Path(directory) / "L=26"
            size_dir.mkdir()
            tau = np.array([0., .1, 1., 10., 100.])
            for index in M_INDICES:
                local = np.array([1., .8, .4, .01, 0.]) / (1 + index)
                np.savez(
                    size_dir / f"autocorrelation_L=26_m_index={index:04d}.npz",
                    L=26, m=index / 50, temperatures=[.5, 0.],
                    imaginary_times=np.column_stack((tau / 50, tau)),
                    local_imaginary_time=np.column_stack((np.ones(5), local)),
                    average_imaginary_time=np.column_stack((np.ones(5), local / 26)),
                )
            fig, fits = plot_autocorrelation(directory)
            self.addCleanup(plt.close, "all")
            self.assertEqual(len(fits), 40)
            self.assertEqual(sum(fit["selected"] for fit in fits), 20)
            self.assertEqual(len(fig.axes), 2)
            np.testing.assert_allclose(fig.axes[0].get_xlim(), [.1, 100.])
            np.testing.assert_allclose(fig.axes[1].get_xlim(), [.1, 1e3])
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                             [rf"$m={index / 50:g}$" for index in M_INDICES])
            for observable, ax in zip(("local", "average"), fig.axes):
                self.assertIsNone(ax.get_legend())
                self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
                self.assertEqual(len(ax.lines), 20)
                self.assertEqual(len({line.get_color() for line in ax.lines}), 10)
                for index, line in zip(M_INDICES, ax.lines):
                    np.testing.assert_array_equal(line.get_xdata(), tau)
                    values = line.get_ydata()
                    np.testing.assert_array_equal(values.mask, [True, False, False, False, True])
                    expected = np.array([.8, .4, .01]) / (1 + index)
                    if observable == "average":
                        expected /= 26
                    np.testing.assert_allclose(values.compressed(), expected)
                    self.assertEqual(line.get_label(), rf"$m={index / 50:g}$")
                for solid, dashed in zip(ax.lines[:10], ax.lines[10:]):
                    self.assertEqual(dashed.get_color(), solid.get_color())
                    self.assertEqual(dashed.get_linestyle(), "--")
                    np.testing.assert_allclose(dashed.get_xdata()[[0, -1]], [.1, 10.])
            fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()
