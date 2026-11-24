"""Check: python -m unittest pxp_model.tests.test_plot_pxp_autocorrelation"""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..plot_pxp_autocorrelation import M_INDICES, fit_decay_models, main, plot_autocorrelation


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

    def test_both_observables_use_selected_temperature_and_log_axes(self):
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
            fig, fits, _ = plot_autocorrelation(directory, T=0)
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
                self.assertEqual((ax.xaxis.get_transform().base, ax.yaxis.get_transform().base),
                                 (10, 10))
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

    def test_selects_size_and_temperature_by_value_or_defaults_to_first(self):
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory:
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            tau = np.array([0., .1, 1., 10., 100.])
            for temperatures in ([.5, 0., 1.], [.5, 1.]):
                times = np.column_stack([tau / (1 + T) for T in temperatures])
                local = np.column_stack([np.exp(-tau / (2 + T)) for T in temperatures])
                for index in M_INDICES:
                    np.savez(
                        size_dir / f"autocorrelation_L=12_m_index={index:04d}.npz",
                        L=12, m=index / 50, temperatures=temperatures,
                        imaginary_times=times, local_imaginary_time=local,
                        average_imaginary_time=local / 12,
                    )
                for requested in [None, *temperatures]:
                    T = temperatures[0] if requested is None else requested
                    T_index = temperatures.index(T)
                    with self.subTest(temperatures=temperatures, T=requested):
                        fig, fits, _ = plot_autocorrelation(directory, L=12, T=requested)
                        self.assertIn(f"L=12, T={T:g}", fig._suptitle.get_text())
                        self.assertEqual({(fit["L"], fit["T_index"], fit["T"]) for fit in fits},
                                         {(12, T_index, T)})
                        for ax, factor in zip(fig.axes, (1, 12)):
                            line = ax.lines[0]
                            np.testing.assert_allclose(line.get_xdata(), tau / (1 + T))
                            if T > 0:
                                np.testing.assert_allclose(ax.get_xlim(), tau[[1, -1]] / (1 + T))
                            np.testing.assert_allclose(line.get_ydata().compressed(),
                                                       np.exp(-tau[1:] / (2 + T)) / factor)
                        plt.close(fig)

            output_dir = Path(directory) / "figures"
            with patch("sys.argv", ["plot_pxp_autocorrelation.py", "--data-dir", directory,
                                    "--L", "12", "--T", "1.00000000001",
                                    "--xscale", "linear", "--yscale", "linear",
                                    "--output-dir", str(output_dir)]), \
                 patch("pxp_model.plot_pxp_autocorrelation.plot_autocorrelation",
                       wraps=plot_autocorrelation) as plot:
                main()
                self.assertEqual(plot.call_args.kwargs["xscale"], "linear")
                self.assertEqual(plot.call_args.kwargs["yscale"], "linear")
            self.assertTrue((output_dir / "pxp_autocorrelation_T=1_L=12.png").is_file())
            with (output_dir / "pxp_autocorrelation_fits_T=1_L=12.csv").open(
                    newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 40)
            self.assertEqual({(row["L"], row["T_index"], row["T"]) for row in records},
                             {("12", "1", "1.0")})

    def test_axis_scales_keep_saved_samples_and_fit_results(self):
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory, \
             patch("pxp_model.plot_pxp_autocorrelation.M_INDICES", (0,)):
            size_dir = Path(directory) / "L=26"
            size_dir.mkdir()
            tau = np.array([0., .1, 1., 2., 5., 8.])
            local = np.array([1., .8, .5, .2, 0., -.01])
            np.savez(size_dir / "autocorrelation_L=26_m_index=0000.npz",
                     L=26, m=0., temperatures=[.1], imaginary_times=tau[:, None],
                     local_imaginary_time=local[:, None],
                     average_imaginary_time=local[:, None] / 26)
            default_fig, expected_fits, _ = plot_autocorrelation(directory)
            plt.close(default_fig)
            for xscale in ("log10", "linear"):
                for yscale in ("log10", "linear"):
                    with self.subTest(xscale=xscale, yscale=yscale):
                        fig, fits, _ = plot_autocorrelation(directory, xscale=xscale, yscale=yscale)
                        self.assertEqual(fits, expected_fits)
                        for ax, factor in zip(fig.axes, (1, 26)):
                            self.assertEqual(ax.get_xscale(), "log" if xscale == "log10" else "linear")
                            self.assertEqual(ax.get_yscale(), "log" if yscale == "log10" else "linear")
                            line = ax.lines[0]
                            np.testing.assert_array_equal(line.get_xdata(), tau)
                            np.testing.assert_array_equal(line.get_ydata().data, local / factor)
                            mask = ((tau <= 0) & (xscale == "log10")) | ((local <= 0) & (yscale == "log10"))
                            np.testing.assert_array_equal(np.ma.getmaskarray(line.get_ydata()), mask)
                            np.testing.assert_allclose(ax.get_xlim(), [.1 if xscale == "log10" else 0., 8.])
                            if yscale == "linear":
                                self.assertLess(ax.get_ylim()[0], local.min() / factor)
                                self.assertLess(ax.get_ylim()[1], 1.1 / factor)
                        fig.canvas.draw()
                        plt.close(fig)
            for scales in ({"xscale": "invalid"}, {"yscale": "invalid"}):
                with self.assertRaisesRegex(ValueError, "xscale and yscale"):
                    plot_autocorrelation(directory, **scales)

    def test_matching_uses_saved_values_and_each_files_column(self):
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory, \
             patch("pxp_model.plot_pxp_autocorrelation.M_INDICES", (0, 1)):
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            saved_T = np.nextafter(.1, np.inf)
            tau = np.array([.1, 1., 10.])
            for index, temperatures in enumerate(([.5, saved_T], [saved_T, .5])):
                times = np.column_stack([tau / T for T in temperatures])
                local = np.exp(-times)
                np.savez(size_dir / f"autocorrelation_L=12_m_index={index:04d}.npz",
                         L=12, m=float(index), temperatures=temperatures,
                         imaginary_times=times, local_imaginary_time=local,
                         average_imaginary_time=local / 12)
            fig, fits, _ = plot_autocorrelation(directory, L=12, T=.1)
            self.assertEqual({(fit["m"], fit["T_index"], fit["T"]) for fit in fits},
                             {(0., 1, saved_T), (1., 0, saved_T)})
            for line in fig.axes[0].lines[:2]:
                np.testing.assert_array_equal(line.get_xdata(), tau / saved_T)
                np.testing.assert_allclose(line.get_ydata(), np.exp(-tau / saved_T))

    def test_index_takes_precedence_over_temperature_in_api_and_cli(self):
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory, \
             patch("pxp_model.plot_pxp_autocorrelation.M_INDICES", (0,)):
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            temperatures = [.5, 0., 1.]
            tau = np.array([.1, 1., 10.])
            times = np.column_stack([tau / (1 + T) for T in temperatures])
            local = np.exp(-times)
            np.savez(size_dir / "autocorrelation_L=12_m_index=0000.npz",
                     L=12, m=0., temperatures=temperatures, imaginary_times=times,
                     local_imaginary_time=local, average_imaginary_time=local / 12)
            for T, column in ((None, 0), (.1, 0), (-1., 1), (np.nan, 2), (0., 2)):
                with self.subTest(T=T, column=column):
                    fig, fits, _ = plot_autocorrelation(directory, L=12, T=T, T_index=column)
                    self.assertEqual({(fit["T_index"], fit["T"]) for fit in fits},
                                     {(column, temperatures[column])})
                    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), times[:, column])
                    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), local[:, column])
                    plt.close(fig)
            for column, message in ((-1, "must be >= 0"), (3, "out of range")):
                with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                    plot_autocorrelation(directory, L=12, T=.5, T_index=column)
            output_dir = Path(directory) / "figures"
            for selectors in (["--T", ".5", "--T_index", "1"],
                              ["--T_index", "1", "--T", ".5"]):
                with patch("sys.argv", ["plot_pxp_autocorrelation.py", "--data-dir", directory,
                                        "--L", "12", *selectors, "--output-dir", str(output_dir)]):
                    main()
                self.assertTrue((output_dir / "pxp_autocorrelation_T=0_L=12.png").is_file())
                with (output_dir / "pxp_autocorrelation_fits_T=0_L=12.csv").open(
                        newline="", encoding="utf-8") as stream:
                    self.assertEqual({(row["T_index"], row["T"]) for row in csv.DictReader(stream)},
                                     {("1", "0.0")})

    def test_disabled_fitting_keeps_data_without_calling_models(self):
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory, \
             patch("pxp_model.plot_pxp_autocorrelation.M_INDICES", (0,)), \
             patch("pxp_model.plot_pxp_autocorrelation.fit_decay_models") as fit:
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            tau = np.array([.1, 1.])
            local = np.exp(-tau)
            np.savez(size_dir / "autocorrelation_L=12_m_index=0000.npz",
                     L=12, m=0., temperatures=[.5], imaginary_times=tau[:, None],
                     local_imaginary_time=local[:, None],
                     average_imaginary_time=local[:, None] / 12)
            fig, fits, temperature = plot_autocorrelation(directory, L=12, fitting=False)
            fit.assert_not_called()
            self.assertEqual(fits, [])
            self.assertEqual(temperature, .5)
            for ax, factor in zip(fig.axes, (1, 12)):
                self.assertEqual(len(ax.lines), 1)
                self.assertEqual(len(ax.texts), 0)
                self.assertEqual(ax.lines[0].get_linestyle(), "-")
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), tau)
                np.testing.assert_allclose(ax.lines[0].get_ydata(), local / factor)

    def test_cli_fitting_switch_controls_csv_output(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("pxp_model.plot_pxp_autocorrelation.M_INDICES", (0,)):
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            times = np.array([[.1, .2], [1., 2.], [10., 20.]])
            local = np.exp(-times)
            np.savez(size_dir / "autocorrelation_L=12_m_index=0000.npz",
                     L=12, m=0., temperatures=[.5, 0.], imaginary_times=times,
                     local_imaginary_time=local, average_imaginary_time=local / 12)
            cases = (("true", ["--T", ".50000000001"], .5),
                     ("false", ["--T", ".50000000001"], .5),
                     ("false", ["--T", ".5", "--T_index", "1"], 0.),
                     ("false", [], .5))
            for index, (fitting, selectors, temperature) in enumerate(cases):
                output_dir = Path(directory) / f"figures_{index}"
                with self.subTest(fitting=fitting, selectors=selectors), \
                     patch("sys.argv", ["plot_pxp_autocorrelation.py", "--data-dir", directory,
                                        "--L", "12", *selectors, "--fitting", fitting,
                                        "--output-dir", str(output_dir)]):
                    main()
                    output = output_dir / f"pxp_autocorrelation_T={temperature:g}_L=12.png"
                    self.assertTrue(output.is_file())
                    fit_output = output_dir / f"pxp_autocorrelation_fits_T={temperature:g}_L=12.csv"
                    self.assertEqual(fit_output.is_file(), fitting == "true")
                    if fitting == "true":
                        with fit_output.open(newline="", encoding="utf-8") as stream:
                            records = list(csv.DictReader(stream))
                        self.assertEqual({row["model"] for row in records},
                                         {"exponential", "power_law"})
                        self.assertEqual(len(records), 4)
            with patch("sys.argv", ["plot_pxp_autocorrelation.py", "--fitting", "invalid"]), \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_invalid_missing_and_ambiguous_temperatures_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            size_dir = Path(directory) / "L=12"
            size_dir.mkdir()
            for T, temperatures, message in (
                (-1., [.1], "T must be >= 0"),
                (np.nan, [.1], "T must be >= 0"),
                (.2, [.1, .5], "No saved temperature matches"),
                (.1, [.1, .1 + 5e-12], "matches multiple"),
                (0., [0., 0.], "matches multiple"),
                (0., [1e-14], "No saved temperature matches"),
                (None, [], "nonempty"),
                (.1, [], "nonempty"),
            ):
                np.savez(size_dir / "autocorrelation_L=12_m_index=0000.npz",
                         L=12, temperatures=temperatures)
                with self.subTest(T=T, temperatures=temperatures), \
                     self.assertRaisesRegex(ValueError, message):
                    plot_autocorrelation(directory, L=12, T=T)


if __name__ == "__main__":
    unittest.main()
