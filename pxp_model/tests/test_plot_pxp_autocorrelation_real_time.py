"""Check: python -m unittest pxp_model.tests.test_plot_pxp_autocorrelation_real_time"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..plot_pxp_autocorrelation_real_time import main, plot_autocorrelation_real_time


MODULE = "pxp_model.plot_pxp_autocorrelation_real_time"
FIELDS = ("local_real_time_real", "average_real_time_real",
          "local_real_time_imag", "average_real_time_imag")


class RealTimeAutocorrelationPlotTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        (self.directory / "L=12").mkdir()
        indices = patch(f"{MODULE}.M_INDICES", (0,))
        indices.start()
        self.addCleanup(indices.stop)
        self.addCleanup(plt.close, "all")
        times = np.array([-1., 0., .1, 1., 10.])
        self.data = dict(L=12, m=.655, temperatures=np.array([.5, 0.]), real_times=times)
        values = np.array([.7, .5, 0., -.2, .3])
        for index, key in enumerate(FIELDS):
            self.data[key] = np.column_stack((values * (index + 1), -values * (index + 2)))
        self.save()

    def save(self, index=0, **updates):
        np.savez(self.directory / "L=12" / f"autocorrelation_L=12_m_index={index:04d}.npz",
                 **(self.data | updates))

    def test_panel_order_keeps_signed_samples_and_axis_masks(self):
        for xscale in ("log10", "linear"):
            with self.subTest(xscale=xscale):
                fig, temperature = plot_autocorrelation_real_time(
                    self.directory, L=12, xscale=xscale)
                self.assertEqual(temperature, .5)
                self.assertEqual(len(fig.axes), 4)
                for ax, key in zip(fig.axes, FIELDS):
                    self.assertEqual(ax.get_xscale(), "log" if xscale == "log10" else "linear")
                    self.assertEqual(ax.get_yscale(), "linear")
                    self.assertEqual(len(ax.lines), 1)
                    line = ax.lines[0]
                    np.testing.assert_array_equal(line.get_xdata(), self.data["real_times"])
                    np.testing.assert_array_equal(np.ma.getdata(line.get_ydata()), self.data[key][:, 0])
                    np.testing.assert_array_equal(np.ma.getmaskarray(line.get_ydata()),
                                                  (self.data["real_times"] <= 0) & (xscale == "log10"))
                    np.testing.assert_allclose(ax.get_xlim(), [.1 if xscale == "log10" else -1., 10.])
                self.assertEqual(len(fig.legends), 1)
                fig.canvas.draw()
                plt.close(fig)

    def test_temperature_selection_uses_saved_value_and_index_precedence(self):
        for arguments, column in (({"T": .50000000001}, 0),
                                  ({"T": np.nan, "T_index": 1}, 1)):
            with self.subTest(arguments=arguments):
                fig, temperature = plot_autocorrelation_real_time(self.directory, L=12, **arguments)
                self.assertEqual(temperature, self.data["temperatures"][column])
                for ax, key in zip(fig.axes, FIELDS):
                    np.testing.assert_array_equal(np.ma.getdata(ax.lines[0].get_ydata()),
                                                  self.data[key][:, column])
                plt.close(fig)
        for arguments in ({"T": -.1}, {"T": .1}, {"T_index": -1}, {"T_index": 2}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                plot_autocorrelation_real_time(self.directory, L=12, **arguments)
        self.save(temperatures=np.array([.5, .5 + 1e-12]))
        with self.assertRaises(ValueError):
            plot_autocorrelation_real_time(self.directory, L=12, T=.5)

    def test_inconsistent_temperatures_across_files_are_rejected(self):
        self.save(index=1, temperatures=np.array([.1, 0.]))
        with patch(f"{MODULE}.M_INDICES", (0, 1)), self.assertRaises(ValueError):
            plot_autocorrelation_real_time(self.directory, L=12)

    def test_cli_saves_only_png_and_rejects_fitting(self):
        output_dir = self.directory / "figures"
        with patch("sys.argv", ["plot_pxp_autocorrelation_real_time.py",
                                "--data-dir", str(self.directory), "--L", "12",
                                "--T", ".5", "--T_index", "1", "--xscale", "linear",
                                "--output-dir", str(output_dir)]):
            main()
        self.assertEqual([path.name for path in output_dir.iterdir()],
                         ["pxp_autocorrelation_real_time_T=0_L=12.png"])
        with patch("sys.argv", ["plot_pxp_autocorrelation_real_time.py", "--fitting", "false"]), \
             patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)

    def test_malformed_or_nonfinite_data_are_rejected(self):
        invalid = ({"real_times": np.ones((5, 2))},
                   {"real_times": np.array([.1, 1., 2., 3., np.nan])},
                   {"local_real_time_real": np.ones((5, 1))},
                   {"average_real_time_imag": np.full((5, 2), np.inf)})
        for updates in invalid:
            with self.subTest(field=next(iter(updates))), self.assertRaises(ValueError):
                self.save(**updates)
                plot_autocorrelation_real_time(self.directory, L=12)


if __name__ == "__main__":
    unittest.main()
