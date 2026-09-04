import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import linregress


SOURCE_PATH = Path(__file__).with_name("pxp_capacity_fit.py")
tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "fit_power_law_capacity"
]
namespace = {"np": np, "linregress": linregress}
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"),
    namespace,
)


class FitPowerLawCapacityTest(unittest.TestCase):
    def test_fits_points_through_half_the_peak_temperature(self):
        Ts = np.geomspace(0.01, 1.0, 81)
        capacities = 3.0 * Ts**0.65
        peak_index = np.searchsorted(Ts, 0.2, side="right") - 1
        capacities[peak_index + 1:] = capacities[peak_index] * np.linspace(
            0.9, 0.1, Ts.size - peak_index - 1
        )

        parameters, errors, fit_indices, r_squared = namespace[
            "fit_power_law_capacity"
        ](Ts, capacities)

        np.testing.assert_allclose(parameters, [3.0, 0.65], rtol=1e-12)
        np.testing.assert_array_equal(
            fit_indices, np.flatnonzero(Ts <= Ts[peak_index] / 2.0)
        )
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_rejects_nonpositive_fit_data(self):
        Ts = np.geomspace(0.01, 1.0, 20)
        capacities = Ts**0.5
        capacities[3] = 0.0

        with self.assertRaisesRegex(ValueError, "positive"):
            namespace["fit_power_law_capacity"](Ts, capacities)

    def test_rejects_fit_ranges_with_fewer_than_two_points(self):
        Ts = np.array([0.01, 0.1, 1.0])
        capacities = np.array([1.0, 0.5, 0.25])

        with self.assertRaisesRegex(ValueError, "at least two"):
            namespace["fit_power_law_capacity"](Ts, capacities)


if __name__ == "__main__":
    unittest.main()
