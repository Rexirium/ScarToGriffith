import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress


SOURCE_PATH = Path(__file__).with_name("pxp_correlation.py")
tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
function_names = {
    "exponential_decay",
    "power_law_decay",
    "upper_envelope_indices",
    "fit_power_law_decay",
}
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in function_names
]
namespace = {
    "np": np,
    "find_peaks": find_peaks,
    "linregress": linregress,
}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)


class FitPowerLawDecayTest(unittest.TestCase):
    def test_fits_every_point_up_to_t_fit_and_ignores_later_data(self):
        fit_power_law_decay = namespace["fit_power_law_decay"]
        ts = np.geomspace(0.1, 30.0, 50)
        expected = np.array([2.0, 0.7])
        corrs = expected[0] * ts ** (-expected[1])
        corrs[ts > 20.0] = 100.0

        parameters, _, fit_indices, r_squared = fit_power_law_decay(
            ts, corrs, t_fit=20.0
        )

        np.testing.assert_allclose(parameters, expected, rtol=1e-5)
        np.testing.assert_array_equal(fit_indices, np.flatnonzero(ts <= 20.0))
        self.assertAlmostEqual(r_squared, 1.0)

    def test_rejects_nonpositive_fit_data(self):
        ts = np.geomspace(0.1, 20.0, 20)
        corrs = 2.0 * ts ** (-0.7)
        corrs[5] = 0.0

        with self.assertRaisesRegex(ValueError, "positive"):
            namespace["fit_power_law_decay"](ts, corrs, t_fit=20.0)


if __name__ == "__main__":
    unittest.main()
