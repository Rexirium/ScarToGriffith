"""State matching checks: python -m unittest pxp_model.test_utils."""

import unittest

import numpy as np

from .utils import max_overlap_indices


class MaxOverlapIndicesTest(unittest.TestCase):
    def test_complex_columns_repeated_matches_and_ties(self):
        candidates = np.column_stack([
            np.array([1, 1j]) / np.sqrt(2),
            np.array([1, -1j]) / np.sqrt(2),
            [1, 0],
        ])
        states = np.column_stack([
            candidates[:, 1], 1j * candidates[:, 0],
            -candidates[:, 1], [0, 1],
        ])
        np.testing.assert_array_equal(
            max_overlap_indices(states, candidates), [1, 0, 1, 0],
        )

    def test_empty_input_states(self):
        matches = max_overlap_indices(np.empty((2, 0)), np.eye(2))
        self.assertEqual(matches.shape, (0,))
        self.assertTrue(np.issubdtype(matches.dtype, np.integer))

    def test_invalid_shapes(self):
        for states, candidates in (
            (np.ones(2), np.eye(2)),
            (np.eye(2), np.ones(2)),
            (np.eye(2), np.eye(3)),
            (np.eye(2), np.empty((2, 0))),
        ):
            with self.subTest(shapes=(states.shape, candidates.shape)):
                with self.assertRaises(ValueError):
                    max_overlap_indices(states, candidates)


if __name__ == "__main__":
    unittest.main()
