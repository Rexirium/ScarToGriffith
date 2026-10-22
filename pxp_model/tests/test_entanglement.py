"""Entropy checks against independent tensor-product constructions."""

import unittest

import numpy as np

from ..pxp_basis import pxp_basis_1d


def entropy(probabilities):
    probabilities = probabilities[probabilities > 1e-14]
    return -np.sum(probabilities * np.log(probabilities))


class EntanglementTest(unittest.TestCase):
    def test_pure_batches_against_full_tensor_product(self):
        basis = pxp_basis_1d(6)
        rng = np.random.default_rng(43)
        for complex_state in (False, True):
            states = rng.normal(size=(basis.Ns, basis.Ns))
            if complex_state:
                states = states + 1j * rng.normal(size=states.shape)
            states /= np.linalg.norm(states, axis=0)
            for cut in (1, 3, 5):
                expected = []
                for state in states.T:
                    embedded = np.zeros(2**basis.N, dtype=state.dtype)
                    embedded[basis.states] = state
                    s = np.linalg.svd(embedded.reshape(2**cut, -1), compute_uv=False)
                    expected.append(entropy(s**2))
                actual = basis.my_ent_entropy(states, cut=cut, enforce_pure=True)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                self.assertAlmostEqual(basis.my_ent_entropy(states[:, :1], cut=cut), expected[0])

if __name__ == "__main__":
    unittest.main()
