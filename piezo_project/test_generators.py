"""Автотесты генераторов спринта 2. Те же проверки, что в ноутбуке, отдельным файлом.

    python -m unittest test_generators -v
"""
import unittest

import numpy as np
from scipy import stats

import generators as G

ALPHA, LAM = 1.568, 0.4734          # Ломакс
C, MU, SIGMA = -0.2807, 1.8716, 0.5126   # GEV в параметризации SciPy
S, THETA, SCALE = 0.5351, 3.9019, 12.8137  # сдвинутое логнормальное
N = 50_000


class GeneratorChecks(unittest.TestCase):
    def test_contract(self):
        for fn, args, bad in [
            (G.lomax_inverse, (ALPHA, LAM), (-2, 1, 1)),
            (G.lomax_gamma_mixture, (ALPHA, LAM), (5, 0, 1)),
            (G.gev_inverse, (C, MU, SIGMA), None),
            (G.shifted_lognormal_box_muller, (S, THETA, SCALE), None),
        ]:
            with self.subTest(generator=fn.__name__):
                rows = G.contract_tests(fn, args, bad_args=bad)
                self.assertGreaterEqual(len(rows), 3)

    def test_support(self):
        rng = np.random.default_rng(G.SEED)
        self.assertTrue(np.all(G.lomax_inverse(N, ALPHA, LAM, rng) >= 0))
        self.assertTrue(np.all(G.lomax_gamma_mixture(N, ALPHA, LAM, rng) >= 0))
        self.assertTrue(np.all(G.gev_inverse(N, C, MU, SIGMA, rng) >= MU - SIGMA / (-C) - 1e-10))
        self.assertTrue(np.all(G.shifted_lognormal_box_muller(N, S, THETA, SCALE, rng) > THETA))

    def test_box_muller_moments(self):
        z = G.standard_normal_box_muller(200_000, np.random.default_rng(1234))
        G.normal_moments_test(z)

    def test_odd_sample_size(self):
        self.assertEqual(len(G.standard_normal_box_muller(7, np.random.default_rng(1))), 7)

    def test_median_matches_theory(self):
        # медиана устойчива к тяжёлому хвосту, поэтому сравниваем её, а не среднее
        rng = np.random.default_rng(G.SEED)
        cases = [
            (G.lomax_inverse(N, ALPHA, LAM, rng), stats.lomax.median(ALPHA, scale=LAM)),
            (G.lomax_gamma_mixture(N, ALPHA, LAM, rng), stats.lomax.median(ALPHA, scale=LAM)),
            (G.gev_inverse(N, C, MU, SIGMA, rng), stats.genextreme.median(C, loc=MU, scale=SIGMA)),
            (G.shifted_lognormal_box_muller(N, S, THETA, SCALE, rng), stats.lognorm.median(S, loc=THETA, scale=SCALE)),
        ]
        for sample, median in cases:
            self.assertAlmostEqual(np.median(sample) / median, 1, delta=0.03)


if __name__ == '__main__':
    unittest.main(verbosity=2)
