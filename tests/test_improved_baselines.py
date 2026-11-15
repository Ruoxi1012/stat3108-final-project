import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from improved_baselines import bootstrap_macro_f1, richer_features


class ImprovedBaselineTests(unittest.TestCase):
    def test_richer_features_are_finite_and_include_breed(self):
        mfcc = np.ones((2, 82), dtype=np.float32)
        logmel = np.arange(2 * 48 * 400, dtype=np.float32).reshape(2, 48, 400)
        rows = [{"breed": "husky"}, {"breed": "shiba"}]
        features = richer_features(mfcc, logmel, rows)
        self.assertEqual(features.shape, (2, 804))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:, -2:], [[1, 0], [0, 1]])

    def test_bootstrap_interval_contains_perfect_score(self):
        labels = np.asarray([0, 1, 2] * 20)
        interval = bootstrap_macro_f1(labels, labels, samples=100)
        np.testing.assert_allclose(interval, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
