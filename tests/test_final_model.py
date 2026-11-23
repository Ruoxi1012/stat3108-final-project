import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from final_model import acoustic_feature, checked_breed, score_distribution


class FinalModelTests(unittest.TestCase):
    def test_acoustic_feature_matches_final_dimension(self):
        time = np.arange(16000, dtype=np.float32) / 16000
        audio = 0.1 * np.sin(2 * np.pi * 440 * time)
        values = acoustic_feature(audio, "husky")
        self.assertEqual(values.shape, (1, 804))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[0, -2:], [1, 0])

    def test_unseen_breed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the model scope"):
            checked_breed("poodle")

    def test_tree_score_distribution_has_three_ordered_classes(self):
        x = np.arange(18, dtype=float).reshape(6, 3)
        y = np.asarray([0, 1, 2, 0, 1, 2])
        model = ExtraTreesClassifier(n_estimators=5, random_state=3108).fit(x, y)
        scores = score_distribution(model, x[:2])
        self.assertEqual(scores.shape, (2, 3))
        np.testing.assert_allclose(scores.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
