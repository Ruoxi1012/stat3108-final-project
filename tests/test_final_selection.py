import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from final_selection import evaluate, valence_model


class FinalSelectionTests(unittest.TestCase):
    def test_valence_model_fits_three_classes(self):
        x = np.arange(36, dtype=float).reshape(12, 3)
        y = np.asarray([0, 1, 2] * 4)
        model = valence_model().fit(x, y)
        self.assertEqual(model.predict(x[:2]).shape, (2,))

    def test_evaluate_reports_accuracy_and_macro_f1(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        record = evaluate("valence", labels, labels)
        self.assertEqual(record["accuracy"], 1.0)
        self.assertEqual(record["macro_f1"], 1.0)
        # Tiny bootstrap samples can omit a class, so the lower endpoint may
        # drop even for perfect predictions; the observed perfect score remains
        # the upper endpoint.
        self.assertEqual(record["macro_f1_95_ci"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
