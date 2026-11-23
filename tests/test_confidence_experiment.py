import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from confidence_experiment import model_identity, split_validation


class ConfidenceExperimentTests(unittest.TestCase):
    def test_policy_split_is_deterministic_disjoint_and_stratified(self):
        rows = []
        for breed in ("husky", "shiba"):
            for arousal in ("Low", "Medium", "High"):
                for valence in ("Negative", "Neutral", "Positive"):
                    for copy in range(4):
                        rows.append({
                            "audio_id": f"{breed}_{arousal}_{valence}_{copy}",
                            "breed": breed,
                            "arousal": arousal,
                            "valence": valence,
                        })
        calibration, selection = split_validation(rows)
        repeated = split_validation(rows)
        self.assertEqual((calibration, selection), repeated)
        self.assertFalse(set(calibration) & set(selection))
        self.assertEqual(set(calibration) | set(selection), {row["audio_id"] for row in rows})
        self.assertEqual(len(calibration), len(selection))

    def test_model_identity_changes_with_target_or_training_ids(self):
        first = model_identity("arousal", ["a", "b"], "manifest")
        self.assertEqual(first, model_identity("arousal", ["b", "a"], "manifest"))
        self.assertNotEqual(first, model_identity("valence", ["a", "b"], "manifest"))
        self.assertNotEqual(first, model_identity("arousal", ["a", "c"], "manifest"))


if __name__ == "__main__":
    unittest.main()
