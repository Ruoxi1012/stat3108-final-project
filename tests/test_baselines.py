import importlib.util
import unittest
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "baselines", Path(__file__).resolve().parents[1] / "code/baselines.py"
)
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


class BaselineTests(unittest.TestCase):
    def test_mel_and_mfcc_features_are_finite(self):
        time = np.arange(16000) / 16000
        audio = 0.2 * np.sin(2 * np.pi * 440 * time)
        filters = baselines.mel_filterbank(40)
        self.assertEqual(filters.shape, (40, 257))
        self.assertTrue(np.isfinite(filters).all())
        mfcc = baselines.mfcc_summary(audio)
        self.assertEqual(mfcc.shape, (82,))
        self.assertTrue(np.isfinite(mfcc).all())

    def test_fixed_logmel_shape_for_short_and_long_audio(self):
        short = np.sin(np.arange(800) * 0.1)
        long = np.sin(np.arange(160000) * 0.1)
        for audio in (short, long):
            feature = baselines.fixed_log_mel(audio)
            self.assertEqual(feature.shape, (baselines.CNN_MELS, baselines.CNN_FRAMES))
            self.assertTrue(np.isfinite(feature).all())

    def test_fixed_split_is_deterministic_and_disjoint(self):
        rows = []
        for breed in ("husky", "shiba"):
            for arousal in baselines.AROUSAL_LABELS:
                for valence in baselines.VALENCE_LABELS:
                    for number in range(6):
                        rows.append({
                            "audio_id": f"{breed}-{arousal}-{valence}-{number}",
                            "breed": breed,
                            "arousal": arousal,
                            "valence": valence,
                        })
        first = baselines.fixed_split(rows)
        second = baselines.fixed_split(rows)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(len(first[0]) + len(first[1]), len(rows))

    def test_metrics_include_ordinal_errors(self):
        metrics = baselines.metric_values(np.array([0, 1, 2]), np.array([2, 1, 1]))
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["severe_error_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
