import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from pretrained_baselines import feature_matrix, fixed_audio, pool_layers


class PretrainedBaselineTests(unittest.TestCase):
    def test_fixed_audio_center_crops_only_long_clips(self):
        self.assertEqual(len(fixed_audio(np.arange(5), 10)), 5)
        np.testing.assert_array_equal(fixed_audio(np.arange(10), 4), [3, 4, 5, 6])

    def test_pooling_excludes_padding(self):
        layer = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ])
        pooled = pool_layers([layer], torch.tensor([2, 3]))
        np.testing.assert_allclose(pooled[0, 0], [2, 3, 1, 1])
        np.testing.assert_allclose(pooled[1, 0, :2], [6, 8])

    def test_feature_matrix_layer_pooling_and_breed(self):
        embeddings = np.zeros((2, 12, 1536), dtype=np.float32)
        embeddings[:, 5, :768] = 2
        rows = [{"breed": "husky"}, {"breed": "shiba"}]
        features = feature_matrix(embeddings, rows, 6, "mean", True)
        self.assertEqual(features.shape, (2, 770))
        np.testing.assert_array_equal(features[:, -2:], [[1, 0], [0, 1]])
        self.assertTrue((features[:, :768] == 2).all())


if __name__ == "__main__":
    unittest.main()
