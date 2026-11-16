import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bioacoustic_baselines import load_aves


class BioacousticBaselineTests(unittest.TestCase):
    def test_official_config_builds_expected_architecture(self):
        config = {
            "extractor_mode": "group_norm",
            "extractor_conv_layer_config": [[4, 4, 2], [4, 2, 2]],
            "extractor_conv_bias": False,
            "encoder_embed_dim": 4,
            "encoder_projection_dropout": 0.0,
            "encoder_pos_conv_kernel": 4,
            "encoder_pos_conv_groups": 1,
            "encoder_num_layers": 1,
            "encoder_num_heads": 1,
            "encoder_attention_dropout": 0.0,
            "encoder_ff_interm_features": 8,
            "encoder_ff_interm_dropout": 0.0,
            "encoder_dropout": 0.0,
            "encoder_layer_norm_first": False,
            "encoder_layer_drop": 0.0,
        }
        from torchaudio.models import wav2vec2_model
        original = wav2vec2_model(**config, aux_num_out=None)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "config.json").write_text(json.dumps(config))
            torch.save(original.state_dict(), folder / "model.pt")
            loaded = load_aves(folder / "config.json", folder / "model.pt")
            self.assertFalse(loaded.training)
            self.assertEqual(
                sum(parameter.numel() for parameter in original.parameters()),
                sum(parameter.numel() for parameter in loaded.parameters()),
            )


if __name__ == "__main__":
    unittest.main()
