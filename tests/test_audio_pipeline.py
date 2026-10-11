import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

spec = importlib.util.spec_from_file_location('audio_pipeline', Path(__file__).resolve().parents[1]/'code/audio_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class AudioTests(unittest.TestCase):
    def test_snr_and_repeatability(self):
        x = np.sin(np.arange(16000)*.1)
        n = np.random.default_rng(4).normal(size=2000)
        a, mixed, added, gain = pipeline.mix_noise(x,n,5,3108)
        measured = 10*np.log10(np.mean(a*a)/np.mean((mixed-a)**2))
        self.assertAlmostEqual(measured,5,places=10)
        np.testing.assert_array_equal(mixed,pipeline.mix_noise(x,n,5,3108)[1])
        self.assertLessEqual(np.max(np.abs(mixed)),.9900001)
        with self.assertRaises(ValueError):
            pipeline.mix_noise(np.zeros(10),n,5,0)

    def test_enhancement_preserves_length_and_is_finite(self):
        x = np.sin(np.arange(16000)*.1)*.2
        n = np.random.default_rng(3).normal(size=len(x))*.05
        result = pipeline.enhance(x+n,n,16000)
        self.assertEqual(len(result),len(x))
        self.assertTrue(np.isfinite(result).all())

    def test_audit_and_split_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for breed in ('husky','shiba'):
                for split in ('train','test'):
                    audio_id = f'{breed}_{split}_00000'
                    folder = root/breed/split
                    folder.mkdir(parents=True)
                    wavfile.write(folder/f'{audio_id}.wav',16000,np.sin(np.arange(1000)*.1).astype(np.float32))
                    (root/f'{breed}_{split}_labels.csv').write_text(f'audio_id,arousal,valence\n{audio_id},High,Positive\n')
            summary = pipeline.audit(root,root/'audit')
            self.assertEqual(summary['readable'],4)
            self.assertTrue(any('crosses official splits' in p['reason'] for p in summary['issues']))
            noise = root/'noise.wav'
            wavfile.write(noise,8000,np.random.default_rng(9).normal(0,.05,1000).astype(np.float32))
            records = pipeline.pairs(root/'audit/manifest.json',noise,root/'pairs',5,3108)
            self.assertEqual(len(records),16)
            for row in records:
                self.assertIn('/'+row['split']+'/',row['path'])
                rate, audio, _ = pipeline.read_audio(Path(row['path']))
                self.assertEqual(rate,16000)
                self.assertEqual(len(audio),1000)
            with self.assertRaises(ValueError):
                pipeline.pairs(root/'audit/manifest.json',noise,root/'pairs',5,3108)
            source = root/'husky/train/husky_train_00000.wav'
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                pipeline.pairs(root/'audit/manifest.json',noise,root/'changed-pairs',5,3108)

    def test_lfs_pointer_is_not_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'husky/train'
            folder.mkdir(parents=True)
            (folder/'husky_train_00000.wav').write_text('version https://git-lfs.github.com/spec/v1\n')
            (root/'husky_train_labels.csv').write_text('audio_id,arousal,valence\nhusky_train_00000,High,Positive\n')
            summary = pipeline.audit(root,root/'audit')
            self.assertEqual(summary['readable'],0)
            self.assertTrue(any('Git LFS pointer' in p['reason'] for p in summary['issues']))


if __name__ == '__main__':
    unittest.main()
