import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from confidence_cli import apply_policy, fit_policy


def export(prefix, split='validation'):
    return {'model_id': 'synthetic-frozen-model', 'target': 'arousal',
            'class_names': ['Low', 'Medium', 'High'], 'split': split,
            'fit_ids': ['husky_train_fit'], 'source_ids': [prefix+str(i) for i in range(6)],
            'logits': [[4, 0, 0], [0, 4, 0], [0, 0, 4]]*2, 'labels': [0, 1, 2]*2}


class WorkflowTests(unittest.TestCase):
    def test_overlap_and_model_refit_are_rejected(self):
        a, b = export('cal'), export('select')
        for key, value in [('source_ids', a['source_ids']), ('model_id', 'refit-model'),
                           ('fit_ids', ['different-training']), ('split', 'official_test'),
                           ('class_names', ['High', 'Medium', 'Low'])]:
            changed = copy.deepcopy(b)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_policy(a, changed, .2, 2)
        changed = copy.deepcopy(a)
        changed['fit_ids'] = [a['source_ids'][0]]
        with self.assertRaises(ValueError):
            fit_policy(changed, b, .2, 2)
        changed = copy.deepcopy(a)
        changed['source_ids'][0] = 'husky_test_00000'
        with self.assertRaises(ValueError):
            fit_policy(changed, b, .2, 2)

    def test_label_free_inference_and_evaluation_overlap_guard(self):
        a, b, test = export('cal'), export('select'), export('test', 'official_test')
        policy = fit_policy(a, b, .2, 2)
        del test['labels']
        self.assertEqual(len(apply_policy(policy, test)), 6)
        with self.assertRaises(ValueError):
            apply_policy(policy, a)

    def test_cli_fit_predict_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, data in [('cal', export('cal')), ('select', export('select')),
                               ('test', export('test', 'official_test'))]:
                (folder/f'{name}.json').write_text(json.dumps(data))
            script = str(ROOT/'code/confidence_cli.py')
            fit = [sys.executable, script, 'fit', '--calibration', str(folder/'cal.json'),
                   '--selection', str(folder/'select.json'), '--min-retained', '2',
                   '--output', str(folder/'policy.json')]
            subprocess.run(fit, check=True, capture_output=True)
            subprocess.run([sys.executable, script, 'predict', '--policy', str(folder/'policy.json'),
                            '--scores', str(folder/'test.json'), '--output', str(folder/'output.json')],
                           check=True, capture_output=True)
            output = json.loads((folder/'output.json').read_text())
            self.assertEqual(output[0]['status'], 'reliable')
            self.assertAlmostEqual(sum(output[0]['probabilities'].values()), 1)
            self.assertNotEqual(subprocess.run(fit, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
