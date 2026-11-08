import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('confidence', Path(__file__).resolve().parents[1]/'code/confidence.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CalibrationTests(unittest.TestCase):
    def test_temperature_reduces_overconfidence_preserving_predictions(self):
        scores = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 8]] * 10)
        labels = np.array([0, 1, 2] * 5 + [1, 2, 0] * 5)
        temperature = c.fit_temperature(scores, labels, split='validation')
        self.assertGreater(temperature, 1)
        self.assertLess(c.nll(scores, labels, temperature), c.nll(scores, labels))
        np.testing.assert_array_equal(c.probabilities(scores).argmax(1),
                                      c.probabilities(scores, temperature).argmax(1))
        np.testing.assert_allclose(c.probabilities(scores, temperature).sum(1), 1)

    def test_uniform_metrics(self):
        metrics = c.reliability_metrics(np.zeros((3, 3)), [0, 1, 2])
        self.assertAlmostEqual(metrics['nll'], np.log(3))
        self.assertAlmostEqual(metrics['brier'], 2/3)
        self.assertAlmostEqual(metrics['ece'], 0)

    def test_validation_guard_and_invalid_inputs(self):
        for split in ('train', 'official_test', 'test'):
            with self.assertRaises(ValueError):
                c.fit_temperature([[1, 2, 3]], [0], split=split)
        for scores in ([], [[1, 2]], [[float('nan'), 1, 2]]):
            with self.assertRaises(ValueError):
                c.probabilities(scores)
        for labels in ([3], [0.5], []):
            with self.assertRaises(ValueError):
                c.nll([[1, 2, 3]], labels)
        with self.assertRaises(ValueError):
            c.probabilities([[1, 2, 3]], 0)

    def test_common_offset_does_not_change_probability(self):
        np.testing.assert_allclose(c.probabilities([[10001, 10002, 10003]]),
                                   c.probabilities([[1, 2, 3]]))


class SelectivePredictionTests(unittest.TestCase):
    def test_threshold_maximizes_coverage_and_keeps_ties(self):
        scores = [[8, 0, 0], [4, 0, 0], [1, 0, 0], [1, 0, 0]]
        labels = [0, 0, 1, 0]
        policy = c.select_threshold(scores, labels, split='validation', max_error=0, min_retained=2)
        self.assertEqual(policy['retained'], 2)
        self.assertEqual(policy['coverage'], .5)
        output = c.predict(scores, 'arousal', 1, policy['threshold'])
        self.assertEqual([r['status'] for r in output], ['reliable', 'reliable', 'uncertain', 'uncertain'])
        self.assertIsNone(output[-1]['prediction'])
        self.assertEqual(output[-1]['top_class'], 'Low')

    def test_no_threshold_abstains_all_without_nan(self):
        policy = c.select_threshold([[1, 0, 0]], [1], split='validation', min_retained=1, max_error=0)
        self.assertIsNone(policy['threshold'])
        self.assertIsNone(policy['error_rate'])
        self.assertEqual(c.predict([[1, 0, 0]], 'valence', 1, policy['threshold'])[0]['status'], 'uncertain')

    def test_uniform_ties_cannot_be_selectively_split(self):
        curve = c.risk_coverage(np.zeros((3, 3)), [0, 1, 2])
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve[0]['coverage'], 1)
        with self.assertRaises(ValueError):
            c.select_threshold([[1, 0, 0]], [0], split='official_test')
        with self.assertRaises(ValueError):
            c.predict([[1, 0, 0]], 'happy', 1, .5)
        with self.assertRaises(ValueError):
            c.select_threshold([[1, 0, 0]], [0], split='validation', min_retained=0)


if __name__ == '__main__':
    unittest.main()
