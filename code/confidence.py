"""Validation-only temperature calibration for three-class canine affect scores."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

LABELS = {'arousal': ('Low', 'Medium', 'High'),
          'valence': ('Negative', 'Neutral', 'Positive')}


def checked_scores(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError('scores must be a nonempty finite (n, 3) matrix')
    return values


def checked_labels(labels, count):
    values = np.asarray(labels)
    if values.shape != (count,) or not np.isin(values, [0, 1, 2]).all():
        raise ValueError('labels must be one class index (0, 1, 2) per row')
    return values.astype(int)


def log_probabilities(scores, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    values = checked_scores(scores)
    # Center first to avoid overflow with large common logit offsets.
    values = (values - values.max(axis=1, keepdims=True)) / temperature
    return values - logsumexp(values, axis=1, keepdims=True)


def probabilities(scores, temperature=1.0):
    return np.exp(log_probabilities(scores, temperature))


def nll(scores, labels, temperature=1.0):
    logs = log_probabilities(scores, temperature)
    labels = checked_labels(labels, len(logs))
    return float(-logs[np.arange(len(logs)), labels].mean())


def fit_temperature(scores, labels, *, split):
    """Fit only on held-out validation scores from the SAME frozen classifier."""
    if split != 'validation':
        raise ValueError('temperature fitting requires validation data, never official test')
    scores = checked_scores(scores)
    labels = checked_labels(labels, len(scores))
    optimum = minimize_scalar(lambda log_t: nll(scores, labels, np.exp(log_t)),
                              bounds=(-4, 4), method='bounded')
    if not optimum.success:
        raise ValueError('temperature optimization failed')
    # Include the identity so optimization cannot worsen validation NLL.
    candidates = (1.0, float(np.exp(optimum.x)), float(np.exp(-4)), float(np.exp(4)))
    return min(candidates, key=lambda temperature: nll(scores, labels, temperature))


def reliability_metrics(scores, labels, temperature=1.0, bins=10):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('bins must be a positive integer')
    probs = probabilities(scores, temperature)
    labels = checked_labels(labels, len(probs))
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {'n': len(labels), 'nll': nll(scores, labels, temperature),
            'brier': float(np.mean(np.sum((probs - np.eye(3)[labels])**2, axis=1))),
            'ece': float(ece), 'ece_bins': bins,
            'accuracy': float(correct.mean())}


def risk_coverage(scores, labels, temperature=1.0):
    """Threshold curve with tied confidences retained or rejected together."""
    probs = probabilities(scores, temperature)
    labels = checked_labels(labels, len(probs))
    confidence = probs.max(axis=1)
    errors = probs.argmax(axis=1) != labels
    curve = []
    for threshold in sorted(set(confidence.tolist()), reverse=True):
        keep = confidence >= threshold
        curve.append({'threshold': threshold, 'retained': int(keep.sum()),
                      'coverage': float(keep.mean()), 'error_rate': float(errors[keep].mean())})
    return curve


def select_threshold(scores, labels, temperature=1.0, *, split, max_error=0.2, min_retained=20):
    if split != 'validation':
        raise ValueError('threshold selection requires held-out validation data')
    if not np.isfinite(max_error) or not 0 <= max_error <= 1:
        raise ValueError('max_error must be between zero and one')
    if not isinstance(min_retained, int) or min_retained < 1:
        raise ValueError('min_retained must be a positive integer')
    candidates = [row for row in risk_coverage(scores, labels, temperature)
                  if row['retained'] >= min_retained and row['error_rate'] <= max_error]
    if not candidates:
        return {'threshold': None, 'retained': 0, 'coverage': 0.0, 'error_rate': None,
                'reason': 'No validation threshold met the empirical error and sample-count criteria.'}
    return dict(max(candidates, key=lambda row: row['coverage']),
                reason='Maximum validation coverage meeting the empirical criteria; no test-risk guarantee.')


def predict(scores, target, temperature, threshold):
    if target not in LABELS:
        raise ValueError('target must be arousal or valence')
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('threshold must be null (abstain all) or a probability')
    probs = probabilities(scores, temperature)
    results = []
    for row in probs:
        index = int(row.argmax())
        confidence = float(row[index])
        reliable = threshold is not None and confidence >= threshold
        results.append({'prediction': LABELS[target][index] if reliable else None,
                        'top_class': LABELS[target][index],
                        'probabilities': dict(zip(LABELS[target], row.tolist())),
                        'confidence': confidence, 'status': 'reliable' if reliable else 'uncertain'})
    return results
