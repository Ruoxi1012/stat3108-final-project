"""Fit a confidence policy on validation exports, then apply it without labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from confidence import (LABELS, checked_labels, checked_scores, fit_temperature,
                        predict, reliability_metrics, risk_coverage, select_threshold)


def load(path):
    return json.loads(Path(path).read_text())


def unique_ids(values, name, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f'{name} must be a list of source audio IDs')
    if any(not isinstance(item, str) or not item for item in values) or len(values) != len(set(values)):
        raise ValueError(f'{name} must contain unique nonempty strings')
    return set(values)


def validate_export(data, validation=False):
    if data.get('target') not in LABELS or data.get('class_names') != list(LABELS[data['target']]):
        raise ValueError('explicit class_names must match the ordered target labels')
    if not isinstance(data.get('model_id'), str) or not data['model_id']:
        raise ValueError('model_id must identify the exact frozen model/checkpoint')
    ids = unique_ids(data.get('source_ids'), 'source_ids')
    scores = checked_scores(data.get('logits'))
    if len(ids) != len(scores):
        raise ValueError('one source ID is required per score row')
    if validation:
        if data.get('split') != 'validation':
            raise ValueError('fitting requires declared validation exports')
        fit_ids = unique_ids(data.get('fit_ids'), 'fit_ids')
        if ids & fit_ids or any('_test_' in item for item in ids | fit_ids):
            raise ValueError('validation/training overlap or official-test ID in fitting data')
        checked_labels(data.get('labels'), len(scores))
    return ids


def fit_policy(calibration, selection, max_error, min_retained):
    calibration_ids = validate_export(calibration, validation=True)
    selection_ids = validate_export(selection, validation=True)
    for key in ('model_id', 'target', 'class_names'):
        if calibration[key] != selection[key]:
            raise ValueError(f'calibration and selection must use the same {key}')
    if set(calibration['fit_ids']) != set(selection['fit_ids']):
        raise ValueError('exports must describe the same model training IDs')
    if calibration_ids & selection_ids:
        raise ValueError('calibration and threshold-selection source IDs must be disjoint')
    temperature = fit_temperature(calibration['logits'], calibration['labels'], split='validation')
    policy = select_threshold(selection['logits'], selection['labels'], temperature,
                              split='validation', max_error=max_error, min_retained=min_retained)
    return {'schema_version': 1, 'target': calibration['target'], 'class_names': calibration['class_names'],
            'model_id': calibration['model_id'], 'temperature': temperature,
            'policy': policy, 'max_error': max_error, 'min_retained': min_retained,
            'fit_ids': calibration['fit_ids'], 'calibration_ids': calibration['source_ids'],
            'selection_ids': selection['source_ids'],
            'selection_metrics_before': reliability_metrics(selection['logits'], selection['labels']),
            'selection_metrics_after': reliability_metrics(selection['logits'], selection['labels'], temperature),
            'selection_risk_coverage': risk_coverage(selection['logits'], selection['labels'], temperature),
            'limitation': 'Empirical validation rule, not a guarantee of correctness or unseen-breed detection.'}


def apply_policy(artifact, data):
    ids = validate_export(data)
    for key in ('model_id', 'target', 'class_names'):
        if data[key] != artifact[key]:
            raise ValueError(f'prediction export does not match policy {key}')
    used = set(artifact['fit_ids'] + artifact['calibration_ids'] + artifact['selection_ids'])
    if ids & used:
        raise ValueError('evaluation source IDs overlap fitting, calibration, or threshold selection')
    output = predict(data['logits'], artifact['target'], artifact['temperature'], artifact['policy']['threshold'])
    for audio_id, row in zip(data['source_ids'], output):
        row['source_id'] = audio_id
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--calibration', required=True)
    fit.add_argument('--selection', required=True)
    fit.add_argument('--max-error', type=float, default=.2)
    fit.add_argument('--min-retained', type=int, default=20)
    fit.add_argument('--output', required=True)
    apply = commands.add_parser('predict')
    apply.add_argument('--policy', required=True)
    apply.add_argument('--scores', required=True)
    apply.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('output exists; choose a fresh path to preserve experiment history')
    if args.command == 'fit':
        output = fit_policy(load(args.calibration), load(args.selection), args.max_error, args.min_retained)
        output['input_sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in [('calibration', args.calibration), ('selection', args.selection)]}
        output['numpy_version'] = np.__version__
    else:
        output = apply_policy(load(args.policy), load(args.scores))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
