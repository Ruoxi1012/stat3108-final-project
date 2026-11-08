# Confidence-aware predictions (Cici, November 3)

This component supports the project's third innovation: return arousal/valence
probabilities and decline uncertain predictions. It accepts three-class model
scores; it does not train a dog-audio classifier. AI assistance: Codex helped
implement and test the code; team review and understanding are required.

## What is implemented

- Positive temperature scaling minimizes validation negative log-likelihood
  (NLL); it changes confidence without changing the highest-scoring class.
- NLL, multiclass Brier score (sum over three classes), and top-label expected
  calibration error (10 equal-width confidence bins).
- A threshold chosen for maximum validation coverage satisfying a predeclared
  empirical error limit and minimum retained count. Equal-confidence clips
  stay together. If no threshold qualifies, all predictions are uncertain.
- A threshold-based risk/coverage table; each point reports the error fraction
  among retained predictions. No statistical risk guarantee is claimed.
- Output: class probabilities, top class, confidence, reliable/uncertain status,
  and a prediction that is null when uncertain.

## Integration contract — prevent leakage and model mismatch

Freeze model choice and weights first. Divide held-out development clips into
**calibration** and **threshold-selection** sets before inspecting their scores.
Neither set can have been used for fitting the classifier. Validation used for
model selection is not an independent final evaluation. Only the official test
supports the final dog-disjoint evaluation; public validation dog IDs are absent.

Both JSON exports use this schema (illustrative single row, not a real result):

```json
{
  "model_id": "exact-checkpoint-sha256-or-version",
  "target": "arousal",
  "class_names": ["Low", "Medium", "High"],
  "split": "validation",
  "fit_ids": ["husky_train_00001"],
  "source_ids": ["husky_train_00002"],
  "logits": [[1.2, 0.3, -0.4]],
  "labels": [0]
}
```

Use independent exports for arousal and valence; valence order is Negative,
Neutral, Positive. IDs must identify original source clips, not augmented file
names. Group all derivatives with their source and apply the audit's exclusion
policy upstream. The CLI checks declared IDs and cannot prove metadata honesty,
detect undeclared duplicates, or infer dog identity.

Use raw CNN logits. For logistic regression, log of predict_proba (floored at
1e-12) can be used as logits, with columns explicitly reordered to class_names.
Linear SVM margins can be scaled as scores but are not preexisting probabilities.
Ruoxi's current baseline branch exports aggregate metrics, so per-clip score
exports remain an integration task. Do not reconstruct probabilities from
accuracy, a confusion matrix, or hard class predictions.

**Do not calibrate a development model and apply its policy to a refitted model.**
The baseline branch currently refits on all eligible training clips, including
validation; that refitted model needs new independent calibration data. Instead,
keep a classifier trained without the calibration/selection clips, and use that
same frozen classifier for all exports and official testing. model_id and fit_ids
must refer to its actual weights and training sources.

## Commands

Use the repository's existing pinned NumPy/SciPy environment:

```bash
python -m unittest discover -s tests -v
python code/confidence_cli.py fit \
  --calibration results/local/calibration.json \
  --selection results/local/selection.json \
  --max-error 0.2 --min-retained 20 \
  --output results/local/confidence-policy.json
python code/confidence_cli.py predict \
  --policy results/local/confidence-policy.json \
  --scores results/local/official-scores.json \
  --output results/local/confidence-predictions.json
```

Prediction input uses the same metadata, source_ids and logits; labels are not
required or read. The CLI rejects overlap with any fitting/calibration/selection
source, mismatched model IDs, or changed class order. Output paths must be fresh.
Policy artifacts record parameters, source IDs, input hashes, selection metrics,
and the validation risk/coverage curve. Keep model exports/artifacts local.

The library functions in confidence.py also support in-process demo inference;
the CLI's stricter source-overlap check is for honest held-out evaluation.

## Interpretation and remaining work

“Reliable” means this confidence exceeds an empirically chosen threshold, not
that the dog's internal emotion is verified, the audio is clean, or the breed is
known. Low confidence is not a detector for all unusual sounds. Error can rise on
new breeds or noisy audio. Calibration need not improve Brier/ECE on another set,
and abstention need not improve risk at every coverage. Report those outcomes.

Current verification uses synthetic scores only. No new EmotionalCanines result,
accuracy gain, or calibrated model is claimed. Next: export held-out scores from
one frozen baseline, fit separate policies per target, evaluate on untouched
official clips and all paired audio conditions, and report per-breed reliability
and risk/coverage alongside macro F1. Include uncertainty intervals before making
comparative claims. Noise enhancement and pretrained encoders remain separate tasks.
