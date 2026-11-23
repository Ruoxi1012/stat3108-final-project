# Final target-specific model

The final system is frozen as a two-head model: **AVES-bio + linear SVM for
arousal** and **richer acoustic features + ExtraTrees for valence**. This choice
combines the best already-frozen official point estimate for each target without
performing another search on the official labels.

| Target | Frozen model | Accuracy | Official unseen-dog macro F1 (95% bootstrap CI) | Week 3 baseline | Paired difference CI |
|---|---|---:|---:|---:|---:|
| Arousal | AVES-bio L6 mean+std + breed, linear SVM | 0.535 | 0.535 (0.465, 0.602) | 0.490 | -0.025 to +0.113 |
| Valence | Rich acoustic + breed, ExtraTrees | 0.480 | 0.472 (0.402, 0.541) | 0.419 | -0.033 to +0.137 |

Both paired difference intervals include zero. The final model is therefore a
reproducible best point-estimate system, not evidence of a statistically clear
improvement or precise emotion reading.

The final accuracy-oriented follow-up compared four valence candidates with
five-fold development CV. Its stable RBF winner reached 0.465 official accuracy
and 0.462 macro F1, below ExtraTrees. The full negative result is retained in
[`results/final_selection`](../final_selection/README.md). Because multiple
official results have now been inspected, the table above is an exploratory
best-point-estimate summary rather than a pristine held-out test claim.

## Build the local model bundle

The bundle is generated under ignored `results/local/`; dataset audio, AVES
weights, caches, and fitted estimators are not committed.

```bash
python code/final_model.py train \
  --manifest results/local/audit/manifest.json \
  --aves-cache results/local/aves_base_bio \
  --acoustic-cache results/local/week3_features.npz \
  --output results/local/final_model.joblib
```

## Predict one WAV

Input must be mono/stereo WAV at 16 kHz and the breed must be one of the two
released dataset breeds.

```bash
python code/final_model.py predict \
  --bundle results/local/final_model.joblib \
  --model results/local/models/aves/aves-base-bio.torchaudio.pt \
  --config results/local/models/aves/aves-base-bio.torchaudio.model_config.json \
  --audio path/to/example.wav \
  --breed husky
```

The output includes one class and a score distribution per target. These scores
are explicitly uncalibrated and have `research_output_only` status. The strict
validation-only confidence experiment found no qualifying threshold and
honestly abstained on all official clips; this final wrapper does not relabel
uncalibrated scores as reliable probabilities.

See [model_card.json](model_card.json) for the frozen configuration, metrics,
scope, and intended-use restrictions.
