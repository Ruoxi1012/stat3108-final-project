# Experiment Blueprint

Status: protocol proposed before data access or model training. Any later change must be recorded with date and reason.

## 1. Exact task

Input: one short dog vocalization sequence. Any padding, cropping, or target-duration rule will be frozen only after a label-blind duration audit and then applied identically across models.

Outputs:

- arousal: low, medium, or high;
- valence: negative, neutral, or positive;
- calibrated confidence and optional abstention.

Scientific claim: predict labels inferred by annotators from visible behavior/context using audio alone. The project does not claim to recover a dog's true private emotion.

## 2. Hypotheses

- **H1:** adding realistic background noise will reduce macro F1 relative to the same original recordings.
- **H2:** audio enhancement will recover part of the noise-related performance loss without materially reducing performance on the original recordings.
- **H3:** performance on official unseen-dog tests will be materially lower than clip-level validation performance, but pretrained representations will retain more performance than from-scratch baselines.
- **H4:** temperature scaling will reduce ECE/Brier/NLL without changing class predictions, and abstention will reduce error on retained predictions as coverage decreases.

Any hypothesis may be rejected. A carefully measured negative result is still a valid contribution.

## 3. Data protocol

Primary dataset: EmotionalCanines.

| Breed | Train dogs | Train clips | Test dogs | Test clips |
|---|---:|---:|---:|---:|
| Husky | 44 | 600 | 18 | 100 |
| Shiba Inu | 101 | 600 | 36 | 100 |
| Total | 145 | 1,200 | 54 | 200 |

The released labels do not expose dog IDs inside each split. Therefore:

- never tune on the official test clips;
- use a fixed, seeded train/validation split only for tuning and state that it may contain dog/source overlap;
- use the official test sets as the only verified dog-disjoint evaluation;
- fit calibration parameters on validation only;
- keep all augmentations derived from a source clip on the source side of the split.

## 4. Evaluation regimes

### A. Within-breed unseen dog (primary)

- Husky train -> Husky official test.
- Shiba train -> Shiba official test.

### B. Cross-breed zero shot

- Husky train -> Shiba official test.
- Shiba train -> Husky official test.

This measures breed shift but also includes label-prior and recording-source shift. Interpret it as domain transfer, not a pure causal breed effect.

### C. Pooled multi-breed

- Husky + Shiba training clips -> both official test sets.
- Report each breed separately and pooled using macro averaging.

### D. Clip-level validation audit

Use fixed seeded stratified validation within the released training set. Compare it with the official test only to quantify the distribution/generalization gap. Because public dog IDs are absent, do not call the difference proven identity leakage.

## 5. Model ladder

Every more complex model must beat or clarify a simpler model.

| ID | Representation | Head | Purpose |
|---|---|---|---|
| B0 | None | Majority / stratified random | Sanity baseline |
| B1 | MFCC summary | Logistic regression and/or linear SVM | Traditional baseline from proposal |
| B2 | eGeMAPS | Linear SVM | Literature-aligned affect baseline |
| B3 | Log-Mel spectrogram | Small 2D CNN | From-scratch deep baseline |
| T1 | HuBERT or wav2vec 2.0 frozen embedding | Same MLP | Human-speech transfer |
| T2 | AVES or BEATs frozen embedding | Same MLP | Bio/general-audio transfer |
| T3 | Dog2vec frozen embedding | Same MLP | Canine-specific transfer |
| P1 | Best validated representation | Calibrated classifier | Reliable class probabilities |
| P2 | P1 probabilities | Validation-selected abstention rule | Reliable/uncertain output |

Freeze encoders for the main comparison to control compute and isolate representation quality. Fine-tuning is optional only after frozen baselines are stable.

## 6. Training controls

- Fixed random seeds: at least 3 for neural heads if compute allows.
- Same validation partition, head capacity, optimizer search budget, and early-stopping rule across frozen encoders.
- Class-weighted loss or balanced sampling chosen on validation; document one choice.
- Normalize amplitude consistently, but retain a version before normalization for stress tests.
- Initial augmentation: conservative background noise and gain only.
- Pitch shift and time stretch are separate ablations because they may modify affect-relevant cues.
- Save configs, package versions, hashes of label files, and exact commands.

## 7. Metrics

### Primary discrimination metrics

- Macro F1 for arousal and valence separately.
- Balanced accuracy as a secondary check.
- Per-class precision and recall.

### Ordinal metrics

- Mean absolute class error after mapping labels to 0, 1, 2.
- Quadratic weighted kappa.
- Severe error rate: low predicted high or high predicted low; negative predicted positive or positive predicted negative.

### Reliability metrics

- Negative log-likelihood.
- Multiclass Brier score.
- Expected calibration error with bins declared in advance.
- Risk-coverage curve and area under risk-coverage curve.

### Uncertainty

- Stratified bootstrap 95% confidence intervals on each official test set.
- Paired bootstrap differences for the proposed model versus the strongest baseline.
- Avoid “significant improvement” language when intervals are wide or overlap zero.

## 8. Required ablations

1. Original vs noise-added vs noise-added-then-enhanced versions of the same clips.
2. HuBERT/wav2vec vs AVES/BEATs vs Dog2vec with frozen encoders.
3. No noise augmentation vs conservative noise augmentation during training.
4. Uncalibrated vs temperature-scaled probabilities.
5. Forced prediction vs validation-selected abstention at several coverage levels.
6. Within-breed vs cross-breed vs pooled training.

Stretch ablations:

- encoder layer choice for Dog2vec;
- pitch/time augmentation;
- independent cross-entropy vs joint or ordinal heads;
- shallow fine-tuning of the strongest encoder.

## 9. Robustness tests

The primary noise experiment is paired: each selected source clip produces three versions with the same dog, breed, sex, vocal content, and label.

1. **Original/reference:** the released recording before our added corruption. “Original” does not imply studio-clean audio.
2. **Noisy:** the same recording with a reproducible background-noise sample added at a predeclared signal-to-noise ratio.
3. **Enhanced:** the noisy version processed by the selected audio-enhancement method.

This paired design changes only the audio condition. Noise files, mixing code, SNR, enhancement model/version, and random seed must be recorded. Noisy and enhanced derivatives must stay in the same data split as their source.

Apply the following transformations only at test time, at predeclared mild and moderate levels:

- background noise at two SNR levels;
- gain change;
- MP3/AAC-style compression if reproducible;
- resampling or bandwidth reduction;
- optional pitch shift and time stretch as semantic-sensitivity probes.

Report paired original-to-noisy degradation and noisy-to-enhanced recovery using macro F1, per-class recall, and bootstrap confidence intervals. Also run the enhancer on original recordings as a safety check: enhancement is not useful if it removes affect-related cues. A model that is accurate only on relatively clean web clips is not robust enough for a demo claim.

## 10. Interpretation and error analysis

For eGeMAPS models, report permutation importance or SHAP values and compare them with known cues: F0/pitch, HNR/tonality, energy, duration, and rhythm.

For deep representations, use time-frequency occlusion rather than a visually attractive but unvalidated saliency map. Test whether masking the highlighted region changes the prediction.

Review at least:

- five correct high-confidence cases;
- five incorrect high-confidence cases;
- five low-confidence/abstained cases;
- errors involving the middle classes;
- cross-breed failures.

Do not publish or redistribute source media outside its license.

## 11. Success criteria

### Minimum viable project

- Fully reproducible data audit and loader.
- B0, B1/B2, B3, and one pretrained encoder.
- Official unseen-dog evaluation with macro F1 and confidence intervals.
- Clear error analysis and presentation-ready plots.

### Strong project

- Controlled three-family representation comparison.
- Cross-breed transfer matrix.
- Paired noise-degradation and enhancement-recovery analysis.
- Calibration, uncertain output, and risk-coverage analysis.
- Stable 60-second demo that shows probabilities, confidence, and abstention.

### What is not required

- State-of-the-art accuracy on a private dataset.
- Training a foundation model from scratch.
- Claiming veterinary utility.
- Collecting private pet recordings.

## 12. Presentation-first outputs

The final presentation should be built around five visuals:

1. “95% headline vs unseen-dog reality” evidence slide.
2. Train/test design diagram showing dog-disjoint and cross-breed evaluation.
3. Representation comparison with bootstrap intervals.
4. 2x2 or 3x3 generalization matrix across breeds/regimes.
5. Calibration/risk-coverage plot plus two audio error cases.

One conclusion should remain understandable without audio playback: whether the model learned a transferable affect signal and when it knows that it does not know.
