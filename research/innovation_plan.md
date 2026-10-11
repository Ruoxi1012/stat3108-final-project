# Confirmed Innovation and Measurement Plan

Decision date: **2026-10-11**

This document records the three innovation ideas selected by the team and defines what evidence is required to support each claim.

## 1. Noise-robust prediction

### Idea

Test whether noise augmentation or audio enhancement improves affect classification in realistic listening conditions.

### Controlled comparison

For each selected source recording, create a matched set:

1. the original/reference recording;
2. the same recording with reproducible background noise added;
3. the noisy recording after audio enhancement.

Because all three conditions come from the same clip, the dog, breed, sex, vocalization, and label remain fixed. The independent variable is the audio condition. “Original” does not necessarily mean perfectly clean; it means that the project has not added noise to it.

### Evidence to report

- macro F1 and per-class recall for all three conditions;
- performance loss from original to noisy;
- performance recovered from noisy to enhanced;
- results at mild and moderate predeclared noise levels;
- a safety check that enhancement does not damage original recordings;
- examples where enhancement helps and where it removes useful information.

### Claim boundary

The project may claim improved robustness only for the tested noise types, levels, models, and dataset. It must not claim removal of all real-world noise.

## 2. Generalization to unseen dogs

### Idea

Investigate whether a model trained on some dogs can classify vocalizations from dogs it has never encountered.

### Evaluation

- Husky training set to official unseen-Husky test set;
- Shiba Inu training set to official unseen-Shiba test set;
- Husky training set to Shiba Inu official test set;
- Shiba Inu training set to Husky official test set;
- pooled training set to both official test sets, reported separately by breed.

The official test labels must never be used for model selection, threshold selection, or calibration.

### Evidence to report

- macro F1 and confusion matrices for each setting;
- comparison with clip-level validation to show the generalization gap;
- bootstrap confidence intervals;
- separate results for arousal and valence;
- analysis of high-confidence failures on unseen dogs.

### Claim boundary

EmotionalCanines contains only Huskies and Shiba Inus. The project can demonstrate generalization to held-out individual dogs and measure transfer between these two breeds. It cannot prove that the model works for every dog or every breed.

## 3. Confidence-aware predictions

### Idea

Instead of always forcing an answer, the system will provide:

- the predicted affect-related state;
- the probability of every possible class;
- an indication of whether the prediction is sufficiently reliable.

If confidence is below a threshold selected using validation data, the output will be **uncertain**.

### Example output

```text
Arousal prediction: High
Probabilities: Low 0.08 | Medium 0.17 | High 0.75
Reliability: Reliable
```

```text
Valence prediction: Uncertain
Probabilities: Negative 0.36 | Neutral 0.34 | Positive 0.30
Reliability: Insufficient confidence
```

### Evidence to report

- calibration error and Brier score before and after calibration;
- accuracy or error among retained predictions at different coverage levels;
- the validation-only procedure used to choose the confidence threshold;
- examples of correct, incorrect, and uncertain predictions;
- comparison of confidence on original, noisy, and enhanced recordings.

### Claim boundary

A high probability is not proof that a prediction is correct. The project must test calibration and clearly distinguish model confidence from biological certainty.

## One-sentence project contribution

> We study whether dog vocal affect recognition can remain useful under background noise, generalize to dogs absent from training, and communicate when its predictions are not reliable enough to trust.
