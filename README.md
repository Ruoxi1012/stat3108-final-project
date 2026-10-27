# Bark Beyond Identity

**Auditing and Improving Cross-Dog Generalization in Canine Vocal Affect Recognition**

STAT UN3108 Applied Deep Learning and AI Final Project, Fall 2026

Team members: Ruoxi Zhao and Cici Zeng

## Project overview

Can a model infer affect-related information from a short dog vocalization, or is it mainly memorizing the dog, breed, recording source, or background? We will study audio-only prediction of **arousal** (low, medium, high) and **valence** (negative, neutral, positive) using the open EmotionalCanines dataset.

We deliberately use the phrase **vocal affect recognition**, not “dog emotion translation.” The labels are human annotations derived from video context and observable behavior; they are not direct measurements of a dog's private emotional state and are not suitable for veterinary diagnosis.

## Research questions

1. How much does realistic background noise reduce classification performance, and can audio enhancement recover that performance without removing useful vocal cues?
2. Can a model trained on some dogs accurately classify vocalizations from dogs it has never encountered, including transfer between the two available breeds?
3. Can the system provide useful probabilities and decline to make a prediction when its confidence is too low?

## Confirmed innovation plan

The project is a **robustness, generalization, and reliability study**, not another architecture-only accuracy contest. Our three confirmed innovations are:

1. **Noise-robust prediction:** compare the same recording in original, noise-added, and noise-added-then-enhanced forms. This paired design keeps the dog, breed, sex, vocalization, and label unchanged while varying audio quality.
2. **Generalization to unseen dogs:** use official test sets containing dogs absent from training, and test Husky-to-Shiba and Shiba-to-Husky transfer. With only two breeds, this evaluates unseen-dog and cross-breed generalization; it does not prove performance on every dog breed.
3. **Confidence-aware output:** return the predicted affect-related state, the probability of each class, and a reliability decision. If confidence is below a threshold selected on validation data, return “uncertain” rather than forcing a label.

Model comparisons—including MFCC/eGeMAPS, a small log-Mel CNN, general pretrained encoders, and canine-specific Dog2vec—support these innovations but are not presented as separate innovations. As of the literature review dated 2026-10-11, we did not find prior work that evaluates this complete combination on EmotionalCanines. This is a bounded novelty claim, not a claim that no related method exists anywhere.

## Data

The primary dataset will be [EmotionalCanines](https://doi.org/10.1145/3746027.3758286): 1,400 short vocalization sequences from 199 Huskies and Shiba Inus, with separate train/test dogs and three-level arousal and valence labels. The dataset repository is [tmdang1101/EmotionalCanines](https://github.com/tmdang1101/EmotionalCanines), licensed CC BY-NC-SA 4.0.

Audio files and model checkpoints will not be committed. See [data/README.md](data/README.md) for provenance, license, and retrieval instructions.

## Planned models

1. Majority and stratified random baselines.
2. eGeMAPS or MFCC features with logistic regression/SVM.
3. A small CNN trained on log-Mel spectrograms.
4. Frozen pretrained embeddings with the same lightweight prediction head:
   - a human-speech representation (HuBERT or wav2vec 2.0);
   - a general/bioacoustic representation (AVES or BEATs);
   - [Dog2vec](https://doi.org/10.21437/Interspeech.2025-1287), pretrained specifically on canine vocalizations.
5. Probability calibration and an abstention rule on the strongest validated classifier.

The minimum viable comparison uses one model from each representation family. Additional encoders and ordinal-learning heads are stretch experiments, not requirements.

## Evaluation contract

All splits are fixed before model comparison. Augmented versions of a clip remain in the same split as their source.

- **Primary:** macro F1 on the official dog-disjoint test sets.
- **Noise experiment:** paired results on original, noise-added, and enhanced versions at predeclared noise levels.
- **Ordinal quality:** mean absolute class error and quadratic weighted kappa.
- **Reliability:** negative log-likelihood, Brier score, expected calibration error, and risk-coverage curves.
- **Stress tests:** background noise, gain change, compression/resampling, and cross-breed transfer.
- **Analysis:** per-class confusion matrices, bootstrap confidence intervals, severe-error counts, and reviewed examples of correct, incorrect, and abstained predictions.

We will report Husky and Shiba results separately as well as pooled results. Accuracy is secondary because class distributions differ.

## Current status

- [x] Project scope and scientific claim refined.
- [x] Initial field review and evidence matrix completed.
- [x] Primary dataset and license verified.
- [x] Leakage-safe evaluation protocol specified.
- [x] Reproducible environment and data audit (2026-10-18; issues documented below).
- [x] Honest majority, random, MFCC-linear, and small Log-Mel CNN baselines (2026-10-27).
- [ ] Pretrained representation comparison.
- [ ] Full paired original/noisy/enhanced classification experiment (the generation smoke example is complete).
- [ ] Calibrated confidence and uncertain-output rule.
- [ ] Robustness, ablations, and error analysis.
- [ ] Demo, presentation, and two-page report.

## Repository map

```text
README.md                         Project snapshot
research/field_review.md          Prior work, limitations, and research gap
research/experiment_blueprint.md  Frozen protocol and success criteria
planning/weekly_github_plan.md    Weekly branches, PRs, and deliverables
data/README.md                    Dataset provenance and retrieval
code/                             Reproducible audit and paired-audio pipeline
results/                          Audits, small tables, and figures
poster/                           Presentation artifact and checkpoints
report/                           Two-page report
```

## Responsible AI use

Pretrained audio models are part of the scientific method. Generative AI tools may assist with literature triage, code scaffolding, tests, debugging, and documentation, but every citation will be checked against its source, every experiment will be reproducible from code, and both team members must understand all submitted work. AI-generated labels will not be treated as ground truth. No secrets, private recordings, unavailable datasets, or large generated artifacts will be committed.

## Research documentation

- [Field review and evidence matrix](research/field_review.md)
- [Confirmed innovation and measurement plan](research/innovation_plan.md)
- [Experiment blueprint](research/experiment_blueprint.md)
- [Weekly GitHub plan](planning/weekly_github_plan.md)
- [Team responsibilities and contribution ledger](planning/team_contributions.md)


## First executable contribution (Cici)

The [data audit and paired noise pipeline](code/README.md) now provides executable
WAV quality checks, label counts, duplicate checks, source hashes, and reproducible
original/noisy/enhanced audio conditions. Follow its setup and test commands.
The current enhancer is a spectral-subtraction control using known added noise;
AI enhancement and classification comparisons remain planned experiments.

## Week 2 data checkpoint

The full 1,400-clip release was audited on 2026-10-18. Every file decoded, but
the audit found five groups of bitwise-equivalent decoded audio. Two groups cross
the official Shiba train/test boundary, three groups carry conflicting labels,
and one group falls into both categories.
The predeclared resolution removes train-side leakage before fitting and retains
the official test set for the primary score with a duplicate-sensitivity analysis.
See [the Week 2 audit record](results/week2_data_audit.md) for exact IDs, hashes,
class counts, audio QA, and the paired 10 dB / 0 dB smoke example.

## Week 3 baseline checkpoint

The fixed clip-level validation split contains 956 fit and 239 validation clips;
the 200 official test clips remain untouched during selection. On the official
unseen-dog test, the MFCC linear SVM is the strongest current baseline, with
pooled macro F1 of **0.490 for arousal** and **0.419 for valence**. The three-seed
small Log-Mel CNN reaches 0.424 ± 0.064 and 0.354 ± 0.026, so the initial deep
baseline does not beat the simpler model. This negative result is retained rather
than hidden. See the [generated Week 3 record](results/week3_baselines/README.md)
for the exact split, selection trace, per-breed metrics, confusion matrices, and
reproduction command.
