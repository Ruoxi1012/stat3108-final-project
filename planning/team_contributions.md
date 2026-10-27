# Team Responsibilities and Contribution Ledger

Last updated: **2026-10-27**

This file makes individual ownership visible while preserving peer review and shared understanding. It records completed work using Git commit evidence and assigns the next concrete deliverables. Responsibilities may change through a reviewed pull request rather than silently.

## Ruoxi Zhao

### Completed contributions

- Defined the three confirmed innovations and their bounded claims.
- Wrote the literature review, evidence matrix, and research-gap analysis.
- Designed the unseen-dog, cross-breed, paired-noise, and confidence-aware evaluation protocol.
- Added the weekly GitHub plan and dataset governance documentation.
- Implemented the fixed development split, MFCC linear baselines, and the
  three-seed small Log-Mel CNN.
- Ran the first real official unseen-dog experiment and retained the finding
  that the small CNN does not beat the MFCC linear SVM.
- Generated per-breed metrics, ordinal metrics, confusion matrices, selection
  traces, and a presentation-ready baseline figure.

Git evidence:

- `836f3c5` — `docs: define confirmed project innovations`
- `d329015` — `research: add evidence review and evaluation protocol`
- `5944fa0` — `chore: add weekly plan and data governance`
- `83b05e4` — `feat: add leakage-aware baseline experiment`
- `c8745d0` — `docs: explain honest baseline reproduction`
- `b68e590` — `fix: average stochastic baseline across seeds`
- `2108cfc` — `exp: record honest Week 3 baseline results`
- Pull request: [#2](https://github.com/Ruoxi1012/stat3108-final-project/pull/2)

### Primary next responsibilities

- Run official unseen-dog and Husky-to-Shiba/Shiba-to-Husky evaluations without test-set tuning.
- Add bootstrap confidence intervals and the official-test duplicate-sensitivity analysis.
- Compare model performance on original, noise-added, and enhanced audio produced by the shared pipeline.
- Produce confidence intervals, confusion matrices, and presentation-ready generalization/robustness figures.

Working branch: `ruoxi/honest-baselines` (Week 3, awaiting peer review)

## Cici Zeng

### Completed contributions

- Added pinned dependencies for the audio pipeline.
- Implemented the reproducible audio audit and paired noise-generation pipeline.
- Added tests for SNR, provenance, and audit failures.
- Documented the pipeline and excluded generated artifacts from Git.

Git evidence:

- `2e5364a` — `chore: add pinned audio pipeline dependencies`
- `bf16c0d` — `feat: add reproducible data audit and paired noise pipeline`
- `b69f9f7` — `test: verify SNR, provenance, and audit failures`
- `3ba59ee` — `docs: explain data QA and paired noise controls`
- `26ed191` — `chore: exclude generated audio, audits, and Python caches`
- `60132f9` — `docs: record Cici's first executable contribution`
- Pull request: [#3](https://github.com/Ruoxi1012/stat3108-final-project/pull/3)

### Primary next responsibilities

- Extend the paired pipeline with a reproducible audio-enhancement method and safety checks on original recordings.
- Implement calibrated class probabilities and the reliable/uncertain output interface.
- Add automated tests that prevent data leakage and verify confidence-threshold behavior.
- Prepare demo examples covering reliable, incorrect, and uncertain predictions.

Working branch: create a task-specific `cici/...` branch for each new unit of work.

## Shared responsibilities

- Review each other's pull requests and run the documented smoke tests before approval.
- Understand and be able to explain every submitted method, result, and limitation.
- Jointly select final claims after reviewing confidence intervals and failure cases.
- Rehearse the 10-minute presentation and share speaking and Q&A responsibilities.
- Keep secrets, raw audio, checkpoints, and generated artifacts out of Git.

## Week 2 checkpoint

Pull request [#4](https://github.com/Ruoxi1012/stat3108-final-project/pull/4)
merged the AI-assisted Week 2 checkpoint: a full-dataset audit, explicit duplicate and conflicting-label
findings, aggregate audio QA, selective paired-audio generation, a licensed
wind-noise smoke example at 10 dB and 0 dB, documentation updates, and a
presentation draft.

## Week 3 checkpoint prepared for review

The `ruoxi/honest-baselines` branch contains the first real model experiment.
All eight tests pass. The strongest official-test result is the MFCC linear SVM
(arousal macro F1 0.490; valence 0.419); the small CNN is weaker and has visible
seed variance. Cici should run the documented reproduction or smoke tests,
review the fixed split and no-test-tuning logic, and confirm that both members
can explain the negative CNN result before merge.

## Contribution rule

A branch name alone is not evidence of contribution. Each contribution must end in a small, explainable commit and a pull request that identifies the question answered, files changed, verification command, result, and remaining limitation.
