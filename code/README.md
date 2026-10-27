# Cici's data audit and paired noise pipeline

Implemented with AI assistance; both members should review and understand the code.

## Setup and verification

Python 3.9 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download EmotionalCanines using the Git LFS instructions in `data/README.md`.
The tool reads the four released `{husky,shiba}_{train,test}_labels.csv` files
and audio under `{breed}/{split}/{audio_id}.wav`.

```bash
python code/audio_pipeline.py audit --root data/raw/EmotionalCanines --output results/local/audit
```

Outputs: `manifest.json` (per-record labels, hashes, audio QA, exclusions) and
`audit.json` (class counts, label hashes, aggregate audio QA, exact decoded
duplicates, and issues).
A nonzero exit means the audit found issues, including deviations from expected
600 train / 100 test clips per breed. Inspect all issues before proceeding.
No audio is silently repaired. Stereo is averaged for QA and paired generation.
Dog IDs are unavailable; this tool does not certify new dog-disjoint splits.
Near-duplicate detection and source-video review remain future work.

For model fitting, use only manifest rows with `model_status: include`. The
audit automatically excludes train-side copies of official-test audio, both
members of conflicting-label training duplicates, and all but the canonical
lexicographically first ID in a same-label training duplicate group. It retains
the released test rows for the official score and records the policy in
`audit.json`.

## Paired noise experiment

Use an appropriately licensed background-noise WAV and record its attribution
in your experiment notes. Choose SNRs before inspecting test results (proposed
controls: 10 dB and 0 dB, each in a separate output directory).

```bash
python code/audio_pipeline.py pairs --manifest results/local/audit/manifest.json --noise data/external/background.wav --output data/processed/noise-10db --snr-db 10 --seed 3108
```

For a smoke example instead of the full dataset, select one or more sources:

```bash
python code/audio_pipeline.py pairs \
  --manifest results/local/audit/manifest.json \
  --noise data/external/esc50/1-137296-A-16.wav \
  --output data/processed/week2-example-10db \
  --snr-db 10 --seed 3108 \
  --audio-id husky_train_00000
```

The source audio hash is checked before processing. A stable per-source seed
selects noise; noise is resampled to source sample rate. RMS mixing achieves the
requested SNR. A common gain applies to reference and mixture to prevent mixture
clipping without changing SNR. This gain is recorded, so `original` is a matched,
possibly attenuated reference rather than a bitwise copy of the source file.

Four conditions preserve source labels and official split:

- `original`: matched reference;
- `noisy`: reference plus noise;
- `enhanced`: spectral subtraction on the noisy version;
- `original_enhanced`: enhancement safety check on the matched reference.

`pairs.json` records provenance, requested and achieved SNR, and condition for
every WAV. `config.json`
records the seed and processing settings. A fresh output directory is required.
Audio and local outputs are ignored by Git.

**This enhancer is an oracle noise-profile signal-processing baseline, not a
trained AI denoiser.** It knows the added noise profile, so it is an optimistic
control, unsuitable for claiming real-world noise suppression. Compare it later
with a documented AI enhancer and evaluate affect classification on all four
conditions. No classification improvement or real-dataset result is claimed.

The synthetic tests verify achieved SNR, determinism, output duration, split
provenance, source hash enforcement, duplicate leakage detection, silent-input
rejection, and Git LFS pointer detection. They do not measure model accuracy.

## Week 3 honest baselines

Install the expanded pinned environment, regenerate the audit, and run the
baseline experiment:

```bash
pip install -r requirements.txt
python code/audio_pipeline.py audit \
  --root data/raw/EmotionalCanines \
  --output results/local/audit
python code/baselines.py \
  --manifest results/local/audit/manifest.json \
  --output results/week3_baselines \
  --cache results/local/week3_features.npz
```

The script creates one fixed 80/20 development split stratified by breed and
the joint arousal/valence label. Because public dog and source IDs are absent,
this validation set is explicitly called clip-level rather than dog-disjoint.
It tunes classical-model regularization and CNN early stopping only on this
validation set. Classical models are then refit on all 1,195 eligible training
clips; each of three CNN seeds is retrained on all eligible clips for its
validation-selected epoch count. The untouched 200-clip official test remains
the only verified unseen-dog evaluation.

Implemented Week 3 models are majority, stratified random, MFCC-summary
logistic regression, MFCC-summary linear SVM, and a small multi-task Log-Mel
CNN. Results include macro F1, balanced accuracy, ordinal MAE, quadratic
weighted kappa, severe-error rate, per-breed breakdowns, and confusion
matrices. The feature cache stays under ignored `results/local/`; small result
tables, protocol metadata, the exact split, and the presentation figure are
written to `results/week3_baselines/`.
