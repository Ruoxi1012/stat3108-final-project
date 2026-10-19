# Week 2 data audit and paired-audio checkpoint

Audit date: **2026-10-18**  
Dataset commit: `6b73c5fce55fc132e97db6a76899f8594a955f59`  
Pipeline seed: `3108`

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/audio_pipeline.py audit \
  --root data/raw/EmotionalCanines \
  --output results/local/audit
```

The clean environment used Python 3.11.7, NumPy 2.0.2, and SciPy 1.13.1. All
four tests passed. The audit exits with status 1 because it deliberately treats
the duplicate findings below as unresolved issues.

## Dataset inventory

| Split | Clips | Arousal: Low / Medium / High | Valence: Negative / Neutral / Positive |
|---|---:|---:|---:|
| Husky train | 600 | 154 / 262 / 184 | 138 / 189 / 273 |
| Husky test | 100 | 33 / 34 / 33 | 33 / 33 / 34 |
| Shiba train | 600 | 224 / 171 / 205 | 159 / 218 / 223 |
| Shiba test | 100 | 34 / 33 / 33 | 33 / 34 / 33 |

All 1,400 files decode as mono, 16 kHz WAV audio. Their combined duration is
0.5845 hours. Clip duration ranges from 0.064 to 30.016 seconds, with a median
of 0.960 seconds and a 95th percentile of 4.032 seconds. Seventy-four files
contain samples at full scale. This is an audit finding, not evidence that every
one of those recordings has audible clipping.

Label-file SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `husky_train_labels.csv` | `263b4c486a073a3e5a641e6db488182d11ed48d58a74b561ba12b0653d4a6152` |
| `husky_test_labels.csv` | `5e9dc4f521b94f4c9727fbb69aea8f3cfebfbd3ea92dcef4b68cd99c6c0edc77` |
| `shiba_train_labels.csv` | `579c4e1890a76dd0682b70f65d1a83c2c1a6b49744a5fafaa135014145dd7a75` |
| `shiba_test_labels.csv` | `eb4c938a84d466fea00c88f51406c51b50e2161c56c541947fda5f4b52697c3b` |

## Exact decoded-audio duplicate findings

The audit hashes decoded mono samples together with their sample rate. It found
five exact-duplicate groups:

| IDs | Location | Labels | Finding |
|---|---|---|---|
| `husky_train_00291`, `husky_train_00577` | Husky train | Medium/Positive vs High/Negative | Conflicting labels |
| `husky_train_00457`, `husky_train_00572` | Husky train | High/Negative for both | Repeated training audio |
| `husky_test_00027`, `husky_test_00063` | Husky test | Low/Positive vs Medium/Positive | Conflicting test labels |
| `shiba_train_00238`, `shiba_test_00042` | Shiba train/test | Medium/Negative for both | Exact train/test leakage |
| `shiba_train_00421`, `shiba_test_00039` | Shiba train/test | High/Positive vs Medium/Negative | Leakage and conflicting labels |

### Resolution fixed before modeling

- Remove `shiba_train_00238` and `shiba_train_00421` from every model-fitting
  dataset. Keep the corresponding test examples in the official test score.
- Exclude both members of the conflicting Husky training pair
  (`husky_train_00291`, `husky_train_00577`) from model fitting.
- Keep one canonical member of the same-label Husky training pair and record
  the retained ID in the training manifest.
- Keep both Husky test rows for the official test score because they belong to
  the released test set. Also report a sensitivity result that gives each exact
  duplicate group one total unit of weight.
- Preserve the untouched release and the full audit. Never delete or overwrite
  source files silently.

The audit writes this decision into `manifest.json` as `model_status` and
`model_exclusion_reason`. It marks five training rows for exclusion and leaves
1,195 training rows eligible for model fitting:

| Excluded training ID | Reason |
|---|---|
| `husky_train_00291` | Conflicting labels on identical decoded training audio |
| `husky_train_00577` | Conflicting labels on identical decoded training audio |
| `husky_train_00572` | Same-label duplicate; canonical ID `husky_train_00457` retained |
| `shiba_train_00238` | Exact decoded duplicate appears in the official test set |
| `shiba_train_00421` | Exact decoded duplicate appears in the official test set |

The public labels do not expose dog IDs. The team can verify only the authors'
official dog-disjoint statement, not construct a new certified dog-disjoint
validation split. Near-duplicate and source-video review remain open tasks.

## Paired-audio smoke example

Source: `husky_train_00000`  
Noise: ESC-50 `1-137296-A-16.wav` (`wind`)  
Noise SHA-256: `73e2dd5e66dda29f76a54b53c3c8e33441d105fc35c1d231bfbd8f7a18442969`

ESC-50 identifies the file as derived from “Rain And Wind On Glass” by
Freesound user Benboncan under CC BY. ESC-50 is distributed under CC BY-NC 3.0.
The local audio and license are ignored by Git.

```bash
python code/audio_pipeline.py pairs \
  --manifest results/local/audit/manifest.json \
  --noise data/external/esc50/1-137296-A-16.wav \
  --output data/processed/week2-example-10db \
  --snr-db 10 --seed 3108 \
  --audio-id husky_train_00000

python code/audio_pipeline.py pairs \
  --manifest results/local/audit/manifest.json \
  --noise data/external/esc50/1-137296-A-16.wav \
  --output data/processed/week2-example-0db \
  --snr-db 0 --seed 3108 \
  --audio-id husky_train_00000
```

Both outputs contain four 16 kHz, 16,384-sample conditions: `original`,
`noisy`, `enhanced`, and `original_enhanced`. Measured SNR equals 10.0 dB and
0.0 dB respectively. This verifies generation and provenance only. It does not
show that enhancement improves affect classification.
