# Data

## Primary dataset: EmotionalCanines

- Paper: [EmotionalCanines: A Dataset for Analysis of Arousal and Valence in Dog Vocalization](https://doi.org/10.1145/3746027.3758286)
- Repository: [tmdang1101/EmotionalCanines](https://github.com/tmdang1101/EmotionalCanines)
- License: CC BY-NC-SA 4.0
- Contents: 1,400 `.wav` clips, 700 Husky and 700 Shiba Inu; 1,200 train and 200 fixed test clips.
- Labels: arousal (`Low`, `Medium`, `High`) and valence (`Negative`, `Neutral`, `Positive`).
- Split guarantee from the paper: the dogs in the fixed test sets do not occur in the corresponding training sets.

The public CSV files contain `audio_id`, `arousal`, and `valence`, but not source-video or dog identity. Preserve the authors' official split. Do not claim that a new random validation split is dog-disjoint.

## Retrieval

The audio is stored with Git LFS. Do not add a copy to this repository.

```bash
git lfs install
git clone https://github.com/tmdang1101/EmotionalCanines.git data/raw/EmotionalCanines
```

The implemented audit command records:

- label-file SHA-256 hashes;
- number of readable clips and missing IDs;
- sampling-rate, channel, duration, and amplitude distributions;
- exact train/test class counts;
- exact decoded-audio duplicates;
- any decoding failures.

Run it with:

```bash
python code/audio_pipeline.py audit \
  --root data/raw/EmotionalCanines \
  --output results/local/audit
```

The 2026-10-18 audit decoded all 1,400 clips and found five exact-duplicate
groups. Two groups cross the official Shiba train/test boundary, three groups
contain conflicting labels, and one group has both problems. The nonzero command exit is intentional:
it makes these findings impossible to miss. The checked-in summary at
`results/week2_data_audit.md` records the exact IDs and the resolution policy.
Near-duplicate and source-video checks remain future work.

The dataset must not be silently repaired. Local manifests and any exclusions
must preserve the original IDs and state the reason. The generated manifest
applies the fixed policy through `model_status`; 1,195 of the 1,200 released
training rows remain eligible for fitting, while all 200 official test rows are
retained for evaluation.

## Background-noise source for the Week 2 smoke example

The local example uses ESC-50 clip `1-137296-A-16.wav`, category `wind`, derived
from “Rain And Wind On Glass” by Freesound user Benboncan under CC BY. ESC-50 is
distributed under CC BY-NC 3.0. The clip and license stay under ignored
`data/external/`; only provenance and hashes are committed.

## Data governance

- Do not commit raw audio or copies of online source videos.
- Preserve attribution and the noncommercial/share-alike license.
- Do not collect private recordings for the minimum viable project.
- Do not infer owner identity or publish personally identifying background speech.
- Do not treat model predictions as veterinary advice.

## Candidate pretrained resources

- [Dog2vec model/code/data index](https://github.com/fispresent/dog2vec) and [paper](https://doi.org/10.21437/Interspeech.2025-1287). Note: the repository did not expose a software license when checked on 2026-10-11; use for course research only unless licensing is clarified.
- [AVES animal vocalization encoder](https://github.com/earthspecies/aves).
- [BEATs official implementation](https://github.com/microsoft/unilm/tree/master/beats).

Model checkpoints belong in an ignored local cache, not in Git.
