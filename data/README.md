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

Before training, the project will add a reproducible audit command that records:

- label-file SHA-256 hashes;
- number of readable clips and missing IDs;
- sampling-rate, channel, duration, and amplitude distributions;
- exact train/test class counts;
- duplicate and near-duplicate checks;
- any decoding failures.

The dataset must not be silently repaired. Exclusions will be written to a manifest with a reason.

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
