"""Evaluate frozen wav2vec 2.0 embeddings without fine-tuning on test labels.

All twelve encoder layers are pooled once and stored in an ignored, resumable
local cache. Layer, pooling, breed-feature use, and linear-head regularization
are selected on the fixed development validation split. Only the frozen winner
is refit on all eligible training clips and evaluated on the official test.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baselines import (SAMPLE_RATE, SPLIT_SEED, classical_model, encode_labels,
                       fixed_split, load_manifest, metric_values, read_audio, sha256)
from improved_baselines import bootstrap_difference, bootstrap_macro_f1


BUNDLE_NAME = "WAV2VEC2_BASE"
MODEL_FILE = "wav2vec2_fairseq_base_ls960.pth"
MAX_SECONDS = 4
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS
LAYERS = (3, 6, 9, 12)
POOLING = ("mean", "mean_std")
BREED_OPTIONS = (False, True)
C_GRID = (0.0001, 0.001, 0.01, 0.1)


def fixed_audio(audio, max_samples=MAX_SAMPLES):
    values = np.asarray(audio, dtype=np.float32)
    if len(values) > max_samples:
        offset = (len(values) - max_samples) // 2
        values = values[offset:offset + max_samples]
    return values


def pool_layers(layers, lengths):
    """Return batch x layer x (mean+std) while excluding padded frames."""
    output = []
    for layer in layers:
        pooled = []
        for index, length in enumerate(lengths.tolist()):
            valid = layer[index, :int(length)]
            pooled.append(torch.cat((valid.mean(0), valid.std(0, unbiased=False))))
        output.append(torch.stack(pooled))
    return torch.stack(output, dim=1).cpu().numpy().astype(np.float32)


def cache_metadata(rows):
    return {
        "schema_version": 1,
        "bundle": BUNDLE_NAME,
        "model_file": MODEL_FILE,
        "sample_rate": SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "pooling": "per-layer valid-frame mean and population standard deviation",
        "audio_ids": [row["audio_id"] for row in rows],
    }


def open_embedding_cache(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    expected = cache_metadata(rows)
    shape = (len(rows), 12, 1536)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != expected:
            raise ValueError("embedding cache metadata mismatch; use a fresh cache directory")
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r+")
        done = np.load(directory / "done.npy", mmap_mode="r+")
        if embeddings.shape != shape or done.shape != (len(rows),):
            raise ValueError("embedding cache shape mismatch")
    else:
        metadata_path.write_text(json.dumps(expected, indent=2) + "\n")
        embeddings = np.lib.format.open_memmap(
            directory / "embeddings.npy", mode="w+", dtype=np.float32, shape=shape
        )
        done = np.lib.format.open_memmap(
            directory / "done.npy", mode="w+", dtype=bool, shape=(len(rows),)
        )
        done[:] = False
        embeddings.flush()
        done.flush()
    return embeddings, done


def extract_embeddings(rows, directory, batch_size=8):
    import torchaudio

    embeddings, done = open_embedding_cache(rows, directory)
    pending = [index for index in range(len(rows)) if not done[index]]
    if not pending:
        return embeddings
    pending.sort(key=lambda index: min(rows[index]["duration_seconds"], MAX_SECONDS))
    model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().eval()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    completed = int(done.sum())
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            audio = [fixed_audio(read_audio(rows[index]["path"])) for index in indices]
            lengths = torch.tensor([len(item) for item in audio], dtype=torch.long)
            padded = torch.zeros((len(audio), int(lengths.max())), dtype=torch.float32)
            for row_index, values in enumerate(audio):
                padded[row_index, :len(values)] = torch.from_numpy(values)
            layers, output_lengths = model.extract_features(padded, lengths)
            embeddings[indices] = pool_layers(layers, output_lengths)
            done[indices] = True
            embeddings.flush()
            done.flush()
            completed += len(indices)
            print(f"Extracted wav2vec embeddings for {completed}/{len(rows)} clips", flush=True)
    return embeddings


def feature_matrix(embeddings, rows, layer, pooling, include_breed):
    values = np.asarray(embeddings[:, layer - 1])
    if pooling == "mean":
        values = values[:, :768]
    if include_breed:
        breed = np.asarray([
            [row["breed"] == "husky", row["breed"] == "shiba"] for row in rows
        ], dtype=np.float32)
        values = np.concatenate((values, breed), axis=1)
    return values


def make_head(c_value):
    return make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c_value, class_weight="balanced", random_state=SPLIT_SEED,
            max_iter=3000, tol=1e-3,
        ),
    )


def select_configuration(embeddings, rows, fit_indices, validation_indices, target):
    y_fit = encode_labels([rows[index] for index in fit_indices], target)
    y_validation = encode_labels([rows[index] for index in validation_indices], target)
    trials = []
    for layer in LAYERS:
        for pooling in POOLING:
            for include_breed in BREED_OPTIONS:
                features = feature_matrix(embeddings, rows, layer, pooling, include_breed)
                for c_value in C_GRID:
                    model = make_head(c_value).fit(features[fit_indices], y_fit)
                    score = float(f1_score(
                        y_validation, model.predict(features[validation_indices]), average="macro"
                    ))
                    trials.append({
                        "layer": layer,
                        "pooling": pooling,
                        "include_breed": include_breed,
                        "C": c_value,
                        "validation_macro_f1": score,
                    })
                    if len(trials) % 16 == 0:
                        print(
                            f"Completed {len(trials)}/64 validation trials for {target}",
                            flush=True,
                        )
    selected = max(
        trials,
        key=lambda row: (
            row["validation_macro_f1"],
            row["pooling"] == "mean",
            not row["include_breed"],
            -row["C"],
            -row["layer"],
        ),
    )
    return selected, trials


def evaluate(target, rows, official_rows, labels, predictions, baseline_predictions):
    record = {
        "target": target,
        "n": len(labels),
        **metric_values(labels, predictions),
        "macro_f1_95_ci": bootstrap_macro_f1(labels, predictions),
        "week3_macro_f1": float(f1_score(labels, baseline_predictions, average="macro")),
        "difference_95_ci": bootstrap_difference(labels, predictions, baseline_predictions),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=(0, 1, 2)).tolist(),
        "per_breed": {},
    }
    record["macro_f1_difference"] = record["macro_f1"] - record["week3_macro_f1"]
    for breed in ("husky", "shiba"):
        selected = np.asarray([row["breed"] == breed for row in official_rows])
        record["per_breed"][breed] = {
            "n": int(selected.sum()),
            **metric_values(labels[selected], predictions[selected]),
        }
    return record


def plot(records, destination):
    x = np.arange(len(records))
    width = 0.34
    baseline = [row["week3_macro_f1"] for row in records]
    pretrained = [row["macro_f1"] for row in records]
    errors = np.asarray([
        [row["macro_f1"] - row["macro_f1_95_ci"][0] for row in records],
        [row["macro_f1_95_ci"][1] - row["macro_f1"] for row in records],
    ])
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    old = axis.bar(x - width / 2, baseline, width, label="Week 3 MFCC SVM", color="#7A8FA6")
    new = axis.bar(x + width / 2, pretrained, width, yerr=errors, capsize=4,
                   label="Frozen wav2vec 2.0 + linear SVM", color="#2A6F97")
    axis.bar_label(old, fmt="%.3f", padding=2)
    axis.bar_label(new, fmt="%.3f", padding=7)
    axis.set_xticks(x, [row["target"].capitalize() for row in records])
    axis.set_ylim(0, 0.75)
    axis.set_ylabel("Official unseen-dog macro F1")
    axis.set_title("Frozen pretrained audio representation\nError bars: 95% bootstrap interval")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown(records, selections):
    lines = [
        "# Frozen wav2vec 2.0 representation experiment",
        "",
        "A 94M-parameter wav2vec 2.0 Base encoder pretrained on LibriSpeech is",
        "kept frozen. Per-layer valid-frame means and standard deviations are",
        "cached locally; only a class-balanced linear SVM is trained. Layer, pooling,",
        "breed metadata, and regularization are selected on development validation.",
        "",
        "## Official unseen-dog result",
        "",
        "| Target | Selected configuration | Validation F1 | Official F1 (95% CI) | Week 3 F1 | Difference (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        selected = selections[record["target"]]["selected"]
        config = (
            f'L{selected["layer"]}, {selected["pooling"]}, '
            f'breed={selected["include_breed"]}, C={selected["C"]}'
        )
        ci, diff = record["macro_f1_95_ci"], record["difference_95_ci"]
        lines.append(
            f'| {record["target"].capitalize()} | {config} | '
            f'{selected["validation_macro_f1"]:.3f} | '
            f'{record["macro_f1"]:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | '
            f'{record["week3_macro_f1"]:.3f} | {record["macro_f1_difference"]:+.3f} '
            f'({diff[0]:+.3f}, {diff[1]:+.3f}) |'
        )
    lines.extend([
        "",
        "![Pretrained comparison](pretrained_comparison.png)",
        "",
        "## Reproduction",
        "",
        "The first run downloads about 360 MB of weights and writes roughly 103 MB",
        "of resumable embedding cache under `results/local/`; neither is committed.",
        "",
        "```bash",
        "python code/pretrained_baselines.py \\",
        "  --manifest results/local/audit/manifest.json \\",
        "  --cache results/local/wav2vec2_base \\",
        "  --output results/week6_pretrained",
        "```",
        "",
        "## Boundaries",
        "",
        "- The encoder is pretrained on human speech, not canine affect.",
        "- Audio is center-cropped to at most four seconds; no fine-tuning is performed.",
        "- Public dog/source IDs are unavailable for dog-grouped development validation.",
        "- This is the planned pretrained-model-family confirmation, not permission to tune repeatedly on official labels.",
        "- The labels describe observed behavior/context, not a dog's private emotional state.",
        "",
    ])
    return "\n".join(lines)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output exists; choose a fresh path to preserve experiment history")
    eligible, official = load_manifest(args.manifest)
    rows = eligible + sorted(official, key=lambda row: row["audio_id"])
    embeddings = extract_embeddings(rows, args.cache, args.batch_size)
    index = {row["audio_id"]: position for position, row in enumerate(rows)}
    fit_ids, validation_ids = fixed_split(eligible)
    fit_indices = np.asarray([index[audio_id] for audio_id in fit_ids])
    validation_indices = np.asarray([index[audio_id] for audio_id in validation_ids])
    eligible_indices = np.asarray([index[row["audio_id"]] for row in eligible])
    official_rows = rows[len(eligible):]
    official_indices = np.arange(len(eligible), len(rows))
    selections, records = {}, []
    for target in ("arousal", "valence"):
        selected, trials = select_configuration(
            embeddings, rows, fit_indices, validation_indices, target
        )
        selections[target] = {"selected": selected, "trials": trials}
        features = feature_matrix(
            embeddings, rows, selected["layer"], selected["pooling"],
            selected["include_breed"]
        )
        y_eligible = encode_labels(eligible, target)
        y_official = encode_labels(official_rows, target)
        model = make_head(selected["C"]).fit(features[eligible_indices], y_eligible)
        predictions = model.predict(features[official_indices])

        baseline_features = np.load(args.week3_cache, allow_pickle=False)
        baseline_ids = list(baseline_features["audio_ids"])
        baseline_index = {audio_id: position for position, audio_id in enumerate(baseline_ids)}
        mfcc = baseline_features["mfcc"]
        baseline = classical_model("MFCC linear SVM", 0.01).fit(
            mfcc[[baseline_index[row["audio_id"]] for row in eligible]], y_eligible
        )
        baseline_predictions = baseline.predict(
            mfcc[[baseline_index[row["audio_id"]] for row in official_rows]]
        )
        records.append(evaluate(
            target, rows, official_rows, y_official, predictions, baseline_predictions
        ))

    output.mkdir(parents=True)
    (output / "run.json").write_text(json.dumps({
        "manifest_sha256": sha256(args.manifest),
        "bundle": BUNDLE_NAME,
        "model_file": MODEL_FILE,
        "encoder_parameters": 94370944,
        "max_seconds": MAX_SECONDS,
        "selection": selections,
        "results": records,
    }, indent=2, allow_nan=False) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "target", "validation_macro_f1", "official_macro_f1", "ci_low", "ci_high",
            "week3_macro_f1", "difference", "difference_ci_low", "difference_ci_high",
        ), lineterminator="\n")
        writer.writeheader()
        for record in records:
            selected = selections[record["target"]]["selected"]
            writer.writerow({
                "target": record["target"],
                "validation_macro_f1": selected["validation_macro_f1"],
                "official_macro_f1": record["macro_f1"],
                "ci_low": record["macro_f1_95_ci"][0],
                "ci_high": record["macro_f1_95_ci"][1],
                "week3_macro_f1": record["week3_macro_f1"],
                "difference": record["macro_f1_difference"],
                "difference_ci_low": record["difference_95_ci"][0],
                "difference_ci_high": record["difference_95_ci"][1],
            })
    plot(records, output / "pretrained_comparison.png")
    (output / "README.md").write_text(markdown(records, selections))
    print(json.dumps(records, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="results/local/wav2vec2_base")
    parser.add_argument("--week3-cache", default="results/local/week3_features.npz")
    parser.add_argument("--output", default="results/week6_pretrained")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
