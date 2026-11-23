"""Evaluate the frozen AVES-bio animal-vocalization representation.

Official AVES TorchAudio weights are loaded directly from their published model
configuration, avoiding optional ONNX/fairseq dependencies. Embeddings are
computed one clip at a time because the AVES authors warn that waveform padding
can change valid-frame embeddings. Model-head selection remains validation-only.
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
from torchaudio.models import wav2vec2_model

from baselines import (SPLIT_SEED, classical_model, encode_labels, fixed_split,
                       load_manifest, read_audio, sha256)
from pretrained_baselines import (evaluate, feature_matrix, fixed_audio,
                                  pool_layers, select_configuration)


MODEL_NAME = "AVES-base-bio"
MODEL_URL = (
    "https://storage.googleapis.com/esp-public-files/ported_aves/"
    "aves-base-bio.torchaudio.pt"
)
CONFIG_URL = (
    "https://storage.googleapis.com/esp-public-files/ported_aves/"
    "aves-base-bio.torchaudio.model_config.json"
)


def load_aves(config_path, model_path):
    config = json.loads(Path(config_path).read_text())
    model = wav2vec2_model(**config, aux_num_out=None)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.eval()


def cache_metadata(rows, config_path, model_path):
    return {
        "schema_version": 1,
        "model": MODEL_NAME,
        "model_sha256": sha256(model_path),
        "config_sha256": sha256(config_path),
        "max_seconds": 4,
        "batch_size": 1,
        "pooling": "per-layer valid-frame mean and population standard deviation",
        "audio_ids": [row["audio_id"] for row in rows],
    }


def open_cache(rows, directory, config_path, model_path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    expected = cache_metadata(rows, config_path, model_path)
    shape = (len(rows), 12, 1536)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != expected:
            raise ValueError("AVES cache metadata mismatch; use a fresh directory")
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r+")
        done = np.load(directory / "done.npy", mmap_mode="r+")
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
    if embeddings.shape != shape or done.shape != (len(rows),):
        raise ValueError("AVES cache shape mismatch")
    return embeddings, done


def extract_embeddings(rows, directory, config_path, model_path):
    embeddings, done = open_cache(rows, directory, config_path, model_path)
    pending = [index for index in range(len(rows)) if not done[index]]
    if not pending:
        return embeddings
    pending.sort(key=lambda index: min(rows[index]["duration_seconds"], 4))
    model = load_aves(config_path, model_path)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    completed = int(done.sum())
    with torch.inference_mode():
        for index in pending:
            audio = torch.from_numpy(fixed_audio(read_audio(rows[index]["path"]))).unsqueeze(0)
            layers, _ = model.extract_features(audio)
            length = torch.tensor([layers[0].shape[1]])
            embeddings[index:index + 1] = pool_layers(layers, length)
            done[index] = True
            completed += 1
            if completed % 10 == 0 or completed == len(rows):
                embeddings.flush()
                done.flush()
                print(f"Extracted AVES-bio embeddings for {completed}/{len(rows)} clips", flush=True)
    return embeddings


def plot(records, destination):
    x = np.arange(len(records))
    width = 0.34
    baseline = [row["week3_macro_f1"] for row in records]
    aves = [row["macro_f1"] for row in records]
    errors = np.asarray([
        [row["macro_f1"] - row["macro_f1_95_ci"][0] for row in records],
        [row["macro_f1_95_ci"][1] - row["macro_f1"] for row in records],
    ])
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    old = axis.bar(x - width / 2, baseline, width, label="Week 3 MFCC SVM", color="#7A8FA6")
    new = axis.bar(x + width / 2, aves, width, yerr=errors, capsize=4,
                   label="Frozen AVES-bio + linear SVM", color="#3A7D44")
    axis.bar_label(old, fmt="%.3f", padding=2)
    axis.bar_label(new, fmt="%.3f", padding=7)
    axis.set_xticks(x, [row["target"].capitalize() for row in records])
    axis.set_ylim(0, 0.75)
    axis.set_ylabel("Official unseen-dog macro F1")
    axis.set_title("Animal-vocalization pretrained representation\nError bars: 95% bootstrap interval")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown(records, selections):
    lines = [
        "# Frozen AVES-bio representation experiment",
        "",
        "AVES-bio is a HuBERT-style encoder pretrained on 360 hours including animal",
        "vocalizations. Official TorchAudio weights are frozen, and embeddings are",
        "computed one clip at a time to avoid the authors' documented padding effect.",
        "Only a class-balanced linear SVM is fitted on EmotionalCanines.",
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
        "![AVES comparison](aves_comparison.png)",
        "",
        "## Reproduction",
        "",
        "Download the official MIT-licensed AVES-bio TorchAudio model and config:",
        "",
        "```bash",
        f"curl -L --output results/local/models/aves/aves-base-bio.torchaudio.pt {MODEL_URL}",
        f"curl -L --output results/local/models/aves/aves-base-bio.torchaudio.model_config.json {CONFIG_URL}",
        "python code/bioacoustic_baselines.py \\",
        "  --manifest results/local/audit/manifest.json \\",
        "  --model results/local/models/aves/aves-base-bio.torchaudio.pt \\",
        "  --config results/local/models/aves/aves-base-bio.torchaudio.model_config.json \\",
        "  --cache results/local/aves_base_bio \\",
        "  --output results/week6_aves_bio",
        "```",
        "",
        "## Boundaries",
        "",
        "- Audio is center-cropped to at most four seconds; the encoder is not fine-tuned.",
        "- AVES is general bioacoustic pretraining, not canine-affect supervision.",
        "- Public dog/source IDs remain unavailable for grouped development validation.",
        "- Official labels are used only for final reporting, not configuration selection.",
        "- These behavior/context labels do not reveal a dog's private emotional state.",
        "",
    ])
    return "\n".join(lines)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output exists; choose a fresh path to preserve experiment history")
    eligible, official = load_manifest(args.manifest)
    official_rows = sorted(official, key=lambda row: row["audio_id"])
    rows = eligible + official_rows
    embeddings = extract_embeddings(rows, args.cache, args.config, args.model)
    index = {row["audio_id"]: position for position, row in enumerate(rows)}
    fit_ids, validation_ids = fixed_split(eligible)
    fit_indices = np.asarray([index[audio_id] for audio_id in fit_ids])
    validation_indices = np.asarray([index[audio_id] for audio_id in validation_ids])
    eligible_indices = np.arange(len(eligible))
    official_indices = np.arange(len(eligible), len(rows))

    baseline_cache = np.load(args.week3_cache, allow_pickle=False)
    baseline_ids = list(baseline_cache["audio_ids"])
    baseline_index = {audio_id: position for position, audio_id in enumerate(baseline_ids)}
    mfcc = baseline_cache["mfcc"]
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
        from pretrained_baselines import make_head
        y_eligible = encode_labels(eligible, target)
        y_official = encode_labels(official_rows, target)
        model = make_head(selected["C"]).fit(features[eligible_indices], y_eligible)
        predictions = model.predict(features[official_indices])
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
        "model": MODEL_NAME,
        "model_url": MODEL_URL,
        "config_url": CONFIG_URL,
        "model_sha256": sha256(args.model),
        "config_sha256": sha256(args.config),
        "encoder_parameters": 94370944,
        "batch_size": 1,
        "selection": selections,
        "results": records,
    }, indent=2, allow_nan=False) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        fields = (
            "target", "validation_macro_f1", "official_macro_f1", "ci_low", "ci_high",
            "week3_macro_f1", "difference", "difference_ci_low", "difference_ci_high",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
    plot(records, output / "aves_comparison.png")
    (output / "README.md").write_text(markdown(records, selections))
    print(json.dumps(records, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default="results/local/aves_base_bio")
    parser.add_argument("--week3-cache", default="results/local/week3_features.npz")
    parser.add_argument("--output", default="results/week6_aves_bio")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
