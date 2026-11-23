"""Train and run the frozen target-specific final canine-affect model.

The final system deliberately uses different validation-selected representations
for the two targets: AVES-bio for arousal and richer acoustic features for
valence. Official-test labels are never read by this script during training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from baselines import (TARGET_LABELS, encode_labels, fixed_log_mel, load_manifest,
                       mfcc_summary, read_audio, sha256)
from bioacoustic_baselines import load_aves
from improved_baselines import make_model, richer_features
from pretrained_baselines import feature_matrix, fixed_audio, make_head, pool_layers


SCHEMA_VERSION = 1
EXPECTED_AROUSAL = {
    "layer": 6,
    "pooling": "mean_std",
    "include_breed": True,
    "C": 0.0001,
}
EXPECTED_VALENCE = {"family": "extra_trees", "min_samples_leaf": 3}


def checked_breed(breed):
    value = str(breed).lower()
    if value not in ("husky", "shiba"):
        raise ValueError("breed must be husky or shiba; unseen breeds are outside the model scope")
    return value


def acoustic_feature(audio, breed):
    """Return the exact 804-dimensional valence representation for one clip."""
    row = {"breed": checked_breed(breed)}
    mfcc = mfcc_summary(audio)[None, :]
    logmel = fixed_log_mel(audio)[None, :, :]
    return richer_features(mfcc, logmel, [row])


def aves_feature(audio, breed, encoder):
    """Return the selected AVES layer-6 mean+std representation for one clip."""
    waveform = torch.from_numpy(fixed_audio(audio)).to(torch.float32).unsqueeze(0)
    with torch.inference_mode():
        layers, _ = encoder.extract_features(waveform)
        pooled = pool_layers(layers, torch.tensor([layers[0].shape[1]]))
    return feature_matrix(
        pooled, [{"breed": checked_breed(breed)}],
        EXPECTED_AROUSAL["layer"], EXPECTED_AROUSAL["pooling"],
        EXPECTED_AROUSAL["include_breed"],
    )


def score_distribution(model, features):
    """Return display scores; these are explicitly not calibrated probabilities."""
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(features), dtype=float)
        ordered = np.zeros((len(features), 3), dtype=float)
        ordered[:, np.asarray(model.classes_, dtype=int)] = values
        return ordered
    scores = np.asarray(model.decision_function(features), dtype=float)
    scores -= scores.max(axis=1, keepdims=True)
    values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def _load_cache_ids(cache):
    return [str(value) for value in cache["audio_ids"].tolist()]


def _selection(path, target):
    data = json.loads(Path(path).read_text())
    selected = dict(data["selection"][target]["selected"])
    selected.pop("validation_macro_f1", None)
    return data, selected


def train_bundle(args):
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("output exists; choose a fresh path to preserve model provenance")

    eligible, _ = load_manifest(args.manifest)
    row_by_id = {row["audio_id"]: row for row in eligible}
    train_ids = [row["audio_id"] for row in eligible]

    aves_run, arousal_config = _selection(args.aves_run, "arousal")
    improved_run, valence_config = _selection(args.improved_run, "valence")
    if arousal_config != EXPECTED_AROUSAL:
        raise ValueError(f"unexpected frozen arousal configuration: {arousal_config}")
    if valence_config != EXPECTED_VALENCE:
        raise ValueError(f"unexpected frozen valence configuration: {valence_config}")
    manifest_hash = sha256(args.manifest)
    if aves_run["manifest_sha256"] != manifest_hash or improved_run["manifest_sha256"] != manifest_hash:
        raise ValueError("experiment records do not match the supplied manifest")

    aves_directory = Path(args.aves_cache)
    aves_metadata = json.loads((aves_directory / "metadata.json").read_text())
    aves_ids = [str(value) for value in aves_metadata["audio_ids"]]
    aves_index = {audio_id: index for index, audio_id in enumerate(aves_ids)}
    if not set(train_ids).issubset(aves_index):
        raise ValueError("AVES cache is missing eligible training clips")
    aves_embeddings = np.load(aves_directory / "embeddings.npy", mmap_mode="r")
    ordered_rows = [row_by_id[audio_id] for audio_id in train_ids]
    selected_embeddings = aves_embeddings[[aves_index[audio_id] for audio_id in train_ids]]
    arousal_x = feature_matrix(
        selected_embeddings, ordered_rows, arousal_config["layer"],
        arousal_config["pooling"], arousal_config["include_breed"],
    )
    arousal_model = make_head(arousal_config["C"]).fit(
        arousal_x, encode_labels(ordered_rows, "arousal")
    )

    acoustic_cache = np.load(args.acoustic_cache, allow_pickle=False)
    acoustic_ids = _load_cache_ids(acoustic_cache)
    acoustic_index = {audio_id: index for index, audio_id in enumerate(acoustic_ids)}
    if not set(train_ids).issubset(acoustic_index):
        raise ValueError("acoustic cache is missing eligible training clips")
    positions = [acoustic_index[audio_id] for audio_id in train_ids]
    valence_x = richer_features(
        acoustic_cache["mfcc"][positions], acoustic_cache["logmel"][positions], ordered_rows
    )
    valence_model = make_model(valence_config).fit(
        valence_x, encode_labels(ordered_rows, "valence")
    )

    identity = hashlib.sha256(json.dumps({
        "manifest": manifest_hash,
        "arousal": arousal_config,
        "valence": valence_config,
        "train_ids": train_ids,
    }, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_id": f"final-target-specific-{identity}",
        "manifest_sha256": manifest_hash,
        "train_count": len(train_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "targets": {
            "arousal": {"labels": TARGET_LABELS["arousal"], "config": arousal_config,
                        "model": arousal_model},
            "valence": {"labels": TARGET_LABELS["valence"], "config": valence_config,
                        "model": valence_model},
        },
        "encoder": {
            "name": aves_run["model"],
            "model_sha256": aves_run["model_sha256"],
            "config_sha256": aves_run["config_sha256"],
        },
        "limitations": [
            "Only Husky and Shiba breed metadata are supported.",
            "Labels are context/behavior annotations, not private emotional states.",
            "Displayed score distributions are not calibrated reliability guarantees.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return {key: bundle[key] for key in ("schema_version", "model_id", "train_count")}


def predict_one(args):
    bundle = joblib.load(args.bundle)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final-model bundle schema")
    if sha256(args.model) != bundle["encoder"]["model_sha256"]:
        raise ValueError("AVES model checksum does not match the final bundle")
    if sha256(args.config) != bundle["encoder"]["config_sha256"]:
        raise ValueError("AVES config checksum does not match the final bundle")
    audio = read_audio(args.audio)
    breed = checked_breed(args.breed)
    encoder = load_aves(args.config, args.model)
    features = {
        "arousal": aves_feature(audio, breed, encoder),
        "valence": acoustic_feature(audio, breed),
    }
    predictions = {}
    for target in ("arousal", "valence"):
        record = bundle["targets"][target]
        model = record["model"]
        distribution = score_distribution(model, features[target])[0]
        index = int(model.predict(features[target])[0])
        predictions[target] = {
            "prediction": record["labels"][index],
            "score_distribution": dict(zip(record["labels"], distribution.tolist())),
            "status": "research_output_only",
        }
    output = {
        "model_id": bundle["model_id"],
        "audio": str(Path(args.audio)),
        "breed": breed,
        "predictions": predictions,
        "warning": (
            "Scores are uncalibrated and are not a reliability or veterinary claim; "
            "the strict validated confidence policy abstained on all official-test clips."
        ),
    }
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise ValueError("output exists; choose a fresh prediction path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit both frozen final heads and export a local bundle")
    train.add_argument("--manifest", required=True)
    train.add_argument("--aves-cache", default="results/local/aves_base_bio")
    train.add_argument("--acoustic-cache", default="results/local/week3_features.npz")
    train.add_argument("--aves-run", default="results/week6_aves_bio/run.json")
    train.add_argument("--improved-run", default="results/week6_improved_models/run.json")
    train.add_argument("--output", default="results/local/final_model.joblib")
    predict = commands.add_parser("predict", help="run the final two-head model on one 16 kHz WAV")
    predict.add_argument("--bundle", default="results/local/final_model.joblib")
    predict.add_argument("--model", required=True, help="local official AVES-bio .pt weights")
    predict.add_argument("--config", required=True, help="local official AVES-bio config JSON")
    predict.add_argument("--audio", required=True)
    predict.add_argument("--breed", required=True, choices=("husky", "shiba"))
    predict.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train_bundle(arguments) if arguments.command == "train" else predict_one(arguments)
    print(json.dumps(result, indent=2, allow_nan=False))
