"""Run the final development-only comparison and one frozen official confirmation.

This script records the last accuracy-oriented iteration. Candidate ranking uses
five-fold development CV over eligible training clips. The official unseen-dog
set is evaluated only after the configurations are frozen.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from baselines import encode_labels, load_manifest, metric_values, sha256
from improved_baselines import (bootstrap_macro_f1, richer_features)
from pretrained_baselines import feature_matrix, make_head


CV_FOLDS = 5
SEED = 3108
VALENCE_WINNER = {"family": "rbf_svm", "C": 3.0, "gamma": 0.0003}
AROUSAL_WINNER = {
    "family": "linear_svm", "layer": 6, "pooling": "mean_std",
    "include_breed": True, "C": 0.0001,
}


def load_features(manifest, acoustic_cache, aves_cache):
    eligible, official = load_manifest(manifest)
    official = sorted(official, key=lambda row: row["audio_id"])
    rows = eligible + official
    ids = [row["audio_id"] for row in rows]

    acoustic = np.load(acoustic_cache, allow_pickle=False)
    acoustic_index = {str(value): index for index, value in enumerate(acoustic["audio_ids"].tolist())}
    acoustic_positions = [acoustic_index[audio_id] for audio_id in ids]
    acoustic_features = richer_features(
        acoustic["mfcc"][acoustic_positions], acoustic["logmel"][acoustic_positions], rows
    )

    metadata = json.loads((Path(aves_cache) / "metadata.json").read_text())
    aves_index = {audio_id: index for index, audio_id in enumerate(metadata["audio_ids"])}
    embeddings = np.load(Path(aves_cache) / "embeddings.npy", mmap_mode="r")
    ordered_embeddings = embeddings[[aves_index[audio_id] for audio_id in ids]]
    aves_l3 = feature_matrix(ordered_embeddings, rows, 3, "mean_std", True)
    aves_l6 = feature_matrix(ordered_embeddings, rows, 6, "mean_std", True)
    return eligible, official, {
        "acoustic": acoustic_features,
        "aves_l3": aves_l3,
        "aves_l6": aves_l6,
        "fusion": np.concatenate((aves_l6, acoustic_features), axis=1),
    }


def valence_candidates():
    return (
        ("acoustic_extra_trees", "acoustic", lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=4,
        )),
        ("aves_l3_linear_svm", "aves_l3", lambda: make_pipeline(
            StandardScaler(), LinearSVC(
                C=0.0003, class_weight="balanced", random_state=SEED, max_iter=10000,
            ),
        )),
        ("aves_acoustic_extra_trees", "fusion", lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=4,
        )),
        ("aves_l3_rbf_svm", "aves_l3", lambda: make_pipeline(
            StandardScaler(), SVC(
                C=VALENCE_WINNER["C"], gamma=VALENCE_WINNER["gamma"], kernel="rbf",
                class_weight="balanced", random_state=SEED,
            ),
        )),
    )


def development_comparison(eligible, features):
    labels = encode_labels(eligible, "valence")
    strata = np.asarray([
        f'{row["breed"]}|{row["arousal"]}|{row["valence"]}' for row in eligible
    ])
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    records = []
    for name, feature_name, factory in valence_candidates():
        scores = []
        oof = np.zeros(len(labels), dtype=int)
        for train, validation in folds.split(np.zeros(len(labels)), strata):
            model = factory().fit(features[feature_name][train], labels[train])
            oof[validation] = model.predict(features[feature_name][validation])
            scores.append(float(f1_score(labels[validation], oof[validation], average="macro")))
        records.append({
            "name": name,
            "feature": feature_name,
            "fold_macro_f1": scores,
            "mean_macro_f1": float(np.mean(scores)),
            "sd_macro_f1": float(np.std(scores)),
            "oof_macro_f1": float(f1_score(labels, oof, average="macro")),
            "oof_accuracy": float(accuracy_score(labels, oof)),
        })
    winner = max(records, key=lambda row: row["mean_macro_f1"])
    if winner["name"] != "aves_l3_rbf_svm":
        raise ValueError(f"predeclared final candidate was not the CV winner: {winner['name']}")
    return records


def valence_model():
    return make_pipeline(
        StandardScaler(), SVC(
            C=VALENCE_WINNER["C"], gamma=VALENCE_WINNER["gamma"], kernel="rbf",
            class_weight="balanced", random_state=SEED,
        ),
    )


def evaluate(target, labels, predictions):
    return {
        "target": target,
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        **metric_values(labels, predictions),
        "macro_f1_95_ci": bootstrap_macro_f1(labels, predictions),
    }


def plot(records, path):
    x = np.arange(2)
    values = [record["macro_f1"] for record in records]
    errors = np.asarray([
        [record["macro_f1"] - record["macro_f1_95_ci"][0] for record in records],
        [record["macro_f1_95_ci"][1] - record["macro_f1"] for record in records],
    ])
    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(x, values, yerr=errors, capsize=5, color=("#3A7D44", "#8A5A44"))
    axis.bar_label(bars, fmt="%.3f", padding=7)
    axis.set_xticks(x, ("Arousal\nAVES linear", "Valence\nAVES RBF"))
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Official unseen-dog macro F1")
    axis.set_title("Frozen final models (95% bootstrap intervals)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output exists; use a fresh directory to preserve experiment history")
    eligible, official, features = load_features(args.manifest, args.acoustic_cache, args.aves_cache)
    development = development_comparison(eligible, {
        name: values[:len(eligible)] for name, values in features.items()
    })
    train = np.arange(len(eligible))
    test = np.arange(len(eligible), len(eligible) + len(official))

    arousal = make_head(AROUSAL_WINNER["C"]).fit(
        features["aves_l6"][train], encode_labels(eligible, "arousal")
    )
    valence = valence_model().fit(
        features["aves_l3"][train], encode_labels(eligible, "valence")
    )
    records = [
        evaluate("arousal", encode_labels(official, "arousal"),
                 arousal.predict(features["aves_l6"][test])),
        evaluate("valence", encode_labels(official, "valence"),
                 valence.predict(features["aves_l3"][test])),
    ]
    output.mkdir(parents=True)
    run_record = {
        "manifest_sha256": sha256(args.manifest),
        "selection_protocol": "five-fold development CV; official confirmation after freeze",
        "cv_folds": CV_FOLDS,
        "seed": SEED,
        "development_valence_candidates": development,
        "selected": {"arousal": AROUSAL_WINNER, "valence": VALENCE_WINNER},
        "official_results": records,
    }
    (output / "run.json").write_text(json.dumps(run_record, indent=2, allow_nan=False) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        fields = ("target", "accuracy", "macro_f1", "ci_low", "ci_high",
                  "balanced_accuracy", "mae", "quadratic_weighted_kappa", "severe_error_rate")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                **{key: record[key] for key in fields if key in record},
                "ci_low": record["macro_f1_95_ci"][0],
                "ci_high": record["macro_f1_95_ci"][1],
            })
    plot(records, output / "final_model_results.png")
    print(json.dumps(run_record, indent=2, allow_nan=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--acoustic-cache", default="results/local/week3_features.npz")
    parser.add_argument("--aves-cache", default="results/local/aves_base_bio")
    parser.add_argument("--output", default="results/final_selection")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
