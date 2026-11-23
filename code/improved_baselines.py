"""Select richer acoustic baselines on validation, then test once on unseen dogs.

This is an exploratory iteration after the Week 3 MFCC baseline. Candidate
selection uses only the fixed development validation split. The selected model
is refit on all eligible training clips before the official test is evaluated.
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baselines import (SPLIT_SEED, build_features, classical_model, encode_labels,
                       fixed_split, load_manifest, metric_values, sha256)


LINEAR_C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
TREE_LEAF_GRID = (1, 3, 5, 10)
BOOTSTRAP_SAMPLES = 5000


def richer_features(mfcc, logmel, rows):
    """Pool spectral distribution, dynamics, and four temporal regions."""
    deltas = np.diff(logmel, axis=2, prepend=logmel[:, :, :1])
    parts = [
        logmel.mean(axis=2),
        logmel.std(axis=2),
        np.quantile(logmel, 0.1, axis=2),
        np.quantile(logmel, 0.5, axis=2),
        np.quantile(logmel, 0.9, axis=2),
        deltas.mean(axis=2),
        deltas.std(axis=2),
    ]
    for chunk in np.array_split(logmel, 4, axis=2):
        parts.extend([chunk.mean(axis=2), chunk.std(axis=2)])
    breed = np.asarray([
        [row["breed"] == "husky", row["breed"] == "shiba"] for row in rows
    ], dtype=np.float32)
    return np.concatenate([mfcc, *parts, breed], axis=1).astype(np.float32)


def candidate_models():
    for c_value in LINEAR_C_GRID:
        yield (
            {"family": "linear_svm", "C": c_value},
            make_pipeline(
                StandardScaler(),
                LinearSVC(
                    C=c_value, class_weight="balanced", random_state=SPLIT_SEED,
                    max_iter=20000,
                ),
            ),
        )
    for leaf in TREE_LEAF_GRID:
        yield (
            {"family": "extra_trees", "min_samples_leaf": leaf},
            ExtraTreesClassifier(
                n_estimators=250,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SPLIT_SEED,
                n_jobs=2,
            ),
        )


def make_model(config):
    if config["family"] == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=config["C"], class_weight="balanced", random_state=SPLIT_SEED,
                max_iter=20000,
            ),
        )
    return ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced",
        random_state=SPLIT_SEED,
        n_jobs=2,
    )


def select_model(x_fit, y_fit, x_validation, y_validation):
    trials = []
    for config, model in candidate_models():
        model.fit(x_fit, y_fit)
        score = float(f1_score(y_validation, model.predict(x_validation), average="macro"))
        trials.append({**config, "validation_macro_f1": score})
    selected = max(trials, key=lambda row: (row["validation_macro_f1"],
                                            row["family"] == "linear_svm"))
    config = {key: value for key, value in selected.items() if key != "validation_macro_f1"}
    return config, trials


def sampled_macro_f1(labels, predictions, indices):
    sampled_labels = np.asarray(labels)[indices]
    sampled_predictions = np.asarray(predictions)[indices]
    class_scores = []
    for class_index in (0, 1, 2):
        true_positive = np.sum(
            (sampled_labels == class_index) & (sampled_predictions == class_index), axis=1
        )
        false_positive = np.sum(
            (sampled_labels != class_index) & (sampled_predictions == class_index), axis=1
        )
        false_negative = np.sum(
            (sampled_labels == class_index) & (sampled_predictions != class_index), axis=1
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(np.divide(
            2 * true_positive, denominator,
            out=np.zeros_like(denominator, dtype=float), where=denominator != 0,
        ))
    return np.mean(class_scores, axis=0)


def bootstrap_macro_f1(labels, predictions, seed=SPLIT_SEED, samples=BOOTSTRAP_SAMPLES):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    generator = np.random.default_rng(seed)
    selected = generator.integers(0, len(labels), (samples, len(labels)))
    values = sampled_macro_f1(labels, predictions, selected)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def bootstrap_difference(labels, improved, baseline, seed=SPLIT_SEED,
                         samples=BOOTSTRAP_SAMPLES):
    labels, improved, baseline = map(np.asarray, (labels, improved, baseline))
    generator = np.random.default_rng(seed)
    selected = generator.integers(0, len(labels), (samples, len(labels)))
    differences = (
        sampled_macro_f1(labels, improved, selected)
        - sampled_macro_f1(labels, baseline, selected)
    )
    return [float(value) for value in np.quantile(differences, (0.025, 0.975))]


def evaluation_record(target, rows, labels, predictions, baseline_predictions):
    record = {
        "target": target,
        "n": len(labels),
        **metric_values(labels, predictions),
        "macro_f1_95_ci": bootstrap_macro_f1(labels, predictions),
        "week3_macro_f1": float(f1_score(labels, baseline_predictions, average="macro")),
        "macro_f1_difference": float(
            f1_score(labels, predictions, average="macro")
            - f1_score(labels, baseline_predictions, average="macro")
        ),
        "difference_95_ci": bootstrap_difference(labels, predictions, baseline_predictions),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=(0, 1, 2)).tolist(),
        "per_breed": {},
    }
    for breed in ("husky", "shiba"):
        selected = np.asarray([row["breed"] == breed for row in rows])
        record["per_breed"][breed] = {
            "n": int(selected.sum()),
            **metric_values(labels[selected], predictions[selected]),
        }
    return record


def write_csv(path, rows):
    fields = (
        "target", "validation_macro_f1", "official_macro_f1", "ci_low", "ci_high",
        "week3_macro_f1", "difference", "difference_ci_low", "difference_ci_high",
        "balanced_accuracy", "mae", "quadratic_weighted_kappa", "severe_error_rate",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, destination):
    targets = [row["target"].capitalize() for row in rows]
    baseline = [row["week3_macro_f1"] for row in rows]
    improved = [row["macro_f1"] for row in rows]
    lower = [row["macro_f1"] - row["macro_f1_95_ci"][0] for row in rows]
    upper = [row["macro_f1_95_ci"][1] - row["macro_f1"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    old_bars = axis.bar(x - width / 2, baseline, width, label="Week 3 MFCC SVM", color="#7A8FA6")
    new_bars = axis.bar(x + width / 2, improved, width, yerr=np.asarray([lower, upper]),
                        capsize=4, label="Richer validation-selected model", color="#2A6F97")
    axis.bar_label(old_bars, fmt="%.3f", padding=2, fontsize=9)
    axis.bar_label(new_bars, fmt="%.3f", padding=7, fontsize=9)
    axis.set_xticks(x, targets)
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Official unseen-dog macro F1")
    axis.set_title("Richer acoustic models vs. Week 3 baseline\nError bars: 95% bootstrap interval")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(records, selections):
    lines = [
        "# Richer acoustic baseline iteration",
        "",
        "Candidate models were selected only on the fixed Week 3 validation split,",
        "then refit on all 1,195 eligible training clips and evaluated once on the",
        "200 official unseen-dog clips. This is an exploratory improvement, not a",
        "claim of precise emotion recognition.",
        "",
        "## Official unseen-dog results",
        "",
        "| Target | Selected model | Validation F1 | Official F1 (95% CI) | Week 3 F1 | Difference (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        selected = selections[record["target"]]["selected"]
        name = "Linear SVM" if selected["family"] == "linear_svm" else "ExtraTrees"
        ci = record["macro_f1_95_ci"]
        diff_ci = record["difference_95_ci"]
        lines.append(
            f'| {record["target"].capitalize()} | {name} | '
            f'{selections[record["target"]]["validation_macro_f1"]:.3f} | '
            f'{record["macro_f1"]:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | '
            f'{record["week3_macro_f1"]:.3f} | {record["macro_f1_difference"]:+.3f} '
            f'({diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}) |'
        )
    lines.extend([
        "",
        "![Official comparison](improved_model_comparison.png)",
        "",
        "## What changed",
        "",
        "The representation concatenates the Week 3 MFCC summary with Log-Mel",
        "distribution statistics, temporal deltas, four-region temporal pooling,",
        "and the known Husky/Shiba breed indicator. A small validation-only grid",
        "compares strongly regularized linear SVMs and ExtraTrees models.",
        "",
        "## Boundaries",
        "",
        "- The official test was not used to choose the feature set or hyperparameters.",
        "- Breed is available metadata here; the two-value encoding cannot represent unseen breeds.",
        "- The development split is clip-level because public dog/source IDs are unavailable.",
        "- Bootstrap intervals describe sampling uncertainty on these 200 clips, not all dogs or breeds.",
        "- These are context/behavior annotations, not direct measurements of private emotion.",
        "- A higher point estimate is not automatically a statistically clear improvement; inspect the paired difference interval.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python code/improved_baselines.py \\",
        "  --manifest results/local/audit/manifest.json \\",
        "  --cache results/local/week3_features.npz \\",
        "  --output results/week6_improved_models",
        "```",
        "",
    ])
    return "\n".join(lines)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("output exists; choose a fresh path to preserve experiment history")
    output.mkdir(parents=True)
    eligible, official = load_manifest(args.manifest)
    fit_ids, validation_ids = fixed_split(eligible)
    all_rows = eligible + official
    ids, mfcc, logmel = build_features(all_rows, args.cache, sha256(args.manifest))
    row_by_id = {row["audio_id"]: row for row in all_rows}
    feature_by_id = {audio_id: index for index, audio_id in enumerate(ids)}
    ordered_rows = [row_by_id[audio_id] for audio_id in ids]
    features = richer_features(mfcc, logmel, ordered_rows)

    def rows_for(source_ids):
        return [row_by_id[audio_id] for audio_id in source_ids]

    def matrix(rows):
        return features[[feature_by_id[row["audio_id"]] for row in rows]]

    fit_rows = rows_for(fit_ids)
    validation_rows = rows_for(validation_ids)
    official_rows = sorted(official, key=lambda row: row["audio_id"])
    selections, records, summary_rows = {}, [], []
    for target in ("arousal", "valence"):
        y_fit = encode_labels(fit_rows, target)
        y_validation = encode_labels(validation_rows, target)
        selected, trials = select_model(
            matrix(fit_rows), y_fit, matrix(validation_rows), y_validation
        )
        validation_score = max(row["validation_macro_f1"] for row in trials)
        selections[target] = {
            "selected": selected,
            "validation_macro_f1": validation_score,
            "trials": trials,
        }
        y_all = encode_labels(eligible, target)
        y_official = encode_labels(official_rows, target)
        model = make_model(selected).fit(matrix(eligible), y_all)
        predictions = model.predict(matrix(official_rows))
        week3 = classical_model("MFCC linear SVM", 0.01).fit(
            mfcc[[feature_by_id[row["audio_id"]] for row in eligible]], y_all
        )
        week3_predictions = week3.predict(
            mfcc[[feature_by_id[row["audio_id"]] for row in official_rows]]
        )
        record = evaluation_record(
            target, official_rows, y_official, predictions, week3_predictions
        )
        records.append(record)
        summary_rows.append({
            "target": target,
            "validation_macro_f1": validation_score,
            "official_macro_f1": record["macro_f1"],
            "ci_low": record["macro_f1_95_ci"][0],
            "ci_high": record["macro_f1_95_ci"][1],
            "week3_macro_f1": record["week3_macro_f1"],
            "difference": record["macro_f1_difference"],
            "difference_ci_low": record["difference_95_ci"][0],
            "difference_ci_high": record["difference_95_ci"][1],
            "balanced_accuracy": record["balanced_accuracy"],
            "mae": record["mae"],
            "quadratic_weighted_kappa": record["quadratic_weighted_kappa"],
            "severe_error_rate": record["severe_error_rate"],
        })

    (output / "run.json").write_text(json.dumps({
        "manifest_sha256": sha256(args.manifest),
        "fit_count": len(fit_rows),
        "validation_count": len(validation_rows),
        "eligible_refit_count": len(eligible),
        "official_test_count": len(official_rows),
        "feature_dimension": int(features.shape[1]),
        "selection": selections,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "results": records,
    }, indent=2, allow_nan=False) + "\n")
    write_csv(output / "summary.csv", summary_rows)
    plot_results(records, output / "improved_model_comparison.png")
    (output / "README.md").write_text(markdown_report(records, selections))
    print(json.dumps(summary_rows, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="results/local/week3_features.npz")
    parser.add_argument("--output", default="results/week6_improved_models")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
