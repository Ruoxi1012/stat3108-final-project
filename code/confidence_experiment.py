"""Run the real-data Week 6 confidence experiment with one frozen MFCC SVM.

The Week 3 development split is retained.  The 956 fit clips train the model;
the 239 validation clips are deterministically divided into disjoint temperature-
calibration and threshold-selection subsets.  The same frozen model is then
evaluated on the 200 official unseen-dog clips.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from baselines import (FEATURE_VERSION, SPLIT_SEED, build_features, classical_model,
                       encode_labels, fixed_split, load_manifest, sha256)
from confidence import reliability_metrics, risk_coverage
from confidence_cli import apply_policy, fit_policy


CLASS_NAMES = {
    "arousal": ["Low", "Medium", "High"],
    "valence": ["Negative", "Neutral", "Positive"],
}
MODEL_KIND = "MFCC linear SVM"
MODEL_C = 0.01


def split_validation(rows, seed=SPLIT_SEED):
    """Split Week 3 validation clips into balanced, disjoint policy subsets."""
    ids = np.asarray([row["audio_id"] for row in rows])
    strata = np.asarray([
        f'{row["breed"]}|{row["arousal"]}|{row["valence"]}' for row in rows
    ])
    calibration, selection = train_test_split(
        ids, test_size=0.5, random_state=seed, stratify=strata
    )
    return sorted(calibration.tolist()), sorted(selection.tolist())


def model_identity(target, fit_ids, manifest_hash, model=None):
    record = {
        "model": MODEL_KIND,
        "C": MODEL_C,
        "target": target,
        "fit_ids": sorted(fit_ids),
        "manifest_sha256": manifest_hash,
        "feature_version": FEATURE_VERSION,
        "split_seed": SPLIT_SEED,
    }
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode())
    if model is not None:
        classifier = model.named_steps["linearsvc"]
        identity.update(np.ascontiguousarray(classifier.coef_).tobytes())
        identity.update(np.ascontiguousarray(classifier.intercept_).tobytes())
    digest = identity.hexdigest()[:16]
    return f"week6-mfcc-linear-svm-{target}-{digest}"


def score_export(model, model_id, target, rows, features, fit_ids, split, labels=True):
    output = {
        "schema_version": 1,
        "model_id": model_id,
        "target": target,
        "class_names": CLASS_NAMES[target],
        "split": split,
        "fit_ids": list(fit_ids),
        "source_ids": [row["audio_id"] for row in rows],
        # LinearSVC decision_function values are uncalibrated class scores.  The
        # confidence schema calls this field logits so it can share one interface.
        "logits": model.decision_function(features).tolist(),
    }
    if labels:
        output["labels"] = encode_labels(rows, target).tolist()
    return output


def policy_metrics(scores, labels, policy, predictions):
    before = reliability_metrics(scores, labels)
    after = reliability_metrics(scores, labels, policy["temperature"])
    reliable = np.asarray([row["status"] == "reliable" for row in predictions])
    predicted = np.asarray([
        CLASS_NAMES[policy["target"]].index(row["top_class"]) for row in predictions
    ])
    labels = np.asarray(labels)
    retained = int(reliable.sum())
    return {
        "target": policy["target"],
        "temperature": policy["temperature"],
        "threshold": policy["policy"]["threshold"],
        "official_n": int(len(labels)),
        "official_macro_f1": float(f1_score(labels, predicted, average="macro")),
        "nll_before": before["nll"],
        "nll_after": after["nll"],
        "brier_before": before["brier"],
        "brier_after": after["brier"],
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "retained": retained,
        "coverage": float(reliable.mean()),
        "retained_accuracy": float((predicted[reliable] == labels[reliable]).mean())
        if retained else None,
        "uncertain": int((~reliable).sum()),
        "selection_coverage": policy["policy"]["coverage"],
        "selection_error_rate": policy["policy"]["error_rate"],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_summary_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, official_curves, destination):
    targets = [row["target"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    positions = np.arange(len(rows))
    width = 0.32
    before_bars = axes[0].bar(
        positions - width / 2, [row["nll_before"] for row in rows], width,
        label="Before calibration", color="#7A8FA6"
    )
    after_bars = axes[0].bar(
        positions + width / 2, [row["nll_after"] for row in rows], width,
        label="After calibration", color="#2A6F97"
    )
    axes[0].bar_label(before_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].bar_label(after_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].set_xticks(positions, [target.capitalize() for target in targets])
    axes[0].set_ylabel("Official-test negative log-likelihood")
    axes[0].set_title("Calibration on unseen dogs")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    colors = {"arousal": "#2A6F97", "valence": "#B23A48"}
    for target in targets:
        curve = official_curves[target]
        axes[1].plot([point["coverage"] for point in curve],
                     [1 - point["error_rate"] for point in curve],
                     label=target.capitalize(), color=colors[target])
    axes[1].set(xlabel="Coverage (fraction receiving a label)",
                ylabel="Accuracy among retained predictions", ylim=(0, 1.02), xlim=(0, 1.02),
                title="Official-test risk–coverage diagnostic")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    if all(row["threshold"] is None for row in rows):
        axes[1].text(
            0.98, 0.05,
            "Strict validation rule selected no threshold:\nall official outputs are uncertain",
            transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
                  "edgecolor": "#666666"},
        )
    figure.suptitle("Week 6 confidence-aware MFCC SVM")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(rows, max_error, min_retained):
    lines = [
        "# Week 6 real-model confidence experiment",
        "",
        "One frozen MFCC linear SVM per target was trained on the 956 Week 3 fit",
        "clips. The 239 development-validation clips were divided into 119",
        "temperature-calibration and 120 threshold-selection clips. The official",
        "200-clip unseen-dog test was used only after both policies were frozen.",
        "",
        "## Official unseen-dog result",
        "",
        "| Target | Temperature | Threshold | NLL before → after | Coverage | Retained accuracy | Uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        threshold = "none" if row["threshold"] is None else f'{row["threshold"]:.3f}'
        accuracy = "n/a" if row["retained_accuracy"] is None else f'{row["retained_accuracy"]:.3f}'
        lines.append(
            f'| {row["target"].capitalize()} | {row["temperature"]:.3f} | {threshold} | '
            f'{row["nll_before"]:.3f} → {row["nll_after"]:.3f} | '
            f'{row["coverage"]:.3f} | {accuracy} | {row["uncertain"]}/{row["official_n"]} |'
        )
    lines.extend([
        "",
        f"The threshold rule maximizes selection-set coverage subject to empirical error ≤ {max_error:.0%}",
        f"and at least {min_retained} retained validation examples. This is not a test-risk guarantee.",
        "",
        "![Calibration and risk-coverage results](confidence_results.png)",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python code/confidence_experiment.py \\",
        "  --manifest results/local/audit/manifest.json \\",
        "  --cache results/local/week3_features.npz \\",
        "  --local-output results/local/week6_confidence \\",
        "  --output results/week6_confidence",
        "```",
        "",
        "## Interpretation boundaries",
        "",
        "- `LinearSVC.decision_function` values are treated as three class scores; they are not native probabilities.",
        "- Temperature and threshold selection use disjoint clips and the exact same frozen model.",
        "- The Week 3 value `C=0.01` is frozen rather than retuned in this experiment.",
        "- The development partition is clip-level because public dog/source IDs are unavailable.",
        "- `reliable` means the score passed an empirical validation rule, not that the prediction is guaranteed correct.",
        "- The official labels are used only for final reporting, never for temperature or threshold selection.",
        "",
    ])
    return "\n".join(lines)


def run(args):
    manifest = Path(args.manifest)
    output = Path(args.output)
    local_output = Path(args.local_output)
    if output.exists() or local_output.exists():
        raise ValueError("output exists; choose fresh paths to preserve experiment history")
    eligible_train, official_test = load_manifest(manifest)
    fit_ids, validation_ids = fixed_split(eligible_train)
    row_by_id = {row["audio_id"]: row for row in eligible_train + official_test}
    fit_rows = [row_by_id[audio_id] for audio_id in fit_ids]
    validation_rows = [row_by_id[audio_id] for audio_id in validation_ids]
    calibration_ids, selection_ids = split_validation(validation_rows)
    calibration_rows = [row_by_id[audio_id] for audio_id in calibration_ids]
    selection_rows = [row_by_id[audio_id] for audio_id in selection_ids]
    official_rows = sorted(official_test, key=lambda row: row["audio_id"])

    ids, mfcc, _ = build_features(
        eligible_train + official_test, args.cache, sha256(manifest)
    )
    feature_index = {audio_id: index for index, audio_id in enumerate(ids)}

    def matrix(rows):
        return mfcc[[feature_index[row["audio_id"]] for row in rows]]

    local_output.mkdir(parents=True)
    output.mkdir(parents=True)
    summaries, curves = [], {}
    manifest_hash = sha256(manifest)
    for target in ("arousal", "valence"):
        model = classical_model(MODEL_KIND, MODEL_C)
        model.fit(matrix(fit_rows), encode_labels(fit_rows, target))
        identity = model_identity(target, fit_ids, manifest_hash, model)
        calibration = score_export(
            model, identity, target, calibration_rows, matrix(calibration_rows),
            fit_ids, "validation"
        )
        selection = score_export(
            model, identity, target, selection_rows, matrix(selection_rows),
            fit_ids, "validation"
        )
        official_with_labels = score_export(
            model, identity, target, official_rows, matrix(official_rows),
            fit_ids, "official_test"
        )
        official_for_inference = dict(official_with_labels)
        official_for_inference.pop("labels")
        policy = fit_policy(calibration, selection, args.max_error, args.min_retained)
        predictions = apply_policy(policy, official_for_inference)

        write_json(local_output / f"{target}_calibration_scores.json", calibration)
        write_json(local_output / f"{target}_selection_scores.json", selection)
        write_json(local_output / f"{target}_official_scores.json", official_with_labels)
        write_json(local_output / f"{target}_official_predictions.json", predictions)
        write_json(output / f"{target}_policy.json", policy)
        summary = policy_metrics(
            official_with_labels["logits"], official_with_labels["labels"],
            policy, predictions
        )
        summaries.append(summary)
        curves[target] = risk_coverage(
            official_with_labels["logits"], official_with_labels["labels"],
            policy["temperature"]
        )

    examples = {
        target: json.loads(
            (local_output / f"{target}_official_predictions.json").read_text()
        )[:2]
        for target in ("arousal", "valence")
    }
    write_json(output / "example_predictions.json", examples)

    write_summary_csv(output / "summary.csv", summaries)
    write_json(output / "run.json", {
        "manifest_sha256": manifest_hash,
        "model": MODEL_KIND,
        "C": MODEL_C,
        "fit_count": len(fit_rows),
        "calibration_count": len(calibration_rows),
        "selection_count": len(selection_rows),
        "official_test_count": len(official_rows),
        "split_seed": SPLIT_SEED,
        "max_error": args.max_error,
        "min_retained": args.min_retained,
        "local_score_exports": str(local_output),
    })
    plot_results(summaries, curves, output / "confidence_results.png")
    (output / "README.md").write_text(
        markdown_report(summaries, args.max_error, args.min_retained)
    )
    print(f"Wrote Week 6 summary to {output} and score exports to {local_output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="results/local/week3_features.npz")
    parser.add_argument("--local-output", default="results/local/week6_confidence")
    parser.add_argument("--output", default="results/week6_confidence")
    parser.add_argument("--max-error", type=float, default=0.20)
    parser.add_argument("--min-retained", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
