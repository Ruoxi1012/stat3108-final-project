"""Run leakage-aware Week 3 baselines on EmotionalCanines.

The script creates a fixed stratified development split from eligible released
training clips, tunes only on that validation partition, refits classical
models on all eligible training clips, and evaluates the untouched official
test clips.  The official split is the only verified dog-disjoint evaluation.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import dct
from scipy.io import wavfile
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from torch import nn


AROUSAL_LABELS = ("Low", "Medium", "High")
VALENCE_LABELS = ("Negative", "Neutral", "Positive")
TARGET_LABELS = {"arousal": AROUSAL_LABELS, "valence": VALENCE_LABELS}
SPLIT_SEED = 3108
VALIDATION_FRACTION = 0.20
SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512
MFCC_MELS = 40
CNN_MELS = 48
CNN_FRAMES = 400  # About four seconds; covers about 95% in the label-blind audit.
CLASSICAL_C_GRID = (0.01, 0.1, 1.0, 10.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_audio(path):
    rate, raw = wavfile.read(path)
    if raw.size == 0:
        raise ValueError(f"empty audio: {path}")
    if raw.dtype == np.uint8:
        audio = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.signedinteger):
        audio = raw.astype(np.float64) / (2 ** (np.iinfo(raw.dtype).bits - 1))
    elif np.issubdtype(raw.dtype, np.floating):
        audio = raw.astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV encoding: {raw.dtype}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, found {rate} Hz: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite samples: {path}")
    return audio


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_mels, sample_rate=SAMPLE_RATE, n_fft=N_FFT, low_hz=50.0):
    high_hz = sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = bins[index:index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index, frequency_bin] = (frequency_bin - left) / (center - left)
        for frequency_bin in range(center, min(right, filters.shape[1])):
            filters[index, frequency_bin] = (right - frequency_bin) / (right - center)
    return filters


def frame_power(audio):
    centered = np.asarray(audio, dtype=np.float64) - float(np.mean(audio))
    peak = float(np.max(np.abs(centered)))
    if peak > 1e-12:
        centered = centered / peak
    if len(centered) < FRAME_LENGTH:
        centered = np.pad(centered, (0, FRAME_LENGTH - len(centered)))
    frame_count = 1 + int(math.ceil((len(centered) - FRAME_LENGTH) / HOP_LENGTH))
    required = (frame_count - 1) * HOP_LENGTH + FRAME_LENGTH
    centered = np.pad(centered, (0, required - len(centered)))
    indices = np.arange(FRAME_LENGTH)[None, :] + HOP_LENGTH * np.arange(frame_count)[:, None]
    frames = centered[indices] * np.hanning(FRAME_LENGTH)[None, :]
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2


def log_mel(audio, n_mels):
    energies = frame_power(audio) @ mel_filterbank(n_mels).T
    return np.log(np.maximum(energies, 1e-10)).T


def mfcc_summary(audio):
    mel = log_mel(audio, MFCC_MELS)
    coefficients = dct(mel, type=2, axis=0, norm="ortho")[:20]
    deltas = np.diff(coefficients, axis=1, prepend=coefficients[:, :1])
    duration = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    summary = np.concatenate([
        coefficients.mean(axis=1),
        coefficients.std(axis=1),
        deltas.mean(axis=1),
        deltas.std(axis=1),
        np.asarray([np.log1p(duration), np.log(rms + 1e-10)]),
    ])
    return summary.astype(np.float32)


def fixed_log_mel(audio):
    spectrogram = log_mel(audio, CNN_MELS)
    mean = float(spectrogram.mean())
    standard_deviation = float(spectrogram.std())
    spectrogram = (spectrogram - mean) / max(standard_deviation, 1e-6)
    if spectrogram.shape[1] > CNN_FRAMES:
        offset = (spectrogram.shape[1] - CNN_FRAMES) // 2
        spectrogram = spectrogram[:, offset:offset + CNN_FRAMES]
    elif spectrogram.shape[1] < CNN_FRAMES:
        total = CNN_FRAMES - spectrogram.shape[1]
        left = total // 2
        spectrogram = np.pad(spectrogram, ((0, 0), (left, total - left)))
    return spectrogram.astype(np.float32)


def load_manifest(path):
    rows = json.loads(Path(path).read_text())
    if len(rows) != len({row["audio_id"] for row in rows}):
        raise ValueError("manifest audio IDs must be unique")
    eligible_train = [
        row for row in rows
        if row["split"] == "train" and row["status"] == "ok" and row["model_status"] == "include"
    ]
    official_test = [row for row in rows if row["split"] == "test" and row["status"] == "ok"]
    if len(eligible_train) != 1195 or len(official_test) != 200:
        raise ValueError(
            f"expected 1195 eligible train and 200 official test clips; "
            f"found {len(eligible_train)} and {len(official_test)}"
        )
    return eligible_train, official_test


def fixed_split(rows, seed=SPLIT_SEED, validation_fraction=VALIDATION_FRACTION):
    ids = [row["audio_id"] for row in rows]
    # Preserve breed and the joint label combination in the clip-level split.
    strata = [f'{row["breed"]}|{row["arousal"]}|{row["valence"]}' for row in rows]
    train_ids, validation_ids = train_test_split(
        ids,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return sorted(train_ids), sorted(validation_ids)


def build_features(rows, cache_path, manifest_hash):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["manifest_hash"]) == manifest_hash:
            return list(cached["audio_ids"]), cached["mfcc"], cached["logmel"]
    audio_ids, mfcc_features, logmel_features = [], [], []
    for index, row in enumerate(rows, start=1):
        audio = read_audio(row["path"])
        audio_ids.append(row["audio_id"])
        mfcc_features.append(mfcc_summary(audio))
        logmel_features.append(fixed_log_mel(audio))
        if index % 100 == 0:
            print(f"Extracted features for {index}/{len(rows)} clips", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        manifest_hash=np.asarray(manifest_hash),
        audio_ids=np.asarray(audio_ids),
        mfcc=np.stack(mfcc_features),
        logmel=np.stack(logmel_features),
    )
    return audio_ids, np.stack(mfcc_features), np.stack(logmel_features)


def encode_labels(rows, target):
    mapping = {label: index for index, label in enumerate(TARGET_LABELS[target])}
    return np.asarray([mapping[row[target]] for row in rows], dtype=np.int64)


def metric_values(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_error_rate": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) == 2)),
    }


def evaluation_rows(model, seed, target, source_rows, y_true, y_pred, evaluation_set):
    output = []
    for breed in ("pooled", "husky", "shiba"):
        selected = np.arange(len(source_rows)) if breed == "pooled" else np.asarray([
            index for index, row in enumerate(source_rows) if row["breed"] == breed
        ])
        metrics = metric_values(y_true[selected], y_pred[selected])
        output.append({
            "model": model,
            "seed": seed,
            "target": target,
            "evaluation_set": evaluation_set,
            "breed": breed,
            "n": int(len(selected)),
            **metrics,
            "confusion_matrix": confusion_matrix(
                y_true[selected], y_pred[selected], labels=(0, 1, 2)
            ).tolist(),
        })
    return output


def classical_model(kind, c_value):
    if kind == "MFCC logistic regression":
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=SPLIT_SEED,
        )
    elif kind == "MFCC linear SVM":
        classifier = LinearSVC(C=c_value, class_weight="balanced", random_state=SPLIT_SEED)
    else:
        raise ValueError(f"unknown model: {kind}")
    return make_pipeline(StandardScaler(), classifier)


def tune_classical(kind, x_fit, y_fit, x_validation, y_validation):
    trials = []
    best = None
    for c_value in CLASSICAL_C_GRID:
        model = classical_model(kind, c_value)
        model.fit(x_fit, y_fit)
        score = float(f1_score(y_validation, model.predict(x_validation), average="macro"))
        trials.append({"C": c_value, "validation_macro_f1": score})
        candidate = (score, -c_value, model)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return -best[1], best[2], trials


class SmallLogMelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=(2, 4), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.20)
        self.arousal = nn.Linear(32, 3)
        self.valence = nn.Linear(32, 3)

    def forward(self, inputs):
        hidden = self.dropout(self.features(inputs).flatten(1))
        return self.arousal(hidden), self.valence(hidden)


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def class_weights(labels):
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    return torch.tensor(len(labels) / (3.0 * counts), dtype=torch.float32)


def cnn_predict(model, x, batch_size=64):
    model.eval()
    arousal, valence = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            inputs = torch.from_numpy(x[start:start + batch_size, None])
            outputs = model(inputs)
            arousal.extend(outputs[0].argmax(1).cpu().numpy().tolist())
            valence.extend(outputs[1].argmax(1).cpu().numpy().tolist())
    return np.asarray(arousal), np.asarray(valence)


def train_cnn(x, y_arousal, y_valence, seed, epochs, validation=None, patience=5):
    set_torch_seed(seed)
    model = SmallLogMelCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = (
        nn.CrossEntropyLoss(weight=class_weights(y_arousal)),
        nn.CrossEntropyLoss(weight=class_weights(y_valence)),
    )
    generator = torch.Generator().manual_seed(seed)
    best_state, best_score, best_epoch, stale = None, -1.0, epochs, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x), generator=generator).numpy()
        total_loss = 0.0
        for start in range(0, len(x), 32):
            selected = permutation[start:start + 32]
            inputs = torch.from_numpy(x[selected, None])
            target_arousal = torch.from_numpy(y_arousal[selected])
            target_valence = torch.from_numpy(y_valence[selected])
            optimizer.zero_grad(set_to_none=True)
            output_arousal, output_valence = model(inputs)
            loss = losses[0](output_arousal, target_arousal) + losses[1](output_valence, target_valence)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
        record = {"epoch": epoch, "train_loss": total_loss / len(x)}
        if validation is not None:
            x_validation, validation_arousal, validation_valence = validation
            predicted_arousal, predicted_valence = cnn_predict(model, x_validation)
            scores = (
                f1_score(validation_arousal, predicted_arousal, average="macro"),
                f1_score(validation_valence, predicted_valence, average="macro"),
            )
            record.update(validation_arousal_macro_f1=float(scores[0]),
                          validation_valence_macro_f1=float(scores[1]))
            score = float(np.mean(scores))
            if score > best_score + 1e-12:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.append(record)
                break
        history.append(record)
    if validation is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def baseline_predictions(y_fit, count, kind, seed):
    if kind == "majority":
        majority = int(np.bincount(y_fit, minlength=3).argmax())
        return np.full(count, majority, dtype=np.int64)
    probabilities = np.bincount(y_fit, minlength=3).astype(float)
    probabilities /= probabilities.sum()
    return np.random.default_rng(seed).choice(3, size=count, p=probabilities)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows):
    metric_names = (
        "macro_f1", "balanced_accuracy", "mae",
        "quadratic_weighted_kappa", "severe_error_rate",
    )
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model"], row["target"], row["evaluation_set"], row["breed"], row["n"])
        grouped[key].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(("model", "target", "evaluation_set", "breed", "n"), key))
        record["runs"] = len(group)
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def plot_official(summary, destination):
    selected = [
        row for row in summary
        if row["evaluation_set"] == "official_test" and row["breed"] == "pooled"
    ]
    models = ["Majority", "Stratified random", "MFCC logistic regression", "MFCC linear SVM", "Small Log-Mel CNN"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, target in zip(axes, ("arousal", "valence")):
        by_model = {row["model"]: row for row in selected if row["target"] == target}
        means = [by_model[model]["macro_f1_mean"] for model in models]
        errors = [by_model[model]["macro_f1_sd"] for model in models]
        positions = np.arange(len(models))
        axis.bar(positions, means, yerr=errors, capsize=3, color="#2A6F97")
        axis.axhline(1 / 3, color="#B23A48", linestyle="--", linewidth=1, label="chance reference")
        axis.set_xticks(positions, [name.replace(" ", "\n", 1) for name in models], fontsize=8)
        axis.set_title(target.capitalize())
        axis.set_ylim(0, 1)
        axis.set_ylabel("Macro F1" if target == "arousal" else "")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Official unseen-dog baseline performance (n=200)\nCNN error bars show SD across three seeds")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(summary, selection, cnn_details):
    rows = [
        row for row in summary
        if row["evaluation_set"] in ("validation", "official_test") and row["breed"] == "pooled"
    ]
    lookup = {(row["model"], row["target"], row["evaluation_set"]): row for row in rows}
    models = ["Majority", "Stratified random", "MFCC logistic regression", "MFCC linear SVM", "Small Log-Mel CNN"]
    lines = [
        "# Week 3 honest baselines",
        "",
        "This directory is generated by `code/baselines.py`. The fixed validation",
        "partition is stratified by breed plus the joint arousal/valence label. It",
        "may contain the same dogs or recording sources on both sides because public",
        "dog IDs are unavailable. Only the official test is verified dog-disjoint.",
        "",
        "## Pooled macro F1",
        "",
        "| Model | Arousal validation | Arousal official | Valence validation | Valence official |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        values = []
        for target, evaluation in (("arousal", "validation"), ("arousal", "official_test"),
                                   ("valence", "validation"), ("valence", "official_test")):
            row = lookup[(model, target, evaluation)]
            value = f'{row["macro_f1_mean"]:.3f}'
            if row["runs"] > 1:
                value += f' ± {row["macro_f1_sd"]:.3f}'
            values.append(value)
        lines.append(f'| {model} | {" | ".join(values)} |')
    lines.extend([
        "",
        "The classical-model `C` values were selected separately for each target on",
        "validation, then the models were refit on all 1,195 eligible training clips",
        "before official testing. The CNN selected an epoch on validation for each of",
        "three seeds and was retrained on all eligible training clips for that many epochs.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python code/audio_pipeline.py audit --root data/raw/EmotionalCanines --output results/local/audit",
        "python code/baselines.py --manifest results/local/audit/manifest.json --output results/week3_baselines --cache results/local/week3_features.npz",
        "```",
        "",
        "## Boundaries",
        "",
        "- Five released training rows are excluded under the predeclared duplicate policy.",
        "- No official-test label is used for model or epoch selection.",
        "- The MFCC baseline is not an eGeMAPS implementation.",
        "- The CNN uses per-clip normalized, center-cropped/padded four-second Log-Mel inputs.",
        "- This is a Week 3 baseline, not evidence that the model reads private emotion or is ready for deployment.",
        "- Test-set duplicate sensitivity, confidence intervals, and cross-breed transfer remain Week 4 work.",
        "",
        "## Selection record",
        "",
        f"Classical choices: `{json.dumps(selection, sort_keys=True)}`",
        "",
        f"CNN best epochs: `{json.dumps(cnn_details, sort_keys=True)}`",
        "",
        "![Official baseline comparison](baseline_comparison.png)",
        "",
    ])
    return "\n".join(lines)


def run(args):
    manifest = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eligible_train, official_test = load_manifest(manifest)
    train_ids, validation_ids = fixed_split(eligible_train, args.split_seed, args.validation_fraction)
    train_set, validation_set = set(train_ids), set(validation_ids)
    if train_set & validation_set:
        raise AssertionError("development split overlap")
    if set(row["audio_id"] for row in official_test) & (train_set | validation_set):
        raise AssertionError("official test leaked into development IDs")

    all_rows = eligible_train + official_test
    ids, mfcc, logmel = build_features(all_rows, args.cache, sha256(manifest))
    feature_index = {audio_id: index for index, audio_id in enumerate(ids)}
    row_by_id = {row["audio_id"]: row for row in all_rows}
    fit_rows = [row_by_id[audio_id] for audio_id in train_ids]
    validation_rows = [row_by_id[audio_id] for audio_id in validation_ids]
    official_rows = sorted(official_test, key=lambda row: row["audio_id"])

    def matrix(rows, features):
        return features[[feature_index[row["audio_id"]] for row in rows]]

    x_fit_mfcc, x_validation_mfcc = matrix(fit_rows, mfcc), matrix(validation_rows, mfcc)
    x_all_mfcc, x_official_mfcc = matrix(eligible_train, mfcc), matrix(official_rows, mfcc)
    x_fit_logmel, x_validation_logmel = matrix(fit_rows, logmel), matrix(validation_rows, logmel)
    x_all_logmel, x_official_logmel = matrix(eligible_train, logmel), matrix(official_rows, logmel)

    raw_metrics, selection = [], {}
    for target in TARGET_LABELS:
        y_fit = encode_labels(fit_rows, target)
        y_validation = encode_labels(validation_rows, target)
        y_all = encode_labels(eligible_train, target)
        y_official = encode_labels(official_rows, target)
        for display, kind in (("Majority", "majority"), ("Stratified random", "stratified")):
            raw_metrics.extend(evaluation_rows(
                display, args.split_seed, target, validation_rows, y_validation,
                baseline_predictions(y_fit, len(y_validation), kind, args.split_seed + (0 if target == "arousal" else 1)),
                "validation",
            ))
            raw_metrics.extend(evaluation_rows(
                display, args.split_seed, target, official_rows, y_official,
                baseline_predictions(y_all, len(y_official), kind, args.split_seed + (10 if target == "arousal" else 11)),
                "official_test",
            ))
        for kind in ("MFCC logistic regression", "MFCC linear SVM"):
            c_value, development_model, trials = tune_classical(
                kind, x_fit_mfcc, y_fit, x_validation_mfcc, y_validation
            )
            selection[f"{kind}/{target}"] = {"selected_C": c_value, "trials": trials}
            raw_metrics.extend(evaluation_rows(
                kind, args.split_seed, target, validation_rows, y_validation,
                development_model.predict(x_validation_mfcc), "validation",
            ))
            final_model = classical_model(kind, c_value).fit(x_all_mfcc, y_all)
            raw_metrics.extend(evaluation_rows(
                kind, args.split_seed, target, official_rows, y_official,
                final_model.predict(x_official_mfcc), "official_test",
            ))

    y_fit_arousal, y_fit_valence = encode_labels(fit_rows, "arousal"), encode_labels(fit_rows, "valence")
    y_validation_arousal = encode_labels(validation_rows, "arousal")
    y_validation_valence = encode_labels(validation_rows, "valence")
    y_all_arousal, y_all_valence = encode_labels(eligible_train, "arousal"), encode_labels(eligible_train, "valence")
    y_official_arousal = encode_labels(official_rows, "arousal")
    y_official_valence = encode_labels(official_rows, "valence")
    cnn_details = []
    for seed in args.cnn_seeds:
        print(f"Training CNN development model with seed {seed}", flush=True)
        development_model, best_epoch, history = train_cnn(
            x_fit_logmel, y_fit_arousal, y_fit_valence, seed, args.cnn_epochs,
            validation=(x_validation_logmel, y_validation_arousal, y_validation_valence),
            patience=args.cnn_patience,
        )
        validation_predictions = cnn_predict(development_model, x_validation_logmel)
        raw_metrics.extend(evaluation_rows(
            "Small Log-Mel CNN", seed, "arousal", validation_rows,
            y_validation_arousal, validation_predictions[0], "validation",
        ))
        raw_metrics.extend(evaluation_rows(
            "Small Log-Mel CNN", seed, "valence", validation_rows,
            y_validation_valence, validation_predictions[1], "validation",
        ))
        print(f"Retraining CNN on all eligible training clips for {best_epoch} epochs", flush=True)
        final_model, _, final_history = train_cnn(
            x_all_logmel, y_all_arousal, y_all_valence, seed, best_epoch,
            validation=None,
        )
        official_predictions = cnn_predict(final_model, x_official_logmel)
        raw_metrics.extend(evaluation_rows(
            "Small Log-Mel CNN", seed, "arousal", official_rows,
            y_official_arousal, official_predictions[0], "official_test",
        ))
        raw_metrics.extend(evaluation_rows(
            "Small Log-Mel CNN", seed, "valence", official_rows,
            y_official_valence, official_predictions[1], "official_test",
        ))
        cnn_details.append({
            "seed": seed,
            "selected_epoch": best_epoch,
            "development_history": history,
            "final_history": final_history,
        })

    split_record = {
        "seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "stratification": "breed + joint arousal/valence label",
        "limitation": "clip-level only; public dog/source IDs are unavailable",
        "eligible_train_count": len(eligible_train),
        "fit_count": len(fit_rows),
        "validation_count": len(validation_rows),
        "official_test_count": len(official_rows),
        "fit_ids": train_ids,
        "validation_ids": validation_ids,
    }
    (output / "split.json").write_text(json.dumps(split_record, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps({
        "manifest_sha256": sha256(manifest),
        "feature_protocol": {
            "sample_rate": SAMPLE_RATE,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "mfcc_mels": MFCC_MELS,
            "cnn_mels": CNN_MELS,
            "cnn_frames": CNN_FRAMES,
            "amplitude": "DC removal then per-clip peak normalization for spectral features",
        },
        "selection": selection,
        "cnn": cnn_details,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "torch": torch.__version__,
        },
    }, indent=2) + "\n")
    raw_fields = (
        "model", "seed", "target", "evaluation_set", "breed", "n",
        "macro_f1", "balanced_accuracy", "mae", "quadratic_weighted_kappa",
        "severe_error_rate", "confusion_matrix",
    )
    write_csv(output / "metrics.csv", raw_metrics, raw_fields)
    summary = aggregate_metrics(raw_metrics)
    summary_fields = tuple(summary[0].keys())
    write_csv(output / "summary.csv", summary, summary_fields)
    plot_official(summary, output / "baseline_comparison.png")
    (output / "README.md").write_text(markdown_report(summary, selection, [
        {"seed": item["seed"], "selected_epoch": item["selected_epoch"]}
        for item in cnn_details
    ]))
    print(f"Wrote Week 3 results to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="results/week3_baselines")
    parser.add_argument("--cache", default="results/local/week3_features.npz")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--cnn-seeds", type=int, nargs="+", default=(3108, 3109, 3110))
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-patience", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
