#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from divesensei.detection.audio_model import MODEL_FEATURES


def load_review_labels(path: Path) -> dict[tuple[str, float], int]:
    labels: dict[tuple[str, float], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = (row.get("review_label") or "").strip().lower()
            if raw not in {"dive", "non-dive"}:
                continue
            timestamp = round(float(row["timestamp"]), 3)
            clip = str(row.get("source_file") or row.get("session_file") or "")
            labels[(clip, timestamp)] = 1 if raw == "dive" else 0
    return labels


def load_rows(path: Path, labels: dict[tuple[str, float], int]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row.get("file", "")), round(float(row["timestamp"]), 3))
            if key not in labels:
                continue
            xs.append([float(row[name]) for name in MODEL_FEATURES])
            ys.append(labels[key])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, epochs: int = 4000, lr: float = 0.05) -> tuple[np.ndarray, float]:
    means = x.mean(axis=0)
    stds = np.maximum(x.std(axis=0), 1e-6)
    z = (x - means) / stds
    weights = np.zeros(z.shape[1], dtype=np.float32)
    bias = 0.0
    for _ in range(epochs):
        logits = z @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = preds - y
        weights -= lr * ((z.T @ error) / len(z))
        bias -= lr * float(np.mean(error))
    return np.concatenate([means, stds, weights]), bias


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print("Usage: train_audio_candidate_model.py <features.jsonl> <review_labels.csv> <output_model.json>")
        return 1

    features_path = Path(argv[0]).resolve()
    labels_path = Path(argv[1]).resolve()
    output_path = Path(argv[2]).resolve()

    labels = load_review_labels(labels_path)
    x, y = load_rows(features_path, labels)
    if len(x) < 4 or len(np.unique(y)) < 2:
        print("Need labeled examples for both classes before training.")
        return 1

    packed, bias = fit_logistic_regression(x, y)
    n = len(MODEL_FEATURES)
    model = {
        "feature_names": MODEL_FEATURES,
        "means": packed[:n].tolist(),
        "stds": packed[n : 2 * n].tolist(),
        "weights": packed[2 * n :].tolist(),
        "bias": bias,
        "training_rows": int(len(x)),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(len(y) - np.sum(y)),
    }
    output_path.write_text(json.dumps(model, indent=2))
    print(json.dumps({"output": str(output_path), "rows": int(len(x))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
