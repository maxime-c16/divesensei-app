#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from divesensei.detection.audio_clip_model import AudioClipModel
from divesensei.detection.audio_features import AUDIO_CLIP_FEATURES, extract_clip_feature_map, load_wav_mono_float32


def load_label_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, epochs: int = 3000, lr: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    return means, stds, weights, bias


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: train_audio_clip_model.py <labels.jsonl> <output_model.json>")
        return 1

    labels_path = Path(argv[0]).resolve()
    output_path = Path(argv[1]).resolve()
    rows = load_label_rows(labels_path)
    x_rows = []
    y_rows = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label not in {"dive", "non-dive"}:
            continue
        audio_path = Path(str(row.get("audio_path", ""))).resolve()
        if not audio_path.exists():
            continue
        signal, sample_rate = load_wav_mono_float32(audio_path)
        center_time = float(row.get("timestamp_seconds", 0.0)) - float(row.get("clip_start_seconds", 0.0))
        features = extract_clip_feature_map(
            signal,
            sample_rate,
            center_time,
            window_seconds=float(row.get("clip_duration_seconds", 3.0)),
            frame_length=1024,
            hop_length=256,
        )
        x_rows.append([float(features[name]) for name in AUDIO_CLIP_FEATURES])
        y_rows.append(1.0 if label == "dive" else 0.0)

    if len(x_rows) < 4 or len(set(y_rows)) < 2:
        print("Need at least four labeled examples across both classes.")
        return 1

    x = np.array(x_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    means, stds, weights, bias = fit_logistic_regression(x, y)
    model = {
        "feature_names": AUDIO_CLIP_FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_rows": int(len(x)),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(len(y) - np.sum(y)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2))
    clip_model = AudioClipModel.load(output_path)
    probs = [clip_model.predict_probability(dict(zip(AUDIO_CLIP_FEATURES, row))) for row in x_rows]
    summary = {
        "output": str(output_path),
        "rows": int(len(x)),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(len(y) - np.sum(y)),
        "training_accuracy": float(np.mean(((np.array(probs) >= 0.5).astype(np.float32)) == y)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
