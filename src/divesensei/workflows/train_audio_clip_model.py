#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei train-audio-clip-model",
        description="Train the short-window audio clip classifier.",
    )
    parser.add_argument("labels_path", help="Input labels.jsonl produced by label-audio")
    parser.add_argument("output_model", help="Output JSON model path")
    parser.add_argument("--epochs", type=int, default=3000, help="Gradient-descent epochs")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Gradient-descent learning rate")
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced"],
        default="none",
        help="Optional sample-weighting scheme for class imbalance.",
    )
    parser.add_argument("--hist-bins", type=int, default=10, help="Histogram bins for per-class probability summaries")
    return parser


def compute_sample_weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode != "balanced":
        return np.ones_like(y, dtype=np.float32)
    positive_count = float(np.sum(y))
    negative_count = float(len(y) - np.sum(y))
    pos_weight = (len(y) / (2.0 * positive_count)) if positive_count > 0 else 1.0
    neg_weight = (len(y) / (2.0 * negative_count)) if negative_count > 0 else 1.0
    weights = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
    return weights


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int = 3000,
    lr: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    means = x.mean(axis=0)
    stds = np.maximum(x.std(axis=0), 1e-6)
    z = (x - means) / stds
    weights = np.zeros(z.shape[1], dtype=np.float32)
    bias = 0.0
    normalizer = max(float(np.sum(sample_weights)), 1e-6)
    for _ in range(epochs):
        logits = z @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = (preds - y) * sample_weights
        weights -= lr * ((z.T @ error) / normalizer)
        bias -= lr * float(np.sum(error) / normalizer)
    return means, stds, weights, bias


def probability_histogram(values: np.ndarray, bins: int) -> dict[str, list[float] | list[int]]:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {
        "bins": [float(edge) for edge in edges.tolist()],
        "counts": [int(count) for count in hist.tolist()],
    }


def feature_importance(feature_names: list[str], weights: np.ndarray) -> list[dict[str, float | str]]:
    ranked = sorted(
        [
            {
                "feature": name,
                "normalized_weight": float(weight),
                "abs_normalized_weight": float(abs(weight)),
                "odds_ratio_per_std": float(np.exp(np.clip(weight, -20.0, 20.0))),
            }
            for name, weight in zip(feature_names, weights, strict=True)
        ],
        key=lambda item: item["abs_normalized_weight"],
        reverse=True,
    )
    return ranked


def build_training_summary(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    hist_bins: int,
    class_weight_mode: str,
    sample_weights: np.ndarray,
) -> dict:
    z = (x - means) / stds
    logits = z @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probs >= 0.5).astype(np.float32)
    positive_mask = y > 0.5
    negative_mask = ~positive_mask
    positive_probs = probs[positive_mask]
    negative_probs = probs[negative_mask]
    summary = {
        "training_accuracy": float(np.mean(predictions == y)),
        "mean_positive_probability": float(np.mean(positive_probs)) if positive_probs.size else None,
        "mean_negative_probability": float(np.mean(negative_probs)) if negative_probs.size else None,
        "probability_histogram": {
            "positive": probability_histogram(positive_probs, hist_bins) if positive_probs.size else {"bins": [], "counts": []},
            "negative": probability_histogram(negative_probs, hist_bins) if negative_probs.size else {"bins": [], "counts": []},
        },
        "feature_importance": feature_importance(feature_names, weights),
        "class_weight_mode": class_weight_mode,
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
        "sample_weight_mean": float(np.mean(sample_weights)),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))

    labels_path = Path(args.labels_path).resolve()
    output_path = Path(args.output_model).resolve()
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
    sample_weights = compute_sample_weights(y, args.class_weight_mode)
    means, stds, weights, bias = fit_logistic_regression(
        x,
        y,
        sample_weights=sample_weights,
        epochs=args.epochs,
        lr=args.learning_rate,
    )
    training_summary = build_training_summary(
        x,
        y,
        list(AUDIO_CLIP_FEATURES),
        means,
        stds,
        weights,
        bias,
        hist_bins=max(2, int(args.hist_bins)),
        class_weight_mode=args.class_weight_mode,
        sample_weights=sample_weights,
    )
    model = {
        "feature_names": AUDIO_CLIP_FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_rows": int(len(x)),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(len(y) - np.sum(y)),
        "training_summary": training_summary,
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
        "class_weight_mode": args.class_weight_mode,
        "feature_importance": training_summary["feature_importance"],
        "probability_histogram": training_summary["probability_histogram"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
