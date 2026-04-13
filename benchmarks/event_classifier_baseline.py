from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MANIFEST = Path("outputs/event_window_manifest_preview.jsonl")
DEFAULT_JSON = Path("outputs/event_classifier_baseline.json")
DEFAULT_MD = Path("outputs/event_classifier_baseline.md")

LABELS = ["springboard_dive", "springboard_rebound_only", "platform_dive", "noise_or_other"]
PAIRWISE_TASKS = [
    ("springboard_dive", "springboard_rebound_only"),
    ("platform_dive", "noise_or_other"),
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_label_for_row(row: dict) -> str | None:
    label = row.get("final_human_event_label")
    if label:
        return str(label)
    label = row.get("suggested_event_label")
    if label:
        return str(label)
    label = row.get("event_label")
    if label:
        return str(label)
    return None


def to_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return float(default)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[index[str(truth)]][index[str(pred)]] += 1
    return matrix.tolist()


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float]:
    tp = int(np.sum((y_true == label) & (y_pred == label)))
    fp = int(np.sum((y_true != label) & (y_pred == label)))
    fn = int(np.sum((y_true == label) & (y_pred != label)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    prfs = [precision_recall_f1(y_true, y_pred, label) for label in labels]
    return {
        "precision": float(np.mean([item["precision"] for item in prfs])) if prfs else 0.0,
        "recall": float(np.mean([item["recall"] for item in prfs])) if prfs else 0.0,
        "f1": float(np.mean([item["f1"] for item in prfs])) if prfs else 0.0,
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray, positive_label: str) -> float:
    pos = scores[y_true == positive_label]
    neg = scores[y_true != positive_label]
    total = len(pos) * len(neg)
    if total == 0:
        return 0.5
    gt = 0
    eq = 0
    for value in pos:
        diff = value - neg
        gt += int(np.sum(diff > 0))
        eq += int(np.sum(diff == 0))
    auc = (gt + 0.5 * eq) / total
    return float(max(auc, 1.0 - auc))


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(np.clip(z, -40.0, 40.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def train_multinomial_logreg(x: np.ndarray, y: np.ndarray, class_names: list[str], epochs: int = 1200, lr: float = 0.08, l2: float = 0.01) -> dict:
    n_samples, n_features = x.shape
    n_classes = len(class_names)
    weights = np.zeros((n_classes, n_features), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    for _ in range(epochs):
        logits = x @ weights.T + bias
        probs = softmax(logits)
        target = np.zeros_like(probs)
        target[np.arange(n_samples), y] = 1.0
        error = probs - target
        grad_w = error.T @ x / n_samples + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= lr * grad_w
        bias -= lr * grad_b
    return {"weights": weights, "bias": bias, "classes": class_names}


def predict_multinomial(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = x @ model["weights"].T + model["bias"]
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    return pred, probs


def train_binary_logreg(x: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.1, l2: float = 0.01) -> dict:
    n_samples, n_features = x.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = probs - y
        grad_w = (x.T @ error) / n_samples + l2 * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return {"weights": weights, "bias": bias}


def predict_binary_logreg(model: dict, x: np.ndarray) -> np.ndarray:
    logits = x @ model["weights"] + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def feature_vector(row: dict) -> list[float]:
    duration = to_float(row.get("event_window_end_seconds")) - to_float(row.get("event_window_start_seconds"))
    anchor = to_float(row.get("event_anchor_timestamp_seconds"))
    proposal = to_float(row.get("proposal_timestamp_seconds"))
    return [
        duration,
        anchor,
        proposal,
        abs(anchor - proposal),
        to_float(row.get("raw_proposal_score")),
        to_float(row.get("audio_score")),
        to_float(row.get("combined_score")),
        to_float(row.get("audio_model_probability")),
        to_float(row.get("audio_clip_probability")),
        1.0 if row.get("threshold_passed") else 0.0,
        1.0 if row.get("is_false_negative_window") else 0.0,
        1.0 if row.get("session_type") == "springboard" else 0.0,
        1.0 if row.get("session_type") == "platform" else 0.0,
        1.0 if row.get("event_label_provenance") == "session_type_inferred" else 0.0,
        1.0 if row.get("event_label_provenance") == "subtype_mapped" else 0.0,
        1.0 if row.get("event_label_provenance") == "uncertain" else 0.0,
    ]


def collect_dataset(rows: list[dict], strict: bool) -> list[dict]:
    selected = []
    for row in rows:
        if strict:
            if row.get("session_type_provenance") != "direct_review":
                continue
        else:
            if event_label_for_row(row) is None:
                continue
        selected.append(row)
    return selected


def split_by_session(rows: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[str(row["source_session_id"])].append(row)
    sessions = sorted(by_session)
    splits = []
    for held_out in sessions:
        train = [row for sid, subset in by_session.items() if sid != held_out for row in subset]
        test = list(by_session[held_out])
        splits.append((held_out, train, test))
    return splits


def train_eval_four_class(train_rows: list[dict], test_rows: list[dict], labels: list[str]) -> dict:
    x_train = np.asarray([feature_vector(row) for row in train_rows], dtype=np.float64)
    x_test = np.asarray([feature_vector(row) for row in test_rows], dtype=np.float64)
    y_train = np.asarray([labels.index(event_label_for_row(row)) for row in train_rows], dtype=int)
    y_test = np.asarray([labels.index(event_label_for_row(row)) for row in test_rows], dtype=int)
    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_test = standardize_apply(x_test, mean, std)
    model = train_multinomial_logreg(x_train, y_train, labels)
    pred_idx, probs = predict_multinomial(model, x_test)
    pred = np.asarray([labels[i] for i in pred_idx], dtype=object)
    truth = np.asarray([labels[i] for i in y_test], dtype=object)
    return {
        "accuracy": accuracy_score(truth, pred),
        "macro": macro_metrics(truth, pred, labels),
        "confusion_matrix": confusion_matrix(truth, pred, labels),
        "per_class": {label: precision_recall_f1(truth, pred, label) for label in labels},
        "predictions": pred.tolist(),
        "truth": truth.tolist(),
        "probabilities": probs.tolist(),
    }


def train_eval_binary(train_rows: list[dict], test_rows: list[dict], positive: str, negative: str) -> dict:
    subset_train = [row for row in train_rows if event_label_for_row(row) in {positive, negative}]
    subset_test = [row for row in test_rows if event_label_for_row(row) in {positive, negative}]
    x_train = np.asarray([feature_vector(row) for row in subset_train], dtype=np.float64)
    x_test = np.asarray([feature_vector(row) for row in subset_test], dtype=np.float64)
    y_train = np.asarray([1 if event_label_for_row(row) == positive else 0 for row in subset_train], dtype=np.float64)
    y_test = np.asarray([1 if event_label_for_row(row) == positive else 0 for row in subset_test], dtype=object)
    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_test = standardize_apply(x_test, mean, std)
    model = train_binary_logreg(x_train, y_train)
    scores = predict_binary_logreg(model, x_test)
    pred = np.asarray([positive if score >= 0.5 else negative for score in scores], dtype=object)
    truth = np.asarray([positive if event_label_for_row(row) == positive else negative for row in subset_test], dtype=object)
    return {
        "train_count": len(subset_train),
        "test_count": len(subset_test),
        "accuracy": accuracy_score(truth, pred),
        "macro": macro_metrics(truth, pred, [positive, negative]),
        "per_class": {label: precision_recall_f1(truth, pred, label) for label in [positive, negative]},
        "auc": binary_auc(truth, scores, positive),
        "confusion_matrix": confusion_matrix(truth, pred, [positive, negative]),
    }


def evaluate(rows: list[dict]) -> dict:
    strict_rows = collect_dataset(rows, strict=True)
    practical_rows = collect_dataset(rows, strict=False)
    split_rows = split_by_session(practical_rows)

    strict_pairwise: dict[str, dict] = {}
    for positive, negative in PAIRWISE_TASKS:
        strict_subset = [row for row in strict_rows if event_label_for_row(row) in {positive, negative}]
        if len(strict_subset) >= 2 and len(set(event_label_for_row(row) for row in strict_subset)) == 2:
            x = np.asarray([feature_vector(row) for row in strict_subset], dtype=np.float64)
            y = np.asarray([1 if event_label_for_row(row) == positive else 0 for row in strict_subset], dtype=np.float64)
            mean, std = standardize_fit(x)
            x = standardize_apply(x, mean, std)
            model = train_binary_logreg(x, y)
            scores = predict_binary_logreg(model, x)
            pred = np.asarray([positive if score >= 0.5 else negative for score in scores], dtype=object)
            truth = np.asarray([positive if event_label_for_row(row) == positive else negative for row in strict_subset], dtype=object)
            strict_pairwise[f"{positive}_vs_{negative}"] = {
                "train_count": len(strict_subset),
                "eval_count": len(strict_subset),
                "accuracy": accuracy_score(truth, pred),
                "macro": macro_metrics(truth, pred, [positive, negative]),
                "per_class": {label: precision_recall_f1(truth, pred, label) for label in [positive, negative]},
                "auc": binary_auc(truth, scores, positive),
            }
        else:
            strict_pairwise[f"{positive}_vs_{negative}"] = {
                "train_count": len(strict_subset),
                "eval_count": len(strict_subset),
                "accuracy": None,
                "macro": None,
                "per_class": {},
                "auc": None,
            }

    pairwise: dict[str, dict] = {}
    for positive, negative in PAIRWISE_TASKS:
        fold_results = []
        for held_out, train_rows, test_rows in split_rows:
            result = train_eval_binary(train_rows, test_rows, positive, negative)
            result["held_out_session"] = held_out
            fold_results.append(result)
        pairwise[f"{positive}_vs_{negative}"] = {
            "folds": fold_results,
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in fold_results])) if fold_results else 0.0,
            "mean_auc": float(np.mean([fold["auc"] for fold in fold_results])) if fold_results else 0.5,
            "mean_macro_f1": float(np.mean([fold["macro"]["f1"] for fold in fold_results])) if fold_results else 0.0,
            "mean_precision": float(np.mean([fold["macro"]["precision"] for fold in fold_results])) if fold_results else 0.0,
            "mean_recall": float(np.mean([fold["macro"]["recall"] for fold in fold_results])) if fold_results else 0.0,
        }

    four_class_rows = [row for row in practical_rows if event_label_for_row(row) in LABELS]
    fold_results = []
    for held_out, train_rows, test_rows in split_rows:
        train_subset = [row for row in train_rows if event_label_for_row(row) in LABELS]
        test_subset = [row for row in test_rows if event_label_for_row(row) in LABELS]
        if len(set(event_label_for_row(row) for row in train_subset)) < 2 or len(test_subset) == 0:
            continue
        fold_results.append({"held_out_session": held_out, **train_eval_four_class(train_subset, test_subset, LABELS)})

    if fold_results:
        all_truth = np.asarray([item for fold in fold_results for item in fold["truth"]], dtype=object)
        all_pred = np.asarray([item for fold in fold_results for item in fold["predictions"]], dtype=object)
        four_class_summary = {
            "available": True,
            "folds": fold_results,
            "macro": macro_metrics(all_truth, all_pred, LABELS),
            "confusion_matrix": confusion_matrix(all_truth, all_pred, LABELS),
            "per_class": {label: precision_recall_f1(all_truth, all_pred, label) for label in LABELS},
        }
    else:
        four_class_summary = {"available": False, "folds": [], "macro": None, "confusion_matrix": None, "per_class": {}}

    return {
        "dataset": {
            "total_rows": len(rows),
            "strict_rows": len(strict_rows),
            "practical_rows": len(practical_rows),
            "row_counts_by_event_label": dict(Counter(event_label_for_row(row) or "None" for row in practical_rows)),
            "row_counts_by_event_label_provenance": dict(Counter(row.get("final_human_event_label_provenance") or row.get("event_label_provenance") or "None" for row in practical_rows)),
            "row_counts_by_session_type": dict(Counter(row["session_type"] for row in practical_rows)),
            "row_counts_by_session_type_provenance": dict(Counter(row["session_type_provenance"] for row in practical_rows)),
            "strict_note": "direct_review session-type provenance only; this subset is too small for a four-class fit",
            "practical_note": "uses final human event labels when present, otherwise suggestion labels; missing labels are retained in summaries",
        },
        "split_strategy": {
            "type": "leave_one_session_out",
            "sessions": sorted(set(row["source_session_id"] for row in practical_rows)),
            "held_out_folds": len(split_rows),
        },
        "strict_subset": {
            "note": "direct_review session-type provenance only; useful only for the springboard contrast and not enough for a full four-class fit",
            "pairwise": strict_pairwise,
        },
        "feature_set": [
            "event_window_duration",
            "event_anchor_timestamp_seconds",
            "proposal_timestamp_seconds",
            "anchor_minus_proposal_abs",
            "raw_proposal_score",
            "audio_score",
            "combined_score",
            "audio_model_probability",
            "audio_clip_probability",
            "threshold_passed",
            "is_false_negative_window",
            "session_type_is_springboard",
            "session_type_is_platform",
            "event_label_provenance_is_session_type_inferred",
            "event_label_provenance_is_subtype_mapped",
            "event_label_provenance_is_uncertain",
        ],
        "supervision_source": {
            "primary": "final_human_event_label",
            "fallback": "suggested_event_label",
        },
        "pairwise": pairwise,
        "four_class": four_class_summary,
        "strongest_confusions": _strongest_confusions(four_class_summary, pairwise),
        "decision": _decision(pairwise, four_class_summary),
    }


def _strongest_confusions(four_class_summary: dict, pairwise: dict) -> list[dict]:
    confusions: list[dict] = []
    for task_name, task in pairwise.items():
        if not task["folds"]:
            continue
        worst_fold = min(task["folds"], key=lambda fold: fold["accuracy"])
        confusions.append(
            {
                "task": task_name,
                "held_out_session": worst_fold["held_out_session"],
                "accuracy": worst_fold["accuracy"],
                "confusion_matrix": worst_fold["confusion_matrix"],
            }
        )
    if four_class_summary.get("available") and four_class_summary.get("confusion_matrix") is not None:
        confusions.append(
            {
                "task": "four_class",
                "confusion_matrix": four_class_summary["confusion_matrix"],
                "macro_f1": four_class_summary["macro"]["f1"],
            }
        )
    return confusions


def _decision(pairwise: dict, four_class_summary: dict) -> str:
    springboard = pairwise["springboard_dive_vs_springboard_rebound_only"]
    platform = pairwise["platform_dive_vs_noise_or_other"]
    if springboard["mean_auc"] >= 0.75 and platform["mean_auc"] >= 0.70:
        if not four_class_summary.get("available") or four_class_summary["macro"]["f1"] >= 0.45:
            return "GO_PHASE_5"
    return "STOP_AND_ADDRESS_LABEL_GAPS"


def write_report(path_json: Path, path_md: Path, report: dict) -> None:
    path_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Event Classifier Baseline",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Dataset",
        "",
        f"- total rows: `{report['dataset']['total_rows']}`",
        f"- strict rows: `{report['dataset']['strict_rows']}`",
        f"- practical rows: `{report['dataset']['practical_rows']}`",
        f"- row counts by class: `{json.dumps(report['dataset']['row_counts_by_event_label'], sort_keys=True)}`",
        f"- row counts by provenance: `{json.dumps(report['dataset']['row_counts_by_event_label_provenance'], sort_keys=True)}`",
        f"- session type counts: `{json.dumps(report['dataset']['row_counts_by_session_type'], sort_keys=True)}`",
        f"- session type provenance counts: `{json.dumps(report['dataset']['row_counts_by_session_type_provenance'], sort_keys=True)}`",
        "",
        "## Split",
        "",
        f"- strategy: `{report['split_strategy']['type']}`",
        f"- sessions: `{', '.join(report['split_strategy']['sessions'])}`",
        f"- folds: `{report['split_strategy']['held_out_folds']}`",
        "",
        "## Strict Subset",
        "",
        f"- note: {report['strict_subset']['note']}",
        "",
        "## Supervision",
        "",
        f"- primary: `{report['supervision_source']['primary']}`",
        f"- fallback: `{report['supervision_source']['fallback']}`",
        "",
        "## Baseline",
        "",
        "- model: `numpy logistic regression`",
        "- features: event-window duration, anchor/proposal offsets, legacy detector scores, and provenance/session flags",
        "",
        "## Pairwise Results",
        "",
    ]
    for task_name, task in report["strict_subset"]["pairwise"].items():
        lines.extend(
            [
                f"### {task_name} (strict)",
                f"- train/eval rows: `{task['train_count']}`",
                f"- accuracy: `{task['accuracy']}`",
                f"- AUC: `{task['auc']}`",
                "",
            ]
        )
    for task_name, task in report["pairwise"].items():
        lines.extend(
            [
                f"### {task_name}",
                f"- mean AUC: `{task['mean_auc']:.4f}`",
                f"- mean accuracy: `{task['mean_accuracy']:.4f}`",
                f"- mean macro F1: `{task['mean_macro_f1']:.4f}`",
                f"- mean precision: `{task['mean_precision']:.4f}`",
                f"- mean recall: `{task['mean_recall']:.4f}`",
                "",
            ]
        )

    lines.extend(["## Four-Class", ""])
    if report["four_class"]["available"]:
        lines.extend(
            [
                f"- macro precision: `{report['four_class']['macro']['precision']:.4f}`",
                f"- macro recall: `{report['four_class']['macro']['recall']:.4f}`",
                f"- macro F1: `{report['four_class']['macro']['f1']:.4f}`",
                f"- confusion matrix: `{json.dumps(report['four_class']['confusion_matrix'])}`",
                "",
                "Per-class metrics:",
            ]
        )
        for label, metrics in report["four_class"]["per_class"].items():
            lines.append(
                f"- `{label}`: precision `{metrics['precision']:.4f}`, recall `{metrics['recall']:.4f}`, f1 `{metrics['f1']:.4f}`, support `{metrics['support']}`"
            )
        lines.append("")
    else:
        lines.append("- not enough train/test support for a stable four-class fit")
        lines.append("")

    lines.extend(
        [
            "## Strongest Confusions",
            "",
        ]
    )
    for item in report["strongest_confusions"]:
        lines.append(f"- `{item['task']}`")
        if "held_out_session" in item:
            lines.append(f"  - held out: `{item['held_out_session']}`")
        if "accuracy" in item:
            lines.append(f"  - accuracy: `{item['accuracy']:.4f}`")
        if "macro_f1" in item:
            lines.append(f"  - macro F1: `{item['macro_f1']:.4f}`")
        lines.append(f"  - confusion matrix: `{json.dumps(item['confusion_matrix'])}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `{report['decision']}`",
        ]
    )
    path_md.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an offline event-window baseline.")
    parser.add_argument("--manifest", nargs="+", default=[str(DEFAULT_MANIFEST)])
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)
    rows: list[dict] = []
    for manifest_path in args.manifest:
        rows.extend(load_rows(Path(manifest_path)))
    report = evaluate(rows)
    write_report(Path(args.output_json), Path(args.output_md), report)
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
