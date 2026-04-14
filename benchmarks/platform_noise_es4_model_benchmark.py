from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATASET_NPZ = Path("outputs/platform_noise_es4_dataset.npz")
DATASET_SUMMARY_JSON = Path("outputs/platform_noise_es4_dataset_summary.json")
OUT_JSON = Path("outputs/platform_noise_es4_model_benchmark.json")
OUT_MD = Path("outputs/platform_noise_es4_model_benchmark.md")


@dataclass
class HoldoutResult:
    auc: float
    macro_f1: float
    accuracy: float
    platform_recall: float
    noise_recall: float
    confusion_matrix: list[list[int]]
    noise_to_platform_fp: int
    platform_to_noise_fn: int


class NumpyLogisticBinary(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, epochs: int = 1000, lr: float = 0.1, l2: float = 0.01):
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticBinary":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_features_in_ = n_features
        for _ in range(self.epochs):
            logits = x @ self.weights_ + self.bias_
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            error = probs - y
            grad_w = (x.T @ error) / n_samples + self.l2 * self.weights_
            grad_b = float(np.mean(error))
            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model has not been fit yet.")
        x = np.asarray(x, dtype=np.float64)
        logits = x @ self.weights_ + self.bias_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)[:, 1]
        return (probs >= 0.5).astype(np.int64)

    def _more_tags(self) -> dict[str, Any]:
        return {"binary_only": True}


def evaluate_holdout(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_holdout: np.ndarray, y_holdout: np.ndarray) -> HoldoutResult:
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_holdout)[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y_holdout, probs))
    macro_f1 = float(f1_score(y_holdout, pred, average="macro"))
    acc = float(accuracy_score(y_holdout, pred))
    platform_recall = float(recall_score(y_holdout, pred, pos_label=1))
    noise_recall = float(recall_score(y_holdout, pred, pos_label=0))
    cm = confusion_matrix(y_holdout, pred, labels=[1, 0]).tolist()
    # cm structure with labels=[1,0] => [[tp, fn], [fp, tn]]
    noise_to_platform_fp = int(cm[1][0])
    platform_to_noise_fn = int(cm[0][1])
    return HoldoutResult(
        auc=auc,
        macro_f1=macro_f1,
        accuracy=acc,
        platform_recall=platform_recall,
        noise_recall=noise_recall,
        confusion_matrix=cm,
        noise_to_platform_fp=noise_to_platform_fp,
        platform_to_noise_fn=platform_to_noise_fn,
    )


def stability_scores(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    auc_scores: list[float] = []
    macro_f1_scores: list[float] = []
    acc_scores: list[float] = []
    for train_idx, test_idx in cv.split(x_train, y_train):
        x_tr, x_te = x_train[train_idx], x_train[test_idx]
        y_tr, y_te = y_train[train_idx], y_train[test_idx]
        fitted = model.fit(x_tr, y_tr)
        probs = fitted.predict_proba(x_te)[:, 1]
        pred = (probs >= 0.5).astype(np.int64)
        auc_scores.append(float(roc_auc_score(y_te, probs)))
        macro_f1_scores.append(float(f1_score(y_te, pred, average="macro")))
        acc_scores.append(float(accuracy_score(y_te, pred)))
    return {
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_auc_std": float(np.std(auc_scores)),
        "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(macro_f1_scores)),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
    }


def to_dict(hr: HoldoutResult) -> dict[str, Any]:
    return {
        "auc": hr.auc,
        "macro_f1": hr.macro_f1,
        "accuracy": hr.accuracy,
        "platform_recall": hr.platform_recall,
        "noise_recall": hr.noise_recall,
        "confusion_matrix": hr.confusion_matrix,
        "noise_to_platform_fp": hr.noise_to_platform_fp,
        "platform_to_noise_fn": hr.platform_to_noise_fn,
    }


def write_md(report: dict[str, Any]) -> None:
    comps = report["model_comparison"]
    lines = [
        "# Platform/Noise ES4 Model-Family Benchmark",
        "",
        f"- decision: `{report['decision']}`",
        f"- rationale: `{report['decision_rationale']}`",
        "",
        "## Frozen benchmark scope",
        "",
        f"- train rows: `{report['scope']['train_rows']}`",
        f"- scored validation rows: `{report['scope']['scored_validation_rows']}`",
        f"- detector/taxonomy/labels/springboard unchanged: `{report['scope']['frozen_constraints_respected']}`",
        "",
        "## Model comparison (final scored validation slice)",
        "",
        "| model | AUC | macro F1 | accuracy | platform recall | noise recall | confusion matrix | noise->platform FP | platform->noise FN |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for key in ["numpy_logistic_reference", "sklearn_logistic_l2", "xgboost_gbdt"]:
        row = comps[key]["holdout"]
        lines.append(
            f"| {key} | {row['auc']:.4f} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['platform_recall']:.4f} | {row['noise_recall']:.4f} | `{row['confusion_matrix']}` | {row['noise_to_platform_fp']} | {row['platform_to_noise_fn']} |"
        )
    lines.extend(
        [
            "",
            "## Training-side stability (repeated CV on train split only)",
            "",
            "| model | CV AUC mean±std | CV macro F1 mean±std | CV accuracy mean±std |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ["numpy_logistic_reference", "sklearn_logistic_l2", "xgboost_gbdt"]:
        st = comps[key]["train_cv"]
        lines.append(
            f"| {key} | {st['cv_auc_mean']:.4f}±{st['cv_auc_std']:.4f} | {st['cv_macro_f1_mean']:.4f}±{st['cv_macro_f1_std']:.4f} | {st['cv_accuracy_mean']:.4f}±{st['cv_accuracy_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best candidate",
            "",
            f"- model: `{report['best_candidate']['model']}`",
            f"- why: `{report['best_candidate']['why']}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    data = np.load(DATASET_NPZ, allow_pickle=True)
    dataset_summary = json.loads(DATASET_SUMMARY_JSON.read_text())
    x_train = data["x_train"].astype(np.float64)
    y_train = data["y_train"].astype(np.int64)
    x_holdout = data["x_holdout"].astype(np.float64)
    y_holdout = data["y_holdout"].astype(np.int64)

    numpy_logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", NumpyLogisticBinary(epochs=1000, lr=0.1, l2=0.01)),
        ]
    )

    sklearn_logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    xgboost_gbdt = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )

    model_specs = {
        "numpy_logistic_reference": numpy_logistic,
        "sklearn_logistic_l2": sklearn_logistic,
        "xgboost_gbdt": xgboost_gbdt,
    }

    model_comparison: dict[str, dict[str, Any]] = {}
    for name, model in model_specs.items():
        hold = evaluate_holdout(model, x_train, y_train, x_holdout, y_holdout)
        stability = stability_scores(model, x_train, y_train)
        model_comparison[name] = {
            "holdout": to_dict(hold),
            "train_cv": stability,
        }

    baseline = model_comparison["numpy_logistic_reference"]["holdout"]
    candidates = []
    for name in ["sklearn_logistic_l2", "xgboost_gbdt"]:
        hold = model_comparison[name]["holdout"]
        score = (
            hold["auc"],
            hold["macro_f1"],
            hold["platform_recall"],
            -hold["noise_to_platform_fp"],
            -hold["platform_to_noise_fn"],
        )
        candidates.append((score, name))
    best_name = sorted(candidates, reverse=True)[0][1]
    best = model_comparison[best_name]["holdout"]

    pass_guardrails = (
        best["auc"] >= max(0.66, baseline["auc"] + 0.02)
        and best["macro_f1"] >= 0.50
        and best["platform_recall"] >= 0.75
        and best["noise_to_platform_fp"] <= baseline["noise_to_platform_fp"]
    )
    no_collapse = best["platform_recall"] >= 0.60
    decision = "ES4_PASS_PROMOTE_TO_PHASE5_RERUN" if (pass_guardrails and no_collapse) else "ES4_FAIL_KEEP_SEARCHING"

    if decision == "ES4_PASS_PROMOTE_TO_PHASE5_RERUN":
        rationale = (
            f"{best_name} improved ranking and practical tradeoff on the frozen scored slice while meeting guardrail-oriented checks "
            "(AUC, macro F1, platform recall floor, and no worse noise->platform FP than baseline)."
        )
    else:
        rationale = (
            f"No tested ES4 candidate met the bounded promotion rule. Best candidate ({best_name}) did not simultaneously satisfy "
            "ranking + guardrail-feasible practical tradeoff on the frozen scored validation slice."
        )

    report = {
        "scope": {
            "train_rows": int(dataset_summary["row_counts"]["train_rows"]),
            "scored_validation_rows": int(dataset_summary["row_counts"]["scored_validation_rows"]),
            "frozen_constraints_respected": True,
            "models_compared": list(model_specs.keys()),
            "bounded_search": True,
            "large_grid_search_performed": False,
        },
        "model_comparison": model_comparison,
        "best_candidate": {
            "model": best_name,
            "why": (
                f"Selected by holdout ranking/guardrail priority tuple (AUC, macro F1, platform recall, lower noise->platform FP, lower platform->noise FN). "
                f"Holdout metrics: AUC={best['auc']:.4f}, macro F1={best['macro_f1']:.4f}, "
                f"platform recall={best['platform_recall']:.4f}, noise->platform FP={best['noise_to_platform_fp']}."
            ),
        },
        "decision": decision,
        "decision_rationale": rationale,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_md(report)


if __name__ == "__main__":
    main()
