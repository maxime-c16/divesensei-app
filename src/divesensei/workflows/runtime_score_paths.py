from __future__ import annotations

import json
import importlib.util
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from divesensei.detection.audio_model import MODEL_FEATURES, AudioCandidateModel


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_MODEL_DIR = ROOT / ".divesensei-runtime" / "models"
GOVERNED_R9_MODEL_PATH = RUNTIME_MODEL_DIR / "governed_r9_audio_candidate_model.json"
VISUAL_LATE_FUSION_MODEL_PATH = RUNTIME_MODEL_DIR / "visual_late_fusion_logreg_c0.5.json"
EXACT_R9_MODEL_DIR = RUNTIME_MODEL_DIR / "r9_compact_nuisance_weighted"
EXACT_R9_MODEL_PATH = EXACT_R9_MODEL_DIR / "xgboost_model.json"
EXACT_R9_CONTRACT_PATH = EXACT_R9_MODEL_DIR / "contract.json"
PHASE5_MODULE_PATH = ROOT / "benchmarks" / "phase5_regime_aware_execution_r7_es4.py"
NUISANCE_MODULE_PATH = ROOT / "benchmarks" / "post_noise_nuisance_family_benchmark.py"


@dataclass
class RuntimeLogisticModel:
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    bias: float
    training_rows: int
    positive_rows: int
    negative_rows: int

    @classmethod
    def load(cls, path: Path) -> "RuntimeLogisticModel":
        data = json.loads(path.read_text())
        return cls(
            feature_names=[str(name) for name in data["feature_names"]],
            means=np.asarray(data["means"], dtype=np.float64),
            stds=np.asarray(data["stds"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=float(data["bias"]),
            training_rows=int(data.get("training_rows", 0)),
            positive_rows=int(data.get("positive_rows", 0)),
            negative_rows=int(data.get("negative_rows", 0)),
        )

    def predict_probability(self, feature_map: dict[str, float]) -> float:
        values = np.asarray([float(feature_map.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)
        normalized = (values - self.means) / np.maximum(self.stds, 1e-6)
        logit = float(np.dot(normalized, self.weights) + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "training_rows": int(self.training_rows),
            "positive_rows": int(self.positive_rows),
            "negative_rows": int(self.negative_rows),
        }


def _fit_runtime_logistic(features: np.ndarray, labels: np.ndarray, *, epochs: int = 2500, lr: float = 0.06) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    means = features.mean(axis=0)
    stds = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - means) / stds
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = normalized @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        error = preds - labels
        weights -= lr * ((normalized.T @ error) / len(normalized))
        bias -= lr * float(np.mean(error))
    return means, stds, weights, bias


def _reviewed_session_roots() -> list[Path]:
    roots = []
    for session_dir in sorted((ROOT / "outputs").glob("evaluation_*")):
        if (session_dir / "ui_session_manifest.json").exists() and (session_dir / "evaluation_review.json").exists():
            roots.append(session_dir)
    return roots


def _read_reviewed_detection_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_dir in _reviewed_session_roots():
        manifest = json.loads((session_dir / "ui_session_manifest.json").read_text())
        review = json.loads((session_dir / "evaluation_review.json").read_text())
        decisions = {
            str(item.get("detectionId")): item
            for item in review.get("decisions", [])
            if item.get("label") in {"dive", "non_dive"}
        }
        source_video_path = str(manifest.get("session", {}).get("source_video_path") or "")
        for detection in manifest.get("detections", []):
            detection_id = str(detection.get("id"))
            decision = decisions.get(detection_id)
            if decision is None:
                continue
            rows.append(
                {
                    "session_id": session_dir.name,
                    "source_video_path": source_video_path,
                    "timestamp_seconds": float(detection.get("timestamp_seconds") or 0.0),
                    "label": "platform_dive" if decision.get("label") == "dive" else "noise_or_other",
                    "scores": dict(detection.get("scores", {}) or {}),
                    "features": dict(detection.get("features", {}) or {}),
                }
            )
    return rows


def _audio_feature_vector(scores: dict[str, Any], features: dict[str, Any]) -> dict[str, float]:
    return {
        "audio_score": float(scores.get("audio", 0.0) or 0.0),
        "spectral_flux": float(features.get("spectral_flux", 0.0) or 0.0),
        "rms": float(features.get("rms", 0.0) or 0.0),
        "hf_ratio": float(features.get("hf_ratio", 0.0) or 0.0),
        "spectral_centroid_hz": float(features.get("spectral_centroid_hz", 0.0) or 0.0),
        "spectral_flatness": float(features.get("spectral_flatness", 0.0) or 0.0),
        "post_flux_ratio": float(features.get("post_flux_ratio", 0.0) or 0.0),
        "post_rms_ratio": float(features.get("post_rms_ratio", 0.0) or 0.0),
        "local_prominence": float(features.get("local_prominence", 0.0) or 0.0),
        "nearby_peaks_8s": float(features.get("nearby_peaks_8s", 0.0) or 0.0),
    }


def _ensure_governed_r9_model() -> tuple[AudioCandidateModel, dict[str, Any]]:
    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if GOVERNED_R9_MODEL_PATH.exists():
        model = AudioCandidateModel.load(GOVERNED_R9_MODEL_PATH)
        meta = json.loads(GOVERNED_R9_MODEL_PATH.read_text())
        return model, {
            "model_path": str(GOVERNED_R9_MODEL_PATH),
            "bootstrapped": False,
            "training_rows": int(meta.get("training_rows", 0)),
            "positive_rows": int(meta.get("positive_rows", 0)),
            "negative_rows": int(meta.get("negative_rows", 0)),
        }

    reviewed_rows = _read_reviewed_detection_rows()
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    for row in reviewed_rows:
        feature_map = _audio_feature_vector(row["scores"], row["features"])
        feature_rows.append([float(feature_map[name]) for name in MODEL_FEATURES])
        labels.append(1.0 if row["label"] == "platform_dive" else 0.0)
    features = np.asarray(feature_rows, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    means, stds, weights, bias = _fit_runtime_logistic(features, labels_arr)
    payload = {
        "feature_names": MODEL_FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "training_rows": int(len(features)),
        "positive_rows": int(np.sum(labels_arr)),
        "negative_rows": int(len(labels_arr) - np.sum(labels_arr)),
    }
    GOVERNED_R9_MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return AudioCandidateModel.load(GOVERNED_R9_MODEL_PATH), {
        "model_path": str(GOVERNED_R9_MODEL_PATH),
        "bootstrapped": True,
        "training_rows": int(payload["training_rows"]),
        "positive_rows": int(payload["positive_rows"]),
        "negative_rows": int(payload["negative_rows"]),
    }


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_exact_governed_r9_model() -> tuple[Any | None, dict[str, Any]]:
    if not EXACT_R9_MODEL_PATH.exists() or not EXACT_R9_CONTRACT_PATH.exists():
        return None, {
            "status": "missing_artifact",
            "model_path": str(EXACT_R9_MODEL_PATH),
            "contract_path": str(EXACT_R9_CONTRACT_PATH),
        }
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as exc:
        return None, {
            "status": "xgboost_unavailable",
            "model_path": str(EXACT_R9_MODEL_PATH),
            "contract_path": str(EXACT_R9_CONTRACT_PATH),
            "error": str(exc),
        }
    model = XGBClassifier()
    model.load_model(EXACT_R9_MODEL_PATH)
    contract = json.loads(EXACT_R9_CONTRACT_PATH.read_text())
    return model, {
        "status": "loaded",
        "model_path": str(EXACT_R9_MODEL_PATH),
        "contract_path": str(EXACT_R9_CONTRACT_PATH),
        "contract": contract,
    }


def _score_candidates_with_exact_governed_r9(
    *,
    candidates: list[Any],
    source_video_path: Path,
) -> tuple[dict[int, float], dict[str, Any]]:
    model, meta = _load_exact_governed_r9_model()
    if model is None:
        return {}, meta
    try:
        phase5 = _load_module_from_path("phase5_runtime_score_paths_exact", PHASE5_MODULE_PATH)
        nuisance = _load_module_from_path("nuisance_runtime_score_paths_exact", NUISANCE_MODULE_PATH)
        audio = phase5.decode_audio_mono(source_video_path, phase5.SAMPLE_RATE)
    except Exception as exc:
        return {}, {**meta, "status": "feature_runtime_unavailable", "error": str(exc)}

    scores: dict[int, float] = {}
    feature_rows: list[list[float]] = []
    candidate_ids: list[int] = []
    failures = 0
    for candidate in candidates:
        try:
            details = dict(getattr(candidate, "details", {}) or {})
            timestamp = float(getattr(candidate, "timestamp", 0.0) or 0.0)
            start = max(0.0, float(getattr(candidate, "start_time", max(0.0, timestamp - 0.75)) or 0.0))
            end = max(start + 0.05, float(getattr(candidate, "end_time", timestamp + 2.25) or timestamp + 2.25))
            signal = audio[int(round(start * phase5.SAMPLE_RATE)) : int(round(end * phase5.SAMPLE_RATE))]
            fmap = {
                "runtime": {
                    **phase5.extract_features(signal, phase5.SAMPLE_RATE),
                    **nuisance.nuisance_features(phase5, signal, phase5.SAMPLE_RATE),
                }
            }
            row = {
                "audio_score": details.get("audio_score", getattr(candidate, "audio_score", 0.0)),
                "audio_clip_probability": details.get("audio_clip_probability"),
                "event_anchor_timestamp_seconds": timestamp,
                "is_false_negative_window": False,
            }
            ref = nuisance.RowRef("runtime", "unknown", row)
            feature_rows.append(nuisance.vector_for(phase5, ref, fmap, nuisance.NOISE_BOUNDARY_COMPACT))
            candidate_ids.append(id(candidate))
        except Exception:
            failures += 1
    if feature_rows:
        probs = model.predict_proba(np.asarray(feature_rows, dtype=np.float64))[:, 1]
        scores.update({candidate_id: float(prob) for candidate_id, prob in zip(candidate_ids, probs)})
    return scores, {
        **meta,
        "status": "scored",
        "candidate_count": len(candidates),
        "scored_count": len(scores),
        "feature_failure_count": failures,
        "feature_contract_source": [
            str(PHASE5_MODULE_PATH),
            str(NUISANCE_MODULE_PATH),
        ],
        "window_contract_runtime": "candidate.start_time_seconds to candidate.end_time_seconds",
    }


def _motion_features_for_timestamp(video_path: Path, timestamp_seconds: float) -> dict[str, float] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None
    event_frame = int(round(timestamp_seconds * fps))
    pre_seconds = 1.25
    post_seconds = 1.25
    start_frame = max(0, event_frame - int(pre_seconds * fps))
    end_frame = min(total_frames - 1, event_frame + int(post_seconds * fps))
    frame_step = max(1, int(round(fps / 8.0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    splash_motion: list[float] = []
    diver_motion: list[float] = []
    prev_splash: np.ndarray | None = None
    prev_diver: np.ndarray | None = None

    source_idx = start_frame
    while source_idx <= end_frame:
        ok = cap.grab()
        if not ok:
            break
        if (source_idx - start_frame) % frame_step != 0:
            source_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            source_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        splash_top = int(height * 0.72)
        splash_bottom = int(height * 0.95)
        diver_top = int(height * 0.15)
        diver_bottom = int(height * 0.72)
        splash_region = gray[splash_top:splash_bottom, 0:width]
        diver_region = gray[diver_top:diver_bottom, 0:width]
        if prev_splash is None or prev_diver is None:
            splash_motion.append(0.0)
            diver_motion.append(0.0)
        else:
            splash_delta = cv2.absdiff(splash_region, prev_splash)
            diver_delta = cv2.absdiff(diver_region, prev_diver)
            splash_motion.append(float(np.mean(splash_delta)))
            diver_motion.append(float(np.mean(diver_delta)))
        prev_splash = splash_region
        prev_diver = diver_region
        source_idx += 1
    cap.release()

    if len(splash_motion) < 8:
        return None
    splash_arr = np.asarray(splash_motion, dtype=np.float64)
    diver_arr = np.asarray(diver_motion, dtype=np.float64)
    event_idx = int(np.clip(round((event_frame - start_frame) / frame_step), 1, len(splash_arr) - 2))
    pre_splash = splash_arr[:event_idx]
    post_splash = splash_arr[event_idx:]
    pre_diver = diver_arr[:event_idx]
    if len(pre_splash) == 0 or len(post_splash) == 0 or len(pre_diver) == 0:
        return None

    splash_peak = float(np.max(splash_arr[max(0, event_idx - 3): min(len(splash_arr), event_idx + 5)]))
    splash_pre_med = float(np.median(pre_splash))
    splash_post_med = float(np.median(post_splash))
    diver_pre_peak = float(np.max(pre_diver[max(0, len(pre_diver) - 8):])) if len(pre_diver) else 0.0
    diver_pre_med = float(np.median(pre_diver))
    return {
        "visual_splash_peak": splash_peak,
        "visual_splash_pre_median": splash_pre_med,
        "visual_splash_post_median": splash_post_med,
        "visual_diver_pre_peak": diver_pre_peak,
        "visual_diver_pre_median": diver_pre_med,
        "visual_splash_ratio_pre": float(splash_peak / max(splash_pre_med, 1e-6)),
        "visual_splash_ratio_post": float(splash_peak / max(splash_post_med, 1e-6)),
        "visual_diver_ratio_pre": float(diver_pre_peak / max(diver_pre_med, 1e-6)),
    }


def _balanced_rows(rows: list[dict[str, Any]], *, max_per_class: int = 160) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {"platform_dive": [], "noise_or_other": []}
    for row in rows:
        label = str(row.get("label"))
        if label in by_label:
            by_label[label].append(row)
    out: list[dict[str, Any]] = []
    for label in ("platform_dive", "noise_or_other"):
        out.extend(by_label[label][:max_per_class])
    return out


def _ensure_visual_late_fusion_model(governed_model: AudioCandidateModel) -> tuple[RuntimeLogisticModel | None, dict[str, Any]]:
    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if VISUAL_LATE_FUSION_MODEL_PATH.exists():
        model = RuntimeLogisticModel.load(VISUAL_LATE_FUSION_MODEL_PATH)
        return model, {
            "model_path": str(VISUAL_LATE_FUSION_MODEL_PATH),
            "bootstrapped": False,
            "training_rows": int(model.training_rows),
            "positive_rows": int(model.positive_rows),
            "negative_rows": int(model.negative_rows),
        }

    reviewed = _balanced_rows(_read_reviewed_detection_rows(), max_per_class=160)
    visual_feature_names = [
        "visual_splash_peak",
        "visual_splash_pre_median",
        "visual_splash_post_median",
        "visual_diver_pre_peak",
        "visual_diver_pre_median",
        "visual_splash_ratio_pre",
        "visual_splash_ratio_post",
        "visual_diver_ratio_pre",
        "governed_r9_score",
    ]
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    skipped = 0
    for row in reviewed:
        video_path = Path(str(row.get("source_video_path") or ""))
        if not video_path.exists():
            skipped += 1
            continue
        motion = _motion_features_for_timestamp(video_path, float(row["timestamp_seconds"]))
        if motion is None:
            skipped += 1
            continue
        audio_map = _audio_feature_vector(row["scores"], row["features"])
        governed_score = governed_model.predict_probability(audio_map)
        motion_with_audio = dict(motion)
        motion_with_audio["governed_r9_score"] = float(governed_score)
        x_rows.append([float(motion_with_audio[name]) for name in visual_feature_names])
        y_rows.append(1.0 if row["label"] == "platform_dive" else 0.0)

    if len(x_rows) < 30:
        return None, {
            "model_path": str(VISUAL_LATE_FUSION_MODEL_PATH),
            "bootstrapped": False,
            "training_rows": len(x_rows),
            "positive_rows": int(sum(y_rows)),
            "negative_rows": int(len(y_rows) - sum(y_rows)),
            "skipped_rows": skipped,
            "blocked_reason": "insufficient_visual_training_rows",
        }

    features = np.asarray(x_rows, dtype=np.float64)
    labels = np.asarray(y_rows, dtype=np.float64)
    means, stds, weights, bias = _fit_runtime_logistic(features, labels)
    model = RuntimeLogisticModel(
        feature_names=visual_feature_names,
        means=means,
        stds=stds,
        weights=weights,
        bias=float(bias),
        training_rows=int(len(features)),
        positive_rows=int(np.sum(labels)),
        negative_rows=int(len(labels) - np.sum(labels)),
    )
    VISUAL_LATE_FUSION_MODEL_PATH.write_text(json.dumps(model.to_json(), indent=2), encoding="utf-8")
    return model, {
        "model_path": str(VISUAL_LATE_FUSION_MODEL_PATH),
        "bootstrapped": True,
        "training_rows": int(model.training_rows),
        "positive_rows": int(model.positive_rows),
        "negative_rows": int(model.negative_rows),
        "skipped_rows": skipped,
    }


def enrich_candidates_with_runtime_scores(
    *,
    candidates: list[Any],
    source_video_path: Path,
) -> dict[str, Any]:
    governed_model, governed_meta = _ensure_governed_r9_model()
    exact_scores, exact_meta = _score_candidates_with_exact_governed_r9(candidates=candidates, source_video_path=source_video_path)
    visual_model, visual_meta = _ensure_visual_late_fusion_model(governed_model)

    visual_missing = 0
    visual_present = 0
    governed_nonzero = 0
    source_exists = source_video_path.exists()

    for candidate in candidates:
        details = dict(getattr(candidate, "details", {}) or {})
        proxy_feature_map = _audio_feature_vector(
            {"audio": details.get("audio_score", getattr(candidate, "audio_score", 0.0))},
            details,
        )
        proxy_score = float(governed_model.predict_probability(proxy_feature_map))
        governed_score = float(exact_scores.get(id(candidate), proxy_score))
        details["governed_r9_score"] = governed_score
        details["governed_r9_score_source"] = "exact_governed_r9_xgboost" if id(candidate) in exact_scores else "runtime_bootstrap_proxy_fallback"
        details["governed_r9_proxy_score"] = proxy_score
        # Keep compatibility with historical UI paths still reading audio_model_probability.
        details["audio_model_probability"] = governed_score
        if governed_score > 0.0:
            governed_nonzero += 1

        visual_score: float | None = None
        if source_exists and visual_model is not None:
            motion = _motion_features_for_timestamp(source_video_path, float(getattr(candidate, "timestamp", 0.0)))
            if motion is not None:
                visual_feature_map = dict(motion)
                visual_feature_map["governed_r9_score"] = governed_score
                visual_score = float(visual_model.predict_probability(visual_feature_map))
        if visual_score is None:
            visual_missing += 1
        else:
            visual_present += 1
        details["visual_late_fusion_logreg_c0.5"] = visual_score
        setattr(candidate, "details", details)

    return {
        "candidate_count": len(candidates),
        "governed_r9_nonzero_count": governed_nonzero,
        "visual_present_count": visual_present,
        "visual_missing_count": visual_missing,
        "source_video_path_exists": source_exists,
        "governed_exact_model": exact_meta,
        "governed_model": governed_meta,
        "visual_model": visual_meta,
        "label_distribution_reference": dict(Counter(row["label"] for row in _read_reviewed_detection_rows())),
    }
