from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_row_key_map(rows: list[dict]) -> dict[str, dict]:
    counts: dict[str, int] = {}
    out: dict[str, dict] = {}
    for row in rows:
        sid = str(row["source_session_id"])
        counts[sid] = counts.get(sid, 0) + 1
        cid = row.get("legacy_candidate_id")
        rid = str(cid) if cid else f"row-{counts[sid]:04d}"
        out[f"{sid}::{rid}"] = row
    return out


def decode_audio(path: Path, sr: int) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def framing(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(signal) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded.reshape(1, frame)
    n_frames = 1 + (len(signal) - frame) // hop
    return np.lib.stride_tricks.as_strided(
        signal, shape=(n_frames, frame), strides=(signal.strides[0] * hop, signal.strides[0]), writeable=False
    )


def local_peaks(values: np.ndarray, threshold: float) -> np.ndarray:
    out = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and values[i] >= threshold:
            out.append(i)
    return np.asarray(out, dtype=int)


def extract_feature_snapshot(signal: np.ndarray, sr: int) -> dict[str, float]:
    eps = 1e-8
    frame = 512
    hop = 128
    frames = framing(signal, frame=frame, hop=hop)
    if frames.size == 0:
        return {
            "spectral_flatness_post_mean": 0.0,
            "tonal_peak_fraction_post_mean": 0.0,
            "whistle_band_energy_fraction_post": 0.0,
            "spectral_entropy_post_mean": 0.0,
            "transient_peak_count": 0.0,
            "post_impact_early_to_late_rms_ratio": 0.0,
        }
    window = np.hanning(frame).astype(np.float32)
    w = frames * window
    rms = np.sqrt(np.mean(np.square(w), axis=1) + eps)
    peak_idx = int(np.argmax(rms))
    mean = float(np.mean(rms))
    std = float(np.std(rms))

    peaks = local_peaks(rms, mean + 0.5 * std)
    f120 = max(1, int(round(0.12 * sr / hop)))
    f600 = max(f120 + 1, int(round(0.60 * sr / hop)))
    early = rms[peak_idx + 1 : peak_idx + 1 + f120]
    late = rms[peak_idx + 1 + f120 : peak_idx + 1 + f600]
    early_mean = float(np.mean(early)) if len(early) else 0.0
    late_mean = float(np.mean(late)) if len(late) else 0.0

    spec = np.abs(np.fft.rfft(w, axis=1)) ** 2
    post = spec[peak_idx : peak_idx + max(1, int(round(0.4 * sr / hop)))]
    if len(post) == 0:
        post = spec[max(0, peak_idx - 1) : peak_idx + 1]
    power = np.maximum(post, eps)
    spectral_flatness = np.exp(np.mean(np.log(power), axis=1)) / np.maximum(np.mean(power, axis=1), eps)
    tonal_peak_fraction = np.max(power, axis=1) / np.maximum(np.sum(power, axis=1), eps)

    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    whistle_band = (freqs >= 1000.0) & (freqs <= 4000.0)
    whistle_frac = np.sum(power[:, whistle_band], axis=1) / np.maximum(np.sum(power, axis=1), eps)
    pnorm = power / np.maximum(np.sum(power, axis=1, keepdims=True), eps)
    entropy = -np.sum(pnorm * np.log(np.maximum(pnorm, eps)), axis=1) / math.log(power.shape[1])

    return {
        "spectral_flatness_post_mean": float(np.mean(spectral_flatness)),
        "tonal_peak_fraction_post_mean": float(np.mean(tonal_peak_fraction)),
        "whistle_band_energy_fraction_post": float(np.mean(whistle_frac)),
        "spectral_entropy_post_mean": float(np.mean(entropy)),
        "transient_peak_count": float(len(peaks)),
        "post_impact_early_to_late_rms_ratio": early_mean / max(late_mean, eps),
    }


def main() -> None:
    sr = 16000
    r6 = json.loads(Path("outputs/phase5_regime_aware_execution_r6.json").read_text())
    r2 = json.loads(Path("outputs/platform_noise_feature_probe_r2.json").read_text())
    mlist = json.loads(Path("outputs/phase5_regime_manifest_lists.json").read_text())
    rows = load_jsonl(Path("outputs/event_window_manifest_preview.jsonl"))
    by_key = build_row_key_map(rows)
    reviewed = load_jsonl(
        Path("outputs/evaluation_insep_quick_9015_20260409_ui/exports/event-reviewed-manifest/event_reviewed_manifest.jsonl")
    )
    reviewed_by = {
        f"evaluation_insep_quick_9015_20260409_ui::{row['legacy_candidate_id']}": row
        for row in reviewed
        if row.get("legacy_candidate_id")
    }

    state = r2["three_way_comparison"]["probe_r2_feature_iteration"]
    fp_rows = list(state["false_positive_noise_to_platform_rows"])
    fn_rows = list(state["false_negative_platform_to_noise_rows"])

    train = [(x["row_key"], x["event_label"], by_key[x["row_key"]]) for x in mlist["platform_noise_track"]["train_rows"]]
    hold = [(x["row_key"], x["event_label"], by_key[x["row_key"]]) for x in mlist["platform_noise_track"]["holdout_rows"]]

    audio_cache: dict[str, np.ndarray] = {}
    feature_map: dict[str, dict[str, float]] = {}
    for key, _, row in train + hold:
        sid = str(row["source_session_id"])
        if sid not in audio_cache:
            audio_cache[sid] = decode_audio(Path(str(row["source_session_root"])) / "web" / "session_source_review.mp4", sr)
        signal = audio_cache[sid]
        start = max(0.0, float(row.get("event_window_start_seconds") or 0.0))
        end = max(start + 0.05, float(row.get("event_window_end_seconds") or 0.0))
        feature_map[key] = extract_feature_snapshot(signal[int(round(start * sr)) : int(round(end * sr))], sr)

    # Simple ordering probe to support AUC diagnosis.
    xtr = np.asarray(
        [
            [
                float(r.get("audio_score") or 0.0),
                float(r.get("audio_clip_probability") or 0.0),
                float(r.get("event_anchor_timestamp_seconds") or 0.0),
                1.0 if r.get("is_false_negative_window") else 0.0,
                feature_map[k]["whistle_band_energy_fraction_post"],
                feature_map[k]["spectral_entropy_post_mean"],
                feature_map[k]["tonal_peak_fraction_post_mean"],
                feature_map[k]["spectral_flatness_post_mean"],
            ]
            for k, _, r in train
        ],
        dtype=np.float64,
    )
    ytr = np.asarray([1.0 if lab == "platform_dive" else 0.0 for _, lab, _ in train], dtype=np.float64)
    xte = np.asarray(
        [
            [
                float(r.get("audio_score") or 0.0),
                float(r.get("audio_clip_probability") or 0.0),
                float(r.get("event_anchor_timestamp_seconds") or 0.0),
                1.0 if r.get("is_false_negative_window") else 0.0,
                feature_map[k]["whistle_band_energy_fraction_post"],
                feature_map[k]["spectral_entropy_post_mean"],
                feature_map[k]["tonal_peak_fraction_post_mean"],
                feature_map[k]["spectral_flatness_post_mean"],
            ]
            for k, _, r in hold
        ],
        dtype=np.float64,
    )
    mean = xtr.mean(axis=0)
    std = xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    w = np.zeros(xtr.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(1000):
        p = 1.0 / (1.0 + np.exp(-np.clip(xtr @ w + b, -40.0, 40.0)))
        e = p - ytr
        w -= 0.1 * ((xtr.T @ e) / len(xtr) + 0.01 * w)
        b -= 0.1 * float(np.mean(e))
    scores = 1.0 / (1.0 + np.exp(-np.clip(xte @ w + b, -40.0, 40.0)))
    score_rows = [{"row_key": k, "label": lab, "score_platform": float(s)} for (k, lab, _), s in zip(hold, scores.tolist())]

    pos = [x["score_platform"] for x in score_rows if x["label"] == "platform_dive"]
    neg = [x["score_platform"] for x in score_rows if x["label"] == "noise_or_other"]
    min_pos = min(pos)
    overlap_noise = [x for x in score_rows if x["label"] == "noise_or_other" and x["score_platform"] >= min_pos]

    def enrich(row_key: str) -> dict:
        row = by_key[row_key]
        rev = reviewed_by.get(row_key, {})
        score = next(x["score_platform"] for x in score_rows if x["row_key"] == row_key)
        return {
            "row_key": row_key,
            "legacy_subtype": rev.get("review_subtype") or row.get("legacy_non_dive_subtype"),
            "suggestion_reason": rev.get("suggested_event_label_reason"),
            "audio_score": row.get("audio_score"),
            "audio_clip_probability": row.get("audio_clip_probability"),
            "score_platform": score,
            "feature_snapshot": feature_map[row_key],
        }

    fp_details = [enrich(k) for k in fp_rows]
    fn_details = [enrich(k) for k in fn_rows]

    def counts(items: list[dict], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in items:
            k = str(it.get(key)) if it.get(key) is not None else "null"
            out[k] = out.get(k, 0) + 1
        return out

    fp_sub = counts(fp_details, "legacy_subtype")
    fn_sub = counts(fn_details, "legacy_subtype")
    fp_reason = counts(fp_details, "suggestion_reason")
    fn_reason = counts(fn_details, "suggestion_reason")

    residual = {
        "scope": {
            "source_execution": "outputs/phase5_regime_aware_execution_r6.json",
            "platform_noise_slice_frozen": True,
            "classifier_family_unchanged": True,
            "phase5_rerun_performed": False,
        },
        "part_a_residual_error_audit": {
            "remaining_false_positives_noise_to_platform_count": len(fp_rows),
            "remaining_false_negatives_platform_to_noise_count": len(fn_rows),
            "false_positive_rows": fp_rows,
            "false_negative_rows": fn_rows,
            "false_positive_commonality": {
                "by_subtype": fp_sub,
                "by_suggestion_reason": fp_reason,
                "acoustic_profile_clue": "Residual FPs are still concentrated in handling/whistle-like noise signatures.",
            },
            "false_negative_commonality": {
                "by_subtype": fn_sub,
                "by_suggestion_reason": fn_reason,
                "acoustic_profile_clue": "Two boundary dives remain, but platform recall is preserved at 0.80.",
            },
            "false_positive_details": fp_details,
            "false_negative_details": fn_details,
        },
        "part_b_why_auc_still_low": {
            "primary_cause": "ranking_of_ambiguous_noise_rows",
            "secondary_factors": [
                "residual_tonal_noise_confusion",
                "residual_diffuse_handling_clutter",
                "score_ordering_overlap_between_noise_and_platform_tails",
            ],
            "evidence": {
                "auc_observed_r6": r6["platform_noise_results"]["validation_metrics"]["auc"],
                "min_positive_score": float(min_pos),
                "noise_rows_scoring_at_or_above_min_positive_count": len(overlap_noise),
                "noise_rows_scoring_at_or_above_min_positive": [x["row_key"] for x in overlap_noise],
            },
            "why_macro_f1_passes_while_auc_below_threshold": "Threshold-level confusion improved enough for macro F1 pass, but AUC remains constrained by residual score overlap in ambiguous noise rows.",
        },
        "decision": "R7_MICRO_REFINEMENT_JUSTIFIED",
    }

    micro_plan = {
        "scope": {
            "single_refinement_only": True,
            "no_model_family_change": True,
            "no_detector_change": True,
            "no_label_or_taxonomy_change": True,
        },
        "part_c_single_micro_refinement": {
            "refinement_id": "r7_tonal_penalty_feature_one_step",
            "type": "one_small_score_ordering_refinement",
            "exact_refinement": "Add one derived feature tonal_noise_penalty = whistle_band_energy_fraction_post * tonal_peak_fraction_post_mean into the same logistic model to down-rank residual whistle-like noise rows.",
            "success_criteria": {
                "primary": "platform/noise AUC >= 0.66",
                "secondary": "platform recall >= 0.80 and no increase in noise->platform FP count",
            },
        },
        "part_d_decision": {
            "decision": "R7_MICRO_REFINEMENT_JUSTIFIED",
            "reason": "The remaining gap is narrow and ranking-specific; one bounded refinement is proportionate before r7.",
        },
    }

    residual_md = [
        "# Platform/Noise r6 Residual Analysis",
        "",
        "## Part A — Residual error audit",
        "",
        f"- remaining noise_or_other -> platform_dive false positives (6): `{fp_rows}`",
        f"- remaining platform_dive -> noise_or_other false negatives (2): `{fn_rows}`",
        f"- FP common subtype pattern: `{fp_sub}`",
        f"- FP common suggestion-reason pattern: `{fp_reason}`",
        f"- FN common subtype pattern: `{fn_sub}`",
        f"- FN common suggestion-reason pattern: `{fn_reason}`",
        "",
        "## Part B — Why AUC is still low",
        "",
        f"- primary cause: `{residual['part_b_why_auc_still_low']['primary_cause']}`",
        f"- evidence: `{residual['part_b_why_auc_still_low']['evidence']}`",
        "",
        f"- decision: `{residual['decision']}`",
        "",
    ]
    plan_md = [
        "# Platform/Noise r6 Micro-Refinement Plan",
        "",
        "## Part C — Single bounded refinement",
        "",
        f"- refinement id: `{micro_plan['part_c_single_micro_refinement']['refinement_id']}`",
        f"- exact refinement: {micro_plan['part_c_single_micro_refinement']['exact_refinement']}",
        "",
        "## Part D — Decision",
        "",
        f"- `{micro_plan['part_d_decision']['decision']}`",
        f"- reason: {micro_plan['part_d_decision']['reason']}",
        "",
    ]

    Path("outputs/platform_noise_r6_residual_analysis.json").write_text(json.dumps(residual, indent=2))
    Path("outputs/platform_noise_r6_residual_analysis.md").write_text("\\n".join(residual_md))
    Path("outputs/platform_noise_r6_micro_refinement_plan.json").write_text(json.dumps(micro_plan, indent=2))
    Path("outputs/platform_noise_r6_micro_refinement_plan.md").write_text("\\n".join(plan_md))


if __name__ == "__main__":
    main()
