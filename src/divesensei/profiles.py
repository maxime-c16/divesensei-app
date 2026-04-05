from __future__ import annotations

from typing import Any, Dict


LONG_SESSION_BASELINE_OVERRIDES: Dict[str, Any] = {
    "audio_min_score": 4.0,
    "audio_pattern_min_score": 0.2,
    "audio_peak_separation": 4.0,
    "audio_visual_merge_seconds": 2.0,
    # Large local/NAS recordings can take significantly longer to decode audio.
    "audio_decode_timeout_seconds": 900.0,
}

LONG_SESSION_ADVANCED_OVERRIDES: Dict[str, Any] = {
    "audio_peak_threshold": 2.8,
    "audio_min_score": 2.8,
    "audio_pattern_min_score": -0.75,
    "audio_peak_separation": 0.3,
    "audio_visual_merge_seconds": 0.45,
    "audio_clip_model_min_probability": 0.9,
    # Keep the same relaxed timeout for the advanced profile.
    "audio_decode_timeout_seconds": 900.0,
}


def apply_named_profile(defaults: Dict[str, Any], profile_name: str, detector_id: str = "audio_v1_heuristic") -> Dict[str, Any]:
    merged = dict(defaults)
    if profile_name == "long-session":
        if detector_id == "audio_v1_heuristic":
            merged.update(LONG_SESSION_BASELINE_OVERRIDES)
        else:
            merged.update(LONG_SESSION_ADVANCED_OVERRIDES)
    return merged
