from __future__ import annotations

from typing import Any, Dict


LONG_SESSION_OVERRIDES: Dict[str, Any] = {
    "audio_min_score": 4.0,
    "audio_pattern_min_score": 0.2,
    "audio_decode_timeout_seconds": 180.0,
}


def apply_named_profile(defaults: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    merged = dict(defaults)
    if profile_name == "long-session":
        merged.update(LONG_SESSION_OVERRIDES)
    return merged

