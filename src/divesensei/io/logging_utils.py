from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class StructuredLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emit_stdout = os.environ.get("DIVESENSEI_EMIT_PROGRESS") == "1"

    def log(self, event: str, **fields: Any) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        payload.update({key: _to_jsonable(value) for key, value in fields.items()})
        line = json.dumps(payload, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.emit_stdout:
            print(line, file=sys.stdout, flush=True)


def build_candidate_debug_summary(candidates: Sequence[Any]) -> dict[str, Any]:
    timestamps = [float(candidate.timestamp) for candidate in candidates]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    confidences = {
        "high": sum(1 for candidate in candidates if getattr(candidate, "confidence", None) == "high"),
        "medium": sum(1 for candidate in candidates if getattr(candidate, "confidence", None) == "medium"),
        "low": sum(1 for candidate in candidates if getattr(candidate, "confidence", None) == "low"),
    }
    return {
        "candidate_count": len(candidates),
        "confidence_counts": confidences,
        "timestamp_range": {
            "first": timestamps[0] if timestamps else None,
            "last": timestamps[-1] if timestamps else None,
        },
        "spacing_seconds": {
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "mean": (sum(deltas) / len(deltas)) if deltas else None,
        },
        "top_audio_scores": sorted((float(candidate.audio_score) for candidate in candidates), reverse=True)[:10],
    }
