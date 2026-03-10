from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


MODEL_FEATURES = [
    "audio_score",
    "spectral_flux",
    "rms",
    "hf_ratio",
    "spectral_centroid_hz",
    "spectral_flatness",
    "post_flux_ratio",
    "post_rms_ratio",
    "local_prominence",
    "nearby_peaks_8s",
]


@dataclass
class AudioCandidateModel:
    feature_names: Sequence[str]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def load(cls, path: str | Path) -> "AudioCandidateModel":
        data = json.loads(Path(path).read_text())
        return cls(
            feature_names=data["feature_names"],
            means=np.array(data["means"], dtype=np.float32),
            stds=np.array(data["stds"], dtype=np.float32),
            weights=np.array(data["weights"], dtype=np.float32),
            bias=float(data["bias"]),
        )

    def predict_probability(self, feature_map: Dict[str, float]) -> float:
        values = np.array([float(feature_map[name]) for name in self.feature_names], dtype=np.float32)
        normalized = (values - self.means) / np.maximum(self.stds, 1e-6)
        logit = float(np.dot(normalized, self.weights) + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
