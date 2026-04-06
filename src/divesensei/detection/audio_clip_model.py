from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


@dataclass
class AudioClipModel:
    feature_names: Sequence[str]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def load(cls, path: str | Path) -> "AudioClipModel":
        data = json.loads(Path(path).read_text())
        return cls(
            feature_names=data["feature_names"],
            means=np.array(data["means"], dtype=np.float32),
            stds=np.array(data["stds"], dtype=np.float32),
            weights=np.array(data["weights"], dtype=np.float32),
            bias=float(data["bias"]),
        )

    def predict_probability(self, feature_map: Dict[str, float]) -> float:
        values = np.array([float(feature_map.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)
        normalized = (values - self.means) / np.maximum(self.stds, 1e-6)
        logit = float(np.dot(normalized, self.weights) + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))

    def explain_feature_map(self, feature_map: Dict[str, float]) -> list[dict[str, float | str]]:
        values = np.array([float(feature_map.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)
        normalized = (values - self.means) / np.maximum(self.stds, 1e-6)
        contributions = normalized * self.weights
        ranked = sorted(
            [
                {
                    "feature": name,
                    "value": float(value),
                    "normalized_value": float(z_value),
                    "weight": float(weight),
                    "contribution": float(contribution),
                    "abs_contribution": float(abs(contribution)),
                }
                for name, value, z_value, weight, contribution in zip(
                    self.feature_names,
                    values,
                    normalized,
                    self.weights,
                    contributions,
                    strict=True,
                )
            ],
            key=lambda item: item["abs_contribution"],
            reverse=True,
        )
        return ranked
