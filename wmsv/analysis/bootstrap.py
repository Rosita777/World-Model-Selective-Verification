from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    if not values:
        raise ValueError("values must not be empty")
    data = np.array(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        indices = rng.integers(0, len(data), size=len(data))
        means.append(float(data[indices].mean()))
    alpha = 1.0 - confidence
    return {
        "mean": float(data.mean()),
        "low": float(np.quantile(means, alpha / 2.0)),
        "high": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def bootstrap_paired_delta_ci(
    left_values: list[float],
    right_values: list[float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    if len(left_values) != len(right_values):
        raise ValueError("left_values and right_values must have the same length")
    if not left_values:
        raise ValueError("values must not be empty")
    deltas = [
        float(left) - float(right)
        for left, right in zip(left_values, right_values)
    ]
    return bootstrap_mean_ci(deltas, samples=samples, confidence=confidence, seed=seed)
