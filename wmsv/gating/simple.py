from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidGate:
    feature_names: list[str]
    positive_centroid: np.ndarray
    negative_centroid: np.ndarray

    def score(self, row: dict) -> float:
        x = np.array([float(row[name]) for name in self.feature_names], dtype=np.float32)
        dist_pos = float(np.linalg.norm(x - self.positive_centroid))
        dist_neg = float(np.linalg.norm(x - self.negative_centroid))
        return dist_neg - dist_pos


def fit_centroid_gate(rows: list[dict], feature_names: list[str]) -> CentroidGate:
    positives = [
        [float(row[name]) for name in feature_names]
        for row in rows
        if int(row["label"]) == 1
    ]
    negatives = [
        [float(row[name]) for name in feature_names]
        for row in rows
        if int(row["label"]) == 0
    ]
    if not positives or not negatives:
        raise ValueError("CentroidGate requires positive and negative examples")
    return CentroidGate(
        feature_names=list(feature_names),
        positive_centroid=np.array(positives, dtype=np.float32).mean(axis=0),
        negative_centroid=np.array(negatives, dtype=np.float32).mean(axis=0),
    )


def top_fraction_mask(scores: list[float], fraction: float) -> list[bool]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if not scores:
        return []
    count = int(round(len(scores) * fraction))
    if count <= 0:
        return [False] * len(scores)
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    chosen = set(ranked[:count])
    return [idx in chosen for idx in range(len(scores))]


def mean_policy_return(rows: list[dict], verify_mask: list[bool]) -> float:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have the same length")
    if not rows:
        return 0.0
    returns = [
        float(row["r_v"] if verify else row["r_c"])
        for row, verify in zip(rows, verify_mask)
    ]
    return sum(returns) / len(returns)


def selection_summary(rows: list[dict], verify_mask: list[bool], epsilon: float = 0.01) -> dict:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have the same length")
    selected = 0
    helpful = 0
    harmful = 0
    spurious = 0
    wasted = 0
    for row, verify in zip(rows, verify_mask):
        if not verify:
            continue
        selected += 1
        action_changed = row["a_v"] != row["a_c"]
        delta = float(row["r_v"]) - float(row["r_c"])
        if action_changed and delta > epsilon:
            helpful += 1
        elif action_changed and delta < -epsilon:
            harmful += 1
        elif action_changed:
            spurious += 1
        else:
            wasted += 1
    return {
        "selected": selected,
        "helpful_selected": helpful,
        "harmful_selected": harmful,
        "spurious_selected": spurious,
        "wasted_selected": wasted,
        "helpful_precision": helpful / selected if selected else 0.0,
    }
