from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VerificationLabels:
    y_change: int
    y_helpful: int
    y_harm: int
    y_waste: int
    delta_r: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerificationLabels):
            return NotImplemented
        return (
            self.y_change == other.y_change
            and self.y_helpful == other.y_helpful
            and self.y_harm == other.y_harm
            and self.y_waste == other.y_waste
            and math.isclose(self.delta_r, other.delta_r)
        )


def _action_changed(a_c, a_v, action_delta: float | None) -> bool:
    if action_delta is None:
        return a_c != a_v
    c = np.asarray(a_c, dtype=np.float64)
    v = np.asarray(a_v, dtype=np.float64)
    return bool(np.linalg.norm(v - c) > float(action_delta))


def classify_verification(
    a_c,
    a_v,
    r_c: float,
    r_v: float,
    epsilon: float = 0.01,
    action_delta: float | None = None,
) -> VerificationLabels:
    changed = _action_changed(a_c, a_v, action_delta)
    delta_r = float(r_v) - float(r_c)
    helpful = changed and delta_r > float(epsilon)
    harmful = changed and delta_r < -float(epsilon)
    wasted = (not changed) or abs(delta_r) <= float(epsilon)
    return VerificationLabels(
        y_change=int(changed),
        y_helpful=int(helpful),
        y_harm=int(harmful),
        y_waste=int(wasted),
        delta_r=delta_r,
    )


def add_verification_labels(
    row: dict, epsilon: float = 0.01, action_delta: float | None = None
) -> dict:
    labels = classify_verification(
        row["a_c"],
        row["a_v"],
        float(row["r_c"]),
        float(row["r_v"]),
        epsilon=epsilon,
        action_delta=action_delta,
    )
    enriched = dict(row)
    enriched.update(
        {
            "y_change": labels.y_change,
            "y_helpful": labels.y_helpful,
            "y_harm": labels.y_harm,
            "y_waste": labels.y_waste,
            "delta_r": labels.delta_r,
            "label": labels.y_helpful,
        }
    )
    return enriched


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels)]
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def positive_label_rate(rows: Sequence[dict], label_key: str = "y_helpful") -> float:
    if not rows:
        return 0.0
    return sum(int(row[label_key]) for row in rows) / len(rows)


def go_no_go_status(
    cheap_success: float,
    always_verify_success: float,
    positive_label_rate: float,
    helpful_auroc: float,
    budget20_gain: float,
    min_always_gain: float,
    min_auroc: float,
) -> dict:
    failures: list[str] = []
    if not 0.15 < float(cheap_success) < 0.85:
        failures.append("cheap_success_outside_sweet_spot")
    if float(always_verify_success) - float(cheap_success) < float(min_always_gain):
        failures.append("always_verify_gap_too_small")
    if not 0.15 <= float(positive_label_rate) <= 0.50:
        failures.append("positive_label_rate_outside_range")
    if math.isnan(float(helpful_auroc)) or float(helpful_auroc) < float(min_auroc):
        failures.append("helpful_auroc_too_low")
    if float(budget20_gain) < 0.03:
        failures.append("budget20_gain_too_small")
    return {"passed": not failures, "failures": failures}
