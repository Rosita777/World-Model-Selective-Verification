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


@dataclass(frozen=True)
class StandardizedCentroidGate:
    feature_names: list[str]
    positive_centroid: np.ndarray
    negative_centroid: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def score(self, row: dict) -> float:
        x = np.array([float(row[name]) for name in self.feature_names], dtype=np.float32)
        z = (x - self.mean) / self.scale
        dist_pos = float(np.linalg.norm(z - self.positive_centroid))
        dist_neg = float(np.linalg.norm(z - self.negative_centroid))
        return dist_neg - dist_pos


@dataclass(frozen=True)
class LogisticGate:
    feature_names: list[str]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray

    def score(self, row: dict) -> float:
        x = np.array([float(row[name]) for name in self.feature_names], dtype=np.float32)
        z = (x - self.mean) / self.scale
        return float(np.dot(z, self.weights) + self.bias)


def _feature_matrix_and_labels(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int32)
    if not bool((labels == 1).any()) or not bool((labels == 0).any()):
        raise ValueError("Gate requires positive and negative examples")
    return features, labels


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, mean, scale


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


def fit_standardized_centroid_gate(rows: list[dict], feature_names: list[str]) -> StandardizedCentroidGate:
    features, labels = _feature_matrix_and_labels(rows, feature_names)
    standardized, mean, scale = _standardize(features)
    return StandardizedCentroidGate(
        feature_names=list(feature_names),
        positive_centroid=standardized[labels == 1].mean(axis=0),
        negative_centroid=standardized[labels == 0].mean(axis=0),
        mean=mean,
        scale=scale,
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def fit_logistic_gate(
    rows: list[dict],
    feature_names: list[str],
    learning_rate: float = 0.1,
    steps: int = 1000,
    l2: float = 1e-3,
) -> LogisticGate:
    features, labels = _feature_matrix_and_labels(rows, feature_names)
    standardized, mean, scale = _standardize(features)
    x = standardized.astype(np.float64)
    y = labels.astype(np.float64)
    positive_count = float((labels == 1).sum())
    negative_count = float((labels == 0).sum())
    sample_weight = np.where(
        labels == 1,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    ).astype(np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        error = (probabilities - y) * sample_weight
        weights_gradient = (x.T @ error) / len(labels) + l2 * weights
        bias_gradient = float(error.mean())
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient
    return LogisticGate(
        feature_names=list(feature_names),
        weights=weights.astype(np.float32),
        bias=float(bias),
        mean=mean,
        scale=scale,
    )


def fit_gate(rows: list[dict], feature_names: list[str], gate_model: str):
    if gate_model == "centroid":
        return fit_centroid_gate(rows, feature_names)
    if gate_model == "standardized_centroid":
        return fit_standardized_centroid_gate(rows, feature_names)
    if gate_model == "logistic":
        return fit_logistic_gate(rows, feature_names)
    raise ValueError("gate_model must be 'centroid', 'standardized_centroid', or 'logistic'")


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
