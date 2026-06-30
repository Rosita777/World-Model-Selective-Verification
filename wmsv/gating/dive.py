from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DIVEFeatureSchema:
    feature_names: list[str]

    def vector(self, row: dict) -> np.ndarray:
        return np.array([float(row[name]) for name in self.feature_names], dtype=np.float64)


@dataclass(frozen=True)
class DIVEV0Gate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    change_weights: np.ndarray
    change_bias: float
    harm_weights: np.ndarray
    harm_bias: float
    waste_weights: np.ndarray
    waste_bias: float
    delta_weights: np.ndarray
    delta_bias: float
    alpha: float = 1.0
    verification_cost: float = 0.0

    def _standardized(self, row: dict) -> np.ndarray:
        return (self.schema.vector(row) - self.mean) / self.scale

    def predict_heads(self, row: dict) -> dict:
        x = self._standardized(row)
        return {
            "p_change": _sigmoid_scalar(float(x @ self.change_weights + self.change_bias)),
            "delta_hat": float(x @ self.delta_weights + self.delta_bias),
            "p_harm": _sigmoid_scalar(float(x @ self.harm_weights + self.harm_bias)),
            "p_waste": _sigmoid_scalar(float(x @ self.waste_weights + self.waste_bias)),
        }

    def score(self, row: dict) -> float:
        heads = self.predict_heads(row)
        return (
            heads["p_change"] * max(heads["delta_hat"], 0.0)
            - self.alpha * heads["p_harm"]
            - self.verification_cost
        )


@dataclass(frozen=True)
class BinaryLabelGate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def score(self, row: dict) -> float:
        x = (self.schema.vector(row) - self.mean) / self.scale
        return float(x @ self.weights + self.bias)


@dataclass(frozen=True)
class ValueRankGate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def _standardized(self, row: dict) -> np.ndarray:
        return (self.schema.vector(row) - self.mean) / self.scale

    def predict_delta(self, row: dict) -> float:
        x = self._standardized(row)
        return float(x @ self.weights + self.bias)

    def score(self, row: dict) -> float:
        return max(self.predict_delta(row), 0.0)


@dataclass(frozen=True)
class RiskAwareValueGate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    change_weights: np.ndarray
    change_bias: float
    harm_weights: np.ndarray
    harm_bias: float
    delta_weights: np.ndarray
    delta_bias: float
    alpha: float = 1.0

    def _standardized(self, row: dict) -> np.ndarray:
        return (self.schema.vector(row) - self.mean) / self.scale

    def predict_heads(self, row: dict) -> dict:
        x = self._standardized(row)
        return {
            "p_change": _sigmoid_scalar(float(x @ self.change_weights + self.change_bias)),
            "value_delta": float(x @ self.delta_weights + self.delta_bias),
            "p_harm": _sigmoid_scalar(float(x @ self.harm_weights + self.harm_bias)),
        }

    def score(self, row: dict) -> float:
        heads = self.predict_heads(row)
        positive_value = max(heads["value_delta"], 0.0)
        return positive_value * (1.0 - heads["p_harm"]) - self.alpha * heads["p_harm"]


def _feature_matrix(rows: Sequence[dict], schema: DIVEFeatureSchema) -> np.ndarray:
    return np.stack([schema.vector(row) for row in rows]).astype(np.float64)


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _sigmoid_scalar(value: float) -> float:
    return float(_sigmoid(np.array([value], dtype=np.float64))[0])


def _fit_binary_head(x: np.ndarray, y: np.ndarray, steps: int = 800, learning_rate: float = 0.1) -> tuple[np.ndarray, float]:
    if len(set(int(v) for v in y)) < 2:
        return np.zeros(x.shape[1], dtype=np.float64), float(np.mean(y))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive = max(float((y == 1).sum()), 1.0)
    negative = max(float((y == 0).sum()), 1.0)
    sample_weight = np.where(y == 1, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
    for _ in range(steps):
        probs = _sigmoid(x @ weights + bias)
        error = (probs - y) * sample_weight
        weights -= learning_rate * (x.T @ error) / len(y)
        bias -= learning_rate * float(error.mean())
    return weights, bias


def _fit_regression_head(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, float]:
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef[:-1], float(coef[-1])


def fit_dive_v0(
    rows: Sequence[dict],
    schema: DIVEFeatureSchema,
    alpha: float = 1.0,
    verification_cost: float = 0.0,
) -> DIVEV0Gate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    change_w, change_b = _fit_binary_head(x, np.array([int(row["y_change"]) for row in rows], dtype=np.float64))
    harm_w, harm_b = _fit_binary_head(x, np.array([int(row["y_harm"]) for row in rows], dtype=np.float64))
    waste_w, waste_b = _fit_binary_head(x, np.array([int(row["y_waste"]) for row in rows], dtype=np.float64))
    delta_w, delta_b = _fit_regression_head(x, np.array([float(row["delta_r"]) for row in rows], dtype=np.float64))
    return DIVEV0Gate(
        schema=schema,
        mean=mean,
        scale=scale,
        change_weights=change_w,
        change_bias=change_b,
        harm_weights=harm_w,
        harm_bias=harm_b,
        waste_weights=waste_w,
        waste_bias=waste_b,
        delta_weights=delta_w,
        delta_bias=delta_b,
        alpha=float(alpha),
        verification_cost=float(verification_cost),
    )


def fit_uncertainty_label_gate(rows: Sequence[dict], schema: DIVEFeatureSchema) -> BinaryLabelGate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    weights, bias = _fit_binary_head(
        x,
        np.array([int(row["uncertainty_label"]) for row in rows], dtype=np.float64),
    )
    return BinaryLabelGate(schema=schema, mean=mean, scale=scale, weights=weights, bias=bias)


def fit_value_rank_gate(rows: Sequence[dict], schema: DIVEFeatureSchema) -> ValueRankGate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    weights, bias = _fit_regression_head(
        x,
        np.array([float(row["delta_r"]) for row in rows], dtype=np.float64),
    )
    return ValueRankGate(schema=schema, mean=mean, scale=scale, weights=weights, bias=bias)


def fit_risk_aware_value_gate(
    rows: Sequence[dict],
    schema: DIVEFeatureSchema,
    alpha: float = 1.0,
) -> RiskAwareValueGate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    change_w, change_b = _fit_binary_head(x, np.array([int(row["y_change"]) for row in rows], dtype=np.float64))
    harm_w, harm_b = _fit_binary_head(x, np.array([int(row["y_harm"]) for row in rows], dtype=np.float64))
    delta_w, delta_b = _fit_regression_head(x, np.array([float(row["delta_r"]) for row in rows], dtype=np.float64))
    return RiskAwareValueGate(
        schema=schema,
        mean=mean,
        scale=scale,
        change_weights=change_w,
        change_bias=change_b,
        harm_weights=harm_w,
        harm_bias=harm_b,
        delta_weights=delta_w,
        delta_bias=delta_b,
        alpha=float(alpha),
    )
