from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wmsv.analysis.budgeting import threshold_budget_mask


def feature_correlations(rows: list[dict], reference: str, features: Sequence[str]) -> dict:
    reference_values = np.array([float(row[reference]) for row in rows], dtype=np.float64)
    correlations: dict[str, float] = {}
    for feature in features:
        values = np.array([float(row[feature]) for row in rows], dtype=np.float64)
        correlations[feature] = _pearson(reference_values, values)
    return {
        "reference": reference,
        "correlations": correlations,
    }


def compare_ranked_selections(
    rows: list[dict],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    budget_fraction: float,
) -> dict:
    left_mask = threshold_budget_mask(left_scores, budget_fraction)
    right_mask = threshold_budget_mask(right_scores, budget_fraction)
    left_indices = {idx for idx, selected in enumerate(left_mask) if selected}
    right_indices = {idx for idx, selected in enumerate(right_mask) if selected}
    return {
        "budget_fraction": float(budget_fraction),
        "left_selected": len(left_indices),
        "right_selected": len(right_indices),
        "overlap": len(left_indices & right_indices),
        "jaccard": _jaccard(left_indices, right_indices),
        "left_mean_improvement": _mean_improvement(rows, left_indices),
        "right_mean_improvement": _mean_improvement(rows, right_indices),
        "left_helpful_precision": _helpful_precision(rows, left_indices),
        "right_helpful_precision": _helpful_precision(rows, right_indices),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or float(left.std()) < 1e-12 or float(right.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _mean_improvement(rows: list[dict], indices: set[int]) -> float:
    if not indices:
        return 0.0
    values = [float(rows[idx]["r_v"]) - float(rows[idx]["r_c"]) for idx in indices]
    return sum(values) / len(values)


def _helpful_precision(rows: list[dict], indices: set[int]) -> float:
    if not indices:
        return 0.0
    return sum(int(rows[idx].get("y_helpful", rows[idx].get("label", 0))) for idx in indices) / len(indices)
