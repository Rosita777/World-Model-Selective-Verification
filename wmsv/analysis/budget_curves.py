from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from wmsv.analysis.budgeting import compute_summary, mean_return_for_mask, threshold_budget_mask
from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0, fit_value_rank_gate


def evaluate_budget_curve(
    train_rows: list[dict],
    eval_rows: list[dict],
    feature_names: list[str],
    budgets: Sequence[float],
    random_seed: int = 0,
) -> list[dict]:
    schema = DIVEFeatureSchema(feature_names)
    gate = fit_dive_v0(train_rows, schema)
    value_rank_gate = fit_value_rank_gate(train_rows, schema)
    dive_scores = [gate.score(row) for row in eval_rows]
    value_rank_scores = [value_rank_gate.score(row) for row in eval_rows]
    uncertainty_scores = [
        float(row.get("ensemble_uncertainty", row.get("uncertainty_proxy", 0.0)))
        for row in eval_rows
    ]
    oracle_scores = [float(row["r_v"]) - float(row["r_c"]) for row in eval_rows]
    rng = random.Random(random_seed)
    random_scores = [rng.random() for _ in eval_rows]

    return [
        _evaluate_budget(
            eval_rows=eval_rows,
            budget_fraction=float(budget),
            dive_scores=dive_scores,
            value_rank_scores=value_rank_scores,
            uncertainty_scores=uncertainty_scores,
            oracle_scores=oracle_scores,
            random_scores=random_scores,
        )
        for budget in budgets
    ]


def aggregate_budget_curves(curves: Sequence[Sequence[dict]], metrics: Sequence[str]) -> list[dict]:
    if not curves:
        return []
    budgets = [float(item["budget_fraction"]) for item in curves[0]]
    aggregate: list[dict] = []
    for idx, budget in enumerate(budgets):
        row = {"budget_fraction": budget}
        for metric in metrics:
            values = np.array([float(curve[idx][metric]) for curve in curves], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        aggregate.append(row)
    return aggregate


def _evaluate_budget(
    eval_rows: list[dict],
    budget_fraction: float,
    dive_scores: list[float],
    value_rank_scores: list[float],
    uncertainty_scores: list[float],
    oracle_scores: list[float],
    random_scores: list[float],
) -> dict:
    masks = {
        "dive": threshold_budget_mask(dive_scores, budget_fraction),
        "value_rank": threshold_budget_mask(value_rank_scores, budget_fraction),
        "uncertainty": threshold_budget_mask(uncertainty_scores, budget_fraction),
        "oracle": threshold_budget_mask(oracle_scores, budget_fraction),
        "random": threshold_budget_mask(random_scores, budget_fraction),
    }
    result = {
        "budget_fraction": budget_fraction,
        "cheap_return": _mean_key(eval_rows, "r_c"),
        "always_verify_return": _mean_key(eval_rows, "r_v"),
        "think_longer_return": _mean_key(eval_rows, "r_t"),
        "uniform_true_return": _mean_key(eval_rows, "r_u"),
        "think_longer_nodes": _mean_key(eval_rows, "think_longer_nodes"),
        "uniform_true_nodes": _mean_key(eval_rows, "uniform_true_nodes"),
    }
    for name, mask in masks.items():
        result[f"{name}_return"] = mean_return_for_mask(eval_rows, mask)
        result[f"{name}_selection"] = _selection_summary(eval_rows, mask)
        result[f"{name}_nodes"] = compute_summary(eval_rows, mask)["total_nodes"] / max(len(eval_rows), 1)
    return result


def _mean_key(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _selection_summary(rows: list[dict], mask: list[bool]) -> dict:
    selected = [row for row, use_verifier in zip(rows, mask) if use_verifier]
    helpful = sum(int(row.get("y_helpful", row.get("label", 0))) for row in selected)
    harmful = sum(int(row.get("y_harm", 0)) for row in selected)
    wasted = sum(int(row.get("y_waste", 0)) for row in selected)
    total_delta = sum(float(row.get("delta_r", float(row["r_v"]) - float(row["r_c"]))) for row in selected)
    return {
        "selected": len(selected),
        "helpful_selected": helpful,
        "harmful_selected": harmful,
        "wasted_selected": wasted,
        "helpful_precision": helpful / len(selected) if selected else 0.0,
        "mean_delta_r": total_delta / len(selected) if selected else 0.0,
        "total_delta_r": total_delta,
    }
