from __future__ import annotations

from pathlib import Path

from wmsv.analysis.stage_a import evaluate_plan_rollout, make_plan_label_row
from wmsv.analysis.uncertainty import ensemble_uncertainty
from wmsv.analysis.verification_value import add_verification_labels
from wmsv.data.boxoban import iter_boxoban_levels
from wmsv.envs.sokoban import parse_level
from wmsv.envs.sokoban import SokobanState
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import (
    DegradedPushEvaluator,
    PotentialEvaluator,
    TrueEvaluator,
    has_corner_deadlock,
)


TINY_LEVELS = [
    ["#####", "#@$.#", "#   #", "#####"],
    ["######", "# @$ #", "#   .#", "######"],
    ["#######", "#@ $. #", "#     #", "#######"],
    ["#######", "# @ $ #", "#   . #", "#######"],
]


def build_boxoban_pilot_rows(
    limit: int = 20,
    seed: int = 0,
    levels_path: str | None = None,
    train_level_count: int = 4,
) -> list[dict]:
    level_limit = max(int(limit), int(train_level_count), 8)
    levels, source = _load_levels(levels_path, limit=level_limit)
    cheap_evaluator = PotentialEvaluator(
        DegradedPushEvaluator(push_error_rate=0.5, seed=seed),
        scale=0.1,
    )
    cheap = BeamPlanner(cheap_evaluator, depth=3, width=8)
    think_longer = BeamPlanner(cheap_evaluator, depth=5, width=16)
    verifier_evaluator = PotentialEvaluator(TrueEvaluator(), scale=0.1)
    verifier = BeamPlanner(verifier_evaluator, depth=5, width=16)
    uniform_true = BeamPlanner(verifier_evaluator, depth=4, width=8)
    uncertainty_planners = [
        BeamPlanner(
            PotentialEvaluator(
                DegradedPushEvaluator(push_error_rate=0.5, seed=seed + offset),
                scale=0.1,
            ),
            depth=3,
            width=8,
        )
        for offset in range(3)
    ]
    rows: list[dict] = []
    idx = 0
    while len(rows) < int(limit):
        level_id, state = levels[idx % len(levels)]
        row = make_plan_label_row(
            f"{level_id}:r{idx}",
            state,
            cheap,
            verifier,
            verifier_evaluator,
        )
        row = add_verification_labels(row)
        think_longer_result = think_longer.plan(state)
        uniform_true_result = uniform_true.plan(state)
        uncertainty = ensemble_uncertainty(state, uncertainty_planners)
        row.update(
            {
                "base_source": source,
                "a_t": think_longer_result.action,
                "plan_t": think_longer_result.plan,
                "r_t": evaluate_plan_rollout(state, think_longer_result.plan, verifier_evaluator),
                "think_longer_score": think_longer_result.score,
                "think_longer_nodes": think_longer_result.nodes_expanded,
                "a_u": uniform_true_result.action,
                "plan_u": uniform_true_result.plan,
                "r_u": evaluate_plan_rollout(state, uniform_true_result.plan, verifier_evaluator),
                "uniform_true_score": uniform_true_result.score,
                "uniform_true_nodes": uniform_true_result.nodes_expanded,
                "ensemble_action_disagreement": uncertainty["action_disagreement"],
                "ensemble_plan_disagreement": uncertainty["plan_disagreement"],
                "ensemble_score_variance": uncertainty["score_variance"],
                "ensemble_num_planners": uncertainty["num_planners"],
                "ensemble_uncertainty": uncertainty["action_disagreement"] + uncertainty["score_variance"],
            }
        )
        _add_plan_features(row, state, cheap)
        rows.append(row)
        idx += 1
    return rows


def _load_levels(levels_path: str | None, limit: int) -> tuple[list[tuple[str, SokobanState]], str]:
    if levels_path is not None and Path(levels_path).exists():
        levels = [
            (level.level_id, parse_level(level.lines))
            for level in iter_boxoban_levels(levels_path, limit=limit)
        ]
        if levels:
            return levels, "boxoban"
    return [(f"tiny-{idx}", parse_level(lines)) for idx, lines in enumerate(TINY_LEVELS)], "tiny"


def _same_state(left: SokobanState, right: SokobanState) -> bool:
    return bool((left.encode() == right.encode()).all())


def _plan_turns(plan: list[int]) -> int:
    return sum(1 for idx in range(1, len(plan)) if plan[idx] != plan[idx - 1])


def _add_plan_features(row: dict, state: SokobanState, cheap: BeamPlanner) -> None:
    plan = list(row.get("plan_c", []))
    plan_length = len(plan)
    row["cheap_plan_length"] = float(plan_length)
    row["cheap_plan_turns"] = float(_plan_turns(plan))
    row["cheap_plan_unique_actions"] = float(len(set(plan))) if plan else 0.0
    row["cheap_plan_score_per_step"] = (
        float(row["cheap_score"]) / float(plan_length)
        if plan_length else 0.0
    )

    current = state
    state_changes = 0
    box_changes = 0
    deadlock_steps = 0
    uncertainty_values: list[float] = []
    replan_matches = 0
    for action in plan:
        uncertainty = (
            float(cheap.evaluator.uncertainty(current, int(action)))
            if hasattr(cheap.evaluator, "uncertainty")
            else 0.0
        )
        uncertainty_values.append(uncertainty)
        replan = cheap.plan(current)
        replan_matches += int(int(replan.action) == int(action))
        step = cheap.evaluator.step(current, int(action))
        state_changes += int(not _same_state(current, step.state))
        box_changes += int(not bool((current.boxes == step.state.boxes).all()))
        deadlock_steps += int(has_corner_deadlock(step.state))
        current = step.state
        if step.done:
            break

    denominator = float(plan_length) if plan_length else 1.0
    row["cheap_plan_final_progress"] = float(current.boxes_on_goals_fraction())
    row["cheap_plan_state_change_fraction"] = float(state_changes) / denominator
    row["cheap_plan_box_change_fraction"] = float(box_changes) / denominator
    row["cheap_plan_deadlock_fraction"] = float(deadlock_steps) / denominator
    row["temporal_consistency"] = float(replan_matches) / denominator
    row["temporal_inconsistency"] = 1.0 - row["temporal_consistency"]
    row["cheap_plan_uncertainty_mean"] = (
        sum(uncertainty_values) / denominator if uncertainty_values else 0.0
    )
    row["cheap_plan_uncertainty_max"] = max(uncertainty_values) if uncertainty_values else 0.0
    first_uncertain = next(
        (idx for idx, value in enumerate(uncertainty_values) if value > 0.0),
        plan_length,
    )
    row["cheap_plan_early_uncertainty"] = (
        1.0 - (float(first_uncertain) / denominator)
        if uncertainty_values else 0.0
    )
    inverse_margin = 1.0 / (1.0 + max(float(row.get("score_margin", 0.0)), 0.0))
    row["decision_instability"] = float(row.get("ensemble_action_disagreement", 0.0)) * inverse_margin
    row["impact_uncertainty"] = (
        float(row.get("ensemble_plan_disagreement", 0.0))
        * row["cheap_plan_box_change_fraction"]
    )
    row["irreversibility_risk"] = (
        row["cheap_plan_uncertainty_mean"]
        * max(row["cheap_plan_box_change_fraction"], row["cheap_plan_deadlock_fraction"])
    )
    row["counterfactual_action_gap"] = _counterfactual_action_gap(state, cheap)
    row["counterfactual_gap_confidence"] = (
        row["counterfactual_action_gap"]
        / (1.0 + abs(float(row.get("cheap_score", 0.0))))
    )


def _counterfactual_action_gap(state: SokobanState, cheap: BeamPlanner) -> float:
    scores: list[float] = []
    for action in range(4):
        step = cheap.evaluator.step(state, action)
        continuation = cheap.plan(step.state)
        scores.append(float(step.reward + continuation.score))
    ordered = sorted(scores, reverse=True)
    return max(ordered[0] - ordered[1], 0.0)
