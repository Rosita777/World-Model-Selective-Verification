from __future__ import annotations

import numpy as np

from wmsv.envs.sokoban import SokobanState
from wmsv.planning.beam import BeamPlanner


def ensemble_uncertainty(state: SokobanState, planners: list[BeamPlanner]) -> dict:
    if not planners:
        raise ValueError("planners must not be empty")

    results = [planner.plan(state) for planner in planners]
    actions = [result.action for result in results]
    plans = [tuple(result.plan) for result in results]
    scores = np.array([result.score for result in results], dtype=np.float32)
    pairs = 0
    disagreeing_pairs = 0
    plan_disagreeing_pairs = 0
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            pairs += 1
            disagreeing_pairs += int(actions[left] != actions[right])
            plan_disagreeing_pairs += int(plans[left] != plans[right])
    disagreement = disagreeing_pairs / pairs if pairs else 0.0
    plan_disagreement = plan_disagreeing_pairs / pairs if pairs else 0.0

    return {
        "action_disagreement": float(disagreement),
        "plan_disagreement": float(plan_disagreement),
        "score_variance": float(scores.var()),
        "num_planners": len(planners),
    }
