from __future__ import annotations

from wmsv.analysis.feasibility import label_from_decisions
from wmsv.envs.sokoban import Action, SokobanState
from wmsv.planning.beam import BeamPlanner


def evaluate_first_action(state: SokobanState, action: int, evaluator: BeamPlanner) -> float:
    if hasattr(evaluator, "evaluator"):
        step = evaluator.evaluator.step(state, int(action))
        next_state = step.state
        first_reward = float(step.reward)
        done = step.done
    else:
        next_state, first_reward, done, _ = state.step(Action(action))

    if done:
        return 1.0
    result = evaluator.plan(next_state)
    return float(first_reward + result.score + next_state.boxes_on_goals_fraction())


def score_margin(top_action_scores: dict[int, float]) -> float:
    ordered = sorted(top_action_scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    if ordered[0] == float("-inf") or ordered[1] == float("-inf"):
        return 0.0
    return float(ordered[0] - ordered[1])


def make_label_row(
    level_id: str,
    state: SokobanState,
    cheap: BeamPlanner,
    verifier: BeamPlanner,
    evaluator: BeamPlanner,
    epsilon: float = 0.01,
) -> dict:
    cheap_result = cheap.plan(state)
    verifier_result = verifier.plan(state)
    r_c = evaluate_first_action(state, cheap_result.action, evaluator)
    r_v = evaluate_first_action(state, verifier_result.action, evaluator)
    uncertainty = 0.0
    if hasattr(cheap.evaluator, "uncertainty"):
        uncertainty = float(cheap.evaluator.uncertainty(state, cheap_result.action))

    return {
        "level_id": level_id,
        "a_c": cheap_result.action,
        "a_v": verifier_result.action,
        "r_c": r_c,
        "r_v": r_v,
        "label": label_from_decisions(cheap_result.action, verifier_result.action, r_c, r_v, epsilon),
        "cheap_score": cheap_result.score,
        "verifier_score": verifier_result.score,
        "score_margin": score_margin(cheap_result.top_action_scores),
        "uncertainty_proxy": uncertainty,
        "cheap_nodes": cheap_result.nodes_expanded,
        "verifier_nodes": verifier_result.nodes_expanded,
    }
