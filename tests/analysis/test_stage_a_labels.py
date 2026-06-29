from wmsv.analysis.stage_a import (
    evaluate_first_action,
    evaluate_plan_rollout,
    make_label_row,
    make_plan_label_row,
)
from wmsv.envs.sokoban import Action, parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import DegradedPushEvaluator, PotentialEvaluator, TrueEvaluator


def test_evaluate_first_action_scores_immediate_solution():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    evaluator = BeamPlanner(TrueEvaluator(), depth=1, width=4)

    score = evaluate_first_action(state, int(Action.RIGHT), evaluator)

    assert score == 1.0


def test_evaluate_first_action_uses_dense_planning_return():
    state = parse_level(["######", "# @$ #", "#   .#", "######"])
    evaluator = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=1, width=4)

    score = evaluate_first_action(state, int(Action.RIGHT), evaluator)

    assert score > 0.0


def test_make_label_row_marks_degraded_cheap_as_helpfully_fixable():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    cheap = BeamPlanner(DegradedPushEvaluator(push_error_rate=1.0, corrupt_push_penalty=1.0), depth=1, width=4)
    verifier = BeamPlanner(TrueEvaluator(), depth=1, width=4)
    evaluator = BeamPlanner(TrueEvaluator(), depth=1, width=4)

    row = make_label_row("tiny-0", state, cheap, verifier, evaluator)

    assert row["a_c"] != row["a_v"]
    assert row["r_v"] > row["r_c"]
    assert row["label"] == 1
    assert row["uncertainty_proxy"] >= 0.0


def test_evaluate_plan_rollout_executes_fixed_action_sequence():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    evaluator = TrueEvaluator()

    score = evaluate_plan_rollout(state, [int(Action.RIGHT)], evaluator)

    assert score == 1.0


def test_evaluate_plan_rollout_accepts_beam_planner_evaluator():
    state = parse_level(["######", "# @$ #", "#   .#", "######"])
    evaluator = BeamPlanner(PotentialEvaluator(TrueEvaluator()), depth=1, width=4)

    score = evaluate_plan_rollout(state, [int(Action.RIGHT)], evaluator)

    assert score > 0.0


def test_make_plan_label_row_compares_full_candidate_plans():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    cheap = BeamPlanner(DegradedPushEvaluator(push_error_rate=1.0, corrupt_push_penalty=1.0), depth=1, width=4)
    verifier = BeamPlanner(TrueEvaluator(), depth=1, width=4)
    evaluator = TrueEvaluator()

    row = make_plan_label_row("tiny-0", state, cheap, verifier, evaluator)

    assert row["plan_c"] != row["plan_v"]
    assert row["r_v"] > row["r_c"]
    assert row["label"] == 1
