from wmsv.analysis.stage_a import evaluate_first_action, make_label_row
from wmsv.envs.sokoban import Action, parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import DegradedPushEvaluator, TrueEvaluator


def test_evaluate_first_action_scores_immediate_solution():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    evaluator = BeamPlanner(TrueEvaluator(), depth=1, width=4)

    score = evaluate_first_action(state, int(Action.RIGHT), evaluator)

    assert score == 1.0


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
