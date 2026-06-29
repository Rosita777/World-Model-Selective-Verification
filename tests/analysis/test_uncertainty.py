from wmsv.analysis.uncertainty import ensemble_uncertainty
from wmsv.envs.sokoban import parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import DegradedPushEvaluator, PotentialEvaluator


def test_ensemble_uncertainty_reports_action_disagreement_and_score_variance():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    planners = [
        BeamPlanner(PotentialEvaluator(DegradedPushEvaluator(0.0, seed=0, corrupt_push_penalty=1.0)), depth=1, width=4),
        BeamPlanner(PotentialEvaluator(DegradedPushEvaluator(1.0, seed=1, corrupt_push_penalty=1.0)), depth=1, width=4),
    ]

    result = ensemble_uncertainty(state, planners)

    assert result["action_disagreement"] == 1.0
    assert result["plan_disagreement"] == 1.0
    assert result["score_variance"] > 0.0
    assert result["num_planners"] == 2
