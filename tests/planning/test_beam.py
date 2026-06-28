from wmsv.envs.sokoban import Action, parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import DegradedPushEvaluator, PotentialEvaluator, TrueEvaluator


def test_true_evaluator_scores_one_step_solution_highest():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    planner = BeamPlanner(evaluator=TrueEvaluator(), depth=1, width=4)

    result = planner.plan(state)

    assert result.action == int(Action.RIGHT)
    assert result.score == 1.0
    assert result.nodes_expanded == 4
    assert result.top_action_scores[int(Action.RIGHT)] == 1.0


def test_beam_planner_expands_multiple_depths_with_width_cap():
    state = parse_level(["#####", "#@ $.#", "#  . #", "######"])
    planner = BeamPlanner(evaluator=TrueEvaluator(), depth=2, width=2)

    result = planner.plan(state)

    assert len(result.plan) == 2
    assert result.nodes_expanded == 12


def test_degraded_push_evaluator_can_hide_successful_pushes():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    evaluator = DegradedPushEvaluator(push_error_rate=1.0)

    step = evaluator.step(state, int(Action.RIGHT))

    assert step.done is False
    assert step.reward == 0.0
    assert step.state.boxes[1, 2]
    assert evaluator.uncertainty(state, int(Action.RIGHT)) == 1.0
    assert evaluator.uncertainty(state, int(Action.UP)) == 0.0


def test_potential_evaluator_rewards_moving_box_closer_to_goal():
    state = parse_level(["######", "# @$ #", "#   .#", "######"])
    evaluator = PotentialEvaluator(TrueEvaluator())

    step = evaluator.step(state, int(Action.RIGHT))

    assert step.reward > 0.0
