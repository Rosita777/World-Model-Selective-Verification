from wmsv.envs.sokoban import Action, parse_level
from wmsv.models.sokoban_memory import MemorizedSokobanModel, collect_sokoban_transitions


def test_memorized_sokoban_model_replays_seen_transition():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    transitions = collect_sokoban_transitions([("tiny", state)], action_sequences=[[int(Action.RIGHT)]])
    model = MemorizedSokobanModel.fit(transitions)

    step = model.step(state, int(Action.RIGHT))

    assert step.done is True
    assert step.reward == 1.0


def test_memorized_sokoban_model_uses_noop_for_unseen_transition():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    model = MemorizedSokobanModel.fit([])

    step = model.step(state, int(Action.RIGHT))

    assert step.state.encode().tolist() == state.encode().tolist()
    assert step.reward == 0.0
    assert step.done is False
