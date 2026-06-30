import numpy as np

from wmsv.data.transitions import Transition, split_transitions


def test_transition_stores_arrays_and_metadata():
    transition = Transition(
        state=np.array([0.0, 1.0]),
        action=np.array([1.0]),
        next_state=np.array([0.5, 1.0]),
        reward=0.2,
        done=False,
        info={"level_id": "x"},
    )

    assert transition.reward == 0.2
    assert transition.info["level_id"] == "x"


def test_split_transitions_is_deterministic():
    transitions = [
        Transition(np.array([i]), np.array([0]), np.array([i + 1]), 0.0, False, {})
        for i in range(10)
    ]

    train, valid = split_transitions(transitions, valid_fraction=0.3, seed=7)

    assert len(train) == 7
    assert len(valid) == 3
    assert [int(t.state[0]) for t in valid] == [8, 3, 1]
