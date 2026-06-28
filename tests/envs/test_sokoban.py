import numpy as np

from wmsv.envs.sokoban import Action, parse_level


LEVEL = [
    "#####",
    "#@$.#",
    "#   #",
    "#####",
]


def test_parse_level_tracks_basic_entities():
    state = parse_level(LEVEL)

    assert state.height == 4
    assert state.width == 5
    assert tuple(state.player) == (1, 1)
    assert state.walls[0, 0]
    assert state.boxes[1, 2]
    assert state.goals[1, 3]


def test_push_box_onto_goal_solves_level():
    state = parse_level(LEVEL)

    next_state, reward, done, info = state.step(Action.RIGHT)

    assert tuple(next_state.player) == (1, 2)
    assert next_state.boxes[1, 3]
    assert reward == 1.0
    assert done is True
    assert info["pushed"] is True


def test_wall_blocks_player_motion():
    state = parse_level(LEVEL)

    next_state, reward, done, info = state.step(Action.UP)

    assert tuple(next_state.player) == (1, 1)
    assert np.array_equal(next_state.boxes, state.boxes)
    assert reward == 0.0
    assert done is False
    assert info["blocked"] is True

