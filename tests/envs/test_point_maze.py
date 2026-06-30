import numpy as np

from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec


def test_point_maze_moves_toward_goal():
    env = PointMazeEnv(PointMazeSpec.empty(size=5, start=(1.0, 1.0), goal=(3.0, 1.0)))

    step = env.step(env.reset(), np.array([1.0, 0.0]))

    assert step.state[0] > 1.0
    assert step.reward > -3.0


def test_point_maze_blocks_wall_crossing():
    spec = PointMazeSpec(
        size=5,
        start=np.array([1.0, 1.0], dtype=np.float64),
        goal=np.array([3.0, 1.0], dtype=np.float64),
        walls={(2, 1)},
    )
    env = PointMazeEnv(spec)

    step = env.step(env.reset(), np.array([1.0, 0.0]))

    assert step.state[0] < 2.0
    assert step.info["blocked"] is True
