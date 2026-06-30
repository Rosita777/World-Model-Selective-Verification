import numpy as np

from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec
from wmsv.planning.shooting import ShootingPlanner


def test_shooting_planner_returns_action_and_plan():
    env = PointMazeEnv(PointMazeSpec.empty(size=5, start=(1.0, 1.0), goal=(3.0, 1.0)))
    planner = ShootingPlanner(env, horizon=3, candidates=16, seed=0)

    result = planner.plan(env.reset())

    assert result.action.shape == (2,)
    assert result.plan.shape == (3, 2)
    assert result.nodes_expanded == 48
