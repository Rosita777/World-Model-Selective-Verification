from __future__ import annotations

import numpy as np

from wmsv.analysis.verification_value import add_verification_labels
from wmsv.data.transitions import Transition
from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec
from wmsv.models.linear_dynamics import LinearDynamicsModel
from wmsv.planning.shooting import ShootingPlanner


def build_maze_pilot_rows(
    variants: int = 3, episodes_per_variant: int = 20, seed: int = 0
) -> list[dict]:
    rows: list[dict] = []
    for variant in range(int(variants)):
        env = _make_env(variant)
        transitions = _collect_maze_transitions(env, seed=seed + variant, count=80)
        learned = LinearDynamicsModel.fit(transitions)
        for episode in range(int(episodes_per_variant)):
            state = _episode_start(env, episode)
            cheap = ShootingPlanner(
                learned,
                horizon=4,
                candidates=16,
                seed=seed + variant * 1000 + episode,
                goal=env.spec.goal,
                goal_bias=0.6,
            )
            verifier = ShootingPlanner(
                env,
                horizon=8,
                candidates=64,
                seed=seed + 10000 + variant * 1000 + episode,
                goal=env.spec.goal,
                goal_bias=0.0,
            )
            think_longer = ShootingPlanner(
                learned,
                horizon=8,
                candidates=64,
                seed=seed + 20000 + variant * 1000 + episode,
                goal=env.spec.goal,
                goal_bias=0.6,
            )
            uniform_true = ShootingPlanner(
                env,
                horizon=4,
                candidates=16,
                seed=seed + 30000 + variant * 1000 + episode,
                goal=env.spec.goal,
                goal_bias=0.0,
            )
            cheap_result = cheap.plan(state)
            verifier_result = verifier.plan(state)
            think_longer_result = think_longer.plan(state)
            uniform_true_result = uniform_true.plan(state)
            r_c = _execute_plan(env, state, cheap_result.plan, max_steps=4)
            r_v = _execute_plan(env, state, verifier_result.plan, max_steps=4)
            r_t = _execute_plan(env, state, think_longer_result.plan, max_steps=4)
            r_u = _execute_plan(env, state, uniform_true_result.plan, max_steps=4)
            row = {
                "level_id": f"maze-{variant}:e{episode}",
                "a_c": cheap_result.action,
                "a_v": verifier_result.action,
                "a_t": think_longer_result.action,
                "a_u": uniform_true_result.action,
                "r_c": r_c,
                "r_v": r_v,
                "r_t": r_t,
                "r_u": r_u,
                "cheap_score": cheap_result.score,
                "verifier_score": verifier_result.score,
                "think_longer_score": think_longer_result.score,
                "uniform_true_score": uniform_true_result.score,
                "score_margin": cheap_result.score_margin,
                "ensemble_uncertainty": abs(verifier_result.score - cheap_result.score),
                "uncertainty_proxy": abs(verifier_result.score - cheap_result.score),
                "cheap_nodes": cheap_result.nodes_expanded,
                "verifier_nodes": verifier_result.nodes_expanded,
                "think_longer_nodes": think_longer_result.nodes_expanded,
                "uniform_true_nodes": uniform_true_result.nodes_expanded,
            }
            rows.append(add_verification_labels(row, epsilon=0.01, action_delta=0.10))
    return rows


def _make_env(variant: int) -> PointMazeEnv:
    size = 7
    walls = {(3, y) for y in range(1, 6) if y != 1 + (variant % 5)}
    spec = PointMazeSpec(
        size=size,
        start=np.array([1.0, 1.0], dtype=np.float64),
        goal=np.array([5.0, 5.0], dtype=np.float64),
        walls=walls,
    )
    return PointMazeEnv(spec)


def _episode_start(env: PointMazeEnv, episode: int) -> np.ndarray:
    starts = [
        env.spec.start + np.array([0.0, 0.0], dtype=np.float64),
        env.spec.start + np.array([0.0, 1.0], dtype=np.float64),
        env.spec.start + np.array([1.0, 0.0], dtype=np.float64),
        env.spec.start + np.array([1.0, 1.0], dtype=np.float64),
        env.spec.start + np.array([0.5, 0.0], dtype=np.float64),
        env.spec.start + np.array([0.0, 0.5], dtype=np.float64),
        env.spec.goal + np.array([-0.55, 0.0], dtype=np.float64),
        env.spec.goal + np.array([0.0, -0.55], dtype=np.float64),
        env.spec.goal + np.array([-0.55, -0.55], dtype=np.float64),
        env.spec.goal + np.array([-0.90, 0.0], dtype=np.float64),
        env.spec.goal + np.array([0.0, -0.90], dtype=np.float64),
        env.spec.goal + np.array([-0.90, -0.45], dtype=np.float64),
    ]
    state = starts[int(episode) % len(starts)]
    return np.clip(state, 0.0, float(env.spec.size - 1))


def _collect_maze_transitions(
    env: PointMazeEnv, seed: int, count: int
) -> list[Transition]:
    rng = np.random.default_rng(int(seed))
    transitions: list[Transition] = []
    state = env.reset()
    for idx in range(int(count)):
        action = rng.normal(size=2)
        norm = np.linalg.norm(action)
        action = action / max(norm, 1.0)
        step = env.step(state, action)
        transitions.append(
            Transition(state, action, step.state, step.reward, step.done, {"idx": idx})
        )
        state = env.reset() if step.done else step.state
    return transitions


def _execute_plan(env: PointMazeEnv, state: np.ndarray, plan: np.ndarray, max_steps: int | None = None) -> float:
    current = np.asarray(state, dtype=np.float64).copy()
    total = 0.0
    horizon = len(plan) if max_steps is None else min(len(plan), int(max_steps))
    for action in plan[:horizon]:
        step = env.step(current, action)
        total += float(step.reward)
        current = step.state
        if step.done:
            break
    return total
