from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wmsv.data.transitions import Transition
from wmsv.envs.point_maze import MazeStep


@dataclass(frozen=True)
class LinearDynamicsModel:
    state_dim: int
    action_dim: int
    next_weights: np.ndarray
    reward_weights: np.ndarray

    @classmethod
    def fit(cls, transitions: Sequence[Transition], ridge: float = 1e-3) -> "LinearDynamicsModel":
        if not transitions:
            raise ValueError("LinearDynamicsModel requires at least one transition")
        state_dim = int(transitions[0].state.shape[0])
        action_dim = int(transitions[0].action.shape[0])
        x = np.stack([_features(t.state, t.action) for t in transitions]).astype(np.float64)
        y_next = np.stack([t.next_state for t in transitions]).astype(np.float64)
        y_reward = np.array([float(t.reward) for t in transitions], dtype=np.float64)
        penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        next_weights = np.linalg.solve(x.T @ x + penalty, x.T @ y_next)
        reward_weights = np.linalg.solve(x.T @ x + penalty, x.T @ y_reward)
        return cls(state_dim, action_dim, next_weights, reward_weights)

    def step(self, state: np.ndarray, action: np.ndarray) -> MazeStep:
        x = _features(state, action)
        next_state = x @ self.next_weights
        reward = float(x @ self.reward_weights)
        return MazeStep(np.asarray(next_state, dtype=np.float64), reward, False, {"model": "linear"})


def _features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float64),
            np.asarray(action, dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ]
    )
