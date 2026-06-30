from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MazeStep:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class PointMazeSpec:
    size: int
    start: np.ndarray
    goal: np.ndarray
    walls: set[tuple[int, int]]

    @classmethod
    def empty(cls, size: int, start: tuple[float, float], goal: tuple[float, float]) -> "PointMazeSpec":
        return cls(
            size=int(size),
            start=np.array(start, dtype=np.float64),
            goal=np.array(goal, dtype=np.float64),
            walls=set(),
        )


class PointMazeEnv:
    def __init__(self, spec: PointMazeSpec, step_size: float = 0.5, goal_radius: float = 0.35):
        self.spec = spec
        self.step_size = float(step_size)
        self.goal_radius = float(goal_radius)

    def reset(self) -> np.ndarray:
        return self.spec.start.copy()

    def step(self, state: np.ndarray, action: np.ndarray) -> MazeStep:
        action = np.asarray(action, dtype=np.float64)
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action = action / norm
        proposed = np.asarray(state, dtype=np.float64) + self.step_size * action
        proposed = np.clip(proposed, 0.0, float(self.spec.size - 1))
        blocked = self._is_wall(proposed)
        next_state = np.asarray(state, dtype=np.float64).copy() if blocked else proposed
        distance = float(np.linalg.norm(next_state - self.spec.goal))
        done = distance <= self.goal_radius
        reward = 1.0 if done else -distance
        return MazeStep(next_state, reward, done, {"blocked": blocked, "distance": distance})

    def _is_wall(self, state: np.ndarray) -> bool:
        cell = (int(round(float(state[0]))), int(round(float(state[1]))))
        return cell in self.spec.walls
