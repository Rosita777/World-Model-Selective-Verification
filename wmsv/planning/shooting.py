from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShootingPlanResult:
    action: np.ndarray
    score: float
    plan: np.ndarray
    nodes_expanded: int
    score_margin: float


class ShootingPlanner:
    def __init__(
        self,
        dynamics,
        horizon: int,
        candidates: int,
        seed: int = 0,
        goal: np.ndarray | None = None,
        goal_bias: float = 0.0,
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if candidates < 2:
            raise ValueError("candidates must be at least 2")
        self.dynamics = dynamics
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.seed = int(seed)
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64)
        self.goal_bias = float(goal_bias)

    def plan(self, state: np.ndarray) -> ShootingPlanResult:
        rng = np.random.default_rng(self.seed)
        plans = rng.normal(size=(self.candidates, self.horizon, 2))
        if self.goal is not None and self.goal_bias > 0.0:
            direction = self.goal - np.asarray(state, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                biased_count = max(1, self.candidates // 2)
                plans[:biased_count] = (
                    self.goal_bias * direction / norm
                    + (1.0 - self.goal_bias) * plans[:biased_count]
                )
        norms = np.linalg.norm(plans, axis=2, keepdims=True)
        plans = plans / np.maximum(norms, 1.0)
        scores = np.array([self._rollout_score(state, plan) for plan in plans], dtype=np.float64)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        second = int(order[1])
        return ShootingPlanResult(
            action=plans[best, 0].copy(),
            score=float(scores[best]),
            plan=plans[best].copy(),
            nodes_expanded=self.candidates * self.horizon,
            score_margin=float(scores[best] - scores[second]),
        )

    def _rollout_score(self, state: np.ndarray, plan: np.ndarray) -> float:
        current = np.asarray(state, dtype=np.float64).copy()
        total = 0.0
        for action in plan:
            step = self.dynamics.step(current, action)
            total += float(step.reward)
            current = step.state
            if step.done:
                break
        return total
