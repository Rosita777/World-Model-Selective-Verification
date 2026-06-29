from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from wmsv.envs.sokoban import Action, SokobanState


@dataclass(frozen=True)
class EvaluatedStep:
    state: SokobanState
    reward: float
    done: bool


class TrueEvaluator:
    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        next_state, reward, done, _ = state.step(Action(action))
        return EvaluatedStep(next_state, reward, done)

    def uncertainty(self, state: SokobanState, action: int) -> float:
        return 0.0


def is_blocked_for_deadlock(state: SokobanState, y: int, x: int) -> bool:
    if y < 0 or y >= state.height or x < 0 or x >= state.width:
        return True
    return bool(state.walls[y, x])


def has_corner_deadlock(state: SokobanState) -> bool:
    boxes = np.argwhere(state.boxes)
    for box in boxes:
        y, x = map(int, box)
        if state.goals[y, x]:
            continue
        vertical_blocked = (
            is_blocked_for_deadlock(state, y - 1, x)
            or is_blocked_for_deadlock(state, y + 1, x)
        )
        horizontal_blocked = (
            is_blocked_for_deadlock(state, y, x - 1)
            or is_blocked_for_deadlock(state, y, x + 1)
        )
        if vertical_blocked and horizontal_blocked:
            return True
    return False


class DeadlockAwareEvaluator:
    """Adds a penalty for obvious irreversible Sokoban corner deadlocks."""

    def __init__(self, base_evaluator, deadlock_penalty: float = 1.0):
        self.base_evaluator = base_evaluator
        self.deadlock_penalty = float(deadlock_penalty)

    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        step = self.base_evaluator.step(state, action)
        if step.done:
            return step
        if has_corner_deadlock(step.state):
            return EvaluatedStep(step.state, step.reward - self.deadlock_penalty, False)
        return step

    def uncertainty(self, state: SokobanState, action: int) -> float:
        if hasattr(self.base_evaluator, "uncertainty"):
            return float(self.base_evaluator.uncertainty(state, action))
        return 0.0


class DegradedPushEvaluator:
    """Evaluator that deterministically hides some legal push outcomes.

    This is a Stage A synthetic cheap world model. It creates a controlled
    cheap/verifier gap before we train neural dynamics models.
    """

    def __init__(self, push_error_rate: float, seed: int = 0, corrupt_push_penalty: float = 0.0):
        if not 0.0 <= push_error_rate <= 1.0:
            raise ValueError("push_error_rate must be in [0, 1]")
        self.push_error_rate = float(push_error_rate)
        self.seed = int(seed)
        self.corrupt_push_penalty = float(corrupt_push_penalty)

    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        next_state, reward, done, info = state.step(Action(action))
        if info["pushed"] and self._should_corrupt(state, action):
            return EvaluatedStep(state.clone(), -self.corrupt_push_penalty, False)
        return EvaluatedStep(next_state, reward, done)

    def uncertainty(self, state: SokobanState, action: int) -> float:
        _, _, _, info = state.step(Action(action))
        return self.push_error_rate if info["pushed"] else 0.0

    def _should_corrupt(self, state: SokobanState, action: int) -> bool:
        if self.push_error_rate <= 0.0:
            return False
        if self.push_error_rate >= 1.0:
            return True
        digest = hashlib.sha256()
        digest.update(str(self.seed).encode("utf-8"))
        digest.update(state.encode().tobytes())
        digest.update(str(int(action)).encode("utf-8"))
        value = int.from_bytes(digest.digest()[:8], "big") / float(2**64 - 1)
        return value < self.push_error_rate


class PotentialEvaluator:
    """Adds dense progress reward to a base evaluator."""

    def __init__(self, base_evaluator, scale: float = 0.1):
        self.base_evaluator = base_evaluator
        self.scale = float(scale)

    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        before = self.potential(state)
        step = self.base_evaluator.step(state, action)
        after = self.potential(step.state)
        shaped_reward = step.reward + self.scale * (after - before)
        return EvaluatedStep(step.state, shaped_reward, step.done)

    def uncertainty(self, state: SokobanState, action: int) -> float:
        if hasattr(self.base_evaluator, "uncertainty"):
            return float(self.base_evaluator.uncertainty(state, action))
        return 0.0

    @staticmethod
    def potential(state: SokobanState) -> float:
        boxes = np.argwhere(state.boxes)
        goals = np.argwhere(state.goals)
        if len(boxes) == 0 or len(goals) == 0:
            return 0.0
        total = 0.0
        for box in boxes:
            distances = np.abs(goals - box).sum(axis=1)
            total += float(distances.min())
        return -total
