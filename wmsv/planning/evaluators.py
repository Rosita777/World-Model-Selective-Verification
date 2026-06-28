from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
