from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from wmsv.data.transitions import Transition
from wmsv.envs.sokoban import Action, SokobanState
from wmsv.planning.evaluators import EvaluatedStep


def _state_key(state: SokobanState, action: int) -> bytes:
    return state.encode().astype(np.float32).tobytes() + int(action).to_bytes(2, "little", signed=False)


def _decode_state(reference: SokobanState, encoded: np.ndarray) -> SokobanState:
    player = np.argwhere(encoded[3] > 0.5)[0].astype(np.int64)
    return SokobanState(
        walls=reference.walls.copy(),
        goals=reference.goals.copy(),
        boxes=encoded[2] > 0.5,
        player=player,
    )


@dataclass(frozen=True)
class SokobanTransitionRecord:
    reference: SokobanState
    next_encoded: np.ndarray
    reward: float
    done: bool


class MemorizedSokobanModel:
    def __init__(self, table: dict[bytes, SokobanTransitionRecord]):
        self.table = dict(table)

    @classmethod
    def fit(cls, transitions: Sequence[Transition]) -> "MemorizedSokobanModel":
        table: dict[bytes, SokobanTransitionRecord] = {}
        for transition in transitions:
            state = transition.info["state_obj"]
            action = int(transition.action[0])
            table[_state_key(state, action)] = SokobanTransitionRecord(
                reference=state,
                next_encoded=transition.next_state.astype(np.float32),
                reward=float(transition.reward),
                done=bool(transition.done),
            )
        return cls(table)

    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        record = self.table.get(_state_key(state, int(action)))
        if record is None:
            return EvaluatedStep(state.clone(), 0.0, False)
        return EvaluatedStep(
            _decode_state(record.reference, record.next_encoded),
            record.reward,
            record.done,
        )

    def uncertainty(self, state: SokobanState, action: int) -> float:
        return 0.0 if _state_key(state, int(action)) in self.table else 1.0


def collect_sokoban_transitions(
    levels: Sequence[tuple[str, SokobanState]],
    action_sequences: Sequence[Sequence[int]],
) -> list[Transition]:
    transitions: list[Transition] = []
    for level_id, state in levels:
        for sequence_id, actions in enumerate(action_sequences):
            current = state.clone()
            for step_id, action in enumerate(actions):
                next_state, reward, done, info = current.step(Action(action))
                transitions.append(
                    Transition(
                        state=current.encode(),
                        action=np.array([int(action)], dtype=np.int64),
                        next_state=next_state.encode(),
                        reward=float(reward),
                        done=bool(done),
                        info={
                            "level_id": level_id,
                            "sequence_id": sequence_id,
                            "step_id": step_id,
                            "state_obj": current,
                            "pushed": bool(info["pushed"]),
                            "blocked": bool(info["blocked"]),
                        },
                    )
                )
                current = next_state
                if done:
                    break
    return transitions
