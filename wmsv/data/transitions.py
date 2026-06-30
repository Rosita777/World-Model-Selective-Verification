from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict


def split_transitions(
    transitions: Sequence[Transition],
    valid_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Transition], list[Transition]]:
    if not 0.0 < float(valid_fraction) < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    indices = list(range(len(transitions)))
    random.Random(int(seed)).shuffle(indices)
    valid_count = int(round(len(indices) * float(valid_fraction)))
    train = [transitions[idx] for idx in indices[valid_count:]]
    valid = [transitions[idx] for idx in indices[:valid_count]]
    return train, valid
