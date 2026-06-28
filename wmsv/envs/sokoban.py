from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


DELTAS = {
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.UP: (-1, 0),
}


@dataclass(frozen=True)
class SokobanState:
    walls: np.ndarray
    goals: np.ndarray
    boxes: np.ndarray
    player: np.ndarray

    @property
    def height(self) -> int:
        return int(self.walls.shape[0])

    @property
    def width(self) -> int:
        return int(self.walls.shape[1])

    def clone(self) -> "SokobanState":
        return SokobanState(
            walls=self.walls.copy(),
            goals=self.goals.copy(),
            boxes=self.boxes.copy(),
            player=self.player.copy(),
        )

    def is_solved(self) -> bool:
        return bool(np.all(self.boxes <= self.goals) and self.boxes.sum() == self.goals.sum())

    def boxes_on_goals_fraction(self) -> float:
        total = max(float(self.boxes.sum()), 1.0)
        return float(np.logical_and(self.boxes, self.goals).sum()) / total

    def encode(self) -> np.ndarray:
        player_channel = np.zeros_like(self.walls, dtype=np.float32)
        player_channel[tuple(self.player)] = 1.0
        boxes_on_goals = np.logical_and(self.boxes, self.goals).astype(np.float32)
        return np.stack(
            [
                self.walls.astype(np.float32),
                self.goals.astype(np.float32),
                self.boxes.astype(np.float32),
                player_channel,
                boxes_on_goals,
            ],
            axis=0,
        )

    def step(self, action: Action | int) -> tuple["SokobanState", float, bool, dict[str, bool]]:
        action = Action(int(action))
        dy, dx = DELTAS[action]
        y, x = map(int, self.player)
        ny, nx = y + dy, x + dx

        if self.walls[ny, nx]:
            return self.clone(), 0.0, False, {"blocked": True, "pushed": False}

        boxes = self.boxes.copy()
        player = self.player.copy()
        pushed = False

        if boxes[ny, nx]:
            by, bx = ny + dy, nx + dx
            if self.walls[by, bx] or boxes[by, bx]:
                return self.clone(), 0.0, False, {"blocked": True, "pushed": False}
            boxes[ny, nx] = False
            boxes[by, bx] = True
            pushed = True

        player[:] = (ny, nx)
        next_state = SokobanState(
            walls=self.walls.copy(),
            goals=self.goals.copy(),
            boxes=boxes,
            player=player,
        )
        done = next_state.is_solved()
        reward = 1.0 if done else 0.0
        return next_state, reward, done, {"blocked": False, "pushed": pushed}


def parse_level(lines: list[str]) -> SokobanState:
    height = len(lines)
    width = max(len(line) for line in lines)
    walls = np.zeros((height, width), dtype=bool)
    goals = np.zeros((height, width), dtype=bool)
    boxes = np.zeros((height, width), dtype=bool)
    player = None

    for y, raw_line in enumerate(lines):
        line = raw_line.ljust(width)
        for x, char in enumerate(line):
            if char == "#":
                walls[y, x] = True
            elif char == ".":
                goals[y, x] = True
            elif char == "$":
                boxes[y, x] = True
            elif char == "@":
                player = np.array([y, x], dtype=np.int64)
            elif char == "*":
                boxes[y, x] = True
                goals[y, x] = True
            elif char == "+":
                player = np.array([y, x], dtype=np.int64)
                goals[y, x] = True
            elif char == " ":
                continue
            else:
                raise ValueError(f"Unsupported Sokoban character {char!r} at {(y, x)}")

    if player is None:
        raise ValueError("Level does not contain a player")

    return SokobanState(walls=walls, goals=goals, boxes=boxes, player=player)
