from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from wmsv.envs.sokoban import Action, DELTAS


@dataclass(frozen=True)
class GeneratedSokobanLevel:
    level_id: str
    lines: list[str]
    solution: tuple[int, ...]


def generate_sokoban_levels(
    count: int,
    seed: int = 0,
    min_size: int = 6,
    max_size: int = 9,
    max_pushes: int = 4,
) -> list[GeneratedSokobanLevel]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if min_size < 5:
        raise ValueError("min_size must be at least 5")
    if max_size < min_size:
        raise ValueError("max_size must be at least min_size")
    if max_pushes < 1:
        raise ValueError("max_pushes must be at least 1")

    rng = random.Random(int(seed))
    levels: list[GeneratedSokobanLevel] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = max(1000, int(count) * 200)
    while len(levels) < int(count) and attempts < max_attempts:
        attempts += 1
        height = rng.randint(int(min_size), int(max_size))
        width = rng.randint(int(min_size), int(max_size))
        pushes = rng.randint(1, int(max_pushes))
        action = Action(rng.randrange(4))
        try:
            lines = _sample_straight_push_level(rng, height, width, pushes, action)
        except RuntimeError:
            continue
        key = tuple(lines)
        if key in seen:
            continue
        seen.add(key)
        levels.append(
            GeneratedSokobanLevel(
                level_id=f"generated-{int(seed)}-{len(levels):04d}",
                lines=lines,
                solution=tuple([int(action)] * pushes),
            )
        )
    if len(levels) < int(count):
        raise RuntimeError(f"only generated {len(levels)} unique levels after {attempts} attempts")
    return levels


def write_boxoban_levels(levels: Sequence[GeneratedSokobanLevel], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for level in levels:
        blocks.append(f"; {level.level_id}")
        blocks.extend(level.lines)
    out.write_text("\n".join(blocks) + "\n")


def _sample_straight_push_level(
    rng: random.Random,
    height: int,
    width: int,
    pushes: int,
    action: Action,
) -> list[str]:
    dy, dx = DELTAS[action]
    valid_goals: list[tuple[int, int]] = []
    for gy in range(1, height - 1):
        for gx in range(1, width - 1):
            box_y = gy - dy * pushes
            box_x = gx - dx * pushes
            player_y = box_y - dy
            player_x = box_x - dx
            if _inside_floor(player_y, player_x, height, width) and _inside_floor(box_y, box_x, height, width):
                valid_goals.append((gy, gx))
    if not valid_goals:
        raise RuntimeError("no valid straight-push level geometry")

    goal_y, goal_x = rng.choice(valid_goals)
    box_y = goal_y - dy * pushes
    box_x = goal_x - dx * pushes
    player_y = box_y - dy
    player_x = box_x - dx

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for y in range(height):
        grid[y][0] = "#"
        grid[y][width - 1] = "#"
    for x in range(width):
        grid[0][x] = "#"
        grid[height - 1][x] = "#"

    grid[goal_y][goal_x] = "."
    grid[box_y][box_x] = "$"
    grid[player_y][player_x] = "@"
    return ["".join(row) for row in grid]


def _inside_floor(y: int, x: int, height: int, width: int) -> bool:
    return 1 <= int(y) < int(height) - 1 and 1 <= int(x) < int(width) - 1
