# Boxoban Kill-Or-Continue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Boxoban/Sokoban kill-or-continue experiment that tests whether decision-aware selective verification beats a strong uncertainty baseline before running the full paper experiment suite.

**Architecture:** Implement a small Python research package, `wmsv`, with isolated modules for Sokoban state dynamics, transition data, learned world models, planners/verifiers, label generation, gates, and analysis. The first milestone is not a polished benchmark; it is a reproducible signal check: healthy verification-value base rates and a Pareto comparison between our gate and uncertainty gating.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, scikit-learn, matplotlib, pytest. Boxoban levels are external data from the DeepMind `boxoban-levels` repository and must not be committed.

---

## 2026-06-28 Execution Override

After Opus review, we are tightening the first execution pass. The original plan is still useful as the full v0 roadmap, but the immediate kill-or-continue should be lighter:

```text
Stage A:
    cloneable Sokoban stepper
    Boxoban level loader
    bounded true-simulator verifier
    synthetically degraded cheap planner / cheap model
    decision-aware gate vs strong uncertainty proxy
    feasibility base-rate sweep over degradation levels

Stage B:
    learned cheap world model
    learned ensemble verifier
    full world-model framing validation
```

Rationale:

```text
If decision-aware gating cannot beat uncertainty-style routing in a
controlled degradation setting where we can tune the cheap/verifier gap,
then training learned world models first is wasted effort.
```

During Stage A, defer Tasks 4, 5, and 8 unless they are needed for the synthetic-degradation experiment. Continue to use TDD for all code.

## Scope

This plan builds the first experiment only:

```text
Boxoban/Sokoban
cheap learned grid dynamics model + shallow planner
learned ensemble verifier + deeper search
decision-aware gate
strong uncertainty gate baseline
feasibility and Pareto analysis
```

This plan does not implement:

```text
MiniGrid controlled benchmark
continuous-control extension
full paper figure suite
large-scale training
GPU cluster orchestration
```

The implementation should stop early if the feasibility precondition fails:

```text
helpful correction rate < 15%
or wasted verification rate < 20%
or V_learned is not meaningfully better than M_c on relevant states
```

## File Structure

Create:

```text
pyproject.toml
requirements.txt
scripts/download_boxoban_levels.py
scripts/run_collect_transitions.py
scripts/run_train_world_model.py
scripts/run_generate_labels.py
scripts/run_train_gate.py
scripts/run_eval_pareto.py
scripts/run_feasibility_report.py
wmsv/__init__.py
wmsv/envs/__init__.py
wmsv/envs/sokoban.py
wmsv/data/__init__.py
wmsv/data/boxoban.py
wmsv/data/transitions.py
wmsv/models/__init__.py
wmsv/models/grid_dynamics.py
wmsv/planning/__init__.py
wmsv/planning/beam.py
wmsv/planning/evaluators.py
wmsv/gating/__init__.py
wmsv/gating/features.py
wmsv/gating/train.py
wmsv/analysis/__init__.py
wmsv/analysis/feasibility.py
wmsv/analysis/pareto.py
tests/envs/test_sokoban.py
tests/data/test_boxoban.py
tests/models/test_grid_dynamics.py
tests/planning/test_beam.py
tests/gating/test_features.py
tests/analysis/test_feasibility.py
docs/plans/2026-06-28-boxoban-kill-or-continue.md
experiments/boxoban_kill_or_continue_v0/README.md
```

Generated but untracked:

```text
data/external/boxoban-levels/
data/processed/boxoban_kill_or_continue_v0/
outputs/boxoban_kill_or_continue_v0/
experiments/boxoban_kill_or_continue_v0/checkpoints/
```

The existing `.gitignore` excludes these large/generated paths.

## Milestone Gates

Do not advance to the next milestone until the current gate passes.

```text
Gate 1:
    Sokoban stepper passes deterministic rule tests.

Gate 2:
    Cheap world model overfits a 256-transition smoke dataset.

Gate 3:
    Learned verifier ensemble is measurably stronger than cheap model
    on held-out transition and planning-state diagnostics.

Gate 4:
    Label dataset has non-degenerate verification value.

Gate 5:
    Decision-aware gate beats uncertainty baseline on at least one
    fixed budget point or shows a clear reason why it fails.
```

## Task 1: Project Packaging

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `wmsv/__init__.py`
- Test: import smoke via `python -c "import wmsv"`

- [ ] **Step 1: Write package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wmsv"
version = "0.1.0"
description = "Decision-aware selective verification for budgeted world-model planning"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "tqdm>=4.66",
    "pyyaml>=6.0",
]

[tool.setuptools.packages.find]
include = ["wmsv*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Write runtime requirements**

Create `requirements.txt`:

```text
numpy>=1.24
torch>=2.0
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.66
pyyaml>=6.0
pytest>=8.0
```

- [ ] **Step 3: Create package init**

Create `wmsv/__init__.py`:

```python
"""World-model selective verification research package."""

__all__ = []
```

- [ ] **Step 4: Run import smoke test**

Run:

```bash
python -c "import wmsv; print('ok')"
```

Expected:

```text
ok
```

- [ ] **Step 5: Commit**

Run:

```bash
git add pyproject.toml requirements.txt wmsv/__init__.py
git commit -m "chore: add Python package skeleton"
git push
```

## Task 2: Sokoban State And Rules

**Files:**
- Create: `wmsv/envs/__init__.py`
- Create: `wmsv/envs/sokoban.py`
- Test: `tests/envs/test_sokoban.py`

The implementation uses a compact local Sokoban stepper for cloneable planning states. External Sokoban libraries should be treated as references, but the experiment needs exact state copying, controlled node budgets, and direct feature extraction.

- [ ] **Step 1: Write failing Sokoban tests**

Create `tests/envs/test_sokoban.py`:

```python
import numpy as np

from wmsv.envs.sokoban import Action, SokobanState, parse_level


LEVEL = [
    "#####",
    "#@$.#",
    "#   #",
    "#####",
]


def test_parse_level_tracks_player_box_and_goal():
    state = parse_level(LEVEL)

    assert state.height == 4
    assert state.width == 5
    assert tuple(state.player) == (1, 1)
    assert state.boxes[1, 2]
    assert state.goals[1, 3]
    assert state.walls[0, 0]


def test_push_box_onto_goal_solves_level():
    state = parse_level(LEVEL)

    next_state, reward, done, info = state.step(Action.RIGHT)

    assert tuple(next_state.player) == (1, 2)
    assert next_state.boxes[1, 3]
    assert reward == 1.0
    assert done is True
    assert info["pushed"] is True


def test_wall_blocks_player_motion():
    state = parse_level(LEVEL)

    next_state, reward, done, info = state.step(Action.UP)

    assert tuple(next_state.player) == (1, 1)
    assert np.array_equal(next_state.boxes, state.boxes)
    assert reward == 0.0
    assert done is False
    assert info["blocked"] is True


def test_encode_has_stable_channel_order():
    state = parse_level(LEVEL)
    encoded = state.encode()

    assert encoded.shape == (5, 4, 5)
    assert encoded[0, 0, 0] == 1.0  # walls
    assert encoded[1, 1, 3] == 1.0  # goals
    assert encoded[2, 1, 2] == 1.0  # boxes
    assert encoded[3, 1, 1] == 1.0  # player
    assert encoded[4].sum() == 1.0  # boxes on goals
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/envs/test_sokoban.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'wmsv.envs'
```

- [ ] **Step 3: Implement Sokoban state**

Create `wmsv/envs/__init__.py`:

```python
"""Environment utilities."""
```

Create `wmsv/envs/sokoban.py`:

```python
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

    def key(self) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
        box_positions = tuple(map(tuple, np.argwhere(self.boxes)))
        return tuple(self.player.tolist()), tuple(sorted(box_positions))

    def step(self, action: Action | int) -> tuple["SokobanState", float, bool, dict[str, bool]]:
        action = Action(int(action))
        dy, dx = DELTAS[action]
        y, x = map(int, self.player)
        ny, nx = y + dy, x + dx

        if self.walls[ny, nx]:
            return self.clone(), 0.0, False, {"blocked": True, "pushed": False}

        boxes = self.boxes.copy()
        player = self.player.copy()

        if boxes[ny, nx]:
            by, bx = ny + dy, nx + dx
            if self.walls[by, bx] or boxes[by, bx]:
                return self.clone(), 0.0, False, {"blocked": True, "pushed": False}
            boxes[ny, nx] = False
            boxes[by, bx] = True

        player[:] = (ny, nx)
        next_state = SokobanState(
            walls=self.walls.copy(),
            goals=self.goals.copy(),
            boxes=boxes,
            player=player,
        )
        done = next_state.is_solved()
        reward = 1.0 if done else 0.0
        return next_state, reward, done, {"blocked": False, "pushed": bool(self.boxes[ny, nx])}


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
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/envs/test_sokoban.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/envs tests/envs/test_sokoban.py
git commit -m "feat: add cloneable Sokoban dynamics"
git push
```

## Task 3: Boxoban Level Loading

**Files:**
- Create: `scripts/download_boxoban_levels.py`
- Create: `wmsv/data/__init__.py`
- Create: `wmsv/data/boxoban.py`
- Test: `tests/data/test_boxoban.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/data/test_boxoban.py`:

```python
from pathlib import Path

from wmsv.data.boxoban import iter_boxoban_levels, parse_boxoban_text


BOXOBAN_TEXT = """
; 0
#####
#@$.#
#   #
#####
; 1
#####
#.@$#
#   #
#####
""".strip()


def test_parse_boxoban_text_returns_level_blocks():
    levels = parse_boxoban_text(BOXOBAN_TEXT)

    assert len(levels) == 2
    assert levels[0].level_id == "0"
    assert levels[0].lines[1] == "#@$.#"
    assert levels[1].level_id == "1"


def test_iter_boxoban_levels_reads_multiple_files(tmp_path: Path):
    folder = tmp_path / "levels"
    folder.mkdir()
    (folder / "000.txt").write_text(BOXOBAN_TEXT)

    levels = list(iter_boxoban_levels(folder, limit=1))

    assert len(levels) == 1
    assert levels[0].source_file.endswith("000.txt")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/data/test_boxoban.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'wmsv.data'
```

- [ ] **Step 3: Implement Boxoban parser**

Create `wmsv/data/__init__.py`:

```python
"""Data loading utilities."""
```

Create `wmsv/data/boxoban.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class BoxobanLevel:
    level_id: str
    lines: list[str]
    source_file: str = ""


def parse_boxoban_text(text: str, source_file: str = "") -> list[BoxobanLevel]:
    levels: list[BoxobanLevel] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id is not None and current_lines:
            levels.append(BoxobanLevel(current_id, current_lines, source_file))
        current_id = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        if line.startswith(";"):
            flush()
            current_id = line[1:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return levels


def iter_boxoban_levels(folder: str | Path, limit: int | None = None) -> Iterator[BoxobanLevel]:
    count = 0
    for path in sorted(Path(folder).glob("*.txt")):
        for level in parse_boxoban_text(path.read_text(), source_file=str(path)):
            yield level
            count += 1
            if limit is not None and count >= limit:
                return
```

- [ ] **Step 4: Add download helper**

Create `scripts/download_boxoban_levels.py`:

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default="data/external/boxoban-levels",
        help="Destination directory for external Boxoban levels.",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Boxoban levels already exist at {dest}")
        return

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/google-deepmind/boxoban-levels.git",
            str(dest),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/data/test_boxoban.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/download_boxoban_levels.py wmsv/data tests/data/test_boxoban.py
git commit -m "feat: add Boxoban level loading"
git push
```

## Task 4: Transition Dataset Collection

**Files:**
- Create: `wmsv/data/transitions.py`
- Create: `scripts/run_collect_transitions.py`
- Test: add transition tests to `tests/data/test_boxoban.py`

Use a small random/heuristic policy to collect transitions. The first goal is not expert play; it is enough state-action coverage for training a cheap dynamics model.

- [ ] **Step 1: Add transition serialization tests**

Append to `tests/data/test_boxoban.py`:

```python
import numpy as np

from wmsv.data.transitions import Transition, transitions_to_npz
from wmsv.envs.sokoban import Action, parse_level


def test_transitions_to_npz_shapes(tmp_path: Path):
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    next_state, reward, done, _ = state.step(Action.RIGHT)
    transition = Transition(state.encode(), int(Action.RIGHT), next_state.encode(), reward, done)
    out_path = tmp_path / "transitions.npz"

    transitions_to_npz([transition], out_path)
    data = np.load(out_path)

    assert data["states"].shape == (1, 5, 4, 5)
    assert data["actions"].tolist() == [0]
    assert data["next_states"].shape == (1, 5, 4, 5)
    assert data["rewards"].tolist() == [1.0]
    assert data["dones"].tolist() == [True]
```

- [ ] **Step 2: Implement transition utilities**

Create `wmsv/data/transitions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


def transitions_to_npz(transitions: list[Transition], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=np.stack([t.state for t in transitions]).astype(np.float32),
        actions=np.array([t.action for t in transitions], dtype=np.int64),
        next_states=np.stack([t.next_state for t in transitions]).astype(np.float32),
        rewards=np.array([t.reward for t in transitions], dtype=np.float32),
        dones=np.array([t.done for t in transitions], dtype=bool),
    )
```

- [ ] **Step 3: Implement collection script**

Create `scripts/run_collect_transitions.py`:

```python
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from wmsv.data.boxoban import iter_boxoban_levels
from wmsv.data.transitions import Transition, transitions_to_npz
from wmsv.envs.sokoban import Action, parse_level


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="data/external/boxoban-levels/unfiltered/train")
    parser.add_argument("--out", default="data/processed/boxoban_kill_or_continue_v0/transitions_smoke.npz")
    parser.add_argument("--num-levels", type=int, default=100)
    parser.add_argument("--steps-per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    transitions: list[Transition] = []

    for level in tqdm(list(iter_boxoban_levels(args.levels, limit=args.num_levels))):
        state = parse_level(level.lines)
        for _ in range(args.steps_per_level):
            action = Action(rng.randrange(4))
            next_state, reward, done, _ = state.step(action)
            transitions.append(Transition(state.encode(), int(action), next_state.encode(), reward, done))
            state = parse_level(level.lines) if done else next_state

    transitions_to_npz(transitions, Path(args.out))
    print(f"Wrote {len(transitions)} transitions to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/data/test_boxoban.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/data/transitions.py scripts/run_collect_transitions.py tests/data/test_boxoban.py
git commit -m "feat: add transition dataset collection"
git push
```

## Task 5: Cheap Grid Dynamics Model

**Files:**
- Create: `wmsv/models/__init__.py`
- Create: `wmsv/models/grid_dynamics.py`
- Create: `scripts/run_train_world_model.py`
- Test: `tests/models/test_grid_dynamics.py`

The first model predicts next-state channels plus reward/done heads. For the kill-or-continue experiment, exact state prediction is enough; policy/value quality can be added later.

- [ ] **Step 1: Write model smoke tests**

Create `tests/models/test_grid_dynamics.py`:

```python
import torch

from wmsv.models.grid_dynamics import GridDynamicsModel


def test_grid_dynamics_output_shapes():
    model = GridDynamicsModel(in_channels=5, hidden_channels=16, num_actions=4)
    states = torch.zeros(2, 5, 10, 10)
    actions = torch.tensor([0, 3])

    out = model(states, actions)

    assert out.next_state_logits.shape == (2, 5, 10, 10)
    assert out.reward.shape == (2,)
    assert out.done_logit.shape == (2,)


def test_grid_dynamics_loss_runs_backward():
    model = GridDynamicsModel(in_channels=5, hidden_channels=16, num_actions=4)
    states = torch.zeros(2, 5, 10, 10)
    actions = torch.tensor([0, 3])
    next_states = torch.zeros(2, 5, 10, 10)
    rewards = torch.zeros(2)
    dones = torch.zeros(2)

    loss = model.loss(states, actions, next_states, rewards, dones)
    loss.backward()

    assert loss.item() >= 0.0
```

- [ ] **Step 2: Implement model**

Create `wmsv/models/__init__.py`:

```python
"""Learned world models."""
```

Create `wmsv/models/grid_dynamics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DynamicsOutput:
    next_state_logits: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor


class GridDynamicsModel(nn.Module):
    def __init__(self, in_channels: int = 5, hidden_channels: int = 64, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.action_embed = nn.Embedding(num_actions, hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.next_head = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reward_head = nn.Linear(hidden_channels, 1)
        self.done_head = nn.Linear(hidden_channels, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> DynamicsOutput:
        hidden = self.encoder(states)
        action = self.action_embed(actions).view(actions.shape[0], -1, 1, 1)
        hidden = hidden + action
        pooled = self.pool(hidden).flatten(1)
        return DynamicsOutput(
            next_state_logits=self.next_head(hidden),
            reward=self.reward_head(pooled).squeeze(-1),
            done_logit=self.done_head(pooled).squeeze(-1),
        )

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        out = self(states, actions)
        next_loss = F.binary_cross_entropy_with_logits(out.next_state_logits, next_states)
        reward_loss = F.mse_loss(out.reward, rewards)
        done_loss = F.binary_cross_entropy_with_logits(out.done_logit, dones)
        return next_loss + 0.1 * reward_loss + 0.1 * done_loss
```

- [ ] **Step 3: Add minimal training script**

Create `scripts/run_train_world_model.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from wmsv.models.grid_dynamics import GridDynamicsModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="experiments/boxoban_kill_or_continue_v0/checkpoints/cheap_model.pt")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = np.load(args.data)
    dataset = TensorDataset(
        torch.tensor(data["states"], dtype=torch.float32),
        torch.tensor(data["actions"], dtype=torch.long),
        torch.tensor(data["next_states"], dtype=torch.float32),
        torch.tensor(data["rewards"], dtype=torch.float32),
        torch.tensor(data["dones"], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = GridDynamicsModel(hidden_channels=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            opt.zero_grad()
            loss = model.loss(*batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden": args.hidden}, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/models/test_grid_dynamics.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/models scripts/run_train_world_model.py tests/models/test_grid_dynamics.py
git commit -m "feat: add cheap grid dynamics model"
git push
```

## Task 6: Beam Planner And Evaluators

**Files:**
- Create: `wmsv/planning/__init__.py`
- Create: `wmsv/planning/beam.py`
- Create: `wmsv/planning/evaluators.py`
- Test: `tests/planning/test_beam.py`

The first planner should support both true-state transitions and learned-model transitions through a common evaluator interface.

- [ ] **Step 1: Write planner tests**

Create `tests/planning/test_beam.py`:

```python
from wmsv.envs.sokoban import Action, parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import TrueEvaluator


def test_true_evaluator_scores_solving_action_highest():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    planner = BeamPlanner(evaluator=TrueEvaluator(), depth=1, width=4)

    result = planner.plan(state)

    assert result.action == int(Action.RIGHT)
    assert result.score == 1.0
    assert result.nodes_expanded == 4


def test_beam_planner_returns_top_scores():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    planner = BeamPlanner(evaluator=TrueEvaluator(), depth=1, width=4)

    result = planner.plan(state)

    assert len(result.top_action_scores) == 4
    assert result.top_action_scores[int(Action.RIGHT)] == 1.0
```

- [ ] **Step 2: Implement planner interfaces**

Create `wmsv/planning/__init__.py`:

```python
"""Planning and verifier utilities."""
```

Create `wmsv/planning/evaluators.py`:

```python
from __future__ import annotations

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
```

Create `wmsv/planning/beam.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from wmsv.envs.sokoban import SokobanState


@dataclass(frozen=True)
class PlanResult:
    action: int
    score: float
    plan: list[int]
    top_action_scores: dict[int, float]
    nodes_expanded: int


class BeamPlanner:
    def __init__(self, evaluator, depth: int, width: int, step_penalty: float = 0.0):
        self.evaluator = evaluator
        self.depth = depth
        self.width = width
        self.step_penalty = step_penalty

    def plan(self, state: SokobanState) -> PlanResult:
        beams = [(0.0, [], state)]
        top_action_scores: dict[int, float] = {0: float("-inf"), 1: float("-inf"), 2: float("-inf"), 3: float("-inf")}
        nodes_expanded = 0

        for _ in range(self.depth):
            candidates = []
            for score, plan, current in beams:
                for action in range(4):
                    step = self.evaluator.step(current, action)
                    nodes_expanded += 1
                    new_score = score + step.reward - self.step_penalty
                    new_plan = plan + [action]
                    first_action = new_plan[0]
                    top_action_scores[first_action] = max(top_action_scores[first_action], new_score)
                    candidates.append((new_score, new_plan, step.state))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[: self.width]

        best_score, best_plan, _ = beams[0]
        return PlanResult(
            action=best_plan[0],
            score=best_score,
            plan=best_plan,
            top_action_scores=top_action_scores,
            nodes_expanded=nodes_expanded,
        )
```

- [ ] **Step 3: Run tests**

Run:

```bash
pytest tests/planning/test_beam.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Commit**

Run:

```bash
git add wmsv/planning tests/planning/test_beam.py
git commit -m "feat: add bounded beam planner"
git push
```

## Task 7: Feasibility Label Generation

**Files:**
- Create: `scripts/run_generate_labels.py`
- Create: `wmsv/analysis/feasibility.py`
- Test: `tests/analysis/test_feasibility.py`

For the first kill-or-continue run, label quality matters more than scale. Use a common true-state evaluator to compare the deployed first actions from `P_c` and `V`.

Initial return definition:

```text
R(a) = 1.0 if, after taking action a, a bounded true-state beam search
       can solve the level within eval_depth and eval_width.
       Otherwise use fraction of boxes on goals as a dense tie-breaker.
```

- [ ] **Step 1: Write feasibility metric tests**

Create `tests/analysis/test_feasibility.py`:

```python
from wmsv.analysis.feasibility import FeasibilityCounts, summarize_labels


def test_summarize_labels_counts_fix_flip_waste():
    labels = [
        {"a_c": 0, "a_v": 1, "r_c": 0.0, "r_v": 1.0},
        {"a_c": 0, "a_v": 1, "r_c": 1.0, "r_v": 0.0},
        {"a_c": 0, "a_v": 0, "r_c": 1.0, "r_v": 1.0},
    ]

    counts = summarize_labels(labels, epsilon=0.01)

    assert counts == FeasibilityCounts(total=3, helpful=1, harmful=1, wasted=1, spurious=0)
    assert counts.helpful_rate == 1 / 3
```

- [ ] **Step 2: Implement feasibility summary**

Create `wmsv/analysis/__init__.py`:

```python
"""Analysis utilities."""
```

Create `wmsv/analysis/feasibility.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilityCounts:
    total: int
    helpful: int
    harmful: int
    wasted: int
    spurious: int

    @property
    def helpful_rate(self) -> float:
        return self.helpful / self.total if self.total else 0.0

    @property
    def harmful_rate(self) -> float:
        return self.harmful / self.total if self.total else 0.0

    @property
    def wasted_rate(self) -> float:
        return self.wasted / self.total if self.total else 0.0


def summarize_labels(labels: list[dict], epsilon: float) -> FeasibilityCounts:
    helpful = harmful = wasted = spurious = 0
    for row in labels:
        action_changed = row["a_c"] != row["a_v"]
        delta = row["r_v"] - row["r_c"]
        if action_changed and delta > epsilon:
            helpful += 1
        elif action_changed and delta < -epsilon:
            harmful += 1
        elif action_changed:
            spurious += 1
        else:
            wasted += 1
    return FeasibilityCounts(len(labels), helpful, harmful, wasted, spurious)
```

- [ ] **Step 3: Create label-generation script skeleton**

Create `scripts/run_generate_labels.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from wmsv.analysis.feasibility import summarize_labels
from wmsv.data.boxoban import iter_boxoban_levels
from wmsv.envs.sokoban import Action, parse_level
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import TrueEvaluator


def boxes_on_goals_fraction(state) -> float:
    total = max(float(state.boxes.sum()), 1.0)
    return float((state.boxes & state.goals).sum()) / total


def evaluate_first_action(state, action: int, eval_planner: BeamPlanner) -> float:
    next_state, reward, done, _ = state.step(Action(action))
    if done:
        return 1.0
    result = eval_planner.plan(next_state)
    if result.score >= 1.0:
        return 1.0
    return boxes_on_goals_fraction(next_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="data/external/boxoban-levels/unfiltered/train")
    parser.add_argument("--out", default="data/processed/boxoban_kill_or_continue_v0/labels_smoke.jsonl")
    parser.add_argument("--num-levels", type=int, default=100)
    parser.add_argument("--cheap-depth", type=int, default=3)
    parser.add_argument("--cheap-width", type=int, default=4)
    parser.add_argument("--verifier-depth", type=int, default=8)
    parser.add_argument("--verifier-width", type=int, default=16)
    parser.add_argument("--eval-depth", type=int, default=8)
    parser.add_argument("--eval-width", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    cheap = BeamPlanner(TrueEvaluator(), depth=args.cheap_depth, width=args.cheap_width)
    verifier = BeamPlanner(TrueEvaluator(), depth=args.verifier_depth, width=args.verifier_width)
    evaluator = BeamPlanner(TrueEvaluator(), depth=args.eval_depth, width=args.eval_width)

    rows = []
    for level in tqdm(list(iter_boxoban_levels(args.levels, limit=args.num_levels))):
        state = parse_level(level.lines)
        cheap_result = cheap.plan(state)
        verifier_result = verifier.plan(state)
        r_c = evaluate_first_action(state, cheap_result.action, evaluator)
        r_v = evaluate_first_action(state, verifier_result.action, evaluator)
        rows.append(
            {
                "level_id": level.level_id,
                "source_file": level.source_file,
                "a_c": cheap_result.action,
                "a_v": verifier_result.action,
                "r_c": r_c,
                "r_v": r_v,
                "cheap_score": cheap_result.score,
                "verifier_score": verifier_result.score,
                "cheap_nodes": cheap_result.nodes_expanded,
                "verifier_nodes": verifier_result.nodes_expanded,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    counts = summarize_labels(rows, epsilon=args.epsilon)
    print(counts)
    print(
        {
            "helpful_rate": counts.helpful_rate,
            "harmful_rate": counts.harmful_rate,
            "wasted_rate": counts.wasted_rate,
        }
    )


if __name__ == "__main__":
    main()
```

This script initially uses true evaluators for both cheap and verifier to debug label plumbing. Replace the cheap and learned verifier evaluators with learned-model evaluators in Task 8 and Task 9.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/analysis/test_feasibility.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/analysis scripts/run_generate_labels.py tests/analysis/test_feasibility.py
git commit -m "feat: add verification label feasibility tooling"
git push
```

## Task 8: Learned Evaluator For Planning

**Files:**
- Modify: `wmsv/planning/evaluators.py`
- Test: extend `tests/planning/test_beam.py`

Add a learned evaluator that applies `GridDynamicsModel` to a Sokoban state and decodes the predicted next state back into a planning state. This is intentionally crude for v0; its job is to create a cheap imperfect imagination path.

- [ ] **Step 1: Add learned-evaluator smoke test**

Append to `tests/planning/test_beam.py`:

```python
from wmsv.models.grid_dynamics import GridDynamicsModel
from wmsv.planning.evaluators import LearnedEvaluator


def test_learned_evaluator_step_returns_sokoban_state():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    model = GridDynamicsModel(hidden_channels=16)
    evaluator = LearnedEvaluator(model)

    step = evaluator.step(state, int(Action.RIGHT))

    assert step.state.walls.shape == state.walls.shape
    assert isinstance(step.reward, float)
    assert isinstance(step.done, bool)
```

- [ ] **Step 2: Implement learned evaluator**

Append to `wmsv/planning/evaluators.py`:

```python
import torch


class LearnedEvaluator:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model.eval()
        self.threshold = threshold

    def step(self, state: SokobanState, action: int) -> EvaluatedStep:
        encoded = torch.tensor(state.encode()[None], dtype=torch.float32)
        actions = torch.tensor([action], dtype=torch.long)
        with torch.no_grad():
            out = self.model(encoded, actions)
        pred = torch.sigmoid(out.next_state_logits[0]).cpu().numpy() > self.threshold

        walls = state.walls.copy()
        goals = state.goals.copy()
        boxes = pred[2].astype(bool)
        player_mask = pred[3].astype(bool)
        player_positions = player_mask.nonzero()
        if len(player_positions[0]) == 0:
            player = state.player.copy()
        else:
            player = torch.tensor([player_positions[0][0], player_positions[1][0]]).numpy()

        next_state = SokobanState(walls=walls, goals=goals, boxes=boxes, player=player)
        reward = float(out.reward.item())
        done = bool(torch.sigmoid(out.done_logit).item() > self.threshold or next_state.is_solved())
        return EvaluatedStep(next_state, reward, done)
```

- [ ] **Step 3: Run tests**

Run:

```bash
pytest tests/planning/test_beam.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 4: Commit**

Run:

```bash
git add wmsv/planning/evaluators.py tests/planning/test_beam.py
git commit -m "feat: add learned model planner evaluator"
git push
```

## Task 9: Gate Features And Baselines

**Files:**
- Create: `wmsv/gating/__init__.py`
- Create: `wmsv/gating/features.py`
- Create: `wmsv/gating/train.py`
- Create: `scripts/run_train_gate.py`
- Test: `tests/gating/test_features.py`

Start with tabular features and logistic regression. This keeps the first result interpretable and fast.

- [ ] **Step 1: Write feature tests**

Create `tests/gating/test_features.py`:

```python
from wmsv.gating.features import label_from_row, row_to_features


def test_label_from_row_requires_action_change_and_gain():
    assert label_from_row({"a_c": 0, "a_v": 1, "r_c": 0.0, "r_v": 1.0}, epsilon=0.01) == 1
    assert label_from_row({"a_c": 0, "a_v": 0, "r_c": 0.0, "r_v": 1.0}, epsilon=0.01) == 0
    assert label_from_row({"a_c": 0, "a_v": 1, "r_c": 1.0, "r_v": 0.0}, epsilon=0.01) == 0


def test_row_to_features_has_stable_keys():
    row = {
        "cheap_score": 0.2,
        "verifier_score": 0.7,
        "cheap_nodes": 12,
        "verifier_nodes": 48,
        "a_c": 0,
        "a_v": 1,
    }

    names, values = row_to_features(row)

    assert names == ["cheap_score", "score_margin_proxy", "cheap_nodes"]
    assert values == [0.2, 0.5, 12.0]
```

- [ ] **Step 2: Implement feature utilities**

Create `wmsv/gating/__init__.py`:

```python
"""Gate training and evaluation."""
```

Create `wmsv/gating/features.py`:

```python
from __future__ import annotations


def label_from_row(row: dict, epsilon: float) -> int:
    return int(row["a_v"] != row["a_c"] and row["r_v"] > row["r_c"] + epsilon)


def row_to_features(row: dict) -> tuple[list[str], list[float]]:
    names = ["cheap_score", "score_margin_proxy", "cheap_nodes"]
    values = [
        float(row.get("cheap_score", 0.0)),
        float(row.get("verifier_score", 0.0)) - float(row.get("cheap_score", 0.0)),
        float(row.get("cheap_nodes", 0.0)),
    ]
    return names, values
```

Create `wmsv/gating/train.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from wmsv.gating.features import label_from_row, row_to_features


def load_feature_matrix(path: str | Path, epsilon: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    feature_rows = []
    labels = []
    names: list[str] | None = None
    for row in rows:
        current_names, values = row_to_features(row)
        names = current_names if names is None else names
        feature_rows.append(values)
        labels.append(label_from_row(row, epsilon))
    return np.array(feature_rows, dtype=np.float32), np.array(labels, dtype=np.int64), names or []


def train_logistic_gate(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(x, y)
    return model
```

- [ ] **Step 3: Add training script**

Create `scripts/run_train_gate.py`:

```python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from wmsv.gating.train import load_feature_matrix, train_logistic_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="experiments/boxoban_kill_or_continue_v0/checkpoints/logistic_gate.pkl")
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    x, y, names = load_feature_matrix(args.labels, args.epsilon)
    model = train_logistic_gate(x, y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model": model, "feature_names": names}, f)
    print({"num_examples": int(len(y)), "positive_rate": float(y.mean()), "features": names})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/gating/test_features.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/gating scripts/run_train_gate.py tests/gating/test_features.py
git commit -m "feat: add decision-aware gate training"
git push
```

## Task 10: Pareto Evaluation

**Files:**
- Create: `wmsv/analysis/pareto.py`
- Create: `scripts/run_eval_pareto.py`

The first Pareto script can operate on label rows and gate scores. It does not need full environment rollout until the label/gate signal is healthy.

- [ ] **Step 1: Implement Pareto utilities**

Create `wmsv/analysis/pareto.py`:

```python
from __future__ import annotations


def policy_return(rows: list[dict], verify_mask: list[bool]) -> float:
    returns = []
    for row, verify in zip(rows, verify_mask):
        returns.append(float(row["r_v"] if verify else row["r_c"]))
    return sum(returns) / len(returns) if returns else 0.0


def budget_fraction(verify_mask: list[bool]) -> float:
    return sum(bool(v) for v in verify_mask) / len(verify_mask) if verify_mask else 0.0
```

- [ ] **Step 2: Add Pareto script**

Create `scripts/run_eval_pareto.py`:

```python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wmsv.analysis.pareto import budget_fraction, policy_return
from wmsv.gating.train import load_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--out", default="outputs/boxoban_kill_or_continue_v0/pareto.png")
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    x, _, _ = load_feature_matrix(args.labels, args.epsilon)
    with Path(args.gate).open("rb") as f:
        gate = pickle.load(f)["model"]
    scores = gate.predict_proba(x)[:, 1]

    thresholds = np.linspace(0.0, 1.0, 21)
    xs, ys = [], []
    for threshold in thresholds:
        mask = [score >= threshold for score in scores]
        xs.append(budget_fraction(mask))
        ys.append(policy_return(rows, mask))

    cheap = policy_return(rows, [False] * len(rows))
    always = policy_return(rows, [True] * len(rows))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o", label="decision-aware gate")
    plt.scatter([0], [cheap], label="cheap only")
    plt.scatter([1], [always], label="always verify")
    plt.xlabel("verifier call fraction")
    plt.ylabel("mean evaluation return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

Run:

```bash
git add wmsv/analysis/pareto.py scripts/run_eval_pareto.py
git commit -m "feat: add first Pareto evaluation script"
git push
```

## Task 11: Experiment README And Runbook

**Files:**
- Create: `experiments/boxoban_kill_or_continue_v0/README.md`

- [ ] **Step 1: Write runbook**

Create `experiments/boxoban_kill_or_continue_v0/README.md`:

```markdown
# Boxoban Kill-Or-Continue v0

Status: planned.

Purpose: test whether decision-aware selective verification has a real signal before running the full paper experiment suite.

## Commands

Download external Boxoban levels:

```bash
python scripts/download_boxoban_levels.py
```

Collect smoke transitions:

```bash
python scripts/run_collect_transitions.py \
  --levels data/external/boxoban-levels/unfiltered/train \
  --out data/processed/boxoban_kill_or_continue_v0/transitions_smoke.npz \
  --num-levels 100 \
  --steps-per-level 100
```

Train cheap model:

```bash
python scripts/run_train_world_model.py \
  --data data/processed/boxoban_kill_or_continue_v0/transitions_smoke.npz \
  --out experiments/boxoban_kill_or_continue_v0/checkpoints/cheap_model.pt \
  --epochs 5
```

Generate initial labels:

```bash
python scripts/run_generate_labels.py \
  --levels data/external/boxoban-levels/unfiltered/train \
  --out data/processed/boxoban_kill_or_continue_v0/labels_smoke.jsonl \
  --num-levels 100
```

Train gate:

```bash
python scripts/run_train_gate.py \
  --labels data/processed/boxoban_kill_or_continue_v0/labels_smoke.jsonl \
  --out experiments/boxoban_kill_or_continue_v0/checkpoints/logistic_gate.pkl
```

Plot first Pareto curve:

```bash
python scripts/run_eval_pareto.py \
  --labels data/processed/boxoban_kill_or_continue_v0/labels_smoke.jsonl \
  --gate experiments/boxoban_kill_or_continue_v0/checkpoints/logistic_gate.pkl \
  --out outputs/boxoban_kill_or_continue_v0/pareto.png
```

## Kill-Or-Continue Criteria

Continue if:

- helpful correction rate is at least 15%;
- wasted verification rate is at least 20%;
- learned verifier condition is better than cheap planner in relevant states;
- decision-aware gate beats strong uncertainty gating at one or more matched budgets.

Pause and revise if:

- labels are degenerate;
- uncertainty gating matches or beats decision-aware gating everywhere;
- V_learned shares too many failures with the cheap model;
- only V_sim works.
```

- [ ] **Step 2: Commit**

Run:

```bash
git add experiments/boxoban_kill_or_continue_v0/README.md
git commit -m "docs: add Boxoban kill-or-continue runbook"
git push
```

## Task 12: Opus Review Checkpoint

**Files:**
- Create: `notes/2026-06-28-opus-review-boxoban-plan.md`

Use Opus as an external reviewer after Task 11 and before long runs.

- [ ] **Step 1: Ask Opus only for critique**

Prompt content:

```text
We are implementing the first kill-or-continue experiment for
Decision-Aware Selective Verification for Budgeted World-Model Planning.

Core experiment:
- Boxoban/Sokoban.
- Cheap learned grid dynamics model + shallow beam/MPC.
- Learned ensemble/larger model verifier + deeper search.
- Strong uncertainty baseline using model ensemble disagreement and
  top-action/value disagreement.
- Primary goal: decide whether decision-aware gate beats uncertainty
  baseline before full experiments.

Please critique:
1. Is the first implementation too weak to test the paper claim?
2. What one baseline must not be skipped?
3. What failure signal should make us stop?
4. What simplification should we make for speed?

Do not rewrite the project. Give concrete reviewer-style risks.
```

- [ ] **Step 2: Write a note**

Create `notes/2026-06-28-opus-review-boxoban-plan.md` with:

```markdown
# Opus Review: Boxoban Kill-Or-Continue Plan

Date: 2026-06-28

Status: external-model critique. Treat as input, not authority.

## Useful Critiques

## Rejected Or Deferred Suggestions

## Adopted Changes
```

- [ ] **Step 3: Commit**

Run:

```bash
git add notes/2026-06-28-opus-review-boxoban-plan.md
git commit -m "notes: add Opus review of Boxoban plan"
git push
```

## First Execution Order

Run tasks in this order:

```text
Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6 -> Task 7
```

Then stop and inspect:

```text
Can we generate a non-degenerate label dataset with true/planner debugging?
Are the base rates plausible?
```

Only then proceed:

```text
Task 8 -> Task 9 -> Task 10 -> Task 11 -> Task 12
```

The first serious research decision happens after Task 12:

```text
If the uncertainty baseline is already as good as decision-aware gating,
do not scale up. Revise the story or design the uncertainty-relevance
stress test first.
```
