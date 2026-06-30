# Phase 1 Dual Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 dual pilot for DIVE-v0 selective verification: shared labels/budgeting, DIVE-v0 gate, a Boxoban learned-dynamics pilot, and a Maze learned-dynamics pilot.

**Architecture:** Keep Stage A intact and add Phase 1 modules beside it. DIVE-v0 is a small numpy multi-head gate trained from offline verifier labels. Boxoban starts with a dependency-free learned transition model from offline transitions; Maze uses a dependency-free vector-state environment with learned linear dynamics and random-shooting planners.

**Tech Stack:** Python 3.10+, numpy, pytest, existing `wmsv` package, no PyTorch/Gymnasium dependency for Phase 1.

---

## Scope

This plan implements the Phase 1 pilot only. It does not run full dual-main experiments, paper plots, or neural world-model training.

Phase 1 success means the repository can produce two pilot JSON reports:

```text
outputs/phase1_dual_pilot/boxoban_pilot.json
outputs/phase1_dual_pilot/maze_pilot.json
```

Each report must include:

```text
cheap-only return/success
always-verify return/success
positive label rate
DIVE-v0 helpful-label AUROC
DIVE-v0 at 20% budget vs random
compute summary
go/no-go status
```

## File Structure

Create or modify these files.

```text
wmsv/analysis/verification_value.py
    Shared DIVE label definitions, helpful/harm/waste classification, AUROC,
    go/no-go checks.

wmsv/analysis/budgeting.py
    Budget masks, policy return summaries, compute accounting, Pareto helpers.

wmsv/gating/dive.py
    DIVE-v0 numpy multi-head gate and uncertainty-label gate baseline.

wmsv/gating/__init__.py
    Export DIVE-v0 symbols.

wmsv/data/transitions.py
    Generic transition record and split helpers for learned model pilots.

wmsv/models/__init__.py
    New models package.

wmsv/models/sokoban_memory.py
    Dependency-free learned Sokoban transition model from offline transitions.

wmsv/models/linear_dynamics.py
    Dependency-free learned vector dynamics for Maze pilot.

wmsv/envs/point_maze.py
    Small vector-state maze environment with walls and deterministic dynamics.

wmsv/planning/shooting.py
    Random-shooting planner for vector dynamics and true Maze simulator.

wmsv/pilots/__init__.py
    Pilot package marker.

wmsv/pilots/boxoban.py
    Boxoban learned-model pilot row builder and evaluator.

wmsv/pilots/maze.py
    Maze learned-model pilot row builder and evaluator.

scripts/run_boxoban_learned_pilot.py
    CLI for Boxoban pilot JSON report.

scripts/run_maze_learned_pilot.py
    CLI for Maze pilot JSON report.

experiments/phase1_dual_pilot/README.md
    Experiment log and commands.

tests/analysis/test_verification_value.py
tests/analysis/test_budgeting.py
tests/gating/test_dive.py
tests/data/test_transitions.py
tests/models/test_sokoban_memory.py
tests/models/test_linear_dynamics.py
tests/envs/test_point_maze.py
tests/planning/test_shooting.py
tests/pilots/test_boxoban_pilot.py
tests/pilots/test_maze_pilot.py
tests/scripts/test_phase1_pilot_scripts.py
```

## Task 1: Verification-Value Labels

**Files:**
- Create: `wmsv/analysis/verification_value.py`
- Test: `tests/analysis/test_verification_value.py`

- [ ] **Step 1: Write failing tests**

Create `tests/analysis/test_verification_value.py`:

```python
import math

from wmsv.analysis.verification_value import (
    VerificationLabels,
    auroc,
    classify_verification,
    go_no_go_status,
)


def test_classify_verification_marks_helpful_change():
    labels = classify_verification(a_c=0, a_v=1, r_c=0.2, r_v=0.8, epsilon=0.05)

    assert labels == VerificationLabels(
        y_change=1,
        y_helpful=1,
        y_harm=0,
        y_waste=0,
        delta_r=0.6,
    )


def test_classify_verification_marks_harmful_change():
    labels = classify_verification(a_c=0, a_v=2, r_c=0.8, r_v=0.1, epsilon=0.05)

    assert labels.y_change == 1
    assert labels.y_helpful == 0
    assert labels.y_harm == 1
    assert labels.y_waste == 0
    assert labels.delta_r == -0.7000000000000001


def test_classify_verification_marks_waste_when_action_same():
    labels = classify_verification(a_c=0, a_v=0, r_c=0.4, r_v=0.9, epsilon=0.05)

    assert labels.y_change == 0
    assert labels.y_helpful == 0
    assert labels.y_harm == 0
    assert labels.y_waste == 1


def test_classify_verification_supports_continuous_controls():
    labels = classify_verification(
        a_c=[0.0, 0.0],
        a_v=[0.2, 0.0],
        r_c=0.1,
        r_v=0.4,
        epsilon=0.05,
        action_delta=0.1,
    )

    assert labels.y_change == 1
    assert labels.y_helpful == 1


def test_auroc_returns_pairwise_probability():
    value = auroc(scores=[0.1, 0.9, 0.8, 0.2], labels=[0, 1, 1, 0])

    assert value == 1.0


def test_auroc_returns_nan_for_single_class():
    value = auroc(scores=[0.1, 0.2], labels=[1, 1])

    assert math.isnan(value)


def test_go_no_go_status_checks_phase1_thresholds():
    status = go_no_go_status(
        cheap_success=0.40,
        always_verify_success=0.55,
        positive_label_rate=0.25,
        helpful_auroc=0.66,
        budget20_gain=0.04,
        min_always_gain=0.10,
        min_auroc=0.60,
    )

    assert status["passed"] is True
    assert status["failures"] == []
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/analysis/test_verification_value.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.analysis.verification_value'`.

- [ ] **Step 3: Implement labels and AUROC**

Create `wmsv/analysis/verification_value.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VerificationLabels:
    y_change: int
    y_helpful: int
    y_harm: int
    y_waste: int
    delta_r: float


def _action_changed(a_c, a_v, action_delta: float | None) -> bool:
    if action_delta is None:
        return a_c != a_v
    c = np.asarray(a_c, dtype=np.float64)
    v = np.asarray(a_v, dtype=np.float64)
    return bool(np.linalg.norm(v - c) > float(action_delta))


def classify_verification(
    a_c,
    a_v,
    r_c: float,
    r_v: float,
    epsilon: float = 0.01,
    action_delta: float | None = None,
) -> VerificationLabels:
    changed = _action_changed(a_c, a_v, action_delta)
    delta_r = float(r_v) - float(r_c)
    helpful = changed and delta_r > float(epsilon)
    harmful = changed and delta_r < -float(epsilon)
    wasted = (not changed) or abs(delta_r) <= float(epsilon)
    return VerificationLabels(
        y_change=int(changed),
        y_helpful=int(helpful),
        y_harm=int(harmful),
        y_waste=int(wasted),
        delta_r=delta_r,
    )


def add_verification_labels(row: dict, epsilon: float = 0.01, action_delta: float | None = None) -> dict:
    labels = classify_verification(
        row["a_c"],
        row["a_v"],
        float(row["r_c"]),
        float(row["r_v"]),
        epsilon=epsilon,
        action_delta=action_delta,
    )
    enriched = dict(row)
    enriched.update(
        {
            "y_change": labels.y_change,
            "y_helpful": labels.y_helpful,
            "y_harm": labels.y_harm,
            "y_waste": labels.y_waste,
            "delta_r": labels.delta_r,
            "label": labels.y_helpful,
        }
    )
    return enriched


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels)]
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def positive_label_rate(rows: Sequence[dict], label_key: str = "y_helpful") -> float:
    if not rows:
        return 0.0
    return sum(int(row[label_key]) for row in rows) / len(rows)


def go_no_go_status(
    cheap_success: float,
    always_verify_success: float,
    positive_label_rate: float,
    helpful_auroc: float,
    budget20_gain: float,
    min_always_gain: float,
    min_auroc: float,
) -> dict:
    failures: list[str] = []
    if not 0.15 < float(cheap_success) < 0.85:
        failures.append("cheap_success_outside_sweet_spot")
    if float(always_verify_success) - float(cheap_success) < float(min_always_gain):
        failures.append("always_verify_gap_too_small")
    if not 0.15 <= float(positive_label_rate) <= 0.50:
        failures.append("positive_label_rate_outside_range")
    if math.isnan(float(helpful_auroc)) or float(helpful_auroc) < float(min_auroc):
        failures.append("helpful_auroc_too_low")
    if float(budget20_gain) < 0.03:
        failures.append("budget20_gain_too_small")
    return {"passed": not failures, "failures": failures}
```

- [ ] **Step 4: Run the tests**

Run:

```bash
PYTHONPATH=. pytest tests/analysis/test_verification_value.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/analysis/verification_value.py tests/analysis/test_verification_value.py
git commit -m "feat: add verification value labels"
```

## Task 2: Budgeting And Compute Accounting

**Files:**
- Create: `wmsv/analysis/budgeting.py`
- Test: `tests/analysis/test_budgeting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/analysis/test_budgeting.py`:

```python
from wmsv.analysis.budgeting import (
    compute_summary,
    mean_return_for_mask,
    threshold_budget_mask,
)


ROWS = [
    {"r_c": 0.0, "r_v": 1.0, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 0.5, "r_v": 0.5, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 0.2, "r_v": 0.8, "cheap_nodes": 10, "verifier_nodes": 100},
    {"r_c": 1.0, "r_v": 0.0, "cheap_nodes": 10, "verifier_nodes": 100},
]


def test_threshold_budget_mask_selects_highest_scores():
    mask = threshold_budget_mask([0.2, 0.9, 0.4, 0.1], budget_fraction=0.5)

    assert mask == [False, True, True, False]


def test_mean_return_for_mask_uses_verified_rows():
    value = mean_return_for_mask(ROWS, [True, False, True, False])

    assert value == (1.0 + 0.5 + 0.8 + 1.0) / 4.0


def test_compute_summary_counts_verifier_calls_and_nodes():
    summary = compute_summary(ROWS, [True, False, True, False])

    assert summary["verifier_call_fraction"] == 0.5
    assert summary["cheap_nodes"] == 40
    assert summary["verifier_nodes"] == 200
    assert summary["total_nodes"] == 240
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/analysis/test_budgeting.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.analysis.budgeting'`.

- [ ] **Step 3: Implement budgeting helpers**

Create `wmsv/analysis/budgeting.py`:

```python
from __future__ import annotations

from collections.abc import Sequence


def threshold_budget_mask(scores: Sequence[float], budget_fraction: float) -> list[bool]:
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0, 1]")
    n = len(scores)
    if n == 0:
        return []
    count = int(round(n * float(budget_fraction)))
    if count <= 0:
        return [False] * n
    order = sorted(range(n), key=lambda idx: float(scores[idx]), reverse=True)
    chosen = set(order[:count])
    return [idx in chosen for idx in range(n)]


def mean_return_for_mask(rows: Sequence[dict], verify_mask: Sequence[bool]) -> float:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have equal length")
    if not rows:
        return 0.0
    total = 0.0
    for row, verify in zip(rows, verify_mask):
        total += float(row["r_v"] if verify else row["r_c"])
    return total / len(rows)


def compute_summary(rows: Sequence[dict], verify_mask: Sequence[bool]) -> dict:
    if len(rows) != len(verify_mask):
        raise ValueError("rows and verify_mask must have equal length")
    cheap_nodes = sum(int(row.get("cheap_nodes", 0)) for row in rows)
    verifier_nodes = sum(int(row.get("verifier_nodes", 0)) for row, verify in zip(rows, verify_mask) if verify)
    selected = sum(1 for value in verify_mask if value)
    return {
        "verifier_calls": selected,
        "verifier_call_fraction": selected / len(rows) if rows else 0.0,
        "cheap_nodes": cheap_nodes,
        "verifier_nodes": verifier_nodes,
        "total_nodes": cheap_nodes + verifier_nodes,
    }
```

- [ ] **Step 4: Run the tests**

Run:

```bash
PYTHONPATH=. pytest tests/analysis/test_budgeting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/analysis/budgeting.py tests/analysis/test_budgeting.py
git commit -m "feat: add budget accounting helpers"
```

## Task 3: DIVE-v0 Gate

**Files:**
- Create: `wmsv/gating/dive.py`
- Modify: `wmsv/gating/__init__.py`
- Test: `tests/gating/test_dive.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gating/test_dive.py`:

```python
import math

from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0, fit_uncertainty_label_gate


ROWS = [
    {
        "score_margin": 0.1,
        "ensemble_uncertainty": 0.8,
        "cheap_score": 0.1,
        "y_change": 1,
        "y_helpful": 1,
        "y_harm": 0,
        "y_waste": 0,
        "delta_r": 0.8,
        "uncertainty_label": 1,
    },
    {
        "score_margin": 0.2,
        "ensemble_uncertainty": 0.7,
        "cheap_score": 0.2,
        "y_change": 1,
        "y_helpful": 1,
        "y_harm": 0,
        "y_waste": 0,
        "delta_r": 0.7,
        "uncertainty_label": 1,
    },
    {
        "score_margin": 1.0,
        "ensemble_uncertainty": 0.1,
        "cheap_score": 0.9,
        "y_change": 0,
        "y_helpful": 0,
        "y_harm": 0,
        "y_waste": 1,
        "delta_r": 0.0,
        "uncertainty_label": 0,
    },
    {
        "score_margin": 1.1,
        "ensemble_uncertainty": 0.2,
        "cheap_score": 0.8,
        "y_change": 1,
        "y_helpful": 0,
        "y_harm": 1,
        "y_waste": 0,
        "delta_r": -0.5,
        "uncertainty_label": 0,
    },
]


def test_dive_scores_helpful_rows_above_wasted_rows():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_dive_v0(ROWS, schema)

    helpful = gate.score({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})
    wasted = gate.score({"score_margin": 1.05, "ensemble_uncertainty": 0.15, "cheap_score": 0.9})

    assert helpful > wasted


def test_dive_predict_heads_returns_named_outputs():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_dive_v0(ROWS, schema)

    heads = gate.predict_heads({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})

    assert set(heads) == {"p_change", "delta_hat", "p_harm", "p_waste"}
    assert 0.0 <= heads["p_change"] <= 1.0
    assert 0.0 <= heads["p_harm"] <= 1.0
    assert 0.0 <= heads["p_waste"] <= 1.0
    assert math.isfinite(heads["delta_hat"])


def test_uncertainty_label_gate_uses_same_schema():
    schema = DIVEFeatureSchema(["score_margin", "ensemble_uncertainty", "cheap_score"])
    gate = fit_uncertainty_label_gate(ROWS, schema)

    high = gate.score({"score_margin": 0.15, "ensemble_uncertainty": 0.75, "cheap_score": 0.1})
    low = gate.score({"score_margin": 1.05, "ensemble_uncertainty": 0.15, "cheap_score": 0.9})

    assert high > low
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/gating/test_dive.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.gating.dive'`.

- [ ] **Step 3: Implement DIVE-v0**

Create `wmsv/gating/dive.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class DIVEFeatureSchema:
    feature_names: list[str]

    def vector(self, row: dict) -> np.ndarray:
        return np.array([float(row[name]) for name in self.feature_names], dtype=np.float64)


@dataclass(frozen=True)
class DIVEV0Gate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    change_weights: np.ndarray
    change_bias: float
    harm_weights: np.ndarray
    harm_bias: float
    waste_weights: np.ndarray
    waste_bias: float
    delta_weights: np.ndarray
    delta_bias: float
    alpha: float = 1.0
    verification_cost: float = 0.0

    def _standardized(self, row: dict) -> np.ndarray:
        return (self.schema.vector(row) - self.mean) / self.scale

    def predict_heads(self, row: dict) -> dict:
        x = self._standardized(row)
        return {
            "p_change": _sigmoid_scalar(float(x @ self.change_weights + self.change_bias)),
            "delta_hat": float(x @ self.delta_weights + self.delta_bias),
            "p_harm": _sigmoid_scalar(float(x @ self.harm_weights + self.harm_bias)),
            "p_waste": _sigmoid_scalar(float(x @ self.waste_weights + self.waste_bias)),
        }

    def score(self, row: dict) -> float:
        heads = self.predict_heads(row)
        return (
            heads["p_change"] * max(heads["delta_hat"], 0.0)
            - self.alpha * heads["p_harm"]
            - self.verification_cost
        )


@dataclass(frozen=True)
class BinaryLabelGate:
    schema: DIVEFeatureSchema
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def score(self, row: dict) -> float:
        x = (self.schema.vector(row) - self.mean) / self.scale
        return float(x @ self.weights + self.bias)


def _feature_matrix(rows: Sequence[dict], schema: DIVEFeatureSchema) -> np.ndarray:
    return np.stack([schema.vector(row) for row in rows]).astype(np.float64)


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _sigmoid_scalar(value: float) -> float:
    return float(_sigmoid(np.array([value], dtype=np.float64))[0])


def _fit_binary_head(x: np.ndarray, y: np.ndarray, steps: int = 800, learning_rate: float = 0.1) -> tuple[np.ndarray, float]:
    if len(set(int(v) for v in y)) < 2:
        return np.zeros(x.shape[1], dtype=np.float64), float(np.mean(y))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive = max(float((y == 1).sum()), 1.0)
    negative = max(float((y == 0).sum()), 1.0)
    sample_weight = np.where(y == 1, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
    for _ in range(steps):
        probs = _sigmoid(x @ weights + bias)
        error = (probs - y) * sample_weight
        weights -= learning_rate * (x.T @ error) / len(y)
        bias -= learning_rate * float(error.mean())
    return weights, bias


def _fit_regression_head(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, float]:
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef[:-1], float(coef[-1])


def fit_dive_v0(
    rows: Sequence[dict],
    schema: DIVEFeatureSchema,
    alpha: float = 1.0,
    verification_cost: float = 0.0,
) -> DIVEV0Gate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    change_w, change_b = _fit_binary_head(x, np.array([int(row["y_change"]) for row in rows], dtype=np.float64))
    harm_w, harm_b = _fit_binary_head(x, np.array([int(row["y_harm"]) for row in rows], dtype=np.float64))
    waste_w, waste_b = _fit_binary_head(x, np.array([int(row["y_waste"]) for row in rows], dtype=np.float64))
    delta_w, delta_b = _fit_regression_head(x, np.array([float(row["delta_r"]) for row in rows], dtype=np.float64))
    return DIVEV0Gate(
        schema=schema,
        mean=mean,
        scale=scale,
        change_weights=change_w,
        change_bias=change_b,
        harm_weights=harm_w,
        harm_bias=harm_b,
        waste_weights=waste_w,
        waste_bias=waste_b,
        delta_weights=delta_w,
        delta_bias=delta_b,
        alpha=float(alpha),
        verification_cost=float(verification_cost),
    )


def fit_uncertainty_label_gate(rows: Sequence[dict], schema: DIVEFeatureSchema) -> BinaryLabelGate:
    features = _feature_matrix(rows, schema)
    x, mean, scale = _standardize(features)
    weights, bias = _fit_binary_head(
        x,
        np.array([int(row["uncertainty_label"]) for row in rows], dtype=np.float64),
    )
    return BinaryLabelGate(schema=schema, mean=mean, scale=scale, weights=weights, bias=bias)
```

- [ ] **Step 4: Export DIVE symbols**

Modify `wmsv/gating/__init__.py` to contain:

```python
from wmsv.gating.dive import DIVEFeatureSchema, DIVEV0Gate, fit_dive_v0, fit_uncertainty_label_gate

__all__ = [
    "DIVEFeatureSchema",
    "DIVEV0Gate",
    "fit_dive_v0",
    "fit_uncertainty_label_gate",
]
```

- [ ] **Step 5: Run DIVE tests**

Run:

```bash
PYTHONPATH=. pytest tests/gating/test_dive.py -q
```

Expected: PASS.

- [ ] **Step 6: Run existing gate tests**

Run:

```bash
PYTHONPATH=. pytest tests/gating/test_simple_gate.py tests/gating/test_dive.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add wmsv/gating/dive.py wmsv/gating/__init__.py tests/gating/test_dive.py
git commit -m "feat: add DIVE v0 gate"
```

## Task 4: Transition Data Utilities

**Files:**
- Create: `wmsv/data/transitions.py`
- Test: `tests/data/test_transitions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_transitions.py`:

```python
import numpy as np

from wmsv.data.transitions import Transition, split_transitions


def test_transition_stores_arrays_and_metadata():
    transition = Transition(
        state=np.array([0.0, 1.0]),
        action=np.array([1.0]),
        next_state=np.array([0.5, 1.0]),
        reward=0.2,
        done=False,
        info={"level_id": "x"},
    )

    assert transition.reward == 0.2
    assert transition.info["level_id"] == "x"


def test_split_transitions_is_deterministic():
    transitions = [
        Transition(np.array([i]), np.array([0]), np.array([i + 1]), 0.0, False, {})
        for i in range(10)
    ]

    train, valid = split_transitions(transitions, valid_fraction=0.3, seed=7)

    assert len(train) == 7
    assert len(valid) == 3
    assert [int(t.state[0]) for t in valid] == [8, 3, 1]
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/data/test_transitions.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.data.transitions'`.

- [ ] **Step 3: Implement transitions**

Create `wmsv/data/transitions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

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
    rng = np.random.default_rng(int(seed))
    indices = list(range(len(transitions)))
    rng.shuffle(indices)
    valid_count = int(round(len(indices) * float(valid_fraction)))
    valid_indices = set(indices[:valid_count])
    train = [transition for idx, transition in enumerate(transitions) if idx not in valid_indices]
    valid = [transition for idx, transition in enumerate(transitions) if idx in valid_indices]
    return train, valid
```

- [ ] **Step 4: Run the tests**

Run:

```bash
PYTHONPATH=. pytest tests/data/test_transitions.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/data/transitions.py tests/data/test_transitions.py
git commit -m "feat: add transition data utilities"
```

## Task 5: Boxoban Learned Transition Model

**Files:**
- Create: `wmsv/models/__init__.py`
- Create: `wmsv/models/sokoban_memory.py`
- Test: `tests/models/test_sokoban_memory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_sokoban_memory.py`:

```python
from wmsv.envs.sokoban import Action, parse_level
from wmsv.models.sokoban_memory import MemorizedSokobanModel, collect_sokoban_transitions


def test_memorized_sokoban_model_replays_seen_transition():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    transitions = collect_sokoban_transitions([("tiny", state)], action_sequences=[[int(Action.RIGHT)]])
    model = MemorizedSokobanModel.fit(transitions)

    step = model.step(state, int(Action.RIGHT))

    assert step.done is True
    assert step.reward == 1.0


def test_memorized_sokoban_model_uses_noop_for_unseen_transition():
    state = parse_level(["#####", "#@$.#", "#   #", "#####"])
    model = MemorizedSokobanModel.fit([])

    step = model.step(state, int(Action.RIGHT))

    assert step.state.encode().tolist() == state.encode().tolist()
    assert step.reward == 0.0
    assert step.done is False
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/models/test_sokoban_memory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.models'`.

- [ ] **Step 3: Implement memorized Sokoban dynamics**

Create `wmsv/models/__init__.py`:

```python
"""Learned model utilities for Phase 1 pilots."""
```

Create `wmsv/models/sokoban_memory.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/models/test_sokoban_memory.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/models/__init__.py wmsv/models/sokoban_memory.py tests/models/test_sokoban_memory.py
git commit -m "feat: add memorized Sokoban dynamics model"
```

## Task 6: PointMaze Environment

**Files:**
- Create: `wmsv/envs/point_maze.py`
- Test: `tests/envs/test_point_maze.py`

- [ ] **Step 1: Write failing tests**

Create `tests/envs/test_point_maze.py`:

```python
import numpy as np

from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec


def test_point_maze_moves_toward_goal():
    env = PointMazeEnv(PointMazeSpec.empty(size=5, start=(1.0, 1.0), goal=(3.0, 1.0)))

    step = env.step(env.reset(), np.array([1.0, 0.0]))

    assert step.state[0] > 1.0
    assert step.reward > -3.0


def test_point_maze_blocks_wall_crossing():
    spec = PointMazeSpec(
        size=5,
        start=np.array([1.0, 1.0], dtype=np.float64),
        goal=np.array([3.0, 1.0], dtype=np.float64),
        walls={(2, 1)},
    )
    env = PointMazeEnv(spec)

    step = env.step(env.reset(), np.array([1.0, 0.0]))

    assert step.state[0] < 2.0
    assert step.info["blocked"] is True
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/envs/test_point_maze.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.envs.point_maze'`.

- [ ] **Step 3: Implement PointMaze**

Create `wmsv/envs/point_maze.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/envs/test_point_maze.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/envs/point_maze.py tests/envs/test_point_maze.py
git commit -m "feat: add point maze environment"
```

## Task 7: Linear Learned Dynamics

**Files:**
- Create: `wmsv/models/linear_dynamics.py`
- Test: `tests/models/test_linear_dynamics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_linear_dynamics.py`:

```python
import numpy as np

from wmsv.data.transitions import Transition
from wmsv.models.linear_dynamics import LinearDynamicsModel


def test_linear_dynamics_fits_simple_translation():
    transitions = []
    for x in range(5):
        state = np.array([float(x), 0.0])
        action = np.array([1.0, 0.0])
        transitions.append(
            Transition(state, action, state + np.array([0.5, 0.0]), -1.0, False, {})
        )
    model = LinearDynamicsModel.fit(transitions, ridge=1e-6)

    step = model.step(np.array([2.0, 0.0]), np.array([1.0, 0.0]))

    assert abs(step.state[0] - 2.5) < 1e-4
    assert step.reward < 0.0
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/models/test_linear_dynamics.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.models.linear_dynamics'`.

- [ ] **Step 3: Implement linear dynamics**

Create `wmsv/models/linear_dynamics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

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
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/models/test_linear_dynamics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/models/linear_dynamics.py tests/models/test_linear_dynamics.py
git commit -m "feat: add linear learned dynamics"
```

## Task 8: Random-Shooting Planner

**Files:**
- Create: `wmsv/planning/shooting.py`
- Test: `tests/planning/test_shooting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/planning/test_shooting.py`:

```python
import numpy as np

from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec
from wmsv.planning.shooting import ShootingPlanner


def test_shooting_planner_returns_action_and_plan():
    env = PointMazeEnv(PointMazeSpec.empty(size=5, start=(1.0, 1.0), goal=(3.0, 1.0)))
    planner = ShootingPlanner(env, horizon=3, candidates=16, seed=0)

    result = planner.plan(env.reset())

    assert result.action.shape == (2,)
    assert result.plan.shape == (3, 2)
    assert result.nodes_expanded == 48
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/planning/test_shooting.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.planning.shooting'`.

- [ ] **Step 3: Implement shooting planner**

Create `wmsv/planning/shooting.py`:

```python
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
    def __init__(self, dynamics, horizon: int, candidates: int, seed: int = 0):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if candidates < 2:
            raise ValueError("candidates must be at least 2")
        self.dynamics = dynamics
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.seed = int(seed)

    def plan(self, state: np.ndarray) -> ShootingPlanResult:
        rng = np.random.default_rng(self.seed)
        plans = rng.normal(size=(self.candidates, self.horizon, 2))
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/planning/test_shooting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/planning/shooting.py tests/planning/test_shooting.py
git commit -m "feat: add random shooting planner"
```

## Task 9: Boxoban Pilot Row Builder

**Files:**
- Create: `wmsv/pilots/__init__.py`
- Create: `wmsv/pilots/boxoban.py`
- Test: `tests/pilots/test_boxoban_pilot.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pilots/test_boxoban_pilot.py`:

```python
from wmsv.pilots.boxoban import build_boxoban_pilot_rows


def test_build_boxoban_pilot_rows_returns_dive_labels_and_features():
    rows = build_boxoban_pilot_rows(limit=4, seed=0)

    assert len(rows) == 4
    row = rows[0]
    assert {"a_c", "a_v", "r_c", "r_v", "y_helpful", "delta_r"}.issubset(row)
    assert {"score_margin", "uncertainty_proxy", "cheap_nodes", "verifier_nodes"}.issubset(row)
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/pilots/test_boxoban_pilot.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.pilots'`.

- [ ] **Step 3: Implement Boxoban pilot rows**

Create `wmsv/pilots/__init__.py`:

```python
"""Phase 1 pilot experiment builders."""
```

Create `wmsv/pilots/boxoban.py`:

```python
from __future__ import annotations

from wmsv.analysis.stage_a import make_plan_label_row
from wmsv.analysis.verification_value import add_verification_labels
from wmsv.envs.sokoban import parse_level
from wmsv.models.sokoban_memory import MemorizedSokobanModel, collect_sokoban_transitions
from wmsv.planning.beam import BeamPlanner
from wmsv.planning.evaluators import PotentialEvaluator, TrueEvaluator


TINY_LEVELS = [
    ["#####", "#@$.#", "#   #", "#####"],
    ["######", "# @$ #", "#   .#", "######"],
    ["#######", "#@ $. #", "#     #", "#######"],
    ["#######", "# @ $ #", "#   . #", "#######"],
]


def build_boxoban_pilot_rows(limit: int = 20, seed: int = 0) -> list[dict]:
    levels = [(f"tiny-{idx}", parse_level(lines)) for idx, lines in enumerate(TINY_LEVELS)]
    action_sequences = _action_sequences(seed=seed, count=8, length=4)
    transitions = collect_sokoban_transitions(levels, action_sequences)
    cheap_model = MemorizedSokobanModel.fit(transitions[: max(1, len(transitions) // 2)])
    cheap = BeamPlanner(cheap_model, depth=2, width=4)
    verifier = BeamPlanner(TrueEvaluator(), depth=4, width=8)
    evaluator = PotentialEvaluator(TrueEvaluator(), scale=0.1)
    rows: list[dict] = []
    idx = 0
    while len(rows) < int(limit):
        level_id, state = levels[idx % len(levels)]
        row = make_plan_label_row(f"{level_id}:r{idx}", state, cheap, verifier, evaluator)
        row = add_verification_labels(row)
        row.update(
            {
                "ensemble_uncertainty": float(row["uncertainty_proxy"]),
                "cheap_plan_length": len(row["plan_c"]),
                "cheap_plan_score_per_step": float(row["cheap_score"]) / max(len(row["plan_c"]), 1),
            }
        )
        rows.append(row)
        idx += 1
    return rows


def _action_sequences(seed: int, count: int, length: int) -> list[list[int]]:
    value = int(seed)
    sequences: list[list[int]] = []
    for idx in range(count):
        sequences.append([(value + idx + step) % 4 for step in range(length)])
    return sequences
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/pilots/test_boxoban_pilot.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/pilots/__init__.py wmsv/pilots/boxoban.py tests/pilots/test_boxoban_pilot.py
git commit -m "feat: add Boxoban pilot row builder"
```

## Task 10: Maze Pilot Row Builder

**Files:**
- Create: `wmsv/pilots/maze.py`
- Test: `tests/pilots/test_maze_pilot.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pilots/test_maze_pilot.py`:

```python
from wmsv.pilots.maze import build_maze_pilot_rows


def test_build_maze_pilot_rows_returns_continuous_labels_and_features():
    rows = build_maze_pilot_rows(variants=2, episodes_per_variant=3, seed=0)

    assert len(rows) == 6
    row = rows[0]
    assert {"a_c", "a_v", "r_c", "r_v", "y_helpful", "delta_r"}.issubset(row)
    assert {"score_margin", "ensemble_uncertainty", "cheap_nodes", "verifier_nodes"}.issubset(row)
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/pilots/test_maze_pilot.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wmsv.pilots.maze'`.

- [ ] **Step 3: Implement Maze pilot rows**

Create `wmsv/pilots/maze.py`:

```python
from __future__ import annotations

import numpy as np

from wmsv.analysis.verification_value import add_verification_labels
from wmsv.data.transitions import Transition
from wmsv.envs.point_maze import PointMazeEnv, PointMazeSpec
from wmsv.models.linear_dynamics import LinearDynamicsModel
from wmsv.planning.shooting import ShootingPlanner


def build_maze_pilot_rows(variants: int = 3, episodes_per_variant: int = 20, seed: int = 0) -> list[dict]:
    rows: list[dict] = []
    for variant in range(int(variants)):
        env = _make_env(variant)
        transitions = _collect_maze_transitions(env, seed=seed + variant, count=80)
        learned = LinearDynamicsModel.fit(transitions)
        cheap = ShootingPlanner(learned, horizon=4, candidates=16, seed=seed + variant)
        verifier = ShootingPlanner(env, horizon=8, candidates=64, seed=seed + 100 + variant)
        for episode in range(int(episodes_per_variant)):
            state = env.reset()
            cheap_result = cheap.plan(state)
            verifier_result = verifier.plan(state)
            r_c = _execute_plan(env, state, cheap_result.plan)
            r_v = _execute_plan(env, state, verifier_result.plan)
            row = {
                "level_id": f"maze-{variant}:e{episode}",
                "a_c": cheap_result.action,
                "a_v": verifier_result.action,
                "r_c": r_c,
                "r_v": r_v,
                "cheap_score": cheap_result.score,
                "verifier_score": verifier_result.score,
                "score_margin": cheap_result.score_margin,
                "ensemble_uncertainty": abs(verifier_result.score - cheap_result.score),
                "uncertainty_proxy": abs(verifier_result.score - cheap_result.score),
                "cheap_nodes": cheap_result.nodes_expanded,
                "verifier_nodes": verifier_result.nodes_expanded,
            }
            rows.append(add_verification_labels(row, epsilon=0.01, action_delta=0.10))
    return rows


def _make_env(variant: int) -> PointMazeEnv:
    size = 7
    walls = {(3, y) for y in range(1, 6) if y != 1 + (variant % 5)}
    spec = PointMazeSpec(
        size=size,
        start=np.array([1.0, 1.0], dtype=np.float64),
        goal=np.array([5.0, 5.0], dtype=np.float64),
        walls=walls,
    )
    return PointMazeEnv(spec)


def _collect_maze_transitions(env: PointMazeEnv, seed: int, count: int) -> list[Transition]:
    rng = np.random.default_rng(int(seed))
    transitions: list[Transition] = []
    state = env.reset()
    for idx in range(int(count)):
        action = rng.normal(size=2)
        norm = np.linalg.norm(action)
        action = action / max(norm, 1.0)
        step = env.step(state, action)
        transitions.append(Transition(state, action, step.state, step.reward, step.done, {"idx": idx}))
        state = env.reset() if step.done else step.state
    return transitions


def _execute_plan(env: PointMazeEnv, state: np.ndarray, plan: np.ndarray) -> float:
    current = np.asarray(state, dtype=np.float64).copy()
    total = 0.0
    for action in plan:
        step = env.step(current, action)
        total += float(step.reward)
        current = step.state
        if step.done:
            break
    return total
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. pytest tests/pilots/test_maze_pilot.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wmsv/pilots/maze.py tests/pilots/test_maze_pilot.py
git commit -m "feat: add maze pilot row builder"
```

## Task 11: Pilot CLIs

**Files:**
- Create: `scripts/run_boxoban_learned_pilot.py`
- Create: `scripts/run_maze_learned_pilot.py`
- Test: `tests/scripts/test_phase1_pilot_scripts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scripts/test_phase1_pilot_scripts.py`:

```python
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import main as boxoban_main
from scripts.run_maze_learned_pilot import main as maze_main


def test_boxoban_pilot_script_writes_report(tmp_path):
    out = tmp_path / "boxoban.json"

    boxoban_main(["--limit", "8", "--out", str(out)])

    data = json.loads(out.read_text())
    assert data["environment"] == "boxoban"
    assert "go_no_go" in data
    assert data["num_rows"] == 8


def test_maze_pilot_script_writes_report(tmp_path):
    out = tmp_path / "maze.json"

    maze_main(["--variants", "2", "--episodes-per-variant", "3", "--out", str(out)])

    data = json.loads(out.read_text())
    assert data["environment"] == "maze"
    assert "go_no_go" in data
    assert data["num_rows"] == 6
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
PYTHONPATH=. pytest tests/scripts/test_phase1_pilot_scripts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_boxoban_learned_pilot'`.

- [ ] **Step 3: Implement shared report logic inside each script**

Create `scripts/run_boxoban_learned_pilot.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmsv.analysis.budgeting import mean_return_for_mask, threshold_budget_mask
from wmsv.analysis.verification_value import auroc, go_no_go_status, positive_label_rate
from wmsv.gating.dive import DIVEFeatureSchema, fit_dive_v0
from wmsv.pilots.boxoban import build_boxoban_pilot_rows


FEATURES = ["score_margin", "ensemble_uncertainty", "cheap_score", "cheap_plan_score_per_step"]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = build_boxoban_pilot_rows(limit=args.limit)
    report = _report(rows, environment="boxoban", min_always_gain=0.12, min_auroc=0.62)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _report(rows: list[dict], environment: str, min_always_gain: float, min_auroc: float) -> dict:
    split = max(2, len(rows) // 2)
    train = rows[:split]
    eval_rows = rows[split:] or rows
    schema = DIVEFeatureSchema(FEATURES)
    gate = fit_dive_v0(train, schema)
    scores = [gate.score(row) for row in eval_rows]
    mask20 = threshold_budget_mask(scores, 0.20)
    random_mask20 = [(idx % 5) == 0 for idx in range(len(eval_rows))]
    cheap_return = sum(float(row["r_c"]) for row in eval_rows) / len(eval_rows)
    always_return = sum(float(row["r_v"]) for row in eval_rows) / len(eval_rows)
    dive_return = mean_return_for_mask(eval_rows, mask20)
    random_return = mean_return_for_mask(eval_rows, random_mask20)
    helpful_auroc = auroc(scores, [int(row["y_helpful"]) for row in eval_rows])
    status = go_no_go_status(
        cheap_success=cheap_return,
        always_verify_success=always_return,
        positive_label_rate=positive_label_rate(eval_rows),
        helpful_auroc=helpful_auroc,
        budget20_gain=dive_return - random_return,
        min_always_gain=min_always_gain,
        min_auroc=min_auroc,
    )
    return {
        "environment": environment,
        "num_rows": len(rows),
        "cheap_return": cheap_return,
        "always_verify_return": always_return,
        "dive_budget20_return": dive_return,
        "random_budget20_return": random_return,
        "positive_label_rate": positive_label_rate(eval_rows),
        "helpful_auroc": helpful_auroc,
        "go_no_go": status,
    }


if __name__ == "__main__":
    main()
```

Create `scripts/run_maze_learned_pilot.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_boxoban_learned_pilot import _report
from wmsv.pilots.maze import build_maze_pilot_rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--episodes-per-variant", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = build_maze_pilot_rows(
        variants=args.variants,
        episodes_per_variant=args.episodes_per_variant,
    )
    report = _report(rows, environment="maze", min_always_gain=0.10, min_auroc=0.60)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script tests**

Run:

```bash
PYTHONPATH=. pytest tests/scripts/test_phase1_pilot_scripts.py -q
```

Expected: PASS.

- [ ] **Step 5: Run both pilot scripts manually**

Run:

```bash
PYTHONPATH=. python scripts/run_boxoban_learned_pilot.py \
  --limit 20 \
  --out outputs/phase1_dual_pilot/boxoban_pilot_smoke.json

PYTHONPATH=. python scripts/run_maze_learned_pilot.py \
  --variants 2 \
  --episodes-per-variant 5 \
  --out outputs/phase1_dual_pilot/maze_pilot_smoke.json
```

Expected: both commands exit 0 and write JSON reports.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/run_boxoban_learned_pilot.py scripts/run_maze_learned_pilot.py tests/scripts/test_phase1_pilot_scripts.py
git commit -m "feat: add phase1 pilot scripts"
```

## Task 12: Experiment README

**Files:**
- Create: `experiments/phase1_dual_pilot/README.md`

- [ ] **Step 1: Write experiment README**

Create `experiments/phase1_dual_pilot/README.md`:

````markdown
# Phase 1 Dual Pilot

Status: planned implementation stage.

Purpose:

```text
Validate DIVE-v0 and the shared selective-verification protocol before
scaling to full dual-main experiments.
```

Pilot commands:

```bash
PYTHONPATH=. python scripts/run_boxoban_learned_pilot.py \
  --limit 200 \
  --out outputs/phase1_dual_pilot/boxoban_pilot.json

PYTHONPATH=. python scripts/run_maze_learned_pilot.py \
  --variants 3 \
  --episodes-per-variant 100 \
  --out outputs/phase1_dual_pilot/maze_pilot.json
```

Go/no-go thresholds:

```text
cheap-only return/success:
    > 15% and < 85%

always-verify improvement:
    Boxoban >= +12 percentage points
    Maze >= +10 percentage points

positive label rate:
    15% to 50%

DIVE-v0 helpful-label AUROC:
    Boxoban >= 0.62
    Maze >= 0.60

20% budget DIVE vs random:
    DIVE >= random + 3 percentage points
```

Fallback rule:

```text
If PointMaze/Maze2D has no verification gap, run one AntMaze/OGBench-style
maze pilot. If that also fails, pause continuous-environment scaling and
redesign the second co-main environment.
```
````

- [ ] **Step 2: Commit**

Run:

```bash
git add experiments/phase1_dual_pilot/README.md
git commit -m "docs: add phase1 dual pilot experiment log"
```

## Task 13: Final Phase 1 Verification

**Files:**
- Modify only if a previous task failed a test and required a fix.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/analysis/test_verification_value.py \
  tests/analysis/test_budgeting.py \
  tests/gating/test_dive.py \
  tests/data/test_transitions.py \
  tests/models/test_sokoban_memory.py \
  tests/models/test_linear_dynamics.py \
  tests/envs/test_point_maze.py \
  tests/planning/test_shooting.py \
  tests/pilots/test_boxoban_pilot.py \
  tests/pilots/test_maze_pilot.py \
  tests/scripts/test_phase1_pilot_scripts.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
PYTHONPATH=. pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run pilot smoke commands**

Run:

```bash
PYTHONPATH=. python scripts/run_boxoban_learned_pilot.py \
  --limit 20 \
  --out outputs/phase1_dual_pilot/boxoban_pilot_smoke.json

PYTHONPATH=. python scripts/run_maze_learned_pilot.py \
  --variants 2 \
  --episodes-per-variant 5 \
  --out outputs/phase1_dual_pilot/maze_pilot_smoke.json
```

Expected: both JSON files exist and include `go_no_go`.

- [ ] **Step 4: Inspect generated reports**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path("outputs/phase1_dual_pilot/boxoban_pilot_smoke.json"),
    Path("outputs/phase1_dual_pilot/maze_pilot_smoke.json"),
]:
    data = json.loads(path.read_text())
    print(path)
    print("  cheap_return:", data["cheap_return"])
    print("  always_verify_return:", data["always_verify_return"])
    print("  helpful_auroc:", data["helpful_auroc"])
    print("  go_no_go:", data["go_no_go"])
PY
```

Expected: command prints both reports without traceback.

- [ ] **Step 5: Commit any final fixes**

If previous steps required fixes, run:

```bash
git status --short
git add wmsv scripts tests experiments
git commit -m "fix: stabilize phase1 dual pilot"
```

If there are no changes, do not create an empty commit.

## Self-Review Checklist

Before handing implementation back:

```text
Spec coverage:
    DIVE-v0 implemented.
    Boxoban pilot implemented.
    Maze pilot implemented.
    Verification labels implemented.
    Uncertainty-label gate baseline implemented.
    Compute accounting implemented.

No implementation dependency on PyTorch or Gymnasium:
    Phase 1 remains numpy-only.

No Stage A regression:
    Existing tests pass.

No committed outputs:
    JSON files under outputs/ remain untracked or ignored.
```
