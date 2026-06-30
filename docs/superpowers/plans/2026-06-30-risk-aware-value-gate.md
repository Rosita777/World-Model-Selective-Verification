# Risk-Aware Value Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a final gate candidate that ranks verification by predicted value while penalizing harmful flips.

**Architecture:** Reuse the existing DIVE-style harm head and direct value-rank regressor. Train one gate with shared features, score rows with `max(predicted_delta, 0) * (1 - p_harm) - alpha * p_harm`, and report it beside DIVE, value-rank, uncertainty, random, and oracle in the existing budget curves.

**Tech Stack:** Python, NumPy ridge/logistic heads, pytest, existing WMSV budget-curve scripts.

---

### Task 1: Gate API

**Files:**
- Modify: `wmsv/gating/dive.py`
- Modify: `wmsv/gating/__init__.py`
- Test: `tests/gating/test_dive.py`

- [ ] Write a failing test importing `fit_risk_aware_value_gate` and checking that a high-value safe row scores above a high-value harmful row.
- [ ] Run `pytest tests/gating/test_dive.py::test_risk_aware_value_gate_penalizes_harmful_flips -q` and confirm import failure.
- [ ] Implement `RiskAwareValueGate` with change/harm binary heads and a delta regression head.
- [ ] Run `pytest tests/gating/test_dive.py -q`.

### Task 2: Budget Curves

**Files:**
- Modify: `wmsv/analysis/budget_curves.py`
- Modify: `scripts/run_phase1_budget_curves.py`
- Modify: `scripts/run_boxoban_gate_ablation.py`
- Test: `tests/analysis/test_budget_curves.py`
- Test: `tests/scripts/test_phase1_budget_curves.py`
- Test: `tests/scripts/test_boxoban_gate_ablation.py`

- [ ] Write failing tests requiring `risk_aware_value_return`, `risk_aware_value_nodes`, and `risk_aware_value_budget20_return`.
- [ ] Run focused tests and confirm failure.
- [ ] Fit the new gate in `evaluate_budget_curve` and include it in masks and aggregate metrics.
- [ ] Run focused tests and then full `pytest -q`.

### Task 3: Experiment Check

**Files:**
- Write output JSON under `outputs/phase1_dual_pilot/`.

- [ ] Run Boxoban gate ablation for uncertainty, impact, and all feature sets.
- [ ] Run 3-seed dual-environment budget curves.
- [ ] Extract return and selected-delta summaries for DIVE, value-rank, risk-aware value, and uncertainty.
- [ ] Commit code and tests after the tests pass.
