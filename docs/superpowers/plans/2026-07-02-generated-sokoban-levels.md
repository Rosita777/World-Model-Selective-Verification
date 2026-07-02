# Generated Sokoban Levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic generated Sokoban level source so Boxoban-style experiments can scale beyond the current one-file sample.

**Architecture:** Implement a small generator in `wmsv/data/generated_sokoban.py` that writes Boxoban-compatible `; id` level blocks. Integrate it into `build_boxoban_pilot_rows` as an explicit fallback/source mode when the external Boxoban folder has fewer unique levels than needed.

**Tech Stack:** Python standard library, existing `parse_level`, existing Boxoban text parser, pytest.

---

### Task 1: Generated Level Library

**Files:**
- Create: `wmsv/data/generated_sokoban.py`
- Test: `tests/data/test_generated_sokoban.py`

- [ ] Write tests that generated levels are deterministic, parseable, unique, and have a simple known solution.
- [ ] Run `pytest tests/data/test_generated_sokoban.py -q` and confirm import failure.
- [ ] Implement `generate_sokoban_levels` and `write_boxoban_levels`.
- [ ] Run `pytest tests/data/test_generated_sokoban.py -q`.

### Task 2: Pilot Integration

**Files:**
- Modify: `wmsv/pilots/boxoban.py`
- Test: `tests/pilots/test_boxoban_pilot.py`

- [ ] Add a test that `build_boxoban_pilot_rows` can request generated levels when the data folder is too small.
- [ ] Run the focused pilot tests and confirm failure.
- [ ] Add `generated_level_count` and `generated_seed` parameters, and make `_load_levels` augment the external folder with generated levels when needed.
- [ ] Run focused tests and then full `pytest -q`.

### Task 3: Smoke Run

**Files:**
- Write local outputs under `outputs/phase1_dual_pilot/`.

- [ ] Run a 1000-row single-seed Boxoban smoke if runtime is acceptable.
- [ ] Generate a markdown summary with `scripts/summarize_main_curves.py` for any multi-seed result produced.
- [ ] Commit code, tests, and plan. Do not commit generated output JSON.
