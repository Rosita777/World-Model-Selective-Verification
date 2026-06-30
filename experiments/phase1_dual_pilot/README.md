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
