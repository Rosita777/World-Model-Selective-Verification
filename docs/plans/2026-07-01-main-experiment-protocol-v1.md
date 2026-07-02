# Main Experiment Protocol v1

Date: 2026-07-01

Status: executable protocol for the next experimental stage.

## Decision

Stop tuning new gate formulas for now. The main method family is:

```text
DIVE:
    conservative decision-aware gate, strong at low verification budgets

ValueRank:
    direct verification-value ranker, strong at mid/high budgets
```

`risk_aware_value` stays as an ablation. It did not improve over ValueRank on
the current dual-environment pilot, so it should not be presented as the final
method.

## Core Story

World models should not only imagine futures; they should learn when an
imagined future is worth verifying.

The experiments test whether decision-aware selective verification improves the
compute-return frontier compared with:

```text
cheap-only planning
always verifying
random verification under the same budget
uncertainty-triggered verification
uniformly thinking longer with the cheap model
uniformly using a fixed true-model budget
```

The key comparison is not "does verification help?" The key comparison is:

```text
At the same verification budget or rollout/search-node budget, does a learned
decision-aware gate spend verification calls on states where verification
changes and improves the deployed decision?
```

## Main Environments

### Environment A: Boxoban / Sokoban

Role: discrete combinatorial planning with irreversible actions and deadlocks.

Current implementation:

```text
cheap planner:
    BeamPlanner(depth=3, width=8) over a degraded push evaluator

expensive verifier:
    BeamPlanner(depth=5, width=16) over the true evaluator

think-longer baseline:
    BeamPlanner(depth=5, width=16) over the degraded evaluator

uniform-true baseline:
    BeamPlanner(depth=4, width=8) over the true evaluator
```

Current limitation:

```text
data/external/boxoban-sample/medium/train currently contains one sample file
with 1000 levels. This is enough for 1000-row smoke runs, but 3000-row
final-scale runs require either a larger external Boxoban subset or generated
Sokoban augmentation.
```

Main-experiment target:

```text
rows:
    1000 smoke-scale rows for quick iteration
    3000 final-scale rows if runtime is acceptable

splits:
    level-group split preferred over row-index split
    no states from the same Boxoban level should appear in both train and eval

seeds:
    5 seeds for main tables
    10 seeds only if variance remains high or final runtime is low
```

Required improvement before paper tables:

```text
For 1000-row smoke experiments, use the existing sample file.
For 3000-row final-scale experiments, augment it with deterministic generated
Sokoban levels unless a larger external Boxoban subset is added.
```

### Environment B: PointMaze / Maze

Role: continuous-control world-model planning with learned linear dynamics and
MPC-style shooting.

Current implementation:

```text
cheap model:
    LinearDynamicsModel fit from random transitions

cheap planner:
    ShootingPlanner(horizon=4, candidates=16)

expensive verifier:
    true PointMazeEnv with ShootingPlanner(horizon=8, candidates=64)

think-longer baseline:
    learned model with horizon=8, candidates=64

uniform-true baseline:
    true PointMazeEnv with horizon=4, candidates=16
```

Main-experiment target:

```text
variants:
    5 to 10 maze layouts

episodes per variant:
    200 smoke-scale
    500 final-scale if runtime is acceptable

seeds:
    5 seeds for main tables
```

Escalation rule:

```text
If reviewers would see this as too synthetic, keep PointMaze as a controlled
main environment and add one extension experiment on D4RL/OGBench-style Maze2D
or AntMaze. Do this only after Boxoban and PointMaze main curves are stable.
```

## Gate Policies To Report

Main methods:

```text
DIVE
ValueRank
```

Baselines:

```text
Cheap only
Always verify
Random gate, same verification-call budget
Uncertainty gate, same verification-call budget
Think longer
Uniform true budget
Oracle gate, upper bound only
```

Ablations:

```text
risk_aware_value:
    report in appendix or ablation table; do not call it the final method

Boxoban feature sets:
    score
    uncertainty
    plan
    trajectory
    impact
    all
```

## Budgets

Use the same budget grid for both environments:

```text
0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00
```

Primary operating points:

```text
low budget:
    0.10

medium budget:
    0.20

high budget:
    0.50
```

Do not select a single favorable budget as the main result. Report curves and
area-under-budget-curve style summaries.

## Metrics

Primary metrics:

```text
return or success rate
average verifier calls
average total search nodes / rollout candidates
compute-return Pareto curve
```

Verification-quality metrics:

```text
helpful correction count and precision
harmful flip count
wasted verification count
mean selected delta_R
total selected delta_R
helpful-label AUROC
```

Reporting rule:

```text
For each environment, show mean +/- standard deviation over seeds.
For each budget, compare methods at the same verification-call fraction.
For compute-matched claims, also compare total nodes against Think Longer and
Uniform True.
```

## Current Pilot Readout

Current 3-seed pilot from
`outputs/phase1_dual_pilot/budget_curves_3seed_risk_aware_value.json`:

```text
Boxoban:
    10% budget: DIVE 0.158, ValueRank 0.153, Uncertainty 0.148
    20% budget: DIVE 0.198, ValueRank 0.196, Uncertainty 0.190
    50% budget: DIVE 0.247, ValueRank 0.279, Uncertainty 0.276

Maze:
    10% budget: DIVE -8.476, ValueRank -8.474, Uncertainty -8.715
    20% budget: DIVE -8.315, ValueRank -8.308, Uncertainty -8.782
    50% budget: DIVE -7.964, ValueRank -7.989, Uncertainty -8.802
```

Interpretation:

```text
DIVE is the conservative low-budget gate.
ValueRank is the direct value-ranking gate and helps more at mid/high budgets,
especially in Boxoban.
Uncertainty is a strong but conceptually different baseline; it must remain in
the main tables.
Risk-aware-value did not add useful signal beyond ValueRank.
```

## Next Execution Steps

### Step 1: freeze the reporting script

Create a script that reads budget-curve JSON and produces a compact markdown
summary with:

```text
main budget table
selected-delta table
helpful/harmful/waste table
compute-node table
```

This prevents manual copying mistakes while experiments scale.

### Step 2: improve Boxoban data scale

Choose one:

```text
Option A:
    add a larger Boxoban subset under data/external, kept out of git if large

Option B:
    implement deterministic Sokoban level generation and document it
```

Preferred: Option A if data download is easy; Option B if network/data access is
annoying.

Current implementation status:

```text
wmsv.data.generated_sokoban provides deterministic generated Sokoban levels.
build_boxoban_pilot_rows augments a too-small external folder with generated
levels. The 1000-level local sample is used first; generated levels fill any
remaining requested level count.
```

### Step 3: run smoke-scale main curves

Commands:

```bash
PYTHONPATH=. python scripts/run_phase1_budget_curves.py \
  --boxoban-limit 1000 \
  --maze-variants 5 \
  --maze-episodes-per-variant 200 \
  --budgets 0,0.05,0.1,0.2,0.3,0.5,0.75,1 \
  --seeds 0,1,2,3,4 \
  --out outputs/phase1_dual_pilot/main_curves_smoke_5seed.json
```

### Step 4: diagnose failure modes

Run Boxoban feature-set ablation at 1000 rows:

```bash
PYTHONPATH=. python scripts/run_boxoban_gate_ablation.py \
  --limit 1000 \
  --feature-sets score,uncertainty,plan,trajectory,impact,all \
  --budgets 0,0.05,0.1,0.2,0.3,0.5,0.75,1 \
  --seed 0 \
  --out outputs/phase1_dual_pilot/boxoban_feature_ablation_1000_seed0.json
```

### Step 5: decide final scale

Proceed to final-scale runs only if smoke-scale results satisfy:

```text
Boxoban:
    DIVE or ValueRank beats Random at 10%, 20%, and 50% budgets.
    At least one of DIVE/ValueRank beats Uncertainty at two of three primary
    budgets, or matches it while using clearer helpful/harmful diagnostics.

Maze:
    DIVE or ValueRank beats Random and Uncertainty at 10%, 20%, and 50%.

Both:
    Think Longer does not dominate the selective-verification curve under
    comparable compute.
```

If these conditions fail, do not tune gate formulas first. Diagnose whether the
cheap/verifier gap, data split, or feature set is the limiting factor.
