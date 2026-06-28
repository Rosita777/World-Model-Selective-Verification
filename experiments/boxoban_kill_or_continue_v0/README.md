# Boxoban Kill-Or-Continue v0

Status: Stage A smoke implemented.

Date: 2026-06-28

## Purpose

Test whether the selective-verification pipeline has a non-degenerate signal before training learned world models.

Stage A deliberately uses:

```text
cheap planner:
    synthetically degraded Sokoban evaluator + shallow beam search

verifier:
    true Sokoban evaluator + deeper bounded beam search

gate:
    simple supervised centroid score over cheap-planner features

baseline:
    random, uncertainty ranking, decision-aware gate, oracle upper bound
```

This is not a paper result. It is a kill-or-continue engineering check.

## Data

External data is not committed.

For the first Boxoban smoke, one raw level file was downloaded locally:

```text
data/external/boxoban-sample/medium/train/000.txt
```

Source:

```text
https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master/medium/train/000.txt
```

## Command

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 20 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_boxoban20_dense.json \
  --budget 0.25 \
  --penalty 1.0 \
  --cheap-depth 3 \
  --cheap-width 8 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16
```

## Result Summary

| push error | helpful | wasted | spurious | cheap return | always verify | uncertainty | decision-aware |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.00 | 0.00 | 0.200 | 0.200 | 0.200 | n/a |
| 0.25 | 0.10 | 0.80 | 0.10 | 0.190 | 0.200 | 0.190 | 0.200 |
| 0.50 | 0.10 | 0.75 | 0.15 | 0.190 | 0.200 | 0.190 | 0.200 |
| 0.75 | 0.20 | 0.55 | 0.25 | 0.175 | 0.200 | 0.175 | 0.185 |
| 1.00 | 0.25 | 0.40 | 0.35 | 0.170 | 0.200 | 0.185 | 0.185 |

## Interpretation

The pipeline now reaches a non-degenerate regime:

```text
push_error_rate = 0.75 or 1.0
helpful correction rate = 20-25%
wasted verification rate = 40-55%
```

This is enough to continue Stage A implementation.

However, the current decision-aware gate is weak:

```text
At 0.75, decision-aware improves over cheap and uncertainty, but not close
to always verify.

At 1.0, decision-aware ties uncertainty.
```

The next research step is not scaling. It is improving the Stage A gate features and baseline fairness:

```text
1. add features for action-change likelihood and cheap planner instability;
2. split train/test levels instead of fitting and evaluating on the same rows;
3. add a stronger uncertainty proxy based on multiple degradation seeds;
4. then re-run on 100-500 Boxoban levels.
```

## Current Judgment

Continue implementation, but do not claim success yet.

This smoke establishes:

```text
the code path works;
the feasibility base rates can be tuned into a useful range;
the first decision-aware gate is not yet strong enough.
```

## 2026-06-28 Update: 500-Level Split With Stronger Baselines

After the first smoke, the script was upgraded with:

```text
train/eval split:
    fit the decision gate on train rows, report returns on eval rows

ensemble uncertainty:
    rank verification calls by action disagreement + score variance across
    5 degraded cheap planners

random baseline:
    fixed-seed random selection under the same verification budget

oracle baseline:
    upper bound that ranks eval rows by true r_v - r_c
```

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 500 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_boxoban500_weakcheap.json \
  --budget 0.25 \
  --penalty 1.0 \
  --cheap-depth 1 \
  --cheap-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --train-fraction 0.5 \
  --random-seed 0
```

Result summary:

| push error | helpful | harmful | wasted | cheap | random | uncertainty | decision-aware | oracle | always verify |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.060 | 0.002 | 0.760 | 0.199 | 0.200 | 0.200 | 0.202 | 0.210 | 0.210 |
| 0.25 | 0.108 | 0.004 | 0.648 | 0.194 | 0.195 | 0.198 | 0.199 | 0.210 | 0.210 |
| 0.50 | 0.158 | 0.004 | 0.546 | 0.186 | 0.191 | 0.192 | 0.198 | 0.210 | 0.210 |
| 0.75 | 0.200 | 0.004 | 0.462 | 0.176 | 0.182 | 0.190 | 0.193 | 0.210 | 0.210 |
| 1.00 | 0.242 | 0.012 | 0.338 | 0.169 | 0.180 | 0.176 | 0.177 | 0.212 | 0.210 |

Helpful precision among selected verification calls:

| push error | random | uncertainty | decision-aware | oracle |
|---:|---:|---:|---:|---:|
| 0.00 | 0.03 | 0.02 | 0.08 | 0.21 |
| 0.25 | 0.03 | 0.10 | 0.13 | 0.34 |
| 0.50 | 0.13 | 0.16 | 0.29 | 0.52 |
| 0.75 | 0.16 | 0.31 | 0.37 | 0.71 |
| 1.00 | 0.24 | 0.16 | 0.18 | 0.87 |

Interpretation:

```text
The cleanest regime is push_error_rate = 0.50 or 0.75.

In those regimes, decision-aware selection beats random and ensemble
uncertainty under the same 25% verification budget.

The oracle upper bound remains much higher, so the current gate is not
near the ceiling.

At push_error_rate = 1.00 the current centroid gate degrades and should
not be used as the main claim.
```

External-review critique from Opus:

```text
The most dangerous weakness is not the gate architecture. It is whether
the experiment creates enough cheap-vs-verifier decision gap and enough
eval examples for the comparison to be statistically meaningful.

Recommended next move: keep Sokoban/Boxoban, but run larger samples,
report oracle upper bound, and focus the main analysis on Goldilocks
regimes where helpful verification is neither too rare nor too trivial.
```

Current judgment:

```text
Continue. The 500-level split gives a real positive signal against
random and uncertainty baselines at intermediate degradation levels, but
the current gate is still a first diagnostic controller, not a final
paper-quality method.
```
