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

## 2026-06-28 Update: 1000-Level Budget Sweep

The runner was extended with:

```text
think-longer baseline:
    spend more cheap-planner compute everywhere instead of selectively
    calling the verifier

budget sweep:
    report each selection policy at multiple verification budgets

compute accounting:
    report average search nodes per eval state for each policy
```

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 1000 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_boxoban1000_budget_sweep.json \
  --budget 0.25 \
  --budgets 0.05,0.10,0.25,0.50,1.00 \
  --penalty 1.0 \
  --cheap-depth 1 \
  --cheap-width 4 \
  --think-longer-depth 3 \
  --think-longer-width 16 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --train-fraction 0.5 \
  --random-seed 0
```

Base rates:

| push error | helpful | harmful | wasted |
|---:|---:|---:|---:|
| 0.00 | 0.070 | 0.006 | 0.761 |
| 0.25 | 0.112 | 0.008 | 0.665 |
| 0.50 | 0.156 | 0.010 | 0.563 |
| 0.75 | 0.196 | 0.010 | 0.473 |
| 1.00 | 0.248 | 0.012 | 0.347 |

Budget sweep at `push_error_rate = 0.50`:

| budget | cheap | think-longer | random | uncertainty | decision-aware | oracle | always verify |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.195 | 0.199 | 0.197 | 0.197 | 0.198 | 0.208 | 0.220 |
| 0.10 | 0.195 | 0.199 | 0.198 | 0.199 | 0.204 | 0.217 | 0.220 |
| 0.25 | 0.195 | 0.199 | 0.203 | 0.207 | 0.211 | 0.223 | 0.220 |
| 0.50 | 0.195 | 0.199 | 0.210 | 0.212 | 0.212 | 0.223 | 0.220 |
| 1.00 | 0.195 | 0.199 | 0.220 | 0.220 | 0.220 | 0.220 | 0.220 |

Average search nodes per eval state at `push_error_rate = 0.50`:

| budget | cheap | think-longer | uncertainty | decision-aware | always verify |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 4 | 84 | 18 | 18 | 280 |
| 0.10 | 4 | 84 | 32 | 32 | 280 |
| 0.25 | 4 | 84 | 73 | 73 | 280 |
| 0.50 | 4 | 84 | 142 | 142 | 280 |
| 1.00 | 4 | 84 | 280 | 280 | 280 |

Budget sweep at `push_error_rate = 0.75`:

| budget | cheap | think-longer | random | uncertainty | decision-aware | oracle | always verify |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.186 | 0.188 | 0.189 | 0.189 | 0.190 | 0.201 | 0.220 |
| 0.10 | 0.186 | 0.188 | 0.192 | 0.191 | 0.194 | 0.212 | 0.220 |
| 0.25 | 0.186 | 0.188 | 0.198 | 0.200 | 0.204 | 0.223 | 0.220 |
| 0.50 | 0.186 | 0.188 | 0.207 | 0.206 | 0.209 | 0.223 | 0.220 |
| 1.00 | 0.186 | 0.188 | 0.220 | 0.220 | 0.220 | 0.220 | 0.220 |

Interpretation:

```text
The main positive signal survives at 1000 levels.

At push_error_rate = 0.50 and 0.75, decision-aware selection improves
over random, ensemble uncertainty, and think-longer at the same reported
budget points.

The think-longer baseline is weak in this synthetic degraded-planner
setting: spending more compute inside the same corrupted evaluator does
not repair the decision as effectively as selectively calling the true
verifier.

The oracle policy can slightly exceed always verify at partial budgets.
This is expected because always verify can include harmful flips, while
oracle selective verification avoids them.
```

Current judgment:

```text
Stage A now has a paper-shaped controlled result: budget sweep,
think-longer baseline, random baseline, uncertainty baseline, oracle
upper bound, and compute accounting.

The next required addition is uncertainty over the reported numbers:
bootstrap confidence intervals or multiple random seeds.
```

## 2026-06-28 Update: Bootstrap Confidence Intervals

Bootstrap CI command:

```bash
PYTHONPATH=. python scripts/summarize_stage_a_ci.py \
  outputs/boxoban_kill_or_continue_v0/stage_a_boxoban1000_budget_sweep.json \
  --rates 0.50,0.75 \
  --budgets 0.25 \
  --samples 1000 \
  --seed 0 \
  --random-seed 0 \
  > outputs/boxoban_kill_or_continue_v0/stage_a_boxoban1000_budget025_ci.json
```

Mean return and 95% bootstrap CI at `budget = 0.25`:

| push error | policy | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | cheap | 0.1946 | [0.1805, 0.2095] |
| 0.50 | think-longer | 0.1985 | [0.1843, 0.2135] |
| 0.50 | random | 0.2032 | [0.1882, 0.2180] |
| 0.50 | uncertainty | 0.2073 | [0.1927, 0.2214] |
| 0.50 | decision-aware | 0.2113 | [0.1966, 0.2261] |
| 0.50 | oracle | 0.2229 | [0.2077, 0.2377] |
| 0.50 | always verify | 0.2203 | [0.2053, 0.2349] |
| 0.75 | cheap | 0.1864 | [0.1728, 0.2002] |
| 0.75 | think-longer | 0.1884 | [0.1753, 0.2022] |
| 0.75 | random | 0.1979 | [0.1837, 0.2125] |
| 0.75 | uncertainty | 0.1998 | [0.1854, 0.2144] |
| 0.75 | decision-aware | 0.2039 | [0.1892, 0.2183] |
| 0.75 | oracle | 0.2229 | [0.2077, 0.2377] |
| 0.75 | always verify | 0.2203 | [0.2053, 0.2349] |

Paired bootstrap deltas for decision-aware at `budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - random | 0.0081 | [0.0027, 0.0135] |
| 0.50 | decision - uncertainty | 0.0040 | [0.0016, 0.0070] |
| 0.50 | decision - think-longer | 0.0128 | [0.0064, 0.0194] |
| 0.50 | decision - cheap | 0.0167 | [0.0108, 0.0227] |
| 0.75 | decision - random | 0.0060 | [0.0001, 0.0125] |
| 0.75 | decision - uncertainty | 0.0041 | [0.0017, 0.0068] |
| 0.75 | decision - think-longer | 0.0155 | [0.0097, 0.0216] |
| 0.75 | decision - cheap | 0.0175 | [0.0121, 0.0235] |

Interpretation:

```text
The per-policy confidence intervals overlap, which is expected because
all policies are evaluated on the same levels and the absolute returns
are noisy.

The paired bootstrap deltas are the more relevant comparison. At
budget = 0.25, decision-aware improves over random, uncertainty, and
think-longer with positive paired CI in the two main Goldilocks regimes.

The uncertainty comparison is statistically positive here, but the mean
gap is small. It should be framed carefully as a controlled Stage A
finding, not yet as a broad claim about all uncertainty-aware planning.
```

Opus critique after this result:

```text
The direction is promising, but the result is still Stage A only.

The next likely reviewer concern is fairness of think-longer. Our
current think-longer baseline spends more compute inside the same
degraded cheap evaluator, which matches the original "think longer with
the cheap model" baseline. A stronger future addition is an iso-compute
uniform true-simulator baseline, which would test whether uniformly
spending a smaller amount of verifier-quality compute can match
selective deep verification.
```
