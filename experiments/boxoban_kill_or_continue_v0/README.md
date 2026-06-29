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

## 2026-06-28 Update: Uniform True-Simulator Baseline

To address the strongest fairness critique, the runner now includes:

```text
uniform_true baseline:
    every eval state gets a shallow beam search with the true simulator
    instead of selecting a subset of states for a deep verifier call

main setting:
    uniform_true depth = 3, width = 12
    average nodes per eval state = 68

comparison point:
    decision-aware selective verification at budget = 0.25
    average nodes per eval state = 73
```

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 1000 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_boxoban1000_uniform_true_w12.json \
  --budget 0.25 \
  --budgets 0.05,0.10,0.25,0.50,1.00 \
  --penalty 1.0 \
  --cheap-depth 1 \
  --cheap-width 4 \
  --think-longer-depth 3 \
  --think-longer-width 16 \
  --uniform-true-depth 3 \
  --uniform-true-width 12 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --train-fraction 0.5 \
  --random-seed 0
```

Budget sweep at `push_error_rate = 0.50`:

| budget | cheap | think-longer | uniform true | uncertainty | decision-aware | oracle | always verify |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.195 | 0.199 | 0.213 | 0.197 | 0.198 | 0.208 | 0.220 |
| 0.10 | 0.195 | 0.199 | 0.213 | 0.199 | 0.204 | 0.217 | 0.220 |
| 0.25 | 0.195 | 0.199 | 0.213 | 0.207 | 0.211 | 0.223 | 0.220 |
| 0.50 | 0.195 | 0.199 | 0.213 | 0.212 | 0.212 | 0.223 | 0.220 |
| 1.00 | 0.195 | 0.199 | 0.213 | 0.220 | 0.220 | 0.220 | 0.220 |

Budget sweep at `push_error_rate = 0.75`:

| budget | cheap | think-longer | uniform true | uncertainty | decision-aware | oracle | always verify |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.186 | 0.188 | 0.213 | 0.189 | 0.190 | 0.201 | 0.220 |
| 0.10 | 0.186 | 0.188 | 0.213 | 0.191 | 0.194 | 0.212 | 0.220 |
| 0.25 | 0.186 | 0.188 | 0.213 | 0.200 | 0.204 | 0.223 | 0.220 |
| 0.50 | 0.186 | 0.188 | 0.213 | 0.206 | 0.209 | 0.223 | 0.220 |
| 1.00 | 0.186 | 0.188 | 0.213 | 0.220 | 0.220 | 0.220 | 0.220 |

Paired bootstrap CI at `budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uniform true | -0.0017 | [-0.0062, 0.0026] |
| 0.50 | decision - uncertainty | 0.0040 | [0.0016, 0.0070] |
| 0.50 | decision - think-longer | 0.0128 | [0.0064, 0.0194] |
| 0.75 | decision - uniform true | -0.0091 | [-0.0151, -0.0035] |
| 0.75 | decision - uncertainty | 0.0041 | [0.0017, 0.0068] |
| 0.75 | decision - think-longer | 0.0155 | [0.0097, 0.0216] |

Interpretation:

```text
This is a negative but important result.

Decision-aware selection remains better than random, uncertainty, and
cheap-model think-longer at budget = 0.25.

However, a uniform shallow true-simulator planner is stronger than the
current decision-aware selective verifier at similar compute. At
push_error_rate = 0.50 they are statistically tied; at push_error_rate =
0.75 uniform true is significantly better.

Therefore the current Stage A result should not claim that selective
verification beats all uniform compute allocation. It supports a narrower
claim: when the alternative is spending more compute inside the degraded
cheap model, or triggering verification by uncertainty, decision-aware
selection is better. If verifier-quality simulation is cheap enough to
apply uniformly, uniform true search is a very strong baseline.
```

Current judgment:

```text
Continue cautiously, but the next experiment must directly compare
strictly compute-matched true-simulator allocation curves.

The strongest current research question becomes:

    When is selective deep verification better than uniform shallow
    verifier-quality search?

The current Boxoban Stage A setting does not yet answer this in favor of
decision-aware selection.
```

## 2026-06-29 Update: Low-Budget Uniform-True Diagnostic

The runner now supports targeted sweeps:

```text
--rates:
    run only selected degradation rates, e.g. 0.50 and 0.75

--states-per-level:
    optionally evaluate multiple planning states sampled from each level
    instead of only the initial state
```

Initial-state low-budget diagnostic, 100 Boxoban levels:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 100 \
  --rates 0.50,0.75 \
  --budgets 0.01,0.02,0.05,0.10,0.25 \
  --cheap-depth 1 \
  --cheap-width 4 \
  --think-longer-depth 3 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_compute_curve_100_maincheap_u_d1w4.json
```

Initial-state result:

| push error | policy | avg nodes | return |
|---:|---|---:|---:|
| 0.50 | cheap | 4.00 | 0.206 |
| 0.50 | decision-aware, budget 0.25 | 70.24 | 0.232 |
| 0.50 | uniform true, depth 1 width 4 | 4.00 | 0.246 |
| 0.50 | always deep verifier | 280.00 | 0.246 |
| 0.75 | cheap | 4.00 | 0.199 |
| 0.75 | decision-aware, budget 0.25 | 70.24 | 0.232 |
| 0.75 | uniform true, depth 1 width 4 | 4.00 | 0.246 |
| 0.75 | always deep verifier | 280.00 | 0.246 |

This is a stronger negative diagnostic than the depth-3 uniform baseline:

```text
On initial states, a one-step true-simulator beam already matches the
deep verifier's return in this 100-level sample.

The likely reason is that the current dense potential evaluation makes
good first actions easy to identify with very shallow true search.
```

Action-agreement diagnostic:

| setting | push error | P(uniform true action = deep verifier action) |
|---|---:|---:|
| initial states | 0.50 | 0.84 |
| initial states | 0.75 | 0.84 |
| sampled mid-states | 0.50 | 0.783 |
| sampled mid-states | 0.75 | 0.783 |

Mid-state diagnostic command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 100 \
  --states-per-level 5 \
  --state-sampler-depth 3 \
  --state-sampler-width 8 \
  --rates 0.50,0.75 \
  --budgets 0.01,0.02,0.05,0.10,0.25 \
  --cheap-depth 1 \
  --cheap-width 4 \
  --think-longer-depth 3 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --out outputs/boxoban_kill_or_continue_v0/stage_a_midstates_100_true_sampler_u_d1w4.json
```

Mid-state result:

| push error | policy | avg nodes | return |
|---:|---|---:|---:|
| 0.50 | cheap | 4.00 | 0.253 |
| 0.50 | uniform true, depth 1 width 4 | 4.00 | 0.279 |
| 0.50 | decision-aware, budget 0.25 | 72.63 | 0.282 |
| 0.50 | always deep verifier | 280.00 | 0.289 |
| 0.75 | cheap | 4.00 | 0.242 |
| 0.75 | uniform true, depth 1 width 4 | 4.00 | 0.279 |
| 0.75 | decision-aware, budget 0.25 | 72.63 | 0.270 |
| 0.75 | always deep verifier | 280.00 | 0.289 |

Interpretation:

```text
Mid-states are less degenerate than initial states, but the same core
problem remains: uniform one-step true search is very strong.

At rate 0.50, decision-aware slightly exceeds uniform one-step true
search, but only by using much more search. At rate 0.75, it still loses.

This points to an experiment-design issue rather than a gate-architecture
issue. Making the gate more complex now would not address the strongest
baseline.
```

Current next move:

```text
Do not immediately upgrade the gate.

First change the task/evaluation so that shallow true search is not
already equivalent to deep verification. The two best candidates are:

1. sparse success/deadlock-sensitive evaluation instead of dense
   potential-only evaluation;
2. candidate-plan verification or hard-state sampling, where the
   verifier checks multi-step consequences rather than just selecting
   a good immediate action.

Only after this degeneracy is fixed should we spend effort on a stronger
learned gate.
```

## 2026-06-29 Update: Boxoban-B Candidate-Plan Diagnostic

We tested two fixes for the uniform true-simulator degeneracy.

First, we added a simple deadlock-aware evaluator:

```text
evaluator_mode = deadlock
    terminal success reward
    penalty for boxes pushed into non-goal static corners
    no dense potential shaping
```

This did not fix the problem. On 100 levels with sampled mid-states:

| push error | helpful rate | uniform true action = deep verifier action | cheap | uniform true 1-step | decision-aware 0.25 | always verifier |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.023 | 1.000 | 0.007 | 0.016 | 0.014 | 0.016 |
| 0.75 | 0.023 | 1.000 | 0.007 | 0.016 | 0.012 | 0.016 |

Interpretation:

```text
Simple corner-deadlock penalties make the task too sparse and still do
not create useful separation between shallow true search and deep true
verification.
```

The stronger fix is to change the decision unit from first action to
candidate plan:

```text
decision_unit = action:
    compare policies by first action, then use a deep evaluator after
    that first action

decision_unit = plan:
    each planner submits a full candidate plan
    the policy return is the true rollout of that fixed plan
```

This better matches the project story: the cheap world model imagines a
future plan, and verification checks whether that imagined future is
worth trusting.

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 500 \
  --states-per-level 5 \
  --state-sampler-depth 3 \
  --state-sampler-width 8 \
  --rates 0.50,0.75 \
  --budgets 0.05,0.10,0.25,0.50 \
  --cheap-depth 3 \
  --cheap-width 8 \
  --think-longer-depth 6 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --evaluator-mode dense \
  --decision-unit plan \
  --out outputs/boxoban_kill_or_continue_v0/stage_b_plan_dense_midstates_500_u_d1w4.json
```

500-level candidate-plan result:

| push error | helpful rate | cheap | uniform true 1-step | think-longer | uncertainty | decision-aware 0.25 | always verifier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.331 | 0.154 | 0.144 | 0.187 | 0.241 | 0.247 | 0.373 |
| 0.75 | 0.471 | 0.102 | 0.144 | 0.112 | 0.195 | 0.214 | 0.373 |

Budget sweep:

| push error | budget | decision-aware | uniform true 1-step | random | uncertainty | think-longer | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.05 | 0.170 | 0.144 | 0.163 | 0.173 | 0.187 | 0.195 |
| 0.50 | 0.10 | 0.192 | 0.144 | 0.176 | 0.192 | 0.187 | 0.227 |
| 0.50 | 0.25 | 0.247 | 0.144 | 0.208 | 0.241 | 0.187 | 0.296 |
| 0.50 | 0.50 | 0.313 | 0.144 | 0.262 | 0.306 | 0.187 | 0.367 |
| 0.75 | 0.05 | 0.120 | 0.144 | 0.113 | 0.123 | 0.112 | 0.144 |
| 0.75 | 0.10 | 0.144 | 0.144 | 0.128 | 0.143 | 0.112 | 0.179 |
| 0.75 | 0.25 | 0.214 | 0.144 | 0.170 | 0.195 | 0.112 | 0.257 |
| 0.75 | 0.50 | 0.288 | 0.144 | 0.235 | 0.287 | 0.112 | 0.351 |

Paired bootstrap CI at `budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uniform true | 0.1025 | [0.0896, 0.1156] |
| 0.50 | decision - random | 0.0387 | [0.0229, 0.0539] |
| 0.50 | decision - uncertainty | 0.0056 | [-0.0040, 0.0162] |
| 0.50 | decision - think-longer | 0.0594 | [0.0453, 0.0736] |
| 0.75 | decision - uniform true | 0.0695 | [0.0562, 0.0828] |
| 0.75 | decision - random | 0.0441 | [0.0281, 0.0611] |
| 0.75 | decision - uncertainty | 0.0191 | [0.0045, 0.0342] |
| 0.75 | decision - think-longer | 0.1018 | [0.0875, 0.1175] |

Interpretation:

```text
Candidate-plan evaluation fixes the main degeneracy.

The one-step true baseline is no longer close to the deep verifier,
because a one-step plan is not allowed to receive free deep evaluation
after its first action.

At budget = 0.25, decision-aware selection beats uniform true 1-step,
random, and cheap-model think-longer with positive paired CI in both
main regimes. It also beats uncertainty significantly at push_error_rate
= 0.75, while the 0.50 uncertainty margin is positive but not yet
statistically decisive.
```

Current judgment:

```text
This is the better Stage B direction.

The paper should shift the main Boxoban experiment from action-level
first-step verification to candidate-plan verification. This is also a
cleaner match to the phrase "verifying imagined futures".

The 500-level result is promising, but should be checked at 1000 levels
before changing the main experiment.
```

## 2026-06-29 Update: 1000-Level Candidate-Plan Check

The 1000-level candidate-plan check used the same configuration as the
500-level run, with `--limit 1000`.

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 1000 \
  --states-per-level 5 \
  --state-sampler-depth 3 \
  --state-sampler-width 8 \
  --rates 0.50,0.75 \
  --budgets 0.05,0.10,0.25,0.50 \
  --cheap-depth 3 \
  --cheap-width 8 \
  --think-longer-depth 6 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --evaluator-mode dense \
  --decision-unit plan \
  --out outputs/boxoban_kill_or_continue_v0/stage_b_plan_dense_midstates_1000_u_d1w4.json
```

1000-level candidate-plan result:

| push error | helpful rate | cheap | uniform true 1-step | think-longer | uncertainty | decision-aware 0.25 | always verifier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.322 | 0.151 | 0.136 | 0.195 | 0.231 | 0.243 | 0.362 |
| 0.75 | 0.459 | 0.103 | 0.136 | 0.120 | 0.191 | 0.211 | 0.362 |

Budget sweep:

| push error | budget | decision-aware | uniform true 1-step | random | uncertainty | think-longer | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.05 | 0.169 | 0.136 | 0.162 | 0.168 | 0.195 | 0.190 |
| 0.50 | 0.10 | 0.188 | 0.136 | 0.171 | 0.187 | 0.195 | 0.218 |
| 0.50 | 0.25 | 0.243 | 0.136 | 0.204 | 0.231 | 0.195 | 0.282 |
| 0.50 | 0.50 | 0.301 | 0.136 | 0.257 | 0.300 | 0.195 | 0.354 |
| 0.75 | 0.05 | 0.126 | 0.136 | 0.116 | 0.124 | 0.120 | 0.144 |
| 0.75 | 0.10 | 0.146 | 0.136 | 0.129 | 0.143 | 0.120 | 0.177 |
| 0.75 | 0.25 | 0.211 | 0.136 | 0.170 | 0.191 | 0.120 | 0.251 |
| 0.75 | 0.50 | 0.274 | 0.136 | 0.234 | 0.281 | 0.120 | 0.341 |

Paired bootstrap CI at `budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uniform true | 0.1065 | [0.0977, 0.1153] |
| 0.50 | decision - random | 0.0382 | [0.0287, 0.0474] |
| 0.50 | decision - uncertainty | 0.0119 | [0.0041, 0.0204] |
| 0.50 | decision - think-longer | 0.0479 | [0.0379, 0.0581] |
| 0.75 | decision - uniform true | 0.0751 | [0.0659, 0.0838] |
| 0.75 | decision - random | 0.0414 | [0.0297, 0.0517] |
| 0.75 | decision - uncertainty | 0.0204 | [0.0111, 0.0300] |
| 0.75 | decision - think-longer | 0.0912 | [0.0805, 0.1011] |

Interpretation:

```text
The candidate-plan result holds at 1000 levels.

At budget = 0.25, decision-aware selection beats uniform true 1-step,
random, uncertainty, and cheap-model think-longer with positive paired
bootstrap CIs in both main regimes.

This is now the strongest current Boxoban result and should replace the
action-level setting as the main Stage B experiment.
```

Updated next work:

```text
1. add plan-specific gate features, such as plan length, number of
   pushes, cheap plan score trajectory, and cheap-vs-ensemble plan
   disagreement;
2. compare additional uniform true baselines with longer plan depths
   under matched compute;
3. make plots for compute-return Pareto curves;
4. keep the failed action-level result as internal ablation evidence
   explaining why candidate-plan verification is the right formulation.
```

## 2026-06-29 Update: Simple Plan-Feature Gate Ablation

We added a switchable gate feature set:

```text
gate_feature_set = base:
    score_margin
    uncertainty_proxy
    cheap_score
    ensemble_action_disagreement
    ensemble_score_variance

gate_feature_set = plan:
    base features plus simple cheap-plan descriptors:
        cheap_plan_length
        cheap_plan_turns
        cheap_plan_unique_actions
        cheap_plan_score_per_step
```

Important guardrail:

```text
The plan feature set does not use verifier outputs. Features derived
from verifier scores would leak the result of verification into the
controller and are therefore invalid.
```

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 1000 \
  --states-per-level 5 \
  --state-sampler-depth 3 \
  --state-sampler-width 8 \
  --rates 0.50,0.75 \
  --budgets 0.05,0.10,0.25,0.50 \
  --cheap-depth 3 \
  --cheap-width 8 \
  --think-longer-depth 6 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --evaluator-mode dense \
  --decision-unit plan \
  --gate-feature-set plan \
  --out outputs/boxoban_kill_or_continue_v0/stage_b_plan_features_midstates_1000_u_d1w4.json
```

Result at `budget = 0.25`:

| push error | gate features | decision-aware | uncertainty | uniform true 1-step | random | think-longer |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | base | 0.243 | 0.231 | 0.136 | 0.204 | 0.195 |
| 0.50 | plan | 0.237 | 0.231 | 0.136 | 0.204 | 0.195 |
| 0.75 | base | 0.211 | 0.191 | 0.136 | 0.170 | 0.120 |
| 0.75 | plan | 0.209 | 0.191 | 0.136 | 0.170 | 0.120 |

Paired bootstrap CI for `gate_feature_set = plan` at `budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uniform true | 0.1008 | [0.0916, 0.1090] |
| 0.50 | decision - random | 0.0325 | [0.0226, 0.0415] |
| 0.50 | decision - uncertainty | 0.0062 | [-0.0032, 0.0155] |
| 0.50 | decision - think-longer | 0.0421 | [0.0320, 0.0530] |
| 0.75 | decision - uniform true | 0.0729 | [0.0631, 0.0819] |
| 0.75 | decision - random | 0.0392 | [0.0274, 0.0497] |
| 0.75 | decision - uncertainty | 0.0181 | [0.0078, 0.0288] |
| 0.75 | decision - think-longer | 0.0889 | [0.0782, 0.0991] |

Interpretation:

```text
The simple plan-feature set is not better than the base gate. It remains
positive against uniform true, random, and think-longer, but slightly
reduces decision-aware return compared with the base feature set.

This suggests that naive aggregate plan descriptors are not enough.
Future gate improvements should use richer plan-level evidence, such as:
    cheap plan rollout score trajectory;
    per-step score margins;
    ensemble disagreement over full plans, not only first actions;
    explicit plan validation features.

For now, the main reported controller should remain the base gate in the
candidate-plan setting.
```

## 2026-06-29 Update: Trajectory Feature Gate Ablation

We tested a richer feature set that still uses only information available
before verification:

```text
gate_feature_set = trajectory:
    base features
    simple plan descriptors
    cheap_plan_final_progress
    cheap_plan_state_change_fraction
    cheap_plan_box_change_fraction
    ensemble_plan_disagreement
```

The new features are intended to capture whether the cheap imagined plan
actually changes the state, moves boxes, makes progress, and whether the
cheap-model ensemble agrees on the full plan.

Command:

```bash
PYTHONPATH=. python scripts/run_stage_a_smoke.py \
  --levels data/external/boxoban-sample/medium/train \
  --limit 1000 \
  --states-per-level 5 \
  --state-sampler-depth 3 \
  --state-sampler-width 8 \
  --rates 0.50,0.75 \
  --budgets 0.05,0.10,0.25,0.50 \
  --cheap-depth 3 \
  --cheap-width 8 \
  --think-longer-depth 6 \
  --think-longer-width 16 \
  --uniform-true-depth 1 \
  --uniform-true-width 4 \
  --verifier-depth 6 \
  --verifier-width 16 \
  --eval-depth 6 \
  --eval-width 16 \
  --uncertainty-seeds 5 \
  --evaluator-mode dense \
  --decision-unit plan \
  --gate-feature-set trajectory \
  --out outputs/boxoban_kill_or_continue_v0/stage_b_trajectory_features_midstates_1000_u_d1w4.json
```

Result at `budget = 0.25`:

| push error | gate features | decision-aware | uncertainty | uniform true 1-step | random | think-longer |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | base | 0.243 | 0.231 | 0.136 | 0.204 | 0.195 |
| 0.50 | simple plan | 0.237 | 0.231 | 0.136 | 0.204 | 0.195 |
| 0.50 | trajectory | 0.235 | 0.231 | 0.136 | 0.204 | 0.195 |
| 0.75 | base | 0.211 | 0.191 | 0.136 | 0.170 | 0.120 |
| 0.75 | simple plan | 0.209 | 0.191 | 0.136 | 0.170 | 0.120 |
| 0.75 | trajectory | 0.204 | 0.191 | 0.136 | 0.170 | 0.120 |

Paired bootstrap CI for `gate_feature_set = trajectory` at
`budget = 0.25`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uniform true | 0.0988 | [0.0898, 0.1068] |
| 0.50 | decision - random | 0.0305 | [0.0214, 0.0397] |
| 0.50 | decision - uncertainty | 0.0042 | [-0.0053, 0.0135] |
| 0.50 | decision - think-longer | 0.0401 | [0.0300, 0.0512] |
| 0.75 | decision - uniform true | 0.0680 | [0.0578, 0.0772] |
| 0.75 | decision - random | 0.0343 | [0.0228, 0.0452] |
| 0.75 | decision - uncertainty | 0.0133 | [0.0033, 0.0239] |
| 0.75 | decision - think-longer | 0.0841 | [0.0735, 0.0938] |

Interpretation:

```text
Trajectory features remain positive against the main non-oracle
baselines, but they do not improve over the base gate.

This suggests the current centroid gate is not the right model for
larger heterogeneous feature sets. Adding hand-crafted features without
normalization or a more suitable classifier can hurt selection quality.

The next gate-improvement step should be model-side rather than feature
side:
    1. standardize features before centroid distance; or
    2. train a small logistic-regression / linear classifier gate;
    3. compare these on the same saved candidate-plan rows.

Until then, the base gate remains the main controller for the paper
prototype.
```

## 2026-06-29 Update: Gate Model Ablation

We added switchable gate models while keeping the same labels, rows, and
candidate-plan decision unit:

```text
gate_model = centroid:
    original raw feature-space centroid distance

gate_model = standardized_centroid:
    centroid distance after train-set feature standardization

gate_model = logistic:
    deterministic small logistic gate trained with balanced labels
```

This tests whether the earlier feature ablations were limited by the
controller model rather than by the verification target.

Result at `budget = 0.25`:

| features | gate model | error 0.50 decision | error 0.75 decision |
|---|---|---:|---:|
| base | centroid | 0.243 | 0.211 |
| base | standardized centroid | 0.248 | 0.213 |
| base | logistic | 0.246 | 0.212 |
| plan | centroid | 0.237 | 0.209 |
| plan | standardized centroid | 0.243 | 0.212 |
| plan | logistic | 0.246 | 0.213 |
| trajectory | centroid | 0.235 | 0.204 |
| trajectory | standardized centroid | 0.241 | 0.212 |
| trajectory | logistic | 0.248 | 0.212 |

For reference, the non-decision baselines at this budget are unchanged:

| push error | uncertainty | random | think-longer | uniform true 1-step |
|---:|---:|---:|---:|---:|
| 0.50 | 0.231 | 0.204 | 0.195 | 0.136 |
| 0.75 | 0.191 | 0.170 | 0.120 | 0.136 |

Paired bootstrap CI for the best simple choice,
`gate_feature_set = base`, `gate_model = standardized_centroid`:

| push error | delta | mean | 95% CI |
|---:|---|---:|---:|
| 0.50 | decision - uncertainty | 0.0173 | [0.0091, 0.0258] |
| 0.50 | decision - random | 0.0437 | [0.0342, 0.0538] |
| 0.50 | decision - think-longer | 0.0533 | [0.0425, 0.0643] |
| 0.50 | decision - uniform true | 0.1119 | [0.1031, 0.1214] |
| 0.75 | decision - uncertainty | 0.0219 | [0.0122, 0.0314] |
| 0.75 | decision - random | 0.0429 | [0.0312, 0.0533] |
| 0.75 | decision - think-longer | 0.0927 | [0.0821, 0.1027] |
| 0.75 | decision - uniform true | 0.0766 | [0.0670, 0.0854] |

Interpretation:

```text
Standardizing the centroid gate is the cleanest current improvement.
It improves over the original centroid without changing the story,
labels, verifier, or environment.

Logistic gates help recover some value from larger feature sets, but do
not clearly dominate standardized centroid. For the paper prototype, the
main controller should be:

    decision_unit = plan
    gate_feature_set = base
    gate_model = standardized_centroid

This is a useful result: the controller is still lightweight, but no
longer obviously too naive due to raw feature scale.
```

## 2026-06-29 Update: Standardized Gate Budget Sweep

We swept the verification budget for the current main controller:

```text
decision_unit = plan
gate_feature_set = base
gate_model = standardized_centroid
```

The sweep reuses the saved 1000-level candidate-plan rows. It does not
regenerate planner rollouts.

Result:

| push error | budget | decision-aware | uncertainty | random | think-longer | uniform true 1-step |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.05 | 0.172 | 0.168 | 0.162 | 0.195 | 0.136 |
| 0.50 | 0.10 | 0.189 | 0.187 | 0.171 | 0.195 | 0.136 |
| 0.50 | 0.15 | 0.215 | 0.207 | 0.181 | 0.195 | 0.136 |
| 0.50 | 0.20 | 0.235 | 0.219 | 0.192 | 0.195 | 0.136 |
| 0.50 | 0.25 | 0.248 | 0.231 | 0.204 | 0.195 | 0.136 |
| 0.50 | 0.30 | 0.261 | 0.242 | 0.214 | 0.195 | 0.136 |
| 0.50 | 0.40 | 0.286 | 0.272 | 0.235 | 0.195 | 0.136 |
| 0.50 | 0.50 | 0.296 | 0.300 | 0.257 | 0.195 | 0.136 |
| 0.75 | 0.05 | 0.126 | 0.124 | 0.116 | 0.120 | 0.136 |
| 0.75 | 0.10 | 0.146 | 0.143 | 0.129 | 0.120 | 0.136 |
| 0.75 | 0.15 | 0.167 | 0.158 | 0.142 | 0.120 | 0.136 |
| 0.75 | 0.20 | 0.187 | 0.173 | 0.155 | 0.120 | 0.136 |
| 0.75 | 0.25 | 0.213 | 0.191 | 0.170 | 0.120 | 0.136 |
| 0.75 | 0.30 | 0.226 | 0.209 | 0.183 | 0.120 | 0.136 |
| 0.75 | 0.40 | 0.254 | 0.248 | 0.207 | 0.120 | 0.136 |
| 0.75 | 0.50 | 0.271 | 0.281 | 0.234 | 0.120 | 0.136 |

Paired bootstrap CI for `decision - uncertainty`:

| push error | budget | mean | 95% CI |
|---:|---:|---:|---:|
| 0.50 | 0.05 | 0.0041 | [-0.0015, 0.0099] |
| 0.50 | 0.10 | 0.0018 | [-0.0064, 0.0104] |
| 0.50 | 0.15 | 0.0078 | [-0.0002, 0.0163] |
| 0.50 | 0.20 | 0.0154 | [0.0061, 0.0247] |
| 0.50 | 0.25 | 0.0173 | [0.0091, 0.0258] |
| 0.50 | 0.30 | 0.0189 | [0.0114, 0.0272] |
| 0.50 | 0.40 | 0.0139 | [0.0066, 0.0214] |
| 0.50 | 0.50 | -0.0036 | [-0.0101, 0.0031] |
| 0.75 | 0.05 | 0.0017 | [-0.0052, 0.0084] |
| 0.75 | 0.10 | 0.0036 | [-0.0052, 0.0132] |
| 0.75 | 0.15 | 0.0080 | [-0.0019, 0.0186] |
| 0.75 | 0.20 | 0.0135 | [0.0038, 0.0242] |
| 0.75 | 0.25 | 0.0219 | [0.0122, 0.0314] |
| 0.75 | 0.30 | 0.0169 | [0.0085, 0.0266] |
| 0.75 | 0.40 | 0.0062 | [-0.0011, 0.0137] |
| 0.75 | 0.50 | -0.0103 | [-0.0173, -0.0035] |

Compute note:

```text
At a fixed budget fraction, decision-aware, uncertainty, and random
selection make the same number of verifier calls and have the same mean
node count. The gain is therefore from choosing different states to
verify, not from spending more compute.
```

Interpretation:

```text
The standardized gate is strongest under limited but not tiny
verification budgets. Against uncertainty, the improvement is
statistically clear around budget 0.20-0.30. At budgets 0.05-0.10,
both methods are too verifier-starved for the gap to be reliable. At
budget 0.50, uncertainty catches up, and in the harder error 0.75 regime
it significantly overtakes the gate.

This is the right paper claim: decision-aware selective verification is
most useful when expensive verification is limited enough that allocation
matters, but not so limited that almost no corrections are possible. The
benefit is not monotonic in budget; it has a crossover with uncertainty
gating once verification becomes abundant.
```
