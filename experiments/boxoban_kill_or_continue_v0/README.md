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
    uncertainty-proxy ranking from the degraded evaluator
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

