# Opus Review: Stage A 1000-Level Budget Sweep

Date: 2026-06-28

Context sent to Opus:

```text
1000 Boxoban medium/train levels.
Train/eval split: 50/50.
Cheap planner: degraded evaluator, depth 1, width 4.
Think-longer baseline: same degraded evaluator, depth 3, width 16.
Verifier/evaluator: true simulator, depth 6, width 16.
Budgets: 0.05, 0.10, 0.25, 0.50, 1.00.
```

Opus judgment:

```text
The signal is directionally right and enough to continue, but the raw
decision-vs-uncertainty gap is small. Bootstrap confidence intervals are
required before treating the numbers as a confirmed result.
```

Critiques adopted:

```text
1. Add bootstrap CI immediately.
2. Treat decision-aware > uncertainty as a Stage A controlled finding,
   not a broad final claim.
3. Keep learned world models for Stage B.
4. Add a stronger iso-compute baseline later.
```

Critiques interpreted with caution:

```text
The current think-longer baseline uses the same degraded cheap evaluator
with more search. This matches the project's original "think longer with
the cheap model" baseline.

Opus suggested a same-quality true-simulator think-longer baseline. That
is useful, but it should be framed separately as a uniform verifier-compute
baseline rather than replacing the cheap-model think-longer baseline.
```

Follow-up implemented:

```text
1. Added paired bootstrap utilities.
2. Added a CI summary script for Stage A JSON outputs.
3. Ran 1000-sample bootstrap CI at rates 0.50 and 0.75, budget 0.25.
```
