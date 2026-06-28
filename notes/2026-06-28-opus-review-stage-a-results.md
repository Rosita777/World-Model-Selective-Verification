# Opus Review: Stage A 100-Level Results

Date: 2026-06-28

Context sent to Opus:

```text
Stage A Boxoban/Sokoban prototype.
Cheap planner: degraded push evaluator + shallow beam.
Verifier/evaluator: true simulator + deeper beam.
Gate label: verifier changes the cheap action and improves evaluated return.
Budget: verify 25% of eval states.
Baseline: ensemble uncertainty from 5 degraded cheap planners.
```

Opus judgment:

```text
Continue only if the experiment creates a meaningful decision gap.
The dangerous failure mode is evaluating a verification gate when
always-verify barely improves over cheap-only.
```

Critiques worth adopting:

```text
1. Do not over-interpret tiny 100-level / 50-eval runs.
2. Report an oracle selective-verification upper bound.
3. Focus on Goldilocks degradation regimes where helpful verification is
   common enough to matter but not so common that the task is trivial.
4. Increase the number of eval levels before making claims.
5. Keep Sokoban/Boxoban for Stage A; do not jump to MuJoCo or learned
   world models before the controlled setting works.
```

Critiques not adopted as immediate work:

```text
1. Do not train a learned world model yet.
2. Do not replace the gate with a neural architecture yet.
3. Do not spend time on many uncertainty aggregation variants yet.
4. Do not switch environments yet.
```

Follow-up implemented in response:

```text
1. Added train/eval split to the smoke runner.
2. Added ensemble uncertainty baseline.
3. Added random baseline.
4. Added oracle upper bound.
5. Re-ran a 500-level split experiment.
```
