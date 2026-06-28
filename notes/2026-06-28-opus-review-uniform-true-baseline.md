# Opus Review: Uniform True-Simulator Baseline

Date: 2026-06-28

Context sent to Opus:

```text
1000 Boxoban levels, train/eval 50/50.
Decision-aware budget 0.25: about 73 nodes per eval state.
Uniform true baseline: true simulator shallow beam, depth 3, width 12,
about 68 nodes per eval state.
```

Observed result:

```text
rate = 0.50:
decision - uniform_true = -0.0017, 95% CI [-0.0062, 0.0026]

rate = 0.75:
decision - uniform_true = -0.0091, 95% CI [-0.0151, -0.0035]
```

Opus judgment:

```text
This is a structural negative result for any broad claim that selective
verification beats uniform compute allocation. Uniform true-simulator
search is too strong in this Stage A setting.
```

Critiques adopted:

```text
1. uniform_true must be a main baseline, not hidden as a diagnostic.
2. The current Stage A claim must be narrowed.
3. The next experiment should compare strictly compute-matched true
   simulator allocation curves.
```

Our interpretation:

```text
The result does not kill the whole project, but it changes the framing.

The current evidence supports:
    decision-aware selection > random / uncertainty / cheap-model
    think-longer in the degraded-world-model setting.

The current evidence does not support:
    decision-aware selective verification > uniform shallow
    verifier-quality search.

The next controlled question is whether selective deep verification can
beat uniform shallow verifier-quality search on harder levels, lower
budgets, or settings where verifier-quality calls have larger fixed cost.
```
