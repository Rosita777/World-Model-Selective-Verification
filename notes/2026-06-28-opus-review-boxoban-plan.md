# Opus Review: Boxoban Kill-Or-Continue Plan

Date: 2026-06-28

Status: external-model critique. Treat as input, not authority.

## Context

We asked Opus to critique the first implementation plan for:

```text
Decision-Aware Selective Verification for Budgeted World-Model Planning
```

The plan originally built package infrastructure, a Sokoban stepper, Boxoban loading, transition collection, a learned cheap world model, a planner, label generation, a gate, and Pareto evaluation.

## Useful Critiques

### The Plan Was Too Heavy For A Kill-Or-Continue

Opus argued that the package structure, per-module tests, download script, transition collection, and learned-model training are too much before we know whether the gating idea has signal.

Our judgment:

```text
Partly adopt.
```

We will keep a lightweight package structure because this repository is intended to become the experiment codebase, but we will not train learned world models before the first signal check.

### First Sweep Should Use A Controlled Degraded Cheap Model

Opus suggested:

```text
Use true Sokoban rules as the bounded verifier,
and create a synthetically degraded cheap model/planner first.
```

This lets us sweep the cheap/verifier gap directly and test whether decision-aware gating can beat uncertainty-like routing before spending time on neural dynamics.

Our judgment:

```text
Adopt for Stage A.
```

This does weaken the immediate world-model claim, so it must be treated as a kill-or-continue engineering stage, not the paper's final experiment.

### Biggest Technical Risk

The main risk is failing to land in the Goldilocks zone:

```text
cheap model too strong:
    verification is almost never useful.

cheap model too weak:
    always verify dominates.

middle regime:
    selective verification can matter.
```

Our first feasibility report must sweep degradation levels and report:

```text
helpful correction rate
wasted verification rate
harmful flip rate
best-action disagreement rate between cheap planner and verifier
```

### Strong Baseline Must Not Be Skipped

The most important first baseline is not random gating. It is an uncertainty-style or cheap-failure proxy at matched verifier budget.

In Stage A, before learned ensembles exist, use:

```text
cheap planner instability / corruption risk score
score-margin proxy
degradation-aware uncertainty proxy if available
```

In Stage B, replace this with:

```text
ensemble top-action disagreement
predicted-return variance
dynamics rollout disagreement
```

## Adopted Changes

```text
1. Added a Stage A execution override to the implementation plan.
2. Stage A uses a synthetically degraded cheap planner before learned dynamics.
3. Learned cheap world model and learned verifier move to Stage B.
4. First report focuses on base-rate feasibility and decision-aware vs uncertainty-proxy gating.
```

## Rejected Or Deferred Suggestions

Opus suggested flattening everything into one or two scripts. We defer that suggestion because:

```text
the project already has a clean repository,
we need reusable Sokoban/planning/gating utilities,
and TDD is easier with small modules.
```

But we accept the spirit:

```text
avoid overbuilding learned-model infrastructure before the first signal.
```

