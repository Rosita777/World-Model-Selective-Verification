# World Model Selective Verification

Working directory for the research project:

```text
When Should a World Model Think Again?
Decision-Aware Selective Verification for Budgeted Planning
```

## Core Idea

World models should not only imagine futures; they should learn when an imagined future is worth verifying.

This project studies inference-time compute allocation in world-model planning. A cheap learned world model first produces an imagined rollout or candidate plan. A lightweight controller then decides whether it is worth invoking a more expensive, bounded verifier such as a stronger learned world model, ensemble, deeper search, or high-fidelity simulator.

The controller should not merely predict uncertainty. It should predict decision-aware verification value:

```text
Will verification change the deployed decision, and will that change improve outcome enough to justify cost?
```

## Current Framing

We do not claim to invent a new world-model architecture or to verify every rollout. The contribution target is selective verifier invocation under a compute budget.

Primary experimental direction:

```text
cheap learned world model + shallow planning
decision-aware gate/controller
bounded expensive verifier
compute-return Pareto evaluation
```

## Directory Layout

- `notes/`: working notes, reading notes, external-model critiques, meeting summaries.
- `docs/proposals/`: research proposal drafts.
- `docs/paper/`: paper outline, related work, figures, rebuttal notes.
- `experiments/`: experiment records, logs, result summaries.
- `scripts/`: runnable code and utilities.
- `configs/`: experiment configs.
- `data/`: small manifests or metadata only. Do not store large datasets blindly.
- `outputs/`: generated plots, tables, and inspection artifacts.
- `tests/`: tests for reusable code.

## Project Hygiene

Keep this project separate from the diffusion personalization repository. Do not store API keys, private tokens, raw large datasets, or model checkpoints in this directory.
