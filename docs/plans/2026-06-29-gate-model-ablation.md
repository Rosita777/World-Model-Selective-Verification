# Gate Model Ablation Implementation Plan

**Goal:** Test whether the current decision-aware gate is limited by the centroid classifier rather than by the verification labels or feature sets.

**Design:** Keep the Stage B plan-level Boxoban setup fixed. Add two gate models beside the existing raw centroid gate: a standardized centroid gate and a small deterministic logistic gate. Both consume only pre-verification row features and train on offline verifier-derived labels.

**Execution Steps:**
- Add tests for standardized centroid scoring, logistic scoring, and experiment-script `gate_model` reporting.
- Implement the new gate models in `wmsv/gating/simple.py`.
- Add `--gate-model centroid|standardized_centroid|logistic` to Stage A/Stage B evaluation and CI summarization scripts.
- Recompute budget and CI summaries from existing 1000-level plan rows, without regenerating expensive rows.
- Record the outcome in `experiments/boxoban_kill_or_continue_v0/README.md`.
