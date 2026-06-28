# External Review Summary

Date: 2026-06-28

Status: summary of external-model critique. Treat as reviewer-style input, not authority.

No API key or secret is stored in this note.

## Context

We asked an external Claude model to critique the project:

```text
When Should a World Model Think Again?
Decision-Aware Selective Verification for Budgeted Planning
```

The critique focused on whether the proposal legitimately counts as world-model planning, whether the verifier design is fair, whether the label is correct, and what a skeptical reviewer would attack.

## Useful Critiques

### 1. The Cheap Path Must Be A Real World-Model Path

The project is only a world-model planning paper if the cheap planner actually uses a learned predictive model to imagine future states.

Bad framing:

```text
weak policy directly outputs an action
```

Better framing:

```text
cheap learned dynamics model + shallow planning over imagined rollouts
```

This critique is adopted.

### 2. True Simulator Verification Cannot Be The Only Main Verifier

If the verifier is only the true rules simulator, reviewers may say the method is just learning when to fall back to ground truth.

Adopted design:

```text
V_learned:
    main verifier condition, using a larger/ensemble learned world model.

V_sim:
    high-fidelity upper-bound analysis, not the core claim.
```

### 3. The Verifier Must Be Bounded And Fallible

An unlimited solver would make verification look like an oracle. The verifier should have a fixed node, rollout, or wall-clock budget.

This preserves the SEVRA-style story:

```text
verification can help,
verification can waste compute,
verification can sometimes harm.
```

### 4. Distribution Shift Must Be Measured

A gated policy visits different states from a cheap-only policy. Therefore gate metrics should be reported not only on the offline dataset but also on states visited by the gated policy.

Adopted plan:

```text
collect mixed-policy states
optionally do one DAgger-style relabeling round
report gate precision/recall on gated-policy states
```

### 5. The Hardest Reviewer Attack Is The Uncertainty Collapse

In deterministic fully observable puzzles, uncertainty and decision relevance may be highly correlated. If uncertainty thresholds match our gate, the core distinction weakens.

Adopted response:

```text
include uncertainty-relevance disentanglement stress tests
compare against dynamics uncertainty and value/Q uncertainty
measure uncertain-but-irrelevant and confident-but-dangerous cases
```

## Label Feedback

The external critique agreed that the primary positive label should keep the action-change condition:

```text
y = 1 if a_v != a_c and R_v > R_c + epsilon
```

Reason:

```text
If verification does not change the deployed decision,
then it is not a helpful decision correction in the SEVRA sense.
```

But the critique also recommended reporting:

```text
Delta = R_v - R_c
```

and training a regression variant.

This is adopted.

## Result Bar Suggested By External Review

A publishable result should roughly satisfy:

```text
>= 5% relative improvement over uncertainty gating
or >= 30% fewer verifier calls at matched performance
in at least two non-trivial settings.
```

Additional desired conditions:

```text
close >= 50% of the cheap-only to oracle-selective gap
controller overhead < 15% of one verifier call
main gains hold for V_learned, not only V_sim
```

## Our Judgment

The critique usefully tightened the proposal. The main changes are:

```text
1. Make V_learned the primary verifier.
2. Keep V_sim only as upper-bound analysis.
3. Add distribution-shift measurement.
4. Add uncertainty-relevance stress tests.
5. Preserve the decision-changing helpful-correction label.
```

The project should proceed only if early experiments show that decision-aware gating beats strong uncertainty and think-longer baselines at matched compute.

## Post-Proposal-v0 Review

After writing proposal v0, we sent the draft excerpt for another strict review. The useful additional points were:

```text
1. The viable regime may be narrow.
   If the cheap model is too good, verification value is too sparse.
   If it is too weak, always verify dominates.

2. Sokoban is strong for irreversible planning, but risky for broad
   world-model framing because it is deterministic, fully observable,
   and has a compact rules simulator.

3. V_learned may share failure modes with M_c.
   The first experiments must test whether the learned verifier is
   genuinely better in the relevant states.

4. Uncertainty baselines must be specified strongly.
   Ensemble disagreement over dynamics and top-action/value disagreement
   are the key baselines, not weak entropy heuristics.

5. The proposal should hedge general claims until experiments show
   robustness across cheap-model quality levels.
```

Adopted edits:

```text
1. Hedged the abstract claim from "improves" to "can improve".
2. Added a feasibility-precondition section.
3. Made uncertainty baselines concrete.
4. Added threshold-sweep, compute, and statistical reporting commitments.
```

The strongest next-step recommendation was:

```text
Start with the primary benchmark and the strongest uncertainty baseline.
If decision-aware gating does not beat that comparison, pause and revise
the story before running the full experiment suite.
```
