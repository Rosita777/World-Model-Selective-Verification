# Proposal v0: When Should a World Model Think Again?

Date: 2026-06-28

Status: working research proposal. This draft fixes the project direction, story, method boundary, and first experiment package. It is not yet an implementation plan.

## Title

Primary title:

```text
When Should a World Model Think Again?
Decision-Aware Selective Verification for Budgeted Planning
```

Short title:

```text
Decision-Aware Selective Verification for World-Model Planning
```

## One-Sentence Story

World models should not only imagine futures; they should learn when an imagined future is worth verifying.

## Abstract Draft

World-model planning systems use learned dynamics to imagine future outcomes before acting. However, imagined rollouts are imperfect, and uniformly increasing rollout depth or verifying every candidate plan can be computationally expensive and sometimes unnecessary. We study inference-time compute allocation in world-model planning: after a cheap learned world model proposes an action or plan, when should the system invoke a more expensive verifier?

We propose decision-aware selective verification. A lightweight controller observes the current state, the cheap imagined rollout, planner scores, and search metadata, then predicts whether expensive verification will change and improve the deployed decision. Unlike uncertainty-threshold methods, the controller is trained on verification value: whether a specific verifier would produce a better decision under a compute budget. We evaluate this idea in controlled grid-world planning and irreversible puzzle planning, with Sokoban/Boxoban as the headline domain. The main claim is that decision-aware selective verification can improve the compute-return Pareto frontier over cheap planning, always verifying, random gating, uncertainty-based gating, and uniformly spending more compute on cheap rollouts. This claim only holds if the improvement is robust across a meaningful range of cheap-model quality levels, not just a single tuned configuration.

## Motivation

Modern world-model agents can use learned dynamics to plan in imagination. This is useful because the agent can evaluate possible futures before committing to an action. But imagined futures are not equally trustworthy and, more importantly, not all mistakes matter equally for the final decision.

The naive choices are unsatisfactory:

```text
Never verify:
    cheap, but critical imagined mistakes can cause bad actions.

Always verify:
    safer in some cases, but expensive and sometimes unnecessary.

Verify when uncertain:
    plausible, but uncertainty is not the same as decision value.

Think longer uniformly:
    spends extra compute everywhere, even when only a few states need it.
```

The central observation is that verification is an intervention with cost and risk. It can produce:

```text
helpful correction:
    a bad cheap decision is changed into a better decision.

wasted verification:
    the verifier agrees or changes nothing important.

harmful flip:
    a good cheap decision is changed into a worse decision.
```

Therefore, the right question is not:

```text
Is the world model uncertain?
```

but:

```text
If I invoke this verifier now, will the deployed decision change,
and will that change improve outcome enough to justify the cost?
```

This is the planning analogue of selective verification in budget-aware LLM reasoning, but the object being verified is not a text answer. It is an imagined future or candidate plan produced by a cheap world model.

## Positioning

This project should not be framed as:

```text
We propose a new world-model architecture.
We solve world-model verification.
We show verification is always useful.
We replace MCTS, MPC, or model-based RL.
```

It should be framed as:

```text
We study when expensive verification should be invoked
in world-model planning under a compute budget.
```

Recent world-model work studies how to learn dynamics, plan in latent imagination, or use verifiers to improve planning. Our contribution is orthogonal: given a cheap imagined rollout and an available verifier, learn when verification has decision value.

## Connection To SEVRA

SEVRA studies budget-aware LLM reasoning. A model first generates a base answer. A gate then predicts whether active verification is worth invoking. The gate is not trained to predict correctness directly; it is trained to predict recoverability, meaning whether verification can fix the base answer.

Our migration keeps the deeper pattern:

| SEVRA / LLM reasoning | World-model planning |
|---|---|
| base answer | cheap plan / cheap imagined rollout |
| active verification | expensive rollout / stronger model / deeper search |
| gate | decision-aware verification controller |
| helpful fix | bad action changed to better action |
| harmful flip | good action changed to worse action |
| token budget | rollout steps / search nodes / simulator calls / wall-clock |

The key inheritance is not the LLM reflection format. It is the inference-time control pattern:

```text
first produce a cheap result,
then decide whether a more expensive intervention has incremental value.
```

## Problem Setup

At each decision state `s`, the agent has:

```text
M_c:
    a cheap learned world model.

P_c:
    a cheap planner that searches or rolls out in M_c.

V:
    a bounded expensive verifier.

g_phi:
    a lightweight controller that decides whether to call V.
```

The cheap world model predicts:

```text
M_c(s, a) -> predicted next state, reward, done, optional value/risk
```

The cheap planner produces:

```text
a_c:
    first action selected by the cheap planner.

pi_c:
    candidate plan.

tau_hat_c:
    predicted rollout under M_c.

J_hat_c:
    predicted plan score.

metadata:
    top-k action scores, score margin, predicted terminal flags,
    predicted risk/deadlock flags, search depth, beam width, nodes expanded.
```

The verifier produces:

```text
a_v:
    first action selected after verification.

pi_v:
    verified or refined plan.

J_v:
    verifier score.
```

The deployed decision is:

```text
if g_phi says verify:
    deploy a_v
else:
    deploy a_c
```

For MPC-style planning, the first action is the primary deployed decision. The system replans at the next state.

## Method

### Cheap World Model

For grid-based domains, use a small CNN or ResNet dynamics model.

Inputs:

```text
grid state channels + action
```

Outputs:

```text
next grid state
reward
done
optional value / risk / deadlock proxy
```

The cheap model should be intentionally imperfect but not useless. If it is too weak, the optimal policy is close to always verify. If it is too strong, verification value becomes rare. The experiments should sweep cheap model quality by varying training data, model size, or corruption/noise.

### Cheap Planner

Use shallow beam search or MPC in the cheap learned world model.

Candidate configuration:

```text
beam width: 8-32
depth: 5-20 depending on domain
score: predicted reward + value/risk heuristic
```

The planner outputs both the selected decision and features useful for gating:

```text
top-k action scores
score margins
rollout summary
predicted terminal outcome
predicted deadlock/risk
search tree statistics
```

### Expensive Verifier

Use bounded, fallible verifiers. The verifier must be more reliable or more thorough than the cheap planner, but it should not be an unlimited oracle.

Primary verifier condition:

```text
V_learned:
    larger learned world model or ensemble of learned world models
    plus deeper/wider search.
```

This is the main world-model claim. It tests whether selective verification works when the verifier is still learned and fallible.

Supporting verifier condition:

```text
V_sim:
    true rules simulator plus deeper bounded search / A* / beam search
    with a fixed node or rollout budget.
```

This is an upper-bound or high-fidelity comparator, not the sole main result. If results only work with `V_sim` and fail with `V_learned`, the world-model story is weak.

### Feasibility Precondition

Before training the controller, the first experiment must establish that selective verification is worth learning. Across training states, the verification-value distribution should not be degenerate.

Useful target regime:

```text
helpful correction rate:
    at least 15% of states have y = 1.

wasted verification rate:
    at least 20% of states have y = 0.

harmful flip rate:
    measurable, ideally above 2%, but not dominant.
```

If nearly all states have `y = 0`, the cheap model is already too strong or the verifier adds little value. If nearly all states have `y = 1`, the cheap model is too weak and always verifying may dominate. The first deliverable should be this distribution plotted across cheap world-model quality levels and verifier budgets.

### Controller

The controller predicts verification value.

Possible inputs:

```text
current state embedding
cheap imagined rollout summary
cheap top-k action scores
score margin between best and second-best action
predicted reward / value / terminal flags
predicted risk or deadlock indicators
search nodes, depth reached, beam diversity
optional dynamics uncertainty or ensemble disagreement
```

Possible architectures:

```text
small CNN + MLP
MLP over handcrafted features
logistic regression baseline
```

The controller should be cheap enough that its overhead is small compared with one verifier call.

### Labels

Offline, run both the cheap planner and the verifier on training states.

For each state:

```text
a_c = first action from P_c
a_v = first action from V
R_c = actual return or success estimate from deploying a_c
R_v = actual return or success estimate from deploying a_v
```

Primary binary label for MPC:

```text
y = 1 if a_v != a_c and R_v > R_c + epsilon else 0
```

Interpretation:

```text
verification changed the deployed decision,
and the changed decision was better.
```

Also record continuous verification gain:

```text
Delta = R_v - R_c
```

This enables a regression version of the gate:

```text
call verifier if predicted Delta > cost threshold
```

Diagnostics:

```text
helpful correction:
    a_v != a_c and R_v > R_c + epsilon

harmful flip:
    a_v != a_c and R_v < R_c - epsilon

wasted verification:
    a_v == a_c or |R_v - R_c| <= epsilon

spurious correction:
    a_v != a_c but |R_v - R_c| <= epsilon
```

For whole-plan deployment, replace first-action difference with plan difference. For MPC, first-action difference remains primary.

## Data Collection And Distribution Shift

A naive offline dataset collected only from the cheap planner may not match the states visited by the gated policy. This matters because a corrected early action can send the agent into different later states.

Training states should come from a mixture:

```text
cheap-only rollouts
always-verify rollouts
random-gated rollouts
preliminary learned-gate rollouts
```

Recommended procedure:

```text
1. collect initial mixed-policy states
2. run P_c and V on each state
3. compute labels and train g_phi
4. deploy g_phi on held-out training levels
5. collect newly visited states
6. label them and optionally retrain once
```

The paper should report gate performance on:

```text
offline dataset states
states visited by the gated policy
held-out levels
```

If distribution shift is small, report it. If it is large but one relabeling round helps, that becomes useful evidence.

## Experiments

The experiments should be presented as formal domains, not as a small toy followed by a real experiment.

### Experiment 1: Controlled Grid-World World-Model Planning

Candidate domains:

```text
MiniGrid DoorKey
MiniGrid LavaCrossing
```

Purpose:

```text
mechanism clarity
visual examples of helpful correction and harmful flip
debugging the label/gate/verifier pipeline
```

This experiment should answer:

```text
Can the pipeline learn selective verification at all?
Does the gate beat random gating at matched verifier budget?
Can we visualize states where verification is useful vs. wasteful?
```

This should not be described as a throwaway sanity check. It is a controlled benchmark with interpretable states.

### Experiment 2: Irreversible Puzzle Planning

Headline domain:

```text
Sokoban / Boxoban
```

Motivation:

```text
Sokoban contains irreversible pushes and deadlocks.
An action can look locally good but make future success impossible.
This makes verification value sparse, state-dependent, and decision-relevant.
```

Main setup:

```text
M_c:
    small learned grid dynamics model.

P_c:
    shallow beam/MPC in M_c.

V_learned:
    ensemble or larger learned world model + deeper/wider search.

V_sim:
    true rules simulator + bounded deeper search as upper-bound analysis.

g_phi:
    decision-aware selective verification controller.
```

Generalization:

```text
train on training levels
select thresholds on validation levels
evaluate on held-out test levels
```

This is the main experiment that should carry the paper.

### Experiment 3: Uncertainty-Relevance Disentanglement

This experiment exists to answer the strongest reviewer objection:

```text
Is decision-aware verification really different from uncertainty thresholding?
```

Possible forms:

```text
partially observable grid-world
stochastic/noisy transition grid-world
synthetic planning trees with controlled uncertainty relevance
Boxoban variants with distractor uncertainty
```

The goal is to create or identify cases where:

```text
uncertain predictions do not affect the root decision
confident predictions can still lead to bad irreversible decisions
```

Key measurements:

```text
fraction of uncertain-but-decision-irrelevant states
fraction of low-uncertainty-but-high-verification-value states
oracle gap between uncertainty-only verification and decision-aware verification
```

This experiment can be framed as a stress test or boundary analysis:

```text
when decision-aware verification helps,
and when it collapses to uncertainty-based routing.
```

### Optional Extension: Continuous Control

Only add after the discrete domains work.

Candidate domains:

```text
CartPole
Pendulum
simple DMControl task
```

Possible setup:

```text
cheap learned dynamics + short MPC
expensive learned ensemble / longer MPC
decision-aware gate
```

Purpose:

```text
show that the framework is not tied to grid puzzles.
```

This is not required for the first working paper unless reviewers or early results demand broader coverage.

## Baselines

Required baselines:

```text
Cheap only:
    always deploy a_c.

Always verify:
    always call V and deploy a_v.

Random gate:
    call V randomly at the same verifier-call budget.

Dynamics uncertainty threshold:
    train an ensemble of K cheap world models, K = 3-5.
    Call V when predicted next-state / rollout disagreement is high.

Action or value uncertainty threshold:
    run the cheap planner in each ensemble member.
    Call V when top-1 actions disagree, or when predicted-return
    variance exceeds a threshold.

Think longer:
    spend matched compute on deeper/wider cheap-model search,
    without post-hoc verification.

Uncertainty-trained controller:
    same architecture as ours, but trained to predict model error or uncertainty labels.

Oracle selective gate:
    upper bound that calls V exactly when verification would help.
```

The most important comparisons are:

```text
ours vs uncertainty threshold
ours vs value/Q uncertainty threshold
ours vs think longer
ours vs always verify at matched or reported compute
```

All threshold-based baselines should be swept to produce budget-performance curves, not compared at a single hand-picked threshold.

## Metrics

Primary:

```text
success rate / return vs verifier-call fraction
success rate / return vs total compute
compute-return Pareto curve
fixed-budget performance at 10%, 25%, 50% verification
```

Compute accounting:

```text
verifier calls
rollout steps
search nodes
wall-clock time
controller overhead
FLOPs when practical
```

Gate quality:

```text
AUROC
AUPRC
precision / recall at operating points
calibration of predicted verification value
```

Verification diagnostics:

```text
helpful correction rate
harmful flip rate
wasted verification rate
spurious correction rate
oracle gap closure
```

World-model-specific analysis:

```text
single-step model error vs multi-step rollout divergence
compounding-error regimes
feature ablation for rollout features vs score-margin-only features
decision relevance of uncertainty
```

Statistical reporting:

```text
mean +/- standard deviation over at least 5 seeds when feasible
paired bootstrap or Wilcoxon signed-rank tests for key comparisons
per-level or per-difficulty breakdown for Sokoban/Boxoban
```

## Key Figures

Minimum convincing figure set:

1. Compute-return Pareto curve on Boxoban.
2. Fixed-budget table at 10%, 25%, and 50% verification.
3. Helpful correction / harmful flip / wasted verification decomposition.
4. Gate vs uncertainty baseline on uncertainty-relevance stress test.
5. Qualitative Sokoban examples:

```text
gate calls verifier and avoids deadlock
gate skips verifier correctly
gate misses a useful verification
gate calls verifier unnecessarily
harmful flip example
```

## Expected Result Bar

The project is worth writing up if the following are approximately true:

```text
1. Ours beats uncertainty-based gating by at least 5% relative return/solve-rate
   or saves at least 30% verifier calls at similar performance.

2. The advantage appears in at least two non-trivial domains or settings.

3. Ours closes at least 50% of the gap between cheap-only and oracle selective gate.

4. Controller overhead is small, ideally below 15% of one verifier call.

5. The strongest results hold for V_learned, not only V_sim.
```

If results only work with true simulator verification, the claim should be weakened to high-fidelity fallback control. If uncertainty baselines match the method everywhere, the paper should become a boundary study of when decision-aware verification collapses to uncertainty routing.

## Reviewer Objections And Responses

### Objection: This Is Just Uncertainty-Aware Planning

Response:

```text
Uncertainty can be a feature, but it is not the target.
The target is the value of invoking a specific verifier:
whether verification changes and improves the deployed decision.
```

Required evidence:

```text
compare against dynamics uncertainty and value/Q uncertainty thresholds
compare against same controller trained on uncertainty labels
show uncertain-but-irrelevant and confident-but-dangerous cases
```

### Objection: This Is Just MCTS Or Search Budget Allocation

Response:

```text
MCTS allocates search internally using a hand-designed tree policy.
We study plan-level or rollout-level verifier invocation in a world-model planning pipeline.
The controller is trained to predict decision-aware verification value after seeing a cheap imagined rollout.
```

Required evidence:

```text
think-longer baseline with matched compute
ablation separating model quality from search depth
```

### Objection: The Verifier Is An Oracle

Response:

```text
The main verifier is V_learned: a stronger learned model or ensemble with bounded search.
V_sim is reported only as a high-fidelity upper-bound condition.
```

Required evidence:

```text
main Pareto gains hold for V_learned
V_sim results are labeled as analysis / upper bound
```

### Objection: Grid Worlds Are Too Simple

Response:

```text
The controlled grid-world experiment is for mechanism clarity.
The headline experiment is Boxoban/Sokoban with irreversible actions and held-out levels.
Stress tests examine when decision-aware verification differs from uncertainty routing.
```

Possible extension:

```text
simple continuous-control MPC after discrete experiments work.
```

### Objection: Gate Training Requires Running The Verifier Everywhere

Response:

```text
The gate is trained offline from labeled states, like a learned router or cascade.
At deployment, it amortizes verifier calls on held-out levels.
```

Required evidence:

```text
generalization to held-out levels
measurement on states visited by the gated policy
optional one-round DAgger-style relabeling
```

## Work Plan After Proposal

This proposal should be followed by an implementation plan, not immediate large-scale experimentation.

Recommended next steps:

```text
1. Build a minimal Boxoban/Sokoban environment wrapper.
2. Implement data collection for transitions and planning states.
3. Train the first cheap grid dynamics model.
4. Implement cheap beam/MPC planner in the learned model.
5. Implement bounded V_learned and V_sim.
6. Generate label dataset for gate training.
7. Train simple gate baselines.
8. Produce the first Pareto curve.
```

Before writing code, decide the exact first domain and verifier:

```text
Option A:
    start with MiniGrid as controlled benchmark, then Boxoban.

Option B:
    start directly with Boxoban and use a tiny subset for fast iteration.

Recommended:
    start directly with Boxoban infrastructure,
    but keep a small controlled grid-world as a debugging benchmark.
```

The paper-facing story should not call any part a toy sanity check. Internally, we can still use small subsets for debugging.

## Current Decision

Proceed with:

```text
Boxoban/Sokoban as the headline environment.
Small learned grid world model as M_c.
Learned ensemble / larger learned model as V_learned.
True simulator bounded search as V_sim upper-bound analysis.
Decision-aware gate trained on action-changing return improvement.
Uncertainty and think-longer baselines as non-negotiable comparisons.
```
