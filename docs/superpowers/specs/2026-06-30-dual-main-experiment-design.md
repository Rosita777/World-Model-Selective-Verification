# Dual Main Experiment Design

Date: 2026-06-30

Status: design spec for the next main experimental stage.

## Goal

Turn the current selective-verification prototype into a credible
world-model planning method evaluation with two co-main environments.

The paper should not look like a Sokoban-only trick. The main claim should
be evaluated in:

```text
Environment A:
    Boxoban / Sokoban
    discrete combinatorial planning with irreversible actions and traps

Environment B:
    PointMaze / Maze2D / AntMaze-style navigation
    trajectory planning with learned dynamics and MPC-style control
```

Both environments use the same research protocol:

```text
cheap learned world model proposes a candidate plan;
lightweight gate decides whether to call an expensive verifier;
the agent executes the first action/control and replans next step.
```

## Core Claim

Decision-aware selective verification improves the compute-return Pareto
frontier in world-model planning.

The main controller is `DIVE`: a Decision-Impact Verification Estimator. DIVE
is not trained to predict model uncertainty directly. It is trained to
estimate verification value:

```text
Will expensive verification change the deployed decision?
If it changes the decision, will the new decision improve outcome?
```

This claim must be tested against uncertainty-based routing and
compute-matched think-longer baselines in both main environments.

## Shared Planning Protocol

At each decision state `s_t`:

1. The cheap planner uses a learned world model `M_c` to produce a candidate
   plan `pi_c` and first action/control `a_c`.
2. The gate observes state, cheap-plan features, model/planner scores, and
   uncertainty features.
3. If the gate fires, the expensive verifier produces a verified plan `pi_v`
   and first action/control `a_v`.
4. The system executes one first action/control.
5. The environment advances to `s_{t+1}` and the system replans.

The default deployment rule is:

```text
if gate(s_t, pi_c, features) == verify:
    execute a_v
else:
    execute a_c
```

This is a closed-loop MPC-style protocol. Plans are evaluated, but only the
first action/control is deployed before replanning.

## DIVE Gate Design

`DIVE(s_t, pi_c)` estimates whether expensive verification has positive
decision value at the current state.

The design goal is to make the gate methodologically meaningful without
turning the project into a large gate-architecture paper.

### Gate Inputs

DIVE uses three groups of features.

Generic state features:

```text
state embedding
goal/progress features
time step / remaining horizon
domain-agnostic risk indicators if available
```

Cheap-plan features:

```text
predicted return of pi_c
candidate plan length
top-k action or plan scores
score margin between best and runner-up plan
planner search depth, width, and nodes
first-action stability under cheap planner perturbations
```

Uncertainty and disagreement features:

```text
ensemble return variance
ensemble top-action disagreement
rollout state disagreement
predicted terminal/success disagreement
```

Uncertainty is an input signal, not the training target.

### Gate Outputs

DIVE is a small multi-head predictor.

```text
change head:
    p_change = P(a_v differs from a_c)

improvement head:
    delta_hat = E[R(pi_v) - R(pi_c)]

harm head:
    p_harm = P(verification changes the decision and makes return worse)

waste head:
    p_waste = P(verification does not change the deployed decision enough)
```

The default verification score is:

```text
score_DIVE =
    p_change * max(delta_hat, 0)
    - alpha * p_harm
    - lambda * verification_cost
```

At a fixed verifier budget, verify the states/plans with highest
`score_DIVE`.

### Gate Model Classes

Use a staged gate family.

```text
DIVE-v0:
    tabular planning features + small MLP
    multi-head outputs
    formula-composed utility score

DIVE-v1:
    add a compact plan/trajectory encoder
    summarize cheap rollout sequences before the MLP heads

DIVE-v2:
    add budget-aware ranking loss
    directly optimize top-budget selection quality
```

The Phase 1 pilot should implement DIVE-v0 only. DIVE-v1 and DIVE-v2 are
reserved for later ablations if DIVE-v0 leaves clear headroom.

### Training Objective

Train DIVE offline from states where both cheap planner and verifier outputs
are recorded.

Default losses:

```text
change head:
    binary cross-entropy on y_change

improvement head:
    Huber or MSE loss on delta_R = R(pi_v) - R(pi_c)

harm head:
    binary cross-entropy on y_harm

waste head:
    binary cross-entropy on y_waste
```

Use class weighting or balanced minibatches if positive verification-value
labels are rare. Validation thresholds and loss weights should be selected
only on pilot validation splits, then locked before full evaluation.

### Cost Constraint

DIVE must be cheap relative to the verifier.

Pilot target:

```text
DIVE inference cost:
    less than 5% of one verifier call in rollout-equivalent compute

DIVE features:
    computed from artifacts already produced by cheap planning, except for
    explicitly budgeted ensemble features used by both DIVE and uncertainty
    baselines
```

This prevents the gate from hiding extra verification-like computation inside
feature extraction.

## Label Definition

DIVE records several labels, not only one binary verify/not-verify label.

For discrete actions:

```text
y_change = 1 iff
    a_v != a_c

y_helpful = 1 iff
    a_v != a_c
    and R(pi_v) > R(pi_c) + epsilon

y_harm = 1 iff
    a_v != a_c
    and R(pi_v) < R(pi_c) - epsilon

y_waste = 1 iff
    a_v == a_c
    or |R(pi_v) - R(pi_c)| <= epsilon

delta_R = R(pi_v) - R(pi_c)
```

For continuous controls:

```text
y_change = 1 iff
    ||a_v - a_c||_2 > delta

y_helpful = 1 iff
    ||a_v - a_c||_2 > delta
    and R(pi_v) > R(pi_c) + epsilon

y_harm = 1 iff
    ||a_v - a_c||_2 > delta
    and R(pi_v) < R(pi_c) - epsilon

y_waste = 1 iff
    ||a_v - a_c||_2 <= delta
    or |R(pi_v) - R(pi_c)| <= epsilon

delta_R = R(pi_v) - R(pi_c)
```

`R(pi_c)` and `R(pi_v)` are measured using the evaluation protocol for the
domain, not merely the cheap model's own predicted score.

For Boxoban / Sokoban, use a shaped evaluation return for labels:

```text
terminal success bonus
step cost
box-on-target progress
deadlock or irreversible-trap penalty if available
```

The main reported metric can still include binary success. The label return
should not collapse to success-only labels because that would turn the gate
into a coarse failure predictor.

For Maze / continuous planning, use cumulative environment reward or a fixed
horizon goal-progress return:

```text
success bonus
negative goal distance
collision or wall penalty if available
control cost if the environment defines one
```

`epsilon` and continuous-action `delta` are selected on pilot validation data
and locked before full evaluation. They must be reported in each experiment
README.

This label is collected offline by running the verifier during training data
generation. At deployment time, the gate is cheap and does not know the
verifier output unless it chooses to invoke the verifier.

## Main Experiment A: Boxoban / Sokoban

### Purpose

Boxoban tests the method in discrete, combinatorial planning with irreversible
actions, deadlocks, and long-horizon consequences.

This environment should demonstrate:

```text
verification is valuable because cheap imagined rollouts can miss traps;
the gate learns where verifier calls have decision value;
selective verification beats uncertainty and uniform extra cheap search.
```

### Environment

Use Boxoban / Sokoban levels with held-out difficulty splits.

Primary evaluation:

```text
medium split:
    1000 evaluation levels

hard / harder split:
    1000 evaluation levels
```

The exact source files and train/eval splits must be recorded in the
experiment README. External level data should not be committed.

### Cheap World Model

Train a lightweight learned model from offline Sokoban trajectories.

Acceptable model outputs:

```text
next grid state
reward / progress score
done flag
optional value, risk, or deadlock proxy
```

The world model should be imperfect but useful. Model quality is controlled
by training data amount, training steps, capacity, or regularization.

Planned quality levels:

```text
weak
medium
strong
```

The weak model should not collapse. The strong model should not saturate the
task so much that verification value disappears.

### Cheap Planner

Use shallow beam search or MPC over the learned model.

The cheap planner returns:

```text
candidate plan
first action
predicted return
top-k plan/action scores
search metadata
```

### Expensive Verifier

Use true Sokoban simulator dynamics with deeper bounded search or a
solver-like planner.

The verifier is not assumed to be perfect. It is simply more accurate or more
expensive than the cheap planner.

### Boxoban Pilot

Run this before any full-scale Boxoban experiment:

```text
levels:
    200 medium levels

model:
    one learned cheap world model

planner/verifier:
    one cheap planner and one verifier setting
```

Go/no-go thresholds:

```text
cheap-only success:
    > 15% and < 85% on medium

always-verify improvement:
    at least +12 percentage points over cheap-only on medium

positive label rate:
    between 15% and 50%

DIVE-v0 helpful-label AUROC:
    at least 0.62

20% budget decision-aware vs random:
    at least +3 percentage points success or return-equivalent gain
```

If the pilot fails, tune model quality or verifier strength before scaling.

## Main Experiment B: Maze / Continuous Planning

### Purpose

The second main environment must show the method is not a Sokoban-specific
grid-search artifact.

It should test trajectory planning with learned dynamics and MPC-style
control.

Preferred starting point:

```text
PointMaze / Maze2D / Wall-style navigation
```

Fallbacks if the preferred environment has too little verification gap:

```text
AntMaze / OGBench maze
Reacher / Pusher-style manipulation
MiniGrid partial-observation tasks only as a last-resort backup
```

MiniGrid is not preferred as the co-main environment because it is too close
to Boxoban in representation and action structure.

### Environment Requirements

The second environment must satisfy:

```text
learned dynamics can make decision-relevant rollout errors;
always-verify clearly improves cheap-only;
uncertainty alone is plausible but not sufficient;
verification has measurable compute cost;
the same gate/budget protocol can be applied.
```

### Cheap World Model

Train a learned dynamics model from offline rollouts:

```text
input:
    state and action/control

output:
    next state
    reward or goal progress
    done/success proxy if available
```

The first implementation may use vector states instead of pixels. Pixel-based
planning is out of scope for the next stage unless the vector-state setting is
too easy.

### Cheap Planner

Use short-horizon MPC or CEM over the learned dynamics.

If actions are continuous, the planner returns a first control vector. If the
environment is discretized for engineering simplicity, the planner returns a
first discrete action or waypoint.

### Expensive Verifier

Use the true simulator with a higher-budget planner:

```text
longer horizon rollout
more CEM samples/iterations
higher-fidelity collision checking
or bounded true-simulator search
```

The verifier must be stronger than the cheap planner but still budgeted.

### Maze Pilot

Run this in parallel with the Boxoban pilot:

```text
variants:
    3 maze/task variants

episodes:
    100-200 evaluation episodes per variant

model:
    one learned cheap dynamics model
```

Go/no-go thresholds:

```text
cheap-only success:
    > 15% and < 85%

always-verify improvement:
    at least +10 percentage points over cheap-only

positive label rate:
    between 15% and 50%

DIVE-v0 helpful-label AUROC:
    at least 0.60

20% budget decision-aware vs random:
    at least +3 percentage points success or return-equivalent gain
```

If PointMaze / Maze2D fails because cheap-only and always-verify are too
close, switch to AntMaze / OGBench maze before doing full-scale runs.

Continuous-environment fallback has a fixed resource cap:

```text
attempt 1:
    PointMaze / Maze2D / Wall-style navigation pilot
    maximum 1 week after shared infrastructure exists

attempt 2:
    AntMaze / OGBench maze pilot
    maximum 1 additional week

if both fail:
    pause continuous scaling and redesign the second co-main environment
    rather than silently downgrading to a weak supplementary experiment
```

MiniGrid partial-observation tasks may be used as a debugging fallback, but
they should not replace the trajectory-planning co-main environment without a
separate design review.

### Full Maze Scale

After pilot success:

```text
task variants:
    3-5

evaluation:
    500 episodes per variant

seeds:
    3 evaluation seeds where feasible

model qualities:
    weak and medium required
    strong required for the full dual-main result unless it saturates the task
    during pilot validation
```

## Required Baselines

The following baselines are required in both main environments.

### Cheap Only

Never call the verifier. This is the lower-cost reference point.

### Always Verify

Call the verifier at every decision state. This is the expensive reference
point and defines the maximum attainable verifier benefit under the chosen
verifier.

### Random Same-Budget

Verify a randomly selected subset of decision states at the same verifier
budget. This tests whether selectivity matters at all.

### Ensemble Uncertainty / Disagreement

Train or sample multiple cheap models/planners and route verification to
states/plans with largest disagreement.

This is non-negotiable because the paper claims decision-aware verification
value is different from uncertainty thresholding.

### Uncertainty-Label Gate

Train the same gate architecture as DIVE, but replace the decision-aware
labels with generic prediction-error or model-disagreement labels.

Example labels:

```text
state prediction error > threshold
return prediction error > threshold
ensemble disagreement > threshold
```

This baseline isolates the value of the supervision target. If DIVE only beats
raw uncertainty ranking but not an uncertainty-label gate with the same model
capacity, the decision-aware claim is weak.

### Compute-Matched Think-Longer

Spend the extra compute on the cheap planner rather than on the verifier.

Examples:

```text
Boxoban:
    increase cheap beam depth/width or number of cheap rollout nodes

Maze:
    increase CEM samples, iterations, or planning horizon in the learned model
```

This baseline is non-negotiable because it tests whether selective
verification beats uniform extra inference-time computation.

Compute matching should be defined before running a full experiment.

Primary accounting:

```text
cheap model forward calls
true simulator calls
rollout steps
search nodes
```

For each budget point, report the total compute used by DIVE and choose a
think-longer configuration whose cheap-model rollout budget is as close as
possible without calling the verifier.

The table should report both:

```text
total rollout-equivalent calls
privileged verifier / true-simulator calls
```

This avoids hiding the fact that verifier calls are qualitatively different
from cheap model calls.

### Oracle Gate

Select the states with largest realized verifier benefit. This provides an
upper bound and helps interpret remaining gate headroom.

### Optional Value / Risk Threshold

If the cheap model has a value/risk head, add a threshold baseline. This is
useful but not required for the first full result.

## DIVE Ablations

At minimum, compare:

```text
binary decision-aware gate:
    one-head verify/not-verify classifier using y_helpful

uncertainty-label gate:
    same architecture trained on prediction error or disagreement labels

DIVE without harm head:
    change + improvement only

DIVE full:
    change + improvement + harm + waste + cost-aware utility
```

The first full ablation can be run on Boxoban only. The second environment
should include the strongest DIVE variant and the fatal baselines before
feature ablations are expanded.

## Metrics

Primary metrics:

```text
success rate
return
verifier call rate
total rollout steps / search nodes / simulator calls
compute-return Pareto curve
```

Gate metrics:

```text
helpful-label AUROC
change-head AUROC
harm-head AUROC
improvement-head regression error
precision / recall
helpful correction rate
harmful flip rate
wasted verification rate
rank correlation with ensemble uncertainty
```

Analysis metrics:

```text
performance by difficulty bucket
performance by model quality
oracle gap
budget sensitivity
```

Wall-clock timing is useful but should not block the main experiments. Report
it in appendix or camera-ready if implementation timing is stable.

## Budget Protocol

Use threshold sweeps over `score_DIVE` and baseline scores to generate Pareto
curves.

Report the following budget points in tables:

```text
0.05
0.10
0.20
0.30
0.50
```

The x-axis should include:

```text
verifier call fraction
search nodes / rollout steps / simulator calls
cheap model forward calls
privileged true-simulator calls
```

When comparing to think-longer, make the compute accounting explicit. The
comparison should answer:

```text
At a fixed inference-time compute budget, is it better to selectively verify
or to uniformly think longer with the cheap world model?
```

The full experiment README must state the exact compute units used for every
baseline before reporting results.

## Execution Phases

### Phase 0: Shared Protocol Spec

Lock this design and write an implementation plan.

Deliverables:

```text
dual-main experiment design
implementation plan
runner interfaces for both environments
DIVE-v0 feature schema and labels
domain-specific R, epsilon, and delta definitions
```

### Phase 1: Dual Pilot

Run both pilots before scaling either environment deeply.

Deliverables:

```text
Boxoban 200-level pilot report
Maze 3-variant pilot report
go/no-go decision for each environment
DIVE-v0 pilot metrics
locked pilot choices for R, epsilon, delta, and compute units
```

### Phase 2: Core Result In One Passing Environment

Scale the environment that passes pilot first while fixing the other if needed.

Deliverables:

```text
budget curve
required baselines
basic gate diagnostics
```

### Phase 3: Full Dual-Main Result

Run both environments with the required baselines and budget curves.

Deliverables:

```text
Boxoban full result
Maze/AntMaze full result
compute-return Pareto curves
```

### Phase 4: Robustness And Analysis

Add seeds, model-quality trends, feature ablations, and difficulty analysis.

Minimum robustness target:

```text
3 gate seeds
1 canonical dynamics seed plus 1-2 replication dynamics seeds where feasible
```

Feature ablations should be done first on Boxoban:

```text
all features
uncertainty-only features
generic-only features
state-only features
plan-only features
DIVE full vs DIVE without harm head
decision-aware labels vs uncertainty labels
```

## Reviewer Risks And Required Defenses

### Risk 1: "This only wins because the verifier uses true simulator."

Defense:

```text
include compute-matched think-longer;
include random same-budget verification;
report compute-return Pareto curves;
show selective verifier calls are better than uniform extra cheap search.
report privileged true-simulator calls separately from cheap model calls.
```

### Risk 2: "This is just uncertainty thresholding."

Defense:

```text
include ensemble disagreement baseline;
include same-architecture uncertainty-label gate;
show helpful-correction precision, not just uncertainty correlation;
report DIVE score correlation with uncertainty scores;
include gate feature ablation.
```

### Risk 3: "The gate label is hindsight oracle supervision."

Defense:

```text
explain offline label collection vs online deployment;
the verifier is the teacher during training;
the gate is the cheap student at inference time.
```

### Risk 4: "The environments are too toy."

Defense:

```text
use two co-main environments with different planning structure;
make the second environment trajectory/MPC-style rather than another grid puzzle;
report nontrivial verification gaps and compute-budget tradeoffs.
```

### Risk 5: "The gate uses domain-specific features."

Defense:

```text
separate generic features from domain features;
report all-features vs generic-only ablation;
keep the main gate interface environment-agnostic.
make DIVE-v0 operate on generic planning statistics before adding
domain-specific features.
```

## Non-Goals For The Next Stage

Do not train a full Dreamer, TD-MPC2, or video world model in the next stage.

Do not use real robots or large-scale pixel control as the next requirement.

Do not claim a new world-model architecture.

Do not claim verification always helps.

Do not treat uncertainty as the target. Uncertainty is a feature or baseline,
not the objective.

Do not implement DIVE-v1 or DIVE-v2 before DIVE-v0 has a passing pilot result.

## Immediate Next Step

After this spec is approved, write an implementation plan for Phase 1:

```text
DIVE-v0 feature schema, labels, and utility score
Boxoban learned-world-model pilot
Maze/Maze2D learned-dynamics pilot
shared label and budget protocol
uncertainty-label gate baseline
compute-matched think-longer definition
required pilot reports
```

No full-scale experiment should start until both pilots have clear
go/no-go outcomes.
