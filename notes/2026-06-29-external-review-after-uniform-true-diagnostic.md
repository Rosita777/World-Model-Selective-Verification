# External Review After Uniform-True Diagnostic

Model used: gpt-5.5 via 987xyz.com/v1. Claude/Opus channel was unavailable under the current token.

1. **Fatal flaw if you continue current setup**

Your verifier is not meaningfully valuable. A **uniform true 1-step simulator with 4 nodes matches the deep verifier** on initial states and agrees with it 84% of the time. That means the benchmark is mostly testing whether you can avoid calling an expensive module that is unnecessary. Any selective-verification win will look artificial or be dominated by “just use the true simulator shallowly.”

The dense potential likely makes Boxoban locally myopic: the first move is too easy.

2. **Best next move**

Do **not** build a stronger gate yet. The signal is not there.

Best next move: **make a harder evaluation setting using sparse/deadlock-aware evaluation plus hard mid-states.**

Ranking:

1. **Sparse/deadlock eval + hard mid-states** — best. Forces cases where shallow true search fails and deep verification matters.  
2. **Candidate-plan verification** — useful after verifier value is established; closer to the real story.  
3. **Switch env** — do this if Boxoban still collapses after the above.  
4. **Stronger gate** — premature; it may just overfit weak artifacts.

3. **Exact next experiment and success criterion**

Run a “Boxoban-B” diagnostic:

- Sample **mid-states enriched for traps/deadlocks**, e.g. states after 5–30 cheap-planner actions, filtered to include:
  - cheap top action disagrees with deep verifier, or
  - shallow true 1-step disagrees with deep verifier, or
  - known deadlock/potential-trap patterns.
- Use **sparser reward/eval**:
  - terminal box-on-target reward,
  - explicit deadlock penalties,
  - remove or heavily downweight dense centroid/potential shaping.
- Compare:
  1. cheap planner only,
  2. cheap-think-longer,
  3. uniform true 1-step,
  4. uniform true shallow/deep at matched node budgets,
  5. deep verifier always,
  6. decision-aware selective verifier at rates 25/50/75%.
- Measure return vs node budget, plus action agreement with deep verifier.

Success criterion:

- Uniform true 1-step should **no longer match deep verifier**:
  - action agreement with deep verifier preferably drops below ~65%;
  - return gap between 1-step and deep verifier is statistically clear.
- Selective verification should dominate compute-matched baselines:
  - at ≤50% verifier-call rate, recover at least **50% of the deep-verifier gain over cheap**, while using substantially fewer nodes than always-verify;
  - beat uniform true 1-step and cheap-think-longer with bootstrap CIs.

If uniform true 1-step is still near-optimal, abandon this Boxoban setup for the paper.

4. **What not to claim**

Do **not** claim:

- “Decision-aware verification beats uncertainty” in general.
- “Selective verification is useful for world-model planning” broadly.
- “The gate learns when the model is wrong” — you explicitly say this is not uncertainty.
- “Verifier calls are valuable” under current Stage A; the data says shallow true search is already enough.
- “Compute-efficient planning” unless you beat **uniform true simulator at matched compute**, not just random gating or cheap baselines.

Current honest claim is only:  
**In a toy Boxoban setting, a simple decision-aware gate can sometimes allocate verifier calls better than naive baselines, but the current task is too locally solvable to demonstrate real verifier value.**
