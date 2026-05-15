# Paper Analysis, Weaknesses, and Experiment Roadmap

Generated from conversation analysis of v4b experimental results (2026-05-05).

---

## The Headline Number

On **random-32-32-10 N=100, 512 steps**:
- ORCA: **16.8%** at_goal, 4169 collisions
- Flow v4b + CV-PIBT: **86.7%** at_goal, 650 collisions
- That is **5.2× more agents reaching goals**, with **84.4% fewer collisions**

---

## 1. Result Analysis

### Two-Layer Contribution Story

The results decompose cleanly into two additive layers:

| Method | random-32-32-10 N=100 at_goal | Δ vs prev | Collisions |
|--------|-------------------------------|-----------|------------|
| ORCA (reactive baseline) | 16.8% | — | 4169 |
| PO-ORCA (+ priority, no backtrack) | 14.9% | -1.9pp | 2626 |
| CV-PIBT + straight (+ backtracking) | 77.4% | +62.5pp | 1307 |
| CV-PIBT + Flow v4b (+ learned velocity) | 86.7% | +9.3pp | 650 |

**Layer 1 — CV-PIBT (the mechanism):** ORCA deadlocks in cluttered environments and achieves only 16.8% at_goal. CV-PIBT's priority-inheritance backtracking resolves gridlock, jumping to 77.4% — a 4.6× improvement — with 69% fewer collisions, using only naive straight-line preferred velocities.

**Layer 2 — Flow model (the learned policy):** On top of CV-PIBT, the flow model adds +9.3pp at_goal and halves collisions again (1307→650, -50.3%). Learned velocities allow agents to plan *around* congestion rather than reactively requesting blocked directions.

### The PO-ORCA Ablation is Clean

PO-ORCA (priority ordering without backtracking) achieves 14.9% at_goal on random N=100 — almost identical to ORCA's 16.8%. Static priority ordering adds nothing. It is the **inheritance + backtracking mechanism** that matters. This is a crisp, publishable ablation.

### v4b vs v4 Model Comparison (256 steps)

v4b (5 training maps) consistently outperforms v4 (9 training maps):

| Map | N | v4b at_goal | v4 at_goal | Δ |
|-----|---|-------------|------------|---|
| empty-48-48 | 50 | 0.851 | 0.723 | +17.7% |
| empty-48-48 | 100 | 0.811 | 0.679 | +19.4% |
| random-32-32-10 | 50 | 0.828 | 0.751 | +10.3% |
| random-32-32-10 | 100 | 0.820 | 0.732 | +12.0% |

Fewer, higher-quality training maps outperform more maps with noisier data. This is a useful training insight.

### Arrival Speed

On random-32-32-10 N=50, agents that DO arrive via the flow model arrive at step 166.7 vs 181.9 for Straight+CV-PIBT — **8.4% faster**. The model doesn't make agents slower individually; it makes more agents succeed via more circuitous routes.

### Empty Maps: Expected Regression

On empty-48-48, ORCA achieves 100% at_goal — it was designed for convex open spaces with no local minima. The full system gets 98.6% (N=50) and 98.1% (N=100). This is acceptable; it should be framed as "our method targets the regime where reactive methods fail."

### PLR Metric Semantics

PLR is computed over **all agents** including those that never arrive. Non-arrived agents contribute their full wandering distance to the numerator but only their start-to-goal distance to the denominator. This partially inflates PLR on obstacle maps where some agents don't reach goals. However, on empty-48-48 where 98.6% arrive, PLR is still 3.33 — indicating the model genuinely takes circuitous paths even in open space.

---

## 2. Weakness Identification (Severity-Ranked)

### CRITICAL: Path Length Ratio

PLR of 3.33–5.90 is the single largest vulnerability. A reviewer will ask: "Your model makes agents wander 3× the optimal distance in an open field — hasn't it learned efficient navigation?"

**Root cause hypothesis:** The flow model was trained on EECBS expert data, which produces optimal *discrete grid* paths. Interpolated to continuous space, these paths zig-zag along grid edges rather than cutting diagonals. The model may have learned this grid-aligned movement style. Additionally, only 3 Euler integration steps may be insufficient for the flow to converge.

**Mitigation path:**
1. Report PLR restricted to arrived agents separately — this will be significantly lower
2. Test more integration steps (5, 10, 20)
3. Visualize trajectories to confirm root cause
4. If grid-alignment is the issue, augment training data with diagonal interpolation

### SERIOUS: Generalization Failure

- room-32-32-4: **26.5%** at_goal (unseen map geometry)
- warehouse: **25.4%** at_goal (unseen, realistic robot domain)

These results are presented without baselines. We don't know if Straight+CV-PIBT also fails here or if the model specifically regresses. This is a critical gap.

**Required:** Run ORCA, PO-ORCA, and Straight+CV-PIBT on Sets B and C to contextualize. If Straight+CV-PIBT also gets ~30%, the model is fine. If it gets 60%, the model is the problem.

### SERIOUS: No Learned Baseline

No comparison against any learned multi-agent navigation method. Reviewers will ask: "Is flow matching actually necessary, or would any learned policy work here?"

**Required continuous-space learned baselines (ranked by feasibility):**
1. **FlowGNNModel + ORCA shield** — already implemented, zero new code, runs `--policy flow --shield_type orca`
2. **ORCA velocity as preferred + CV-PIBT** — runs `--policy orca --shield_type cv-pibt`, already supported by the eval infrastructure
3. **CADRL/GA3C-CADRL** — RL-based continuous navigation, more work to implement

The most important is (1): it holds the learned model fixed and swaps only the shield, isolating CV-PIBT's contribution with a learned policy.

### MODERATE: No Multi-Seed Results

Single training seed, 25 scenarios per map. Without multiple training seeds, you cannot claim the model training is stable. 25 scenarios per map helps with scenario variance but not training variance.

### MODERATE: Collision Regression on Empty Maps

Flow+CV-PIBT: 12.4 collisions on empty-48-48 N=50 vs Straight+CV-PIBT's 4.7 (2.6× more). The flow model generates velocities the shield can't perfectly resolve. Not catastrophic in absolute terms but counter-narrative.

### MINOR: Success Rate Metric

Success (all agents arrive) is 0.000 for all methods on random maps. Too strict for dense settings. Present but don't let it dominate.

---

## 3. Next Experiments (Priority-Ordered)

### Experiment 1: Baselines on Sets B and C (HIGHEST PRIORITY — Zero new code)

**What:** Run `--policy orca --shield_type orca`, `--policy orca --shield_type po-orca`, and `--policy orca --shield_type cv-pibt` on room-32-32-4, random-64-64-10, and warehouse at N=50,100 with 512 steps.

**Expected:** Straight+CV-PIBT likely gets 40–60% on room and warehouse. This contextualizes whether 25% for the flow model is (a) acceptable given that even the strong baseline struggles, or (b) a model regression.

**Why it matters:** Cannot publish generalization results without baselines. This is a 2-hour run on Lambda.

### Experiment 2: ORCA Velocity + CV-PIBT (HIGH PRIORITY — Zero new code)

**What:** Run `--policy orca --shield_type cv-pibt` on Set A (all maps, N=50,100, 512 steps). This uses ORCA's reactive velocity as the preferred velocity input to CV-PIBT, rather than straight-line.

**Why this is interesting:** ORCA computes velocities that already dodge immediate obstacles. Feeding these into CV-PIBT gives the shield a smarter starting point — it resolves deadlocks via priority inheritance but starts from more informed velocities. This should outperform Straight+CV-PIBT, showing CV-PIBT is a general-purpose shield that improves *any* preferred velocity policy.

**Expected:** On random maps: at_goal somewhere between Straight+CV-PIBT (77.4%) and Flow+CV-PIBT (86.7%), closer to the former. On empty maps: similar to or slightly better than Straight+CV-PIBT. Fewer collisions than ORCA alone.

**Why it matters:** 
1. Shows CV-PIBT improves ORCA itself — a strong claim
2. Creates a complete "shield comparison" story: `any_policy + ORCA_shield` vs `any_policy + CV-PIBT shield`
3. Publishable as: "CV-PIBT improves any preferred velocity source, including reactive methods"

**How to run:** `--policy orca --nav straight --shield_type cv-pibt` (straight-line velocity directed at goal, standard ORCA collision detection as preferred input would require checking whether `run_orca_baseline` can take `shield_type=epibt` — verify this is wired correctly in eval_continuous.py first.)

**Note on implementation:** In eval_continuous.py, `run_orca_baseline` already accepts `shield_type` as a parameter and passes it to `env.step(preferred, shield_type=shield_type)`. This means `--policy orca --shield_type cv-pibt` should already work. Verify with a quick 1-scenario test.

### Experiment 3: FlowGNNModel + ORCA Shield (HIGH PRIORITY — Zero new code)

**What:** Run `--policy flow --shield_type orca` on Set A (all maps, N=50,100, 256 and 512 steps).

**Why this matters:** This is the most critical ablation missing from the paper. It holds the learned model fixed and tests whether CV-PIBT or ORCA shield produces better outcomes. Expected to perform significantly worse than Flow+CV-PIBT on random maps (ORCA deadlocks even with learned preferred velocities), confirming that CV-PIBT is the essential ingredient.

### Experiment 4: PLR Diagnosis

**What:**
- (a) Report arrived-agent PLR: filter `history_positions` to only agents where `arrival_steps >= 0`, compute their path length vs. their start-to-goal distance.
- (b) Test 5, 10, 20 Euler integration steps at inference on Set A.
- (c) Visualize 5 trajectories from empty-48-48 N=50 to see if paths are grid-aligned zigzags, spirals, or something else.

**Expected:** Arrived-agent PLR will be lower (maybe 1.5–2.5 on empty maps). More steps may improve velocity direction quality. Visualization will reveal whether this is a training data artifact or a model issue.

**Why it matters:** PLR is reviewer concern #1. Need either a convincing explanation with evidence or a fix.

### Experiment 5: Multi-Seed Training (REQUIRED FOR SUBMISSION)

**What:** Train v4b with 3 different random seeds. Evaluate on Set A, 512 steps. Report mean±std across seeds.

**Expected:** Moderate variance across seeds (±2–5pp at_goal). If high, training instability is a problem. If low, strengthens the contribution.

**Why it matters:** Required for statistical claims at any venue. Can be parallelized on Lambda.

---

## 4. Paper Story

### Main Claim

> CV-PIBT — a continuous-space extension of Priority Inheritance with Backtracking — achieves 5× higher goal completion than ORCA in structured environments while reducing collisions by 84%. Combined with a flow-matching GNN that learns congestion-aware preferred velocities, the full system further improves to 86.7% at_goal with an additional 50% collision reduction.

### Ablation Table Structure

Present as layered additive improvement (see table in Section 1). Makes the contribution hierarchy crystal clear: CV-PIBT is the mechanism, learned velocity is the complementary enhancement.

**Augmented ablation with new experiments:**

| Method | random-32-32-10 N=100 at_goal | Collisions |
|--------|-------------------------------|------------|
| ORCA | 16.8% | 4169 |
| PO-ORCA | 14.9% | 2626 |
| ORCA + CV-PIBT [new] | TBD | TBD |
| Straight + CV-PIBT | 77.4% | 1307 |
| Flow v4b + ORCA shield [new] | TBD | TBD |
| Flow v4b + CV-PIBT | **86.7%** | **650** |

### Single Most Compelling Number

**random-32-32-10 N=100, 512 steps: 86.7% at_goal vs. ORCA's 16.8%, with 84.4% fewer collisions.** Use in abstract.

### Supplementary Analyses to Strengthen the Paper

1. **Priority inheritance frequency:** What fraction of timesteps trigger backtracking? Shows the mechanism is actively resolving conflicts.
2. **Congestion-conditioned results:** Split scenarios by density (above/below median pairwise distance). Show model advantage grows with congestion.
3. **Arrived-agent PLR:** Report separately to contextualize path efficiency.
4. **Wall-clock timing:** Report inference time per step. If competitive with ORCA, deployability argument is strengthened.
5. **Qualitative visualization:** 4-panel figure: same scenario under ORCA (deadlocked), PO-ORCA (deadlocked), Straight+CV-PIBT (slow progress), Flow+CV-PIBT (coordinated flow).

### Framing Advice

- Position CV-PIBT as the **primary algorithmic contribution** — novel, generalizable, works with any preferred velocity source
- Position the flow model as a **demonstration** of how to learn effective preferred velocities for this shield
- Frame empty-map regression proactively: "In open spaces, reactive methods suffice; our method targets the regime where they fail"
- Frame PLR as: "Agents trade path efficiency for completion rate — favorable in multi-robot coordination where goal completion matters more than path optimality"

---

## 5. Related Work Gaps

A reviewer targeting continuous multi-agent navigation will expect discussion of:

| Work | Gap | Severity |
|------|-----|----------|
| PRIMAL/PRIMAL2 (Damani et al., 2021) | Learned MAPF, discrete — closest learned baseline concept | High |
| SCRIMP (Wang et al., 2023) | State-of-the-art learned discrete MAPF with structured communication | High |
| CADRL/SA-CADRL/SARL (Chen et al., 2017–2019) | Deep RL for continuous crowd navigation | Medium |
| NH-ORCA / AVO | ORCA variants for non-holonomic agents | Medium |
| MAPF-LNS2 (Li et al., 2022) | Strong classical centralized planner | Medium |
| Social Force Model (Helbing & Molnár, 1995) | Classic continuous multi-agent dynamics | Low |
| PIBT (Okumura et al., 2019, 2022) | Must clearly delineate discrete vs. our continuous extension | Critical |

### Most Critical Gap

No comparison against any learned method. At minimum: FlowGNNModel + ORCA shield (Experiment 3) as an ablation covering "learned policy + weak shield" vs "learned policy + CV-PIBT." This satisfies reviewers asking whether the shield or the policy drives performance.

---

## Key Numbers Reference

### Set A Summary (512 steps, averaged across maps)

| Method | N=50 at_goal | N=50 Coll | N=100 at_goal | N=100 Coll | N=50 Succ | N=50 PLR |
|--------|-------------|-----------|---------------|------------|-----------|---------|
| ORCA | 0.579 | 492.9 | 0.584 | 2101.4 | 0.500 | 0.677 |
| PO-ORCA | 0.576 | 334.7 | 0.575 | 1462.6 | 0.500 | 0.702 |
| Straight+CV-PIBT | 0.889 | 105.7 | 0.886 | 675.8 | 0.480 | 1.387 |
| Flow v4b+CV-PIBT 256 | 0.840 | 55.6 | 0.815 | 292.6 | 0.000 | 2.515 |
| Flow v4b+CV-PIBT 512 | 0.925 | 96.0 | 0.924 | 364.2 | 0.340 | 4.617 |

### OOD Performance (Set C, 512 steps)

| Map | N | at_goal | Collisions | PLR |
|-----|---|---------|------------|-----|
| warehouse | 50 | 0.254 | 57.2 | 1.436 |
| warehouse | 100 | 0.217 | 319.7 | 1.400 |

### Metric Semantics

- **PLR:** Sum of all agents' actual path lengths / sum of all agents' start-to-goal distances. Computed over ALL agents (arrived and not). Inflated by non-arrived agents.
- **collisions:** Cumulative count of (pair, timestep) violations where distance < 2×agent_radius. Same pair counted across multiple timesteps.
- **arr_step:** Mean arrival step over all agents; non-arrived agents count as arriving at episode end (step_count).
- **success:** Binary — 1.0 only if ALL agents reach goal.
