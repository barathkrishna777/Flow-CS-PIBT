# Flow-CS-PIBT Current State and SSIL Improvement Handoff

Date: 2026-05-03  
Branch: `barath/learned-wait-logit`  
Local Mac repo: `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT`  
Lambda repo: `/home/anushree_mattlab/barath/Flow-CS-PIBT` or `~/barath/Flow-CS-PIBT`  
Lambda env: `conda activate 3dposehsx_env`

## Executive Summary

Flow-CS-PIBT / FLOMAP is close to SSIL overall, but it does not currently beat SSIL on the headline evaluation. The best current FLOMAP variant is the learned wait-logit model evaluated with `--wait-mode learned --wait-logit-scale 1.0 --wait-logit-bias -5.0`.

The best learned wait model improves the original flow-vector baseline:

| Method | Overall | Coordination stress | Non-stress | Mean agents at goal | Stress agents at goal |
|---|---:|---:|---:|---:|---:|
| Baseline FLOMAP wave9 compact | `1195/1850 = 64.59%` | `146/350 = 41.71%` | `1049/1500 = 69.93%` | `91.93%` | `76.98%` |
| Learned wait-logit, `bias=-5` | `1222/1850 = 66.05%` | `161/350 = 46.00%` | `1061/1500 = 70.73%` | `93.83%` | `84.36%` |
| SSIL primary8 reference | `1229/1850 = 66.43%` | `181/350 = 51.71%` | approximately tied outside stress | `95.69%` | about `96.36%` |

The learned wait-logit model is only `7` successes behind SSIL overall on primary8, but it is still `20` successes behind on the coordination-stress slice. This means the remaining gap is concentrated in dense random and maze bottleneck regimes.

The latest planner-aligned wait ranking experiments did not help. Medium evaluation looked barely promising, but the 540-case result underperformed prior failed alternatives:

| Variant | Medium overall | Medium stress | 540-case overall | 540-case stress | Decision |
|---|---:|---:|---:|---:|---|
| Ranking loss, `bias=-5` | `70/110 = 63.64%` | `3/10 = 30.00%` | `301/540 = 55.74%` | `15/50 = 30.00%` | reject |
| Ranking loss, stress multiplier 3 | `61/110 = 55.45%` | `2/10 = 20.00%` | not run | not run | reject |
| Gate BCE previous run | `70/110 = 63.64%` | `3/10 = 30.00%` | `305/540 = 56.48%` | unknown here | not enough |
| Hybrid CE previous run | `73/110 = 66.36%` | `4/10 = 40.00%` | `308/540 = 57.04%` | unknown here | not enough |

Bottom line: the best model remains the learned wait-logit DDP checkpoint, not the ranking, gate, or hybrid refinements.

## Current Best Checkpoints and Evaluation Modes

Best full learned wait checkpoint:

```text
large_scale_flow_learned_wait_head_ddp_best.pt
```

Best known full evaluation mode:

```bash
--wait-mode learned \
--wait-logit-scale 1.0 \
--wait-logit-bias -5.0 \
--movement-logit-scale 1.0
```

Best known full primary8 result:

```text
1222/1850 = 66.05%
stress: 161/350 = 46.00%
non-stress: 1061/1500 = 70.73%
```

Original baseline anchor:

```text
evals/rishi_full_wave9_compact_best.csv
1195/1850 = 64.59%
stress: 146/350 = 41.71%
```

SSIL reference:

```text
primary8 overall: 1229/1850 = 66.43%
primary8 stress: 181/350 = 51.71%
```

BD-PIBT is also important. It is not a weak baseline:

```text
BD-PIBT primary8: 1229/1850 = 66.43%
BD-PIBT rishi12: 1606/2700 = 59.48%
SSIL rishi12: 1491/2700 = 55.22%
FLOMAP rishi12: 1433/2700 = 53.07%
```

Any paper claim must be careful because the heuristic BD-PIBT baseline is very strong.

## Where FLOMAP Beats or Matches SSIL

There is no clean headline "FLOMAP beats SSIL" result. However, FLOMAP is competitive and wins some map slices.

Primary8 per-map wins or near-wins from the report data:

| Map | FLOMAP | SSIL | Gap |
|---|---:|---:|---:|
| `den312d` | `35.60%` | `32.00%` | FLOMAP +3.60 pp |
| `random-64-64-10` | `86.40%` | `85.20%` | FLOMAP +1.20 pp |
| `warehouse-10-20-10-2-1` | `2.80%` | `2.40%` | FLOMAP +0.40 pp, but this is a floor-case map |

Extended12 per-map wins:

| Map | FLOMAP | SSIL | Gap |
|---|---:|---:|---:|
| `den312d` | `35.60%` | `32.00%` | FLOMAP +3.60 pp |
| `maze-32-32-4` | `21.33%` | `14.67%` | FLOMAP +6.66 pp |
| `random-64-64-10` | `86.40%` | `85.20%` | FLOMAP +1.20 pp |
| `warehouse-10-20-10-2-1` | `2.80%` | `2.40%` | FLOMAP +0.40 pp |
| `warehouse-20-40-10-2-1` | `14.80%` | `11.20%` | FLOMAP +3.60 pp |

Important slice result:

```text
Excluding coordination stress maps:
FLOMAP: 1049/1500 = 69.93%
SSIL:   1048/1500 = 69.87%

Excluding stress and floor-case maps:
FLOMAP: 1042/1250 = 83.36%
SSIL:   1042/1250 = 83.36%
```

This is the strongest honest story: FLOMAP is competitive with SSIL outside coordination-stress regimes. SSIL wins the headline because it is better on dense random and bottleneck maps.

## Main Challenge

The failure mode is concentrated in coordination-heavy settings.

Stress maps used in current analyses:

```python
{"random-32-32-10", "maze-128-128-2"}
```

These maps require wait behavior, short-horizon coordination, bottleneck negotiation, and sometimes recovery from local deadlocks. SSIL likely benefits because it is trained directly on the same five-way discrete action interface that CS-PIBT consumes:

```text
0 = wait
1 = right
2 = down
3 = up
4 = left
```

FLOMAP predicts continuous velocity. Inference must convert that velocity into a discrete action ranking for PIBT. The velocity-to-action interface can distort useful short-horizon behavior, especially wait decisions.

The learned wait-logit work confirms that wait handling matters, but the remaining gap is not solved by simply improving a binary wait head.

## Things Already Tried

### 1. Flow-vector baseline

The original model predicts velocity and converts it to action preferences by dot product with cardinal directions plus a fixed wait threshold. This is the strong baseline:

```text
1195/1850 = 64.59%
stress: 146/350 = 41.71%
```

Takeaway: continuous flow guidance is viable, but the action interface is a bottleneck.

### 2. Wait threshold sweeps

Changing the fixed wait threshold showed sensitivity but did not close the SSIL gap. Wait handling matters, but static thresholding is too blunt.

### 3. Learned wait-logit interface

Added a `wait_head` to `FlowGNNModel` and inference mode:

```text
learned: [wait, v dot right, v dot down, v dot up, v dot left]
```

Calibration flags:

```bash
--wait-logit-scale
--wait-logit-bias
--movement-logit-scale
```

Best result:

```text
large_scale_flow_learned_wait_head_ddp_best.pt
--wait-mode learned --wait-logit-scale 1.0 --wait-logit-bias -5.0
1222/1850 = 66.05%
stress: 161/350 = 46.00%
```

Takeaway: learned wait improves FLOMAP and is currently the best path so far, but it still trails SSIL on stress.

### 4. Binary gate BCE

Mode:

```text
learned_gate: P(wait) gates movement direction
```

Checkpoint:

```text
large_scale_flow_learned_wait_gate_bce_teacher_best.pt
```

Best medium:

```text
bias around -3 / -4
540-case rishi12 s5 at b=-3: 305/540 = 56.48%
```

Takeaway: not enough.

### 5. Hybrid action CE

Checkpoint:

```text
large_scale_flow_learned_wait_hybrid_teacher_best.pt
```

Observed behavior:

```text
Model calibration was poor.
Default-ish model calibration: 84/540 = 15.56%
Manual bias -5 with movement scale 1 helped medium: 73/110 = 66.36%
540-case: 308/540 = 57.04%
```

Takeaway: medium looked better than gate/ranking, but broad 540-case did not justify full-scale follow-up.

### 6. Planner-aligned wait ranking loss

Implemented CLI flags:

```bash
--wait-ranking-loss-weight
--wait-ranking-margin
--wait-ranking-velocity-source teacher_x1|predicted_x1
--stress-loss-multiplier
--stress-agent-threshold
```

Loss:

```text
If target is wait:
  softplus(max(movement_logits) + margin - calibrated_wait)

If target is move:
  softplus(calibrated_wait + margin - movement_logits[target_action - 1])
```

Results:

```text
Medium b=-5: 70/110 = 63.64%, stress 3/10 = 30.00%
540-case b=-5: 301/540 = 55.74%, stress 15/50 = 30.00%
Stress multiplier 3 medium: 61/110 = 55.45%, stress 2/10 = 20.00%
```

Takeaway: reject. It did not beat gate BCE or hybrid CE, and it is much worse than the best learned wait full result.

### 7. Action heads, hybrid heads, smaller hybrid capacity, custom coordination data

Earlier experiments tried action-head repair, hybrid action policies, smaller SSIL-like capacity, and custom coordination-heavy data. Some quick diagnostics improved, but medium/full results generally regressed or failed to robustly beat the flow-vector baseline. The existing report summarizes these as evidence that direct action supervision can affect target behavior, but naive action-head or hybrid replacements are not enough.

## Current Limitations

1. **No headline SSIL win.** Best learned wait result is `66.05%`, slightly below SSIL `66.43%`.
2. **Stress gap remains large.** Learned wait stress is `46.00%`; SSIL stress is `51.71%`.
3. **Manual calibration is required.** The best learned wait mode needs `--wait-logit-bias -5.0`, indicating calibration mismatch.
4. **540-case gates can be misleading.** Some medium runs looked promising but failed on 540-case evaluation.
5. **Runtime is worse than SSIL and BD-PIBT.** FLOMAP uses multiple integration steps and samples; SSIL emits logits in one classifier pass.
6. **BD-PIBT is very strong.** The non-learned heuristic baseline ties SSIL on primary8 and beats SSIL/FLOMAP on extended12.
7. **Local development is constrained.** Heavy training and evaluation cannot be run locally on the Mac. Local work should be limited to code changes, syntax checks, smoke tests, and small scripts.

## Environment and Workflow Constraints

Local Mac:

```bash
cd /Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT
```

Use local Mac only for:

```text
code inspection
syntax checks
unit/smoke tests
git diffs
small CSV analysis if files are local
```

Do not run heavy training or full eval locally.

Lambda:

```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
```

Use Lambda for:

```text
training
medium eval
540-case eval
full 1850/2700 eval
multi-GPU sharded eval
```

Preferred eval discipline:

1. Start with a small smoke only if code changed.
2. Run medium eval before 540-case.
3. Run 540-case before full 1850/2700.
4. Only run full eval if the smaller gates beat the best relevant references.
5. Always summarize overall, stress, non-stress, per-map, per-agent, and agents-at-goal.

## Useful Commands

Local syntax checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile \
  main_pys/generative_model.py \
  main_pys/train_flow.py \
  main_pys/simulator.py \
  eval_rishi_paper.py \
  eval_rishi_paper_parallel.py \
  test_wait_logit_interface.py

PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 test_wait_logit_interface.py
git diff --check
```

Medium eval template on Lambda:

```bash
python eval_rishi_paper_parallel.py \
  -m CHECKPOINT.pt \
  --map-set rishi12 \
  --max-scenario 2 \
  --agents 100 200 400 600 800 \
  --output evals/NAME_medium.csv \
  --wait-mode learned \
  --wait-logit-scale 1.0 \
  --wait-logit-bias -5.0 \
  --movement-logit-scale 1.0 \
  --gpus 0 1 2 3 \
  --jobs-per-gpu 1 \
  --resume-shards
```

540-case eval template:

```bash
python eval_rishi_paper_parallel.py \
  -m CHECKPOINT.pt \
  --map-set rishi12 \
  --max-scenario 5 \
  --agents 100 200 300 400 500 600 700 800 900 1000 \
  --output evals/NAME_rishi12_s5.csv \
  --wait-mode learned \
  --wait-logit-scale 1.0 \
  --wait-logit-bias -5.0 \
  --movement-logit-scale 1.0 \
  --gpus 0 1 2 3 \
  --jobs-per-gpu 1 \
  --resume-shards
```

Summary snippet:

```bash
python - <<'PY'
import pandas as pd

files = [
    "evals/CANDIDATE.csv",
]
stress_maps = {"random-32-32-10", "maze-128-128-2"}

for path in files:
    df = pd.read_csv(path)
    ok = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    stress = df["mapName"].isin(stress_maps)
    print(
        f"{path}: overall {ok.sum()}/{len(df)} = {100*ok.mean():.2f}% | "
        f"stress {ok[stress].sum()}/{stress.sum()} = {100*ok[stress].mean():.2f}% | "
        f"goal {100*(df['num_agents_at_goal']/df['agentNum']).mean():.2f}%"
    )
PY
```

## Possible Future Directions

These are ideas, not marching orders. Future work should be willing to ignore this list and invent better approaches.

### 1. Case-level paired analysis before more training

Analyze cases where SSIL succeeds and FLOMAP fails, especially on stress maps. For matched CSVs, compute:

- wait rate by map and agent count
- action disagreement patterns
- agents-at-goal when not successful
- runtime/time-limit failures
- whether failures are concentrated in specific scenarios/seeds
- differences by density bucket
- cases where FLOMAP succeeds and SSIL fails

This may reveal whether the remaining gap is mostly wait calibration, direction ranking, priority interaction, deadlock recovery, or map-specific topology.

### 2. Map/agent-conditioned inference calibration

The best learned wait model needs a global `bias=-5`. A global bias may be too crude. Try a schedule conditioned on:

- map family
- agent count
- current fraction at goal
- step count
- congestion estimate
- local obstacle density
- number of nearby occupied cells
- priority rank or priority age

This could be implemented as a lightweight inference-only policy before any new training.

### 3. Flow as residual over BD-PIBT rather than replacement

BD-PIBT is very strong. Instead of comparing flow and BD as separate policies, try hybrid action rankings:

```text
score(action) = alpha * BD_preference(action) + beta * flow_dot(action) + gamma * wait_score
```

or use flow only as a tie-breaker when BD has multiple plausible actions. This could preserve BD-PIBT's robust topology behavior while adding learned corrections.

### 4. PIBT-aware one-step or short-horizon action scoring

The model currently scores each agent's action preference locally, then PIBT resolves conflicts. Consider an inference wrapper that evaluates candidate rankings by a cheap one-step or few-step proxy:

- candidate action validity
- whether action worsens BD distance
- local collision pressure
- expected PIBT recursion conflicts
- bottleneck occupancy
- whether waiting unblocks higher-priority agents

This need not be differentiable at first. A heuristic wrapper might beat another learned head.

### 5. Learn an action value or reranker, not just wait

Train a model to score candidate actions after generating them from flow/BD. Targets could come from:

- expert action labels
- whether the action agrees with successful expert trajectory
- rollouts labeled by success/failure
- PIBT conflict outcomes
- hindsight labels from failed vs successful cases

The model could consume features for each candidate action instead of predicting one global velocity.

### 6. DAgger or failure-focused data collection

The model may fail because training data is only expert state distribution, while eval sees model-induced states and PIBT side effects. Try collecting data from model rollouts:

- run current best model
- identify failure states or near-deadlock states
- query expert/BD/SSIL-like target for those states
- fine-tune on those states with high weight

This is more likely to target the actual deployment distribution than more static supervised loss variations.

### 7. Better features for coordination

Potential features:

- current priority and priority age
- local BD gradient and distance-to-goal delta for each action
- congestion / occupancy in local patch
- at-goal mask
- bottleneck/corridor indicator
- local free-space degree
- time since last progress
- previous action or short memory
- predicted blockers from PIBT recursion

The current model may lack information that SSIL implicitly exploits through action-label training and local occupancy patterns.

### 8. Architecture changes

Candidates:

- graph transformer or attention over nearby agents
- larger field-of-view radius for stress maps
- explicit edge features in message passing
- recurrent state or temporal memory
- multi-head model: flow, action logits, wait, uncertainty, value
- uncertainty-aware ensemble or consensus over samples

Do not assume "bigger model" is the answer; first identify whether the failure is representational, objective-level, or planner-interface-level.

### 9. Mixture-of-experts or policy switching

Since different methods win different regimes, consider a gate that chooses among:

- learned wait FLOMAP
- baseline flow threshold FLOMAP
- BD-PIBT
- action-head/hybrid policy if useful on a narrow slice

The gate could start as an explicit map/agent heuristic and later become learned. This may produce a publishable system even if no single monolithic model dominates.

### 10. Publication framing

Current honest framing:

```text
FLOMAP shows that rectified-flow velocity guidance can be competitive with a discrete SSIL classifier when wrapped with CS-PIBT, and learned wait calibration nearly closes the overall primary8 gap. The remaining failures are concentrated in coordination-heavy regimes, revealing an interface-alignment bottleneck between continuous guidance and discrete PIBT action ranking.
```

Not currently supported:

```text
FLOMAP beats SSIL.
FLOMAP is state of the art.
Ranking loss solves wait behavior.
Hybrid/action-head training closes the gap.
```

NeurIPS/CoRL main would likely require at least one of:

- a clear SSIL win on a meaningful benchmark
- a principled interface fix that improves stress without hurting non-stress
- a new analysis or theory of continuous-to-discrete planner interfaces
- a robotics/continuous-control benchmark where velocity guidance has an advantage SSIL cannot match
- a robust hybrid with BD-PIBT or PIBT-aware reranking that beats both learned and heuristic baselines

## Recommended Next Decision

Stop spending full-eval time on ranking-loss wait refinement. The evidence is negative.

The highest-value next step is not another blind loss variant. It is paired failure analysis plus an intervention that targets the actual failure mode. The most promising families are:

1. inference-time mixture with BD/flow/wait calibration,
2. PIBT-aware candidate reranking,
3. failure-state DAgger,
4. map/agent-conditioned calibration,
5. action-value/reranker model trained on deployment-like states.

