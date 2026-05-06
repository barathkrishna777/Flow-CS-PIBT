# Final Experiment Runbook and Paper Outline

Current state: 2026-05-05. Branch: `barath/continuous_space_epibt`.

## Immediate Lambda Runbook

Training status:

- Seed 42 should use the existing continuous v4b checkpoint:
  `checkpoints/continuous_v4b/continuous_flow_v4b_best.pt`.
- Seeds 123 and 456 should be trained with `main_pys.train_continuous`,
  producing:
  `checkpoints/continuous_v4b/continuous_flow_v4b_seed123_best.pt` and
  `checkpoints/continuous_v4b/continuous_flow_v4b_seed456_best.pt`.
- Do not use `large_scale_flow_v4b_seed*_best.pt` for these continuous-space
  evals. Those are produced by `main_pys.train_flow` and use the older
  3-channel input format.

Before launching evals, run this preflight from the repo root on Lambda:

```bash
ls -lh checkpoints/continuous_v4b/continuous_flow_v4b_best.pt
ls -lh checkpoints/continuous_v4b/continuous_flow_v4b_seed123_best.pt
ls -lh checkpoints/continuous_v4b/continuous_flow_v4b_seed456_best.pt
```

Then run, in order:

```bash
bash scripts/run_agent3_evals.sh
bash scripts/run_v4b_setBC_evals.sh
```

Check the one known fragile artifact:

```bash
ls evals/baselines/straight_po-orca_512_setB.csv
```

If it is missing, rerun only that baseline:

```bash
python eval_parallel.py \
  --map-dir data/mapf-map \
  --scen-dir data/mapf-scen-random \
  --maps random-64-64-10 room-32-32-4 \
  --agent-counts 50 \
  --max-scenarios 25 \
  --policy orca \
  --nav straight \
  --shield-type po-orca \
  --max-steps 512 \
  --output-csv evals/baselines/straight_po-orca_512_setB.csv \
  --num-gpus 4
```

Finally:

```bash
python analyze_evals.py --markdown
```

If `python` is not available in the active environment, use `python3`.

## Expected Row Counts

Use these counts as a quick sanity check after each run:

| CSV family | Expected rows |
|---|---:|
| Set A per method or per seed: 2 maps x 2 agent counts x 25 scenarios | 100 |
| Set B per method: 2 maps x 1 agent count x 25 scenarios | 50 |
| Set C per method: 1 map x 2 agent counts x 25 scenarios | 50 |

`eval_parallel.py` now removes stale final CSVs, shard CSVs, and GPU logs for the requested output before launching. This prevents old pre-arrPLR shards from contaminating reruns.

## Result Story To Preserve

Primary Set A result on `random-32-32-10`, `N=100`, 512 steps:

| Method | AtGoal | Collisions | arrPLR |
|---|---:|---:|---:|
| ORCA | 16.8% | 4169 | - |
| PO-ORCA | 14.9% | 2626 | - |
| Straight + EPIBTShield | 77.4% | 1307 | 1.122 |
| Flow v4b + ORCA | 48.3% | about 0 | - |
| Flow v4b + EPIBTShield | 87.2% | 653 | 1.469 |

The clean decomposition:

- Shield matters: same flow policy with ORCA gets 48.3%, while EPIBTShield gets 87.2%.
- Learning helps: straight preferred velocities with EPIBTShield get 77.4%, while learned flow guidance reaches 87.2%.
- PO-ORCA is not enough: priority ordering alone does not solve the structured deadlock regime.
- arrPLR reframes efficiency: arrived agents take roughly 1.2x to 1.5x optimal paths, while all-agent PLR is inflated by agents that do not arrive.

Generalization:

- Set B `random-64-64-10`, `N=50`: ORCA 4.8% -> Straight+EPIBT 50.5% -> Flow+EPIBT 61.5%.
- Set B `room-32-32-4`, `N=50`: ORCA 1.2% -> Straight+EPIBT 11.4% -> Flow+EPIBT 26.5%.
- Set C warehouse, `N=50`: ORCA 13.2% -> Straight+EPIBT 41.1% -> Flow+EPIBT 25.4%.

Warehouse framing:

> EPIBTShield generalizes strongly out-of-domain, improving warehouse completion from 13.2% to 41.1% without learning. Learned flow guidance adds clear gains on in-distribution and near-distribution maps, but does not improve over the shield-only policy on fully OOD warehouse maps, which is expected because warehouse layouts are absent from training data.

## Paper Structure

Working title:

> Flow-Guided Priority-Inheritance Shielding for Continuous-Space Multi-Agent Path Finding

Recommended contribution order:

1. A continuous-space EPIBTShield that takes arbitrary preferred velocities and resolves conflicts with priority inheritance and backtracking.
2. A rectified-flow GNN policy that generates congestion-aware preferred velocities from EECBS expert demonstrations.
3. An empirical study showing that shielding and learning contribute separately: EPIBTShield resolves the deadlock regime, and flow guidance further improves goal completion and collision reduction.

Main sections:

1. Introduction
   State the continuous-space MAPF gap: ORCA-style reactive methods work in open spaces but fail in cluttered, dense, structured maps; learned policies need a reliable collision-resolution layer.

2. Problem Formulation
   Define continuous positions, radii, velocity bounds, timestep dynamics, goals, obstacles, completion metrics, and collision metrics.

3. Method
   Present the system as `preferred velocity policy -> EPIBTShield -> executed velocity`.

4. Learned Flow Policy
   Explain FlowGNNModel, rectified flow target, 3 Euler integration steps, EECBS supervision, and training maps.

5. EPIBTShield
   Describe priority inheritance, candidate velocity search, backtracking, and how the shield is policy-agnostic.

6. Experiments
   Organize around Set A primary, Set B generalization, Set C warehouse OOD, integration-step ablation, Flow+ORCA shield ablation, and multi-seed robustness.

7. Discussion and Limitations
   Be explicit that warehouse exposes a domain gap for the learned policy. Also avoid overclaiming "hard safety" unless the paper states the precise assumptions under which the shield guarantees collision-free motion, because empirical collision counts are nonzero.

## Tables and Figures

Minimum paper figures:

- Method diagram: FlowGNNModel preferred velocities feeding EPIBTShield.
- Main Set A table: ORCA, PO-ORCA, Straight+EPIBT, Flow+ORCA, Flow+EPIBT.
- Generalization bar chart: Set B and Set C by map.
- Integration-step ablation: 3, 5, 10, 20 Euler steps with AtGoal and arrPLR.
- Qualitative trajectory figure: same scenario under ORCA, Straight+EPIBT, Flow+EPIBT.
- Multi-seed table: seed 42, 123, 456 plus mean and std.

## Submission Gates

Do not freeze paper numbers until all are true:

- All three seed CSVs exist under `evals/v4b_seed*/flow_epibt_512_setA.csv`.
- Set B/C v4b CSVs were regenerated after `arrived_path_length_ratio` was added.
- `evals/baselines/straight_po-orca_512_setB.csv` exists.
- `python analyze_evals.py --markdown` produces a nonempty `docs/eval_results.md`.
- The markdown row counts match the expected row counts above.
- The final text distinguishes shield-only OOD generalization from learned-policy OOD behavior.
