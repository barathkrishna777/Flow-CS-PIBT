# Continuous MAPF Implementation Details

## Overview

This file summarizes the current continuous-space MAPF implementation in this repository, the intended paper framing, the scripts currently used for data generation / training / evaluation, and the status of the latest open-space benchmark run.

The continuous stack is intentionally separate from the original grid-world simulator and paper comparison code. The core continuous comparison is:

- ORCA
- continuous flow policy
- discretized continuous-action baseline

The continuous baseline is **not** the original grid-trained discrete model forced into continuous execution.

## Current Paper Plan

The current paper plan separates the project into two method-comparison stories:

- Grid section:
  - original discrete grid policy vs grid flow
- Continuous section:
  - ORCA vs continuous flow vs discretized continuous-action baseline

### Main continuous claims intended for the paper

- The primary continuous benchmark is currently `empty-48-48`.
- The primary reported metric is `agent_fraction_at_goal`.
- `success` is treated as a stricter secondary metric.
- The current obstacle-map story is **not** paper-ready yet.

### Current continuous benchmark defaults

- Flow inference:
  - `num_integration_steps = 3`
  - `num_consensus_samples = 3`
- Discrete baseline:
  - 8-direction continuous-action classifier
- Shield:
  - ORCA
- Benchmark agent counts:
  - `32`, `64`, `96`, `128`
- Phase A dataset plan:
  - multiple `empty-48-48` scenarios
  - 3 training seeds for each learned method
  - 30 epochs per seed

## Continuous Implementation

### 1. Environment and dynamics

File:

- `main_pys/continuous_env.py`

This file implements:

- continuous agent positions and velocities on top of the grid maps
- circular-agent collision checking
- obstacle intersection checks using grid obstacles as continuous occupied geometry
- path metrics such as:
  - `success`
  - `agents_at_goal`
  - `agent_fraction_at_goal`
  - `path_length`
  - `path_length_ratio`
  - `smoothness`
  - `collisions`
  - `near_collisions`
  - `obstacle_hits`
  - `mean_arrival_step`

### 2. Safety shield

Also in `main_pys/continuous_env.py`:

- `ORCAStyleShield`

This supports:

- true ORCA through `rvo2` when available
- heuristic ORCA-style fallback when `rvo2` is unavailable or fails

The shield:

- clips speeds to `max_speed`
- projects preferred velocities to safer ones
- handles agent-agent interactions
- handles obstacle constraints

### 3. Continuous supervision generation

File:

- `scripts/generate_continuous_data.py`

This script builds training rollouts from:

- `EECBS-flow` path plans when available
- ORCA fallback when EECBS is unavailable or replay validation fails

Current data generation flow:

1. Parse `.scen` starts/goals.
2. Convert grid cell centers to continuous positions via `+0.5`.
3. Run EECBS to obtain discrete paths.
4. Densify those paths into continuous positions / velocities using `dt` and `max_speed`.
5. Validate replay in the continuous environment.
6. If needed, fall back to ORCA rollout.
7. Save compressed `.npz` files containing:
   - `map_name`
   - `scenario_name`
   - `positions`
   - `velocities`
   - `goals`
   - `dt`
   - `expert_source`
   - `action_labels`

### 4. Continuous dataset construction

File:

- `main_pys/dataset_continuous.py`

This dataset:

- loads continuous rollout `.npz` files
- treats each rollout timestep as one training example
- builds continuous graph inputs via `create_continuous_data_object`
- computes per-node weights to rebalance moving vs waiting actions
- supports a weighted sampler across agent-count buckets

### 5. Learned policies

File:

- `main_pys/train_continuous.py`

Two policy types are supported:

- `flow`
- `discrete`

Both use `FlowGNNModel`, with different loss formulations.

#### Flow policy training

- Uses a flow-matching objective on continuous velocity vectors.
- Samples noise `x_0` and interpolation time `t`.
- Predicts the flow field from noisy/intermediate states.
- Includes an auxiliary action classification loss.

#### Discrete baseline training

- Uses only action classification loss.
- Predicts an 8-direction plus wait action label.

#### Current training features

- reproducible seeding via `--seed`
- deterministic worker seeding
- optional weighted sampling
- checkpoint output directory via `--output-dir`
- saved checkpoint metadata includes:
  - model config
  - dataset config
  - policy type
  - epoch
  - seed
  - train / validation loss

### 6. Evaluation

File:

- `scripts/eval_continuous.py`

This script evaluates:

- `orca`
- `flow`
- `discrete`

For flow inference, evaluation uses:

- Euler integration with `num_integration_steps`
- multi-sample averaging with `num_consensus_samples`

For the discrete baseline, evaluation:

- predicts logits
- applies temperature `tau`
- picks the argmax action
- converts labels to continuous direction vectors

The evaluation CSV now records:

- map
- scenario
- agents
- policy
- shield type
- run name
- model name / path
- train seed
- eval seed
- integration steps
- consensus samples
- tau
- all continuous metrics listed above

It can also save trajectory visualizations.

## Benchmark Orchestration

File:

- `scripts/run_continuous_benchmark.py`

This script was added to automate the Phase A open-space benchmark workflow.

### Supported stages

- `generate`
- `train`
- `eval`
- `summarize`

### What it does

- generates shared continuous training data
- trains flow and discrete policies across multiple seeds
- evaluates ORCA and learned policies on the same benchmark set
- optionally runs flow consensus sweeps for `1/3/5`
- optionally generates visualization cases
- writes summary CSVs
- writes scaling and runtime plots

### Main outputs

- data:
  - `data/continuous_phase_a/`
- checkpoints:
  - `checkpoints/continuous_phase_a/`
- eval CSVs:
  - `benchmarks/continuous_phase_a/evals/main/`
  - `benchmarks/continuous_phase_a/evals/sweeps/`
- summaries:
  - `benchmarks/continuous_phase_a/summaries/`
- plots:
  - `benchmarks/continuous_phase_a/summaries/plots/`
- visualization assets:
  - `benchmarks/continuous_phase_a/viz/`

## Scripts Currently Being Used

### Data generation

```bash
python -m scripts.generate_continuous_data \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 32 64 96 128 \
  --output-dir data/continuous_phase_a \
  --expert-source hybrid \
  --max-scenarios 5
```

### Direct training

```bash
python -m main_pys.train_continuous \
  --data-dir data/continuous_phase_a \
  --map-dir data/mapf-map \
  --policy-type flow \
  --run-name phaseA_empty48_s0 \
  --output-dir checkpoints/continuous_phase_a \
  --epochs 30 \
  --seed 0
```

```bash
python -m main_pys.train_continuous \
  --data-dir data/continuous_phase_a \
  --map-dir data/mapf-map \
  --policy-type discrete \
  --run-name phaseA_empty48_s0 \
  --output-dir checkpoints/continuous_phase_a \
  --epochs 30 \
  --seed 0
```

### Direct evaluation

```bash
python -m scripts.eval_continuous \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 32 64 96 128 \
  --policy flow \
  --model-path checkpoints/continuous_phase_a/continuous_flow_phaseA_empty48_s0_best.pt \
  --num-integration-steps 3 \
  --num-consensus-samples 3 \
  --shield-type orca \
  --output-csv benchmarks/continuous_phase_a/evals/main/flow_seed0.csv
```

### Full Phase A runner

```bash
python -m scripts.run_continuous_benchmark \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 32 64 96 128 \
  --max-scenarios 5 \
  --epochs 30 \
  --seeds 0 1 2 \
  --enable-consensus-sweep \
  --make-viz
```

## Current Benchmark Run Status

The latest run shared in the terminal log completed:

- data generation
- 3 flow training runs
- 3 discrete baseline training runs
- ORCA evaluation
- flow evaluation for seeds `0/1/2`
- flow consensus sweeps for `c=1` and `c=5`
- discrete baseline evaluation for seeds `0/1/2`
- visualization runs for ORCA and flow

### Dataset actually generated in the latest run

Generated files include:

- `empty-48-48-random-1`
- `empty-48-48-random-10`
- `empty-48-48-random-11`
- `empty-48-48-random-12`
- `empty-48-48-random-13`

This is an important caveat:

- the intended Phase A plan was to use scenarios `1-5`
- the current script sorts filenames lexicographically, so `--max-scenarios 5` picked `1,10,11,12,13`
- this means the current benchmark is valid as a multi-scenario open-space run, but it is **not yet the exact intended `1-5` protocol**

## Latest Training Snapshot

### Flow validation losses

Approximate best validation losses from the latest run:

- seed 0:
  - best val about `0.1843`
- seed 1:
  - best val about `0.1807`
- seed 2:
  - best val about `0.1819`

### Discrete validation losses

Approximate best validation losses from the latest run:

- seed 0:
  - best val about `0.1906`
- seed 1:
  - best val about `0.1857`
- seed 2:
  - best val about `0.1844`

Interpretation:

- both learned methods train stably
- the discrete baseline reaches slightly lower classification validation loss
- despite that, the flow policy is much stronger on the actual open-space benchmark metric of interest

## Latest Evaluation Summary

### ORCA

Observed behavior on `empty-48-48`:

- very strong `agent_fraction_at_goal`
- often near-perfect completion
- `success` is high at low density, but no longer always perfect for `64/96/128`

From the logged runs:

- `32` agents:
  - consistently `1.000`
- `64` agents:
  - often `0.969-1.000`
- `96` agents:
  - often `0.969-0.990`
- `128` agents:
  - often `0.977-0.992`

### Flow policy

Observed behavior across seeds with `steps=3`, `consensus=3`:

- `32` agents:
  - usually around `0.875-0.969`
- `64` agents:
  - usually around `0.844-0.938`
- `96` agents:
  - usually around `0.854-0.927`
- `128` agents:
  - usually around `0.836-0.914`

Key qualitative result:

- flow is clearly below ORCA
- flow is clearly above the discrete baseline
- `success` remains `0` in the logged flow runs, so `agent_fraction_at_goal` remains the right primary metric

### Discrete continuous-action baseline

Observed behavior across seeds:

- `32` agents:
  - roughly `0.500-0.812`
- `64` agents:
  - roughly `0.594-0.750`
- `96` agents:
  - roughly `0.604-0.781`
- `128` agents:
  - roughly `0.617-0.711`

Key qualitative result:

- the discrete continuous-action baseline is materially weaker than flow on every tested density

### Consensus sweep behavior

The flow sweeps for `c=1`, `c=3`, and `c=5` show:

- only modest differences in `agent_fraction_at_goal`
- no obvious evidence from the pasted logs that `c=1` beats `c=3`
- no strong upside from `c=5`

Current interpretation:

- `consensus = 3` still looks like a reasonable default
- final judgment should come from the summary CSVs and runtime tradeoff plots

## What Is Ready vs Not Ready

### Ready enough to claim

- On open-space `empty-48-48`, continuous flow is better than the discretized continuous-action baseline.
- ORCA remains stronger than the learned flow policy.
- `agent_fraction_at_goal` is the most informative continuous metric right now.

### Not ready yet

- Obstacle-map continuous claims.
- Any comparison that treats the original grid-trained discrete model as the continuous baseline.
- Using `success` alone as the headline continuous metric.
- Declaring the final Phase A protocol frozen until the scenario-selection issue is fixed.

## Recommended Immediate Next Steps

1. Fix scenario selection so `--max-scenarios 5` uses `random-1` through `random-5` numerically.
2. Re-run the Phase A benchmark on the intended scenario set.
3. Inspect the generated summary CSVs and plots under `benchmarks/continuous_phase_a/summaries/`.
4. Freeze `steps=3`, likely keep `consensus=3` unless the summary shows a clear reason to change it.
5. Keep obstacle-map work as a separate engineering milestone until ORCA obstacle handling is trustworthy.

## Relevant Files

- `docs/FLOW_ADVANCED_PLAN.md`
- `README.md`
- `docs/implementation_details.md`
- `main_pys/continuous_env.py`
- `main_pys/dataset_continuous.py`
- `scripts/generate_continuous_data.py`
- `main_pys/train_continuous.py`
- `scripts/eval_continuous.py`
- `scripts/run_continuous_benchmark.py`
