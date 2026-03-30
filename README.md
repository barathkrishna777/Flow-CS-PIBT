# Flow-CS-PIBT: Continuous Flow Matching for Multi-Agent Path Finding

This project extends Veerapaneni et al.'s CS-PIBT framework by replacing the discrete classification policy with a **Continuous Flow Matching (Rectified Flow)** generative model. Instead of predicting action labels directly, the model learns to generate continuous velocity vectors via an ODE flow field, which are then mapped to discrete actions for the CS-PIBT collision shield.

## Approach

This work is inspired by the following papers:
1. [Improving Learnt Local MAPF Policies with Heuristic Search (ICAPS 2024)](https://arxiv.org/abs/2403.20300)
2. [Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF (ICRA 2025)](https://arthurjakobsson.github.io/ssil_mapf/)
3. [Real-Time LaCAM (SoCS 2025)](https://arxiv.org/abs/2504.06091)

### Pipeline

1. **Data Generation**: 1.2M expert trajectory graphs generated using EECBS (suboptimality factor 2.0). Discrete paths are converted to continuous expert velocities using a Savitzky-Golay filter.

2. **Model** (`main_pys/generative_model.py`): A PyTorch Geometric GNN (6-layer SAGEConv, 1024-dim hidden) with a CNN encoder for local map context and Backward Dijkstra (BD) heuristic features.

3. **Training** (`main_pys/train_flow.py`): Rectified Flow training — the model predicts the vector field `x_1 - x_0` where `x_1` is the expert velocity and `x_0` is Gaussian noise. Uses per-graph time sampling, goal-weighting, gradient clipping, and mixed-precision (AMP).

4. **Inference** (`main_pys/simulator.py`): The flow is integrated over 5 Euler steps (dt=0.2) to produce a velocity vector. A magnitude-based wait detection mechanism identifies near-stationary agents. Action probabilities are computed via dot-product with cardinal direction vectors followed by temperature-scaled softmax, then passed to CS-PIBT/LaCAM.

### Key Fixes and Improvements

- **Wait Action Fix**: The dot product of any velocity with `[0,0]` is always 0, so the "wait" action was never prioritized. A magnitude threshold (`||v|| < 0.25`) now correctly routes near-stationary agents to wait, improving agent completion rates from ~40% to ~94%.
- **Per-Graph Time Sampling**: Flow matching time `t` is sampled once per graph (not per node), matching the inference-time integration where all nodes in a graph share the same `t`.
- **Removed Collision Loss**: The repulsive collision loss operated on the flow field rather than the final velocity, which is mathematically incorrect for flow matching and hurt convergence.
- **Mixed Precision Training**: AMP with FP16 for ~2x throughput on GPU (A100/RTX 4060 compatible).

## Installation

```sh
git clone https://github.com/barathkrishna777/Flow-CS-PIBT.git
cd Flow-CS-PIBT
git checkout barath_gnn
```

Install dependencies:
```sh
conda config --set channel_priority flexible
conda env create -f environment.yml
conda activate mlmapf
```

## Data Setup

Place the following zip files in an accessible location:
- `data.zip` — maps, BD heuristics, scenario files
- `massive_flow_dataset_large_scale.zip` — expert trajectory data (1.2M samples)

Then use the training script which handles extraction automatically:
```sh
python train_full.py --trajectories /path/to/massive_flow_dataset_large_scale.zip --base-data /path/to/data.zip
```

## Training

### Full Training (GPU recommended)
```sh
python train_full.py --trajectories /path/to/trajectories.zip --base-data /path/to/data.zip
```
This extracts data and launches `main_pys.train_flow` with the correct configuration. On an A100, training uses batch size 128, 8 workers, and AMP.

### Overfit Sanity Check (local GPU)
To verify the model can memorize a small dataset:
```sh
python -m analysis_scripts.train_overfit_big
python -m analysis_scripts.eval_overfit_big
```

## Evaluation

### Batch Evaluation
```sh
python run_experiments.py
```
Runs the model across multiple maps (empty-48-48, random-32-32-10, den312d) and agent densities (50, 100, 200), outputting results to `logs/batch_results.csv`.

### Single Scenario
```sh
python -m main_pys.simulator --mapNpzFile=data/all_maps.npz \
      --mapName=empty-48-48 --scenFile=data/scen-random/empty-48-48-random-1.scen \
      --bdNpzFile=data/bd_npzs/large_scale/empty-48-48-random-1_bds.npz \
      --modelPath=large_scale_flow_epoch_1.pt \
      --outputCSVFile=logs/results.csv \
      --maxSteps=3x --seed=0 --useGPU=True \
      --agentNum=50 --shieldType=CS-PIBT
```

Replace `--shieldType=CS-PIBT` with `LaCAM` or `Real-Time-LaCAM` for other collision shields.

## Continuous MAPF (Phase 3)

The continuous stack is intentionally separate from the grid simulator:

- `main_pys/continuous_env.py` implements continuous dynamics, obstacle checks, and an ORCA-style local safety shield.
- `generate_continuous_data.py` converts `EECBS-flow` plans into continuous trajectories and can fall back to an ORCA-style expert.
- `main_pys/train_continuous.py` trains either a continuous flow model or an 8-direction discrete baseline on continuous `.npz` rollouts.
- `eval_continuous.py` evaluates ORCA, continuous flow, or the discrete baseline and can save trajectory plots.

Example commands:

```sh
# Generate continuous supervision from EECBS-flow with ORCA fallback
python3 generate_continuous_data.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 random-32-32-10 \
  --agent-counts 100 200 \
  --output-dir data/continuous \
  --expert-source hybrid

# Train the continuous flow model
python3 -m main_pys.train_continuous \
  --data-dir data/continuous \
  --map-dir data/mapf-map \
  --policy-type flow \
  --run-name conti_flow_v1

# Train the continuous discrete baseline
python3 -m main_pys.train_continuous \
  --data-dir data/continuous \
  --map-dir data/mapf-map \
  --policy-type discrete \
  --run-name conti_disc_v1

# Evaluate a learned continuous policy
python3 eval_continuous.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 random-32-32-10 \
  --agent-counts 100 200 \
  --policy flow \
  --model-path continuous_flow_conti_flow_v1_best.pt \
  --output-csv evals/continuous_flow_eval.csv \
  --viz-dir logs/continuous_viz
```

### Visualization
```sh
python -m main_pys.visualize_path empty-48-48 logs/paths.npy --scenName=empty-48-48-random-1.scen
```

## Project Structure

```
Flow-CS-PIBT/
├── main_pys/
│   ├── generative_model.py   # Flow GNN model (6-layer SAGEConv + CNN)
│   ├── train_flow.py         # Rectified Flow training loop
│   ├── dataset.py            # FlowMAPFDataset with BD heuristics
│   ├── simulator.py          # Inference + CS-PIBT/LaCAM integration
│   ├── model.py              # Original SSIL model (for reference)
│   └── model_inputs.py       # Graph construction and normalization
├── analysis_scripts/
│   ├── train_overfit_big.py   # Overfit test for big model
│   ├── eval_overfit_big.py    # Evaluate overfit model
│   └── diagnose_action_mapping.py  # Action mapping analysis
├── train_full.py              # End-to-end training script
├── run_experiments.py         # Batch evaluation
└── data/                      # Maps, scenarios, BDs, trajectories
```

## Citation

If you use this repository, please cite the original works:

```bibtex
@article{veerapaneni2024improving_mapf_policies_with_search,
  title = {Improving Learnt Local MAPF Policies with Heuristic Search},
  volume = {34},
  url = {https://ojs.aaai.org/index.php/ICAPS/article/view/31522},
  doi = {10.1609/icaps.v34i1.31522},
  number = {1},
  journal = {International Conference on Automated Planning and Scheduling (ICAPS)},
  author = {Veerapaneni, Rishi and Wang, Qian and Ren, Kevin and Jakobsson, Arthur and Li, Jiaoyang and Likhachev, Maxim},
  year = {2024},
  pages = {597-606},
}

@inproceedings{veerapaneni2025work_smart_not_harder,
  author = {Veerapaneni, Rishi and Jakobsson, Arthur and Ren, Kevin and Kim, Samuel and Li, Jiaoyang and Likhachev, Maxim},
  booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title = {Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large-Scale Imitation Learning for MAPF},
  year = {2025},
  pages = {10229-10236},
  doi = {10.1109/ICRA55743.2025.11128836},
}

@inproceedings{liang2025real_time_lacam,
  title = {Real-Time LaCAM for Real-Time MAPF},
  author = {Liang, Runzhe and Veerapaneni, Rishi and Harabor, Daniel and Li, Jiaoyang and Likhachev, Maxim},
  booktitle = {Proceedings of the International Symposium on Combinatorial Search (SoCS)},
  volume = {18},
  pages = {196-200},
  year = {2025},
  doi = {10.1609/socs.v18i1.35993},
  url = {https://ojs.aaai.org/index.php/SOCS/article/view/35993},
}
```
