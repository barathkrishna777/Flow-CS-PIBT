# Flow-CS-PIBT

Flow-CS-PIBT studies learned local policies for multi-agent path finding (MAPF)
inside the CS-PIBT collision shield. The main grid-world model is a continuous
flow matching policy: instead of predicting one discrete action directly, it
generates a continuous velocity vector and converts that vector into action
preferences for CS-PIBT or LaCAM-style planning.

The repository also keeps compatibility with Rishi Veerapaneni's SSIL
classifier model, so the same simulator can evaluate both the flow model and
the classifier baseline on the paper's held-out maps.

## What Is Here

- `generate_flow_data_multi.py`: generate grid-world expert trajectories with
  EECBS and backward-Dijkstra (BD) heuristic files.
- `preprocess_dataset.py`: optionally convert raw trajectory `.npz` files into
  ready-to-load PyTorch Geometric `.pt` files.
- `main_pys/train_flow.py`: train the flow matching policy.
- `train_full.py`: extract zipped assets and launch a standard full training
  run.
- `main_pys/simulator.py`: run a trained flow or classifier policy inside the
  grid-world CS-PIBT simulator.
- `eval_full.py`: smaller ablation/evaluation sweep over maps, agents, and
  flow integration steps.
- `eval_rishi_paper.py`: Rishi-paper held-out grid-world benchmark.
- `analysis_scripts/summarize_grid_eval.py`: summarize simulator CSVs into
  readable tables.

## Setup

Clone the repository and create the conda environment:

```sh
git clone https://github.com/barathkrishna777/Flow-CS-PIBT.git
cd Flow-CS-PIBT

conda config --set channel_priority flexible
conda env create -f environment.yml
conda activate mlmapf
```

The environment uses Python 3.11, PyTorch, PyTorch Geometric, NumPy, SciPy, and
the other dependencies needed for training and evaluation.

For downloading the public Google Drive assets:

```sh
python -m pip install gdown
```

## Download Standard Assets

For most evaluation and training workflows, start with the maps, random
scenarios, BD heuristics, `all_maps.npz`, and the SSIL classifier checkpoint:

```sh
bash download_assets.bash
```

This creates the usual `data/` layout. Depending on the asset source, scenarios
may land in either `data/scen-random` or `data/mapf-scen-random`; the newer eval
scripts check both.

The Rishi held-out BD bundle can also be downloaded directly:

```sh
mkdir -p data/constant_npzs
gdown --folder "https://drive.google.com/drive/folders/1S3md2fHR2cahc_yoeJxNeKl_gti-JU0h?usp=drive_link" \
  -O data/constant_npzs \
  --continue
```

Expected important files include:

```text
data/all_maps.npz
data/mapf-map/*.map
data/scen-random/*.scen
data/constant_npzs/*_bds.npz
data/model/ssil_model.pt
```

If `data/all_maps.npz` is missing but `data/constant_npzs/all_maps.npz` exists,
either copy it into place or pass the path explicitly to lower-level simulator
commands.

If scenarios were downloaded as `data/mapf-scen-random` and you want to run the
raw data generator, create the expected alias:

```sh
ln -s mapf-scen-random data/scen-random
```

## Dataset Options

You can train from either raw trajectory `.npz` files or preprocessed `.pt`
files.

Raw data is easier to inspect and regenerate. Preprocessed data is much faster
for training because graph construction, BD lookup, normalization, and target
creation are done once ahead of time.

### Option A: Use A Zipped Dataset

If you already have the large trajectory zip and base data zip:

```sh
python train_full.py \
  --trajectories /path/to/massive_flow_dataset_large_scale.zip \
  --base-data /path/to/data.zip \
  --run-name my_run
```

`train_full.py` extracts the data and launches `main_pys.train_flow` with the
standard large model configuration.

If the data is already extracted:

```sh
python train_full.py \
  --trajectories /path/to/massive_flow_dataset_large_scale.zip \
  --base-data /path/to/data.zip \
  --skip-extract \
  --run-name my_run
```

### Option B: Generate Raw Grid-World Data Yourself

First build or place the EECBS binary at:

```text
build/eecbs
```

Then make sure these directories exist:

```text
data/mapf-map
data/scen-random
```

If your random scenarios are in `data/mapf-scen-random`, use the symlink command
from the asset section above before generating trajectories.

Generate BD heuristics and EECBS expert trajectories:

```sh
python generate_flow_data_multi.py
```

Outputs:

```text
data/bd_npzs/large_scale/*_bds.npz
data/flow_training_data_multi/*.npz
```

By default, `generate_flow_data_multi.py` follows the Rishi split: it generates
BD files for held-out test maps, but does not generate training trajectories
from those held-out maps.

### Option C: Preprocess Raw Data

Preprocess raw trajectory `.npz` files into PyG `.pt` samples:

```sh
python preprocess_dataset.py \
  --data-dir data/flow_training_data_multi \
  --map-dir data/mapf-map \
  --out data/preprocessed \
  --workers 32
```

To exclude specific maps:

```sh
python preprocess_dataset.py \
  --data-dir data/flow_training_data_multi \
  --map-dir data/mapf-map \
  --out data/preprocessed \
  --workers 32 \
  --exclude-maps den312d empty-48-48
```

## Training

### Train From Preprocessed Data

This is the recommended route for serious runs:

```sh
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed \
  --run-name my_flow \
  --hidden-dim 1024 \
  --num-layers 6
```

The best checkpoint is saved as:

```text
large_scale_flow_my_flow_best.pt
```

Epoch checkpoints are saved as:

```text
large_scale_flow_my_flow_epoch_1.pt
large_scale_flow_my_flow_epoch_2.pt
...
```

### Train From Raw `.npz` Data

If no preprocessed directory is supplied, training falls back to
`data/flow_training_data_multi` and constructs graphs on the fly:

```sh
python -m main_pys.train_flow \
  --run-name raw_flow \
  --hidden-dim 1024 \
  --num-layers 6
```

This is slower, but useful when testing new data quickly.

### Quick Training Smoke Test

```sh
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed \
  --run-name smoke \
  --quick \
  --no-wandb
```

### Resume Training

```sh
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed \
  --run-name my_flow \
  --resume large_scale_flow_my_flow_epoch_3.pt
```

## Grid-World Evaluation

### Single Simulator Run

```sh
python -m main_pys.simulator \
  --mapNpzFile=data/all_maps.npz \
  --mapName=empty-48-48 \
  --scenFile=data/scen-random/empty-48-48-random-1.scen \
  --bdNpzFile=data/constant_npzs/empty-48-48_bds.npz \
  --modelPath=large_scale_flow_my_flow_best.pt \
  --outputCSVFile=evals/single_empty48.csv \
  --maxSteps=3x \
  --seed=0 \
  --useGPU=True \
  --agentNum=100 \
  --shieldType=CS-PIBT \
  --policyType=flow
```

For Rishi's classifier checkpoint, use:

```sh
python -m main_pys.simulator \
  --mapNpzFile=data/all_maps.npz \
  --mapName=empty-48-48 \
  --scenFile=data/scen-random/empty-48-48-random-1.scen \
  --bdNpzFile=data/constant_npzs/empty-48-48_bds.npz \
  --modelPath=data/model/ssil_model.pt \
  --outputCSVFile=evals/single_empty48_ssil.csv \
  --maxSteps=3x \
  --seed=0 \
  --useGPU=True \
  --agentNum=100 \
  --shieldType=CS-PIBT \
  --policyType=classifier
```

You can replace `--shieldType=CS-PIBT` with `LaCAM` or `Real-Time-LaCAM` when
testing other planners.

### Smaller Flow Ablation Sweep

`eval_full.py` sweeps maps, agent counts, and flow integration steps:

```sh
python eval_full.py large_scale_flow_my_flow_best.pt \
  --output evals/flow_ablation.csv
```

Useful variants:

```sh
python eval_full.py large_scale_flow_my_flow_best.pt \
  --maps empty-48-48 random-32-32-10 den312d \
  --agents 100 400 800 \
  --steps 1 2 3 5 \
  --output evals/flow_steps_sweep.csv
```

```sh
python eval_full.py large_scale_flow_my_flow_best.pt \
  --extended \
  --output evals/flow_extended.csv
```

### Rishi Paper Held-Out Benchmark

`eval_rishi_paper.py` supports named map presets:

- `rishi8` is the default strict held-out protocol.
- `rishi12` is the 12-map panel from Rishi's reported evals, useful for
  checking whether familiar map topology helps Flow.

The default `rishi8` held-out test maps are:

```text
Paris_1_256
empty-48-48
maze-128-128-2
random-64-64-10
random-32-32-10
warehouse-10-20-10-2-1
den312d
den520d
```

The `rishi12` panel maps are:

```text
Berlin_1_256
empty-32-32
maze-32-32-4
random-64-64-20
warehouse-20-40-10-2-1
room-64-64-16
Paris_1_256
empty-48-48
maze-128-128-2
random-64-64-10
warehouse-10-20-10-2-1
den312d
```

`rishi12` is not a pure held-out-topology test if any of those maps appeared in
training trajectories. Use it as a topology-familiarity comparison against the
strict `rishi8` run.

A quick smoke test uses one scenario and agent counts `100, 400, 800`:

```sh
python eval_rishi_paper.py \
  -m large_scale_flow_my_flow_best.pt \
  -o evals/smoke_flow.csv \
  --quick \
  --policy-type flow
```

```sh
python eval_rishi_paper.py \
  -m data/model/ssil_model.pt \
  -o evals/smoke_ssil_classifier.csv \
  --quick \
  --policy-type classifier
```

The full benchmark uses scenarios `random-1` through `random-25` and agent
counts `100, 200, ..., 1000`:

```sh
mkdir -p logs evals

CUDA_VISIBLE_DEVICES=0 nohup python eval_rishi_paper.py \
  -m large_scale_flow_my_flow_best.pt \
  -o evals/rishi_full_flow.csv \
  --map-set rishi8 \
  --policy-type flow \
  > logs/rishi_full_flow.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python eval_rishi_paper.py \
  -m data/model/ssil_model.pt \
  -o evals/rishi_full_ssil_classifier.csv \
  --map-set rishi8 \
  --policy-type classifier \
  > logs/rishi_full_ssil_classifier.log 2>&1 &
```

For the 12-map topology comparison on Lambda:

```sh
mkdir -p logs evals

CUDA_VISIBLE_DEVICES=0 nohup python eval_rishi_paper.py \
  -m large_scale_flow_my_flow_best.pt \
  -o evals/rishi12_full_flow.csv \
  --map-set rishi12 \
  --policy-type flow \
  > logs/rishi12_full_flow.log 2>&1 &
```

Monitor:

```sh
tail -f logs/rishi_full_flow.log
tail -f logs/rishi_full_ssil_classifier.log
```

Summarize:

```sh
python -m analysis_scripts.summarize_grid_eval \
  evals/rishi_full_flow.csv \
  evals/rishi_full_ssil_classifier.csv \
  --labels flow ssil_classifier
```

The summarizer prints per-run, per-map, and overall success tables.

## Visualization

Visualize saved grid paths:

```sh
python -m main_pys.visualize_path \
  empty-48-48 \
  logs/paths.npy \
  --scenName=empty-48-48-random-1.scen
```

## Troubleshooting

### Missing BD Files

If evaluation prints missing BD warnings, first check where the BD files landed:

```sh
find data -name "*_bds.npz" | head
```

The newer Rishi eval script searches common locations including:

```text
data/bd_npzs/large_scale
data/constant_npzs
data/constant_npzs/bd_npzs
data/bd_npzs
```

If you are using a lower-level simulator command, pass the exact BD path with
`--bdNpzFile`.

### Scenario Directory Names

MovingAI's random scenarios may be named `data/scen-random` or
`data/mapf-scen-random`. The Rishi eval script checks both. For manual
simulator commands, pass the exact `.scen` path.

### WandB

Training logs to Weights & Biases by default. Disable it with:

```sh
python -m main_pys.train_flow --no-wandb
```

### CPU vs GPU

Training and large evals are designed for a CUDA GPU. Small smoke tests can run
on CPU, but full Rishi-paper evals should be run on a GPU machine.

## Project Structure

```text
Flow-CS-PIBT/
├── main_pys/
│   ├── generative_model.py
│   ├── train_flow.py
│   ├── dataset.py
│   ├── dataset_preprocessed.py
│   ├── simulator.py
│   ├── model.py
│   └── model_inputs.py
├── analysis_scripts/
│   └── summarize_grid_eval.py
├── generate_flow_data_multi.py
├── preprocess_dataset.py
├── train_full.py
├── eval_full.py
├── eval_rishi_paper.py
└── download_assets.bash
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
