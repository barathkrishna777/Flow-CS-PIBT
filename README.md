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
- `analysis_scripts/compare_grid_1v1.py`: generate a Markdown report and PNG
  figure comparing two simulator CSVs.
- `generate_grid_visualizations.py`: mine successful rows from an eval CSV,
  rerun those cases with path logging, and render polished GIFs.
- `demo_planner_gif.py`: run a self-contained CS-PIBT-style planner demo and
  render a GIF without downloaded assets or a trained checkpoint.
- `main_pys/visualize_path.py`: render a saved simulator path `.npy` as an
  animated grid-world GIF.
- `generate_and_preprocess_continuous.py`, `main_pys/train_continuous.py`, and
  `eval_continuous.py`: separate continuous-space experiments.

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

## Presentation Demo GIF

If you need a quick planner visualization without downloading the full map,
scenario, BD, and checkpoint assets, run:

```sh
python demo_planner_gif.py
```

This uses the `Berlin_1_256.map` benchmark map automatically when it exists at
`data/mapf-map/Berlin_1_256.map` and the random scenario at
`data/mapf-scen-random/Berlin_1_256-random-1.scen`. By default, it solves the
first 300 agents from that scenario with a PIBT-style collision shield using
backward-Dijkstra preferences, opens a Matplotlib window showing the final
paths, prints planner statistics such as agents-at-goal, runtime, arrival
steps, path length, wait fraction, and collision counts, and writes an animated
GIF of the agents moving:

```text
logs/planner_demo.gif
logs/planner_demo_paths.npy
logs/planner_demo.log
```

For the fastest presentation run, precompute the Berlin backward-Dijkstra cache
once:

```sh
python demo_planner_gif.py --no-show --precompute-only
```

The default cache path is:

```text
data/demo-cache/Berlin_1_256-random-1_N300_bd_distances.npz
```

Useful variants:

```sh
python demo_planner_gif.py --no-show
python demo_planner_gif.py --quiet
python demo_planner_gif.py --progress-interval 100 --render-progress-interval 20
python demo_planner_gif.py --rebuild-cache
python demo_planner_gif.py --no-cache
python demo_planner_gif.py --scenario crossing --agents 8 --output logs/crossing_demo.gif
python demo_planner_gif.py --agents 12 --frame-stride 1 --duration-ms 70
python demo_planner_gif.py --map-file data/mapf-map/Berlin_1_256.map --scen-file data/mapf-scen-random/Berlin_1_256-random-1.scen
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

## Continuous MAPF

The continuous-space stack is separate from the grid simulator.

- `main_pys/continuous_env.py` implements continuous dynamics, obstacle checks,
  and an ORCA-style local safety shield.
- `generate_and_preprocess_continuous.py` creates a continuous-space dataset
  root and manifest.
- `generate_continuous_data.py` converts EECBS-flow plans into continuous
  trajectories and can fall back to an ORCA-style expert.
- `main_pys/train_continuous.py` trains either a continuous flow model or an
  8-direction discrete baseline.
- `eval_continuous.py` evaluates ORCA, continuous flow, or the discrete
  baseline and can save trajectory plots.

Generate a continuous dataset:

```sh
python generate_and_preprocess_continuous.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 16 32 64 96 128 160 \
  --dataset-root data/continuous_main \
  --expert-source hybrid
```

Train a continuous flow policy:

```sh
python -m main_pys.train_continuous \
  --data-dir data/continuous_main/raw \
  --map-dir data/mapf-map \
  --policy-type flow \
  --run-name conti_flow_v1 \
  --output-dir checkpoints/continuous \
  --seed 0
```

Train the continuous discrete baseline:

```sh
python -m main_pys.train_continuous \
  --data-dir data/continuous_main/raw \
  --map-dir data/mapf-map \
  --policy-type discrete \
  --run-name conti_disc_v1 \
  --output-dir checkpoints/continuous \
  --seed 0
```

Evaluate a learned continuous policy:

```sh
python eval_continuous.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 random-32-32-10 \
  --agent-counts 100 200 \
  --policy flow \
  --model-path checkpoints/continuous/continuous_flow_conti_flow_v1_best.pt \
  --run-name conti_flow_v1 \
  --train-seed 0 \
  --output-csv evals/continuous_flow_eval.csv \
  --viz-dir logs/continuous_viz
```

Run the packaged benchmark:

```sh
python run_continuous_benchmark.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 32 64 96 128 \
  --max-scenarios 5 \
  --data-dir data/continuous_phase_a \
  --checkpoint-dir checkpoints/continuous_phase_a \
  --benchmark-dir benchmarks/continuous_phase_a \
  --run-prefix phaseA_empty48 \
  --epochs 30 \
  --seeds 0 1 2 \
  --enable-consensus-sweep \
  --make-viz
```

To evaluate with the external `picbf-cs` CBF shield, install the
`continuous-collision-shield` package or point `PICBF_CS_PATH` at that repo:

```sh
PICBF_CS_PATH=/path/to/picbf-cs python eval_continuous.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 32 64 \
  --policy orca \
  --shield-type picbf-cs \
  --output-csv evals/continuous_picbf_cs_orca.csv
```

## Visualization

There are three useful visualization paths:

- turn one or more evaluation CSVs into text summaries;
- compare two evaluation CSVs with a Markdown report and PNG figure;
- render solved grid-world cases from a CSV into animated GIFs.

The CSV-based tools expect simulator-format CSVs, such as files produced by
`eval_full.py`, `eval_rishi_paper.py`, `main_pys.simulator`, or the checked-in
examples under `evals/`. The most important columns are `mapName`, `scenFile`,
`agentNum`, `success`, `num_agents_at_goal`, `runtime`, and a cost column such
as `total_cost_true` or `total_cost_not_resting_at_goal`.

### Visualization Setup

Use the project conda environment first:

```sh
conda activate mlmapf
```

For GIF rendering and comparison figures, make sure these packages are
available:

```sh
python -m pip install matplotlib pillow tqdm pandas
```

For animated grid visualizations, your friend also needs the same map/scenario
assets used to make the CSV:

```text
data/all_maps.npz
data/mapf-map/*.map
data/scen-random/*.scen or data/mapf-scen-random/*.scen
data/bd_npzs/large_scale/*_bds.npz or data/constant_npzs/*_bds.npz
```

If the CSV was produced on another machine, pass the local checkpoint with
`--model-path` when generating GIFs.

### Summarize Eval CSVs

Print overall and per-map success tables:

```sh
python -m analysis_scripts.summarize_grid_eval \
  evals/rishi_full_wave9_compact_best.csv \
  evals/rishi_quick_wave9_best.csv \
  --labels flow_full flow_quick
```

Compare older batch-style CSVs by average at-goal percentage:

```sh
python compare_results.py \
  --csv evals/wave8_best_full_eval.csv evals/wave8_heldout_full_eval.csv
```

### Compare Two CSVs With A Figure

Create a Markdown report plus a multi-panel PNG figure:

```sh
python -m analysis_scripts.compare_grid_1v1 \
  evals/rishi_full_wave9_compact_best.csv \
  evals/rishi_quick_wave9_best.csv \
  --labels "Flow full" "Flow quick" \
  --out-dir visualizations/grid_1v1
```

Outputs:

```text
visualizations/grid_1v1/flow_full_vs_flow_quick.md
visualizations/grid_1v1/flow_full_vs_flow_quick.png
```

Useful option:

```sh
python -m analysis_scripts.compare_grid_1v1 \
  evals/a.csv evals/b.csv \
  --labels "Run A" "Run B" \
  --cost-column total_cost_true
```

### Make Showcase GIFs From A CSV

`generate_grid_visualizations.py` is the easiest way to make nice GIFs from
existing CSV results. It selects successful rows from the CSV, reruns those
cases while saving paths, then renders the paths as GIFs.

Make three polished showcase GIFs at roughly 50, 200, and 1000 agents:

```sh
python generate_grid_visualizations.py \
  --eval-csv evals/rishi_full_wave9_compact_best.csv \
  --model-path large_scale_flow_my_flow_best.pt \
  --output-dir visualizations/showcase \
  --showcase \
  --use-gpu true
```

Outputs are organized as:

```text
visualizations/showcase/paths/*.npy
visualizations/showcase/metrics/*.csv
visualizations/showcase/gifs/*.gif
```

Pick exact agent counts yourself:

```sh
python generate_grid_visualizations.py \
  --eval-csv evals/rishi_full_wave9_compact_best.csv \
  --model-path large_scale_flow_my_flow_best.pt \
  --output-dir visualizations/custom \
  --agent-counts 100 400 800 \
  --distinct-maps \
  --agent-count-selection fastest \
  --map-preferences empty den random \
  --soft-style \
  --frame-stride 8 \
  --trail-length 32 \
  --figure-size 10 \
  --dpi 180 \
  --use-gpu true
```

Pick the hardest successful cases in a CSV:

```sh
python generate_grid_visualizations.py \
  --eval-csv evals/rishi_full_wave9_compact_best.csv \
  --model-path large_scale_flow_my_flow_best.pt \
  --output-dir visualizations/hard_cases \
  --top-k 5 \
  --min-agents 400 \
  --soft-style \
  --use-gpu true
```

Preview what would run without launching simulator reruns:

```sh
python generate_grid_visualizations.py \
  --eval-csv evals/rishi_full_wave9_compact_best.csv \
  --model-path large_scale_flow_my_flow_best.pt \
  --showcase \
  --dry-run
```

Reuse already-generated `.npy` path files and only rerender GIFs:

```sh
python generate_grid_visualizations.py \
  --eval-csv evals/rishi_full_wave9_compact_best.csv \
  --model-path large_scale_flow_my_flow_best.pt \
  --output-dir visualizations/showcase \
  --showcase \
  --reuse-paths
```

Common GIF style flags:

```text
--frame-stride N             render every Nth timestep; larger is faster/smaller
--trail-length N             number of previous positions drawn behind agents
--agent-size FLOAT           marker size for agents
--goal-size FLOAT            marker size for goal stars
--trail-width FLOAT          path trail width
--figure-size FLOAT          Matplotlib figure size in inches
--dpi INT                    output resolution
--frame-duration-ms INT      GIF frame duration
--end-frame-duration-ms INT  pause on the final frame
--soft-style                 use the softer presentation palette
```

### Render One Saved Path File

If you already have a saved simulator path file, render it directly:

```sh
python -m main_pys.visualize_path \
  empty-48-48 \
  visualizations/showcase/paths/empty-48-48_empty-48-48-random-1_N100_seed0.npy \
  --scenName=empty-48-48-random-1.scen \
  --mapFolder=data/mapf-map \
  --sceneFile=data/scen-random \
  --outputGif=visualizations/empty48_N100.gif \
  --softStyle \
  --frameStride=6 \
  --trailLength=30 \
  --figureSize=9 \
  --dpi=160
```

To create a path file from one simulator run, include `--outputPathsFile`:

```sh
python -m main_pys.simulator \
  --mapNpzFile=data/all_maps.npz \
  --mapName=empty-48-48 \
  --scenFile=data/scen-random/empty-48-48-random-1.scen \
  --bdNpzFile=data/constant_npzs/empty-48-48_bds.npz \
  --modelPath=large_scale_flow_my_flow_best.pt \
  --outputCSVFile=evals/single_empty48.csv \
  --outputPathsFile=logs/empty48_paths.npy \
  --maxSteps=3x \
  --seed=0 \
  --useGPU=True \
  --agentNum=100 \
  --shieldType=CS-PIBT \
  --policyType=flow
```

Then render:

```sh
python -m main_pys.visualize_path \
  empty-48-48 \
  logs/empty48_paths.npy \
  --scenName=empty-48-48-random-1.scen \
  --sceneFile=data/scen-random \
  --outputGif=logs/empty48_paths.gif
```

### Continuous Trajectory Plots

Continuous evals can save trajectory PNGs directly with `--viz-dir`:

```sh
python eval_continuous.py \
  --map-dir data/mapf-map \
  --scen-dir data/scen-random \
  --maps empty-48-48 \
  --agent-counts 100 \
  --policy orca \
  --output-csv evals/continuous_orca_viz.csv \
  --viz-dir visualizations/continuous_orca
```

For learned policies, add `--model-path`, `--run-name`, and the policy-specific
flow options exactly as in the continuous evaluation section above.

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
│   ├── compare_grid_1v1.py
│   └── summarize_grid_eval.py
├── generate_flow_data_multi.py
├── generate_grid_visualizations.py
├── preprocess_dataset.py
├── train_full.py
├── eval_full.py
├── eval_rishi_paper.py
├── eval_continuous.py
├── run_continuous_benchmark.py
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
