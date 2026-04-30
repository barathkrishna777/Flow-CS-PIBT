# PIBT Baseline and Fair Comparison Notes

Date: 2026-04-30

This note records the non-learned baseline added for the report and the rules
for comparing Flow-CS-PIBT against external learned MAPF solvers.

## Baseline Definition

Use `--policy-type pibt` for the pure non-learned baseline. This policy does
not load a neural checkpoint and writes `BD-PIBT` in the simulator `modelPath`
column. It uses backward-Dijkstra action preferences from `get_bd_prefs` with
the existing `add_noise=True` tie-breaking, then passes those preferences
through the same CS-PIBT shield, priority updates, time limit, max-step
multiplier, maps, scenarios, and agent counts as the learned policies.

For at-goal agents, the simulator forces wait to be the first preference. For
invalid moves into obstacles or map padding, the simulator moves invalid actions
to the end of the preference order before CS-PIBT is called. The BD grids are
already padded with high values, so obstacle and out-of-bounds moves should be
unattractive; the explicit preference cleanup is a robustness guard and mirrors
the invalid-action protection used for neural probabilities.

## Lambda Sync Options

Git workflow:

```bash
# local Mac
cd /Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT
git status --short
git add eval_rishi_paper.py main_pys/simulator.py docs/pibt_baseline_and_fair_comparisons.md
git commit -m "Add BD-guided PIBT eval baseline"
git push

# Lambda
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
git status --short
git pull --ff-only
```

Rsync/scp workflow:

```bash
# local Mac; replace <lambda-host>
cd /Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT
ssh <lambda-host> 'mkdir -p ~/barath/Flow-CS-PIBT/main_pys ~/barath/Flow-CS-PIBT/docs'
rsync -av eval_rishi_paper.py <lambda-host>:~/barath/Flow-CS-PIBT/eval_rishi_paper.py
rsync -av main_pys/simulator.py <lambda-host>:~/barath/Flow-CS-PIBT/main_pys/simulator.py
rsync -av docs/pibt_baseline_and_fair_comparisons.md <lambda-host>:~/barath/Flow-CS-PIBT/docs/pibt_baseline_and_fair_comparisons.md

# Lambda
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
python3 -m py_compile eval_rishi_paper.py main_pys/simulator.py
```

## Lambda Preflight

```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
git status --short
python eval_rishi_paper.py --help | grep -E "pibt|policy-type"

test -f data/all_maps.npz
test -d data/scen-random
test -d data/bd_npzs/large_scale
for m in Paris_1_256 empty-48-48 maze-128-128-2 random-64-64-10 random-32-32-10 warehouse-10-20-10-2-1 den312d den520d Berlin_1_256 empty-32-32 maze-32-32-4 random-64-64-20 warehouse-20-40-10-2-1 room-64-64-16; do
  ls "data/scen-random/${m}-random-1.scen" "data/bd_npzs/large_scale/${m}-random-1_bds.npz"
done
```

## Rishi-8 PIBT Shards

```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
OUT_ROOT=evals/lambda_sharded_pibt_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/rishi8" "$OUT_ROOT/rishi12"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python eval_rishi_paper.py --policy-type pibt --maps Paris_1_256 empty-48-48 --output "$OUT_ROOT/rishi8/rishi8_pibt_shard0.csv" > "$OUT_ROOT/logs/rishi8_pibt_shard0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_rishi_paper.py --policy-type pibt --maps maze-128-128-2 random-64-64-10 --output "$OUT_ROOT/rishi8/rishi8_pibt_shard1.csv" > "$OUT_ROOT/logs/rishi8_pibt_shard1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_rishi_paper.py --policy-type pibt --maps random-32-32-10 warehouse-10-20-10-2-1 --output "$OUT_ROOT/rishi8/rishi8_pibt_shard2.csv" > "$OUT_ROOT/logs/rishi8_pibt_shard2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_rishi_paper.py --policy-type pibt --maps den312d den520d --output "$OUT_ROOT/rishi8/rishi8_pibt_shard3.csv" > "$OUT_ROOT/logs/rishi8_pibt_shard3.log" 2>&1 &
wait
```

## Rishi-12 PIBT Shards

Run this in the same shell so `OUT_ROOT` is still set.

```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python eval_rishi_paper.py --policy-type pibt --map-set rishi12 --maps Berlin_1_256 empty-32-32 maze-32-32-4 --output "$OUT_ROOT/rishi12/rishi12_pibt_shard0.csv" > "$OUT_ROOT/logs/rishi12_pibt_shard0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_rishi_paper.py --policy-type pibt --map-set rishi12 --maps random-64-64-20 warehouse-20-40-10-2-1 room-64-64-16 --output "$OUT_ROOT/rishi12/rishi12_pibt_shard1.csv" > "$OUT_ROOT/logs/rishi12_pibt_shard1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_rishi_paper.py --policy-type pibt --map-set rishi12 --maps Paris_1_256 empty-48-48 maze-128-128-2 --output "$OUT_ROOT/rishi12/rishi12_pibt_shard2.csv" > "$OUT_ROOT/logs/rishi12_pibt_shard2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_rishi_paper.py --policy-type pibt --map-set rishi12 --maps random-64-64-10 warehouse-10-20-10-2-1 den312d --output "$OUT_ROOT/rishi12/rishi12_pibt_shard3.csv" > "$OUT_ROOT/logs/rishi12_pibt_shard3.log" 2>&1 &
wait
```

The `CUDA_VISIBLE_DEVICES` assignments are only for process bookkeeping. The
`pibt` policy forces `useGPU=False` internally.

## Monitor

```bash
nvidia-smi
pgrep -af "eval_rishi_paper.py|main_pys.simulator"
tail -f "$OUT_ROOT/logs/rishi8_pibt_shard0.log"
tail -f "$OUT_ROOT/logs/rishi12_pibt_shard0.log"

for f in "$OUT_ROOT"/rishi8/*.csv "$OUT_ROOT"/rishi12/*.csv; do
  [ -f "$f" ] && printf "%s rows=" "$f" && awk 'END{print NR > 0 ? NR - 1 : 0}' "$f"
done
```

Expected combined data rows: Rishi-8 has 1850 rows and Rishi-12 has 2700 rows.
Individual shard row counts can differ because not every map has every agent
count available.

## Combine and Summarize

```bash
python analysis_scripts/combine_simulator_csvs.py \
  "$OUT_ROOT"/rishi8/rishi8_pibt_shard*.csv \
  -o "$OUT_ROOT/rishi8/rishi8_pibt_combined.csv" \
  --expect-rows 1850

python analysis_scripts/combine_simulator_csvs.py \
  "$OUT_ROOT"/rishi12/rishi12_pibt_shard*.csv \
  -o "$OUT_ROOT/rishi12/rishi12_pibt_combined.csv" \
  --expect-rows 2700

python -m analysis_scripts.summarize_grid_eval \
  "$OUT_ROOT/rishi8/rishi8_pibt_combined.csv" \
  --labels pibt_rishi8

python -m analysis_scripts.summarize_grid_eval \
  "$OUT_ROOT/rishi12/rishi12_pibt_combined.csv" \
  --labels pibt_rishi12
```

## Compare Against Existing CSVs

```bash
python -m analysis_scripts.summarize_grid_eval \
  "$OUT_ROOT/rishi8/rishi8_pibt_combined.csv" \
  evals/rishi_full_wave9_compact_best.csv \
  evals/final_evals/rishi_full_ssil_classifier.csv \
  --labels pibt flow_vector ssil

if [ -f evals/lambda_sharded_rishi_lambda_20260429_191606/rishi12_wave9/rishi12_wave9_flow_combined.csv ]; then
  python -m analysis_scripts.summarize_grid_eval \
    "$OUT_ROOT/rishi12/rishi12_pibt_combined.csv" \
    evals/lambda_sharded_rishi_lambda_20260429_191606/rishi12_wave9/rishi12_wave9_flow_combined.csv \
    evals/final_evals/rishi12_ssil_classifier.csv \
    --labels pibt flow_vector ssil
else
  python -m analysis_scripts.summarize_grid_eval \
    "$OUT_ROOT/rishi12/rishi12_pibt_combined.csv" \
    evals/final_evals/rishi12_ssil_classifier.csv \
    --labels pibt ssil
fi
```

## Copy Results Back To Local Mac

```bash
# local Mac; replace <lambda-host> and the timestamped directory name
cd /Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT
rsync -av <lambda-host>:~/barath/Flow-CS-PIBT/evals/lambda_sharded_pibt_YYYYMMDD_HHMMSS/ evals/lambda_sharded_pibt_YYYYMMDD_HHMMSS/
```

## Report Framing

Pasteable text:

> We include `BD-PIBT` as a non-learned classical baseline. It uses
> backward-Dijkstra distance-to-goal preferences with random tie-breaking and
> the same CS-PIBT shield, timeout, max-step multiplier, scenarios, maps, and
> agent counts as the learned policies. This baseline isolates the value of the
> learned guidance: if Flow-CS-PIBT improves all-agents success over BD-PIBT,
> the result suggests that the learned velocity field provides useful guidance
> beyond shortest-path greedy preferences. At the same time, all-agents success
> and mean agents-at-goal should be interpreted separately. A heuristic policy
> may move most agents to goal while leaving one or a few agents stuck; those
> cases have high partial completion but still count as failures under the
> all-agents success metric.
>
> We treat SSIL as the strongest learned baseline evaluated on the same
> protocol. Flow-CS-PIBT does not beat SSIL overall in the current results, but
> is competitive with SSIL and should be compared against BD-PIBT to quantify
> improvement over heuristic-only guidance. The gap to SSIL appears
> concentrated in coordination-heavy maps. Because Flow-CS-PIBT predicts
> continuous velocities while SSIL predicts discrete actions, the SSIL action
> objective may better match wait and local coordination behavior; this is
> consistent with our ablations but should be framed as a hypothesis rather
> than a proven causal claim.

External ML MAPF baselines should not be mixed into the main quantitative
table unless they use the same or a clearly comparable benchmark, metric,
timeout, density range, and collision-shield setting. The WSNH paper and
project page emphasize that prior ML MAPF work often used random maps, lower
agent densities, and different shields; WSNH directly recommends including the
greedy-action PIBT baseline when using CS-PIBT. Therefore, keep the main
quantitative table limited to methods evaluated in this repo on the same
protocol: `BD-PIBT`, Flow-CS-PIBT, SSIL, and the hybrid/action-head ablations.
Discuss PRIMAL, DHC, MAGAT, SCRIMP, MAPF-GPT, SILLM, and related methods in
related work unless you can cite directly comparable numbers and label them as
"reported from prior work" rather than rerun results.
