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

## Completed Lambda Results

Timestamped output root:

```text
evals/lambda_sharded_pibt_20260430_103517
```

Combined row counts matched the expected protocol sizes:

| Eval | Rows | All-agents success | Mean agents at goal | Avg runtime |
|---|---:|---:|---:|---:|
| Rishi-8 `BD-PIBT` | 1850 | 66.43% (1229/1850) | 98.46% | 6.47s |
| Rishi-12 `BD-PIBT` | 2700 | 59.48% (1606/2700) | 98.57% | 6.00s |

Comparison against current Flow-CS-PIBT and SSIL CSVs:

| Eval | Method | Rows | All-agents success | Mean agents at goal | Avg runtime |
|---|---|---:|---:|---:|---:|
| Rishi-8 | `BD-PIBT` | 1850 | 66.43% | 98.46% | 6.47s |
| Rishi-8 | Flow-CS-PIBT flow-vector | 1850 | 64.59% | 91.93% | 31.34s |
| Rishi-8 | SSIL classifier | 1850 | 66.43% | 95.69% | 18.96s |
| Rishi-12 | `BD-PIBT` | 2700 | 59.48% | 98.57% | 6.00s |
| Rishi-12 | Flow-CS-PIBT flow-vector | 2700 | 53.07% | 90.56% | 32.16s |
| Rishi-12 | SSIL classifier | 2700 | 55.22% | 94.83% | 18.62s |

These results change the headline interpretation. Under this exact eval path,
the non-learned BD-guided CS-PIBT baseline is not weak: it ties SSIL on Rishi-8
all-agents success, beats both learned methods on Rishi-12 all-agents success,
has the highest mean agents-at-goal, and is substantially faster. Flow-CS-PIBT
does not demonstrate improvement over heuristic-only BD guidance in these
results.

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
> agent counts as the learned policies. In the completed full sweeps, this
> heuristic-only baseline is surprisingly strong: it reaches 66.43% all-agents
> success on Rishi-8, tying SSIL and exceeding the Flow-CS-PIBT flow-vector
> policy, and 59.48% on Rishi-12, exceeding both Flow-CS-PIBT and SSIL. It also
> obtains the highest mean agents-at-goal and the lowest runtime among the
> compared methods. These results show that the current learned flow guidance
> does not improve over BD-guided CS-PIBT under this protocol.
>
> We therefore frame Flow-CS-PIBT as an informative negative/diagnostic result
> rather than as a strict improvement over the classical baseline. The learned
> flow-vector policy improves over `BD-PIBT` on a few maps, notably
> `random-64-64-10`, but loses substantially on maps such as `den312d`,
> `empty-32-32`, `maze-32-32-4`, and `room-64-64-16`. The comparison suggests
> that shortest-path BD preferences plus CS-PIBT already encode a strong
> coordination prior, while the current flow policy may distort useful wait or
> local routing behavior. This interpretation is supported by the eval results
> but should not be claimed as a causal proof without further ablation.

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
