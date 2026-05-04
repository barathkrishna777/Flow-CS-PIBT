# FLOMAP Research Report

This folder contains a standalone IEEE-style LaTeX research report for the
FLOMAP experiments.

## Contents

- `main.tex`: main report.
- `references.bib`: BibTeX references.
- `figures/`: generated grouped metric panels and TikZ figure snippets.
- `tables/`: generated CSV and LaTeX summary tables.
- `scripts/`: reproducible figure/table generation scripts.

## Generate Tables and Figures

Run from the repository root:

```sh
python3 research_report/scripts/summarize_eval_slices.py
mkdir -p /tmp/flow_cs_pibt_mplconfig /tmp/flow_cs_pibt_xdg_cache
env MPLCONFIGDIR=/tmp/flow_cs_pibt_mplconfig \
  XDG_CACHE_HOME=/tmp/flow_cs_pibt_xdg_cache \
  .venv-plot/bin/python research_report/scripts/plot_density_panels.py \
  --include-12-map
python3 research_report/scripts/plot_coordination_slices.py
python3 research_report/scripts/plot_ablation_summary.py
```

The summary scripts prefer local CSVs.  If an expected remote CSV is missing,
they use the numeric fallback summaries recorded in the task prompt and mark the
source as `remote summary` in `tables/overall_summary.csv`.  The density-panel
plot script needs row-level CSVs, so it skips 12-map density plots until the
remote 12-map flow CSV is copied locally.

The PNG density-panel script uses matplotlib and emits
`figures/primary8_grouped_metric_panel.png` as the paper-facing 8-map metric
plate.  It uses local map preview images when present and schematic thumbnails
only when no preview asset is available.  The local repo has matplotlib in
`.venv-plot`; system `python3` may not.

To require CSV-backed generation only:

```sh
python3 research_report/scripts/summarize_eval_slices.py --no-fallback
```

## Compile

```sh
cd research_report
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```sh
cd research_report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Remote CSV TODO

The following combined remote CSVs were not present in the local Mac repo when
this report was generated:

```text
evals/lambda_sharded_rishi_lambda_20260429_191606/rishi12_wave9/rishi12_wave9_flow_combined.csv
evals/lambda_sharded_rishi_lambda_20260429_191606/rishi8_hybrid/rishi8_hybrid_action_head_combined.csv
```

Copy them into the local repo with:

```sh
mkdir -p evals/lambda_sharded_rishi_lambda_20260429_191606
rsync -av <lambda-host>:/home/anushree_mattlab/barath/Flow-CS-PIBT/evals/lambda_sharded_rishi_lambda_20260429_191606/ evals/lambda_sharded_rishi_lambda_20260429_191606/
```

If the CSVs cannot be copied locally, run the same scripts on the remote
evaluation machine from
`~/barath/Flow-CS-PIBT`, then copy back:

```sh
rsync -av <lambda-host>:/home/anushree_mattlab/barath/Flow-CS-PIBT/research_report/figures/ research_report/figures/
rsync -av <lambda-host>:/home/anushree_mattlab/barath/Flow-CS-PIBT/research_report/tables/ research_report/tables/
```

## Source Status

Generated source status is written to:

```text
tables/csv_source_status.csv
tables/missing_lambda_csvs.txt
```
