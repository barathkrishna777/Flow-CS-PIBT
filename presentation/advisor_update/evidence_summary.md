# Evidence Summary

## Found in repo and used directly

- Template: `powerpoint-template.potx`
- Core grid-world implementation:
  - `main_pys/generative_model.py`
  - `main_pys/train_flow.py`
  - `main_pys/model_inputs.py`
  - `main_pys/simulator.py`
  - `main_pys/rishi_like_model.py`
  - `eval_rishi_paper.py`
- Repo-grounded related-work context:
  - `README.md`
  - `FLOW_VS_SSIL_ABLATION_PLAN.md`
  - `PAPER_PLAN.md`
- Headline grid-world result assets:
  - `evals/final_evals/slide_outputs/takeaways.md`
  - `evals/final_evals/slide_outputs/tables/per_agent_summary.md`
  - `evals/final_evals/slide_outputs/tables/per_map_summary.md`
  - `evals/final_evals/rishi12_ours_vs_rishi_1v1/rishi12_ours_vs_ssil_summary_2row_highres.png`
  - `evals/final_evals/topology_exposure/topology_exposure_slide_ready_highres.png`
  - `evals/final_evals/topology_exposure/topology_exposure_slide_ready_summary.md`
- Ongoing-work result assets:
  - `evals/continuous_evals/slide_outputs/figures/continuous_empty48_pilot_summary.png`
  - `evals/continuous_evals/slide_outputs/continuous_summary.md`
  - `evals/transformer/slide_outputs/full_comparison/figures/shared_case_overall_comparison_clean_labels.png`
  - `evals/transformer/slide_outputs/full_comparison/figures/shared_case_success_by_agents.png`
  - `evals/transformer/slide_outputs/transformer_summary.md`
- Rollout media:
  - `logs/slide_gifs/progress_smoke.gif`
  - `logs/slide_gifs/planner_near_miss_warehouse_100_agents.gif`
  - `logs/slide_gifs/planner_demo_100_agents.gif`
  - `logs/slide_gifs/planner_demo_100_agents_random64.gif`
  - `logs/slide_gifs/planner_near_miss_maze_25_agents.gif`

## Numbers pulled from repo outputs

- Clean held-out Flow-GNN success: `59.73%` (`1105 / 1850`)
- Topology-exposed reference: `62.27%`
- SSIL baseline: `66.43%` (`1229 / 1850`)
- Mean agents at goal for clean held-out Flow-GNN: `90.42%`
- Mean agents at goal within failed clean Flow-GNN runs: `76.22%`
- Mean agents at goal within failed SSIL runs: `87.17%`
- Runtime: Flow-GNN `32.68s` vs SSIL `18.96s`
- Topology exposure overall gain: `+0.32 pp`
- Transformer + action head on shared 8-map cases: `42.1%`

## Generated for this deck

- `presentation/advisor_update/generated_plots/grid_summary.json`
  - consolidated per-agent, per-map, failure-at-goal, and overall summary numbers for deck use
- `presentation/advisor_update/media/motivation_montage.png`
  - map montage for the motivation slide
- `presentation/advisor_update/media/behavior_success_strip.png`
- `presentation/advisor_update/media/behavior_failure_strip.png`
- `presentation/advisor_update/speaker_notes.md`
  - short speaker-style notes for each slide
- `presentation/advisor_update/outputs/flow_cs_pibt_advisor_update_2026-04-24.pptx`
  - final advisor deck built from the repo template

## What was intentionally not claimed

- No new quantitative claim beyond what could be traced to repo files or generated summaries from repo outputs
- No claim that Flow-GNN beats SSIL on the main held-out benchmark
- No claim that continuous-space or transformer results are paper-ready replacements for the grid-world headline
