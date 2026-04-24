## Audience

- Primary audience: research advisors
- Goal: deliver a 20-25 minute research update centered on the grid-world `barath/flow-gnn` work, with evidence-backed results, limitations, and next steps

## Story Arc

1. Why learned decentralized MAPF still matters
2. What prior CS-PIBT / SSIL-style work established
3. Where our flow-based formulation changes the interface
4. What was actually implemented in this repo
5. What the held-out evidence says
6. Why the remaining gap is informative
7. How the grid-world work motivates the continuous-space and transformer directions
8. Why lattice-based planning is the most promising return path for grid-world

## Slide List

1. Title: `Flow-CS-PIBT Research Update`
2. Motivation: large-agent coordination, planner cost, and why grid-world remains the right testbed
3. Related work: Rishi Veerapaneni's `Work Smarter Not Harder` versus our flow-based policy
4. Problem setup: MAPF state, action, objective, and shielding constraints
5. Method overview: expert trajectories to Flow-GNN to CS-PIBT / LaCAM action preferences
6. Training and inference details: flow loss, auxiliary action head, Euler steps, consensus, wait threshold
7. Experimental protocol: held-out maps, agent ladder, baselines, metrics, and clean-vs-exposed training regimes
8. Planner behavior: solved rollout and hard-case rollout
9. Headline results: overall success and scaling with agent count
10. Topology exposure: training on held-out topologies barely moves the headline number
11. Limitations: near-solved failures, deadlocks, and planner-interface mismatch
12. Ongoing work I: continuous-space pilot and why flow is a natural fit there
13. Ongoing work II: transformer/action-head exploration
14. Future work: lattice-based planning instead of cardinal actions
15. Takeaways / advisor discussion points
16. Backup: 12-map panel comparison
17. Backup: transformer detailed success-vs-agents comparison

## Source Plan

- Template: `powerpoint-template.potx`
- Primary repo narrative: `README.md`, `PAPER_PLAN.md`, `PROJECT_CONTEXT_HANDOFF.md`, `FLOW_VS_SSIL_ABLATION_PLAN.md`
- Core implementation:
  - `main_pys/generative_model.py`
  - `main_pys/train_flow.py`
  - `main_pys/model_inputs.py`
  - `main_pys/simulator.py`
  - `main_pys/rishi_like_model.py`
  - `eval_rishi_paper.py`
- Grid results and visuals:
  - `evals/final_evals/slide_outputs/*`
  - `evals/final_evals/topology_exposure/*`
  - `evals/final_evals/rishi12_*`
  - `evals/rishi_full_wave9_compact_best.csv`
  - `evals/wave8_heldout_full_eval.csv`
- Rollout media:
  - `logs/slide_gifs/progress_smoke.gif`
  - `logs/slide_gifs/planner_demo_100_agents.gif`
  - `logs/slide_gifs/planner_demo_100_agents_random64.gif`
  - `logs/slide_gifs/planner_near_miss_maze_25_agents.gif`
  - `logs/slide_gifs/planner_near_miss_warehouse_100_agents.gif`
- Ongoing work:
  - `evals/continuous_evals/slide_outputs/figures/continuous_empty48_pilot_summary.png`
  - `evals/transformer/slide_outputs/transformer_summary.md`
  - `evals/transformer/slide_outputs/full_comparison/figures/shared_case_overall_comparison_clean_labels.png`

## Visual System

- Use the Carnegie Mellon template palette: CMU red, tartan accent, dark blue, gold, white
- Keep content slides clean and editorial: large title, one focal visual, short evidence callouts
- Favor split layouts: chart/image on one side, 2-3 short speaker-oriented lines on the other
- Use native PowerPoint text, tables, and charts for labels/numbers

## Generated Assets

- `generated_plots/grid_summary.json`: compact repo-backed summaries for slide charts
- `media/motivation_montage.png`: cropped rollout/map montage for the motivation slide
- `media/behavior_success_strip.png`: solved rollout storyboard
- `media/behavior_failure_strip.png`: hard-case rollout storyboard

## Editability Plan

- All slide titles, captions, notes, callouts, and comparison tables remain editable PowerPoint objects
- Data-backed charts for headline results and limitations are authored as native chart objects
- Existing repo figures are used as images where they already encode multi-panel evidence cleanly
- Speaker notes are written into slide notes and mirrored in a separate markdown outline
