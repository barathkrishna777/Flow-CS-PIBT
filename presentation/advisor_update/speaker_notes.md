# Speaker Notes

## Slide 1: Flow-CS-PIBT Research Update

- Frame this as a research update, not a final paper pitch.
- The talk centers on the grid-world branch results, then uses those results to motivate the continuous-space and transformer directions.
- The headline tension is simple: flow is promising, but the discrete planner interface is still the bottleneck.

Sources:
- powerpoint-template.potx
- README.md
- PAPER_PLAN.md

## Slide 2: Why learned local MAPF still matters

- Open by separating two failure sources: finding a locally sensible direction versus resolving conflicts under a decentralized shield.
- Grid-world is still the right place to debug that interface because the state, action set, and benchmark protocol are controlled.
- That is why the rest of the deck uses the held-out Rishi benchmark as the main reference point.

Sources:
- README.md
- PROJECT_CONTEXT_HANDOFF.md
- logs/slide_gifs/*

## Slide 3: Closest prior work: SSIL + CS-PIBT

- Emphasize that we are not discarding the successful CS-PIBT line; we are changing the policy object that feeds it.
- That makes the comparison fair: same shield, same maps, same scenario ladder, same success criterion.
- The key hypothesis is that flow should be a better inductive bias for geometry and later continuous-space transfer, even if the current grid interface is imperfect.

Sources:
- README.md
- PAPER_PLAN.md
- FLOW_VS_SSIL_ABLATION_PLAN.md

## Slide 4: Problem setup and target interface

- Keep the math light: what matters is the interface from a continuous prediction back into a discrete coordinated planner.
- The repo still evaluates strict all-agents success, not only average motion quality.
- That is why agents-at-goal fraction becomes an important secondary diagnostic later in the deck.

Sources:
- main_pys/model_inputs.py
- main_pys/simulator.py
- eval_rishi_paper.py

## Slide 5: What was implemented on barath/flow-gnn

- Use this slide to orient advisors to actual repo scope: this is not just one model file, but a full train/eval/analysis branch.
- The important architectural move is the shared encoder feeding both flow output and an auxiliary action head.
- That auxiliary head becomes central when we interpret the later transformer and action-head results.

Sources:
- main_pys/generative_model.py
- main_pys/train_flow.py
- eval_rishi_paper.py
- analysis_scripts/*

## Slide 6: Training objective and inference interface

- This is the key technical slide: training lives in continuous velocity space, but deployment still has to collapse back into 5 planner actions.
- That is why the branch adds the action head, inference sweeps, and later transformer action-head experiments.
- When results break at high density, this slide gives the mechanism: the learned geometry must still survive a discrete ranking bottleneck.

Sources:
- main_pys/train_flow.py
- main_pys/generative_model.py
- main_pys/simulator.py
- eval_rishi_paper.py

## Slide 7: Experimental protocol

- Be explicit that the clean held-out number and the topology-exposed comparisons are different protocols, and the deck keeps them separate.
- This slide is where you reassure advisors the comparison to SSIL is fair.
- The later topology slide then uses the exposure experiment to argue that training-data coverage alone is not the main issue.

Sources:
- eval_rishi_paper.py
- evals/final_evals/slide_outputs/*
- evals/final_evals/topology_exposure/*

## Slide 8: Planner behavior: solved vs hard cases

- Use the solved case to show that the model and shield can produce clean coordinated motion.
- Use the hard warehouse case to foreshadow the main limitation: narrow bottlenecks still break the continuous-to-discrete interface.
- If the GIFs animate in slideshow mode, let them run briefly; if not, the first frame still anchors the discussion.

Sources:
- logs/slide_gifs/progress_smoke.gif
- logs/slide_gifs/planner_near_miss_warehouse_100_agents.gif

## Slide 9: Headline held-out results

- Lead with the fairness point: this is the clean held-out comparison, not the topology-exposed reference.
- The encouraging result is that Flow-GNN is already competitive at 100 to 300 agents.
- The problem is high-density ranking and runtime, not that the model cannot move agents in the right direction at all.

Sources:
- presentation/advisor_update/generated_plots/grid_summary.json
- evals/final_evals/slide_outputs/tables/per_agent_summary.md

## Slide 10: Topology exposure does not close the gap

- This is the cleanest argument that the remaining deficit is not just missing map families in training.
- There are small wins on Empty 48x48 and Random 32x32, but the overall number barely moves.
- That points back to planner alignment and action interface as the higher-leverage target.

Sources:
- evals/final_evals/topology_exposure/topology_exposure_slide_ready_summary.md
- evals/final_evals/topology_exposure/topology_exposure_slide_ready_highres.png

## Slide 11: Where failures come from

- Avoid overclaiming: warehouse and random failures are often near-complete, but maze and den maps still reflect real coordination difficulty.
- This is the slide that turns a disappointing strict-success gap into a constructive diagnosis.
- Tie it directly to the next-step agenda: richer planner interfaces rather than only more training data.

Sources:
- presentation/advisor_update/generated_plots/grid_summary.json
- evals/final_evals/slide_outputs/tables/per_map_summary.md

## Slide 12: Continuous-space flow is the natural next step

- This is the cleanest argument for the flow formulation itself: it transfers naturally into continuous control, where direct velocity prediction is a better fit.
- Be honest that ORCA still wins and obstacle maps are deferred.
- The success here is conceptual and directional: flow already beats the discretized continuous-action baseline on the empty-map pilot.

Sources:
- implementation_details.md
- evals/continuous_evals/slide_outputs/continuous_summary.md

## Slide 13: Transformer action selection is informative, but not the fix

- Use this as a negative but useful result: more expressive sequence models did not automatically solve the planner-interface problem.
- The action-head transformer is clearly the best transformer variant, which again points to action alignment.
- That keeps the future-work focus disciplined rather than chasing architecture alone.

Sources:
- evals/transformer/slide_outputs/transformer_summary.md
- evals/transformer/slide_outputs/full_comparison/shared_case_comparison_summary.md

## Slide 14: Future work: replace cardinal actions with a lattice interface

- This is the main future-work thesis for advisors: the next grid-world experiment should change the action interface, not only retrain the same 5-way bottleneck harder.
- A lattice interface is also a good bridge conceptually between the grid-world and continuous-space branches.
- Frame this as the most promising route to turn the current diagnosis into an actionable grid-world experiment.

Sources:
- PAPER_PLAN.md
- main_pys/simulator.py
- FLOW_VS_SSIL_ABLATION_PLAN.md

## Slide 15: Takeaways and advisor discussion

- Close by making the diagnosis actionable rather than defensive.
- The positive story is strong enough to justify the branch: flow is not a dead end, it is revealing where the interface really matters.
- End by asking for guidance on whether to pursue the lattice interface immediately or do one more action-alignment round first.

Sources:
- PAPER_PLAN.md
- FLOW_VS_SSIL_ABLATION_PLAN.md
- evals/final_evals/slide_outputs/takeaways.md

## Slide 16: 12-map panel comparison

- Use only if the discussion shifts toward topology familiarity or the broader 12-map panel.
- The main point remains the same: the gap persists and is not obviously erased by more familiar map families.

Sources:
- evals/final_evals/rishi12_ours_vs_rishi_1v1/rishi12_ours_vs_ssil_highres_2x6_report.md

## Slide 17: Transformer detailed scaling

- Keep this as a technical backup if advisors ask whether the transformer effort is competitive already.
- The answer in the repo today is no: useful diagnostic, not yet a headline model.

Sources:
- evals/transformer/slide_outputs/full_comparison/figures/shared_case_success_by_agents.png

