# Next Steps After Continuous Evaluation Review

## Summary
- The current continuous results support one strong claim already: on `empty-48-48`, **continuous flow is clearly better than the discretized continuous-action baseline** across `32/64/96/128` agents, while still trailing ORCA.
- The open-space continuous story is now credible; the obstacle-space story is **not** ready yet. The `random-32-32-10` CSVs show ORCA itself failing with extreme `obstacle_hits` and collisions, so obstacle handling is still a systems bug / geometry issue, not a model comparison result.
- For the paper, do **not** compare the continuous section against the original grid-trained discrete model forced into continuous execution. Use:
  - **Grid section**: original discrete grid policy vs grid flow
  - **Continuous section**: ORCA vs continuous flow vs discretized continuous-action baseline trained on the same continuous data

## Key Changes / Next Research Moves
- **Freeze the open-space continuous benchmark protocol first.**
  - Use `empty-48-48` as the primary continuous benchmark until the obstacle stack is fixed.
  - Primary reported method settings:
    - flow: `num_integration_steps=3`
    - flow consensus: **3** samples as default
    - discrete baseline: 8-direction continuous-action model
    - shield: ORCA
  - Treat `agent_fraction_at_goal` as the primary performance metric, with `success` as a strict secondary metric.

- **Rebuild the continuous comparison on a larger and fairer open-space dataset.**
  - Generate `empty-48-48` data for multiple scenarios, not just `random-1`.
  - Use the same training dataset and same epoch budget for flow and the discretized continuous baseline.
  - Retrain both methods from scratch on that shared dataset before freezing the comparison table.
  - Run at least **3 seeds** per learned method for the open-space benchmark.

- **Lock the best flow inference setting before scaling.**
  - Use the current results to choose consensus `=3` as the default.
  - Do one final inference sweep on the larger open-space dataset:
    - consensus `1/3/5`
    - steps `3` only unless a new sweep shows a reason to revisit
  - Acceptance criterion: the chosen setting improves or preserves `agent_fraction_at_goal` relative to `c1` without an unjustified runtime penalty.

- **Unblock obstacle maps as a separate engineering milestone.**
  - Do not run new obstacle-map learning experiments until ORCA + shield behavior is sane on obstacle maps.
  - Investigate `main_pys/continuous_env.py` obstacle geometry and coordinate conventions for the `rvo2` obstacle encoding.
  - Validate with one sparse obstacle map first, then retry `random-32-32-10`.
  - Acceptance criterion for obstacle readiness:
    - `obstacle_hits` near zero for ORCA
    - dramatically fewer collisions than the current `random-32-32-10` smoke runs
    - meaningful goal completion at `32/64` agents

## Concrete Experiment Sequence
- **Phase A: Open-space benchmark expansion**
  - Generate `empty-48-48` continuous data for scenarios `1-5` and agents `32/64/96/128`
  - Train:
    - flow, 3 seeds, 30 epochs each
    - discretized continuous baseline, 3 seeds, 30 epochs each
  - Evaluate ORCA, flow, and discretized continuous baseline on the same scenario set.
  - Produce:
    - main table for `32/64/96/128`
    - scaling plot: agent count vs `agent_fraction_at_goal`
    - runtime plot
    - trajectory visualizations for one low-density and one high-density case

- **Phase B: Obstacle-stack repair**
  - Add a small obstacle-only validation battery:
    - 2-agent obstacle bypass
    - corridor
    - sparse random clutter
  - Re-run ORCA-only eval on one easy obstacle map, then `random-32-32-10`.
  - Only after ORCA is trustworthy, train/evaluate continuous flow on one obstacle map.

- **Phase C: Reconnect to the grid-paper story**
  - Keep the continuous section method comparison separate from the grid section.
  - Use the grid world to compare the original discrete grid model vs grid flow.
  - Use the continuous world to compare ORCA vs continuous flow vs discretized continuous-action baseline.

## Test Plan / Acceptance Criteria
- **Open-space benchmark acceptance**
  - Flow must remain above the discretized continuous baseline across `32/64/96/128` on mean `agent_fraction_at_goal`.
  - Flow should remain within a defensible gap to ORCA on open maps.
  - Consensus `=3` should remain the default unless a broader sweep disproves it.

- **Obstacle readiness acceptance**
  - ORCA obstacle eval no longer shows the current pathological `obstacle_hits`.
  - Collision counts and path-length ratios become plausible before any obstacle-map learning claims are made.

- **Paper-readiness acceptance**
  - Every continuous comparison uses methods defined for continuous MAPF.
  - The original grid discrete model is only compared in the grid section.
  - Report both:
    - `agent_fraction_at_goal` as primary
    - `success` as secondary strict metric
  - Also report path length ratio, smoothness, runtime, collisions, and near-collisions.

## Assumptions and Defaults
- Default continuous flow inference for the next round: **3 Euler steps, 3 consensus samples**
- Primary continuous benchmark for the next phase: `empty-48-48`
- Obstacle-map claims are explicitly deferred until ORCA obstacle handling is repaired
- The current continuous discrete baseline is treated as a **discretized continuous-action baseline**, not as the original grid-trained discrete model
- The next paper-quality milestone is: **multi-scenario, multi-seed open-space benchmark showing flow > discretized continuous baseline and approaching ORCA**
