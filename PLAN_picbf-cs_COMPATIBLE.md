# Flow-CS-PIBT Compatibility With Local-Joint picbf-cs

## Summary
Update Flow-CS-PIBT so `--shield-type picbf-cs` uses the new decentralized-facing `LocalJointCBFShield` API from `picbf-cs`, not the old centralized `ContinuousCBFShield` path. Keep `picbf-cs` itself unchanged. Target Python 3.11 for evals, matching `picbf-cs`’s declared `requires-python >=3.11`.

## Key Changes
- In Flow’s `PICBFCSShield` adapter, import and require:
  - `LocalJointCBFShield`
  - `LocalAgentState`
  - `AgentState`
  - `AABBObstacle`
  - `ObstacleIndex`
  - `ShieldConfig`
- Make `--shield-type picbf-cs` instantiate `LocalJointCBFShield` and call:
  - local agents as `LocalAgentState(agent_id=i, state=AgentState(...))`
  - nominal velocities as `{i: (vx, vy)}`
  - output velocities as `result.velocities_by_id[i]` in original agent-array order
- Preserve Flow’s current `(row, col)` continuous coordinate convention for agents and AABB obstacles.
- Cache per-map obstacle data as both:
  - merged AABB obstacle tuple
  - `ObstacleIndex(obstacles)`
- Add an optional eval/benchmark flag:
  - `--picbf-communication-radius FLOAT`
  - default `None`, meaning auto-compute:
    `2 * agent_radius + safety_margin + 2 * max_speed * dt + max(0.4, 4 * safety_margin)`
  - validate provided radius is positive
- Thread this option through `eval_continuous.py`, `run_continuous_benchmark.py`, and `ContinuousMAPFEnv`.
- Store lightweight shield debug info after each picbf step:
  - component count
  - max component size
  - total local solver time
  - optional status counts
- Include that debug info in eval progress logs when available, so the smoke eval can show whether the new local components are actually small.

## Runtime And Compatibility
- Do not modify `picbf-cs` in this implementation.
- Assume Lambda evals run with Python 3.11.
- If Python 3.8 is used, Flow should fail fast with a clear message that the current `picbf-cs` package requires Python 3.11 or a separate picbf-cs compatibility patch.
- If `LocalJointCBFShield` is missing from the imported package, raise an actionable error asking the user to pull/update `picbf-cs`.

## Test Plan
- Add a focused Flow-side synthetic test for `picbf-cs` local adapter:
  - 4-agent crossing on empty grid
  - stable ID-keyed output returns one velocity per agent
  - zero collisions and zero obstacle hits
  - debug info reports at least one local component and max component size no larger than agent count
- Run:
  - `python -m py_compile eval_continuous.py run_continuous_benchmark.py main_pys/continuous_env.py`
  - the new focused picbf local adapter test
  - existing `picbf-cs` tests in the `picbf-cs` repo under Python 3.11
- Smoke eval on Lambda after pulling both repos:
  ```sh
  python3.11 eval_continuous.py \
    --map-dir data/mapf-map \
    --scen-dir data/scen-random \
    --maps empty-48-48 \
    --agent-counts 32 \
    --policy orca \
    --shield-type picbf-cs \
    --max-steps 64 \
    --log-interval 1 \
    --output-csv evals/continuous_picbf_cs_local_orca_smoke.csv
  ```
- Acceptance criteria for the smoke eval:
  - logs show local component stats
  - per-step runtime is substantially lower than the old centralized run’s multi-second spikes
  - collisions and obstacle hits remain zero
  - at-goal fraction is no worse than the old `0.219` baseline, with improvement expected but not hard-coded as a correctness gate

## Assumptions
- `--shield-type picbf-cs` should now mean the local-joint decentralized-facing shield.
- No separate centralized `picbf-cs` option is needed for this pass.
- Agent array index is a stable agent ID for the duration of each Flow eval rollout.
- The automatic communication radius should track Flow’s `agent_radius`, `max_speed`, `dt`, and `safety_margin` rather than being fixed at `2.0`.
