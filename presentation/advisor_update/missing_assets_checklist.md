# Missing Assets Checklist

## Highest priority

- A cleaner single benchmark table exported directly from the held-out evaluation code with:
  - success
  - agents at goal
  - runtime
  - cost per agent
  - one row each for clean Flow-GNN, topology-exposed reference, and SSIL
- One or two polished rollout GIFs for:
  - a strong success case at moderate density
  - a failure case that clearly deadlocks at a bottleneck
- A more presentation-ready topology-exposure figure with larger labels than the current composite image

## Nice to have

- A compact ablation plot isolating:
  - flow-only
  - flow + action head
  - direct classifier
  - any planner-interface variants already evaluated
- A diagram or still frame showing how CS-PIBT shielding/conflict resolution actually intervenes during rollout
- A single architecture figure for the continuous-space stack and a matching one for the transformer branch

## If there is time tonight

- One backup slide comparing success, runtime, and cost on the same chart family with larger fonts
- A short qualitative failure taxonomy with 3 categories:
  - late deadlock after near-complete progress
  - maze coordination miss
  - warehouse aisle contention
