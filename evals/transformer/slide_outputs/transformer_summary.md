# Transformer Held-Out Ablation Summary

All three transformer CSVs cover the same 8 held-out maps; flow-dot has 6 fewer rows, so paired comparisons use 1,844 common cases.

## Overall

| Variant | Cases | Success | Agents at Goal | Mean Runtime | Median Runtime |
|---|---:|---:|---:|---:|---:|
| Flow-dot aux | 1844 | 20.2% | 67.0% | 53.0s | 60.1s |
| Action head | 1850 | 42.1% | 79.4% | 42.4s | 49.8s |
| Flow only | 1850 | 30.9% | 68.5% | 50.4s | 60.0s |

## Paired Wins

- Action head vs Flow only: 1128-211 wins, 505 ties, comparing success first and fraction of agents at goal second.
- Action head vs Flow-dot aux: 1302-195 wins, 347 ties, comparing success first and fraction of agents at goal second.
- Flow only vs Flow-dot aux: 856-660 wins, 328 ties, comparing success first and fraction of agents at goal second.

## Slide Takeaways

- Add a small transformer ablation slide if you want to show the training-objective exploration: the action head is the clear best transformer variant.
- Do not position this as beating the current main Flow model: existing 8-heldout Flow CSVs in this repo are around 60-62% success, while transformer-action-head is 42.1%.
- Action head keeps nonzero success up to 1000 agents overall, while flow-only and flow-dot collapse at the highest crowd sizes.
- Maze-128-128-2 is unsolved by all transformer variants, which is a useful failure-mode caveat.
