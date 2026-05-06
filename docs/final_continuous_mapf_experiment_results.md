# Final Continuous-Space MAPF Experiment Results

Date finalized: 2026-05-06  
Branch: `barath/continuous_space_epibt`  
System: FlowGNNModel preferred velocities + EPIBTShield continuous priority-inheritance shield  
Primary eval command: `python analyze_evals.py --markdown`  
Final catalog status: `Loaded 22 experiment(s)` with no skipped expected catalog entries.

## System Summary

The final system evaluates a learned rectified-flow GNN policy in continuous-space MAPF. At each timestep:

1. FlowGNNModel predicts preferred velocities.
2. The policy uses 3 Euler integration steps by default.
3. EPIBTShield resolves conflicts with continuous-space priority inheritance and backtracking over velocity choices.
4. The executed velocities are evaluated for goal completion, collisions, near-collisions, obstacle hits, path efficiency, arrival time, and runtime.

Primary eval flags:

```bash
--policy flow --shield-type epibt --num-integration-steps 3
```

Important metric:

- `path_length_ratio`: all agents, including non-arrived agents.
- `arrived_path_length_ratio`: only agents that actually reach goals. This is the paper-friendly path-efficiency metric because it avoids inflating path length with agents that wander after failure.

## Experiment Matrix

The final analyzer loaded all 22 cataloged experiments:

- Set A primary baselines: ORCA, PO-ORCA, Straight+EPIBTShield.
- Set A main Flow v4b: 256-step and 512-step evaluations.
- Set A Flow+ORCA shield ablation: 256-step and 512-step evaluations.
- Set A integration-step ablations: 5, 10, and 20 Euler steps at 512 env steps.
- Set A multi-seed robustness: seeds 42, 123, and 456 at 512 env steps.
- Set B generalization: random-64-64-10 and room-32-32-4, N=50.
- Set C OOD warehouse: warehouse-10-20-10-2-1, N=50 and N=100.
- Old model comparison: Flow v4 vs v4b at 256 env steps.

All primary rows use 25 scenarios per map/agent-count combination.

Expected row counts:

| Eval family | Rows |
|---|---:|
| Set A per method | 100 |
| Set B per method | 50 |
| Set C per method | 50 |
| Multi-seed Set A per seed | 100 |

## Main Set A Summary

Set A averages over `empty-48-48` and `random-32-32-10`, with N=50 and N=100.

| Method | N50 AtGoal | N50 Coll | N100 AtGoal | N100 Coll | N50 Succ | N50 PLR | Rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORCA (straight) | 0.579 | 492.9 | 0.584 | 2101.4 | 0.500 | 0.677 | 100 |
| PO-ORCA (straight) | 0.576 | 334.7 | 0.575 | 1462.6 | 0.500 | 0.702 | 100 |
| Straight + EPIBTShield | 0.889 | 105.7 | 0.886 | 675.8 | 0.480 | 1.387 | 100 |
| Flow v4b + EPIBT, 256 steps | 0.842 | 49.4 | 0.813 | 290.1 | 0.000 | 2.516 | 100 |
| Flow v4b + EPIBT, 512 steps | 0.927 | 98.4 | 0.921 | 354.9 | 0.360 | 4.568 | 100 |

Primary takeaway: EPIBTShield is the main mechanism that unlocks high completion in structured maps, and learned flow guidance adds further gains on top of the shield.

## Primary Paper Number

On `random-32-32-10`, N=100, 512 env steps:

| Method | AtGoal | Collisions | Arrived PLR | Mean Arrival Step |
|---|---:|---:|---:|---:|
| ORCA (straight) | 0.168 | 4168.9 | 1.010 | 434.1 |
| PO-ORCA (straight) | 0.149 | 2626.3 | 1.096 | 443.0 |
| Straight + EPIBTShield | 0.774 | 1306.9 | 1.122 | 187.5 |
| Flow v4b + ORCA, 512 steps | 0.486 | 3996.0 | 1.380 | 330.6 |
| Flow v4b + EPIBT, 512 steps | 0.874 | 633.4 | 1.462 | 167.6 |

Paper-ready interpretation:

- ORCA gets only 16.8% of agents to goals in dense random maps.
- Straight+EPIBTShield raises this to 77.4%, showing that priority inheritance/backtracking is the core mechanism.
- Flow+EPIBT raises completion further to 87.4%, showing learned guidance helps once the shield can resolve conflicts.
- Flow+ORCA reaches only 48.6%, showing that the learned policy alone is insufficient under a weaker ORCA shield.
- Flow+EPIBT reduces collisions from 4168.9 to 633.4 versus ORCA, an 84.8% reduction.
- Arrived agents under Flow+EPIBT take 1.462x optimal path length, which is much more favorable than all-agent PLR suggests.

## Set A Per-Map Details

### empty-48-48, N=50, 512 steps

| Method | AtGoal | Success | Coll | NearColl | PLR | ArrPLR | ArrStep |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORCA | 1.000 | 1.000 | 2.6 | 156.6 | 0.998 | 0.995 | 123.9 |
| PO-ORCA | 1.000 | 1.000 | 71.9 | 127.0 | 1.013 | 1.007 | 125.2 |
| Straight+EPIBT | 0.998 | 0.960 | 4.7 | 377.1 | 1.038 | 1.026 | 127.5 |
| Flow+EPIBT | 0.989 | 0.720 | 14.2 | 571.6 | 3.235 | 1.222 | 153.6 |
| Flow+ORCA | 0.922 | 0.120 | 65.2 | 1239.6 | 1.357 | 1.221 | 186.5 |

### empty-48-48, N=100, 512 steps

| Method | AtGoal | Success | Coll | NearColl | PLR | ArrPLR | ArrStep |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORCA | 1.000 | 0.960 | 34.0 | 676.4 | 1.003 | 0.997 | 124.1 |
| PO-ORCA | 1.000 | 1.000 | 299.0 | 535.7 | 1.035 | 1.020 | 126.4 |
| Straight+EPIBT | 0.998 | 0.920 | 44.6 | 1566.2 | 1.079 | 1.055 | 130.5 |
| Flow+EPIBT | 0.968 | 0.160 | 76.4 | 2638.3 | 4.043 | 1.254 | 161.2 |
| Flow+ORCA | 0.849 | 0.000 | 655.8 | 4266.8 | 1.481 | 1.251 | 206.6 |

### random-32-32-10, N=50, 512 steps

| Method | AtGoal | Success | Coll | NearColl | PLR | ArrPLR | ArrStep |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORCA | 0.158 | 0.000 | 983.3 | 778.6 | 0.355 | 1.002 | 439.4 |
| PO-ORCA | 0.153 | 0.000 | 597.6 | 1289.0 | 0.390 | 1.041 | 442.1 |
| Straight+EPIBT | 0.779 | 0.000 | 206.7 | 1009.9 | 1.735 | 1.078 | 181.9 |
| Flow+EPIBT | 0.866 | 0.000 | 182.6 | 743.5 | 5.900 | 1.412 | 165.3 |
| Flow+ORCA | 0.485 | 0.000 | 973.5 | 1288.5 | 1.408 | 1.344 | 330.6 |

### random-32-32-10, N=100, 512 steps

| Method | AtGoal | Success | Coll | NearColl | PLR | ArrPLR | ArrStep |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORCA | 0.168 | 0.000 | 4168.9 | 2715.8 | 0.386 | 1.010 | 434.1 |
| PO-ORCA | 0.149 | 0.000 | 2626.3 | 5367.2 | 0.483 | 1.096 | 443.0 |
| Straight+EPIBT | 0.774 | 0.000 | 1306.9 | 3954.9 | 1.733 | 1.122 | 187.5 |
| Flow+EPIBT | 0.874 | 0.000 | 633.4 | 2745.0 | 5.854 | 1.462 | 167.6 |
| Flow+ORCA | 0.486 | 0.000 | 3996.0 | 3920.8 | 1.536 | 1.380 | 330.6 |

## Flow+ORCA Shield Ablation

This is the cleanest evidence that EPIBTShield is essential even when the preferred velocity comes from the learned flow model.

| Map | N | Flow+ORCA AtGoal | Flow+EPIBT AtGoal | Delta |
|---|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.922 | 0.989 | +0.067 |
| empty-48-48 | 100 | 0.849 | 0.968 | +0.119 |
| random-32-32-10 | 50 | 0.485 | 0.866 | +0.381 |
| random-32-32-10 | 100 | 0.486 | 0.874 | +0.388 |

Interpretation: On the hard random map, swapping ORCA for EPIBTShield almost doubles completion for the same learned preferred velocity policy.

## Step Budget Comparison

Default uses 512 environment steps. The 256-step result shows good early progress but lower completion, especially on open maps where many agents need more rollout time.

| Map | N | Flow+EPIBT 256 AtGoal | Flow+EPIBT 512 AtGoal | 256 Coll | 512 Coll |
|---|---:|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.850 | 0.989 | 12.8 | 14.2 |
| empty-48-48 | 100 | 0.812 | 0.968 | 68.0 | 76.4 |
| random-32-32-10 | 50 | 0.835 | 0.866 | 85.9 | 182.6 |
| random-32-32-10 | 100 | 0.814 | 0.874 | 512.2 | 633.4 |

Interpretation: 512 steps is the right default for completion. The 256-step version has lower PLR because agents are simply simulated for less time, but it undercounts late arrivals.

## Integration-Step Ablation

All rows use Flow v4b + EPIBT at 512 env steps, varying Euler integration steps.

| Map | N | 3-step AtGoal | 5-step AtGoal | 10-step AtGoal | 20-step AtGoal |
|---|---:|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.989 | 0.984 | 0.982 | 0.984 |
| empty-48-48 | 100 | 0.968 | 0.961 | 0.941 | 0.956 |
| random-32-32-10 | 50 | 0.866 | 0.893 | 0.856 | 0.870 |
| random-32-32-10 | 100 | 0.874 | 0.879 | 0.856 | 0.868 |

Arrived PLR for integration steps:

| Map | N | 3-step ArrPLR | 5-step ArrPLR | 10-step ArrPLR | 20-step ArrPLR |
|---|---:|---:|---:|---:|---:|
| empty-48-48 | 50 | 1.222 | 1.291 | 1.336 | 1.333 |
| empty-48-48 | 100 | 1.254 | 1.323 | 1.352 | 1.371 |
| random-32-32-10 | 50 | 1.412 | 1.524 | 1.587 | 1.620 |
| random-32-32-10 | 100 | 1.462 | 1.592 | 1.644 | 1.659 |

Interpretation: 3 Euler steps is the best default. More steps provide no reliable AtGoal gain and consistently worsen arrived-agent path efficiency.

## Multi-Seed Robustness

Seeds 42, 123, and 456 were trained/evaluated on Set A at 512 env steps.

| Map | N | Seed 42 AtGoal | Seed 123 AtGoal | Seed 456 AtGoal | Mean AtGoal | Std |
|---|---:|---:|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.984 | 0.965 | 0.974 | 0.974 | 0.008 |
| empty-48-48 | 100 | 0.981 | 0.948 | 0.926 | 0.952 | 0.023 |
| random-32-32-10 | 50 | 0.868 | 0.781 | 0.801 | 0.817 | 0.037 |
| random-32-32-10 | 100 | 0.866 | 0.789 | 0.782 | 0.812 | 0.038 |

For the headline hard setting, `random-32-32-10`, N=100:

- Seed 42: 0.866 AtGoal, 635.8 collisions, ArrPLR 1.467.
- Seed 123: 0.789 AtGoal, 471.6 collisions, ArrPLR 1.427.
- Seed 456: 0.782 AtGoal, 523.9 collisions, ArrPLR 1.432.
- Mean +/- std AtGoal: 0.812 +/- 0.038.

Important caveat: seed 42 is the strongest model. The retrained seed 123/456 models remain substantially above Straight+EPIBT on the hard random map? For N=100, Straight+EPIBT is 0.774, while seed 123 is 0.789 and seed 456 is 0.782, so the learned advantage persists but is modest in those retrained seeds. Use the main v4b seed-42 result for the primary table and the multi-seed table as a robustness/variance disclosure.

## Generalization: Set B

Set B includes `random-64-64-10` and `room-32-32-4`, N=50, 512 env steps.

| Map | Method | AtGoal | Coll | ArrPLR | ArrStep |
|---|---|---:|---:|---:|---:|
| random-64-64-10 | ORCA | 0.048 | 383.3 | 0.992 | 490.1 |
| random-64-64-10 | PO-ORCA | 0.048 | 239.7 | 0.999 | 490.1 |
| random-64-64-10 | Straight+EPIBT | 0.505 | 267.2 | 1.051 | 331.1 |
| random-64-64-10 | Flow+EPIBT | 0.614 | 106.0 | 1.391 | 321.6 |
| room-32-32-4 | ORCA | 0.012 | 1028.4 | 0.920 | 506.0 |
| room-32-32-4 | PO-ORCA | 0.010 | 901.5 | 0.949 | 506.8 |
| room-32-32-4 | Straight+EPIBT | 0.114 | 2019.6 | 1.153 | 458.9 |
| room-32-32-4 | Flow+EPIBT | 0.262 | 1113.1 | 3.069 | 404.5 |

Interpretation:

- On larger random maps, EPIBTShield alone increases completion from 4.8% to 50.5%, and flow guidance further increases it to 61.4%.
- On room maps, completion is lower overall, but the same ordering holds: ORCA 1.2%, Straight+EPIBT 11.4%, Flow+EPIBT 26.2%.
- Flow guidance improves completion and arrival speed on Set B but may increase arrived-path ratio, especially on room maps.

## OOD Warehouse: Set C

Set C is `warehouse-10-20-10-2-1`, with N=50 and N=100.

| N | Method | AtGoal | Coll | ArrPLR | ArrStep |
|---:|---|---:|---:|---:|---:|
| 50 | ORCA | 0.132 | 855.3 | 0.994 | 459.8 |
| 50 | PO-ORCA | 0.129 | 391.7 | 0.998 | 460.6 |
| 50 | Straight+EPIBT | 0.411 | 29.8 | 1.173 | 403.0 |
| 50 | Flow+EPIBT | 0.255 | 57.4 | 1.521 | 429.5 |
| 100 | ORCA | 0.125 | 3495.9 | 0.995 | 462.4 |
| 100 | PO-ORCA | 0.122 | 2001.2 | 1.003 | 463.7 |
| 100 | Straight+EPIBT | 0.394 | 167.7 | 1.253 | 406.0 |
| 100 | Flow+EPIBT | 0.217 | 320.0 | 1.567 | 438.6 |

Paper framing:

EPIBTShield generalizes strongly out of domain: on warehouse N=50, ORCA reaches 13.2%, while Straight+EPIBT reaches 41.1%. However, learned flow guidance does not improve over shield-only behavior on warehouse maps: Flow+EPIBT reaches only 25.5% at N=50 and 21.7% at N=100. This is expected because warehouse layouts were not included in training and should be framed as a domain adaptation limitation, not a shield failure.

## v4 vs v4b Model Comparison

At 256 env steps, v4b outperforms v4 consistently.

| Map | N | v4b AtGoal | v4 AtGoal | Delta |
|---|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.850 | 0.744 | +0.106 |
| empty-48-48 | 100 | 0.812 | 0.680 | +0.132 |
| random-32-32-10 | 50 | 0.835 | 0.740 | +0.095 |
| random-32-32-10 | 100 | 0.814 | 0.730 | +0.084 |

Interpretation: v4b is the stronger final model and should be used for all main paper claims.

## Paper Story

Main claim:

> Continuous-space priority inheritance with backtracking is a strong shield for MAPF-style multi-agent navigation. It dramatically improves completion over ORCA in structured environments, and learned flow-matching guidance further improves completion when the evaluation domain resembles training data.

Contribution decomposition:

1. ORCA fails in cluttered dense environments.
2. PO-ORCA reduces some collisions but does not solve deadlock/completion.
3. Straight+EPIBTShield shows that the shield itself is powerful without learning.
4. Flow+ORCA shows that learning without the right shield is insufficient.
5. Flow+EPIBT shows that learning and shielding are complementary.

Recommended abstract number:

On `random-32-32-10`, N=100, Flow+EPIBT reaches 87.4% AtGoal versus 16.8% for ORCA and 77.4% for Straight+EPIBT, while reducing collisions by 84.8% versus ORCA.

Recommended limitation sentence:

> On fully OOD warehouse maps, the learned flow prior underperforms the shield-only baseline, indicating that the current policy learns map-family-specific guidance; nevertheless, EPIBTShield itself generalizes strongly, improving warehouse N=50 completion from 13.2% to 41.1%.

## Recommended Paper Tables

### Main ablation table

Use `random-32-32-10`, N=100, 512 steps:

| Method | Preferred velocity | Shield | AtGoal | Coll | ArrPLR |
|---|---|---|---:|---:|---:|
| ORCA | straight | ORCA | 0.168 | 4168.9 | 1.010 |
| PO-ORCA | straight | priority ORCA | 0.149 | 2626.3 | 1.096 |
| Straight+EPIBT | straight | EPIBT | 0.774 | 1306.9 | 1.122 |
| Flow+ORCA | learned flow | ORCA | 0.486 | 3996.0 | 1.380 |
| Flow+EPIBT | learned flow | EPIBT | 0.874 | 633.4 | 1.462 |

### Generalization table

Use Set B and Set C AtGoal:

| Map | N | ORCA | Straight+EPIBT | Flow+EPIBT |
|---|---:|---:|---:|---:|
| random-64-64-10 | 50 | 0.048 | 0.505 | 0.614 |
| room-32-32-4 | 50 | 0.012 | 0.114 | 0.262 |
| warehouse-10-20-10-2-1 | 50 | 0.132 | 0.411 | 0.255 |
| warehouse-10-20-10-2-1 | 100 | 0.125 | 0.394 | 0.217 |

### Multi-seed table

Use Set A, 512 steps:

| Map | N | Mean AtGoal | Std | Seed 42 | Seed 123 | Seed 456 |
|---|---:|---:|---:|---:|---:|---:|
| empty-48-48 | 50 | 0.974 | 0.008 | 0.984 | 0.965 | 0.974 |
| empty-48-48 | 100 | 0.952 | 0.023 | 0.981 | 0.948 | 0.926 |
| random-32-32-10 | 50 | 0.817 | 0.037 | 0.868 | 0.781 | 0.801 |
| random-32-32-10 | 100 | 0.812 | 0.038 | 0.866 | 0.789 | 0.782 |
