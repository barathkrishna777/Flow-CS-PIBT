# Continuous-Space Pilot Results

These CSVs are small smoke/pilot evaluations, mostly on one empty-48-48 random scenario. Treat them as preliminary, not as a full benchmark.

## Best slide use

- Use as a backup/appendix figure showing continuous-space infrastructure and early trends.
- ORCA is the strongest continuous baseline here: highest agents-at-goal, shortest paths, smoothest trajectories, and fastest runtime.
- Learned flow variants avoid most obstacle hits compared with the discrete policy, and larger consensus generally improves path-length ratio/smoothness for 32-128 agents, but they do not beat ORCA.
- Avoid claiming continuous-space SOTA; the data is too small and contains collisions/near-collisions at higher density.

## Files

- `figures/continuous_empty48_pilot_summary.png`
- `figures/continuous_flow_consensus_sweep.png`
- `tables/continuous_empty48_summary.csv`
