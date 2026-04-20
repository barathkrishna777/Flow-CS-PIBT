# Shared 8-Map Model Comparison Including Transformers

Comparison uses 1844 cases common to SSIL, two Flow GNN CSVs, and all three transformer CSVs. Flow-dot is missing 6 rows, so the common set has 1,844 rather than 1,850 cases.

| Model | Success | Agents at Goal | Mean Runtime | Median Runtime |
|---|---:|---:|---:|---:|
| SSIL classifier | 66.3% | 95.7% | 19.0s | 9.3s |
| Flow GNN (wave10) | 59.6% | 90.4% | 32.7s | 21.2s |
| Flow GNN (Parth e14) | 62.1% | 92.7% | 32.8s | 21.3s |
| Transformer + action head | 42.0% | 79.4% | 42.4s | 49.9s |
| Transformer flow only | 31.0% | 68.5% | 50.4s | 60.0s |
| Transformer flow-dot aux | 20.2% | 67.0% | 53.0s | 60.1s |

## Interpretation

- The transformer action-head variant is the strongest transformer, but it trails both Flow GNN checkpoints and SSIL on success rate.
- SSIL remains the best success-rate baseline in this shared 8-map slice and is much faster, but Flow GNN Parth e14 is close on success and has the highest agents-at-goal fraction among learned Flow models.
- Transformer action-head may be worth one sentence in the main comparison as negative/ablation evidence: transformer architecture alone did not improve held-out MAPF; the action head helped, but not enough to beat the GNN flow policy.
- If slide space is tight, omit flow-only and flow-dot from the main comparison and mention them verbally: action head was the best transformer variant.
