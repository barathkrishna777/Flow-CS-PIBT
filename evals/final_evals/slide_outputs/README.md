# Final Eval Slide Outputs

Generated from the five CSVs available in `evals/final_evals`.
Parth's 12-map result was not available, so the 12-map plots compare only Rishi and Ours.

## Key Figures

- `figures/overall_success.svg`: high-level success-rate bars for available eval panels.
- `figures/heldout_success_by_agents.svg`: heldout success vs. agent count.
- `figures/heldout_success_by_map.svg`: heldout success by map.
- `figures/all12_success_by_agents.svg`: 12-map success vs. agent count.
- `figures/all12_success_by_map.svg`: 12-map success by map.
- `tables/comparison_deltas.md`: compact deltas against Rishi for each panel.

## Overall Table

| Panel | Run | Training note | Rows | Maps | Success | Mean agents at goal | Avg runtime | Avg cost / agent | CSV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 heldout maps | Rishi classifier | classifier baseline | 1850 | 8 | 66.4% (1229/1850) | 95.7% | 18.96s | 250.3 | rishi_full_ssil_classifier.csv |
| 8 heldout maps | Ours, trained off heldout | trained on non-heldout maps | 1850 | 8 | 59.7% (1105/1850) | 90.4% | 32.68s | 334.3 | rishi8_wave10_newdata_clean_20260416_224144.csv |
| 8 heldout maps | Parth, trained on all data | trained on full dataset | 1850 | 8 | 62.3% (1152/1850) | 92.7% | 32.80s | 304.9 | rishi8_parth_wave2_ft_v2_epoch14.csv |
| 12-map panel | Rishi classifier | classifier baseline | 2700 | 12 | 55.2% (1491/2700) | 94.8% | 18.62s | 232.4 | rishi12_ssil_classifier.csv |
| 12-map panel | Ours, trained off heldout | trained on non-heldout maps | 2700 | 12 | 47.5% (1282/2700) | 88.8% | 32.73s | 302.6 | rishi12_wave10_newdata_clean_20260416_224144.csv |
