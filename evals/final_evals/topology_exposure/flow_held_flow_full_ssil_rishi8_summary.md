# flow_held vs flow_full vs SSIL

![flow held full ssil highres](flow_held_flow_full_ssil_rishi8_highres.png)

## Overall

| Model | Rows | Success | Delta vs flow_held |
| --- | ---: | ---: | ---: |
| flow_held (ours) | 1850 | 59.73% | +0.00 pp |
| flow_full (Parth's) | 1850 | 60.05% | +0.32 pp |
| SSIL (Rishi's) | 1850 | 66.43% | +6.70 pp |

## Per Map

| Map | Runs | flow_held | flow_full | flow_full delta | SSIL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paris_1_256 | 250 | 100.00% | 100.00% | +0.00 pp | 100.00% |
| empty-48-48 | 250 | 92.40% | 95.60% | +3.20 pp | 100.00% |
| maze-128-128-2 | 250 | 25.60% | 25.60% | +0.00 pp | 40.00% |
| random-64-64-10 | 250 | 66.40% | 66.80% | +0.40 pp | 85.20% |
| random-32-32-10 | 100 | 56.00% | 61.00% | +5.00 pp | 81.00% |
| warehouse-10-20-10-2-1 | 250 | 4.40% | 2.80% | -1.60 pp | 2.40% |
| den312d | 250 | 31.20% | 32.40% | +1.20 pp | 32.00% |
| den520d | 250 | 99.60% | 96.80% | -2.80 pp | 99.60% |

## By Agent Count

| Agents | Runs | flow_held | flow_full | flow_full delta | SSIL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 200 | 91.50% | 88.50% | -3.00 pp | 89.00% |
| 200 | 200 | 80.00% | 79.50% | -0.50 pp | 81.50% |
| 300 | 200 | 74.50% | 70.50% | -4.00 pp | 80.00% |
| 400 | 200 | 56.00% | 63.50% | +7.50 pp | 68.00% |
| 500 | 175 | 52.57% | 54.86% | +2.29 pp | 61.71% |
| 600 | 175 | 53.14% | 52.00% | -1.14 pp | 61.71% |
| 700 | 175 | 49.71% | 49.71% | +0.00 pp | 54.86% |
| 800 | 175 | 47.43% | 46.86% | -0.57 pp | 53.71% |
| 900 | 175 | 46.86% | 45.71% | -1.14 pp | 54.86% |
| 1000 | 175 | 36.57% | 40.57% | +4.00 pp | 51.43% |

## Interpretation

- `flow_full` is only slightly better than `flow_held` overall on this epoch20 checkpoint: +0.32 pp.
- The clearest topology-exposure wins are `random-32-32-10` (+5.00 pp) and `empty-48-48` (+3.20 pp).
- `SSIL (Rishi's)` remains strongest overall on this rishi8 panel.
- This supports a cautious claim: topology exposure can help on familiar/random/open maps, but checkpoint selection matters.
