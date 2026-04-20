# Rishi12 1v1: flow_held vs SSIL

Models: `flow_held (ours)` vs `SSIL (Rishi's)`. Figures are high-resolution and split the 12 maps into 2 rows of 6 maps where applicable.

![Success 2x6](rishi12_ours_vs_rishi_success_2x6_highres.png)

![Runtime 2x6](rishi12_ours_vs_rishi_runtime_2x6_highres.png)

![Cost 2x6](rishi12_ours_vs_rishi_cost_2x6_highres.png)

## Overall

| Model | Rows | Success | Delta vs flow_held |
| --- | ---: | ---: | ---: |
| flow_held (ours) | 2700 | 47.48% | +0.00 pp |
| SSIL (Rishi's) | 2700 | 55.22% | +7.74 pp |

## Per Map

| Map | Runs | flow_held success | SSIL success | SSIL delta |
| --- | ---: | ---: | ---: | ---: |
| Berlin_1_256 | 250 | 100.00% | 100.00% | +0.00 pp |
| empty-32-32 | 125 | 76.80% | 100.00% | +23.20 pp |
| maze-32-32-4 | 75 | 12.00% | 14.67% | +2.67 pp |
| random-64-64-20 | 250 | 22.00% | 42.00% | +20.00 pp |
| warehouse-20-40-10-2-1 | 250 | 13.20% | 11.20% | -2.00 pp |
| room-64-64-16 | 250 | 15.60% | 27.20% | +11.60 pp |
| Paris_1_256 | 250 | 100.00% | 100.00% | +0.00 pp |
| empty-48-48 | 250 | 92.40% | 100.00% | +7.60 pp |
| maze-128-128-2 | 250 | 25.60% | 42.00% | +16.40 pp |
| random-64-64-10 | 250 | 66.40% | 85.20% | +18.80 pp |
| warehouse-10-20-10-2-1 | 250 | 4.40% | 2.40% | -2.00 pp |
| den312d | 250 | 31.20% | 32.00% | +0.80 pp |

## By Agent Count

| Agents | Runs | flow_held success | SSIL success | SSIL delta |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 300 | 86.00% | 84.67% | -1.33 pp |
| 200 | 300 | 73.67% | 74.67% | +1.00 pp |
| 300 | 300 | 57.67% | 67.33% | +9.67 pp |
| 400 | 275 | 46.18% | 58.55% | +12.36 pp |
| 500 | 275 | 33.82% | 52.36% | +18.55 pp |
| 600 | 250 | 37.20% | 46.80% | +9.60 pp |
| 700 | 250 | 34.80% | 40.00% | +5.20 pp |
| 800 | 250 | 33.20% | 40.00% | +6.80 pp |
| 900 | 250 | 32.80% | 38.80% | +6.00 pp |
| 1000 | 250 | 26.00% | 36.80% | +10.80 pp |
