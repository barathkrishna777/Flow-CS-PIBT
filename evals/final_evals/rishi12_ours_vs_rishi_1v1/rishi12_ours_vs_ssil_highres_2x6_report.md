# Rishi12 1v1: flow_held (ours) vs SSIL (Rishi's)

## Figures

![Success by map 2x6](rishi12_ours_vs_ssil_success_by_map_2x6_highres.png)

![Summary two-row](rishi12_ours_vs_ssil_summary_2row_highres.png)

## Overall

| Model | Rows | Success | Mean agents at goal | Avg runtime | Avg cost / agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| flow_held (ours) | 2700 | 47.48% | 88.79% | 32.73s | 302.59 |
| SSIL (Rishi's) | 2700 | 55.22% | 94.83% | 18.62s | 232.45 |

## Per Map

| Map | Runs | flow_held success | SSIL success | SSIL - flow_held | flow_held at-goal | SSIL at-goal | flow_held runtime | SSIL runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Berlin_1_256 | 250 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | 28.22s | 13.23s |
| empty-32-32 | 125 | 76.80% | 100.00% | +23.20 pp | 98.39% | 100.00% | 6.91s | 2.16s |
| maze-32-32-4 | 75 | 12.00% | 14.67% | +2.67 pp | 81.30% | 88.62% | 9.40s | 3.29s |
| random-64-64-20 | 250 | 22.00% | 42.00% | +20.00 pp | 95.17% | 99.75% | 19.73s | 10.45s |
| warehouse-20-40-10-2-1 | 250 | 13.20% | 11.20% | -2.00 pp | 97.41% | 97.64% | 58.02s | 34.02s |
| room-64-64-16 | 250 | 15.60% | 27.20% | +11.60 pp | 63.34% | 80.14% | 28.95s | 15.73s |
| Paris_1_256 | 250 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | 30.76s | 15.07s |
| empty-48-48 | 250 | 92.40% | 100.00% | +7.60 pp | 99.86% | 100.00% | 11.84s | 4.85s |
| maze-128-128-2 | 250 | 25.60% | 42.00% | +16.40 pp | 65.11% | 96.78% | 105.09s | 71.02s |
| random-64-64-10 | 250 | 66.40% | 85.20% | +18.80 pp | 99.91% | 99.97% | 15.65s | 7.29s |
| warehouse-10-20-10-2-1 | 250 | 4.40% | 2.40% | -2.00 pp | 91.06% | 91.47% | 29.37s | 16.10s |
| den312d | 250 | 31.20% | 32.00% | +0.80 pp | 73.51% | 81.79% | 19.53s | 11.24s |
