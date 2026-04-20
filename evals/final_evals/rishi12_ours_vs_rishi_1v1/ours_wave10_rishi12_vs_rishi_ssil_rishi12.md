# Grid Eval 1v1 Comparison

- A: `Ours wave10 rishi12` from `evals/final_evals/rishi12_wave10_newdata_clean_20260416_224144.csv`
- B: `Rishi SSIL rishi12` from `evals/final_evals/rishi12_ssil_classifier.csv`
- Cost-per-agent column: `total_cost_not_resting_at_goal`
- Figure: `evals/final_evals/rishi12_ours_vs_rishi_1v1/ours_wave10_rishi12_vs_rishi_ssil_rishi12.png`

## Overall

| Run | Rows | Success | Mean agents at goal | Avg runtime | Avg cost / agent |
| --- | --- | --- | --- | --- | --- |
| Ours wave10 rishi12 | 2700 | 47.48% (1282/2700) | 88.79% | 32.73s | 302.59 |
| Rishi SSIL rishi12 | 2700 | 55.22% (1491/2700) | 94.83% | 18.62s | 232.45 |

## Overall Delta

| Metric | Rishi SSIL rishi12 - Ours wave10 rishi12 | Winner |
| --- | --- | --- |
| Success | +7.74 pp | Rishi SSIL rishi12 |
| Mean agents at goal | +6.03 pp | Rishi SSIL rishi12 |
| Avg runtime | -14.11s | Rishi SSIL rishi12 |
| Avg cost / agent | -70.14 | Rishi SSIL rishi12 |

## Per Map

| Map | Ours wave10 rishi12 success | Rishi SSIL rishi12 success | Success delta | Ours wave10 rishi12 at-goal | Rishi SSIL rishi12 at-goal | Ours wave10 rishi12 runtime | Rishi SSIL rishi12 runtime | Ours wave10 rishi12 cost/agent | Rishi SSIL rishi12 cost/agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Berlin_1_256 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | 28.22s | 13.23s | 218.24 | 184.81 |
| Paris_1_256 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | 30.76s | 15.07s | 230.16 | 194.68 |
| den312d | 31.20% | 32.00% | +0.80 pp | 73.51% | 81.79% | 19.53s | 11.24s | 218.72 | 189.83 |
| empty-32-32 | 76.80% | 100.00% | +23.20 pp | 98.39% | 100.00% | 6.91s | 2.16s | 65.26 | 42.45 |
| empty-48-48 | 92.40% | 100.00% | +7.60 pp | 99.86% | 100.00% | 11.84s | 4.85s | 69.82 | 52.03 |
| maze-128-128-2 | 25.60% | 42.00% | +16.40 pp | 65.11% | 96.78% | 105.09s | 71.02s | 1431.13 | 968.53 |
| maze-32-32-4 | 12.00% | 14.67% | +2.67 pp | 81.30% | 88.62% | 9.40s | 3.29s | 198.88 | 160.61 |
| random-64-64-10 | 66.40% | 85.20% | +18.80 pp | 99.91% | 99.97% | 15.65s | 7.29s | 80.40 | 61.33 |
| random-64-64-20 | 22.00% | 42.00% | +20.00 pp | 95.17% | 99.75% | 19.73s | 10.45s | 138.62 | 87.55 |
| room-64-64-16 | 15.60% | 27.20% | +11.60 pp | 63.34% | 80.14% | 28.95s | 15.73s | 342.63 | 279.63 |
| warehouse-10-20-10-2-1 | 4.40% | 2.40% | -2.00 pp | 91.06% | 91.47% | 29.37s | 16.10s | 202.74 | 197.56 |
| warehouse-20-40-10-2-1 | 13.20% | 11.20% | -2.00 pp | 97.41% | 97.64% | 58.02s | 34.02s | 243.22 | 225.06 |

## Per Agent Count

| Agents | Ours wave10 rishi12 success | Rishi SSIL rishi12 success | Success delta | Ours wave10 rishi12 runtime | Rishi SSIL rishi12 runtime | Ours wave10 rishi12 cost/agent | Rishi SSIL rishi12 cost/agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 86.00% | 84.67% | -1.33 pp | 9.76s | 2.04s | 175.06 | 140.37 |
| 200 | 73.67% | 74.67% | +1.00 pp | 13.80s | 3.61s | 210.75 | 158.88 |
| 300 | 57.67% | 67.33% | +9.67 pp | 18.83s | 6.13s | 256.53 | 181.28 |
| 400 | 46.18% | 58.55% | +12.36 pp | 25.52s | 10.14s | 301.96 | 195.12 |
| 500 | 33.82% | 52.36% | +18.55 pp | 30.43s | 15.11s | 334.06 | 215.85 |
| 600 | 37.20% | 46.80% | +9.60 pp | 35.67s | 21.54s | 365.01 | 251.59 |
| 700 | 34.80% | 40.00% | +5.20 pp | 42.25s | 28.02s | 357.24 | 275.84 |
| 800 | 33.20% | 40.00% | +6.80 pp | 46.66s | 31.73s | 360.61 | 298.81 |
| 900 | 32.80% | 38.80% | +6.00 pp | 53.99s | 36.46s | 358.15 | 320.88 |
| 1000 | 26.00% | 36.80% | +10.80 pp | 62.44s | 41.41s | 356.53 | 334.59 |
