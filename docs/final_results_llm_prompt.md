# Strong LLM Prompt for Paper Drafting and Visualization Planning

Use this prompt with a strong LLM to draft paper sections, LaTeX tables, figure plans, captions, and visualization code/design guidance.

```text
You are helping write an ICRA-level robotics/AI paper about continuous-space Multi-Agent Path Finding (MAPF). The method combines a learned rectified-flow GNN velocity policy with a continuous extension of PIBT called EPIBTShield.

System:
- FlowGNNModel predicts preferred velocities.
- At evaluation time, preferred velocities are passed to EPIBTShield.
- EPIBTShield resolves conflicts using priority inheritance and backtracking in continuous velocity space.
- Default learned policy uses 3 Euler integration steps.
- Main eval flags: --policy flow --shield-type epibt.
- Metrics:
  - AtGoal: fraction of agents reaching goals.
  - Coll: cumulative pair-timestep collision count.
  - ArrPLR: path length ratio only over agents that arrived.
  - PLR over all agents is less paper-friendly because failed agents can wander.

Evaluation sets:
- Set A: random-32-32-10 and empty-48-48, N=50 and N=100, 25 scenarios each.
- Set B: random-64-64-10 and room-32-32-4, N=50, 25 scenarios each.
- Set C: warehouse-10-20-10-2-1, N=50 and N=100, 25 scenarios each.
- Main horizon: 512 environment steps; also evaluated 256-step budget.
- Main model: continuous v4b.
- Multi-seed robustness: seeds 42, 123, 456.

Core Set A hard-case result, random-32-32-10, N=100, 512 steps:
- ORCA: AtGoal 0.168, Coll 4168.9, ArrPLR 1.010, ArrStep 434.1.
- PO-ORCA: AtGoal 0.149, Coll 2626.3, ArrPLR 1.096, ArrStep 443.0.
- Straight+EPIBTShield: AtGoal 0.774, Coll 1306.9, ArrPLR 1.122, ArrStep 187.5.
- Flow v4b + ORCA: AtGoal 0.486, Coll 3996.0, ArrPLR 1.380, ArrStep 330.6.
- Flow v4b + EPIBTShield: AtGoal 0.874, Coll 633.4, ArrPLR 1.462, ArrStep 167.6.

Main interpretation:
- ORCA fails in dense/cluttered random maps.
- PO-ORCA reduces collisions somewhat but does not improve completion.
- Straight+EPIBTShield demonstrates that priority inheritance and backtracking are the core mechanism: 16.8% -> 77.4% AtGoal on random N=100.
- Flow+ORCA demonstrates learned preferred velocities alone are insufficient with a weak shield: only 48.6% AtGoal.
- Flow+EPIBT demonstrates the combination is strongest: 87.4% AtGoal, 84.8% fewer collisions than ORCA, and +10.0 percentage points over Straight+EPIBT.

Set A average summary:
- ORCA: N50 AtGoal 0.579, Coll 492.9; N100 AtGoal 0.584, Coll 2101.4.
- PO-ORCA: N50 0.576, Coll 334.7; N100 0.575, Coll 1462.6.
- Straight+EPIBT: N50 0.889, Coll 105.7; N100 0.886, Coll 675.8.
- Flow+EPIBT 256: N50 0.842, Coll 49.4; N100 0.813, Coll 290.1.
- Flow+EPIBT 512: N50 0.927, Coll 98.4; N100 0.921, Coll 354.9.

Integration-step ablation:
- 3 steps is the default and best overall.
- AtGoal for Flow+EPIBT at 3/5/10/20 steps:
  - empty N50: 0.989 / 0.984 / 0.982 / 0.984
  - empty N100: 0.968 / 0.961 / 0.941 / 0.956
  - random N50: 0.866 / 0.893 / 0.856 / 0.870
  - random N100: 0.874 / 0.879 / 0.856 / 0.868
- ArrPLR worsens as steps increase:
  - random N100: 1.462 at 3 steps, 1.592 at 5, 1.644 at 10, 1.659 at 20.
- Conclusion: more integration steps do not reliably improve AtGoal and increase route inefficiency.

Multi-seed robustness, Set A 512:
- empty N50 AtGoal: seed42 0.984, seed123 0.965, seed456 0.974, mean 0.974, std 0.008.
- empty N100: 0.981, 0.948, 0.926, mean 0.952, std 0.023.
- random N50: 0.868, 0.781, 0.801, mean 0.817, std 0.037.
- random N100: 0.866, 0.789, 0.782, mean 0.812, std 0.038.
- Caveat: seed42 is strongest. Seeds 123/456 still slightly beat Straight+EPIBT on random N=100, but the learned gain is modest for retrained seeds.

Set B generalization:
- random-64-64-10 N50:
  - ORCA AtGoal 0.048, Coll 383.3, ArrPLR 0.992.
  - Straight+EPIBT AtGoal 0.505, Coll 267.2, ArrPLR 1.051.
  - Flow+EPIBT AtGoal 0.614, Coll 106.0, ArrPLR 1.391.
- room-32-32-4 N50:
  - ORCA AtGoal 0.012, Coll 1028.4.
  - Straight+EPIBT AtGoal 0.114, Coll 2019.6.
  - Flow+EPIBT AtGoal 0.262, Coll 1113.1.

Set C OOD warehouse:
- warehouse N50:
  - ORCA AtGoal 0.132, Coll 855.3.
  - Straight+EPIBT AtGoal 0.411, Coll 29.8.
  - Flow+EPIBT AtGoal 0.255, Coll 57.4.
- warehouse N100:
  - ORCA AtGoal 0.125, Coll 3495.9.
  - Straight+EPIBT AtGoal 0.394, Coll 167.7.
  - Flow+EPIBT AtGoal 0.217, Coll 320.0.
- Interpretation: EPIBTShield generalizes strongly OOD, but the learned flow prior does not improve warehouse behavior because warehouse layouts were absent from training. Frame this as a domain adaptation limitation and a motivation for future training data expansion.

v4 vs v4b at 256 steps:
- v4b beats v4 on all Set A map/N combinations:
  - empty N50: 0.850 vs 0.744.
  - empty N100: 0.812 vs 0.680.
  - random N50: 0.835 vs 0.740.
  - random N100: 0.814 vs 0.730.

Please produce:
1. A concise ICRA-style experimental results section.
2. A main ablation table and a generalization table in LaTeX.
3. A paragraph explaining why EPIBTShield is the primary algorithmic contribution.
4. A paragraph explaining the learned flow contribution.
5. A limitations paragraph that honestly discusses warehouse OOD degradation and multi-seed variance.
6. A publication-quality visualization plan and, if code is requested, plotting code that can produce the figures.

Visualization requirements:
- Do not make generic AI-looking plots, decorative dashboards, cluttered rainbow charts, or low-information grouped bars by default.
- Every figure must answer a specific scientific question in one glance.
- Prefer clean, paper-native visual encodings: small multiples, slope charts, dumbbell plots, connected dot plots, annotated deltas, and compact heatmaps when they clarify comparisons.
- Use restrained, colorblind-safe palettes. Suggested semantic color mapping:
  - ORCA: neutral gray.
  - PO-ORCA: muted slate.
  - Straight+EPIBTShield: teal or blue-green.
  - Flow+ORCA: amber or muted orange.
  - Flow+EPIBT: deep blue or high-emphasis indigo.
- Keep backgrounds white, gridlines subtle, labels direct, and legends minimal. Prefer direct labeling when possible.
- Avoid 3D plots, gradient fills, glossy effects, unnecessary icons, oversized titles, and decorative shapes.
- Use consistent scales across comparable panels so gains are visually honest.
- Include uncertainty where available: use scenario std/error bars for per-method aggregates and seed spread for multi-seed summaries.
- Put the key numeric takeaway directly on the figure as a concise annotation, e.g. "+70.6 pp AtGoal vs ORCA" or "84.8% fewer collisions".
- Show both completion and safety/efficiency tradeoffs, not only AtGoal.
- Make figures readable in grayscale print and at two-column paper width.
- Use font sizes appropriate for conference papers: no tiny axis labels; no labels that overlap.
- Prefer vector outputs, such as PDF/SVG, and high-DPI PNG only for previews.

Recommended figures:

Figure 1: Mechanism Decomposition on Set A Hard Case
- Question: What contributes what?
- Data: random-32-32-10, N=100, 512 steps.
- Recommended design: horizontal connected dot/slope plot for AtGoal, ordered ORCA -> PO-ORCA -> Straight+EPIBT -> Flow+ORCA -> Flow+EPIBT. Add collision count as a second aligned panel below, preferably log-scaled or annotated directly.
- Must show:
  - ORCA 0.168
  - Straight+EPIBT 0.774
  - Flow+ORCA 0.486
  - Flow+EPIBT 0.874
- Key annotation: "Shield: +60.6 pp over ORCA; learning on shield: +10.0 pp; ORCA shield with learned flow remains poor."

Figure 2: Generalization and OOD Behavior
- Question: Where does learning help, and where does the shield alone generalize?
- Data: random64 N50, room N50, warehouse N50, warehouse N100.
- Recommended design: small-multiple slope charts or dumbbell charts per map, comparing ORCA, Straight+EPIBT, and Flow+EPIBT. Each panel should share a 0-1 AtGoal scale.
- Must make the warehouse reversal obvious without looking like a failure of the whole method:
  - random64: 0.048 -> 0.505 -> 0.614
  - room: 0.012 -> 0.114 -> 0.262
  - warehouse N50: 0.132 -> 0.411 -> 0.255
  - warehouse N100: 0.125 -> 0.394 -> 0.217
- Key annotation: "EPIBTShield transfers OOD; learned flow needs warehouse-like training data."

Figure 3: Integration Steps Tradeoff
- Question: Why use 3 Euler steps?
- Data: 3/5/10/20 steps for AtGoal and ArrPLR.
- Recommended design: two-row small multiple. Top row AtGoal by steps; bottom row ArrPLR by steps. Use one line per map/N setting with subtle colors or split into random vs empty panels.
- Must highlight that higher integration steps do not reliably improve AtGoal and increase ArrPLR.
- Key annotation: "3 steps offers the best completion-efficiency tradeoff."

Figure 4: Multi-Seed Robustness
- Question: Is training stable?
- Recommended design: compact dot plot with one row per map/N setting. Show each seed as a small dot, mean as a larger mark, and a horizontal interval spanning min-max or +/- std.
- Must expose, not hide, that seed 42 is strongest on random maps.
- Key annotation: "random N=100: 0.812 +/- 0.038 AtGoal across 3 seeds."

Figure 5: Qualitative Trajectory Figure
- Question: What does the behavior look like?
- Recommended design: same scenario shown across ORCA, Straight+EPIBT, Flow+EPIBT. Use map obstacles in light gray, trajectories in transparent lines, starts/goals with subtle markers, and collisions/deadlock region annotated sparingly.
- Avoid spaghetti: choose a representative scenario, draw at most a manageable subset of agents or use opacity and endpoint emphasis.
- Caption should explain behavior, not just describe colors.

Tone:
- Rigorous, precise, and not overclaiming safety.
- Avoid saying collision-free guarantee unless assumptions are explicitly stated, because empirical collision counts are nonzero.
- Write like a robotics paper, not a marketing page.
```
