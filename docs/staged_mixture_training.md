# Staged Mixture Training Scaffold

This branch sets up a future curriculum experiment for the coordination-heavy
FLOMAP failure modes. It is intentionally parked until the velocity-to-action
interface and wait-action sensitivity analyses are complete.

## Hypothesis

FLOMAP currently loses most of its learned-policy gap on dense random and maze
bottleneck maps. A staged mixture may help the model learn rare coordination
behaviors first, then recover broad generalization with full-distribution
training.

## Proposed Curriculum

1. **Stage A: coordination-heavy warm start**
   - Train briefly on a subset dominated by `maze-128-128-2`,
     `random-32-32-10`, and `random-64-64-10`.
   - Goal: force the trunk to see waiting/yielding/bottleneck negotiation early.

2. **Stage B: mixed curriculum**
   - Resume from Stage A on `coordination subset + full base dataset`.
   - Goal: keep coordination examples frequent while reintroducing broad maps.

3. **Stage C: full-distribution polish**
   - Resume from Stage B on the full base dataset only.
   - Goal: reduce specialization and recover city/open/general behavior.

## Lambda Entry Point

```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
bash scripts/run_lambda_staged_mixture_training.sh
```

The script is a starting point, not a final experiment protocol. Before using it
for reportable results, decide the exact epoch/step budget and evaluate after
each stage on the full Rishi-8 panel.

## Success Criteria

Use the current `wave9_compact` result as the baseline. A useful staged-mixture
checkpoint should:

- improve the coordination-stress slice (`random-32-32-10` and
  `maze-128-128-2`);
- preserve non-stress success within about 1 percentage point;
- avoid reducing overall Rishi-8 success relative to the current FLOMAP
  checkpoint;
- keep the same velocity-to-action inference settings unless the experiment is
  explicitly paired with a separate interface sweep.

## Notes

`preprocess_dataset.py` now supports `--include-maps`, which lets us build a
coordination-heavy preprocessed directory without changing the base dataset.
