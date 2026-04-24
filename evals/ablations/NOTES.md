# Ablation Notes

| change | checkpoint | policy | conditioning | overall success | mean@goal | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (inference fix) | large_scale_flow_wave8_heldout_best | flow_action_head | integrated | 43.48% (10/23) | 95.93% | 25.81s |
| 1 (inference fix) | large_scale_flow_wave8_heldout_best | flow_action_head | zero_t0 | 0.00% (0/23) | 91.48% | 21.26s |
| baseline | large_scale_flow_wave8_heldout_best | flow | — | 65.22% (15/23) | 94.95% | 27.41s |
| baseline | ssil_model | classifier | — | 69.57% (16/23) | 97.54% | 13.94s |

## Change 1 findings (2026-04-24, quick eval)

**Hypothesis confirmed**: distribution mismatch (v=0, t=0.5 during inference vs v_t=t·x_1+(1-t)·ε during training) was the root cause of 0% action-head success. Replacing the dummy forward with a flow-integrated v≈x_1 at t=0.99 lifts success to 43.48%.

**zero_t0 stays at 0%**: confirms v=0 is always out-of-distribution regardless of t value — the GNN trunk has never seen v=0 in training.

**Remaining 22pp gap vs flow_vector (65%)**: the action head learned to shortcut by reading v_t during training rather than learning true discrete action distributions. The zero-v forward in Change 2 is needed to force the head to learn without the hint.

**Paris_1_256 anomaly (0% with integrated)**: the head is making systematically wrong predictions on this large map even with in-distribution conditioning. Expected to recover after Change 2 repair retrain.

**Decision**: proceed to Change 2 (repair retrain). 43% < 60% stop threshold, so no early promotion to full eval.
