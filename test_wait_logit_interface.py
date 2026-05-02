"""Smoke test for FlowGNNModel learned wait-logit interface.

Run: python test_wait_logit_interface.py
No MAPF data or CUDA required. If PyTorch/PyG are not installed, the test
skips cleanly so local lightweight environments can still run syntax checks.
"""

PASS = 0
FAIL = 0
SKIP = 0


def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def report_skip(name: str, detail: str = ""):
    global SKIP
    SKIP += 1
    msg = f"[SKIP] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def main() -> int:
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError as exc:
        report_skip("wait-logit shape smoke", f"missing optional dependency: {exc.name}")
        print(f"\nSummary: {PASS} passed, {FAIL} failed, {SKIP} skipped")
        return 0

    from main_pys.generative_model import (
        FlowGNNModel,
        binary_gate_action_probs_from_velocity,
        hybrid_action_logits_from_velocity,
    )

    torch.manual_seed(7)
    n_agents = 3
    k = 2
    patch = 2 * k + 1
    data = Data(
        x=torch.randn(n_agents, 3, patch, patch),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        aux_features=torch.randn(n_agents, 5),
        batch=torch.zeros(n_agents, dtype=torch.long),
    )
    model = FlowGNNModel(k=k, hidden_dim=32, num_layers=1)
    model.eval()

    v_t = torch.randn(n_agents, 2)
    t = torch.full((n_agents, 1), 0.99)
    with torch.no_grad():
        flow, action_logits, wait_logit = model(
            v_t,
            t,
            data,
            return_action_logits=True,
            return_wait_logit=True,
        )
        hybrid_logits = hybrid_action_logits_from_velocity(flow, wait_logit)
        model_hybrid_logits = model.hybrid_action_logits(flow, wait_logit)
        gate_probs = binary_gate_action_probs_from_velocity(flow, wait_logit, tau=0.3)
        gate_probs_unclipped = binary_gate_action_probs_from_velocity(flow, wait_logit, tau=0.3, eps=0.0)
        model_gate_probs = model.binary_gate_action_probs(flow, wait_logit, tau=0.3)

    report("flow output shape", tuple(flow.shape) == (n_agents, 2), str(tuple(flow.shape)))
    report("action head shape", tuple(action_logits.shape) == (n_agents, 5), str(tuple(action_logits.shape)))
    report("wait logit shape", tuple(wait_logit.shape) == (n_agents,), str(tuple(wait_logit.shape)))
    report("hybrid logits shape", tuple(hybrid_logits.shape) == (n_agents, 5), str(tuple(hybrid_logits.shape)))
    report("binary gate probs shape", tuple(gate_probs.shape) == (n_agents, 5), str(tuple(gate_probs.shape)))
    report("model calibration default scale", torch.allclose(model.wait_logit_scale.detach(), torch.tensor(1.0)))
    report("model calibration default bias", torch.allclose(model.wait_logit_bias.detach(), torch.tensor(0.0)))
    report("model movement default scale", torch.allclose(model.movement_logit_scale.detach(), torch.tensor(1.0)))
    report("model calibrated logits default to identity", torch.allclose(model_hybrid_logits, hybrid_logits))
    report("model gate probs default to identity helper", torch.allclose(model_gate_probs, gate_probs))

    expected_moves = flow @ torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=flow.dtype).T
    report(
        "hybrid movement logits preserve dot products",
        torch.allclose(hybrid_logits[:, 1:], expected_moves),
    )

    calibrated = hybrid_action_logits_from_velocity(
        flow,
        wait_logit,
        wait_logit_scale=torch.tensor(2.0),
        wait_logit_bias=torch.tensor(-3.0),
        movement_logit_scale=torch.tensor(0.5),
    )
    report(
        "hybrid wait calibration applies scale and bias",
        torch.allclose(calibrated[:, 0], 2.0 * wait_logit - 3.0),
    )
    report(
        "hybrid movement calibration applies scale",
        torch.allclose(calibrated[:, 1:], 0.5 * expected_moves),
    )

    report("binary gate probs sum to one", torch.allclose(gate_probs.sum(dim=1), torch.ones(n_agents)))
    report(
        "binary gate wait prob is sigmoid wait logit",
        torch.allclose(gate_probs_unclipped[:, 0], torch.sigmoid(wait_logit), atol=1e-6),
    )
    report(
        "binary gate move mass is one minus wait",
        torch.allclose(gate_probs_unclipped[:, 1:].sum(dim=1), 1.0 - gate_probs_unclipped[:, 0], atol=1e-6),
    )
    with torch.no_grad():
        model.movement_logit_scale.fill_(9.0)
        gate_probs_after_move_scale = model.binary_gate_action_probs(flow, wait_logit, tau=0.3)
    report(
        "binary gate ignores five-logit movement calibration",
        torch.allclose(gate_probs_after_move_scale, gate_probs),
    )

    targets = torch.tensor([0, 1, 4], dtype=torch.long)
    ce = torch.nn.functional.cross_entropy(model_hybrid_logits, targets)
    report("hybrid logits support action CE", torch.isfinite(ce).item(), str(float(ce)))

    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in {"wait_logit_scale", "wait_logit_bias", "movement_logit_scale"}
    }
    fresh = FlowGNNModel(k=k, hidden_dim=32, num_layers=1)
    incompatible = fresh.load_state_dict(legacy_state, strict=False)
    expected_missing = {"wait_logit_scale", "wait_logit_bias", "movement_logit_scale"}
    report(
        "legacy checkpoints miss only calibration params",
        set(incompatible.missing_keys) == expected_missing,
        str(incompatible.missing_keys),
    )
    report("legacy load keeps wait scale default", torch.allclose(fresh.wait_logit_scale.detach(), torch.tensor(1.0)))
    report("legacy load keeps wait bias default", torch.allclose(fresh.wait_logit_bias.detach(), torch.tensor(0.0)))
    report("legacy load keeps movement scale default", torch.allclose(fresh.movement_logit_scale.detach(), torch.tensor(1.0)))

    print(f"\nSummary: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
