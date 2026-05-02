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

    report("flow output shape", tuple(flow.shape) == (n_agents, 2), str(tuple(flow.shape)))
    report("action head shape", tuple(action_logits.shape) == (n_agents, 5), str(tuple(action_logits.shape)))
    report("wait logit shape", tuple(wait_logit.shape) == (n_agents,), str(tuple(wait_logit.shape)))
    report("hybrid logits shape", tuple(hybrid_logits.shape) == (n_agents, 5), str(tuple(hybrid_logits.shape)))

    expected_moves = flow @ torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=flow.dtype).T
    report(
        "hybrid movement logits preserve dot products",
        torch.allclose(hybrid_logits[:, 1:], expected_moves),
    )

    print(f"\nSummary: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
