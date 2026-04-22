import torch

from main_pys.train_continuous import _chunk_step_weights, _reduce_chunked_step_losses


def test_chunk_step_weights_zero_later_steps_when_first_step_only_enabled():
    weights = _chunk_step_weights(
        chunk_horizon=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        first_step_only_loss=True,
    )

    assert torch.equal(weights, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))


def test_reduce_chunked_step_losses_can_ignore_future_steps():
    per_step_loss = torch.tensor(
        [
            [1.0, 100.0],
            [3.0, 200.0],
        ],
        dtype=torch.float32,
    )
    node_weights = torch.ones(2, 1, dtype=torch.float32)

    full_loss = _reduce_chunked_step_losses(
        per_step_loss,
        node_weights,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    first_step_only_loss = _reduce_chunked_step_losses(
        per_step_loss,
        node_weights,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    assert torch.isclose(full_loss, torch.tensor(76.0, dtype=torch.float32))
    assert torch.isclose(first_step_only_loss, torch.tensor(2.0, dtype=torch.float32))

