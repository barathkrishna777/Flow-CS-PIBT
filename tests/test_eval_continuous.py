from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_continuous import (
    finalize_preferred_velocity_stats,
    init_preferred_velocity_stats,
    initial_flow_state,
    update_preferred_velocity_stats,
)


def test_initial_flow_state_prev_repeats_previous_velocity_across_chunk():
    prev_velocities = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    state = initial_flow_state(
        init_mode="prev",
        n_agents=2,
        flow_dim=4,
        device=torch.device("cpu"),
        prev_velocities=prev_velocities,
        chunk_horizon=2,
        velocity_dim=2,
        max_speed=2.0,
    )

    expected = torch.tensor(
        [
            [0.5, -0.5, 0.5, -0.5],
            [0.25, 0.125, 0.25, 0.125],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(state.cpu(), expected)


def test_preferred_velocity_stats_capture_first_step_progress_and_later_speed():
    env = SimpleNamespace(
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        goals=np.array([[1.0, 0.0]], dtype=np.float32),
        dt=0.2,
        max_speed=1.0,
        goal_tolerance=0.25,
    )
    velocity_chunk = np.array([[[1.0, 0.0], [0.5, 0.0]]], dtype=np.float32)

    stats = init_preferred_velocity_stats()
    update_preferred_velocity_stats(stats, env, velocity_chunk)
    summary = finalize_preferred_velocity_stats(stats)

    assert summary["preferred_first_speed_mean"] == 1.0
    assert summary["preferred_first_speed_p95"] == 1.0
    assert np.isclose(summary["preferred_first_progress_mean"], 0.2)
    assert np.isclose(summary["preferred_first_goal_cosine_mean"], 1.0)
    assert summary["preferred_later_speed_mean"] == 0.5
