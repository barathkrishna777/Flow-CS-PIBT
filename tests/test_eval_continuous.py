from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval_continuous import (
    finalize_executed_velocity_stats,
    finalize_preferred_velocity_stats,
    init_executed_velocity_stats,
    init_preferred_velocity_stats,
    initial_flow_state,
    update_executed_velocity_stats,
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


def test_executed_velocity_stats_capture_shield_projection_gap():
    pre_positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    post_positions = np.array([[0.1, 0.05], [1.0, 0.0]], dtype=np.float32)
    env = SimpleNamespace(
        positions=post_positions,
        goals=np.array([[1.0, 0.0], [1.5, 0.0]], dtype=np.float32),
        max_speed=1.0,
        goal_tolerance=0.25,
        history_velocities=[
            np.array([[0.5, 0.25], [0.0, 0.0]], dtype=np.float32),
        ],
    )
    preferred_first = np.array([[1.0, 0.0], [0.8, 0.0]], dtype=np.float32)

    stats = init_executed_velocity_stats()
    update_executed_velocity_stats(stats, env, pre_positions, preferred_first)
    summary = finalize_executed_velocity_stats(stats)

    assert np.isclose(summary["executed_first_speed_mean"], (np.sqrt(0.5 ** 2 + 0.25 ** 2) + 0.0) / 2)
    # Only the active agent (0) contributes to progress; agent 1 is at goal within tolerance? dist=0.5>0.25 so both active.
    # pre_dist: agent0=1.0, agent1=0.5. post_dist: agent0=sqrt(0.81+0.0025)=0.9014, agent1=0.5.
    expected_progress = ((1.0 - np.sqrt((1.0 - 0.1) ** 2 + 0.05 ** 2)) + 0.0) / 2
    assert np.isclose(summary["executed_first_progress_mean"], expected_progress, atol=1e-5)
    # Only agent 0 is moving, cosine with goal dir (1,0) of vel (0.5,0.25) ≈ 0.894
    expected_cos = 0.5 / np.sqrt(0.5 ** 2 + 0.25 ** 2)
    assert np.isclose(summary["executed_first_goal_cosine_mean"], expected_cos, atol=1e-5)
    # Only agent 0 has both preferred and executed moving; preferred dir (1,0), exec dir normalized
    assert np.isclose(summary["preferred_vs_executed_cosine_mean"], expected_cos, atol=1e-5)
    # Projection magnitudes: agent0 |(1,0)-(0.5,0.25)| = sqrt(0.25+0.0625), agent1 |(0.8,0)-(0,0)| = 0.8
    expected_proj = (np.sqrt(0.25 + 0.0625) + 0.8) / 2
    assert np.isclose(summary["preferred_vs_executed_projection_mean"], expected_proj, atol=1e-5)
