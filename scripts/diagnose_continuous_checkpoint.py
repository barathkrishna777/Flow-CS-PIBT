import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from main_pys.dataset_continuous import ContinuousFlowDataset
from main_pys.dataset_continuous_preprocessed import PreprocessedContinuousShardDataset
from main_pys.model_inputs import align_continuous_aux_features
from scripts.eval_continuous import initial_flow_state, load_model


def _reshape_velocity_targets(targets: torch.Tensor, velocity_dim: int) -> tuple[torch.Tensor, int]:
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    targets = targets.reshape(targets.shape[0], -1).float()
    if targets.shape[1] % velocity_dim != 0:
        raise ValueError(
            f"Velocity target width {targets.shape[1]} is not divisible by velocity_dim={velocity_dim}"
        )
    return targets, targets.shape[1] // velocity_dim


def _clip_velocity_rows(velocities: np.ndarray, max_speed: float) -> np.ndarray:
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    scale = np.maximum(norms / max(max_speed, 1e-6), 1.0)
    return velocities / scale


def _batch_string_attr(batch, attr_name: str, num_graphs: int) -> List[str]:
    values = getattr(batch, attr_name, None)
    if values is None:
        return [""] * num_graphs
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)] * num_graphs


def _split_batch_rows(batch) -> Iterable[tuple[int, int, str]]:
    ptr = batch.ptr.cpu().tolist()
    map_names = _batch_string_attr(batch, "map_name", len(ptr) - 1)
    for graph_idx in range(len(ptr) - 1):
        yield ptr[graph_idx], ptr[graph_idx + 1], map_names[graph_idx]


def _prev_velocities_from_batch(batch, max_speed: float) -> np.ndarray:
    prev = getattr(batch, "prev_velocities", None)
    if prev is not None:
        return prev.detach().cpu().numpy().astype(np.float32)
    aux = getattr(batch, "aux_features", None)
    if aux is None or aux.shape[1] < 2:
        raise ValueError("Unable to recover previous velocities from batch")
    return (aux[:, -2:].detach().cpu().numpy().astype(np.float32) * float(max_speed))


def _accumulate_metric(store: Dict[str, List[float]], key: str, values: np.ndarray) -> None:
    if values.size:
        store[key].extend(float(v) for v in values.tolist())


def _goal_cosine(velocities: np.ndarray, goal_delta: np.ndarray) -> np.ndarray:
    vel_norm = np.linalg.norm(velocities, axis=1)
    goal_norm = np.linalg.norm(goal_delta, axis=1)
    moving = (vel_norm > 1e-6) & (goal_norm > 1e-6)
    if not np.any(moving):
        return np.zeros(0, dtype=np.float32)
    vel_dir = velocities[moving] / np.maximum(vel_norm[moving, None], 1e-6)
    goal_dir = goal_delta[moving] / np.maximum(goal_norm[moving, None], 1e-6)
    return np.sum(vel_dir * goal_dir, axis=1).astype(np.float32)


def _progress(positions: np.ndarray, goals: np.ndarray, velocities: np.ndarray, dt: float, max_speed: float) -> np.ndarray:
    current_dist = np.linalg.norm(goals - positions, axis=1)
    next_positions = positions + _clip_velocity_rows(velocities, max_speed) * dt
    next_dist = np.linalg.norm(goals - next_positions, axis=1)
    return (current_dist - next_dist).astype(np.float32)


def _summaries(values: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, raw in values.items():
        arr = np.asarray(raw, dtype=np.float32)
        out[key] = float(arr.mean()) if arr.size else 0.0
    return out


def resolve_target_velocity_source(args: argparse.Namespace, checkpoint: Dict[str, object]) -> str:
    if args.target_velocity_source is not None:
        return str(args.target_velocity_source)
    data_cfg = checkpoint.get("dataset_config", {}) or {}
    return str(data_cfg.get("target_velocity_source", "executed"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose continuous checkpoint vs expert targets")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=None, help="Raw continuous .npz rollout directory")
    parser.add_argument("--preprocessed-dir", default=None, help="Preprocessed continuous shard directory")
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--expert-sources", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--flow-init-mode", choices=["randn", "zeros", "prev"], default="randn")
    parser.add_argument("--target-velocity-source", choices=["executed", "tracker-preferred"], default=None)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if bool(args.data_dir) == bool(args.preprocessed_dir):
        raise ValueError("Pass exactly one of --data-dir or --preprocessed-dir")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model, checkpoint = load_model(args.model_path, device)
    model.eval()
    model_cfg = checkpoint.get("model_config", {}) or {}
    data_cfg = checkpoint.get("dataset_config", {}) or {}
    target_velocity_source = resolve_target_velocity_source(args, checkpoint)

    velocity_dim = int(getattr(model, "velocity_dim", 2))
    chunk_horizon = int(getattr(model, "chunk_horizon", 1))
    max_speed = float(data_cfg.get("max_speed", 1.0))
    wait_threshold = float(data_cfg.get("wait_threshold", 0.1))

    if args.preprocessed_dir:
        if target_velocity_source != "executed":
            raise ValueError(
                "Preprocessed continuous shards only store executed targets; "
                "use --data-dir when diagnosing tracker-preferred supervision."
            )
        dataset = PreprocessedContinuousShardDataset(
            preprocessed_dir=args.preprocessed_dir,
            map_dir=args.map_dir,
            expert_sources=args.expert_sources,
            scenario_ids=args.scenario_ids,
            scenario_start=args.scenario_start,
            scenario_end=args.scenario_end,
        )
    else:
        dataset = ContinuousFlowDataset(
            data_dir=args.data_dir,
            map_dir=args.map_dir,
            k=int(model_cfg.get("k", 4)),
            m=int(model_cfg.get("m", 5)),
            wait_threshold=wait_threshold,
            max_speed=max_speed,
            chunk_horizon=chunk_horizon,
            target_velocity_source=target_velocity_source,
            expert_sources=args.expert_sources,
            scenario_ids=args.scenario_ids,
            scenario_start=args.scenario_start,
            scenario_end=args.scenario_end,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    per_map: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    overall: Dict[str, List[float]] = defaultdict(list)
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        batch = batch.to(device)
        batch = align_continuous_aux_features(batch, getattr(model, "aux_feature_dim", None))
        x1, parsed_chunk_horizon = _reshape_velocity_targets(batch.y, velocity_dim)
        if parsed_chunk_horizon != chunk_horizon:
            raise ValueError(
                f"Dataset chunk_horizon={parsed_chunk_horizon} does not match model chunk_horizon={chunk_horizon}"
            )

        n_agents = x1.shape[0]
        flow_dim = velocity_dim * chunk_horizon
        prev_velocities_np = _prev_velocities_from_batch(batch, max_speed=max_speed)

        with torch.no_grad():
            dt = 1.0 / max(args.num_integration_steps, 1)
            v = initial_flow_state(
                args.flow_init_mode,
                n_agents,
                flow_dim,
                device,
                prev_velocities_np,
                chunk_horizon,
                velocity_dim,
                max_speed,
            )
            for step in range(args.num_integration_steps):
                t = torch.full((n_agents, 1), step * dt, device=device)
                flow = model(v, t, batch)
                v = v + flow * dt

        pred_chunk = (v.reshape(n_agents, chunk_horizon, velocity_dim).detach().cpu().numpy() * max_speed).astype(np.float32)
        target_chunk = (x1.reshape(n_agents, chunk_horizon, velocity_dim).detach().cpu().numpy() * max_speed).astype(np.float32)
        positions = batch.positions.detach().cpu().numpy().astype(np.float32)
        goals = batch.goals.detach().cpu().numpy().astype(np.float32)
        goal_delta = goals - positions

        pred_first = pred_chunk[:, 0, :]
        target_first = target_chunk[:, 0, :]
        pred_speed = np.linalg.norm(pred_first, axis=1)
        target_speed = np.linalg.norm(target_first, axis=1)
        pred_progress = _progress(positions, goals, pred_first, dt=float(data_cfg.get("dt", 0.2)), max_speed=max_speed)
        target_progress = _progress(positions, goals, target_first, dt=float(data_cfg.get("dt", 0.2)), max_speed=max_speed)
        pred_goal_cos = _goal_cosine(pred_first, goal_delta)
        target_goal_cos = _goal_cosine(target_first, goal_delta)

        pred_vs_target_cos_per_agent = np.full(n_agents, np.nan, dtype=np.float32)
        target_speed_mask = target_speed >= wait_threshold
        if np.any(target_speed_mask):
            pred_vs_target_cos_per_agent[target_speed_mask] = np.sum(
                (pred_first[target_speed_mask] / np.maximum(pred_speed[target_speed_mask, None], 1e-6))
                * (target_first[target_speed_mask] / np.maximum(target_speed[target_speed_mask, None], 1e-6)),
                axis=1,
            ).astype(np.float32)

        per_agent_mse = np.mean((pred_first - target_first) ** 2, axis=1).astype(np.float32)
        later_pred_speed = np.linalg.norm(pred_chunk[:, 1:, :].reshape(-1, velocity_dim), axis=1) if chunk_horizon > 1 else np.zeros(0, dtype=np.float32)
        later_target_speed = np.linalg.norm(target_chunk[:, 1:, :].reshape(-1, velocity_dim), axis=1) if chunk_horizon > 1 else np.zeros(0, dtype=np.float32)

        for start, end, map_name in _split_batch_rows(batch):
            store = per_map[map_name]
            _accumulate_metric(store, "pred_first_speed_mean", pred_speed[start:end])
            _accumulate_metric(store, "expert_first_speed_mean", target_speed[start:end])
            _accumulate_metric(store, "pred_first_progress_mean", pred_progress[start:end])
            _accumulate_metric(store, "expert_first_progress_mean", target_progress[start:end])
            _accumulate_metric(store, "pred_first_goal_cosine_mean", _goal_cosine(pred_first[start:end], goal_delta[start:end]))
            _accumulate_metric(store, "expert_first_goal_cosine_mean", _goal_cosine(target_first[start:end], goal_delta[start:end]))
            _accumulate_metric(
                store,
                "pred_vs_expert_first_cosine_mean",
                pred_vs_target_cos_per_agent[start:end][np.isfinite(pred_vs_target_cos_per_agent[start:end])],
            )
            _accumulate_metric(store, "first_step_mse_mean", per_agent_mse[start:end])

        _accumulate_metric(overall, "pred_first_speed_mean", pred_speed)
        _accumulate_metric(overall, "expert_first_speed_mean", target_speed)
        _accumulate_metric(overall, "pred_first_progress_mean", pred_progress)
        _accumulate_metric(overall, "expert_first_progress_mean", target_progress)
        _accumulate_metric(overall, "pred_first_goal_cosine_mean", pred_goal_cos)
        _accumulate_metric(overall, "expert_first_goal_cosine_mean", target_goal_cos)
        _accumulate_metric(
            overall,
            "pred_vs_expert_first_cosine_mean",
            pred_vs_target_cos_per_agent[np.isfinite(pred_vs_target_cos_per_agent)],
        )
        _accumulate_metric(overall, "first_step_mse_mean", per_agent_mse)
        _accumulate_metric(overall, "pred_later_speed_mean", later_pred_speed)
        _accumulate_metric(overall, "expert_later_speed_mean", later_target_speed)

    summary_rows = []
    summary_rows.append(
        {
            "map": "__overall__",
            "target_velocity_source": target_velocity_source,
            **_summaries(overall),
        }
    )
    for map_name in sorted(per_map):
        summary_rows.append(
            {
                "map": map_name,
                "target_velocity_source": target_velocity_source,
                **_summaries(per_map[map_name]),
            }
        )

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    for row in summary_rows:
        print(row["map"])
        for key in fieldnames:
            if key == "map":
                continue
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, (int, float)):
                print(f"  {key}: {float(value):.6f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
