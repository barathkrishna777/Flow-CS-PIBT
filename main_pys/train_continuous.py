import argparse
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from main_pys.continuous_env import ORCAStyleShield
from main_pys.dataset_continuous import ContinuousFlowDataset, build_continuous_weighted_sampler
from main_pys.dataset_continuous_preprocessed import (
    PreprocessedContinuousShardDataset,
    build_preprocessed_continuous_weighted_sampler,
)
from main_pys.generative_model import FlowGNNModel
from main_pys.model_inputs import align_continuous_aux_features
from main_pys.transformer_model import FlowTransformerModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _batch_string_attr(batch, attr_name: str, num_graphs: int) -> List[str]:
    values = getattr(batch, attr_name, None)
    if values is None:
        return [""] * num_graphs
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)] * num_graphs


def _shield_project_batch(predicted_velocity, batch, map_cache, max_speed, shield_type, dt, agent_radius):
    ptr = batch.ptr.cpu().tolist()
    num_graphs = len(ptr) - 1
    map_names = _batch_string_attr(batch, "map_name", num_graphs)
    shielded_chunks = []
    for graph_idx in range(num_graphs):
        start = ptr[graph_idx]
        end = ptr[graph_idx + 1]
        positions = batch.positions[start:end].detach().cpu().numpy()
        preferred = predicted_velocity[start:end].detach().cpu().numpy() * max_speed
        obstacle_map = map_cache.get(map_names[graph_idx])
        if obstacle_map is None:
            obstacle_map = np.zeros((48, 48), dtype=np.int8)
        shield = ORCAStyleShield(
            agent_radius=agent_radius,
            max_speed=max_speed,
            dt=dt,
        )
        projected = shield.project(
            positions,
            preferred,
            obstacle_map,
            use_true_orca=(shield_type == "orca"),
        )
        shielded_chunks.append(torch.from_numpy(projected / max(max_speed, 1e-6)))
    return torch.cat(shielded_chunks, dim=0).to(predicted_velocity.device, dtype=predicted_velocity.dtype)


def _prepare_batch_for_model(batch, model, device):
    batch = batch.to(device)
    return align_continuous_aux_features(batch, getattr(model, "aux_feature_dim", None))


def _reshape_velocity_targets(targets: torch.Tensor, velocity_dim: int) -> tuple[torch.Tensor, int]:
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    targets = targets.reshape(targets.shape[0], -1).float()
    if targets.shape[1] % velocity_dim != 0:
        raise ValueError(
            f"Velocity target width {targets.shape[1]} is not divisible by velocity_dim={velocity_dim}"
        )
    return targets, targets.shape[1] // velocity_dim


def _compute_action_loss(action_logits, action_labels, weights, action_dim: int):
    if action_labels.dim() == 1:
        action_targets = action_labels.unsqueeze(1)
    else:
        action_targets = action_labels.reshape(action_labels.shape[0], -1)

    action_chunk_horizon = action_logits.shape[1] // action_dim
    if action_logits.shape[1] % action_dim != 0:
        raise ValueError(
            f"Action logits width {action_logits.shape[1]} is not divisible by action_dim={action_dim}"
        )
    if action_targets.shape[1] != action_chunk_horizon:
        raise ValueError(
            f"Action target horizon {action_targets.shape[1]} does not match logits horizon {action_chunk_horizon}"
        )

    flat_logits = action_logits.reshape(-1, action_chunk_horizon, action_dim).reshape(-1, action_dim)
    flat_targets = action_targets.reshape(-1)
    per_step_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(-1, action_chunk_horizon)
    return (per_step_loss * weights.expand(-1, action_chunk_horizon)).mean()


def compute_flow_loss(model, batch, device, use_amp, args, map_cache):
    batch = _prepare_batch_for_model(batch, model, device)
    velocity_dim = int(getattr(model, "velocity_dim", 2))
    action_dim = int(getattr(model, "action_dim", args.num_directions + 1))
    x_1, chunk_horizon = _reshape_velocity_targets(batch.y, velocity_dim)
    weights = batch.node_weights.float().view(-1, 1)

    num_graphs = int(batch.batch.max().item()) + 1
    t_per_graph = torch.sigmoid(torch.randn(num_graphs, 1, device=device)).clamp(0.01, 0.99)
    t = t_per_graph[batch.batch]
    x_0 = torch.randn_like(x_1)
    x_t = t * x_1 + (1.0 - t) * x_0

    with torch.cuda.amp.autocast(enabled=use_amp):
        predicted_flow, action_logits = model(x_t, t, batch, return_action_logits=True)
        target_flow = x_1 - x_0
        flow_loss = (F.mse_loss(predicted_flow, target_flow, reduction="none") * weights).mean()

        predicted_velocity = x_t + (1.0 - t) * predicted_flow
        shield_loss = torch.tensor(0.0, device=device)
        if args.shield_aware_loss:
            predicted_velocity_steps = predicted_velocity.reshape(-1, chunk_horizon, velocity_dim)
            target_velocity_steps = x_1.reshape(-1, chunk_horizon, velocity_dim)
            predicted_first = predicted_velocity_steps[:, 0]
            target_first = target_velocity_steps[:, 0]
            shielded_velocity = _shield_project_batch(
                predicted_velocity=predicted_first,
                batch=batch,
                map_cache=map_cache,
                max_speed=args.max_speed,
                shield_type=args.train_shield_type,
                dt=args.dt,
                agent_radius=args.agent_radius,
            )
            shielded_velocity = predicted_first + (shielded_velocity - predicted_first).detach()
            shield_loss = (F.mse_loss(shielded_velocity, target_first, reduction="none") * weights).mean()

        action_loss = _compute_action_loss(action_logits, batch.action_label, weights, action_dim)

        total_loss = (
            args.flow_loss_weight * flow_loss
            + args.shield_loss_weight * shield_loss
            + args.action_loss_weight * action_loss
        )
    return total_loss


def compute_discrete_loss(model, batch, device, use_amp):
    batch = _prepare_batch_for_model(batch, model, device)
    n = batch.action_label.shape[0]
    velocity_dim = int(getattr(model, "velocity_dim", 2))
    chunk_horizon = int(getattr(model, "chunk_horizon", 1))
    action_dim = int(getattr(model, "action_dim", 5))
    zero_v = torch.zeros(n, velocity_dim * chunk_horizon, device=device)
    zero_t = torch.zeros(n, 1, device=device)
    weights = batch.node_weights.float().view(-1, 1)

    with torch.cuda.amp.autocast(enabled=use_amp):
        _, action_logits = model(zero_v, zero_t, batch, return_action_logits=True)
        loss = _compute_action_loss(action_logits, batch.action_label, weights, action_dim)
    return loss


def validate(model, loader, device, use_amp, policy_type, args, map_cache):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            if policy_type == "flow":
                loss = compute_flow_loss(model, batch, device, use_amp, args, map_cache)
            else:
                loss = compute_discrete_loss(model, batch, device, use_amp)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def train(args):
    if hasattr(torch.multiprocessing, "set_sharing_strategy"):
        torch.multiprocessing.set_sharing_strategy("file_system")

    set_seed(args.seed)
    if getattr(args, "chunk_horizon", 1) > 1 and getattr(args, "model_type", "gnn") != "transformer":
        raise ValueError("--chunk-horizon > 1 is currently supported only with --model-type transformer")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    data_loader_generator = torch.Generator().manual_seed(args.seed)

    base_dataset = None
    dataset_cls = PreprocessedContinuousShardDataset if args.preprocessed_dir else ContinuousFlowDataset
    dataset_kwargs = dict(
        map_dir=args.map_dir,
        expert_sources=args.expert_sources,
    )
    if args.preprocessed_dir:
        if args.target_velocity_source != "executed":
            raise ValueError(
                "--target-velocity-source tracker-preferred currently requires raw .npz data; "
                "existing preprocessed shards only store executed-velocity supervision."
            )
        dataset_kwargs["preprocessed_dir"] = args.preprocessed_dir
    else:
        dataset_kwargs.update(
            data_dir=args.data_dir,
            k=args.k,
            m=args.m,
            num_directions=args.num_directions,
            wait_threshold=args.wait_threshold,
            max_speed=args.max_speed,
            chunk_horizon=args.chunk_horizon,
            target_velocity_source=args.target_velocity_source,
        )

    preload = getattr(args, "preload_shards", False) and args.preprocessed_dir is not None
    val_preprocessed_dir = getattr(args, "val_preprocessed_dir", None)
    if args.val_scenario_start is not None or args.val_scenario_end is not None or args.val_scenario_ids:
        train_dataset = dataset_cls(
            **dataset_kwargs,
            scenario_ids=args.train_scenario_ids,
            scenario_start=args.train_scenario_start,
            scenario_end=args.train_scenario_end,
            **({"preload": True} if preload and args.preprocessed_dir else {}),
        )
        # If a separate val shard dir was provided, use it (ignoring scenario filters
        # since those were baked in during preprocessing).
        if val_preprocessed_dir and args.preprocessed_dir:
            val_kwargs = dict(dataset_kwargs)
            val_kwargs["preprocessed_dir"] = val_preprocessed_dir
            if preload:
                val_kwargs["preload"] = True
            val_dataset = PreprocessedContinuousShardDataset(**val_kwargs)
        else:
            val_dataset = dataset_cls(
                **dataset_kwargs,
                scenario_ids=args.val_scenario_ids,
                scenario_start=args.val_scenario_start,
                scenario_end=args.val_scenario_end,
            )
        map_cache = train_dataset.maps
    else:
        base_dataset = dataset_cls(
            **dataset_kwargs,
            **({"preload": True} if preload and args.preprocessed_dir else {}),
        )
        train_indices, val_indices = base_dataset.split_indices_by_rollout(args.val_split, args.seed)
        train_dataset = Subset(base_dataset, train_indices)
        val_dataset = Subset(base_dataset, val_indices) if val_indices else None
        map_cache = base_dataset.maps

    train_size = len(train_dataset)
    val_size = len(val_dataset) if val_dataset is not None else 0
    if train_size <= 0:
        raise RuntimeError("Training dataset is empty after applying filters")

    sample_graph = train_dataset[0]
    sample_input_channels = int(sample_graph.x.shape[1])
    sample_aux_dim = int(getattr(sample_graph, "aux_features").shape[1])
    sample_target_dim = int(sample_graph.y.shape[1] if sample_graph.y.dim() > 1 else sample_graph.y.shape[0])
    sample_chunk_horizon = sample_target_dim // 2
    if sample_target_dim % 2 != 0:
        raise RuntimeError(f"Unexpected target width {sample_target_dim}; expected a multiple of 2")
    if sample_chunk_horizon != int(args.chunk_horizon):
        raise RuntimeError(
            f"Dataset emits chunk_horizon={sample_chunk_horizon}, but --chunk-horizon={args.chunk_horizon}. "
            "Rebuild raw/preprocessed continuous data for the requested chunk horizon."
        )

    sampler = None
    if not args.no_weighted_sampling:
        sampler_base = base_dataset if isinstance(train_dataset, Subset) else train_dataset
        sampler_subset = train_dataset if isinstance(train_dataset, Subset) else None
        sampler_builder = (
            build_preprocessed_continuous_weighted_sampler
            if isinstance(sampler_base, PreprocessedContinuousShardDataset)
            else build_continuous_weighted_sampler
        )
        sampler = sampler_builder(
            sampler_base,
            subset=sampler_subset,
            balance_agent_counts=True,
            balance_expert_sources=True,
            oversample_difficult=args.oversample_difficult,
        )

    batch_size = args.batch_size or (128 if device.type == "cuda" else 16)
    workers = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 2)
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker if workers > 0 else None,
        "generator": data_loader_generator,
    }
    if workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**train_loader_kwargs)
    val_batch_size = getattr(args, "val_batch_size", None) or batch_size
    # Val workers: default to 0 (single-threaded) to avoid workers each holding
    # a full shard (~1.7 GB) in RAM while the main process runs the forward pass.
    val_workers = getattr(args, "val_num_workers", 0)
    val_loader = None
    if val_dataset is not None:
        val_loader_kwargs = {
            "dataset": val_dataset,
            "batch_size": val_batch_size,
            "shuffle": False,
            "num_workers": val_workers,
            "pin_memory": (device.type == "cuda"),
            "persistent_workers": val_workers > 0,
            "worker_init_fn": seed_worker if val_workers > 0 else None,
            "generator": data_loader_generator,
        }
        if val_workers > 0:
            val_loader_kwargs["prefetch_factor"] = 2
        val_loader = DataLoader(**val_loader_kwargs)

    model_type = getattr(args, "model_type", "gnn")
    if model_type == "transformer":
        model = FlowTransformerModel(
            k=args.k,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=getattr(args, "num_heads", 8),
            num_input_channels=sample_input_channels,
            aux_feature_dim=sample_aux_dim,
            action_dim=args.num_directions + 1,
            chunk_horizon=getattr(args, "chunk_horizon", 1),
        ).to(device)
    else:
        model = FlowGNNModel(
            k=args.k,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_input_channels=sample_input_channels,
            aux_feature_dim=sample_aux_dim,
            action_dim=args.num_directions + 1,
        ).to(device)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Loaded init checkpoint "
            f"{args.init_checkpoint} | missing={len(missing)} unexpected={len(unexpected)}"
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = f"continuous_{args.policy_type}_{args.run_name}_" if args.run_name else f"continuous_{args.policy_type}_"
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            if args.policy_type == "flow":
                loss = compute_flow_loss(model, batch, device, use_amp, args, map_cache)
            else:
                loss = compute_discrete_loss(model, batch, device, use_amp)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train = total_loss / max(count, 1)
        val_loss = validate(model, val_loader, device, use_amp, args.policy_type, args, map_cache) if val_loader else avg_train
        print(f"Epoch {epoch + 1}: train={avg_train:.4f} val={val_loss:.4f}")

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "policy_type": args.policy_type,
            "model_type": model_type,
            "model_config": {
                "k": args.k,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": getattr(args, "num_heads", 8),
                "num_input_channels": sample_input_channels,
                "aux_feature_dim": sample_aux_dim,
                "action_dim": args.num_directions + 1,
                "velocity_dim": 2,
                "chunk_horizon": getattr(args, "chunk_horizon", 1),
            },
            "dataset_config": {
                "num_directions": args.num_directions,
                "wait_threshold": args.wait_threshold,
                "max_speed": args.max_speed,
                "expert_sources": args.expert_sources,
                "chunk_horizon": args.chunk_horizon,
                "target_velocity_source": args.target_velocity_source,
                "preprocessed_dir": args.preprocessed_dir,
                "val_preprocessed_dir": args.val_preprocessed_dir,
                "preload_shards": bool(args.preload_shards),
                "train_scenario_ids": args.train_scenario_ids,
                "train_scenario_start": args.train_scenario_start,
                "train_scenario_end": args.train_scenario_end,
                "val_scenario_ids": args.val_scenario_ids,
                "val_scenario_start": args.val_scenario_start,
                "val_scenario_end": args.val_scenario_end,
            },
            "loss_config": {
                "shield_aware_loss": args.shield_aware_loss,
                "train_shield_type": args.train_shield_type,
                "flow_loss_weight": args.flow_loss_weight,
                "shield_loss_weight": args.shield_loss_weight,
                "action_loss_weight": args.action_loss_weight,
            },
            "seed": args.seed,
            "train_loss": avg_train,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"{prefix}epoch_{epoch + 1}.pt"))
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, f"{prefix}best.pt"))

    print(f"Completed training | train_samples={train_size} val_samples={val_size} best_val={best_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train continuous MAPF models")
    parser.add_argument("--data-dir", required=True, help="Directory of continuous .npz files")
    parser.add_argument("--map-dir", required=True, help="Directory of .map files")
    parser.add_argument("--policy-type", choices=["flow", "discrete"], default="flow")
    parser.add_argument("--model-type", choices=["gnn", "transformer"], default="gnn")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to load before training/fine-tuning")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8, help="Transformer attention heads (transformer model only)")
    parser.add_argument("--chunk-horizon", type=int, default=1, help="Number of future steps to predict (action chunking)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument(
        "--target-velocity-source",
        choices=["executed", "tracker-preferred"],
        default="executed",
        help=(
            "Continuous supervision target. "
            "'executed' matches shielded realized velocities in the dataset. "
            "'tracker-preferred' uses pre-shield tracker preferred velocities when present."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--oversample-difficult", action="store_true")
    parser.add_argument("--expert-sources", nargs="*", default=None)
    parser.add_argument("--train-scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--train-scenario-start", type=int, default=None)
    parser.add_argument("--train-scenario-end", type=int, default=None)
    parser.add_argument("--val-scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--val-scenario-start", type=int, default=None)
    parser.add_argument("--val-scenario-end", type=int, default=None)
    parser.add_argument("--shield-aware-loss", action="store_true")
    parser.add_argument("--train-shield-type", choices=["orca", "heuristic-orca"], default="orca")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--shield-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--preprocessed-dir", default=None, help="Directory of compact continuous shard files")
    parser.add_argument("--val-preprocessed-dir", default=None, help="Separate shard directory for validation (e.g. built with different scenario range)")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Batch size for validation (defaults to --batch-size; reduce for large-agent val sets)")
    parser.add_argument("--val-num-workers", type=int, default=0, help="DataLoader workers for validation (default 0 to avoid per-worker shard RAM overhead)")
    parser.add_argument("--preload-shards", action="store_true", help="Load all needed shards into RAM at startup")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
