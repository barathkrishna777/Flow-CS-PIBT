import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
import contextlib
import math
import random

from main_pys.dataset import FlowMAPFDataset
from main_pys.dataset_preprocessed import PreprocessedFlowMAPFDataset, build_weighted_sampler
from main_pys.generative_model import FlowGNNModel
from main_pys.lattice_primitives import CARDINAL_TO_LATTICE, NUM_PRIMITIVES

PREPROCESSED_DIRS = [
    "/media/anushree_mattlab/Seagate Por/preprocessed_data",  # external drive (primary)
    "data/preprocessed",                                       # local fallback
]


WAIT_SPEED_THRESHOLD = 0.1  # velocities below this magnitude → wait action
ACTION_VECTORS = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype=torch.float32)  # right, down, up, left


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def setup_distributed(distributed=False, local_rank=None):
    """Initialize torch.distributed when launched by torchrun.

    Normal `python -m main_pys.train_flow ...` runs are intentionally left alone.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    inferred = world_size > 1
    enabled = distributed or inferred

    if local_rank is None:
        local_rank = _env_int("LOCAL_RANK", 0)

    if enabled:
        if world_size <= 1:
            raise ValueError("--distributed requires torchrun with WORLD_SIZE > 1")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training currently requires CUDA GPUs")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def cleanup_distributed(ddp_info):
    if ddp_info["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def reduce_average(value, device, ddp_info):
    if not ddp_info["enabled"]:
        return value
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / ddp_info["world_size"]).item()


@contextlib.contextmanager
def suppress_stdout(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        yield


class DistributedWeightedSampler(Sampler):
    """WeightedRandomSampler-style sampling sharded across DDP ranks."""

    def __init__(self, weights, num_samples, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples_global = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = replacement
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.num_samples_global / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def dataset_indices_in_full(dataset, full_dataset):
    """Resolve indices for nested Subset/random_split datasets."""
    if dataset is full_dataset:
        return torch.arange(len(full_dataset), dtype=torch.long)
    if isinstance(dataset, Subset):
        parent_indices = dataset_indices_in_full(dataset.dataset, full_dataset)
        subset_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return parent_indices[subset_indices]
    raise TypeError("Weighted sampling only supports the full dataset or torch.utils.data.Subset")


def velocity_to_action_labels(expert_velocities, device):
    """Convert expert velocity vectors to discrete action labels (0-4).

    Actions: 0=wait, 1=right, 2=down, 3=up, 4=left
    """
    speeds = expert_velocities.norm(dim=1)
    is_wait = speeds < WAIT_SPEED_THRESHOLD

    # Dot product with cardinal directions for non-wait agents
    dots = expert_velocities @ ACTION_VECTORS.to(device).T  # (N, 4)
    best_dir = dots.argmax(dim=1) + 1  # +1 because action 0 is wait

    labels = torch.where(is_wait, torch.zeros_like(best_dir), best_dir)
    return labels


def _reduce_node_loss(loss, node_weights, unweighted, extra_weights=None):
    weights = torch.ones_like(node_weights) if unweighted else node_weights
    if extra_weights is not None:
        weights = weights * extra_weights
    return (loss * weights.squeeze(1)).mean()


def _stress_node_weights(batch, threshold, multiplier, device, dtype):
    if multiplier == 1.0:
        return None
    if not hasattr(batch, "batch") or batch.batch is None:
        return None
    batch_index = batch.batch.to(device)
    if batch_index.numel() == 0:
        return None
    num_graphs = int(batch_index.max().item()) + 1
    graph_counts = torch.bincount(batch_index, minlength=num_graphs)
    graph_multipliers = torch.ones(num_graphs, device=device, dtype=dtype)
    graph_multipliers = torch.where(
        graph_counts.to(device) >= threshold,
        torch.full_like(graph_multipliers, float(multiplier)),
        graph_multipliers,
    )
    return graph_multipliers[batch_index].view(-1, 1)


def wait_ranking_loss_from_logits(ranking_logits, expert_actions, margin=0.5):
    """Pairwise wait-vs-move ranking loss for scores [wait, right, down, up, left]."""
    wait_score = ranking_logits[:, 0]
    movement_scores = ranking_logits[:, 1:]
    is_wait = expert_actions == 0

    wait_target_loss = F.softplus(movement_scores.max(dim=1).values + margin - wait_score)
    move_indices = (expert_actions - 1).clamp(min=0)
    preferred_move_score = movement_scores.gather(1, move_indices.view(-1, 1)).squeeze(1)
    move_target_loss = F.softplus(wait_score + margin - preferred_move_score)
    return torch.where(is_wait, wait_target_loss, move_target_loss)


def compute_flow_loss(
    model,
    batch,
    device,
    use_amp,
    action_loss_weight=0.3,
    wait_head_loss_weight=0.0,
    hybrid_action_loss_weight=0.0,
    hybrid_velocity_source="teacher_x1",
    wait_ranking_loss_weight=0.0,
    wait_ranking_margin=0.5,
    wait_ranking_velocity_source="teacher_x1",
    stress_loss_multiplier=1.0,
    stress_agent_threshold=300,
    unweighted_action_loss=False,
    unweighted_flow_loss=False,
    discrete_forward_mode="both",
    lattice_loss_weight=0.0,
):
    """Shared flow matching loss computation for train and val."""
    batch = batch.to(device)
    x_1 = batch.y.view(-1, 2)
    if hasattr(batch, "node_weights") and batch.node_weights is not None:
        node_weights = batch.node_weights.view(-1, 1)
    else:
        node_weights = torch.ones(x_1.shape[0], 1, device=device)

    num_graphs = batch.batch.max().item() + 1
    t_per_graph = torch.sigmoid(torch.randn(num_graphs, 1, device=device))
    t_per_graph = t_per_graph.clamp(0.01, 0.99)
    t = t_per_graph[batch.batch]
    x_0 = torch.randn_like(x_1)
    x_t = t * x_1 + (1 - t) * x_0

    with torch.cuda.amp.autocast(enabled=use_amp):
        need_action_loss = action_loss_weight > 0
        need_wait_head_loss = wait_head_loss_weight > 0
        need_hybrid_loss = hybrid_action_loss_weight > 0
        need_wait_ranking_loss = wait_ranking_loss_weight > 0
        need_wait_logit = need_wait_head_loss or need_hybrid_loss or need_wait_ranking_loss
        need_lattice_loss = lattice_loss_weight > 0
        need_discrete_loss = need_action_loss or need_wait_logit
        use_shared_discrete = need_discrete_loss and discrete_forward_mode in ("shared", "both")

        def velocity_arg(active, source):
            if not active:
                return None
            if source == "teacher_x1":
                return x_1
            if source == "predicted_x1":
                return None
            raise ValueError(f"Unknown velocity source: {source}")

        def unpack_discrete_forward(outputs):
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            idx = 0
            flow = outputs[idx]
            idx += 1
            logits = None
            wait = None
            calibrated_wait = None
            hybrid_logits = None
            ranking_logits = None
            lattice_logits = None
            if need_action_loss:
                logits = outputs[idx]
                idx += 1
            if need_wait_logit:
                wait = outputs[idx]
                idx += 1
            if need_wait_head_loss:
                calibrated_wait = outputs[idx]
                idx += 1
            if need_hybrid_loss:
                hybrid_logits = outputs[idx]
                idx += 1
            if need_wait_ranking_loss:
                ranking_logits = outputs[idx]
                idx += 1
            if need_lattice_loss:
                lattice_logits = outputs[idx]
            return flow, logits, wait, calibrated_wait, hybrid_logits, ranking_logits, lattice_logits

        def forward_discrete_heads(v, t_in):
            outputs = model(
                v,
                t_in,
                batch,
                return_action_logits=need_action_loss,
                return_wait_logit=need_wait_logit,
                return_calibrated_wait_logit=need_wait_head_loss,
                return_hybrid_logits=need_hybrid_loss,
                hybrid_velocity_for_logits=velocity_arg(need_hybrid_loss, hybrid_velocity_source),
                return_wait_ranking_logits=need_wait_ranking_loss,
                wait_ranking_velocity_for_logits=velocity_arg(
                    need_wait_ranking_loss,
                    wait_ranking_velocity_source,
                ),
                return_lattice_logits=need_lattice_loss,
            )
            return unpack_discrete_forward(outputs)

        if use_shared_discrete or need_lattice_loss:
            (
                predicted_flow,
                action_logits,
                wait_logit,
                calibrated_wait_logit,
                hybrid_logits,
                ranking_logits,
                lattice_logits_shared,
            ) = (
                forward_discrete_heads(x_t, t)
            )
        else:
            predicted_flow = model(x_t, t, batch)
            action_logits = None
            wait_logit = None
            calibrated_wait_logit = None
            hybrid_logits = None
            ranking_logits = None
            lattice_logits_shared = None

        target_flow = x_1 - x_0
        base_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
        if unweighted_flow_loss:
            flow_loss = base_loss.mean()
        else:
            flow_loss = (base_loss * node_weights).mean()

        zero = torch.tensor(0.0, device=device)
        action_loss = zero
        wait_head_loss = zero
        hybrid_action_loss = zero
        wait_ranking_loss = zero

        if need_discrete_loss:
            # Auxiliary labels: 0=wait, 1=right, 2=down, 3=up, 4=left.
            if hasattr(batch, "action_y") and batch.action_y is not None:
                expert_actions = batch.action_y.view(-1).long().to(device)
            else:
                expert_actions = velocity_to_action_labels(x_1, device)
            target_wait = (expert_actions == 0).to(dtype=x_1.dtype)
            stress_weights = _stress_node_weights(
                batch,
                stress_agent_threshold,
                stress_loss_multiplier,
                device,
                node_weights.dtype,
            )

            def compute_discrete_losses(logits, calibrated_wait, hybrid, ranking):
                cur_action_loss = zero
                cur_wait_loss = zero
                cur_hybrid_loss = zero
                cur_ranking_loss = zero
                if need_action_loss:
                    ce = F.cross_entropy(logits, expert_actions, reduction='none')
                    cur_action_loss = _reduce_node_loss(ce, node_weights, unweighted_action_loss)
                if need_wait_head_loss:
                    bce = F.binary_cross_entropy_with_logits(
                        calibrated_wait.view(-1),
                        target_wait.to(dtype=calibrated_wait.dtype),
                        reduction='none',
                    )
                    cur_wait_loss = _reduce_node_loss(bce, node_weights, unweighted_action_loss)
                if need_hybrid_loss:
                    hybrid_ce = F.cross_entropy(hybrid, expert_actions, reduction='none')
                    cur_hybrid_loss = _reduce_node_loss(
                        hybrid_ce,
                        node_weights,
                        unweighted_action_loss,
                    )
                if need_wait_ranking_loss:
                    ranking_node_loss = wait_ranking_loss_from_logits(
                        ranking,
                        expert_actions,
                        margin=wait_ranking_margin,
                    )
                    cur_ranking_loss = _reduce_node_loss(
                        ranking_node_loss,
                        node_weights,
                        unweighted_action_loss,
                        extra_weights=stress_weights,
                    )
                return cur_action_loss, cur_wait_loss, cur_hybrid_loss, cur_ranking_loss

            # shared: use logits from the noisy forward (original behaviour)
            if use_shared_discrete:
                action_loss, wait_head_loss, hybrid_action_loss, wait_ranking_loss = compute_discrete_losses(
                    action_logits,
                    calibrated_wait_logit,
                    hybrid_logits,
                    ranking_logits,
                )

            # zero_v: second forward with v=0, t=0 so heads can't read answer from v_t
            if discrete_forward_mode in ("zero_v", "both"):
                zero_v = torch.zeros_like(x_1)
                zero_t = torch.zeros(x_1.shape[0], 1, device=device)
                (
                    flow_zero,
                    action_logits_zero,
                    wait_logit_zero,
                    calibrated_wait_logit_zero,
                    hybrid_logits_zero,
                    ranking_logits_zero,
                    _,
                ) = forward_discrete_heads(zero_v, zero_t)
                action_loss_zero, wait_loss_zero, hybrid_loss_zero, ranking_loss_zero = compute_discrete_losses(
                    action_logits_zero,
                    calibrated_wait_logit_zero,
                    hybrid_logits_zero,
                    ranking_logits_zero,
                )

            # x1_t1: forward with (x_1, t=0.99) — matches integrated inference distribution.
            # The trunk sees (v≈x_1, t≈1) at inference; training on ground-truth x_1 at
            # t=0.99 teaches the discrete heads on the same conditioning.
            if discrete_forward_mode in ("x1_t1",):
                t_high = torch.full((x_1.shape[0], 1), 0.99, device=device)
                (
                    flow_x1,
                    action_logits_x1,
                    wait_logit_x1,
                    calibrated_wait_logit_x1,
                    hybrid_logits_x1,
                    ranking_logits_x1,
                    _,
                ) = forward_discrete_heads(x_1, t_high)
                action_loss_x1, wait_loss_x1, hybrid_loss_x1, ranking_loss_x1 = compute_discrete_losses(
                    action_logits_x1,
                    calibrated_wait_logit_x1,
                    hybrid_logits_x1,
                    ranking_logits_x1,
                )

            if discrete_forward_mode == "zero_v":
                action_loss = action_loss_zero
                wait_head_loss = wait_loss_zero
                hybrid_action_loss = hybrid_loss_zero
                wait_ranking_loss = ranking_loss_zero
            elif discrete_forward_mode == "both":
                action_loss = 0.5 * action_loss + 0.5 * action_loss_zero
                wait_head_loss = 0.5 * wait_head_loss + 0.5 * wait_loss_zero
                hybrid_action_loss = 0.5 * hybrid_action_loss + 0.5 * hybrid_loss_zero
                wait_ranking_loss = 0.5 * wait_ranking_loss + 0.5 * ranking_loss_zero
            elif discrete_forward_mode == "x1_t1":
                action_loss = action_loss_x1
                wait_head_loss = wait_loss_x1
                hybrid_action_loss = hybrid_loss_x1
                wait_ranking_loss = ranking_loss_x1
            # else "shared": keep losses from the noisy forward

        # ── Lattice primitive head loss ──
        lattice_loss = zero
        if lattice_loss_weight > 0 and lattice_logits_shared is not None:
            if hasattr(batch, "lattice_action_y") and batch.lattice_action_y is not None:
                lattice_targets = batch.lattice_action_y.view(-1).long().to(device)
            else:
                cardinal_labels = (
                    batch.action_y.view(-1).long().to(device)
                    if hasattr(batch, "action_y") and batch.action_y is not None
                    else velocity_to_action_labels(x_1, device)
                )
                _c2l = torch.from_numpy(CARDINAL_TO_LATTICE).long().to(device)
                lattice_targets = _c2l[cardinal_labels]

            lattice_ce = F.cross_entropy(lattice_logits_shared, lattice_targets, reduction='none')
            lattice_loss = _reduce_node_loss(lattice_ce, node_weights, unweighted_action_loss)

        loss = (
            flow_loss
            + action_loss_weight * action_loss
            + wait_head_loss_weight * wait_head_loss
            + hybrid_action_loss_weight * hybrid_action_loss
            + wait_ranking_loss_weight * wait_ranking_loss
            + lattice_loss_weight * lattice_loss
        )

    return loss


def validate(
    model,
    val_loader,
    device,
    use_amp,
    action_loss_weight,
    wait_head_loss_weight,
    hybrid_action_loss_weight,
    hybrid_velocity_source,
    wait_ranking_loss_weight,
    wait_ranking_margin,
    wait_ranking_velocity_source,
    stress_loss_multiplier,
    stress_agent_threshold,
    unweighted_action_loss,
    unweighted_flow_loss,
    discrete_forward_mode="both",
    lattice_loss_weight=0.0,
    ddp_info=None,
):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_flow_loss(
                model,
                batch,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                wait_head_loss_weight=wait_head_loss_weight,
                hybrid_action_loss_weight=hybrid_action_loss_weight,
                hybrid_velocity_source=hybrid_velocity_source,
                wait_ranking_loss_weight=wait_ranking_loss_weight,
                wait_ranking_margin=wait_ranking_margin,
                wait_ranking_velocity_source=wait_ranking_velocity_source,
                stress_loss_multiplier=stress_loss_multiplier,
                stress_agent_threshold=stress_agent_threshold,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
                discrete_forward_mode=discrete_forward_mode,
                lattice_loss_weight=lattice_loss_weight,
            )
            total_loss += loss.item()
            num_batches += 1
    if ddp_info and ddp_info["enabled"]:
        stats = torch.tensor([total_loss, num_batches], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        num_batches = int(stats[1].item())
    return total_loss / max(num_batches, 1)


def train(run_name="", quick=False, use_wandb=True, wandb_project="flow-mapf", wandb_entity=None,
          preprocessed_dir=None, no_weighted_sampling=False, val_split=0.05, patience=0,
          max_train_samples=None, max_val_samples=None,
          resume=None, start_epoch=0, hidden_dim=1024, num_layers=6,
          action_loss_weight=0.3, wait_head_loss_weight=0.0, hybrid_action_loss_weight=0.0,
          hybrid_velocity_source="teacher_x1", wait_ranking_loss_weight=0.0,
          wait_ranking_margin=0.5, wait_ranking_velocity_source="teacher_x1",
          stress_loss_multiplier=1.0, stress_agent_threshold=300, unweighted_action_loss=False,
          unweighted_flow_loss=False, epochs=10, discrete_forward_mode="both",
          reset_best_val_loss=False, action_head_only=False, action_head_lr=5e-4,
          wait_head_only=False, wait_head_lr=5e-4, calibration_only=False,
          freeze_movement_logit_scale=False,
          lattice_loss_weight=0.0, lattice_head_only=False, lattice_head_lr=5e-4,
          extra_feature_dim=0,
          distributed=False, local_rank=None, seed=42,
          batch_size=None):
    ddp_info = setup_distributed(distributed=distributed, local_rank=local_rank)
    is_main = ddp_info["is_main"]
    log = print if is_main else (lambda *args, **kwargs: None)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{ddp_info['local_rank']}" if ddp_info["enabled"] else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    log(
        f"Device: {device} | AMP: {use_amp}"
        + (f" | DDP world_size={ddp_info['world_size']}" if ddp_info["enabled"] else "")
    )

    if wait_head_only and wait_head_loss_weight <= 0 and hybrid_action_loss_weight <= 0 and wait_ranking_loss_weight <= 0:
        raise ValueError("--wait-head-only requires a positive wait, hybrid, or ranking loss weight")
    if lattice_head_only and lattice_loss_weight <= 0:
        raise ValueError("--lattice-head-only requires --lattice-loss-weight > 0")
    if calibration_only and (action_head_only or wait_head_only):
        raise ValueError("--calibration-only cannot be combined with --action-head-only or --wait-head-only")
    if calibration_only and wait_head_loss_weight <= 0 and hybrid_action_loss_weight <= 0 and wait_ranking_loss_weight <= 0:
        raise ValueError("--calibration-only requires a positive wait, hybrid, or ranking loss weight")

    # Find preprocessed data: CLI override > external drive > local
    pp_dir = preprocessed_dir
    if pp_dir is None:
        for candidate in PREPROCESSED_DIRS:
            if os.path.isdir(candidate) and len(os.listdir(candidate)) > 0:
                pp_dir = candidate
                break

    with suppress_stdout(not is_main):
        if pp_dir:
            dirs = [d.strip() for d in pp_dir.split(",")] if "," in pp_dir else [pp_dir]
            log(f"Using PREPROCESSED dataset from {' + '.join(dirs)}")
            full_dataset = PreprocessedFlowMAPFDataset(pp_dir)
        else:
            log(f"No preprocessed data found — using on-the-fly dataset (slow)")
            full_dataset = FlowMAPFDataset(data_dir="data/flow_training_data_multi",
                                      map_dir="data/mapf-map",
                                      bd_dir="data/bd_npzs",
                                      k=4, m=5)

    # ── Validation split ──
    val_size = int(len(full_dataset) * val_split) if val_split > 0 else 0
    train_size = len(full_dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        log(f"Train/Val split: {train_size:,} / {val_size:,} ({val_split*100:.0f}%)")
    else:
        train_dataset = full_dataset
        val_dataset = None

    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, range(max_train_samples))
        train_size = len(train_dataset)
        log(f"Smoke subset: train limited to {train_size:,} samples")
    if (
        val_dataset is not None
        and max_val_samples is not None
        and max_val_samples > 0
        and max_val_samples < len(val_dataset)
    ):
        val_dataset = Subset(val_dataset, range(max_val_samples))
        val_size = len(val_dataset)
        log(f"Smoke subset: val limited to {val_size:,} samples")

    # GPU: more workers + bigger batches to keep GPU saturated; CPU: stay conservative
    cpu_cores = min(12, os.cpu_count() or 2) if device.type == "cuda" else min(4, os.cpu_count() or 2)
    if batch_size is None:
        batch_size = 256 if device.type == "cuda" else 32

    # ── Weighted sampling (upsamples high-agent-count scenarios) ──
    sampler = None
    train_sampler = None
    if not no_weighted_sampling and pp_dir and isinstance(full_dataset, PreprocessedFlowMAPFDataset):
        with suppress_stdout(not is_main):
            sampler = build_weighted_sampler(full_dataset)
            train_indices = dataset_indices_in_full(train_dataset, full_dataset)
            train_weights = sampler.weights[train_indices]
        if ddp_info["enabled"]:
            train_sampler = DistributedWeightedSampler(
                weights=train_weights,
                num_samples=len(train_dataset),
                num_replicas=ddp_info["world_size"],
                rank=ddp_info["rank"],
                replacement=True,
                seed=seed,
            )
            sampler = train_sampler
        else:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=train_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
    elif ddp_info["enabled"]:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ddp_info["world_size"],
            rank=ddp_info["rank"],
            shuffle=True,
        )
        sampler = train_sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Don't shuffle when using sampler
        sampler=sampler,
        num_workers=cpu_cores,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = None
    val_sampler = None
    if val_dataset is not None:
        if ddp_info["enabled"]:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=ddp_info["world_size"],
                rank=ddp_info["rank"],
                shuffle=False,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=min(4, cpu_cores),
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
            persistent_workers=True
        )

    model = FlowGNNModel(hidden_dim=hidden_dim, num_layers=num_layers, extra_feature_dim=extra_feature_dim).to(device)
    effective_freeze_movement_logit_scale = freeze_movement_logit_scale or (
        wait_ranking_loss_weight > 0 and hybrid_action_loss_weight <= 0
    )
    if effective_freeze_movement_logit_scale:
        model.movement_logit_scale.requires_grad = False

    if action_head_only or wait_head_only or calibration_only or lattice_head_only:
        trainable_groups = []
        calibration_names = {"wait_logit_scale", "wait_logit_bias"}
        if not effective_freeze_movement_logit_scale:
            calibration_names.add("movement_logit_scale")
        for name, param in model.named_parameters():
            train_action = action_head_only and name.startswith("action_head.")
            train_wait = wait_head_only and name.startswith("wait_head.")
            train_calibration = (wait_head_only or calibration_only) and name in calibration_names
            train_lattice = lattice_head_only and name.startswith("lattice_head.")
            param.requires_grad = train_action or train_wait or train_calibration or train_lattice
        if action_head_only:
            action_params = [p for n, p in model.named_parameters() if n.startswith("action_head.") and p.requires_grad]
            if action_params:
                trainable_groups.append({"params": action_params, "lr": action_head_lr})
        if wait_head_only:
            wait_params = [p for n, p in model.named_parameters() if n.startswith("wait_head.") and p.requires_grad]
            if wait_params:
                trainable_groups.append({"params": wait_params, "lr": wait_head_lr})
        if wait_head_only or calibration_only:
            calibration_params = [p for n, p in model.named_parameters() if n in calibration_names and p.requires_grad]
            if calibration_params:
                trainable_groups.append({"params": calibration_params, "lr": wait_head_lr})
        if lattice_head_only:
            lattice_params = [p for n, p in model.named_parameters() if n.startswith("lattice_head.") and p.requires_grad]
            if lattice_params:
                trainable_groups.append({"params": lattice_params, "lr": lattice_head_lr})
        trainable = [p for group in trainable_groups for p in group["params"]]
        if not trainable:
            raise RuntimeError("No trainable parameters selected")
        log(
            "head_only: freezing trunk, training "
            f"{sum(p.numel() for p in trainable):,} params "
            f"(action_head={action_head_only}, wait_head={wait_head_only}, "
            f"calibration_only={calibration_only}, lattice_head={lattice_head_only})"
        )
        optimizer = AdamW(trainable_groups, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 1 if quick else epochs
    # Scheduler is rebuilt after resume so T_max reflects the actual run length.
    # Placeholder T_max here; overwritten below if resuming.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    # Mixed precision: ~2x throughput on A100 Tensor Cores
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint
    if resume and os.path.exists(resume):
        log(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full checkpoint (model + optimizer + scheduler + metadata)
            incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if incompatible.missing_keys:
                log(f"  Missing checkpoint keys initialized from scratch: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                log(f"  Unexpected checkpoint keys ignored: {incompatible.unexpected_keys}")
            if action_head_only or wait_head_only or calibration_only or lattice_head_only:
                # Optimizer only covers selected head params — skip incompatible full-model state
                start_epoch = ckpt['epoch']
                epochs = start_epoch + epochs
                scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - start_epoch, 1), eta_min=1e-6)
                best_val_loss = float('inf')
                log(f"  Loaded model weights (head_only: skipping optimizer/scheduler state). "
                    f"Resuming from epoch {start_epoch + 1}, running until epoch {epochs}")
            else:
                loaded_optimizer = False
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    loaded_optimizer = True
                except (KeyError, ValueError) as exc:
                    log(f"  Skipping optimizer state (architecture changed): {exc}")
                if loaded_optimizer:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except (KeyError, ValueError) as exc:
                        log(f"  Skipping scheduler state: {exc}")
                    if ckpt.get('scaler_state_dict'):
                        scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt['epoch']  # epoch is already 1-indexed, use as start
                epochs = start_epoch + epochs  # --epochs means additional epochs when resuming
                # Rebuild scheduler so T_max matches the actual number of epochs to run
                scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - start_epoch, 1), eta_min=1e-6)
                best_val_loss = float('inf') if reset_best_val_loss else ckpt.get('best_val_loss', float('inf'))
                state_msg = "full state" if loaded_optimizer else "model weights"
                log(f"  Restored {state_msg}: resuming from epoch {start_epoch + 1}, running until epoch {epochs}, best_val={'reset' if reset_best_val_loss else f'{best_val_loss:.4f}'}")
        else:
            # Legacy checkpoint (model weights only)
            incompatible = model.load_state_dict(ckpt, strict=False)
            if incompatible.missing_keys:
                log(f"  Missing checkpoint keys initialized from scratch: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                log(f"  Unexpected checkpoint keys ignored: {incompatible.unexpected_keys}")
            log(f"  Loaded model weights only (legacy checkpoint). Fast-forwarding scheduler {start_epoch} steps.")
            for _ in range(start_epoch):
                scheduler.step()

    if ddp_info["enabled"]:
        find_unused = (
            action_loss_weight <= 0
            or wait_head_loss_weight <= 0
            or hybrid_action_loss_weight <= 0
            or wait_ranking_loss_weight <= 0
            or lattice_loss_weight <= 0
            or action_head_only
            or wait_head_only
            or calibration_only
            or lattice_head_only
        )
        model = DDP(
            model,
            device_ids=[ddp_info["local_rank"]],
            output_device=ddp_info["local_rank"],
            find_unused_parameters=find_unused,
        )
        # DDP calls _sync_params() (in-place broadcast) at the start of every
        # model.forward(). Two forward calls per backward step (discrete_forward_mode="both")
        # increment parameter version counters between the first forward and backward,
        # causing "modified by an inplace operation" errors. Fall back to single forward.
        if discrete_forward_mode == "both":
            log("[DDP] discrete_forward_mode='both' unsupported under DDP (double forward/single backward); using 'shared'.")
            discrete_forward_mode = "shared"

    # WandB setup
    if use_wandb and is_main:
        import wandb
        wandb_kwargs = {"project": wandb_project, "config": {}}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        wandb.init(**wandb_kwargs)
        wandb.config.update({
            "run_name": run_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dataset_size": len(full_dataset),
            "train_size": train_size,
            "val_size": val_size,
            "num_workers": cpu_cores,
            "quick": quick,
            "seed": seed,
            "use_amp": use_amp,
            "device": str(device),
            "weighted_sampling": sampler is not None,
            "val_split": val_split,
            "patience": patience,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "action_loss_weight": action_loss_weight,
            "wait_head_loss_weight": wait_head_loss_weight,
            "hybrid_action_loss_weight": hybrid_action_loss_weight,
            "hybrid_velocity_source": hybrid_velocity_source,
            "wait_ranking_loss_weight": wait_ranking_loss_weight,
            "wait_ranking_margin": wait_ranking_margin,
            "wait_ranking_velocity_source": wait_ranking_velocity_source,
            "stress_loss_multiplier": stress_loss_multiplier,
            "stress_agent_threshold": stress_agent_threshold,
            "unweighted_action_loss": unweighted_action_loss,
            "unweighted_flow_loss": unweighted_flow_loss,
            "discrete_forward_mode": discrete_forward_mode,
            "action_head_only": action_head_only,
            "wait_head_only": wait_head_only,
            "calibration_only": calibration_only,
            "freeze_movement_logit_scale": effective_freeze_movement_logit_scale,
        })

    log_batch_every = 10
    log(f"Batch size: {batch_size} | Workers: {cpu_cores} | Epochs: {epochs}")
    log(f"Weighted sampling: {'ON' if sampler else 'OFF'}")
    log(
        "Loss weighting: "
        f"flow={'unweighted' if unweighted_flow_loss else 'weighted'}, "
        f"action={'unweighted' if unweighted_action_loss else 'weighted'}, "
        f"action_loss_weight={action_loss_weight}, "
        f"wait_head_loss_weight={wait_head_loss_weight}, "
        f"hybrid_action_loss_weight={hybrid_action_loss_weight}, "
        f"hybrid_velocity_source={hybrid_velocity_source}, "
        f"wait_ranking_loss_weight={wait_ranking_loss_weight}, "
        f"wait_ranking_margin={wait_ranking_margin}, "
        f"wait_ranking_velocity_source={wait_ranking_velocity_source}, "
        f"stress_loss_multiplier={stress_loss_multiplier}, "
        f"stress_agent_threshold={stress_agent_threshold}, "
        f"freeze_movement_logit_scale={effective_freeze_movement_logit_scale}, "
        f"discrete_forward_mode={discrete_forward_mode}"
    )

    # Initialize best_val_loss (may be overridden by resume checkpoint above)
    if 'best_val_loss' not in dir():
        best_val_loss = float('inf')
    if not isinstance(best_val_loss, float) or best_val_loss is None:
        best_val_loss = float('inf')
    epochs_without_improvement = 0
    prefix = f"large_scale_flow_{run_name}_" if run_name else "large_scale_flow_"

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        # ── Training ──
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_grad_norms = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_main)

        for batch_idx, batch in enumerate(pbar):
            loss = compute_flow_loss(
                model,
                batch,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                wait_head_loss_weight=wait_head_loss_weight,
                hybrid_action_loss_weight=hybrid_action_loss_weight,
                hybrid_velocity_source=hybrid_velocity_source,
                wait_ranking_loss_weight=wait_ranking_loss_weight,
                wait_ranking_margin=wait_ranking_margin,
                wait_ranking_velocity_source=wait_ranking_velocity_source,
                stress_loss_multiplier=stress_loss_multiplier,
                stress_agent_threshold=stress_agent_threshold,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
                discrete_forward_mode=discrete_forward_mode,
                lattice_loss_weight=lattice_loss_weight,
            )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient norm (for wandb) before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)

            # Gradient clipping — flow matching targets (x_1 - x_0) can be large
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            if is_main:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Per-batch wandb logging
            if use_wandb and is_main and (batch_idx + 1) % log_batch_every == 0:
                import wandb
                global_step = epoch * len(train_loader) + batch_idx + 1
                wandb.log({"train/batch_loss": loss.item()}, step=global_step)

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        avg_train_loss = reduce_average(avg_train_loss, device, ddp_info)
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        avg_grad_norm = reduce_average(avg_grad_norm, device, ddp_info)

        # ── Validation ──
        val_loss = None
        if val_loader is not None:
            val_loss = validate(
                model,
                val_loader,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                wait_head_loss_weight=wait_head_loss_weight,
                hybrid_action_loss_weight=hybrid_action_loss_weight,
                hybrid_velocity_source=hybrid_velocity_source,
                wait_ranking_loss_weight=wait_ranking_loss_weight,
                wait_ranking_margin=wait_ranking_margin,
                wait_ranking_velocity_source=wait_ranking_velocity_source,
                stress_loss_multiplier=stress_loss_multiplier,
                stress_agent_threshold=stress_agent_threshold,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
                discrete_forward_mode=discrete_forward_mode,
                lattice_loss_weight=lattice_loss_weight,
                ddp_info=ddp_info,
            )
            val_str = f" | Val Loss: {val_loss:.4f}"

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = f"{prefix}best.pt"
                if is_main:
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'model_config': {
                            'hidden_dim': hidden_dim,
                            'num_layers': num_layers,
                        },
                        'loss_config': {
                            'action_loss_weight': action_loss_weight,
                            'wait_head_loss_weight': wait_head_loss_weight,
                            'hybrid_action_loss_weight': hybrid_action_loss_weight,
                            'hybrid_velocity_source': hybrid_velocity_source,
                            'wait_ranking_loss_weight': wait_ranking_loss_weight,
                            'wait_ranking_margin': wait_ranking_margin,
                            'wait_ranking_velocity_source': wait_ranking_velocity_source,
                            'stress_loss_multiplier': stress_loss_multiplier,
                            'stress_agent_threshold': stress_agent_threshold,
                            'freeze_movement_logit_scale': effective_freeze_movement_logit_scale,
                            'discrete_forward_mode': discrete_forward_mode,
                        },
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                    }, best_path)
                val_str += " (best)"
            else:
                epochs_without_improvement += 1
                val_str += f" (no improvement for {epochs_without_improvement} epochs)"
        else:
            val_str = ""

        log(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_str} | LR: {current_lr:.2e}")

        # Save epoch checkpoint (full state for resumability)
        ckpt_path = f"{prefix}epoch_{epoch+1}.pt"
        if is_main:
            raw_model = model.module if hasattr(model, "module") else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'model_config': {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                },
                'loss_config': {
                    'action_loss_weight': action_loss_weight,
                    'wait_head_loss_weight': wait_head_loss_weight,
                    'hybrid_action_loss_weight': hybrid_action_loss_weight,
                    'hybrid_velocity_source': hybrid_velocity_source,
                    'wait_ranking_loss_weight': wait_ranking_loss_weight,
                    'wait_ranking_margin': wait_ranking_margin,
                    'wait_ranking_velocity_source': wait_ranking_velocity_source,
                    'stress_loss_multiplier': stress_loss_multiplier,
                    'stress_agent_threshold': stress_agent_threshold,
                    'freeze_movement_logit_scale': effective_freeze_movement_logit_scale,
                    'discrete_forward_mode': discrete_forward_mode,
                },
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, ckpt_path)

        # Per-epoch wandb logging
        if use_wandb and is_main:
            import wandb
            epoch_step = (epoch + 1) * len(train_loader)
            log_dict = {
                "epoch/train_loss": avg_train_loss,
                "epoch/lr": current_lr,
                "epoch/grad_norm_mean": avg_grad_norm,
                "epoch": epoch + 1,
            }
            if val_loss is not None:
                log_dict["epoch/val_loss"] = val_loss
                log_dict["epoch/best_val_loss"] = best_val_loss
            wandb.log(log_dict, step=epoch_step)
            wandb.save(ckpt_path, base_path=".")

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            log(f"Early stopping: val loss hasn't improved for {patience} epochs.")
            break

    # Final summary
    if use_wandb and is_main:
        import wandb
        wandb.run.summary["final_train_loss"] = avg_train_loss
        if val_loss is not None:
            wandb.run.summary["final_val_loss"] = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["final_checkpoint"] = ckpt_path

    if val_loader is not None:
        log(f"\nBest val loss: {best_val_loss:.4f} (saved to {prefix}best.pt)")

    cleanup_distributed(ddp_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="",
                        help="Suffix for checkpoint names, e.g. wave2 -> large_scale_flow_wave2_epoch_N.pt")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 1 epoch only (for testing)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="flow-mapf",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="flow-cspibt",
                        help="W&B entity/team (default: flow-cspibt)")
    parser.add_argument("--preprocessed-dir", type=str, default=None,
                        help="Preprocessed .pt directory, or comma-separated list (e.g. "
                             "data/preprocessed,data/preprocessed_heldout)")
    parser.add_argument("--no-weighted-sampling", action="store_true",
                        help="Disable agent-count weighted sampling")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of data for validation (0 to disable)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit train samples for smoke tests only (default: use all)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit val samples for smoke tests only (default: use all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. large_scale_flow_wave4_epoch_1.pt)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch to resume from (0-indexed, e.g. 1 means start at epoch 2)")
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="Model hidden dimension (default: 1024)")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of GNN layers (default: 6)")
    parser.add_argument("--action-loss-weight", type=float, default=0.3,
                        help="Weight for auxiliary/exact discrete action cross entropy")
    parser.add_argument("--wait-head-loss-weight", type=float, default=0.0,
                        help="Weight for learned wait-head BCE loss. Default 0 keeps legacy training unchanged")
    parser.add_argument("--hybrid-action-loss-weight", type=float, default=0.0,
                        help="Weight for CE over inference logits [calibrated wait, calibrated movement]")
    parser.add_argument("--hybrid-velocity-source", type=str,
                        choices=["teacher_x1", "predicted_x1"], default="teacher_x1",
                        help="Velocity used for hybrid movement logits: teacher_x1 uses expert x_1; "
                             "predicted_x1 uses x_t + (1-t)*predicted_flow")
    parser.add_argument("--wait-ranking-loss-weight", type=float, default=0.0,
                        help="Weight for planner-aligned wait-vs-move ranking loss. Default 0 disables it")
    parser.add_argument("--wait-ranking-margin", type=float, default=0.5,
                        help="Soft ranking margin between calibrated wait and preferred movement scores. "
                             "Default 0.5 asks for a modest planner-facing gap; use 0 for pure ordering")
    parser.add_argument("--wait-ranking-velocity-source", type=str,
                        choices=["teacher_x1", "predicted_x1"], default="teacher_x1",
                        help="Velocity used for ranking movement scores: teacher_x1 uses expert x_1 first; "
                             "predicted_x1 uses x_t + (1-t)*predicted_flow")
    parser.add_argument("--stress-loss-multiplier", type=float, default=1.0,
                        help="Multiplier for ranking-loss nodes from graphs with agent count >= threshold")
    parser.add_argument("--stress-agent-threshold", type=int, default=300,
                        help="Agent-count threshold for --stress-loss-multiplier")
    parser.add_argument("--unweighted-action-loss", action="store_true",
                        help="Do not apply node_weights to the action/wait-head losses")
    parser.add_argument("--unweighted-flow-loss", action="store_true",
                        help="Do not apply node_weights to the flow MSE loss")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs; --quick still forces 1 epoch")
    parser.add_argument("--discrete-forward-mode", type=str,
                        choices=["shared", "zero_v", "both", "x1_t1"], default="both",
                        help="Action CE forward mode: shared=noisy forward only, zero_v=v=0 t=0 only, "
                             "both=average of shared+zero_v, x1_t1=use ground-truth x_1 at t=0.99 "
                             "(matches integrated inference; best for action_head_only). Default: both")
    parser.add_argument("--reset-best-val-loss", action="store_true",
                        help="Reset best_val_loss to inf on resume (use when loss function changes, e.g. repair retrains)")
    parser.add_argument("--action-head-only", action="store_true",
                        help="Freeze all trunk params, train only action_head (for isolated head repair)")
    parser.add_argument("--action-head-lr", type=float, default=5e-4,
                        help="LR for action_head_only mode (default: 5e-4)")
    parser.add_argument("--wait-head-only", action="store_true",
                        help="Freeze all trunk params, train only wait_head (for learned wait-logit finetuning)")
    parser.add_argument("--wait-head-lr", type=float, default=5e-4,
                        help="LR for wait_head_only mode (default: 5e-4)")
    parser.add_argument("--calibration-only", action="store_true",
                        help="Freeze all model params except wait/movement calibration scalars")
    parser.add_argument("--freeze-movement-logit-scale", action="store_true",
                        help="Keep movement_logit_scale fixed at its checkpoint/default value during training")
    parser.add_argument("--lattice-loss-weight", type=float, default=0.0,
                        help="Weight for lattice primitive classification CE loss (17-class). Default 0 disables it")
    parser.add_argument("--lattice-head-only", action="store_true",
                        help="Freeze trunk, train only lattice_head (17-class). Requires --lattice-loss-weight > 0")
    parser.add_argument("--lattice-head-lr", type=float, default=5e-4,
                        help="LR for lattice_head_only mode (default: 5e-4)")
    parser.add_argument("--extra-feature-dim", type=int, default=0,
                        help="Dim of extra agent features (goal disp + BD dist = 3). 0 disables (default 0)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size (default: 256 on GPU). With DDP, set to 64 for effective batch=256 matching single-GPU.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable DistributedDataParallel; also auto-enabled under torchrun WORLD_SIZE>1")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None,
                        help="Local GPU rank for DDP (torchrun usually provides LOCAL_RANK env var)")
    args = parser.parse_args()
    train(run_name=args.run_name, quick=args.quick, use_wandb=not args.no_wandb,
          wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
          preprocessed_dir=args.preprocessed_dir,
          no_weighted_sampling=args.no_weighted_sampling,
          val_split=args.val_split, patience=args.patience,
          max_train_samples=args.max_train_samples,
          max_val_samples=args.max_val_samples,
          resume=args.resume, start_epoch=args.start_epoch,
          hidden_dim=args.hidden_dim, num_layers=args.num_layers,
          action_loss_weight=args.action_loss_weight,
          wait_head_loss_weight=args.wait_head_loss_weight,
          hybrid_action_loss_weight=args.hybrid_action_loss_weight,
          hybrid_velocity_source=args.hybrid_velocity_source,
          wait_ranking_loss_weight=args.wait_ranking_loss_weight,
          wait_ranking_margin=args.wait_ranking_margin,
          wait_ranking_velocity_source=args.wait_ranking_velocity_source,
          stress_loss_multiplier=args.stress_loss_multiplier,
          stress_agent_threshold=args.stress_agent_threshold,
          unweighted_action_loss=args.unweighted_action_loss,
          unweighted_flow_loss=args.unweighted_flow_loss,
          epochs=args.epochs,
          discrete_forward_mode=args.discrete_forward_mode,
          reset_best_val_loss=args.reset_best_val_loss,
          action_head_only=args.action_head_only,
          action_head_lr=args.action_head_lr,
          wait_head_only=args.wait_head_only,
          wait_head_lr=args.wait_head_lr,
          calibration_only=args.calibration_only,
          freeze_movement_logit_scale=args.freeze_movement_logit_scale,
          lattice_loss_weight=args.lattice_loss_weight,
          lattice_head_only=args.lattice_head_only,
          lattice_head_lr=args.lattice_head_lr,
          extra_feature_dim=args.extra_feature_dim,
          distributed=args.distributed,
          local_rank=args.local_rank,
          seed=args.seed,
          batch_size=args.batch_size)
