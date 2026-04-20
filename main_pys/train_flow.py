import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import argparse

from main_pys.dataset import FlowMAPFDataset
from main_pys.dataset_preprocessed import PreprocessedFlowMAPFDataset, build_weighted_sampler
from main_pys.generative_model import FlowGNNModel


def _clean_state_dict(state_dict):
    """Strip _orig_mod. prefix added by torch.compile so checkpoints are portable."""
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        cleaned[new_k] = v
    return cleaned


def _default_num_workers(device):
    """Match dataloader workers to available CPUs (SLURM-aware; avoids 12 workers on 5 CPUs)."""
    env_w = os.environ.get("FLOW_NUM_WORKERS")
    if env_w is not None and env_w.isdigit():
        return max(0, int(env_w))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        # Reserve one CPU for the training process
        return max(0, int(slurm) - 1)
    if device.type == "cuda":
        return min(12, max(1, (os.cpu_count() or 2) - 1))
    return min(4, os.cpu_count() or 2)

PREPROCESSED_DIRS = [
    "/media/anushree_mattlab/Seagate Por/preprocessed_data",  # external drive (primary)
    "data/preprocessed",                                       # local fallback
]


WAIT_SPEED_THRESHOLD = 0.1  # velocities below this magnitude → wait action
ACTION_VECTORS = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype=torch.float32)  # right, down, up, left


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


def compute_flow_loss(
    model,
    batch,
    device,
    use_amp,
    action_loss_weight=0.3,
    unweighted_action_loss=False,
    unweighted_flow_loss=False,
    return_components=False,
):
    """Shared flow matching loss computation for train and val, with optional auxiliary action loss."""
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
        predicted_flow, action_logits = model(x_t, t, batch, return_action_logits=True)
        target_flow = x_1 - x_0
        base_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
        if unweighted_flow_loss:
            flow_loss = base_loss.mean()
        else:
            flow_loss = (base_loss * node_weights).mean()

        # Auxiliary action classification loss
        if hasattr(batch, "action_y") and batch.action_y is not None:
            expert_actions = batch.action_y.view(-1).long().to(device)
        else:
            expert_actions = velocity_to_action_labels(x_1, device)
        action_loss = F.cross_entropy(action_logits, expert_actions, reduction='none')
        if unweighted_action_loss:
            action_loss = action_loss.mean()
        else:
            action_loss = (action_loss * node_weights.squeeze(1)).mean()

        loss = flow_loss + action_loss_weight * action_loss

    if return_components:
        return loss, flow_loss.detach(), action_loss.detach()
    return loss


def validate(
    model,
    val_loader,
    device,
    use_amp,
    action_loss_weight,
    unweighted_action_loss,
    unweighted_flow_loss,
):
    """Run validation and return (avg_total, avg_flow, avg_action) — all scalars."""
    model.eval()
    total_loss = 0.0
    total_flow = 0.0
    total_action = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, flow_l, act_l = compute_flow_loss(
                model,
                batch,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
                return_components=True,
            )
            total_loss += loss.item()
            total_flow += float(flow_l.item())
            total_action += float(act_l.item())
            num_batches += 1
    n = max(num_batches, 1)
    return total_loss / n, total_flow / n, total_action / n


def train(run_name="", quick=False, use_wandb=True, wandb_project="flow-mapf", wandb_entity=None,
          preprocessed_dir=None, no_weighted_sampling=False, val_split=0.05, patience=0,
          resume=None, start_epoch=0, hidden_dim=1024, num_layers=6,
          action_loss_weight=0.3, unweighted_action_loss=False,
          unweighted_flow_loss=False, epochs=10,
          batch_size=None, num_workers=None, val_every=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | AMP: {use_amp}")

    # Find preprocessed data: CLI override > external drive > local
    pp_dir = preprocessed_dir
    if pp_dir is None:
        for candidate in PREPROCESSED_DIRS:
            if os.path.isdir(candidate) and len(os.listdir(candidate)) > 0:
                pp_dir = candidate
                break

    if pp_dir:
        dirs = [d.strip() for d in pp_dir.split(",")] if "," in pp_dir else [pp_dir]
        print(f"Using PREPROCESSED dataset from {' + '.join(dirs)}")
        full_dataset = PreprocessedFlowMAPFDataset(pp_dir, validate=False)
    else:
        print(f"No preprocessed data found — using on-the-fly dataset (slow)")
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
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train/Val split: {train_size:,} / {val_size:,} ({val_split*100:.0f}%)")
    else:
        train_dataset = full_dataset
        val_dataset = None

    cpu_cores = num_workers if num_workers is not None else _default_num_workers(device)
    if batch_size is None:
        batch_size = 512 if device.type == "cuda" else 32
    val_every = max(1, int(val_every))

    # ── Weighted sampling (upsamples high-agent-count scenarios) ──
    sampler = None
    if not no_weighted_sampling and pp_dir and isinstance(full_dataset, PreprocessedFlowMAPFDataset):
        sampler = build_weighted_sampler(full_dataset)
        # If using val split with weighted sampler, we need to remap indices
        if val_size > 0:
            # Build sampler on full dataset, but only sample from train indices
            train_indices = train_dataset.indices
            full_weights = sampler.weights
            train_weights = full_weights[train_indices]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=train_weights,
                num_samples=len(train_dataset),
                replacement=True
            )

    _train_kw = dict(
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cpu_cores,
        pin_memory=(device.type == "cuda"),
    )
    if cpu_cores > 0:
        _train_kw["prefetch_factor"] = 2
        _train_kw["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, **_train_kw)

    val_loader = None
    if val_dataset is not None:
        val_w = min(4, cpu_cores) if cpu_cores > 0 else 0
        _val_kw = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_w,
            pin_memory=(device.type == "cuda"),
        )
        if val_w > 0:
            _val_kw["prefetch_factor"] = 2
            _val_kw["persistent_workers"] = True
        val_loader = DataLoader(val_dataset, **_val_kw)

    model = FlowGNNModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 1 if quick else epochs
    # Gentle cosine decay: LR goes from 1e-4 -> ~0 over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision: ~2x throughput on A100 Tensor Cores
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint (must happen BEFORE torch.compile)
    if resume and os.path.exists(resume):
        print(f"Loading checkpoint for fine-tuning: {resume}")
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full checkpoint — load model weights only, reset training state for fine-tuning
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded model weights (fine-tune mode: resetting epoch/optimizer/scheduler)")
        else:
            # Legacy checkpoint (model weights only)
            model.load_state_dict(ckpt)
            print(f"  Loaded model weights (legacy checkpoint).")

    # torch.compile AFTER checkpoint load — compile wraps keys with _orig_mod. prefix
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # WandB setup
    if use_wandb:
        import wandb
        wandb_kwargs = {"project": wandb_project, "name": run_name, "config": {}}
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
            "use_amp": use_amp,
            "device": str(device),
            "weighted_sampling": sampler is not None,
            "val_split": val_split,
            "patience": patience,
            "action_loss_weight": action_loss_weight,
            "unweighted_action_loss": unweighted_action_loss,
            "unweighted_flow_loss": unweighted_flow_loss,
            "val_every": val_every,
        })
        # Charts: batch metrics vs global_step; epoch metrics vs epoch index
        wandb.define_metric("global_step")
        for _m in (
            "train/batch_loss", "train/batch_flow_loss", "train/batch_action_loss", "train/lr",
        ):
            wandb.define_metric(_m, step_metric="global_step")
        wandb.define_metric("epoch")
        for _m in (
            "train/loss", "train/flow_loss", "train/action_loss",
            "val/loss", "val/flow_loss", "val/action_loss", "val/best_val_loss",
            "train/grad_norm_mean", "train/grad_norm_max",
            "train/epoch_time_sec", "train/batches_per_sec", "train/samples_per_sec",
            "optim/lr", "early_stopping/epochs_without_improvement",
        ):
            wandb.define_metric(_m, step_metric="epoch")

    log_batch_every = 10
    print(f"Batch size: {batch_size} | Workers: {cpu_cores} | Epochs: {epochs}")
    if val_loader is not None and val_every > 1:
        print(f"Validation: every {val_every} epoch(s) (faster; early stopping uses validated epochs only)")
    print(f"Weighted sampling: {'ON' if sampler else 'OFF'}")
    print(
        "Loss weighting: "
        f"flow={'unweighted' if unweighted_flow_loss else 'weighted'}, "
        f"action={'unweighted' if unweighted_action_loss else 'weighted'}, "
        f"action_loss_weight={action_loss_weight}"
    )

    # Initialize best_val_loss (may be overridden by resume checkpoint above)
    if 'best_val_loss' not in dir():
        best_val_loss = float('inf')
    if not isinstance(best_val_loss, float) or best_val_loss is None:
        best_val_loss = float('inf')
    epochs_without_improvement = 0
    prefix = f"large_scale_flow_{run_name}_" if run_name else "large_scale_flow_"

    for epoch in range(start_epoch, epochs):
        # ── Training ──
        model.train()
        total_loss = 0.0
        total_flow = 0.0
        total_action = 0.0
        num_batches = 0
        epoch_grad_norms = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_t0 = time.perf_counter()

        for batch_idx, batch in enumerate(pbar):
            loss, flow_b, act_b = compute_flow_loss(
                model,
                batch,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
                return_components=True,
            )
            total_flow += float(flow_b.item())
            total_action += float(act_b.item())

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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Per-batch wandb logging
            if use_wandb and (batch_idx + 1) % log_batch_every == 0:
                import wandb
                global_step = epoch * len(train_loader) + batch_idx + 1
                wandb.log({
                    "global_step": global_step,
                    "train/batch_loss": loss.item(),
                    "train/batch_flow_loss": float(flow_b.item()),
                    "train/batch_action_loss": float(act_b.item()),
                    "train/lr": optimizer.param_groups[0]["lr"],
                })

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        avg_train_flow = total_flow / max(num_batches, 1)
        avg_train_action = total_action / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        max_grad_norm = max(epoch_grad_norms) if epoch_grad_norms else 0.0
        epoch_wall_s = time.perf_counter() - epoch_t0
        train_samples = num_batches * batch_size  # approximate graphs per epoch

        # ── Validation (optional subsampling via val_every to save time) ──
        val_loss = None
        val_flow = None
        val_action = None
        do_val = val_loader is not None and (epoch % val_every == 0)
        if do_val:
            val_loss, val_flow, val_action = validate(
                model,
                val_loader,
                device,
                use_amp,
                action_loss_weight=action_loss_weight,
                unweighted_action_loss=unweighted_action_loss,
                unweighted_flow_loss=unweighted_flow_loss,
            )
            val_str = f" | Val Loss: {val_loss:.4f}"

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = f"{prefix}best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': _clean_state_dict(model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, best_path)
                val_str += " (best)"
            else:
                epochs_without_improvement += 1
                val_str += f" (no improvement for {epochs_without_improvement} epochs)"
        elif val_loader is not None:
            val_str = f" | Val: skipped (val_every={val_every})"
        else:
            val_str = ""

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_str} | LR: {current_lr:.2e}")

        # Save epoch checkpoint (full state for resumability)
        ckpt_path = f"{prefix}epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': _clean_state_dict(model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, ckpt_path)

        # Per-epoch wandb logging (x-axis = epoch via wandb.define_metric)
        if use_wandb:
            import wandb
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/flow_loss": avg_train_flow,
                "train/action_loss": avg_train_action,
                "train/grad_norm_mean": avg_grad_norm,
                "train/grad_norm_max": max_grad_norm,
                "train/epoch_time_sec": epoch_wall_s,
                "train/batches_per_sec": num_batches / max(epoch_wall_s, 1e-6),
                "train/samples_per_sec": train_samples / max(epoch_wall_s, 1e-6),
                "optim/lr": current_lr,
            }
            if val_loss is not None:
                log_dict["val/loss"] = val_loss
                log_dict["val/flow_loss"] = val_flow
                log_dict["val/action_loss"] = val_action
                log_dict["val/best_val_loss"] = best_val_loss
                log_dict["early_stopping/epochs_without_improvement"] = epochs_without_improvement
            wandb.log(log_dict)

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: val loss hasn't improved for {patience} epochs.")
            break

    # Final summary
    if use_wandb:
        import wandb
        wandb.run.summary["final_train_loss"] = avg_train_loss
        wandb.run.summary["final_train_flow"] = avg_train_flow
        wandb.run.summary["final_train_action"] = avg_train_action
        if val_loss is not None:
            wandb.run.summary["final_val_loss"] = val_loss
            wandb.run.summary["final_val_flow"] = val_flow
            wandb.run.summary["final_val_action"] = val_action
            wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["final_checkpoint"] = ckpt_path

    if val_loader is not None:
        print(f"\nBest val loss: {best_val_loss:.4f} (saved to {prefix}best.pt)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="",
                        help="Suffix for checkpoint names, e.g. wave2 -> large_scale_flow_wave2_epoch_N.pt")
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
    parser.add_argument("--unweighted-action-loss", action="store_true",
                        help="Do not apply node_weights to the action cross entropy loss")
    parser.add_argument("--unweighted-flow-loss", action="store_true",
                        help="Do not apply node_weights to the flow MSE loss")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs; --quick still forces 1 epoch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 512 cuda / 32 cpu)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: SLURM_CPUS_PER_TASK-1 or auto)")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run validation every N epochs (1=every epoch; 2 saves ~half val time)")
    args = parser.parse_args()
    train(run_name=args.run_name, quick=args.quick, use_wandb=not args.no_wandb,
          wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
          preprocessed_dir=args.preprocessed_dir,
          no_weighted_sampling=args.no_weighted_sampling,
          val_split=args.val_split, patience=args.patience,
          resume=args.resume, start_epoch=args.start_epoch,
          hidden_dim=args.hidden_dim, num_layers=args.num_layers,
          action_loss_weight=args.action_loss_weight,
          unweighted_action_loss=args.unweighted_action_loss,
          unweighted_flow_loss=args.unweighted_flow_loss,
          epochs=args.epochs,
          batch_size=args.batch_size, num_workers=args.num_workers, val_every=args.val_every)
