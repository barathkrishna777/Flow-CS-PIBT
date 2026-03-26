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


def compute_flow_loss(model, batch, device, use_amp, action_loss_weight=0.3):
    """Shared flow matching loss computation for train and val, with optional auxiliary action loss."""
    batch = batch.to(device)
    x_1 = batch.y.view(-1, 2)
    node_weights = batch.node_weights.view(-1, 1)

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
        flow_loss = (base_loss * node_weights).mean()

        # Auxiliary action classification loss
        expert_actions = velocity_to_action_labels(x_1, device)
        action_loss = F.cross_entropy(action_logits, expert_actions, reduction='none')
        action_loss = (action_loss * node_weights.squeeze(1)).mean()

        loss = flow_loss + action_loss_weight * action_loss

    return loss


def validate(model, val_loader, device, use_amp):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_flow_loss(model, batch, device, use_amp)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


def train(run_name="", quick=False, use_wandb=True, wandb_project="flow-mapf", wandb_entity=None,
          preprocessed_dir=None, no_weighted_sampling=False, val_split=0.05, patience=0,
          resume=None, start_epoch=0, hidden_dim=1024, num_layers=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
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
        full_dataset = PreprocessedFlowMAPFDataset(pp_dir)
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

    # GPU: more workers + bigger batches to keep GPU saturated; CPU: stay conservative
    cpu_cores = min(12, os.cpu_count() or 2) if device.type == "cuda" else min(4, os.cpu_count() or 2)
    batch_size = 256 if device.type == "cuda" else 32

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
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, cpu_cores),
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
            persistent_workers=True
        )

    model = FlowGNNModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 1 if quick else 10
    # Gentle cosine decay: LR goes from 1e-4 -> ~0 over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision: ~2x throughput on A100 Tensor Cores
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint
    if resume and os.path.exists(resume):
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full checkpoint (model + optimizer + scheduler + metadata)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if ckpt.get('scaler_state_dict'):
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch']  # epoch is already 1-indexed, use as start
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"  Restored full state: resuming from epoch {start_epoch + 1}, best_val={best_val_loss:.4f}")
        else:
            # Legacy checkpoint (model weights only)
            model.load_state_dict(ckpt)
            print(f"  Loaded model weights only (legacy checkpoint). Fast-forwarding scheduler {start_epoch} steps.")
            for _ in range(start_epoch):
                scheduler.step()

    # WandB setup
    if use_wandb:
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
            "use_amp": use_amp,
            "device": str(device),
            "weighted_sampling": sampler is not None,
            "val_split": val_split,
            "patience": patience,
        })

    log_batch_every = 10
    print(f"Batch size: {batch_size} | Workers: {cpu_cores} | Epochs: {epochs}")
    print(f"Weighted sampling: {'ON' if sampler else 'OFF'}")

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
        num_batches = 0
        epoch_grad_norms = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            loss = compute_flow_loss(model, batch, device, use_amp)

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
                wandb.log({"train/batch_loss": loss.item()}, step=global_step)

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0

        # ── Validation ──
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, use_amp)
            val_str = f" | Val Loss: {val_loss:.4f}"

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = f"{prefix}best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
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
        else:
            val_str = ""

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_str} | LR: {current_lr:.2e}")

        # Save epoch checkpoint (full state for resumability)
        ckpt_path = f"{prefix}epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, ckpt_path)

        # Per-epoch wandb logging
        if use_wandb:
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
            print(f"Early stopping: val loss hasn't improved for {patience} epochs.")
            break

    # Final summary
    if use_wandb:
        import wandb
        wandb.run.summary["final_train_loss"] = avg_train_loss
        if val_loss is not None:
            wandb.run.summary["final_val_loss"] = val_loss
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
                        help="Override preprocessed data directory")
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
    args = parser.parse_args()
    train(run_name=args.run_name, quick=args.quick, use_wandb=not args.no_wandb,
          wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
          preprocessed_dir=args.preprocessed_dir,
          no_weighted_sampling=args.no_weighted_sampling,
          val_split=args.val_split, patience=args.patience,
          resume=args.resume, start_epoch=args.start_epoch,
          hidden_dim=args.hidden_dim, num_layers=args.num_layers)
