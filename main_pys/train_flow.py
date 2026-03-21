import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import argparse

from main_pys.dataset import FlowMAPFDataset
from main_pys.dataset_preprocessed import PreprocessedFlowMAPFDataset
from main_pys.generative_model import FlowGNNModel

PREPROCESSED_DIR = "data/preprocessed"


def train(run_name="", quick=False, use_wandb=True, wandb_project="flow-mapf", wandb_entity=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # Use preprocessed .pt files if available (50x faster); fall back to on-the-fly
    if os.path.isdir(PREPROCESSED_DIR) and len(os.listdir(PREPROCESSED_DIR)) > 0:
        print(f"Using PREPROCESSED dataset from {PREPROCESSED_DIR}")
        dataset = PreprocessedFlowMAPFDataset(PREPROCESSED_DIR)
    else:
        print(f"No preprocessed data found — using on-the-fly dataset (slow)")
        dataset = FlowMAPFDataset(data_dir="data/flow_training_data_multi",
                                  map_dir="data/mapf-map",
                                  bd_dir="data/bd_npzs",
                                  k=4, m=5)

    # GPU: more workers + bigger batches to keep GPU saturated; CPU: stay conservative
    cpu_cores = min(12, os.cpu_count() or 2) if device.type == "cuda" else min(4, os.cpu_count() or 2)
    batch_size = 256 if device.type == "cuda" else 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_cores,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )

    model = FlowGNNModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 1 if quick else 10
    # Gentle cosine decay: LR goes from 1e-4 -> ~0 over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision: ~2x throughput on A100 Tensor Cores
    # Compatible with both old (torch.cuda.amp) and new (torch.amp) APIs
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            "weight_decay": 1e-5,
            "dataset_size": len(dataset),
            "num_workers": cpu_cores,
            "quick": quick,
            "use_amp": use_amp,
            "device": str(device),
        })

    log_batch_every = 10
    print(f"Batch size: {batch_size} | Workers: {cpu_cores} | Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_grad_norms = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)

            x_1 = batch.y.view(-1, 2)
            node_weights = batch.node_weights.view(-1, 1)
            graph_data = batch

            # Sample one t per GRAPH (not per node) to match inference
            num_graphs = batch.batch.max().item() + 1
            t_per_graph = torch.rand(num_graphs, 1, device=device)
            t = t_per_graph[batch.batch]
            x_0 = torch.randn_like(x_1)
            x_t = t * x_1 + (1 - t) * x_0

            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted_flow = model(x_t, t, graph_data)
                target_flow = x_1 - x_0

                # Pure weighted MSE — no collision loss (it targets the flow field
                # not the final velocity, which is mathematically incorrect for
                # flow matching and was shown to hurt convergence in overfit tests)
                base_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
                loss = (base_loss * node_weights).mean()

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
                global_step = epoch * len(dataloader) + batch_idx + 1
                wandb.log({"train/batch_loss": loss.item()}, step=global_step)

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        prefix = f"large_scale_flow_{run_name}_" if run_name else "large_scale_flow_"
        ckpt_path = f"{prefix}epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Per-epoch wandb logging
        if use_wandb:
            import wandb
            epoch_step = (epoch + 1) * len(dataloader)
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/lr": current_lr,
                "epoch/grad_norm_mean": avg_grad_norm,
                "epoch": epoch + 1,
            }, step=epoch_step)
            wandb.save(ckpt_path, base_path=".")

    # Final summary
    if use_wandb:
        import wandb
        wandb.run.summary["final_epoch_loss"] = avg_loss
        wandb.run.summary["total_epochs"] = epochs
        wandb.run.summary["final_checkpoint"] = ckpt_path


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
    args = parser.parse_args()
    train(run_name=args.run_name, quick=args.quick, use_wandb=not args.no_wandb,
          wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)