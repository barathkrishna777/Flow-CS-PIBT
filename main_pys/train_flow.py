import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os

from main_pys.dataset import FlowMAPFDataset
from main_pys.generative_model import FlowGNNModel


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    dataset = FlowMAPFDataset(data_dir="data/flow_training_data_multi",
                              map_dir="data/mapf-map",
                              bd_dir="data/bd_npzs",
                              k=4, m=5)

    # A100: 8-12 workers to keep GPU saturated; CPU: stay conservative
    cpu_cores = min(8, os.cpu_count() or 2) if device.type == "cuda" else min(4, os.cpu_count() or 2)
    # A100: batch_size=128 fits comfortably in 40GB; CPU: keep small
    batch_size = 128 if device.type == "cuda" else 32
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

    epochs = 10
    # Gentle cosine decay: LR goes from 1e-4 -> ~0 over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision: ~2x throughput on A100 Tensor Cores
    # Compatible with both old (torch.cuda.amp) and new (torch.amp) APIs
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Batch size: {batch_size} | Workers: {cpu_cores} | Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
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
            # Gradient clipping — flow matching targets (x_1 - x_0) can be large
            # when x_0 is far from x_1, causing gradient spikes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        torch.save(model.state_dict(), f"large_scale_flow_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    train()