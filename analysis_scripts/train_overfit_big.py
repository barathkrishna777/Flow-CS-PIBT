"""
Overfit the BIG model (6-layer, 1024-dim) on a small subset of empty-48-48
trajectories, then check if it can memorize them perfectly.

Uses the same training loop as train_flow.py (with wait fix, AMP, etc.)
but on a tiny dataset with aggressive LR and no regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import glob
import numpy as np

from main_pys.dataset import FlowMAPFDataset
from main_pys.generative_model import FlowGNNModel


class SingleFileDataset(FlowMAPFDataset):
    """Override to only load a specific list of trajectory files."""

    def __init__(self, file_list, map_dir, bd_dir, k=4, m=5):
        self.map_dir = map_dir
        self.bd_dir = bd_dir
        self.k = k
        self.m = m
        self.npz_files = file_list

        print("Preloading Maps...")
        self.maps = {}
        for map_path in glob.glob(os.path.join(map_dir, "*.map")):
            map_name = os.path.basename(map_path).replace(".map", "")
            self.maps[map_name] = self._read_map(map_path)

        print("Building flattened timestep index...")
        self.index = []
        for f in self.npz_files:
            try:
                with np.load(f) as data:
                    T = data['discrete_positions'].shape[1]
                    for t in range(T):
                        self.index.append((f, t))
            except Exception as e:
                print(f"Skipping: {f} ({e})")

        print(f"Overfit dataset: {len(self.npz_files)} file(s), {len(self.index)} samples")


def train_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # Pick a few empty-48-48 trajectory files with varying agent counts
    overfit_files = []
    data_dir = os.path.join("data", "flow_training_data_multi")
    for agent_count in ["20", "50", "100"]:
        f = os.path.join(data_dir, f"empty-48-48-random-1_{agent_count}.npz")
        if os.path.exists(f):
            overfit_files.append(f)
            print(f"  Using: {f}")
        else:
            print(f"  Missing (skipping): {f}")

    if not overfit_files:
        raise FileNotFoundError("No trajectory files found!")

    dataset = SingleFileDataset(
        file_list=overfit_files,
        map_dir="data/mapf-map",
        bd_dir="data/bd_npzs",
        k=4, m=5,
    )

    batch_size = 16 if device.type == "cuda" else 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = FlowGNNModel().to(device)
    # Aggressive LR, NO weight decay — we WANT to overfit
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = 200
    best_loss = float('inf')

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            x_1 = batch.y.view(-1, 2)
            graph_data = batch

            num_graphs = batch.batch.max().item() + 1
            t_per_graph = torch.rand(num_graphs, 1, device=device)
            t = t_per_graph[batch.batch]
            x_0 = torch.randn_like(x_1)
            x_t = t * x_1 + (1 - t) * x_0

            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted_flow = model(x_t, t, graph_data)
                target_flow = x_1 - x_0
                loss = F.mse_loss(predicted_flow, target_flow)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/overfit_big_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

        # Save periodic checkpoints
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints/overfit_big_epoch_{epoch}.pt")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Saved: checkpoints/overfit_big_best.pt")
    torch.save(model.state_dict(), "checkpoints/overfit_big_final.pt")


if __name__ == "__main__":
    train_overfit()
