"""
Overfit sanity check: Train on a SINGLE tiny scenario (empty-32-32, 20 agents)
for many epochs with no regularization, then evaluate on that same scenario.

If the model can't overfit this, the architecture or training loop is broken.
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
        # Bypass parent __init__ but reuse its helpers
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
    print(f"Device: {device}")

    # --- Single tiny file ---
    overfit_files = [
        os.path.join("data", "flow_training_data_multi", "empty-32-32-random-10_20.npz"),
    ]
    for f in overfit_files:
        assert os.path.exists(f), f"Missing: {f}"

    dataset = SingleFileDataset(
        file_list=overfit_files,
        map_dir="data/mapf-map",
        bd_dir="data/bd_npzs",
        k=4, m=5,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,       # small batch to see every sample many times
        shuffle=True,
        num_workers=0,      # no workers — deterministic, no IPC issues
        pin_memory=False,
    )

    model = FlowGNNModel().to(device)
    # Higher LR, NO weight decay — we WANT to overfit
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)

    epochs = 100
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            x_1 = batch.y.view(-1, 2)
            graph_data = batch

            # Flow matching: sample t, interpolate, predict field
            # Logit-normal sampling + clamp to avoid t→1 singularity
            num_graphs = batch.batch.max().item() + 1
            t_per_graph = torch.sigmoid(torch.randn(num_graphs, 1, device=device))
            t_per_graph = t_per_graph.clamp(0.01, 0.99)
            t = t_per_graph[batch.batch]
            x_0 = torch.randn_like(x_1)
            x_t = t * x_1 + (1 - t) * x_0

            predicted_flow = model(x_t, t, graph_data)
            target_flow = x_1 - x_0

            # Pure MSE — no weighting, no collision loss, no distractions
            loss = F.mse_loss(predicted_flow, target_flow)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "overfit_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Saved: overfit_best.pt")
    torch.save(model.state_dict(), "overfit_final.pt")
    print(f"Saved: overfit_final.pt")


if __name__ == "__main__":
    train_overfit()
