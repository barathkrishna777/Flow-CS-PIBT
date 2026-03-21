"""
Direct velocity prediction baseline (NO flow matching).
Same GNN architecture, but trained to directly predict the expert velocity x_1.
v_t and t are set to zero — the model learns a pure mapping from graph features to velocity.

This isolates: is the bottleneck in the GNN, or in the flow matching formulation?
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
    """Override to only load specific trajectory files."""

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


def train_direct():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = FlowGNNModel().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)

    epochs = 100
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            x_1 = batch.y.view(-1, 2)  # expert velocity
            graph_data = batch

            # ---- KEY DIFFERENCE: No flow matching ----
            # Pass zeros for v_t and t — model predicts velocity purely from graph features
            num_nodes = x_1.shape[0]
            v_t = torch.zeros(num_nodes, 2, device=device)
            t = torch.zeros(num_nodes, 1, device=device)

            predicted_velocity = model(v_t, t, graph_data)

            # Direct MSE against expert velocity
            loss = F.mse_loss(predicted_velocity, x_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "direct_overfit_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    torch.save(model.state_dict(), "direct_overfit_final.pt")
    print("Saved: direct_overfit_best.pt, direct_overfit_final.pt")


if __name__ == "__main__":
    train_direct()
