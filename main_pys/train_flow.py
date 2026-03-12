import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os

from main_pys.dataset import FlowMAPFDataset
from main_pys.generative_model import FlowGNNModel

def collision_loss(predicted_flow, graph_data, min_dist=1.5):
    """Repulsive loss to teach the flow field basic social distancing."""
    edge_index = graph_data.edge_index
    rel_pos = graph_data.edge_attr 
    v_source = predicted_flow[edge_index[0]]
    v_target = predicted_flow[edge_index[1]]
    rel_vel = v_source - v_target
    dist = torch.norm(rel_pos, dim=1)
    mask = dist < min_dist
    approach_speed = torch.sum(rel_vel * rel_pos, dim=1) / (dist + 1e-8)
    loss = torch.where(mask & (approach_speed < 0), 
                       torch.square(approach_speed), 
                       torch.zeros_like(approach_speed))
    return loss.mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FlowMAPFDataset(data_dir="data/flow_training_data_multi",
                              map_dir="data/mapf-map",
                              bd_dir="data/bd_npzs", 
                              k=4, m=5)

    # Scaled up batch size and I/O workers for massive dataset
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    model = FlowGNNModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 10 
    safety_weight = 0.1

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Unpacking 3 variables now: graph, target velocity, and agent weights
        for graph_data, x_1, node_weights in pbar:
            graph_data = graph_data.to(device)
            x_1 = x_1.to(device).view(-1, 2)
            node_weights = node_weights.to(device).view(-1, 1)

            N = x_1.shape[0]
            t = torch.rand(N, 1).to(device)
            x_0 = torch.randn_like(x_1).to(device)
            x_t = t*x_1 + (1-t)*x_0

            predicted_flow = model(x_t, t, graph_data)
            target_flow = x_1 - x_0

            # --- WEIGHTED MSE LOSS ---
            # Down-weights the loss for agents that are just sitting at their goals
            base_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
            weighted_mse = (base_loss * node_weights).mean()

            # Collision Loss
            safe_loss = collision_loss(predicted_flow, graph_data)

            # Combined Objective
            loss = weighted_mse + (safety_weight * safe_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"large_scale_flow_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()