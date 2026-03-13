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

    cpu_cores = min(16, os.cpu_count() or 4)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=cpu_cores, 
        pin_memory=True,
        prefetch_factor=4, # 2. Tell the workers to prepare 4 batches in advance so the GPU never waits
        persistent_workers=True # 3. Keep the workers alive between epochs to save startup time
    )

    model = FlowGNNModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 2 
    safety_weight = 0.1

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            x_1 = batch.y.view(-1, 2)
            node_weights = batch.node_weights.view(-1, 1)
            graph_data = batch 

            N = x_1.shape[0]
            t = torch.rand(N, 1).to(device)
            x_0 = torch.randn_like(x_1).to(device)
            x_t = t*x_1 + (1-t)*x_0

            predicted_flow = model(x_t, t, graph_data)
            target_flow = x_1 - x_0

            base_loss = F.mse_loss(predicted_flow, target_flow, reduction='none')
            weighted_mse = (base_loss * node_weights).mean()

            safe_loss = collision_loss(predicted_flow, graph_data)

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