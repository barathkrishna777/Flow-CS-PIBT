import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from main_pys.dataset import FlowMAPFDataset
from main_pys.generative_model import ContextEncoder, VelocityFlowNetwork, FlowMAPFModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FlowMAPFDataset(data_dir="data/flow_training_data_multi",
                              map_dir="data/mapf-map",
                              bd_dir="data/bd_npzs",
                              k=4, m=5)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = FlowMAPFModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 150
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for graph_data, x_1 in pbar:
            graph_data = graph_data.to(device)
            x_1 = x_1.to(device)
            x_1 = x_1.view(-1, 2)

            N = x_1.shape[0]

            t = torch.rand(N, 1).to(device)

            x_0 = torch.randn_like(x_1).to(device)

            x_t = t*x_1 + (1-t)*x_0

            predicted_flow = model(x_t, t, graph_data)
            target_flow = x_1 - x_0

            loss = criterion(predicted_flow, target_flow)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"flow_model_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    train()