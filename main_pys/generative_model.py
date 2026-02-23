import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, input_channels, grid_size, hidden_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        final_spatial_dim = (grid_size // 2) // 2
        self.flat_size = 128 * final_spatial_dim * final_spatial_dim
        self.projection = nn.Linear(self.flat_size, hidden_dim)

    def forward(self, x):
        return self.projection(self.cnn(x))


class VelocityFlowNetwork(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 + 1 + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t, context):
        if len(t.shape) == 1: t = t.unsqueeze(1)
        return self.net(torch.cat([x, t, context], dim=-1))

class FlowMAPFModel(nn.Module):
    def __init__(self, k=4):
        super().__init__()

        grid_dim = 2*k + 1
        self.encoder = ContextEncoder(input_channels=3, grid_size=grid_dim, hidden_dim=128)
        self.flow_net = VelocityFlowNetwork(hidden_dim=512, context_dim=128)

    def forward(self, v_t, t, data):
        context = self.encoder(data.x)
        return self.flow_net(v_t, t, context)