import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, input_channels, grid_size, hidden_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.flat_size = 32 * (grid_size // 2) * (grid_size // 2)
        self.projection = nn.Linear(self.flat_size, hidden_dim)

    def forward(self, x):
        return self.project(self.cnn(x))


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