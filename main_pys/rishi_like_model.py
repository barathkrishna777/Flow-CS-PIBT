import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class RishiLikeClassifier(nn.Module):
    """SageConv discrete-action classifier for the MAPF grid benchmark."""

    def __init__(
        self,
        k=4,
        hidden_dim=128,
        num_layers=3,
        dropout=0.25,
        in_channels=3,
        aux_feature_dim=5,
        action_dim=5,
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.aux_feature_dim = aux_feature_dim
        self.action_dim = action_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        patch_width = 2 * k + 1
        cnn_out_dim = in_channels * (patch_width - 2) * (patch_width - 2)
        self.input_proj = nn.Linear(cnn_out_dim + aux_feature_dim, hidden_dim)
        self.message_proj = nn.Linear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList([
            pyg_nn.SAGEConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        aux_features = getattr(data, "aux_features", None)
        if aux_features is None:
            aux_features = data.bd_pred

        x = self.cnn(x)
        x = torch.hstack([x, aux_features])
        x = F.leaky_relu(self.input_proj(x))
        x = self.message_proj(x)

        for conv, ln in zip(self.convs, self.lns):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.output(x)
        return x, logits
