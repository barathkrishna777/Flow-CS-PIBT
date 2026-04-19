import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class FlowGNNModel(nn.Module):
    def __init__(
        self,
        k=4,
        hidden_dim=1024,
        num_layers=6,
        num_input_channels=3,
        aux_feature_dim=5,
        action_dim=5,
        velocity_dim=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_input_channels = num_input_channels
        self.aux_feature_dim = aux_feature_dim
        self.action_dim = action_dim
        self.velocity_dim = velocity_dim
        
        # --- 1. Visual Context Encoder ---
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Flatten()
        )
        
        spatial_size = ((2 * k + 1) // 2) // 2 
        cnn_out_dim = 256 * spatial_size * spatial_size
        
        self.visual_proj = nn.Sequential(
            nn.Linear(cnn_out_dim + aux_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15) 
        )
        
        # --- 2. GNN with Residual Connections ---
        gnn_input_dim = hidden_dim + velocity_dim + 1 
        self.input_proj = nn.Linear(gnn_input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        for _ in range(self.num_layers): 
            self.convs.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
            
        # --- 3. Output Head (flow velocity) ---
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, velocity_dim)
        )

        # --- 4. Auxiliary Action Head (grid4/grid8 action logits) ---
        action_head_dim = min(hidden_dim, 256)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_head_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(action_head_dim, action_dim)
        )

    def forward(self, v_t, t, data, return_action_logits=False):
        x, edge_index = data.x, data.edge_index
        aux_features = getattr(data, "aux_features", None)
        if aux_features is None:
            aux_features = data.bd_pred

        # 1. Process visual grid
        cnn_out = self.conv(x)
        visual_features = torch.hstack([cnn_out, aux_features])
        visual_emb = self.visual_proj(visual_features)

        # 2. Inject Flow variables
        if len(t.shape) == 1: t = t.unsqueeze(1)
        node_features = torch.cat([visual_emb, v_t, t], dim=-1)

        node_features = self.input_proj(node_features)

        # 3. Pass messages with Residual (Skip) Connections
        for i in range(self.num_layers):
            identity = node_features
            node_features = self.convs[i](node_features, edge_index)
            node_features = self.lns[i](node_features)
            node_features = F.silu(node_features)
            node_features = node_features + identity # The crucial residual addition

        # 4. Predict velocity (flow output)
        flow_output = self.post_mp(node_features)

        if return_action_logits:
            # Auxiliary action logits from shared GNN features (detached from flow head)
            action_logits = self.action_head(node_features)
            return flow_output, action_logits

        return flow_output
