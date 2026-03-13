import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class FlowGNNModel(nn.Module):
    def __init__(self, k=4, hidden_dim=1024): # Scaled up from 512 to 1024
        super().__init__()
        
        # --- 1. Wider Visual Context Encoder ---
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
            nn.Linear(cnn_out_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2) 
        )
        
        # --- 2. Deeper GNN with Residual Connections ---
        gnn_input_dim = hidden_dim + 2 + 1 
        
        # Project inputs to match hidden_dim for residual addition
        self.input_proj = nn.Linear(gnn_input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.num_layers = 6 # Scaled from 4 to 6 for a larger receptive field
        
        for _ in range(self.num_layers): 
            self.convs.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
            
        # --- 3. Expressive Output Head ---
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Dropout(0.2), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, v_t, t, data):
        x, edge_index, bd_pred = data.x, data.edge_index, data.bd_pred
        
        # 1. Process visual grid
        cnn_out = self.conv(x)
        visual_features = torch.hstack([cnn_out, bd_pred]) 
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
            
        # 4. Predict velocity
        return self.post_mp(node_features)