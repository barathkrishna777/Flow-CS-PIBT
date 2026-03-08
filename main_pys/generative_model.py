import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class FlowGNNModel(nn.Module):
    def __init__(self, k=4, hidden_dim=512):
        super().__init__()
        
        # --- 1. Scaled-Up Visual Context Encoder ---
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 9x9 -> 4x4
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4 -> 2x2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Math: 128 channels * 2 * 2 spatial dimensions
        spatial_size = ((2 * k + 1) // 2) // 2 
        cnn_out_dim = 128 * spatial_size * spatial_size
        
        # 512 (CNN) + 5 (bd_pred) projected to 512
        self.visual_proj = nn.Sequential(
            nn.Linear(cnn_out_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2) # Regularization to prevent memorizing paths
        )
        
        # --- 2. Deeper GNN Message Passing ---
        gnn_input_dim = hidden_dim + 2 + 1 # Visual Context + Velocity (2) + Time (1)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        # Layer 1
        self.convs.append(pyg_nn.SAGEConv(gnn_input_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        
        # Layers 2, 3, 4 (Deeper graph for complex coordination)
        for _ in range(3): 
            self.convs.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
            
        # --- 3. Regularized Flow Output Head ---
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Dropout(0.2), # Regularization to prevent ODE overshoot
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, v_t, t, data):
        x, edge_index, bd_pred = data.x, data.edge_index, data.bd_pred
        
        # 1. Process visual grid
        cnn_out = self.conv(x)
        
        # Append Backward Dijkstra heuristic
        visual_features = torch.hstack([cnn_out, bd_pred]) 
        visual_emb = self.visual_proj(visual_features) # (N, hidden_dim)
        
        # 2. Inject Flow variables
        if len(t.shape) == 1: t = t.unsqueeze(1)
        node_features = torch.cat([visual_emb, v_t, t], dim=-1) # (N, hidden_dim + 3)
        
        # 3. Pass messages (SAGEConv)
        for i in range(len(self.convs)):
            node_features = self.convs[i](node_features, edge_index)
            node_features = F.relu(node_features)
            node_features = self.lns[i](node_features)
            
        # 4. Predict velocity
        return self.post_mp(node_features)