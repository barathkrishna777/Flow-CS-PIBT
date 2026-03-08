import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class FlowGNNModel(nn.Module):
    def __init__(self, k=4, hidden_dim=128):
        super().__init__()
        
        # --- 1. Visual Context Encoder (Mimicking Rishi's CustomConv) ---
        # 3 channels, 3x3 kernel, stride 1, padding 0
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=1, padding=0)
        
        # Calculate linear dimension dynamically based on k
        # A 9x9 grid with a 3x3 conv (no padding) becomes 7x7. 
        spatial_size = (2 * k + 1) - 2 
        linear_dim = 3 * spatial_size * spatial_size + 5 # 3 channels + 5 for bd_pred
        
        self.visual_proj = nn.Linear(linear_dim, hidden_dim)
        
        # --- 2. GNN Message Passing (Mimicking Rishi's SAGEConv Stack) ---
        # The input to the graph is now: Visual Context + Velocity (2) + Time (1)
        gnn_input_dim = hidden_dim + 2 + 1 
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        # Layer 1
        self.convs.append(pyg_nn.SAGEConv(gnn_input_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        
        # Layers 2 & 3
        for _ in range(2): 
            self.convs.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
            
        # --- 3. Flow Output Head ---
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), # SiLU remains best for smooth continuous vector fields
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, v_t, t, data):
        # data contains the PyTorch Geometric graph: x, edge_index, bd_pred
        x, edge_index, bd_pred = data.x, data.edge_index, data.bd_pred
        
        # 1. Process visual grid and Backward Dijkstra (bd_pred)
        conv_out = self.conv(x)
        flattened = torch.flatten(conv_out, start_dim=1)
        
        # Append Rishi's goal-directional heuristic
        visual_features = torch.hstack([flattened, bd_pred]) 
        visual_emb = F.relu(self.visual_proj(visual_features)) # (N, hidden_dim)
        
        # 2. Inject Flow variables into the graph nodes!
        if len(t.shape) == 1: t = t.unsqueeze(1)
        
        # Each agent's node now contains its visual state AND its kinematic state
        node_features = torch.cat([visual_emb, v_t, t], dim=-1) # (N, hidden_dim + 3)
        
        # 3. Pass messages between agents! (SAGEConv)
        # Agents now "communicate" their intentions (v_t) and environment (visual_emb)
        for i in range(len(self.convs)):
            node_features = self.convs[i](node_features, edge_index)
            node_features = F.relu(node_features)
            node_features = self.lns[i](node_features)
            
        # 4. Predict the final smooth velocity vector
        return self.post_mp(node_features)