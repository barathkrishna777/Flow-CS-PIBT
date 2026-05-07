import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from main_pys.lattice_primitives import (
    NUM_PRIMITIVES,
    PRIMITIVE_VELOCITY_VECTORS_TORCH,
    PRIMITIVE_SPEED_CLASS,
)


CARDINAL_ACTION_VECTORS = torch.tensor(
    [[0, 1], [1, 0], [-1, 0], [0, -1]],
    dtype=torch.float32,
)


def hybrid_action_logits_from_velocity(
    predicted_velocity,
    wait_logit,
    wait_logit_scale=1.0,
    wait_logit_bias=0.0,
    movement_logit_scale=1.0,
):
    """Build logits as [wait, right, down, up, left].

    Movement logits preserve the flow geometry via velocity dot products; the
    wait logit is supplied by the learned wait head. Optional scalar
    calibration mirrors learned inference.
    """
    action_vectors = CARDINAL_ACTION_VECTORS.to(
        device=predicted_velocity.device,
        dtype=predicted_velocity.dtype,
    )
    calibrated_wait = wait_logit_scale * wait_logit.view(-1, 1) + wait_logit_bias
    movement_logits = movement_logit_scale * (predicted_velocity @ action_vectors.T)
    return torch.cat([calibrated_wait, movement_logits], dim=1)


def wait_ranking_logits_from_velocity(
    predicted_velocity,
    wait_logit,
    wait_logit_scale=1.0,
    wait_logit_bias=0.0,
):
    """Build planner-aligned scores as [calibrated wait, right, down, up, left].

    Unlike the five-logit hybrid interface, movement scores are raw velocity dot
    products. The ranking loss compares the calibrated wait score directly
    against the movement ordering that PIBT consumes.
    """
    action_vectors = CARDINAL_ACTION_VECTORS.to(
        device=predicted_velocity.device,
        dtype=predicted_velocity.dtype,
    )
    calibrated_wait = wait_logit_scale * wait_logit.view(-1, 1) + wait_logit_bias
    movement_logits = predicted_velocity @ action_vectors.T
    return torch.cat([calibrated_wait, movement_logits], dim=1)


def binary_gate_action_probs_from_velocity(
    predicted_velocity,
    wait_logit,
    wait_logit_scale=1.0,
    wait_logit_bias=0.0,
    movement_logit_scale=1.0,
    tau=1.0,
    eps=1e-8,
):
    """Compose P(wait) with P(direction | move).

    The wait head owns only the binary wait-vs-move decision. The velocity
    geometry owns the direction distribution when the agent moves.
    """
    action_vectors = CARDINAL_ACTION_VECTORS.to(
        device=predicted_velocity.device,
        dtype=predicted_velocity.dtype,
    )
    calibrated_wait = wait_logit_scale * wait_logit.view(-1, 1) + wait_logit_bias
    wait_prob = torch.sigmoid(calibrated_wait)
    movement_logits = movement_logit_scale * (predicted_velocity @ action_vectors.T)
    move_probs = torch.softmax(movement_logits / tau, dim=1)
    probs = torch.cat([wait_prob, (1.0 - wait_prob) * move_probs], dim=1)
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=1, keepdim=True)


def lattice_action_logits_from_velocity(
    predicted_velocity,
    wait_logit=None,
    wait_logit_scale=1.0,
    wait_logit_bias=0.0,
    speed_bonus=0.3,
):
    """Score all lattice primitives from velocity + optional wait logit.

    Movement primitives are scored by dot product with their characteristic
    direction vector.  Double-step primitives get a speed bonus proportional
    to velocity magnitude so that fast-moving agents prefer longer strides.
    """
    device = predicted_velocity.device
    dtype = predicted_velocity.dtype
    prim_vecs = PRIMITIVE_VELOCITY_VECTORS_TORCH.to(device=device, dtype=dtype)
    speed_cls = torch.tensor(PRIMITIVE_SPEED_CLASS, device=device, dtype=dtype)

    dir_scores = predicted_velocity @ prim_vecs.T  # (N, NUM_PRIMITIVES)

    vel_mag = predicted_velocity.norm(dim=1, keepdim=True)
    speed_mod = speed_bonus * vel_mag * (speed_cls.unsqueeze(0) - 1.0)
    scores = dir_scores + speed_mod

    if wait_logit is not None:
        calibrated = wait_logit_scale * wait_logit.view(-1, 1) + wait_logit_bias
        scores[:, 0] = calibrated.squeeze(1)
    else:
        scores[:, 0] = -vel_mag.squeeze(1)

    return scores


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
        extra_feature_dim=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_input_channels = num_input_channels
        self.aux_feature_dim = aux_feature_dim
        self.action_dim = action_dim
        self.velocity_dim = velocity_dim
        self.extra_feature_dim = extra_feature_dim
        
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

        # --- 4. Auxiliary Action Head (5-class: wait, right, down, up, left) ---
        action_head_dim = min(hidden_dim, 256)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_head_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(action_head_dim, action_dim)
        )

        # --- 5. Learned Wait Head (1-logit binary wait classifier) ---
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_dim, action_head_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(action_head_dim, 1)
        )

        # --- 6. Lattice Primitive Head (NUM_PRIMITIVES-class) ---
        # extra_features (goal disp + BD dist) are concatenated directly here,
        # giving a short gradient path without touching the frozen trunk.
        lattice_in_dim = hidden_dim + extra_feature_dim
        self.lattice_head = nn.Sequential(
            nn.Linear(lattice_in_dim, action_head_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(action_head_dim, NUM_PRIMITIVES),
        )

        # Scalar calibration for the learned wait-logit inference interface.
        # Old checkpoints load with strict=False and keep these identity values.
        self.wait_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.wait_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.movement_logit_scale = nn.Parameter(torch.tensor(1.0))

    def calibrated_wait_logit(self, wait_logit):
        return self.wait_logit_scale * wait_logit + self.wait_logit_bias

    def hybrid_action_logits(self, velocity_for_logits, wait_logit):
        return hybrid_action_logits_from_velocity(
            velocity_for_logits,
            wait_logit,
            wait_logit_scale=self.wait_logit_scale,
            wait_logit_bias=self.wait_logit_bias,
            movement_logit_scale=self.movement_logit_scale,
        )

    def wait_ranking_logits(self, velocity_for_logits, wait_logit):
        return wait_ranking_logits_from_velocity(
            velocity_for_logits,
            wait_logit,
            wait_logit_scale=self.wait_logit_scale,
            wait_logit_bias=self.wait_logit_bias,
        )

    def binary_gate_action_probs(self, velocity_for_logits, wait_logit, tau=1.0):
        return binary_gate_action_probs_from_velocity(
            velocity_for_logits,
            wait_logit,
            wait_logit_scale=self.wait_logit_scale,
            wait_logit_bias=self.wait_logit_bias,
            tau=tau,
        )

    def forward(
        self,
        v_t,
        t,
        data,
        return_action_logits=False,
        return_wait_logit=False,
        return_calibrated_wait_logit=False,
        return_hybrid_logits=False,
        hybrid_velocity_for_logits=None,
        return_wait_ranking_logits=False,
        wait_ranking_velocity_for_logits=None,
        return_lattice_logits=False,
    ):
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

        outputs = [flow_output]
        if return_action_logits:
            # Auxiliary action logits from shared GNN features (detached from flow head)
            outputs.append(self.action_head(node_features))

        need_wait_logit = (
            return_wait_logit
            or return_calibrated_wait_logit
            or return_hybrid_logits
            or return_wait_ranking_logits
        )
        wait_logit = None
        if need_wait_logit:
            wait_logit = self.wait_head(node_features).squeeze(-1)
        if return_wait_logit:
            outputs.append(wait_logit)
        if return_calibrated_wait_logit:
            outputs.append(self.calibrated_wait_logit(wait_logit))
        if return_hybrid_logits:
            if hybrid_velocity_for_logits is None:
                hybrid_velocity_for_logits = v_t + (1 - t) * flow_output
            outputs.append(self.hybrid_action_logits(hybrid_velocity_for_logits, wait_logit))
        if return_wait_ranking_logits:
            if wait_ranking_velocity_for_logits is None:
                wait_ranking_velocity_for_logits = v_t + (1 - t) * flow_output
            outputs.append(self.wait_ranking_logits(wait_ranking_velocity_for_logits, wait_logit))

        if return_lattice_logits:
            if self.extra_feature_dim > 0:
                extra_feat = getattr(data, "extra_features", None)
                if extra_feat is not None:
                    lat_in = torch.cat([node_features, extra_feat.to(node_features.device).float()], dim=1)
                else:
                    lat_in = torch.cat([node_features, node_features.new_zeros(node_features.shape[0], self.extra_feature_dim)], dim=1)
            else:
                lat_in = node_features
            outputs.append(self.lattice_head(lat_in))

        if len(outputs) == 1:
            return flow_output
        return tuple(outputs)
