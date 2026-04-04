"""Transformer-based flow matching model for continuous MAPF.

Replaces the GNN backbone with masked local self-attention.
Key design choices (per MAPF expert guidance):
  - No biases on any nn.Linear (bias=False everywhere)
  - Spatially-aware communication via relative position encoding in attention
  - Masked attention: each agent only attends to k-nearest neighbours + self
  - Supports action chunking: predict H velocity steps per forward pass
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Embed diffusion timestep t ∈ [0,1] into a fixed-dim vector (no params)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N, 1) or (N,)
        t = t.view(-1, 1).float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t * freqs.unsqueeze(0)  # (N, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (N, dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class RelativePosEncoding(nn.Module):
    """Map 2-D relative position deltas to per-head attention biases.

    Linear(2 → num_heads) with no bias — produces one scalar per head per edge.
    """
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, num_heads, bias=False)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """edge_attr: (E, 2) → (E, num_heads)."""
        return self.proj(edge_attr)


class LocalMaskedAttention(nn.Module):
    """Multi-head self-attention with:
      - Hard local mask from edge_index (only neighbours attend)
      - Additive relative-position bias per edge
      - bias=False on all projections
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_enc = RelativePosEncoding(num_heads)

    def forward(
        self,
        x: torch.Tensor,            # (N, hidden_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, 2) normalised relative positions
        batch_ptr: torch.Tensor,    # (num_graphs + 1,) cumulative node counts
    ) -> torch.Tensor:
        N = x.shape[0]
        device = x.device
        num_graphs = batch_ptr.shape[0] - 1

        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # Relative position bias: (E, num_heads)
        rel_bias = self.rel_pos_enc(edge_attr)  # (E, H)

        # Build block-diagonal attention per graph to keep graphs isolated.
        # For each graph, build a (n_i, n_i, H) attention logit matrix, then
        # softmax and aggregate.
        outputs = torch.zeros_like(x)

        for g in range(num_graphs):
            gs = int(batch_ptr[g].item())
            ge = int(batch_ptr[g + 1].item())
            n = ge - gs
            if n == 0:
                continue

            # Local node slices
            Q_g = Q[gs:ge]  # (n, H, D)
            K_g = K[gs:ge]
            V_g = V[gs:ge]

            # Attention logits: (n, n, H)
            # QK^T / sqrt(d): einsum over head dimension D
            attn = torch.einsum("ihd,jhd->ijh", Q_g, K_g) * self.scale  # (n, n, H)

            # Build -inf mask: everything masked except self + neighbours
            mask = torch.full((n, n), float("-inf"), device=device)
            mask.fill_diagonal_(0.0)  # self-attention always allowed

            # Find edges belonging to this graph
            edge_mask = (edge_index[0] >= gs) & (edge_index[0] < ge)
            e_src = edge_index[0][edge_mask] - gs
            e_dst = edge_index[1][edge_mask] - gs
            if e_src.numel() > 0:
                mask[e_src, e_dst] = 0.0
                mask[e_dst, e_src] = 0.0  # symmetric

                # Add relative position bias for edges in this graph
                rb = rel_bias[edge_mask]  # (E_g, H)
                attn[e_src, e_dst] += rb
                attn[e_dst, e_src] += rb  # symmetric

            # mask: (n, n) → broadcast to (n, n, H)
            attn = attn + mask.unsqueeze(-1)  # add -inf where masked

            # Permute to (H, n, n) for softmax over j dimension
            attn = attn.permute(2, 0, 1)  # (H, n, n)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Weighted sum of values: (H, n, D)
            out_g = torch.einsum("hij,jhd->ihd", attn, V_g)  # (n, H, D)
            outputs[gs:ge] = out_g.reshape(n, self.hidden_dim)

        return self.out_proj(outputs)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with local masked attention and bias-free FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = LocalMaskedAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_ptr: torch.Tensor,
    ) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x), edge_index, edge_attr, batch_ptr)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FlowTransformerModel(nn.Module):
    """Transformer flow-matching model for continuous MAPF.

    Matches the same forward() interface as FlowGNNModel so it is a drop-in
    replacement in train_continuous.py and eval_continuous.py.

    Args:
        k: Local patch half-size (visual field radius in grid cells).
        hidden_dim: Width of transformer hidden states.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads (must divide hidden_dim).
        num_input_channels: CNN input channels (typically 4).
        aux_feature_dim: Scalar feature dimension (typically 5).
        action_dim: Number of discrete action classes (num_directions + 1).
        velocity_dim: Velocity dimensionality (2 for 2-D navigation).
        chunk_horizon: Number of future velocity steps to predict (≥1).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        k: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_input_channels: int = 4,
        aux_feature_dim: int = 5,
        action_dim: int = 9,
        velocity_dim: int = 2,
        chunk_horizon: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_input_channels = num_input_channels
        self.aux_feature_dim = aux_feature_dim
        self.action_dim = action_dim
        self.velocity_dim = velocity_dim
        self.chunk_horizon = chunk_horizon

        # ----- 1. Visual patch encoder (CNN) -----
        # Processes (N, num_input_channels, 2k+1, 2k+1) local patches
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
            nn.Flatten(),
        )
        spatial_size = ((2 * k + 1) // 2) // 2
        cnn_out_dim = 256 * spatial_size * spatial_size

        # Project visual + aux → hidden_dim (no bias)
        self.visual_proj = nn.Sequential(
            nn.Linear(cnn_out_dim + aux_feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ----- 2. Flow & time conditioning -----
        # v_t: (N, velocity_dim * chunk_horizon), t: (N, 1)
        flow_cond_dim = velocity_dim * chunk_horizon + 1
        self.flow_proj = nn.Linear(hidden_dim + flow_cond_dim, hidden_dim, bias=False)

        # ----- 3. Transformer backbone -----
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ----- 4. Output heads -----
        out_vel_dim = velocity_dim * chunk_horizon
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_vel_dim, bias=False),
        )

        out_action_dim = action_dim * chunk_horizon
        action_head_dim = min(hidden_dim, 256)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_head_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(action_head_dim, out_action_dim, bias=False),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        v_t: torch.Tensor,
        t: torch.Tensor,
        data,
        return_action_logits: bool = False,
    ):
        """
        Args:
            v_t: (N, velocity_dim * chunk_horizon) noisy velocity at time t.
            t:   (N, 1) or (N,) diffusion timestep.
            data: PyG Data/Batch with x, edge_index, edge_attr, aux_features, batch.
            return_action_logits: If True, also return discrete action logits.

        Returns:
            flow_output: (N, velocity_dim * chunk_horizon) predicted flow vector.
            action_logits (optional): (N, action_dim * chunk_horizon).
        """
        x_patches = data.x          # (N, C, H, W)
        edge_index = data.edge_index  # (2, E)
        edge_attr = data.edge_attr    # (E, 2)
        aux_features = getattr(data, "aux_features", None)
        if aux_features is None:
            aux_features = data.bd_pred

        # Build batch_ptr from data.batch (or assume single graph)
        if hasattr(data, "ptr"):
            batch_ptr = data.ptr  # already computed by PyG DataLoader
        elif hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            num_graphs = int(batch.max().item()) + 1
            counts = torch.bincount(batch, minlength=num_graphs)
            batch_ptr = torch.cat([
                torch.zeros(1, dtype=torch.long, device=counts.device),
                counts.cumsum(0),
            ])
        else:
            batch_ptr = torch.tensor([0, x_patches.shape[0]], dtype=torch.long, device=x_patches.device)

        # 1. Visual + auxiliary encoding
        cnn_out = self.conv(x_patches)               # (N, cnn_out_dim)
        visual_feats = torch.cat([cnn_out, aux_features], dim=-1)  # (N, cnn_out_dim + aux)
        node_emb = self.visual_proj(visual_feats)    # (N, hidden_dim)

        # 2. Inject flow conditioning (v_t, t)
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (N, 1)
        node_emb = self.flow_proj(
            torch.cat([node_emb, v_t.float(), t.float()], dim=-1)
        )  # (N, hidden_dim)

        # 3. Transformer blocks
        for block in self.blocks:
            node_emb = block(node_emb, edge_index, edge_attr, batch_ptr)

        # 4. Output
        flow_output = self.post_mp(node_emb)  # (N, velocity_dim * chunk_horizon)

        if return_action_logits:
            action_logits = self.action_head(node_emb)  # (N, action_dim * chunk_horizon)
            return flow_output, action_logits

        return flow_output
