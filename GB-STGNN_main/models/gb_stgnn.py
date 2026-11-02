from __future__ import annotations

import torch
from torch import nn
from .graph_ops import build_diffusion_operators, validate_similarity_weight


def aggregate_nodes_to_balls(
    node_features: torch.Tensor,
    node_to_ball: torch.Tensor,
    num_balls: int,
    ball_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool node representations within each granular ball."""
    ball_features = node_features.new_zeros(
        node_features.shape[0], num_balls, node_features.shape[2]
    )
    index = node_to_ball.view(1, -1, 1).expand_as(node_features)
    ball_features.scatter_add_(1, index, node_features)
    if ball_sizes is None:
        ball_sizes = torch.bincount(node_to_ball, minlength=num_balls).clamp_min(1)
    ball_sizes = ball_sizes.to(node_features).view(1, -1, 1)
    return ball_features / ball_sizes


def project_balls_to_nodes(
    ball_features: torch.Tensor,
    node_to_ball: torch.Tensor,
) -> torch.Tensor:
    """Broadcast each granular-ball representation to its member nodes."""
    return ball_features[:, node_to_ball, :]


class DiffConv(nn.Module):
    """Fixed two-step forward/reverse diffusion, including zero-hop support."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.diffusion_steps = 2
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim * 5, out_dim)

    @staticmethod
    def _batch_operator(A: torch.Tensor, batch_size: int, num_nodes: int,
                        dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if A.ndim == 2:
            if A.shape != (num_nodes, num_nodes):
                raise ValueError(f"A must have shape {(num_nodes, num_nodes)}, got {tuple(A.shape)}")
            A_batch = A.unsqueeze(0).expand(batch_size, -1, -1)
        elif A.ndim == 3:
            if A.shape[-2:] != (num_nodes, num_nodes):
                raise ValueError(f"A must end with shape {(num_nodes, num_nodes)}, got {tuple(A.shape)}")
            A_batch = A.expand(batch_size, -1, -1) if A.shape[0] == 1 else A
            if A_batch.shape[0] != batch_size:
                raise ValueError(f"A batch dimension must be 1 or B={batch_size}, got {A.shape[0]}")
        else:
            raise ValueError(f"A must have shape [N,N] or [B,N,N], got {tuple(A.shape)}")
        A_batch = A_batch.to(device=device, dtype=dtype)
        return A_batch

    def forward(self, x: torch.Tensor, G_fwd: torch.Tensor, G_rev: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, N, d], got {tuple(x.shape)}")
        batch_size, num_nodes, _ = x.shape
        G_fwd = self._batch_operator(G_fwd, batch_size, num_nodes, x.dtype, x.device)
        G_rev = self._batch_operator(G_rev, batch_size, num_nodes, x.dtype, x.device)
        supports = [x]
        h_fwd = h_rev = x
        for _ in range(2):
            h_fwd = torch.bmm(G_fwd, h_fwd)
            h_rev = torch.bmm(G_rev, h_rev)
            supports.extend([h_fwd, h_rev])
        return self.linear(self.dropout(torch.cat(supports, dim=-1)))


class GBSTGNN(nn.Module):
    """GB-STGNN with mean-pooled balls; static decoder inputs are opt-in."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_gru_layers: int,
        num_diffconv_layers: int,
        horizon_dim: int,
        dropout: float,
        static_dim: int = 0,
        lambda_sim: float = 0.1,
        add_self_loops: bool = False,
        node_only_decoder: bool = False,
        quality_modulation: bool = True,
    ) -> None:
        super().__init__()
        if num_diffconv_layers < 1:
            raise ValueError("num_diffconv_layers must be at least 1.")
        self.static_dim = static_dim
        if static_dim < 0:
            raise ValueError("static_dim must be non-negative")
        validate_similarity_weight(lambda_sim)
        self.lambda_sim = lambda_sim
        self.add_self_loops = add_self_loops
        self.node_only_decoder = node_only_decoder
        self.quality_modulation = quality_modulation
        gru_dropout = dropout if num_gru_layers > 1 else 0.0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_gru_layers, batch_first=True, dropout=gru_dropout)
        self.ball_diffusion_layers = nn.ModuleList(
            [DiffConv(hidden_dim, hidden_dim, dropout)
             for _ in range(0 if node_only_decoder else num_diffconv_layers)]
        )
        self.activation = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * (1 if node_only_decoder else 3) + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_dim),
        )
        self._graph_prepared = False
        for name in ("_fixed_node_to_ball", "_fixed_G_fwd", "_fixed_G_rev", "_fixed_ball_sizes", "_fixed_static"):
            self.register_buffer(name, None, persistent=False)

    @property
    def graph_is_prepared(self) -> bool:
        return self._graph_prepared

    @torch.no_grad()
    def prepare_graph(self, node_to_ball, A_phy, A_sim, ball_quality, static=None) -> None:
        """Snapshot a fixed graph once. Call again when topology/quality/weights change.

        Buffers follow model device/dtype moves but are not stored in checkpoints.
        Differentiable/dynamic graphs must use the explicit forward arguments.
        """
        for tensor in (A_phy, A_sim, ball_quality, static):
            if tensor is not None and tensor.requires_grad:
                raise ValueError("prepare_graph only supports fixed, non-learnable attributes")
        if A_phy.ndim != 2 or A_sim.ndim != 2:
            raise ValueError("prepare_graph requires one fixed graph with shape [M,M]")
        parameter = next(self.parameters())
        node = node_to_ball.to(device=parameter.device, dtype=torch.long).detach().clone()
        if node.ndim != 1 or not node.numel():
            raise ValueError("node_to_ball must be a nonempty vector")
        num_balls = A_phy.shape[-1]
        if torch.any(node < 0) or torch.any(node >= num_balls):
            raise ValueError("node_to_ball contains invalid ball indices")
        if A_phy.shape != (num_balls, num_balls) or A_sim.shape != A_phy.shape:
            raise ValueError("Fixed graph dimensions must match the quality vector")
        forward, reverse = build_diffusion_operators(
            A_phy.to(parameter), A_sim.to(parameter),
            ball_quality if self.quality_modulation else torch.ones_like(ball_quality),
            self.lambda_sim,
            add_self_loops=self.add_self_loops,
        )
        if self.static_dim and (static is None or static.shape != (node.numel(), self.static_dim)):
            raise ValueError("Fixed static attributes must have shape [N,static_dim]")
        if not self.static_dim and static is not None:
            raise ValueError("Do not cache static attributes for a decoder without static inputs")
        self._fixed_node_to_ball = node
        self._fixed_G_fwd, self._fixed_G_rev = forward.detach().clone(), reverse.detach().clone()
        self._fixed_ball_sizes = torch.bincount(node, minlength=num_balls).clamp_min(1).to(parameter)
        self._fixed_static = static.to(parameter).detach().clone() if self.static_dim else None
        self._prepared_lambda_sim = self.lambda_sim
        self._prepared_add_self_loops = self.add_self_loops
        self._prepared_quality_modulation = self.quality_modulation
        self.gru.flatten_parameters()
        self._graph_prepared = True

    def clear_prepared_graph(self) -> None:
        self._graph_prepared = False
        for name in ("_fixed_node_to_ball", "_fixed_G_fwd", "_fixed_G_rev", "_fixed_ball_sizes", "_fixed_static"):
            setattr(self, name, None)

    def forward(
        self,
        x: torch.Tensor,
        node_to_ball: torch.Tensor | None = None,
        hydro_dist_to_ball: torch.Tensor | None = None,
        A_phy: torch.Tensor | None = None,
        A_sim: torch.Tensor | None = None,
        ball_quality: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, L, N, F], got {tuple(x.shape)}")
        batch_size, seq_len, num_nodes, feature_dim = x.shape
        _ = hydro_dist_to_ball
        graph_args = (node_to_ball, A_phy, A_sim, ball_quality)
        ball_sizes = None
        if all(value is None for value in graph_args):
            if not self.graph_is_prepared:
                raise ValueError("Call prepare_graph or provide all raw graph arguments")
            if (self._prepared_lambda_sim != self.lambda_sim
                    or self._prepared_add_self_loops != self.add_self_loops
                    or self._prepared_quality_modulation != self.quality_modulation
):
                raise ValueError("Graph mixture weights changed; call prepare_graph again")
            node_to_ball = self._fixed_node_to_ball
            G_fwd, G_rev = self._fixed_G_fwd, self._fixed_G_rev
            ball_sizes = self._fixed_ball_sizes
            num_balls = G_fwd.shape[-1]
            if static is None:
                static = self._fixed_static
        else:
            if any(value is None for value in graph_args):
                raise ValueError("Provide all raw graph arguments together")
            node_to_ball = node_to_ball.to(device=x.device, dtype=torch.long)
            num_balls = A_phy.shape[-1]
            for name, matrix in (("A_phy", A_phy), ("A_sim", A_sim)):
                if matrix.shape[-2:] != (num_balls, num_balls):
                    raise ValueError(f"{name} must match the ball count {num_balls}")
            G_fwd, G_rev = build_diffusion_operators(
                A_phy.to(device=x.device, dtype=x.dtype),
                A_sim.to(device=x.device, dtype=x.dtype),
                ball_quality if self.quality_modulation else torch.ones_like(ball_quality),
                self.lambda_sim,
                add_self_loops=self.add_self_loops,
                )
        if node_to_ball.shape != (num_nodes,):
            raise ValueError("Node input dimensions do not match the graph partition")

        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, feature_dim)
        _, hidden = self.gru(x_flat)
        h_node = hidden[-1].reshape(batch_size, num_nodes, -1)
        h_fused = h_node
        if not self.node_only_decoder:
            h_ball_local = aggregate_nodes_to_balls(h_node, node_to_ball, num_balls, ball_sizes)
            h_ball_diff = h_ball_local
            for layer in self.ball_diffusion_layers:
                h_ball_diff = self.activation(
                    h_ball_diff + layer(h_ball_diff, G_fwd, G_rev)
                )
            ball_to_node = project_balls_to_nodes(torch.cat([h_ball_local, h_ball_diff], dim=-1), node_to_ball)
            h_fused = torch.cat([h_node, ball_to_node], dim=-1)
        if self.static_dim:
            if static is None:
                raise ValueError("Static attributes are required for this model setting")
            static = static.to(device=x.device, dtype=h_node.dtype)
            if static.ndim == 2:
                static = static.unsqueeze(0).expand(batch_size, -1, -1)
            if static.shape != (batch_size, num_nodes, self.static_dim):
                raise ValueError("Static attribute dimensions do not match the model")
            h_fused = torch.cat([h_fused, static], dim=-1)
        return self.decoder(h_fused)

# Backward-compatible name for existing checkpoint/inference callers.
GBSTGNNDiffusion = GBSTGNN

# Compatibility name for callers of the original diffusion layer.
DirectedDiffusionConv = DiffConv
