"""Component-wise graph normalization and post-normalization quality gating."""
from __future__ import annotations

import math
import torch

GRAPH_OPERATOR_VERSION = "component_row_norm_dual_quality_fixed_two_step_v7"


def validate_similarity_weight(lambda_sim: float) -> None:
    if not math.isfinite(lambda_sim) or lambda_sim < 0:
        raise ValueError("lambda_sim must be finite and non-negative")


def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize positive rows exactly; zero rows stay zero, without epsilon bias."""
    total = matrix.sum(dim=-1, keepdim=True)
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return matrix / denominator


def validate_quality(quality: torch.Tensor, num_balls: int) -> None:
    if quality.ndim != 1 or quality.shape[0] != num_balls:
        raise ValueError("Quality must have shape [M]")
    if not torch.isfinite(quality).all() or torch.any(quality < 0):
        raise ValueError("Quality must contain finite non-negative values")


def apply_quality(operator: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Apply Q[m] Q[n] without normalization; caller validates fixed quality."""
    quality = quality.to(operator)
    return operator * quality[:, None] * quality[None, :]


def build_transition_matrices(
    A_phy: torch.Tensor, A_sim: torch.Tensor,
    lambda_sim: float = 0.1,
    *, add_self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P_fwd/rev from raw source-row, target-column component matrices.

    Remove component self-loops. Reverse diffusion starts from raw transposes,
    not a transpose of normalized forward matrices. Supports [M,M] and [B,M,M].
    """
    validate_similarity_weight(lambda_sim)
    for name, matrix in (("A_phy", A_phy), ("A_sim", A_sim)):
        if matrix.ndim not in (2, 3) or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"{name} must have shape [M,M] or [B,M,M]")
        if not matrix.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(matrix).all() or torch.any(matrix < 0):
            raise ValueError(f"{name} must contain finite non-negative weights")
    if A_phy.shape[-2:] != A_sim.shape[-2:]:
        raise ValueError("Physical and similarity node dimensions must match")
    if A_phy.device != A_sim.device or A_phy.dtype != A_sim.dtype:
        raise ValueError("Physical and similarity matrices must share device/dtype")
    if A_phy.ndim == A_sim.ndim == 3 and A_phy.shape[0] != A_sim.shape[0] and 1 not in (A_phy.shape[0], A_sim.shape[0]):
        raise ValueError("Physical and similarity batch dimensions must broadcast")
    identity = torch.eye(A_phy.shape[-1], device=A_phy.device, dtype=A_phy.dtype)
    physical = A_phy * (1 - identity)
    similarity = A_sim * (1 - identity)

    def combine(physical, similarity):
        return row_normalize(row_normalize(physical)
                             + lambda_sim * row_normalize(similarity)
                             + (identity if add_self_loops else torch.zeros_like(identity)))

    return (combine(physical, similarity),
            combine(physical.transpose(-1, -2), similarity.transpose(-1, -2)))


def build_diffusion_operators(A_phy, A_sim, quality, lambda_sim=0.1, *, add_self_loops=False):
    """Normalize raw components, then apply dual-end quality once; no renormalization."""
    validate_quality(quality, A_phy.shape[-1])
    forward, reverse = build_transition_matrices(A_phy, A_sim, lambda_sim, add_self_loops=add_self_loops)
    return apply_quality(forward, quality), apply_quality(reverse, quality)


def graph_operator_config(lambda_sim: float, *, add_self_loops: bool = False) -> dict:
    self_loop_term = " + I" if add_self_loops else ""
    return {
        "version": GRAPH_OPERATOR_VERSION,
        "graph_convolution": "DiffConv", "diffusion_steps": 2,
        "lambda_sim": lambda_sim, "physical_coefficient": 1.0,
        "add_self_loops": add_self_loops,
        "adjacency_orientation": "raw rows=source, columns=target",
        "feature_aggregation": "row m receives neighbor-column n by left multiplication",
        "physical_weights": "sum physical edge weights between distinct balls",
        "component_self_loops": "removed before component normalization",
        "zero_row_policy": "remain zero",
        "forward": f"P_fwd = RowNorm(RowNorm(A_phy) + lambda_sim*RowNorm(A_sim){self_loop_term})",
        "reverse": f"P_rev = RowNorm(RowNorm(A_phy.T) + lambda_sim*RowNorm(A_sim.T){self_loop_term})",
        "quality_modulation": "G_fwd/rev = diag(Q) @ P_fwd/rev @ diag(Q)",
        "normalize_after_quality": False,
        "supports": "[H, G_fwd H, G_rev H, G_fwd^2 H, G_rev^2 H]",
        "root_policy": "zero-hop and residual features are not scaled by quality",
    }
