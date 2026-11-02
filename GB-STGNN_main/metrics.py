from __future__ import annotations

import torch


EPS = 1e-8


def _valid_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if y_pred.shape != y_true.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"y_pred, y_true, and mask must have the same shape, got "
            f"{tuple(y_pred.shape)}, {tuple(y_true.shape)}, {tuple(mask.shape)}"
        )
    valid = mask.bool() & torch.isfinite(y_pred) & torch.isfinite(y_true)
    return y_pred[valid], y_true[valid]


def masked_mae(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, true = _valid_values(y_pred, y_true, mask)
    if pred.numel() == 0:
        return y_pred.new_tensor(float("nan"))
    return torch.mean(torch.abs(pred - true))


def masked_rmse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, true = _valid_values(y_pred, y_true, mask)
    if pred.numel() == 0:
        return y_pred.new_tensor(float("nan"))
    return torch.sqrt(torch.mean((pred - true) ** 2) + EPS)


def masked_nse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, true = _valid_values(y_pred, y_true, mask)
    if pred.numel() == 0:
        return y_pred.new_tensor(float("nan"))
    numerator = torch.sum((true - pred) ** 2)
    denominator = torch.sum((true - torch.mean(true)) ** 2)
    return 1.0 - numerator / (denominator + EPS)


def masked_kge(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, true = _valid_values(y_pred, y_true, mask)
    if pred.numel() < 2:
        return y_pred.new_tensor(float("nan"))

    pred_mean = torch.mean(pred)
    true_mean = torch.mean(true)
    pred_std = torch.std(pred, unbiased=False)
    true_std = torch.std(true, unbiased=False)

    covariance = torch.mean((pred - pred_mean) * (true - true_mean))
    correlation = covariance / (pred_std * true_std + EPS)
    alpha = pred_std / (true_std + EPS)
    beta = pred_mean / (true_mean + EPS)
    return 1.0 - torch.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
