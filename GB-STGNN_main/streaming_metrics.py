"""Global masked metrics using bounded-memory, float64 sufficient statistics."""
import torch
from metrics import EPS


class StreamingMetrics:
    """Accumulate each horizon and the overall metrics without copying predictions."""
    def __init__(self):
        self.statistics = None

    @torch.no_grad()
    def update(self, prediction, target, mask):
        if prediction.shape != target.shape or target.shape != mask.shape or target.ndim != 3:
            raise ValueError("Expected matching [B,N,H] predictions, targets and masks")
        valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
        pred = torch.where(valid, prediction, 0).to(torch.float64)
        true = torch.where(valid, target, 0).to(torch.float64)
        residual = pred - true
        dims = (0, 1)
        statistics = torch.stack([
            valid.sum(dims).to(torch.float64), residual.abs().sum(dims),
            residual.square().sum(dims), pred.sum(dims), true.sum(dims),
            pred.square().sum(dims), true.square().sum(dims), (pred * true).sum(dims),
        ], dim=-1)
        if self.statistics is None:
            self.statistics = statistics
        else:
            self.statistics.add_(statistics)

    def compute(self):
        if self.statistics is None:
            raise ValueError("Evaluation loader has no batches")
        # Row zero pools ALL valid values; remaining rows represent individual horizons.
        stats = torch.cat([self.statistics.sum(0, keepdim=True), self.statistics], dim=0)
        count, absolute, squared, p_sum, t_sum, p_square, t_square, cross = stats.unbind(-1)
        divisor = count.clamp_min(1)
        p_mean, t_mean = p_sum / divisor, t_sum / divisor
        p_var = (p_square / divisor - p_mean.square()).clamp_min(0)
        t_var = (t_square / divisor - t_mean.square()).clamp_min(0)
        p_std, t_std = p_var.sqrt(), t_var.sqrt()
        correlation = (cross / divisor - p_mean * t_mean) / (p_std * t_std + EPS)
        alpha = p_std / (t_std + EPS)
        beta = p_mean / (t_mean + EPS)
        mae = absolute / divisor
        rmse = (squared / divisor + EPS).sqrt()
        nse = 1 - squared / ((t_square - t_sum.square() / divisor).clamp_min(0) + EPS)
        kge = 1 - ((correlation - 1).square() + (alpha - 1).square() + (beta - 1).square()).sqrt()
        values = torch.stack([mae, rmse, nse, kge], dim=-1)
        values = torch.where((count > 0)[:, None], values, torch.full_like(values, float("nan")))
        values[:, 3] = torch.where(count >= 2, values[:, 3], float("nan"))
        result = [dict(zip(("mae", "rmse", "nse", "kge"), row)) for row in values.cpu().tolist()]
        return result[0], result[1:]
