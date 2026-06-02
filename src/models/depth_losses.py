from __future__ import annotations

import torch


def valid_depth_mask(depth_m: torch.Tensor, min_depth_m: float = 1.0, max_depth_m: float = 10.0) -> torch.Tensor:
    return torch.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)


def masked_l1_loss(pred_m: torch.Tensor, target_m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    if mask.sum() == 0:
        return pred_m.sum() * 0.0
    return torch.abs(pred_m[mask] - target_m[mask]).mean()


def silog_loss(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    mask: torch.Tensor,
    variance_weight: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    if mask.sum() == 0:
        return pred_m.sum() * 0.0
    diff = torch.log(pred_m[mask].clamp_min(eps)) - torch.log(target_m[mask].clamp_min(eps))
    return torch.sqrt((diff**2).mean() - variance_weight * diff.mean() ** 2 + eps)


def depth_loss(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str = "l1_silog",
) -> torch.Tensor:
    if loss_name == "l1":
        return masked_l1_loss(pred_m, target_m, mask)
    if loss_name == "silog":
        return silog_loss(pred_m, target_m, mask)
    if loss_name == "l1_silog":
        return masked_l1_loss(pred_m, target_m, mask) + 0.2 * silog_loss(pred_m, target_m, mask)
    raise ValueError(f"unsupported depth loss: {loss_name}")
