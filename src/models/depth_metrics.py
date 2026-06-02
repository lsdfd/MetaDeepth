from __future__ import annotations

import torch


@torch.no_grad()
def depth_metrics(
    pred_m: torch.Tensor,
    target_m: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    mask = mask.to(dtype=torch.bool)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "abs_rel": float("nan"), "delta1": float("nan")}

    pred = pred_m[mask].clamp_min(eps)
    target = target_m[mask].clamp_min(eps)
    error = pred - target
    ratio = torch.maximum(pred / target, target / pred)
    return {
        "mae": float(torch.abs(error).mean().detach().cpu()),
        "rmse": float(torch.sqrt((error**2).mean()).detach().cpu()),
        "abs_rel": float((torch.abs(error) / target).mean().detach().cpu()),
        "delta1": float((ratio < 1.25).to(torch.float32).mean().detach().cpu()),
    }


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float32)
        out[key] = float(torch.nanmean(values).item())
    return out
