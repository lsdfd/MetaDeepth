from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.models.simple_unet import INPUT_CHANNELS


@dataclass(frozen=True)
class DepthSample:
    image: torch.Tensor
    depth_m: torch.Tensor
    valid_mask: torch.Tensor


class SyntheticDepthDataset(Dataset):
    """Deterministic smoke dataset for train/eval plumbing only."""

    def __init__(
        self,
        length: int = 16,
        image_size: int = 128,
        input_mode: str = "rgb_meta",
        min_depth_m: float = 1.0,
        max_depth_m: float = 10.0,
    ):
        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"input_mode must be one of {tuple(INPUT_CHANNELS)}")
        self.length = int(length)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        h = w = self.image_size
        y = torch.linspace(0.0, 1.0, h).view(1, h, 1).expand(1, h, w)
        x = torch.linspace(0.0, 1.0, w).view(1, 1, w).expand(1, h, w)
        phase = 0.15 * float(index)

        depth_unit = 0.5 + 0.25 * torch.sin(2 * torch.pi * (x + phase)) + 0.2 * y
        depth_unit = depth_unit.clamp(0.0, 1.0)
        depth_m = self.min_depth_m + (self.max_depth_m - self.min_depth_m) * depth_unit

        rgb = torch.cat(
            [
                x,
                y,
                (0.5 + 0.5 * torch.sin(2 * torch.pi * (x + y + phase))).clamp(0.0, 1.0),
            ],
            dim=0,
        )
        rgb = (rgb + 0.03 * torch.randn(rgb.shape, generator=generator)).clamp(0.0, 1.0)
        meta = (1.0 / depth_m + 0.05 * torch.randn(depth_m.shape, generator=generator)).clamp(0.0, 1.0)

        if self.input_mode == "rgb":
            image = rgb
        elif self.input_mode == "meta":
            image = meta
        else:
            image = torch.cat([rgb, meta], dim=0)

        valid_mask = torch.ones_like(depth_m, dtype=torch.bool)
        return {"image": image, "depth_m": depth_m, "valid_mask": valid_mask}


class ManifestDepthDataset(Dataset):
    """Minimal future hook for HM3D/Hypersim processed samples.

    Manifest rows should eventually point to tensors or arrays containing:
    image, depth_m, valid_mask. It is intentionally not implemented until the
    processed data format is fixed.
    """

    def __init__(self, manifest_path: str | Path, input_mode: str = "rgb_meta"):
        self.manifest_path = Path(manifest_path)
        self.input_mode = input_mode
        raise NotImplementedError(
            "Create the processed RGB-D/Meta manifest after the PSF forward model is stable."
        )


def build_depth_dataset(cfg: dict[str, Any], split: str, input_mode: str) -> Dataset:
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "synthetic")
    if dataset_name == "synthetic":
        length_key = "train_length" if split == "train" else "val_length"
        return SyntheticDepthDataset(
            length=int(data_cfg.get(length_key, 16 if split == "train" else 4)),
            image_size=int(data_cfg.get("image_size", 128)),
            input_mode=input_mode,
            min_depth_m=float(cfg["model"].get("min_depth_m", 1.0)),
            max_depth_m=float(cfg["model"].get("max_depth_m", 10.0)),
        )
    if dataset_name == "manifest":
        return ManifestDepthDataset(data_cfg[f"{split}_manifest"], input_mode=input_mode)
    raise ValueError(f"unsupported depth dataset: {dataset_name}")
