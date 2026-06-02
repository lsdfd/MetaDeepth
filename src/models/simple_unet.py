from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


INPUT_CHANNELS = {
    "rgb": 3,
    "meta": 1,
    "rgb_meta": 4,
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthUNet(nn.Module):
    """Small metric-depth baseline with UNet/FastDepth-style skip fusion.

    This is intentionally compact: it gives the project a trainable depth
    decoder now, while leaving DPT/Depth-Anything style encoders for later.
    """

    def __init__(
        self,
        input_mode: str = "rgb_meta",
        min_depth_m: float = 1.0,
        max_depth_m: float = 10.0,
        width: int = 32,
    ):
        super().__init__()
        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"input_mode must be one of {tuple(INPUT_CHANNELS)}")

        self.input_mode = input_mode
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        channels = INPUT_CHANNELS[input_mode]
        self.enc1 = ConvBlock(channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bridge = ConvBlock(width * 4, width * 8)

        self.dec3 = ConvBlock(width * 8 + width * 4, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bridge(F.avg_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        unit_depth = torch.sigmoid(self.head(d1))
        return self.min_depth_m + (self.max_depth_m - self.min_depth_m) * unit_depth


class TinyDepthNet(nn.Module):
    """Backward-compatible tiny decoder from one encoded image to depth."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FoundationDepthAdapter(nn.Module):
    """Placeholder for Depth Anything / DPT / MiDaS style backbones.

    Keep this dependency-free until the physical forward model and dataset
    choice are stable enough to justify large pretrained weights.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        raise NotImplementedError(
            "Foundation model adapters are reserved for the later RGB+Meta stage."
        )


def build_depth_model(
    input_mode: str = "rgb_meta",
    min_depth_m: float = 1.0,
    max_depth_m: float = 10.0,
    width: int = 32,
) -> DepthUNet:
    return DepthUNet(
        input_mode=input_mode,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        width=width,
    )
