from __future__ import annotations

import json
from pathlib import Path

import h5py
from PIL import Image
import torch

from src.optics.psf import compute_point_source_psf


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values in [0, 1] to linear intensity."""
    threshold = 0.04045
    return torch.where(rgb <= threshold, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def green_channel_intensity(rgb: torch.Tensor) -> torch.Tensor:
    """First narrowband approximation around green-yellow wavelengths."""
    rgb_lin = srgb_to_linear(rgb)
    return rgb_lin[..., 1]


def load_hypersim_pilot_sample(root: str | Path, index: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    root = Path(root)
    manifest = json.loads((root / "pilot_manifest.json").read_text(encoding="utf-8"))
    item = manifest[index]
    rgb = Image.open(root / item["rgb"]).convert("RGB")
    rgb_tensor = torch.from_numpy(__import__("numpy").array(rgb)).to(torch.float32) / 255.0
    with h5py.File(root / item["depth_meters"], "r") as f:
        depth = torch.from_numpy(f["dataset"][:].astype("float32"))
    return rgb_tensor, depth, item


def ray_angle_maps(height: int, width: int, fov_x_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    fov_x = torch.deg2rad(torch.tensor(float(fov_x_deg)))
    fx = (width / 2.0) / torch.tan(fov_x / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    angle_x = torch.atan((x - cx) / fx)
    angle_y = torch.atan((y - cy) / fx)
    return angle_x, angle_y


def crop_center(image: torch.Tensor, size: int) -> torch.Tensor:
    height, width = image.shape[:2]
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return image[top : top + size, left : left + size]


def render_spatially_varying_psf(
    intensity: torch.Tensor,
    depth: torch.Tensor,
    angle_x: torch.Tensor,
    angle_y: torch.Tensor,
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    grid_size_m: float,
    psf_size: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Direct per-pixel spatially varying PSF rendering for small review patches."""
    if intensity.shape != depth.shape:
        raise ValueError("intensity and depth must have the same shape")
    height, width = intensity.shape
    out = torch.zeros_like(intensity)
    half = psf_size // 2
    for row in range(height):
        for col in range(width):
            z = depth[row, col]
            z_value = float(z.detach().cpu())
            if not bool(torch.isfinite(z.detach()).cpu()) or z_value <= 0:
                continue
            result = compute_point_source_psf(
                phase=phase,
                wavelength_m=wavelength_m,
                aperture_diameter_m=aperture_diameter_m,
                metasurface_sensor_distance_m=metasurface_sensor_distance_m,
                grid_size_m=grid_size_m,
                depth_m=z,
                angle_x_rad=angle_x[row, col],
                angle_y_rad=angle_y[row, col],
                crop_size=psf_size,
                normalize=True,
            )
            kernel = result.psf * intensity[row, col]
            y0 = max(row - half, 0)
            y1 = min(row + half + 1, height)
            x0 = max(col - half, 0)
            x1 = min(col + half + 1, width)
            ky0 = y0 - (row - half)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (col - half)
            kx1 = kx0 + (x1 - x0)
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] + kernel[ky0:ky1, kx0:kx1]
    return out / (out.sum() + eps)


def poisson_mahalanobis(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.sum((a - b) ** 2 / (a + eps))


def patch_depth_fisher_scalar(
    intensity: torch.Tensor,
    depth: torch.Tensor,
    angle_x: torch.Tensor,
    angle_y: torch.Tensor,
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    grid_size_m: float,
    psf_size: int,
    eps: float = 1e-9,
) -> torch.Tensor:
    def render_with_delta(delta: torch.Tensor) -> torch.Tensor:
        return render_spatially_varying_psf(
            intensity=intensity,
            depth=depth + delta,
            angle_x=angle_x,
            angle_y=angle_y,
            phase=phase,
            wavelength_m=wavelength_m,
            aperture_diameter_m=aperture_diameter_m,
            metasurface_sensor_distance_m=metasurface_sensor_distance_m,
            grid_size_m=grid_size_m,
            psf_size=psf_size,
        ).reshape(-1)

    delta = torch.tensor(0.0, dtype=depth.dtype)
    mu = render_with_delta(delta)
    jac = torch.func.jacfwd(render_with_delta)(delta)
    return torch.sum(jac**2 / (mu + eps))


def naive_depth_binned_forward(intensity: torch.Tensor, depth: torch.Tensor, depth_values: torch.Tensor, psfs: torch.Tensor) -> torch.Tensor:
    """Very small first forward model: bin pixels by depth and convolve each bin with its PSF.

    intensity/depth shape: [H, W].
    psfs shape: [Z, Kh, Kw].
    """
    if intensity.ndim != 2 or depth.ndim != 2:
        raise ValueError("intensity and depth must be 2D tensors")

    image = torch.zeros_like(intensity)
    for i, z in enumerate(depth_values):
        if i == 0:
            lower = -torch.inf
        else:
            lower = (depth_values[i - 1] + z) / 2
        if i == len(depth_values) - 1:
            upper = torch.inf
        else:
            upper = (z + depth_values[i + 1]) / 2

        mask = ((depth >= lower) & (depth < upper)).to(intensity.dtype)
        source = intensity * mask
        kernel = psfs[i].to(source.device, source.dtype)
        kernel = kernel / (kernel.sum() + 1e-12)
        pad_y = kernel.shape[-2] // 2
        pad_x = kernel.shape[-1] // 2
        image = image + torch.nn.functional.conv2d(
            source[None, None],
            kernel[None, None],
            padding=(pad_y, pad_x),
        )[0, 0, : intensity.shape[0], : intensity.shape[1]]

    return image
