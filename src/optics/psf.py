from dataclasses import dataclass

import torch

from .propagation import angular_spectrum
from .pupil import circular_aperture, make_xy_grid


@dataclass(frozen=True)
class PsfResult:
    psf: torch.Tensor
    full_psf_sum: float
    crop_energy_ratio: float
    dx_m: float
    grid_size_m: float
    aperture_diameter_m: float
    depth_m: float
    angle_x_rad: float
    angle_y_rad: float
    wavelength_m: float
    metasurface_sensor_distance_m: float


def center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds image shape {tuple(image.shape)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return image[..., top : top + size, left : left + size]


def point_source_field(
    x: torch.Tensor,
    y: torch.Tensor,
    wavelength_m: float,
    depth_m: float,
    angle_x_rad: float = 0.0,
    angle_y_rad: float = 0.0,
) -> torch.Tensor:
    """Field from an off-axis point source at the metasurface plane.

    Implements the documented spherical-wave model:
    R_o = sqrt((x - z tan(alpha_x))^2 + (y - z tan(alpha_y))^2 + z^2),
    U_in = exp(i k R_o) / R_o.
    """
    k = 2 * torch.pi / wavelength_m

    x0 = depth_m * torch.tan(torch.as_tensor(angle_x_rad, device=x.device, dtype=x.dtype))
    y0 = depth_m * torch.tan(torch.as_tensor(angle_y_rad, device=x.device, dtype=x.dtype))
    r = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + depth_m**2)
    return torch.exp(1j * k * r) / r


def compute_point_source_psf(
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    grid_size_m: float,
    depth_m: float,
    angle_x_rad: float = 0.0,
    angle_y_rad: float = 0.0,
    crop_size: int | None = None,
    normalize: bool = True,
) -> PsfResult:
    """Compute a scalar point-source PSF for one depth and 2D field angle."""
    n = phase.shape[-1]
    x, y, dx = make_xy_grid(n, grid_size_m, device=str(phase.device))
    aperture = circular_aperture(x, y, aperture_diameter_m)

    u_in = point_source_field(x, y, wavelength_m, depth_m, angle_x_rad, angle_y_rad)
    u0 = aperture * torch.exp(1j * phase) * u_in
    us = angular_spectrum(u0, dx, wavelength_m, metasurface_sensor_distance_m)
    full_psf = torch.abs(us) ** 2
    full_sum_tensor = full_psf.sum()

    psf = center_crop(full_psf, crop_size) if crop_size is not None else full_psf
    crop_sum_tensor = psf.sum()
    if normalize:
        psf = psf / (crop_sum_tensor + 1e-12)

    full_sum = float(full_sum_tensor.detach().cpu())
    crop_sum = float(crop_sum_tensor.detach().cpu())
    crop_energy_ratio = crop_sum / full_sum if full_sum > 0.0 else 0.0

    return PsfResult(
        psf=psf,
        full_psf_sum=full_sum,
        crop_energy_ratio=crop_energy_ratio,
        dx_m=dx,
        grid_size_m=grid_size_m,
        aperture_diameter_m=aperture_diameter_m,
        depth_m=depth_m,
        angle_x_rad=angle_x_rad,
        angle_y_rad=angle_y_rad,
        wavelength_m=wavelength_m,
        metasurface_sensor_distance_m=metasurface_sensor_distance_m,
    )


def compute_psf(
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    pupil_size_m: float,
    depth_m: float,
    angle_x_rad: float = 0.0,
    angle_y_rad: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Backward-compatible PSF tensor API."""
    result = compute_point_source_psf(
        phase=phase,
        wavelength_m=wavelength_m,
        aperture_diameter_m=aperture_diameter_m,
        metasurface_sensor_distance_m=metasurface_sensor_distance_m,
        grid_size_m=pupil_size_m,
        depth_m=depth_m,
        angle_x_rad=angle_x_rad,
        angle_y_rad=angle_y_rad,
        crop_size=None,
        normalize=normalize,
    )
    return result.psf
