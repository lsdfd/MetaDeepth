import torch

from .propagation import angular_spectrum
from .pupil import circular_aperture, make_xy_grid


def point_source_field(
    x: torch.Tensor,
    y: torch.Tensor,
    wavelength_m: float,
    depth_m: float,
    angle_x_rad: float = 0.0,
    angle_y_rad: float = 0.0,
) -> torch.Tensor:
    """Approximate the field from an off-axis point source at the metasurface plane."""
    k = 2 * torch.pi / wavelength_m

    # Point position under a simple optical-axis depth convention.
    x0 = depth_m * torch.tan(torch.as_tensor(angle_x_rad, device=x.device))
    y0 = depth_m * torch.tan(torch.as_tensor(angle_y_rad, device=x.device))
    r = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + depth_m**2)
    return torch.exp(1j * k * r) / r


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
    """Compute a scalar PSF for one depth and field angle."""
    n = phase.shape[-1]
    x, y, dx = make_xy_grid(n, pupil_size_m, device=str(phase.device))
    aperture = circular_aperture(x, y, aperture_diameter_m)

    u_in = point_source_field(x, y, wavelength_m, depth_m, angle_x_rad, angle_y_rad)
    transmittance = aperture * torch.exp(1j * phase)
    u0 = u_in * transmittance
    us = angular_spectrum(u0, dx, wavelength_m, metasurface_sensor_distance_m)
    psf = torch.abs(us) ** 2

    if normalize:
        psf = psf / (psf.sum() + 1e-12)
    return psf

