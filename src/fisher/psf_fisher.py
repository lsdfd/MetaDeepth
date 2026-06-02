import torch

from src.optics.psf import compute_psf


def poisson_fisher_from_jacobian(mu: torch.Tensor, jacobian: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute F = J^T diag(1 / mu) J for Poisson noise.

    mu has shape [pixels].
    jacobian has shape [num_params, pixels].
    """
    weight = 1.0 / (mu + eps)
    return (jacobian * weight.unsqueeze(0)) @ jacobian.T


def finite_difference_psf_jacobian(
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    pupil_size_m: float,
    depth_m: float,
    angle_x_rad: float,
    angle_y_rad: float,
    dz_m: float = 1e-3,
    dangle_rad: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference Jacobian of PSF wrt [z, angle_x, angle_y]."""
    base = compute_psf(
        phase,
        wavelength_m,
        aperture_diameter_m,
        metasurface_sensor_distance_m,
        pupil_size_m,
        depth_m,
        angle_x_rad,
        angle_y_rad,
    )

    samples = [
        (depth_m + dz_m, angle_x_rad, angle_y_rad, depth_m - dz_m, angle_x_rad, angle_y_rad, 2 * dz_m),
        (depth_m, angle_x_rad + dangle_rad, angle_y_rad, depth_m, angle_x_rad - dangle_rad, angle_y_rad, 2 * dangle_rad),
        (depth_m, angle_x_rad, angle_y_rad + dangle_rad, depth_m, angle_x_rad, angle_y_rad - dangle_rad, 2 * dangle_rad),
    ]

    rows = []
    for zp, axp, ayp, zm, axm, aym, denom in samples:
        plus = compute_psf(
            phase,
            wavelength_m,
            aperture_diameter_m,
            metasurface_sensor_distance_m,
            pupil_size_m,
            zp,
            axp,
            ayp,
        )
        minus = compute_psf(
            phase,
            wavelength_m,
            aperture_diameter_m,
            metasurface_sensor_distance_m,
            pupil_size_m,
            zm,
            axm,
            aym,
        )
        rows.append(((plus - minus) / denom).reshape(-1))

    return base.reshape(-1), torch.stack(rows, dim=0)

