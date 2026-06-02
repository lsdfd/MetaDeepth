from __future__ import annotations

import torch
import torch.nn.functional as F

from src.optics.psf import compute_point_source_psf


def phase_from_control(control: torch.Tensor, out_size: int) -> torch.Tensor:
    """Map a low-dimensional trainable control grid to a [0, 2pi] phase map."""
    phase = 2 * torch.pi * torch.sigmoid(control)
    phase = F.interpolate(
        phase[None, None],
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return phase


def psf_mu(
    theta: torch.Tensor,
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    grid_size_m: float,
    crop_size: int,
    background_photons: float,
) -> torch.Tensor:
    """Poisson mean image for theta=[z, angle_x, angle_y, photon_count]."""
    result = compute_point_source_psf(
        phase=phase,
        wavelength_m=wavelength_m,
        aperture_diameter_m=aperture_diameter_m,
        metasurface_sensor_distance_m=metasurface_sensor_distance_m,
        grid_size_m=grid_size_m,
        depth_m=theta[0],
        angle_x_rad=theta[1],
        angle_y_rad=theta[2],
        crop_size=crop_size,
        normalize=True,
    )
    return theta[3] * result.psf.reshape(-1) + background_photons


def poisson_fisher(mu: torch.Tensor, jacobian: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """F_mn = sum_i (d mu_i/d theta_m)(d mu_i/d theta_n)/(mu_i + eps)."""
    weighted_j = jacobian / torch.sqrt(mu + eps).unsqueeze(1)
    return weighted_j.T @ weighted_j


def fisher_for_theta(
    theta: torch.Tensor,
    phase: torch.Tensor,
    wavelength_m: float,
    aperture_diameter_m: float,
    metasurface_sensor_distance_m: float,
    grid_size_m: float,
    crop_size: int,
    background_photons: float,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Poisson Fisher matrix and mean image for theta=[z, ax, ay, photons]."""

    def fn(local_theta: torch.Tensor) -> torch.Tensor:
        return psf_mu(
            theta=local_theta,
            phase=phase,
            wavelength_m=wavelength_m,
            aperture_diameter_m=aperture_diameter_m,
            metasurface_sensor_distance_m=metasurface_sensor_distance_m,
            grid_size_m=grid_size_m,
            crop_size=crop_size,
            background_photons=background_photons,
        )

    mu = fn(theta)
    jacobian = torch.func.jacfwd(fn)(theta)
    return poisson_fisher(mu, jacobian, eps=eps), mu


def effective_depth_fisher(fisher: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Schur complement for depth after nuisance [angle_x, angle_y, photons]."""
    f_zz = fisher[0, 0]
    f_zeta = fisher[0, 1:]
    f_etaz = fisher[1:, 0]
    f_etaeta = fisher[1:, 1:]
    inv_etaeta = torch.linalg.pinv(f_etaeta + eps * torch.eye(3, device=fisher.device, dtype=fisher.dtype))
    return f_zz - f_zeta @ inv_etaeta @ f_etaz


def crlb_from_information(info: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return 1.0 / (info + eps)


def psf_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """PSF separability metrics beyond pixel L1."""
    pa = a.reshape(-1)
    pb = b.reshape(-1)
    pa = pa / (pa.sum() + eps)
    pb = pb / (pb.sum() + eps)
    mix = 0.5 * (pa + pb)
    js = 0.5 * (
        (pa * (torch.log(pa + eps) - torch.log(mix + eps))).sum()
        + (pb * (torch.log(pb + eps) - torch.log(mix + eps))).sum()
    )
    cosine = 1.0 - torch.dot(pa, pb) / (
        (torch.linalg.vector_norm(pa) + eps) * (torch.linalg.vector_norm(pb) + eps)
    )
    fa = torch.abs(torch.fft.fft2(a))
    fb = torch.abs(torch.fft.fft2(b))
    freq_l1 = torch.mean(torch.abs(fa - fb))
    return {
        "l1": float(torch.mean(torch.abs(pa - pb)).detach().cpu()),
        "cosine_distance": float(cosine.detach().cpu()),
        "js_divergence": float(js.detach().cpu()),
        "freq_l1": float(freq_l1.detach().cpu()),
    }
