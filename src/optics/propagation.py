import torch


def angular_spectrum(
    field: torch.Tensor,
    dx_m: float,
    wavelength_m: float,
    distance_m: float,
) -> torch.Tensor:
    """Propagate a scalar complex field with the angular spectrum method."""
    if not torch.is_complex(field):
        raise TypeError("field must be a complex tensor")

    ny, nx = field.shape[-2:]
    fy = torch.fft.fftfreq(ny, d=dx_m, device=field.device)
    fx = torch.fft.fftfreq(nx, d=dx_m, device=field.device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    k = 2 * torch.pi / wavelength_m
    kz_sq = k**2 - (2 * torch.pi * fx_grid) ** 2 - (2 * torch.pi * fy_grid) ** 2
    kz = torch.sqrt(kz_sq.to(torch.complex64))
    transfer = torch.exp(1j * kz * distance_m)

    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)

