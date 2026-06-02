import torch


def make_xy_grid(num: int, size_m: float, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create a square physical coordinate grid.

    Returns x, y coordinates in meters and the sample spacing dx.
    """
    dx = size_m / num
    coords = (torch.arange(num, device=device, dtype=torch.float32) - num // 2) * dx
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return x, y, dx


def circular_aperture(x: torch.Tensor, y: torch.Tensor, diameter_m: float) -> torch.Tensor:
    radius = diameter_m / 2
    return ((x**2 + y**2) <= radius**2).to(x.dtype)


def random_phase(shape: tuple[int, int], device: str = "cpu", seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return 2 * torch.pi * torch.rand(shape, generator=generator, device=device)


def exact_lens_phase(x: torch.Tensor, y: torch.Tensor, wavelength_m: float, focal_length_m: float) -> torch.Tensor:
    """Equal-optical-path phase for focusing a normally incident wave at distance f."""
    k = 2 * torch.pi / wavelength_m
    return -k * (torch.sqrt(x**2 + y**2 + focal_length_m**2) - focal_length_m)


def thin_lens_phase(x: torch.Tensor, y: torch.Tensor, wavelength_m: float, focal_length_m: float) -> torch.Tensor:
    """Backward-compatible alias for the exact lens baseline used in this project."""
    return exact_lens_phase(x, y, wavelength_m, focal_length_m)


def spiral_phase(x: torch.Tensor, y: torch.Tensor, charge: int = 1) -> torch.Tensor:
    theta = torch.atan2(y, x)
    return charge * theta


def double_helix_phase(
    x: torch.Tensor,
    y: torch.Tensor,
    charge: int = 1,
    phase_offset_rad: float = torch.pi / 2,
) -> torch.Tensor:
    theta = torch.atan2(y, x)
    field = torch.exp(1j * charge * theta) + torch.exp(
        1j * (-charge * theta + phase_offset_rad)
    )
    return torch.angle(field)


def multiring_spiral_phase(
    x: torch.Tensor,
    y: torch.Tensor,
    aperture_diameter_m: float,
    charges: tuple[int, ...] = (1, -1, 2, -2),
) -> torch.Tensor:
    radius = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    aperture_radius = aperture_diameter_m / 2
    phase = torch.zeros_like(x)
    for idx, charge in enumerate(charges):
        inner = aperture_radius * idx / len(charges)
        outer = aperture_radius * (idx + 1) / len(charges)
        mask = (radius >= inner) & (radius < outer)
        phase = torch.where(mask, charge * theta, phase)
    return phase
