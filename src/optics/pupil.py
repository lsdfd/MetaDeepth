import torch


def make_xy_grid(num: int, size_m: float, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create a square physical coordinate grid.

    Returns x, y coordinates in meters and the sample spacing dx.
    """
    coords = torch.linspace(-size_m / 2, size_m / 2, num, device=device)
    dx = float(coords[1] - coords[0])
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return x, y, dx


def circular_aperture(x: torch.Tensor, y: torch.Tensor, diameter_m: float) -> torch.Tensor:
    radius = diameter_m / 2
    return ((x**2 + y**2) <= radius**2).to(x.dtype)


def random_phase(shape: tuple[int, int], device: str = "cpu", seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return 2 * torch.pi * torch.rand(shape, generator=generator, device=device)


def thin_lens_phase(x: torch.Tensor, y: torch.Tensor, wavelength_m: float, focal_length_m: float) -> torch.Tensor:
    """Paraxial thin-lens phase, useful as a sanity-check baseline."""
    k = 2 * torch.pi / wavelength_m
    return -k * (x**2 + y**2) / (2 * focal_length_m)

