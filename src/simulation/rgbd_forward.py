import torch


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values in [0, 1] to linear intensity."""
    threshold = 0.04045
    return torch.where(rgb <= threshold, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def green_channel_intensity(rgb: torch.Tensor) -> torch.Tensor:
    """First narrowband approximation around green-yellow wavelengths."""
    rgb_lin = srgb_to_linear(rgb)
    return rgb_lin[..., 1]


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

