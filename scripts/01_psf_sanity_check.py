from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.optics.psf import compute_psf
from src.optics.pupil import make_xy_grid, random_phase, thin_lens_phase
from src.utils.config import load_config
from src.utils.units import mm, nm


def center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    """Crop the central PSF window used for first-stage visual checks."""
    height, width = image.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds image shape {tuple(image.shape)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return image[..., top : top + size, left : left + size]


def psf_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).detach().cpu())


def save_psf_grid(
    psfs: dict[tuple[float, float], torch.Tensor],
    depths_m: list[float],
    angles_deg: list[float],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(depths_m),
        len(angles_deg),
        figsize=(3.0 * len(angles_deg), 2.8 * len(depths_m)),
        squeeze=False,
    )

    vmax = max(float(psf.max().detach().cpu()) for psf in psfs.values())
    for row, depth_m in enumerate(depths_m):
        for col, angle_deg in enumerate(angles_deg):
            ax = axes[row][col]
            psf = psfs[(depth_m, angle_deg)].detach().cpu().numpy()
            ax.imshow(psf, cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"z={depth_m:g} m, ax={angle_deg:g} deg", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "baseline.yaml")

    system = cfg["system"]
    scene = cfg["scene"]
    simulation = cfg["simulation"]

    device = simulation.get("device", "cpu")
    wavelength_m = nm(float(system["wavelength_nm"]))
    aperture_m = mm(float(system["aperture_mm"]))
    distance_m = mm(float(system["metasurface_sensor_distance_mm"]))
    grid = int(system.get("debug_pupil_grid", system["pupil_grid"]))
    psf_grid = int(system["psf_grid"])
    depths_m = [float(v) for v in scene["depth_samples_m"]]
    angles_deg = [float(v) for v in scene["angle_samples_deg"]]

    x, y, _ = make_xy_grid(grid, aperture_m, device=device)
    phase_bank = {
        "lens": thin_lens_phase(x, y, wavelength_m, focal_length_m=distance_m),
        "random": random_phase((grid, grid), device=device, seed=0),
    }

    out_dir = root / "outputs" / "psf_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("First-stage PSF sanity check")
    print(f"lambda={system['wavelength_nm']} nm, aperture={system['aperture_mm']} mm, d={system['metasurface_sensor_distance_mm']} mm")
    print(f"depths={depths_m} m, angles={angles_deg} deg, pupil_grid={grid}, psf_grid={psf_grid}")

    for phase_name, phase in phase_bank.items():
        cropped_psfs: dict[tuple[float, float], torch.Tensor] = {}
        full_psfs: dict[tuple[float, float], torch.Tensor] = {}

        for depth_m in depths_m:
            for angle_deg in angles_deg:
                full = compute_psf(
                    phase=phase,
                    wavelength_m=wavelength_m,
                    aperture_diameter_m=aperture_m,
                    metasurface_sensor_distance_m=distance_m,
                    pupil_size_m=aperture_m,
                    depth_m=depth_m,
                    angle_x_rad=torch.deg2rad(torch.tensor(angle_deg)).item(),
                    angle_y_rad=0.0,
                    normalize=True,
                )
                crop = center_crop(full, psf_grid)
                crop = crop / (crop.sum() + 1e-12)
                full_psfs[(depth_m, angle_deg)] = full
                cropped_psfs[(depth_m, angle_deg)] = crop

                min_value = float(crop.min().detach().cpu())
                psf_sum = float(crop.sum().detach().cpu())
                print(
                    f"{phase_name:>6s} z={depth_m:>4g} m angle={angle_deg:>5g} deg "
                    f"min={min_value:.3e} sum={psf_sum:.6f} max={float(crop.max().detach().cpu()):.3e}"
                )

        center_angle = min(angles_deg, key=abs)
        center_depth = depths_m[len(depths_m) // 2]
        depth_delta = psf_distance(cropped_psfs[(depths_m[0], center_angle)], cropped_psfs[(depths_m[-1], center_angle)])
        angle_delta = psf_distance(cropped_psfs[(center_depth, angles_deg[0])], cropped_psfs[(center_depth, angles_deg[-1])])
        print(f"{phase_name:>6s} depth-change mean_abs_delta={depth_delta:.3e}")
        print(f"{phase_name:>6s} angle-change mean_abs_delta={angle_delta:.3e}")

        save_psf_grid(
            cropped_psfs,
            depths_m,
            angles_deg,
            title=f"{phase_name.capitalize()} phase PSFs",
            out_path=out_dir / f"{phase_name}_phase_psf_grid.png",
        )

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
