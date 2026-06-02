from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.optics.psf import compute_psf
from src.optics.pupil import random_phase
from src.utils.config import load_config
from src.utils.units import mm, nm


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "baseline.yaml")

    system = cfg["system"]
    scene = cfg["scene"]
    wavelength_m = nm(system["wavelength_nm"])
    aperture_m = mm(system["aperture_mm"])
    distance_m = mm(system["metasurface_sensor_distance_mm"])
    grid = int(system.get("debug_pupil_grid", system["pupil_grid"]))

    phase = random_phase((grid, grid), seed=0)
    psf = compute_psf(
        phase=phase,
        wavelength_m=wavelength_m,
        aperture_diameter_m=aperture_m,
        metasurface_sensor_distance_m=distance_m,
        pupil_size_m=aperture_m,
        depth_m=float(scene["depth_samples_m"][1]),
        angle_x_rad=0.0,
        angle_y_rad=0.0,
    )

    out_dir = root / "outputs" / "psf_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "random_phase_psf_demo.png"

    plt.figure(figsize=(5, 4))
    plt.imshow(psf.detach().cpu().numpy(), cmap="magma")
    plt.colorbar(label="normalized intensity")
    plt.title("Random phase PSF demo")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
