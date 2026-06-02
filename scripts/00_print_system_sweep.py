from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.system_config import load_first_stage_system_config


def _join(values: tuple[float, ...], unit: str) -> str:
    return ", ".join(f"{value:g} {unit}" for value in values)


def main() -> None:
    cfg = load_first_stage_system_config(ROOT / "configs" / "system_first_stage.yaml")
    sweep_rows = cfg.iter_psf_sweep()

    print("First-stage metasurface depth imaging system")
    print(f"wavelength: {cfg.wavelength_nm:g} nm")
    print(f"bandwidth: {cfg.bandwidth_nm:g} nm")
    print(f"FOV scan: {_join(cfg.fov_scan_deg, 'deg')}")
    print(f"depth range: {cfg.depth_range_m[0]:g}-{cfg.depth_range_m[1]:g} m")
    print(f"depth samples: {_join(cfg.depth_samples_m, 'm')}")
    print(f"aperture scan: {_join(cfg.aperture_scan_mm, 'mm')}")
    print(
        "metasurface-CMOS distance scan: "
        f"{_join(cfg.metasurface_sensor_distance_scan_mm, 'mm')}"
    )
    print(
        "sensor: "
        f"{cfg.sensor.width} x {cfg.sensor.height}, "
        f"{cfg.sensor.pixel_pitch_um:g} um pixel pitch"
    )
    print(
        "pupil grids: "
        f"main={cfg.pupil_grid_main}, debug={cfg.pupil_grid_debug}, psf={cfg.psf_grid}"
    )
    print("angle samples by FOV:")
    for fov, angle_samples in cfg.angle_samples_by_fov_deg.items():
        rendered = ", ".join(
            f"{sample['label']}=({float(sample['angle_x_deg']):g}, {float(sample['angle_y_deg']):g}) deg"
            for sample in angle_samples
        )
        print(f"  FOV {fov:g} deg -> {rendered}")

    print("estimated meta-atoms across aperture diameter:")
    for aperture_mm, count in cfg.meta_atom_counts.items():
        print(f"  {aperture_mm:g} mm / {cfg.metasurface_period_nm:g} nm -> about {count} x {count}")

    print(f"PSF sweep rows: {len(sweep_rows)}")
    print(f"PSF figures: {cfg.psf_figures_dir}")
    print(f"PSF cache: {cfg.psf_cache_dir}")


if __name__ == "__main__":
    main()
