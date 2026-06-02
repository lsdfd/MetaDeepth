from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import torch

from src.optics.psf import PsfResult, compute_point_source_psf
from src.optics.pupil import circular_aperture, exact_lens_phase, make_xy_grid, random_phase
from src.utils.system_config import FirstStageSystemConfig, load_first_stage_system_config
from src.utils.units import mm, nm


def _pixel_coordinates(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    y = torch.arange(height, device=image.device, dtype=image.dtype)
    x = torch.arange(width, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def _summarize_psf(psf: torch.Tensor) -> dict[str, float | int]:
    xx, yy = _pixel_coordinates(psf)
    total = psf.sum() + 1e-12
    centroid_x = (psf * xx).sum() / total
    centroid_y = (psf * yy).sum() / total
    flat_index = int(torch.argmax(psf).detach().cpu())
    peak_x = flat_index % psf.shape[-1]
    peak_y = flat_index // psf.shape[-1]
    radius_sq = ((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2) * psf
    return {
        "psf_sum": float(psf.sum().detach().cpu()),
        "psf_min": float(psf.min().detach().cpu()),
        "psf_max": float(psf.max().detach().cpu()),
        "centroid_x_px": float(centroid_x.detach().cpu()),
        "centroid_y_px": float(centroid_y.detach().cpu()),
        "peak_x_px": peak_x,
        "peak_y_px": peak_y,
        "second_moment_radius_px": float(torch.sqrt(radius_sq.sum() / total).detach().cpu()),
    }


def _mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).detach().cpu())


def _select_debug_values(cfg: FirstStageSystemConfig) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    fovs = (90.0,) if 90.0 in cfg.fov_scan_deg else (cfg.fov_scan_deg[-1],)
    depths = (cfg.depth_samples_m[0], cfg.depth_samples_m[-1])
    apertures = (3.0,) if 3.0 in cfg.aperture_scan_mm else (cfg.aperture_scan_mm[0],)
    distances = (10.0,) if 10.0 in cfg.metasurface_sensor_distance_scan_mm else (cfg.metasurface_sensor_distance_scan_mm[-1],)
    return fovs, depths, apertures, distances


def _phase_for_baseline(
    phase_name: str,
    grid: int,
    grid_size_m: float,
    wavelength_m: float,
    focal_length_m: float,
    device: str,
    seed: int,
) -> torch.Tensor:
    x, y, _ = make_xy_grid(grid, grid_size_m, device=device)
    if phase_name == "lens":
        return exact_lens_phase(x, y, wavelength_m, focal_length_m)
    if phase_name == "random":
        return random_phase((grid, grid), device=device, seed=seed)
    raise ValueError(f"unsupported phase baseline: {phase_name}")


def _max_phase_step_rad(phase: torch.Tensor, aperture_mask: torch.Tensor) -> float:
    mask_x = aperture_mask[:, 1:] * aperture_mask[:, :-1]
    mask_y = aperture_mask[1:, :] * aperture_mask[:-1, :]
    dx = torch.abs(phase[:, 1:] - phase[:, :-1])[mask_x > 0]
    dy = torch.abs(phase[1:, :] - phase[:-1, :])[mask_y > 0]
    values = [value for value in (dx, dy) if value.numel() > 0]
    if not values:
        return 0.0
    return float(torch.max(torch.cat(values)).detach().cpu())


def _angle_sort_key(row: dict[str, float | str]) -> tuple[float, float]:
    return (-float(row["angle_y_deg"]), float(row["angle_x_deg"]))


def _save_angle_grid(
    results: dict[tuple[str, float, float, float, float, float, float], PsfResult],
    cfg: FirstStageSystemConfig,
    phase_name: str,
    fov_deg: float,
    depth_m: float,
    aperture_mm: float,
    distance_mm: float,
    out_path: Path,
) -> None:
    angle_samples = sorted(cfg.angle_samples_by_fov_deg[fov_deg], key=_angle_sort_key)
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 8.2), squeeze=False)
    psfs = [
        results[(phase_name, fov_deg, float(sample["angle_x_deg"]), float(sample["angle_y_deg"]), depth_m, aperture_mm, distance_mm)].psf
        for sample in angle_samples
    ]
    vmax = max(float(psf.max().detach().cpu()) for psf in psfs)

    for ax, sample, psf in zip(axes.reshape(-1), angle_samples, psfs, strict=True):
        ax.imshow(psf.detach().cpu().numpy(), cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(
            f"ax={float(sample['angle_x_deg']):g}, ay={float(sample['angle_y_deg']):g}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{phase_name} PSF | FOV {fov_deg:g} deg | z={depth_m:g} m | "
        f"D={aperture_mm:g} mm | d={distance_mm:g} mm"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    result: PsfResult,
    cfg: FirstStageSystemConfig,
    phase_name: str,
    fov_deg: float,
    angle_label: str,
    angle_x_deg: float,
    angle_y_deg: float,
    aperture_mm: float,
    distance_mm: float,
    grid: int,
    phase_max_step_rad: float | str,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "phase": phase_name,
        "fov_deg": fov_deg,
        "fov_definition": cfg.fov_definition,
        "angle_label": angle_label,
        "angle_x_deg": angle_x_deg,
        "angle_y_deg": angle_y_deg,
        "depth_m": result.depth_m,
        "aperture_mm": aperture_mm,
        "metasurface_sensor_distance_mm": distance_mm,
        "wavelength_nm": cfg.wavelength_nm,
        "grid": grid,
        "grid_size_mm": result.grid_size_m / 1e-3,
        "dx_um": result.dx_m / 1e-6,
        "sensor_pixel_pitch_um": cfg.sensor.pixel_pitch_um,
        "psf_grid": cfg.psf_grid,
        "phase_max_step_rad": phase_max_step_rad,
        "full_psf_sum": result.full_psf_sum,
        "crop_energy_ratio": result.crop_energy_ratio,
    }
    row.update(_summarize_psf(result.psf))
    return row


def run(args: argparse.Namespace) -> None:
    cfg = load_first_stage_system_config(args.config)
    device = args.device
    grid = cfg.pupil_grid_main if args.full else cfg.pupil_grid_debug
    wavelength_m = nm(cfg.wavelength_nm)

    if args.full:
        fovs = cfg.fov_scan_deg
        depths = cfg.depth_samples_m
        apertures = cfg.aperture_scan_mm
        distances = cfg.metasurface_sensor_distance_scan_mm
    else:
        fovs, depths, apertures, distances = _select_debug_values(cfg)

    out_dir = ROOT / cfg.psf_figures_dir
    summary_rows: list[dict[str, float | int | str]] = []
    results: dict[tuple[str, float, float, float, float, float, float], PsfResult] = {}

    print("First-stage point-source PSF sanity check")
    print(f"config: {args.config}")
    print(f"mode: {'full' if args.full else 'debug'}")
    print(f"grid={grid}, crop={cfg.psf_grid}, padding={cfg.pupil_padding_factor:g}")
    print(f"FOVs={fovs}, depths={depths}, apertures={apertures}, distances={distances}")

    for aperture_mm in apertures:
        aperture_m = mm(aperture_mm)
        grid_size_m = aperture_m * cfg.pupil_padding_factor
        x, y, _ = make_xy_grid(grid, grid_size_m, device=device)
        aperture_mask = circular_aperture(x, y, aperture_m)
        for distance_mm in distances:
            distance_m = mm(distance_mm)
            phase_bank = {
                phase_name: _phase_for_baseline(
                    phase_name=phase_name,
                    grid=grid,
                    grid_size_m=grid_size_m,
                    wavelength_m=wavelength_m,
                    focal_length_m=distance_m,
                    device=device,
                    seed=cfg.random_seed,
                )
                for phase_name in cfg.phase_baselines
            }
            phase_steps = {
                phase_name: _max_phase_step_rad(phase, aperture_mask) if phase_name == "lens" else ""
                for phase_name, phase in phase_bank.items()
            }

            for phase_name, phase in phase_bank.items():
                for fov_deg in fovs:
                    for depth_m in depths:
                        for angle_sample in cfg.angle_samples_by_fov_deg[fov_deg]:
                            angle_x_deg = float(angle_sample["angle_x_deg"])
                            angle_y_deg = float(angle_sample["angle_y_deg"])
                            result = compute_point_source_psf(
                                phase=phase,
                                wavelength_m=wavelength_m,
                                aperture_diameter_m=aperture_m,
                                metasurface_sensor_distance_m=distance_m,
                                grid_size_m=grid_size_m,
                                depth_m=depth_m,
                                angle_x_rad=torch.deg2rad(torch.tensor(angle_x_deg)).item(),
                                angle_y_rad=torch.deg2rad(torch.tensor(angle_y_deg)).item(),
                                crop_size=cfg.psf_grid,
                                normalize=cfg.normalize_psf,
                            )
                            key = (phase_name, fov_deg, angle_x_deg, angle_y_deg, depth_m, aperture_mm, distance_mm)
                            results[key] = result
                            summary_rows.append(
                                _summary_row(
                                    result=result,
                                    cfg=cfg,
                                    phase_name=phase_name,
                                    fov_deg=fov_deg,
                                    angle_label=str(angle_sample["label"]),
                                    angle_x_deg=angle_x_deg,
                                    angle_y_deg=angle_y_deg,
                                    aperture_mm=aperture_mm,
                                    distance_mm=distance_mm,
                                    grid=grid,
                                    phase_max_step_rad=phase_steps[phase_name],
                                )
                            )

                        figure_path = (
                            out_dir
                            / "single_case"
                            / f"{phase_name}_fov{fov_deg:g}_z{depth_m:g}m_D{aperture_mm:g}mm_d{distance_mm:g}mm.png"
                        )
                        _save_angle_grid(
                            results=results,
                            cfg=cfg,
                            phase_name=phase_name,
                            fov_deg=fov_deg,
                            depth_m=depth_m,
                            aperture_mm=aperture_mm,
                            distance_mm=distance_mm,
                            out_path=figure_path,
                        )

    center_key = next(
        row for row in summary_rows if row["angle_label"] == "center" and row["phase"] == cfg.phase_baselines[0]
    )
    print(
        "example center PSF: "
        f"phase={center_key['phase']}, z={center_key['depth_m']} m, "
        f"sum={center_key['psf_sum']:.6f}, crop_energy={center_key['crop_energy_ratio']:.3e}"
    )

    if len(depths) >= 2:
        for phase_name in cfg.phase_baselines:
            fov_deg = fovs[0]
            aperture_mm = apertures[0]
            distance_mm = distances[0]
            depth_a, depth_b = depths[0], depths[-1]
            key_a = (phase_name, fov_deg, 0.0, 0.0, depth_a, aperture_mm, distance_mm)
            key_b = (phase_name, fov_deg, 0.0, 0.0, depth_b, aperture_mm, distance_mm)
            if key_a in results and key_b in results:
                delta = _mean_abs_delta(results[key_a].psf, results[key_b].psf)
                print(f"{phase_name} center depth-change mean_abs_delta={delta:.3e}")

    summary_path = out_dir / ("psf_sanity_full_summary.csv" if args.full else "psf_sanity_debug_summary.csv")
    _write_summary(summary_rows, summary_path)
    print(f"Saved figures to {out_dir}")
    print(f"Saved summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-stage point-source PSF sanity check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "system_first_stage.yaml",
        help="Path to first-stage system config.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full", action="store_true", help="Run the full configured discrete PSF sweep.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
