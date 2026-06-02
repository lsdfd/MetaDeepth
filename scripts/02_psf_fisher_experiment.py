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

from src.fisher.psf_fisher import (
    crlb_from_information,
    effective_depth_fisher,
    fisher_for_theta,
    phase_from_control,
    psf_similarity,
)
from src.optics.psf import compute_point_source_psf
from src.optics.pupil import (
    double_helix_phase,
    exact_lens_phase,
    make_xy_grid,
    multiring_spiral_phase,
    random_phase,
    spiral_phase,
)
from src.utils.config import load_config
from src.utils.system_config import load_first_stage_system_config
from src.utils.units import mm, nm


def _deg(value: float) -> float:
    return float(torch.deg2rad(torch.tensor(value)))


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_phase(
    name: str,
    grid: int,
    grid_size_m: float,
    wavelength_m: float,
    distance_m: float,
    aperture_m: float,
) -> torch.Tensor:
    if name == "lens":
        x, y, _ = make_xy_grid(grid, grid_size_m)
        return exact_lens_phase(x, y, wavelength_m, distance_m)
    if name == "random":
        return random_phase((grid, grid), seed=0)
    x, y, _ = make_xy_grid(grid, grid_size_m)
    if name == "spiral":
        return spiral_phase(x, y, charge=1)
    if name == "double_helix":
        return double_helix_phase(x, y, charge=1)
    if name == "multiring_spiral":
        return multiring_spiral_phase(x, y, aperture_diameter_m=aperture_m)
    raise ValueError(f"unsupported phase: {name}")


def _small_cases(cfg) -> list[dict[str, float | str]]:
    fov = 90.0 if 90.0 in cfg.fov_scan_deg else cfg.fov_scan_deg[-1]
    depths = (1.0, 3.0, 5.0, 7.0, 10.0)
    angle_samples = cfg.angle_samples_by_fov_deg[fov]
    return [
        {
            "fov_deg": fov,
            "angle_label": str(sample["label"]),
            "angle_x_deg": float(sample["angle_x_deg"]),
            "angle_y_deg": float(sample["angle_y_deg"]),
            "depth_m": depth,
        }
        for depth in depths
        for sample in angle_samples
    ]


def _fisher_row(case, phase_name: str, phase: torch.Tensor, cfg, fisher_cfg, grid_size_m: float, aperture_m: float, distance_m: float) -> dict[str, float | str]:
    theta = torch.tensor(
        [
            float(case["depth_m"]),
            _deg(float(case["angle_x_deg"])),
            _deg(float(case["angle_y_deg"])),
            float(fisher_cfg["photon_count"]),
        ],
        dtype=torch.float32,
    )
    fisher, mu = fisher_for_theta(
        theta=theta,
        phase=phase,
        wavelength_m=nm(cfg.wavelength_nm),
        aperture_diameter_m=aperture_m,
        metasurface_sensor_distance_m=distance_m,
        grid_size_m=grid_size_m,
        crop_size=cfg.psf_grid,
        background_photons=float(fisher_cfg["background_photons"]),
        eps=float(fisher_cfg["eps"]),
    )
    i_raw = fisher[0, 0]
    i_eff = effective_depth_fisher(fisher, eps=float(fisher_cfg["eps"]))
    crlb_raw = crlb_from_information(i_raw, eps=float(fisher_cfg["eps"]))
    crlb_eff = crlb_from_information(i_eff, eps=float(fisher_cfg["eps"]))
    return {
        "phase": phase_name,
        "fov_deg": float(case["fov_deg"]),
        "angle_label": str(case["angle_label"]),
        "angle_x_deg": float(case["angle_x_deg"]),
        "angle_y_deg": float(case["angle_y_deg"]),
        "depth_m": float(case["depth_m"]),
        "I_z_raw": float(i_raw.detach().cpu()),
        "I_z_eff": float(i_eff.detach().cpu()),
        "CRLB_z_raw": float(crlb_raw.detach().cpu()),
        "CRLB_z_eff": float(crlb_eff.detach().cpu()),
        "F_z_ax": float(fisher[0, 1].detach().cpu()),
        "F_z_ay": float(fisher[0, 2].detach().cpu()),
        "F_z_a": float(fisher[0, 3].detach().cpu()),
        "mu_sum": float(mu.sum().detach().cpu()),
    }


def run_baseline(args, cfg, fisher_cfg) -> None:
    grid = cfg.pupil_grid_debug
    aperture_m = mm(3.0)
    distance_m = mm(10.0)
    grid_size_m = aperture_m * cfg.pupil_padding_factor
    cases = _small_cases(cfg)
    out_dir = ROOT / "outputs" / "fisher" / "first_stage"

    rows = []
    psfs: dict[tuple[str, str, float], torch.Tensor] = {}
    for phase_name in cfg.phase_baselines:
        phase = _make_phase(phase_name, grid, grid_size_m, nm(cfg.wavelength_nm), distance_m, aperture_m)
        for case in cases:
            rows.append(_fisher_row(case, phase_name, phase, cfg, fisher_cfg, grid_size_m, aperture_m, distance_m))
            result = compute_point_source_psf(
                phase=phase,
                wavelength_m=nm(cfg.wavelength_nm),
                aperture_diameter_m=aperture_m,
                metasurface_sensor_distance_m=distance_m,
                grid_size_m=grid_size_m,
                depth_m=float(case["depth_m"]),
                angle_x_rad=_deg(float(case["angle_x_deg"])),
                angle_y_rad=_deg(float(case["angle_y_deg"])),
                crop_size=cfg.psf_grid,
                normalize=True,
            )
            psfs[(phase_name, str(case["angle_label"]), float(case["depth_m"]))] = result.psf

    sim_rows = []
    for phase_name in cfg.phase_baselines:
        for angle_label in sorted({str(case["angle_label"]) for case in cases}):
            depths = sorted({float(case["depth_m"]) for case in cases})
            for left, right in zip(depths[:-1], depths[1:], strict=True):
                metrics = psf_similarity(psfs[(phase_name, angle_label, left)], psfs[(phase_name, angle_label, right)])
                sim_rows.append(
                    {
                        "phase": phase_name,
                        "angle_label": angle_label,
                        "depth_left_m": left,
                        "depth_right_m": right,
                        **metrics,
                    }
                )

    _write_csv(rows, out_dir / "baseline_fisher_summary.csv")
    _write_csv(sim_rows, out_dir / "baseline_psf_similarity_depth.csv")
    print(f"saved {out_dir / 'baseline_fisher_summary.csv'}")
    print(f"saved {out_dir / 'baseline_psf_similarity_depth.csv'}")


def run_optimize(args, cfg, fisher_cfg) -> None:
    grid = 256
    aperture_m = mm(3.0)
    distance_m = mm(10.0)
    grid_size_m = aperture_m * cfg.pupil_padding_factor
    cases = [case for case in _small_cases(cfg) if str(case["angle_label"]) in {"center", "x_pos_edge", "y_pos_edge", "corner_pos_pos"}]
    control_size = int(fisher_cfg["phase_control_grid"])
    control = torch.zeros((control_size, control_size), requires_grad=True)
    opt = torch.optim.Adam([control], lr=float(fisher_cfg["learning_rate"]))
    out_dir = ROOT / "outputs" / "fisher" / "first_stage"
    history = []

    for step in range(int(fisher_cfg["optimize_steps"])):
        opt.zero_grad()
        phase = phase_from_control(control, grid)
        infos = []
        for case in cases:
            theta = torch.tensor(
                [float(case["depth_m"]), _deg(float(case["angle_x_deg"])), _deg(float(case["angle_y_deg"])), float(fisher_cfg["photon_count"])],
                dtype=torch.float32,
            )
            fisher, _ = fisher_for_theta(
                theta=theta,
                phase=phase,
                wavelength_m=nm(cfg.wavelength_nm),
                aperture_diameter_m=aperture_m,
                metasurface_sensor_distance_m=distance_m,
                grid_size_m=grid_size_m,
                crop_size=cfg.psf_grid,
                background_photons=float(fisher_cfg["background_photons"]),
                eps=float(fisher_cfg["eps"]),
            )
            infos.append(effective_depth_fisher(fisher, eps=float(fisher_cfg["eps"])))
        log_infos = torch.log(torch.stack(infos) + float(fisher_cfg["eps"]))
        robust = -torch.logsumexp(-log_infos, dim=0)
        loss = -log_infos.mean() - float(fisher_cfg["robust_weight"]) * robust
        loss.backward()
        opt.step()
        history.append({"step": step, "loss": float(loss.detach().cpu()), "mean_log_I": float(log_infos.mean().detach().cpu())})
        print(f"step={step} loss={history[-1]['loss']:.4e} mean_log_I={history[-1]['mean_log_I']:.4e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    phase = phase_from_control(control.detach(), grid)
    torch.save({"control": control.detach(), "phase": phase}, out_dir / "optimized_phase.pt")
    _write_csv(history, out_dir / "optimized_phase_history.csv")
    plt.figure(figsize=(5, 4))
    plt.imshow(phase.detach().cpu().numpy(), cmap="twilight")
    plt.colorbar(label="phase rad")
    plt.tight_layout()
    plt.savefig(out_dir / "optimized_phase.png", dpi=160)
    plt.close()
    print(f"saved optimized phase to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PSF Fisher baseline/optimization experiment.")
    parser.add_argument("--mode", choices=("baseline", "optimize", "report"), default="baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "system_first_stage.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_first_stage_system_config(args.config)
    fisher_cfg = load_config(args.config)["fisher"]
    if args.mode in {"baseline", "report"}:
        run_baseline(args, cfg, fisher_cfg)
    elif args.mode == "optimize":
        run_optimize(args, cfg, fisher_cfg)


if __name__ == "__main__":
    main()
