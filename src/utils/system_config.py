from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .config import load_config


@dataclass(frozen=True)
class SensorSpec:
    width: int
    height: int
    pixel_pitch_um: float


@dataclass(frozen=True)
class FirstStageSystemConfig:
    wavelength_nm: float
    bandwidth_nm: float
    metasurface_period_nm: float
    aperture_scan_mm: tuple[float, ...]
    metasurface_sensor_distance_scan_mm: tuple[float, ...]
    fov_scan_deg: tuple[float, ...]
    fov_definition: str
    angle_sample_policy: str
    depth_range_m: tuple[float, float]
    depth_samples_m: tuple[float, ...]
    pupil_grid_main: int
    pupil_grid_debug: int
    pupil_padding_factor: float
    psf_grid: int
    normalize_psf: bool
    random_seed: int
    phase_baselines: tuple[str, ...]
    sensor: SensorSpec
    psf_figures_dir: Path
    psf_cache_dir: Path

    @property
    def angle_samples_by_fov_deg(self) -> dict[float, tuple[dict[str, float | str], ...]]:
        if self.angle_sample_policy != "center_cross_corners":
            raise ValueError(f"unsupported angle_sample_policy: {self.angle_sample_policy}")

        samples: dict[float, tuple[dict[str, float | str], ...]] = {}
        for fov in self.fov_scan_deg:
            half_angle = fov / 2.0
            samples[fov] = (
                {"label": "center", "angle_x_deg": 0.0, "angle_y_deg": 0.0},
                {"label": "x_neg_edge", "angle_x_deg": -half_angle, "angle_y_deg": 0.0},
                {"label": "x_pos_edge", "angle_x_deg": half_angle, "angle_y_deg": 0.0},
                {"label": "y_neg_edge", "angle_x_deg": 0.0, "angle_y_deg": -half_angle},
                {"label": "y_pos_edge", "angle_x_deg": 0.0, "angle_y_deg": half_angle},
                {"label": "corner_neg_neg", "angle_x_deg": -half_angle, "angle_y_deg": -half_angle},
                {"label": "corner_neg_pos", "angle_x_deg": -half_angle, "angle_y_deg": half_angle},
                {"label": "corner_pos_neg", "angle_x_deg": half_angle, "angle_y_deg": -half_angle},
                {"label": "corner_pos_pos", "angle_x_deg": half_angle, "angle_y_deg": half_angle},
            )
        return samples

    @property
    def meta_atom_counts(self) -> dict[float, int]:
        return {
            aperture_mm: round(aperture_mm * 1e6 / self.metasurface_period_nm)
            for aperture_mm in self.aperture_scan_mm
        }

    def iter_psf_sweep(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for phase, fov, aperture_mm, sensor_distance_mm, depth_m in product(
            self.phase_baselines,
            self.fov_scan_deg,
            self.aperture_scan_mm,
            self.metasurface_sensor_distance_scan_mm,
            self.depth_samples_m,
        ):
            for angle_sample in self.angle_samples_by_fov_deg[fov]:
                rows.append(
                    {
                        "phase": phase,
                        "fov_deg": fov,
                        "fov_definition": self.fov_definition,
                        "angle_label": angle_sample["label"],
                        "angle_x_deg": angle_sample["angle_x_deg"],
                        "angle_y_deg": angle_sample["angle_y_deg"],
                        "aperture_mm": aperture_mm,
                        "metasurface_sensor_distance_mm": sensor_distance_mm,
                        "depth_m": depth_m,
                    }
                )
        return rows


def _as_float_tuple(values: list[Any] | tuple[Any, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def load_first_stage_system_config(path: str | Path) -> FirstStageSystemConfig:
    cfg = load_config(path)
    optics = cfg["optics"]
    scene = cfg["scene"]
    sensor = cfg["sensor"]
    sampling = cfg["sampling"]
    outputs = cfg["outputs"]

    depth_range = _as_float_tuple(scene["depth_range_m"])
    if len(depth_range) != 2:
        raise ValueError("scene.depth_range_m must contain [min_m, max_m]")

    depth_samples = _as_float_tuple(scene["depth_samples_m"])
    if min(depth_samples) < depth_range[0] or max(depth_samples) > depth_range[1]:
        raise ValueError("scene.depth_samples_m must stay inside scene.depth_range_m")

    resolution = sensor["resolution"]
    return FirstStageSystemConfig(
        wavelength_nm=float(optics["wavelength_nm"]),
        bandwidth_nm=float(optics["bandwidth_nm"]),
        metasurface_period_nm=float(optics["metasurface_period_nm"]),
        aperture_scan_mm=_as_float_tuple(optics["aperture_scan_mm"]),
        metasurface_sensor_distance_scan_mm=_as_float_tuple(
            optics["metasurface_sensor_distance_scan_mm"]
        ),
        fov_scan_deg=_as_float_tuple(scene["fov_scan_deg"]),
        fov_definition=str(scene["fov_definition"]),
        angle_sample_policy=str(scene["angle_sample_policy"]),
        depth_range_m=(depth_range[0], depth_range[1]),
        depth_samples_m=depth_samples,
        pupil_grid_main=int(sampling["pupil_grid_main"]),
        pupil_grid_debug=int(sampling["pupil_grid_debug"]),
        pupil_padding_factor=float(sampling["pupil_padding_factor"]),
        psf_grid=int(sampling["psf_grid"]),
        normalize_psf=bool(sampling["normalize_psf"]),
        random_seed=int(sampling["random_seed"]),
        phase_baselines=tuple(str(value) for value in optics["phase_baselines"]),
        sensor=SensorSpec(
            width=int(resolution["width"]),
            height=int(resolution["height"]),
            pixel_pitch_um=float(sensor["pixel_pitch_um"]),
        ),
        psf_figures_dir=Path(outputs["psf_figures_dir"]),
        psf_cache_dir=Path(outputs["psf_cache_dir"]),
    )
