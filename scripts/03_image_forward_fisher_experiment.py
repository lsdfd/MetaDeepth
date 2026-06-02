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
from skimage.metrics import structural_similarity

from src.optics.pupil import exact_lens_phase, make_xy_grid, random_phase
from src.simulation.rgbd_forward import (
    crop_center,
    green_channel_intensity,
    load_hypersim_pilot_sample,
    patch_depth_fisher_scalar,
    poisson_mahalanobis,
    ray_angle_maps,
    render_spatially_varying_psf,
)
from src.utils.system_config import load_first_stage_system_config
from src.utils.units import mm, nm


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _js(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    pa = a.reshape(-1)
    pb = b.reshape(-1)
    pa = pa / (pa.sum() + eps)
    pb = pb / (pb.sum() + eps)
    mix = 0.5 * (pa + pb)
    return float(
        0.5
        * (
            (pa * (torch.log(pa + eps) - torch.log(mix + eps))).sum()
            + (pb * (torch.log(pb + eps) - torch.log(mix + eps))).sum()
        )
    )


def _image_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    va = a.reshape(-1)
    vb = b.reshape(-1)
    cosine = 1.0 - torch.dot(va, vb) / (
        (torch.linalg.vector_norm(va) + 1e-12) * (torch.linalg.vector_norm(vb) + 1e-12)
    )
    min_side = min(a.shape[-2:])
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    ssim_value = float("nan")
    if win_size >= 3:
        ssim_value = float(
            structural_similarity(
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                data_range=float((torch.max(torch.stack([a, b])) - torch.min(torch.stack([a, b]))).detach().cpu()) + 1e-12,
                win_size=win_size,
            )
        )
    return {
        "l1": float(torch.mean(torch.abs(a - b))),
        "cosine_distance": float(cosine),
        "js_divergence": _js(a, b),
        "poisson_mahalanobis": float(poisson_mahalanobis(a, b)),
        "ssim": ssim_value,
        "freq_l1": float(torch.mean(torch.abs(torch.abs(torch.fft.fft2(a)) - torch.abs(torch.fft.fft2(b))))),
    }


def _tiny_probe(images: torch.Tensor, labels: torch.Tensor, steps: int = 50) -> float:
    model = torch.nn.Linear(images[0].numel(), 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    x = images.reshape(images.shape[0], -1)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        opt.step()
    pred = model(x).argmax(dim=1)
    return float((pred == labels).to(torch.float32).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatially varying RGB-D forward and image-level Fisher smoke test.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "system_first_stage.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "hypersim_pilot_100")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--psf-size", type=int, default=31)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--phase", choices=("lens", "random"), default="random")
    parser.add_argument("--delta-depth", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_first_stage_system_config(args.config)
    rgb, depth, item = load_hypersim_pilot_sample(args.data_root, index=args.sample_index)
    intensity = crop_center(green_channel_intensity(rgb), args.patch_size)
    depth_patch = crop_center(depth, args.patch_size)
    ax, ay = ray_angle_maps(depth.shape[0], depth.shape[1], fov_x_deg=60.0)
    ax_patch = crop_center(ax, args.patch_size)
    ay_patch = crop_center(ay, args.patch_size)

    aperture_m = mm(3.0)
    distance_m = mm(10.0)
    grid_size_m = aperture_m * cfg.pupil_padding_factor
    x, y, _ = make_xy_grid(args.grid, grid_size_m)
    phase = (
        exact_lens_phase(x, y, nm(cfg.wavelength_nm), distance_m)
        if args.phase == "lens"
        else random_phase((args.grid, args.grid), seed=cfg.random_seed)
    )

    common = dict(
        intensity=intensity,
        angle_x=ax_patch,
        angle_y=ay_patch,
        phase=phase,
        wavelength_m=nm(cfg.wavelength_nm),
        aperture_diameter_m=aperture_m,
        metasurface_sensor_distance_m=distance_m,
        grid_size_m=grid_size_m,
        psf_size=args.psf_size,
    )
    encoded = render_spatially_varying_psf(depth=depth_patch, **common)
    encoded_plus = render_spatially_varying_psf(depth=depth_patch + args.delta_depth, **common)
    encoded_minus = render_spatially_varying_psf(depth=torch.clamp(depth_patch - args.delta_depth, min=0.1), **common)
    fisher = patch_depth_fisher_scalar(depth=depth_patch, **common)

    rows = [
        {
            "scene": item["scene"],
            "frame_id": item["frame_id"],
            "phase": args.phase,
            "patch_size": args.patch_size,
            "psf_size": args.psf_size,
            "depth_delta_m": args.delta_depth,
            "patch_depth_min_m": float(depth_patch.min()),
            "patch_depth_max_m": float(depth_patch.max()),
            "patch_depth_fisher_scalar": float(fisher.detach().cpu()),
            **{f"plus_{k}": v for k, v in _image_metrics(encoded, encoded_plus).items()},
            **{f"minus_{k}": v for k, v in _image_metrics(encoded, encoded_minus).items()},
        }
    ]

    probe_images = torch.stack([encoded_plus, encoded_minus, encoded_plus * 0.98, encoded_minus * 1.02])
    probe_labels = torch.tensor([1, 0, 1, 0])
    rows[0]["tiny_probe_train_accuracy"] = _tiny_probe(probe_images, probe_labels)

    out_dir = ROOT / "outputs" / "image_forward" / "first_stage"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "image_forward_fisher_summary.csv")

    fig, axes = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    for axis, image, title in zip(
        axes[0],
        (intensity, encoded, encoded_plus, torch.abs(encoded_plus - encoded)),
        ("linear G", "encoded", "+depth", "|diff|"),
        strict=True,
    ):
        axis.imshow(image.detach().cpu().numpy(), cmap="magma")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "image_forward_smoke.png", dpi=160)
    plt.close(fig)
    print(f"saved {out_dir / 'image_forward_fisher_summary.csv'}")
    print(f"saved {out_dir / 'image_forward_smoke.png'}")


if __name__ == "__main__":
    main()
