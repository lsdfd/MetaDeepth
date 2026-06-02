from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.depth_data import build_depth_dataset
from src.models.depth_metrics import average_metric_rows, depth_metrics
from src.models.simple_unet import build_depth_model
from src.utils.config import load_config


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    input_mode = str(cfg["model"].get("input_mode", "rgb_meta"))
    device = str(cfg["train"].get("device", "cpu"))

    dataset = build_depth_dataset(cfg, split="val", input_mode=input_mode)
    loader = DataLoader(dataset, batch_size=int(cfg["data"].get("batch_size", 2)), shuffle=False)
    model = build_depth_model(
        input_mode=input_mode,
        min_depth_m=float(cfg["model"].get("min_depth_m", 1.0)),
        max_depth_m=float(cfg["model"].get("max_depth_m", 10.0)),
        width=int(cfg["model"].get("width", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device)
            depth_m = batch["depth_m"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            pred_m = model(image)
            row = {"batch": batch_index, **depth_metrics(pred_m, depth_m, valid_mask)}
            rows.append(row)

    summary = average_metric_rows([{k: v for k, v in row.items() if k != "batch"} for row in rows])
    out_path = args.output or (Path(checkpoint["config"]["outputs"]["run_dir"]) / "eval_metrics.csv")
    out_path = ROOT / out_path if not out_path.is_absolute() else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"mae={summary['mae']:.4f} rmse={summary['rmse']:.4f} "
        f"abs_rel={summary['abs_rel']:.4f} delta1={summary['delta1']:.4f}"
    )
    print(f"saved metrics to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the depth reconstruction scaffold.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "depth_learning.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "depth_learning" / "smoke" / "depth_model.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
