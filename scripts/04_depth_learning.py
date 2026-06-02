from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import torch
from torch.utils.data import DataLoader

from src.models.depth_data import build_depth_dataset
from src.models.depth_losses import depth_loss
from src.models.depth_metrics import average_metric_rows, depth_metrics
from src.models.simple_unet import build_depth_model
from src.utils.config import load_config


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_loader(cfg: dict, split: str, input_mode: str) -> DataLoader:
    dataset = build_depth_dataset(cfg, split=split, input_mode=input_mode)
    return DataLoader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 2)),
        shuffle=(split == "train"),
    )


def _make_model(cfg: dict, input_mode: str, device: str) -> torch.nn.Module:
    return build_depth_model(
        input_mode=input_mode,
        min_depth_m=float(cfg["model"].get("min_depth_m", 1.0)),
        max_depth_m=float(cfg["model"].get("max_depth_m", 10.0)),
        width=int(cfg["model"].get("width", 32)),
    ).to(device)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            depth_m = batch["depth_m"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            pred_m = model(image)
            rows.append(depth_metrics(pred_m, depth_m, valid_mask))
    return average_metric_rows(rows)


def train(cfg: dict) -> None:
    torch.manual_seed(int(cfg["train"].get("seed", 0)))
    input_mode = str(cfg["model"].get("input_mode", "rgb_meta"))
    device = str(cfg["train"].get("device", "cpu"))
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = _make_loader(cfg, split="train", input_mode=input_mode)
    val_loader = _make_loader(cfg, split="val", input_mode=input_mode)
    model = _make_model(cfg, input_mode, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("learning_rate", 1e-3)))

    rows: list[dict[str, float | int | str]] = []
    for epoch in range(int(cfg["train"].get("epochs", 1))):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            depth_m = batch["depth_m"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            pred_m = model(image)
            loss = depth_loss(pred_m, depth_m, valid_mask, loss_name=str(cfg["train"].get("loss", "l1_silog")))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **metrics}
        rows.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"abs_rel={row['abs_rel']:.4f} rmse={row['rmse']:.4f} delta1={row['delta1']:.4f}"
        )

    torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "depth_model.pt")
    _write_csv(rows, run_dir / "train_metrics.csv")
    print(f"saved checkpoint to {run_dir / 'depth_model.pt'}")
    print(f"saved metrics to {run_dir / 'train_metrics.csv'}")


def evaluate(cfg: dict, checkpoint_path: Path, output_path: Path | None) -> None:
    input_mode = str(cfg["model"].get("input_mode", "rgb_meta"))
    device = str(cfg["train"].get("device", "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    loader = _make_loader(cfg, split="val", input_mode=input_mode)
    model = _make_model(cfg, input_mode, device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate_model(model, loader, device)

    out_path = output_path or (ROOT / cfg["outputs"]["run_dir"] / "eval_metrics.csv")
    out_path = ROOT / out_path if not out_path.is_absolute() else out_path
    _write_csv([metrics], out_path)
    print(
        f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
        f"abs_rel={metrics['abs_rel']:.4f} delta1={metrics['delta1']:.4f}"
    )
    print(f"saved metrics to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth reconstruction scaffold train/eval entry.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "depth_learning.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "depth_learning" / "smoke" / "depth_model.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
