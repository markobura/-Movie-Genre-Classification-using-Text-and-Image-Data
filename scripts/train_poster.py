#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poster.config import BACKBONES, ROOT as PROJECT_ROOT
from poster.dataset import PosterDataset
from poster.models import build_model
from poster.transforms import get_transforms


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vgg16_stanford", choices=BACKBONES)
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "train_data")
    parser.add_argument("--val-dir", type=Path, default=PROJECT_ROOT / "val_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    transform = get_transforms(args.backbone, train=True)

    train_ds = PosterDataset(args.train_dir, transform=transform)
    val_ds = PosterDataset(
        args.val_dir, transform=get_transforms(args.backbone, train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )

    model = build_model(args.backbone).to(device)

    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")
    print(f"model on {next(model.parameters()).device}")


if __name__ == "__main__":
    main()
