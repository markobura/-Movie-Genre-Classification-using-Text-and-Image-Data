#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poster.config import BACKBONES, GENRES, ROOT as PROJECT_ROOT
from poster.dataset import PosterDataset
from poster.models import build_model
from poster.transforms import get_transforms


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pos_weight(labels_df: pd.DataFrame) -> torch.Tensor:
    n = len(labels_df)
    counts = labels_df[GENRES].sum().values.astype(np.float32)
    weights = (n - counts) / np.maximum(counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, device, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels, _ in tqdm(loader, leave=False, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vgg16_stanford", choices=BACKBONES)
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "train_data")
    parser.add_argument("--val-dir", type=Path, default=PROJECT_ROOT / "val_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
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

    results_dir = PROJECT_ROOT / "results" / "poster"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_labels = pd.read_csv(args.train_dir / "train_labels.csv")
    pos_weight = compute_pos_weight(train_labels)

    weights_path = results_dir / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(
            {g: float(w) for g, w in zip(GENRES, pos_weight.tolist())},
            f,
            indent=2,
        )

    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")
    print(f"model on {next(model.parameters()).device}")
    print(f"class weights saved to {weights_path}")
    for genre in GENRES[:3]:
        print(f"  {genre}: pos_weight={pos_weight[GENRES.index(genre)].item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.0,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, device, optimizer
        )
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f}")


if __name__ == "__main__":
    main()
