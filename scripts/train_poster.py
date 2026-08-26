#!/usr/bin/env python3
import argparse
import json
import random
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
from poster.metrics import compute_ap_metrics
from poster.models import build_model
from poster.transforms import get_transforms


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, epoch: int, backbone: str, val_macro_ap: float):
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone,
            "val_macro_ap": val_macro_ap,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_batches = []
    y_score_batches = []

    for images, labels, _ in tqdm(loader, leave=False, desc="val"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        y_true_batches.append(labels.cpu().numpy())
        y_score_batches.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_score = np.concatenate(y_score_batches, axis=0)
    val_loss = total_loss / len(loader.dataset)
    metrics = compute_ap_metrics(y_true, y_score)
    return val_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vgg16_stanford", choices=BACKBONES)
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "train_data")
    parser.add_argument("--val-dir", type=Path, default=PROJECT_ROOT / "val_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint dir (default: models/poster/<backbone>)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir or PROJECT_ROOT / "models" / "poster" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"

    device = resolve_device(args.device)
    transform = get_transforms(args.backbone, train=True)

    train_ds = PosterDataset(args.train_dir, transform=transform)
    val_ds = PosterDataset(
        args.val_dir, transform=get_transforms(args.backbone, train=False)
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        generator=train_generator,
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
    print(f"output_dir={output_dir}")
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

    best_macro_ap = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        val_macro_ap = metrics["macro_ap"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_ap": val_macro_ap,
            }
        )

        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_ap={val_macro_ap:.4f}"
        )

        if val_macro_ap > best_macro_ap:
            best_macro_ap = val_macro_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, epoch, args.backbone, val_macro_ap
            )
            print(f"  saved best checkpoint (val_macro_ap={val_macro_ap:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping at epoch {epoch} "
                f"(best val_macro_ap={best_macro_ap:.4f} at epoch {best_epoch})"
            )
            break

    with open(history_path, "w") as f:
        json.dump(
            {
                "backbone": args.backbone,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_macro_ap": best_macro_ap,
                "epochs": history,
            },
            f,
            indent=2,
        )
    print(f"history saved to {history_path}")
    print(f"best checkpoint: {checkpoint_path} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
