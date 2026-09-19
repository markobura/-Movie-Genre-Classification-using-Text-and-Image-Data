#!/usr/bin/env python3
"""Train CLIP + MiniLM + fusion head on train_data.

    python scripts/train_joint_fusion.py --device cpu
    python scripts/train_joint_fusion.py --unfreeze-vision 0 --unfreeze-text 0
"""

import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fusion.joint_data import MultimodalMovieDataset, collate_multimodal
from fusion.joint_model import JointFusionModel, TEXT_MODEL_ID
from fusion.metrics import metrics_payload
from poster.config import GENRES, ROOT as PROJECT_ROOT
from poster.inference import resolve_device
from poster.metrics import compute_ap_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(labels_df: pd.DataFrame) -> torch.Tensor:
    n = len(labels_df)
    counts = labels_df[GENRES].sum().values.astype(np.float32)
    return torch.tensor((n - counts) / np.maximum(counts, 1.0), dtype=torch.float32)


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        # picklable for DataLoader workers on the spawn start method
        collate_fn=functools.partial(
            collate_multimodal, pad_token_id=dataset.pad_token_id
        ),
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup, then linear decay."""
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm,
                    scheduler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False, desc="train"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_split(model, loader, device, desc="eval"):
    model.eval()
    y_true, y_score, movie_ids = [], [], []
    for batch in tqdm(loader, leave=False, desc=desc):
        logits = model(
            batch["pixel_values"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        y_score.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(batch["labels"].numpy())
        movie_ids.append(batch["movie_id"].numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_score),
        np.concatenate(movie_ids),
    )


def save_predictions(path: Path, movie_ids, scores) -> None:
    df = pd.DataFrame(scores, columns=GENRES)
    df.insert(0, "movie_id", movie_ids)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "train_data")
    parser.add_argument("--val-dir", type=Path, default=PROJECT_ROOT / "val_data")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "test_data")
    parser.add_argument("--unfreeze-vision", type=int, default=1,
                        help="How many of the 12 CLIP blocks train")
    parser.add_argument("--unfreeze-text", type=int, default=2,
                        help="How many of the 6 MiniLM blocks train")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=256,
                        help="Plot tokens fed to MiniLM")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "fusion" / "joint")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "fusion")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)

    train_ds = MultimodalMovieDataset(
        args.train_dir, tokenizer, max_length=args.max_length, train=True
    )
    val_ds = MultimodalMovieDataset(args.val_dir, tokenizer, max_length=args.max_length)
    test_ds = MultimodalMovieDataset(args.test_dir, tokenizer, max_length=args.max_length)

    train_loader = make_loader(train_ds, args.batch_size, True, args.seed, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.seed, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.seed, args.num_workers)

    model = JointFusionModel(
        unfreeze_vision=args.unfreeze_vision,
        unfreeze_text=args.unfreeze_text,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    counts = model.count_parameters()
    print(f"device={device}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(
        f"parameters: total={counts['total']:,} trainable={counts['trainable']:,} "
        f"(head {counts['head']:,})"
    )

    pos_weight = compute_pos_weight(pd.read_csv(args.train_dir / "train_labels.csv"))
    started = time.time()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr, args.weight_decay)
    )
    scheduler = build_scheduler(
        optimizer, args.epochs * len(train_loader), args.warmup_ratio
    )

    best_macro_ap = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.max_grad_norm, scheduler,
        )
        y_val_np, val_scores, val_ids = predict_split(model, val_loader, device, "val")
        val_metrics = compute_ap_metrics(y_val_np, val_scores)
        val_macro_ap = val_metrics["macro_ap"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_ap": val_macro_ap,
            "val_micro_ap": val_metrics["micro_ap"],
            "seconds": round(time.time() - epoch_start, 1),
        })
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} "
            f"val_macro_ap={val_macro_ap:.4f} ({time.time() - epoch_start:.0f}s)"
        )

        if val_macro_ap > best_macro_ap:
            best_macro_ap = val_macro_ap
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.trainable_state_dict().items()}
            epochs_without_improvement = 0
            print(f"  new best (val_macro_ap={val_macro_ap:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state, strict=False)
    train_info = {
        "trained_on": "train",
        "selected_on": "val",
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_macro_ap,
        "history": history,
    }
    y_val_np, val_scores, val_ids = predict_split(model, val_loader, device, "val")
    y_test_np, test_scores, test_ids = predict_split(model, test_loader, device, "test")

    elapsed = time.time() - started

    save_predictions(args.output_dir / "predictions_val.csv", val_ids, val_scores)
    save_predictions(args.output_dir / "predictions_test.csv", test_ids, test_scores)

    payload = {
        "name": "Joint fusion (CLIP + MiniLM)",
        "config": {
            "unfreeze_vision": args.unfreeze_vision,
            "unfreeze_text": args.unfreeze_text,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "device": str(device),
            "train_size": len(train_ds),
        },
        "parameters": counts,
        "training": train_info,
        "elapsed_seconds": round(elapsed, 1),
        "val": metrics_payload("Joint fusion, val", y_val_np, val_scores),
        "test": metrics_payload("Joint fusion, test", y_test_np, test_scores),
    }
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    checkpoint_path = args.checkpoint_dir / "joint_fusion.pt"
    torch.save(
        {
            "trainable_state_dict": model.trainable_state_dict(),
            "config": payload["config"],
            "best_epoch": train_info["best_epoch"],
            "best_val_macro_ap": train_info["best_val_macro_ap"],
        },
        checkpoint_path,
    )

    print(f"\nbest epoch {train_info['best_epoch']}, "
          f"val macro AP {train_info['best_val_macro_ap']:.4f}")
    print(f"TEST  macro AP {payload['test']['macro_ap']:.4f}  "
          f"micro {payload['test']['micro_ap']:.4f}  "
          f"samples {payload['test']['sample_ap']:.4f}")
    print(f"trained in {elapsed / 60:.1f} min")
    print(f"metrics    -> {metrics_path}")
    print(f"checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
