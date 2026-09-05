#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poster.config import BACKBONES, GENRES, ROOT as PROJECT_ROOT
from poster.dataset import PosterDataset
from poster.inference import load_model_from_checkpoint, resolve_device, run_inference
from poster.metrics import compute_ap_metrics
from poster.transforms import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vgg16_stanford", choices=BACKBONES)
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "test_data")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON path (default: results/poster/test_metrics_<backbone>.json)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT / "models" / "poster" / args.backbone / "best.pt"
    )
    output_path = args.output or (
        PROJECT_ROOT / "results" / "poster" / f"test_metrics_{args.backbone}.json"
    )

    device = resolve_device(args.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    test_ds = PosterDataset(
        args.test_dir, transform=get_transforms(args.backbone, train=False)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )

    y_true, y_score, _ = run_inference(model, test_loader, device)
    metrics = compute_ap_metrics(y_true, y_score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "backbone": checkpoint.get("backbone", args.backbone),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_macro_ap": checkpoint.get("val_macro_ap"),
        "test_samples": len(test_ds),
        "macro_ap": metrics["macro_ap"],
        "micro_ap": metrics["micro_ap"],
        "sample_ap": metrics["sample_ap"],
        "per_genre_ap": metrics["per_genre_ap"],
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"checkpoint={checkpoint_path}")
    print(f"test_samples={len(test_ds)}")
    print(f"macro_ap={metrics['macro_ap']:.4f}")
    print(f"micro_ap={metrics['micro_ap']:.4f}")
    print(f"sample_ap={metrics['sample_ap']:.4f}")
    print(f"metrics saved to {output_path}")


if __name__ == "__main__":
    main()
