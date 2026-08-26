#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poster.config import BACKBONES, GENRES, ROOT as PROJECT_ROOT
from poster.dataset import PosterDataset
from poster.inference import load_model_from_checkpoint, resolve_device, run_inference
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
        help="Predictions CSV (default: results/poster/predictions_test_<backbone>.csv)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT / "models" / "poster" / args.backbone / "best.pt"
    )
    output_path = args.output or (
        PROJECT_ROOT / "results" / "poster" / f"predictions_test_{args.backbone}.csv"
    )

    device = resolve_device(args.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    test_ds = PosterDataset(
        args.test_dir, transform=get_transforms(args.backbone, train=False)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )

    _, y_score, movie_ids = run_inference(model, test_loader, device)

    df = pd.DataFrame(y_score, columns=GENRES)
    df.insert(0, "movie_id", movie_ids.astype(int))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"checkpoint={checkpoint_path}")
    print(f"predictions={len(df)} rows x {len(GENRES)} genres")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
