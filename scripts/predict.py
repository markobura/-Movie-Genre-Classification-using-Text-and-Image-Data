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


def split_name_from_dir(split_dir: Path) -> str:
    name = split_dir.name.replace("_data", "")
    if name not in ("train", "val", "test"):
        raise ValueError(f"Unrecognized split directory: {split_dir}")
    return name


def default_output_path(split: str, backbone: str) -> Path:
    return PROJECT_ROOT / "results" / "poster" / f"predictions_{split}_{backbone}.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vgg16_stanford", choices=BACKBONES)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Split directory (e.g. val_data, test_data). Default: test_data",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --split-dir",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Predictions CSV (default: results/poster/predictions_<split>_<backbone>.csv)",
    )
    args = parser.parse_args()

    split_dir = args.split_dir or args.test_dir or (PROJECT_ROOT / "test_data")
    split = split_name_from_dir(split_dir)

    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT / "models" / "poster" / args.backbone / "best.pt"
    )
    output_path = args.output or default_output_path(split, args.backbone)

    device = resolve_device(args.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    dataset = PosterDataset(
        split_dir, transform=get_transforms(args.backbone, train=False)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )

    _, y_score, movie_ids = run_inference(model, loader, device)

    df = pd.DataFrame(y_score, columns=GENRES)
    df.insert(0, "movie_id", movie_ids.astype(int))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"split={split} samples={len(df)}")
    print(f"checkpoint={checkpoint_path}")
    print(f"predictions={len(df)} rows x {len(GENRES)} genres")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
