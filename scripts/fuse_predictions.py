#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fusion.data import load_fusion_inputs
from fusion.late_fusion import predict_fusion, train_late_fusion
from fusion.metrics import build_comparison_table, metrics_payload, mean_fusion_scores
from poster.config import GENRES, ROOT as PROJECT_ROOT
from poster.inference import resolve_device

FUSION_NOTES = (
    "Stanford (LeBaron) combines four modalities (poster, video, metadata, text) "
    "with a final fully-connected fusion layer. Our fusion uses poster + text only. "
    "Fusion is trained on validation predictions, not train, because base models "
    "are overfit on train (see notebooks/01_text_classification.ipynb section 11). "
    "Limitations: the late fusion head is selected on the same 491 val rows it is "
    "trained on, so best_val_macro_ap is a fit score, not a held-out score."
)


def main():
    parser = argparse.ArgumentParser(
        description="Late fusion of poster and text predictions (Stanford-aligned)."
    )
    parser.add_argument(
        "--text-npz",
        type=Path,
        default=PROJECT_ROOT / "models" / "text_predictions.npz",
    )
    parser.add_argument(
        "--text-key",
        default="logreg",
        choices=["logreg", "glove", "minilm"],
        help="Text model arrays inside npz (logreg = TF-IDF + LogReg)",
    )
    parser.add_argument(
        "--poster-val-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "poster"
        / "predictions_val_clip_vit_b32.csv",
    )
    parser.add_argument(
        "--poster-test-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "poster"
        / "predictions_test_clip_vit_b32.csv",
    )
    parser.add_argument("--poster-backbone", default="clip_vit_b32")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "fusion",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "fusion",
    )
    args = parser.parse_args()

    if not args.text_npz.exists():
        raise FileNotFoundError(
            f"Missing {args.text_npz}. Run notebooks/01_text_classification.ipynb "
            "section 11 to export "
            "models/text_predictions.npz."
        )

    data = load_fusion_inputs(
        args.text_npz,
        args.text_key,
        args.poster_val_csv,
        args.poster_test_csv,
    )

    device = resolve_device(args.device)
    print(f"device={device}")
    print(f"val={len(data['y_val'])} test={len(data['y_test'])}")

    poster_val_t = torch.tensor(data["poster_val"])
    text_val_t = torch.tensor(data["text_val"])
    y_val_t = torch.tensor(data["y_val"])
    pos_weight = torch.tensor(data["pos_weight"], dtype=torch.float32)

    model, train_info = train_late_fusion(
        poster_val_t,
        text_val_t,
        y_val_t,
        pos_weight,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        seed=args.seed,
        device=device,
    )
    print(
        f"late fusion trained: best_epoch={train_info['best_epoch']} "
        f"val_macro_ap={train_info['best_val_macro_ap']:.4f}"
    )

    fusion_test = predict_fusion(
        model,
        torch.tensor(data["poster_test"]),
        torch.tensor(data["text_test"]),
        device,
    ).numpy()

    poster_name = f"Poster ({args.poster_backbone})"
    text_name = "Text (TF-IDF + LogReg)" if args.text_key == "logreg" else f"Text ({args.text_key})"
    fusion_name = f"Late fusion ({args.poster_backbone} + {args.text_key})"

    comparison = build_comparison_table(
        data["y_test"],
        data["poster_test"],
        data["text_test"],
        fusion_test,
        poster_name=poster_name,
        text_name=text_name,
        fusion_name=fusion_name,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = args.output_dir / "comparison_test.csv"
    comparison.to_csv(comparison_path)

    predictions_df = pd.DataFrame(fusion_test, columns=GENRES)
    predictions_df.insert(0, "movie_id", data["test_movie_ids"])
    predictions_path = args.output_dir / "predictions_test_fusion.csv"
    predictions_df.to_csv(predictions_path, index=False)

    checkpoint_path = (
        args.checkpoint_dir / f"late_fusion_{args.text_key}_{args.poster_backbone}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "text_key": args.text_key,
            "poster_backbone": args.poster_backbone,
            "best_epoch": train_info["best_epoch"],
            "best_val_macro_ap": train_info["best_val_macro_ap"],
            "train_history": train_info["history"],
        },
        checkpoint_path,
    )

    metrics = {
        "notes": FUSION_NOTES,
        "text_npz": str(args.text_npz.resolve()),
        "text_key": args.text_key,
        "poster_backbone": args.poster_backbone,
        "fusion_training": train_info,
        "models": {
            "poster_only": metrics_payload(poster_name, data["y_test"], data["poster_test"]),
            "text_only": metrics_payload(text_name, data["y_test"], data["text_test"]),
            "mean_fusion": metrics_payload(
                "Mean fusion (poster + text)",
                data["y_test"],
                mean_fusion_scores(data["poster_test"], data["text_test"]),
            ),
            "late_fusion": metrics_payload(fusion_name, data["y_test"], fusion_test),
        },
        "paper_results_index": list(comparison.index),
    }
    metrics_path = args.output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest comparison (sorted by AP_macro):")
    print(comparison.to_string())
    print(f"\ncomparison saved to {comparison_path}")
    print(f"metrics saved to {metrics_path}")
    print(f"predictions saved to {predictions_path}")
    print(f"checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
