from pathlib import Path

import numpy as np
import pandas as pd

from helpers import GENRES, load_split

TEXT_KEYS = {
    "logreg": ("val_logreg", "test_logreg"),
    "glove": ("val_glove", "test_glove"),
    "minilm": ("val_minilm", "test_minilm"),
}


def load_poster_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [g for g in GENRES if g not in df.columns]
    if missing:
        raise ValueError(f"Poster CSV missing genre columns: {missing}")
    if "movie_id" not in df.columns:
        raise ValueError("Poster CSV must include movie_id column")
    scores = df[GENRES].values.astype(np.float32)
    movie_ids = df["movie_id"].astype(int).values
    return movie_ids, scores


def load_text_predictions(
    npz_path: Path, text_key: str
) -> tuple[np.ndarray, np.ndarray]:
    if text_key not in TEXT_KEYS:
        raise ValueError(f"text_key must be one of {list(TEXT_KEYS)}")

    data = np.load(npz_path, allow_pickle=True)
    val_key, test_key = TEXT_KEYS[text_key]
    for key in (val_key, test_key):
        if key not in data.files:
            raise KeyError(f"{npz_path} missing array '{key}'")

    val_scores = np.asarray(data[val_key], dtype=np.float32)
    test_scores = np.asarray(data[test_key], dtype=np.float32)
    if val_scores.shape[1] != len(GENRES) or test_scores.shape[1] != len(GENRES):
        raise ValueError(
            f"Expected {len(GENRES)} genre columns in text predictions, "
            f"got val={val_scores.shape}, test={test_scores.shape}"
        )
    return val_scores, test_scores


def align_poster_with_split(
    split: str, movie_ids: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels_df = load_split(split)
    expected_ids = labels_df["movie_id"].astype(int).values
    if len(movie_ids) != len(expected_ids):
        raise ValueError(
            f"Poster {split} row count {len(movie_ids)} != labels {len(expected_ids)}"
        )
    if not np.array_equal(movie_ids, expected_ids):
        index = {mid: i for i, mid in enumerate(movie_ids)}
        try:
            order = [index[mid] for mid in expected_ids]
        except KeyError as exc:
            raise ValueError(
                f"Poster predictions missing movie_id present in {split} labels"
            ) from exc
        scores = scores[order]
        movie_ids = expected_ids
    return movie_ids, scores


def load_fusion_inputs(
    text_npz: Path,
    text_key: str,
    poster_val_csv: Path,
    poster_test_csv: Path,
) -> dict:
    val_labels = load_split("val")
    test_labels = load_split("test")
    y_val = val_labels[GENRES].values.astype(np.float32)
    y_test = test_labels[GENRES].values.astype(np.float32)

    val_movie_ids, poster_val = align_poster_with_split(
        "val", *load_poster_predictions(poster_val_csv)
    )
    test_movie_ids, poster_test = align_poster_with_split(
        "test", *load_poster_predictions(poster_test_csv)
    )

    text_val, text_test = load_text_predictions(text_npz, text_key)

    if poster_val.shape != text_val.shape:
        raise ValueError(
            f"Val shape mismatch poster {poster_val.shape} vs text {text_val.shape}"
        )
    if poster_test.shape != text_test.shape:
        raise ValueError(
            f"Test shape mismatch poster {poster_test.shape} vs text {text_test.shape}"
        )

    train_labels = load_split("train")
    n = len(train_labels)
    counts = train_labels[GENRES].sum().values.astype(np.float32)
    pos_weight = (n - counts) / np.maximum(counts, 1.0)

    return {
        "y_val": y_val,
        "y_test": y_test,
        "poster_val": poster_val,
        "poster_test": poster_test,
        "text_val": text_val,
        "text_test": text_test,
        "val_movie_ids": val_movie_ids,
        "test_movie_ids": test_movie_ids,
        "pos_weight": pos_weight,
    }
