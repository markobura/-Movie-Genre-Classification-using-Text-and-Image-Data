from pathlib import Path

import numpy as np
import pandas as pd

from helpers import GENRES, clean_plot, load_split
from poster.config import ROOT

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


def text_train_scores(text_key: str) -> np.ndarray:
    """Text-model probabilities on the train split.

    `models/text_predictions.npz` holds only val and test, so these are
    recomputed from the saved sklearn artifacts through the same pipeline
    notebook 01 used.
    """
    import joblib

    models_dir = ROOT / "models"
    train = load_split("train")

    if text_key == "logreg":
        vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        logreg = joblib.load(models_dir / "text_logreg.joblib")
        features = vectorizer.transform(clean_plot(train["plot"]))
        return logreg.predict_proba(features).astype(np.float32)

    if text_key == "minilm":
        embeddings = np.load(models_dir / "emb_train.npy")
        minilm = joblib.load(models_dir / "text_minilm_logreg.joblib")
        return minilm.predict_proba(embeddings).astype(np.float32)

    raise ValueError(
        f"text_key={text_key!r} has no saved artifact to score the train split; "
        "use 'logreg' or 'minilm'"
    )


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
    poster_train_csv: Path,
    poster_val_csv: Path,
    poster_test_csv: Path,
) -> dict:
    """Predictions of both modalities on all three splits, aligned with the
    label files."""
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

    train_movie_ids, poster_train = align_poster_with_split(
        "train", *load_poster_predictions(poster_train_csv)
    )
    text_train = text_train_scores(text_key)
    if poster_train.shape != text_train.shape:
        raise ValueError(
            f"Train shape mismatch poster {poster_train.shape} "
            f"vs text {text_train.shape}"
        )

    return {
        "y_train": train_labels[GENRES].values.astype(np.float32),
        "y_val": y_val,
        "y_test": y_test,
        "poster_train": poster_train,
        "poster_val": poster_val,
        "poster_test": poster_test,
        "text_train": text_train,
        "text_val": text_val,
        "text_test": text_test,
        "train_movie_ids": train_movie_ids,
        "val_movie_ids": val_movie_ids,
        "test_movie_ids": test_movie_ids,
        "pos_weight": pos_weight,
    }
