import numpy as np
import pandas as pd

from helpers import PAPER_RESULTS, evaluate, per_genre_ap, results_table


def scores_to_row(name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return evaluate(y_true, y_score, thresholds=None, name=name)


def mean_fusion_scores(poster: np.ndarray, text: np.ndarray) -> np.ndarray:
    return ((poster + text) / 2.0).astype(np.float32)


def build_comparison_table(
    y_test: np.ndarray,
    poster_test: np.ndarray,
    text_test: np.ndarray,
    fusion_test: np.ndarray,
    *,
    poster_name: str,
    text_name: str,
    fusion_name: str,
    include_paper: bool = True,
) -> pd.DataFrame:
    rows = [
        scores_to_row(poster_name, y_test, poster_test),
        scores_to_row(text_name, y_test, text_test),
        scores_to_row("Mean fusion (poster + text)", y_test, mean_fusion_scores(poster_test, text_test)),
        scores_to_row(fusion_name, y_test, fusion_test),
    ]

    table = results_table(rows)
    if include_paper:
        paper = PAPER_RESULTS.copy()
        table = pd.concat([table, paper[["AP_micro", "AP_macro", "AP_samples"]]])
    return table


def metrics_payload(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    row = scores_to_row(name, y_true, y_score)
    return {
        "name": name,
        "macro_ap": row["AP_macro"],
        "micro_ap": row["AP_micro"],
        "sample_ap": row["AP_samples"],
        "per_genre_ap": per_genre_ap(y_true, y_score).to_dict(),
    }
