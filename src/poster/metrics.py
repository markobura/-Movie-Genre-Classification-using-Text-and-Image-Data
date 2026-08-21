import numpy as np
from sklearn.metrics import average_precision_score

from .config import GENRES

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
def compute_ap_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Multi-label metrics: macro/micro/sample AP and per-genre AP."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if y_true.shape[1] != len(GENRES):
        raise ValueError(
            f"Expected {len(GENRES)} labels, got {y_true.shape[1]} columns"
        )

    per_genre = {}
    for i, genre in enumerate(GENRES):
        if y_true[:, i].sum() == 0:
            per_genre[genre] = 0.0
        else:
            per_genre[genre] = float(
                average_precision_score(y_true[:, i], y_score[:, i])
            )

    macro_ap = float(np.mean(list(per_genre.values())))
    micro_ap = float(average_precision_score(y_true, y_score, average="micro"))
    sample_ap = float(average_precision_score(y_true, y_score, average="samples"))

    return {
        "macro_ap": macro_ap,
        "micro_ap": micro_ap,
        "sample_ap": sample_ap,
        "per_genre_ap": per_genre,
    }
