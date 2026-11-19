"""Metrics on held-out rats: balanced accuracy, Cohen's kappa, confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix

from .dataset import CLASS_NAMES


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true/y_pred: class indices 0..2 over labeled seconds only."""
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = [c for c in labels if (y_true == c).any()]
    per_class = {CLASS_NAMES[c]: float(cm[c, c] / cm[c].sum()) if cm[c].sum() else float("nan") for c in labels}
    return dict(
        n=int(y_true.size),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),   # mean recall over classes present
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        accuracy=float((y_true == y_pred).mean()),
        recall=per_class,
        classes_present=[CLASS_NAMES[c] for c in present],
        confusion=cm.tolist(),
    )


def pooled_rat_scores(per_session: dict[str, dict]) -> dict:
    """Pool a rat's sessions (true/pred arrays) into one score; each second counts once."""
    yt = np.concatenate([v["y_true"] for v in per_session.values()])
    yp = np.concatenate([v["y_pred"] for v in per_session.values()])
    return scores(yt, yp)
