"""Evaluation metrics for multi-label classification (macro ROC-AUC)."""

from __future__ import annotations

import numpy as np

from ..schema import LABEL_NAMES


def macro_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC over 12 labels.

    Args:
        y_true: (N, 12) binary ground-truth
        y_pred: (N, 12) predicted probabilities [0, 1]

    Returns:
        Mean AUC across labels (scalar). Labels with only one class are skipped.
    """
    from sklearn.metrics import roc_auc_score

    aucs = []
    for i, name in enumerate(LABEL_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        if len(np.unique(t)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(t, p))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return {label_name: auc} for each label."""
    from sklearn.metrics import roc_auc_score

    result = {}
    for i, name in enumerate(LABEL_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        if len(np.unique(t)) < 2:
            result[name] = 0.0
            continue
        try:
            result[name] = float(roc_auc_score(t, p))
        except ValueError:
            result[name] = 0.0
    return result
