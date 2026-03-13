"""
Discrimination Metrics for PRS Evaluation.

AUC-ROC, AUC-PR, and related discrimination measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscriminationResults:
    """Container for discrimination metric results."""

    auc_roc: float
    auc_pr: float
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    thresholds_roc: np.ndarray
    n_cases: int
    n_controls: int

    def summary(self) -> dict[str, float]:
        return {
            "AUC-ROC": self.auc_roc,
            "AUC-PR": self.auc_pr,
            "N_cases": self.n_cases,
            "N_controls": self.n_controls,
        }


def compute_discrimination(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> DiscriminationResults:
    """
    Compute AUC-ROC and AUC-PR for binary classification.

    Args:
        y_true: Binary true labels [n_samples] (0=control, 1=case).
        y_score: Predicted risk scores/probabilities [n_samples].

    Returns:
        DiscriminationResults with AUC values and curve data.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    n_cases = int(y_true.sum())
    n_controls = int(len(y_true) - n_cases)

    if n_cases == 0 or n_controls == 0:
        logger.warning("Only one class present; AUC is undefined.")
        return DiscriminationResults(
            auc_roc=np.nan, auc_pr=np.nan,
            fpr=np.array([]), tpr=np.array([]),
            precision=np.array([]), recall=np.array([]),
            thresholds_roc=np.array([]),
            n_cases=n_cases, n_controls=n_controls,
        )

    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    fpr, tpr, thresh_roc = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    logger.info("AUC-ROC=%.4f, AUC-PR=%.4f (cases=%d, controls=%d)",
                auc_roc, auc_pr, n_cases, n_controls)

    return DiscriminationResults(
        auc_roc=auc_roc, auc_pr=auc_pr,
        fpr=fpr, tpr=tpr,
        precision=prec, recall=rec,
        thresholds_roc=thresh_roc,
        n_cases=n_cases, n_controls=n_controls,
    )
