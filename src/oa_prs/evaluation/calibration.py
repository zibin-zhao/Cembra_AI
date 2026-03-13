"""
Calibration Metrics for PRS Evaluation.

Brier score, Hosmer-Lemeshow test, and calibration curve data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResults:
    """Container for calibration metric results."""

    brier_score: float
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_p: float
    prob_true: np.ndarray       # Calibration curve: observed frequencies
    prob_pred: np.ndarray       # Calibration curve: mean predicted probabilities
    n_bins: int

    def summary(self) -> dict[str, float]:
        return {
            "Brier_score": self.brier_score,
            "HL_statistic": self.hosmer_lemeshow_stat,
            "HL_p_value": self.hosmer_lemeshow_p,
        }


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Groups observations by predicted probability quantiles and tests whether
    observed event rates match predicted rates using a chi-squared test.

    Args:
        y_true: Binary outcomes [n_samples].
        y_prob: Predicted probabilities [n_samples].
        n_groups: Number of quantile groups.

    Returns:
        Tuple of (H-L statistic, p-value).
    """
    sorted_idx = np.argsort(y_prob)
    groups = np.array_split(sorted_idx, n_groups)

    hl_stat = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        n_g = len(group)
        o_g = y_true[group].sum()        # Observed events
        e_g = y_prob[group].sum()         # Expected events
        # Avoid division by zero
        if e_g > 0 and (n_g - e_g) > 0:
            hl_stat += (o_g - e_g) ** 2 / e_g
            hl_stat += ((n_g - o_g) - (n_g - e_g)) ** 2 / (n_g - e_g)

    # Chi-squared test with (n_groups - 2) degrees of freedom
    df = max(n_groups - 2, 1)
    p_value = 1 - stats.chi2.cdf(hl_stat, df)
    return float(hl_stat), float(p_value)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResults:
    """
    Compute calibration metrics.

    Args:
        y_true: Binary true labels [n_samples].
        y_prob: Predicted probabilities [n_samples].
        n_bins: Number of bins for calibration curve and HL test.

    Returns:
        CalibrationResults with Brier score, HL test, and calibration curve.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)
    hl_stat, hl_p = hosmer_lemeshow_test(y_true, y_prob, n_groups=n_bins)

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    logger.info("Brier=%.4f, HL_stat=%.2f (p=%.4f)", brier, hl_stat, hl_p)

    return CalibrationResults(
        brier_score=brier,
        hosmer_lemeshow_stat=hl_stat,
        hosmer_lemeshow_p=hl_p,
        prob_true=prob_true,
        prob_pred=prob_pred,
        n_bins=n_bins,
    )
