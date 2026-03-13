"""
Cross-Ancestry Fairness Evaluation.

Assess PRS model performance consistency across ancestries, sexes, and age groups.
Based on Cell Genomics guidelines for cross-ancestry PRS fairness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .discrimination import compute_discrimination
from .calibration import compute_calibration

logger = logging.getLogger(__name__)


@dataclass
class FairnessResults:
    """Container for fairness evaluation results."""

    subgroup_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    auc_gap: float = 0.0              # Max AUC difference across groups
    calibration_gap: float = 0.0      # Max Brier score difference
    threshold_consistency: float = 0.0  # Proportion of groups where same threshold works

    def summary(self) -> dict[str, float]:
        return {
            "AUC_gap": self.auc_gap,
            "calibration_gap": self.calibration_gap,
            "threshold_consistency": self.threshold_consistency,
            "n_subgroups": len(self.subgroup_metrics),
        }


def evaluate_fairness(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subgroup_labels: pd.Series | np.ndarray,
    subgroup_name: str = "ancestry",
    reference_threshold: float | None = None,
) -> FairnessResults:
    """
    Evaluate PRS fairness across subgroups.

    For each subgroup, computes AUC-ROC, Brier score, and checks whether
    a common threshold gives consistent sensitivity/specificity.

    Args:
        y_true: Binary outcomes [n_samples].
        y_score: Risk scores [n_samples].
        subgroup_labels: Group labels [n_samples] (e.g., "EUR", "EAS").
        subgroup_name: Name of the subgroup dimension (for logging).
        reference_threshold: Optional fixed threshold to test consistency.

    Returns:
        FairnessResults with per-subgroup metrics and gap statistics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    labels = np.asarray(subgroup_labels)
    unique_groups = np.unique(labels)

    if reference_threshold is None:
        # Use overall median as default threshold
        reference_threshold = float(np.median(y_score))

    subgroup_metrics: dict[str, dict[str, float]] = {}
    aucs = []
    briers = []
    sensitivities_at_threshold = []

    for group in unique_groups:
        mask = labels == group
        n_group = mask.sum()
        if n_group < 20:
            logger.warning("Subgroup '%s' has only %d samples; skipping.", group, n_group)
            continue

        y_t = y_true[mask]
        y_s = y_score[mask]
        n_cases = int(y_t.sum())

        if n_cases == 0 or n_cases == n_group:
            logger.warning("Subgroup '%s': no variance in outcome.", group)
            continue

        disc = compute_discrimination(y_t, y_s)
        cal = compute_calibration(y_t, y_s, n_bins=min(10, n_group // 5))

        # Sensitivity at the reference threshold
        predicted_positive = y_s >= reference_threshold
        sensitivity = float(y_t[predicted_positive].sum()) / max(n_cases, 1)

        metrics = {
            "n": n_group,
            "n_cases": n_cases,
            "prevalence": n_cases / n_group,
            "AUC-ROC": disc.auc_roc,
            "AUC-PR": disc.auc_pr,
            "Brier": cal.brier_score,
            "sensitivity_at_threshold": sensitivity,
        }
        subgroup_metrics[str(group)] = metrics
        aucs.append(disc.auc_roc)
        briers.append(cal.brier_score)
        sensitivities_at_threshold.append(sensitivity)

    # Compute gaps
    auc_gap = max(aucs) - min(aucs) if aucs else 0.0
    cal_gap = max(briers) - min(briers) if briers else 0.0

    # Threshold consistency: all subgroups within 10% sensitivity of each other
    if sensitivities_at_threshold:
        sens_range = max(sensitivities_at_threshold) - min(sensitivities_at_threshold)
        threshold_consistency = 1.0 if sens_range < 0.10 else max(0, 1 - sens_range)
    else:
        threshold_consistency = 0.0

    logger.info(
        "Fairness (%s): %d subgroups, AUC_gap=%.4f, cal_gap=%.4f, threshold_consistency=%.2f",
        subgroup_name, len(subgroup_metrics), auc_gap, cal_gap, threshold_consistency,
    )

    return FairnessResults(
        subgroup_metrics=subgroup_metrics,
        auc_gap=auc_gap,
        calibration_gap=cal_gap,
        threshold_consistency=threshold_consistency,
    )
