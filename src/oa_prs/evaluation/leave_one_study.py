"""
Leave-One-Study-Out Cross-Validation.

Validates PRS model robustness by training on one GWAS study and evaluating
on another. For OA: train on UKB 2019, evaluate on MVP+UKB 2022, and vice versa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .discrimination import compute_discrimination
from .calibration import compute_calibration

logger = logging.getLogger(__name__)


@dataclass
class LOSOResults:
    """Leave-One-Study-Out cross-validation results."""

    study_results: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_auc_roc: float = 0.0
    std_auc_roc: float = 0.0
    mean_brier: float = 0.0
    consistency_score: float = 0.0  # 1 - |AUC_study1 - AUC_study2| / mean(AUC)

    def summary(self) -> dict[str, float]:
        return {
            "mean_AUC-ROC": self.mean_auc_roc,
            "std_AUC-ROC": self.std_auc_roc,
            "mean_Brier": self.mean_brier,
            "consistency_score": self.consistency_score,
            "n_studies": len(self.study_results),
        }


def leave_one_study_out(
    studies: dict[str, dict[str, np.ndarray]],
    model_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
) -> LOSOResults:
    """
    Perform leave-one-study-out cross-validation.

    For each study, trains the model on all OTHER studies, then evaluates on
    the held-out study. This tests generalizability across GWAS cohorts.

    Args:
        studies: Dict mapping study name -> {"X": features, "y": labels}.
                 E.g., {"ukb_2019": {"X": ..., "y": ...}, "mvp_ukb_2022": {...}}
        model_fn: Callable(X_train, y_train) -> fitted model.
        predict_fn: Callable(model, X_test) -> predicted probabilities.

    Returns:
        LOSOResults with per-study metrics and consistency score.
    """
    study_names = list(studies.keys())
    study_metrics: dict[str, dict[str, float]] = {}
    aucs = []
    briers = []

    for holdout_name in study_names:
        logger.info("LOSO: holding out %s", holdout_name)

        # Train on all other studies
        X_train_parts = []
        y_train_parts = []
        for name, data in studies.items():
            if name != holdout_name:
                X_train_parts.append(data["X"])
                y_train_parts.append(data["y"])

        if not X_train_parts:
            logger.warning("Only one study available; skipping LOSO.")
            continue

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        X_test = studies[holdout_name]["X"]
        y_test = studies[holdout_name]["y"]

        # Train and predict
        model = model_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)

        # Evaluate
        disc = compute_discrimination(y_test, y_pred)
        cal = compute_calibration(y_test, y_pred, n_bins=min(10, len(y_test) // 10))

        metrics = {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "prevalence_test": float(y_test.mean()),
            "AUC-ROC": disc.auc_roc,
            "AUC-PR": disc.auc_pr,
            "Brier": cal.brier_score,
        }
        study_metrics[holdout_name] = metrics
        aucs.append(disc.auc_roc)
        briers.append(cal.brier_score)

        logger.info(
            "  %s: AUC=%.4f, Brier=%.4f (n_test=%d)",
            holdout_name, disc.auc_roc, cal.brier_score, len(y_test),
        )

    # Aggregate
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    std_auc = float(np.std(aucs)) if aucs else 0.0
    mean_brier = float(np.mean(briers)) if briers else 0.0

    # Consistency: how similar are AUCs across studies
    if len(aucs) >= 2 and mean_auc > 0:
        auc_range = max(aucs) - min(aucs)
        consistency = max(0.0, 1.0 - auc_range / mean_auc)
    else:
        consistency = 1.0 if len(aucs) == 1 else 0.0

    return LOSOResults(
        study_results=study_metrics,
        mean_auc_roc=mean_auc,
        std_auc_roc=std_auc,
        mean_brier=mean_brier,
        consistency_score=consistency,
    )


def cross_study_validation(
    gwas_2019_scores: np.ndarray,
    gwas_2022_scores: np.ndarray,
    y_true: np.ndarray,
    study_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Simplified 2-study cross-validation for OA GWAS.

    Tests whether PRS trained on 2019 GWAS generalizes to 2022 and vice versa.
    This is a quicker alternative to full LOSO when we have exactly 2 studies.

    Args:
        gwas_2019_scores: PRS scores from 2019 weights [n_samples].
        gwas_2022_scores: PRS scores from 2022 weights [n_samples].
        y_true: Binary outcome [n_samples].
        study_labels: Study membership labels [n_samples].

    Returns:
        Dict with cross-validated metrics for each direction.
    """
    results = {}

    # PRS trained on 2019, evaluated on 2022 cohort
    mask_2022 = study_labels == "mvp_ukb_2022"
    if mask_2022.sum() > 20:
        disc = compute_discrimination(y_true[mask_2022], gwas_2019_scores[mask_2022])
        results["2019_weights_on_2022"] = {
            "AUC-ROC": disc.auc_roc,
            "n": int(mask_2022.sum()),
        }

    # PRS trained on 2022, evaluated on 2019 cohort
    mask_2019 = study_labels == "ukb_2019"
    if mask_2019.sum() > 20:
        disc = compute_discrimination(y_true[mask_2019], gwas_2022_scores[mask_2019])
        results["2022_weights_on_2019"] = {
            "AUC-ROC": disc.auc_roc,
            "n": int(mask_2019.sum()),
        }

    return results
