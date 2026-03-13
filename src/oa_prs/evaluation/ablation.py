"""
Ablation Study Module.

Systematically removes branches/features to measure each component's
contribution to the final ensemble performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import numpy as np

from .discrimination import compute_discrimination

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    removed_branches: list[str]
    remaining_branches: list[str]
    auc_roc: float
    delta_auc: float  # Change from full model


@dataclass
class AblationStudy:
    """Complete ablation study results."""

    full_model_auc: float
    results: list[AblationResult] = field(default_factory=list)
    branch_importance: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "full_model_AUC": self.full_model_auc,
            "branch_importance": self.branch_importance,
            "n_experiments": len(self.results),
        }


def run_ablation(
    branch_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    ensemble_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    max_remove: int = 2,
) -> AblationStudy:
    """
    Run ablation study by systematically removing branches.

    For each subset of branches to remove (up to max_remove at a time),
    re-fits the ensemble without those branches and measures AUC change.

    Args:
        branch_predictions: Dict mapping branch name -> predicted scores.
        y_true: Binary outcomes.
        ensemble_fn: Callable that takes a dict of branch predictions and
                     returns combined scores.
        max_remove: Maximum number of branches to remove simultaneously.

    Returns:
        AblationStudy with per-branch importance scores.
    """
    all_branches = list(branch_predictions.keys())
    logger.info("Ablation study: %d branches, max_remove=%d", len(all_branches), max_remove)

    # Full model performance
    full_scores = ensemble_fn(branch_predictions)
    full_disc = compute_discrimination(y_true, full_scores)
    full_auc = full_disc.auc_roc
    logger.info("Full model AUC: %.4f", full_auc)

    results: list[AblationResult] = []
    branch_deltas: dict[str, list[float]] = {b: [] for b in all_branches}

    for n_remove in range(1, min(max_remove + 1, len(all_branches))):
        for to_remove in combinations(all_branches, n_remove):
            remaining = {k: v for k, v in branch_predictions.items() if k not in to_remove}
            if not remaining:
                continue

            try:
                ablated_scores = ensemble_fn(remaining)
                ablated_disc = compute_discrimination(y_true, ablated_scores)
                delta = full_auc - ablated_disc.auc_roc

                result = AblationResult(
                    removed_branches=list(to_remove),
                    remaining_branches=list(remaining.keys()),
                    auc_roc=ablated_disc.auc_roc,
                    delta_auc=delta,
                )
                results.append(result)

                # Track delta per removed branch
                for b in to_remove:
                    branch_deltas[b].append(delta)

                logger.info(
                    "  Remove %s: AUC=%.4f (Δ=%.4f)",
                    to_remove, ablated_disc.auc_roc, delta,
                )
            except Exception as e:
                logger.warning("Ablation failed for %s: %s", to_remove, e)

    # Compute importance as mean AUC drop when branch is removed
    branch_importance = {}
    for branch, deltas in branch_deltas.items():
        if deltas:
            branch_importance[branch] = float(np.mean(deltas))
        else:
            branch_importance[branch] = 0.0

    # Sort by importance (highest first)
    branch_importance = dict(sorted(branch_importance.items(), key=lambda x: -x[1]))

    logger.info("Branch importance ranking:")
    for branch, imp in branch_importance.items():
        logger.info("  %s: %.4f", branch, imp)

    return AblationStudy(
        full_model_auc=full_auc,
        results=results,
        branch_importance=branch_importance,
    )


def run_additive_ablation(
    branch_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    ensemble_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
) -> list[dict[str, Any]]:
    """
    Additive ablation: start from no branches, add one at a time.

    Orders branches by their incremental contribution.

    Args:
        branch_predictions: Dict mapping branch name -> predicted scores.
        y_true: Binary outcomes.
        ensemble_fn: Callable for combining branch predictions.

    Returns:
        List of dicts with cumulative performance as each branch is added.
    """
    all_branches = list(branch_predictions.keys())
    results = []
    current_branches: dict[str, np.ndarray] = {}
    prev_auc = 0.5  # Random baseline

    # Greedily add the branch that gives the biggest improvement
    remaining = set(all_branches)
    for step in range(len(all_branches)):
        best_branch = None
        best_auc = -1.0

        for branch in remaining:
            trial = {**current_branches, branch: branch_predictions[branch]}
            try:
                scores = ensemble_fn(trial)
                disc = compute_discrimination(y_true, scores)
                if disc.auc_roc > best_auc:
                    best_auc = disc.auc_roc
                    best_branch = branch
            except Exception:
                continue

        if best_branch is None:
            break

        current_branches[best_branch] = branch_predictions[best_branch]
        remaining.discard(best_branch)

        results.append({
            "step": step + 1,
            "added_branch": best_branch,
            "cumulative_branches": list(current_branches.keys()),
            "auc_roc": best_auc,
            "delta_auc": best_auc - prev_auc,
        })
        prev_auc = best_auc

        logger.info(
            "  Step %d: +%s → AUC=%.4f (Δ=%.4f)",
            step + 1, best_branch, best_auc, results[-1]["delta_auc"],
        )

    return results
