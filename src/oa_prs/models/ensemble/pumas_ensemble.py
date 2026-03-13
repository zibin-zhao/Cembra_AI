"""
PUMAS-Ensemble: Summary Statistics-Only PRS Benchmarking.

Implements PUMAS (Polygenic prediction via Bayesian regression and
continuous shrinkage priors Using GWAS Summary Statistics) for model
selection and ensemble without requiring individual-level data.

Reference: Zhao et al., Nat Commun 2021.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PUMASResults:
    """Container for PUMAS ensemble results."""

    method_r2: dict[str, float] = field(default_factory=dict)
    best_method: str = ""
    best_r2: float = 0.0
    ensemble_r2: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "best_method": self.best_method,
            "best_r2": self.best_r2,
            "ensemble_r2": self.ensemble_r2,
            "n_methods": len(self.method_r2),
            "weights": self.weights,
        }


class PUMASEnsemble:
    """
    Summary-statistics-based model evaluation and ensembling.

    PUMAS estimates prediction R² using only GWAS summary statistics
    and LD reference panels, without requiring individual-level data.
    This allows model comparison and ensemble weight optimization
    when only summary statistics are available.

    Args:
        ld_ref_dir: Directory with LD reference panel.
        n_gwas: GWAS sample size.
        n_bootstrap: Number of bootstrap samples for R² estimation.
    """

    def __init__(
        self,
        ld_ref_dir: str | Path,
        n_gwas: int,
        n_bootstrap: int = 200,
    ):
        self.ld_ref_dir = Path(ld_ref_dir)
        self.n_gwas = n_gwas
        self.n_bootstrap = n_bootstrap

    def estimate_r2(
        self,
        weights: np.ndarray,
        sumstats: pd.DataFrame,
        ld_matrix: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Estimate PRS prediction R² from summary statistics.

        Uses the PUMAS approach: R² ≈ β_prs' * R_LD * β_prs / Var(y),
        where β_prs are the PRS weights and R_LD is the LD matrix.

        Args:
            weights: Per-SNP PRS weights [n_snps].
            sumstats: Summary statistics with BETA, SE, N columns.
            ld_matrix: LD correlation matrix [n_snps, n_snps].
                       If None, uses diagonal (assumes no LD).

        Returns:
            Tuple of (R² estimate, standard error).
        """
        beta_gwas = sumstats["BETA"].values
        se_gwas = sumstats["SE"].values
        n = sumstats.get("N", pd.Series([self.n_gwas] * len(sumstats))).values

        p = len(weights)

        if ld_matrix is not None:
            # R² ≈ w' * R * β_gwas where R is LD matrix
            r2_est = float(weights @ ld_matrix @ beta_gwas)
        else:
            # No LD: R² ≈ Σ w_j * β_j
            r2_est = float(np.dot(weights, beta_gwas))

        # Bound to [0, 1]
        r2_est = max(0.0, min(1.0, r2_est))

        # Bootstrap SE
        r2_boots = []
        for _ in range(self.n_bootstrap):
            noise = np.random.normal(0, se_gwas)
            beta_boot = beta_gwas + noise
            if ld_matrix is not None:
                r2_b = float(weights @ ld_matrix @ beta_boot)
            else:
                r2_b = float(np.dot(weights, beta_boot))
            r2_boots.append(max(0.0, min(1.0, r2_b)))

        se = float(np.std(r2_boots))

        return r2_est, se

    def compare_methods(
        self,
        method_weights: dict[str, np.ndarray],
        sumstats: pd.DataFrame,
        ld_matrix: np.ndarray | None = None,
    ) -> PUMASResults:
        """
        Compare multiple PRS methods using PUMAS R² estimation.

        Args:
            method_weights: Dict mapping method name -> weight vectors.
            sumstats: GWAS summary statistics.
            ld_matrix: LD correlation matrix (optional).

        Returns:
            PUMASResults with per-method R² and best method selection.
        """
        method_r2 = {}

        for method_name, weights in method_weights.items():
            r2, se = self.estimate_r2(weights, sumstats, ld_matrix)
            method_r2[method_name] = r2
            logger.info("PUMAS R² for %s: %.4f (SE=%.4f)", method_name, r2, se)

        # Best single method
        best_method = max(method_r2, key=method_r2.get)
        best_r2 = method_r2[best_method]

        # Simple ensemble: inverse-variance weighted combination
        ensemble_weights = self._optimize_weights(method_weights, sumstats, ld_matrix)
        ensemble_r2 = self._estimate_ensemble_r2(
            method_weights, ensemble_weights, sumstats, ld_matrix,
        )

        logger.info(
            "PUMAS best: %s (R²=%.4f), ensemble R²=%.4f",
            best_method, best_r2, ensemble_r2,
        )

        return PUMASResults(
            method_r2=method_r2,
            best_method=best_method,
            best_r2=best_r2,
            ensemble_r2=ensemble_r2,
            weights=ensemble_weights,
        )

    def _optimize_weights(
        self,
        method_weights: dict[str, np.ndarray],
        sumstats: pd.DataFrame,
        ld_matrix: np.ndarray | None,
    ) -> dict[str, float]:
        """
        Optimize ensemble weights via grid search on estimated R².
        """
        methods = list(method_weights.keys())
        n_methods = len(methods)

        if n_methods <= 1:
            return {m: 1.0 for m in methods}

        # Simple proportional weighting based on individual R²
        r2_values = {}
        for m, w in method_weights.items():
            r2, _ = self.estimate_r2(w, sumstats, ld_matrix)
            r2_values[m] = max(r2, 0.0)

        total_r2 = sum(r2_values.values())
        if total_r2 > 0:
            return {m: r2 / total_r2 for m, r2 in r2_values.items()}
        else:
            return {m: 1.0 / n_methods for m in methods}

    def _estimate_ensemble_r2(
        self,
        method_weights: dict[str, np.ndarray],
        ensemble_weights: dict[str, float],
        sumstats: pd.DataFrame,
        ld_matrix: np.ndarray | None,
    ) -> float:
        """Estimate R² for the weighted ensemble."""
        combined = np.zeros(len(sumstats))
        for method, w in ensemble_weights.items():
            combined += w * method_weights[method]

        r2, _ = self.estimate_r2(combined, sumstats, ld_matrix)
        return r2
