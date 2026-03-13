"""
PRS Refinement Module.

Feeds fine-mapped posterior probabilities and functional priors back into
PRS weight estimation to improve prediction accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PRSRefiner:
    """
    Refine PRS weights using fine-mapping posteriors and functional annotations.

    Two refinement strategies:
        1. Posterior Direct: Use SuSiE-inf posterior effect sizes as PRS weights,
           filtered by PIP threshold.
        2. Prior Reweight: Inject PIP/functional scores as informative priors
           into PRS-CS/LDpred2, then re-estimate weights.

    Args:
        method: "posterior_direct" or "prior_reweight".
        pip_threshold: Minimum posterior inclusion probability to retain SNP.
        prior_weight: Blending weight between functional prior and data (for reweight).
    """

    def __init__(
        self,
        method: str = "posterior_direct",
        pip_threshold: float = 0.1,
        prior_weight: float = 0.5,
    ) -> None:
        self.method = method
        self.pip_threshold = pip_threshold
        self.prior_weight = prior_weight

    def refine_weights(
        self,
        baseline_weights: pd.DataFrame,
        finemapping_results: pd.DataFrame,
        functional_scores: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Produce refined PRS weights.

        Args:
            baseline_weights: DataFrame with columns [SNP, CHR, BP, A1, A2, BETA_baseline].
            finemapping_results: DataFrame with columns [SNP, CHR, BP, PIP, BETA_posterior].
            functional_scores: Optional DataFrame with [SNP, enformer_score, turf_score, ...].

        Returns:
            DataFrame with columns [SNP, CHR, BP, A1, A2, BETA_refined].
        """
        if self.method == "posterior_direct":
            return self._posterior_direct(baseline_weights, finemapping_results)
        elif self.method == "prior_reweight":
            return self._prior_reweight(
                baseline_weights, finemapping_results, functional_scores
            )
        else:
            raise ValueError(f"Unknown refinement method: {self.method}")

    def _posterior_direct(
        self,
        baseline: pd.DataFrame,
        finemapped: pd.DataFrame,
    ) -> pd.DataFrame:
        """Use fine-mapped posterior effects directly, filtered by PIP threshold."""
        logger.info(
            "Posterior direct refinement: PIP threshold=%.3f", self.pip_threshold
        )

        # Merge baseline and fine-mapping
        merged = baseline.merge(
            finemapped[["SNP", "PIP", "BETA_posterior"]],
            on="SNP",
            how="left",
        )

        # For SNPs with high PIP: use posterior effect
        # For SNPs below threshold: use baseline (possibly shrunk)
        has_pip = merged["PIP"].notna() & (merged["PIP"] >= self.pip_threshold)
        merged["BETA_refined"] = np.where(
            has_pip,
            merged["BETA_posterior"],
            merged["BETA_baseline"] * 0.5,  # Shrink non-fine-mapped SNPs
        )

        n_refined = has_pip.sum()
        logger.info(
            "Refined %d SNPs via posterior direct (%.1f%% of total)",
            n_refined, 100 * n_refined / len(merged),
        )
        return merged[["SNP", "CHR", "BP", "A1", "A2", "BETA_refined"]]

    def _prior_reweight(
        self,
        baseline: pd.DataFrame,
        finemapped: pd.DataFrame,
        functional: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Reweight baseline PRS using functional priors.

        Combines PIP from fine-mapping with functional annotation scores
        to create an informative prior weight, then blends with baseline beta.
        """
        logger.info("Prior reweight refinement: blend=%.2f", self.prior_weight)

        merged = baseline.merge(
            finemapped[["SNP", "PIP"]],
            on="SNP",
            how="left",
        )
        merged["PIP"] = merged["PIP"].fillna(0.0)

        # Compute prior weight from PIP and optional functional scores
        prior = merged["PIP"].copy()
        if functional is not None and not functional.empty:
            score_cols = [c for c in functional.columns if c != "SNP"]
            func_merged = merged.merge(functional, on="SNP", how="left")
            func_merged[score_cols] = func_merged[score_cols].fillna(0.0)
            # Average functional scores as additional prior information
            func_score = func_merged[score_cols].mean(axis=1)
            func_score = (func_score - func_score.mean()) / (func_score.std() + 1e-8)
            from scipy.special import expit
            prior = prior * 0.7 + expit(func_score) * 0.3

        # Blend: refined_beta = w * baseline + (1-w) * prior_adjusted_baseline
        prior_adjustment = 1.0 + prior * 2.0  # Boost high-PIP SNPs
        merged["BETA_refined"] = (
            self.prior_weight * merged["BETA_baseline"]
            + (1 - self.prior_weight) * merged["BETA_baseline"] * prior_adjustment
        )

        return merged[["SNP", "CHR", "BP", "A1", "A2", "BETA_refined"]]
