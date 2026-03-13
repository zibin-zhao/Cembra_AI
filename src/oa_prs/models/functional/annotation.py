"""
Tissue-specific annotation scoring (TURF, TissueLinc) for variant prioritization.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class TissueAnnotator:
    """
    Load and use tissue-specific variant annotations for SNP prioritization.

    Attributes
    ----------
    turf_scores : Optional[pd.DataFrame]
        Loaded TURF scores
    """

    def __init__(self):
        """Initialize tissue annotator."""
        self.turf_scores = None
        log.info("tissue_annotator_initialized")

    def load_turf_scores(self, path: str | Path) -> pd.DataFrame:
        """
        Load TURF (Tissue-specific UnRegulated Function) scores.

        TURF scores quantify the functional importance of variants in specific tissues.

        Expected format:
        SNP CHR BP A1 A2 TURF_BONE TURF_CARTILAGE TURF_SYNOVIUM ...

        Parameters
        ----------
        path : str | Path
            Path to TURF scores file

        Returns
        -------
        pd.DataFrame
            TURF scores with SNP identifier and per-tissue scores

        Raises
        ------
        FileNotFoundError
            If file not found
        ValueError
            If format invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"TURF scores file not found: {path}")

        try:
            df = pd.read_csv(path, sep="\t")

            if "SNP" not in df.columns:
                raise ValueError("TURF file must contain SNP column")

            self.turf_scores = df

            # Identify tissue columns (assumed to start with TURF_)
            tissue_cols = [col for col in df.columns if col.startswith("TURF_")]
            log.info(
                "turf_scores_loaded",
                n_snps=len(df),
                n_tissues=len(tissue_cols),
                tissues=tissue_cols,
            )

            return df

        except Exception as e:
            log.error("turf_scores_load_failed", path=str(path), error=str(e))
            raise

    def load_tissuarc_scores(self, path: str | Path) -> pd.DataFrame:
        """
        Load TissueARC (Tissue-specific Annotated Regulatory Circuit) scores.

        Parameters
        ----------
        path : str | Path
            Path to TissueARC scores

        Returns
        -------
        pd.DataFrame
            TissueARC scores

        Raises
        ------
        FileNotFoundError
            If file not found
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"TissueARC file not found: {path}")

        df = pd.read_csv(path, sep="\t")

        if "SNP" not in df.columns:
            raise ValueError("TissueARC file must contain SNP column")

        log.info("tissuarc_scores_loaded", n_snps=len(df))

        return df

    def prioritize_snps(
        self,
        snp_df: pd.DataFrame,
        tissue: str,
        threshold: float = 0.5,
        score_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter SNPs by tissue-specific annotation threshold.

        Parameters
        ----------
        snp_df : pd.DataFrame
            Input SNPs (must contain SNP column for merging)
        tissue : str
            Tissue name (e.g., "BONE", "CARTILAGE")
        threshold : float
            Score threshold for inclusion (0-1)
        score_col : Optional[str]
            Column name for score (auto-detected if None)

        Returns
        -------
        pd.DataFrame
            Filtered SNPs meeting annotation threshold

        Raises
        ------
        ValueError
            If tissue not available or SNPs not found
        """
        if self.turf_scores is None:
            raise ValueError("TURF scores not loaded. Call load_turf_scores() first.")

        if "SNP" not in snp_df.columns:
            raise ValueError("Input SNP dataframe must contain SNP column")

        # Auto-detect score column if not specified
        if score_col is None:
            score_col = f"TURF_{tissue.upper()}"
            if score_col not in self.turf_scores.columns:
                raise ValueError(
                    f"Tissue {tissue} not found. Available tissues: "
                    f"{[col.replace('TURF_', '') for col in self.turf_scores.columns if col.startswith('TURF_')]}"
                )

        # Merge with tissue scores
        merged = snp_df.merge(
            self.turf_scores[["SNP", score_col]],
            on="SNP",
            how="left",
        )

        # Filter by threshold
        filtered = merged[merged[score_col] >= threshold].copy()

        log.info(
            "snps_prioritized",
            tissue=tissue,
            threshold=threshold,
            n_input=len(snp_df),
            n_above_threshold=len(filtered),
            pct_retained=100 * len(filtered) / len(snp_df),
        )

        return filtered

    def get_tissue_ranks(
        self,
        snp_list: list[str],
        tissues: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get tissue-specific rankings for a set of SNPs.

        Parameters
        ----------
        snp_list : list[str]
            List of SNP IDs
        tissues : Optional[list[str]]
            Tissues to include (if None, use all available)

        Returns
        -------
        pd.DataFrame
            SNP × tissue ranking matrix

        Raises
        ------
        ValueError
            If SNPs not found
        """
        if self.turf_scores is None:
            raise ValueError("TURF scores not loaded")

        # Filter to requested SNPs
        snp_scores = self.turf_scores[
            self.turf_scores["SNP"].isin(snp_list)
        ].copy()

        if len(snp_scores) == 0:
            raise ValueError(f"No SNPs found in TURF scores")

        # Identify tissue columns
        if tissues is None:
            tissues = [
                col.replace("TURF_", "")
                for col in snp_scores.columns
                if col.startswith("TURF_")
            ]

        # Create ranking matrix (1 = highest score per tissue)
        ranking = snp_scores[["SNP"]].copy()

        for tissue in tissues:
            score_col = f"TURF_{tissue.upper()}"
            if score_col in snp_scores.columns:
                ranking[tissue] = snp_scores[score_col].rank(
                    method="min", ascending=False
                )

        log.info(
            "tissue_ranks_computed",
            n_snps=len(ranking),
            tissues=tissues,
        )

        return ranking

    def joint_annotation_score(
        self,
        snp_df: pd.DataFrame,
        tissues: list[str],
        weights: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute joint annotation score across multiple tissues.

        Parameters
        ----------
        snp_df : pd.DataFrame
            Input SNP dataframe
        tissues : list[str]
            Tissues to consider
        weights : Optional[dict[str, float]]
            Per-tissue weights (default equal)

        Returns
        -------
        pd.DataFrame
            Input dataframe with JOINT_ANNOTATION column added

        Raises
        ------
        ValueError
            If tissues not available
        """
        if self.turf_scores is None:
            raise ValueError("TURF scores not loaded")

        result = snp_df.copy()

        # Default equal weights
        if weights is None:
            weights = {tissue: 1 / len(tissues) for tissue in tissues}

        # Initialize joint score
        result["JOINT_ANNOTATION"] = 0.0

        # Add weighted tissue scores
        for tissue in tissues:
            score_col = f"TURF_{tissue.upper()}"

            if score_col not in self.turf_scores.columns:
                log.warning(
                    "tissue_not_available",
                    tissue=tissue,
                    available=list(self.turf_scores.columns),
                )
                continue

            # Merge tissue scores
            tissue_df = self.turf_scores[["SNP", score_col]]
            result = result.merge(tissue_df, on="SNP", how="left")

            # Add weighted score
            weight = weights.get(tissue, 0)
            result["JOINT_ANNOTATION"] += result[score_col].fillna(0) * weight

            # Drop intermediate column
            result = result.drop(score_col, axis=1)

        log.info(
            "joint_annotation_computed",
            n_snps=len(result),
            n_tissues=len(tissues),
        )

        return result
