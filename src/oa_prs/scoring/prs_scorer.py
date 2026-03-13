"""
Classical PRS Individual Scorer.

Computes polygenic risk scores for individuals given genotype data and
SNP weights from any PRS method.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PRSScorer:
    """
    Compute individual-level Polygenic Risk Scores.

    PRS = Σ(weight_i × dosage_i) for all SNPs i

    Supports:
        - PLINK .bed/.bim/.fam genotype files (via pandas-plink)
        - Pre-loaded dosage matrices (numpy arrays)
        - Multiple weight sets for comparison

    Args:
        weights: DataFrame with columns [SNP, A1, BETA] at minimum.
                 A1 is the effect allele whose dosage is counted.
    """

    def __init__(self, weights: pd.DataFrame) -> None:
        required = {"SNP", "A1", "BETA"}
        missing = required - set(weights.columns)
        if missing:
            raise ValueError(f"Weights DataFrame missing columns: {missing}")

        self.weights = weights.copy()
        self.weights = self.weights.dropna(subset=["BETA"])
        logger.info("PRSScorer initialized with %d SNP weights", len(self.weights))

    @classmethod
    def from_file(cls, path: str | Path, beta_col: str = "BETA") -> PRSScorer:
        """
        Load weights from a TSV/CSV file.

        Expected columns: SNP, A1 (effect allele), and a beta/weight column.
        Additional columns (CHR, BP, A2) are preserved but not required.
        """
        path = Path(path)
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep)
        if beta_col != "BETA" and beta_col in df.columns:
            df = df.rename(columns={beta_col: "BETA"})
        return cls(df)

    def score_plink(
        self,
        bed_prefix: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Score individuals from PLINK binary files (.bed/.bim/.fam).

        Args:
            bed_prefix: Path prefix for .bed/.bim/.fam files.
            output_path: Optional path to save scores.

        Returns:
            DataFrame with columns [FID, IID, PRS, N_SNP].
        """
        try:
            from pandas_plink import read_plink1_bin
        except ImportError as exc:
            raise ImportError(
                "pandas-plink required for PLINK scoring. "
                "Install: pip install pandas-plink"
            ) from exc

        bed_prefix = str(bed_prefix)
        logger.info("Reading PLINK files: %s", bed_prefix)

        # Read genotype data
        G = read_plink1_bin(
            f"{bed_prefix}.bed",
            f"{bed_prefix}.bim",
            f"{bed_prefix}.fam",
            verbose=False,
        )

        # Extract sample and variant info
        fam = pd.DataFrame({
            "FID": G.fid.values,
            "IID": G.iid.values,
        })
        bim_snps = G.variant.snp.values
        bim_a1 = G.variant.a0.values  # Note: pandas-plink a0 = reference allele

        # Match weights to genotype SNPs
        weight_dict = dict(zip(self.weights["SNP"], self.weights["BETA"]))
        effect_allele_dict = dict(zip(self.weights["SNP"], self.weights["A1"]))

        matched_indices = []
        matched_betas = []
        for idx, snp in enumerate(bim_snps):
            if snp in weight_dict:
                matched_indices.append(idx)
                beta = weight_dict[snp]
                # Check if alleles need flipping
                if snp in effect_allele_dict and effect_allele_dict[snp] != bim_a1[idx]:
                    beta = -beta  # Flip direction
                matched_betas.append(beta)

        n_matched = len(matched_indices)
        logger.info(
            "Matched %d / %d weight SNPs to genotype data (%.1f%%)",
            n_matched, len(self.weights), 100 * n_matched / max(len(self.weights), 1),
        )

        if n_matched == 0:
            logger.warning("No SNPs matched! Check SNP ID format and allele coding.")
            fam["PRS"] = 0.0
            fam["N_SNP"] = 0
            return fam

        # Extract dosage matrix for matched SNPs and compute PRS
        dosage = G.values[:, matched_indices]  # [n_individuals, n_matched_snps]
        betas = np.array(matched_betas)

        # Handle missing genotypes (NaN → 0)
        dosage_clean = np.nan_to_num(dosage, nan=0.0)

        # PRS = dosage × beta
        prs = dosage_clean @ betas

        fam["PRS"] = prs
        fam["N_SNP"] = n_matched

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fam.to_csv(output_path, sep="\t", index=False)
            logger.info("Saved PRS scores to %s", output_path)

        return fam

    def score_dosage(
        self,
        dosage_matrix: np.ndarray,
        snp_ids: list[str],
        sample_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Score from a pre-loaded dosage matrix.

        Args:
            dosage_matrix: [n_samples, n_snps] dosage values (0/1/2 or continuous).
            snp_ids: SNP identifiers corresponding to columns.
            sample_ids: Optional sample identifiers.

        Returns:
            DataFrame with columns [SAMPLE_ID, PRS, N_SNP].
        """
        snp_to_idx = {s: i for i, s in enumerate(snp_ids)}

        matched_cols = []
        matched_betas = []
        for _, row in self.weights.iterrows():
            if row["SNP"] in snp_to_idx:
                matched_cols.append(snp_to_idx[row["SNP"]])
                matched_betas.append(row["BETA"])

        betas = np.array(matched_betas)
        dosage_sub = np.nan_to_num(dosage_matrix[:, matched_cols], nan=0.0)
        prs = dosage_sub @ betas

        if sample_ids is None:
            sample_ids = [f"SAMPLE_{i}" for i in range(len(prs))]

        return pd.DataFrame({
            "SAMPLE_ID": sample_ids,
            "PRS": prs,
            "N_SNP": len(matched_betas),
        })

    def standardize_scores(self, scores: pd.DataFrame, prs_col: str = "PRS") -> pd.DataFrame:
        """Z-score standardize PRS within the scored cohort."""
        result = scores.copy()
        mean = result[prs_col].mean()
        std = result[prs_col].std()
        result[f"{prs_col}_Z"] = (result[prs_col] - mean) / (std + 1e-10)
        return result
