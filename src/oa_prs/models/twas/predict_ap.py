"""
PredictAP: EAS-Specific Expression Prediction.

Runs PredictDB-style TWAS using EAS-trained expression models,
which are more appropriate than EUR-trained models for Hong Kong Chinese targets.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictAPResults:
    """Container for PredictAP results."""

    gene_scores: pd.DataFrame  # gene, z_score, p_value, tissue
    n_genes_tested: int = 0
    n_significant: int = 0
    tissues_tested: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "n_genes_tested": self.n_genes_tested,
            "n_significant": self.n_significant,
            "n_tissues": len(self.tissues_tested),
        }


class PredictAPRunner:
    """
    Runner for EAS-specific transcriptomic prediction.

    PredictAP uses expression weights trained on EAS populations
    (e.g., from MESA or CARTaGENE EAS panels) rather than GTEx EUR models.
    This improves TWAS accuracy for cross-ancestry transfer to HK Chinese.

    Args:
        gwas_file: Path to harmonized GWAS summary statistics.
        model_dir: Directory containing EAS expression prediction models.
        output_dir: Output directory for results.
        n_threads: Number of parallel threads.
    """

    def __init__(
        self,
        gwas_file: str | Path,
        model_dir: str | Path = "data/external/predict_ap_models",
        output_dir: str | Path = "outputs/twas/predict_ap",
        n_threads: int = 4,
    ):
        self.gwas_file = Path(gwas_file)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.n_threads = n_threads
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tissues: list[str] | None = None,
        p_threshold: float = 5e-6,
    ) -> PredictAPResults:
        """
        Run PredictAP analysis.

        Args:
            tissues: List of tissues to analyze. If None, uses OA-relevant defaults.
            p_threshold: Significance threshold for gene selection.

        Returns:
            PredictAPResults with gene-level association scores.
        """
        if tissues is None:
            tissues = [
                "Muscle_Skeletal",
                "Adipose_Subcutaneous",
                "Whole_Blood",
                "Cells_Cultured_fibroblasts",
            ]

        all_results = []

        for tissue in tissues:
            logger.info("PredictAP: running tissue %s", tissue)
            model_db = self.model_dir / f"eas_{tissue}.db"

            if not model_db.exists():
                logger.warning("EAS model not found for %s, using EUR fallback.", tissue)
                model_db = self.model_dir / f"gtex_v8_{tissue}.db"
                if not model_db.exists():
                    logger.warning("No model available for %s, skipping.", tissue)
                    continue

            output_file = self.output_dir / f"predict_ap_{tissue}.tsv"

            try:
                result_df = self._run_spredixcan_with_eas_models(
                    model_db=model_db,
                    tissue=tissue,
                    output_file=output_file,
                )
                if result_df is not None:
                    result_df["tissue"] = tissue
                    all_results.append(result_df)
            except Exception as e:
                logger.error("PredictAP failed for %s: %s", tissue, e)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["gene", "z_score", "p_value", "tissue"])

        n_sig = int((combined["p_value"] < p_threshold).sum()) if len(combined) > 0 else 0

        logger.info(
            "PredictAP complete: %d genes tested, %d significant (p < %.1e) across %d tissues",
            len(combined), n_sig, p_threshold, len(tissues),
        )

        return PredictAPResults(
            gene_scores=combined,
            n_genes_tested=len(combined),
            n_significant=n_sig,
            tissues_tested=tissues,
        )

    def _run_spredixcan_with_eas_models(
        self,
        model_db: Path,
        tissue: str,
        output_file: Path,
    ) -> pd.DataFrame | None:
        """
        Run S-PrediXcan using EAS expression models.

        Uses the SPrediXcan.py script with EAS-trained prediction models
        instead of default EUR GTEx models.
        """
        cmd = [
            "python", "-m", "metaxcan.SPrediXcan",
            "--gwas_file", str(self.gwas_file),
            "--snp_column", "SNP",
            "--effect_allele_column", "A1",
            "--non_effect_allele_column", "A2",
            "--beta_column", "BETA",
            "--pvalue_column", "P",
            "--model_db_path", str(model_db),
            "--covariance", str(model_db.with_suffix(".txt.gz")),
            "--output_file", str(output_file),
            "--throw",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if output_file.exists():
                df = pd.read_csv(output_file, sep="\t")
                df = df.rename(columns={
                    "gene_name": "gene",
                    "zscore": "z_score",
                    "pvalue": "p_value",
                })
                return df[["gene", "z_score", "p_value"]].dropna()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("S-PrediXcan with EAS model failed: %s", e)

        return None

    def merge_with_eur_twas(
        self,
        eur_twas_file: str | Path,
        method: str = "fisher",
    ) -> pd.DataFrame:
        """
        Combine EAS and EUR TWAS results for meta-analysis.

        Args:
            eur_twas_file: Path to EUR S-PrediXcan results.
            method: Meta-analysis method ("fisher", "stouffer", "min_p").

        Returns:
            DataFrame with combined gene-level p-values.
        """
        eur = pd.read_csv(eur_twas_file, sep="\t")
        eas_file = self.output_dir / "predict_ap_combined.tsv"

        if not eas_file.exists():
            logger.warning("No EAS TWAS results found for merging.")
            return eur

        eas = pd.read_csv(eas_file, sep="\t")

        # Merge on gene name
        merged = eur.merge(eas, on="gene", suffixes=("_eur", "_eas"), how="outer")

        if method == "fisher":
            from scipy import stats
            merged["combined_p"] = merged.apply(
                lambda row: stats.combine_pvalues(
                    [p for p in [row.get("p_value_eur"), row.get("p_value_eas")] if pd.notna(p)],
                    method="fisher",
                )[1] if pd.notna(row.get("p_value_eur")) or pd.notna(row.get("p_value_eas")) else np.nan,
                axis=1,
            )
        elif method == "min_p":
            merged["combined_p"] = merged[["p_value_eur", "p_value_eas"]].min(axis=1)

        output = self.output_dir / "eur_eas_meta_twas.tsv"
        merged.to_csv(output, sep="\t", index=False)
        logger.info("EUR+EAS meta-TWAS: %d genes, saved to %s", len(merged), output)

        return merged
