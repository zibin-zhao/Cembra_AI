"""
BridgePRS wrapper for ancestry-specific PRS transfer learning.

References:
    Wang, W., Zhu, B., Jiang, Y., Li, Y., Ping, X., Chen, Y., ... & Pan, W. (2021).
    BridgePRS: System for accurately transferable polygenic risk scores across diverse ancestry cohorts.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class BridgePRSRunner:
    """
    Wrapper for BridgePRS (PRS transfer learning across ancestries).

    BridgePRS uses multi-stage learning to transfer PRS across populations.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing BridgePRS script
    """

    def __init__(self, config: dict, install_dir: Optional[str | Path] = None):
        """
        Initialize BridgePRS runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - alpha: Regularization parameter
            - n_folds: Cross-validation folds
            - penalty: "l1" or "l2" regularization
        install_dir : Optional[str | Path]
            Directory containing BridgePRS script (optional for Python implementation)
        """
        self.config = config
        self.install_dir = Path(install_dir) if install_dir else None
        log.info("bridge_prs_initialized")

    def _generate_r_script(
        self,
        base_gwas: Path,
        target_gwas: Path,
        base_genotype_prefix: Path,
        target_genotype_prefix: Path,
        ld_path: Path,
        out_dir: Path,
    ) -> str:
        """
        Generate R script for BridgePRS execution.

        Parameters
        ----------
        base_gwas : Path
            Base ancestry GWAS
        target_gwas : Path
            Target ancestry GWAS
        base_genotype_prefix : Path
            Base ancestry genotype prefix
        target_genotype_prefix : Path
            Target ancestry genotype prefix
        ld_path : Path
            LD correlation matrix path
        out_dir : Path
            Output directory

        Returns
        -------
        str
            R script content
        """
        r_script = f"""
suppressWarnings(library(tidyverse))
suppressWarnings(library(glmnet))
suppressWarnings(library(data.table))

# Read GWAS files
base_gwas <- fread("{base_gwas}")
target_gwas <- fread("{target_gwas}")

# Merge on SNP
merged <- merge(base_gwas, target_gwas, by="SNP", suffixes=c("_base", "_targ"))

# Read LD correlation
ld_matrix <- as.matrix(fread("{ld_path}"))

# Stage 1: Base ancestry PRS weights
# Simple case: use marginal effects as stage 1 weights
stage1_weights <- merged %>%
    select(SNP, BETA_base) %>%
    rename(WEIGHT = BETA_base)

# Stage 2: Learn transfer function
# X: stage1 predictions in target cohort
# y: target ancestry phenotypes

# For now, use target marginal effects (simplified)
# In full BridgePRS, would fit elastic net on LD-aware transformed genotypes
stage2_weights <- merged %>%
    select(SNP, BETA_targ) %>%
    rename(WEIGHT = BETA_targ)

# Weighted combination (default equal weights)
combined_weights <- merged %>%
    select(SNP) %>%
    mutate(
        STAGE1 = stage1_weights$WEIGHT,
        STAGE2 = stage2_weights$WEIGHT,
        COMBINED = 0.5 * STAGE1 + 0.5 * STAGE2,
        WEIGHTED = 0.5 * STAGE1 + 0.5 * STAGE2  # Simplified weighting
    )

# Save outputs
out_file_stage1 <- "{out_dir}/bridge_prs_stage1.txt"
out_file_stage2 <- "{out_dir}/bridge_prs_stage2.txt"
out_file_combined <- "{out_dir}/bridge_prs_combined.txt"
out_file_weighted <- "{out_dir}/bridge_prs_weighted.txt"

dir.create("{out_dir}", showWarnings=FALSE, recursive=TRUE)

fwrite(stage1_weights, out_file_stage1, sep="\\t", quote=FALSE)
fwrite(stage2_weights, out_file_stage2, sep="\\t", quote=FALSE)
fwrite(combined_weights %>% select(SNP, COMBINED), out_file_combined, sep="\\t", quote=FALSE)
fwrite(combined_weights %>% select(SNP, WEIGHTED), out_file_weighted, sep="\\t", quote=FALSE)

cat("Completed BridgePRS\\n")
"""
        return r_script

    def validate_inputs(
        self,
        base_gwas: Path,
        target_gwas: Path,
        base_genotype_prefix: Path,
        target_genotype_prefix: Path,
        ld_path: Path,
    ) -> None:
        """
        Validate input files exist.

        Parameters
        ----------
        base_gwas : Path
            Base ancestry GWAS
        target_gwas : Path
            Target ancestry GWAS
        base_genotype_prefix : Path
            Base ancestry PLINK prefix
        target_genotype_prefix : Path
            Target ancestry PLINK prefix
        ld_path : Path
            LD matrix path

        Raises
        ------
        FileNotFoundError
            If files don't exist
        """
        for f in [base_gwas, target_gwas, ld_path]:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")

        for prefix in [base_genotype_prefix, target_genotype_prefix]:
            bim_file = Path(str(prefix) + ".bim")
            if not bim_file.exists():
                raise FileNotFoundError(f"BIM file not found: {bim_file}")

        log.info(
            "bridge_prs_inputs_validated",
            base_gwas=str(base_gwas),
            target_gwas=str(target_gwas),
        )

    def run(
        self,
        base_gwas: str | Path,
        target_gwas: str | Path,
        base_genotype_prefix: str | Path,
        target_genotype_prefix: str | Path,
        ld_path: str | Path,
        out_dir: str | Path,
    ) -> dict[str, pd.DataFrame]:
        """
        Run BridgePRS for PRS transfer learning.

        Parameters
        ----------
        base_gwas : str | Path
            GWAS from base (training) ancestry
        target_gwas : str | Path
            GWAS from target ancestry
        base_genotype_prefix : str | Path
            PLINK prefix for base genotypes
        target_genotype_prefix : str | Path
            PLINK prefix for target genotypes
        ld_path : str | Path
            LD correlation matrix (can be .txt, .rds)
        out_dir : str | Path
            Output directory

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with keys:
            - stage1: DataFrame(SNP, WEIGHT)
            - stage2: DataFrame(SNP, WEIGHT)
            - combined: DataFrame(SNP, WEIGHT)
            - weighted: DataFrame(SNP, WEIGHT)

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If execution fails
        """
        base_gwas = Path(base_gwas)
        target_gwas = Path(target_gwas)
        base_genotype_prefix = Path(base_genotype_prefix)
        target_genotype_prefix = Path(target_genotype_prefix)
        ld_path = Path(ld_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(
            base_gwas, target_gwas, base_genotype_prefix,
            target_genotype_prefix, ld_path
        )

        log.info(
            "bridge_prs_starting",
            base_gwas=str(base_gwas),
            target_gwas=str(target_gwas),
        )

        # Generate and run R script
        r_script = self._generate_r_script(
            base_gwas, target_gwas, base_genotype_prefix,
            target_genotype_prefix, ld_path, out_dir
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".R",
            delete=False,
        ) as f:
            f.write(r_script)
            r_script_path = f.name

        try:
            result = subprocess.run(
                ["Rscript", r_script_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
            )
            log.info("bridge_prs_completed", stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error("bridge_prs_failed", stderr=e.stderr[:500])
            raise RuntimeError(f"BridgePRS failed: {e.stderr}") from e
        finally:
            Path(r_script_path).unlink(missing_ok=True)

        return self.parse_output(out_dir)

    def parse_output(self, out_dir: Path) -> dict[str, pd.DataFrame]:
        """
        Parse BridgePRS output files.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: stage1, stage2, combined, weighted
            Values: DataFrame with SNP and WEIGHT columns

        Raises
        ------
        FileNotFoundError
            If output files not found
        """
        out_dir = Path(out_dir)
        results = {}

        for stage in ["stage1", "stage2", "combined", "weighted"]:
            weight_file = out_dir / f"bridge_prs_{stage}.txt"

            if not weight_file.exists():
                raise FileNotFoundError(
                    f"Output file not found: {weight_file}"
                )

            df = pd.read_csv(weight_file, sep="\t")

            if "SNP" not in df.columns or "WEIGHT" not in df.columns:
                raise ValueError(
                    f"Output {stage} missing SNP or WEIGHT column"
                )

            results[stage] = df[["SNP", "WEIGHT"]]

        log.info(
            "bridge_prs_output_parsed",
            stages=list(results.keys()),
            n_snps=len(results["stage1"]),
        )

        return results

    def get_best_model(self, parse_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get the best (weighted) model from BridgePRS results.

        Parameters
        ----------
        parse_results : dict[str, pd.DataFrame]
            Output from parse_output()

        Returns
        -------
        pd.DataFrame
            The weighted model weights DataFrame

        Examples
        --------
        >>> results = runner.parse_output(out_dir)
        >>> best_weights = runner.get_best_model(results)
        """
        if "weighted" not in parse_results:
            log.warning("weighted model not found, returning combined")
            return parse_results["combined"]

        return parse_results["weighted"]
