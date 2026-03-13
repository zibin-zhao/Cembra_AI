"""
LDpred2 wrapper for LD-based polygenic risk score calculation.

Uses R bigsnpr package via subprocess (R script execution).

References:
    Privé, F., Arbel, J., & Vilhjálmsson, B. J. (2020).
    LDpred2: Better, faster, stronger.
    Bioinformatics, 36(22), 5424–5431.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class LDpred2Runner:
    """
    Wrapper for LDpred2 (LD-based PRS via bigsnpr).

    Executes LDpred2 via R scripts.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    r_script_template : str
        Template R script for LDpred2
    """

    def __init__(self, config: dict):
        """
        Initialize LDpred2 runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - n_iter: Number of iterations
            - p_init: Initial p-value threshold
            - h2_init: Initial heritability estimate
            - shrink_corr: Shrinkage correlation (0-1)
        """
        self.config = config
        self._validate_r_installation()

    def _validate_r_installation(self) -> None:
        """Check that R and bigsnpr are available."""
        try:
            result = subprocess.run(
                ["Rscript", "-e", "library(bigsnpr)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("bigsnpr package not available in R")
            log.info("ldpred2_r_validated")
        except FileNotFoundError:
            raise RuntimeError("Rscript not found. Please install R.")

    def _generate_r_script(
        self,
        gwas_file: Path,
        bim_prefix: Path,
        ld_ref_prefix: Path,
        out_dir: Path,
        method: str,
    ) -> str:
        """
        Generate R script for LDpred2 execution.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics file
        bim_prefix : Path
            PLINK .bim prefix
        ld_ref_prefix : Path
            LD reference prefix
        out_dir : Path
            Output directory
        method : str
            LDpred2 method (auto, grid, inf)

        Returns
        -------
        str
            R script content
        """
        # Use bigsnpr for LD computation
        r_script = f"""
suppressWarnings(library(bigsnpr))
suppressWarnings(library(data.table))

# Read GWAS summary stats
gwas_file <- "{gwas_file}"
gwas <- fread(gwas_file)
setnames(gwas, c("SNP", "A1", "A2", "BETA", "SE"),
         c("rsid", "a1", "a2", "beta", "se"))

# Read LD reference (assuming .rds format from bigsnpr)
ld_ref_file <- "{ld_ref_prefix}.rds"
corr <- readRDS(ld_ref_file)

# Read variant info
bim_file <- "{bim_prefix}.bim"
bim <- fread(bim_file, select=c(2, 1, 4, 5, 6))
setnames(bim, c("rsid", "chrom", "pos", "a1", "a2"))

# Run LDpred2-{method}
if ("{method}" == "auto") {{
    res <- ldpred2_auto(corr, gwas)
}} else if ("{method}" == "inf") {{
    res <- ldpred2_inf(corr, gwas)
}} else if ("{method}" == "grid") {{
    res <- ldpred2_grid(corr, gwas, auto_param = NULL)
}}

# Extract best weights
if ("{method}" == "auto") {{
    weights <- attr(res, "beta_auto")
}} else {{
    # Take maximum likelihood estimate
    best_idx <- which.max(res$score)
    weights <- res$beta[, best_idx]
}}

# Save output
out_file <- "{out_dir}/ldpred2_weights.txt"
dir.create("{out_dir}", showWarnings=FALSE, recursive=TRUE)

output_df <- data.frame(
    SNP = bim$rsid,
    CHR = bim$chrom,
    BP = bim$pos,
    A1 = bim$a1,
    A2 = bim$a2,
    WEIGHT = weights
)

fwrite(output_df, out_file, sep="\\t", quote=FALSE)
cat("Completed LDpred2 {method}\\n")
"""
        return r_script

    def run(
        self,
        gwas_file: str | Path,
        ld_ref: str | Path,
        bim_prefix: str | Path,
        out_dir: str | Path,
        method: Literal["auto", "inf", "grid"] = "auto",
    ) -> pd.DataFrame:
        """
        Run LDpred2 to generate polygenic weights.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics
        ld_ref : str | Path
            LD reference (prefix or .rds file)
        bim_prefix : str | Path
            PLINK .bim prefix
        out_dir : str | Path
            Output directory
        method : Literal["auto", "inf", "grid"]
            LDpred2 method to use

        Returns
        -------
        pd.DataFrame
            Weights dataframe

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If R execution fails
        """
        gwas_file = Path(gwas_file)
        ld_ref = Path(ld_ref)
        bim_prefix = Path(bim_prefix)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        if not gwas_file.exists():
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        bim_file = Path(str(bim_prefix) + ".bim")
        if not bim_file.exists():
            raise FileNotFoundError(f"BIM file not found: {bim_file}")

        if not ld_ref.exists():
            raise FileNotFoundError(f"LD reference not found: {ld_ref}")

        log.info(
            "ldpred2_starting",
            method=method,
            gwas_file=str(gwas_file),
            ld_ref=str(ld_ref),
        )

        # Generate and run R script
        r_script = self._generate_r_script(
            gwas_file, bim_prefix, ld_ref.with_suffix(""), out_dir, method
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
            log.info("ldpred2_completed", method=method, stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error(
                "ldpred2_failed",
                method=method,
                stderr=e.stderr[:500],
            )
            raise RuntimeError(f"LDpred2 R script failed: {e.stderr}") from e
        finally:
            Path(r_script_path).unlink(missing_ok=True)

        return self.parse_output(out_dir)

    def parse_output(self, out_dir: Path) -> pd.DataFrame:
        """
        Parse LDpred2 output weights.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()

        Returns
        -------
        pd.DataFrame
            Columns: SNP, CHR, BP, A1, A2, WEIGHT

        Raises
        ------
        FileNotFoundError
            If output file not found
        """
        weight_file = Path(out_dir) / "ldpred2_weights.txt"

        if not weight_file.exists():
            raise FileNotFoundError(f"Output file not found: {weight_file}")

        df = pd.read_csv(weight_file, sep="\t")

        expected_cols = {"SNP", "CHR", "BP", "A1", "A2", "WEIGHT"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"Output missing expected columns. Expected {expected_cols}, "
                f"got {set(df.columns)}"
            )

        log.info(
            "ldpred2_output_parsed",
            n_snps=len(df),
            n_nonzero=sum(df["WEIGHT"] != 0),
        )

        return df[["SNP", "CHR", "BP", "A1", "A2", "WEIGHT"]]
