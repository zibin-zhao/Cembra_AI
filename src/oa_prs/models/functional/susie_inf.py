"""
SuSiE (Sum of Single Effects) wrapper for fine-mapping and credible sets.

SuSiE performs variable selection in high-dimensional settings to
identify likely causal variants.

References:
    Wang, G., Sarkar, A., Carbonetto, P., & Stephens, M. (2020).
    A simple new approach to variable selection in regression, with application
    to genetic fine mapping. Journal of the Royal Statistical Society, 82(5), 1273-1300.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class SuSiEInfRunner:
    """
    Wrapper for SuSiE fine-mapping for credible set identification.

    Identifies likely causal variants and generates posterior inclusion probabilities.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    """

    def __init__(self, config: dict):
        """
        Initialize SuSiE runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - L: Maximum number of causal SNPs
            - prior_weights: Prior on effect sizes
            - min_abs_corr: Minimum correlation for independent effects
        """
        self.config = config
        log.info("susie_inf_initialized")

    def _generate_r_script(
        self,
        gwas_file: Path,
        ld_matrix_file: Path,
        out_dir: Path,
    ) -> str:
        """
        Generate R script for SuSiE fine-mapping.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics
        ld_matrix_file : Path
            LD correlation matrix
        out_dir : Path
            Output directory

        Returns
        -------
        str
            R script content
        """
        L = self.config.get("L", 10)

        r_script = f"""
suppressWarnings(library(susieR))
suppressWarnings(library(data.table))
suppressWarnings(library(tidyverse))

# Read GWAS summary stats
gwas <- fread("{gwas_file}")
setnames(gwas, c("SNP", "CHR", "BP", "A1", "A2", "BETA", "SE"),
         c("SNP", "CHR", "BP", "A1", "A2", "BETA", "SE"))

# Read LD matrix
ld_matrix <- as.matrix(fread("{ld_matrix_file}"))

# Compute z-scores
z <- gwas$BETA / gwas$SE

# Run SuSiE
fit <- susie_suff_stat(
    z = z,
    R = ld_matrix,
    n = mean(1 / gwas$SE^2, na.rm=TRUE),  # Approximate sample size
    L = {L},
    estimate_residual_variance = TRUE,
    verbose = TRUE
)

# Extract results
pip <- data.frame(
    SNP = gwas$SNP,
    CHR = gwas$CHR,
    BP = gwas$BP,
    A1 = gwas$A1,
    A2 = gwas$A2,
    PIP = fit$pip,
    POSTERIOR_BETA = fit$posterior_mean,
    POSTERIOR_SD = fit$posterior_sd
)

# Assign to credible sets
credible_set_id <- rep(NA, nrow(pip))
for (i in seq_len(fit$n_effects)) {{
    if (!is.null(fit$sets$cs[[i]])) {{
        credible_set_id[fit$sets$cs[[i]]] <- i
    }}
}}
pip$CREDIBLE_SET_ID <- credible_set_id

# Save outputs
out_file <- "{out_dir}/susie_inf_results.txt"
dir.create("{out_dir}", showWarnings=FALSE, recursive=TRUE)

fwrite(pip, out_file, sep="\\t", quote=FALSE)

cat("Completed SuSiE fine-mapping\\n")
"""
        return r_script

    def validate_inputs(
        self,
        gwas_file: Path,
        ld_matrix_file: Path,
    ) -> None:
        """
        Validate input files.

        Parameters
        ----------
        gwas_file : Path
            GWAS file
        ld_matrix_file : Path
            LD matrix file

        Raises
        ------
        FileNotFoundError
            If files don't exist
        """
        if not gwas_file.exists():
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        if not ld_matrix_file.exists():
            raise FileNotFoundError(
                f"LD matrix not found: {ld_matrix_file}"
            )

        log.info(
            "susie_inf_inputs_validated",
            gwas_file=str(gwas_file),
            ld_matrix=str(ld_matrix_file),
        )

    def run(
        self,
        gwas_file: str | Path,
        ld_matrix: str | Path,
        priors: Optional[dict] = None,
        out_dir: Optional[str | Path] = None,
        loci: Optional[list[dict]] = None,
    ) -> pd.DataFrame:
        """
        Run SuSiE fine-mapping.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics
        ld_matrix : str | Path
            LD correlation matrix
        priors : Optional[dict]
            Prior configuration (optional)
        out_dir : Optional[str | Path]
            Output directory
        loci : Optional[list[dict]]
            Locus-specific configurations (optional)

        Returns
        -------
        pd.DataFrame
            Fine-mapping results with columns:
            SNP, CHR, BP, A1, A2, PIP, POSTERIOR_BETA, POSTERIOR_SD, CREDIBLE_SET_ID

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If execution fails
        """
        gwas_file = Path(gwas_file)
        ld_matrix = Path(ld_matrix)
        if out_dir is None:
            out_dir = Path.cwd() / "susie_output"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_file, ld_matrix)

        log.info(
            "susie_inf_starting",
            gwas_file=str(gwas_file),
            ld_matrix=str(ld_matrix),
        )

        # Generate and run R script
        r_script = self._generate_r_script(gwas_file, ld_matrix, out_dir)

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
            log.info("susie_inf_completed", stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error("susie_inf_failed", stderr=e.stderr[:500])
            raise RuntimeError(f"SuSiE fine-mapping failed: {e.stderr}") from e
        finally:
            Path(r_script_path).unlink(missing_ok=True)

        return self.parse_output(out_dir)

    def parse_output(self, out_dir: Path) -> pd.DataFrame:
        """
        Parse SuSiE fine-mapping output.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()

        Returns
        -------
        pd.DataFrame
            Columns: SNP, CHR, BP, A1, A2, PIP, POSTERIOR_BETA,
                    POSTERIOR_SD, CREDIBLE_SET_ID

        Raises
        ------
        FileNotFoundError
            If output file not found
        """
        out_dir = Path(out_dir)
        result_file = out_dir / "susie_inf_results.txt"

        if not result_file.exists():
            raise FileNotFoundError(f"Output file not found: {result_file}")

        df = pd.read_csv(result_file, sep="\t")

        required_cols = {"SNP", "PIP", "POSTERIOR_BETA"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Missing columns. Required: {required_cols}, "
                f"got {set(df.columns)}"
            )

        n_credible_sets = df["CREDIBLE_SET_ID"].notna().sum()
        log.info(
            "susie_inf_output_parsed",
            n_snps=len(df),
            n_with_pip_gt_0_1=sum(df["PIP"] > 0.1),
            n_credible_sets=n_credible_sets,
        )

        return df
