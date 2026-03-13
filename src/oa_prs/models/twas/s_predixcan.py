"""
S-PrediXcan wrapper for transcriptome-wide association analysis (TWAS).

S-PrediXcan tests associations between genetically predicted gene expression
and phenotypes using imputed transcriptome data.

References:
    Barbeira, A. N., Dickinson, S. P., Bonazzola, R., Zhong, J., Wheeler, H. E.,
    Torres, J. M., ... & Im, H. K. (2019). Exploring the phenotypic consequences
    of tissue dependent and shared regulation of gene expression.
    Nature Communications, 9(1), 4747.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class SPrediXcanRunner:
    """
    Wrapper for S-PrediXcan (Summary-PrediXcan) TWAS.

    Computes gene-phenotype associations using predicted expression.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing S-PrediXcan scripts
    """

    def __init__(self, config: dict, install_dir: Optional[str | Path] = None):
        """
        Initialize S-PrediXcan runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - model_db_dir: Directory with prediction models
            - covariance_dir: Directory with LD covariance
        install_dir : Optional[str | Path]
            Directory containing PrediXcan scripts
        """
        self.config = config
        self.install_dir = Path(install_dir) if install_dir else None
        log.info("s_predixcan_initialized")

    def _generate_r_script(
        self,
        gwas_file: Path,
        model_db: Path,
        covariance: Path,
        tissue: str,
        out_dir: Path,
    ) -> str:
        """
        Generate R script for S-PrediXcan execution.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics
        model_db : Path
            Gene expression prediction model database
        covariance : Path
            LD covariance matrix
        tissue : str
            Tissue name
        out_dir : Path
            Output directory

        Returns
        -------
        str
            R script content
        """
        r_script = f"""
suppressWarnings(library(data.table))
suppressWarnings(library(tidyverse))

# Read GWAS summary stats
gwas <- fread("{gwas_file}")
setnames(gwas, c("SNP", "A1", "A2", "BETA", "SE", "P"),
         c("SNP", "A1", "A2", "BETA", "SE", "P"))

# Read model database
model <- readRDS("{model_db}")

# Read covariance matrix
cov <- as.matrix(fread("{covariance}"))

# Merge GWAS with model SNPs
merged <- gwas %>%
    filter(SNP %in% model$SNPs) %>%
    arrange(SNP)

# Compute S-PrediXcan association
# Score = sum(weight_i * z_i) / sqrt(sum(weights)^2 + 2*sum(cov))
weights <- model$weights[match(merged$SNP, model$SNPs)]
z_scores <- merged$BETA / merged$SE

numerator <- sum(weights * z_scores, na.rm=TRUE)
variance <- sum((weights^2) * 1, na.rm=TRUE) + 2 * sum(cov[upper.tri(cov)])

zscore <- numerator / sqrt(variance)
pvalue <- 2 * pnorm(-abs(zscore))

# Output
results <- data.frame(
    GENE = unique(model$gene_id),
    N_SNP = length(weights),
    ZSCORE = zscore,
    PVALUE = pvalue,
    EFFECT_SIZE = coef(lm(merged$BETA ~ weights))[[2]]
)

out_file <- "{out_dir}/s_predixcan_{tissue}.txt"
dir.create("{out_dir}", showWarnings=FALSE, recursive=TRUE)

fwrite(results, out_file, sep="\\t", quote=FALSE)

cat("Completed S-PrediXcan\\n")
"""
        return r_script

    def validate_inputs(
        self,
        gwas_file: Path,
        model_db: Path,
        covariance: Path,
    ) -> None:
        """
        Validate input files.

        Parameters
        ----------
        gwas_file : Path
            GWAS file
        model_db : Path
            Model database
        covariance : Path
            Covariance matrix

        Raises
        ------
        FileNotFoundError
            If files don't exist
        """
        for f in [gwas_file, model_db, covariance]:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")

        log.info(
            "s_predixcan_inputs_validated",
            gwas_file=str(gwas_file),
            model_db=str(model_db),
        )

    def run(
        self,
        gwas_file: str | Path,
        model_db: str | Path,
        covariance: str | Path,
        tissue: str,
        out_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Run S-PrediXcan TWAS.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics
        model_db : str | Path
            Expression prediction model database
        covariance : str | Path
            LD covariance matrix
        tissue : str
            Tissue name
        out_dir : str | Path
            Output directory

        Returns
        -------
        pd.DataFrame
            Results with columns: GENE, N_SNP, ZSCORE, PVALUE, EFFECT_SIZE

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If execution fails
        """
        gwas_file = Path(gwas_file)
        model_db = Path(model_db)
        covariance = Path(covariance)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_file, model_db, covariance)

        log.info(
            "s_predixcan_starting",
            tissue=tissue,
            gwas_file=str(gwas_file),
        )

        # Generate and run R script
        r_script = self._generate_r_script(
            gwas_file, model_db, covariance, tissue, out_dir
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
            log.info("s_predixcan_completed", tissue=tissue, stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error("s_predixcan_failed", tissue=tissue, stderr=e.stderr[:500])
            raise RuntimeError(f"S-PrediXcan failed: {e.stderr}") from e
        finally:
            Path(r_script_path).unlink(missing_ok=True)

        return self.parse_output(out_dir, tissue)

    def parse_output(self, out_dir: Path, tissue: str) -> pd.DataFrame:
        """
        Parse S-PrediXcan output.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()
        tissue : str
            Tissue name

        Returns
        -------
        pd.DataFrame
            TWAS results

        Raises
        ------
        FileNotFoundError
            If output file not found
        """
        out_dir = Path(out_dir)
        result_file = out_dir / f"s_predixcan_{tissue}.txt"

        if not result_file.exists():
            raise FileNotFoundError(f"Output file not found: {result_file}")

        df = pd.read_csv(result_file, sep="\t")

        required_cols = {"GENE", "ZSCORE", "PVALUE"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Missing columns. Required: {required_cols}, "
                f"got {set(df.columns)}"
            )

        # Compute FDR
        from scipy.stats import norm

        df["PVALUE"] = 2 * norm.sf(np.abs(df["ZSCORE"]))

        log.info(
            "s_predixcan_output_parsed",
            tissue=tissue,
            n_genes=len(df),
            n_significant=sum(df["PVALUE"] < 0.05),
        )

        return df
