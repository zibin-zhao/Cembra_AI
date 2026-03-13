"""
PolyFun wrapper for functional fine-mapping and SNP heritability estimation.

PolyFun estimates per-SNP heritability using GWAS summary statistics
and functional annotations.

References:
    Finucane, H. K., Bulik-Sullivan, B., Gusev, A., Trynka, G., Reshef, Y.,
    Loh, P.-R., ... & Price, A. L. (2015). Partitioning heritability by
    functional annotation using genome-wide association summary statistics.
    Nature Genetics, 47(11), 1228-1235.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class PolyFunRunner:
    """
    Wrapper for PolyFun (Polygenic Functionality) for SNP heritability estimation.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing PolyFun scripts
    """

    def __init__(self, config: dict, install_dir: Optional[str | Path] = None):
        """
        Initialize PolyFun runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - method: "polyfun" or "polyfun_inf"
            - n_random_snps: Number of random SNPs for estimation
        install_dir : Optional[str | Path]
            Directory containing PolyFun script
        """
        self.config = config
        self.install_dir = Path(install_dir) if install_dir else None
        log.info("polyfun_initialized")

    def _generate_r_script(
        self,
        gwas_file: Path,
        annotations_file: Path,
        out_dir: Path,
    ) -> str:
        """
        Generate R script for PolyFun execution.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics
        annotations_file : Path
            Functional annotations
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
setnames(gwas, c("SNP", "CHR", "BP", "A1", "A2", "BETA", "SE"),
         c("SNP", "CHR", "BP", "A1", "A2", "BETA", "SE"))

# Read annotations
annot <- fread("{annotations_file}")

# Merge
merged <- merge(gwas, annot, by="SNP", all.x=TRUE)

# Compute per-SNP heritability (simplified PolyFun)
# In full PolyFun, this uses regression of LD score on annotations
# Here we use a simplified approach: functional annotation * effect size
merged <- merged %>%
    mutate(
        # Basic per-SNP heritability: effect size squared
        SNP_HERIT = BETA^2 / nrow(gwas),
        # Annotation-weighted heritability
        SNP_HERIT_ANNOT = SNP_HERIT * (1 + rowMeans(annot[,-1]))
    )

# Normalize
merged <- merged %>%
    mutate(
        SNPVAR = SNP_HERIT_ANNOT / sum(SNP_HERIT_ANNOT, na.rm=TRUE)
    )

# Save output
out_file <- "{out_dir}/polyfun_snpvar.txt"
dir.create("{out_dir}", showWarnings=FALSE, recursive=TRUE)

output_df <- merged %>%
    select(SNP, CHR, BP, A1, A2, BETA, SE, SNPVAR)

fwrite(output_df, out_file, sep="\\t", quote=FALSE)

cat("Completed PolyFun\\n")
"""
        return r_script

    def validate_inputs(
        self,
        gwas_file: Path,
        annotations_file: Path,
    ) -> None:
        """
        Validate input files.

        Parameters
        ----------
        gwas_file : Path
            GWAS file
        annotations_file : Path
            Annotations file

        Raises
        ------
        FileNotFoundError
            If files don't exist
        """
        if not gwas_file.exists():
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        if not annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_file}"
            )

        log.info(
            "polyfun_inputs_validated",
            gwas_file=str(gwas_file),
            annotations_file=str(annotations_file),
        )

    def run(
        self,
        gwas_file: str | Path,
        annotations: str | Path,
        out_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Run PolyFun to estimate per-SNP heritability.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics
        annotations : str | Path
            Functional annotations file
        out_dir : str | Path
            Output directory

        Returns
        -------
        pd.DataFrame
            SNP-level heritability with columns: SNP, CHR, BP, A1, A2, SNPVAR

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If execution fails
        """
        gwas_file = Path(gwas_file)
        annotations = Path(annotations)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_file, annotations)

        log.info(
            "polyfun_starting",
            gwas_file=str(gwas_file),
            annotations=str(annotations),
        )

        # Generate and run R script
        r_script = self._generate_r_script(gwas_file, annotations, out_dir)

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
            log.info("polyfun_completed", stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error("polyfun_failed", stderr=e.stderr[:500])
            raise RuntimeError(f"PolyFun failed: {e.stderr}") from e
        finally:
            Path(r_script_path).unlink(missing_ok=True)

        return self.parse_snpvar(out_dir / "polyfun_snpvar.txt")

    def parse_snpvar(self, path: str | Path) -> pd.DataFrame:
        """
        Parse PolyFun SNP heritability output.

        Parameters
        ----------
        path : str | Path
            Path to snpvar output file

        Returns
        -------
        pd.DataFrame
            Columns: SNP, CHR, BP, A1, A2, SNPVAR

        Raises
        ------
        FileNotFoundError
            If file not found
        ValueError
            If required columns missing
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {path}")

        df = pd.read_csv(path, sep="\t")

        required_cols = {"SNP", "SNPVAR"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Missing columns. Required: {required_cols}, "
                f"got {set(df.columns)}"
            )

        log.info(
            "polyfun_snpvar_parsed",
            n_snps=len(df),
            mean_snpvar=df["SNPVAR"].mean(),
        )

        return df
