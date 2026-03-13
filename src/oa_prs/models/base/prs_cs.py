"""
PRS-CS wrapper for single-ancestry polygenic risk score calculation.

References:
    Ge, T., Chen, C.-Y., Feng, Q., Hibar, D. P., & Thompson, W. K. (2019).
    Polygenic prediction via Bayesian regression and continuous shrinkage priors.
    Nature Communications, 10(1), 1776.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class PRSCSRunner:
    """
    Wrapper for PRS-CS (Polygenic Risk Score - Continuous Shrinkage).

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing PRScs.py
    """

    def __init__(self, config: dict, install_dir: str | Path):
        """
        Initialize PRS-CS runner.

        Parameters
        ----------
        config : dict
            Configuration dictionary with parameters like:
            - phi: Continuous shrinkage parameter (default 1e-2)
            - n_iter: Number of iterations (default 1000)
            - n_burnin: Burn-in iterations (default 500)
        install_dir : str | Path
            Directory containing PRScs.py script
        """
        self.config = config
        self.install_dir = Path(install_dir)
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Check that PRScs.py exists."""
        prscs_path = self.install_dir / "PRScs.py"
        if not prscs_path.exists():
            raise FileNotFoundError(
                f"PRScs.py not found in {self.install_dir}"
            )
        log.info("prs_cs_initialized", install_dir=str(self.install_dir))

    def validate_inputs(
        self,
        gwas_file: Path,
        bim_prefix: Path,
        ld_ref_dir: Path,
        n_gwas: int,
    ) -> None:
        """
        Validate input files exist.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics file
        bim_prefix : Path
            Prefix for PLINK .bim file (will check .bim existence)
        ld_ref_dir : Path
            Directory containing LD reference matrices
        n_gwas : int
            Sample size in GWAS

        Raises
        ------
        FileNotFoundError
            If required files don't exist
        ValueError
            If parameters are invalid
        """
        if not gwas_file.exists():
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        bim_file = Path(str(bim_prefix) + ".bim")
        if not bim_file.exists():
            raise FileNotFoundError(f"BIM file not found: {bim_file}")

        if not ld_ref_dir.exists() or not ld_ref_dir.is_dir():
            raise FileNotFoundError(f"LD reference dir not found: {ld_ref_dir}")

        if n_gwas <= 0:
            raise ValueError(f"n_gwas must be positive, got {n_gwas}")

        log.info(
            "prs_cs_inputs_validated",
            gwas_file=str(gwas_file),
            bim_prefix=str(bim_prefix),
            ld_ref_dir=str(ld_ref_dir),
            n_gwas=n_gwas,
        )

    def _build_command(
        self,
        gwas_file: Path,
        bim_prefix: Path,
        ld_ref_dir: Path,
        n_gwas: int,
        out_dir: Path,
        chrom: Optional[int] = None,
        phi: Optional[float] = None,
    ) -> list[str]:
        """
        Build PRScs.py command.

        Parameters
        ----------
        gwas_file : Path
            GWAS file path
        bim_prefix : Path
            PLINK .bim prefix
        ld_ref_dir : Path
            LD reference directory
        n_gwas : int
            GWAS sample size
        out_dir : Path
            Output directory
        chrom : Optional[int]
            Chromosome (if None, process all)
        phi : Optional[float]
            Shrinkage parameter (default from config)

        Returns
        -------
        list[str]
            Command as list of strings
        """
        cmd = [
            "python",
            str(self.install_dir / "PRScs.py"),
            "--ref_dir",
            str(ld_ref_dir),
            "--bim_prefix",
            str(bim_prefix),
            "--sst_file",
            str(gwas_file),
            "--n_gwas",
            str(n_gwas),
            "--out_dir",
            str(out_dir),
        ]

        if chrom is not None:
            cmd.extend(["--chrom", str(chrom)])

        phi_val = phi or self.config.get("phi", 1e-2)
        cmd.extend(["--phi", str(phi_val)])

        if "n_iter" in self.config:
            cmd.extend(["--n_iter", str(self.config["n_iter"])])

        if "n_burnin" in self.config:
            cmd.extend(["--n_burnin", str(self.config["n_burnin"])])

        return cmd

    def run(
        self,
        gwas_file: str | Path,
        bim_prefix: str | Path,
        ld_ref_dir: str | Path,
        n_gwas: int,
        out_dir: str | Path,
        chrom: Optional[int] = None,
        phi: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run PRS-CS to generate polygenic weights.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics (format: SNP A1 A2 BETA SE)
        bim_prefix : str | Path
            PLINK .bim file prefix
        ld_ref_dir : str | Path
            Directory with LD reference matrices
        n_gwas : int
            GWAS sample size
        out_dir : str | Path
            Output directory
        chrom : Optional[int]
            If specified, process only this chromosome
        phi : Optional[float]
            Shrinkage parameter (overrides config)

        Returns
        -------
        pd.DataFrame
            Weights dataframe with columns: SNP, CHR, BP, A1, A2, WEIGHT

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If PRScs.py execution fails
        """
        gwas_file = Path(gwas_file)
        bim_prefix = Path(bim_prefix)
        ld_ref_dir = Path(ld_ref_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_file, bim_prefix, ld_ref_dir, n_gwas)

        cmd = self._build_command(
            gwas_file, bim_prefix, ld_ref_dir, n_gwas, out_dir, chrom, phi
        )

        log.info("prs_cs_starting", command=" ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            log.info("prs_cs_completed", stdout=result.stdout[:500])
        except subprocess.CalledProcessError as e:
            log.error(
                "prs_cs_failed",
                returncode=e.returncode,
                stderr=e.stderr[:500],
            )
            raise RuntimeError(f"PRScs.py failed: {e.stderr}") from e

        return self.parse_output(out_dir, chrom)

    def parse_output(
        self,
        out_dir: Path,
        chrom: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Parse PRS-CS output files.

        PRS-CS outputs weights as:
        SNP CHR BP A1 A2 WEIGHT

        Parameters
        ----------
        out_dir : Path
            Output directory from run()
        chrom : Optional[int]
            If specified, parse only this chromosome

        Returns
        -------
        pd.DataFrame
            Columns: SNP, CHR, BP, A1, A2, WEIGHT

        Raises
        ------
        FileNotFoundError
            If output files not found
        ValueError
            If output format is unexpected
        """
        out_dir = Path(out_dir)
        weights_dfs = []

        if chrom is not None:
            # Single chromosome
            weight_file = out_dir / f"prs_chr{chrom}.txt"
            if not weight_file.exists():
                raise FileNotFoundError(f"Output file not found: {weight_file}")
            weights_dfs.append(pd.read_csv(weight_file, sep="\s+"))
        else:
            # All chromosomes
            for weight_file in sorted(out_dir.glob("prs_chr*.txt")):
                weights_dfs.append(pd.read_csv(weight_file, sep="\s+"))

        if not weights_dfs:
            raise FileNotFoundError(
                f"No PRS output files found in {out_dir}"
            )

        weights_df = pd.concat(weights_dfs, ignore_index=True)

        # Validate columns
        expected_cols = {"SNP", "CHR", "BP", "A1", "A2", "WEIGHT"}
        if not expected_cols.issubset(weights_df.columns):
            raise ValueError(
                f"Output missing expected columns. Expected {expected_cols}, "
                f"got {set(weights_df.columns)}"
            )

        log.info(
            "prs_cs_output_parsed",
            n_snps=len(weights_df),
            cols=list(weights_df.columns),
        )

        return weights_df[["SNP", "CHR", "BP", "A1", "A2", "WEIGHT"]]
