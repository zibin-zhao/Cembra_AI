"""
PRS-CSx wrapper for cross-ancestry polygenic risk score calculation.

References:
    Ge, T., Chen, C.-Y., Feng, Q., Hibar, D. P., & Thompson, W. K. (2022).
    Polygenic prediction via Bayesian regression and continuous shrinkage priors.
    Nature Communications, 13(1), 7636.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class PRSCSxRunner:
    """
    Wrapper for PRS-CSx (cross-ancestry PRS using continuous shrinkage).

    Handles multiple ancestries simultaneously using ancestry-specific
    LD reference matrices.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing PRScsx.py
    """

    def __init__(self, config: dict, install_dir: str | Path):
        """
        Initialize PRS-CSx runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - phi_dict: dict[str, float] per-ancestry shrinkage parameters
            - n_iter: Number of iterations
            - n_burnin: Burn-in iterations
        install_dir : str | Path
            Directory containing PRScsx.py script
        """
        self.config = config
        self.install_dir = Path(install_dir)
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Check that PRScsx.py exists."""
        prscs_path = self.install_dir / "PRScsx.py"
        if not prscs_path.exists():
            raise FileNotFoundError(
                f"PRScsx.py not found in {self.install_dir}"
            )
        log.info("prs_csx_initialized", install_dir=str(self.install_dir))

    def validate_inputs(
        self,
        gwas_files: dict[str, Path],
        n_gwas: dict[str, int],
        bim_prefix: Path,
        ld_ref_dir: Path,
        populations: list[str],
    ) -> None:
        """
        Validate input files and parameters.

        Parameters
        ----------
        gwas_files : dict[str, Path]
            GWAS file per ancestry
        n_gwas : dict[str, int]
            Sample size per ancestry
        bim_prefix : Path
            PLINK .bim prefix
        ld_ref_dir : Path
            LD reference directory
        populations : list[str]
            List of population codes (EUR, EAS, AFR, SAS, AMR)

        Raises
        ------
        FileNotFoundError
            If required files don't exist
        ValueError
            If parameters are invalid
        """
        for pop in populations:
            if pop not in gwas_files:
                raise ValueError(f"Missing GWAS file for {pop}")
            if pop not in n_gwas:
                raise ValueError(f"Missing n_gwas for {pop}")

            gwas_file = gwas_files[pop]
            if not gwas_file.exists():
                raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

            if n_gwas[pop] <= 0:
                raise ValueError(f"n_gwas[{pop}] must be positive")

        bim_file = Path(str(bim_prefix) + ".bim")
        if not bim_file.exists():
            raise FileNotFoundError(f"BIM file not found: {bim_file}")

        if not ld_ref_dir.exists() or not ld_ref_dir.is_dir():
            raise FileNotFoundError(f"LD reference dir not found: {ld_ref_dir}")

        log.info(
            "prs_csx_inputs_validated",
            populations=populations,
            bim_prefix=str(bim_prefix),
        )

    def _build_command(
        self,
        gwas_files: dict[str, Path],
        n_gwas: dict[str, int],
        bim_prefix: Path,
        ld_ref_dir: Path,
        populations: list[str],
        out_dir: Path,
        chrom: Optional[int] = None,
        phi: Optional[dict[str, float]] = None,
    ) -> list[str]:
        """
        Build PRScsx.py command.

        Parameters
        ----------
        gwas_files : dict[str, Path]
            GWAS file paths per ancestry
        n_gwas : dict[str, int]
            GWAS sample sizes per ancestry
        bim_prefix : Path
            PLINK .bim prefix
        ld_ref_dir : Path
            LD reference directory
        populations : list[str]
            List of populations
        out_dir : Path
            Output directory
        chrom : Optional[int]
            Chromosome (if None, process all)
        phi : Optional[dict[str, float]]
            Shrinkage parameters per ancestry

        Returns
        -------
        list[str]
            Command as list of strings
        """
        # Build sst_file argument: pop1,path1,pop2,path2,...
        sst_parts = []
        for pop in populations:
            sst_parts.append(f"{pop},{gwas_files[pop]}")
        sst_arg = " ".join(sst_parts)

        # Build n_gwas argument
        n_gwas_parts = []
        for pop in populations:
            n_gwas_parts.append(str(n_gwas[pop]))
        n_gwas_arg = ",".join(n_gwas_parts)

        cmd = [
            "python",
            str(self.install_dir / "PRScsx.py"),
            "--ref_dir",
            str(ld_ref_dir),
            "--bim_prefix",
            str(bim_prefix),
            "--sst_file",
            sst_arg,
            "--n_gwas",
            n_gwas_arg,
            "--pop",
            ",".join(populations),
            "--out_dir",
            str(out_dir),
        ]

        if chrom is not None:
            cmd.extend(["--chrom", str(chrom)])

        # Add phi parameters
        if phi:
            phi_parts = []
            for pop in populations:
                phi_val = phi.get(pop, self.config.get("phi", 1e-2))
                phi_parts.append(str(phi_val))
            cmd.extend(["--phi", ",".join(phi_parts)])

        if "n_iter" in self.config:
            cmd.extend(["--n_iter", str(self.config["n_iter"])])

        if "n_burnin" in self.config:
            cmd.extend(["--n_burnin", str(self.config["n_burnin"])])

        return cmd

    def run(
        self,
        gwas_files: dict[str, str | Path],
        n_gwas: dict[str, int],
        bim_prefix: str | Path,
        ld_ref_dir: str | Path,
        populations: list[str],
        out_dir: str | Path,
        chrom: Optional[int] = None,
        phi: Optional[dict[str, float]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run PRS-CSx to generate cross-ancestry polygenic weights.

        Parameters
        ----------
        gwas_files : dict[str, str | Path]
            GWAS files per ancestry (pop → path)
        n_gwas : dict[str, int]
            Sample sizes per ancestry
        bim_prefix : str | Path
            PLINK .bim prefix
        ld_ref_dir : str | Path
            LD reference directory
        populations : list[str]
            Population codes (e.g., ["EUR", "EAS"])
        out_dir : str | Path
            Output directory
        chrom : Optional[int]
            If specified, process only this chromosome
        phi : Optional[dict[str, float]]
            Shrinkage parameters per ancestry

        Returns
        -------
        dict[str, pd.DataFrame]
            Weights per ancestry: {pop → DataFrame(SNP, CHR, BP, A1, A2, WEIGHT)}

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        ValueError
            If parameters invalid
        RuntimeError
            If PRScsx.py execution fails
        """
        # Convert to Path objects
        gwas_files = {pop: Path(p) for pop, p in gwas_files.items()}
        bim_prefix = Path(bim_prefix)
        ld_ref_dir = Path(ld_ref_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_files, n_gwas, bim_prefix, ld_ref_dir, populations)

        cmd = self._build_command(
            gwas_files, n_gwas, bim_prefix, ld_ref_dir, populations, out_dir,
            chrom, phi
        )

        log.info(
            "prs_csx_starting",
            populations=populations,
            command=" ".join(cmd[:5]),
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            log.info("prs_csx_completed", stdout=result.stdout[:500])
        except subprocess.CalledProcessError as e:
            log.error(
                "prs_csx_failed",
                returncode=e.returncode,
                stderr=e.stderr[:500],
            )
            raise RuntimeError(f"PRScsx.py failed: {e.stderr}") from e

        return self.parse_output(out_dir, populations, chrom)

    def parse_output(
        self,
        out_dir: Path,
        populations: list[str],
        chrom: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Parse PRS-CSx output files.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()
        populations : list[str]
            Population codes
        chrom : Optional[int]
            If specified, parse only this chromosome

        Returns
        -------
        dict[str, pd.DataFrame]
            Weights per ancestry

        Raises
        ------
        FileNotFoundError
            If output files not found
        ValueError
            If output format unexpected
        """
        out_dir = Path(out_dir)
        results = {}

        for pop in populations:
            weights_dfs = []

            if chrom is not None:
                # Single chromosome
                weight_file = out_dir / f"prs_{pop}_chr{chrom}.txt"
                if not weight_file.exists():
                    raise FileNotFoundError(
                        f"Output file not found: {weight_file}"
                    )
                weights_dfs.append(pd.read_csv(weight_file, sep="\s+"))
            else:
                # All chromosomes
                pattern = f"prs_{pop}_chr*.txt"
                for weight_file in sorted(out_dir.glob(pattern)):
                    weights_dfs.append(pd.read_csv(weight_file, sep="\s+"))

            if not weights_dfs:
                raise FileNotFoundError(
                    f"No output files found for {pop} in {out_dir}"
                )

            weights_df = pd.concat(weights_dfs, ignore_index=True)

            # Validate columns
            expected_cols = {"SNP", "CHR", "BP", "A1", "A2", "WEIGHT"}
            if not expected_cols.issubset(weights_df.columns):
                raise ValueError(
                    f"Output missing expected columns for {pop}. "
                    f"Expected {expected_cols}, got {set(weights_df.columns)}"
                )

            results[pop] = weights_df[["SNP", "CHR", "BP", "A1", "A2", "WEIGHT"]]

        log.info(
            "prs_csx_output_parsed",
            populations=populations,
            n_snps_per_pop={pop: len(results[pop]) for pop in populations},
        )

        return results
