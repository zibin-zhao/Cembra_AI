"""
SMR-HEIDI wrapper for causal inference in TWAS.

SMR (Summary-data-based Mendelian Randomization) with HEIDI (Heterogeneity
in Dependent Instruments) test distinguishes causal variants from horizontal
pleiotropy.

References:
    Zhu, Z., Zheng, Z., Zhang, F., Wang, Z., Zhao, J., Zhang, Q., ... & Yang, J.
    (2018). Causal associations between risk factors and common diseases inferred
    from GWAS summary data. Nature Communications, 9(1), 224.
"""

import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class SMRHEIDIRunner:
    """
    Wrapper for SMR-HEIDI (Summary-data Mendelian Randomization).

    Tests for causal associations between gene expression and phenotypes.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    install_dir : Path
        Directory containing smr executable
    """

    def __init__(self, config: dict, install_dir: Optional[str | Path] = None):
        """
        Initialize SMR-HEIDI runner.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - heidi_mtd: HEIDI test method
            - smr_thresh: SMR p-value threshold
            - heidi_thresh: HEIDI p-value threshold
        install_dir : Optional[str | Path]
            Directory containing smr executable
        """
        self.config = config
        self.install_dir = Path(install_dir) if install_dir else None
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Check that SMR executable exists if install_dir provided."""
        if self.install_dir:
            smr_path = self.install_dir / "smr"
            if not smr_path.exists():
                log.warning(
                    "smr_executable_not_found",
                    expected_path=str(smr_path),
                )
        else:
            log.info("smr_install_dir_not_specified")

    def validate_inputs(
        self,
        gwas_file: Path,
        eqtl_besd_prefix: Path,
    ) -> None:
        """
        Validate input files.

        Parameters
        ----------
        gwas_file : Path
            GWAS summary statistics
        eqtl_besd_prefix : Path
            eQTL BESD file prefix

        Raises
        ------
        FileNotFoundError
            If files don't exist
        """
        if not gwas_file.exists():
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        # Check for BESD files
        for suffix in [".esd.gz", ".esi.gz", ".epi.gz"]:
            besd_file = Path(str(eqtl_besd_prefix) + suffix)
            if not besd_file.exists():
                raise FileNotFoundError(f"BESD file not found: {besd_file}")

        log.info(
            "smr_heidi_inputs_validated",
            gwas_file=str(gwas_file),
            eqtl_prefix=str(eqtl_besd_prefix),
        )

    def _build_command(
        self,
        gwas_file: Path,
        eqtl_besd_prefix: Path,
        out_dir: Path,
    ) -> list[str]:
        """
        Build SMR command.

        Parameters
        ----------
        gwas_file : Path
            GWAS file path
        eqtl_besd_prefix : Path
            eQTL BESD prefix
        out_dir : Path
            Output directory

        Returns
        -------
        list[str]
            Command as list
        """
        smr_cmd = str(self.install_dir / "smr") if self.install_dir else "smr"

        cmd = [
            smr_cmd,
            "--bfile",
            str(eqtl_besd_prefix),
            "--gwas-summary",
            str(gwas_file),
            "--out",
            str(out_dir / "smr_results"),
            "--trans",
        ]

        # Add HEIDI test
        if self.config.get("heidi_mtd", True):
            cmd.append("--heidi-mtd")
            cmd.append(str(self.config.get("heidi_method", 1)))

        # Add p-value thresholds
        if "smr_thresh" in self.config:
            cmd.append("--smr-thresh")
            cmd.append(str(self.config["smr_thresh"]))

        if "heidi_thresh" in self.config:
            cmd.append("--heidi-thresh")
            cmd.append(str(self.config["heidi_thresh"]))

        return cmd

    def run(
        self,
        gwas_file: str | Path,
        eqtl_besd_prefix: str | Path,
        out_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Run SMR-HEIDI analysis.

        Parameters
        ----------
        gwas_file : str | Path
            GWAS summary statistics
        eqtl_besd_prefix : str | Path
            eQTL BESD file prefix
        out_dir : str | Path
            Output directory

        Returns
        -------
        pd.DataFrame
            Results with columns: gene, b_smr, p_smr, p_heidi

        Raises
        ------
        FileNotFoundError
            If input files don't exist
        RuntimeError
            If execution fails
        """
        gwas_file = Path(gwas_file)
        eqtl_besd_prefix = Path(eqtl_besd_prefix)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.validate_inputs(gwas_file, eqtl_besd_prefix)

        cmd = self._build_command(gwas_file, eqtl_besd_prefix, out_dir)

        log.info(
            "smr_heidi_starting",
            command=" ".join(cmd[:5]),
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
            )
            log.info("smr_heidi_completed", stdout=result.stdout[:200])
        except subprocess.CalledProcessError as e:
            log.error("smr_heidi_failed", stderr=e.stderr[:500])
            raise RuntimeError(f"SMR-HEIDI failed: {e.stderr}") from e

        return self.parse_output(out_dir)

    def parse_output(self, out_dir: Path) -> pd.DataFrame:
        """
        Parse SMR-HEIDI output.

        Parameters
        ----------
        out_dir : Path
            Output directory from run()

        Returns
        -------
        pd.DataFrame
            Columns: gene, b_smr, p_smr, p_heidi

        Raises
        ------
        FileNotFoundError
            If output file not found
        """
        out_dir = Path(out_dir)
        result_file = out_dir / "smr_results.smr"

        if not result_file.exists():
            raise FileNotFoundError(f"Output file not found: {result_file}")

        df = pd.read_csv(result_file, sep="\s+")

        # Standardize column names (SMR output uses different naming)
        df = df.rename(
            columns={
                "probeID": "gene",
                "b_SMR": "b_smr",
                "p_SMR": "p_smr",
                "p_HEIDI": "p_heidi",
            }
        )

        required_cols = {"gene", "b_smr", "p_smr"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Missing columns. Required: {required_cols}, "
                f"got {set(df.columns)}"
            )

        log.info(
            "smr_heidi_output_parsed",
            n_genes=len(df),
            n_significant_smr=sum(df["p_smr"] < 0.05),
        )

        return df

    def filter_causal(
        self,
        parse_results: pd.DataFrame,
        p_heidi_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Filter results to likely causal associations (HEIDI test pass).

        Parameters
        ----------
        parse_results : pd.DataFrame
            Output from parse_output()
        p_heidi_threshold : float
            HEIDI p-value threshold for causal association

        Returns
        -------
        pd.DataFrame
            Filtered to genes passing HEIDI test

        Examples
        --------
        >>> results = runner.parse_output(out_dir)
        >>> causal = runner.filter_causal(results, p_heidi_threshold=0.01)
        """
        if "p_heidi" not in parse_results.columns:
            log.warning("p_heidi column not found, returning all results")
            return parse_results

        filtered = parse_results[
            parse_results["p_heidi"] > p_heidi_threshold
        ].copy()

        log.info(
            "causal_genes_filtered",
            n_input=len(parse_results),
            n_causal=len(filtered),
            threshold=p_heidi_threshold,
        )

        return filtered
