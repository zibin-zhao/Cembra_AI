"""
Input/output utilities for reading and writing genomic data.
"""

import gzip
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd


def read_gwas(
    path: str | Path,
    format: Optional[Literal["parquet", "tsv", "csv", "gz"]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read GWAS summary statistics with automatic format detection.

    Automatically detects format from file extension if not specified.
    Supports: parquet, TSV, CSV, and gzipped TSV.

    Parameters
    ----------
    path : str | Path
        Path to GWAS file
    format : Optional[Literal["parquet", "tsv", "csv", "gz"]]
        File format. If None, auto-detect from extension
    **kwargs
        Additional arguments passed to pandas read function

    Returns
    -------
    pd.DataFrame
        GWAS summary statistics dataframe

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If format cannot be determined
    IOError
        If file cannot be read

    Examples
    --------
    >>> gwas_df = read_gwas("gwas.parquet")
    >>> gwas_df = read_gwas("gwas.tsv.gz")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect format
    if format is None:
        if path.suffix == ".parquet":
            format = "parquet"
        elif path.suffix == ".gz":
            format = "gz"
        elif path.suffix == ".tsv":
            format = "tsv"
        elif path.suffix == ".csv":
            format = "csv"
        else:
            raise ValueError(
                f"Cannot auto-detect format for {path}. "
                "Please specify format explicitly."
            )

    # Set default delimiter for text formats
    if format in ["tsv", "gz"] and "sep" not in kwargs:
        kwargs["sep"] = "\t"
    elif format == "csv" and "sep" not in kwargs:
        kwargs["sep"] = ","

    try:
        if format == "parquet":
            return pd.read_parquet(path)
        elif format == "gz":
            with gzip.open(path, "rt") as f:
                return pd.read_csv(f, **kwargs)
        elif format in ["tsv", "csv"]:
            return pd.read_csv(path, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
    except Exception as e:
        raise IOError(f"Failed to read {path}: {e}") from e


def write_gwas(
    df: pd.DataFrame,
    path: str | Path,
    format: Literal["parquet", "tsv", "csv"] = "parquet",
    **kwargs,
) -> None:
    """
    Write GWAS summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        GWAS dataframe to write
    path : str | Path
        Output path
    format : Literal["parquet", "tsv", "csv"]
        Output format
    **kwargs
        Additional arguments passed to pandas write function

    Raises
    ------
    ValueError
        If format is invalid
    IOError
        If write fails

    Examples
    --------
    >>> write_gwas(gwas_df, "output.parquet")
    >>> write_gwas(gwas_df, "output.tsv.gz", sep="\\t")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "parquet":
            df.to_parquet(path, **kwargs)
        elif format == "tsv":
            df.to_csv(path, sep="\t", index=False, **kwargs)
        elif format == "csv":
            df.to_csv(path, index=False, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
    except Exception as e:
        raise IOError(f"Failed to write {path}: {e}") from e


def read_plink_bim(path: str | Path) -> pd.DataFrame:
    """
    Read PLINK .bim file (variant information).

    PLINK .bim format: CHR SNP BP A1 A2 (and sometimes 6th column)

    Parameters
    ----------
    path : str | Path
        Path to .bim file

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: CHR, SNP, BP, A1, A2

    Raises
    ------
    FileNotFoundError
        If file not found
    IOError
        If file cannot be read

    Examples
    --------
    >>> bim_df = read_plink_bim("data.bim")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"BIM file not found: {path}")

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["CHR", "SNP", "BP", "A1", "A2"],
            usecols=[0, 1, 2, 3, 4],
            dtype={"CHR": "int8", "SNP": str, "BP": "int64", "A1": str, "A2": str},
        )
        return df
    except Exception as e:
        raise IOError(f"Failed to read .bim file {path}: {e}") from e


def read_plink_fam(path: str | Path) -> pd.DataFrame:
    """
    Read PLINK .fam file (sample information).

    PLINK .fam format: FID IID PAT MAT SEX PHENO

    Parameters
    ----------
    path : str | Path
        Path to .fam file

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: FID, IID, PAT, MAT, SEX, PHENO

    Raises
    ------
    FileNotFoundError
        If file not found
    IOError
        If file cannot be read

    Examples
    --------
    >>> fam_df = read_plink_fam("data.fam")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"FAM file not found: {path}")

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
            dtype={"FID": str, "IID": str, "PAT": str, "MAT": str, "SEX": "int8"},
        )
        return df
    except Exception as e:
        raise IOError(f"Failed to read .fam file {path}: {e}") from e


def read_h5_scores(
    path: str | Path,
    snp_list: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Read pre-computed variant scores from HDF5 file.

    Expected HDF5 structure:
    - Datasets: "snps", "scores" (1D arrays) or variants × scores (2D)
    - Attributes: metadata

    Parameters
    ----------
    path : str | Path
        Path to HDF5 file
    snp_list : Optional[list[str]]
        If provided, only read these SNPs

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: SNP, SCORE

    Raises
    ------
    FileNotFoundError
        If file not found
    IOError
        If HDF5 cannot be read

    Examples
    --------
    >>> scores_df = read_h5_scores("enformer_scores.h5")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    try:
        with h5py.File(path, "r") as f:
            # Read SNP identifiers
            if "snps" in f:
                snps = f["snps"][:]
                if isinstance(snps[0], bytes):
                    snps = [s.decode("utf-8") for s in snps]
            else:
                raise KeyError("HDF5 file must contain 'snps' dataset")

            # Read scores
            if "scores" in f:
                scores = f["scores"][:]
            else:
                raise KeyError("HDF5 file must contain 'scores' dataset")

            # Filter to requested SNPs if provided
            if snp_list is not None:
                mask = np.isin(snps, snp_list)
                snps = snps[mask]
                scores = scores[mask]

            return pd.DataFrame({"SNP": snps, "SCORE": scores})
    except Exception as e:
        raise IOError(f"Failed to read HDF5 file {path}: {e}") from e


def write_h5_scores(
    df: pd.DataFrame,
    path: str | Path,
    snp_col: str = "SNP",
    score_col: str = "SCORE",
    metadata: Optional[dict] = None,
) -> None:
    """
    Write variant scores to HDF5 file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with SNP and score columns
    path : str | Path
        Output HDF5 path
    snp_col : str
        Name of SNP column
    score_col : str
        Name of score column
    metadata : Optional[dict]
        Metadata to store as HDF5 attributes

    Raises
    ------
    KeyError
        If required columns not found
    IOError
        If write fails

    Examples
    --------
    >>> write_h5_scores(scores_df, "output.h5", metadata={"source": "Enformer"})
    """
    if snp_col not in df.columns or score_col not in df.columns:
        raise KeyError(f"Columns {snp_col} or {score_col} not found")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("snps", data=df[snp_col].values.astype("S"))
            f.create_dataset("scores", data=df[score_col].values.astype("float32"))

            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = str(value)
    except Exception as e:
        raise IOError(f"Failed to write H5 file {path}: {e}") from e
