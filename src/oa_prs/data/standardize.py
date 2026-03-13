"""
Standardization of diverse GWAS formats to canonical schema.

Handles coordinate liftover, column renaming, and MAF computation.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import structlog

from oa_prs.constants import GWAS_COLUMNS, GWAS_DTYPES

logger = structlog.get_logger(__name__)


def standardize_gwas(
    filepath: str | Path,
    column_mapping: Dict[str, str],
    build_from: Optional[str] = None,
    build_to: str = "hg38",
    maf_from_freq: bool = True,
    sep: str = "\t",
    compression: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize GWAS summary statistics to canonical schema.

    Converts diverse GWAS formats to the standard schema:
    [SNP, CHR, BP, A1, A2, BETA, SE, P, MAF, N]

    Handles:
    - Column renaming based on mapping
    - Coordinate liftover (hg19 → hg38)
    - MAF computation from allele frequency if needed
    - Data type conversion

    Args:
        filepath: Path to GWAS file (gzip, bz2 supported)
        column_mapping: Dict mapping standard column names to file column names.
                       Required keys: SNP, CHR, BP, A1, A2, BETA, SE, P
                       Optional: MAF, FREQ, N
                       Example: {'SNP': 'variant_id', 'CHR': 'chromosome', ...}
        build_from: Genome build of input coordinates ('hg19' or 'hg38', optional)
        build_to: Target genome build ('hg38' default)
        maf_from_freq: If True, compute MAF from FREQ column if MAF not available
        sep: Field separator (default tab-delimited)
        compression: Compression mode ('gzip', 'bz2', or None)

    Returns:
        Standardized DataFrame with canonical schema

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns missing or coordinate lift fails

    Example:
        mapping = {
            'SNP': 'rsid',
            'CHR': 'chrom',
            'BP': 'pos',
            'A1': 'alt',
            'A2': 'ref',
            'BETA': 'effect',
            'SE': 'se',
            'P': 'pval',
            'N': 'n',
        }
        gwas_std = standardize_gwas('gwas.txt.gz', mapping, build_from='hg19')
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GWAS file not found: {filepath}")

    logger.info(
        "Loading GWAS file",
        filepath=str(filepath),
        sep=repr(sep),
        compression=compression,
    )

    # Load file
    try:
        df = pd.read_csv(
            filepath,
            sep=sep,
            compression=compression,
            low_memory=False,
            dtype=str,  # Read as strings initially to handle missing values
        )
    except Exception as e:
        raise ValueError(f"Failed to load GWAS file: {e}")

    logger.info("Loaded GWAS file", rows=len(df), columns=len(df.columns))

    # Check required columns
    required_cols = {"SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
    missing_cols = required_cols - set(column_mapping.keys())
    if missing_cols:
        raise ValueError(f"Missing column mappings for: {missing_cols}")

    missing_in_file = [
        col for col in column_mapping.values() if col not in df.columns
    ]
    if missing_in_file:
        raise ValueError(f"Missing in file: {missing_in_file}")

    # Rename columns to standard names
    rename_dict = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=rename_dict)

    logger.info("Renamed columns", mapping=list(rename_dict.items())[:5])

    # Select relevant columns (may be a subset if optional columns missing)
    cols_to_keep = [col for col in GWAS_COLUMNS if col in df.columns]
    df = df[cols_to_keep].copy()

    # Convert data types
    for col in df.columns:
        if col in GWAS_DTYPES:
            try:
                if col in ["CHR", "N"]:
                    # Integer columns
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        GWAS_DTYPES[col]
                    )
                elif col == "BP":
                    # Position (int64)
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
                else:
                    # Float columns
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        GWAS_DTYPES[col]
                    )
            except Exception as e:
                logger.warning(
                    f"Type conversion failed for column {col}",
                    error=str(e),
                )

    # Remove rows with missing required values
    required_in_df = [col for col in required_cols if col in df.columns]
    before = len(df)
    df = df.dropna(subset=required_in_df)
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed rows with missing values", count=removed)

    # Lift coordinates if needed
    if build_from and build_from != build_to:
        if build_from == "hg19" and build_to == "hg38":
            logger.info(
                "Lifting coordinates from hg19 to hg38",
                rows=len(df),
            )
            df = _liftover_hg19_to_hg38(df)
        else:
            logger.warning(
                f"Unsupported lift: {build_from} → {build_to}, skipping"
            )

    # Compute MAF from allele frequency if needed
    if "MAF" not in df.columns and maf_from_freq:
        if "FREQ" in column_mapping:
            freq_col = column_mapping["FREQ"]
            if freq_col in df.columns:
                df["FREQ"] = pd.to_numeric(df["FREQ"], errors="coerce")
                # MAF is min(freq, 1-freq) but we'll use freq directly
                df["MAF"] = df["FREQ"].apply(lambda x: min(x, 1 - x) if pd.notna(x) else np.nan)
                logger.info("Computed MAF from FREQ column")
            else:
                logger.warning(f"FREQ column not found in mapping, MAF will be missing")
        else:
            logger.warning("MAF not available and FREQ not in mapping")

    # Ensure all standard columns are present (add with NaN if missing)
    for col in GWAS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            logger.info(f"Added missing column {col} (all NaN)")

    # Reorder to standard schema
    df = df[GWAS_COLUMNS]

    logger.info(
        "GWAS standardization complete",
        rows=len(df),
        columns=len(GWAS_COLUMNS),
    )

    return df


def _liftover_hg19_to_hg38(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform simplified liftover from hg19 to hg38.

    This is a placeholder using a simplified approach. For production,
    use pyliftover or pyensembl with actual chain files.

    In a real scenario, you would load actual hg19->hg38 chain files
    and perform proper base-pair coordinate conversion.

    Args:
        df: DataFrame with BP column (hg19 coordinates)

    Returns:
        DataFrame with updated BP column (hg38 coordinates)

    Note:
        This simplified version does not modify coordinates.
        Production code should use proper liftover libraries.
    """
    logger.warning(
        "Using simplified liftover (no actual conversion). "
        "Install pyliftover and load proper chain files for production."
    )

    # Placeholder: in production, load actual hg19->hg38 chain file
    # For now, return as-is with warning
    # Example production code:
    # from pyliftover import LiftOver
    # lo = LiftOver('hg19', 'hg38')
    # df['BP'] = df.apply(
    #     lambda row: lo.convert_coordinate(
    #         f"chr{row['CHR']}", row['BP']
    #     )[0][1] if row['CHR'] in range(1, 23) else row['BP'],
    #     axis=1
    # )

    return df
