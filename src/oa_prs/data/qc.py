"""
Quality control functions for GWAS summary statistics.

Applies filters for MAF, INFO score, HWE, duplicates, ambiguous strand SNPs,
indels, and multi-allelic variants.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
import structlog

from oa_prs.constants import (
    VALID_ALLELES,
    COMPLEMENT,
    AMBIGUOUS_PAIRS,
    DEFAULT_QC,
)

logger = structlog.get_logger(__name__)


def run_qc(
    df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply quality control filters to GWAS summary statistics.

    Filters applied in order:
    1. Remove duplicates (by SNP ID)
    2. Remove indels (non-ACGT alleles)
    3. Remove multi-allelic variants
    4. Remove strand-ambiguous SNPs (A/T or C/G)
    5. Remove SNPs with MAF below threshold
    6. Remove SNPs with INFO score below threshold (if available)
    7. Remove SNPs failing HWE test (if available)

    Args:
        df: DataFrame with columns SNP, A1, A2, MAF, and optionally INFO, HWE_P
        config: QC config dict with keys:
            - maf_threshold: float, default 0.01
            - info_threshold: float, default 0.8
            - hwe_p_threshold: float, default 1e-6
            - remove_duplicates: bool, default True
            - remove_indels: bool, default True
            - remove_ambiguous: bool, default True

    Returns:
        Tuple of:
        - Filtered DataFrame
        - QC report dict with counts at each filtering step

    Example:
        df, qc_report = run_qc(gwas_df, config={'maf_threshold': 0.01})
        print(f"Variants after QC: {len(df)}")
        print(qc_report)
    """
    if config is None:
        config = DEFAULT_QC.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_QC.copy()
        merged_config.update(config)
        config = merged_config

    logger.info("Starting GWAS QC", initial_snps=len(df), config=config)

    # Initialize report
    report = {"initial": len(df)}
    df_qc = df.copy()

    # Check required columns
    required_cols = {"SNP", "A1", "A2", "MAF"}
    if not required_cols.issubset(df_qc.columns):
        missing = required_cols - set(df_qc.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # 1. Remove duplicates
    if config.get("remove_duplicates", True):
        before = len(df_qc)
        df_qc = df_qc.drop_duplicates(subset=["SNP"], keep="first")
        after = len(df_qc)
        removed = before - after
        report["after_dedup"] = after
        logger.info(
            "Removed duplicates",
            removed=removed,
            remaining=after,
        )

    # 2. Remove indels (check for non-ACGT characters)
    if config.get("remove_indels", True):
        before = len(df_qc)
        mask = df_qc["A1"].isin(VALID_ALLELES) & df_qc["A2"].isin(VALID_ALLELES)
        df_qc = df_qc[mask]
        after = len(df_qc)
        removed = before - after
        report["after_indel_filter"] = after
        logger.info(
            "Removed indels",
            removed=removed,
            remaining=after,
        )

    # 3. Remove multi-allelic variants (A1 == A2 or extra alleles)
    before = len(df_qc)
    mask = df_qc["A1"] != df_qc["A2"]
    df_qc = df_qc[mask]
    after = len(df_qc)
    removed = before - after
    report["after_multiallelic_filter"] = after
    logger.info(
        "Removed multi-allelic variants",
        removed=removed,
        remaining=after,
    )

    # 4. Remove strand-ambiguous SNPs (A/T or C/G)
    if config.get("remove_ambiguous", True):
        before = len(df_qc)
        mask = ~df_qc.apply(
            lambda row: frozenset({row["A1"], row["A2"]}) in AMBIGUOUS_PAIRS,
            axis=1,
        )
        df_qc = df_qc[mask]
        after = len(df_qc)
        removed = before - after
        report["after_ambiguous_filter"] = after
        logger.info(
            "Removed strand-ambiguous SNPs",
            removed=removed,
            remaining=after,
        )

    # 5. MAF filter
    maf_threshold = config.get("maf_threshold", 0.01)
    before = len(df_qc)
    df_qc = df_qc[df_qc["MAF"] >= maf_threshold]
    after = len(df_qc)
    removed = before - after
    report["after_maf_filter"] = after
    logger.info(
        "Applied MAF filter",
        threshold=maf_threshold,
        removed=removed,
        remaining=after,
    )

    # 6. INFO filter (if available and threshold > 0)
    info_threshold = config.get("info_threshold", 0.8)
    if "INFO" in df_qc.columns and info_threshold > 0:
        before = len(df_qc)
        df_qc = df_qc[df_qc["INFO"] >= info_threshold]
        after = len(df_qc)
        removed = before - after
        report["after_info_filter"] = after
        logger.info(
            "Applied INFO filter",
            threshold=info_threshold,
            removed=removed,
            remaining=after,
        )

    # 7. HWE filter (if available and threshold > 0)
    hwe_threshold = config.get("hwe_p_threshold", 1e-6)
    if "HWE_P" in df_qc.columns and hwe_threshold > 0:
        before = len(df_qc)
        df_qc = df_qc[df_qc["HWE_P"] >= hwe_threshold]
        after = len(df_qc)
        removed = before - after
        report["after_hwe_filter"] = after
        logger.info(
            "Applied HWE filter",
            threshold=hwe_threshold,
            removed=removed,
            remaining=after,
        )

    report["final"] = len(df_qc)
    report["total_removed"] = report["initial"] - report["final"]
    report["retention_pct"] = (report["final"] / report["initial"]) * 100

    logger.info(
        "QC completed",
        initial_snps=report["initial"],
        final_snps=report["final"],
        removed=report["total_removed"],
        retention_pct=f"{report['retention_pct']:.1f}%",
    )

    return df_qc.reset_index(drop=True), report
