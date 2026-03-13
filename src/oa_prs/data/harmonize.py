"""
Allele harmonization across GWAS datasets.

Aligns alleles to a reference panel, handles strand flips, removes unresolvable variants.
"""

from typing import Tuple
import pandas as pd
import numpy as np
import structlog

from oa_prs.constants import COMPLEMENT

logger = structlog.get_logger(__name__)


def harmonize_gwas(
    gwas_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Harmonize GWAS alleles to a reference panel.

    Aligns alleles from GWAS data to a reference dataset. Handles:
    - Matching SNPs by position and alleles
    - Flipping alleles if A1 is swapped with A2
    - Detecting strand flips (complement swaps)
    - Removing unresolvable SNPs

    Args:
        gwas_df: GWAS DataFrame with columns: SNP, CHR, BP, A1, A2, BETA, SE, P
        ref_df: Reference DataFrame with columns: SNP, CHR, BP, A1, A2

    Returns:
        Tuple of:
        - Harmonized GWAS DataFrame
        - Report dict with harmonization statistics

    Example:
        gwas_harm, report = harmonize_gwas(gwas_data, reference_data)
        print(f"SNPs removed: {report['unresolvable']}")
        print(f"SNPs with flipped alleles: {report['allele_swaps']}")
    """
    logger.info(
        "Starting GWAS harmonization",
        gwas_snps=len(gwas_df),
        ref_snps=len(ref_df),
    )

    # Ensure required columns
    required_gwas = {"SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
    if not required_gwas.issubset(gwas_df.columns):
        raise ValueError(f"GWAS missing columns: {required_gwas - set(gwas_df.columns)}")

    required_ref = {"SNP", "CHR", "BP", "A1", "A2"}
    if not required_ref.issubset(ref_df.columns):
        raise ValueError(f"Reference missing columns: {required_ref - set(ref_df.columns)}")

    gwas = gwas_df.copy()
    ref = ref_df[["SNP", "CHR", "BP", "A1", "A2"]].copy()
    ref.columns = ["SNP", "REF_CHR", "REF_BP", "REF_A1", "REF_A2"]

    # Merge GWAS with reference by SNP ID
    merged = gwas.merge(ref, on="SNP", how="inner")
    logger.info("SNPs matched by ID", matched=len(merged))

    report = {
        "initial_gwas": len(gwas_df),
        "matched_by_id": len(merged),
        "allele_match": 0,
        "allele_swap": 0,
        "strand_flip": 0,
        "strand_flip_swap": 0,
        "unresolvable": 0,
        "final": 0,
    }

    # Check chromosome and position match
    position_mismatch = (merged["CHR"] != merged["REF_CHR"]) | (
        merged["BP"] != merged["REF_BP"]
    )
    before_pos = len(merged)
    merged = merged[~position_mismatch]
    report["position_mismatch"] = before_pos - len(merged)
    logger.info("Position mismatches removed", count=report["position_mismatch"])

    # Initialize allele status column
    merged["allele_status"] = "unresolvable"
    merged["beta_flipped"] = False

    # 1. Check for exact allele match (A1/A2 same orientation)
    exact_match = (
        (merged["A1"] == merged["REF_A1"]) & (merged["A2"] == merged["REF_A2"])
    )
    merged.loc[exact_match, "allele_status"] = "match"
    report["allele_match"] = exact_match.sum()

    # 2. Check for allele swap (A1/A2 reversed)
    allele_swap = (
        (merged["A1"] == merged["REF_A2"]) & (merged["A2"] == merged["REF_A1"])
    )
    merged.loc[allele_swap, "allele_status"] = "swap"
    merged.loc[allele_swap, "BETA"] = -merged.loc[allele_swap, "BETA"]
    merged.loc[allele_swap, "beta_flipped"] = True
    report["allele_swap"] = allele_swap.sum()

    # 3. Check for strand flip (complement alleles match)
    complement_a1 = merged["A1"].apply(lambda x: COMPLEMENT.get(x, x))
    complement_a2 = merged["A2"].apply(lambda x: COMPLEMENT.get(x, x))

    strand_flip = (complement_a1 == merged["REF_A1"]) & (complement_a2 == merged["REF_A2"])
    merged.loc[strand_flip, "allele_status"] = "strand_flip"
    merged.loc[strand_flip, "A1"] = complement_a1[strand_flip]
    merged.loc[strand_flip, "A2"] = complement_a2[strand_flip]
    report["strand_flip"] = strand_flip.sum()

    # 4. Check for strand flip + allele swap
    strand_flip_swap = (complement_a1 == merged["REF_A2"]) & (
        complement_a2 == merged["REF_A1"]
    )
    merged.loc[strand_flip_swap, "allele_status"] = "strand_flip_swap"
    merged.loc[strand_flip_swap, "A1"] = complement_a2[strand_flip_swap]
    merged.loc[strand_flip_swap, "A2"] = complement_a1[strand_flip_swap]
    merged.loc[strand_flip_swap, "BETA"] = -merged.loc[strand_flip_swap, "BETA"]
    merged.loc[strand_flip_swap, "beta_flipped"] = True
    report["strand_flip_swap"] = strand_flip_swap.sum()

    # Remove unresolvable SNPs
    before_unresolvable = len(merged)
    merged = merged[merged["allele_status"] != "unresolvable"]
    report["unresolvable"] = before_unresolvable - len(merged)

    # Clean up and return
    harmonized = merged[gwas_df.columns.tolist()].copy()
    report["final"] = len(harmonized)

    logger.info(
        "Harmonization completed",
        exact_match=report["allele_match"],
        swaps=report["allele_swap"],
        strand_flips=report["strand_flip"],
        strand_flip_swaps=report["strand_flip_swap"],
        unresolvable=report["unresolvable"],
        final=report["final"],
    )

    return harmonized.reset_index(drop=True), report


def harmonize_multi_gwas(
    gwas_list: list,
) -> Tuple[pd.DataFrame, dict]:
    """
    Harmonize multiple GWAS to a common SNP set.

    Finds the intersection of SNPs across all GWAS datasets after standardization.
    Uses the first GWAS as the reference for alleles.

    Args:
        gwas_list: List of DataFrames, each with columns: SNP, CHR, BP, A1, A2, BETA, SE, P

    Returns:
        Tuple of:
        - Combined harmonized DataFrame with columns: SNP, CHR, BP, A1, A2, and BETA_1, BETA_2, etc.
        - Report dict with statistics

    Raises:
        ValueError: If fewer than 2 GWAS provided or if required columns missing

    Example:
        gwas_combined, report = harmonize_multi_gwas([gwas1, gwas2, gwas3])
        print(f"SNPs in all datasets: {len(gwas_combined)}")
    """
    if len(gwas_list) < 2:
        raise ValueError("Need at least 2 GWAS to harmonize")

    logger.info("Starting multi-GWAS harmonization", num_gwas=len(gwas_list))

    for i, gwas in enumerate(gwas_list):
        required = {"SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
        if not required.issubset(gwas.columns):
            raise ValueError(f"GWAS {i} missing columns: {required - set(gwas.columns)}")

    # Use first GWAS as reference
    ref_gwas = gwas_list[0][["SNP", "CHR", "BP", "A1", "A2"]].copy()
    combined = ref_gwas.copy()

    report = {
        "initial_snps": [len(gwas) for gwas in gwas_list],
        "after_harmonization": [],
    }

    # Harmonize each GWAS to reference
    for i, gwas in enumerate(gwas_list):
        if i == 0:
            # First dataset: just use as-is after selecting common SNPs
            combined[f"BETA_{i}"] = ref_gwas.merge(
                gwas[["SNP", "BETA"]], on="SNP", how="left"
            )["BETA"]
            combined[f"SE_{i}"] = ref_gwas.merge(
                gwas[["SNP", "SE"]], on="SNP", how="left"
            )["SE"]
            combined[f"P_{i}"] = ref_gwas.merge(
                gwas[["SNP", "P"]], on="SNP", how="left"
            )["P"]
        else:
            # Harmonize to reference
            harmonized, _ = harmonize_gwas(gwas, ref_gwas)

            # Merge harmonized data
            combined = combined.merge(
                harmonized[["SNP", "BETA"]].rename(columns={"BETA": f"BETA_{i}"}),
                on="SNP",
                how="left",
            )
            combined = combined.merge(
                harmonized[["SNP", "SE"]].rename(columns={"SE": f"SE_{i}"}),
                on="SNP",
                how="left",
            )
            combined = combined.merge(
                harmonized[["SNP", "P"]].rename(columns={"P": f"P_{i}"}),
                on="SNP",
                how="left",
            )

        report["after_harmonization"].append(len(combined.dropna(subset=[f"BETA_{i}"])))

    # Keep only SNPs with data from all GWAS
    beta_cols = [f"BETA_{i}" for i in range(len(gwas_list))]
    combined = combined.dropna(subset=beta_cols)

    logger.info(
        "Multi-GWAS harmonization completed",
        final_snps=len(combined),
        snps_in_all=len(combined),
    )

    report["final_snps"] = len(combined)
    return combined.reset_index(drop=True), report
