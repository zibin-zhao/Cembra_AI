"""
Genetic utility functions for allele manipulation and QC.
"""

from typing import Literal

import pandas as pd

from oa_prs.constants import COMPLEMENT, VALID_ALLELES


def flip_alleles(a1: str, a2: str) -> tuple[str, str]:
    """
    Flip alleles to their complement (A↔T, C↔G).

    Parameters
    ----------
    a1 : str
        First allele
    a2 : str
        Second allele

    Returns
    -------
    tuple[str, str]
        Complemented alleles (a1_complement, a2_complement)

    Raises
    ------
    ValueError
        If alleles are not valid DNA bases

    Examples
    --------
    >>> flip_alleles("A", "G")
    ('T', 'C')
    """
    if a1 not in VALID_ALLELES or a2 not in VALID_ALLELES:
        raise ValueError(
            f"Invalid alleles: {a1}, {a2}. Must be A, T, C, or G"
        )
    return COMPLEMENT[a1], COMPLEMENT[a2]


def is_ambiguous(a1: str, a2: str) -> bool:
    """
    Check if SNP is strand-ambiguous (A/T or C/G).

    Ambiguous SNPs cannot be reliably flipped strand.

    Parameters
    ----------
    a1 : str
        First allele
    a2 : str
        Second allele

    Returns
    -------
    bool
        True if SNP is ambiguous

    Examples
    --------
    >>> is_ambiguous("A", "T")
    True
    >>> is_ambiguous("A", "G")
    False
    """
    return frozenset({a1, a2}) in {
        frozenset({"A", "T"}),
        frozenset({"C", "G"}),
    }


def compute_maf(freq: float) -> float:
    """
    Compute minor allele frequency.

    Parameters
    ----------
    freq : float
        Allele frequency (0-1)

    Returns
    -------
    float
        Minor allele frequency (0-0.5)

    Examples
    --------
    >>> compute_maf(0.05)
    0.05
    >>> compute_maf(0.95)
    0.05
    """
    if not 0 <= freq <= 1:
        raise ValueError(f"Frequency must be between 0 and 1, got {freq}")
    return min(freq, 1 - freq)


def allele_match(
    a1: str,
    a2: str,
    ref_a1: str,
    ref_a2: str,
) -> Literal["exact", "flip", "complement", "reverse_complement", "no_match"]:
    """
    Determine allele matching type between two SNP representations.

    Checks multiple matching scenarios:
    - exact: a1==ref_a1 and a2==ref_a2
    - flip: a1==ref_a2 and a2==ref_a1
    - complement: a1==complement(ref_a1) and a2==complement(ref_a2)
    - reverse_complement: complement(a1)==ref_a2 and complement(a2)==ref_a1
    - no_match: no match found

    Parameters
    ----------
    a1 : str
        First allele from query
    a2 : str
        Second allele from query
    ref_a1 : str
        First allele from reference
    ref_a2 : str
        Second allele from reference

    Returns
    -------
    Literal["exact", "flip", "complement", "reverse_complement", "no_match"]
        Type of allele match

    Examples
    --------
    >>> allele_match("A", "G", "A", "G")
    'exact'
    >>> allele_match("G", "A", "A", "G")
    'flip'
    >>> allele_match("T", "C", "A", "G")
    'complement'
    """
    try:
        comp_a1, comp_a2 = flip_alleles(a1, a2)
        comp_ref_a1, comp_ref_a2 = flip_alleles(ref_a1, ref_a2)
    except ValueError:
        return "no_match"

    if a1 == ref_a1 and a2 == ref_a2:
        return "exact"
    elif a1 == ref_a2 and a2 == ref_a1:
        return "flip"
    elif comp_a1 == ref_a1 and comp_a2 == ref_a2:
        return "complement"
    elif comp_a1 == ref_a2 and comp_a2 == ref_a1:
        return "reverse_complement"
    else:
        return "no_match"


def harmonize_alleles(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    snp_col: str = "SNP",
    a1_col: str = "A1",
    a2_col: str = "A2",
) -> pd.DataFrame:
    """
    Harmonize alleles in dataframe against reference.

    Adds columns:
    - MATCH_TYPE: type of match (exact/flip/complement/reverse_complement/no_match)
    - NEED_FLIP_BETA: whether to flip BETA sign
    - HARMONIZED: boolean indicating successful harmonization

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with SNP and allele columns
    ref_df : pd.DataFrame
        Reference dataframe with SNP and allele columns
    snp_col : str
        Name of SNP ID column
    a1_col : str
        Name of first allele column
    a2_col : str
        Name of second allele column

    Returns
    -------
    pd.DataFrame
        Input dataframe with harmonization columns added

    Raises
    ------
    KeyError
        If required columns not found
    """
    if snp_col not in df.columns or a1_col not in df.columns or a2_col not in df.columns:
        raise KeyError(
            f"Required columns not found: {snp_col}, {a1_col}, {a2_col}"
        )

    # Merge with reference
    merged = df.merge(
        ref_df[[snp_col, a1_col, a2_col]],
        on=snp_col,
        suffixes=("", "_ref"),
        how="left",
    )

    # Get match types
    match_types = []
    for _, row in merged.iterrows():
        if pd.isna(row.get(f"{a1_col}_ref")):
            match_types.append("no_match")
        else:
            match_types.append(
                allele_match(
                    row[a1_col],
                    row[a2_col],
                    row[f"{a1_col}_ref"],
                    row[f"{a2_col}_ref"],
                )
            )

    merged["MATCH_TYPE"] = match_types
    merged["NEED_FLIP_BETA"] = merged["MATCH_TYPE"].isin(
        ["flip", "reverse_complement"]
    )
    merged["HARMONIZED"] = merged["MATCH_TYPE"] != "no_match"

    return merged
