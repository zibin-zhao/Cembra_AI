"""
Global constants: column schemas, file naming conventions, and genetic reference values.
"""

from dataclasses import dataclass

# =============================================================================
# Standardized GWAS column schema (all input GWAS are converted to this)
# =============================================================================
GWAS_COLUMNS = ["SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P", "MAF", "N"]

GWAS_DTYPES = {
    "SNP": str,
    "CHR": "int8",
    "BP": "int64",
    "A1": str,
    "A2": str,
    "BETA": "float64",
    "SE": "float64",
    "P": "float64",
    "MAF": "float64",
    "N": "int64",
}

# =============================================================================
# Valid alleles and complement mapping
# =============================================================================
VALID_ALLELES = {"A", "T", "C", "G"}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# Ambiguous SNPs (A/T or C/G) — these are strand-ambiguous
AMBIGUOUS_PAIRS = {frozenset({"A", "T"}), frozenset({"C", "G"})}

# =============================================================================
# Chromosomes
# =============================================================================
AUTOSOMES = list(range(1, 23))

# =============================================================================
# Population codes (1000 Genomes)
# =============================================================================

@dataclass(frozen=True)
class Population:
    code: str
    name: str
    superpop: str
    description: str


POPULATIONS = {
    "EUR": Population("EUR", "European", "EUR", "CEU+TSI+FIN+GBR+IBS"),
    "EAS": Population("EAS", "East Asian", "EAS", "CHB+CHS+CDX+JPT+KHV"),
    "AFR": Population("AFR", "African", "AFR", "YRI+LWK+GWD+MSL+ESN+ASW+ACB"),
    "SAS": Population("SAS", "South Asian", "SAS", "GIH+PJL+BEB+STU+ITU"),
    "AMR": Population("AMR", "Admixed American", "AMR", "MXL+PUR+CLM+PEL"),
}

# Sub-populations for Hong Kong Chinese target
HK_CHINESE_SUBPOPS = ["CHB", "CHS"]  # Han Chinese Beijing + Han Chinese South

# =============================================================================
# QC thresholds
# =============================================================================
DEFAULT_QC = {
    "maf_threshold": 0.01,        # Minimum MAF
    "info_threshold": 0.8,        # Minimum INFO score (if available)
    "hwe_p_threshold": 1e-6,      # HWE p-value threshold
    "max_missing_rate": 0.05,     # Maximum per-SNP missingness
    "remove_ambiguous": True,     # Remove strand-ambiguous SNPs
    "remove_duplicates": True,    # Remove duplicate SNP IDs
    "remove_indels": True,        # Remove insertion/deletion variants
}

# =============================================================================
# Phenotype codes
# =============================================================================
PHENOTYPES = {
    "knee_oa": {
        "name": "Knee Osteoarthritis",
        "icd10": ["M17"],
        "description": "Primary: gonarthrosis (knee OA)",
    },
    "hip_oa": {
        "name": "Hip Osteoarthritis",
        "icd10": ["M16"],
        "description": "Coxarthrosis (hip OA)",
    },
    "any_oa": {
        "name": "Any Osteoarthritis",
        "icd10": ["M15", "M16", "M17", "M18", "M19"],
        "description": "Any OA diagnosis",
    },
    "thr": {
        "name": "Total Hip Replacement",
        "opcs4": ["W37", "W38", "W39"],
        "description": "Surgical endpoint: THR",
    },
    "tkr": {
        "name": "Total Knee Replacement",
        "opcs4": ["W40", "W41", "W42"],
        "description": "Surgical endpoint: TKR",
    },
}

# =============================================================================
# File naming conventions
# =============================================================================
def gwas_filename(ancestry: str, phenotype: str, suffix: str = "harmonized") -> str:
    """Generate standardized GWAS filename."""
    return f"gwas_{ancestry.lower()}_{phenotype}_{suffix}.parquet"


def prs_weights_filename(method: str, ancestry: str, chrom: int | None = None) -> str:
    """Generate PRS weights filename."""
    base = f"prs_{method}_{ancestry.lower()}"
    if chrom is not None:
        return f"{base}_chr{chrom}.tsv"
    return f"{base}_all.tsv"
