"""
Data processing module for genomics pipeline.

Provides utilities for downloading, quality control, harmonization, standardization,
and simulation of GWAS summary statistics and LD reference panels.
"""

from oa_prs.data.download import DataDownloader
from oa_prs.data.qc import run_qc
from oa_prs.data.harmonize import harmonize_gwas, harmonize_multi_gwas
from oa_prs.data.standardize import standardize_gwas
from oa_prs.data.ld_utils import load_ld_matrix, compute_ld_blocks, get_block_snps
from oa_prs.data.simulate import (
    simulate_genotypes_from_ld,
    simulate_phenotype,
    create_training_dataset,
)
from oa_prs.data.datasets import SNPBlockDataset, IndividualGenotypeDataset

__all__ = [
    "DataDownloader",
    "run_qc",
    "harmonize_gwas",
    "harmonize_multi_gwas",
    "standardize_gwas",
    "load_ld_matrix",
    "compute_ld_blocks",
    "get_block_snps",
    "simulate_genotypes_from_ld",
    "simulate_phenotype",
    "create_training_dataset",
    "SNPBlockDataset",
    "IndividualGenotypeDataset",
]
