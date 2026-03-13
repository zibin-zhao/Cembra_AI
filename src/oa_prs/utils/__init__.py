"""
Utility functions for OA-PRS transfer analysis.
"""

from oa_prs.utils.genetics import (
    allele_match,
    compute_maf,
    flip_alleles,
    is_ambiguous,
)
from oa_prs.utils.io import (
    read_gwas,
    read_h5_scores,
    read_plink_bim,
    write_gwas,
)
from oa_prs.utils.logging_config import get_logger, setup_logging
from oa_prs.utils.reproducibility import (
    compute_file_hash,
    log_environment,
    set_all_seeds,
)
from oa_prs.utils.slurm import (
    check_job_status,
    generate_slurm_header,
    submit_job,
)

__all__ = [
    "flip_alleles",
    "is_ambiguous",
    "compute_maf",
    "allele_match",
    "read_gwas",
    "write_gwas",
    "read_plink_bim",
    "read_h5_scores",
    "setup_logging",
    "get_logger",
    "set_all_seeds",
    "compute_file_hash",
    "log_environment",
    "generate_slurm_header",
    "submit_job",
    "check_job_status",
]
