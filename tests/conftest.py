"""Shared pytest fixtures for OA-PRS Transfer Learning project."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def toy_sumstats():
    """
    Return a toy summary statistics DataFrame.

    Columns: SNP, CHR, BP, A1, A2, BETA, SE, P, N, MAF
    100 rows with mix of chr1-3.
    """
    np.random.seed(42)
    n_snps = 100

    snps = [f"rs{i+1000}" for i in range(n_snps)]
    chrs = np.random.choice([1, 2, 3], size=n_snps)
    bps = np.random.randint(1000000, 100000000, size=n_snps)
    a1 = np.random.choice(['A', 'T', 'G', 'C'], size=n_snps)
    a2 = np.random.choice(['A', 'T', 'G', 'C'], size=n_snps)
    betas = np.random.normal(0, 0.1, size=n_snps)
    ses = np.abs(np.random.normal(0.05, 0.01, size=n_snps))
    ps = np.random.uniform(1e-8, 0.05, size=n_snps)
    ns = np.full(n_snps, 500000)
    mafs = np.random.uniform(0.01, 0.5, size=n_snps)

    df = pd.DataFrame({
        'SNP': snps,
        'CHR': chrs,
        'BP': bps,
        'A1': a1,
        'A2': a2,
        'BETA': betas,
        'SE': ses,
        'P': ps,
        'N': ns,
        'MAF': mafs
    })

    return df


@pytest.fixture
def toy_genotypes():
    """
    Return toy genotypes as numpy array.

    Shape: (50 samples, 100 SNPs)
    Values: float32 dosages in [0, 2]
    """
    np.random.seed(42)
    genotypes = np.random.uniform(0, 2, size=(50, 100)).astype(np.float32)
    return genotypes


@pytest.fixture
def toy_phenotype():
    """
    Return binary toy phenotype.

    Shape: (50,)
    ~15% prevalence
    """
    np.random.seed(42)
    phenotype = np.random.binomial(n=1, p=0.15, size=50).astype(np.float32)
    return phenotype


@pytest.fixture
def toy_ld_matrix():
    """
    Return block-diagonal positive-definite correlation matrix.

    Shape: (100, 100)
    Block structure: three 33-34 blocks
    """
    np.random.seed(42)
    n_snps = 100
    block_size = 33

    ld_matrix = np.zeros((n_snps, n_snps))

    # Create block-diagonal structure
    for block_start in range(0, n_snps, block_size):
        block_end = min(block_start + block_size, n_snps)
        block_len = block_end - block_start

        # Generate random correlation block
        A = np.random.randn(block_len, block_len)
        block = np.corrcoef(A)

        # Ensure positive-definite by adding small diagonal offset
        block += np.eye(block_len) * 0.1

        ld_matrix[block_start:block_end, block_start:block_end] = block

    return ld_matrix


@pytest.fixture
def toy_config():
    """
    Return minimal config dict for OA-PRS pipeline.

    Contains: seed, phenotype, etc.
    """
    config = {
        'seed': 42,
        'phenotype': 'knee_oa',
        'ancestry': 'EUR',
        'n_snps': 100,
        'n_samples': 50,
        'maf_threshold': 0.01,
        'info_threshold': 0.8,
        'device': 'cpu'
    }
    return config


@pytest.fixture
def tmp_output_dir():
    """Return a temporary directory path for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
