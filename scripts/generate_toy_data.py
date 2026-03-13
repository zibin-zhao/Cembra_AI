#!/usr/bin/env python3
"""
Generate Synthetic Toy Data for Pipeline Testing.

Creates small, realistic-looking GWAS summary statistics, LD matrices,
genotype files, and phenotype data to enable end-to-end pipeline testing
without requiring real data.

Usage:
    python scripts/generate_toy_data.py --output-dir data/toy/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def generate_gwas_sumstats(
    n_snps: int = 1000,
    ancestry: str = "EUR",
    n_gwas: int = 400000,
    n_causal: int = 50,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate synthetic GWAS summary statistics.

    Simulates realistic betas with a mixture of causal and null SNPs,
    appropriate MAFs, and p-values derived from beta/SE.
    """
    rng = np.random.RandomState(seed)

    snp_ids = [f"rs{100000 + i}" for i in range(n_snps)]
    chroms = rng.choice(range(1, 23), size=n_snps)
    bps = np.sort(rng.randint(1_000_000, 250_000_000, size=n_snps))
    alleles = list("ACGT")
    a1 = [alleles[i % 4] for i in range(n_snps)]
    a2 = [alleles[(i + 2) % 4] for i in range(n_snps)]

    # MAF: realistic distribution (many rare, some common)
    maf = rng.beta(1.5, 5, size=n_snps)
    maf = np.clip(maf, 0.01, 0.49)

    # Effect sizes: mixture of null + causal
    betas = np.zeros(n_snps)
    causal_idx = rng.choice(n_snps, size=n_causal, replace=False)
    betas[causal_idx] = rng.normal(0, 0.05, size=n_causal)

    # SE depends on MAF and sample size
    se = 1.0 / np.sqrt(2 * maf * (1 - maf) * n_gwas)
    se *= rng.uniform(0.9, 1.1, size=n_snps)  # Add noise

    # Observed beta = true beta + noise
    beta_obs = betas + rng.normal(0, se)
    z_scores = beta_obs / se
    from scipy.stats import norm
    p_values = 2 * norm.sf(np.abs(z_scores))

    df = pd.DataFrame({
        "SNP": snp_ids,
        "CHR": chroms,
        "BP": bps,
        "A1": a1,
        "A2": a2,
        "BETA": beta_obs,
        "SE": se,
        "P": p_values,
        "MAF": maf,
        "N": n_gwas,
    })

    logger.info("Generated %s GWAS sumstats: %d SNPs, %d causal", ancestry, n_snps, n_causal)
    return df


def generate_ld_matrix(n_snps: int = 1000, n_blocks: int = 20, seed: int = SEED) -> np.ndarray:
    """
    Generate a synthetic block-diagonal LD matrix.

    Each LD block has moderate correlations; between blocks, correlations are ~0.
    """
    rng = np.random.RandomState(seed)
    ld = np.eye(n_snps)
    block_size = n_snps // n_blocks

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_snps)
        block_n = end - start
        # Random positive-definite block
        A = rng.randn(block_n, block_n) * 0.3
        block_cov = A @ A.T + np.eye(block_n) * 0.5
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(block_cov))
        block_corr = block_cov / np.outer(d, d)
        np.fill_diagonal(block_corr, 1.0)
        ld[start:end, start:end] = block_corr

    logger.info("Generated LD matrix: %d × %d, %d blocks", n_snps, n_snps, n_blocks)
    return ld


def generate_genotype_data(
    n_individuals: int = 200,
    n_snps: int = 1000,
    maf: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic genotype data (0/1/2 dosages) and phenotype.

    Returns:
        genotype: [n_individuals, n_snps] dosage matrix.
        fam: PLINK .fam-like DataFrame.
        bim: PLINK .bim-like DataFrame.
    """
    rng = np.random.RandomState(seed)

    if maf is None:
        maf = rng.beta(1.5, 5, size=n_snps)
        maf = np.clip(maf, 0.01, 0.49)

    # Genotypes: binomial(2, maf) for each SNP
    genotype = np.column_stack([
        rng.binomial(2, p=f, size=n_individuals) for f in maf
    ]).astype(np.float32)

    # fam file
    fam = pd.DataFrame({
        "FID": [f"FAM{i:04d}" for i in range(n_individuals)],
        "IID": [f"IND{i:04d}" for i in range(n_individuals)],
        "PAT": 0,
        "MAT": 0,
        "SEX": rng.choice([1, 2], size=n_individuals),
        "PHENO": -9,
    })

    # bim file
    bim = pd.DataFrame({
        "CHR": rng.choice(range(1, 23), size=n_snps),
        "SNP": [f"rs{100000 + i}" for i in range(n_snps)],
        "CM": 0.0,
        "BP": np.sort(rng.randint(1_000_000, 250_000_000, size=n_snps)),
        "A1": ["A" if i % 2 == 0 else "C" for i in range(n_snps)],
        "A2": ["G" if i % 2 == 0 else "T" for i in range(n_snps)],
    })

    logger.info("Generated genotype data: %d individuals × %d SNPs", n_individuals, n_snps)
    return genotype, fam, bim


def generate_phenotype(
    genotype: np.ndarray,
    n_causal: int = 50,
    prevalence: float = 0.15,
    seed: int = SEED,
) -> np.ndarray:
    """
    Generate binary phenotype from genotype using logistic model.

    Args:
        genotype: [n_individuals, n_snps] dosage matrix.
        n_causal: Number of causal SNPs.
        prevalence: Target disease prevalence.

    Returns:
        Binary phenotype array [n_individuals].
    """
    rng = np.random.RandomState(seed)
    n_individuals, n_snps = genotype.shape

    # Select causal SNPs and assign effects
    causal_idx = rng.choice(n_snps, size=n_causal, replace=False)
    betas = np.zeros(n_snps)
    betas[causal_idx] = rng.normal(0, 0.3, size=n_causal)

    # Genetic liability
    liability = genotype @ betas

    # Set intercept to achieve target prevalence
    from scipy.optimize import brentq
    from scipy.special import expit

    def prevalence_at_intercept(intercept):
        return expit(intercept + liability).mean() - prevalence

    try:
        intercept = brentq(prevalence_at_intercept, -10, 10)
    except ValueError:
        intercept = np.log(prevalence / (1 - prevalence))

    prob = expit(intercept + liability)
    phenotype = rng.binomial(1, prob)

    logger.info(
        "Generated phenotype: prevalence=%.3f, n_cases=%d, n_controls=%d",
        phenotype.mean(), phenotype.sum(), len(phenotype) - phenotype.sum(),
    )
    return phenotype


def generate_annotations(n_snps: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic functional annotation scores."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "SNP": [f"rs{100000 + i}" for i in range(n_snps)],
        "enformer_score": rng.exponential(0.5, size=n_snps),
        "turf_score": rng.beta(2, 5, size=n_snps),
        "conservation": rng.beta(1, 3, size=n_snps),
        "regulomedb_rank": rng.randint(1, 8, size=n_snps),
    })


def generate_all(
    output_dir: Path,
    n_snps: int = 1000,
    n_individuals: int = 200,
    seed: int = SEED,
) -> None:
    """Generate all toy data files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GWAS summary stats
    gwas_eur = generate_gwas_sumstats(n_snps, "EUR", n_gwas=400000, seed=seed)
    gwas_eas = generate_gwas_sumstats(n_snps, "EAS", n_gwas=50000, seed=seed + 1)
    gwas_eur.to_csv(output_dir / "toy_gwas_eur.tsv", sep="\t", index=False)
    gwas_eas.to_csv(output_dir / "toy_gwas_eas.tsv", sep="\t", index=False)

    # LD matrices
    ld_eur = generate_ld_matrix(n_snps, seed=seed)
    ld_eas = generate_ld_matrix(n_snps, seed=seed + 1)
    np.savez_compressed(output_dir / "toy_ld_eur.npz", ld=ld_eur)
    np.savez_compressed(output_dir / "toy_ld_eas.npz", ld=ld_eas)

    # Genotype and phenotype
    genotype, fam, bim = generate_genotype_data(n_individuals, n_snps, seed=seed)
    phenotype = generate_phenotype(genotype, seed=seed)
    fam["PHENO"] = phenotype

    # Save as TSV (PLINK binary writing requires plink; we save readable format)
    np.save(output_dir / "toy_genotype.npy", genotype)
    fam.to_csv(output_dir / "toy_phenotype.tsv", sep="\t", index=False)
    bim.to_csv(output_dir / "toy_bim.tsv", sep="\t", index=False)

    # Annotations
    annotations = generate_annotations(n_snps, seed=seed)
    annotations.to_csv(output_dir / "toy_annotations.tsv", sep="\t", index=False)

    # README
    readme = f"""# Toy Data for Pipeline Testing

Generated with `scripts/generate_toy_data.py`

## Contents
- `toy_gwas_eur.tsv`: Simulated EUR GWAS summary statistics ({n_snps} SNPs, N=400,000)
- `toy_gwas_eas.tsv`: Simulated EAS GWAS summary statistics ({n_snps} SNPs, N=50,000)
- `toy_ld_eur.npz`: EUR LD matrix ({n_snps} × {n_snps}, block-diagonal)
- `toy_ld_eas.npz`: EAS LD matrix
- `toy_genotype.npy`: Genotype dosage matrix ({n_individuals} × {n_snps})
- `toy_phenotype.tsv`: Sample info with binary phenotype (knee OA)
- `toy_bim.tsv`: SNP information (like PLINK .bim)
- `toy_annotations.tsv`: Functional annotation scores

## Parameters
- Seed: {seed}
- Causal SNPs: 50
- Prevalence: ~15%
- LD blocks: 20
"""
    (output_dir / "README.md").write_text(readme)
    logger.info("All toy data written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate toy data for pipeline testing")
    parser.add_argument("--output-dir", default="data/toy/", help="Output directory")
    parser.add_argument("--n-snps", type=int, default=1000, help="Number of SNPs")
    parser.add_argument("--n-individuals", type=int, default=200, help="Number of individuals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_all(Path(args.output_dir), args.n_snps, args.n_individuals, args.seed)
