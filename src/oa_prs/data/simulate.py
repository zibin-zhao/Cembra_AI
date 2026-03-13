"""
Simulate genotypes and phenotypes from GWAS summary statistics and LD matrices.

Critical for training CATN models when individual-level data is unavailable.
Uses multivariate normal distribution conditioned on observed LD structure.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import structlog
from scipy.stats import norm

logger = structlog.get_logger(__name__)


def simulate_genotypes_from_ld(
    ld_matrix: np.ndarray,
    maf: np.ndarray,
    n_individuals: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate genotypes from LD matrix and allele frequencies.

    Uses multivariate normal approximation to simulate correlated genotypes
    that maintain observed LD structure and MAF.

    Args:
        ld_matrix: LD correlation matrix (n_snps × n_snps)
        maf: Minor allele frequencies (n_snps,)
        n_individuals: Number of individuals to simulate
        seed: Random seed for reproducibility

    Returns:
        Simulated genotypes (n_individuals × n_snps), values in {0, 1, 2}

    Raises:
        ValueError: If LD matrix invalid or maf out of range

    Example:
        genotypes = simulate_genotypes_from_ld(ld_mat, maf_array, n_individuals=5000)
        print(genotypes.shape)  # (5000, n_snps)
        print(np.mean(genotypes, axis=0) / 2)  # Should match MAF
    """
    if seed is not None:
        np.random.seed(seed)

    n_snps = ld_matrix.shape[0]
    if len(maf) != n_snps:
        raise ValueError(f"MAF length ({len(maf)}) != n_snps ({n_snps})")

    if np.any(maf < 0) or np.any(maf > 0.5):
        raise ValueError("MAF must be in [0, 0.5]")

    logger.info(
        "Simulating genotypes from LD",
        n_snps=n_snps,
        n_individuals=n_individuals,
    )

    # For each SNP, compute threshold for converting std normal to 0/1/2
    # P(G=0) = (1-p)^2, P(G=1) = 2p(1-p), P(G=2) = p^2
    # where p is MAF
    p = maf
    thresholds = np.array(
        [
            [norm.ppf((1 - p[i]) ** 2) for i in range(n_snps)],
            [norm.ppf(1 - p[i] ** 2) for i in range(n_snps)],
        ]
    )

    # Generate correlated standard normal variables
    try:
        # Use Cholesky decomposition of LD matrix
        L = np.linalg.cholesky(ld_matrix)
    except np.linalg.LinAlgError:
        logger.warning("LD matrix not positive definite, using eigendecomposition")
        # Fallback: eigendecomposition
        evals, evecs = np.linalg.eigh(ld_matrix)
        evals = np.maximum(evals, 0)
        L = evecs @ np.diag(np.sqrt(evals))

    # Simulate standard normals
    Z = np.random.standard_normal((n_individuals, n_snps))

    # Apply LD correlation structure
    Z_corr = Z @ L.T

    # Convert to genotypes
    genotypes = np.zeros((n_individuals, n_snps), dtype=np.int8)
    for i in range(n_snps):
        # Count how many thresholds each value crosses
        genotypes[:, i] = (Z_corr[:, i] > thresholds[0, i]).astype(int)
        genotypes[:, i] += (Z_corr[:, i] > thresholds[1, i]).astype(int)

    logger.info(
        "Genotype simulation completed",
        shape=genotypes.shape,
        mean_maf=np.mean(np.sum(genotypes, axis=0) / (2 * n_individuals)),
    )

    return genotypes


def simulate_phenotype(
    genotypes: np.ndarray,
    betas: np.ndarray,
    prevalence: float = 0.1,
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate phenotype (binary disease) from genotypes and effect sizes.

    Uses logistic regression model:
    logit(P(disease)) = intercept + sum(beta_i * G_i) + noise

    Args:
        genotypes: Simulated genotypes (n_individuals × n_snps)
        betas: Effect sizes / regression coefficients (n_snps,)
        prevalence: Disease prevalence (proportion with disease)
        noise_scale: Environmental noise standard deviation
        seed: Random seed

    Returns:
        Tuple of:
        - phenotypes: Binary disease status (n_individuals,) in {0, 1}
        - liability_scores: Continuous liability (n_individuals,)

    Example:
        pheno, liability = simulate_phenotype(genotypes, betas, prevalence=0.1)
        print(f"Disease rate: {np.mean(pheno):.3f}")
    """
    if seed is not None:
        np.random.seed(seed)

    n_individuals, n_snps = genotypes.shape
    if len(betas) != n_snps:
        raise ValueError(f"Beta length ({len(betas)}) != n_snps ({n_snps})")

    logger.info(
        "Simulating phenotypes",
        n_individuals=n_individuals,
        n_snps=n_snps,
        prevalence=prevalence,
    )

    # Compute genetic liability (standardized)
    genetic_score = genotypes @ betas
    genetic_score = (genetic_score - np.mean(genetic_score)) / np.std(genetic_score)

    # Add environmental noise
    env_noise = np.random.standard_normal(n_individuals) * noise_scale
    liability = genetic_score + env_noise

    # Set threshold to achieve desired prevalence
    threshold = np.percentile(liability, (1 - prevalence) * 100)

    # Binary phenotype
    phenotypes = (liability > threshold).astype(np.int8)

    realized_prevalence = np.mean(phenotypes)
    logger.info(
        "Phenotype simulation completed",
        realized_prevalence=f"{realized_prevalence:.3f}",
        target_prevalence=prevalence,
    )

    return phenotypes, liability


def create_training_dataset(
    gwas_df: pd.DataFrame,
    ld_blocks: Dict[int, Tuple[int, int]],
    ld_matrices: Dict[int, np.ndarray],
    n_sim: int = 5000,
    ancestry: str = "EUR",
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create full training dataset by simulating genotypes and phenotypes.

    Critical for CATN training when individual-level data unavailable.
    Simulates genotypes with realistic LD structure, computes effects,
    and generates binary phenotypes.

    Args:
        gwas_df: GWAS summary stats with columns: SNP, CHR, BP, A1, A2, BETA, MAF
        ld_blocks: Dict mapping block ID to (start_bp, end_bp)
        ld_matrices: Dict mapping block ID to LD correlation matrix
        n_sim: Number of individuals to simulate
        ancestry: Population ancestry code (for logging)
        config: Optional config dict with:
            - prevalence: Disease prevalence (default 0.1)
            - noise_scale: Environmental noise (default 1.0)
        seed: Random seed

    Returns:
        Tuple of:
        - genotypes: Simulated genotypes (n_individuals × n_snps), dtype int8
        - phenotypes: Binary disease status (n_individuals,), dtype int8
        - snp_info: DataFrame with SNP metadata (SNP, BETA, MAF)

    Raises:
        ValueError: If GWAS data incomplete or LD blocks mismatch

    Example:
        genotypes, phenotypes, snp_info = create_training_dataset(
            gwas_df, ld_blocks, ld_matrices,
            n_sim=5000, ancestry='EUR',
            config={'prevalence': 0.05, 'noise_scale': 1.0}
        )
        print(f"Training set: {len(phenotypes)} individuals, {genotypes.shape[1]} SNPs")
    """
    if config is None:
        config = {}

    prevalence = config.get("prevalence", 0.1)
    noise_scale = config.get("noise_scale", 1.0)

    logger.info(
        "Creating training dataset",
        n_sim=n_sim,
        ancestry=ancestry,
        n_snps=len(gwas_df),
        n_blocks=len(ld_blocks),
        prevalence=prevalence,
    )

    # Validate GWAS data
    required_cols = {"SNP", "BETA", "MAF"}
    if not required_cols.issubset(gwas_df.columns):
        raise ValueError(f"GWAS missing columns: {required_cols - set(gwas_df.columns)}")

    # Collect all genotypes and SNP info
    all_genotypes = []
    all_snp_info = []

    # Simulate genotypes block by block
    for block_id, (start_bp, end_bp) in ld_blocks.items():
        if block_id not in ld_matrices:
            logger.warning(f"LD matrix not found for block {block_id}, skipping")
            continue

        # Get SNPs in this block
        block_gwas = gwas_df[
            (gwas_df["BP"] >= start_bp) & (gwas_df["BP"] <= end_bp)
        ].copy()

        if len(block_gwas) == 0:
            continue

        ld_matrix = ld_matrices[block_id]
        maf = block_gwas["MAF"].values

        # Simulate genotypes for this block
        try:
            block_genotypes = simulate_genotypes_from_ld(
                ld_matrix,
                maf,
                n_individuals=n_sim,
                seed=seed + block_id if seed is not None else None,
            )
            all_genotypes.append(block_genotypes)
            all_snp_info.append(
                block_gwas[["SNP", "BETA", "MAF"]].reset_index(drop=True)
            )

        except Exception as e:
            logger.warning(
                f"Failed to simulate block {block_id}",
                error=str(e),
            )

    # Concatenate all blocks
    if not all_genotypes:
        raise ValueError("No blocks could be simulated")

    genotypes = np.hstack(all_genotypes)
    snp_info = pd.concat(all_snp_info, ignore_index=True)

    logger.info(
        "Genotypes simulated for all blocks",
        shape=genotypes.shape,
        n_unique_snps=len(snp_info),
    )

    # Simulate phenotypes
    betas = snp_info["BETA"].values
    phenotypes, _ = simulate_phenotype(
        genotypes,
        betas,
        prevalence=prevalence,
        noise_scale=noise_scale,
        seed=seed,
    )

    logger.info(
        "Training dataset created",
        n_individuals=len(phenotypes),
        n_snps=genotypes.shape[1],
        n_cases=np.sum(phenotypes),
        n_controls=np.sum(phenotypes == 0),
        ancestry=ancestry,
    )

    return genotypes, phenotypes, snp_info
