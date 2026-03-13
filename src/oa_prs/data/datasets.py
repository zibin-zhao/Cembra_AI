"""
PyTorch Dataset classes for genomics data.

Provides efficient data loading for CATN (LD blocks) and traditional
(individual genotypes) approaches.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import structlog

logger = structlog.get_logger(__name__)


class SNPBlockDataset(Dataset):
    """
    PyTorch Dataset for SNP features grouped by LD blocks.

    Loads genotypes organized by LD block structure, ideal for CATN models
    that operate on block-level representations.

    Supports:
    - Variable-length blocks
    - Efficient batching with custom collate function
    - Multiple ancestry populations
    """

    def __init__(
        self,
        genotypes: np.ndarray | str | Path,
        phenotypes: np.ndarray | str | Path,
        ld_blocks: Dict[int, List[int]],
        block_indices: Optional[List[int]] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize SNPBlockDataset.

        Args:
            genotypes: Genotype matrix (n_individuals × n_snps) or path to .npy file
            phenotypes: Binary phenotype vector (n_individuals,) or path to file
            ld_blocks: Dict mapping block_id to list of SNP indices in that block
            block_indices: Optional list of block IDs to use. If None, uses all blocks.
            transform: Optional transform function for preprocessing

        Example:
            dataset = SNPBlockDataset(
                genotypes='genotypes.npy',
                phenotypes='phenotypes.npy',
                ld_blocks={0: [0, 1, 2], 1: [3, 4, 5, 6]},
            )
            loader = DataLoader(
                dataset,
                batch_size=16,
                collate_fn=dataset.collate_fn,
            )
        """
        # Load genotypes
        if isinstance(genotypes, (str, Path)):
            logger.info("Loading genotypes from file", path=str(genotypes))
            self.genotypes = np.load(genotypes)
        else:
            self.genotypes = np.asarray(genotypes, dtype=np.float32)

        # Load phenotypes
        if isinstance(phenotypes, (str, Path)):
            logger.info("Loading phenotypes from file", path=str(phenotypes))
            pheno_data = np.load(phenotypes)
            if pheno_data.ndim > 1:
                pheno_data = pheno_data.flatten()
            self.phenotypes = pheno_data
        else:
            pheno_data = np.asarray(phenotypes)
            if pheno_data.ndim > 1:
                pheno_data = pheno_data.flatten()
            self.phenotypes = pheno_data

        # Validate shapes
        if self.genotypes.shape[0] != self.phenotypes.shape[0]:
            raise ValueError(
                f"Genotype ({self.genotypes.shape[0]}) and phenotype "
                f"({self.phenotypes.shape[0]}) sample mismatch"
            )

        self.ld_blocks = ld_blocks
        self.transform = transform

        # Use provided block indices or all blocks
        if block_indices is None:
            self.block_indices = sorted(ld_blocks.keys())
        else:
            self.block_indices = block_indices

        logger.info(
            "SNPBlockDataset initialized",
            n_individuals=self.genotypes.shape[0],
            n_snps=self.genotypes.shape[1],
            n_blocks=len(self.block_indices),
            case_rate=f"{np.mean(self.phenotypes):.3f}",
        )

    def __len__(self) -> int:
        """Return number of blocks (not individuals)."""
        return len(self.block_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single LD block's data.

        Args:
            idx: Index in block list

        Returns:
            Tuple of:
            - genotypes_block: Tensor of shape (n_individuals, block_size)
            - phenotypes: Tensor of shape (n_individuals,)
            - block_id: Original block ID
        """
        block_id = self.block_indices[idx]
        snp_indices = self.ld_blocks[block_id]

        # Extract genotypes for this block
        block_geno = self.genotypes[:, snp_indices].astype(np.float32)

        if self.transform:
            block_geno = self.transform(block_geno)

        return (
            torch.from_numpy(block_geno),
            torch.from_numpy(self.phenotypes.astype(np.int64)),
            block_id,
        )

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Custom collate function for variable-length blocks.

        Handles variable block sizes without padding.

        Args:
            batch: List of (genotypes, phenotypes, block_id) tuples

        Returns:
            Tuple of:
            - genotypes_list: List of tensors (variable sizes)
            - phenotypes: Stacked tensor (n_individuals,)
        """
        genotypes_list = [item[0] for item in batch]
        phenotypes = batch[0][1]  # All have same phenotype vector

        return genotypes_list, phenotypes


class IndividualGenotypeDataset(Dataset):
    """
    PyTorch Dataset for individual-level genotypes and phenotypes.

    Traditional dataset class for loading individual genotypes paired with
    phenotype labels. Suitable for Phase 3 experiments with actual data.

    Supports:
    - Efficient storage with various compression formats
    - Per-individual sampling
    - Feature normalization
    """

    def __init__(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        snp_ids: Optional[List[str]] = None,
        individual_ids: Optional[List[str]] = None,
        normalize: bool = True,
        dtype: str = "float32",
    ):
        """
        Initialize IndividualGenotypeDataset.

        Args:
            genotype_file: Path to genotype matrix (.npy, .npz, .parquet)
            phenotype_file: Path to phenotype vector or DataFrame
            snp_ids: Optional list of SNP IDs (for metadata)
            individual_ids: Optional list of individual IDs
            normalize: Whether to standardize genotypes (mean 0, std 1)
            dtype: NumPy dtype for genotypes

        Example:
            dataset = IndividualGenotypeDataset(
                genotype_file='genotypes.npy',
                phenotype_file='phenotypes.npy',
                normalize=True,
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        # Load genotypes
        genotype_file = Path(genotype_file)
        if not genotype_file.exists():
            raise FileNotFoundError(f"Genotype file not found: {genotype_file}")

        logger.info("Loading genotypes", path=str(genotype_file))
        if genotype_file.suffix == ".npy":
            self.genotypes = np.load(genotype_file)
        elif genotype_file.suffix == ".npz":
            data = np.load(genotype_file)
            self.genotypes = data["arr_0"] if "arr_0" in data else list(data.values())[0]
        elif genotype_file.suffix == ".parquet":
            geno_df = pd.read_parquet(genotype_file)
            self.genotypes = geno_df.values
        else:
            raise ValueError(f"Unsupported genotype format: {genotype_file.suffix}")

        # Load phenotypes
        phenotype_file = Path(phenotype_file)
        if not phenotype_file.exists():
            raise FileNotFoundError(f"Phenotype file not found: {phenotype_file}")

        logger.info("Loading phenotypes", path=str(phenotype_file))
        if phenotype_file.suffix == ".npy":
            phenotypes = np.load(phenotype_file)
        elif phenotype_file.suffix == ".csv":
            pheno_df = pd.read_csv(phenotype_file, index_col=0)
            phenotypes = pheno_df.values.flatten()
        elif phenotype_file.suffix == ".parquet":
            pheno_df = pd.read_parquet(phenotype_file)
            phenotypes = pheno_df.values.flatten()
        else:
            raise ValueError(f"Unsupported phenotype format: {phenotype_file.suffix}")

        # Validate and store
        if self.genotypes.shape[0] != phenotypes.shape[0]:
            raise ValueError(
                f"Genotype and phenotype sample mismatch: "
                f"{self.genotypes.shape[0]} vs {phenotypes.shape[0]}"
            )

        self.genotypes = self.genotypes.astype(dtype)
        self.phenotypes = phenotypes.astype(np.int64)

        # Normalize genotypes if requested
        if normalize:
            logger.info("Normalizing genotypes")
            means = np.nanmean(self.genotypes, axis=0)
            stds = np.nanstd(self.genotypes, axis=0)
            stds[stds == 0] = 1.0  # Avoid division by zero
            self.genotypes = (self.genotypes - means) / stds

        self.snp_ids = snp_ids
        self.individual_ids = individual_ids

        logger.info(
            "IndividualGenotypeDataset initialized",
            n_individuals=self.genotypes.shape[0],
            n_snps=self.genotypes.shape[1],
            case_rate=f"{np.mean(self.phenotypes):.3f}",
        )

    def __len__(self) -> int:
        """Return number of individuals."""
        return self.genotypes.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get genotype and phenotype for an individual.

        Args:
            idx: Individual index

        Returns:
            Tuple of (genotype, phenotype) as torch tensors
        """
        genotype = torch.from_numpy(self.genotypes[idx])
        phenotype = torch.tensor(self.phenotypes[idx], dtype=torch.int64)
        return genotype, phenotype

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.

        Returns:
            Dict with dataset statistics
        """
        return {
            "n_individuals": self.genotypes.shape[0],
            "n_snps": self.genotypes.shape[1],
            "n_cases": np.sum(self.phenotypes),
            "n_controls": np.sum(self.phenotypes == 0),
            "case_rate": float(np.mean(self.phenotypes)),
            "snp_ids": self.snp_ids,
            "individual_ids": self.individual_ids,
        }
