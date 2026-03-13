"""
LD matrix and block utilities.

Load LD matrices in various formats, compute LD blocks, and assign SNPs to blocks.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def load_ld_matrix(
    path: str | Path,
    chrom: int,
    format: str = "auto",
) -> np.ndarray:
    """
    Load LD matrix from file.

    Supports multiple formats:
    - .npz: Sparse or dense numpy arrays (load with np.load)
    - .npy: Dense numpy array
    - .h5/.hdf5: HDF5 format (requires h5py)

    Args:
        path: Path to LD matrix file
        chrom: Chromosome number (for reference)
        format: File format ('auto' detects from extension)

    Returns:
        LD correlation matrix as 2D numpy array (n_snps × n_snps)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format unsupported or file corrupt

    Example:
        ld_mat = load_ld_matrix('1kg_ld_eur_chr1.npz', chrom=1)
        print(ld_mat.shape)  # (n_snps, n_snps)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LD matrix not found: {path}")

    # Auto-detect format
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npz":
            format = "npz"
        elif suffix == ".npy":
            format = "npy"
        elif suffix in [".h5", ".hdf5"]:
            format = "hdf5"
        else:
            raise ValueError(f"Unknown file format: {suffix}")

    logger.info(
        "Loading LD matrix",
        path=str(path),
        chrom=chrom,
        format=format,
    )

    try:
        if format == "npz":
            data = np.load(path, allow_pickle=False)
            # Handle both sparse and dense formats
            if "data" in data:
                # Sparse format
                logger.warning("Sparse LD matrix detected, converting to dense")
                from scipy.sparse import csr_matrix
                ld_matrix = csr_matrix(
                    (data["data"], data["indices"], data["indptr"]),
                    shape=data["shape"],
                ).toarray()
            elif "arr_0" in data:
                ld_matrix = data["arr_0"]
            else:
                # Try first array
                ld_matrix = data[list(data.files)[0]]

        elif format == "npy":
            ld_matrix = np.load(path, allow_pickle=False)

        elif format == "hdf5":
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py required for HDF5 format")

            with h5py.File(path, "r") as f:
                # Try common dataset names
                if "ld_matrix" in f:
                    ld_matrix = f["ld_matrix"][:]
                elif "LD" in f:
                    ld_matrix = f["LD"][:]
                else:
                    # Use first dataset
                    key = list(f.keys())[0]
                    ld_matrix = f[key][:]

        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        raise ValueError(f"Failed to load LD matrix: {e}")

    logger.info(
        "LD matrix loaded",
        shape=ld_matrix.shape,
        dtype=ld_matrix.dtype,
    )

    return ld_matrix


def compute_ld_blocks(
    bim_file: str | Path,
    method: str = "fixed_mb",
    window_mb: float = 1.0,
    min_snps: int = 2,
) -> Dict[int, Tuple[int, int]]:
    """
    Compute or load LD blocks from a PLINK .bim file.

    Creates LD blocks using a sliding window or other method.

    Args:
        bim_file: Path to PLINK .bim file (SNP, CHR, CM, BP, A1, A2)
        method: Block definition method:
            - 'fixed_mb': Fixed sliding window (window_mb)
            - 'fixed_snps': Fixed number of SNPs
        window_mb: Window size in Mb (for fixed_mb method)
        min_snps: Minimum SNPs per block

    Returns:
        Dict mapping block ID to tuple (start_bp, end_bp)

    Raises:
        FileNotFoundError: If .bim file doesn't exist
        ValueError: If file format invalid

    Example:
        blocks = compute_ld_blocks('data.bim', method='fixed_mb', window_mb=1.0)
        for block_id, (start, end) in blocks.items():
            print(f"Block {block_id}: {start}-{end}")
    """
    bim_file = Path(bim_file)
    if not bim_file.exists():
        raise FileNotFoundError(f".bim file not found: {bim_file}")

    logger.info(
        "Computing LD blocks",
        bim_file=str(bim_file),
        method=method,
        window_mb=window_mb,
    )

    # Load .bim file
    # Format: CHR, SNP, CM, BP, A1, A2
    try:
        bim_df = pd.read_csv(
            bim_file,
            sep=r"\s+",
            header=None,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
            dtype={"CHR": int, "BP": int},
        )
    except Exception as e:
        raise ValueError(f"Failed to parse .bim file: {e}")

    blocks = {}
    block_id = 0

    # Process each chromosome separately
    for chrom in sorted(bim_df["CHR"].unique()):
        chrom_df = bim_df[bim_df["CHR"] == chrom].sort_values("BP")

        if len(chrom_df) < min_snps:
            logger.warning(f"Chromosome {chrom} has fewer than {min_snps} SNPs, skipping")
            continue

        if method == "fixed_mb":
            window_bp = int(window_mb * 1e6)

            # Create blocks with sliding window
            for i in range(0, len(chrom_df), max(1, len(chrom_df) // 10)):
                start_bp = chrom_df.iloc[i]["BP"]
                end_bp = min(start_bp + window_bp, chrom_df["BP"].max())

                # Get SNPs in this window
                snps_in_block = chrom_df[
                    (chrom_df["BP"] >= start_bp) & (chrom_df["BP"] <= end_bp)
                ]

                if len(snps_in_block) >= min_snps:
                    blocks[block_id] = (start_bp, end_bp)
                    block_id += 1

        elif method == "fixed_snps":
            # Create blocks with fixed number of SNPs
            n_snps_per_block = window_mb  # Repurpose for SNP count
            for i in range(0, len(chrom_df), int(n_snps_per_block)):
                snps_in_block = chrom_df.iloc[i : i + int(n_snps_per_block)]
                if len(snps_in_block) >= min_snps:
                    start_bp = snps_in_block["BP"].min()
                    end_bp = snps_in_block["BP"].max()
                    blocks[block_id] = (start_bp, end_bp)
                    block_id += 1

    logger.info("LD blocks computed", num_blocks=len(blocks), chromosomes=len(set(bim_df["CHR"])))
    return blocks


def get_block_snps(
    ld_blocks: Dict[int, Tuple[int, int]],
    snp_positions: pd.DataFrame,
) -> Dict[int, List[str]]:
    """
    Assign SNPs to LD blocks based on genomic position.

    Args:
        ld_blocks: Dict mapping block ID to (start_bp, end_bp)
        snp_positions: DataFrame with columns: SNP, CHR, BP

    Returns:
        Dict mapping block ID to list of SNP IDs in that block

    Example:
        block_snps = get_block_snps(blocks, gwas_df[['SNP', 'CHR', 'BP']])
        for block_id, snps in block_snps.items():
            print(f"Block {block_id}: {len(snps)} SNPs")
    """
    if "SNP" not in snp_positions.columns or "BP" not in snp_positions.columns:
        raise ValueError("snp_positions must have SNP and BP columns")

    logger.info(
        "Assigning SNPs to LD blocks",
        num_blocks=len(ld_blocks),
        num_snps=len(snp_positions),
    )

    block_snps = {block_id: [] for block_id in ld_blocks.keys()}

    for block_id, (start_bp, end_bp) in ld_blocks.items():
        snps_in_range = snp_positions[
            (snp_positions["BP"] >= start_bp) & (snp_positions["BP"] <= end_bp)
        ]
        block_snps[block_id] = snps_in_range["SNP"].tolist()

    # Log statistics
    total_assigned = sum(len(snps) for snps in block_snps.values())
    logger.info(
        "SNP assignment completed",
        total_assigned=total_assigned,
        unassigned=len(snp_positions) - total_assigned,
        avg_snps_per_block=total_assigned / len(ld_blocks) if ld_blocks else 0,
    )

    return block_snps
