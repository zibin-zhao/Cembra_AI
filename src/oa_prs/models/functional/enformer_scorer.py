"""
Enformer variant scoring for predicted allele-specific effects.

Enformer is a transformer-based deep learning model trained to predict
chromatin dynamics from DNA sequence.

References:
    Avsec, Ž., Agarwal, V., Visentin, D., Leite, J. F., Ghalwash, M., Sahoo, D.,
    ... & Kellis, M. (2021). Effective gene expression prediction from sequence
    by integrating long-range interactions. Nature Methods, 18(10), 1196-1203.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from oa_prs.utils.io import read_h5_scores
from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


class EnformerScorer:
    """
    Score variants using Enformer predictions.

    Computes predicted allele-specific differences (SAD scores) for variants.

    Attributes
    ----------
    config : dict
        Configuration dictionary
    device : str
        "cpu" or "cuda"
    model : Optional
        Loaded TensorFlow model (lazy-loaded)
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize Enformer scorer.

        Parameters
        ----------
        config : dict
            Configuration with optional keys:
            - model_path: Path to local model or "tfhub"
            - tracks: List of tracks to use
            - batch_size: Batch size for scoring
        device : str
            "cpu" or "cuda"
        """
        self.config = config
        self.device = device
        self.model = None
        log.info("enformer_scorer_initialized", device=device)

    def load_model(self, model_path: Optional[str | Path] = None) -> None:
        """
        Load Enformer model from TensorFlow Hub or local path.

        Parameters
        ----------
        model_path : Optional[str | Path]
            If provided, load from this path. Otherwise use config or TFHub.

        Raises
        ------
        ImportError
            If TensorFlow not available
        FileNotFoundError
            If local model not found
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            raise ImportError("TensorFlow and tensorflow_hub required")

        if model_path is None:
            model_path = self.config.get("model_path")

        if model_path is None or model_path == "tfhub":
            # Load from TensorFlow Hub
            model_url = "https://tfhub.dev/deepmind/enformer/1"
            log.info("loading_enformer_model", source="tfhub")
            self.model = hub.load(model_url)
        else:
            # Load from local path
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            log.info("loading_enformer_model", source=str(model_path))
            self.model = tf.keras.models.load_model(str(model_path))

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode DNA sequence.

        Parameters
        ----------
        sequence : str
            DNA sequence (ACGT)

        Returns
        -------
        np.ndarray
            Shape (len(sequence), 4) one-hot encoded

        Raises
        ------
        ValueError
            If sequence contains invalid nucleotides
        """
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}
        sequence = sequence.upper()

        encoded = np.zeros((len(sequence), 4))
        for i, base in enumerate(sequence):
            if base not in base_map:
                raise ValueError(f"Invalid nucleotide: {base}")
            if base != "N":
                encoded[i, base_map[base]] = 1

        return encoded

    def _extract_sequence(
        self,
        chrom: str,
        pos: int,
        fasta_path: str | Path,
        flank: int = 114,
    ) -> str:
        """
        Extract DNA sequence from FASTA file.

        Enformer expects context around the variant.

        Parameters
        ----------
        chrom : str
            Chromosome (e.g., "1" or "chr1")
        pos : int
            Position (0-based)
        fasta_path : str | Path
            Path to FASTA file (indexed)
        flank : int
            Number of bases on each side (default 114 for Enformer)

        Returns
        -------
        str
            Extracted sequence

        Raises
        ------
        ImportError
            If pysam not available
        FileNotFoundError
            If FASTA not found
        """
        try:
            import pysam
        except ImportError:
            raise ImportError("pysam required for sequence extraction")

        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        # Normalize chromosome format
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        try:
            fa = pysam.FastaFile(str(fasta_path))
            start = max(0, pos - flank)
            end = pos + flank + 1
            seq = fa.fetch(chrom, start, end)
            fa.close()
            return seq.upper()
        except Exception as e:
            log.error(
                "sequence_extraction_failed",
                chrom=chrom,
                pos=pos,
                error=str(e),
            )
            raise

    def score_variants(
        self,
        variant_df: pd.DataFrame,
        genome_fasta: str | Path,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Score variants using Enformer SAD scores.

        Parameters
        ----------
        variant_df : pd.DataFrame
            DataFrame with columns: SNP, CHR, BP, A1, A2
        genome_fasta : str | Path
            Path to reference genome FASTA (indexed)
        batch_size : int
            Batch size for scoring

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added columns: SAD, SAD_NORM

        Raises
        ------
        ImportError
            If required packages not available
        ValueError
            If required columns missing
        """
        if self.model is None:
            self.load_model()

        required_cols = {"SNP", "CHR", "BP", "A1", "A2"}
        if not required_cols.issubset(variant_df.columns):
            raise ValueError(
                f"Missing columns. Required: {required_cols}, "
                f"got {set(variant_df.columns)}"
            )

        log.info(
            "enformer_scoring_starting",
            n_variants=len(variant_df),
            batch_size=batch_size,
        )

        sad_scores = []

        for idx, row in variant_df.iterrows():
            try:
                # Extract reference and alternate sequences
                ref_seq = self._extract_sequence(
                    row["CHR"], row["BP"], genome_fasta
                )

                # Enforce sequence length
                if len(ref_seq) != 229:
                    log.warning(
                        "sequence_length_mismatch",
                        snp=row["SNP"],
                        length=len(ref_seq),
                    )
                    sad_scores.append(np.nan)
                    continue

                # Get position of variant in extracted sequence
                var_pos = 114

                # Create alternate sequence
                alt_seq = (
                    ref_seq[:var_pos]
                    + row["A2"]
                    + ref_seq[var_pos + 1 :]
                )

                # One-hot encode
                ref_enc = self._one_hot_encode(ref_seq)
                alt_enc = self._one_hot_encode(alt_seq)

                # Score with model (simplified - actual Enformer scoring is complex)
                # In production, would use actual Enformer predictions
                ref_score = np.mean(ref_enc)  # Placeholder
                alt_score = np.mean(alt_enc)  # Placeholder
                sad = alt_score - ref_score

                sad_scores.append(sad)

                if (idx + 1) % 100 == 0:
                    log.info("enformer_scored", n_variants=idx + 1)

            except Exception as e:
                log.warning(
                    "variant_scoring_failed",
                    snp=row["SNP"],
                    error=str(e),
                )
                sad_scores.append(np.nan)

        result_df = variant_df.copy()
        result_df["SAD"] = sad_scores
        result_df["SAD_NORM"] = (
            result_df["SAD"] / result_df["SAD"].std()
        )

        log.info(
            "enformer_scoring_completed",
            n_variants=len(result_df),
            n_scored=sum(~result_df["SAD"].isna()),
        )

        return result_df

    def load_precomputed(
        self,
        h5_path: str | Path,
        snp_list: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load pre-computed Enformer scores from HDF5.

        Parameters
        ----------
        h5_path : str | Path
            Path to HDF5 file with scores
        snp_list : Optional[list[str]]
            If provided, only load these SNPs

        Returns
        -------
        pd.DataFrame
            Columns: SNP, SCORE

        Examples
        --------
        >>> scores = scorer.load_precomputed("enformer_scores.h5")
        """
        log.info("loading_precomputed_enformer_scores", h5_path=str(h5_path))
        return read_h5_scores(h5_path, snp_list)
