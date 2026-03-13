"""
CATN Model Inference and Prediction.

This module provides utilities for loading trained CATN models and performing
inference on new genomic data.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import structlog

from .catn_model import CrossAncestryTransferNet

log = structlog.get_logger(__name__)


def load_checkpoint(
    path: str, device: torch.device = torch.device("cpu")
) -> Tuple[CrossAncestryTransferNet, Dict[str, Any]]:
    """
    Load a saved CATN checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load model onto. Default: CPU

    Returns:
        Tuple of (model, config).

    Raises:
        FileNotFoundError: If checkpoint file not found.
        KeyError: If checkpoint is corrupted.
    """
    try:
        checkpoint = torch.load(path, map_location=device)
    except FileNotFoundError as e:
        log.error("Checkpoint file not found", path=path)
        raise

    # Extract config and initialize model
    config = checkpoint.get("config")
    if config is None:
        raise KeyError("Checkpoint does not contain 'config' key")

    model = CrossAncestryTransferNet(config).to(device)

    # Load state dict
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        log.error("Error loading model state dict", error=str(e))
        raise

    log.info("Loaded checkpoint successfully", path=path)

    return model, config


class CATNPredictor:
    """
    Inference interface for trained CATN models.

    Provides methods for:
    - Predicting risk probabilities on new genomic data
    - Extracting learned SNP weights
    - Computing individual-level risk scores

    Args:
        model_path: Path to trained CATN checkpoint.
        config: Model configuration (if not in checkpoint).
        device: Device for inference ('cuda' or 'cpu'). Default: 'cpu'
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize CATN predictor.

        Args:
            model_path: Path to checkpoint.
            config: Optional config (overrides checkpoint config).
            device: Device for inference.
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load checkpoint
        self.model, checkpoint_config = load_checkpoint(model_path, device)

        # Use provided config or checkpoint config
        self.config = config if config is not None else checkpoint_config

        # Set model to eval mode
        self.model.eval()

        log.info(
            "Initialized CATN predictor",
            model_path=model_path,
            device=str(device),
        )

    def predict(
        self,
        snp_features: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
        return_probs: bool = True,
    ) -> torch.Tensor:
        """
        Predict risk on new genomic data.

        Args:
            snp_features: Per-SNP features [batch, max_snps, input_dim].
            block_indices: LD block assignments [batch, max_snps].
            block_masks: SNP validity masks [batch, max_snps].
            return_probs: Whether to return probabilities or logits.
                Default: True

        Returns:
            Risk predictions [batch, 1].
        """
        # Move inputs to device
        snp_features = snp_features.to(self.device)
        block_indices = block_indices.to(self.device)
        block_masks = block_masks.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.model(
                snp_features=snp_features,
                block_indices=block_indices,
                block_masks=block_masks,
            )

            logits = output["risk_logits"]

            # Convert to probabilities
            if return_probs:
                probs = torch.sigmoid(logits)
                return probs
            else:
                return logits

    def predict_batch(
        self,
        snp_features_batch: torch.Tensor,
        block_indices_batch: torch.Tensor,
        block_masks_batch: torch.Tensor,
        batch_size: int = 32,
        return_probs: bool = True,
    ) -> torch.Tensor:
        """
        Predict on batches of data.

        Useful for memory efficiency with large datasets.

        Args:
            snp_features_batch: Features [n_samples, max_snps, input_dim].
            block_indices_batch: Block indices [n_samples, max_snps].
            block_masks_batch: Block masks [n_samples, max_snps].
            batch_size: Processing batch size. Default: 32
            return_probs: Return probabilities or logits. Default: True

        Returns:
            Risk predictions [n_samples, 1].
        """
        n_samples = snp_features_batch.shape[0]
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)

            batch_features = snp_features_batch[i:batch_end]
            batch_indices = block_indices_batch[i:batch_end]
            batch_masks = block_masks_batch[i:batch_end]

            batch_preds = self.predict(
                batch_features, batch_indices, batch_masks, return_probs
            )
            predictions.append(batch_preds.cpu())

        return torch.cat(predictions, dim=0)

    def extract_weights(
        self,
        snp_features: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
        snp_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Extract learned SNP importance weights.

        Args:
            snp_features: Per-SNP features [batch, max_snps, input_dim].
            block_indices: LD block assignments [batch, max_snps].
            block_masks: SNP validity masks [batch, max_snps].
            snp_ids: Optional SNP identifiers. Default: None

        Returns:
            DataFrame with columns: snp_id, weight, block_id
        """
        # Move inputs to device
        snp_features = snp_features.to(self.device)
        block_indices = block_indices.to(self.device)
        block_masks = block_masks.to(self.device)

        with torch.no_grad():
            # Extract weights (uses first sample if batch > 1)
            weights = self.model.extract_snp_weights(
                snp_features, block_indices, block_masks
            )

        # Convert to numpy
        weights_np = weights[0].cpu().numpy()
        block_indices_np = block_indices[0].cpu().numpy()
        block_masks_np = block_masks[0].cpu().numpy()

        # Create dataframe
        valid_mask = block_masks_np.astype(bool)
        weights_valid = weights_np[valid_mask]
        blocks_valid = block_indices_np[valid_mask]

        df_data = {
            "weight": weights_valid,
            "block_id": blocks_valid,
        }

        if snp_ids is not None:
            snp_ids_valid = [snp_ids[i] for i in range(len(snp_ids)) if valid_mask[i]]
            df_data["snp_id"] = snp_ids_valid

        df = pd.DataFrame(df_data)

        return df.sort_values("weight", ascending=False).reset_index(drop=True)

    def predict_individual(
        self,
        genotype_matrix: np.ndarray,
        snp_info: pd.DataFrame,
        ld_block_map: Dict[int, list],
        features_generator,
    ) -> np.ndarray:
        """
        Predict risk for individuals given genotype data.

        Args:
            genotype_matrix: Genotype matrix [n_individuals, n_snps].
                Values: 0, 1, 2 (alternate alleles).
            snp_info: DataFrame with SNP information (must have columns:
                'snp_id', 'position', 'allele0', 'allele1').
            ld_block_map: Dictionary mapping block_id -> list of SNP indices.
            features_generator: Callable that generates features from genotypes.
                Signature: features_generator(genotypes) -> [batch, n_snps, input_dim]

        Returns:
            Risk predictions [n_individuals].
        """
        n_individuals = genotype_matrix.shape[0]

        # Generate features
        features = features_generator(genotype_matrix)

        # Prepare block indices and masks
        max_snps = features.shape[1]
        block_indices = np.zeros((n_individuals, max_snps), dtype=np.int64)
        block_masks = np.zeros((n_individuals, max_snps), dtype=np.float32)

        # Map SNPs to LD blocks
        for block_id, snp_indices in ld_block_map.items():
            for snp_idx in snp_indices:
                if snp_idx < max_snps:
                    block_indices[:, snp_idx] = block_id
                    block_masks[:, snp_idx] = 1.0

        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        block_indices_tensor = torch.from_numpy(block_indices).long()
        block_masks_tensor = torch.from_numpy(block_masks).bool()

        # Predict
        predictions = self.predict(
            features_tensor, block_indices_tensor, block_masks_tensor, return_probs=True
        )

        return predictions.squeeze(-1).cpu().numpy()

    def get_representations(
        self,
        snp_features: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
        level: str = "global",
    ) -> torch.Tensor:
        """
        Extract intermediate representations.

        Useful for analyzing learned representations or using as features
        for downstream tasks.

        Args:
            snp_features: Per-SNP features [batch, max_snps, input_dim].
            block_indices: LD block assignments [batch, max_snps].
            block_masks: SNP validity masks [batch, max_snps].
            level: Representation level:
                - 'encoded_snps': Encoded SNP features
                - 'block_representations': Block-pooled representations
                - 'global': Global representation (default)

        Returns:
            Representations tensor of appropriate shape.
        """
        # Move inputs to device
        snp_features = snp_features.to(self.device)
        block_indices = block_indices.to(self.device)
        block_masks = block_masks.to(self.device)

        with torch.no_grad():
            output = self.model(
                snp_features=snp_features,
                block_indices=block_indices,
                block_masks=block_masks,
            )

            if level == "encoded_snps":
                # Return encoded SNP features
                encoded = self.model.snp_encoder(snp_features)
                return encoded.cpu()

            elif level == "block_representations":
                # Return block representations
                return output["block_representations"].cpu()

            elif level == "global":
                # Return global representation
                return output["global_representation"].cpu()

            else:
                raise ValueError(f"Unknown representation level: {level}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()

    def save_predictions(
        self,
        predictions: np.ndarray,
        output_path: str,
        individual_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Save predictions to file.

        Args:
            predictions: Risk predictions [n_individuals].
            output_path: Path to save predictions.
            individual_ids: Optional individual IDs. Default: None
        """
        data = {"risk_score": predictions}

        if individual_ids is not None:
            data["individual_id"] = individual_ids

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        log.info("Saved predictions", path=output_path, n_predictions=len(predictions))
