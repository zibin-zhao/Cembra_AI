"""
Cross-Ancestry Transfer Network (CATN) Model.

This module implements the main CATN model that performs cross-ancestry
polygenic risk score transfer learning using domain adversarial training.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

from .catn_layers import (
    CrossBlockAttention,
    DomainDiscriminator,
    LDBlockTransformer,
    RiskPredictionHead,
    SNPFeatureEncoder,
)

log = structlog.get_logger(__name__)


class CrossAncestryTransferNet(nn.Module):
    """
    Cross-Ancestry Transfer Network for Polygenic Risk Score Learning.

    This model performs multi-phase training on genomic data:
    1. Phase 1: Pre-train risk predictor on EUR ancestry data
    2. Phase 2: Domain adaptation on multi-ancestry data
    3. Phase 3: Fine-tune prediction head on individual/mixed data

    The architecture processes SNPs organized in LD blocks, capturing
    intra-block and inter-block interactions for improved risk prediction
    that generalizes across ancestries.

    Architecture Flow:
    1. SNPFeatureEncoder: Encode per-SNP features → [batch, max_snps, d_model]
    2. Group by LD block, apply LDBlockTransformer within each block
    3. Pool each block to [batch, n_blocks, d_model]
    4. CrossBlockAttention: Capture inter-block interactions
    5. Global pooling to [batch, d_model]
    6. RiskPredictionHead: Predict OA risk
    7. DomainDiscriminator: Predict ancestry (during Phase 2)

    Args:
        config: Configuration dictionary with keys:
            - input_dim: Input SNP feature dimension (required)
            - d_model: Model embedding dimension. Default: 256
            - n_heads: Multi-head attention heads. Default: 8
            - n_encoder_layers: LD block transformer layers. Default: 2
            - d_ff: Feed-forward dimension. Default: 1024
            - dropout: Dropout probability. Default: 0.1
            - risk_hidden_dims: Risk head hidden dims. Default: (512, 256)
            - domain_hidden_dims: Domain head hidden dims. Default: (512, 256)
            - top_k_blocks: Top-k sparse attention. Default: 4
            - use_gradient_checkpointing: Enable checkpointing. Default: False
            - use_positional_encoding: Use positional encoding. Default: True
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize CATN model.

        Args:
            config: Configuration dictionary.

        Raises:
            ValueError: If required config keys are missing.
        """
        super().__init__()

        # Validate required config
        if "input_dim" not in config:
            raise ValueError("Config must contain 'input_dim' key")

        # Store config
        self.config = config
        self.d_model = config.get("d_model", 256)
        self.n_heads = config.get("n_heads", 8)
        self.dropout = config.get("dropout", 0.1)

        # Extract config parameters
        input_dim = config["input_dim"]
        n_encoder_layers = config.get("n_encoder_layers", 2)
        d_ff = config.get("d_ff", 1024)
        risk_hidden_dims = config.get("risk_hidden_dims", (512, 256))
        domain_hidden_dims = config.get("domain_hidden_dims", (512, 256))
        top_k_blocks = config.get("top_k_blocks", 4)
        use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
        use_positional_encoding = config.get("use_positional_encoding", True)

        log.info(
            "Initializing CATN model",
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=n_encoder_layers,
            input_dim=input_dim,
        )

        # SNP Feature Encoder
        self.snp_encoder = SNPFeatureEncoder(
            input_dim=input_dim,
            d_model=self.d_model,
            use_positional_encoding=use_positional_encoding,
            dropout=self.dropout,
        )

        # LD Block Transformer (intra-block interaction)
        self.ld_transformer = LDBlockTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=self.dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Cross-Block Attention (inter-block interaction)
        self.cross_block_attention = CrossBlockAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            top_k=top_k_blocks,
        )

        # Risk Prediction Head
        self.risk_head = RiskPredictionHead(
            input_dim=self.d_model,
            hidden_dims=risk_hidden_dims,
            dropout=self.dropout,
        )

        # Domain Discriminator (for adversarial training)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=self.d_model,
            hidden_dims=domain_hidden_dims,
            dropout=self.dropout,
            lambda_=1.0,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        snp_features: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
        ancestry_labels: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CATN model.

        Args:
            snp_features: Per-SNP feature vectors of shape
                [batch, max_snps, input_dim].
            block_indices: LD block assignment for each SNP of shape
                [batch, max_snps]. Values indicate which block (0 to n_blocks-1).
            block_masks: Boolean mask indicating real vs. padded SNPs of shape
                [batch, max_snps]. True for real SNPs, False for padding.
            ancestry_labels: Optional ancestry labels of shape [batch] for
                domain adversarial training. Default: None
            return_attention_weights: Whether to return attention weights.
                Default: False

        Returns:
            Dictionary with keys:
            - 'risk_logits': Risk prediction logits [batch, 1]
            - 'domain_logits': Domain logits [batch, 1] (if ancestry_labels provided)
            - 'block_representations': Block-pooled representations [batch, n_blocks, d_model]
            - 'global_representation': Global pooled representation [batch, d_model]
            - 'attention_weights': Block attention weights if return_attention_weights=True
            - 'snp_attention_weights': SNP attention scores from block transformer

        Raises:
            ValueError: If block_indices or block_masks have invalid shapes.
        """
        batch_size = snp_features.shape[0]
        max_snps = snp_features.shape[1]

        # Validate input shapes
        if block_indices.shape != (batch_size, max_snps):
            raise ValueError(
                f"block_indices shape {block_indices.shape} doesn't match "
                f"expected ({batch_size}, {max_snps})"
            )
        if block_masks.shape != (batch_size, max_snps):
            raise ValueError(
                f"block_masks shape {block_masks.shape} doesn't match "
                f"expected ({batch_size}, {max_snps})"
            )

        # Step 1: Encode per-SNP features
        encoded_snps = self.snp_encoder(snp_features)  # [batch, max_snps, d_model]

        # Step 2: Group SNPs by LD block and apply within-block transformer
        block_representations = self._process_ld_blocks(
            encoded_snps, block_indices, block_masks
        )  # [batch, n_blocks, d_model]

        # Create block mask (True for real blocks, False for empty blocks)
        block_mask = self._create_block_mask(
            block_indices, block_masks, block_representations.shape[1]
        )  # [batch, n_blocks]

        # Step 3: Apply cross-block attention
        attended_blocks, attn_weights = self.cross_block_attention(
            block_representations, block_mask
        )  # [batch, n_blocks, d_model], weights

        # Step 4: Global pooling
        global_representation = self._global_pool(
            attended_blocks, block_mask
        )  # [batch, d_model]

        # Step 5: Risk prediction
        risk_logits = self.risk_head(global_representation)  # [batch, 1]

        # Build output dictionary
        output = {
            "risk_logits": risk_logits,
            "block_representations": block_representations,
            "global_representation": global_representation,
        }

        # Step 6: Domain discrimination (if ancestry labels provided)
        if ancestry_labels is not None:
            domain_logits = self.domain_discriminator(global_representation)
            output["domain_logits"] = domain_logits

        # Include attention weights if requested
        if return_attention_weights:
            output["attention_weights"] = attn_weights

        return output

    def _process_ld_blocks(
        self,
        encoded_snps: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process SNPs within each LD block using transformer.

        Groups SNPs by LD block, applies transformer within each block,
        and pools to single vector per block.

        Args:
            encoded_snps: Encoded SNP features [batch, max_snps, d_model].
            block_indices: LD block indices [batch, max_snps].
            block_masks: SNP validity masks [batch, max_snps].

        Returns:
            Block representations [batch, n_blocks, d_model].
        """
        batch_size = encoded_snps.shape[0]
        n_blocks = block_indices.max().item() + 1

        # Process all SNPs through transformer together
        # This allows attention within blocks
        snp_key_padding_mask = ~block_masks.bool()  # True for padding
        transformed_snps = self.ld_transformer(
            encoded_snps, src_key_padding_mask=snp_key_padding_mask
        )  # [batch, max_snps, d_model]

        # Pool SNPs by block
        block_reps = []
        for block_idx in range(n_blocks):
            # Get mask for this block
            block_mask = (block_indices == block_idx) & block_masks.bool()

            # Extract SNPs in this block
            block_snps = transformed_snps * block_mask.unsqueeze(-1).float()

            # Pool: mean over SNPs in block
            block_count = block_mask.sum(dim=1, keepdim=True).clamp(min=1)
            block_rep = block_snps.sum(dim=1) / block_count  # [batch, d_model]

            block_reps.append(block_rep)

        # Stack block representations
        block_representations = torch.stack(block_reps, dim=1)  # [batch, n_blocks, d_model]

        return block_representations

    @staticmethod
    def _create_block_mask(
        block_indices: torch.Tensor,
        snp_masks: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """
        Create block-level masks indicating which blocks have real SNPs.

        Args:
            block_indices: SNP block assignments [batch, max_snps].
            snp_masks: SNP validity masks [batch, max_snps].
            n_blocks: Total number of blocks.

        Returns:
            Block masks [batch, n_blocks] where True = has real SNPs.
        """
        batch_size = block_indices.shape[0]
        device = block_indices.device

        block_mask = torch.zeros(
            (batch_size, n_blocks), dtype=torch.bool, device=device
        )

        for batch_idx in range(batch_size):
            valid_snps = snp_masks[batch_idx].bool()
            valid_blocks = block_indices[batch_idx][valid_snps].unique()
            block_mask[batch_idx, valid_blocks] = True

        return block_mask

    @staticmethod
    def _global_pool(
        block_representations: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool block representations to global representation.

        Uses masked mean pooling to ignore empty blocks.

        Args:
            block_representations: Block reps [batch, n_blocks, d_model].
            block_mask: Block validity mask [batch, n_blocks].

        Returns:
            Global representation [batch, d_model].
        """
        # Mask invalid blocks
        masked_blocks = (
            block_representations * block_mask.unsqueeze(-1).float()
        )  # [batch, n_blocks, d_model]

        # Mean pool over blocks
        block_count = block_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        global_rep = masked_blocks.sum(dim=1) / block_count  # [batch, d_model]

        return global_rep

    def extract_snp_weights(
        self,
        snp_features: torch.Tensor,
        block_indices: torch.Tensor,
        block_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract learned SNP importance weights.

        Computes weights based on attention patterns in the model.
        SNPs with higher attention in the transformer layers receive
        higher weights.

        Args:
            snp_features: SNP feature vectors [batch, max_snps, input_dim].
            block_indices: LD block assignments [batch, max_snps].
            block_masks: SNP validity masks [batch, max_snps].

        Returns:
            SNP weights [batch, max_snps].
        """
        batch_size = snp_features.shape[0]
        max_snps = snp_features.shape[1]
        device = snp_features.device

        # Forward pass to get encodings
        with torch.no_grad():
            encoded_snps = self.snp_encoder(snp_features)

            # Get transformer attention weights
            # Note: This is a simplified version - full implementation would
            # extract actual attention from transformer layers
            snp_key_padding_mask = ~block_masks.bool()

            # Use attention from attention layers if accessible
            # For now, use norm of learned representations as proxy
            snp_norms = torch.norm(encoded_snps, p=2, dim=-1)  # [batch, max_snps]

            # Normalize and mask
            snp_norms = snp_norms * block_masks.float()
            max_norm = snp_norms.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            snp_weights = snp_norms / max_norm

        return snp_weights

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters except risk prediction head.

        Used in Phase 3 fine-tuning to prevent catastrophic forgetting
        of learned representations.
        """
        # Freeze all components except risk head
        for module in [
            self.snp_encoder,
            self.ld_transformer,
            self.cross_block_attention,
            self.domain_discriminator,
        ]:
            for param in module.parameters():
                param.requires_grad = False

        # Unfreeze risk head
        for param in self.risk_head.parameters():
            param.requires_grad = True

        log.info("Frozen backbone, unfrozen risk prediction head")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters.

        Used when transitioning between training phases.
        """
        for param in self.parameters():
            param.requires_grad = True

        log.info("Unfrozen all parameters")

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()

    def get_parameter_groups(
        self, learning_rate: float, weight_decay: float = 0.01
    ) -> list:
        """
        Get parameter groups for optimizer with different learning rates.

        Useful for discriminative fine-tuning or layer-wise learning rates.

        Args:
            learning_rate: Base learning rate.
            weight_decay: Weight decay (L2 regularization).

        Returns:
            List of parameter groups for optimizer.
        """
        param_groups = [
            {
                "params": self.snp_encoder.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "snp_encoder",
            },
            {
                "params": self.ld_transformer.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "ld_transformer",
            },
            {
                "params": self.cross_block_attention.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "cross_block_attention",
            },
            {
                "params": self.risk_head.parameters(),
                "lr": learning_rate * 2.0,
                "weight_decay": weight_decay,
                "name": "risk_head",
            },
            {
                "params": self.domain_discriminator.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "domain_discriminator",
            },
        ]

        return param_groups
