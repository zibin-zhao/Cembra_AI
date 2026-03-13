"""
CATN Custom Layers and Components.

This module implements all custom PyTorch layers used in the Cross-Ancestry
Transfer Network, including domain adversarial components, attention mechanisms,
and specialized neural network modules for genomic data.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for Domain Adversarial Training.

    During forward pass, acts as identity. During backward pass, negates
    gradients by a scale factor lambda to enable adversarial domain adaptation.

    This enables the discriminator to push back against the feature extractor,
    forcing it to learn domain-invariant representations.

    Attributes:
        lambda_: Scale factor for gradient reversal (default: 1.0)
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """
        Initialize Gradient Reversal Layer.

        Args:
            lambda_: Scale factor for gradient reversal. Default: 1.0
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: identity function.

        Args:
            x: Input tensor of any shape.

        Returns:
            Same tensor unchanged.
        """
        return x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: negate gradients by lambda.

        Args:
            grad_output: Upstream gradients.

        Returns:
            Negated and scaled gradients.
        """
        return grad_output.neg() * self.lambda_

    def set_lambda(self, lambda_: float) -> None:
        """
        Update the gradient reversal scale factor.

        Useful for annealing the domain adaptation strength during training.

        Args:
            lambda_: New scale factor value.
        """
        self.lambda_ = lambda_


class SNPFeatureEncoder(nn.Module):
    """
    SNP Feature Encoder with Positional Information.

    Encodes per-SNP feature vectors with optional positional encoding
    for genomic position within LD blocks. Applies LayerNorm for
    stable training.

    The encoder helps contextualize each SNP within its local
    linkage disequilibrium block.

    Args:
        input_dim: Dimension of input SNP features.
        d_model: Output embedding dimension.
        use_positional_encoding: Whether to add positional encoding. Default: True
        dropout: Dropout probability. Default: 0.1
        max_snps_per_block: Maximum SNPs per LD block (for PE). Default: 100
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        use_positional_encoding: bool = True,
        dropout: float = 0.1,
        max_snps_per_block: int = 100,
    ) -> None:
        """
        Initialize SNP Feature Encoder.

        Args:
            input_dim: Dimension of input SNP features.
            d_model: Output embedding dimension.
            use_positional_encoding: Whether to add positional encoding.
            dropout: Dropout probability.
            max_snps_per_block: Maximum SNPs per LD block.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # Linear projection to model dimension
        self.projection = nn.Linear(input_dim, d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Positional encoding for SNPs within LD blocks
        if use_positional_encoding:
            self.register_buffer(
                "positional_encoding",
                self._create_positional_encoding(max_snps_per_block, d_model),
            )
        else:
            self.positional_encoding = None

    @staticmethod
    def _create_positional_encoding(
        seq_len: int, d_model: int
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.

        Args:
            seq_len: Sequence length.
            d_model: Model dimension.

        Returns:
            Positional encoding tensor of shape [seq_len, d_model].
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self, snp_features: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode SNP features with optional positional information.

        Args:
            snp_features: Input tensor of shape [batch, max_snps, input_dim].
            positions: Optional tensor of shape [batch, max_snps] with positions
                within LD blocks. If None, uses sequential positions.

        Returns:
            Encoded features of shape [batch, max_snps, d_model].
        """
        # Project to model dimension
        x = self.projection(snp_features)  # [batch, max_snps, d_model]

        # Add positional encoding if enabled
        if self.use_positional_encoding and self.positional_encoding is not None:
            seq_len = x.size(1)
            if positions is None:
                # Use sequential positions
                pe = self.positional_encoding[:seq_len, :]  # [seq_len, d_model]
            else:
                # Use custom positions from input
                pe = self.positional_encoding[positions.long()]  # [batch, seq_len, d_model]
            x = x + pe

        # Layer normalization
        x = self.norm(x)

        # Dropout
        x = self.dropout(x)

        return x


class LDBlockTransformer(nn.Module):
    """
    Transformer Encoder for LD Block Processing.

    Standard Transformer encoder (multi-head self-attention + feed-forward)
    that operates within a single linkage disequilibrium block. Captures
    intra-block SNP interactions.

    Args:
        d_model: Model dimension. Default: 256
        n_heads: Number of attention heads. Default: 8
        n_layers: Number of transformer layers. Default: 2
        d_ff: Feed-forward hidden dimension. Default: 1024
        dropout: Dropout probability. Default: 0.1
        activation: Activation function ('relu' or 'gelu'). Default: 'gelu'
        use_gradient_checkpointing: Enable gradient checkpointing. Default: False
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """
        Initialize LD Block Transformer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout probability.
            activation: Activation function ('relu' or 'gelu').
            use_gradient_checkpointing: Enable gradient checkpointing.
        """
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Standard transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        # Stack of encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process SNPs within an LD block using transformer.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            src_key_padding_mask: Boolean mask where True indicates padding.
                Shape: [batch, seq_len]. Default: None

        Returns:
            Processed tensor of shape [batch, seq_len, d_model].
        """
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            return torch.utils.checkpoint.checkpoint(
                self.transformer,
                x,
                None,
                src_key_padding_mask,
                use_reentrant=False,
            )
        else:
            return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class CrossBlockAttention(nn.Module):
    """
    Sparse Cross-Block Attention Module.

    Captures long-range genetic interactions across LD blocks using
    sparse attention for memory efficiency. Each LD block is first pooled
    to a single vector, then sparse top-k attention is applied across blocks.

    Args:
        d_model: Model dimension. Default: 256
        n_heads: Number of attention heads. Default: 8
        dropout: Dropout probability. Default: 0.1
        top_k: Number of top blocks to attend to. Default: 4
        use_mean_pooling: Use mean pooling for block aggregation. Default: True
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        top_k: int = 4,
        use_mean_pooling: bool = True,
    ) -> None:
        """
        Initialize Cross-Block Attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            top_k: Number of top blocks for sparse attention.
            use_mean_pooling: Whether to use mean pooling for blocks.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.top_k = top_k
        self.use_mean_pooling = use_mean_pooling

        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        block_representations: torch.Tensor,
        block_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparse cross-block attention.

        Args:
            block_representations: Block-pooled representations of shape
                [batch, n_blocks, d_model].
            block_masks: Boolean mask where True indicates valid blocks.
                Shape: [batch, n_blocks]. Default: None

        Returns:
            Tuple of (attended_output, attention_weights):
            - attended_output: Shape [batch, n_blocks, d_model]
            - attention_weights: Shape [batch, n_heads, n_blocks, n_blocks]
        """
        batch_size, n_blocks, d_model = block_representations.shape

        # Project to Q, K, V
        Q = self.q_proj(block_representations)  # [batch, n_blocks, d_model]
        K = self.k_proj(block_representations)
        V = self.v_proj(block_representations)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_blocks, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_blocks, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_blocks, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch, n_heads, n_blocks, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, n_heads, n_blocks, n_blocks]

        # Apply sparse top-k attention
        if self.top_k and self.top_k < n_blocks:
            # Get top-k values and indices
            topk_scores, topk_indices = torch.topk(
                scores, k=min(self.top_k, n_blocks), dim=-1
            )
            # [batch, n_heads, n_blocks, top_k]

            # Create sparse mask
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_indices, topk_scores)
            scores = mask

        # Apply block mask if provided
        if block_masks is not None:
            # block_masks: [batch, n_blocks] -> expand for attention
            block_mask_expanded = (
                block_masks.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch_size, self.n_heads, n_blocks, n_blocks)
            )
            scores = scores.masked_fill(~block_mask_expanded, float("-inf"))

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, n_blocks, n_blocks]
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch, n_heads, n_blocks, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch, n_blocks, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, n_blocks, d_model)
        # [batch, n_blocks, d_model]

        # Final output projection
        output = self.out_proj(attn_output)  # [batch, n_blocks, d_model]

        return output, attn_weights


class RiskPredictionHead(nn.Module):
    """
    Risk Prediction Head.

    Multi-layer perceptron that predicts osteoarthritis risk probability
    from learned representations. Applies sigmoid activation for binary
    classification.

    Args:
        input_dim: Input dimension (typically d_model). Default: 256
        hidden_dims: Tuple of hidden layer dimensions. Default: (512, 256)
        dropout: Dropout probability. Default: 0.2
        use_batch_norm: Whether to use batch normalization. Default: True
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Initialize Risk Prediction Head.

        Args:
            input_dim: Input dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict risk probability.

        Args:
            x: Input representation of shape [batch, input_dim].

        Returns:
            Risk logits (pre-sigmoid) of shape [batch, 1].
        """
        logits = self.mlp(x)
        return logits

    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get risk probabilities with sigmoid applied.

        Args:
            x: Input representation of shape [batch, input_dim].

        Returns:
            Risk probabilities of shape [batch, 1].
        """
        logits = self.mlp(x)
        return self.sigmoid(logits)


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for Adversarial Training.

    Multi-layer perceptron with Gradient Reversal Layer that predicts
    ancestry/domain from learned representations. Used for domain adversarial
    training to encourage domain-invariant feature learning.

    Args:
        input_dim: Input dimension (typically d_model). Default: 256
        hidden_dims: Tuple of hidden layer dimensions. Default: (512, 256)
        dropout: Dropout probability. Default: 0.2
        lambda_: Initial gradient reversal scale factor. Default: 1.0
        use_batch_norm: Whether to use batch normalization. Default: True
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        lambda_: float = 1.0,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Initialize Domain Discriminator.

        Args:
            input_dim: Input dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout probability.
            lambda_: Initial gradient reversal scale factor.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_)

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer for binary domain classification
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict domain/ancestry.

        Args:
            x: Input representation of shape [batch, input_dim].

        Returns:
            Domain logits (pre-sigmoid) of shape [batch, 1].
        """
        # Apply gradient reversal
        x = self.grl(x)

        # Predict domain
        logits = self.mlp(x)
        return logits

    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get domain predictions with sigmoid applied.

        Args:
            x: Input representation of shape [batch, input_dim].

        Returns:
            Domain probabilities of shape [batch, 1].
        """
        x = self.grl(x)
        logits = self.mlp(x)
        return self.sigmoid(logits)

    def set_lambda(self, lambda_: float) -> None:
        """
        Update the gradient reversal scale factor.

        Args:
            lambda_: New scale factor value.
        """
        self.grl.set_lambda(lambda_)
