"""Unit tests for CATN neural network layers."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from oa_prs.models.catn_layers import (
    GradientReversal,
    SNPEncoder,
    LDBlockTransformer,
    RiskHead,
    DomainDiscriminator
)


class TestGradientReversal:
    """Test gradient reversal layer."""

    def test_gradient_reversal_negates_gradients(self):
        """Test that gradient reversal negates gradients."""
        gr = GradientReversal(alpha=1.0)

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = gr(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be negated
        assert torch.allclose(x.grad, torch.tensor([-1.0, -1.0, -1.0])), \
            "Gradient reversal did not negate gradients correctly"

    def test_gradient_reversal_forward_pass(self):
        """Test that forward pass returns input unchanged."""
        gr = GradientReversal(alpha=1.0)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = gr(x)

        assert torch.allclose(y, x), \
            "Forward pass should return input unchanged"

    def test_gradient_reversal_alpha(self):
        """Test gradient reversal with different alpha."""
        gr = GradientReversal(alpha=0.5)

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = gr(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be negated and scaled by alpha
        assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5, -0.5])), \
            "Alpha scaling not applied correctly"


class TestSNPEncoder:
    """Test SNP encoder layer."""

    def test_snp_encoder_output_shape(self):
        """Test encoder produces correct d_model dims."""
        batch_size = 16
        n_snps = 100
        d_model = 64

        encoder = SNPEncoder(n_snps=n_snps, d_model=d_model)
        x = torch.randn(batch_size, n_snps)

        output = encoder(x)

        assert output.shape == (batch_size, n_snps, d_model), \
            f"Expected shape {(batch_size, n_snps, d_model)}, got {output.shape}"

    def test_snp_encoder_output_shape_different_dims(self):
        """Test encoder with different dimensions."""
        test_cases = [
            (8, 50, 32),
            (32, 200, 128),
            (1, 100, 64)
        ]

        for batch_size, n_snps, d_model in test_cases:
            encoder = SNPEncoder(n_snps=n_snps, d_model=d_model)
            x = torch.randn(batch_size, n_snps)
            output = encoder(x)

            assert output.shape == (batch_size, n_snps, d_model)

    def test_snp_encoder_requires_grad(self):
        """Test that encoder parameters require gradients."""
        encoder = SNPEncoder(n_snps=100, d_model=64)

        for param in encoder.parameters():
            assert param.requires_grad, "Encoder parameters should require gradients"


class TestLDBlockTransformer:
    """Test LD block transformer."""

    def test_ld_block_transformer_output_shape(self):
        """Test transformer attention output shape."""
        batch_size = 16
        n_snps = 100
        d_model = 64

        transformer = LDBlockTransformer(
            n_snps=n_snps,
            d_model=d_model,
            n_heads=4,
            n_layers=2
        )

        x = torch.randn(batch_size, n_snps, d_model)
        output = transformer(x)

        assert output.shape == (batch_size, n_snps, d_model), \
            f"Expected shape {(batch_size, n_snps, d_model)}, got {output.shape}"

    def test_ld_block_transformer_different_sizes(self):
        """Test transformer with different input sizes."""
        test_cases = [
            (8, 50, 32, 2),
            (16, 100, 64, 4),
            (32, 200, 128, 8)
        ]

        for batch_size, n_snps, d_model, n_heads in test_cases:
            transformer = LDBlockTransformer(
                n_snps=n_snps,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=2
            )
            x = torch.randn(batch_size, n_snps, d_model)
            output = transformer(x)

            assert output.shape == (batch_size, n_snps, d_model)

    def test_ld_block_transformer_requires_grad(self):
        """Test that transformer parameters require gradients."""
        transformer = LDBlockTransformer(
            n_snps=100,
            d_model=64,
            n_heads=4,
            n_layers=2
        )

        for param in transformer.parameters():
            assert param.requires_grad


class TestRiskHead:
    """Test risk prediction head."""

    def test_risk_head_output_shape(self):
        """Test risk prediction head output shape (batch, 1)."""
        batch_size = 16
        n_snps = 100
        d_model = 64

        head = RiskHead(d_model=d_model, n_snps=n_snps)
        x = torch.randn(batch_size, n_snps, d_model)

        output = head(x)

        assert output.shape == (batch_size, 1), \
            f"Expected shape (batch, 1), got {output.shape}"

    def test_risk_head_output_shape_different_batch(self):
        """Test risk head with different batch sizes."""
        d_model = 64
        n_snps = 100

        head = RiskHead(d_model=d_model, n_snps=n_snps)

        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, n_snps, d_model)
            output = head(x)

            assert output.shape == (batch_size, 1)

    def test_risk_head_output_range(self):
        """Test that risk head output is in valid range."""
        head = RiskHead(d_model=64, n_snps=100)
        x = torch.randn(16, 100, 64)

        output = head(x)

        # Output should be bounded (e.g., sigmoid output)
        assert torch.all(output >= 0) and torch.all(output <= 1), \
            "Risk head output not in [0, 1]"


class TestDomainDiscriminator:
    """Test domain discriminator head."""

    def test_domain_discriminator_output_shape(self):
        """Test domain head output shape (batch, 1)."""
        batch_size = 16
        n_snps = 100
        d_model = 64

        head = DomainDiscriminator(d_model=d_model, n_snps=n_snps)
        x = torch.randn(batch_size, n_snps, d_model)

        output = head(x)

        assert output.shape == (batch_size, 1), \
            f"Expected shape (batch, 1), got {output.shape}"

    def test_domain_discriminator_output_shape_different_batch(self):
        """Test domain head with different batch sizes."""
        d_model = 64
        n_snps = 100

        head = DomainDiscriminator(d_model=d_model, n_snps=n_snps)

        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, n_snps, d_model)
            output = head(x)

            assert output.shape == (batch_size, 1)

    def test_domain_discriminator_output_range(self):
        """Test that domain head output is in valid range."""
        head = DomainDiscriminator(d_model=64, n_snps=100)
        x = torch.randn(16, 100, 64)

        output = head(x)

        # Output should be bounded (e.g., sigmoid output)
        assert torch.all(output >= 0) and torch.all(output <= 1), \
            "Domain head output not in [0, 1]"

    def test_domain_discriminator_requires_grad(self):
        """Test that domain discriminator parameters require gradients."""
        head = DomainDiscriminator(d_model=64, n_snps=100)

        for param in head.parameters():
            assert param.requires_grad
