"""Unit tests for full CATN model."""

import numpy as np
import pytest
import torch

from oa_prs.models.catn_model import CATN


class TestCATNForwardPass:
    """Test full forward pass of CATN model."""

    def test_forward_pass(self):
        """Test full forward pass with random input."""
        batch_size = 16
        n_snps = 100
        device = 'cpu'

        model = CATN(
            n_snps=n_snps,
            d_model=64,
            n_heads=4,
            n_layers=2,
            device=device
        ).to(device)

        # Create random input
        genotypes = torch.randn(batch_size, n_snps).to(device)

        # Forward pass
        risk_pred, domain_pred = model(genotypes)

        assert risk_pred.shape == (batch_size, 1)
        assert domain_pred.shape == (batch_size, 1)

    def test_forward_pass_output_range(self):
        """Test that forward pass outputs are in valid range."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        genotypes = torch.randn(8, 100)

        risk_pred, domain_pred = model(genotypes)

        assert torch.all(risk_pred >= 0) and torch.all(risk_pred <= 1)
        assert torch.all(domain_pred >= 0) and torch.all(domain_pred <= 1)

    def test_forward_pass_batch_consistency(self):
        """Test that forward pass is consistent across batch sizes."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        model.eval()

        # Same genotypes, different batch sizes
        x = torch.randn(1, 100)

        with torch.no_grad():
            risk_1, domain_1 = model(x)
            risk_4, domain_4 = model(torch.cat([x] * 4, dim=0))

        # First element of batch should match single sample
        assert torch.allclose(risk_1, risk_4[:1], atol=1e-5)
        assert torch.allclose(domain_1, domain_4[:1], atol=1e-5)


class TestExtractWeights:
    """Test SNP weight extraction."""

    def test_extract_weights(self):
        """Test SNP weight extraction."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)

        weights = model.extract_snp_weights()

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (100,), \
            f"Expected shape (100,), got {weights.shape}"

    def test_extract_weights_range(self):
        """Test that extracted weights are finite."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)

        weights = model.extract_snp_weights()

        assert np.all(np.isfinite(weights)), \
            "Extracted weights contain NaN or inf"

    def test_extract_weights_after_training(self):
        """Test weight extraction after model update."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # One training step
        x = torch.randn(16, 100)
        y = torch.randint(0, 2, (16, 1)).float()

        risk_pred, _ = model(x)
        loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)
        loss.backward()
        optimizer.step()

        # Extract weights
        weights = model.extract_snp_weights()

        assert weights.shape == (100,)
        assert np.all(np.isfinite(weights))


class TestPhase1TrainingStep:
    """Test pre-training phase on EUR data."""

    def test_phase1_training_step(self):
        """Test a single EUR pre-training step."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate EUR pre-training data
        batch_size = 16
        x_eur = torch.randn(batch_size, 100)
        y_eur = torch.randint(0, 2, (batch_size, 1)).float()

        # Forward pass
        risk_pred, domain_pred = model(x_eur)

        # Loss computation
        risk_loss = torch.nn.functional.binary_cross_entropy(risk_pred, y_eur)

        # No domain loss in phase 1 (single source)
        total_loss = risk_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Loss should be finite
        assert np.isfinite(total_loss.item())

    def test_phase1_convergence(self):
        """Test that phase 1 training reduces loss."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x = torch.randn(32, 100)
        y = torch.randint(0, 2, (32, 1)).float()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            risk_pred, _ = model(x)
            loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allowing for some variance)
        assert losses[-1] <= losses[0] * 1.1, \
            "Loss did not decrease over training steps"


class TestFreezeBackbone:
    """Test freezing backbone parameters."""

    def test_freeze_backbone(self):
        """Test that backbone params are frozen."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)

        # Freeze encoder and transformer
        model.freeze_backbone()

        # Check that encoder parameters are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad, \
                "Encoder parameters should be frozen"

        # Check that transformer parameters are frozen
        for param in model.transformer.parameters():
            assert not param.requires_grad, \
                "Transformer parameters should be frozen"

        # Check that heads are still trainable
        for param in model.risk_head.parameters():
            assert param.requires_grad, \
                "Risk head parameters should be trainable"

    def test_unfreeze_backbone(self):
        """Test unfreezing backbone parameters."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)

        model.freeze_backbone()
        model.unfreeze_backbone()

        # All parameters should be trainable again
        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_affects_training(self):
        """Test that frozen parameters don't update during training."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial encoder weights
        initial_encoder_weights = [p.clone() for p in model.encoder.parameters()]

        # Freeze backbone
        model.freeze_backbone()

        # Training step
        x = torch.randn(16, 100)
        y = torch.randint(0, 2, (16, 1)).float()
        optimizer.zero_grad()
        risk_pred, _ = model(x)
        loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)
        loss.backward()
        optimizer.step()

        # Check that encoder weights are unchanged
        for initial, current in zip(initial_encoder_weights,
                                    model.encoder.parameters()):
            assert torch.allclose(initial, current), \
                "Frozen parameters were updated during training"


class TestCATNProperties:
    """Test general properties of CATN model."""

    def test_model_device_placement(self):
        """Test that model can be placed on different devices."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)

        # CPU
        model = model.to('cpu')
        x = torch.randn(8, 100)
        risk_pred, domain_pred = model(x)
        assert risk_pred.device.type == 'cpu'

    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        model.eval()

        x = torch.randn(8, 100)
        with torch.no_grad():
            risk_pred, domain_pred = model(x)

        assert risk_pred.grad_fn is None, \
            "Model should not build computation graph in eval mode with no_grad"

    def test_model_train_mode(self):
        """Test model in training mode."""
        model = CATN(n_snps=100, d_model=64, n_heads=4, n_layers=2)
        model.train()

        x = torch.randn(8, 100)
        risk_pred, domain_pred = model(x)

        assert risk_pred.requires_grad, \
            "Model should build computation graph in train mode"
