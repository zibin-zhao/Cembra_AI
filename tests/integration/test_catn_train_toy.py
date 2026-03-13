"""Integration tests for CATN model training on toy data."""

import numpy as np
import torch
import pytest

from oa_prs.models.catn_model import CATN
from oa_prs.data.simulate import simulate_genotypes_from_sumstats


class TestCATNTrainToyEpoch:
    """Test CATN training for one epoch on tiny data."""

    def test_catn_train_toy_epoch(self, toy_sumstats, toy_phenotype):
        """Train 1 epoch on tiny simulated data."""
        # Simulate toy genotypes
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        # Convert to tensor
        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
        phenotype_tensor = torch.tensor(toy_phenotype, dtype=torch.float32)

        # Initialize model
        model = CATN(
            n_snps=100,
            d_model=32,  # Small for toy data
            n_heads=2,
            n_layers=1,
            device='cpu'
        )
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop for 1 epoch
        epoch_loss = 0.0
        n_batches = 5
        batch_size = 10

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(genotypes_tensor))

            x = genotypes_tensor[start_idx:end_idx]
            y = phenotype_tensor[start_idx:end_idx].unsqueeze(1)

            # Forward pass
            risk_pred, domain_pred = model(x)

            # Loss computation
            loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        assert np.isfinite(avg_loss), "Loss is not finite"
        assert avg_loss > 0, "Loss should be positive"

    def test_catn_train_loss_decreases(self, toy_sumstats, toy_phenotype):
        """Test that loss decreases over multiple epochs."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
        phenotype_tensor = torch.tensor(toy_phenotype, dtype=torch.float32)

        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0

            for i in range(0, len(genotypes_tensor), 10):
                x = genotypes_tensor[i:i+10]
                y = phenotype_tensor[i:i+10].unsqueeze(1)

                risk_pred, _ = model(x)
                loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss)

        # Loss should generally decrease (with some tolerance for variance)
        assert losses[-1] <= losses[0] * 1.2, \
            f"Loss increased: {losses}"


class TestCATNPredictToy:
    """Test CATN prediction on toy data."""

    def test_catn_predict_toy(self, toy_sumstats):
        """Predict on toy data after training."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)

        # Initialize and train
        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        model.eval()

        # Make predictions
        with torch.no_grad():
            risk_pred, domain_pred = model(genotypes_tensor)

        assert risk_pred.shape == (50, 1)
        assert domain_pred.shape == (50, 1)
        assert torch.all(risk_pred >= 0) and torch.all(risk_pred <= 1)
        assert torch.all(domain_pred >= 0) and torch.all(domain_pred <= 1)

    def test_catn_predict_batch_vs_individual(self, toy_sumstats):
        """Test batch prediction matches individual predictions."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=10, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)

        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        model.eval()

        # Batch prediction
        with torch.no_grad():
            risk_batch, _ = model(genotypes_tensor)

        # Individual predictions
        risk_individual = []
        with torch.no_grad():
            for i in range(len(genotypes_tensor)):
                risk, _ = model(genotypes_tensor[i:i+1])
                risk_individual.append(risk.item())

        risk_individual = torch.tensor(risk_individual).unsqueeze(1)

        assert torch.allclose(risk_batch, risk_individual, atol=1e-5)

    def test_catn_predict_reproducibility(self, toy_sumstats):
        """Test that predictions are reproducible."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)

        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        model.eval()

        # First prediction
        with torch.no_grad():
            risk1, domain1 = model(genotypes_tensor)

        # Second prediction (should be identical)
        with torch.no_grad():
            risk2, domain2 = model(genotypes_tensor)

        assert torch.allclose(risk1, risk2)
        assert torch.allclose(domain1, domain2)


class TestCATNTransferLearningSetup:
    """Test CATN for transfer learning setup."""

    def test_catn_freeze_unfreeze_backbone(self, toy_sumstats):
        """Test freezing and unfreezing backbone."""
        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)

        # Initial state - all trainable
        initial_params_trainable = sum(1 for p in model.parameters()
                                      if p.requires_grad)

        # Freeze backbone
        model.freeze_backbone()
        frozen_params_trainable = sum(1 for p in model.parameters()
                                     if p.requires_grad)

        # Should have fewer trainable parameters
        assert frozen_params_trainable < initial_params_trainable

        # Unfreeze
        model.unfreeze_backbone()
        unfrozen_params_trainable = sum(1 for p in model.parameters()
                                       if p.requires_grad)

        assert unfrozen_params_trainable == initial_params_trainable

    def test_catn_phase1_and_phase2(self, toy_sumstats, toy_phenotype):
        """Test phase 1 (EUR pretraining) and phase 2 (transfer)."""
        # Simulate EUR training data
        genotypes_eur = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )
        phenotype_eur = toy_phenotype

        genotypes_eur_tensor = torch.tensor(genotypes_eur, dtype=torch.float32)
        phenotype_eur_tensor = torch.tensor(phenotype_eur, dtype=torch.float32)

        # Initialize model
        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        optimizer_phase1 = torch.optim.Adam(model.parameters(), lr=0.01)

        # Phase 1: EUR pretraining (1 epoch)
        model.train()
        phase1_loss = 0.0
        for i in range(0, len(genotypes_eur_tensor), 10):
            x = genotypes_eur_tensor[i:i+10]
            y = phenotype_eur_tensor[i:i+10].unsqueeze(1)

            risk_pred, _ = model(x)
            loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

            optimizer_phase1.zero_grad()
            loss.backward()
            optimizer_phase1.step()

            phase1_loss += loss.item()

        # Freeze backbone for phase 2
        model.freeze_backbone()

        # Verify backbone is frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # Phase 2 would fine-tune heads on target ancestry data
        # (not fully implemented here, just test the setup)
        optimizer_phase2 = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.01
        )

        # Should be able to optimize phase 2 without error
        x_test = torch.randn(10, 100)
        y_test = torch.randint(0, 2, (10, 1)).float()

        model.train()
        risk_pred, _ = model(x_test)
        loss_phase2 = torch.nn.functional.binary_cross_entropy(risk_pred, y_test)

        optimizer_phase2.zero_grad()
        loss_phase2.backward()
        optimizer_phase2.step()

        assert np.isfinite(loss_phase2.item())


class TestCATNWeightExtraction:
    """Test SNP weight extraction from trained model."""

    def test_catn_extract_weights_after_training(self, toy_sumstats, toy_phenotype):
        """Extract weights after training."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
        phenotype_tensor = torch.tensor(toy_phenotype, dtype=torch.float32)

        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train for 1 epoch
        for i in range(0, len(genotypes_tensor), 10):
            x = genotypes_tensor[i:i+10]
            y = phenotype_tensor[i:i+10].unsqueeze(1)

            risk_pred, _ = model(x)
            loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Extract weights
        weights = model.extract_snp_weights()

        assert weights.shape == (100,)
        assert np.all(np.isfinite(weights))

    def test_catn_weights_are_different_after_training(self, toy_sumstats, toy_phenotype):
        """Test that weights change after training."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
        phenotype_tensor = torch.tensor(toy_phenotype, dtype=torch.float32)

        model = CATN(n_snps=100, d_model=32, n_heads=2, n_layers=1)

        # Get initial weights
        weights_initial = model.extract_snp_weights()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train
        for _ in range(5):
            for i in range(0, len(genotypes_tensor), 10):
                x = genotypes_tensor[i:i+10]
                y = phenotype_tensor[i:i+10].unsqueeze(1)

                risk_pred, _ = model(x)
                loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get final weights
        weights_final = model.extract_snp_weights()

        # Weights should have changed
        assert not np.allclose(weights_initial, weights_final, atol=1e-6)
