"""Unit tests for genotype simulation module."""

import numpy as np
import pandas as pd
import pytest
from oa_prs.data.simulate import simulate_genotypes_from_sumstats


class TestSimulationOutputShape:
    """Test output shape of simulation."""

    def test_output_shape(self):
        """Check (n_samples, n_snps) shape."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(100)],
            'MAF': np.random.uniform(0.01, 0.5, 100)
        })
        n_samples = 50

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=n_samples, seed=42
        )

        assert genotypes.shape == (50, 100), \
            f"Expected shape (50, 100), got {genotypes.shape}"

    def test_output_shape_different_sizes(self):
        """Test output shape with different input sizes."""
        for n_snps, n_samples in [(50, 20), (100, 100), (200, 30)]:
            sumstats = pd.DataFrame({
                'SNP': [f'rs{i}' for i in range(n_snps)],
                'MAF': np.random.uniform(0.01, 0.5, n_snps)
            })

            genotypes = simulate_genotypes_from_sumstats(
                sumstats, n_samples=n_samples, seed=42
            )

            assert genotypes.shape == (n_samples, n_snps)


class TestDosageRange:
    """Test that dosages are in valid range."""

    def test_dosage_range(self, toy_sumstats):
        """Check that values are in [0, 2]."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        assert np.all(genotypes >= 0), "Some dosages are < 0"
        assert np.all(genotypes <= 2), "Some dosages are > 2"

    def test_dosage_range_boundary(self, toy_sumstats):
        """Test that dosages approach boundaries appropriately."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=1000, seed=42
        )

        # With 1000 samples, should have some values near 0 and near 2
        min_val = np.min(genotypes)
        max_val = np.max(genotypes)

        assert min_val >= 0
        assert max_val <= 2


class TestMAFConsistency:
    """Test that simulated MAF matches input MAF."""

    def test_maf_consistency(self):
        """Check that simulated MAF ≈ input MAF."""
        maf_true = 0.3
        sumstats = pd.DataFrame({
            'SNP': ['rs1'],
            'MAF': [maf_true]
        })

        # Simulate with large sample for better approximation
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=5000, seed=42
        )

        # Calculate realized MAF (convert dosage to allele frequency)
        # Dosage = 0,1,2 with frequency determined by p
        maf_simulated = np.mean(genotypes[:, 0]) / 2.0

        # Allow 5% relative error
        assert abs(maf_simulated - maf_true) / maf_true < 0.05, \
            f"Simulated MAF {maf_simulated} differs from true {maf_true}"

    def test_maf_consistency_multiple_snps(self, toy_sumstats):
        """Test MAF consistency across multiple SNPs."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=1000, seed=42
        )

        maf_simulated = np.mean(genotypes, axis=0) / 2.0
        maf_true = toy_sumstats['MAF'].values

        # Check correlation between true and simulated MAF
        correlation = np.corrcoef(maf_true, maf_simulated)[0, 1]
        assert correlation > 0.8, \
            f"Simulated and true MAF not correlated (r={correlation})"


class TestReproducibility:
    """Test that same seed produces same output."""

    def test_reproducibility(self, toy_sumstats):
        """Same seed → same output."""
        gen1 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )
        gen2 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        assert np.allclose(gen1, gen2), \
            "Same seed produced different genotypes"

    def test_different_seeds_different_output(self, toy_sumstats):
        """Different seeds → different output."""
        gen1 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )
        gen2 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=123
        )

        assert not np.allclose(gen1, gen2), \
            "Different seeds produced identical genotypes"

    def test_reproducibility_large_sample(self):
        """Test reproducibility with large sample."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(500)],
            'MAF': np.random.uniform(0.01, 0.5, 500)
        })

        gen1 = simulate_genotypes_from_sumstats(
            sumstats, n_samples=500, seed=999
        )
        gen2 = simulate_genotypes_from_sumstats(
            sumstats, n_samples=500, seed=999
        )

        assert np.array_equal(gen1, gen2)


class TestSimulationProperties:
    """Test general properties of simulated data."""

    def test_dtype_is_float(self, toy_sumstats):
        """Test that output dtype is float."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        assert genotypes.dtype in [np.float32, np.float64]

    def test_no_nan_values(self, toy_sumstats):
        """Test that simulated data contains no NaN values."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        assert not np.isnan(genotypes).any(), "Simulated data contains NaN values"

    def test_no_inf_values(self, toy_sumstats):
        """Test that simulated data contains no infinite values."""
        genotypes = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=42
        )

        assert not np.isinf(genotypes).any(), \
            "Simulated data contains infinite values"
