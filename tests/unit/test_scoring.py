"""Unit tests for PRS scoring module."""

import numpy as np
import pandas as pd
import pytest

from oa_prs.scoring.score import (
    compute_prs_dosage,
    standardize_prs,
    handle_missing_snps
)


class TestScoreDosage:
    """Test dot-product scoring."""

    def test_score_dosage(self):
        """Test dot-product scoring."""
        # 10 samples, 5 SNPs
        genotypes = np.array([
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # PRS = dot product
        expected_prs = genotypes @ weights

        prs = compute_prs_dosage(genotypes, weights)

        assert np.allclose(prs, expected_prs)

    def test_score_dosage_shape(self):
        """Test that score output has correct shape."""
        n_samples = 100
        n_snps = 50

        genotypes = np.random.uniform(0, 2, size=(n_samples, n_snps))
        weights = np.random.randn(n_snps)

        prs = compute_prs_dosage(genotypes, weights)

        assert prs.shape == (n_samples,)

    def test_score_dosage_zero_weights(self):
        """Test scoring with zero weights."""
        genotypes = np.array([[1.0, 1.0, 1.0]])
        weights = np.array([0.0, 0.0, 0.0])

        prs = compute_prs_dosage(genotypes, weights)

        assert prs[0] == 0.0

    def test_score_dosage_single_sample(self):
        """Test scoring with single sample."""
        genotypes = np.array([[1.0, 2.0, 0.5]])
        weights = np.array([0.1, 0.2, 0.3])

        expected = 1.0 * 0.1 + 2.0 * 0.2 + 0.5 * 0.3

        prs = compute_prs_dosage(genotypes, weights)

        assert np.isclose(prs[0], expected)


class TestStandardize:
    """Test z-score standardization."""

    def test_standardize(self):
        """Test z-score standardization."""
        prs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        prs_std = standardize_prs(prs)

        # Check mean ≈ 0
        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)

        # Check std ≈ 1
        assert np.isclose(np.std(prs_std, ddof=0), 1.0)

    def test_standardize_preserves_shape(self):
        """Test that standardization preserves shape."""
        prs = np.random.randn(100)

        prs_std = standardize_prs(prs)

        assert prs_std.shape == prs.shape

    def test_standardize_already_std(self):
        """Test standardizing already standardized data."""
        prs = np.random.randn(100)
        prs = (prs - np.mean(prs)) / np.std(prs, ddof=0)

        prs_std = standardize_prs(prs)

        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)
        assert np.isclose(np.std(prs_std, ddof=0), 1.0)

    def test_standardize_constant(self):
        """Test standardizing constant PRS."""
        prs = np.array([5.0, 5.0, 5.0, 5.0])

        prs_std = standardize_prs(prs)

        # When all values are same, std = 0, so result should be NaN or 0
        # Most implementations handle this by returning 0
        assert np.all(prs_std == 0.0) or np.all(np.isnan(prs_std))

    def test_standardize_single_value(self):
        """Test standardizing single value."""
        prs = np.array([5.0])

        prs_std = standardize_prs(prs)

        # Single value has 0 std
        assert np.isnan(prs_std[0]) or prs_std[0] == 0.0


class TestMissingSNPs:
    """Test handling of missing SNPs."""

    def test_missing_snps_skip(self):
        """Test handling of missing SNPs."""
        genotypes_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs4'],
            'dosage': [1.0, 2.0, 0.5]
        })

        weights_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'weight': [0.1, 0.2, 0.3, 0.4]
        })

        # rs3 is missing in genotypes
        result_snps, result_weights = handle_missing_snps(
            genotypes_df, weights_df,
            snp_col='SNP', genotype_col='dosage', weight_col='weight'
        )

        # Should only have rs1, rs2, rs4
        assert len(result_snps) == 3
        assert len(result_weights) == 3
        assert 'rs3' not in result_snps

    def test_missing_snps_preserves_order(self):
        """Test that order is preserved."""
        genotypes_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'dosage': [1.0, 2.0, 3.0]
        })

        weights_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'weight': [0.1, 0.2, 0.3]
        })

        result_snps, result_weights = handle_missing_snps(
            genotypes_df, weights_df,
            snp_col='SNP', genotype_col='dosage', weight_col='weight'
        )

        assert list(result_snps) == [1.0, 2.0, 3.0]
        assert list(result_weights) == [0.1, 0.2, 0.3]

    def test_missing_snps_all_present(self):
        """Test when all SNPs are present."""
        genotypes_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'dosage': [1.0, 2.0, 3.0]
        })

        weights_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'weight': [0.1, 0.2, 0.3]
        })

        result_snps, result_weights = handle_missing_snps(
            genotypes_df, weights_df,
            snp_col='SNP', genotype_col='dosage', weight_col='weight'
        )

        assert len(result_snps) == 3
        assert len(result_weights) == 3

    def test_missing_snps_none_present(self):
        """Test when no SNPs are present."""
        genotypes_df = pd.DataFrame({
            'SNP': ['rs_a', 'rs_b'],
            'dosage': [1.0, 2.0]
        })

        weights_df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'weight': [0.1, 0.2, 0.3]
        })

        result_snps, result_weights = handle_missing_snps(
            genotypes_df, weights_df,
            snp_col='SNP', genotype_col='dosage', weight_col='weight'
        )

        # No overlap
        assert len(result_snps) == 0
        assert len(result_weights) == 0


class TestScoringPipeline:
    """Test complete scoring pipeline."""

    def test_score_and_standardize(self):
        """Test scoring followed by standardization."""
        genotypes = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0]
        ])
        weights = np.array([0.1, 0.2, 0.3])

        # Score
        prs = compute_prs_dosage(genotypes, weights)

        # Standardize
        prs_std = standardize_prs(prs)

        # Check standardization
        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)
        assert np.isclose(np.std(prs_std, ddof=0), 1.0)

    def test_score_large_data(self):
        """Test scoring on large data."""
        np.random.seed(42)
        n_samples = 10000
        n_snps = 100000

        # Simulate sparse genotypes
        genotypes = np.random.binomial(2, 0.2, size=(n_samples, n_snps)).astype(np.float32)
        weights = np.random.randn(n_snps) * 0.01

        prs = compute_prs_dosage(genotypes, weights)

        assert prs.shape == (n_samples,)
        assert np.all(np.isfinite(prs))
