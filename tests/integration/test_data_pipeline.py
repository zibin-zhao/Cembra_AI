"""Integration tests for complete data pipeline."""

import numpy as np
import pandas as pd
import pytest

from oa_prs.data.qc import apply_qc_filters
from oa_prs.data.harmonize import harmonize_alleles
from oa_prs.scoring.score import (
    compute_prs_dosage,
    standardize_prs
)
from oa_prs.data.simulate import simulate_genotypes_from_sumstats


class TestQCHarmonizePipeline:
    """Integration test: QC → harmonize → standardize pipeline."""

    def test_qc_harmonize_pipeline(self, toy_sumstats):
        """Chain QC → harmonize → standardize on toy data."""
        # Start with toy summary statistics
        sumstats = toy_sumstats.copy()

        # Step 1: QC filtering
        qc_sumstats = apply_qc_filters(
            sumstats,
            maf_threshold=0.01,
            info_threshold=0.0,
            remove_ambiguous=False
        )

        assert len(qc_sumstats) > 0, "QC removed all variants"
        assert all(qc_sumstats['MAF'] >= 0.01), "MAF filter not applied"

        # Step 2: Add target alleles (simulate reference panel)
        qc_sumstats['target_A1'] = qc_sumstats['A1'].copy()
        qc_sumstats['target_A2'] = qc_sumstats['A2'].copy()

        # Step 3: Harmonize alleles
        harmonized = harmonize_alleles(
            qc_sumstats,
            a1_col='A1', a2_col='A2',
            target_a1_col='target_A1',
            target_a2_col='target_A2'
        )

        assert len(harmonized) <= len(qc_sumstats), \
            "Harmonization created more variants"

        # Step 4: Extract weights and standardize
        weights = harmonized['BETA'].values
        weights_std = (weights - np.mean(weights)) / (np.std(weights) + 1e-8)

        assert all(np.isfinite(weights_std)), \
            "Standardization produced non-finite values"

    def test_pipeline_preserves_data_integrity(self, toy_sumstats):
        """Test that pipeline preserves data integrity."""
        sumstats = toy_sumstats.copy()
        n_original = len(sumstats)

        # Apply pipeline
        qc_sumstats = apply_qc_filters(sumstats)
        qc_sumstats['target_A1'] = qc_sumstats['A1'].copy()
        qc_sumstats['target_A2'] = qc_sumstats['A2'].copy()
        harmonized = harmonize_alleles(qc_sumstats,
                                       a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        # Check data types
        assert harmonized['BETA'].dtype in [np.float32, np.float64]
        assert all(harmonized['CHR'].isin([1, 2, 3]))

        # Check that essential columns remain
        required_cols = ['SNP', 'CHR', 'BP', 'A1', 'A2', 'BETA']
        assert all(col in harmonized.columns for col in required_cols)

    def test_pipeline_with_missing_columns(self):
        """Test pipeline handles missing INFO column gracefully."""
        # Create sumstats without INFO column
        df = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(100)],
            'CHR': np.random.choice([1, 2, 3], 100),
            'BP': np.random.randint(1000000, 100000000, 100),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'BETA': np.random.normal(0, 0.1, 100),
            'SE': np.abs(np.random.normal(0.05, 0.01, 100)),
            'P': np.random.uniform(1e-8, 0.05, 100),
            'N': np.full(100, 500000),
            'MAF': np.random.uniform(0.01, 0.5, 100)
        })

        # Should handle missing INFO
        qc_result = apply_qc_filters(df, info_threshold=0.0)
        assert len(qc_result) > 0


class TestSimulateFromSumstats:
    """Test full simulation pipeline."""

    def test_simulate_from_sumstats(self, toy_sumstats):
        """Full simulation pipeline."""
        sumstats = toy_sumstats.copy()
        n_samples = 100

        # Simulate genotypes
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=n_samples, seed=42
        )

        assert genotypes.shape == (n_samples, len(sumstats))
        assert np.all(genotypes >= 0)
        assert np.all(genotypes <= 2)

    def test_simulate_score_pipeline(self, toy_sumstats):
        """Simulate genotypes and compute PRS."""
        sumstats = toy_sumstats.copy()
        weights = sumstats['BETA'].values
        n_samples = 50

        # Simulate
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=n_samples, seed=42
        )

        # Score
        prs = compute_prs_dosage(genotypes, weights)

        assert prs.shape == (n_samples,)
        assert all(np.isfinite(prs))

    def test_simulate_score_standardize_pipeline(self, toy_sumstats):
        """Full pipeline: simulate → score → standardize."""
        sumstats = toy_sumstats.copy()
        weights = sumstats['BETA'].values

        # Simulate
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=100, seed=42
        )

        # Score
        prs = compute_prs_dosage(genotypes, weights)

        # Standardize
        prs_std = standardize_prs(prs)

        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)
        assert np.isclose(np.std(prs_std, ddof=0), 1.0)

    def test_pipeline_reproducibility(self, toy_sumstats):
        """Test that pipeline is reproducible."""
        weights = toy_sumstats['BETA'].values

        # First run
        gen1 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=999
        )
        prs1 = compute_prs_dosage(gen1, weights)
        prs1_std = standardize_prs(prs1)

        # Second run
        gen2 = simulate_genotypes_from_sumstats(
            toy_sumstats, n_samples=50, seed=999
        )
        prs2 = compute_prs_dosage(gen2, weights)
        prs2_std = standardize_prs(prs2)

        assert np.allclose(prs1_std, prs2_std)

    def test_pipeline_with_different_sample_sizes(self, toy_sumstats):
        """Test pipeline with different sample sizes."""
        weights = toy_sumstats['BETA'].values

        for n_samples in [10, 50, 100, 500]:
            genotypes = simulate_genotypes_from_sumstats(
                toy_sumstats, n_samples=n_samples, seed=42
            )
            prs = compute_prs_dosage(genotypes, weights)
            prs_std = standardize_prs(prs)

            assert prs.shape == (n_samples,)
            assert prs_std.shape == (n_samples,)


class TestPipelineEdgeCases:
    """Test edge cases in data pipeline."""

    def test_pipeline_single_variant(self):
        """Test pipeline with single variant."""
        sumstats = pd.DataFrame({
            'SNP': ['rs1'],
            'CHR': [1],
            'BP': [1000000],
            'A1': ['A'],
            'A2': ['G'],
            'BETA': [0.1],
            'SE': [0.05],
            'P': [0.001],
            'N': [500000],
            'MAF': [0.3]
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        assert genotypes.shape == (50, 1)

    def test_pipeline_high_maf(self):
        """Test pipeline with high MAF variants."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(10)],
            'CHR': [1] * 10,
            'BP': np.arange(1000000, 10000000, 1000000),
            'A1': ['A'] * 10,
            'A2': ['G'] * 10,
            'BETA': np.linspace(0.05, 0.3, 10),
            'SE': [0.05] * 10,
            'P': [0.001] * 10,
            'N': [500000] * 10,
            'MAF': np.linspace(0.4, 0.5, 10)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)
        prs_std = standardize_prs(prs)

        assert all(np.isfinite(prs_std))

    def test_pipeline_low_maf(self):
        """Test pipeline with low MAF variants."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(10)],
            'CHR': [1] * 10,
            'BP': np.arange(1000000, 10000000, 1000000),
            'A1': ['A'] * 10,
            'A2': ['G'] * 10,
            'BETA': np.linspace(0.05, 0.3, 10),
            'SE': [0.05] * 10,
            'P': [0.001] * 10,
            'N': [500000] * 10,
            'MAF': np.linspace(0.01, 0.05, 10)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)

        assert all(np.isfinite(prs))
