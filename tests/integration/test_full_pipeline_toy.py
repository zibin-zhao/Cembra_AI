"""Integration tests for full end-to-end pipeline."""

import numpy as np
import pandas as pd
import torch
import pytest

from oa_prs.data.qc import apply_qc_filters
from oa_prs.data.harmonize import harmonize_alleles
from oa_prs.data.simulate import simulate_genotypes_from_sumstats
from oa_prs.scoring.score import compute_prs_dosage, standardize_prs
from oa_prs.evaluation.metrics import compute_auc, compute_brier, compute_fairness_metric
from oa_prs.models.catn_model import CATN


class TestToyPipelineEndToEnd:
    """Full pipeline test: QC → simulate → score → evaluate."""

    def test_toy_pipeline_end_to_end(self, toy_sumstats, toy_phenotype, toy_config):
        """Run toy data → QC → score → evaluate."""
        # Step 1: Quality Control
        sumstats = apply_qc_filters(
            toy_sumstats,
            maf_threshold=toy_config['maf_threshold'],
            info_threshold=toy_config['info_threshold'],
            remove_ambiguous=False
        )

        assert len(sumstats) > 0, "QC removed all variants"

        # Step 2: Add target alleles
        sumstats['target_A1'] = sumstats['A1'].copy()
        sumstats['target_A2'] = sumstats['A2'].copy()

        # Step 3: Harmonize
        harmonized = harmonize_alleles(
            sumstats,
            a1_col='A1', a2_col='A2',
            target_a1_col='target_A1',
            target_a2_col='target_A2'
        )

        # Step 4: Simulate genotypes
        genotypes = simulate_genotypes_from_sumstats(
            harmonized,
            n_samples=len(toy_phenotype),
            seed=toy_config['seed']
        )

        assert genotypes.shape == (len(toy_phenotype), len(harmonized))

        # Step 5: Compute PRS
        weights = harmonized['BETA'].values
        prs = compute_prs_dosage(genotypes, weights)
        prs_std = standardize_prs(prs)

        assert prs_std.shape == (len(toy_phenotype),)
        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)

        # Step 6: Evaluate
        auc = compute_auc(toy_phenotype, prs_std)

        assert 0 <= auc <= 1, f"AUC out of bounds: {auc}"

    def test_toy_pipeline_qc_to_evaluation(self, toy_sumstats):
        """Test complete pipeline from QC to evaluation metrics."""
        # Create synthetic phenotypes
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.binomial(1, 0.15, n_samples)

        # QC
        sumstats = apply_qc_filters(toy_sumstats)

        # Simulate
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=n_samples, seed=42
        )

        # Score
        weights = sumstats['BETA'].values
        prs = compute_prs_dosage(genotypes, weights)
        prs_std = standardize_prs(prs)

        # Evaluate
        auc = compute_auc(y_true, prs_std)
        brier = compute_brier(y_true, prs_std)

        assert 0 <= auc <= 1
        assert 0 <= brier <= 1

    def test_toy_pipeline_multiple_ancestry(self, toy_sumstats):
        """Test pipeline with multiple ancestry groups."""
        # Create labels for two ancestry groups
        ancestry_labels = np.array([0] * 25 + [1] * 25)  # EUR vs AFR
        y_true = np.random.binomial(1, 0.15, 50)

        # QC
        sumstats = apply_qc_filters(toy_sumstats)

        # Simulate
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        # Score
        weights = sumstats['BETA'].values
        prs = compute_prs_dosage(genotypes, weights)
        prs_std = standardize_prs(prs)

        # Evaluate per ancestry
        auc_eur = compute_auc(y_true[:25], prs_std[:25])
        auc_afr = compute_auc(y_true[25:], prs_std[25:])

        # Fairness metric
        auc_gap = compute_fairness_metric(y_true, prs_std, ancestry_labels)

        assert 0 <= auc_eur <= 1
        assert 0 <= auc_afr <= 1
        assert 0 <= auc_gap <= 1

    def test_toy_pipeline_with_catn_model(self, toy_sumstats, toy_phenotype):
        """Test pipeline with CATN model."""
        # QC
        sumstats = apply_qc_filters(toy_sumstats)

        # Simulate
        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=len(toy_phenotype), seed=42
        )

        genotypes_tensor = torch.tensor(genotypes, dtype=torch.float32)
        phenotype_tensor = torch.tensor(toy_phenotype, dtype=torch.float32)

        # Train CATN
        model = CATN(
            n_snps=len(sumstats),
            d_model=32,
            n_heads=2,
            n_layers=1
        )
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(2):  # 2 mini-epochs
            for i in range(0, len(genotypes_tensor), 10):
                x = genotypes_tensor[i:i+10]
                y = phenotype_tensor[i:i+10].unsqueeze(1)

                risk_pred, _ = model(x)
                loss = torch.nn.functional.binary_cross_entropy(risk_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract weights
        model.eval()
        weights_catn = model.extract_snp_weights()

        assert weights_catn.shape == (len(sumstats),)

        # Predict and evaluate
        with torch.no_grad():
            risk_pred, _ = model(genotypes_tensor)

        risk_pred_np = risk_pred.numpy().squeeze()
        auc = compute_auc(toy_phenotype, risk_pred_np)

        assert 0 <= auc <= 1


class TestPipelineRobustness:
    """Test pipeline robustness to edge cases."""

    def test_pipeline_with_extreme_weights(self):
        """Test pipeline with very large/small weights."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(100)],
            'CHR': np.random.choice([1, 2, 3], 100),
            'BP': np.random.randint(1000000, 100000000, 100),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'BETA': np.concatenate([
                np.linspace(-10, -5, 50),
                np.linspace(5, 10, 50)
            ]),
            'SE': [0.05] * 100,
            'P': [0.001] * 100,
            'N': [500000] * 100,
            'MAF': np.random.uniform(0.01, 0.5, 100)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)
        prs_std = standardize_prs(prs)

        assert all(np.isfinite(prs_std))

    def test_pipeline_with_all_positive_effects(self):
        """Test pipeline where all effects are positive."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(100)],
            'CHR': np.random.choice([1, 2, 3], 100),
            'BP': np.random.randint(1000000, 100000000, 100),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'BETA': np.abs(np.random.normal(0.1, 0.05, 100)),
            'SE': [0.05] * 100,
            'P': [0.001] * 100,
            'N': [500000] * 100,
            'MAF': np.random.uniform(0.01, 0.5, 100)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)
        prs_std = standardize_prs(prs)

        assert all(np.isfinite(prs_std))

    def test_pipeline_with_mixed_effect_signs(self):
        """Test pipeline with mixed positive/negative effects."""
        np.random.seed(42)
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(100)],
            'CHR': np.random.choice([1, 2, 3], 100),
            'BP': np.random.randint(1000000, 100000000, 100),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'BETA': np.random.normal(0, 0.1, 100),
            'SE': [0.05] * 100,
            'P': [0.001] * 100,
            'N': [500000] * 100,
            'MAF': np.random.uniform(0.01, 0.5, 100)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)
        prs_std = standardize_prs(prs)

        # PRS distribution should be roughly normal
        assert np.isclose(np.mean(prs_std), 0.0, atol=1e-10)
        assert np.isclose(np.std(prs_std, ddof=0), 1.0)


class TestPipelinePerformance:
    """Test pipeline on data of varying sizes."""

    def test_pipeline_scaling(self):
        """Test that pipeline scales with data size."""
        for n_samples in [10, 50, 100]:
            sumstats = pd.DataFrame({
                'SNP': [f'rs{i}' for i in range(50)],
                'CHR': np.random.choice([1, 2, 3], 50),
                'BP': np.random.randint(1000000, 100000000, 50),
                'A1': np.random.choice(['A', 'T', 'G', 'C'], 50),
                'A2': np.random.choice(['A', 'T', 'G', 'C'], 50),
                'BETA': np.random.normal(0, 0.1, 50),
                'SE': [0.05] * 50,
                'P': [0.001] * 50,
                'N': [500000] * 50,
                'MAF': np.random.uniform(0.01, 0.5, 50)
            })

            genotypes = simulate_genotypes_from_sumstats(
                sumstats, n_samples=n_samples, seed=42
            )

            prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)
            prs_std = standardize_prs(prs)

            assert prs_std.shape == (n_samples,)

    def test_pipeline_with_large_snp_count(self):
        """Test pipeline with many SNPs."""
        sumstats = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(10000)],
            'CHR': np.random.choice([1, 2, 3], 10000),
            'BP': np.random.randint(1000000, 100000000, 10000),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 10000),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 10000),
            'BETA': np.random.normal(0, 0.01, 10000),
            'SE': [0.05] * 10000,
            'P': np.random.uniform(1e-8, 0.05, 10000),
            'N': [500000] * 10000,
            'MAF': np.random.uniform(0.01, 0.5, 10000)
        })

        genotypes = simulate_genotypes_from_sumstats(
            sumstats, n_samples=50, seed=42
        )

        prs = compute_prs_dosage(genotypes, sumstats['BETA'].values)

        assert prs.shape == (50,)
        assert all(np.isfinite(prs))
