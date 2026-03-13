"""Unit tests for quality control module (oa_prs.data.qc)."""

import numpy as np
import pandas as pd
import pytest
from oa_prs.data.qc import (
    filter_maf,
    filter_info,
    remove_ambiguous_snps,
    remove_duplicates,
    apply_qc_filters
)


class TestMAFFilter:
    """Test MAF filtering functionality."""

    def test_maf_filter_removes_low_maf(self, toy_sumstats):
        """Test that variants with MAF < threshold are removed."""
        threshold = 0.05
        filtered = filter_maf(toy_sumstats, threshold=threshold)

        assert all(filtered['MAF'] >= threshold), \
            "Some variants with MAF < threshold were not removed"
        assert len(filtered) < len(toy_sumstats), \
            "No variants were filtered"

    def test_maf_filter_keeps_valid(self, toy_sumstats):
        """Test that variants with MAF >= threshold are kept."""
        threshold = 0.01
        filtered = filter_maf(toy_sumstats, threshold=threshold)

        # Check that all kept variants have MAF >= threshold
        assert all(filtered['MAF'] >= threshold)

    def test_maf_filter_threshold_zero(self, toy_sumstats):
        """Test with threshold = 0 keeps all variants."""
        filtered = filter_maf(toy_sumstats, threshold=0.0)
        assert len(filtered) == len(toy_sumstats)

    def test_maf_filter_threshold_one(self, toy_sumstats):
        """Test with threshold = 1 removes all variants."""
        filtered = filter_maf(toy_sumstats, threshold=1.0)
        assert len(filtered) == 0


class TestInfoFilter:
    """Test INFO score filtering."""

    def test_info_filter_removes_low_info(self):
        """Test that variants with INFO < threshold are removed."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'INFO': [0.95, 0.75, 0.85, 0.65]
        })
        threshold = 0.8
        filtered = filter_info(df, threshold=threshold)

        assert all(filtered['INFO'] >= threshold)
        assert len(filtered) == 2

    def test_info_filter_keeps_valid(self):
        """Test that variants with INFO >= threshold are kept."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'INFO': [0.95, 0.88, 0.92]
        })
        threshold = 0.85
        filtered = filter_info(df, threshold=threshold)

        assert len(filtered) == 3

    def test_info_filter_threshold_zero(self):
        """Test with threshold = 0 keeps all variants."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2'],
            'INFO': [0.5, 0.3]
        })
        filtered = filter_info(df, threshold=0.0)
        assert len(filtered) == len(df)


class TestAmbiguousSNPRemoval:
    """Test removal of ambiguous SNP pairs."""

    def test_ambiguous_snp_at_pair(self):
        """Test that A/T pairs are removed."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['A', 'A', 'G'],
            'A2': ['T', 'G', 'C']
        })
        filtered = remove_ambiguous_snps(df)

        # rs1 (A/T) should be removed
        assert len(filtered) == 2
        assert 'rs1' not in filtered['SNP'].values

    def test_ambiguous_snp_cg_pair(self):
        """Test that C/G pairs are removed."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['C', 'A', 'G'],
            'A2': ['G', 'T', 'C']
        })
        filtered = remove_ambiguous_snps(df)

        # rs1 (C/G) should be removed
        assert len(filtered) == 2
        assert 'rs1' not in filtered['SNP'].values

    def test_ambiguous_snp_reverse_order(self):
        """Test that T/A and G/C pairs are also removed."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['T', 'G', 'A'],
            'A2': ['A', 'C', 'G']
        })
        filtered = remove_ambiguous_snps(df)

        # rs1 (T/A) and rs2 (G/C) should be removed
        assert len(filtered) == 1
        assert 'rs3' in filtered['SNP'].values

    def test_non_ambiguous_snps_kept(self):
        """Test that non-ambiguous SNPs are kept."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['A', 'G', 'C'],
            'A2': ['G', 'C', 'T']
        })
        filtered = remove_ambiguous_snps(df)

        assert len(filtered) == 3


class TestDuplicateRemoval:
    """Test handling of duplicate SNP IDs."""

    def test_duplicate_removal_keeps_first(self):
        """Test that duplicate SNP IDs are handled (keep first occurrence)."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs1', 'rs3'],
            'BETA': [0.1, 0.2, 0.15, 0.3]
        })
        filtered = remove_duplicates(df, keep='first')

        assert len(filtered) == 3
        assert filtered[filtered['SNP'] == 'rs1']['BETA'].values[0] == 0.1

    def test_duplicate_removal_keeps_last(self):
        """Test that duplicate SNP IDs can keep last occurrence."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs1', 'rs3'],
            'BETA': [0.1, 0.2, 0.15, 0.3]
        })
        filtered = remove_duplicates(df, keep='last')

        assert len(filtered) == 3
        assert filtered[filtered['SNP'] == 'rs1']['BETA'].values[0] == 0.15

    def test_no_duplicates_unchanged(self):
        """Test that data with no duplicates is unchanged."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'BETA': [0.1, 0.2, 0.3]
        })
        filtered = remove_duplicates(df)

        assert len(filtered) == len(df)
        assert df.equals(filtered)


class TestQCPreservesValid:
    """Test that QC pipeline preserves valid variants."""

    def test_qc_pipeline_preserves_valid(self, toy_sumstats):
        """Test that valid variants pass through entire QC pipeline."""
        # Start with toy data and apply all QC filters
        original_len = len(toy_sumstats)
        filtered = apply_qc_filters(
            toy_sumstats,
            maf_threshold=0.01,
            info_threshold=0.0,
            remove_ambiguous=False
        )

        # Some variants should pass (all toy data has MAF >= 0.01)
        assert len(filtered) > 0, "No variants passed QC"
        assert len(filtered) <= original_len, "More variants after QC than before"

    def test_qc_preserves_columns(self, toy_sumstats):
        """Test that QC preserves required columns."""
        required_cols = ['SNP', 'CHR', 'BP', 'A1', 'A2', 'BETA', 'SE', 'P']
        filtered = apply_qc_filters(toy_sumstats)

        for col in required_cols:
            assert col in filtered.columns, f"Column {col} missing after QC"

    def test_qc_output_is_dataframe(self, toy_sumstats):
        """Test that QC output is a pandas DataFrame."""
        filtered = apply_qc_filters(toy_sumstats)
        assert isinstance(filtered, pd.DataFrame)
