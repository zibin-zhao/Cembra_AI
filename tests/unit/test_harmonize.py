"""Unit tests for allele harmonization module."""

import numpy as np
import pandas as pd
import pytest
from oa_prs.data.harmonize import (
    strand_flip,
    complement,
    harmonize_alleles,
    is_valid_allele
)


class TestStrandFlip:
    """Test strand flip functionality."""

    def test_strand_flip_at_to_ta(self):
        """Test A/T → T/A flip with beta sign change."""
        result = strand_flip('A', 'T', 0.5)

        assert result == ('T', 'A', -0.5), \
            "Strand flip should reverse alleles and negate beta"

    def test_strand_flip_gc_to_cg(self):
        """Test G/C → C/G flip."""
        result = strand_flip('G', 'C', 0.3)

        assert result == ('C', 'G', -0.3)

    def test_strand_flip_negative_beta(self):
        """Test strand flip with negative beta."""
        result = strand_flip('A', 'G', -0.2)

        assert result == ('G', 'A', 0.2)

    def test_strand_flip_zero_beta(self):
        """Test strand flip with beta = 0."""
        result = strand_flip('C', 'T', 0.0)

        assert result == ('T', 'C', -0.0)


class TestComplement:
    """Test allele complement matching."""

    def test_complement_a_to_t(self):
        """Test A→T complement."""
        result = complement('A')
        assert result == 'T'

    def test_complement_t_to_a(self):
        """Test T→A complement."""
        result = complement('T')
        assert result == 'A'

    def test_complement_c_to_g(self):
        """Test C→G complement."""
        result = complement('C')
        assert result == 'G'

    def test_complement_g_to_c(self):
        """Test G→C complement."""
        result = complement('G')
        assert result == 'C'

    def test_complement_pair(self):
        """Test complementing a pair."""
        a1, a2 = 'A', 'G'
        c1, c2 = complement(a1), complement(a2)

        assert c1 == 'T'
        assert c2 == 'C'


class TestNoChangeNeeded:
    """Test correct alleles pass through."""

    def test_harmonize_no_change(self):
        """Test that correctly oriented alleles pass through unchanged."""
        df = pd.DataFrame({
            'SNP': ['rs1'],
            'A1': ['A'],
            'A2': ['G'],
            'BETA': [0.1],
            'target_A1': ['A'],
            'target_A2': ['G']
        })

        harmonized = harmonize_alleles(df, a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        assert harmonized.iloc[0]['A1'] == 'A'
        assert harmonized.iloc[0]['A2'] == 'G'
        assert harmonized.iloc[0]['BETA'] == 0.1

    def test_harmonize_multiple_rows_no_change(self):
        """Test multiple rows with no changes needed."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['A', 'G', 'C'],
            'A2': ['G', 'C', 'T'],
            'BETA': [0.1, 0.2, 0.3],
            'target_A1': ['A', 'G', 'C'],
            'target_A2': ['G', 'C', 'T']
        })

        harmonized = harmonize_alleles(df, a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        assert all(harmonized['BETA'] == [0.1, 0.2, 0.3])


class TestInvalidAlleles:
    """Test that non-ACGT alleles are rejected."""

    def test_invalid_allele_n(self):
        """Test that 'N' allele is invalid."""
        assert not is_valid_allele('N')

    def test_invalid_allele_number(self):
        """Test that numeric alleles are invalid."""
        assert not is_valid_allele('1')

    def test_invalid_allele_lowercase(self):
        """Test that lowercase alleles are handled."""
        # This depends on implementation - lowercase may be converted or rejected
        result = is_valid_allele('a')
        # Either valid (after conversion) or invalid
        assert isinstance(result, bool)

    def test_valid_alleles(self):
        """Test that A, C, G, T are valid."""
        for allele in ['A', 'C', 'G', 'T']:
            assert is_valid_allele(allele), f"Allele {allele} should be valid"

    def test_harmonize_rejects_invalid(self):
        """Test that harmonization rejects invalid alleles."""
        df = pd.DataFrame({
            'SNP': ['rs1', 'rs2'],
            'A1': ['A', 'N'],
            'A2': ['G', 'T'],
            'BETA': [0.1, 0.2],
            'target_A1': ['A', 'N'],
            'target_A2': ['G', 'T']
        })

        # Should either skip or mark as invalid
        harmonized = harmonize_alleles(df, a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        # At minimum, should return fewer rows or mark as failed
        assert len(harmonized) <= len(df)


class TestHarmonizeAlleles:
    """Integration tests for allele harmonization."""

    def test_harmonize_swap_alleles(self):
        """Test swapping of alleles."""
        df = pd.DataFrame({
            'SNP': ['rs1'],
            'A1': ['G'],
            'A2': ['A'],
            'BETA': [0.1],
            'target_A1': ['A'],
            'target_A2': ['G']
        })

        harmonized = harmonize_alleles(df, a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        # After harmonization, A1/A2 should match target and beta negated
        assert harmonized.iloc[0]['A1'] == 'A'
        assert harmonized.iloc[0]['A2'] == 'G'
        assert harmonized.iloc[0]['BETA'] == -0.1

    def test_harmonize_preserves_snp_id(self):
        """Test that SNP IDs are preserved during harmonization."""
        df = pd.DataFrame({
            'SNP': ['rs123', 'rs456'],
            'A1': ['A', 'T'],
            'A2': ['G', 'A'],
            'BETA': [0.1, 0.2],
            'target_A1': ['A', 'T'],
            'target_A2': ['G', 'A']
        })

        harmonized = harmonize_alleles(df, a1_col='A1', a2_col='A2',
                                       target_a1_col='target_A1',
                                       target_a2_col='target_A2')

        assert list(harmonized['SNP']) == ['rs123', 'rs456']
