"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, brier_score_loss

from oa_prs.evaluation.metrics import (
    compute_auc,
    compute_brier,
    compute_fairness_metric,
    compute_quantile_or
)


class TestAUCPerfect:
    """Test AUC metric with perfect predictions."""

    def test_auc_perfect(self):
        """Perfect predictions → AUC = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        auc = compute_auc(y_true, y_pred)

        assert auc == 1.0, "Perfect predictions should give AUC = 1.0"

    def test_auc_perfect_all_correct(self):
        """Test AUC with all predictions correct."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        auc = compute_auc(y_true, y_pred)

        assert auc == 1.0


class TestAUCRandom:
    """Test AUC metric with random predictions."""

    def test_auc_random(self):
        """Random predictions → AUC ≈ 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)

        auc = compute_auc(y_true, y_pred)

        assert 0.4 < auc < 0.6, \
            f"Random predictions should give AUC ≈ 0.5, got {auc}"

    def test_auc_random_seed(self):
        """Test AUC with random predictions (fixed seed)."""
        np.random.seed(123)
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.uniform(0, 1, 50)

        auc = compute_auc(y_true, y_pred)

        # Compare with sklearn
        sklearn_auc = roc_auc_score(y_true, y_pred)
        assert np.isclose(auc, sklearn_auc)


class TestBrierPerfect:
    """Test Brier score with perfect predictions."""

    def test_brier_perfect(self):
        """Perfect calibration → Brier = 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])

        brier = compute_brier(y_true, y_pred)

        assert brier == 0.0, \
            "Perfect predictions should give Brier = 0.0"

    def test_brier_worst_case(self):
        """Test Brier score with worst predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])

        brier = compute_brier(y_true, y_pred)

        assert brier == 1.0, \
            "Worst predictions should give Brier = 1.0"

    def test_brier_vs_sklearn(self):
        """Test Brier score against sklearn."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)

        brier = compute_brier(y_true, y_pred)
        sklearn_brier = brier_score_loss(y_true, y_pred)

        assert np.isclose(brier, sklearn_brier)


class TestFairnessEqualGroups:
    """Test fairness metric with equal performance."""

    def test_fairness_equal_groups(self):
        """Equal performance → AUC_gap = 0."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        auc_gap = compute_fairness_metric(y_true, y_pred, groups)

        assert auc_gap == 0.0, \
            "Equal performance should give AUC_gap = 0.0"

    def test_fairness_different_groups(self):
        """Test fairness with different group performance."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.4, 0.5])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        auc_gap = compute_fairness_metric(y_true, y_pred, groups)

        assert auc_gap > 0, \
            "Different group performance should give non-zero AUC_gap"

    def test_fairness_metric_range(self):
        """Test that fairness metric is in valid range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)
        groups = np.random.randint(0, 2, 100)

        auc_gap = compute_fairness_metric(y_true, y_pred, groups)

        assert 0 <= auc_gap <= 1, \
            "AUC_gap should be in [0, 1]"


class TestQuantileRisk:
    """Test odds ratio computation."""

    def test_quantile_risk(self):
        """Test OR computation."""
        # Create data with clear separation
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        prs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        or_q1_q4 = compute_quantile_or(prs, y_true, q_low=0.25, q_high=0.75)

        assert or_q1_q4 > 1.0, \
            "OR should be > 1 for well-separated data"

    def test_quantile_risk_no_separation(self):
        """Test OR when no risk separation."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        prs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        or_q1_q4 = compute_quantile_or(prs, y_true, q_low=0.25, q_high=0.75)

        # OR should be close to 1
        assert 0.5 < or_q1_q4 < 2.0, \
            "OR should be close to 1 for no separation"

    def test_quantile_risk_perfect_separation(self):
        """Test OR with perfect separation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        prs = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])

        or_q1_q4 = compute_quantile_or(prs, y_true, q_low=0.25, q_high=0.75)

        # Very high OR expected
        assert or_q1_q4 > 5.0, \
            "OR should be large for perfect separation"

    def test_quantile_risk_custom_quantiles(self):
        """Test OR with custom quantiles."""
        y_true = np.array([0]*50 + [1]*50)
        prs = np.concatenate([np.linspace(0, 0.5, 50),
                             np.linspace(0.5, 1.0, 50)])

        or_deciles = compute_quantile_or(prs, y_true, q_low=0.1, q_high=0.9)

        assert or_deciles > 1.0


class TestMetricsProperties:
    """Test general properties of metrics."""

    def test_auc_deterministic(self):
        """Test that AUC computation is deterministic."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.uniform(0, 1, 50)

        auc1 = compute_auc(y_true, y_pred)
        auc2 = compute_auc(y_true, y_pred)

        assert auc1 == auc2

    def test_brier_deterministic(self):
        """Test that Brier score computation is deterministic."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.uniform(0, 1, 50)

        brier1 = compute_brier(y_true, y_pred)
        brier2 = compute_brier(y_true, y_pred)

        assert brier1 == brier2

    def test_metrics_with_small_samples(self):
        """Test metrics with minimal samples."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.2, 0.8])

        auc = compute_auc(y_true, y_pred)
        brier = compute_brier(y_true, y_pred)

        assert 0 <= auc <= 1
        assert 0 <= brier <= 1
