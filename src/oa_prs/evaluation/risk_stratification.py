"""
Risk Stratification Analysis.

Quantile risk ratios, odds ratios for top PRS percentiles, and
Decision Curve Analysis (DCA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskStratificationResults:
    """Container for risk stratification results."""

    quantile_table: pd.DataFrame    # Quantile-based risk table
    top_percentile_ors: dict[int, dict[str, float]]  # {percentile: {OR, CI_low, CI_high, p}}
    dca_thresholds: np.ndarray | None
    dca_net_benefit: np.ndarray | None

    def summary(self) -> dict:
        return {
            "quantile_table": self.quantile_table.to_dict(),
            "top_percentile_ors": self.top_percentile_ors,
        }


def compute_quantile_risk(
    y_true: np.ndarray,
    prs: np.ndarray,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """
    Compute case rate by PRS quantile.

    Args:
        y_true: Binary outcomes [n_samples].
        prs: Polygenic risk scores [n_samples].
        n_quantiles: Number of quantile groups (e.g., 10 = deciles).

    Returns:
        DataFrame with quantile, n, n_cases, prevalence, odds_ratio (vs lowest quantile).
    """
    y_true = np.asarray(y_true, dtype=int)
    prs = np.asarray(prs, dtype=float)

    quantile_labels = pd.qcut(prs, q=n_quantiles, labels=False, duplicates="drop")

    rows = []
    # Reference: lowest quantile
    ref_mask = quantile_labels == 0
    ref_cases = y_true[ref_mask].sum()
    ref_controls = ref_mask.sum() - ref_cases
    ref_odds = (ref_cases + 0.5) / (ref_controls + 0.5)  # Add 0.5 for Haldane correction

    for q in sorted(np.unique(quantile_labels)):
        mask = quantile_labels == q
        n = mask.sum()
        n_cases = y_true[mask].sum()
        n_controls = n - n_cases
        prevalence = n_cases / n if n > 0 else 0
        odds = (n_cases + 0.5) / (n_controls + 0.5)
        odds_ratio = odds / ref_odds

        rows.append({
            "quantile": int(q) + 1,
            "n": n,
            "n_cases": int(n_cases),
            "prevalence": prevalence,
            "odds_ratio_vs_Q1": odds_ratio,
        })

    return pd.DataFrame(rows)


def compute_top_percentile_risk(
    y_true: np.ndarray,
    prs: np.ndarray,
    percentiles: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute odds ratios for top X% PRS vs. the rest.

    Args:
        y_true: Binary outcomes.
        prs: Risk scores.
        percentiles: List of top percentiles to evaluate (default: [1, 5, 10, 20]).

    Returns:
        Dict mapping percentile → {OR, CI_low, CI_high, p_value}.
    """
    if percentiles is None:
        percentiles = [1, 5, 10, 20]

    y_true = np.asarray(y_true, dtype=int)
    prs = np.asarray(prs, dtype=float)
    results = {}

    for pct in percentiles:
        threshold = np.percentile(prs, 100 - pct)
        top = prs >= threshold
        rest = ~top

        a = y_true[top].sum()       # Top, case
        b = top.sum() - a           # Top, control
        c = y_true[rest].sum()      # Rest, case
        d = rest.sum() - c          # Rest, control

        # Odds ratio with Haldane correction
        OR = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        log_or = np.log(OR)
        se_log_or = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
        ci_low = np.exp(log_or - 1.96 * se_log_or)
        ci_high = np.exp(log_or + 1.96 * se_log_or)
        z = log_or / se_log_or
        from scipy.stats import norm
        p_value = 2 * norm.sf(abs(z))

        results[pct] = {
            "OR": float(OR),
            "CI_low": float(ci_low),
            "CI_high": float(ci_high),
            "p_value": float(p_value),
            "n_top": int(top.sum()),
            "cases_in_top": int(a),
        }
        logger.info("Top %d%%: OR=%.2f (%.2f-%.2f), p=%.2e", pct, OR, ci_low, ci_high, p_value)

    return results


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decision Curve Analysis — net benefit across threshold probabilities.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        thresholds: Array of threshold probabilities to evaluate.

    Returns:
        Tuple of (thresholds, net_benefit).
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    net_benefits = []
    for pt in thresholds:
        predicted_positive = y_prob >= pt
        tp = (predicted_positive & (y_true == 1)).sum()
        fp = (predicted_positive & (y_true == 0)).sum()
        nb = (tp / n) - (fp / n) * (pt / (1 - pt + 1e-10))
        net_benefits.append(nb)

    return thresholds, np.array(net_benefits)


def compute_risk_stratification(
    y_true: np.ndarray,
    prs: np.ndarray,
    n_quantiles: int = 10,
    top_percentiles: list[int] | None = None,
    run_dca: bool = True,
) -> RiskStratificationResults:
    """Full risk stratification analysis."""
    quantile_table = compute_quantile_risk(y_true, prs, n_quantiles)
    top_ors = compute_top_percentile_risk(y_true, prs, top_percentiles)

    dca_thresh = dca_nb = None
    if run_dca:
        dca_thresh, dca_nb = decision_curve_analysis(y_true, prs)

    return RiskStratificationResults(
        quantile_table=quantile_table,
        top_percentile_ors=top_ors,
        dca_thresholds=dca_thresh,
        dca_net_benefit=dca_nb,
    )
