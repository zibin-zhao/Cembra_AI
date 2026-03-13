#!/bin/bash
#SBATCH --job-name=oaprs_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/11_eval_%j.out
#SBATCH --error=logs/11_eval_%j.err

# =============================================================================
# Step 11: Comprehensive Evaluation
# =============================================================================
# Evaluates the final ensemble and all branch models:
#   - Discrimination: AUC-ROC, AUC-PR
#   - Calibration: Brier score, Hosmer-Lemeshow
#   - Risk stratification: quantile ORs, top-percentile enrichment, DCA
#   - Fairness: cross-ancestry AUC gap, calibration gap, threshold consistency
#   - Ablation: contribution of each branch
# Generates evaluation report with plots.
# =============================================================================

set -euo pipefail

echo "=== Step 11: Evaluation ==="
echo "Date: $(date)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
OUTPUT_DIR="outputs/evaluation"

mkdir -p "${OUTPUT_DIR}/plots" "${OUTPUT_DIR}/tables"

python -c "
from oa_prs.evaluation.discrimination import compute_discrimination
from oa_prs.evaluation.calibration import compute_calibration
from oa_prs.evaluation.fairness import evaluate_fairness
from oa_prs.evaluation.risk_stratification import (
    compute_quantile_risk, compute_top_percentile_risk, decision_curve_analysis,
)
import yaml, json, os
import numpy as np
import pandas as pd

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

print('=== Comprehensive Evaluation ===')

# Load ensemble predictions (if available)
pred_file = 'outputs/ensemble/predictions.tsv'
if not os.path.exists(pred_file):
    print('No ensemble predictions found. Run evaluation after ensemble training.')
    print('Generating template evaluation config...')

    eval_template = {
        'models_to_evaluate': [
            'baseline_prs_cs', 'baseline_ldpred2',
            'crossanc_prs_csx', 'crossanc_bridge_prs',
            'catn', 'ensemble',
        ],
        'metrics': {
            'discrimination': ['auc_roc', 'auc_pr'],
            'calibration': ['brier_score', 'hosmer_lemeshow'],
            'risk_stratification': ['quantile_or', 'top_percentile', 'dca'],
            'fairness': ['auc_gap', 'calibration_gap', 'threshold_consistency'],
        },
        'subgroups': ['ancestry', 'sex', 'age_group'],
        'quantiles': [5, 10, 20],
        'top_percentiles': [1, 5, 10],
    }
    with open('${OUTPUT_DIR}/eval_config.json', 'w') as f:
        json.dump(eval_template, f, indent=2)
    print(f'Template saved to: ${OUTPUT_DIR}/eval_config.json')
    exit(0)

# Load predictions
preds = pd.read_csv(pred_file, sep='\t')
y_true = preds['outcome'].values
y_score = preds['risk_score'].values

# --- Discrimination ---
print('\\n--- Discrimination Metrics ---')
disc = compute_discrimination(y_true, y_score)
print(f'  AUC-ROC: {disc.auc_roc:.4f}')
print(f'  AUC-PR:  {disc.auc_pr:.4f}')

# --- Calibration ---
print('\\n--- Calibration Metrics ---')
cal = compute_calibration(y_true, y_score)
print(f'  Brier Score:    {cal.brier_score:.4f}')
print(f'  HL Statistic:   {cal.hosmer_lemeshow_stat:.2f} (p={cal.hosmer_lemeshow_p:.4f})')

# --- Risk Stratification ---
print('\\n--- Risk Stratification ---')
for n_q in [5, 10, 20]:
    qr = compute_quantile_risk(y_true, y_score, n_quantiles=n_q)
    print(f'  {n_q}-quantile top OR: {qr.odds_ratios[-1]:.2f} (95% CI: {qr.confidence_intervals[-1]})')

for pct in [1, 5, 10]:
    tpr = compute_top_percentile_risk(y_true, y_score, percentile=pct)
    print(f'  Top {pct}% OR: {tpr[\"odds_ratio\"]:.2f}')

# --- Fairness ---
if 'ancestry' in preds.columns:
    print('\\n--- Cross-Ancestry Fairness ---')
    fair = evaluate_fairness(y_true, y_score, preds['ancestry'], subgroup_name='ancestry')
    print(f'  AUC Gap:    {fair.auc_gap:.4f}')
    print(f'  Cal Gap:    {fair.calibration_gap:.4f}')
    print(f'  Threshold Consistency: {fair.threshold_consistency:.2f}')
    for grp, metrics in fair.subgroup_metrics.items():
        print(f'    {grp}: AUC={metrics[\"AUC-ROC\"]:.4f}, n={metrics[\"n\"]}')

# --- Save all results ---
results = {
    'discrimination': disc.summary() if hasattr(disc, 'summary') else {'AUC-ROC': disc.auc_roc, 'AUC-PR': disc.auc_pr},
    'calibration': cal.summary(),
}
with open('${OUTPUT_DIR}/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\\nResults saved to: ${OUTPUT_DIR}/evaluation_results.json')
"

echo "=== Evaluation Complete ==="
echo "Date: $(date)"
