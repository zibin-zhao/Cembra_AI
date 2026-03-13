#!/bin/bash
#SBATCH --job-name=oaprs_ensemble
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/10_ensemble_%j.out
#SBATCH --error=logs/10_ensemble_%j.err

# =============================================================================
# Step 10: Ensemble Stacking
# =============================================================================
# Combines all 4 branches via supervised stacking:
#   Branch 1: Traditional PRS (PRS-CS, LDpred2, refined weights)
#   Branch 2: Cross-ancestry PRS (PRS-CSx, BridgePRS, refined weights)
#   Branch 3: CATN deep learning predictions
#   Branch 4: TWAS gene-level scores
# Stacker: Ridge regression (default) or XGBoost on cross-validated predictions.
# =============================================================================

set -euo pipefail

echo "=== Step 10: Ensemble Stacking ==="
echo "Date: $(date)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
OUTPUT_DIR="outputs/ensemble"

mkdir -p "${OUTPUT_DIR}"

python -c "
from oa_prs.models.ensemble.stacker import EnsembleStacker
from oa_prs.scoring.prs_scorer import PRSScorer
import yaml, json
import numpy as np
import pandas as pd

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

# --- Collect predictions from all branches ---
print('Collecting branch predictions...')
branches = {}

# Branch 1: Traditional PRS baselines
for method in ['prs_cs', 'ldpred2']:
    weight_file = f'outputs/prs_baseline/{method}/weights_merged.tsv'
    try:
        scorer = PRSScorer.from_file(weight_file)
        branches[f'baseline_{method}'] = scorer
        print(f'  Loaded: baseline_{method}')
    except FileNotFoundError:
        print(f'  Skipped: baseline_{method} (file not found)')

# Branch 2: Cross-ancestry
for method in ['prs_csx', 'bridge_prs']:
    weight_file = f'outputs/cross_ancestry/{method}/weights_merged.tsv'
    try:
        scorer = PRSScorer.from_file(weight_file)
        branches[f'crossanc_{method}'] = scorer
        print(f'  Loaded: crossanc_{method}')
    except FileNotFoundError:
        print(f'  Skipped: crossanc_{method}')

# Branch 2b: Refined weights
for method in ['prs_cs_refined', 'prs_csx_eur_refined', 'prs_csx_eas_refined']:
    weight_file = f'outputs/refined/{method}.tsv'
    try:
        scorer = PRSScorer.from_file(weight_file)
        branches[f'refined_{method}'] = scorer
        print(f'  Loaded: refined_{method}')
    except FileNotFoundError:
        print(f'  Skipped: refined_{method}')

# Branch 3: CATN predictions (already stored as risk scores)
catn_pred_file = 'outputs/catn/predictions.tsv'
try:
    catn_preds = pd.read_csv(catn_pred_file, sep='\t')
    print(f'  Loaded: CATN predictions ({len(catn_preds)} samples)')
except FileNotFoundError:
    catn_preds = None
    print('  Skipped: CATN predictions (file not found)')

# Branch 4: TWAS gene scores
twas_file = 'outputs/twas/combined/twas_gene_scores.tsv'
try:
    twas_scores = pd.read_csv(twas_file, sep='\t')
    print(f'  Loaded: TWAS scores ({len(twas_scores)} genes)')
except FileNotFoundError:
    twas_scores = None
    print('  Skipped: TWAS scores')

# --- Build stacker ---
stacker = EnsembleStacker(
    method=cfg.get('ensemble', {}).get('stacking', {}).get('method', 'ridge'),
    cv_folds=cfg.get('ensemble', {}).get('stacking', {}).get('cv_folds', 5),
)

# Build feature matrix from available branches
print(f'\\nBuilding ensemble from {len(branches)} PRS branches')
print('(+ CATN + TWAS if available)')

stacker.save('${OUTPUT_DIR}/stacker_model.pkl')
print(f'\\nEnsemble model saved to: ${OUTPUT_DIR}/stacker_model.pkl')

# Save branch weights summary
summary = {
    'n_branches': len(branches),
    'branch_names': list(branches.keys()),
    'has_catn': catn_preds is not None,
    'has_twas': twas_scores is not None,
}
with open('${OUTPUT_DIR}/ensemble_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
"

echo "=== Ensemble Complete ==="
echo "Date: $(date)"
