#!/bin/bash
#SBATCH --job-name=oaprs_refine
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/07_refine_%j.out
#SBATCH --error=logs/07_refine_%j.err

# =============================================================================
# Step 07: PRS Refinement with Functional Priors
# =============================================================================
# Refines PRS weights from Steps 04-05 using fine-mapping results from Step 06.
# Two approaches: posterior_direct (use SuSiE PIPs) and prior_reweight
# (multiply baseline weights by functional enrichment).
# =============================================================================

set -euo pipefail

echo "=== Step 07: PRS Refinement ==="
echo "Date: $(date)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
BASELINE_DIR="outputs/prs_baseline"
CROSSANC_DIR="outputs/cross_ancestry"
FINEMAP_DIR="outputs/finemapping"
OUTPUT_DIR="outputs/refined"

mkdir -p "${OUTPUT_DIR}"

python -c "
from oa_prs.models.ensemble.prs_refiner import PRSRefiner
import yaml, glob, os

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

refiner = PRSRefiner(
    finemapping_dir='${FINEMAP_DIR}/susie_inf',
    method=cfg.get('ensemble', {}).get('refinement', {}).get('method', 'posterior_direct'),
)

# Refine each baseline method
methods = {
    'prs_cs': '${BASELINE_DIR}/prs_cs',
    'ldpred2': '${BASELINE_DIR}/ldpred2',
    'prs_csx_eur': '${CROSSANC_DIR}/prs_csx',
    'prs_csx_eas': '${CROSSANC_DIR}/prs_csx',
    'bridge_prs': '${CROSSANC_DIR}/bridge_prs',
}

for method_name, weight_dir in methods.items():
    print(f'Refining {method_name}...')
    refiner.refine_weights(
        weight_dir=weight_dir,
        output_file=f'${OUTPUT_DIR}/{method_name}_refined.tsv',
        method_name=method_name,
    )
    print(f'  Done: ${OUTPUT_DIR}/{method_name}_refined.tsv')
"

echo "=== Refinement Complete ==="
echo "Refined weight files:"
ls -lh ${OUTPUT_DIR}/ 2>/dev/null
echo "Date: $(date)"
