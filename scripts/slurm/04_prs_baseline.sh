#!/bin/bash
#SBATCH --job-name=oaprs_baseline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/04_baseline_%A_%a.out
#SBATCH --error=logs/04_baseline_%A_%a.err

# =============================================================================
# Step 04: PRS Baseline Methods (per chromosome)
# =============================================================================
# Runs PRS-CS and LDpred2-auto on EUR GWAS, producing per-SNP posterior weights.
# These are single-ancestry baselines before cross-ancestry transfer.
# =============================================================================

set -euo pipefail

CHR=${SLURM_ARRAY_TASK_ID}
echo "=== PRS Baselines: Chromosome ${CHR} ==="
echo "Date: $(date)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
GWAS_FILE="data/processed/ukb_2019_chr${CHR}.tsv"
LD_DIR="data/raw/ld_ref/1kg_eur"
OUTPUT_DIR="outputs/prs_baseline"

mkdir -p "${OUTPUT_DIR}/prs_cs" "${OUTPUT_DIR}/ldpred2"

# --- PRS-CS ---
echo "--- PRS-CS (EUR, chr${CHR}) ---"
python -c "
from oa_prs.models.base.prs_cs import PRSCSRunner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

runner = PRSCSRunner(
    ref_dir='${LD_DIR}',
    n_gwas=cfg['data']['gwas']['eur']['sample_size'],
    phi=cfg.get('models', {}).get('prs_cs', {}).get('phi', 'auto'),
)
runner.run(
    sumstats_file='${GWAS_FILE}',
    output_file='${OUTPUT_DIR}/prs_cs/weights_chr${CHR}.tsv',
    chrom=${CHR},
)
"

# --- LDpred2-auto ---
echo "--- LDpred2-auto (EUR, chr${CHR}) ---"
python -c "
from oa_prs.models.base.ldpred2 import LDpred2Runner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

runner = LDpred2Runner(
    ld_ref_dir='${LD_DIR}',
    n_gwas=cfg['data']['gwas']['eur']['sample_size'],
    sparse=True,
)
runner.run(
    sumstats_file='${GWAS_FILE}',
    output_file='${OUTPUT_DIR}/ldpred2/weights_chr${CHR}.tsv',
    chrom=${CHR},
)
"

echo "=== Baselines Chr${CHR} Complete ==="
echo "Date: $(date)"
