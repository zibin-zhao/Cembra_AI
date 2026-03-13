#!/bin/bash
# =============================================================================
# Master Pipeline Script — Submits all SLURM jobs with dependencies
# =============================================================================
# Usage: bash scripts/slurm/run_full_pipeline.sh
#
# This script submits the entire OA-PRS Transfer Learning pipeline as
# a chain of SLURM jobs, each starting only after its dependencies complete.
# =============================================================================

set -euo pipefail

# Configuration
PARTITION_CPU="${PARTITION_CPU:-cpu}"
PARTITION_GPU="${PARTITION_GPU:-gpu}"
ACCOUNT="${ACCOUNT:-}"
CONTAINER="${CONTAINER:-containers/oa_prs_gpu.sif}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================"
echo " OA-PRS Transfer Learning Pipeline"
echo " Project: ${PROJECT_DIR}"
echo " CPU Partition: ${PARTITION_CPU}"
echo " GPU Partition: ${PARTITION_GPU}"
echo "========================================"

cd "${PROJECT_DIR}"

ACCOUNT_FLAG=""
if [ -n "${ACCOUNT}" ]; then
    ACCOUNT_FLAG="--account=${ACCOUNT}"
fi

# --- Step 01: Download Data ---
JOB_01=$(sbatch --parsable \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_01_download" \
    --cpus-per-task=4 --mem=16G --time=2:00:00 \
    --output=logs/01_download_%j.out \
    scripts/slurm/01_download_data.sh)
echo "Step 01 (Download): Job ${JOB_01}"

# --- Step 02: QC & Harmonize ---
JOB_02=$(sbatch --parsable \
    --dependency=afterok:${JOB_01} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_02_qc" \
    --cpus-per-task=8 --mem=32G --time=1:00:00 \
    --output=logs/02_qc_%j.out \
    scripts/slurm/02_qc_harmonize.sh)
echo "Step 02 (QC): Job ${JOB_02}"

# --- Step 03: Enformer Scoring (GPU, per-chromosome array) ---
JOB_03=$(sbatch --parsable \
    --dependency=afterok:${JOB_02} \
    --partition="${PARTITION_GPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_03_enformer" \
    --gres=gpu:a100:1 --cpus-per-task=4 --mem=64G --time=24:00:00 \
    --array=1-22 \
    --output=logs/03_enformer_%A_%a.out \
    scripts/slurm/03_enformer_scoring.sh)
echo "Step 03 (Enformer): Job ${JOB_03}"

# --- Step 04: PRS Baselines (per-chromosome array) ---
JOB_04=$(sbatch --parsable \
    --dependency=afterok:${JOB_02} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_04_baseline" \
    --cpus-per-task=1 --mem=64G --time=4:00:00 \
    --array=1-22 \
    --output=logs/04_baseline_%A_%a.out \
    scripts/slurm/04_prs_baseline.sh)
echo "Step 04 (PRS Baseline): Job ${JOB_04}"

# --- Step 05: Cross-Ancestry PRS (per-chromosome array) ---
JOB_05=$(sbatch --parsable \
    --dependency=afterok:${JOB_02} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_05_crossanc" \
    --cpus-per-task=1 --mem=64G --time=8:00:00 \
    --array=1-22 \
    --output=logs/05_crossanc_%A_%a.out \
    scripts/slurm/05_cross_ancestry_prs.sh)
echo "Step 05 (Cross-Ancestry PRS): Job ${JOB_05}"

# --- Step 06: Fine-Mapping (depends on QC + Enformer) ---
JOB_06=$(sbatch --parsable \
    --dependency=afterok:${JOB_02}:${JOB_03} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_06_finemap" \
    --cpus-per-task=16 --mem=128G --time=16:00:00 \
    --output=logs/06_finemap_%j.out \
    scripts/slurm/06_polyfun_finemapping.sh)
echo "Step 06 (Fine-Mapping): Job ${JOB_06}"

# --- Step 07: PRS Refinement (depends on baselines + fine-mapping) ---
JOB_07=$(sbatch --parsable \
    --dependency=afterok:${JOB_04}:${JOB_05}:${JOB_06} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_07_refine" \
    --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output=logs/07_refine_%j.out \
    scripts/slurm/07_prs_refinement.sh)
echo "Step 07 (PRS Refinement): Job ${JOB_07}"

# --- Step 08: TWAS/SMR ---
JOB_08=$(sbatch --parsable \
    --dependency=afterok:${JOB_02} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_08_twas" \
    --cpus-per-task=8 --mem=32G --time=4:00:00 \
    --output=logs/08_twas_%j.out \
    scripts/slurm/08_twas_smr.sh)
echo "Step 08 (TWAS/SMR): Job ${JOB_08}"

# --- Step 09: CATN Training (depends on QC + Enformer + Fine-mapping, GPU) ---
JOB_09=$(sbatch --parsable \
    --dependency=afterok:${JOB_02}:${JOB_03}:${JOB_06} \
    --partition="${PARTITION_GPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_09_catn" \
    --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=8:00:00 \
    --output=logs/09_catn_%j.out \
    scripts/slurm/09_train_catn.sh)
echo "Step 09 (CATN Training): Job ${JOB_09}"

# --- Step 10: Ensemble Stacking (depends on all model branches) ---
JOB_10=$(sbatch --parsable \
    --dependency=afterok:${JOB_07}:${JOB_08}:${JOB_09} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_10_ensemble" \
    --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/10_ensemble_%j.out \
    scripts/slurm/10_ensemble.sh)
echo "Step 10 (Ensemble): Job ${JOB_10}"

# --- Step 11: Evaluation ---
JOB_11=$(sbatch --parsable \
    --dependency=afterok:${JOB_10} \
    --partition="${PARTITION_CPU}" ${ACCOUNT_FLAG} \
    --job-name="oaprs_11_eval" \
    --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/11_eval_%j.out \
    scripts/slurm/11_evaluation.sh)
echo "Step 11 (Evaluation): Job ${JOB_11}"

echo ""
echo "========================================"
echo " All jobs submitted!"
echo " Monitor: squeue -u \$USER"
echo " Logs: logs/"
echo "========================================"
