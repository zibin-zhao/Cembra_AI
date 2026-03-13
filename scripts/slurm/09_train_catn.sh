#!/bin/bash
#SBATCH --job-name=oaprs_catn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/09_catn_%j.out
#SBATCH --error=logs/09_catn_%j.err

# =============================================================================
# Step 09: Train CATN (Cross-Ancestry Transfer Network)
# =============================================================================
# GPU job: trains the custom deep learning model in 3 phases.
# Phase 1: EUR pre-training (simulated genotypes from sumstats + LD)
# Phase 2: Domain adaptation (EUR → EAS)
# Phase 3: Individual fine-tuning (when data available)
# =============================================================================

set -euo pipefail

echo "=== CATN Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load modules (adjust to your cluster)
module load cuda/12.1 2>/dev/null || true

# Activate environment
source activate oa_prs_gpu 2>/dev/null || conda activate oa_prs_gpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
DATA_DIR="data/processed"
OUTPUT_DIR="outputs/catn"
mkdir -p "${OUTPUT_DIR}"

echo "--- Phase 1: EUR Pre-training ---"
python -m oa_prs.cli run --config "${CONFIG}" --step catn \
    2>&1 | tee "${OUTPUT_DIR}/phase1_train.log"

echo "--- Phase 2: Domain Adaptation ---"
# Phase 2 is triggered automatically by the trainer when EAS data is present

echo "--- Phase 3: Individual Fine-tuning ---"
# Phase 3 is enabled via config when individual data becomes available

echo "=== CATN Training Complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Date: $(date)"
