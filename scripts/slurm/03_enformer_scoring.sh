#!/bin/bash
#SBATCH --job-name=oaprs_enformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/03_enformer_%A_%a.out
#SBATCH --error=logs/03_enformer_%A_%a.err

# =============================================================================
# Step 03: Enformer Variant Effect Scoring (GPU, per chromosome)
# =============================================================================
# Scores variants using Enformer (DeepMind) for functional annotation.
# Produces SAD (SNP Activity Difference) scores for each variant.
# =============================================================================

set -euo pipefail

CHR=${SLURM_ARRAY_TASK_ID}
echo "=== Enformer Scoring: Chromosome ${CHR} ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load cuda/12.1 2>/dev/null || true
source activate oa_prs_gpu 2>/dev/null || conda activate oa_prs_gpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

VARIANT_FILE="data/processed/variants_chr${CHR}.tsv"
OUTPUT_FILE="outputs/enformer/enformer_sad_chr${CHR}.h5"
GENOME_FASTA="data/external/hg38.fa"

mkdir -p outputs/enformer

python -c "
from oa_prs.models.functional.enformer_scorer import EnformerScorer

scorer = EnformerScorer(
    genome_fasta='${GENOME_FASTA}',
    batch_size=4,
    device='cuda',
)
scorer.score_variants_file(
    variant_file='${VARIANT_FILE}',
    output_file='${OUTPUT_FILE}',
    chromosome=${CHR},
)
"

echo "=== Enformer Chr${CHR} Complete ==="
echo "Output: ${OUTPUT_FILE}"
echo "Date: $(date)"
