#!/bin/bash
#SBATCH --job-name=oaprs_finemap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/06_finemap_%j.out
#SBATCH --error=logs/06_finemap_%j.err

# =============================================================================
# Step 06: Functional Fine-Mapping (PolyFun + SuSiE-inf)
# =============================================================================
# Uses PolyFun to estimate per-SNP functional priors from annotations
# (baselineLF2.2 + Enformer SAD + TURF/TLand), then runs SuSiE-inf
# for fine-mapping with infinitesimal effects (suited for polygenic OA).
# =============================================================================

set -euo pipefail

echo "=== Step 06: Fine-Mapping ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
GWAS_DIR="data/processed"
ENFORMER_DIR="outputs/enformer"
ANNOT_DIR="data/raw/annotations"
OUTPUT_DIR="outputs/finemapping"

mkdir -p "${OUTPUT_DIR}/polyfun" "${OUTPUT_DIR}/susie_inf"

# --- Step 6a: Merge Enformer scores into annotation matrix ---
echo "--- Merging Enformer SAD scores with annotations ---"
python -c "
from oa_prs.models.functional.annotation import AnnotationBuilder
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

builder = AnnotationBuilder(
    annotation_dir='${ANNOT_DIR}',
    enformer_dir='${ENFORMER_DIR}',
    annotation_cfg=cfg.get('annotations', {}),
)
builder.build_annotation_matrix(
    output_file='${OUTPUT_DIR}/annotation_matrix.tsv',
)
"

# --- Step 6b: PolyFun (estimate per-SNP functional priors) ---
echo "--- Running PolyFun ---"
for CHR in $(seq 1 22); do
    echo "  PolyFun chr${CHR}..."
    python -c "
from oa_prs.models.functional.polyfun_runner import PolyFunRunner

runner = PolyFunRunner(
    sumstats_file='${GWAS_DIR}/mvp_ukb_2022_chr${CHR}.tsv',
    annotation_file='${OUTPUT_DIR}/annotation_matrix.tsv',
    output_dir='${OUTPUT_DIR}/polyfun',
    n_threads=${SLURM_CPUS_PER_TASK},
)
runner.estimate_priors(chrom=${CHR})
" &
    # Limit parallelism to available CPUs
    if (( $(jobs -r | wc -l) >= ${SLURM_CPUS_PER_TASK} )); then
        wait -n
    fi
done
wait

# --- Step 6c: SuSiE-inf (fine-mapping with infinitesimal effects) ---
echo "--- Running SuSiE-inf ---"
for CHR in $(seq 1 22); do
    echo "  SuSiE-inf chr${CHR}..."
    python -c "
from oa_prs.models.functional.susie_inf import SuSiEInfRunner

runner = SuSiEInfRunner(
    prior_file='${OUTPUT_DIR}/polyfun/priors_chr${CHR}.tsv',
    ld_dir='data/raw/ld_ref/1kg_eur',
    output_dir='${OUTPUT_DIR}/susie_inf',
)
runner.run(chrom=${CHR})
" &
    if (( $(jobs -r | wc -l) >= ${SLURM_CPUS_PER_TASK} )); then
        wait -n
    fi
done
wait

echo "=== Fine-Mapping Complete ==="
echo "Date: $(date)"
echo "Output files:"
ls -lh ${OUTPUT_DIR}/susie_inf/ 2>/dev/null | head -10
