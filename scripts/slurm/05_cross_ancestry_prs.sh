#!/bin/bash
#SBATCH --job-name=oaprs_crossanc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/05_crossanc_%A_%a.out
#SBATCH --error=logs/05_crossanc_%A_%a.err

# =============================================================================
# Step 05: Cross-Ancestry PRS (per chromosome)
# =============================================================================
# Runs PRS-CSx (multi-ancestry Bayesian) and BridgePRS (ancestry bridging)
# to produce cross-ancestry PRS weights transferring EUR → EAS.
# =============================================================================

set -euo pipefail

CHR=${SLURM_ARRAY_TASK_ID}
echo "=== Cross-Ancestry PRS: Chromosome ${CHR} ==="
echo "Date: $(date)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
EUR_GWAS="data/processed/ukb_2019_chr${CHR}.tsv"
EAS_GWAS="data/processed/eas_chr${CHR}.tsv"
LD_EUR="data/raw/ld_ref/1kg_eur"
LD_EAS="data/raw/ld_ref/1kg_eas"
OUTPUT_DIR="outputs/cross_ancestry"

mkdir -p "${OUTPUT_DIR}/prs_csx" "${OUTPUT_DIR}/bridge_prs"

# --- PRS-CSx (multi-ancestry) ---
echo "--- PRS-CSx (EUR+EAS, chr${CHR}) ---"
python -c "
from oa_prs.models.transfer.prs_csx import PRSCSxRunner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

runner = PRSCSxRunner(
    ref_dirs={'EUR': '${LD_EUR}', 'EAS': '${LD_EAS}'},
    n_gwas={'EUR': cfg['data']['gwas']['eur']['sample_size'],
            'EAS': cfg['data']['gwas']['eas']['sample_size']},
)
runner.run(
    sumstats_files={'EUR': '${EUR_GWAS}', 'EAS': '${EAS_GWAS}'},
    output_dir='${OUTPUT_DIR}/prs_csx',
    chrom=${CHR},
)
"

# --- BridgePRS ---
echo "--- BridgePRS (EUR→EAS, chr${CHR}) ---"
python -c "
from oa_prs.models.transfer.bridge_prs import BridgePRSRunner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

runner = BridgePRSRunner(
    ld_ref_base='${LD_EUR}',
    ld_ref_target='${LD_EAS}',
)
runner.run(
    base_sumstats='${EUR_GWAS}',
    target_sumstats='${EAS_GWAS}',
    output_dir='${OUTPUT_DIR}/bridge_prs',
    chrom=${CHR},
)
"

echo "=== Cross-Ancestry Chr${CHR} Complete ==="
echo "Date: $(date)"
