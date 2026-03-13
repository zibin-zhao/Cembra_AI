#!/bin/bash
#SBATCH --job-name=oaprs_twas
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/08_twas_%j.out
#SBATCH --error=logs/08_twas_%j.err

# =============================================================================
# Step 08: TWAS (S-PrediXcan) & SMR-HEIDI
# =============================================================================
# Identifies causal genes via transcriptome-wide association.
# S-PrediXcan: uses GTEx v8 models for EUR tissues relevant to OA.
# SMR-HEIDI: Mendelian randomization to distinguish pleiotropy from linkage.
# Gene-level scores are used as features in the ensemble.
# =============================================================================

set -euo pipefail

echo "=== Step 08: TWAS & SMR ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"
GWAS_FILE="data/processed/mvp_ukb_2022_harmonized.tsv"
OUTPUT_DIR="outputs/twas"

mkdir -p "${OUTPUT_DIR}/s_predixcan" "${OUTPUT_DIR}/smr" "${OUTPUT_DIR}/combined"

# --- S-PrediXcan across OA-relevant tissues ---
echo "--- Running S-PrediXcan ---"
python -c "
from oa_prs.models.twas.s_predixcan import SPrediXcanRunner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

tissues = cfg.get('twas', {}).get('tissues', [
    'Muscle_Skeletal', 'Adipose_Subcutaneous', 'Nerve_Tibial',
    'Whole_Blood', 'Cells_Cultured_fibroblasts',
])

runner = SPrediXcanRunner(
    gwas_file='${GWAS_FILE}',
    model_db_dir=cfg.get('twas', {}).get('model_db_dir', 'data/external/gtex_v8_models'),
    output_dir='${OUTPUT_DIR}/s_predixcan',
)

for tissue in tissues:
    print(f'  S-PrediXcan: {tissue}')
    runner.run_tissue(tissue=tissue)

# S-MultiXcan (cross-tissue meta-analysis)
print('  S-MultiXcan (cross-tissue)')
runner.run_multixcan(tissues=tissues, output_file='${OUTPUT_DIR}/s_predixcan/multixcan_results.tsv')
"

# --- SMR-HEIDI ---
echo "--- Running SMR-HEIDI ---"
python -c "
from oa_prs.models.twas.smr_heidi import SMRHEIDIRunner
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

runner = SMRHEIDIRunner(
    gwas_file='${GWAS_FILE}',
    eqtl_dir=cfg.get('twas', {}).get('eqtl_dir', 'data/external/gtex_v8_eqtl'),
    ld_ref='data/raw/ld_ref/1kg_eur',
    output_dir='${OUTPUT_DIR}/smr',
)
runner.run(heidi_threshold=0.05)
"

# --- Combine TWAS results into gene-level features ---
echo "--- Combining TWAS results ---"
python -c "
import pandas as pd
import os, glob

# Merge S-PrediXcan and SMR results
spredixcan_files = glob.glob('${OUTPUT_DIR}/s_predixcan/*_results.tsv')
smr_file = '${OUTPUT_DIR}/smr/smr_results.tsv'

dfs = []
for f in spredixcan_files:
    df = pd.read_csv(f, sep='\t')
    dfs.append(df)

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    if os.path.exists(smr_file):
        smr = pd.read_csv(smr_file, sep='\t')
        combined = combined.merge(smr[['gene', 'smr_p', 'heidi_p']], on='gene', how='left')
    combined.to_csv('${OUTPUT_DIR}/combined/twas_gene_scores.tsv', sep='\t', index=False)
    print(f'Combined TWAS: {len(combined)} gene-tissue associations')
"

echo "=== TWAS & SMR Complete ==="
echo "Date: $(date)"
