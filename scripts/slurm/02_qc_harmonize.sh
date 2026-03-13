#!/bin/bash
#SBATCH --job-name=oaprs_qc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/02_qc_%j.out
#SBATCH --error=logs/02_qc_%j.err

# =============================================================================
# Step 02: Quality Control & Harmonization
# =============================================================================
# Applies QC filters (MAF, INFO, ambiguous SNPs), harmonizes alleles to hg38,
# standardizes column names, and splits by chromosome.
# =============================================================================

set -euo pipefail

echo "=== Step 02: QC & Harmonize ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"

mkdir -p data/processed

# --- Run QC pipeline ---
echo "--- Running QC on GWAS summary statistics ---"
python -m oa_prs.cli run --config "${CONFIG}" --step qc 2>&1

# --- Harmonize alleles ---
echo "--- Harmonizing alleles to hg38 ---"
python -m oa_prs.cli run --config "${CONFIG}" --step harmonize 2>&1

# --- Standardize columns ---
echo "--- Standardizing column formats ---"
python -m oa_prs.cli run --config "${CONFIG}" --step standardize 2>&1

# --- Split by chromosome for parallel processing ---
echo "--- Splitting by chromosome ---"
python -c "
from oa_prs.data.standardize import split_by_chromosome
import pandas as pd

for study in ['ukb_2019', 'mvp_ukb_2022', 'eas']:
    fpath = f'data/processed/{study}_harmonized.tsv'
    try:
        df = pd.read_csv(fpath, sep='\t')
        split_by_chromosome(df, output_dir='data/processed', prefix=f'{study}')
        print(f'  Split {study}: {len(df)} variants across {df[\"CHR\"].nunique()} chromosomes')
    except FileNotFoundError:
        print(f'  Skipping {study} (not available)')
"

echo "=== QC & Harmonization Complete ==="
echo "Date: $(date)"
echo "Processed files:"
ls -lh data/processed/*.tsv 2>/dev/null | head -20
