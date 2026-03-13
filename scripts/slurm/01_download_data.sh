#!/bin/bash
#SBATCH --job-name=oaprs_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/01_download_%j.out
#SBATCH --error=logs/01_download_%j.err

# =============================================================================
# Step 01: Download External Data
# =============================================================================
# Downloads GWAS summary statistics, LD reference panels, and annotation files.
# Validates checksums after download.
# =============================================================================

set -euo pipefail

echo "=== Step 01: Download Data ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source activate oa_prs_cpu 2>/dev/null || conda activate oa_prs_cpu 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

CONFIG="configs/config.yaml"

mkdir -p data/raw/gwas data/raw/ld_ref data/raw/annotations data/external

# --- Download GWAS summary statistics ---
echo "--- Downloading GWAS summary statistics ---"
python -m oa_prs.cli run --config "${CONFIG}" --step download_gwas 2>&1

# --- Download LD reference panels (1000 Genomes Phase 3) ---
echo "--- Downloading LD reference panels ---"
python -m oa_prs.cli run --config "${CONFIG}" --step download_ld 2>&1

# --- Download annotation files ---
echo "--- Downloading annotation files ---"
python -m oa_prs.cli run --config "${CONFIG}" --step download_annotations 2>&1

# --- Download reference genome (hg38) ---
echo "--- Downloading hg38 reference genome ---"
if [ ! -f "data/external/hg38.fa" ]; then
    wget -q -O data/external/hg38.fa.gz \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    gunzip data/external/hg38.fa.gz
    samtools faidx data/external/hg38.fa 2>/dev/null || true
fi

echo "=== Download Complete ==="
echo "Date: $(date)"
ls -lh data/raw/gwas/ data/raw/ld_ref/ data/raw/annotations/
