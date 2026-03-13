#!/bin/bash
# =============================================================================
# Build container images for OA-PRS pipeline
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAG="${1:-latest}"

echo "=== Building OA-PRS Containers ==="
echo "Project: ${PROJECT_DIR}"
echo "Tag: ${TAG}"

# --- Docker (for local development / cloud) ---
echo ""
echo "--- Building Docker image ---"
cd "${PROJECT_DIR}"
docker build \
    -t oa_prs_gpu:${TAG} \
    -f containers/Dockerfile \
    .
echo "Docker image: oa_prs_gpu:${TAG}"

# --- Singularity/Apptainer (for HPC) ---
echo ""
echo "--- Building Singularity image ---"
if command -v singularity &> /dev/null; then
    cd "${PROJECT_DIR}"
    sudo singularity build \
        containers/oa_prs_gpu.sif \
        containers/singularity.def
    echo "Singularity image: containers/oa_prs_gpu.sif"
elif command -v apptainer &> /dev/null; then
    cd "${PROJECT_DIR}"
    apptainer build \
        containers/oa_prs_gpu.sif \
        containers/singularity.def
    echo "Apptainer image: containers/oa_prs_gpu.sif"
else
    echo "WARN: Neither singularity nor apptainer found. Skipping SIF build."
    echo "      To build on HPC: singularity build oa_prs_gpu.sif containers/singularity.def"
fi

echo ""
echo "=== Build Complete ==="
