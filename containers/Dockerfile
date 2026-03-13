# =============================================================================
# OA-PRS Transfer Learning — Docker Image (GPU)
# =============================================================================
# Multi-stage build: base system → Python deps → project code
# Compatible with NVIDIA GPU runtime for Enformer and CATN training.
# =============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    tabix \
    samtools \
    bcftools \
    plink1.9 \
    plink2 \
    r-base \
    r-base-dev \
    libhdf5-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages for LDpred2
RUN R -e "install.packages(c('bigsnpr', 'bigstatsr', 'data.table', 'R.utils'), repos='https://cloud.r-project.org/')"

# Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies stage ----
FROM base AS deps

WORKDIR /opt/oa_prs

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[dev]"

# Install additional GPU-specific packages
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Application stage ----
FROM deps AS app

WORKDIR /opt/oa_prs

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create working directories
RUN mkdir -p data/raw data/processed data/external data/toy \
    outputs/enformer outputs/prs_baseline outputs/cross_ancestry \
    outputs/finemapping outputs/catn outputs/twas outputs/ensemble \
    outputs/evaluation logs

# Default entry point
ENTRYPOINT ["python", "-m", "oa_prs.cli"]
CMD ["--help"]

# Labels
LABEL maintainer="OA-PRS Team"
LABEL description="OA-PRS Transfer Learning Pipeline (GPU)"
LABEL version="0.1.0"
LABEL license="Apache-2.0"
