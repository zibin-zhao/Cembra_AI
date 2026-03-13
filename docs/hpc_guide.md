# OA-PRS Transfer Learning: HPC Deployment Guide

**Version**: 1.0
**Date**: 2026-03-13
**License**: Apache 2.0

This guide covers deployment and execution of the OA-PRS pipeline on high-performance computing (HPC) clusters using SLURM job scheduling.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [SLURM Configuration](#slurm-configuration)
4. [Job Submission](#job-submission)
5. [Dependency Management](#dependency-management)
6. [GPU Optimization](#gpu-optimization)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Resource Estimates](#resource-estimates)
9. [Data Management](#data-management)
10. [Best Practices](#best-practices)

---

## System Requirements

### Minimum Specifications

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **CPU** | 16 cores | For parallel processing |
| **RAM** | 64 GB | Per job; 128+ GB for CATN |
| **Storage** | 100 GB | For full pipeline (sumstats + LD + models) |
| **GPU** | Optional | NVIDIA A100/H100 recommended for CATN (4-6 hr vs 30+ hr CPU) |
| **Network** | 10 Mbps+ | For data transfer |
| **Python** | 3.9+ | 3.10+ preferred |
| **CUDA** | 11.8+ | If using GPU |
| **CuDNN** | 8.6+ | For GPU acceleration |

### Cluster Environment Examples

#### Example 1: Slurm on XSEDE/ACCESS

```bash
# Check available nodes
sinfo

# Check available GPU
sinfo --Node --format="NodeList,CPUs,Memory,Gres"

# Account setup
sacctmgr show user $USER
```

#### Example 2: Slurm on University Cluster

```bash
# Get queue info
squeue
qos-info

# Check resource limits
scontrol show config | grep -E "MaxCPU|MaxNodes|MaxMemory"
```

---

## Environment Setup

### Step 1: Load Required Modules

```bash
# Load Python
module load python/3.10
# or: module load anaconda/3/latest

# Load CUDA (if GPU available)
module load cuda/11.8
module load cudnn/8.6

# Load compiler tools (optional, for building from source)
module load gcc/11.2
module load openmpi/4.1.1

# List loaded modules
module list
```

### Step 2: Create Conda Environment

```bash
# Clone the repository
git clone https://github.com/your-org/oa-prs-transfer.git
cd oa-prs-transfer

# Create environment from file
conda env create -f environment.yml --prefix $SCRATCH/conda_envs/oa-prs

# Activate environment
conda activate $SCRATCH/conda_envs/oa-prs

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import oaprs_transfer; print(oaprs_transfer.__version__)"
```

### Step 3: Data Directory Setup

```bash
# Create data directories
mkdir -p /path/to/project/{data,results,logs}

# Data subdirectories
mkdir -p data/raw data/processed data/splits
mkdir -p data/ld_matrices_eur data/ld_matrices_eas
mkdir -p data/annotations

# Results subdirectories
mkdir -p results/{prs_weights,predictions,models,evaluation}

# Set permissions
chmod -R 755 /path/to/project
```

### Step 4: Download Data

```bash
# Download toy data (small, ~5 GB)
python scripts/download_toy_data.py --output-dir data/

# Or download full data (large, ~50 GB) - OPTIONAL
# python scripts/download_full_data.py --output-dir data/

# Verify checksums
python -c "from oaprs_transfer.data import verify_data; verify_data('data/')"
```

---

## SLURM Configuration

### Basic SLURM Parameters

```bash
#SBATCH --job-name=job_name           # Job identifier
#SBATCH --nodes=1                     # Number of compute nodes
#SBATCH --ntasks=1                    # Number of MPI tasks
#SBATCH --cpus-per-task=16            # CPUs per task
#SBATCH --mem=64G                     # Total memory
#SBATCH --time=12:00:00               # Walltime (hours:minutes:seconds)
#SBATCH --partition=gpu               # Queue/partition name
#SBATCH --gpus=1                      # Number of GPUs
#SBATCH --output=logs/%j.out          # Output file (%j = job ID)
#SBATCH --error=logs/%j.err           # Error file
#SBATCH --mail-type=ALL               # Email notifications
#SBATCH --mail-user=user@institution.edu
```

### Partition Selection

```bash
# Determine available partitions
sinfo -l

# Typical partitions:
#   gpu         - GPU nodes (A100, H100)
#   cpu         - CPU-only nodes
#   long        - Long-running jobs (48+ hours)
#   interactive - Interactive jobs
#   mem         - High-memory nodes (512+ GB)
```

### Example Clusters

#### MIT Supercloud

```bash
# Available partitions: work, xgpu, rvl, etc.
#SBATCH --partition=xgpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
```

#### NERSC

```bash
# Available partitions: gpu, cpu, long
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
```

#### University Cluster (Generic)

```bash
# Query available GPUs
sinfo --format "%20N %10c %20C %20m %20G"

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
```

---

## Job Submission

### Single Job: PRS-CS Pipeline

```bash
#!/bin/bash
#SBATCH --job-name=oa-prs-cs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prs_cs_%j.out
#SBATCH --error=logs/prs_cs_%j.err

# Setup
module load python/3.10 cuda/11.8
conda activate $SCRATCH/conda_envs/oa-prs

# Run
cd /path/to/oa-prs-transfer
python scripts/run_pipeline.py \
    --config configs/toy_example.yaml \
    --pipeline prs_cs \
    --output results/prs_cs_run \
    --n-jobs 8
```

**Submit**:
```bash
sbatch jobs/slurm_prs_cs.sh
```

---

### GPU Job: CATN Deep Learning Training

```bash
#!/bin/bash
#SBATCH --job-name=oa-prs-catn
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/catn_%j.out
#SBATCH --error=logs/catn_%j.err

# Setup
module load python/3.10 cuda/11.8 cudnn/8.6
conda activate $SCRATCH/conda_envs/oa-prs

# GPU diagnostics
nvidia-smi
echo "GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Run
cd /path/to/oa-prs-transfer
python scripts/train_catn.py \
    --config configs/production.yaml \
    --output results/catn_run \
    --gpu 0 \
    --mixed-precision \
    --seed 42
```

**Submit**:
```bash
sbatch jobs/slurm_catn.sh
```

**Monitor GPU**:
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

---

### Master Pipeline with Dependency Chain

```bash
#!/bin/bash
#SBATCH --job-name=oa-prs-master
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/master_%j.out

module load python/3.10
conda activate $SCRATCH/conda_envs/oa-prs

cd /path/to/oa-prs-transfer

# Run master script that manages all dependencies
python scripts/slurm_master.py \
    --config configs/production.yaml \
    --output results/full_pipeline \
    --partition gpu \
    --gpu-hours 8 \
    --cpu-hours 24
```

---

### Array Job: Ablation Studies

```bash
#!/bin/bash
#SBATCH --job-name=oa-prs-ablation
#SBATCH --array=1-8%4                # 8 jobs, max 4 simultaneous
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/ablation_%a_%j.out

module load python/3.10 cuda/11.8
conda activate $SCRATCH/conda_envs/oa-prs

cd /path/to/oa-prs-transfer

# Map array index to branch name
declare -a BRANCHES=("prs_cs" "ldpred2" "prs_csx" "bridgeprs" \
                      "enformer" "twas_eur" "twas_eas" "catn")
BRANCH=${BRANCHES[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Running ablation without branch: $BRANCH"

python scripts/ablation_study.py \
    --config configs/production.yaml \
    --exclude-branch "$BRANCH" \
    --output "results/ablation_no_$BRANCH"
```

**Submit**:
```bash
sbatch jobs/slurm_ablation.sh

# Check status
squeue --array

# Cancel all
scancel -w job_id
```

---

## Dependency Management

### Sequential Pipeline (One After Another)

```bash
# Step 1: Data preparation
JOB1=$(sbatch --parsable jobs/slurm_data_prep.sh)
echo "Data prep submitted: $JOB1"

# Step 2: PRS methods (depend on Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/slurm_prs_methods.sh)
echo "PRS methods submitted: $JOB2"

# Step 3: CATN training (depend on Step 1)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/slurm_catn.sh)
echo "CATN submitted: $JOB3"

# Step 4: Ensemble (depend on Steps 2 & 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 jobs/slurm_ensemble.sh)
echo "Ensemble submitted: $JOB4"

# Step 5: Evaluation (depend on Step 4)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/slurm_evaluation.sh)
echo "Evaluation submitted: $JOB5"
```

### Parallel Pipeline (Conditional Dependencies)

```bash
# Master script managing all dependencies
sbatch --job-name=oa-prs-pipeline scripts/slurm_master.sh --config configs/production.yaml
```

---

## GPU Optimization

### CUDA Configuration

```bash
# Check CUDA capability
nvidia-smi
nvidia-smi --query-gpu=name,capability_major,capability_minor --format=csv

# Required: Compute Capability >= 7.0 (for mixed precision)
```

### Mixed Precision Training (Recommended)

```bash
# In code:
python scripts/train_catn.py \
    --mixed-precision \
    --dtype float16  # or bfloat16 on newer GPUs
```

**Advantages**:
- 2-4x faster training
- 50% memory reduction
- Minimal accuracy loss

### Distributed Training (Multi-GPU)

```bash
#!/bin/bash
#SBATCH --gpus=4                      # 4 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Distributed training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train_catn.py \
    --config configs/production.yaml \
    --distributed
```

### Memory Management

```bash
# Check GPU memory during run
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -l 1

# PyTorch memory stats
python -c "
import torch
print('GPU memory allocated:', torch.cuda.memory_allocated() / 1e9, 'GB')
print('GPU memory cached:', torch.cuda.memory_reserved() / 1e9, 'GB')
"

# Automatic memory optimization
export CUDA_LAUNCH_BLOCKING=0  # Async GPU kernel launch
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Deterministic ordering
```

---

## Monitoring & Troubleshooting

### Job Monitoring

```bash
# Check job status
squeue -u $USER

# Get detailed job info
scontrol show job <job_id>

# Real-time job monitoring
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" --sort=-u

# Check job logs
tail -f logs/catn_12345.out
tail -f logs/catn_12345.err

# Monitor GPU during run
watch -n 1 nvidia-smi

# Job accounting
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,Partition
```

### Common Issues & Solutions

#### Issue 1: Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python scripts/train_catn.py \
    --config configs/production.yaml \
    model.catn.batch_size=32  # reduce from 64

# Use gradient accumulation
    model.catn.gradient_accumulation_steps=2

# Enable mixed precision (saves memory)
    --mixed-precision

# Check memory usage
nvidia-smi --format=csv --query-gpu=memory.used,memory.total
```

#### Issue 2: Job Timeout

**Symptom**: `TIMEOUT` in squeue status

**Solution**:
```bash
# Increase walltime
#SBATCH --time=24:00:00  # increase from 08:00:00

# Or submit checkpointing job
python scripts/train_catn.py \
    --config configs/production.yaml \
    --checkpoint results/catn_run/checkpoint.pt \
    --resume-from-checkpoint
```

#### Issue 3: Module Load Errors

**Symptom**: `Module not found` or `conda: command not found`

**Solution**:
```bash
# Check available modules
module avail python

# Load system conda first
module load conda
# Then create environment

# Or use explicit paths
/opt/conda/bin/conda activate $SCRATCH/conda_envs/oa-prs
```

#### Issue 4: Slow Data I/O

**Symptom**: Low GPU utilization during training

**Solution**:
```bash
# Move data to local scratch (faster)
cp -r data/ $TMPDIR/
# Update config to use $TMPDIR

# Increase data loader workers
model.data.num_workers=8  # default 4

# Use faster I/O (NVMe > HDD)
sinfo --Node --format="NodeList,Gres,CPUs" | grep nvme
```

---

## Resource Estimates

### CPU Pipeline

| Step | Time (1 node, 16 CPU) | Memory | Storage |
|------|---|---|---|
| 1. Data prep | 30 min | 16 GB | 1 GB |
| 2. PRS-CS | 45 min | 32 GB | 2 GB |
| 3. LDpred2 | 1.0 h | 32 GB | 2 GB |
| 4. PRS-CSx | 1.5 h | 32 GB | 2 GB |
| 5. BridgePRS | 45 min | 32 GB | 2 GB |
| 6. Functional | 2.0 h | 64 GB | 5 GB |
| 7. TWAS/SMR | 1.5 h | 32 GB | 2 GB |
| 8. Ensemble | 20 min | 32 GB | 1 GB |
| 9. Evaluation | 15 min | 16 GB | 1 GB |
| **Total** | **~9 hours** | **64 GB** | **18 GB** |

### GPU (CATN) Pipeline

| Phase | Time (A100 GPU) | Time (CPU) | Memory | Speedup |
|-------|---|---|---|---|
| Phase 1: EUR pretrain | 2.0 h | 12 h | 64 GB | 6x |
| Phase 2: Domain adapt | 1.5 h | 10 h | 64 GB | 6.7x |
| Phase 3: Fine-tune | 0.25 h | 2 h | 32 GB | 8x |
| **Total CATN** | **3.75 h** | **24 h** | **64 GB** | **6.4x** |

### Full Pipeline (All Methods + Ensemble)

```
Time breakdown:
  - Traditional PRS (parallel): ~2.5 hours
  - Functional annotations: ~2 hours
  - TWAS/SMR: ~1.5 hours
  - CATN (GPU): ~4 hours [or 24 hours on CPU]
  - Ensemble + Evaluation: ~1 hour

Total with GPU: ~10 hours
Total without GPU: ~30 hours
```

### Resource Allocation Recommendations

**Small pilot (toy data)**:
```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gpus=0  # CPU only, no GPU needed
```

**Medium run (1000s samples)**:
```bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
```

**Large production run (10k+ samples)**:
```bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gpus=1  # or 2 for distributed training
```

---

## Data Management

### Data Organization on HPC

```bash
# Recommended structure
/scratch/$USER/oa_prs_transfer/
  ├── data/
  │   ├── raw/           # Original GWAS sumstats (read-only)
  │   ├── processed/     # LD matrices, annotations
  │   ├── splits/        # Train/val/test splits
  │   └── cache/         # Computed embeddings, etc.
  ├── results/           # Pipeline outputs
  │   ├── prs_weights/
  │   ├── predictions/
  │   ├── models/
  │   └── evaluation/
  ├── logs/              # SLURM output logs
  └── configs/           # Configuration files
```

### Data Transfer

```bash
# Download from remote storage
rsync -avz user@data-server:/path/to/data/ ./data/

# Or use GLOBUS (for large files between institutions)
globus transfer \
    --recursive \
    <source-endpoint>:/source/path \
    <dest-endpoint>:/dest/path

# Compress for transfer
tar -czf data_backup.tar.gz data/
scp data_backup.tar.gz user@local-machine:/path/
```

### Disk Quota Management

```bash
# Check quota
quota -s  # or myquota on some clusters

# Find large files
du -sh data/* | sort -rh | head -20

# Compress old results
tar -czf results_backup_2026-03-13.tar.gz results/
rm -rf results/*  # after verification

# Use tiered storage
# Hot (scratch): Current working data
# Warm (project): Active results
# Cold (archive): Historical runs
```

---

## Best Practices

### 1. Job Naming & Organization

```bash
# Use descriptive names
#SBATCH --job-name=oa-prs-catn-phase1-run42

# Organize logs
mkdir -p logs/{prs,catn,ensemble,evaluation}/$(date +%Y%m%d)
#SBATCH --output=logs/catn/$(date +%Y%m%d)/phase1_%j.out
```

### 2. Logging & Reproducibility

```bash
# Save configuration with results
cp configs/production.yaml results/run_42/config_used.yaml

# Log environment
python -c "import sys; print(sys.executable)" > results/run_42/python_path.txt
conda list > results/run_42/environment.txt
nvidia-smi > results/run_42/gpu_info.txt

# Save random seed for reproducibility
echo "seed: 42" >> results/run_42/metadata.yaml
```

### 3. Checkpointing

```bash
# Save intermediate results
python scripts/train_catn.py \
    --config configs/production.yaml \
    --checkpoint results/catn_run/checkpoint_epoch_{}.pt \
    --save-frequency 5  # every 5 epochs
```

### 4. Error Handling

```bash
# Exit on first error
set -e

# Print commands for debugging
set -x

# Full error context
trap "echo 'Error on line $LINENO'" ERR
```

### 5. Resource Monitoring

```bash
# In your job script
echo "Job started at $(date)"
echo "Hostname: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "Memory info:"
free -h
echo "Disk info:"
df -h

# ... run actual computation ...

echo "Job completed at $(date)"
```

### 6. Parallel Processing

```bash
# Use job arrays for parameter sweeps
#SBATCH --array=1-100%20  # 100 jobs, max 20 simultaneous

# Use GNU parallel for embarrassingly parallel tasks
parallel -j $SLURM_CPUS_PER_TASK \
    'python compute_prs.py --variant-chunk {} data/variants' \
    ::: $(seq 1 100)
```

### 7. Container-based Execution (Docker/Singularity)

```bash
# Build Singularity container (on login node)
singularity build oa-prs.sif Singularity

# Run in container
#SBATCH --container-image=oa-prs.sif
singularity exec --nv oa-prs.sif \
    python scripts/run_pipeline.py --config configs/production.yaml
```

---

## Workflow Examples

### Example 1: Complete Production Run

```bash
#!/bin/bash

# Submit master job
echo "Submitting OA-PRS production pipeline..."

MASTER_JOB=$(sbatch --parsable \
    --job-name=oa-prs-master \
    --time=48:00:00 \
    --output=logs/master_%j.out \
    scripts/slurm_master.sh \
    --config configs/production.yaml)

echo "Master job submitted: $MASTER_JOB"

# Check status periodically
for i in {1..100}; do
    sleep 60
    STATUS=$(squeue -j $MASTER_JOB -h -o %T)
    if [ -z "$STATUS" ]; then
        echo "Job completed!"
        break
    fi
    echo "Job $MASTER_JOB status: $STATUS"
done

# Extract results
echo "Pipeline complete. Results in results/full_pipeline/"
cat results/full_pipeline/summary.txt
```

### Example 2: Debugging Session

```bash
#!/bin/bash

# Interactive debugging job
srun --pty --gpus=1 --cpus-per-task=8 --mem=32G \
    --partition=gpu --time=01:00:00 \
    bash

# Now in interactive shell:
module load python/3.10 cuda/11.8
conda activate oa-prs

# Run with verbose output
python scripts/train_catn.py \
    --config configs/toy_example.yaml \
    --verbose \
    --debug
```

---

## Support & Resources

- **SLURM Documentation**: https://slurm.schedmd.com/sbatch.html
- **XSEDE HPC**: https://www.xsede.org/
- **Cluster User Guide**: Check `/opt/cluster/docs/`
- **Local Support**: HPC Support Team email/Slack

---

**Last updated**: 2026-03-13
**License**: Apache 2.0
