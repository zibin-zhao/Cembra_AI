# Cembra AI: OA-PRS Transfer Learning -  Cross-Ancestry Polygenic Risk Score Pipeline

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/your-org/oa-prs-transfer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)


## Overview

**OA-PRS Transfer Learning** is a comprehensive pipeline for developing and evaluating **cross-ancestry Polygenic Risk Scores (PRS) for Osteoarthritis**, enabling equitable genetic risk prediction across diverse ancestry populations. This project bridges traditional statistical PRS methods with modern deep learning and functional genomics, combining:

- **Traditional PRS**: PRS-CS (Bayesian shrinkage), PRS-CSx (multi-population), BridgePRS (ancestry bridging), LDpred2-auto
- **Deep Learning**: CATN (Cross-Ancestry Transfer Network) with domain adversarial training
- **Functional Genomics**: Enformer (deep learning annotations), PolyFun (annotation priors), SuSiE-inf (fine-mapping with infinitesimal effects)
- **TWAS/SMR**: S-PrediXcan, S-MultiXcan, SMR-HEIDI for expression-based predictions

The pipeline evaluates models on discrimination (C-statistic), calibration (expected vs. observed), fairness metrics (performance gaps across populations), and risk stratification across European (EUR) and East Asian (EAS/Hong Kong Chinese) populations.

## Architecture Overview

```
                         ┌─────────────────────────┐
                         │    GWAS Summary Stats   │
                         │ (EUR, EAS, MVP, UKB)    │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
        ┌───────▼──────────┐  ┌──────▼──────────┐  ┌──────▼──────────┐
        │ Traditional PRS  │  │ Cross-Ancestry  │  │ Functional      │
        │ (4 branch)       │  │ Transfer (2)    │  │ Genomics        │
        │                  │  │                 │  │                 │
        │ • PRS-CS         │  │ • PRS-CSx       │  │ • Enformer SAD  │
        │ • LDpred2-auto   │  │ • BridgePRS     │  │ • PolyFun       │
        │ • PRS-CSx (EUR)  │  │   (EUR→EAS)     │  │ • SuSiE-inf     │
        │ • lassosum       │  │                 │  │ • TURF/TLand    │
        └───────┬──────────┘  └────────┬────────┘  └────────┬────────┘
                │                      │                    │
                └──────────────────────┼────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │ TWAS/SMR Expression Scores │
                        │ (S-PrediXcan, SMR-HEIDI)    │
                        └──────────────┬──────────────┘
                                       │
        ┌──────────────────────────────▼──────────────────────────────┐
        │                     CATN Deep Learning                      │
        │         (Cross-Ancestry Transfer Network)                  │
        │  SNP Feature Encoder → LD-Block Transformer                │
        │  → Cross-Block Attention → Risk Head + Domain Discriminator│
        │  Training: EUR Pretrain → Domain Adapt → Fine-tune         │
        └──────────────────────────┬───────────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │        Ensemble Stacking (Ridge/XGBoost)               │
        │    Combine all branch predictions with cross-val       │
        └──────────────────────────┬──────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │              Model Evaluation & Fairness               │
        │  • Discrimination (C-statistic per population)         │
        │  • Calibration (Expected vs. observed)                │
        │  • Fairness metrics (performance gaps)                │
        │  • Risk stratification (decile analysis)              │
        │  • LOSO validation (EUR / EAS / Mixed holdout)         │
        │  • Ablation studies (branch contribution)             │
        └──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- conda or mamba
- 50+ GB free disk space (for LD matrices and model files)
- GPU recommended for CATN training (optional but faster)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/oa-prs-transfer.git
cd oa-prs-transfer

# Create conda environment
conda env create -f environment.yml
conda activate oa-prs-transfer

# Verify installation
python -c "import oaprs_transfer; print(oaprs_transfer.__version__)"
```

### Toy Data & Quick Pipeline Run

```bash
# Download toy datasets (EUR & EAS GWAS sumstats, small LD matrices)
python scripts/download_toy_data.py

# Run minimal pipeline
python scripts/run_pipeline.py \
    --config configs/toy_example.yaml \
    --output results/toy_run

# Expected runtime: ~2 hours on CPU, 30 min on GPU
# Output: PRS predictions, evaluation metrics, figures
```

### Full Pipeline (Production)

```bash
# Download full data (requires 50+ GB)
python scripts/download_full_data.py

# Run complete pipeline
python scripts/run_pipeline.py \
    --config configs/production.yaml \
    --output results/full_run \
    --n-jobs 8

# For HPC (see hpc_guide.md):
sbatch scripts/slurm_master.sh
```

## Pipeline Steps

| Step | Name | Description | Input | Output | Time (GPU) |
|------|------|-------------|-------|--------|-----------|
| 1 | Data Preparation | Harmonize GWAS sumstats, LD pruning, ancestry-specific QC | Raw GWAS, GENCODE | QC'd sumstats, LD matrices | 30 min |
| 2 | PRS-CS | Bayesian shrinkage prior for traditional PRS | Sumstats, LD | PRS weights (EUR-derived) | 45 min |
| 3 | LDpred2-auto | Infinitesimal effects model with automatic shrinkage | Sumstats, LD | PRS weights (auto-tuned) | 1 h |
| 4 | PRS-CSx | Multi-population Bayesian shrinkage (EUR + EAS) | EUR/EAS sumstats, LD | Cross-ancestry weights | 1.5 h |
| 5 | BridgePRS | Ancestry bridging via trans-ancestry meta-analysis | EUR/EAS sumstats, shared SNPs | Bridged weights | 45 min |
| 6 | Functional Annotations | Enformer SAD, PolyFun priors, SuSiE-inf fine-mapping | Sumstats, variant library, GTEx | Functional scores, PIP | 2 h |
| 7 | TWAS/SMR | S-PrediXcan & SMR-HEIDI for expression-based risk | Sumstats, eQTL models, HEIDI | Expression PRS weights | 1.5 h |
| 8 | Genotype Simulation | Simulate genotypes from sumstats + LD for CATN training | Sumstats, LD, test genotypes | Synthetic training genotypes | 30 min |
| 9 | CATN Training | Deep learning transfer learning (3-phase) | Synthetic EUR genotypes, real test set | Trained CATN model | **4-6 h** |
| 10 | Ensemble Stacking | Combine all 8+ branch predictions with meta-learner | Branch predictions (CV folds) | Final ensemble weights | 20 min |
| 11 | Evaluation & Fairness | C-statistic, calibration, fairness gaps, risk strata | Predictions, phenotypes | Report + figures | 15 min |

## CATN Model: Cross-Ancestry Transfer Network

### Motivation

Standard PRS methods struggle with non-EUR populations due to LD differences and allele frequency divergence. CATN addresses this via:

1. **LD-Block aware architecture**: Self-attention within LD blocks captures long-range dependencies while respecting chromosome structure
2. **Cross-ancestry domain adaptation**: Adversarial training on domain discriminator forces EUR-trained features to be ancestry-invariant
3. **Individual fine-tuning**: Rapid adaptation to new populations with minimal EAS samples
4. **Genotype simulation**: Trains on realistic synthetic genotypes derived from GWAS summary statistics

### Architecture

```
SNP Feature Encoder
  └─ Dense(SNP) → BatchNorm → ReLU → Embedding(256)

LD-Block Transformer (4 blocks, 8 heads)
  ├─ Block-level masking (attention only within LD block)
  ├─ Multi-head self-attention
  └─ Feed-forward (512 → 256)

Cross-Block Sparse Attention (2 layers)
  └─ Sparse attention between LD-block representatives

Risk Head
  └─ GlobalAveragePool → Dense(128) → Dropout(0.3) → Output(1)

Domain Discriminator (Gradient Reversal at λ=0.1)
  └─ Dense(64) → ReLU → Dense(1) → Sigmoid
     (Labels: 0=EUR, 1=EAS; Loss forces DANN confusion)

Training Loss = λ_risk · Risk_BCE + λ_domain · Adversarial_BCE
              + λ_l1 · L1(weights) + λ_calib · Calibration_Loss
```

### Training Protocol (3-Phase)

**Phase 1: EUR Pretraining (2 hours)**
- Data: 50K simulated EUR genotypes from EUR GWAS sumstats
- Objective: Maximize risk prediction on EUR individuals
- Hyperparameters: LR=1e-3, batch=128, epochs=100

**Phase 2: Domain Adaptation (1.5 hours)**
- Data: EUR (50K) + EAS synthetic (10K) genotypes
- Objective: Domain-invariant feature learning via gradient reversal
- Domain loss weight λ_domain increases from 0 → 0.1 linearly
- Hyperparameters: LR=5e-4, batch=128, epochs=50

**Phase 3: Individual Fine-tuning (15 min per population)**
- Data: Real genotypes from target population (e.g., HK Chinese)
- Objective: Rapid adaptation with small sample sizes
- Freeze encoder, tune transformer + risk head
- Hyperparameters: LR=1e-4, batch=32, epochs=10

## Project Structure

```
oa-prs-transfer/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── environment.yml                    # Conda dependencies
├── setup.py                           # Package installation
│
├── docs/
│   ├── methods_whitepaper.md          # Detailed technical methods
│   ├── hpc_guide.md                   # HPC/SLURM deployment
│   ├── catn_architecture.md           # CATN model deep-dive
│   ├── data_sources.md                # Data acquisition & preprocessing
│   └── evaluation_metrics.md           # Statistical evaluation framework
│
├── src/oaprs_transfer/
│   ├── __init__.py
│   ├── config.py                      # Hydra config management
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── prs_cs.py                  # PRS-CS implementation
│   │   ├── ldpred2.py                 # LDpred2 wrapper
│   │   ├── prs_csx.py                 # PRS-CSx multi-pop
│   │   ├── bridgeprs.py               # BridgePRS ancestry bridge
│   │   ├── catn.py                    # CATN model & training
│   │   ├── twas_smr.py                # TWAS/SMR integration
│   │   ├── ensemble.py                # Stacking meta-learner
│   │   └── functional_annotations.py  # Enformer, PolyFun, SuSiE-inf
│   │
│   ├── data/
│   │   ├── preprocessing.py           # GWAS QC, harmonization
│   │   ├── ld_matrix.py               # LD matrix utilities
│   │   ├── genotype_sim.py            # Genotype simulation from sumstats
│   │   └── loaders.py                 # Data loaders for PyTorch
│   │
│   ├── evaluation/
│   │   ├── metrics.py                 # C-statistic, calibration
│   │   ├── fairness.py                # Fairness gaps per ancestry
│   │   ├── risk_stratification.py     # Decile analysis
│   │   └── plotting.py                # Visualization utilities
│   │
│   └── models/
│       ├── catn_layers.py             # CATN components (encoder, transformer)
│       ├── dann.py                    # Domain adversarial loss
│       └── lightning_module.py        # PyTorch Lightning training
│
├── configs/
│   ├── toy_example.yaml               # Minimal example config
│   ├── production.yaml                # Full production run
│   ├── ablation.yaml                  # Ablation study (single branch)
│   └── defaults/
│       ├── model_catn.yaml            # CATN hyperparameters
│       ├── data_eur.yaml              # EUR-specific config
│       ├── data_eas.yaml              # EAS-specific config
│       └── evaluation.yaml            # Evaluation settings
│
├── scripts/
│   ├── run_pipeline.py                # Main entry point
│   ├── download_toy_data.py           # Download minimal datasets
│   ├── download_full_data.py          # Download all data
│   ├── slurm_master.sh                # SLURM master submission
│   ├── train_catn.py                  # CATN training script
│   ├── evaluate_model.py              # Evaluation pipeline
│   └── generate_report.py             # HTML report generation
│
├── tests/
│   ├── test_prs_cs.py
│   ├── test_catn.py
│   ├── test_evaluation.py
│   └── conftest.py                    # Pytest fixtures
│
├── results/
│   └── example_run/                   # Example output structure
│       ├── prs_weights/               # All PRS method weights
│       ├── predictions/               # Model predictions (train/val/test)
│       ├── models/                    # Trained model checkpoints
│       ├── evaluation/                # Metrics & plots
│       │   ├── discrimination.csv
│       │   ├── calibration.csv
│       │   ├── fairness_gaps.csv
│       │   ├── c_statistic_by_ancestry.png
│       │   └── roc_curves.png
│       └── reports/
│           └── final_report.html
│
└── data/
    ├── raw/
    │   ├── gwas_eur_knee_oa_2019.txt.gz
    │   ├── gwas_eas_mvp_ukb_2022.txt.gz
    │   └── genotypes_test_set.vcf.gz
    │
    ├── processed/
    │   ├── ld_matrices_1kg_eur/
    │   ├── ld_matrices_1kg_eas/
    │   ├── annotations_enformer_v1.h5
    │   └── eqtl_weights_gtex_v8.db
    │
    └── splits/
        ├── train_eur_50k.txt
        ├── val_eur_10k.txt
        ├── test_eas_hk_chinese_5k.txt
        └── loso_folds.pkl
```

## HPC Deployment

### Conda Environment Setup

```bash
# On HPC cluster
module load python/3.9  # or equivalent for your cluster
module load cuda/11.8   # If GPU available

# Clone and setup
git clone https://github.com/your-org/oa-prs-transfer.git
cd oa-prs-transfer
conda env create -f environment.yml
conda activate oa-prs-transfer
```

### SLURM Job Submission

```bash
# Single GPU job (CATN training)
sbatch scripts/slurm_catn.sh

# Master pipeline (auto-dependencies)
sbatch scripts/slurm_master.sh

# Array job for ablation studies
sbatch --array=1-8 scripts/slurm_ablation_array.sh
```

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=oa-prs-catn
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

module load cuda/11.8
conda activate oa-prs-transfer

python scripts/train_catn.py \
    --config configs/production.yaml \
    --output results/full_run \
    --seed 42
```

### Singularity Container (Optional)

```bash
# Build container
singularity build oa-prs-transfer.sif Singularity

# Run pipeline in container
singularity exec --nv oa-prs-transfer.sif \
    python scripts/run_pipeline.py --config configs/production.yaml
```

## Data Sources

### GWAS Summary Statistics

| Population | Phenotype | Study | Release | N cases | N controls | Source |
|-----------|-----------|-------|---------|---------|-----------|--------|
| EUR | Knee OA | UK Biobank | 2019 | 9,000 | 160,000 | UKB Data Portal |
| EAS + EUR | Knee OA | MVP + UKB | 2022 | ~40,000 | ~400,000 | dbGAP, EGA |

### LD Reference Panels

- **1000 Genomes Phase 3** (1KG3): EUR (N=503), EAS (N=504)
- Downloaded from: https://data.broadinstitute.org/alkesgroup/LDSCORE/
- Format: SNP-major, sparse matrices in `.tar.bz2`

### Functional Annotations

| Resource | Data | Version | Usage |
|----------|------|---------|-------|
| Enformer | Deep learning tissue predictions | V1 | SAD scores for variant effect prediction |
| PolyFun | Annotation priors from LDSC | V1 | SNP-level prior weighting |
| SuSiE-inf | Fine-mapping with infinitesimal effects | — | Identify credible sets, CATN feature selection |
| TURF/TLand | Tissue/cell-type annotations | GTEx v8 | Tissue prioritization for TWAS |
| GTEx | eQTL models | V8 | Expression prediction for S-PrediXcan |

### Test Genotypes

- **Hong Kong Chinese cohort**: N=5,000 (holdout test set)
- **UK Biobank**: N=10,000 EUR (validation), N=5,000 EAS (validation)
- Format: VCF, phased with Eagle2, QC'd (MAC > 5)

## Configuration

All configurations use **Hydra** for flexible YAML-based parameter management. Override configs via CLI:

```bash
python scripts/run_pipeline.py \
    --config configs/production.yaml \
    data.n_samples=25000 \
    model.catn.batch_size=64
```

### Key Configuration Files

**configs/production.yaml**
```yaml
data:
  gwas_eur: data/raw/gwas_eur_knee_oa_2019.txt.gz
  gwas_eas: data/raw/gwas_eas_mvp_ukb_2022.txt.gz
  genotypes_test: data/raw/genotypes_test_set.vcf.gz
  ld_ref_eur: data/processed/ld_matrices_1kg_eur/
  ld_ref_eas: data/processed/ld_matrices_1kg_eas/
  n_samples_eur: 50000
  n_samples_eas: 10000

model:
  catn:
    hidden_dim: 256
    n_transformer_blocks: 4
    dropout: 0.3
    learning_rate: 0.001
  prs_csx:
    phi: [0.1, 0.5, 0.9]

training:
  n_epochs_phase1: 100
  n_epochs_phase2: 50
  n_epochs_phase3: 10
  device: cuda

evaluation:
  fairness_thresholds: [0.05, 0.10]
  calibration_bins: 10
```

## Evaluation Framework

The pipeline includes comprehensive evaluation per **Cell Genomics fairness guidelines**:

### Discrimination
- **Metric**: C-statistic (AUC) by ancestry
- **Target**: C ≥ 0.65 for both EUR & EAS
- **Report**: Performance gap (EUR − EAS)

### Calibration
- **Metric**: Expected vs. observed risk in deciles
- **Target**: Slope ≈ 1.0, intercept ≈ 0 (per population)
- **Report**: Calibration plot with 95% CI

### Fairness
- **Metrics**: Disparate impact ratio, equalized odds gap
- **Target**: < 5% performance difference across populations
- **Report**: Fairness metrics table & heatmaps

### Risk Stratification
- **Metric**: Relative risk in deciles
- **Report**: Forest plot, decile analysis per population

### Ablation Studies
- **Design**: Leave-one-branch-out (LOBO)
- **Report**: Contribution of each method (PRS-CS, CATN, TWAS, etc.)

### Leave-One-Ancestry-Out (LOSO)
- **Folds**: EUR / EAS / Mixed (50/50 EUR+EAS)
- **Report**: Generalization performance across fold combinations

## Contributing

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.



## Support

- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/your-org/oa-prs-transfer/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/your-org/oa-prs-transfer/discussions) for Q&A
- **Documentation**: See [docs/](docs/) for detailed guides
- **Email**: zzhaobz@connect.ust.hk

---

**Last updated**: 2026-03-13
