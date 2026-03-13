# Cembra Bio AI  Pipeline — Architecture Plan

## 1. Project Vision

A pipeline for Osteoarthritis (OA) Polygenic Risk Score (PRS) prediction using transfer learning. The system extracts genetic risk information from large-scale European GWAS, transfers knowledge to East Asian populations through both traditional statistical genetics tools and a **custom deep learning model**, and ultimately scores individual-level disease risk.

**Core Innovation**: Combine established PRS methods (PRS-CSx, BridgePRS) with a novel **Cross-Ancestry Transfer Network (CATN)** — a PyTorch-based deep learning model that learns ancestry-invariant genetic representations through domain adaptation, enabling accurate risk prediction even with very small target-ancestry samples (from ~500K EUR → ~5K EAS).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│                                                                      │
│  EUR GWAS Summary Stats    EAS GWAS Summary Stats    Individual      │
│  (UKB 2019, MVP+UKB 2022) (small sample)            Genotype Data   │
│  SNP | A1 | A2 | BETA | SE | P                      (.bed/.bim/.fam)│
└──────────┬──────────────────────┬──────────────────────┬─────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                              │
│                                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Download │→ │ QC Filter │→ │ Harmonize    │→ │ Standardize    │  │
│  │ Manager  │  │ (MAF,INFO │  │ (Allele flip │  │ (Column names, │  │
│  │          │  │  HWE,dup) │  │  strand,ref) │  │  coords, freq) │  │
│  └──────────┘  └───────────┘  └──────────────┘  └────────────────┘  │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (4 PARALLEL BRANCHES)                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ BRANCH A: Traditional PRS (CPU)                                 │ │
│  │ PRS-CS (EUR baseline) → LDpred2-auto → PRS-CSx (cross-ancestry)│ │
│  │ → BridgePRS (cross-ancestry)                                    │ │
│  │ Output: Per-SNP posterior effect weights (EUR & EAS adapted)    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ BRANCH B: Functional Annotation & Fine-Mapping (GPU + CPU)      │ │
│  │ Enformer/Basenji (zero-shot functional scoring)                 │ │
│  │ → PolyFun (functional priors) → SuSiE-inf (fine-mapping)       │ │
│  │ → TURF/TLand (tissue-specific prioritization)                   │ │
│  │ Output: Per-SNP causal probabilities (PIP), functional scores   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ BRANCH C: Custom Deep Learning — CATN (GPU)                     │ │
│  │ CrossAncestryTransferNet                                        │ │
│  │ Phase 1: Pre-train on EUR (SNP features → risk)                 │ │
│  │ Phase 2: Domain adaptation EUR→EAS (adversarial training)       │ │
│  │ Phase 3: Fine-tune on individual data (if available)            │ │
│  │ Output: Ancestry-adapted risk scores + learned SNP weights      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ BRANCH D: TWAS / SMR (CPU)                                      │ │
│  │ S-PrediXcan / S-MultiXcan (tissue-level gene associations)      │ │
│  │ → PredictAP (EAS-specific expression prediction)                │ │
│  │ → SMR-HEIDI (causal mediation testing)                          │ │
│  │ Output: Gene-level association scores, causal gene list         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE & SCORING LAYER                           │
│                                                                      │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│  │ PRS Refinement    │  │ Stacking Model    │  │ Individual       │  │
│  │ (feed back fine-  │→ │ (Linear/Ridge/    │→ │ Scorer           │  │
│  │  mapped priors    │  │  XGBoost combine  │  │ (genotype ×      │  │
│  │  into PRS weights)│  │  branches A-D)    │  │  final weights)  │  │
│  └──────────────────┘  └───────────────────┘  └──────────────────┘  │
│                                                                      │
│  Optional: PUMAS-ensemble (summary-stat-only tuning/benchmarking)    │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                                   │
│                                                                      │
│  Discrimination: AUC-ROC, AUC-PR, C-index                           │
│  Calibration:    Brier score, Hosmer-Lemeshow, calibration plots     │
│  Risk Strat:     Top 1%/5%/10% quantile risk ratios, DCA curves     │
│  Fairness:       Per-ancestry threshold consistency, calibration gap │
│  Robustness:     Leave-one-study-out (2019 ↔ 2022 cross-validation) │
│  Ablation:       Branch-by-branch contribution analysis              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Custom Deep Learning Model: CATN (Cross-Ancestry Transfer Network)

### 3.1 Problem Statement

Traditional PRS = Σ(β_i × genotype_i) — a simple linear model. This fails across ancestries because:
- LD patterns differ (EUR vs EAS have different correlation structures)
- Allele frequencies differ (MAF varies across populations)
- Effect sizes may differ due to gene-environment interactions
- Causal variants may not be the same tagged SNPs

### 3.2 CATN Architecture

```
INPUT (per SNP, within LD blocks)
┌──────────────────────────────────┐
│ Per-SNP Feature Vector (dim=F):  │
│  • GWAS beta (EUR)               │
│  • GWAS SE (EUR)                 │
│  • -log10(p-value)               │
│  • MAF_EUR, MAF_EAS              │
│  • LD score                      │
│  • Enformer SAD score (top-k)    │
│  • TURF tissue relevance score   │
│  • Conservation (PhyloP/phastCons│
│  • PolyFun causal prior (PIP)    │
│  • Chromosome (positional enc)   │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 1. SNP FEATURE ENCODER           │
│    Linear(F, d_model) + LayerNorm│
│    + Positional Encoding (genomic│
│      position within LD block)   │
│    → [n_snps_in_block, d_model]  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 2. LD-BLOCK TRANSFORMER          │
│    Multi-Head Self-Attention     │
│    within each LD block          │
│    (captures local SNP interact.)│
│    N_layers=4, N_heads=8         │
│    → [n_blocks, d_model] (pooled)│
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 3. CROSS-BLOCK ATTENTION         │
│    Sparse attention across blocks│
│    (captures long-range genetic  │
│     interactions, pathway-level) │
│    → [1, d_model] (global repr.) │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
    ▼                         ▼
┌───────────────┐   ┌─────────────────────┐
│ 4a. RISK HEAD │   │ 4b. DOMAIN          │
│    MLP → P(OA)│   │     DISCRIMINATOR   │
│    (per ancest│   │     (GRL: Gradient   │
│     -ry head) │   │      Reversal Layer) │
│               │   │     MLP → P(ancestry)│
└───────────────┘   └─────────────────────┘
```

### 3.3 Training Strategy

**Phase 1 — EUR Pre-training (large-scale)**
- Data: Simulated individual genotypes from EUR GWAS summary stats + LD matrices
  - Method: Use multivariate normal simulation from LD structure
  - ~100K simulated individuals with realistic genotype-phenotype correlation
- Loss: Binary cross-entropy for OA risk prediction
- Goal: Learn which SNP features and interactions predict OA risk

**Phase 2 — Cross-Ancestry Domain Adaptation (transfer)**
- Data: EUR simulated + small real/simulated EAS genotype-phenotype pairs
- Loss: Risk prediction loss + Domain adversarial loss (λ-weighted)
  - L_total = L_risk(EUR) + α × L_risk(EAS) + λ × L_domain_adversarial
  - λ uses gradient reversal scheduling (ramp up during training)
- Goal: Learn ancestry-invariant genetic representations
- Regularization: L2 + dropout + early stopping on EAS validation set

**Phase 3 — Individual Fine-tuning (when data available)**
- Data: Real individual genotype + phenotype data
- Strategy: Freeze backbone, fine-tune only prediction head
- Few-shot capable: designed to work with as few as 500 individuals

### 3.4 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | PyTorch 2.0+ | Best for research + production, user preference |
| LD block grouping | Pre-computed via PLINK --blocks | Biologically meaningful attention windows |
| Attention type | Standard within-block, sparse cross-block | Memory-efficient for ~1M SNPs |
| Domain adaptation | DANN (gradient reversal) | Proven effective, simpler than optimal transport |
| Genotype simulation | mvnorm from LD + marginal betas | Standard approach when no individual data |
| Training precision | Mixed precision (AMP) | 2x speedup on modern GPUs |
| Checkpointing | Gradient checkpointing | Fits larger models in GPU memory |

---

## 4. Project Structure

```
oa_prs_transfer/
├── README.md                              # Project overview + quick start
├── LICENSE                                # Apache 2.0
├── pyproject.toml                         # Package definition (PEP 621)
├── Makefile                               # Top-level orchestration commands
├── .gitignore
├── .github/
│   └── workflows/
│       ├── lint.yml                       # Ruff + mypy
│       ├── test.yml                       # pytest on toy data
│       └── build_container.yml            # Build Singularity/Docker
│
├── configs/                               # All YAML configs (Hydra)
│   ├── config.yaml                        # Master config (imports all below)
│   ├── data/
│   │   ├── gwas_sources.yaml              # GWAS download URLs + expected schemas
│   │   ├── ld_references.yaml             # 1KG LD panel paths per ancestry
│   │   └── annotations.yaml               # Functional annotation sources
│   ├── models/
│   │   ├── prs_cs.yaml                    # PRS-CS hyperparams
│   │   ├── prs_csx.yaml                   # PRS-CSx multi-pop config
│   │   ├── bridge_prs.yaml                # BridgePRS config
│   │   ├── ldpred2.yaml                   # LDpred2-auto config
│   │   ├── enformer.yaml                  # Enformer batch/GPU settings
│   │   ├── polyfun.yaml                   # PolyFun annotations + params
│   │   ├── susie_inf.yaml                 # SuSiE-inf fine-mapping params
│   │   ├── catn.yaml                      # CATN architecture + training
│   │   ├── twas.yaml                      # S-PrediXcan / SMR config
│   │   └── ensemble.yaml                  # Stacking model config
│   ├── evaluation/
│   │   └── metrics.yaml                   # Which metrics, thresholds, plots
│   └── slurm/
│       ├── cpu_job.yaml                   # CPU partition defaults
│       └── gpu_job.yaml                   # GPU partition defaults
│
├── data/
│   ├── raw/                               # .gitignore'd — actual GWAS files
│   ├── processed/                         # .gitignore'd — QC'd harmonized data
│   ├── external/                          # .gitignore'd — OAOB, GTEx models
│   ├── ld_ref/                            # .gitignore'd — 1KG LD panels
│   └── toy/                               # TRACKED — synthetic test data
│       ├── README.md
│       ├── toy_gwas_eur.tsv               # ~1000 SNPs, simulated betas
│       ├── toy_gwas_eas.tsv               # ~1000 SNPs, smaller N
│       ├── toy_ld_eur.npz                 # Small LD matrix
│       ├── toy_ld_eas.npz
│       ├── toy_genotype.bed/bim/fam       # 200 individuals, 1000 SNPs
│       ├── toy_phenotype.tsv              # Binary OA phenotype
│       ├── toy_annotations.tsv            # Functional annotation scores
│       └── toy_enformer_scores.h5         # Pre-computed SAD scores
│
├── src/
│   └── oa_prs/
│       ├── __init__.py                    # Version, package metadata
│       ├── cli.py                         # Click CLI: `oa-prs run`, `oa-prs score`
│       ├── config.py                      # Hydra/OmegaConf config loader
│       ├── constants.py                   # Column names, file schemas, paths
│       │
│       ├── data/                          # ── Data Processing ──
│       │   ├── __init__.py
│       │   ├── download.py                # Download GWAS sumstats, LD refs, models
│       │   ├── qc.py                      # MAF filter, INFO filter, HWE, duplicates
│       │   ├── harmonize.py               # Allele flip, strand alignment, ref match
│       │   ├── standardize.py             # Uniform column names, coord lift, freq
│       │   ├── ld_utils.py                # LD matrix loading, block computation
│       │   ├── simulate.py                # Genotype simulation from sumstats + LD
│       │   └── datasets.py                # PyTorch Dataset/DataLoader classes
│       │
│       ├── models/                        # ── Model Implementations ──
│       │   ├── __init__.py
│       │   ├── base/                      # Baseline single-ancestry PRS
│       │   │   ├── __init__.py
│       │   │   ├── prs_cs.py              # PRS-CS wrapper (subprocess + parse)
│       │   │   └── ldpred2.py             # LDpred2-auto wrapper (R via rpy2)
│       │   │
│       │   ├── transfer/                  # Cross-ancestry transfer methods
│       │   │   ├── __init__.py
│       │   │   ├── prs_csx.py             # PRS-CSx wrapper
│       │   │   ├── bridge_prs.py          # BridgePRS wrapper
│       │   │   ├── catn_model.py          # CATN PyTorch model definition
│       │   │   ├── catn_layers.py         # Custom layers (GRL, LD-attention)
│       │   │   ├── catn_trainer.py        # Training loop (3 phases)
│       │   │   └── catn_inference.py      # Inference & weight extraction
│       │   │
│       │   ├── functional/                # Functional annotation & fine-mapping
│       │   │   ├── __init__.py
│       │   │   ├── enformer_scorer.py     # Enformer variant effect scoring
│       │   │   ├── polyfun_runner.py      # PolyFun functional prior estimation
│       │   │   ├── susie_inf.py           # SuSiE-inf fine-mapping
│       │   │   └── annotation.py          # TURF/TLand/TITR prioritization
│       │   │
│       │   ├── twas/                      # Transcriptome-wide association
│       │   │   ├── __init__.py
│       │   │   ├── s_predixcan.py         # S-PrediXcan wrapper
│       │   │   ├── smr_heidi.py           # SMR-HEIDI wrapper
│       │   │   └── predict_ap.py          # PredictAP (EAS-specific)
│       │   │
│       │   └── ensemble/                  # Model combination
│       │       ├── __init__.py
│       │       ├── prs_refiner.py         # Feed fine-mapped priors back into PRS
│       │       ├── stacker.py             # Ridge/XGBoost stacking of branches
│       │       └── pumas_ensemble.py      # PUMAS-ensemble (sumstat-only)
│       │
│       ├── scoring/                       # ── Individual Scoring ──
│       │   ├── __init__.py
│       │   ├── prs_scorer.py              # Classical PRS: Σ(β × genotype)
│       │   └── catn_scorer.py             # CATN-based scoring
│       │
│       ├── evaluation/                    # ── Evaluation Framework ──
│       │   ├── __init__.py
│       │   ├── discrimination.py          # AUC-ROC, AUC-PR, C-index
│       │   ├── calibration.py             # Brier, HL test, calibration curves
│       │   ├── risk_stratification.py     # Quantile risk ratios, DCA
│       │   ├── fairness.py                # Cross-ancestry fairness analysis
│       │   ├── leave_one_study.py         # 2019↔2025  cross-validation
│       │   ├── ablation.py                # Per-branch contribution analysis
│       │   └── report_generator.py        # Auto-generate evaluation report
│       │
│       └── utils/                         # ── Utilities ──
│           ├── __init__.py
│           ├── genetics.py                # Allele coding, LD computation, PCA
│           ├── io.py                      # File I/O (PLINK, HDF5, parquet)
│           ├── logging_config.py          # Structured logging (structlog)
│           ├── slurm.py                   # SLURM job script generation
│           └── reproducibility.py         # Seed setting, hash tracking
│
├── scripts/                               # ── HPC Job Scripts ──
│   ├── slurm/
│   │   ├── 00_setup_environment.sh        # Conda/Singularity setup
│   │   ├── 01_download_data.sh            # Download all data sources
│   │   ├── 02_qc_harmonize.sh             # QC + harmonization (CPU)
│   │   ├── 03_enformer_scoring.sh         # Enformer variant scoring (GPU)
│   │   ├── 04_prs_baseline.sh             # PRS-CS + LDpred2 baselines (CPU)
│   │   ├── 05_cross_ancestry_prs.sh       # PRS-CSx + BridgePRS (CPU)
│   │   ├── 06_polyfun_finemapping.sh      # PolyFun + SuSiE-inf (CPU, high-mem)
│   │   ├── 07_prs_refinement.sh           # Feed back priors into PRS (CPU)
│   │   ├── 08_twas_smr.sh                 # S-PrediXcan + SMR-HEIDI (CPU)
│   │   ├── 09_train_catn.sh               # CATN training phases 1-3 (GPU)
│   │   ├── 10_ensemble_stacking.sh        # Combine all branches (CPU)
│   │   ├── 11_evaluation.sh               # Full evaluation suite (CPU)
│   │   └── run_full_pipeline.sh           # Master script: submits all with deps
│   │
│   └── generate_toy_data.py               # Generate synthetic test data
│
├── tests/
│   ├── conftest.py                        # Shared fixtures, toy data paths
│   ├── unit/
│   │   ├── test_qc.py
│   │   ├── test_harmonize.py
│   │   ├── test_simulate.py
│   │   ├── test_catn_model.py
│   │   ├── test_catn_layers.py
│   │   ├── test_metrics.py
│   │   └── test_scoring.py
│   └── integration/
│       ├── test_data_pipeline.py          # End-to-end data processing
│       ├── test_catn_train_toy.py         # CATN training on toy data
│       └── test_full_pipeline_toy.py      # Full pipeline on toy data
│
├── containers/
│   ├── Dockerfile                         # Docker image
│   ├── singularity.def                    # Singularity for HPC
│   └── build.sh                           # Build script
│
├── env/
│   ├── environment_cpu.yml                # Conda env (CPU-only tools)
│   └── environment_gpu.yml                # Conda env (GPU: PyTorch + Enformer)
│
└── docs/
    ├── data_inventory.md                  # All data sources + licenses
    ├── methods_whitepaper.md              # Full methodology description
    ├── catn_architecture.md               # CATN model design document
    ├── fairness_protocol.md               # Cross-ancestry fairness evaluation
    ├── hpc_guide.md                       # HPC setup + job submission guide
    └── api_reference.md                   # Python API docs
```

---

## 5. Data Flow & Dependencies

```
Step 01: Download
    ├── GWAS sumstats (EUR: UKB2019, MVP+UKB2022)
    ├── GWAS sumstats (EAS: if available)
    ├── 1KG LD reference panels (EUR, EAS)
    ├── GTEx v8 expression models
    ├── eQTL summary data (for SMR)
    ├── OAOB database
    └── Functional annotations (TURF/TLand)

Step 02: QC + Harmonize  [depends on: 01]
    ├── Input: raw GWAS files
    └── Output: harmonized_gwas_{eur,eas}.parquet

Step 03: Enformer Scoring  [depends on: 02, GPU]
    ├── Input: variant list from harmonized GWAS
    └── Output: enformer_sad_scores.h5

Step 04: PRS Baselines  [depends on: 02]
    ├── Input: harmonized EUR GWAS + EUR LD
    └── Output: prs_cs_weights.tsv, ldpred2_weights.tsv

Step 05: Cross-Ancestry PRS  [depends on: 02]
    ├── Input: harmonized GWAS (EUR+EAS) + multi-ancestry LD
    └── Output: prs_csx_weights.tsv, bridge_prs_weights.tsv

Step 06: Fine-Mapping  [depends on: 02, 03]
    ├── Input: harmonized GWAS + Enformer scores + annotations
    └── Output: polyfun_priors.tsv, susie_inf_pip.tsv

Step 07: PRS Refinement  [depends on: 04, 05, 06]
    ├── Input: baseline weights + fine-mapped priors
    └── Output: refined_prs_weights.tsv

Step 08: TWAS/SMR  [depends on: 02]
    ├── Input: harmonized GWAS + GTEx models + eQTL data
    └── Output: twas_gene_scores.tsv, smr_heidi_results.tsv

Step 09: CATN Training  [depends on: 02, 03, 06, GPU]
    ├── Input: all features (GWAS + Enformer + priors + LD)
    └── Output: catn_model.pt, catn_weights.tsv

Step 10: Ensemble  [depends on: 07, 08, 09]
    ├── Input: all branch outputs
    └── Output: ensemble_model.pkl, final_weights.tsv

Step 11: Evaluation  [depends on: 10]
    ├── Input: all models + test data
    └── Output: evaluation_report.html
```

---

## 6. SLURM Resource Estimates

| Step | Partition | CPUs | Memory | GPU | Wall Time | Array? |
|------|-----------|------|--------|-----|-----------|--------|
| 01 Download | cpu | 4 | 16G | — | 2h | No |
| 02 QC/Harmonize | cpu | 8 | 32G | — | 1h | No |
| 03 Enformer | gpu | 4 | 64G | 1×A100 | 12-24h | chr1-22 |
| 04 PRS Baselines | cpu | 8 | 64G | — | 4-8h | chr1-22 |
| 05 Cross-Ancestry | cpu | 8 | 64G | — | 6-12h | chr1-22 |
| 06 Fine-Mapping | cpu | 16 | 128G | — | 8-16h | by region |
| 07 PRS Refinement | cpu | 4 | 32G | — | 1h | No |
| 08 TWAS/SMR | cpu | 8 | 32G | — | 2-4h | No |
| 09 CATN Training | gpu | 8 | 64G | 1×A100 | 4-8h | No |
| 10 Ensemble | cpu | 4 | 16G | — | 30min | No |
| 11 Evaluation | cpu | 4 | 16G | — | 30min | No |

---

## 7. Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.10+ | User preference, ecosystem |
| DL Framework | PyTorch 2.0+ | Research-friendly, dynamic graphs |
| Config | Hydra + OmegaConf | Industry-standard, composable configs |
| CLI | Click | Clean interface for HPC scripts |
| Data | pandas, pyarrow, h5py | Fast I/O for genetic data |
| Genetics | plink2, pandas-plink | Standard tools |
| Stats/ML | scikit-learn, scipy, xgboost | Stacking and evaluation |
| Plotting | matplotlib, seaborn | Publication-quality figures |
| Logging | structlog | Structured JSON logs for HPC |
| Testing | pytest, pytest-cov | Comprehensive testing |
| Linting | ruff, mypy | Code quality |
| Container | Docker + Singularity | HPC compatibility |
| CI | GitHub Actions | Automated lint + test |

---

## 8. Key Files to Implement (Priority Order)

### Phase 1: Foundation (must have)
1. `pyproject.toml` + project scaffolding
2. `configs/` — all YAML configurations
3. `src/oa_prs/data/` — download, QC, harmonize, standardize
4. `src/oa_prs/models/base/prs_cs.py` — PRS-CS wrapper
5. `src/oa_prs/models/transfer/prs_csx.py` — PRS-CSx wrapper
6. `src/oa_prs/scoring/prs_scorer.py` — individual scoring
7. `data/toy/` + `scripts/generate_toy_data.py`
8. `tests/` — unit + integration tests

### Phase 2: Novel Components
9. `src/oa_prs/models/transfer/catn_model.py` — CATN architecture
10. `src/oa_prs/models/transfer/catn_layers.py` — custom layers
11. `src/oa_prs/models/transfer/catn_trainer.py` — 3-phase training
12. `src/oa_prs/data/simulate.py` — genotype simulation
13. `src/oa_prs/models/functional/enformer_scorer.py`

### Phase 3: Full Pipeline
14. `src/oa_prs/models/functional/polyfun_runner.py`
15. `src/oa_prs/models/functional/susie_inf.py`
16. `src/oa_prs/models/transfer/bridge_prs.py`
17. `src/oa_prs/models/twas/` — all TWAS/SMR wrappers
18. `src/oa_prs/models/ensemble/` — stacking + refinement
19. `src/oa_prs/evaluation/` — full evaluation suite

### Phase 4: Production
20. `scripts/slurm/` — all HPC job scripts
21. `containers/` — Docker + Singularity
22. `docs/` — whitepaper + guides
23. `.github/workflows/` — CI/CD

---

**License**: Apache 2.0
