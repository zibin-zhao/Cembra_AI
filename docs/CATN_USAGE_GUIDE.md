# CATN (Cross-Ancestry Transfer Network) - Usage Guide

## Overview

The Cross-Ancestry Transfer Network (CATN) is a production-grade PyTorch implementation of a deep learning model for cross-ancestry polygenic risk score (PRS) transfer learning. It uses domain adversarial training to learn ancestry-invariant representations that improve risk prediction generalization across populations.

## Architecture Overview

### Model Components

1. **SNPFeatureEncoder** - Encodes per-SNP features with optional positional encoding
2. **LDBlockTransformer** - Multi-head attention transformer for intra-block SNP interactions
3. **CrossBlockAttention** - Sparse attention across LD blocks for inter-block interactions
4. **RiskPredictionHead** - MLP that predicts osteoarthritis (OA) risk probability
5. **DomainDiscriminator** - MLP with gradient reversal for ancestry prediction (adversarial)

### Processing Flow

```
SNP Features [batch, max_snps, input_dim]
    ↓
SNPFeatureEncoder
    ↓
Encoded Features [batch, max_snps, d_model]
    ↓
LDBlockTransformer (within-block interaction)
    ↓
Transformed SNPs [batch, max_snps, d_model]
    ↓
Pool by LD Block
    ↓
Block Representations [batch, n_blocks, d_model]
    ↓
CrossBlockAttention (between-block interaction)
    ↓
Global Representation [batch, d_model]
    ↓
Risk Head + Domain Discriminator
    ↓
Risk Logits [batch, 1] + Domain Logits [batch, 1]
```

## Installation

```bash
# Clone or navigate to project
cd /sessions/awesome-friendly-johnson/mnt/Cembra/oa_prs_transfer

# Install dependencies
pip install torch torchvision structlog numpy pandas
```

## Quick Start

### 1. Initialize Model

```python
import torch
from src.oa_prs.models.transfer import CrossAncestryTransferNet

# Configuration
config = {
    "input_dim": 128,           # SNP feature dimension
    "d_model": 256,             # Model embedding dimension
    "n_heads": 8,               # Multi-head attention heads
    "n_encoder_layers": 2,      # Transformer layers per block
    "d_ff": 1024,               # Feed-forward hidden dimension
    "dropout": 0.1,
    "risk_hidden_dims": (512, 256),
    "domain_hidden_dims": (512, 256),
    "top_k_blocks": 4,          # Sparse attention top-k
    "use_gradient_checkpointing": True,  # Memory efficiency
    "use_positional_encoding": True,
}

# Create model
model = CrossAncestryTransferNet(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. Phase 1: EUR Pre-training

```python
from src.oa_prs.models.transfer import CATNTrainer

# Training configuration
train_config = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_epochs_phase1": 20,
    "early_stopping_patience": 5,
    "use_amp": True,  # Mixed precision
}

# Initialize trainer
trainer = CATNTrainer(model, train_config, device)

# Train on EUR data
history_phase1 = trainer.train_phase1(
    train_loader_eur,
    val_loader_eur,
    checkpoint_path="best_phase1.pt"
)
```

### 3. Phase 2: Domain Adaptation

```python
# Phase 2 configuration
phase2_config = {
    **train_config,
    "num_epochs_phase2": 30,
    "lambda_domain_init": 0.0,     # Start with domain loss = 0
    "lambda_domain_max": 1.0,      # Ramp up to 1.0
    "alpha_eas": 0.5,              # EAS loss weight
}

trainer = CATNTrainer(model, phase2_config, device)

# Train with multi-ancestry data
history_phase2 = trainer.train_phase2(
    train_loader_eur,
    train_loader_eas,
    val_loader_eas,
    checkpoint_path="best_phase2.pt"
)
```

### 4. Phase 3: Fine-tune Prediction Head

```python
# Phase 3 configuration
phase3_config = {
    **train_config,
    "num_epochs_phase3": 10,
}

trainer = CATNTrainer(model, phase3_config, device)

# Fine-tune on individual/mixed data
history_phase3 = trainer.train_phase3(
    train_loader_individual,
    val_loader_individual,
    checkpoint_path="best_phase3.pt"
)
```

## Data Format

### Input Data Format

The model expects data in a specific format per batch:

```python
batch = {
    "snp_features": torch.Tensor,      # [batch, max_snps, input_dim]
    "block_indices": torch.LongTensor, # [batch, max_snps] - LD block IDs
    "block_masks": torch.BoolTensor,   # [batch, max_snps] - 1=real, 0=padding
    "labels": torch.FloatTensor,       # [batch] - OA status (0/1)
    "ancestry_labels": torch.FloatTensor,  # [batch] - ancestry (0=EUR, 1=EAS)
}
```

### Example DataLoader Creation

```python
from torch.utils.data import Dataset, DataLoader

class PRSDataset(Dataset):
    def __init__(self, snp_features, block_indices, labels,
                 block_masks=None, ancestry_labels=None, max_snps=500):
        self.snp_features = snp_features
        self.block_indices = block_indices
        self.labels = labels
        self.block_masks = block_masks
        self.ancestry_labels = ancestry_labels
        self.max_snps = max_snps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Pad to max_snps
        n_snps = min(len(self.snp_features[idx]), self.max_snps)

        features = torch.zeros(self.max_snps, self.snp_features[idx].shape[-1])
        features[:n_snps] = torch.from_numpy(
            self.snp_features[idx][:self.max_snps]
        ).float()

        blocks = torch.zeros(self.max_snps, dtype=torch.long)
        blocks[:n_snps] = torch.from_numpy(
            self.block_indices[idx][:self.max_snps]
        ).long()

        mask = torch.zeros(self.max_snps, dtype=torch.bool)
        mask[:n_snps] = True

        batch = {
            "snp_features": features,
            "block_indices": blocks,
            "block_masks": mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

        if self.ancestry_labels is not None:
            batch["ancestry_labels"] = torch.tensor(
                self.ancestry_labels[idx], dtype=torch.float
            )

        return batch

# Create dataloaders
train_dataset = PRSDataset(snp_features_train, block_indices_train,
                           labels_train, ancestry_labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Inference

### Basic Prediction

```python
from src.oa_prs.models.transfer import CATNPredictor, load_checkpoint

# Load model
predictor = CATNPredictor(
    model_path="best_phase3.pt",
    device=torch.device("cuda")
)

# Predict on new data
risk_probs = predictor.predict(
    snp_features,      # [batch, max_snps, input_dim]
    block_indices,     # [batch, max_snps]
    block_masks,       # [batch, max_snps]
    return_probs=True
)

print(f"Risk probabilities: {risk_probs}")
```

### Extract SNP Weights

```python
# Get learned SNP importance weights
weights_df = predictor.extract_weights(
    snp_features,
    block_indices,
    block_masks,
    snp_ids=["rs1234567", "rs2345678", ...]  # Optional SNP IDs
)

print(weights_df.head(10))
# Columns: snp_id, weight, block_id
```

### Get Intermediate Representations

```python
# Extract representations at different levels
global_rep = predictor.get_representations(
    snp_features, block_indices, block_masks,
    level="global"  # [batch, d_model]
)

block_rep = predictor.get_representations(
    snp_features, block_indices, block_masks,
    level="block_representations"  # [batch, n_blocks, d_model]
)
```

### Batch Prediction

```python
# Predict on large datasets efficiently
predictions = predictor.predict_batch(
    snp_features_all,  # [n_samples, max_snps, input_dim]
    block_indices_all,
    block_masks_all,
    batch_size=32
)
```

## Advanced Features

### Mixed Precision Training

```python
# Automatically enabled with use_amp=True in config
train_config = {
    "use_amp": True,  # Uses torch.cuda.amp
    ...
}
```

### Gradient Checkpointing

Saves GPU memory during training:

```python
config = {
    "use_gradient_checkpointing": True,  # Enables recomputation
    ...
}
```

### Discriminative Fine-tuning

Use different learning rates for different layers:

```python
# Get parameter groups for layer-wise learning rates
param_groups = model.get_parameter_groups(
    learning_rate=1e-4,
    weight_decay=1e-5
)

optimizer = optim.Adam(param_groups)
```

### Model Freezing

Prevent catastrophic forgetting in Phase 3:

```python
# Freeze backbone, unfreeze risk head
model.freeze_backbone()

# Later: unfreeze all
model.unfreeze_backbone()
```

## Key Architecture Decisions

### 1. Gradient Reversal Layer
- Enables domain adversarial training
- Negates gradients from discriminator to force domain-invariant features
- Lambda parameter can be annealed during training

### 2. Sparse Cross-Block Attention
- Top-k sparse attention for memory efficiency
- Captures long-range genetic interactions without full attention cost
- Each LD block is pooled before cross-block attention

### 3. SNP Positional Encoding
- Adds sinusoidal positional encoding for SNP positions within blocks
- Helps model understand local genomic context
- Can be disabled with `use_positional_encoding=False`

### 4. Domain Adversarial Training
- Phase 2 combines three losses:
  - EUR risk loss
  - EAS risk loss
  - Domain discriminator loss (annealed)
- Encourages learning ancestry-invariant representations

## Configuration Best Practices

### For Small Models (Limited GPU Memory)
```python
config = {
    "d_model": 128,
    "n_heads": 4,
    "n_encoder_layers": 1,
    "d_ff": 512,
    "use_gradient_checkpointing": True,
    "dropout": 0.2,
}
```

### For Large Models (High Capacity)
```python
config = {
    "d_model": 512,
    "n_heads": 16,
    "n_encoder_layers": 4,
    "d_ff": 2048,
    "use_gradient_checkpointing": True,
    "dropout": 0.1,
}
```

### For Production Inference
```python
config = {
    "use_gradient_checkpointing": False,  # Slower but uses same memory as training
    "use_amp": True,  # Faster inference
}
```

## Logging and Monitoring

All components use `structlog` for structured logging:

```python
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

# Logs automatically include:
# - Training loss components
# - Validation metrics
# - Learning rate changes
# - Early stopping events
# - Model state changes
```

## Model Checkpointing

```python
# Checkpoints include:
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "config": model.get_config(),
}

torch.save(checkpoint, "model.pt")

# Load checkpoint
loaded_model, config = load_checkpoint("model.pt", device)
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for 50-100x speedup
2. **Batch Size**: Use largest batch size that fits in GPU memory
3. **Mixed Precision**: Enable `use_amp=True` for 30-40% speedup
4. **Gradient Checkpointing**: Trade speed for memory with `use_gradient_checkpointing=True`
5. **Data Loading**: Use `num_workers` in DataLoader for CPU parallelization
6. **Early Stopping**: Prevents overfitting and saves training time

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Enable gradient checkpointing
- Reduce model dimension (d_model, d_ff)
- Enable mixed precision (use_amp=True)

### Training Divergence
- Reduce learning rate
- Enable gradient clipping (default: 1.0)
- Check input data normalization
- Verify block_indices are valid (0 to n_blocks-1)

### Poor Validation Performance
- Increase training epochs
- Reduce dropout
- Check data imbalance
- Verify train/val data split

### Slow Inference
- Disable gradient checkpointing
- Use larger batch sizes
- Move model to GPU
- Enable mixed precision

## Citation

If you use CATN in your research, please cite:

```bibtex
@software{catn2024,
  title={Cross-Ancestry Transfer Network for Polygenic Risk Score Learning},
  author={OA PRS Team},
  year={2024},
  url={https://github.com/...}
}
```

## License

[Your License Here]
