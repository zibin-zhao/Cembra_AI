# CATN Architecture: Cross-Ancestry Transfer Network for OA-PRS

**Version**: 1.0
**Date**: 2026-03-13
**License**: Apache 2.0

## Table of Contents

1. [Design Motivation](#design-motivation)
2. [Architecture Overview](#architecture-overview)
3. [Layer-by-Layer Component Details](#layer-by-layer-component-details)
4. [Training Protocol](#training-protocol)
5. [Genotype Simulation](#genotype-simulation)
6. [Domain Adversarial Training](#domain-adversarial-training)
7. [Inference & Weight Extraction](#inference--weight-extraction)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Implementation Details](#implementation-details)
10. [Validation & Benchmarking](#validation--benchmarking)

---

## Design Motivation

### The Problem with Standard PRS

**Challenge 1: Ancestry-Dependent Effects**

Standard PRS methods (e.g., PRS-CS, LDpred2) assume:
```
y_i = Σ_j (x_{ij} × β_j) + ε_i

where β_j = ancestry-independent effect size
```

**Reality**: Effect sizes differ across populations:
- European (EUR) GWAS: β_j^EUR ≠ β_j^EAS
- Reasons:
  1. **LD structure differences**: LD blocks vary between populations
  2. **Allele frequency divergence**: FST-dependent effect modification
  3. **Genetic background**: Population-specific rare variants
  4. **Ascertainment**: EUR-enriched GWAS leads to EUR bias

**Consequence**: C-index drops from 0.72 (EUR) to 0.62 (EAS) when applying EUR-trained PRS.

**Challenge 2: Non-linear Risk Accumulation**

Assume: Risk = linear sum of SNP effects

**Reality**: Epistasis and pathways suggest non-linearity:
- Gene-gene interactions (e.g., TP53-MDM2)
- SNP-environment interactions (BMI × genetic risk)
- Tissue-specific effects (cartilage-specific vs. systemic)

**Challenge 3: Limited EAS GWAS**

- EUR GWAS: 400k+ samples (UK Biobank, GIANT)
- EAS GWAS: 50k samples (sparse, mostly East Asian biobanks)
- Cannot train robust models on limited EAS data alone

### The Solution: CATN

**Key Innovation**: Learn **shared, ancestry-invariant representations** that:

1. Capture non-linear risk patterns (via deep learning)
2. Transfer knowledge from EUR (abundant data) to EAS (limited data)
3. Account for population differences via domain adversarial training
4. Respect chromosome structure (LD-block transformer)

**Conceptual Advantage**:
```
Standard PRS:         EUR genotypes → EUR weights → EUR risk
                     EAS genotypes → EUR weights → Poor EAS risk

CATN:                EUR genotypes → Shared features → EUR risk
                                    ↓
                     EAS genotypes → Shared features → EAS risk
                                    (via domain adaptation)
```

---

## Architecture Overview

### High-Level Diagram

```
                    INPUT GENOTYPES (N × M)
                    N samples, M SNPs
                            │
                            ↓
                ┌───────────────────────────┐
                │ SNP FEATURE ENCODER       │
                │ • Input: M (raw dosage)   │
                │ • Dense → BatchNorm → ReLU│
                │ • Output: M × 256         │
                └────────────┬──────────────┘
                             │
                    ┌────────▼────────────────────────┐
                    │ LD-BLOCK TRANSFORMER (4 blocks) │
                    │ • Self-attention within blocks  │
                    │ • 8 heads per block             │
                    │ • Feed-forward networks         │
                    │ • Output: M × 256              │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼──────────────────┐
                    │ CROSS-BLOCK SPARSE ATTENTION  │
                    │ • Block summarization          │
                    │ • Sparse inter-block attention │
                    │ • Output: B × 256 (B ≈ 100)   │
                    └────────┬───────────────────────┘
                             │
            ┌────────────────┼────────────────────┐
            │                │                    │
    ┌───────▼─────┐  ┌──────▼─────┐  ┌──────────▼───────┐
    │ RISK HEAD   │  │DOMAIN HEAD  │  │ UNCERTAINTY HEAD │
    │             │  │ (Grad Rev)  │  │ (Optional)       │
    │Dense(128)   │  │             │  │                  │
    │→ Output(1)  │  │Dense(64)    │  │Quantile Output   │
    │Sigmoid      │  │→ Output(1)  │  │                  │
    │             │  │Sigmoid      │  │                  │
    │ŷ ∈ [0, 1]   │  │d ∈ {0, 1}   │  │Confidence inter. │
    └──────┬──────┘  └──────┬──────┘  └──────────────────┘
           │                │
           └────────┬───────┘
                    │
         ┌──────────▼──────────────┐
         │  LOSS AGGREGATION       │
         │ L = λ_r·L_risk          │
         │   + λ_d·L_domain_adv    │
         │   + λ_l1·L_l1           │
         │   + λ_c·L_calib         │
         └──────────────────────────┘
```

---

## Layer-by-Layer Component Details

### 1. SNP Feature Encoder

**Purpose**: Transform raw genotypes (0, 1, 2 dosage) into dense embeddings for downstream processing.

**Architecture**:
```python
class SNPFeatureEncoder(nn.Module):
    def __init__(self, input_dim=N_SNPS, hidden_dim=1024, embed_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x shape: (batch, n_snps)
        return self.net(x)  # output: (batch, n_snps, 256)
```

**Design Rationale**:
1. **Dense(M → 1024)**: Expand representation space
   - Allows non-linear combinations of SNPs
   - Captures first-order interactions
2. **BatchNorm**: Stabilize training
   - Reduces internal covariate shift
   - Improves gradient flow
3. **ReLU activation**: Introduce non-linearity
   - Standard in deep learning
   - Allows sparse representations
4. **Dropout(0.2)**: Regularization
   - Prevents co-adaptation of neurons
   - Improves generalization
5. **Dense(1024 → 256)**: Dimensionality reduction
   - Projects to embedding space
   - Computational efficiency (future layers)
6. **LayerNorm**: Final normalization
   - Stabilize transformer input

**Input/Output**:
- **Input**: x ∈ ℝ^{N×M} (N samples, M SNPs)
- **Output**: h ∈ ℝ^{N×M×256} (embeddings per SNP)

---

### 2. LD-Block Transformer

**Purpose**: Learn intra-block dependencies respecting genome structure.

**Motivation**: LD blocks represent functional units. Attention only within blocks reduces:
- Quadratic complexity O(M²) → O(B × n_b²)
- Noise from distal, unrelated variants
- Improves biological interpretability

#### 2.1 LD Block Identification

**Algorithm**:
```python
def identify_ld_blocks(
    genotypes,
    ld_threshold=0.5,
    min_block_size=10,
):
    """
    Cluster SNPs into LD blocks based on pairwise LD.

    Args:
        genotypes: (n_samples, n_snps)
        ld_threshold: r² cutoff for grouping

    Returns:
        blocks: list of SNP indices per block
    """
    # Compute LD matrix
    ld_matrix = compute_ld(genotypes)  # (n_snps, n_snps)

    # Greedy clustering
    blocks = []
    visited = set()

    for i in range(len(ld_matrix)):
        if i in visited:
            continue

        # Start new block
        block = [i]
        visited.add(i)
        queue = [i]

        while queue:
            j = queue.pop(0)
            # Find neighbors with r² > threshold
            neighbors = [k for k in range(len(ld_matrix))
                        if ld_matrix[j, k] > ld_threshold and k not in visited]
            block.extend(neighbors)
            visited.update(neighbors)
            queue.extend(neighbors)

        if len(block) >= min_block_size:
            blocks.append(block)

    return blocks
```

**Typical Results**:
- Genome-wide: ~100-500 LD blocks
- Per-chromosome: 5-50 blocks
- Block size: 10-500 SNPs (highly variable)

#### 2.2 Multi-Head Self-Attention (Within Block)

**Architecture**:
```python
class LDBlockTransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, block_len, 256)
        # Self-attention with LD block mask
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,  # Only attend within block
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
```

**LD Block Masking**:
```python
def create_ld_block_mask(block_assignments, embed_dim):
    """
    Create attention mask: True = can attend, False = masked.

    Args:
        block_assignments: (n_snps,) array of block IDs

    Returns:
        mask: (n_snps, n_snps) attention mask
    """
    n_snps = len(block_assignments)
    mask = torch.zeros(n_snps, n_snps, dtype=torch.bool)

    # Same block → can attend
    for i in range(n_snps):
        same_block = block_assignments == block_assignments[i]
        mask[i, same_block] = True

    return mask
```

**Attention Computation**:
```
score_{ij} = (Q_i · K_j) / √(d_k)

where Q_i, K_j = query, key from SNP i, j
      d_k = 256 / 8 = 32 (per-head dimension)

Only compute for pairs in same LD block (via mask)
```

**Multi-Head Mechanism**:
- **8 attention heads** allow parallel representation learning
- Head k learns different aspect of SNP interactions:
  - Head 1: Perfect LD linkage (r² ≈ 1)
  - Head 2: Moderate LD (r² ≈ 0.5)
  - Head 3-8: Other patterns (complex interactions, tag SNPs)

#### 2.3 Feed-Forward Network (Per-Block)

**Purpose**: Non-linear feature transformation after attention.

```
FF(x) = Dense(embed → ff_dim) → ReLU → Dropout → Dense(ff_dim → embed)

Hidden dimension expansion: 256 → 512 (2x) → 256
```

**Rationale**:
- Expands representational capacity
- ReLU introduces non-linearity
- Dropout prevents overfitting

#### 2.4 Residual Connections & Layer Normalization

```
x' = x + Attention(x)      [Residual]
x'' = LayerNorm(x')        [Pre-norm]
x''' = x'' + FF(x'')       [Residual]
output = LayerNorm(x''')
```

**Benefits**:
- Residual connections enable deeper networks (50+ layers)
- Layer normalization stabilizes training
- Improves gradient flow

---

### 3. Cross-Block Sparse Attention

**Purpose**: Capture long-range dependencies between distant LD blocks.

**Motivation**: Some SNPs in different LD blocks may be functionally related (e.g., genes in same pathway). Full attention O(M²) is expensive; sparse attention reduces complexity.

#### 3.1 Block Summarization

```python
def summarize_blocks(block_embeddings, block_assignments):
    """
    Reduce each LD block to a single summary vector.

    Args:
        block_embeddings: (batch, n_snps, 256) from LD-block transformer
        block_assignments: (n_snps,) block IDs

    Returns:
        block_summaries: (batch, n_blocks, 256)
    """
    n_blocks = len(np.unique(block_assignments))
    summaries = []

    for block_id in range(n_blocks):
        mask = block_assignments == block_id
        block_embed = block_embeddings[:, mask, :]  # (batch, block_size, 256)

        # Global average pooling
        summary = block_embed.mean(dim=1)  # (batch, 256)
        summaries.append(summary)

    return torch.stack(summaries, dim=1)  # (batch, n_blocks, 256)
```

#### 3.2 Sparse Attention Patterns

**Option 1: Top-K Attention**
```python
def top_k_attention(query, key, value, k=10):
    """
    Attend to top-k most similar blocks (by cosine similarity).

    Args:
        query, key, value: (batch, n_blocks, 256)
        k: number of blocks to attend to

    Returns:
        output: (batch, n_blocks, 256)
    """
    # Cosine similarity between all block pairs
    scores = torch.mm(query, key.t())  # (batch, n_blocks, n_blocks)
    scores = scores / np.sqrt(256)

    # Mask to keep only top-k
    top_k_values, top_k_indices = torch.topk(scores, k, dim=-1)
    mask = torch.ones_like(scores) * (-1e9)
    mask.scatter_(-1, top_k_indices, 0)

    # Masked softmax
    scores = torch.softmax(scores + mask, dim=-1)

    # Apply attention
    output = torch.bmm(scores, value)  # (batch, n_blocks, 256)
    return output
```

**Option 2: Strided/Local Attention**
```python
def strided_attention(query, key, value, stride=5):
    """
    Attend to nearby blocks (±stride positions on chromosome).
    """
    n_blocks = query.shape[1]
    mask = torch.ones(n_blocks, n_blocks) * (-1e9)

    for i in range(n_blocks):
        # Attend to blocks within stride
        attend_to = list(range(max(0, i - stride),
                               min(n_blocks, i + stride + 1)))
        mask[i, attend_to] = 0

    scores = torch.softmax(scores + mask, dim=-1)
    output = torch.bmm(scores, value)
    return output
```

**Computational Complexity**:
- Full attention: O(n_blocks²) = O(M²) if n_blocks ≈ M
- Top-k: O(n_blocks × k) = O(M) for k << n_blocks
- Strided: O(n_blocks × stride)

#### 3.3 Implementation in Transformer

```python
class CrossBlockSparseAttention(nn.Module):
    def __init__(self, embed_dim=256, n_layers=2, sparsity='top-k', k=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim,
                num_heads=4,  # Fewer heads for computational efficiency
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.sparsity = sparsity
        self.k = k
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

    def forward(self, x):
        # x: (batch, n_blocks, 256)
        for layer, norm in zip(self.layers, self.norms):
            # Create sparse mask
            if self.sparsity == 'top-k':
                mask = create_topk_mask(x, self.k)
            else:
                mask = create_strided_mask(x, stride=5)

            # Apply sparse attention
            attn_out, _ = layer(x, x, x, attn_mask=mask)
            x = norm(x + attn_out)

        return x  # (batch, n_blocks, 256)
```

---

### 4. Risk Head

**Purpose**: Map learned features to risk probability.

**Architecture**:
```python
class RiskHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x):
        # x: (batch, n_blocks, 256)
        return self.net(x)  # output: (batch, 1)
```

**Component Breakdown**:

1. **GlobalAveragePool**: Aggregate block representations
   ```
   (batch, n_blocks, 256) → (batch, 256)
   ```

2. **Dense(256 → 128)**: Compress to task-specific dimension
   - Reduces parameters
   - Projects to risk subspace

3. **ReLU**: Non-linearity
   - Maintains expressiveness
   - Sparse representation

4. **Dropout(0.3)**: Strong regularization
   - High dropout (0.3) because shallow network (2 layers)
   - Prevents overfitting to training samples

5. **Dense(128 → 1)**: Risk prediction
   - Single output

6. **Sigmoid**: Probability mapping
   ```
   σ(z) = 1 / (1 + e^{-z})
   output ∈ [0, 1] = probability of OA
   ```

**Training Loss**:
```
L_risk = BCE(y_true, y_pred)
       = -[y × log(ŷ) + (1 - y) × log(1 - ŷ)]

where ŷ = risk_head(features)
      y ∈ {0, 1} = OA case/control
```

---

### 5. Domain Discriminator (With Gradient Reversal)

**Purpose**: Learn domain-invariant features by encouraging confusion in ancestry classification.

**Architecture**:
```python
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.2, lambda_adv=0.1):
        super().__init__()
        self.grad_reversal = GradientReversalLayer(lambda_adv)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output: P(domain = EAS)
        )

    def forward(self, x, reverse_grad=True):
        if reverse_grad:
            x = self.grad_reversal(x)
        return self.net(x)  # output: (batch, 1)
```

**Domain Labels**:
```
domain_label = 0 if sample is EUR
domain_label = 1 if sample is EAS
```

**Training Loss**:
```
L_domain = BCE(domain_label, domain_pred)
         = -[d × log(ŷ_d) + (1 - d) × log(1 - ŷ_d)]

where ŷ_d = discriminator(features)
      d ∈ {0, 1} = ancestry label
```

#### 5.1 Gradient Reversal Layer

**Key Insight**: During forward pass, GRL is identity. During backward pass, gradients are reversed.

```python
class GradientReversalLayer(nn.Module):
    """
    Layer that reverses gradient during backpropagation.
    """

    def __init__(self, lambda_param=0.1):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        return x  # Identity in forward pass

    def backward(self, grad_output):
        return -self.lambda_param * grad_output  # Reverse & scale in backward
```

**Mathematical Interpretation**:

```
Forward: y = f(x) = x

Backward (with reversal):
∂L/∂x = -λ × ∂L_domain/∂x

Effect on main model (risk head):
The main model learns features that REDUCE domain discriminability
(opposite of what domain loss would normally encourage)
```

**Intuition**:
1. Domain discriminator learns: "EUR = 0, EAS = 1"
2. Main model sees reversed gradient signal: "Learn features that confuse this discriminator"
3. Equilibrium: Features become indistinguishable across domains
4. Result: EUR-trained model generalizes to EAS

---

### 6. Uncertainty Head (Optional)

**Purpose**: Quantify model confidence for risk assessment.

**Architecture**:
```python
class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Lower and upper quantile
        )

    def forward(self, x):
        # x: (batch, n_blocks, 256)
        outputs = self.net(x)  # (batch, 2)

        lower = torch.sigmoid(outputs[:, 0]) * 0.5  # Lower bound
        upper = 0.5 + torch.sigmoid(outputs[:, 1]) * 0.5  # Upper bound

        return lower, upper  # (batch, 1), (batch, 1)
```

**Output**: 90% confidence interval [lower, upper] around risk prediction

**Training Loss**:
```
L_uncertainty = quantile_loss(lower, upper, y_true, quantiles=[0.05, 0.95])
```

---

## Training Protocol

### Overview

**3-Phase Training** with increasing complexity:

| Phase | Duration | Data | Objective | Domain Loss |
|-------|----------|------|-----------|-------------|
| 1 | 2 hours | EUR only (50k) | Learn risk prediction | 0 |
| 2 | 1.5 hours | EUR + EAS (60k) | Learn domain-invariant features | 0 → 0.1 (linear) |
| 3 | 15 min | Target pop (HK Chinese) | Rapid fine-tuning | 0 (disabled) |

### Phase 1: EUR Pretraining

**Goal**: Train on EUR data using standard supervised learning.

**Data**:
```python
# Simulate EUR genotypes from EUR GWAS sumstats
X_eur_synthetic = simulate_genotypes_from_sumstats(
    gwas_sumstats_eur,
    ld_matrix_eur,
    n_samples=50000
)

# Load EUR phenotypes
y_eur = load_phenotypes('ukb_knee_oa_eur.txt')

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(
    X_eur_synthetic, y_eur,
    test_size=0.1,
    stratify=y_eur,
    random_state=42
)
```

**Configuration**:
```yaml
# Phase 1: EUR Pretraining
training:
  phase: 1
  n_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  optimizer: Adam
  loss:
    lambda_risk: 1.0
    lambda_domain: 0.0  # Disabled
    lambda_l1: 1e-5
    lambda_calibration: 0.1
  scheduler:
    type: cosine
    warmup_epochs: 5
  early_stopping:
    patience: 10
    metric: val_loss
```

**Optimization Loop**:
```python
def train_phase_1(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 10 epochs per restart
        T_mult=2,
    )

    for epoch in range(config.n_epochs):
        # Training
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Forward pass
            risk_pred = model.risk_head(model(X_batch))

            # Loss computation
            loss_risk = F.binary_cross_entropy(risk_pred, y_batch)
            loss_l1 = torch.sum(torch.abs(model.parameters()))
            loss_calib = calibration_loss(risk_pred, y_batch)

            loss = (config.lambda_risk * loss_risk +
                   config.lambda_l1 * loss_l1 +
                   config.lambda_calibration * loss_calib)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss = validate(model, val_loader)

        # Learning rate scheduling
        scheduler.step()

        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if should_stop(val_loss):
            break

    return model
```

**Checkpointing**:
```python
# Save best model
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'checkpoints/phase1_best.pt')
```

---

### Phase 2: Domain Adaptation

**Goal**: Learn domain-invariant features using adversarial training.

**Data**:
```python
# EUR data (50k samples)
X_eur = simulate_genotypes_from_sumstats(...)
y_eur = load_phenotypes('ukb_knee_oa_eur.txt')
d_eur = np.zeros(50000)  # Domain label: 0 = EUR

# EAS data (10k samples)
X_eas = simulate_genotypes_from_sumstats(...)
y_eas = load_phenotypes('mvp_ukb_knee_oa_eas.txt')
d_eas = np.ones(10000)  # Domain label: 1 = EAS

# Combine
X_combined = np.vstack([X_eur, X_eas])
y_combined = np.hstack([y_eur, y_eas])
d_combined = np.hstack([d_eur, d_eas])

# Train-val split (stratified by domain)
X_train, X_val, y_train, y_val, d_train, d_val = train_test_split(
    X_combined, y_combined, d_combined,
    test_size=0.1,
    stratify=d_combined,
    random_state=42
)
```

**Configuration**:
```yaml
# Phase 2: Domain Adaptation
training:
  phase: 2
  n_epochs: 50
  batch_size: 128
  learning_rate: 0.0005
  optimizer: Adam
  loss:
    lambda_risk: 1.0
    lambda_domain: [0.0, 0.1]  # Linear schedule 0 → 0.1
    lambda_l1: 1e-5
    lambda_calibration: 0.1
  domain_schedule: linear  # Increase λ_domain linearly
  early_stopping:
    patience: 5
```

**Optimization Loop**:
```python
def train_phase_2(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    lambda_domain_init, lambda_domain_final = config.lambda_domain
    n_epochs = config.n_epochs

    for epoch in range(n_epochs):
        # Linearly increase domain loss weight
        progress = epoch / n_epochs
        lambda_domain = (lambda_domain_init +
                         (lambda_domain_final - lambda_domain_init) * progress)

        # Training with mixed EUR/EAS batches
        train_loss = 0.0
        for X_batch, y_batch, d_batch in train_loader:
            # Forward pass
            features = model(X_batch)
            risk_pred = model.risk_head(features)
            domain_pred = model.domain_discriminator(features)

            # Loss computation
            loss_risk = F.binary_cross_entropy(risk_pred, y_batch)
            loss_domain = F.binary_cross_entropy(domain_pred, d_batch.unsqueeze(-1))
            loss_l1 = torch.sum(torch.abs(model.parameters()))
            loss_calib = calibration_loss(risk_pred, y_batch)

            # Combined loss
            loss = (config.lambda_risk * loss_risk +
                   lambda_domain * loss_domain +  # Increasing weight
                   config.lambda_l1 * loss_l1 +
                   config.lambda_calibration * loss_calib)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss = validate(model, val_loader)

        # Logging
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, " +
                  f"val_loss={val_loss:.4f}, lambda_domain={lambda_domain:.3f}")

    return model
```

**Key Points**:
1. **Alternating optimization**: Risk and domain losses compete
   - Risk loss: Improve prediction on both EUR and EAS
   - Domain loss (reversed): Make features indistinguishable
2. **Linear schedule**: Gradually increase domain pressure
3. **Mixed batches**: Ensure each batch has EUR and EAS samples

---

### Phase 3: Individual Fine-tuning

**Goal**: Rapid adaptation to target population (e.g., HK Chinese).

**Data**:
```python
# Real test set genotypes + phenotypes
X_test = load_genotypes('hk_chinese_cohort.vcf')
y_test = load_phenotypes('hk_chinese_phenotypes.txt')

# Train-val split
X_finetune, X_test_final, y_finetune, y_test_final = train_test_split(
    X_test, y_test,
    test_size=0.2,
    random_state=42
)
```

**Strategy: Freeze Most Layers**

```python
# Freeze encoder and transformer (learned representations)
for param in model.encoder.parameters():
    param.requires_grad = False

for param in model.ld_block_transformer.parameters():
    param.requires_grad = False

for param in model.cross_block_attention.parameters():
    param.requires_grad = False

# Only tune output layers
for param in model.risk_head.parameters():
    param.requires_grad = True

# Discriminator can be frozen or left trainable
# (typically frozen to prevent domain shift)
for param in model.domain_discriminator.parameters():
    param.requires_grad = False
```

**Configuration**:
```yaml
# Phase 3: Fine-tuning
training:
  phase: 3
  n_epochs: 10
  batch_size: 32
  learning_rate: 0.0001
  optimizer: SGD  # More stable than Adam with small data
  momentum: 0.9
  loss:
    lambda_risk: 1.0
    lambda_domain: 0.0  # Disabled
    lambda_l1: 1e-5
    lambda_calibration: 0.1
  freeze_layers:
    - encoder
    - ld_block_transformer
    - cross_block_attention
```

**Optimization Loop**:
```python
def train_phase_3(model, train_loader, val_loader, config):
    # Only optimize unfrozen parameters
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        momentum=config.momentum
    )

    for epoch in range(config.n_epochs):
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Forward pass
            risk_pred = model.risk_head(model(X_batch))

            # Loss (domain loss disabled)
            loss_risk = F.binary_cross_entropy(risk_pred, y_batch)
            loss_calib = calibration_loss(risk_pred, y_batch)

            loss = (config.lambda_risk * loss_risk +
                   config.lambda_calibration * loss_calib)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Early stopping with small patience
        val_loss = validate(model, val_loader)
        if should_stop(val_loss, patience=3):
            break

    return model
```

---

## Genotype Simulation

### Why Simulate Genotypes?

**Problem**: Cannot use real genotypes for training (privacy, limited EAS samples).

**Solution**: Simulate realistic genotypes from GWAS summary statistics + LD matrix.

### Algorithm: Conditional Sampling from LD Structure

**Step 1: Extract LD Structure**

```python
def extract_ld_structure(genotypes, window_size=1000):
    """
    Compute LD matrix from reference cohort genotypes.

    Args:
        genotypes: (n_samples, n_snps) real genotypes
        window_size: compute LD only within window (efficiency)

    Returns:
        ld_matrix: (n_snps, n_snps) correlation matrix
    """
    # Standardize
    genotypes_std = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)

    # Compute correlation
    ld = np.corrcoef(genotypes_std.T)

    # Zero out long-range correlations (likely noise)
    ld[np.abs(np.arange(n_snps)[:, None] - np.arange(n_snps)) > window_size] = 0

    return ld
```

**Step 2: Estimate Effect Sizes (Posterior Inference)**

From summary statistics (β̂, SE), estimate true effects using Bayesian shrinkage:

```python
def estimate_effect_sizes(beta_hat, SE, ld_matrix, h2=0.05):
    """
    Estimate posterior mean effect sizes.

    Uses Bayesian shrinkage with horseshoe prior (approx).
    """
    # Posterior variance (approximate)
    posterior_var = 1 / (1/SE**2 + 1/(h2/M))

    # Posterior mean
    posterior_mean = posterior_var * (beta_hat / SE**2)

    return posterior_mean, posterior_var
```

**Step 3: Sample Effect Sizes**

```python
def sample_effect_sizes(posterior_mean, posterior_var, n_samples=1):
    """
    Sample effect sizes from posterior distribution.
    """
    beta_sampled = np.random.normal(
        posterior_mean,
        np.sqrt(posterior_var),
        size=(n_samples, len(posterior_mean))
    )
    return beta_sampled
```

**Step 4: Simulate Genotypes Preserving LD**

**Method 1: Multivariate Normal Sampling**

```python
def simulate_genotypes_mvn(
    allele_frequencies,
    ld_matrix,
    n_samples=10000,
    centered=True
):
    """
    Simulate genotypes from MVN with LD structure.

    Args:
        allele_frequencies: (n_snps,) allele freq in population
        ld_matrix: (n_snps, n_snps) LD (correlation) matrix
        n_samples: # individuals to simulate

    Returns:
        genotypes: (n_samples, n_snps) simulated dosages [0, 2]
    """
    n_snps = len(allele_frequencies)

    # Ensure LD matrix is positive definite
    eigvals, eigvecs = np.linalg.eigh(ld_matrix)
    eigvals[eigvals < 0] = 0  # Zero out negative eigenvalues
    ld_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Sample from MVN(0, LD_PD)
    Z = np.random.multivariate_normal(
        mean=np.zeros(n_snps),
        cov=ld_pd,
        size=n_samples
    )

    # Convert to allele counts
    # For each SNP, transform standard normal to binomial-like
    genotypes = np.zeros((n_samples, n_snps))
    for j in range(n_snps):
        p = allele_frequencies[j]
        # Quantile transform to approximate binomial
        genotypes[:, j] = 2 * scipy.stats.norm.cdf(Z[:, j])

    return genotypes
```

**Method 2: Conditional Gaussian Approximation (More Accurate)**

```python
def simulate_genotypes_conditional(
    allele_frequencies,
    ld_matrix,
    n_samples=10000,
    reference_genotypes=None,
):
    """
    Simulate genotypes conditional on reference LD structure.

    Args:
        allele_frequencies: (n_snps,) per-SNP allele freq
        ld_matrix: (n_snps, n_snps) from reference cohort
        reference_genotypes: (n_ref, n_snps) optional, for better LD est.

    Returns:
        genotypes: (n_samples, n_snps) simulated dosages
    """
    n_snps = len(allele_frequencies)

    # If reference genotypes provided, compute empirical LD
    if reference_genotypes is not None:
        ld_matrix = empirical_ld(reference_genotypes)

    # Ensure positive definiteness
    ld_matrix = make_psd(ld_matrix)

    # Simulate via iterative block conditioning
    genotypes = np.zeros((n_samples, n_snps))

    # Process in LD blocks
    blocks = identify_blocks(ld_matrix, threshold=0.1)

    for block in blocks:
        # Conditional distribution within block
        block_snps = block

        # Covariance within block
        block_cov = ld_matrix[np.ix_(block_snps, block_snps)]

        # Marginal variances (binomial with allele freq)
        p = allele_frequencies[block_snps]
        marginal_var = 2 * p * (1 - p)

        # Correlation → covariance
        block_cov_scaled = block_cov * np.outer(marginal_var, marginal_var)

        # Sample from multivariate normal
        mean = 2 * p
        Z = np.random.multivariate_normal(
            mean=mean,
            cov=block_cov_scaled,
            size=n_samples
        )

        # Clip to valid range [0, 2]
        genotypes[:, block_snps] = np.clip(Z, 0, 2)

    return genotypes
```

**Step 5: Generate Phenotypes**

```python
def simulate_phenotypes(
    genotypes,
    effect_sizes,
    phenotype_type='binary',
    heritability=0.05,
):
    """
    Simulate phenotypes from genotypes and effects.

    Args:
        genotypes: (n_samples, n_snps) simulated dosages
        effect_sizes: (n_snps,) effect sizes per SNP
        phenotype_type: 'binary' (OA case/control) or 'quantitative'
        heritability: total heritability

    Returns:
        phenotypes: (n_samples,) binary or continuous
    """
    # Linear predictor
    linear_predictor = genotypes @ effect_sizes

    if phenotype_type == 'binary':
        # Prevalence-matched binary phenotype
        risk = scipy.special.expit(linear_predictor)  # Logistic
        phenotypes = np.random.binomial(1, risk)

    else:  # quantitative
        # Residual variance
        sigma_residual = np.sqrt(1 - heritability)
        noise = np.random.normal(0, sigma_residual, n_samples)
        phenotypes = linear_predictor + noise

    return phenotypes
```

---

## Domain Adversarial Training

### The Adversarial Game

**Setup**: Two competing objectives
1. **Main model** (encoder → transformer → risk head):
   - Maximize: Risk prediction accuracy on both EUR and EAS
   - Constraint: Don't let domain discriminator distinguish ancestries

2. **Domain discriminator**:
   - Maximize: Correct ancestry classification (EUR vs EAS)
   - Adversary: Main model tries to confuse it

### Mathematical Formulation

**Total Loss**:
```
L(θ_main, θ_disc) = L_risk(θ_main) + λ_domain × L_adversarial(θ_main, θ_disc)

where:
  L_risk = BCE(y_true, y_pred)
  L_adversarial = -BCE(d_label, d_pred)  [Negative because main model wants to increase it]
```

**Gradient Flow**:
```
For main model parameters θ_main:
  ∂L/∂θ_main = ∂L_risk/∂θ_main - λ_domain × ∂L_domain/∂θ_main
               ↑                   ↑
               Improve risk        Confuse discriminator
               prediction          (reversed gradient)

For discriminator parameters θ_disc:
  ∂L/∂θ_disc = ∂L_domain/∂θ_disc
               ↑
               Improve ancestry
               classification
```

### Training Dynamics

```
Epoch 1-10 (λ_domain ≈ 0):
  - Model learns EUR risk patterns
  - Discriminator learns EUR-specific features
  - Model features: EUR-optimized, EAS-suboptimal

Epoch 11-30 (λ_domain ≈ 0.05):
  - Risk loss decreases slowly (already near optimum)
  - Adversarial loss increases (model confuses discriminator)
  - Model features: Gradually becoming domain-invariant

Epoch 31-50 (λ_domain ≈ 0.1):
  - Risk loss plateaus (can't improve without losing domain-invariance)
  - Adversarial loss continues rising
  - Model features: Domain-invariant, but may sacrifice some EUR accuracy
  - Final compromise: Balanced EUR/EAS performance
```

### Practical Tips

1. **Lambda Schedule**: Linear increase (0 → 0.1) over epochs
   - Smooth convergence
   - Avoid oscillations

2. **Batch Composition**: Mix EUR and EAS equally
   - Each batch: 50% EUR, 50% EAS
   - Ensures discriminator sees both domains

3. **Discriminator Architecture**: Keep simple
   - 2-3 layers sufficient
   - Too complex → overfitting

4. **Monitoring**: Track both losses
   ```python
   L_risk vs. epoch  → Should decrease initially, plateau
   L_domain vs. epoch  → Should oscillate around equilibrium
   ```

---

## Inference & Weight Extraction

### Goal: Deploy Trained CATN as Standard PRS

**Challenge**: CATN is a black-box neural network. How to extract interpretable weights?

### Method 1: Input Jacobian

**Idea**: Compute how much each SNP affects risk prediction.

```python
def compute_snp_importance_jacobian(model, X_batch):
    """
    Compute importance via input-output Jacobian.

    ∂ŷ/∂x_j = how much does SNP j change prediction?
    """
    X_batch.requires_grad_(True)

    # Forward pass
    y_pred = model(X_batch)

    # Backward pass (compute jacobian)
    jacobian = torch.autograd.grad(
        outputs=y_pred.sum(),
        inputs=X_batch,
        create_graph=False
    )[0]  # (batch, n_snps)

    # SNP importance: average jacobian across samples
    snp_importance = jacobian.mean(dim=0).cpu().numpy()

    return snp_importance
```

**Interpretation**:
- Positive values: SNP increases risk
- Negative values: SNP decreases risk
- Magnitude: Importance

### Method 2: Attention Weights

**Idea**: Extract learned attention patterns from transformer.

```python
def extract_attention_weights(model):
    """
    Extract attention patterns from LD-block transformer.
    """
    # Access attention weights from transformer layers
    attention_weights = []

    for layer in model.ld_block_transformer.layers:
        attn_module = layer.self_attn
        # Attention weights: (batch, n_heads, seq_len, seq_len)
        weights = attn_module.in_proj_weight
        attention_weights.append(weights)

    return attention_weights
```

**Interpretation**:
- High attention between SNPs → likely functional interaction
- SNPs with high self-attention (diagonal) → independently important

### Method 3: SHAP (SHapley Additive exPlanations)

**Idea**: Game-theoretic approach to feature importance.

```python
import shap

def compute_shap_importances(model, X_background, X_explain):
    """
    Compute SHAP importance scores.

    Args:
        model: Trained CATN
        X_background: Background data (e.g., 100 random samples)
        X_explain: Data to explain (e.g., test samples)
    """
    # Create explainer
    explainer = shap.DeepExplainer(model, X_background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_explain)

    # SNP importance: average |SHAP| across samples
    snp_importance = np.abs(shap_values).mean(axis=0)

    return snp_importance
```

### Method 4: Direct Weight Extraction

**Simplest**: Extract and use as-is.

```python
def extract_prs_weights(model):
    """
    Extract effective PRS weights from trained CATN.

    Returns a simpler model that approximates CATN.
    """
    # Option 1: Use encoder weights directly
    encoder_weights = model.encoder[0].weight  # Shape: (1024, n_snps)
    encoder_bias = model.encoder[0].bias

    # Reduce to per-SNP effect
    snp_weights = encoder_weights.mean(dim=0).detach().cpu().numpy()

    return snp_weights
```

### Full Inference Pipeline

```python
def compute_prs_catn(genotypes_test, model, method='jacobian'):
    """
    Compute final PRS from trained CATN.

    Args:
        genotypes_test: (n_samples, n_snps) test genotypes
        model: Trained CATN model
        method: 'jacobian', 'attention', or 'shap'

    Returns:
        prs: (n_samples,) risk scores
    """
    # Normalize genotypes
    genotypes_std = (genotypes_test - genotypes_test.mean(axis=0)) / \
                    genotypes_test.std(axis=0)

    # Forward pass through CATN
    with torch.no_grad():
        genotypes_tensor = torch.from_numpy(genotypes_std).float()

        # Extract features
        features = model.encoder(genotypes_tensor)
        block_features = model.ld_block_transformer(features)
        cross_block_features = model.cross_block_attention(block_features)

        # Risk prediction
        risk_scores = model.risk_head(cross_block_features)

    return risk_scores.numpy().flatten()
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Default | Range | Sensitivity |
|-----------|---------|-------|-------------|
| **Encoder hidden dim** | 1024 | 512-2048 | Low |
| **Embedding dim** | 256 | 128-512 | Medium |
| **N transformer blocks** | 4 | 2-8 | Medium |
| **N attention heads** | 8 | 4-16 | Low |
| **FF hidden dim** | 512 | 256-1024 | Low |
| **Dropout rate** | 0.2 | 0.1-0.5 | High |
| **Learning rate** | 0.001 | 1e-4-1e-2 | High |
| **Batch size** | 128 | 32-256 | Medium |
| **λ_domain** | 0.1 | 0-1 | High |
| **λ_l1** | 1e-5 | 1e-6-1e-3 | Low |

### Tuning Strategy

**Step 1: Grid Search** (coarse)
```python
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'dropout': [0.1, 0.2, 0.3],
    'lambda_domain': [0.05, 0.1, 0.2],
}

best_params = {}
best_val_loss = float('inf')

for lr in param_grid['learning_rate']:
    for dropout in param_grid['dropout']:
        for lambda_domain in param_grid['lambda_domain']:
            model = CATN(dropout=dropout)
            val_loss = train_and_validate(model, lr=lr, lambda_domain=lambda_domain)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'lr': lr, 'dropout': dropout, 'lambda_domain': lambda_domain}
```

**Step 2: Bayesian Optimization** (fine)
```python
from skopt import gp_minimize

def objective(params):
    lr, dropout, lambda_domain = params
    model = CATN(dropout=dropout)
    val_loss = train_and_validate(model, lr=lr, lambda_domain=lambda_domain)
    return val_loss

space = [
    (1e-4, 1e-2),  # lr
    (0.1, 0.5),    # dropout
    (0.01, 0.5),   # lambda_domain
]

result = gp_minimize(objective, space, n_calls=50)
```

---

## Implementation Details

### PyTorch Lightning Training (Recommended)

```python
import pytorch_lightning as pl

class CATNModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.catn = CATN(config)
        self.domain_loss_weight = 0.0

    def forward(self, x):
        return self.catn(x)

    def training_step(self, batch, batch_idx):
        X, y, d = batch
        y_pred = self(X)
        d_pred = self.catn.domain_discriminator(self.catn.encode(X))

        loss_risk = F.binary_cross_entropy(y_pred, y)
        loss_domain = F.binary_cross_entropy(d_pred, d)
        loss_l1 = torch.sum(torch.abs(self.catn.encoder.weight))

        loss = (self.config.lambda_risk * loss_risk +
                self.domain_loss_weight * loss_domain +
                self.config.lambda_l1 * loss_l1)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, d = batch
        y_pred = self(X)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def on_epoch_end(self):
        # Increase domain loss weight
        progress = self.current_epoch / self.trainer.max_epochs
        self.domain_loss_weight = self.config.lambda_domain_final * progress
```

### Training Wrapper

```python
def train_catn_full(
    X_eur, y_eur,
    X_eas, y_eas,
    X_test, y_test,
    config,
):
    """
    Train CATN through all 3 phases.
    """
    # Phase 1: EUR Pretraining
    print("Phase 1: EUR Pretraining...")
    model = CATN(config)
    model = train_phase_1(model, X_eur, y_eur, config)
    torch.save(model.state_dict(), 'checkpoints/phase1.pt')

    # Phase 2: Domain Adaptation
    print("Phase 2: Domain Adaptation...")
    model = train_phase_2(model, X_eur, y_eur, X_eas, y_eas, config)
    torch.save(model.state_dict(), 'checkpoints/phase2.pt')

    # Phase 3: Fine-tuning
    print("Phase 3: Fine-tuning...")
    model = train_phase_3(model, X_test, y_test, config)
    torch.save(model.state_dict(), 'checkpoints/phase3.pt')

    return model
```

---

## Validation & Benchmarking

### Cross-Validation Strategy

```python
def cross_validate_catn(X_all, y_all, d_all, n_folds=5):
    """
    Leave-One-Ancestry-Out cross-validation.
    """
    folds = [
        ('EUR', d_all == 0),
        ('EAS', d_all == 1),
        ('Mixed', np.ones_like(d_all, dtype=bool)),
    ]

    results = {}

    for fold_name, fold_mask in folds:
        X_train, X_test = X_all[~fold_mask], X_all[fold_mask]
        y_train, y_test = y_all[~fold_mask], y_all[fold_mask]

        model = CATN(config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        c_stat = roc_auc_score(y_test, y_pred)

        results[fold_name] = c_stat

    return results
```

### Comparison with Baselines

```python
def benchmark_methods(X, y):
    """
    Compare CATN against standard PRS methods.
    """
    results = {}

    # 1. PRS-CS
    prs_cs_weights = run_prs_cs(...)
    results['PRS-CS'] = evaluate(X @ prs_cs_weights, y)

    # 2. LDpred2
    ldpred2_weights = run_ldpred2(...)
    results['LDpred2'] = evaluate(X @ ldpred2_weights, y)

    # 3. CATN
    catn_model = CATN(config).fit(X, y)
    results['CATN'] = evaluate(catn_model.predict(X), y)

    return results
```

---

**Document version**: 1.0
**Last updated**: 2026-03-13
**License**: Apache 2.0
**Questions?** Open an issue on GitHub or email: [contact]
