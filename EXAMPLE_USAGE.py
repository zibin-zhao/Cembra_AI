"""
Example Usage of CATN Model

This script demonstrates the complete workflow for training and using
the Cross-Ancestry Transfer Network model.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import CATN components
from src.oa_prs.models.transfer import (
    CrossAncestryTransferNet,
    CATNTrainer,
    CATNPredictor,
)


# ============================================================================
# 1. Create a Simple Dataset
# ============================================================================
class SimpleDataset(Dataset):
    """Minimal dataset for demonstration."""

    def __init__(self, n_samples=100, max_snps=500, input_dim=128):
        self.n_samples = n_samples
        self.max_snps = max_snps
        self.input_dim = input_dim

        # Generate random data
        self.snp_features = np.random.randn(n_samples, max_snps, input_dim)
        self.block_indices = np.random.randint(0, 10, (n_samples, max_snps))
        self.labels = np.random.binomial(1, 0.3, n_samples)
        self.ancestry_labels = np.random.binomial(1, 0.5, n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "snp_features": torch.from_numpy(self.snp_features[idx]).float(),
            "block_indices": torch.from_numpy(self.block_indices[idx]).long(),
            "block_masks": torch.ones(self.max_snps, dtype=torch.bool),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            "ancestry_labels": torch.tensor(
                self.ancestry_labels[idx], dtype=torch.float
            ),
        }


# ============================================================================
# 2. Initialize Model and Trainer
# ============================================================================
def main():
    # Configuration
    config = {
        "input_dim": 128,
        "d_model": 256,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "d_ff": 1024,
        "dropout": 0.1,
        "risk_hidden_dims": (512, 256),
        "domain_hidden_dims": (512, 256),
        "use_gradient_checkpointing": False,
        "use_positional_encoding": True,
        "top_k_blocks": 4,
    }

    train_config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs_phase1": 3,  # Short for demo
        "num_epochs_phase2": 3,
        "num_epochs_phase3": 2,
        "early_stopping_patience": 5,
        "use_amp": True,
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\n=== Creating Model ===")
    model = CrossAncestryTransferNet(config)
    model.to(device)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Create datasets and dataloaders
    print("\n=== Creating Data ===")
    train_dataset = SimpleDataset(n_samples=100, max_snps=500, input_dim=128)
    val_dataset = SimpleDataset(n_samples=20, max_snps=500, input_dim=128)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=16)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create trainer
    trainer = CATNTrainer(model, train_config, device)

    # ========================================================================
    # 3. Phase 1: EUR Pre-training
    # ========================================================================
    print("\n=== Phase 1: EUR Pre-training ===")
    history_phase1 = trainer.train_phase1(
        train_loader,
        val_loader,
        checkpoint_path="phase1_best.pt",
    )
    print(f"Final val loss: {history_phase1['val_loss'][-1]:.4f}")

    # ========================================================================
    # 4. Phase 2: Domain Adaptation
    # ========================================================================
    print("\n=== Phase 2: Domain Adaptation ===")
    history_phase2 = trainer.train_phase2(
        train_loader,  # EUR data
        train_loader,  # EAS data (same for demo)
        val_loader,    # EAS validation
        checkpoint_path="phase2_best.pt",
    )
    print(f"Final EAS val loss: {history_phase2['val_loss_eas'][-1]:.4f}")

    # ========================================================================
    # 5. Phase 3: Fine-tune Prediction Head
    # ========================================================================
    print("\n=== Phase 3: Fine-tune Prediction Head ===")
    history_phase3 = trainer.train_phase3(
        train_loader,
        val_loader,
        checkpoint_path="phase3_best.pt",
    )
    print(f"Final val loss: {history_phase3['val_loss'][-1]:.4f}")

    # ========================================================================
    # 6. Inference
    # ========================================================================
    print("\n=== Inference ===")
    
    # Load best model
    predictor = CATNPredictor("phase3_best.pt", device=device)

    # Get test data
    test_dataset = SimpleDataset(n_samples=10, max_snps=500, input_dim=128)
    test_loader = DataLoader(test_dataset, batch_size=10)
    batch = next(iter(test_loader))

    # Predict
    predictions = predictor.predict(
        batch["snp_features"],
        batch["block_indices"],
        batch["block_masks"],
        return_probs=True,
    )
    print(f"Risk predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].squeeze().numpy()}")

    # Extract SNP weights
    weights_df = predictor.extract_weights(
        batch["snp_features"],
        batch["block_indices"],
        batch["block_masks"],
    )
    print(f"\nTop SNP weights:\n{weights_df.head()}")

    # Get representations
    global_rep = predictor.get_representations(
        batch["snp_features"],
        batch["block_indices"],
        batch["block_masks"],
        level="global",
    )
    print(f"\nGlobal representation shape: {global_rep.shape}")

    print("\n=== Complete! ===")


if __name__ == "__main__":
    main()
