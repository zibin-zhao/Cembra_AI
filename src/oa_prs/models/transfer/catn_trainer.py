"""
CATN Model Training Loop.

This module implements three-phase training for the CATN model:
- Phase 1: EUR pre-training with risk prediction task
- Phase 2: Domain adaptation with multi-ancestry data
- Phase 3: Fine-tune prediction head on individual data
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import structlog
import numpy as np

from .catn_model import CrossAncestryTransferNet

log = structlog.get_logger(__name__)


class CATNTrainer:
    """
    Trainer for Cross-Ancestry Transfer Network.

    Handles three training phases with domain adversarial training,
    mixed precision, gradient clipping, and early stopping.

    Args:
        model: CATN model instance.
        config: Training configuration dictionary with keys:
            - learning_rate: Learning rate. Default: 1e-4
            - weight_decay: L2 regularization. Default: 1e-5
            - batch_size: Batch size. Default: 32
            - num_epochs_phase1: Phase 1 epochs. Default: 20
            - num_epochs_phase2: Phase 2 epochs. Default: 30
            - num_epochs_phase3: Phase 3 epochs. Default: 10
            - early_stopping_patience: Early stopping patience. Default: 5
            - gradient_clip_val: Gradient clipping value. Default: 1.0
            - warmup_epochs: Warmup epochs. Default: 2
            - use_amp: Enable mixed precision. Default: True
            - use_gradient_checkpointing: Enable checkpointing. Default: False
            - lambda_domain_init: Initial domain adversarial weight. Default: 0.0
            - lambda_domain_max: Max domain adversarial weight. Default: 1.0
            - alpha_eas: EAS risk loss weight. Default: 0.5
        device: Device to train on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: CrossAncestryTransferNet,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """
        Initialize CATN trainer.

        Args:
            model: CATN model instance.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Training hyperparameters
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.num_epochs_phase1 = config.get("num_epochs_phase1", 20)
        self.num_epochs_phase2 = config.get("num_epochs_phase2", 30)
        self.num_epochs_phase3 = config.get("num_epochs_phase3", 10)
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.gradient_clip_val = config.get("gradient_clip_val", 1.0)
        self.warmup_epochs = config.get("warmup_epochs", 2)
        self.use_amp = config.get("use_amp", True)
        self.lambda_domain_init = config.get("lambda_domain_init", 0.0)
        self.lambda_domain_max = config.get("lambda_domain_max", 1.0)
        self.alpha_eas = config.get("alpha_eas", 0.5)

        # Loss functions
        self.risk_loss = nn.BCEWithLogitsLoss()
        self.domain_loss = nn.BCEWithLogitsLoss()

        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

        log.info(
            "Initialized CATN trainer",
            learning_rate=self.learning_rate,
            device=str(device),
            use_amp=self.use_amp,
        )

    def train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 1: EUR Pre-training.

        Trains the risk prediction head on EUR ancestry data using
        standard binary cross-entropy loss.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_path: Path to save best checkpoint. Default: None

        Returns:
            Training history dictionary with 'train_loss' and 'val_loss'.
        """
        log.info("Starting Phase 1: EUR Pre-training")

        # Unfreeze all parameters
        self.model.unfreeze_backbone()

        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create scheduler with warmup
        scheduler = self._create_scheduler(optimizer, self.num_epochs_phase1)

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.num_epochs_phase1):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch_phase1(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader, phase="phase1")
            history["val_loss"].append(val_loss)

            # Step scheduler
            scheduler.step()

            # Log metrics
            log.info(
                "Phase 1 epoch completed",
                epoch=epoch + 1,
                total_epochs=self.num_epochs_phase1,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=scheduler.get_last_lr()[0],
            )

            # Early stopping and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, epoch)
                    log.info("Saved best checkpoint", epoch=epoch, loss=val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    log.info(
                        "Early stopping triggered",
                        epoch=epoch,
                        patience=self.early_stopping_patience,
                    )
                    break

        log.info("Phase 1 training completed", best_loss=self.best_val_loss)

        return history

    def train_phase2(
        self,
        train_loader_eur: DataLoader,
        train_loader_eas: DataLoader,
        val_loader_eas: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Domain Adaptation.

        Trains with multi-ancestry data using domain adversarial training.
        Balances three losses:
        - Risk prediction on EUR data
        - Risk prediction on EAS data
        - Domain prediction (ancestry adversarial)

        Args:
            train_loader_eur: EUR training data loader.
            train_loader_eas: EAS training data loader.
            val_loader_eas: EAS validation data loader.
            checkpoint_path: Path to save best checkpoint. Default: None

        Returns:
            Training history with multiple loss components.
        """
        log.info("Starting Phase 2: Domain Adaptation")

        # Unfreeze all parameters
        self.model.unfreeze_backbone()

        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create scheduler
        scheduler = self._create_scheduler(optimizer, self.num_epochs_phase2)

        # Training loop
        history = {
            "train_loss_eur": [],
            "train_loss_eas": [],
            "train_loss_domain": [],
            "train_loss_total": [],
            "val_loss_eas": [],
        }

        for epoch in range(self.num_epochs_phase2):
            self.current_epoch = epoch

            # Update domain loss weight (ramp up over training)
            lambda_domain = self._compute_domain_weight(
                epoch, self.num_epochs_phase2
            )
            self.model.domain_discriminator.set_lambda(lambda_domain)

            # Train epoch
            train_metrics = self._train_epoch_phase2(
                train_loader_eur, train_loader_eas, optimizer, lambda_domain
            )

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)

            # Validate on EAS data
            val_loss = self._validate(val_loader_eas, phase="phase2")
            history["val_loss_eas"].append(val_loss)

            # Step scheduler
            scheduler.step()

            # Log metrics
            log.info(
                "Phase 2 epoch completed",
                epoch=epoch + 1,
                total_epochs=self.num_epochs_phase2,
                loss_eur=train_metrics.get("loss_eur", 0),
                loss_eas=train_metrics.get("loss_eas", 0),
                loss_domain=train_metrics.get("loss_domain", 0),
                loss_total=train_metrics.get("loss_total", 0),
                lambda_domain=lambda_domain,
                val_loss_eas=val_loss,
            )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    log.info("Early stopping triggered", epoch=epoch)
                    break

        log.info("Phase 2 training completed", best_loss=self.best_val_loss)

        return history

    def train_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 3: Fine-tune Prediction Head.

        Freezes backbone and only trains risk prediction head on
        individual/mixed ancestry data.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_path: Path to save best checkpoint. Default: None

        Returns:
            Training history dictionary.
        """
        log.info("Starting Phase 3: Fine-tune Prediction Head")

        # Freeze backbone, unfreeze risk head
        self.model.freeze_backbone()

        # Create optimizer for risk head only
        optimizer = optim.Adam(
            self.model.risk_head.parameters(),
            lr=self.learning_rate * 10.0,  # Higher LR for fine-tuning
            weight_decay=self.weight_decay,
        )

        # Create scheduler
        scheduler = self._create_scheduler(optimizer, self.num_epochs_phase3)

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.num_epochs_phase3):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch_phase1(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader, phase="phase3")
            history["val_loss"].append(val_loss)

            # Step scheduler
            scheduler.step()

            # Log metrics
            log.info(
                "Phase 3 epoch completed",
                epoch=epoch + 1,
                total_epochs=self.num_epochs_phase3,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    log.info("Early stopping triggered", epoch=epoch)
                    break

        log.info("Phase 3 training completed", best_loss=self.best_val_loss)

        return history

    def _train_epoch_phase1(
        self, train_loader: DataLoader, optimizer: optim.Optimizer
    ) -> float:
        """
        Training epoch for Phase 1.

        Args:
            train_loader: Training data loader.
            optimizer: Optimizer instance.

        Returns:
            Average epoch loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(
                        snp_features=batch["snp_features"],
                        block_indices=batch["block_indices"],
                        block_masks=batch["block_masks"],
                    )

                    # Risk loss
                    risk_loss = self.risk_loss(
                        output["risk_logits"].squeeze(-1), batch["labels"].float()
                    )

                    # Backward pass
                    self.scaler.scale(risk_loss).backward()
                    self.scaler.unscale_(optimizer)

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                    # Optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                output = self.model(
                    snp_features=batch["snp_features"],
                    block_indices=batch["block_indices"],
                    block_masks=batch["block_masks"],
                )

                # Risk loss
                risk_loss = self.risk_loss(
                    output["risk_logits"].squeeze(-1), batch["labels"].float()
                )

                # Backward pass
                risk_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

                # Optimizer step
                optimizer.step()

            total_loss += risk_loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _train_epoch_phase2(
        self,
        train_loader_eur: DataLoader,
        train_loader_eas: DataLoader,
        optimizer: optim.Optimizer,
        lambda_domain: float,
    ) -> Dict[str, float]:
        """
        Training epoch for Phase 2.

        Args:
            train_loader_eur: EUR training data loader.
            train_loader_eas: EAS training data loader.
            optimizer: Optimizer instance.
            lambda_domain: Domain adversarial weight.

        Returns:
            Dictionary with loss components.
        """
        self.model.train()

        loss_eur_total = 0.0
        loss_eas_total = 0.0
        loss_domain_total = 0.0
        num_batches = 0

        # Create iterators
        iter_eas = iter(train_loader_eas)

        for batch_eur in train_loader_eur:
            # Get corresponding EAS batch
            try:
                batch_eas = next(iter_eas)
            except StopIteration:
                iter_eas = iter(train_loader_eas)
                batch_eas = next(iter_eas)

            # Move batches to device
            batch_eur = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_eur.items()}
            batch_eas = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_eas.items()}

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    # EUR forward pass
                    output_eur = self.model(
                        snp_features=batch_eur["snp_features"],
                        block_indices=batch_eur["block_indices"],
                        block_masks=batch_eur["block_masks"],
                    )

                    # EAS forward pass
                    output_eas = self.model(
                        snp_features=batch_eas["snp_features"],
                        block_indices=batch_eas["block_indices"],
                        block_masks=batch_eas["block_masks"],
                        ancestry_labels=batch_eas["ancestry_labels"],
                    )

                    # Loss components
                    loss_eur = self.risk_loss(
                        output_eur["risk_logits"].squeeze(-1),
                        batch_eur["labels"].float(),
                    )

                    loss_eas = self.risk_loss(
                        output_eas["risk_logits"].squeeze(-1),
                        batch_eas["labels"].float(),
                    )

                    loss_domain = self.domain_loss(
                        output_eas["domain_logits"].squeeze(-1),
                        batch_eas["ancestry_labels"].float(),
                    )

                    # Combined loss
                    loss = (
                        loss_eur
                        + self.alpha_eas * loss_eas
                        + lambda_domain * loss_domain
                    )

                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                    # Optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                # EUR forward pass
                output_eur = self.model(
                    snp_features=batch_eur["snp_features"],
                    block_indices=batch_eur["block_indices"],
                    block_masks=batch_eur["block_masks"],
                )

                # EAS forward pass
                output_eas = self.model(
                    snp_features=batch_eas["snp_features"],
                    block_indices=batch_eas["block_indices"],
                    block_masks=batch_eas["block_masks"],
                    ancestry_labels=batch_eas["ancestry_labels"],
                )

                # Loss components
                loss_eur = self.risk_loss(
                    output_eur["risk_logits"].squeeze(-1),
                    batch_eur["labels"].float(),
                )

                loss_eas = self.risk_loss(
                    output_eas["risk_logits"].squeeze(-1),
                    batch_eas["labels"].float(),
                )

                loss_domain = self.domain_loss(
                    output_eas["domain_logits"].squeeze(-1),
                    batch_eas["ancestry_labels"].float(),
                )

                # Combined loss
                loss = (
                    loss_eur
                    + self.alpha_eas * loss_eas
                    + lambda_domain * loss_domain
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

                # Optimizer step
                optimizer.step()

            loss_eur_total += loss_eur.item()
            loss_eas_total += loss_eas.item()
            loss_domain_total += loss_domain.item()
            num_batches += 1

        return {
            "loss_eur": loss_eur_total / num_batches,
            "loss_eas": loss_eas_total / num_batches,
            "loss_domain": loss_domain_total / num_batches,
            "loss_total": (
                loss_eur_total
                + self.alpha_eas * loss_eas_total
                + lambda_domain * loss_domain_total
            )
            / num_batches,
        }

    def _validate(
        self, val_loader: DataLoader, phase: str = "phase1"
    ) -> float:
        """
        Validation epoch.

        Args:
            val_loader: Validation data loader.
            phase: Training phase name for logging.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                if self.use_amp:
                    with autocast():
                        output = self.model(
                            snp_features=batch["snp_features"],
                            block_indices=batch["block_indices"],
                            block_masks=batch["block_masks"],
                        )

                        # Risk loss
                        loss = self.risk_loss(
                            output["risk_logits"].squeeze(-1),
                            batch["labels"].float(),
                        )
                else:
                    output = self.model(
                        snp_features=batch["snp_features"],
                        block_indices=batch["block_indices"],
                        block_masks=batch["block_masks"],
                    )

                    # Risk loss
                    loss = self.risk_loss(
                        output["risk_logits"].squeeze(-1),
                        batch["labels"].float(),
                    )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _compute_domain_weight(self, epoch: int, total_epochs: int) -> float:
        """
        Compute domain adversarial weight with annealing.

        Uses linear schedule from lambda_domain_init to lambda_domain_max.

        Args:
            epoch: Current epoch.
            total_epochs: Total epochs in phase.

        Returns:
            Domain loss weight.
        """
        if total_epochs == 0:
            return self.lambda_domain_max

        progress = epoch / total_epochs
        lambda_domain = (
            self.lambda_domain_init
            + (self.lambda_domain_max - self.lambda_domain_init) * progress
        )

        return lambda_domain

    def _create_scheduler(
        self, optimizer: optim.Optimizer, num_epochs: int
    ) -> optim.lr_scheduler.LRScheduler:
        """
        Create learning rate scheduler with warmup.

        Args:
            optimizer: Optimizer instance.
            num_epochs: Total training epochs.

        Returns:
            Learning rate scheduler.
        """
        # Warmup scheduler
        if self.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_epochs,
            )

            # Cosine annealing scheduler
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - self.warmup_epochs, eta_min=1e-6
            )

            # Combine schedulers
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )

        return scheduler

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        log.info("Loaded checkpoint", path=path, epoch=checkpoint.get("epoch", -1))
