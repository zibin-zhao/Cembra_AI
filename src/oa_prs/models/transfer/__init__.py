"""
Cross-Ancestry Transfer Network (CATN) implementations.

This module contains the CATN model for cross-ancestry polygenic risk score
transfer learning, including model architecture, training, and inference.
"""

from .catn_layers import (
    CrossBlockAttention,
    DomainDiscriminator,
    GradientReversalLayer,
    LDBlockTransformer,
    RiskPredictionHead,
    SNPFeatureEncoder,
)
from .catn_inference import CATNPredictor, load_checkpoint
from .catn_model import CrossAncestryTransferNet
from .catn_trainer import CATNTrainer

__all__ = [
    # Layers
    "GradientReversalLayer",
    "SNPFeatureEncoder",
    "LDBlockTransformer",
    "CrossBlockAttention",
    "RiskPredictionHead",
    "DomainDiscriminator",
    # Model
    "CrossAncestryTransferNet",
    # Training
    "CATNTrainer",
    # Inference
    "CATNPredictor",
    "load_checkpoint",
]
