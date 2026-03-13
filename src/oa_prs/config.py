"""
Hydra Configuration Loader for OA-PRS Pipeline.

Provides centralized config management with validation and defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data-related configuration."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    toy_dir: str = "data/toy"
    genome_build: str = "hg38"
    gwas_sources: dict[str, Any] = field(default_factory=dict)
    ld_references: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model-related configuration."""

    prs_cs: dict[str, Any] = field(default_factory=dict)
    prs_csx: dict[str, Any] = field(default_factory=dict)
    bridge_prs: dict[str, Any] = field(default_factory=dict)
    ldpred2: dict[str, Any] = field(default_factory=dict)
    enformer: dict[str, Any] = field(default_factory=dict)
    polyfun: dict[str, Any] = field(default_factory=dict)
    susie_inf: dict[str, Any] = field(default_factory=dict)
    catn: dict[str, Any] = field(default_factory=dict)
    twas: dict[str, Any] = field(default_factory=dict)
    ensemble: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    metrics: list[str] = field(
        default_factory=lambda: ["discrimination", "calibration", "fairness", "risk_stratification"]
    )
    n_quantiles: list[int] = field(default_factory=lambda: [5, 10, 20])
    top_percentiles: list[int] = field(default_factory=lambda: [1, 5, 10])
    fairness_subgroups: list[str] = field(default_factory=lambda: ["ancestry", "sex"])
    cv_folds: int = 5


@dataclass
class SlurmConfig:
    """SLURM cluster configuration."""

    partition_cpu: str = "cpu"
    partition_gpu: str = "gpu"
    gpu_type: str = "a100"
    account: str = ""
    container: str = "containers/oa_prs_gpu.sif"


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""

    project_name: str = "oa_prs_transfer"
    phenotype: str = "knee_oa"
    target_ancestry: str = "EAS"
    base_ancestry: str = "EUR"
    seed: int = 42
    n_threads: int = 4
    output_dir: str = "outputs"

    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        if self.seed < 0:
            warnings.append("Seed should be non-negative.")
        if self.target_ancestry not in ("EAS", "AFR", "SAS", "AMR"):
            warnings.append(f"Unusual target ancestry: {self.target_ancestry}")
        if self.phenotype not in ("knee_oa", "hip_oa", "any_oa", "thr", "tkr"):
            warnings.append(f"Unrecognized phenotype: {self.phenotype}")
        return warnings


def load_config(config_path: str | Path = "configs/config.yaml") -> PipelineConfig:
    """
    Load and merge pipeline configuration from YAML files.

    Supports Hydra-style defaults where the main config references
    sub-configs in subdirectories.

    Args:
        config_path: Path to the master config YAML.

    Returns:
        Validated PipelineConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s. Using defaults.", config_path)
        return PipelineConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config_dir = config_path.parent

    # Load sub-configs referenced in defaults
    defaults = raw.pop("defaults", [])
    for default in defaults:
        if isinstance(default, dict):
            for category, filename in default.items():
                sub_path = config_dir / category / f"{filename}.yaml"
                if sub_path.exists():
                    with open(sub_path) as f:
                        sub_cfg = yaml.safe_load(f) or {}
                    raw.setdefault(category, {}).update(sub_cfg)

    # Also load any additional config files in known subdirectories
    for sub_dir_name in ["data", "models", "evaluation", "slurm"]:
        sub_dir = config_dir / sub_dir_name
        if sub_dir.is_dir():
            for yaml_file in sorted(sub_dir.glob("*.yaml")):
                with open(yaml_file) as f:
                    sub_cfg = yaml.safe_load(f) or {}
                key = yaml_file.stem
                raw.setdefault(sub_dir_name, {})[key] = sub_cfg

    # Build config dataclass
    data_raw = raw.get("data", {})
    model_raw = raw.get("models", {})
    eval_raw = raw.get("evaluation", {})
    slurm_raw = raw.get("slurm", {})

    cfg = PipelineConfig(
        project_name=raw.get("project_name", "oa_prs_transfer"),
        phenotype=raw.get("phenotype", "knee_oa"),
        target_ancestry=raw.get("target_ancestry", "EAS"),
        base_ancestry=raw.get("base_ancestry", "EUR"),
        seed=raw.get("seed", 42),
        n_threads=raw.get("n_threads", 4),
        output_dir=raw.get("output_dir", "outputs"),
        data=DataConfig(
            raw_dir=data_raw.get("raw_dir", "data/raw"),
            processed_dir=data_raw.get("processed_dir", "data/processed"),
            external_dir=data_raw.get("external_dir", "data/external"),
            genome_build=data_raw.get("genome_build", "hg38"),
            gwas_sources=data_raw.get("gwas_sources", {}),
            ld_references=data_raw.get("ld_references", {}),
        ),
        models=ModelConfig(**{k: v for k, v in model_raw.items() if k in ModelConfig.__dataclass_fields__}),
        evaluation=EvaluationConfig(
            metrics=eval_raw.get("metrics", ["discrimination", "calibration", "fairness"]),
            cv_folds=eval_raw.get("cv_folds", 5),
        ),
        slurm=SlurmConfig(
            partition_cpu=slurm_raw.get("partition_cpu", "cpu"),
            partition_gpu=slurm_raw.get("partition_gpu", "gpu"),
            gpu_type=slurm_raw.get("gpu_type", "a100"),
        ),
    )

    warnings = cfg.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    logger.info(
        "Loaded config: phenotype=%s, target=%s, base=%s",
        cfg.phenotype, cfg.target_ancestry, cfg.base_ancestry,
    )
    return cfg
