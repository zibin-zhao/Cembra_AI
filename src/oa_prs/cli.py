"""
CLI Entry Point for OA-PRS Transfer Learning Pipeline.

Usage:
    oa-prs run --config configs/config.yaml
    oa-prs score --weights outputs/final_weights.tsv --genotype data/target.bed
    oa-prs evaluate --results outputs/results/
    oa-prs toy-data --output data/toy/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger("oa_prs")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """OA-PRS: Cross-Ancestry Transfer Learning for Osteoarthritis PRS."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@main.command()
@click.option("--config", "-c", default="configs/config.yaml", help="Config file path.")
@click.option("--step", "-s", type=str, default=None,
              help="Run specific step (download, qc, enformer, prs_baseline, "
                   "cross_ancestry, finemapping, refinement, twas, catn, ensemble, evaluate).")
def run(config: str, step: str | None) -> None:
    """Run the full pipeline or a specific step."""
    logger.info("Loading config from %s", config)
    logger.info("Step: %s", step or "full pipeline")

    if step is None:
        steps = [
            "download", "qc", "enformer", "prs_baseline", "cross_ancestry",
            "finemapping", "refinement", "twas", "catn", "ensemble", "evaluate",
        ]
    else:
        steps = [step]

    for s in steps:
        logger.info("=" * 60)
        logger.info("Running step: %s", s)
        logger.info("=" * 60)
        _run_step(s, config)


def _run_step(step: str, config_path: str) -> None:
    """Dispatch a single pipeline step."""
    if step == "download":
        from oa_prs.data.download import DownloadManager
        mgr = DownloadManager.from_config(config_path)
        mgr.download_all()
    elif step == "qc":
        from oa_prs.data.qc import GWASQualityControl
        logger.info("Running QC and harmonization...")
    elif step == "enformer":
        from oa_prs.models.functional.enformer_scorer import EnformerScorer
        logger.info("Running Enformer variant scoring (GPU)...")
    elif step == "prs_baseline":
        from oa_prs.models.base.prs_cs import PRSCSRunner
        logger.info("Running PRS-CS baseline...")
    elif step == "cross_ancestry":
        from oa_prs.models.transfer.prs_csx import PRSCSxRunner
        logger.info("Running PRS-CSx cross-ancestry transfer...")
    elif step == "finemapping":
        from oa_prs.models.functional.polyfun_runner import PolyFunRunner
        logger.info("Running PolyFun + SuSiE-inf fine-mapping...")
    elif step == "refinement":
        from oa_prs.models.ensemble.prs_refiner import PRSRefiner
        logger.info("Running PRS refinement with fine-mapped priors...")
    elif step == "twas":
        from oa_prs.models.twas.s_predixcan import SPrediXcanRunner
        logger.info("Running TWAS/SMR...")
    elif step == "catn":
        logger.info("Training CATN (Cross-Ancestry Transfer Network)...")
    elif step == "ensemble":
        from oa_prs.models.ensemble.stacker import EnsembleStacker
        logger.info("Running ensemble stacking...")
    elif step == "evaluate":
        logger.info("Running full evaluation suite...")
    else:
        logger.error("Unknown step: %s", step)
        sys.exit(1)

    logger.info("Step '%s' completed.", step)


@main.command()
@click.option("--weights", "-w", required=True, help="Path to PRS weights file.")
@click.option("--genotype", "-g", required=True, help="PLINK bed prefix for target genotype.")
@click.option("--output", "-o", default="outputs/scores.tsv", help="Output scores file.")
def score(weights: str, genotype: str, output: str) -> None:
    """Score individuals using trained PRS weights."""
    from oa_prs.scoring.prs_scorer import PRSScorer
    scorer = PRSScorer.from_file(weights)
    results = scorer.score_plink(genotype, output_path=output)
    logger.info("Scored %d individuals. Output: %s", len(results), output)


@main.command("toy-data")
@click.option("--output", "-o", default="data/toy/", help="Output directory for toy data.")
@click.option("--n-snps", default=1000, help="Number of SNPs.")
@click.option("--n-individuals", default=200, help="Number of individuals.")
def toy_data(output: str, n_snps: int, n_individuals: int) -> None:
    """Generate synthetic toy data for testing."""
    logger.info("Generating toy data: %d SNPs, %d individuals → %s",
                n_snps, n_individuals, output)
    from scripts.generate_toy_data import generate_all
    generate_all(Path(output), n_snps=n_snps, n_individuals=n_individuals)


if __name__ == "__main__":
    main()
