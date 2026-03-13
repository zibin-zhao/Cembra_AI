"""
Evaluation Report Generator.

Produces comprehensive evaluation reports in JSON and Markdown format,
summarizing all metrics across models and subgroups.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate evaluation reports for the OA-PRS pipeline.

    Collects results from all evaluation modules and produces
    structured reports for publication and review.

    Args:
        output_dir: Directory for saving reports.
        project_name: Name of the project for report headers.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/evaluation",
        project_name: str = "OA-PRS Transfer Learning",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.results: dict[str, Any] = {}

    def add_discrimination(self, model_name: str, disc_results: Any) -> None:
        """Add discrimination metrics for a model."""
        self.results.setdefault("discrimination", {})[model_name] = {
            "AUC-ROC": getattr(disc_results, "auc_roc", None),
            "AUC-PR": getattr(disc_results, "auc_pr", None),
        }

    def add_calibration(self, model_name: str, cal_results: Any) -> None:
        """Add calibration metrics for a model."""
        self.results.setdefault("calibration", {})[model_name] = {
            "Brier_score": getattr(cal_results, "brier_score", None),
            "HL_statistic": getattr(cal_results, "hosmer_lemeshow_stat", None),
            "HL_p_value": getattr(cal_results, "hosmer_lemeshow_p", None),
        }

    def add_fairness(self, fair_results: Any) -> None:
        """Add fairness evaluation results."""
        self.results["fairness"] = {
            "AUC_gap": getattr(fair_results, "auc_gap", None),
            "calibration_gap": getattr(fair_results, "calibration_gap", None),
            "threshold_consistency": getattr(fair_results, "threshold_consistency", None),
            "subgroup_metrics": getattr(fair_results, "subgroup_metrics", {}),
        }

    def add_risk_stratification(
        self, model_name: str, quantile_results: Any, top_pct_results: Any = None
    ) -> None:
        """Add risk stratification results."""
        entry: dict[str, Any] = {}
        if hasattr(quantile_results, "odds_ratios"):
            entry["top_quantile_OR"] = float(quantile_results.odds_ratios[-1])
        if top_pct_results is not None:
            entry["top_percentile"] = top_pct_results
        self.results.setdefault("risk_stratification", {})[model_name] = entry

    def add_ablation(self, ablation_results: Any) -> None:
        """Add ablation study results."""
        self.results["ablation"] = {
            "full_model_AUC": getattr(ablation_results, "full_model_auc", None),
            "branch_importance": getattr(ablation_results, "branch_importance", {}),
        }

    def add_loso(self, loso_results: Any) -> None:
        """Add leave-one-study-out results."""
        self.results["leave_one_study"] = {
            "mean_AUC-ROC": getattr(loso_results, "mean_auc_roc", None),
            "std_AUC-ROC": getattr(loso_results, "std_auc_roc", None),
            "consistency_score": getattr(loso_results, "consistency_score", None),
            "per_study": getattr(loso_results, "study_results", {}),
        }

    def add_custom(self, section_name: str, data: dict[str, Any]) -> None:
        """Add custom section to the report."""
        self.results[section_name] = data

    def generate_json(self, filename: str = "evaluation_report.json") -> Path:
        """Generate JSON report."""
        report = {
            "project": self.project_name,
            "generated_at": datetime.now().isoformat(),
            "results": self._serialize(self.results),
        }

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("JSON report saved to: %s", output_path)
        return output_path

    def generate_markdown(self, filename: str = "evaluation_report.md") -> Path:
        """Generate Markdown report."""
        lines = [
            f"# {self.project_name} — Evaluation Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
        ]

        # Discrimination table
        if "discrimination" in self.results:
            lines.extend(self._make_discrimination_section())

        # Calibration table
        if "calibration" in self.results:
            lines.extend(self._make_calibration_section())

        # Fairness
        if "fairness" in self.results:
            lines.extend(self._make_fairness_section())

        # Risk Stratification
        if "risk_stratification" in self.results:
            lines.extend(self._make_risk_section())

        # Ablation
        if "ablation" in self.results:
            lines.extend(self._make_ablation_section())

        # LOSO
        if "leave_one_study" in self.results:
            lines.extend(self._make_loso_section())

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Markdown report saved to: %s", output_path)
        return output_path

    def _make_discrimination_section(self) -> list[str]:
        disc = self.results["discrimination"]
        lines = [
            "## Discrimination Metrics",
            "",
            "| Model | AUC-ROC | AUC-PR |",
            "|-------|---------|--------|",
        ]
        for model, metrics in disc.items():
            auc_roc = f"{metrics['AUC-ROC']:.4f}" if metrics.get("AUC-ROC") else "N/A"
            auc_pr = f"{metrics['AUC-PR']:.4f}" if metrics.get("AUC-PR") else "N/A"
            lines.append(f"| {model} | {auc_roc} | {auc_pr} |")
        lines.append("")
        return lines

    def _make_calibration_section(self) -> list[str]:
        cal = self.results["calibration"]
        lines = [
            "## Calibration Metrics",
            "",
            "| Model | Brier Score | HL Statistic | HL p-value |",
            "|-------|-------------|-------------|------------|",
        ]
        for model, metrics in cal.items():
            brier = f"{metrics['Brier_score']:.4f}" if metrics.get("Brier_score") else "N/A"
            hl = f"{metrics['HL_statistic']:.2f}" if metrics.get("HL_statistic") else "N/A"
            hlp = f"{metrics['HL_p_value']:.4f}" if metrics.get("HL_p_value") else "N/A"
            lines.append(f"| {model} | {brier} | {hl} | {hlp} |")
        lines.append("")
        return lines

    def _make_fairness_section(self) -> list[str]:
        fair = self.results["fairness"]
        lines = [
            "## Cross-Ancestry Fairness",
            "",
            f"- **AUC Gap:** {fair.get('AUC_gap', 'N/A')}",
            f"- **Calibration Gap:** {fair.get('calibration_gap', 'N/A')}",
            f"- **Threshold Consistency:** {fair.get('threshold_consistency', 'N/A')}",
            "",
        ]
        subgroups = fair.get("subgroup_metrics", {})
        if subgroups:
            lines.extend([
                "| Subgroup | N | AUC-ROC | Brier |",
                "|----------|---|---------|-------|",
            ])
            for grp, metrics in subgroups.items():
                lines.append(
                    f"| {grp} | {metrics.get('n', 'N/A')} | "
                    f"{metrics.get('AUC-ROC', 'N/A'):.4f} | "
                    f"{metrics.get('Brier', 'N/A'):.4f} |"
                )
            lines.append("")
        return lines

    def _make_risk_section(self) -> list[str]:
        risk = self.results["risk_stratification"]
        lines = ["## Risk Stratification", ""]
        for model, data in risk.items():
            top_or = data.get("top_quantile_OR", "N/A")
            lines.append(f"- **{model}**: Top quantile OR = {top_or}")
        lines.append("")
        return lines

    def _make_ablation_section(self) -> list[str]:
        abl = self.results["ablation"]
        lines = [
            "## Ablation Study",
            "",
            f"Full model AUC: {abl.get('full_model_AUC', 'N/A')}",
            "",
            "| Branch | Importance (ΔAUC) |",
            "|--------|-------------------|",
        ]
        for branch, imp in abl.get("branch_importance", {}).items():
            lines.append(f"| {branch} | {imp:.4f} |")
        lines.append("")
        return lines

    def _make_loso_section(self) -> list[str]:
        loso = self.results["leave_one_study"]
        lines = [
            "## Leave-One-Study-Out Validation",
            "",
            f"- **Mean AUC-ROC:** {loso.get('mean_AUC-ROC', 'N/A')}",
            f"- **Std AUC-ROC:** {loso.get('std_AUC-ROC', 'N/A')}",
            f"- **Consistency Score:** {loso.get('consistency_score', 'N/A')}",
            "",
        ]
        return lines

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: ReportGenerator._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReportGenerator._serialize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
