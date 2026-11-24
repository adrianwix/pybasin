"""Documentation artifacts for package-level numerical-accuracy studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from test_support import BasinFractionComparison, ClassificationMetrics, ExecutionTarget


@dataclass
class AccuracyComparison:
    """Serializable evidence from one implementation/reference comparison."""

    system_name: str
    method_name: str
    execution_target: ExecutionTarget
    classification_metrics: ClassificationMetrics
    basins: list[BasinFractionComparison]
    reference_name: str = "Attractors.jl"
    uncertainty_sigma_limit: float = 1.96

    @property
    def case_id(self) -> str:
        """Stable artifact identifier used by documentation macros."""
        return f"{self.system_name}_{self.method_name}_{self.execution_target.name.lower()}"

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-serializable validation record."""
        return {
            "system_name": self.system_name,
            "method_name": self.method_name,
            "execution_target": self.execution_target.value,
            "reference_name": self.reference_name,
            "uncertainty_sigma_limit": self.uncertainty_sigma_limit,
            "classification_metrics": self.classification_metrics.to_dict(),
            "attractors": [basin.to_dict() for basin in self.basins],
        }


class AccuracyArtifactCollector:
    """Collect accuracy comparisons and write stable documentation inputs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.comparisons: list[AccuracyComparison] = []

    def add(self, comparison: AccuracyComparison) -> None:
        """Queue one comparison for export after a successful test session."""
        self.comparisons.append(comparison)

    def write(self) -> None:
        """Write every queued comparison as formatted JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for comparison in self.comparisons:
            output_path = self.output_dir / f"{comparison.case_id}_comparison.json"
            output_path.write_text(json.dumps(comparison.to_dict(), indent=2) + "\n")
