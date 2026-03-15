from pathlib import Path

import torch

from case_studies.comparison_utils import compare_with_expected
from case_studies.lorenz.setup_lorenz_system import lorenz_stop_event, setup_lorenz_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.types import InteractivePlotterOptions
from pybasin.sampler import Sampler
from pybasin.solvers.jax_solver import JaxSolver
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main(sampler_override: Sampler | None = None) -> tuple[BasinStabilityEstimator, StudyResult]:
    props = setup_lorenz_system()
    sampler = sampler_override if sampler_override is not None else props.get("sampler")

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The default solver does not work here because it needs the event_fn to stop
    solver = JaxSolver(
        device=device,
        cache_dir=".pybasin_cache/lorenz",
        event_fn=lorenz_stop_event,
    )

    bse = BasinStabilityEstimator(
        n=props.get("n"),
        ode_system=props.get("ode_system"),
        sampler=sampler,
        solver=solver,
    )

    result = bse.run()
    print("Basin Stability:", {k: float(v) for k, v in result["basin_stability"].items()})

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_lorenz.py", main)

    label_mapping = {"0": "butterfly1", "1": "butterfly2", "unbounded": "unbounded"}
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "lorenz"
        / "main_lorenz.json"
    )
    compare_with_expected(result["basin_stability"], label_mapping, expected_file)

    options: InteractivePlotterOptions = {
        "templates_phase_space": {"x_axis": 1, "y_axis": 2, "exclude_templates": ["unbounded"]},
        "feature_space": {"exclude_labels": ["unbounded"]},
    }
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
    plotter.run(port=8050)
