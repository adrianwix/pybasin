from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DBSCANClusterer
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main() -> tuple[BasinStabilityEstimator, StudyResult]:
    setup = setup_duffing_oscillator_system()

    estimator = DBSCANClusterer(auto_tune=True, assign_noise=True)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=estimator,
        output_dir="results_unsupervised",
        feature_selector=None,
    )

    result = bse.run()

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_duffing_oscillator_unsupervised.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "duffing"
        / "main_duffing_unsupervised.json"
    )

    compare_with_expected_by_size(result["basin_stability"], expected_file, result["errors"])

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
