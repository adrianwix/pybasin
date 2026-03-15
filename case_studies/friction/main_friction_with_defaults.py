from pathlib import Path

from case_studies.comparison_utils import compare_with_expected
from case_studies.friction.setup_friction_system import setup_friction_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.sampler import Sampler
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main(sampler_override: Sampler | None = None) -> tuple[BasinStabilityEstimator, StudyResult]:
    props = setup_friction_system()
    sampler = sampler_override if sampler_override is not None else props["sampler"]

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=sampler,
    )

    result = bse.run()
    print("Basin Stability:", {k: float(v) for k, v in result["basin_stability"].items()})

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_friction_with_defaults.py", main)

    label_mapping = {"0": "LC", "1": "FP"}
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "friction"
        / "main_friction_case1.json"
    )

    compare_with_expected(result["basin_stability"], label_mapping, expected_file)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
