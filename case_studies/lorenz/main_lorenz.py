from pathlib import Path

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.plotters.types import InteractivePlotterOptions
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main() -> tuple[BasinStabilityEstimator, StudyResult]:
    props = setup_lorenz_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        output_dir="results_case1",
        # feature_selector=None,
    )

    result = bse.run()
    print("Basin Stability:", result["basin_stability"])

    # bse.save()

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_lorenz.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "lorenz"
        / "main_lorenz.json"
    )

    compare_with_expected_by_size(result["basin_stability"], expected_file, result["errors"])

    plotter = MatplotlibPlotter(bse)
    plotter.plot_state_space()
    plotter.show()

    options: InteractivePlotterOptions = {
        "templates_phase_space": {"x_axis": 1, "y_axis": 2, "exclude_templates": ["unbounded"]},
        "feature_space": {"exclude_labels": ["unbounded"]},
    }
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
    # plotter.run(port=8050, debug=True)
