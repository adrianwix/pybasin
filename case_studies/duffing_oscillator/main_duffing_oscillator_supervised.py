from pathlib import Path

# from matplotlib import pyplot as plt
from case_studies.comparison_utils import compare_with_expected
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main() -> tuple[BasinStabilityEstimator, StudyResult]:
    setup = setup_duffing_oscillator_system()

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=setup.get("estimator"),
        template_integrator=setup.get("template_integrator"),
        output_dir="results",
        feature_selector=None,
    )

    result = bse.run()

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_duffing_oscillator_supervised.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "duffing"
        / "main_duffing_supervised.json"
    )

    basin_stability = result["basin_stability"]
    label_mapping = {label: label for label in basin_stability}
    compare_with_expected(basin_stability, label_mapping, expected_file, result["errors"])

    # Test stacked trajectory plot with custom axis limits
    plotter = MatplotlibPlotter(bse)

    # Duffing: all y from -1 to 1, all x from 0 to 50
    # plotter.plot_templates_trajectories(
    #     plotted_var=0,
    #     y_limits=(-1.4, 1.4),
    #     x_limits=(0, 50),
    # )
    # plt.show()  # type: ignore[misc]

    # Test 2D phase space plot (y1 vs y2)
    # plotter.plot_templates_phase_space(x_var=0, y_var=1, time_range=(700, 1000))
    # plt.show()  # type: ignore[misc]

    interactive_plotter = InteractivePlotter(
        bse,
        state_labels={0: "x", 1: "v"},
        options={"templates_time_series": {"time_range": (0, 0.15)}},
    )
    interactive_plotter.run(port=8050)
