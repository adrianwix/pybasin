import warnings
from pathlib import Path

import matplotlib

# from matplotlib import pyplot as plt
from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

warnings.filterwarnings("ignore", category=SyntaxWarning, module="nolds")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        output_dir="results_case1",
        feature_selector=None,
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case1.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "pendulum"
        / "main_pendulum_case1.json"
    )

    if bse.result is not None:
        errors = bse.get_errors()
        compare_with_expected_by_size(bse.result["basin_stability"], expected_file, errors)

    # Test matplotlib plotter with new modular functions
    mpl_plotter = MatplotlibPlotter(bse)

    # Test individual plots
    # mpl_plotter.plot_basin_stability_bars()
    # mpl_plotter.plot_state_space()
    # mpl_plotter.plot_feature_space()

    # mpl_plotter.plot_templates_trajectories(plotted_var=1, x_limits=(0, 150))
    # plt.show()  # type: ignore[misc]

    # Test combined plot
    # mpl_plotter.plot_bse_results()

    # Interactive plotter
    plotter = InteractivePlotter(
        bse,
        state_labels={0: "θ", 1: "ω"},
        options={"templates_time_series": {"state_variable": 1}},
    )
    plotter.run(debug=False)
