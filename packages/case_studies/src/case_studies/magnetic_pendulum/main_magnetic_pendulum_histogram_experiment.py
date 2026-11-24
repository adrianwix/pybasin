import matplotlib

from case_studies.magnetic_pendulum.setup_magnetic_pendulum_system import (
    setup_magnetic_pendulum_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.predictors import HistogramClusterer
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

# The damped pendulum settles at one of 3 tight, well-separated equilibria (empirically
# (-0.50, 0.86), (-0.50, -0.86), (0.99, 0.00)), so these edges only need to keep each
# equilibrium comfortably inside a single bin -- validated against real Attractors.jl
# ground truth in packages/pybasin/tests/accuracy/magnetic_pendulum
# (see generate_ground_truth.jl for why a uniform fine grid isn't a safe choice here).
BIN_EDGES_X: list[float] = [-1.5, 0.0, 1.5]
BIN_EDGES_Y: list[float] = [-1.5, -0.5, 0.5, 1.5]


def main() -> BasinStabilityEstimator:
    setup = setup_magnetic_pendulum_system()

    grouper = HistogramClusterer(bin_edges=[BIN_EDGES_X, BIN_EDGES_Y])

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=grouper,
        output_dir="results_magnetic_pendulum_histogram",
        feature_selector=None,
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_magnetic_pendulum_histogram_experiment.py", main)

    if bse.result is not None:
        print("Basin stability:", bse.result["basin_stability"])

    plotter = MatplotlibPlotter(bse)
    plotter.plot_bse_results()
    plotter.show()
