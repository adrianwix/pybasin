import matplotlib

from case_studies.henon_map.setup_henon_map_system import setup_henon_map_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.predictors import HistogramClusterer
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

# Same bin edges as packages/pybasin/tests/accuracy/henon_map/generate_ground_truth.jl
# (range(-10, 240; step=50)): occupied bin 1 is the bounded chaotic attractor, occupied bin
# 25 is the (200, 200) divergence sentinel.
BIN_EDGES: list[float] = [-10.0, 40.0, 90.0, 140.0, 190.0, 240.0]


def main() -> BasinStabilityEstimator:
    setup = setup_henon_map_system()

    grouper = HistogramClusterer(bin_edges=[BIN_EDGES, BIN_EDGES])

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=grouper,
        output_dir="results_henon_histogram",
        feature_selector=None,
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_henon_histogram_experiment.py", main)

    if bse.result is not None:
        print("Basin stability:", bse.result["basin_stability"])

    plotter = MatplotlibPlotter(bse)
    plotter.plot_bse_results()
    plotter.show()
