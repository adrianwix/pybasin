import matplotlib

from case_studies.magnetic_pendulum.setup_magnetic_pendulum_system import (
    setup_magnetic_pendulum_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.predictors import PairwiseComparisonClusterer
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

# Matches Attractors.jl's own ground truth exactly -- validated in
# packages/pybasin/tests/accuracy/magnetic_pendulum (see generate_ground_truth.jl).
THRESHOLD = 1.0


def main() -> BasinStabilityEstimator:
    setup = setup_magnetic_pendulum_system()

    clusterer = PairwiseComparisonClusterer(threshold=THRESHOLD, rescale_features=False)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=clusterer,
        output_dir="results_magnetic_pendulum_pairwise",
        feature_selector=None,
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_magnetic_pendulum_pairwise_experiment.py", main)

    if bse.result is not None:
        print("Basin stability:", bse.result["basin_stability"])

    plotter = MatplotlibPlotter(bse)
    plotter.plot_bse_results()
    plotter.show()
