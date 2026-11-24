import matplotlib

from case_studies.henon_map.setup_henon_map_system import setup_henon_map_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.predictors import PairwiseComparisonClusterer
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

# The bounded chaotic attractor's mean-state features cluster tightly (empirically,
# within ~0.11 units per axis); the (200, 200) divergence sentinel sits ~283 units
# away. threshold=1.0 separates the two with a wide margin on both sides -- validated
# against real Attractors.jl ground truth in
# packages/pybasin/tests/accuracy/henon_map (see generate_ground_truth.jl).
THRESHOLD = 1.0


def main() -> BasinStabilityEstimator:
    setup = setup_henon_map_system()

    clusterer = PairwiseComparisonClusterer(threshold=THRESHOLD, rescale_features=False)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup.get("solver"),
        feature_extractor=setup.get("feature_extractor"),
        predictor=clusterer,
        output_dir="results_henon_pairwise",
        feature_selector=None,
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_henon_pairwise_experiment.py", main)

    if bse.result is not None:
        print("Basin stability:", bse.result["basin_stability"])

    plotter = MatplotlibPlotter(bse)
    plotter.plot_bse_results()
    plotter.show()
