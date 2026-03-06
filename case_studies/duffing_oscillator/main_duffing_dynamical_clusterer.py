from pathlib import Path

import torch

from case_studies.comparison_utils import compare_with_expected
from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE, DuffingParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DynamicalSystemClusterer
from pybasin.sampler import CsvSampler, UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.settings import DYNAMICAL_SYSTEM_FC_PARAMETERS
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.utils import time_execution


def main(csv_path: Path | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Duffing oscillator system on device: {device}")

    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}

    ode_system = DuffingJaxODE(params)

    if csv_path is not None:
        sampler = CsvSampler(
            csv_path, coordinate_columns=["x1", "x2"], label_column="label", device=device
        )
        n = sampler.n_samples
    else:
        n = 2000
        sampler = UniformRandomSampler(
            min_limits=[-1, -0.5],
            max_limits=[1, 1],
            device=device,
        )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/duffing",
    )

    feature_extractor = TorchFeatureExtractor(
        features=DYNAMICAL_SYSTEM_FC_PARAMETERS,
        time_steady=900.0,
        normalize=False,
    )

    clusterer = DynamicalSystemClusterer(fp_variance_threshold=1e-6, lc_periodicity_threshold=0.5)

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=clusterer,
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", {k: float(v) for k, v in basin_stability.items()})

    return bse


if __name__ == "__main__":
    bse = time_execution("main_duffing_dynamical_clusterer.py", main)

    label_mapping = {
        "LC_0": "period-1 LC y_1",
        "LC_1": "period-2 LC y_3",
        "LC_2": "period-2 LC y_4",
        "LC_3": "period-3 LC y_5",
        "LC_4": "period-1 LC y_2",
    }
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "duffing"
        / "main_duffing_supervised.json"
    )

    unsupervised_mapping = {
        "LC_0": "y3",
        "LC_1": "y4",
        "LC_2": "y5",
        "LC_3": "y2",
        "LC_4": "y1",
    }
    unsupervised_file = expected_file.parent / "main_duffing_unsupervised.json"

    if bse.bs_vals is not None:
        print("\n\nMatlab supervised results:")
        compare_with_expected(bse.bs_vals, label_mapping, expected_file)
        print("Matlab unsupervised results:")
        compare_with_expected(bse.bs_vals, unsupervised_mapping, unsupervised_file)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
    # plotter.run(port=8050)
