from pathlib import Path

import numpy as np
import torch

from case_studies.comparison_utils import compare_with_expected
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DynamicalSystemClusterer
from pybasin.sampler import CsvSampler, GridSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.settings import DYNAMICAL_SYSTEM_FC_PARAMETERS
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main(csv_path: Path | None = None) -> tuple[BasinStabilityEstimator, StudyResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up pendulum system on device: {device}")

    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumJaxODE(params)

    if csv_path is not None:
        sampler = CsvSampler(
            csv_path, coordinate_columns=["x1", "x2"], label_column="label", device=device
        )
        n = sampler.n_samples
    else:
        n = 10000
        sampler = GridSampler(
            min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
            max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
            device=device,
        )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/pendulum",
    )

    feature_extractor = TorchFeatureExtractor(
        features=DYNAMICAL_SYSTEM_FC_PARAMETERS,
        time_steady=950.0,
        normalize=False,
    )

    clusterer = DynamicalSystemClusterer(
        fp_variance_threshold=1e-6,
        lc_periodicity_threshold=0.5,
    )

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=clusterer,
        feature_selector=None,
    )

    result = bse.run()
    print("Basin Stability:", {k: float(v) for k, v in result["basin_stability"].items()})

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_pendulum_dynamical_clusterer.py", main)
    label_mapping = {"FP_0": "FP", "LC_1": "LC"}
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "pendulum"
        / "main_pendulum_case1.json"
    )

    compare_with_expected(result["basin_stability"], label_mapping, expected_file)

    plotter = InteractivePlotter(bse, state_labels={0: "theta", 1: "omega"})
    # plotter.run(port=8050)
