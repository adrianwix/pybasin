from pathlib import Path

import torch

from case_studies.comparison_utils import compare_with_expected
from case_studies.friction.friction_jax_ode import FrictionJaxODE, FrictionParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DynamicalSystemClusterer
from pybasin.sampler import CsvSampler, UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.settings import DYNAMICAL_SYSTEM_FC_PARAMETERS
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main(csv_path: Path | None = None) -> tuple[BasinStabilityEstimator, StudyResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up friction system on device: {device}")

    params: FrictionParams = {
        "v_d": 1.5,
        "xi": 0.05,
        "musd": 2.0,
        "mud": 0.5,
        "muv": 0.0,
        "v0": 0.5,
    }

    ode_system = FrictionJaxODE(params)

    if csv_path is not None:
        sampler = CsvSampler(
            csv_path, coordinate_columns=["x1", "x2"], label_column="label", device=device
        )
        n = sampler.n_samples
    else:
        n = 5000
        sampler = UniformRandomSampler(
            min_limits=[-2.0, 0.0],
            max_limits=[2.0, 2.0],
            device=device,
        )

    solver = JaxSolver(
        t_span=(0, 500),
        t_steps=500,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/friction",
    )

    feature_extractor = TorchFeatureExtractor(
        features=DYNAMICAL_SYSTEM_FC_PARAMETERS,
        time_steady=400.0,
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
    bse, result = time_execution("main_friction_dynamical_clusterer.py", main)
    label_mapping = {"FP_0": "FP", "LC_1": "LC"}
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
