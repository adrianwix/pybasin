from pathlib import Path
from typing import Any

import jax.numpy as jnp
import torch
from jax import Array

from case_studies.comparison_utils import compare_with_expected
from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE, LorenzParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.predictors import DynamicalSystemClusterer
from pybasin.sampler import CsvSampler, GridSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.settings import DYNAMICAL_SYSTEM_FC_PARAMETERS
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def lorenz_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Array:
    max_val = 200.0
    max_abs_y = jnp.max(jnp.abs(y))
    return max_val - max_abs_y


def main(csv_path: Path | None = None) -> tuple[BasinStabilityEstimator, StudyResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}

    ode_system = LorenzJaxODE(params)

    if csv_path is not None:
        sampler = CsvSampler(
            csv_path, coordinate_columns=["x1", "x2", "x3"], label_column="label", device=device
        )
        n = sampler.n_samples
    else:
        n = 20000
        sampler = GridSampler(
            min_limits=[-10.0, -20.0, 0.0],
            max_limits=[10.0, 20.0, 0.0],
            device=device,
        )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/lorenz",
        event_fn=lorenz_stop_event,
    )

    feature_extractor = TorchFeatureExtractor(
        features=DYNAMICAL_SYSTEM_FC_PARAMETERS,
        time_steady=900.0,
        normalize=False,
    )

    clusterer = DynamicalSystemClusterer(
        fp_variance_threshold=1e-6,
        lc_periodicity_threshold=0.5,
        tiers=["FP", "LC", "chaos"],
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
    bse, result = time_execution("main_lorenz_dynamical_clusterer.py", main)
    label_mapping = {
        "chaos_0": "butterfly1",
        "chaos_1": "butterfly2",
        "unbounded": "unbounded",
    }
    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "lorenz"
        / "main_lorenz.json"
    )

    compare_with_expected(result["basin_stability"], label_mapping, expected_file)

    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"})
    # plotter.run(port=8050)
