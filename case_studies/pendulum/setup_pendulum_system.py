import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# from tsfresh.feature_extraction import (  # pyright: ignore[reportMissingTypeStubs]
#     MinimalFCParameters,
# )
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator

# from pybasin.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import SetupProperties


def setup_pendulum_system() -> SetupProperties:
    n = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up pendulum system on device: {device}")

    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumJaxODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        # t_eval=(950.0, 1000.0),
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/pendulum",
    )

    feature_extractor = TorchFeatureExtractor(
        time_steady=950.0,
        features=None,
        features_per_state={
            1: {"log_delta": None},
        },
        normalize=False,
    )

    template_y0 = [
        [0.4, 0.0],
        [2.7, 0.0],
    ]
    classifier_labels = ["FP", "LC"]

    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=template_y0,
        labels=classifier_labels,
        ode_params=params,
    )

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "estimator": knn,
        "template_integrator": template_integrator,
    }
