from typing import Any

import jax.numpy as jnp
import torch
from jax import Array
from sklearn.neighbors import KNeighborsClassifier

from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE, LorenzParams
from pybasin.feature_extractors.jax.jax_feature_extractor import JaxFeatureExtractor
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import SetupProperties


def lorenz_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Array:
    """
    Event function to stop integration when amplitude exceeds threshold.

    Replicates lorenzStopFcn.m (bSTAB library) behavior:
    - Returns positive when under threshold (continue integration)
    - Returns negative/zero when over threshold (stop integration)
    """
    max_val = 200.0
    max_abs_y = jnp.max(jnp.abs(y))
    return max_val - max_abs_y


def setup_lorenz_system() -> SetupProperties:
    n = 20_000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}

    ode_system = LorenzJaxODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0], max_limits=[10.0, 20.0, 0.0], device=device
    )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=4000,
        t_eval=(900.0, 1000.0),
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/lorenz",
        event_fn=lorenz_stop_event,
    )

    feature_extractor = JaxFeatureExtractor(
        time_steady=900.0,
        normalize=False,
        features_per_state={
            0: {"mean": None},
            1: None,
            2: None,
        },
    )

    classifier_initial_conditions = [
        [0.8, -3.0, 0.0],
        [-0.8, 3.0, 0.0],
        [10.0, 50.0, 0.0],
    ]

    classifier_labels = ["chaotic attractor 1", "chaotic attractor 2", "unbounded"]

    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=classifier_initial_conditions,
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
