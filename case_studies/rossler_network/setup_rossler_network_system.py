"""Setup for the Rössler network basin stability case study."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax import Array

from case_studies.rossler_network.rossler_network_jax_ode import (
    RosslerNetworkJaxODE,
    RosslerNetworkParams,
)
from case_studies.rossler_network.rossler_network_topology import EDGES_I, EDGES_J, N_NODES
from case_studies.rossler_network.synchronization_classifier import (
    SynchronizationClassifier,
)
from case_studies.rossler_network.synchronization_feature_extractor import (
    SynchronizationFeatureExtractor,
)
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.types import SetupProperties

K_VALUES_FROM_PAPER = np.array(
    [0.119, 0.139, 0.159, 0.179, 0.198, 0.218, 0.238, 0.258, 0.278, 0.297, 0.317]
)
EXPECTED_SB_FROM_PAPER = np.array(
    [0.226, 0.274, 0.330, 0.346, 0.472, 0.496, 0.594, 0.628, 0.656, 0.694, 0.690]
)
EXPECTED_MEAN_SB = 0.490


def rossler_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Array:
    """
    Event function to stop integration when amplitude exceeds threshold.

    Returns positive when under threshold (continue integration),
    negative/zero when over threshold (stop integration).
    """
    max_val = 400
    max_abs_y = jnp.max(jnp.abs(y))
    return max_val - max_abs_y


def setup_rossler_network_system() -> SetupProperties:
    """Setup the Rössler network system for basin stability estimation.

    Uses coupling strength K=0.218 (expected S_B ≈ 0.496 from paper).

    :return: Configuration dictionary for BasinStabilityEstimator.
    """
    k = 0.218
    n = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Rössler network system on device: {device}")
    print(f"  N = {N_NODES} nodes, k = {k}")

    params: RosslerNetworkParams = {
        "a": 0.2,
        "b": 0.2,
        "c": 7.0,
        "K": k,
    }

    ode_system = RosslerNetworkJaxODE(params, n=N_NODES, edges_i=EDGES_I, edges_j=EDGES_J)

    min_limits = (
        [-15.0] * N_NODES  # x_i in [-15, 15]
        + [-15.0] * N_NODES  # y_i in [-15, 15]
        + [-5.0] * N_NODES  # z_i in [-5, 35]
    )
    max_limits = (
        [15.0] * N_NODES  # x_i
        + [15.0] * N_NODES  # y_i
        + [35.0] * N_NODES  # z_i
    )

    sampler = UniformRandomSampler(
        min_limits=min_limits,
        max_limits=max_limits,
        device=device,
    )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        t_eval=(950.0, 1000.0),
        device=device,
        rtol=1e-3,
        atol=1e-6,
        cache_dir=".pybasin_cache/rossler_network",
        event_fn=rossler_stop_event,
    )

    feature_extractor = SynchronizationFeatureExtractor(
        n_nodes=N_NODES,
        time_steady=950,
        device=device,
    )

    sync_classifier = SynchronizationClassifier(
        epsilon=1.5,
    )

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "estimator": sync_classifier,
    }
