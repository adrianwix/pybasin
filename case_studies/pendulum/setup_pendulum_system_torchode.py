import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.pendulum.pendulum_feature_extractor import PendulumFeatureExtractor
from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from pybasin.sampler import GridSampler
from pybasin.solvers import TorchOdeSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import SetupProperties


def setup_pendulum_system_torchode() -> SetupProperties:
    """
    Setup pendulum system using TorchOdeSolver instead of TorchDiffEqSolver.

    This provides an alternative ODE solver based on torchode, which offers:
    - JIT compilation support for better performance
    - Batch parallelization
    - Multiple solver methods (dopri5, tsit5, etc.)
    """
    n = 10000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up pendulum system on device: {device}")

    # Define the parameters of the pendulum
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    # Instantiate ODE system for the pendulum.
    ode_system = PendulumODE(params)

    # Define sampling limits based on the pendulum parameters.
    # Here the angular limits for theta are adjusted using arcsin(T/K).
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )

    # Create the TorchOdeSolver with specified integration time and frequency.
    # Available methods: 'dopri5' (default), 'tsit5', 'euler', 'midpoint', 'heun'
    solver = TorchOdeSolver(
        t_span=(0, 1000),
        t_steps=1000,
        t_eval=(950.0, 1000.0),
        device=device,
        method="dopri5",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/pendulum",
    )

    # Instantiate the feature extractor with a steady state time.
    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Define template initial conditions and labels (e.g., for Fixed Point and Limit Cycle).
    classifier_initial_conditions = [
        [0.5, 0.0],  # FP: fixed point
        [2.7, 0.0],  # LC: limit cycle
    ]
    classifier_labels = ["FP", "LC"]

    # Create a KNeighborsClassifier with k=1.
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
