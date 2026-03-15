import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from case_studies.pendulum.pendulum_feature_extractor import PendulumFeatureExtractor
from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from pybasin.sampler import GridSampler
from pybasin.solvers import ScipyParallelSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import SetupProperties


def setup_pendulum_system_sklearn() -> SetupProperties:
    """
    Setup pendulum system using ScipyParallelSolver.

    This configuration uses the new sklearn-based parallel solver that
    leverages Python 3.14's free-threading capabilities for efficient
    parallel ODE solving.

    Returns:
        SetupProperties dictionary containing all system components
    """
    # Use same sample size as MATLAB for fair comparison
    n = 10000

    # ScipyParallelSolver runs on CPU with parallel processing
    device = "cpu"
    print(f"Setting up pendulum system with ScipyParallelSolver on device: {device}")
    print("Using multiprocessing backend with sklearn for parallel execution")

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

    # Create the sklearn parallel solver with specified integration time
    # n_jobs=-1 uses all available CPU cores
    # Python 3.14's free-threading allows true parallel execution without GIL
    solver = ScipyParallelSolver(
        t_span=(0, 1000),
        t_steps=25001,
        t_eval=(950.0, 1000.0),
        device=device,
        n_jobs=-1,  # Use all available CPUs
        rtol=1e-6,
        atol=1e-8,
    )

    # Instantiate the feature extractor with a steady state time.
    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # Create a KNeighborsClassifier with k=1.
    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=[
            [0.5, 0.0],  # FP: fixed point
            [2.7, 0.0],  # LC: limit cycle
        ],
        labels=["FP", "LC"],
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
