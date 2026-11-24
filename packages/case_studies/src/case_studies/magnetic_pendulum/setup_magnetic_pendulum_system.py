import torch

from case_studies.magnetic_pendulum.magnetic_pendulum_feature_extractor import (
    MagneticPendulumFeatureExtractor,
)
from case_studies.magnetic_pendulum.magnetic_pendulum_ode import (
    MagneticPendulumODE,
    MagneticPendulumParams,
)
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler
from pybasin.solvers.torchode_solver import TorchOdeSolver
from pybasin.types import SetupProperties

PARAMS: MagneticPendulumParams = {"alpha": 0.2, "omega": 0.8, "d": 0.2, "gamma": 1.0}
TTR = 500.0


def setup_magnetic_pendulum_system(device: str | None = None) -> SetupProperties:
    """Shared magnetic pendulum setup (no predictor). `main_magnetic_pendulum_histogram_experiment.py`
    and `main_magnetic_pendulum_pairwise_experiment.py` build on this, each assigning its
    own unsupervised clusterer as `predictor`.

    :param device: Device for sampling and integration. Auto-detected when omitted.
    :return: Configuration shared by the magnetic pendulum experiments.
    """
    n = 1000

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting up magnetic pendulum system on device: {device}")

    ode_system = MagneticPendulumODE(PARAMS, n_magnets=3)

    # State is 4D (x, y, vx, vy); sample position in [-2, 2]^2 and pin velocity to
    # exactly 0 via zero-width bounds, matching Attractors.jl's
    # `ProjectedDynamicalSystem(ds, 1:2, [0.0, 0.0])`.
    sampler = UniformRandomSampler(
        BoxRegion([-2.0, -2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]), device=device
    )

    solver = TorchOdeSolver(
        t_span=(0, TTR),
        t_steps=2,
        device=device,
        cache_dir=".pybasin_cache/magnetic_pendulum",
    )

    feature_extractor = MagneticPendulumFeatureExtractor()

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
    }
