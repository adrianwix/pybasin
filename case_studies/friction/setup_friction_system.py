import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.friction.friction_feature_extractor import FrictionFeatureExtractor
from case_studies.friction.friction_jax_ode import FrictionJaxODE, FrictionParams
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import SetupProperties


def setup_friction_system() -> SetupProperties:
    n = 5000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up friction system on device: {device}")

    params: FrictionParams = {
        "v_d": 1.5,  # Driving velocity
        "xi": 0.05,  # Damping ratio
        "musd": 2.0,  # Ratio static to dynamic friction coefficient
        "mud": 0.5,  # Dynamic coefficient of friction
        "muv": 0.0,  # Linear strengthening parameter
        "v0": 0.5,  # Reference velocity for exponential decay
    }

    ode_system = FrictionJaxODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-2.0, 0.0],
        max_limits=[2.0, 2.0],
        device=device,
    )

    solver = JaxSolver(
        t_span=(0, 500),
        t_steps=500,
        t_eval=(400.0, 500.0),
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/friction",
    )

    feature_extractor = FrictionFeatureExtractor(time_steady=400)

    classifier_initial_conditions = [
        [1.0, 1.0],
        [2.0, 2.0],
    ]

    classifier_labels = ["FP", "LC"]

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
