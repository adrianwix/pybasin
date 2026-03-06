import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE, DuffingParams
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import SetupProperties


def setup_duffing_oscillator_system() -> SetupProperties:
    n = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Duffing oscillator system on device: {device}")

    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    ode_system = DuffingJaxODE(params)

    sampler = UniformRandomSampler(min_limits=[-1, -0.5], max_limits=[1, 1], device=device)

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=5000,
        t_eval=(900.0, 1000.0),
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=".pybasin_cache/duffing",
    )

    feature_extractor = TorchFeatureExtractor(
        time_steady=900.0,
        normalize=False,
        features=None,
        features_per_state={
            0: {"maximum": None, "standard_deviation": None},
        },
    )

    classifier_initial_conditions = [
        [-0.21, 0.02],
        [1.05, 0.77],
        [-0.67, 0.02],
        [-0.46, 0.30],
        [-0.43, 0.12],
    ]

    classifier_labels = [
        "$\\bar{y}_1$",
        "$\\bar{y}_2$",
        "$\\bar{y}_3$",
        "$\\bar{y}_4$",
        "$\\bar{y}_5$",
    ]

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
