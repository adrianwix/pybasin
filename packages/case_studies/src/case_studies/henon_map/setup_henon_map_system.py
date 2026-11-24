import torch

from case_studies.henon_map.henon_feature_extractor import HenonMeanStateFeatureExtractor
from case_studies.henon_map.henon_map_ode import HenonMapODE, HenonMapParams
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler
from pybasin.solvers import DiscreteMapSolver
from pybasin.types import SetupProperties

N_STEPS = 200


def setup_henon_map_system(device: str | None = None) -> SetupProperties:
    """Shared Hénon map setup (no predictor): both `main_henon_histogram_experiment.py`
    and `main_henon_pairwise_experiment.py` build on this, each assigning its own
    unsupervised clusterer as `predictor`.

    :param device: Device for sampling and map iteration. Auto-detected when omitted.
    :return: Configuration shared by the Hénon map experiments.
    """
    n = 1000

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting up Henon map system on device: {device}")

    params: HenonMapParams = {"p1": 1.4, "p2": 0.3}
    ode_system = HenonMapODE(params)

    sampler = UniformRandomSampler(BoxRegion([-2.0, -2.0], [2.0, 2.0]), device=device)

    solver = DiscreteMapSolver(
        t_steps=N_STEPS,
        device=device,
        cache_dir=".pybasin_cache/henon_map",
    )

    feature_extractor = HenonMeanStateFeatureExtractor(time_steady=0.0)

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
    }
