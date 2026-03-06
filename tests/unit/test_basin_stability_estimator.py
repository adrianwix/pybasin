from typing import TypedDict

import torch
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.sampler import UniformRandomSampler
from pybasin.solution import Solution
from pybasin.solvers import TorchOdeSolver
from pybasin.solvers.torch_ode_system import ODESystem
from pybasin.template_integrator import TemplateIntegrator


class LinearParams(TypedDict):
    k: float


class LinearODE(ODESystem[LinearParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.params["k"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['k']} * y"


class FinalStateExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        return solution.y[-1, :, :]

    @property
    def feature_names(self) -> list[str]:
        return ["final_state_0"]


def test_basin_stability_estimator_basic():
    params: LinearParams = {"k": -1.0}
    ode_system = LinearODE(params)

    sampler = UniformRandomSampler(min_limits=[0.5], max_limits=[2.0], device="cpu")
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)
    feature_extractor = FinalStateExtractor(time_steady=0)

    template_ics = [[1.0]]
    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=template_ics,
        labels=["stable"],
        ode_params=params,
    )

    bse = BasinStabilityEstimator(
        n=50,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=knn,
        template_integrator=template_integrator,
        feature_selector=None,
    )

    bs_vals = bse.estimate_bs(parallel_integration=False)

    # Basin stability values calculated
    assert bs_vals is not None
    # Expected label exists in results
    assert "stable" in bs_vals
    # All samples classified as "stable" (100% basin stability)
    assert bs_vals["stable"] == 1.0
    # Solution object created
    assert bse.solution is not None
    # Initial conditions stored
    assert bse.y0 is not None


def test_basin_stability_multiple_classes():
    params: LinearParams = {"k": 0.0}
    ode_system = LinearODE(params)

    sampler = UniformRandomSampler(min_limits=[-2.0], max_limits=[2.0], device="cpu")
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)
    feature_extractor = FinalStateExtractor(time_steady=0)

    template_ics = [[-1.0], [1.0]]
    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=template_ics,
        labels=["neg", "pos"],
        ode_params=params,
    )

    bse = BasinStabilityEstimator(
        n=100,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=knn,
        template_integrator=template_integrator,
        feature_selector=None,
    )

    bs_vals = bse.estimate_bs(parallel_integration=False)

    # Two classes found ("neg" and "pos")
    assert len(bs_vals) == 2
    # Basin stability fractions sum to 100%
    assert sum(bs_vals.values()) == 1.0
