# pyright: basic
from typing import Any, TypedDict

import numpy as np
import pytest
import torch
from sklearn.feature_selection import VarianceThreshold  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_selector import DefaultFeatureSelector
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler
from pybasin.solution import Solution
from pybasin.solvers import TorchOdeSolver
from pybasin.solvers.torch_ode_system import ODESystem
from pybasin.template_integrator import TemplateIntegrator


class LinearParams(TypedDict):
    k: float


class LinearODE(ODESystem[LinearParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return p[..., 0] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['k']} * y"


class FinalStateExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        return solution.y[..., -1]

    @property
    def feature_names(self) -> list[str]:
        return ["final_state_0"]


def test_basin_stability_estimator_basic():
    params: LinearParams = {"k": -1.0}
    ode_system = LinearODE(params)

    sampler = UniformRandomSampler(BoxRegion([0.5], [2.0]), device="cpu")
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
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

    result = bse.run(parallel_integration=False)
    bs_vals = result["basin_stability"]

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

    sampler = UniformRandomSampler(BoxRegion([-2.0], [2.0]), device="cpu")
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
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

    result = bse.run(parallel_integration=False)
    bs_vals = result["basin_stability"]

    # Two classes found ("neg" and "pos")
    assert len(bs_vals) == 2
    # Basin stability fractions sum to 100%
    assert sum(bs_vals.values()) == 1.0


def test_basin_stability_estimator_seed_reproduces_y0():
    params: LinearParams = {"k": -1.0}
    ode_system = LinearODE(params)
    sampler = UniformRandomSampler(BoxRegion([0.5], [2.0]), device="cpu")
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    feature_extractor = FinalStateExtractor(time_steady=0)

    def make_estimator() -> BasinStabilityEstimator:
        return BasinStabilityEstimator(
            n=50,
            ode_system=ode_system,
            sampler=sampler,
            solver=solver,
            feature_extractor=feature_extractor,
            feature_selector=None,
            seed=42,
        )

    bse_a = make_estimator()
    bse_a.run(parallel_integration=False)
    bse_b = make_estimator()
    bse_b.run(parallel_integration=False)

    assert bse_a.y0 is not None
    assert bse_b.y0 is not None
    assert torch.equal(bse_a.y0, bse_b.y0)


def _make_bse(feature_selector: Any) -> BasinStabilityEstimator:
    """Build a BSE instance for exercising _apply_feature_filtering() directly.

    run() is not called; only the pieces needed to construct the estimator matter.
    """
    params: LinearParams = {"k": -1.0}
    ode_system = LinearODE(params)
    sampler = UniformRandomSampler(BoxRegion([0.5], [2.0]), device="cpu")
    return BasinStabilityEstimator(
        n=10,
        ode_system=ode_system,
        sampler=sampler,
        feature_selector=feature_selector,
    )


class _LegacyNumpyOnlySelector(VarianceThreshold):  # type: ignore[misc]
    """Stand-in for an arbitrary sklearn transformer with no `.backend` attribute."""


def _make_feature_data() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    n = 100
    base = rng.standard_normal(n)
    X = np.column_stack([
        np.zeros(n),  # zero variance -> dropped
        base,
        base + rng.standard_normal(n) * 0.01,  # correlated with previous
        rng.standard_normal(n),
    ]).astype(np.float32)
    names = ["f0", "f1", "f2", "f3"]
    return X, names


class TestApplyFeatureFiltering:
    def test_device_native_selector_keeps_cpu_tensor_untouched_type(self):
        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np)
        bse = _make_bse(DefaultFeatureSelector(correlation_threshold=0.95, min_features=2))

        filtered, filtered_names = bse._apply_feature_filtering(features, names)

        assert isinstance(filtered, torch.Tensor)
        assert filtered.device == features.device
        assert filtered.dtype == features.dtype
        assert len(filtered_names) == filtered.shape[1]

    def test_legacy_numpy_only_selector_still_works(self):
        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np)
        bse = _make_bse(_LegacyNumpyOnlySelector(threshold=0.01))

        filtered, filtered_names = bse._apply_feature_filtering(features, names)

        assert isinstance(filtered, torch.Tensor)
        assert filtered.device == features.device
        assert filtered.dtype == features.dtype
        assert len(filtered_names) == filtered.shape[1]
        assert filtered.shape[1] == 3, "zero-variance feature f0 dropped"

    def test_device_native_and_legacy_selector_agree_on_variance_step(self):
        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np)

        # correlation_threshold=1.1 never removes anything (all |corr| <= 1), so this
        # isolates the DefaultFeatureSelector's variance step for comparison against
        # the legacy VarianceThreshold-only selector.
        bse_native = _make_bse(DefaultFeatureSelector(correlation_threshold=1.1, min_features=1))
        filtered_native, _ = bse_native._apply_feature_filtering(features, names)

        bse_legacy = _make_bse(_LegacyNumpyOnlySelector(threshold=0.01))
        filtered_legacy, _ = bse_legacy._apply_feature_filtering(features, names)

        assert filtered_native.shape[1] == 3
        assert filtered_legacy.shape[1] == 3

    def test_cuda_tensor_stays_on_device(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        pytest.importorskip("cupy")

        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np).cuda()
        bse = _make_bse(DefaultFeatureSelector(correlation_threshold=0.95, min_features=2))

        filtered, _ = bse._apply_feature_filtering(features, names)

        assert isinstance(filtered, torch.Tensor)
        assert filtered.device.type == "cuda"
        assert filtered.dtype == torch.float32
