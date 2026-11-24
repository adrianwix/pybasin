# pyright: basic
from typing import Any, TypedDict

import numpy as np
import pytest
import torch
from sklearn.feature_selection import VarianceThreshold  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_selector import DefaultFeatureSelector, SklearnFeatureSelectorAdapter
from pybasin.predictors.histogram_clusterer import HistogramClusterer
from pybasin.predictors.pairwise_comparison_clusterer import PairwiseComparisonClusterer
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


def test_basin_stability_unsupervised_pairwise_comparison_clusterer():
    """`BasinStabilityEstimator`'s `is_clusterer`/`fit_predict` dispatch (as opposed to the
    `is_classifier`/`predict` + `template_integrator` path exercised above) with
    `PairwiseComparisonClusterer` as the unsupervised predictor. No `template_integrator`
    is needed -- unlike a classifier, a clusterer has no template to fit against."""
    params: LinearParams = {"k": 0.0}
    ode_system = LinearODE(params)

    sampler = UniformRandomSampler(BoxRegion([-2.0], [2.0]), device="cpu")
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    feature_extractor = FinalStateExtractor(time_steady=0)

    clusterer = PairwiseComparisonClusterer(threshold=0.5, rescale_features=False)

    bse = BasinStabilityEstimator(
        n=100,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=clusterer,
        feature_selector=None,
        seed=42,
    )

    result = bse.run(parallel_integration=False)
    bs_vals = result["basin_stability"]

    assert len(bs_vals) >= 1
    assert sum(bs_vals.values()) == pytest.approx(1.0)
    # PairwiseComparisonClusterer never emits -1 (§2.1) -- every point joins some group.
    assert "-1" not in bs_vals and -1 not in bs_vals
    assert clusterer.n_groups_ == len(bs_vals)


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

    def test_sklearn_selector_adapter(self):
        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np)
        selector = SklearnFeatureSelectorAdapter(VarianceThreshold(threshold=0.01))
        bse = _make_bse(selector)

        filtered, filtered_names = bse._apply_feature_filtering(features, names)

        assert isinstance(filtered, torch.Tensor)
        assert filtered.device == features.device
        assert filtered.dtype == features.dtype
        assert len(filtered_names) == filtered.shape[1]
        assert filtered.shape[1] == 3, "zero-variance feature f0 dropped"

    def test_builtin_and_sklearn_selector_agree_on_variance_step(self):
        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np)

        # correlation_threshold=1.1 never removes anything (all |corr| <= 1), so this
        # isolates the DefaultFeatureSelector's variance step for comparison against
        # the legacy VarianceThreshold-only selector.
        bse_native = _make_bse(DefaultFeatureSelector(correlation_threshold=1.1, min_features=1))
        filtered_native, _ = bse_native._apply_feature_filtering(features, names)

        sklearn_selector = SklearnFeatureSelectorAdapter(VarianceThreshold(threshold=0.01))
        bse_sklearn = _make_bse(sklearn_selector)
        filtered_sklearn, _ = bse_sklearn._apply_feature_filtering(features, names)

        assert filtered_native.shape[1] == 3
        assert filtered_sklearn.shape[1] == 3

    def test_raw_sklearn_selector_is_rejected(self):
        with pytest.raises(TypeError, match="SklearnFeatureSelectorAdapter"):
            _make_bse(VarianceThreshold())

    def test_cuda_tensor_stays_on_device(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")

        X_np, names = _make_feature_data()
        features = torch.from_numpy(X_np).cuda()
        bse = _make_bse(DefaultFeatureSelector(correlation_threshold=0.95, min_features=2))

        filtered, _ = bse._apply_feature_filtering(features, names)

        assert isinstance(filtered, torch.Tensor)
        assert filtered.device.type == "cuda"
        assert filtered.dtype == torch.float32


class TestHistogramClustererFeatureSelectorEnforcement:
    """`HistogramClusterer`'s bin edges are fixed geometry over a specific feature-column
    order; a `feature_selector` is refit per parameter group and can select a
    different column subset/order each time, so `BasinStabilityEstimator` rejects
    that combination at construction time.
    """

    def _make_estimator(self, feature_selector: Any) -> BasinStabilityEstimator:
        params: LinearParams = {"k": -1.0}
        ode_system = LinearODE(params)
        sampler = UniformRandomSampler(BoxRegion([0.5], [2.0]), device="cpu")
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0]])
        return BasinStabilityEstimator(
            n=10,
            ode_system=ode_system,
            sampler=sampler,
            predictor=grouper,
            feature_selector=feature_selector,
        )

    def test_feature_selector_not_none_raises(self):
        with pytest.raises(ValueError, match="feature_selector=None"):
            self._make_estimator(DefaultFeatureSelector())

    def test_feature_selector_none_is_accepted(self):
        bse = self._make_estimator(None)
        assert isinstance(bse.predictor, HistogramClusterer)
        assert bse.feature_selector is None
