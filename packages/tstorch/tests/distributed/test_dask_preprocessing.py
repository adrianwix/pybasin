"""Distributed preprocessing: NumPy-backed Dask Arrays through `tstorch.preprocessing`."""

import dask.array as da
import numpy as np
import torch
from tstorch.preprocessing import ExtremeImputer, StandardScaler


def _array(data: np.ndarray, chunks: int) -> da.Array:
    return da.from_array(data, chunks=(chunks, data.shape[1]))


class TestExtremeImputerDask:
    def test_matches_eager_result(self) -> None:
        data = np.array(
            [[1.0, np.nan], [np.inf, 2.0], [-np.inf, 3.0], [4.0, 5.0]], dtype=np.float32
        )
        eager_result = ExtremeImputer().transform(torch.from_numpy(data)).numpy()

        dask_result = ExtremeImputer().transform(_array(data, chunks=1)).compute()

        np.testing.assert_array_equal(dask_result, eager_result)

    def test_returns_dask_array(self) -> None:
        data = np.random.default_rng(0).standard_normal((10, 3)).astype(np.float32)
        result = ExtremeImputer().transform(_array(data, chunks=4))
        assert isinstance(result, da.Array)


class TestStandardScalerDask:
    def test_fit_matches_eager_population_variance(self) -> None:
        data = np.random.default_rng(0).standard_normal((100, 4)).astype(np.float64)

        eager_state = StandardScaler().fit(torch.from_numpy(data)).state
        dask_state = StandardScaler().fit(_array(data, chunks=17)).state

        np.testing.assert_allclose(dask_state.mean.numpy(), eager_state.mean.numpy(), rtol=1e-5)
        np.testing.assert_allclose(dask_state.scale.numpy(), eager_state.scale.numpy(), rtol=1e-5)

    def test_state_is_cpu_tensor_regardless_of_backend(self) -> None:
        data = np.random.default_rng(1).standard_normal((50, 2)).astype(np.float32)
        state = StandardScaler().fit(_array(data, chunks=10)).state

        assert isinstance(state.mean, torch.Tensor)
        assert state.mean.device == torch.device("cpu")
        assert state.n_samples_seen == 50

    def test_transform_matches_eager_result(self) -> None:
        data = np.random.default_rng(2).standard_normal((60, 3)).astype(np.float32)
        array = _array(data, chunks=13)

        scaler_dask = StandardScaler().fit(array)
        transformed_dask = scaler_dask.transform(array).compute()

        scaler_eager = StandardScaler().fit(torch.from_numpy(data))
        transformed_eager = scaler_eager.transform(torch.from_numpy(data)).numpy()

        np.testing.assert_allclose(transformed_dask, transformed_eager, rtol=1e-4, atol=1e-5)

    def test_different_chunkings_agree(self) -> None:
        data = np.random.default_rng(3).standard_normal((40, 2)).astype(np.float32)

        state_a = StandardScaler().fit(_array(data, chunks=40)).state
        state_b = StandardScaler().fit(_array(data, chunks=7)).state

        np.testing.assert_allclose(state_a.mean.numpy(), state_b.mean.numpy(), rtol=1e-5)
        np.testing.assert_allclose(state_a.scale.numpy(), state_b.scale.numpy(), rtol=1e-5)

    def test_extreme_imputation_then_scaling_stays_finite(self) -> None:
        data = np.array([[1.0], [np.nan], [np.inf], [-np.inf], [2.0], [3.0]], dtype=np.float32)
        array = _array(data, chunks=2)

        imputed = ExtremeImputer().transform(array)
        scaler = StandardScaler().fit(imputed)
        transformed = scaler.transform(imputed).compute()

        assert np.isfinite(transformed).all()
