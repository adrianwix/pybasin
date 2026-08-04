"""Tests for the independent `tstorch.preprocessing` module."""

import pytest
import torch
from tstorch.preprocessing import ExtremeImputer, StandardScaler


class TestExtremeImputer:
    def test_replaces_nan_and_inf(self) -> None:
        x = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
        result = ExtremeImputer().transform(x)

        assert torch.isfinite(result).all()
        assert result[0] == 1.0
        assert result[1] > 0  # NaN -> positive sentinel
        assert result[2] > 0  # +inf -> positive sentinel
        assert result[3] < 0  # -inf -> negative sentinel
        assert result[1] == result[2]
        assert result[3] == -result[2]

    def test_finite_values_untouched(self) -> None:
        x = torch.tensor([1.0, -2.5, 0.0, 3.14])
        result = ExtremeImputer().transform(x)
        assert torch.equal(result, x)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_sentinel_stays_finite_for_every_floating_dtype(self, dtype: torch.dtype) -> None:
        x = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=dtype)
        result = ExtremeImputer().transform(x)
        assert torch.isfinite(result).all()

    def test_rejects_non_floating_dtype(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        with pytest.raises(TypeError):
            ExtremeImputer().transform(x)

    def test_rejects_complex_dtype(self) -> None:
        x = torch.tensor([1 + 2j], dtype=torch.complex64)
        with pytest.raises(TypeError):
            ExtremeImputer().transform(x)


class TestStandardScaler:
    def test_state_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _ = StandardScaler().state

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StandardScaler().transform(torch.randn(5, 2))

    def test_fit_transform_zscores(self) -> None:
        x = torch.randn(200, 3)
        scaler = StandardScaler().fit(x)
        transformed = scaler.transform(x)

        assert torch.allclose(transformed.mean(dim=0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(transformed.std(dim=0, correction=0), torch.ones(3), atol=1e-5)

    def test_population_variance_matches_manual_ddof0(self) -> None:
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        scaler = StandardScaler().fit(x)
        expected_scale = x.std(dim=0, correction=0)
        assert torch.allclose(scaler.state.scale, expected_scale)

    def test_constant_column_gets_unit_scale(self) -> None:
        x = torch.ones(10, 1) * 5.0
        scaler = StandardScaler().fit(x)
        assert scaler.state.scale.item() == pytest.approx(1.0)

    def test_state_matches_cpu_fit_device(self) -> None:
        x = torch.randn(20, 2)
        scaler = StandardScaler().fit(x)
        assert scaler.state.mean.device == torch.device("cpu")
        assert scaler.state.scale.device == torch.device("cpu")
        assert scaler.state.n_samples_seen == 20

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_state_stays_on_cuda_when_fit_on_cuda(self) -> None:
        x = torch.randn(20, 2, device="cuda")
        scaler = StandardScaler().fit(x)

        assert scaler.state.mean.device.type == "cuda"
        assert scaler.state.scale.device.type == "cuda"

        transformed = scaler.transform(x)
        assert transformed.device.type == "cuda"
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(2, device="cuda"), atol=1e-4)
        assert torch.allclose(
            transformed.std(dim=0, correction=0), torch.ones(2, device="cuda"), atol=1e-4
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_promotes_arithmetic(self, dtype: torch.dtype) -> None:
        x = (torch.randn(50, 2) * 3).to(dtype)
        scaler = StandardScaler().fit(x)
        transformed = scaler.transform(x)
        assert torch.isfinite(transformed.float()).all()

    def test_extreme_imputation_then_scaling_stays_finite(self) -> None:
        x = torch.tensor([[1.0], [float("nan")], [float("inf")], [-float("inf")], [2.0]])
        imputed = ExtremeImputer().transform(x)
        scaler = StandardScaler().fit(imputed)
        transformed = scaler.transform(imputed)
        assert torch.isfinite(transformed).all()
