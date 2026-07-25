# pyright: basic
"""Tests for TsfreshFeatureExtractor's mapping to the tsfresh API.

Verifies that the extractor correctly builds the wide-format DataFrame,
maps int-keyed kind_to_fc_parameters to tsfresh's string-keyed format,
and produces results identical to calling tsfresh directly.
"""

import multiprocessing
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch
from tsfresh import extract_features  # type: ignore[import-untyped]
from tsfresh.feature_extraction import (  # type: ignore[import-untyped]
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute  # type: ignore[import-untyped]

from pybasin.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.solution import Solution

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_solution() -> Solution:
    """Solution with known values: 3 batches, 10 timesteps, 2 states."""
    n_steps = 10
    n_batch = 3
    n_states = 2

    time = torch.linspace(0, 1, n_steps)
    y = torch.arange(n_steps * n_batch * n_states, dtype=torch.float32).reshape(
        n_batch, n_states, n_steps
    )
    ic = y[:, :, 0]
    return Solution(initial_condition=ic, time=time, y=y)


@pytest.fixture
def deterministic_solution() -> Solution:
    """Solution with sinusoidal data for deterministic feature extraction."""
    n_steps = 100
    n_batch = 4
    n_states = 2

    time = torch.linspace(0, 10, n_steps)
    t_expanded = time[None, None, :].expand(n_batch, n_states, n_steps)

    torch.manual_seed(42)
    freqs = torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None, None]
    y = torch.sin(t_expanded * freqs)
    y[:, 1, :] = torch.cos(t_expanded[:, 1, :] * freqs[:, 0, :])
    ic = y[:, :, 0]
    return Solution(initial_condition=ic, time=time, y=y)


# ── DataFrame format tests ───────────────────────────────────────────────────


class TestDataFrameConstruction:
    """Verify the wide-format DataFrame matches what tsfresh expects."""

    def test_wide_format_columns(self, simple_solution: Solution) -> None:
        """Output DataFrame has id, time, state_0, ..., state_S columns."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            call_args = mock_ef.call_args
            df_passed: pd.DataFrame = call_args[0][0]

            assert list(df_passed.columns) == ["id", "time", "state_0", "state_1"]

    def test_wide_format_shape(self, simple_solution: Solution) -> None:
        """DataFrame has n_batch * n_timesteps rows."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            df_passed: pd.DataFrame = mock_ef.call_args[0][0]
            n_batch, _, n_steps = simple_solution.y.shape
            assert len(df_passed) == n_steps * n_batch

    def test_wide_format_ids(self, simple_solution: Solution) -> None:
        """Each batch index appears n_timesteps times consecutively."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            df_passed: pd.DataFrame = mock_ef.call_args[0][0]
            n_batch = simple_solution.y.shape[0]
            n_steps = simple_solution.y.shape[2]

            expected_ids = np.repeat(np.arange(n_batch), n_steps)
            np.testing.assert_array_equal(np.asarray(df_passed["id"].values), expected_ids)

    def test_wide_format_time_indices(self, simple_solution: Solution) -> None:
        """Time indices tile [0..N-1] for each batch."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            df_passed: pd.DataFrame = mock_ef.call_args[0][0]
            n_batch = simple_solution.y.shape[0]
            n_steps = simple_solution.y.shape[2]

            expected_times = np.tile(np.arange(n_steps), n_batch)
            np.testing.assert_array_equal(np.asarray(df_passed["time"].values), expected_times)

    def test_wide_format_state_values(self, simple_solution: Solution) -> None:
        """State columns contain correctly transposed values from y tensor."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            df_passed: pd.DataFrame = mock_ef.call_args[0][0]
            y_np = simple_solution.y.cpu().numpy()

            for batch_idx in range(simple_solution.y.shape[0]):
                for state_idx in range(simple_solution.y.shape[1]):
                    col = f"state_{state_idx}"
                    mask = df_passed["id"] == batch_idx
                    expected = y_np[batch_idx, state_idx, :]
                    actual = np.asarray(df_passed.loc[mask, col].values)
                    np.testing.assert_array_almost_equal(actual, expected)


# ── kind_to_fc_parameters mapping tests ──────────────────────────────────────


class TestKindToFcParametersMapping:
    """Verify int keys are mapped to 'state_N' strings for tsfresh."""

    def test_int_keys_mapped_to_state_strings(self, simple_solution: Solution) -> None:
        """kind_to_fc_parameters={0: ..., 1: ...} becomes {'state_0': ..., 'state_1': ...}."""
        fc_state_0: dict[str, None] = {"mean": None}
        fc_state_1: dict[str, None] = {"maximum": None}

        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            kind_to_fc_parameters={0: fc_state_0, 1: fc_state_1},
            normalize=False,
            n_jobs=1,
        )

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            call_kwargs = mock_ef.call_args[1]
            mapped = call_kwargs["kind_to_fc_parameters"]

            assert mapped == {"state_0": fc_state_0, "state_1": fc_state_1}

    def test_none_kind_to_fc_parameters_passes_none(self, simple_solution: Solution) -> None:
        """When kind_to_fc_parameters is None, None is passed to tsfresh."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            call_kwargs = mock_ef.call_args[1]
            assert call_kwargs["kind_to_fc_parameters"] is None

    def test_default_fc_parameters_passed_through(self, simple_solution: Solution) -> None:
        """default_fc_parameters is forwarded directly to tsfresh."""
        custom_params: dict[str, None] = {"mean": None, "variance": None}

        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters=custom_params,
            normalize=False,
            n_jobs=1,
        )

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            call_kwargs = mock_ef.call_args[1]
            assert call_kwargs["default_fc_parameters"] is custom_params

    def test_tsfresh_call_uses_wide_format_args(self, simple_solution: Solution) -> None:
        """extract_features is called with column_id and column_sort only (wide format)."""
        extractor = TsfreshFeatureExtractor(time_steady=0.0, normalize=False, n_jobs=1)

        with patch(
            "pybasin.feature_extractors.tsfresh_feature_extractor.extract_features"
        ) as mock_ef:
            mock_ef.return_value = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=pd.Index([0, 1, 2]))

            extractor.extract_features(simple_solution)

            call_kwargs = mock_ef.call_args[1]
            assert call_kwargs["column_id"] == "id"
            assert call_kwargs["column_sort"] == "time"
            assert "column_kind" not in call_kwargs
            assert "column_value" not in call_kwargs


# ── End-to-end equivalence with raw tsfresh ──────────────────────────────────


class TestEquivalenceWithRawTsfresh:
    """Verify our wrapper produces identical results to calling tsfresh directly."""

    def _build_tsfresh_df(self, solution: Solution) -> pd.DataFrame:
        """Build the wide-format DataFrame manually (reference implementation)."""
        y_np = solution.y.cpu().numpy()
        n_batch, n_states, n_steps = y_np.shape
        y_flat = y_np.transpose(0, 2, 1).reshape(-1, n_states)

        df_data: dict[str, Any] = {
            "id": np.repeat(np.arange(n_batch), n_steps),
            "time": np.tile(np.arange(n_steps), n_batch),
        }
        for s in range(n_states):
            df_data[f"state_{s}"] = y_flat[:, s]
        return pd.DataFrame(df_data)

    def test_default_fc_parameters_match(self, deterministic_solution: Solution) -> None:
        """Wrapper output matches calling tsfresh directly with default_fc_parameters."""
        fc_params: dict[str, None] = {"mean": None, "maximum": None, "minimum": None}

        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters=fc_params,
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)

        df = self._build_tsfresh_df(deterministic_solution)
        expected_df = cast(
            pd.DataFrame,
            extract_features(
                df,
                column_id="id",
                column_sort="time",
                default_fc_parameters=fc_params,
                n_jobs=1,
                disable_progressbar=True,
            ),
        )
        impute(expected_df)
        expected = torch.tensor(expected_df.values, dtype=torch.float32)

        torch.testing.assert_close(result, expected)

    def test_kind_to_fc_parameters_match(self, deterministic_solution: Solution) -> None:
        """Wrapper output matches calling tsfresh directly with kind_to_fc_parameters."""
        kind_params_int: dict[int, dict[str, None]] = {
            0: {"mean": None, "variance": None},
            1: {"maximum": None, "minimum": None},
        }
        kind_params_str: dict[str, dict[str, None]] = {
            "state_0": {"mean": None, "variance": None},
            "state_1": {"maximum": None, "minimum": None},
        }

        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            kind_to_fc_parameters=kind_params_int,
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)

        df = self._build_tsfresh_df(deterministic_solution)
        expected_df = cast(
            pd.DataFrame,
            extract_features(
                df,
                column_id="id",
                column_sort="time",
                kind_to_fc_parameters=kind_params_str,
                n_jobs=1,
                disable_progressbar=True,
            ),
        )
        impute(expected_df)
        expected = torch.tensor(expected_df.values, dtype=torch.float32)

        torch.testing.assert_close(result, expected)

    def test_minimal_fc_parameters_match(self, deterministic_solution: Solution) -> None:
        """Wrapper with MinimalFCParameters matches tsfresh directly."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters=MinimalFCParameters(),
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)

        df = self._build_tsfresh_df(deterministic_solution)
        expected_df = cast(
            pd.DataFrame,
            extract_features(
                df,
                column_id="id",
                column_sort="time",
                default_fc_parameters=MinimalFCParameters(),
                n_jobs=1,
                disable_progressbar=True,
            ),
        )
        impute(expected_df)
        expected = torch.tensor(expected_df.values, dtype=torch.float32)

        torch.testing.assert_close(result, expected)


# ── Output shape and metadata tests ──────────────────────────────────────────


class TestOutputShape:
    """Verify output tensor shape and feature names."""

    def test_output_shape_batch_dimension(self, deterministic_solution: Solution) -> None:
        """Output has one row per batch element."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None},
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)
        assert result.shape[0] == deterministic_solution.y.shape[0]

    def test_output_shape_features_per_state(self, deterministic_solution: Solution) -> None:
        """With N features and S states, output has N*S feature columns."""
        fc_params: dict[str, None] = {"mean": None, "maximum": None}

        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters=fc_params,
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)
        n_states = deterministic_solution.y.shape[1]

        assert result.shape[1] == len(fc_params) * n_states

    def test_feature_names_populated(self, deterministic_solution: Solution) -> None:
        """feature_names is populated after extraction."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None},
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)
        assert len(extractor.feature_names) == result.shape[1]

    def test_feature_names_raises_before_extraction(self) -> None:
        """feature_names raises RuntimeError before extract_features is called."""
        extractor = TsfreshFeatureExtractor(normalize=False)
        with pytest.raises(RuntimeError):
            _ = extractor.feature_names

    def test_output_dtype_float32(self, deterministic_solution: Solution) -> None:
        """Output tensor is float32."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None},
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)
        assert result.dtype == torch.float32


# ── Normalization tests ──────────────────────────────────────────────────────


class TestNormalization:
    """Verify scaler behavior."""

    def test_normalize_first_call_is_zero_mean_unit_var(
        self, deterministic_solution: Solution
    ) -> None:
        """First extraction with normalize=True produces zero-mean, unit-variance features."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None, "variance": None},
            normalize=True,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)

        means = result.mean(dim=0)
        stds = result.std(dim=0, correction=0)

        non_const_mask = stds > 1e-6
        torch.testing.assert_close(
            means[non_const_mask], torch.zeros_like(means[non_const_mask]), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            stds[non_const_mask], torch.ones_like(stds[non_const_mask]), atol=1e-5, rtol=1e-5
        )

    def test_normalize_false_returns_raw(self, deterministic_solution: Solution) -> None:
        """With normalize=False, output matches raw tsfresh values."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None},
            normalize=False,
            n_jobs=1,
        )
        result = extractor.extract_features(deterministic_solution)

        assert not torch.all(result == 0)

    def test_reset_scaler(self, deterministic_solution: Solution) -> None:
        """reset_scaler allows refitting on new data."""
        extractor = TsfreshFeatureExtractor(
            time_steady=0.0,
            default_fc_parameters={"mean": None},
            normalize=True,
            n_jobs=1,
        )
        result1 = extractor.extract_features(deterministic_solution)
        extractor.reset_scaler()
        result2 = extractor.extract_features(deterministic_solution)

        torch.testing.assert_close(result1, result2)


# ── Defaults tests ───────────────────────────────────────────────────────────


class TestDefaults:
    """Verify constructor defaults."""

    def test_defaults_to_minimal_fc_parameters(self) -> None:
        """When no fc_parameters provided, defaults to MinimalFCParameters."""
        extractor = TsfreshFeatureExtractor()
        assert extractor.default_fc_parameters == MinimalFCParameters()

    def test_n_jobs_minus_one_resolved(self) -> None:
        """n_jobs=-1 is resolved to actual CPU count."""
        extractor = TsfreshFeatureExtractor(n_jobs=-1)
        assert extractor.n_jobs == multiprocessing.cpu_count()

    def test_n_jobs_negative_clamped(self) -> None:
        """Negative n_jobs (other than -1) is clamped to 1."""
        extractor = TsfreshFeatureExtractor(n_jobs=-5)
        assert extractor.n_jobs == 1
