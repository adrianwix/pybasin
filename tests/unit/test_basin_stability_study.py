"""Tests for BasinStabilityStudy."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.study_params import (
    GridStudyParams,
    SweepStudyParams,
    ZipStudyParams,
)
from pybasin.types import StudyResult


@pytest.fixture
def mock_components() -> dict[str, MagicMock]:
    """Create mock components for BasinStabilityStudy."""
    ode_system = MagicMock()
    ode_system.params = {"T": 0.5, "K": 0.1}

    sampler = MagicMock()
    sampler.min_limits = [0.0, 0.0]
    sampler.max_limits = [1.0, 1.0]
    # Remove n_samples so hasattr() returns False (MagicMock returns True by default)
    del sampler.n_samples

    solver = MagicMock()
    feature_extractor = MagicMock()
    estimator = MagicMock()

    return {
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "estimator": estimator,
    }


@pytest.fixture
def mock_bse() -> MagicMock:
    """Create a mock BasinStabilityEstimator."""
    mock = MagicMock()
    mock.run.return_value = StudyResult(
        study_label={"baseline": True},
        basin_stability={"attractor_1": 0.6, "attractor_2": 0.4},
        errors={},
        n_samples=100,
        labels=None,
        orbit_data=None,
        initial_condition=np.zeros((100, 2)),
    )
    mock.solution = None
    mock.y0 = None
    mock.n = 100
    return mock


class TestBasinStabilityStudyWithSweep:
    """Tests for BasinStabilityStudy with SweepStudyParams."""

    def test_calls_bse_correct_number_of_times(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """A pure-param study creates one BSE and calls run_parameter_study once."""
        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2, 0.3]})
        mock_bse.run_parameter_study.return_value = [MagicMock()] * 3

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ) as mock_bse_class:
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            assert mock_bse_class.call_count == 1
            mock_bse.run_parameter_study.assert_called_once_with(study_params)

    def test_updates_ode_param_for_each_run(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """run_parameter_study receives the study_params containing all parameter values."""
        t_values = [0.1, 0.5, 0.9]
        study_params = SweepStudyParams(**{'ode_system.params["T"]': t_values})
        mock_bse.run_parameter_study.return_value = [MagicMock() for _ in t_values]

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            mock_bse.run_parameter_study.assert_called_once_with(study_params)

    def test_nested_param_path_is_evaluated_correctly(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Study labels use short names; run_parameter_study receives the complete study_params."""
        t_values = [0.15, 0.25]
        study_params = SweepStudyParams(**{'ode_system.params["T"]': t_values})
        mock_results = [
            StudyResult(
                study_label={"T": 0.15},
                basin_stability={},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
            StudyResult(
                study_label={"T": 0.25},
                basin_stability={},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
        ]

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            mock_bse.run_parameter_study.return_value = mock_results
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            mock_bse.run_parameter_study.assert_called_once_with(study_params)
            assert results[0]["study_label"] == {"T": 0.15}
            assert results[1]["study_label"] == {"T": 0.25}

    def test_results_contain_correct_study_labels(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Each result dict should contain study_label matching the parameter value."""
        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2]})
        mock_results = [
            StudyResult(
                study_label={"T": 0.1},
                basin_stability={"a": 0.6},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
            StudyResult(
                study_label={"T": 0.2},
                basin_stability={"a": 0.6},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
        ]

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            mock_bse.run_parameter_study.return_value = mock_results
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            assert [r["study_label"] for r in results] == [{"T": 0.1}, {"T": 0.2}]
            assert len(results) == 2


class TestBasinStabilityStudyWithGrid:
    """Tests for BasinStabilityStudy with GridStudyParams."""

    def test_calls_bse_for_cartesian_product(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """A pure-param grid study creates one BSE and calls run_parameter_study once."""
        study_params = GridStudyParams(
            **{
                'ode_system.params["T"]': [0.1, 0.2],
                'ode_system.params["K"]': [1.0, 2.0, 3.0],
            }
        )
        mock_bse.run_parameter_study.return_value = [MagicMock()] * 6

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ) as mock_bse_class:
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            assert mock_bse_class.call_count == 1
            mock_bse.run_parameter_study.assert_called_once_with(study_params)

    def test_grid_produces_correct_study_labels(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Grid should produce all combinations as study_labels."""
        study_params = GridStudyParams(
            **{
                'ode_system.params["T"]': [0.1, 0.2],
                'ode_system.params["K"]': [1.0, 2.0],
            }
        )
        expected_study_labels = [
            {"T": 0.1, "K": 1.0},
            {"T": 0.1, "K": 2.0},
            {"T": 0.2, "K": 1.0},
            {"T": 0.2, "K": 2.0},
        ]
        mock_results = [
            StudyResult(
                study_label=label,
                basin_stability={},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            )
            for label in expected_study_labels
        ]

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            mock_bse.run_parameter_study.return_value = mock_results
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            assert [r["study_label"] for r in results] == expected_study_labels


class TestBasinStabilityStudyWithZip:
    """Tests for BasinStabilityStudy with ZipStudyParams."""

    def test_calls_bse_for_zipped_params(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """A pure-param zip study creates one BSE and calls run_parameter_study once."""
        study_params = ZipStudyParams(
            **{
                'ode_system.params["T"]': [0.1, 0.2, 0.3],
                'ode_system.params["K"]': [1.0, 2.0, 3.0],
            }
        )
        mock_bse.run_parameter_study.return_value = [MagicMock()] * 3

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ) as mock_bse_class:
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            assert mock_bse_class.call_count == 1
            mock_bse.run_parameter_study.assert_called_once_with(study_params)

    def test_zip_produces_paired_study_labels(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Zip should pair parameters at same index."""
        study_params = ZipStudyParams(
            **{
                'ode_system.params["T"]': [0.1, 0.2],
                'ode_system.params["K"]': [1.0, 2.0],
            }
        )
        expected_study_labels = [
            {"T": 0.1, "K": 1.0},
            {"T": 0.2, "K": 2.0},
        ]
        mock_results = [
            StudyResult(
                study_label=label,
                basin_stability={},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            )
            for label in expected_study_labels
        ]

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            mock_bse.run_parameter_study.return_value = mock_results
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            assert [r["study_label"] for r in results] == expected_study_labels


class TestBasinStabilityStudyWithSampler:
    """Tests for varying sampler objects."""

    def test_passes_different_samplers_to_bse(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Each run should receive its own sampler."""
        sampler1 = MagicMock(name="sampler1")
        sampler2 = MagicMock(name="sampler2")
        sampler3 = MagicMock(name="sampler3")

        study_params = SweepStudyParams(sampler=[sampler1, sampler2, sampler3])

        captured_samplers: list[MagicMock] = []

        def capture_bse_args(**kwargs: Any) -> MagicMock:
            captured_samplers.append(kwargs["sampler"])
            return mock_bse

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            side_effect=capture_bse_args,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            assert captured_samplers == [sampler1, sampler2, sampler3]

    def test_zip_param_with_sampler(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """Zip should pair ODE param with corresponding sampler."""
        sampler1 = MagicMock(name="sampler_T0.1")
        sampler2 = MagicMock(name="sampler_T0.2")

        study_params = ZipStudyParams(
            **{
                'ode_system.params["T"]': [0.1, 0.2],
                "sampler": [sampler1, sampler2],
            }
        )

        captured_args: list[dict[str, Any]] = []

        def capture_bse_args(**kwargs: Any) -> MagicMock:
            captured_args.append(
                {"T": kwargs["ode_system"].params["T"], "sampler": kwargs["sampler"]}
            )
            return mock_bse

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            side_effect=capture_bse_args,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            # Verify study_labels have both parameters
            assert results[0]["study_label"] == {"T": 0.1, "sampler": sampler1}
            assert results[1]["study_label"] == {"T": 0.2, "sampler": sampler2}

            # Verify actual nested parameter AND sampler were updated correctly
            assert captured_args[0]["T"] == 0.1
            assert captured_args[0]["sampler"] is sampler1
            assert captured_args[1]["T"] == 0.2
            assert captured_args[1]["sampler"] is sampler2


class TestBasinStabilityStudyBSEArguments:
    """Tests that BSE is called with correct arguments."""

    def test_bse_receives_all_required_arguments(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """BSE should receive all required arguments from AS-BSE."""
        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.5]})

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ) as mock_bse_class:
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            call_kwargs = mock_bse_class.call_args.kwargs
            assert call_kwargs["n"] == 100
            assert call_kwargs["ode_system"] is mock_components["ode_system"]
            assert call_kwargs["solver"] is mock_components["solver"]
            assert call_kwargs["feature_extractor"] is mock_components["feature_extractor"]
            assert call_kwargs["predictor"] is mock_components["estimator"]
            assert call_kwargs["feature_selector"] is None

    def test_bse_run_called_for_each_run(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """run_parameter_study should be called once for a pure-param study."""
        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2, 0.3]})
        mock_bse.run_parameter_study.return_value = [MagicMock()] * 3

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            mock_bse.run_parameter_study.assert_called_once_with(study_params)

    def test_bse_get_errors_called_for_each_run(
        self, mock_components: dict[str, MagicMock], mock_bse: MagicMock
    ) -> None:
        """get_errors is no longer called directly by the study (it's inside run())."""
        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2]})

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            bs_study.run()

            assert mock_bse.get_errors.call_count == 0


class TestBasinStabilityStudyResults:
    """Tests for results storage."""

    def test_basin_stabilities_stored_correctly(
        self, mock_components: dict[str, MagicMock]
    ) -> None:
        """Basin stabilities from BSE should be stored."""
        mock_bse = MagicMock()
        mock_bse.run_parameter_study.return_value = [
            StudyResult(
                study_label={"T": 0.1},
                basin_stability={"a": 0.7, "b": 0.3},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
            StudyResult(
                study_label={"T": 0.2},
                basin_stability={"a": 0.5, "b": 0.5},
                errors={},
                n_samples=100,
                labels=None,
                orbit_data=None,
                initial_condition=np.zeros((100, 2)),
            ),
        ]

        study_params = SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2]})

        with patch(
            "pybasin.basin_stability_study.BasinStabilityEstimator",
            return_value=mock_bse,
        ):
            bs_study = BasinStabilityStudy(
                n=100,
                study_params=study_params,
                output_dir=None,
                **mock_components,
            )
            results = bs_study.run()

            assert results[0]["basin_stability"] == {"a": 0.7, "b": 0.3}
            assert results[1]["basin_stability"] == {"a": 0.5, "b": 0.5}
