"""Tests for study_params module."""

import pytest

from pybasin.study_params import (
    CustomStudyParams,
    GridStudyParams,
    ParamAssignment,
    RunConfig,
    SweepStudyParams,
    ZipStudyParams,
    _extract_short_name,  # pyright: ignore[reportPrivateUsage]
)


class TestExtractShortName:
    """Tests for _extract_short_name helper function."""

    def test_extracts_from_dict_access(self) -> None:
        assert _extract_short_name('ode_system.params["T"]') == "T"
        assert _extract_short_name('ode_system.params["sigma"]') == "sigma"

    def test_extracts_from_attribute_access(self) -> None:
        assert _extract_short_name("solver.rtol") == "rtol"
        assert _extract_short_name("ode_system.params") == "params"

    def test_returns_name_for_simple_names(self) -> None:
        assert _extract_short_name("sampler") == "sampler"
        assert _extract_short_name("n") == "n"


class TestSweepStudyParams:
    """Tests for SweepStudyParams."""

    def test_iterates_over_values(self) -> None:
        params = SweepStudyParams(T=[0.1, 0.2, 0.3])
        configs = list(params)

        assert len(configs) == 3
        assert configs[0].assignments[0].name == "T"
        assert configs[0].assignments[0].value == 0.1
        assert configs[1].assignments[0].value == 0.2
        assert configs[2].assignments[0].value == 0.3

    def test_generates_correct_study_labels(self) -> None:
        params = SweepStudyParams(**{'ode_system.params["T"]': [0.5, 1.0]})
        configs = list(params)

        assert configs[0].study_label == {"T": 0.5}
        assert configs[1].study_label == {"T": 1.0}

    def test_len_returns_correct_count(self) -> None:
        params = SweepStudyParams(x=[1, 2, 3, 4, 5])
        assert len(params) == 5

    def test_handles_object_values(self) -> None:
        class MockSampler:
            def __init__(self, name: str) -> None:
                self.name = name

        samplers = [MockSampler("a"), MockSampler("b")]
        params = SweepStudyParams(sampler=samplers)
        configs = list(params)

        assert len(configs) == 2
        assert configs[0].assignments[0].value.name == "a"
        assert configs[1].assignments[0].value.name == "b"


class TestGridStudyParams:
    """Tests for GridStudyParams."""

    def test_creates_cartesian_product(self) -> None:
        params = GridStudyParams(T=[0.1, 0.2], sigma=[1.0, 2.0])
        configs = list(params)

        assert len(configs) == 4
        labels = [c.study_label for c in configs]
        assert {"T": 0.1, "sigma": 1.0} in labels
        assert {"T": 0.1, "sigma": 2.0} in labels
        assert {"T": 0.2, "sigma": 1.0} in labels
        assert {"T": 0.2, "sigma": 2.0} in labels

    def test_len_returns_product_of_lengths(self) -> None:
        params = GridStudyParams(a=[1, 2, 3], b=[4, 5])
        assert len(params) == 6

    def test_preserves_full_parameter_names(self) -> None:
        params = GridStudyParams(
            **{'ode_system.params["K"]': [0.1], 'ode_system.params["p"]': [0.5]}
        )
        configs = list(params)

        assignments = {a.name: a.value for a in configs[0].assignments}
        assert 'ode_system.params["K"]' in assignments
        assert 'ode_system.params["p"]' in assignments

    def test_single_parameter_equals_sweep(self) -> None:
        grid_params = GridStudyParams(T=[0.1, 0.2, 0.3])
        sweep_params = SweepStudyParams(T=[0.1, 0.2, 0.3])

        grid_configs = list(grid_params)
        sweep_configs = list(sweep_params)

        assert len(grid_configs) == len(sweep_configs)
        for gc, sc in zip(grid_configs, sweep_configs, strict=True):
            assert gc.study_label == sc.study_label


class TestZipStudyParams:
    """Tests for ZipStudyParams."""

    def test_zips_parameters_together(self) -> None:
        params = ZipStudyParams(T=[0.1, 0.2, 0.3], sigma=[1.0, 2.0, 3.0])
        configs = list(params)

        assert len(configs) == 3
        assert configs[0].study_label == {"T": 0.1, "sigma": 1.0}
        assert configs[1].study_label == {"T": 0.2, "sigma": 2.0}
        assert configs[2].study_label == {"T": 0.3, "sigma": 3.0}

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            ZipStudyParams(T=[0.1, 0.2], sigma=[1.0, 2.0, 3.0])

    def test_len_returns_correct_count(self) -> None:
        params = ZipStudyParams(a=[1, 2, 3], b=[4, 5, 6])
        assert len(params) == 3

    def test_handles_sampler_with_param(self) -> None:
        class MockSampler:
            def __init__(self, t: float) -> None:
                self.t = t

        t_values = [0.1, 0.2]
        samplers = [MockSampler(t) for t in t_values]

        params = ZipStudyParams(
            **{
                'ode_system.params["T"]': t_values,
                "sampler": samplers,
            }
        )
        configs = list(params)

        assert len(configs) == 2
        assert configs[0].study_label["T"] == 0.1
        assert configs[0].assignments[1].value.t == 0.1


class TestCustomStudyParams:
    """Tests for CustomStudyParams."""

    def test_yields_provided_configs(self) -> None:
        configs_input = [
            RunConfig(
                assignments=[ParamAssignment("T", 0.1)],
                study_label={"T": 0.1},
            ),
            RunConfig(
                assignments=[ParamAssignment("T", 0.2)],
                study_label={"T": 0.2},
            ),
        ]
        params = CustomStudyParams(configs_input)
        configs = list(params)

        assert len(configs) == 2
        assert configs[0].study_label == {"T": 0.1}
        assert configs[1].study_label == {"T": 0.2}

    def test_len_returns_config_count(self) -> None:
        configs_input = [RunConfig() for _ in range(5)]
        params = CustomStudyParams(configs_input)
        assert len(params) == 5

    def test_from_dicts_creates_configs(self) -> None:
        dicts = [
            {"T": 0.1, "n": 100},
            {"T": 0.2, "n": 200},
        ]
        params = CustomStudyParams.from_dicts(dicts)
        configs = list(params)

        assert len(configs) == 2
        assert configs[0].study_label == {"T": 0.1, "n": 100}
        assert configs[1].study_label == {"T": 0.2, "n": 200}

    def test_from_dicts_handles_full_param_names(self) -> None:
        dicts = [
            {'ode_system.params["K"]': 0.1, 'ode_system.params["p"]': 0.5},
        ]
        params = CustomStudyParams.from_dicts(dicts)
        configs = list(params)

        assert configs[0].study_label == {"K": 0.1, "p": 0.5}
        assert configs[0].assignments[0].name == 'ode_system.params["K"]'
