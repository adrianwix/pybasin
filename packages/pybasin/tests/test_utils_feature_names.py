"""Tests for feature name utility functions."""

from pybasin.feature_extractors.utils import (
    get_feature_indices_by_base_name,
    parse_feature_name,
    validate_feature_names,
)


class TestParseFeatureName:
    """Tests for parse_feature_name function."""

    def test_valid_feature_name(self) -> None:
        result = parse_feature_name("state_0__variance")
        assert result.state_index == 0
        assert result.base_name == "variance"
        assert result.params == {}

    def test_valid_feature_name_state_1(self) -> None:
        result = parse_feature_name("state_1__mean")
        assert result.state_index == 1
        assert result.base_name == "mean"

    def test_valid_feature_name_with_params(self) -> None:
        result = parse_feature_name("state_0__quantile__q_0.1")
        assert result.state_index == 0
        assert result.base_name == "quantile"
        assert result.params == {"q": "0.1"}

    def test_valid_feature_name_high_state_index(self) -> None:
        result = parse_feature_name("state_10__amplitude")
        assert result.state_index == 10
        assert result.base_name == "amplitude"

    def test_feature_name_no_state_prefix(self) -> None:
        result = parse_feature_name("variance")
        assert result.state_index is None
        assert result.base_name == "variance"

    def test_feature_name_with_params_no_state(self) -> None:
        result = parse_feature_name("quantile__q_0.1")
        assert result.state_index is None
        assert result.base_name == "quantile"
        assert result.params == {"q": "0.1"}


class TestValidateFeatureNames:
    """Tests for validate_feature_names function."""

    def test_all_valid_names(self) -> None:
        names = ["state_0__variance", "state_1__mean", "state_0__amplitude"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is True
        assert invalid_names == []

    def test_some_invalid_names(self) -> None:
        names = ["state_0__variance", "invalid_name", "state_1__mean"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is False
        assert invalid_names == ["invalid_name"]

    def test_all_invalid_names(self) -> None:
        names = ["bad_name", "another_bad", "wrong_format"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is False
        assert len(invalid_names) == 3

    def test_empty_list(self) -> None:
        all_valid, invalid_names = validate_feature_names([])
        assert all_valid is True
        assert invalid_names == []

    def test_complex_valid_names(self) -> None:
        names = [
            "state_0__variance",
            "state_0__linear_trend__attr_slope",
            "state_1__quantile__q_0.5",
        ]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is True
        assert invalid_names == []


class TestGetFeatureIndicesByBaseName:
    """Tests for get_feature_indices_by_base_name function."""

    def test_single_match(self) -> None:
        names = ["state_0__variance", "state_0__mean", "state_0__amplitude"]
        indices = get_feature_indices_by_base_name(names, "mean")
        assert indices == [1]

    def test_multiple_matches_across_states(self) -> None:
        names = [
            "state_0__variance",
            "state_1__variance",
            "state_0__mean",
            "state_1__mean",
        ]
        indices = get_feature_indices_by_base_name(names, "variance")
        assert indices == [0, 1]

    def test_feature_with_params(self) -> None:
        names = [
            "state_0__quantile__q_0.1",
            "state_0__quantile__q_0.5",
            "state_1__quantile__q_0.1",
        ]
        indices = get_feature_indices_by_base_name(names, "quantile")
        assert indices == [0, 1, 2]

    def test_no_matches(self) -> None:
        names = ["state_0__variance", "state_0__mean"]
        indices = get_feature_indices_by_base_name(names, "amplitude")
        assert indices == []

    def test_empty_feature_names(self) -> None:
        indices = get_feature_indices_by_base_name([], "variance")
        assert indices == []

    def test_with_invalid_names_in_list(self) -> None:
        names = ["state_0__variance", "invalid_name", "state_1__variance"]
        indices = get_feature_indices_by_base_name(names, "variance")
        assert indices == [0, 2]

    def test_exact_match_preferred(self) -> None:
        names = [
            "state_0__mean",
            "state_0__mean_absolute",
        ]
        indices = get_feature_indices_by_base_name(names, "mean")
        assert indices == [0]
