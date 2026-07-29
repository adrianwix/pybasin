import re
from dataclasses import dataclass


@dataclass
class ParsedFeatureName:
    """Parsed components of an encoded feature name.

    :ivar base_name: The base feature function name (e.g., ``mean``, ``quantile``).
    :ivar params: Dictionary of parameter names to string values. Empty dict if no params.
    :ivar state_index: The 0-based state variable index, or ``None`` if not encoded.
    """

    base_name: str
    params: dict[str, str]
    state_index: int | None


def to_snake_case(name: str) -> str:
    """Convert CamelCase or PascalCase to snake_case.

    ```python
    to_snake_case("FeatureExtractor")  # Returns: "feature_extractor"
    to_snake_case("TsFreshFeatureExtractor")  # Returns: "ts_fresh_feature_extractor"
    ```

    :param name: String in CamelCase or PascalCase format.
    :return: String converted to snake_case.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def format_feature_name(
    base_name: str,
    params: dict[str, object] | None = None,
    state_index: int | None = None,
) -> str:
    """Encode a feature name with optional parameters and state index.

    Creates a structured feature name string in the format used throughout pybasin:
    ``state_{idx}__{base_name}`` or ``state_{idx}__{base_name}__param_value``.

    ```python
    format_feature_name("mean")
    # Returns: "mean"

    format_feature_name("mean", state_index=0)
    # Returns: "state_0__mean"

    format_feature_name("quantile", {"q": 0.1}, state_index=1)
    # Returns: "state_1__quantile__q_0.1"
    ```

    :param base_name: The base feature function name (e.g., ``mean``, ``quantile``).
    :param params: Optional dictionary of parameter names to values. Values are
        converted to strings. Parameters are sorted alphabetically for consistency.
    :param state_index: Optional 0-based state variable index. If provided, prepended
        as ``state_{index}__``.
    :return: Encoded feature name string.
    """
    parts: list[str] = [base_name]

    if params:
        param_str = "__".join(f"{k}_{v}" for k, v in sorted(params.items()))
        parts.append(param_str)

    feature_part = "__".join(parts)

    if state_index is not None:
        return f"state_{state_index}__{feature_part}"

    return feature_part


def parse_feature_name(feature_name: str) -> ParsedFeatureName:
    """Decode an encoded feature name into its components.

    Parses a feature name string in format ``state_{idx}__{base_name}__params``
    to extract the state index, base name, and parameters.

    ```python
    parse_feature_name("state_1__quantile__q_0.1")
    # Returns: ParsedFeatureName(base_name="quantile", params={"q": "0.1"}, state_index=1)

    parse_feature_name("state_0__mean")
    # Returns: ParsedFeatureName(base_name="mean", params={}, state_index=0)

    parse_feature_name("mean")
    # Returns: ParsedFeatureName(base_name="mean", params={}, state_index=None)
    ```

    :param feature_name: Encoded feature name string.
    :return: ParsedFeatureName with base_name, params dict, and state_index.
    """
    state_index: int | None = None
    remaining = feature_name

    # Check for state_ prefix
    state_match = re.match(r"^state_(\d+)__(.+)$", feature_name)
    if state_match:
        state_index = int(state_match.group(1))
        remaining = state_match.group(2)

    parts = remaining.split("__")
    base_name = parts[0]
    params: dict[str, str] = {}

    for part in parts[1:]:
        if "_" in part:
            key, _, value = part.partition("_")
            params[key] = value

    return ParsedFeatureName(base_name=base_name, params=params, state_index=state_index)


def format_feature_for_display(
    feature_name: str,
    state_var_symbol: str = "y",
    use_latex: bool = True,
) -> str:
    """Format a feature name for human-readable display.

    Converts an encoded feature name to a display string suitable for plot
    labels and legends. State indices are converted to 1-based numbering
    for display.

    ```python
    format_feature_for_display("state_0__mean")
    # Returns: "$y_1$ mean"

    format_feature_for_display("state_1__quantile__q_0.1")
    # Returns: "$y_2$ quantile (q=0.1)"

    format_feature_for_display("state_0__mean", use_latex=False)
    # Returns: "y1 mean"
    ```

    :param feature_name: Encoded feature name string.
    :param state_var_symbol: Symbol to use for state variables. Default is ``y``.
    :param use_latex: Whether to wrap state variable in LaTeX math mode.
        Default is ``True``.
    :return: Human-readable display string.
    """
    parsed = parse_feature_name(feature_name)

    display_name = parsed.base_name.replace("_", " ")

    if parsed.params:
        param_strs = [f"{k}={v}" for k, v in sorted(parsed.params.items())]
        display_name = f"{display_name} ({', '.join(param_strs)})"

    if parsed.state_index is not None:
        display_idx = parsed.state_index + 1
        if use_latex:
            state_str = f"${state_var_symbol}_{{{display_idx}}}$"
        else:
            state_str = f"{state_var_symbol}{display_idx}"
        return f"{state_str} {display_name}"

    return display_name


def validate_feature_names(feature_names: list[str]) -> tuple[bool, list[str]]:
    """Validate that all feature names follow the naming convention.

    Feature names must follow: ``state_X__feature_name``
    where X is a non-negative integer.

    ```python
    validate_feature_names(["state_0__mean", "state_1__variance"])
    # Returns: (True, [])

    validate_feature_names(["state_0__mean", "invalid_name"])
    # Returns: (False, ["invalid_name"])
    ```

    :param feature_names: List of feature names to validate.
    :return: Tuple of (all_valid, invalid_names) where all_valid is True if all
        names are valid, and invalid_names is a list of names that don't match
        the convention.
    """
    invalid_names: list[str] = []
    for name in feature_names:
        parsed = parse_feature_name(name)
        if parsed.state_index is None:
            invalid_names.append(name)
    return (len(invalid_names) == 0, invalid_names)


def get_feature_indices_by_base_name(feature_names: list[str], base_feature: str) -> list[int]:
    """Get column indices for features matching a base feature name.

    Finds all features that have the given base name (after the state_X__ prefix).
    Useful for predictors that need to access specific features across multiple states.

    ```python
    feature_names = ["state_0__variance", "state_1__variance", "state_0__mean"]
    get_feature_indices_by_base_name(feature_names, "variance")
    # Returns: [0, 1]

    get_feature_indices_by_base_name(feature_names, "mean")
    # Returns: [2]
    ```

    :param feature_names: List of feature names.
    :param base_feature: Base feature name to search for (e.g., "variance", "mean").
    :return: List of column indices where the base feature appears.
    """
    indices: list[int] = []
    for idx, name in enumerate(feature_names):
        parsed = parse_feature_name(name)
        if parsed.state_index is not None:
            feature_id = parsed.base_name
            if parsed.params:
                param_str = "__".join(f"{k}_{v}" for k, v in sorted(parsed.params.items()))
                feature_id = f"{feature_id}__{param_str}"
            if feature_id == base_feature or feature_id.startswith(base_feature + "__"):
                indices.append(idx)
    return indices
