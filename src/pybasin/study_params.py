"""Study parameter generators for multi-parameter basin stability studies.

This module provides classes for defining parameter variations in basin stability
studies. The generators produce RunConfig objects that can be iterated by the
BasinStabilityStudy.

Classes:
    - SweepStudyParams: Single parameter sweep (1D study)
    - GridStudyParams: Cartesian product of multiple parameters
    - ZipStudyParams: Parallel iteration of multiple parameters
    - CustomStudyParams: User-defined list of configurations
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import Any


@dataclass
class ParamAssignment:
    """A single parameter assignment.

    :ivar name: The parameter path, e.g., 'ode_system.params["T"]' or 'sampler'.
    :ivar value: The value to assign to the parameter.
    """

    name: str
    value: Any


@dataclass
class RunConfig:
    """Configuration for a single BSE run with multiple parameter assignments.

    :ivar assignments: List of parameter assignments to apply for this run.
    :ivar study_label: Dictionary identifying this run's parameter combination,
        e.g., {"T": 0.5} or {"K": 0.1, "sigma": 0.3}.
    """

    assignments: list[ParamAssignment] = field(default_factory=lambda: [])
    study_label: dict[str, Any] = field(default_factory=lambda: {})


class StudyParams(ABC):
    """Base class for study parameter generators.

    Subclasses must implement __iter__ to yield RunConfig objects and __len__
    to return the total number of runs.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[RunConfig]:
        """Yield RunConfig for each parameter combination."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of runs."""
        raise NotImplementedError

    def to_list(self) -> list[RunConfig]:
        """Return all RunConfig objects as a list.

        Useful for inspecting the generated parameter combinations before running
        the study.

        :return: List of all RunConfig objects that would be yielded by iteration.
        """
        return list(self)


def _extract_short_name(name: str) -> str:
    """Extract short parameter name from a full path.

    :param name: Full parameter path, e.g., 'ode_system.params["T"]'.
    :return: Short name, e.g., 'T'.
    """
    if '["' in name:
        return name.split('["')[1].rstrip('"]')
    return name.split(".")[-1]


class SweepStudyParams(StudyParams):
    """Single parameter sweep.

    Iterates over a single parameter's values, yielding one RunConfig per value.
    Accepts exactly one keyword argument whose key is the parameter path and
    whose value is the list of values to sweep.

    Example:

    ```python
    study_params = SweepStudyParams(**{'ode_system.params["T"]': np.arange(0.01, 0.97, 0.05)})
    ```

    :ivar name: The parameter path to vary.
    :ivar values: List of values to sweep through.
    """

    def __init__(self, **params: list[Any]) -> None:
        """Initialize the sweep study parameters.

        :param params: Exactly one keyword argument mapping the parameter path
            to its list of values, e.g.,
            ``SweepStudyParams(**{'ode_system.params["T"]': t_values})``.
        :raises ValueError: If not exactly one keyword argument is provided.
        """
        if len(params) != 1:
            raise ValueError(f"SweepStudyParams takes exactly one parameter, got {len(params)}")
        self.name: str = next(iter(params))
        self.values: list[Any] = list(next(iter(params.values())))

    def __iter__(self) -> Iterator[RunConfig]:
        """Yield RunConfig for each parameter value."""
        short_name = _extract_short_name(self.name)
        for val in self.values:
            yield RunConfig(
                assignments=[ParamAssignment(self.name, val)],
                study_label={short_name: val},
            )

    def __len__(self) -> int:
        """Return the number of parameter values."""
        return len(self.values)


class GridStudyParams(StudyParams):
    """Cartesian product of multiple parameters.

    Creates all combinations of the provided parameter values (grid study).

    Example:

    ```python
    study_params = GridStudyParams(
        **{
            'ode_system.params["K"]': k_values,
            'ode_system.params["sigma"]': sigma_values,
        }
    )
    # Runs: K[0]×sigma[0], K[0]×sigma[1], ..., K[n]×sigma[m]
    ```
    """

    def __init__(self, **params: list[Any]) -> None:
        """Initialize the grid study parameters.

        :param params: Keyword arguments mapping parameter names to value arrays.
                       e.g., GridStudyParams(**{'ode_system.params["T"]': t_values})
        """
        self.param_names: list[str] = list(params.keys())
        self.param_values: list[list[Any]] = [list(v) for v in params.values()]

    def __iter__(self) -> Iterator[RunConfig]:
        """Yield RunConfig for each parameter combination (Cartesian product)."""
        short_names = [_extract_short_name(name) for name in self.param_names]
        for combo in product(*self.param_values):
            assignments = [
                ParamAssignment(name, val)
                for name, val in zip(self.param_names, combo, strict=True)
            ]
            study_label = dict(zip(short_names, combo, strict=True))
            yield RunConfig(assignments=assignments, study_label=study_label)

    def __len__(self) -> int:
        """Return the total number of combinations."""
        result = 1
        for vals in self.param_values:
            result *= len(vals)
        return result


class ZipStudyParams(StudyParams):
    """Parallel iteration of multiple parameters (must have same length).

    Iterates through parameters in parallel (like Python's zip), where values
    at the same index are used together.

    Example:

    ```python
    t_values = np.arange(0.01, 0.97, 0.05)
    samplers = [CsvSampler(f"gt_T_{t:.2f}.csv") for t in t_values]

    study_params = ZipStudyParams(
        **{
            'ode_system.params["T"]': t_values,
            "sampler": samplers,
        }
    )
    ```
    """

    def __init__(self, **params: list[Any]) -> None:
        """Initialize the zip study parameters.

        :param params: Keyword arguments mapping parameter names to value arrays.
                       All arrays must have the same length.
        :raises ValueError: If parameter arrays have different lengths.
        """
        self.param_names: list[str] = list(params.keys())
        self.param_values: list[list[Any]] = [list(v) for v in params.values()]

        lengths = [len(v) for v in self.param_values]
        if len(set(lengths)) > 1:
            raise ValueError(f"All parameter arrays must have same length, got {lengths}")

    def __iter__(self) -> Iterator[RunConfig]:
        """Yield RunConfig for each parameter tuple (parallel iteration)."""
        short_names = [_extract_short_name(name) for name in self.param_names]
        for combo in zip(*self.param_values, strict=True):
            assignments = [
                ParamAssignment(name, val)
                for name, val in zip(self.param_names, combo, strict=True)
            ]
            study_label = dict(zip(short_names, combo, strict=True))
            yield RunConfig(assignments=assignments, study_label=study_label)

    def __len__(self) -> int:
        """Return the number of parameter tuples."""
        return len(self.param_values[0]) if self.param_values else 0


class CustomStudyParams(StudyParams):
    """User-provided list of configurations.

    Allows full control over parameter combinations by providing explicit
    RunConfig objects.

    Example:

    ```python
    configs = [
        RunConfig(
            assignments=[
                ParamAssignment("ode_system", ode),
                ParamAssignment('ode_system.params["K"]', K),
            ],
            study_label={"K": K, "p": p},
        )
        for K, p in product(k_values, p_values)
    ]
    study_params = CustomStudyParams(configs)
    ```
    """

    def __init__(self, configs: list[RunConfig]) -> None:
        """Initialize with a list of RunConfig objects.

        :param configs: List of RunConfig objects defining each run.
        """
        self.configs = configs

    def __iter__(self) -> Iterator[RunConfig]:
        """Yield each RunConfig."""
        yield from self.configs

    def __len__(self) -> int:
        """Return the number of configurations."""
        return len(self.configs)

    @classmethod
    def from_dicts(cls, param_dicts: list[dict[str, Any]]) -> "CustomStudyParams":
        """Create from a list of {param_name: value} dictionaries.

        Example:

        ```python
        study_params = CustomStudyParams.from_dicts(
            [
                {'ode_system.params["K"]': 0.1, "n": 500},
                {'ode_system.params["K"]': 0.2, "n": 1000},
            ]
        )
        ```

        :param param_dicts: List of dictionaries where each dict maps parameter
                            names to values for one run.
        :return: CustomStudyParams instance.
        """
        configs: list[RunConfig] = []
        for d in param_dicts:
            assignments = [ParamAssignment(k, v) for k, v in d.items()]
            study_label = {_extract_short_name(k): v for k, v in d.items()}
            configs.append(RunConfig(assignments=assignments, study_label=study_label))
        return cls(configs)
