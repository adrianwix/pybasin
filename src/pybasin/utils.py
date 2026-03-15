import ast
import inspect
import os
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from json import JSONEncoder
from pathlib import Path
from textwrap import dedent
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import torch

from pybasin.constants import DEFAULT_STEADY_FRACTION
from pybasin.protocols import FeatureSelectorProtocol
from pybasin.solution import Solution
from pybasin.ts_torch.calculators.torch_features_pattern import extract_peak_values

P = ParamSpec("P")
R = TypeVar("R")


def set_seed(seed: int) -> None:
    """
    Set the random seed globally for reproducible experiments.

    Covers PyTorch (CPU and CUDA), NumPy, and Python's built-in random module,
    which together seed all stochastic components in the pipeline (sampling,
    feature extraction, HDBSCAN clustering, etc.).

    :param seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore[no-untyped-call]
    torch.cuda.manual_seed_all(seed)


class DisplayNameMixin:
    """Mixin that provides a computed display_name property from the class name."""

    @property
    def display_name(self) -> str:
        """Human-readable name derived from class name (e.g., 'TorchDiffEqSolver' -> 'Torch Diff Eq Solver')."""
        class_name = self.__class__.__name__
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
        spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
        return spaced


class AutoGetStrMixin:
    """Mixin that auto-generates a string representation from the ``ode()`` method source code.

    Requires the class to have ``ode`` (method) and ``params`` attributes.
    """

    params: Any

    def get_str(self) -> str:
        """Return a string representation of the ODE system for caching and display.

        By default, auto-generates from the ``ode()`` method source code
        (with docstrings stripped). Override this method to provide a custom representation.

        :return: A human-readable description of the ODE system and its parameters.
        """
        return self._auto_get_str()

    def _auto_get_str(self) -> str:
        """Auto-generate string representation from ``ode()`` method source code."""
        try:
            source = dedent(inspect.getsource(self.ode))  # type: ignore[attr-defined]
            tree = ast.parse(source)
            func_def = tree.body[0]
            if isinstance(func_def, ast.FunctionDef) and ast.get_docstring(func_def):
                func_def.body = func_def.body[1:]
            return f"{self.__class__.__name__}:\n{ast.unparse(tree)}"
        except (OSError, TypeError, SyntaxError):
            if isinstance(self.params, dict):
                params_dict = cast(dict[str, Any], self.params)  # pyright: ignore[reportUnknownMemberType]
                params_str = ", ".join(f"{k}={v}" for k, v in params_dict.items())
            else:
                params_str = ""
            return f"{self.__class__.__name__}({params_str})"


def time_execution(script_name: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    start_time = time.time()  # Record the start time
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Get the current time and date in a human-readable format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the elapsed time and current time to a file
    with open("execution_time.txt", "a") as f:
        f.write(f"{current_time} - {script_name}: {elapsed_time} seconds\n")

    return result


def generate_filename(name: str, file_extension: str):
    """
    Generates a unique filename using either a timestamp or a UUID.
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{date}_{name}.{file_extension}"


_PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")


def find_project_root(start: str | None = None) -> str:
    """
    Walk up from *start* (default: cwd) until a marker file is found.

    Markers checked: ``pyproject.toml``, ``.git``.

    :param start: Directory to start searching from. Defaults to ``os.getcwd()``.
    :return: Absolute path to the project root directory.
    :raises FileNotFoundError: If no marker is found before reaching the filesystem root.
    """
    current = os.path.abspath(start or os.getcwd())
    while True:
        if any(os.path.exists(os.path.join(current, m)) for m in _PROJECT_ROOT_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                f"Could not find project root (looked for {_PROJECT_ROOT_MARKERS})"
            )
        current = parent


def resolve_cache_dir(cache_dir: str) -> str:
    """
    Resolve a cache directory path and ensure it exists.

    Relative paths are resolved from the project root (found via marker-file detection).
    Absolute paths are used as-is.

    :param cache_dir: Relative or absolute path to the cache directory.
    :return: Absolute path to the cache directory (created if needed).
    """
    if os.path.isabs(cache_dir):
        full_path = cache_dir
    else:
        full_path = os.path.join(find_project_root(), cache_dir)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def _get_caller_dir() -> str:
    """
    Inspects the call stack to determine the directory of the calling script
    that is outside the pybasin module. This implementation iterates over the
    stack frames and returns the first frame whose __file__ is not within the
    pybasin package directory or the Python standard library.
    """
    # Get the absolute directory of the current (pybasin) module.
    library_dir = os.path.abspath(os.path.dirname(__file__))

    # Get standard library paths to exclude them
    stdlib_paths = {os.path.dirname(os.__file__)}
    if hasattr(sys, "base_prefix"):
        stdlib_paths.add(sys.base_prefix)
    if hasattr(sys, "prefix"):
        stdlib_paths.add(sys.prefix)

    for frame in inspect.stack():
        caller_file = frame.frame.f_globals.get("__file__")
        if caller_file:
            abs_caller_file = os.path.abspath(caller_file)
            if not abs_caller_file.startswith(library_dir):
                is_stdlib = any(
                    abs_caller_file.startswith(stdlib_path) for stdlib_path in stdlib_paths
                )
                if not is_stdlib:
                    return os.path.dirname(abs_caller_file)

    return os.getcwd()


def resolve_folder(output_dir: str | Path) -> Path:
    """
    Resolve an output directory path relative to the caller's directory and ensure it exists.

    :param output_dir: Directory path (relative to calling script, or absolute).
    :return: Resolved absolute path.
    """
    base_dir = Path(_get_caller_dir())
    full_folder = base_dir / output_dir
    full_folder.mkdir(parents=True, exist_ok=True)
    return full_folder


class NumpyEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:  # type: ignore[override]
        if isinstance(o, np.ndarray):
            return o.tolist()  # type: ignore[return-value]
        if isinstance(o, np.integer):
            return int(o)  # type: ignore[arg-type]
        if isinstance(o, np.floating):
            return float(o)  # type: ignore[arg-type]
        if isinstance(o, Solution):
            return {
                "initial_condition": o.initial_condition.tolist(),  # type: ignore[misc]
                "time": o.time.tolist(),  # type: ignore[misc]
                "y": o.y.tolist(),  # type: ignore[misc]
                "features": o.features.tolist() if o.features is not None else None,  # type: ignore[misc]
                "labels": o.labels.tolist() if o.labels is not None else None,  # type: ignore[misc]
            }
        return super().default(o)


def get_feature_names(selector: FeatureSelectorProtocol, original_names: list[str]) -> list[str]:
    """Get feature names after applying a sklearn selector/transformer.

    :param selector: Fitted feature selector satisfying :class:`FeatureSelectorProtocol`.
    :param original_names: List of original feature names before filtering.
    :return: List of feature names that passed the selector's filter.
    """
    mask = selector.get_support(indices=False)
    return [name for name, keep in zip(original_names, mask, strict=True) if keep]


@dataclass
class OrbitData:
    """Data structure for orbit diagram plotting.

    Stores peak amplitude information from steady-state trajectories,
    organized by degree of freedom (DOF). For period-N orbits, trajectories
    will have N distinct peak amplitude levels.

    :ivar peak_values: Peak amplitudes for each DOF. Shape (max_peaks, B, D) where
        D is number of DOFs. Padded with NaN for trajectories with fewer peaks.
    :ivar peak_counts: Number of peaks per trajectory per DOF. Shape (B, D).
    :ivar dof_indices: Which state indices were analyzed.
    :ivar time_steady: Time threshold used for steady-state filtering.
    """

    peak_values: torch.Tensor
    peak_counts: torch.Tensor
    dof_indices: list[int]
    time_steady: float


def extract_orbit_data(
    t: torch.Tensor,
    y: torch.Tensor,
    dof: list[int],
    time_steady: float | None = None,
    peak_support: int = 1,
) -> OrbitData:
    """Extract orbit data from trajectories for orbit diagram plotting.

        Analyzes steady-state portion of trajectories to find peak amplitudes.
        For period-N orbits, each trajectory will have N distinct peak amplitude
        levels per oscillation cycle.

        :param t: Time points tensor of shape (N,).
        :param y: Trajectory tensor of shape (N, B, S) where N=timesteps, B=batch, S=states.
        :param dof: List of state indices (degrees of freedom) to analyze.
        :param time_steady: Time threshold for steady-state. Points with t >= time_steady
            are considered steady-state. If None, uses 85% of the time span.
    :param peak_support: Support for peak detection (window size = 2*peak_support+1).
        :return: OrbitData containing peak amplitudes and counts per trajectory.
    """
    if time_steady is None:
        t_min = float(t[0].item())
        t_max = float(t[-1].item())
        time_steady = t_min + DEFAULT_STEADY_FRACTION * (t_max - t_min)

    steady_mask = t >= time_steady
    t_steady = t[steady_mask]
    y_steady = y[steady_mask]

    if len(t_steady) == 0:
        n_batch = y.shape[1]
        n_dof = len(dof)
        return OrbitData(
            peak_values=torch.zeros(0, n_batch, n_dof, dtype=y.dtype, device=y.device),
            peak_counts=torch.zeros(n_batch, n_dof, dtype=torch.long, device=y.device),
            dof_indices=dof,
            time_steady=time_steady,
        )

    y_dof = y_steady[:, :, dof]

    peak_values, peak_counts = extract_peak_values(y_dof, n=peak_support)

    return OrbitData(
        peak_values=peak_values,
        peak_counts=peak_counts.long(),
        dof_indices=dof,
        time_steady=time_steady,
    )
