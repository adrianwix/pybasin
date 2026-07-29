from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import torch


class Sampler(ABC):
    """Abstract base class for sampling initial conditions using PyTorch."""

    display_name: str = "Sampler"

    def __init__(self, min_limits: list[float], max_limits: list[float], device: str | None = None):
        """
        Initialize the sampler.

        :param min_limits: List of minimum values for each state.
        :param max_limits: List of maximum values for each state.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        assert len(min_limits) == len(max_limits), (
            "min_limits and max_limits must have the same length"
        )

        # Auto-detect device if not specified and normalize cuda to cuda:0
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Normalize "cuda" to "cuda:0" for consistency
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is None:  # type: ignore[comparison-overlap]
                self.device = torch.device("cuda:0")
            else:
                self.device = dev

        # Use float32 for GPU efficiency (5-10x faster than float64)
        self.min_limits = torch.tensor(min_limits, dtype=torch.float32, device=self.device)
        self.max_limits = torch.tensor(max_limits, dtype=torch.float32, device=self.device)
        self.state_dim = len(min_limits)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.state_dim})"

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n samples for the initial conditions.

        :param n: Number of samples.
        :return: Sampled initial conditions as a tensor of shape (n, state_dim).
        """
        pass


class UniformRandomSampler(Sampler):
    """Generates random samples using a uniform distribution within the specified range."""

    display_name: str = "Uniform Random Sampler"

    def sample(self, n: int) -> torch.Tensor:
        return (
            torch.rand(n, self.state_dim, device=self.device) * (self.max_limits - self.min_limits)
            + self.min_limits
        )


class GridSampler(Sampler):
    """Generates evenly spaced samples in a grid pattern within the specified range.

    Handles fixed dimensions (where min == max) by only distributing grid points
    along varying dimensions. For example, with limits [-10, 10], [-20, 20], [0, 0]
    and n=20000, the grid uses n^(1/2) ≈ 142 points per varying dimension (x, y)
    and a single point for the fixed dimension (z), yielding 142 x 142 x 1 = 20164
    unique samples instead of 28 x 28 x 28 = 21952 with many duplicates.
    """

    display_name: str = "Grid Sampler"

    def sample(self, n: int) -> torch.Tensor:
        varying_dims = [
            i for i in range(self.state_dim) if self.min_limits[i] != self.max_limits[i]
        ]
        n_varying = len(varying_dims)

        if n_varying == 0:
            return torch.stack([self.min_limits] * n, dim=0)

        n_per_dim = int(np.ceil(n ** (1 / n_varying)))

        grid_points: list[torch.Tensor] = []
        for i in range(self.state_dim):
            min_val = self.min_limits[i].item()
            max_val = self.max_limits[i].item()
            if i in varying_dims:
                grid_points.append(torch.linspace(min_val, max_val, n_per_dim, device=self.device))
            else:
                grid_points.append(torch.tensor([min_val], device=self.device))

        grid_matrices = torch.meshgrid(*grid_points, indexing="ij")
        points = torch.stack([grid.flatten() for grid in grid_matrices], dim=1)

        return points


class GaussianSampler(Sampler):
    """Generates samples using a Gaussian distribution around the midpoint."""

    display_name: str = "Gaussian Sampler"

    def __init__(
        self,
        min_limits: list[float],
        max_limits: list[float],
        std_factor: float = 0.2,
        device: str | None = None,
    ):
        super().__init__(min_limits, max_limits, device)
        self.std_factor = std_factor

    def sample(self, n: int) -> torch.Tensor:
        mean = (self.min_limits + self.max_limits) / 2
        std = self.std_factor * (self.max_limits - self.min_limits)

        samples = torch.normal(mean.repeat(n, 1), std.repeat(n, 1))

        return torch.clamp(samples, self.min_limits, self.max_limits)


class CsvSampler(Sampler):
    """Loads samples from a CSV file.

    This sampler reads initial conditions from a CSV file, useful for reproducing
    exact results from MATLAB or other reference implementations.
    """

    display_name: str = "CSV Sampler"

    def __init__(
        self,
        csv_path: str | Path,
        coordinate_columns: list[str],
        label_column: str | None = None,
        device: str | None = None,
    ):
        """Initialize the CSV sampler.

        :param csv_path: Path to the CSV file containing samples.
        :param coordinate_columns: List of column names to use as coordinates
            (e.g., ["x1", "x2"] or ["disp", "vel"]).
        :param label_column: Column name containing ground truth labels (e.g., "label").
            If None, no labels are loaded.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)  # pyright: ignore[reportUnknownMemberType]

        missing_cols = [col for col in coordinate_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Columns {missing_cols} not found in CSV. Available: {list(df.columns)}"
            )

        self._coordinate_columns = coordinate_columns
        self._label_column = label_column
        self._data: np.ndarray[tuple[int, int], np.dtype[np.float32]] = df[
            coordinate_columns
        ].values.astype(np.float32)  # type: ignore[reportUnknownMemberType]

        if label_column is not None:
            if label_column not in df.columns:
                raise ValueError(
                    f"Label column '{label_column}' not found in CSV. Available: {list(df.columns)}"
                )
            self._labels: np.ndarray | None = np.asarray(
                df[label_column].values  # type: ignore[reportUnknownMemberType]
            )
        else:
            self._labels = None

        min_limits_list: list[float] = list(self._data.min(axis=0).tolist())  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        max_limits_list: list[float] = list(self._data.max(axis=0).tolist())  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        super().__init__(min_limits_list, max_limits_list, device)

        self._tensor_data = torch.tensor(
            self._data,  # type: ignore[reportUnknownMemberType]
            dtype=torch.float32,
            device=self.device,
        )

    def __repr__(self) -> str:
        return f"CsvSampler({self.csv_path.name}, n={self.n_samples})"

    @property
    def labels(self) -> np.ndarray | None:
        """Return ground truth labels from the CSV file, or None if no label column was specified."""
        return self._labels

    @property
    def n_samples(self) -> int:
        """Return the total number of samples in the CSV file."""
        return len(self._data)

    def sample(self, n: int | None = None) -> torch.Tensor:
        """Return samples from the CSV file.

        :param n: Number of samples to return. If None, returns all samples.
            If n is larger than available samples, raises ValueError.
            If n is smaller than available samples, returns first n samples.
        :return: Sampled initial conditions as a tensor of shape (n, state_dim).
        """
        if n is None:
            return self._tensor_data

        if n > len(self._tensor_data):
            raise ValueError(
                f"Requested {n} samples, but CSV only contains {len(self._tensor_data)}"
            )

        return self._tensor_data[:n]
