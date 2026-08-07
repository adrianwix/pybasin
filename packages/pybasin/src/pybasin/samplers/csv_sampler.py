from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import torch
from torch import Tensor

from pybasin.regions import BoxRegion, Region


class CsvSampler:
    """Loads samples from a CSV file.

    Reads initial conditions from a CSV file, useful for reproducing exact results from
    MATLAB or other reference implementations. The region is whatever the CSV's own data
    spans, not something the caller supplies: a ``BoxRegion`` fitted to each coordinate
    column's min/max.
    """

    is_deterministic = True

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

        resolved_device = (
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        min_limits: list[float] = list(self._data.min(axis=0).tolist())  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        max_limits: list[float] = list(self._data.max(axis=0).tolist())  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        self.region: Region = BoxRegion(
            torch.tensor(min_limits, dtype=torch.float32, device=resolved_device),
            torch.tensor(max_limits, dtype=torch.float32, device=resolved_device),
        )
        self.device: torch.device = self.region.device
        self.dtype: torch.dtype = self.region.dtype

        self._tensor_data = torch.tensor(
            self._data,  # type: ignore[reportUnknownMemberType]
            dtype=torch.float32,
            device=resolved_device,
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

    def sample(self, n: int | None = None, *, generator: torch.Generator | None = None) -> Tensor:
        """Return samples from the CSV file.

        :param n: Number of samples to return. If None, returns all samples.
            If n is larger than available samples, raises ValueError.
            If n is smaller than available samples, returns the first n samples.
        :param generator: Ignored -- CsvSampler is deterministic (always returns the
            CSV's own row order), accepted only for structural compatibility with the
            ``Sampler`` protocol.
        :return: Sampled initial conditions as a tensor of shape (n, state_dim).
        """
        if n is None:
            return self._tensor_data

        if n > len(self._tensor_data):
            raise ValueError(
                f"Requested {n} samples, but CSV only contains {len(self._tensor_data)}"
            )

        return self._tensor_data[:n]
