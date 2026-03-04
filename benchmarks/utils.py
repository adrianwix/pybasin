# pyright: basic
"""Shared utilities for benchmark comparison scripts."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats as scipy_stats

from thesis_utils.docs_plots_utils import thesis_export as docs_export
from thesis_utils.thesis_plots_utils import (
    THESIS_FULL_WIDTH_CM,
    THESIS_SINGLE_HEIGHT_CM,
    thesis_export,
)


def load_json(path: Path) -> dict:
    """Load and return a JSON file as a dict.

    :param path: Path to the JSON file.
    :return: Parsed JSON content.
    """
    with open(path) as f:
        return json.load(f)


def dual_export_figure(
    build_fn: Callable[[], Figure],
    *,
    docs_name: str,
    docs_dir: Path,
    thesis_name: str,
    thesis_dir: Path,
    thesis_width: float = THESIS_FULL_WIDTH_CM,
    thesis_height: float = THESIS_SINGLE_HEIGHT_CM,
) -> None:
    """Build a figure twice and export as PNG (docs) and PDF (thesis).

    :param build_fn: Callable that returns a new matplotlib Figure.
    :param docs_name: Filename for the docs PNG export.
    :param docs_dir: Directory for docs assets.
    :param thesis_name: Filename for the thesis PDF export.
    :param thesis_dir: Directory for thesis artifacts.
    :param thesis_width: Thesis figure width in cm.
    :param thesis_height: Thesis figure height in cm.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir.mkdir(parents=True, exist_ok=True)

    fig_docs = build_fn()
    docs_export(fig_docs, docs_name, docs_dir)
    plt.close(fig_docs)

    fig_thesis = build_fn()
    thesis_export(
        fig_thesis,
        thesis_name,
        thesis_dir,
        width=thesis_width,
        height=thesis_height,
    )
    plt.close(fig_thesis)


def build_scaling_figure(
    df: pd.DataFrame,
    *,
    label_column: str,
    colors: dict[str, str],
    markers: dict[str, str],
    order: list[str],
    title: str,
) -> Figure:
    """Build a log-log scaling plot with linear-regression fit lines.

    :param df: DataFrame with columns ``label_column``, ``N``, and ``mean_time``.
    :param label_column: Name of the column used to group/label series.
    :param colors: Mapping from label to colour hex string.
    :param markers: Mapping from label to matplotlib marker.
    :param order: Desired legend order of labels.
    :param title: Plot title.
    :return: The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    labels: list[str] = [lbl for lbl in order if lbl in df[label_column].unique()]

    for label in labels:
        sub = cast(pd.DataFrame, df[df[label_column] == label]).sort_values(by="N")
        if len(sub) < 3:
            continue

        n_vals: np.ndarray = np.asarray(sub["N"].values, dtype=float)
        t_vals: np.ndarray = np.asarray(sub["mean_time"].values, dtype=float)

        color = colors.get(label, "#333333")
        marker = markers.get(label, "o")

        ax.plot(n_vals, t_vals, marker, color=color, label=label, markersize=8)

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        result = scipy_stats.linregress(log_n, log_t)
        slope: float = result.slope
        intercept: float = result.intercept

        n_fit = np.logspace(np.log10(n_vals.min()), np.log10(n_vals.max()), 100)
        t_fit: np.ndarray = np.exp(intercept) * n_fit**slope
        ax.plot(n_fit, t_fit, "--", color=color, alpha=0.7, label="_nolegend_")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    return fig
