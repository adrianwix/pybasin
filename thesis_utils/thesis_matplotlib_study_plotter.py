"""Parameter Study Basin Stability Plotter."""

import logging
from collections import defaultdict
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.utils import OrbitData, generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class MatplotlibStudyPlotter:
    """
    Matplotlib-based plotter for parameter study basin stability results.

    Supports multi-parameter studies by grouping results along one chosen parameter
    (x-axis) while producing separate curves for each combination of the remaining
    parameters. When ``parameters`` is not passed, one figure is produced per
    studied parameter.

    :ivar bs_study: BasinStabilityStudy instance with computed results.
    """

    def __init__(self, bs_study: BasinStabilityStudy):
        """
        Initialize the plotter with a BasinStabilityStudy instance.

        :param bs_study: An instance of BasinStabilityStudy.
        """
        self.bs_study = bs_study
        self._pending_figures: list[tuple[str, Figure]] = []

    def save(self, dpi: int = 300) -> None:
        """
        Save all pending figures to the output directory.

        Figures are tracked when plot methods create new figures. Call this
        after plotting to save all figures at once.

        :param dpi: Resolution for saved images.
        :raises ValueError: If ``bs_study.output_dir`` is not set or no figures pending.
        """
        if self.bs_study.output_dir is None:
            raise ValueError("bs_study.output_dir is not defined. Set it before calling save().")

        if not self._pending_figures:
            raise ValueError("No figures to save. Call a plot method first.")

        full_folder = resolve_folder(self.bs_study.output_dir)

        for name, fig in self._pending_figures:
            file_name = generate_filename(name, "png")
            full_path = full_folder / file_name
            logger.info("Saving plot to: %s", full_path)
            fig.savefig(full_path, dpi=dpi)  # type: ignore[misc]

        self._pending_figures.clear()

    def show(self) -> None:
        """
        Display all matplotlib figures.

        Convenience wrapper around ``plt.show()`` so users don't need to
        import matplotlib separately.
        """
        plt.show()  # type: ignore[misc]

    def _group_by_parameter(self, param_name: str) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group study result indices by the values of all parameters except ``param_name``.

        Within each group the indices are sorted by ``param_name``'s value so they
        can be plotted as a line.

        :param param_name: The parameter whose values form the x-axis.
        :return: Mapping from a tuple of (other_param, value) pairs to sorted result indices.
        """
        plottable = set(self._plottable_parameter_names())
        other_params: list[str] = [p for p in plottable if p != param_name]

        groups: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
        for i, r in enumerate(self.bs_study.results):
            sl = r["study_label"]
            group_key = tuple((p, sl[p]) for p in other_params) if other_params else ()
            groups[group_key].append(i)

        for group_key in groups:
            groups[group_key].sort(
                key=lambda i: self.bs_study.results[i]["study_label"][param_name]
            )

        return dict(groups)

    def _plottable_parameter_names(self) -> list[str]:
        """Return studied parameter names whose values are numeric (plottable).

        Parameters like ``sampler`` carry object values (e.g. ``CsvSampler``)
        that cannot be placed on a numeric axis, so they are excluded.

        :return: Filtered list of parameter names.
        """
        all_names = self.bs_study.studied_parameter_names
        if not self.bs_study.results:
            return all_names
        first_label = self.bs_study.results[0]["study_label"]
        return [
            n
            for n in all_names
            if isinstance(first_label[n], (int, float, np.floating, np.integer))
        ]

    def _resolve_parameters(self, parameters: list[str] | None) -> list[str]:
        """Resolve which parameters to iterate over.

        :param parameters: Explicit list or None for all plottable parameters.
        :return: List of parameter names.
        :raises ValueError: If a name is not among the studied parameters.
        """
        all_names = self.bs_study.studied_parameter_names
        if parameters is None:
            return self._plottable_parameter_names()
        for p in parameters:
            if p not in all_names:
                raise ValueError(f"Parameter '{p}' not found. Studied parameters: {all_names}")
        return parameters

    def _get_attractor_labels(self) -> list[str]:
        """Collect all unique attractor labels across every run, sorted."""
        labels_set: set[str] = set()
        for r in self.bs_study.results:
            labels_set.update(r["basin_stability"].keys())
        return sorted(labels_set)

    @staticmethod
    def _generate_colors(n: int) -> list[Any]:
        """Generate ``n`` visually distinct colors from a matplotlib colormap.

        Uses ``tab10`` for up to 10 colors, ``tab20`` for up to 20, and
        ``hsv`` for larger counts so that colors never repeat within a figure.

        :param n: Number of distinct colors required.
        :return: List of RGBA color tuples.
        """
        if n <= 1:
            return [plt.cm.tab10(0)]  # type: ignore[misc]
        if n <= 10:
            cmap = plt.cm.tab10  # type: ignore[misc]
        elif n <= 20:
            cmap = plt.cm.tab20  # type: ignore[misc]
        else:
            cmap = plt.cm.hsv  # type: ignore[misc]
        return [cmap(i / n) for i in range(n)]

    def plot_parameter_stability(
        self,
        interval: Literal["linear", "log"] = "linear",
        parameters: list[str] | None = None,
    ) -> list[Figure]:
        """Plot basin stability values against parameter variation.

        Produces one figure per parameter. Colors represent attractors so that
        downstream ``recolor_figure`` calls reassign thesis-palette colors
        correctly. For multi-parameter studies, line style and marker vary per
        group (combination of the other parameters).

        :param interval: x-axis scale — ``'linear'`` or ``'log'``.
        :param parameters: Which studied parameters to plot. ``None`` plots all.
        :return: List of matplotlib Figure objects (one per parameter).
        """
        if not self.bs_study.results:
            raise ValueError("No results available. Run study first.")

        params_to_plot = self._resolve_parameters(parameters)
        attractor_labels = self._get_attractor_labels()

        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
        linestyles = ["-", "--", ":", "-."]
        all_groups = {p: self._group_by_parameter(p) for p in params_to_plot}
        attractor_colors = self._generate_colors(len(attractor_labels))
        figures: list[Figure] = []

        for param_name in params_to_plot:
            groups = all_groups[param_name]
            n_groups = len(groups)
            single_group = n_groups <= 1

            fig = plt.figure(figsize=(10, 6))  # type: ignore[misc]

            if interval == "log":
                plt.xscale("log")  # type: ignore[misc]

            for g_idx, (_group_key, indices) in enumerate(groups.items()):
                x_values = [self.bs_study.results[i]["study_label"][param_name] for i in indices]

                for a_idx, attractor in enumerate(attractor_labels):
                    y_values = [
                        self.bs_study.results[i]["basin_stability"].get(attractor, 0)
                        for i in indices
                    ]
                    plt.plot(  # type: ignore[misc]
                        x_values,
                        y_values,
                        marker=markers[g_idx % len(markers)],
                        linestyle=linestyles[g_idx % len(linestyles)],
                        color=attractor_colors[a_idx],
                        markersize=4,
                        linewidth=1.0,
                        alpha=0.8,
                    )

            attractor_handles = [
                Line2D(
                    [0],
                    [0],
                    color=attractor_colors[a_idx],
                    linewidth=1.0,
                    linestyle="-",
                    marker="o",
                    markersize=4,
                    label=attractor,
                )
                for a_idx, attractor in enumerate(attractor_labels)
            ]

            if single_group:
                plt.legend(  # type: ignore[misc]
                    handles=attractor_handles,
                    loc="best",
                )
            else:
                group_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=1.0,
                        linestyle=linestyles[g_idx % len(linestyles)],
                        marker=markers[g_idx % len(markers)],
                        markersize=4,
                        markevery=[0],
                        label=", ".join(f"{k}={v}" for k, v in group_key),
                    )
                    for g_idx, group_key in enumerate(groups.keys())
                ]
                legend1 = plt.legend(  # type: ignore[misc]
                    handles=attractor_handles,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
                plt.gca().add_artist(legend1)  # type: ignore[misc]
                plt.legend(  # type: ignore[misc]
                    handles=group_handles,
                    title="Parameters",
                    bbox_to_anchor=(1.02, 0.5),
                    loc="upper left",
                    handlelength=4,
                )

            plt.xlabel(param_name)  # type: ignore[misc]
            plt.ylabel("Basin Stability")  # type: ignore[misc]
            plt.title(f"Parameter Stability ({param_name})")  # type: ignore[misc]
            plt.grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]
            plt.tight_layout()

            self._pending_figures.append((f"parameter_stability_{param_name}", fig))
            figures.append(fig)

        return figures

    def plot_orbit_diagram(
        self,
        dof: list[int] | None = None,
        interval: Literal["linear", "log"] = "linear",
        parameters: list[str] | None = None,
    ) -> list[Figure]:
        """Plot orbit diagrams showing attractor amplitude levels over parameter variation.

        Produces one figure per parameter. For each attractor at each parameter value,
        displays the peak amplitudes detected from steady-state trajectories.
        Period-N orbits appear as N distinct amplitude bands.

        Requires ``compute_orbit_data`` when creating the BasinStabilityStudy.

        :param interval: x-axis scale — ``'linear'`` or ``'log'``.
        :param dof: List of state indices to plot. If None, uses the DOFs from orbit_data.
        :param parameters: Which studied parameters to plot. ``None`` plots all.
        :return: List of matplotlib Figure objects (one per parameter).
        :raises ValueError: If orbit_data was not computed during the study.
        """
        if not self.bs_study.results:
            raise ValueError("No results available. Run study first.")

        first_orbit_data = self.bs_study.results[0].get("orbit_data")
        if first_orbit_data is None:
            raise ValueError(
                "No orbit data available. Set compute_orbit_data=True when creating "
                "BasinStabilityStudy to enable orbit diagram plotting."
            )

        if dof is None:
            dof = first_orbit_data.dof_indices

        params_to_plot = self._resolve_parameters(parameters)
        attractor_labels = self._get_attractor_labels()
        n_dofs = len(dof)

        colors = self._generate_colors(len(attractor_labels))
        figures: list[Figure] = []

        for param_name in params_to_plot:
            groups = self._group_by_parameter(param_name)

            fig, axes = plt.subplots(1, n_dofs, figsize=(6 * n_dofs, 5))  # type: ignore[misc]
            if n_dofs == 1:
                axes = [axes]

            scatter_data: dict[int, dict[int, tuple[list[float], list[float]]]] = {
                j: {a_idx: ([], []) for a_idx in range(len(attractor_labels))}
                for j in range(n_dofs)
            }

            for _, indices in groups.items():
                x_values = [self.bs_study.results[i]["study_label"][param_name] for i in indices]

                for a_idx, attractor in enumerate(attractor_labels):
                    for pos, result_idx in enumerate(indices):
                        result = self.bs_study.results[result_idx]
                        orbit_data: OrbitData | None = result.get("orbit_data")
                        labels = result.get("labels")

                        if orbit_data is None or labels is None:
                            continue

                        attractor_mask = labels == attractor
                        if not np.any(attractor_mask):
                            continue

                        x_val = x_values[pos]

                        for j, dof_idx in enumerate(dof):
                            try:
                                dof_pos = orbit_data.dof_indices.index(dof_idx)
                            except ValueError:
                                continue

                            peaks_all = orbit_data.peak_values[:, attractor_mask, dof_pos]
                            peaks_flat = peaks_all.cpu().numpy().flatten()
                            valid_peaks = peaks_flat[np.isfinite(peaks_flat)]

                            if len(valid_peaks) == 0:
                                continue

                            xs, ys = scatter_data[j][a_idx]
                            xs.extend([x_val] * len(valid_peaks))
                            ys.extend(valid_peaks.tolist())

            for j, dof_idx in enumerate(dof):
                ax = axes[j]
                for a_idx, attractor in enumerate(attractor_labels):
                    xs, ys = scatter_data[j][a_idx]
                    if xs:
                        ax.scatter(
                            xs,
                            ys,
                            c=[colors[a_idx]],
                            s=3,
                            alpha=0.5,
                            label=attractor,
                            rasterized=True,
                        )  # type: ignore[misc]

                if interval == "log":
                    ax.set_xscale("log")  # type: ignore[misc]
                ax.set_xlabel(param_name)  # type: ignore[misc]
                ax.set_ylabel(f"peak $y_{{{dof_idx}}}$")  # type: ignore[misc]
                ax.grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]

                handles: list[Any] = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=colors[a_idx],
                        markersize=4,
                        label=attractor,
                    )
                    for a_idx, attractor in enumerate(attractor_labels)
                    if scatter_data[j][a_idx][0]
                ]
                if handles:
                    ax.legend(handles=handles)  # type: ignore[misc]

            plt.suptitle(f"Orbit Diagram ({param_name})")  # type: ignore[misc]
            plt.tight_layout()

            self._pending_figures.append((f"orbit_diagram_{param_name}", fig))
            figures.append(fig)

        return figures
