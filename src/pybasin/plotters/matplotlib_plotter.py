# pyright: basic

import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.utils import format_feature_for_display
from pybasin.plotters.colors import get_color
from pybasin.utils import generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class MatplotlibPlotter:
    def __init__(self, bse: BasinStabilityEstimator):
        """
        Initialize the Plotter with a BasinStabilityEstimator instance.

        :param bse: An instance of BasinStabilityEstimator.
        """
        self.bse = bse
        self._pending_figures: list[tuple[str, Figure]] = []

    def save(self, dpi: int = 300) -> None:
        """
        Save all pending figures to the output directory.

        Figures are tracked when plot methods create new figures (i.e., when
        no ``ax`` parameter is passed). Call this after plotting to save
        all figures at once.

        :param dpi: Resolution for saved images.
        :raises ValueError: If ``bse.output_dir`` is not set or no figures pending.
        """
        if self.bse.output_dir is None:
            raise ValueError("bse.output_dir is not defined. Set it before calling save().")

        if not self._pending_figures:
            raise ValueError("No figures to save. Call a plot method first.")

        full_folder = resolve_folder(self.bse.output_dir)

        for name, fig in self._pending_figures:
            file_name = generate_filename(name, "png")
            full_path = full_folder / file_name
            logger.info("Saving plot to: %s", full_path)
            fig.savefig(full_path, dpi=dpi)

        self._pending_figures.clear()

    def show(self) -> None:
        """
        Display all matplotlib figures.

        Convenience wrapper around ``plt.show()`` so users don't need to
        import matplotlib separately.
        """
        plt.show()  # type: ignore[misc]

    def _track_figure(self, name: str, ax: Axes) -> None:
        """Track a figure for later saving if it was created by this plotter."""
        fig = ax.get_figure()
        if fig is not None:
            self._pending_figures.append((name, fig))  # type: ignore[arg-type]

    def plot_basin_stability_bars(self, ax: Axes | None = None) -> Axes:
        """
        Plot basin stability values as a bar chart.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        :return: The Axes object with the plot.
        """
        if self.bse.bs_vals is None:
            raise ValueError(
                "No basin stability values available. Please run estimate_bs() before plotting."
            )

        # Create standalone figure if no axes provided
        created_figure = ax is None
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]

        # Plot bar chart
        bar_labels, values = zip(*self.bse.bs_vals.items(), strict=True)
        ax.bar(bar_labels, values, color=["#ff7f0e", "#1f77b4"])  # type: ignore[misc]
        ax.set_xticks(bar_labels)  # type: ignore[misc]
        ax.set_ylabel(r"$\mathcal{S}(\bar{y}_i)$")  # type: ignore[misc]
        ax.set_title("Basin Stability")  # type: ignore[misc]

        if created_figure:
            self._track_figure("basin_stability_bars", ax)  # type: ignore[arg-type]

        return ax  # type: ignore[return-value]

    def plot_state_space(self, ax: Axes | None = None) -> Axes:
        """
        Plot initial conditions in state space, colored by their attractor labels.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        :return: The Axes object with the plot.
        """
        if self.bse.y0 is None:
            raise ValueError(
                "No initial conditions available. Please run estimate_bs() before plotting."
            )

        if self.bse.solution is None or self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        # Extract data
        initial_conditions = self.bse.y0.cpu().numpy()
        labels = np.array(self.bse.solution.labels)

        # Create standalone figure if no axes provided
        created_figure = ax is None
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]

        # Plot state space scatter
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            ax.scatter(  # type: ignore[misc]
                initial_conditions[idx, 0],
                initial_conditions[idx, 1],
                s=4,
                alpha=0.5,
                label=str(label),
            )
        ax.set_title("Initial Conditions in State Space")  # type: ignore[misc]
        ax.set_xlabel(r"$y_1$")  # type: ignore[misc]
        ax.set_ylabel(r"$y_2$")  # type: ignore[misc]
        ax.legend(loc="upper left")  # type: ignore[misc]

        if created_figure:
            self._track_figure("state_space", ax)  # type: ignore[arg-type]

        return ax  # type: ignore[return-value]

    def plot_feature_space(self, ax: Axes | None = None) -> Axes:
        """
        Plot feature space with classifier results.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        :return: The Axes object with the plot.
        """
        if self.bse.solution is None:
            raise ValueError("No solutions available. Please run estimate_bs() before plotting.")

        if self.bse.solution.features is None:
            raise ValueError("No features available. Please run estimate_bs() before plotting.")

        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        # Extract data
        features_array = self.bse.solution.features.cpu().numpy()
        all_labels = np.array(self.bse.solution.labels)

        # Features only exist for bounded trajectories.
        # Filter out unbounded trajectories (matching feature_space_aio.py approach)
        bounded_mask = all_labels != "unbounded"
        labels = all_labels[bounded_mask]

        # Verify features array matches bounded trajectory count
        if len(features_array) != len(labels):
            raise ValueError(
                f"Feature array size mismatch: {len(features_array)} features "
                f"vs {len(labels)} bounded trajectories"
            )

        n_features = features_array.shape[1] if features_array.ndim > 1 else 1

        # Ensure features_array is 2D for consistent indexing
        if features_array.ndim == 1:
            features_array = features_array.reshape(-1, 1)

        # Create standalone figure if no axes provided
        created_figure = ax is None
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]

        # Plot feature space scatter using boolean masks (matching feature_space_aio.py)
        unique_labels = np.unique(labels)
        rng = np.random.default_rng(42)

        for label in unique_labels:
            mask = labels == label
            if n_features >= 2:
                ax.scatter(  # type: ignore[misc]
                    features_array[mask, 0],
                    features_array[mask, 1],
                    s=5,
                    alpha=0.5,
                    label=str(label),
                )
            else:
                x_data = features_array[mask, 0]
                y_jitter = rng.uniform(-0.4, 0.4, size=len(x_data))
                ax.scatter(  # type: ignore[misc]
                    x_data,
                    y_jitter,
                    s=5,
                    alpha=0.5,
                    label=str(label),
                )

        if n_features >= 2:
            ax.set_title("Feature Space with Classifier Results")  # type: ignore[misc]
            feature_names = self.bse.solution.feature_names
            if feature_names and len(feature_names) >= 2:
                ax.set_xlabel(format_feature_for_display(feature_names[0]))  # type: ignore[misc]
                ax.set_ylabel(format_feature_for_display(feature_names[1]))  # type: ignore[misc]
            else:
                ax.set_xlabel(r"$\mathcal{X}_1$")  # type: ignore[misc]
                ax.set_ylabel(r"$\mathcal{X}_2$")  # type: ignore[misc]
        else:
            ax.set_title("Feature Space (1D Strip Plot)")  # type: ignore[misc]
            feature_names = self.bse.solution.feature_names
            if feature_names and len(feature_names) >= 1:
                ax.set_xlabel(format_feature_for_display(feature_names[0]))  # type: ignore[misc]
            else:
                ax.set_xlabel(r"$\mathcal{X}_1$")  # type: ignore[misc]
            ax.set_ylabel("")  # type: ignore[misc]
            ax.set_yticks([])  # type: ignore[misc]
            ax.set_ylim(-0.6, 0.6)  # type: ignore[misc]

        ax.legend()  # type: ignore[misc]

        if created_figure:
            self._track_figure("feature_space", ax)  # type: ignore[arg-type]

        return ax  # type: ignore[return-value]

    def plot_bse_results(self) -> Figure:
        """
        Generate diagnostic plots using the data stored in self.solution:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder plot for future use.

        This method combines the individual plotting functions into a 2x2 grid.
        For individual plots, use plot_basin_stability_bars(), plot_state_space(),
        or plot_feature_space() directly.

        :return: The Figure object with the 2x2 grid of plots.
        """
        # Create 2x2 subplot grid
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # type: ignore[misc]

        # Use the individual plotting functions
        self.plot_basin_stability_bars(ax=axs[0, 0])
        self.plot_state_space(ax=axs[0, 1])
        self.plot_feature_space(ax=axs[1, 0])

        # Placeholder for future plotting
        axs[1, 1].set_title("Future Plot")

        plt.tight_layout()

        self._pending_figures.append(("bse_results", fig))

        return fig  # type: ignore[return-value]

    # Plots 2 states over time for the same trajectory in the same space
    def plot_templates_phase_space(
        self,
        x_var: int = 0,
        y_var: int = 1,
        z_var: int | None = None,
        time_range: tuple[float, float] = (700, 1000),
    ) -> Figure:
        """
        Plot trajectories for the template initial conditions in 2D or 3D phase space.

        Creates a CPU copy of the solver with 10x the original n_steps for smoother
        visualization. Caching is disabled to avoid polluting the cache with
        visualization-specific integrations.

        :param x_var: State variable index for x-axis.
        :param y_var: State variable index for y-axis.
        :param z_var: State variable index for z-axis (3D plot if provided).
        :param time_range: Time range (t_start, t_end) to plot.
        """
        if self.bse.template_integrator is None:
            raise ValueError(
                "plot_phase requires a template_integrator with template initial conditions."
            )

        base_solver = self.bse.template_integrator.solver or self.bse.solver  # type: ignore[misc]

        # Create CPU copy with 10x n_steps for smoother plots, no caching
        solver = base_solver.clone(device="cpu", n_steps_factor=10, cache_dir=None)

        # Convert template_y0 list to tensor on solver's device
        template_tensor = torch.tensor(
            self.bse.template_integrator.template_y0,  # type: ignore[misc]
            dtype=torch.float32,
            device=solver.device,
        )

        t, trajectories = solver.integrate(
            self.bse.ode_system,
            template_tensor,
        )

        # Move tensors to CPU for plotting
        t = t.cpu().numpy()
        trajectories = trajectories.cpu().numpy()

        # Filter to time range
        t_min, t_max = time_range
        mask = (t >= t_min) & (t <= t_max)
        trajectories = trajectories[mask, :, :]

        fig = plt.figure(figsize=(8, 6))  # type: ignore[misc]
        if z_var is None:
            ax: Axes = fig.add_subplot(111)  # type: ignore[assignment]
            for i, (label, traj) in enumerate(
                zip(
                    self.bse.template_integrator.labels,
                    np.transpose(trajectories, (1, 0, 2)),
                    strict=True,
                )
            ):  # type: ignore[arg-type]
                ax.plot(
                    traj[:, x_var],
                    traj[:, y_var],
                    label=str(label),
                    color=get_color(i),
                    linewidth=2.5,
                )  # type: ignore[misc]
            ax.set_xlabel(f"$y_{{{x_var + 1}}}$")  # type: ignore[misc]
            ax.set_ylabel(f"$y_{{{y_var + 1}}}$")  # type: ignore[misc]
            ax.set_title("2D Phase Plot")  # type: ignore[misc]
        else:
            ax = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]
            for i, (label, traj) in enumerate(
                zip(
                    self.bse.template_integrator.labels,
                    np.transpose(trajectories, (1, 0, 2)),
                    strict=True,
                )
            ):  # type: ignore[arg-type]
                ax.plot(
                    traj[:, x_var],
                    traj[:, y_var],
                    traj[:, z_var],
                    label=str(label),
                    color=get_color(i),
                    linewidth=2.5,
                )  # type: ignore[misc,attr-defined]
            ax.set_xlabel(f"$y_{{{x_var + 1}}}$")  # type: ignore[misc]
            ax.set_ylabel(f"$y_{{{y_var + 1}}}$")  # type: ignore[misc]
            ax.set_zlabel(f"$y_{{{z_var + 1}}}$")  # type: ignore[misc,attr-defined]
            ax.set_title("3D Phase Plot")  # type: ignore[misc]

        plt.legend()  # type: ignore[misc]
        plt.tight_layout()

        self._pending_figures.append(("templates_phase_space", fig))

        return fig

    def plot_templates_trajectories(
        self,
        plotted_var: int,
        y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    ) -> Figure:
        """
        Plot template trajectories as vertically stacked subplots.

        Each trajectory gets its own row with independent y-axis scaling.
        This handles different amplitudes across attractors cleanly.

        Creates a CPU copy of the solver with 10x the original n_steps for smoother
        visualization. Caching is disabled to avoid polluting the cache with
        visualization-specific integrations.

        :param plotted_var: Index of the state variable to plot.
        :param y_limits: Y-axis limits. Tuple applies to all, dict maps label to (y_min, y_max).
        :param x_limits: X-axis limits. Tuple applies to all, dict maps label to (x_min, x_max).
        """
        if self.bse.template_integrator is None:
            raise ValueError(
                "plot_templates requires a template_integrator with template initial conditions."
            )

        base_solver = self.bse.template_integrator.solver or self.bse.solver  # type: ignore[misc]

        # Create CPU copy with 10x n_steps for smoother plots, no caching
        solver = base_solver.clone(device="cpu", n_steps_factor=10, cache_dir=None)

        template_tensor = torch.tensor(
            self.bse.template_integrator.template_y0,  # type: ignore[misc]
            dtype=torch.float32,
            device=solver.device,
        )

        t, y = solver.integrate(
            self.bse.ode_system,
            template_tensor,
        )

        t = t.cpu().numpy()
        y = y.cpu().numpy()

        labels = self.bse.template_integrator.labels
        trajectories = np.transpose(y, (1, 0, 2))  # (n_templates, n_time, n_states)
        n_templates = len(labels)

        use_shared_x = x_limits is None or isinstance(x_limits, tuple)
        fig, axes = plt.subplots(  # type: ignore[misc]
            n_templates,
            1,
            figsize=(8, 1.5 * n_templates),
            sharex=use_shared_x,
        )
        if n_templates == 1:
            axes = [axes]

        for i, (label, traj) in enumerate(zip(labels, trajectories, strict=True)):
            ax_i: Axes = axes[i]  # type: ignore[misc]
            color = get_color(i)

            x_lim = x_limits[label] if isinstance(x_limits, dict) else x_limits
            if x_lim is not None:
                x_min, x_max = x_lim
                traj_mask = (t >= x_min) & (t <= x_max)
                ax_i.plot(t[traj_mask], traj[traj_mask, plotted_var], linewidth=2.5, color=color)  # type: ignore[misc]
                ax_i.set_xlim(x_min, x_max)  # type: ignore[misc]
            else:
                ax_i.plot(t, traj[:, plotted_var], linewidth=2.5, color=color)  # type: ignore[misc]

            ax_i.set_ylabel(str(label), fontsize=10)  # type: ignore[misc]
            ax_i.tick_params(axis="both", which="major", labelsize=8)  # type: ignore[misc]

            y_lim = y_limits[label] if isinstance(y_limits, dict) else y_limits
            if y_lim is not None:
                ax_i.set_ylim(y_lim)  # type: ignore[misc]

        axes[-1].set_xlabel("time")  # type: ignore[misc]
        plt.tight_layout()

        self._pending_figures.append(("templates_trajectories", fig))

        return fig
