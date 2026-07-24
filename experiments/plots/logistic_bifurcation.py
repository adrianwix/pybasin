# pyright: basic
"""Verify extract_peak_values behavior using the logistic map bifurcation diagram.

The logistic map x_{n+1} = r * x_n * (1 - x_n) has well-known bifurcation behavior:
- r < 3: single fixed point
- r ~ 3: period-2 oscillation
- r ~ 3.45: period-4
- r ~ 3.54: period-8
- r > ~3.57: chaos (with periodic windows)

This script compares:
1. Traditional approach: plot all values after transient
2. Peak extraction: plot only local maxima (peaks)

For maps (discrete iterations), extract_peak_values should still work but will
show different behavior than for continuous ODEs. With discrete maps, "peaks"
are just local maxima in the iteration sequence.
"""

import matplotlib.pyplot as plt
import torch

from pybasin.utils import extract_peak_values


def generate_logistic_trajectories(
    r_values: torch.Tensor,
    n_transient: int = 500,
    n_record: int = 200,
) -> torch.Tensor:
    """Generate logistic map trajectories for given r values.

    :param r_values: Parameter values, shape (n_r,).
    :param n_transient: Iterations to discard (transient).
    :param n_record: Iterations to record after transient.
    :return: Trajectories of shape (n_record, n_r, 1).
    """
    n_r = r_values.shape[0]
    x = torch.ones(n_r) * 0.5

    for _ in range(n_transient):
        x = r_values * x * (1 - x)

    trajectory = torch.zeros(n_record, n_r, 1)
    for i in range(n_record):
        x = r_values * x * (1 - x)
        trajectory[i, :, 0] = x

    return trajectory


def plot_bifurcation_comparison() -> None:
    """Plot logistic map bifurcation using traditional vs peak extraction methods."""

    n_r = 800
    n_transient = 500
    n_record = 200

    r_values = torch.linspace(2.5, 4.0, n_r)
    trajectory = generate_logistic_trajectories(r_values, n_transient, n_record)

    peak_values, peak_counts = extract_peak_values(trajectory, n=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    r_np = r_values.numpy()
    for i in range(n_r):
        ax.scatter(
            [r_np[i]] * n_record,
            trajectory[:, i, 0].numpy(),
            c="black",
            s=0.1,
            alpha=0.3,
        )
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.set_title("Traditional: All iterations after transient")
    ax.set_xlim(2.5, 4.0)

    ax = axes[0, 1]
    for i in range(n_r):
        n_peaks = int(peak_counts[i, 0].item())
        if n_peaks > 0:
            peaks = peak_values[:n_peaks, i, 0].numpy()
            ax.scatter([r_np[i]] * n_peaks, peaks, c="blue", s=0.5, alpha=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel("Peak values")
    ax.set_title("Using extract_peak_values (local maxima)")
    ax.set_xlim(2.5, 4.0)

    ax = axes[1, 0]
    ax.plot(r_np, peak_counts[:, 0].numpy(), ".", markersize=1)
    ax.set_xlabel("r")
    ax.set_ylabel("Number of peaks")
    ax.set_title("Peak count vs r")
    ax.set_xlim(2.5, 4.0)

    ax = axes[1, 1]
    unique_values_per_r: list[int] = []
    for i in range(n_r):
        n_peaks = int(peak_counts[i, 0].item())
        if n_peaks > 0:
            peaks = peak_values[:n_peaks, i, 0]
            unique_peaks = torch.unique(torch.round(peaks * 1000) / 1000)
            unique_values_per_r.append(len(unique_peaks))
        else:
            unique_values_per_r.append(0)

    ax.plot(r_np, unique_values_per_r, ".", markersize=1)
    ax.set_xlabel("r")
    ax.set_ylabel("Unique peak values (rounded)")
    ax.set_title("Approximate period (unique peak values)")
    ax.set_xlim(2.5, 4.0)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig("experiments/plots/logistic_bifurcation.png", dpi=150)
    plt.show()
    print("Saved to experiments/plots/logistic_bifurcation.png")


if __name__ == "__main__":
    plot_bifurcation_comparison()
