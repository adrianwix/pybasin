import torch

from case_studies.lorenz.lorenz_ode import LorenzODE, LorenzParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.types import InteractivePlotterOptions
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import TorchDiffEqSolver
from pybasin.utils import time_execution


def main():
    # Takes 407.9446s using 10.000 points
    n = 10000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    # Parameters for broken butterfly system
    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}

    ode_system = LorenzODE(params)

    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0], max_limits=[10.0, 20.0, 0.0], device=device
    )

    # TorchDiffEqSolver doesn't support event functions, but LorenzODE has built-in masking
    # that stops dynamics when state magnitude exceeds 200
    solver = TorchDiffEqSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device=device,
        cache_dir=".pybasin_cache/lorenz",
    )

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        output_dir="results_case1_torchdiffeq",
    )

    result = bse.run()
    print("Basin Stability:", result["basin_stability"])

    # bse.save()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz_torchdiffeq.py", main)
    options: InteractivePlotterOptions = {
        "templates_phase_space": {"x_axis": 1, "y_axis": 2, "exclude_templates": ["unbounded"]},
        "feature_space": {"exclude_labels": ["unbounded"]},
    }
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y", 2: "z"}, options=options)
    # plotter.run(port=8050)
