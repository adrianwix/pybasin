import warnings
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import matplotlib
from diffrax import Dopri5, ODETerm, PIDController, SaveAt
from matplotlib import pyplot as plt

from case_studies.comparison_utils import compare_with_expected_by_size
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.solvers import JaxSolver
from pybasin.solvers.jax_ode_system import JaxODESystem
from pybasin.utils import time_execution

matplotlib.use("TkAgg")

warnings.filterwarnings("ignore", category=SyntaxWarning, module="nolds")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main():
    props = setup_pendulum_system()

    ode_system = cast(JaxODESystem[Any], props["ode_system"])
    default_p = ode_system.params_to_array()

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return ode_system.ode(t, y, default_p)

    time_span: tuple[float, float] = (0, 1000)
    n_steps = 1000
    t_eval = jnp.linspace(time_span[0], time_span[1], n_steps)  # type: ignore[reportUnknownMemberType]

    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(ode_wrapper),
            "solver": Dopri5(),
            "t0": time_span[0],
            "t1": time_span[1],
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-8, atol=1e-6),
            "max_steps": 16**5,
        },
        cache_dir=".pybasin_cache/pendulum",
    )

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=ode_system,
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        output_dir="results_case1_solver_args",
        feature_selector=None,
    )

    bse.estimate_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case1_jax_solver_args.py", main)

    expected_file = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "integration"
        / "pendulum"
        / "main_pendulum_case1.json"
    )

    if bse.result is not None:
        errors = bse.get_errors()
        compare_with_expected_by_size(bse.result["basin_stability"], expected_file, errors)

    mpl_plotter = MatplotlibPlotter(bse)

    mpl_plotter.plot_basin_stability_bars()
    mpl_plotter.plot_state_space()
    mpl_plotter.plot_feature_space()

    mpl_plotter.plot_templates_trajectories(plotted_var=1, x_limits=(0, 150))
    plt.show()  # type: ignore[misc]

    plotter = InteractivePlotter(
        bse,
        state_labels={0: "θ", 1: "ω"},
        options={"templates_time_series": {"state_variable": 1}},
    )
    plotter.run(debug=True)
