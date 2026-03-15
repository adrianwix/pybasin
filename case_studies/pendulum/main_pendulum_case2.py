import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_study import BasinStabilityStudy

# from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution

CASE2_T_VALUES: list[float] = [float(v) for v in np.arange(0.01, 0.97, 0.05)]


def main():
    props = setup_pendulum_system()

    study_params = SweepStudyParams(
        **{'ode_system.params["T"]': CASE2_T_VALUES},
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert estimator is not None, "estimator is required for BasinStabilityStudy"

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        output_dir="results_case2",
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bss = time_execution("main_pendulum_case2.py", main)

    # plotter = MatplotlibStudyPlotter(bss)
    # plotter.plot_parameter_stability()
    # plotter.plot_orbit_diagram()
    # plotter.show()

    plotter = InteractivePlotter(
        bss,
        state_labels={0: "θ", 1: "ω"},
        options={"templates_time_series": {"state_variable": 1}},
    )
    plotter.run()
