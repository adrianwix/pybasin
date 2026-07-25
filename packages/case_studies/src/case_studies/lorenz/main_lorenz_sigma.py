import matplotlib
import numpy as np

from pybasin.utils import time_execution

matplotlib.use("TkAgg")

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.study_params import SweepStudyParams

LORENZ_SIGMA_VALUES: list[float] = [float(v) for v in np.arange(0.12, 0.1825, 0.0025)]


def main():
    props = setup_lorenz_system()

    study_params = SweepStudyParams(
        **{'ode_system.params["sigma"]': LORENZ_SIGMA_VALUES},
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
        output_dir="results_sigma",
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz_sigma.py", main)

    plotter = MatplotlibStudyPlotter(bse)

    plotter.plot_parameter_stability()
    plotter.plot_orbit_diagram([0, 1, 2])
    plotter.show()

    bse.save()
