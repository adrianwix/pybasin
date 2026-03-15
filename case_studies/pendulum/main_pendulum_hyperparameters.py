"""Hyperparameter sensitivity study for the pendulum case.

This script varies the number of sampling points (N) to study the
sensitivity of basin stability values against this hyperparameter.
Based on the MATLAB bSTAB implementation.
"""

import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution

HYPERPARAMETER_N_VALUES: list[int] = [int(round(v)) for v in 5 * np.logspace(1, 3, 20)]


def main():
    """Run hyperparameter sensitivity study for pendulum system."""
    props = setup_pendulum_system()

    study_params = SweepStudyParams(
        n=HYPERPARAMETER_N_VALUES,
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
        output_dir="results_hyperparameters",
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_hyperparameters.py", main)

    plotter = MatplotlibStudyPlotter(bse)

    plotter.plot_parameter_stability()
    plotter.show()

    bse.save()
